"""Prompt, history, temporal fusion and greedy generation from the verified final evaluator."""
import itertools
import re
import torch
from qwenvl.data.vln_unified_dataset import _build_vln_messages, DEFAULT_IMAGE_TOKEN, DEFAULT_MEMORY_TOKEN
from qwenvl.data.vln_sampling import history_indices
from qwenvl.temporal_token_merge import register_temporal_merge_hook

class ReferencePolicy:
    NAV_PROMPT = (
        "You are an autonomous navigation assistant. Your task is to "
        "<instruction>. Devise an action sequence to follow the instruction "
        "using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 "
        "degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
    )


    def _build_infer_messages(self, instruction, all_frames):
        """Rebuild the training-time prompt for the current decision step.

        This is a 1:1 replay of what ``build_vln_samples.expand_episode`` bakes
        into each per-step record and ``VLNUnifiedDataset`` (label_mode="action")
        feeds the model: ONE user turn = instruction + optional ``<memory>``
        (uniform-over-trajectory history) + exactly ONE ``<image>`` (the current
        frame), and the assistant turn is what we're asking the model to produce.

        Previously this evaluator reconstructed a MULTI-turn window (one user turn
        per decision round, with previously-emitted arrow strings replayed as
        assistant turns, reset every ``num_frames`` steps). That format belonged
        to the windowed ``VLNActionDataset`` retired in 2242b5c; against the
        current per-step data it was a train/eval mismatch — the model saw up to
        8 current frames across ~15 turns, and for the first 32 steps no
        ``<memory>`` block appeared at all (``window_start == 0``), a prompt shape
        that never occurs in training (where any ``c >= 1`` carries ``<memory>``).
        Single-turn is also what OmniNav and Uni-NaVid do at both train and eval.
        """
        nav = self.NAV_PROMPT.replace("<instruction>.", instruction)
        cur = len(all_frames) - 1  # index of the current frame
        hist_idx = history_indices(cur, self.num_history)
        memory_images = [all_frames[j] for j in hist_idx]

        value = nav
        if memory_images:
            # Omitted entirely at cur == 0 — training does the same (an empty
            # <memory> clause is a prompt shape the model never saw).
            value += f" These are your historical observations: {DEFAULT_MEMORY_TOKEN}."
        value += f" {self.conjunction}{DEFAULT_IMAGE_TOKEN}."
        if self.future_query_tokens:
            # The fixed query is policy input. Future RGB/VGGT targets are never
            # present at inference; generated action tokens can attend to this
            # query exactly as they did during training.
            value += " " + " ".join(self.future_query_tokens)
        if self.scob_query_tokens:
            # The five fixed rows are causal inputs immediately before the
            # assistant action.  They are deliberately appended after the
            # current image so ScobController can verify the final RGB run.
            value += " " + " ".join(self.scob_query_tokens)

        sources = [{"from": "human", "value": value}]
        return _build_vln_messages(sources, [all_frames[cur]], memory_images)


    def parse_actions(self, output):
        action_patterns = "|".join(re.escape(a) for a in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[m] for m in matches]
        return list(itertools.chain.from_iterable(actions))


    @torch.inference_mode()
    def _generate_chunk(self, instruction, all_frames):
        messages = self._build_infer_messages(instruction, all_frames)
        auxiliary = None
        if self.temporal_merge_config is not None:
            current = len(all_frames) - 1
            start = max(0, current - self.current_window_size + 1)
            auxiliary = list(all_frames[start:current])
        return self._run_generation(messages, temporal_auxiliary=auxiliary)


    @torch.inference_mode()
    def _run_generation(self, messages, temporal_auxiliary=None):
        """Tokenize a ready messages list, greedily generate, and decode the new
        tokens. Split out of ``_generate_chunk`` so subclasses that construct their
        own conversation (e.g. the episode-level pointing evaluator) can reuse the
        generate/decode core without re-implementing it."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        hook = None
        if self.temporal_merge_config is not None and temporal_auxiliary:
            auxiliary = self.processor.image_processor(
                temporal_auxiliary, return_tensors="pt"
            )
            auxiliary_pixels = auxiliary["pixel_values"].to(
                device=self.device, dtype=self.model.dtype
            )
            auxiliary_grids = auxiliary["image_grid_thw"].to(self.device)
            # Frames are chronological and end at t-1.
            auxiliary_ages = torch.arange(
                len(temporal_auxiliary), 0, -1, device=self.device
            )
            hook = register_temporal_merge_hook(
                self.model,
                aux_pixel_values=auxiliary_pixels,
                aux_grid_thw=auxiliary_grids,
                main_grid_thw=inputs["image_grid_thw"],
                aux_ages=auxiliary_ages,
                config=self.temporal_merge_config,
                residual_gate=self.temporal_residual_gate,
            )
        try:
            generation_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.scob_query_tokens:
                generation_kwargs["suppress_tokens"] = [
                    int(self.tokenizer.convert_tokens_to_ids(token))
                    for token in self.scob_query_tokens
                ]
            gen = self.model.generate(**inputs, **generation_kwargs)
        finally:
            if hook is not None:
                hook.remove()
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

