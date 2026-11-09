"""Final GuidedNav Habitat rollout; prompt/generation retained from the verified evaluator."""
import os
import sys
import re
import json
import gzip
import argparse
import itertools
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import habitat
from habitat import Env
from habitat.config.default import get_config as get_habitat_config
from habitat.utils.visualizations.utils import images_to_video
from habitat_extensions import measures
from qwenvl.data.vln_unified_dataset import _build_vln_messages, DEFAULT_IMAGE_TOKEN, DEFAULT_MEMORY_TOKEN
from qwenvl.data.vln_sampling import history_indices
from qwenvl.temporal_token_merge import TemporalMergeConfig, TemporalResidualGate, parse_temporal_decay, register_temporal_merge_hook
from qwenvl.eval.eval_common import get_rank, get_world_size, get_local_rank, init_distributed, resolve_sim_gpu, RemoteEnv
from qwenvl.inference import load_final_model, validate_temporal_gate

def build_model_and_processor(model_path, min_pixels, max_pixels, device, dtype=torch.bfloat16):
    base_path = os.environ.get('GUIDEDNAV_BASE')
    if not base_path:
        raise ValueError('Set GUIDEDNAV_BASE to the exact released navigation-base directory')
    return load_final_model(model_path, base_path, device, min_pixels, max_pixels, 'flash_attention_2',
                            weights_source=os.environ.get('GUIDEDNAV_WEIGHTS_SOURCE', 'released'))

class VLNEvaluator:
    NAV_PROMPT = (
        "You are an autonomous navigation assistant. Your task is to "
        "<instruction>. Devise an action sequence to follow the instruction "
        "using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 "
        "degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
    )

    def __init__(
        self,
        config_path,
        split,
        env_num,
        output_path,
        model,
        processor,
        tokenizer,
        args,
        epoch=0,
    ):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.video_filter = getattr(args, "video_filter", "all")
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.scob = None
        self.scob_query_tokens = ()

        self.num_history = args.num_history
        self.num_future_steps = args.num_future_steps
        self.num_exec = getattr(args, "num_exec", 0)
        self.max_new_tokens = args.max_new_tokens
        self.max_episodes = args.max_episodes
        self.temporal_merge_config = getattr(args, "temporal_merge_config", None)
        self.temporal_residual_gate = getattr(args, "temporal_residual_gate", None)
        self.current_window_size = getattr(args, "current_window_size", 1)
        self.future_query_tokens = []
        # A single, in-distribution conjunction keeps eval deterministic (the
        # model was trained on all seven uniformly, so any one is in-domain).
        self.conjunction = "you can see "

        # Arrow glyph -> action index, for parsing the model's answer. (The
        # reverse map went away with the multi-turn prompt: nothing re-encodes
        # emitted actions back into text any more.)
        self.actions2idx = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3]})

        self.config = get_habitat_config(config_path)
        # Sim renders on GPU (SIM_GPU_BASE + local_rank). With REMOTE_SIM=1 this
        # is a *different* physical GPU than the model (see RemoteEnv) — the only
        # way to keep env.step ~4 ms instead of ~5 s on this box.
        self.remote_sim = os.environ.get("REMOTE_SIM", "0") == "1"
        self.sim_gpu = resolve_sim_gpu()
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = self.sim_gpu
            # Optional scenes_dir override (train episodes reference scenes that
            # may live outside the shipped scenes_dir). Empty = use config default.
            sd = getattr(args, "scenes_dir", "") or ""
            if sd:
                self.config.habitat.dataset.scenes_dir = sd
        # ``RemoteEnv`` serializes episodes through a small RPC boundary.  The
        # underlying Habitat episode still carries the instruction text, but
        # RxR-specific fields such as language and instruction_id are lost in
        # that round trip.  Recover them from the immutable source JSON so
        # language-stratified summaries are exact rather than silently empty.
        self.rxr_instruction_metadata = self._load_rxr_instruction_metadata()
        os.makedirs(self.output_path, exist_ok=True)

    def _load_rxr_instruction_metadata(self):
        """Map (scene id, episode id) to the source RxR instruction fields.

        This is deliberately a metadata-only read: it neither alters the
        dataset sent to Habitat nor changes the rollout/policy inputs.
        """
        dataset_path = str(self.config.habitat.dataset.data_path).format(
            split=self.split
        )
        if not dataset_path.endswith(".gz") or not os.path.isfile(dataset_path):
            return {}
        try:
            with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[eval] unable to read RxR instruction metadata from "
                f"{dataset_path}: {exc}",
                flush=True,
            )
            return {}

        metadata = {}
        for row in payload.get("episodes", []):
            instruction = row.get("instruction") or {}
            scene_id = str(row.get("scene_id", "")).rstrip("/").split("/")[-2]
            episode_id = str(row.get("episode_id"))
            if not scene_id or episode_id == "None":
                continue
            metadata[scene_id, episode_id] = {
                "instruction_id": instruction.get("instruction_id"),
                "language": instruction.get("language"),
            }
        print(
            f"[eval] restored RxR metadata for {len(metadata)} episodes",
            flush=True,
        )
        return metadata

    # -- prompting ----------------------------------------------------------
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

    # -- rollout ------------------------------------------------------------
    def _on_episode_start(self):
        """Hook called once per episode after the frame buffer is reset. No-op for
        the single-turn discrete evaluator; the episode-level pointing evaluator
        overrides it to reset its growing-conversation state."""
        pass

    def _on_episode_end(self, scene_id, episode_id, instruction, metrics, steps):
        """Hook called once per episode after metrics are computed. No-op by
        default; the pointing evaluator overrides it to dump the per-decision
        point/execution trace (``--viz_dir``)."""
        pass

    def config_env(self):
        if self.remote_sim:
            return RemoteEnv(self.config, self.sim_gpu)
        return Env(config=self.config)

    def eval_action(self, idx):
        env = self.config_env()
        # Subclasses whose action decoding needs the simulator (e.g. the pointing
        # evaluator's ShortestPathFollower point-goal execution) read it here;
        # parse_actions only receives the model text.
        self.env = env

        scene_episode_dict = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        result_file = os.path.join(self.output_path, "result.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                for line in f:
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "scene_id" not in res:
                        continue
                    done_res.append(
                        [res["scene_id"], res["episode_id"], res["episode_instruction"]]
                    )
                    if get_rank() == 0:
                        sucs.append(res["success"])
                        spls.append(res["spl"])
                        oss.append(res["os"])
                        ones.append(res["ne"])

        n_done = 0
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split("/")[-2]
            for episode in episodes[idx :: self.env_num]:
                if self.max_episodes and n_done >= self.max_episodes:
                    break
                episode_instruction = episode.instruction.instruction_text
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                import time as _time
                _rst0 = _time.perf_counter()
                observations = env.reset()
                _t_reset = _time.perf_counter() - _rst0
                _t_gen = 0.0; _t_step = 0.0; _n_gen = 0
                _ep0 = _time.perf_counter()

                # -- per-episode streaming state --
                # Single-turn prompting means there is no window/round state to
                # carry: every decision is rebuilt from `all_frames` alone (the
                # uniform history + the current frame), exactly as in training.
                all_frames = []      # every observed RGB frame (PIL), one per step
                pending = []         # queued action indices left to execute
                step_id = 0
                vis_frames = []
                self._on_episode_start()

                while not env.episode_over:
                    rgb = observations["rgb"]
                    img = Image.fromarray(rgb).convert("RGB")
                    all_frames.append(img)

                    if not pending:
                        _g0 = _time.perf_counter()
                        text = self._generate_chunk(episode_instruction, all_frames)
                        _t_gen += _time.perf_counter() - _g0
                        _n_gen += 1
                        acts = self.parse_actions(text)
                        if not acts:  # model emitted no parsable action -> STOP
                            acts = [0]
                        acts = acts[: self.num_future_steps]
                        if self.num_exec > 0:
                            # Re-plan after num_exec steps instead of running the
                            # whole emitted chunk (parity knob with the continuous
                            # evaluator's --num_exec). 0 = execute the full chunk.
                            acts = acts[: self.num_exec]
                        pending = list(acts)

                    if self.save_video:
                        # RGB-only capture (aligned with the continuous evaluator):
                        # no TopDownMap — that re-renders the whole scene each step
                        # (CPU-bound, ~seconds/step on large train scenes). Overlay
                        # step + the action about to execute.
                        act_name = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT"}.get(
                            pending[0], str(pending[0])
                        )
                        frame = Image.fromarray(np.asarray(rgb)).convert("RGB")
                        ImageDraw.Draw(frame).text(
                            (5, 4), f"step {step_id}  {act_name}", fill=(255, 255, 0)
                        )
                        vis_frames.append(np.asarray(frame))

                    action = pending.pop(0)
                    _s0 = _time.perf_counter()
                    observations = env.step(action)
                    _t_step += _time.perf_counter() - _s0
                    step_id += 1

                metrics = env.get_metrics()
                # SDTW is not enabled in the official RxR task config, but its
                # definition is exactly Success * nDTW. Derive it once at the
                # episode end to avoid forcing nDTW recomputation every step.
                if "ndtw" in metrics and "sdtw" not in metrics:
                    metrics["sdtw"] = float(metrics["success"] * metrics["ndtw"])
                if self.save_video and vis_frames:
                    sr, osr = float(metrics["success"]), float(metrics["oracle_success"])
                    keep = (
                        self.video_filter == "all"
                        or (self.video_filter == "fail" and sr == 0.0)
                        or (self.video_filter == "sr0_osr" and sr == 0.0 and osr > 0.0)
                    )
                    if keep:
                        images_to_video(
                            vis_frames,
                            os.path.join(self.output_path, f"vis_{self.epoch}"),
                            f"{scene_id}_{episode_id}",
                            fps=6,
                            quality=9,
                        )

                self._on_episode_end(
                    scene_id, episode_id, episode_instruction, metrics, step_id)

                sucs.append(metrics["success"])
                spls.append(metrics["spl"])
                oss.append(metrics["oracle_success"])
                ones.append(metrics["distance_to_goal"])
                n_done += 1
                if os.environ.get("VLN_TIMING"):
                    _wall = _time.perf_counter() - _ep0
                    print(
                        f"[TIMING r{idx}] steps={step_id} n_gen={_n_gen} "
                        f"gen={_t_gen:.1f}s({_t_gen/max(_n_gen,1):.2f}/gen) "
                        f"env_step={_t_step:.1f}s({_t_step/max(step_id,1)*1000:.0f}ms) "
                        f"reset={_t_reset:.1f}s wall={_wall:.1f}s "
                        f"other={_wall-_t_gen-_t_step-_t_reset:.1f}s",
                        flush=True,
                    )
                print(
                    f"[rank {idx}] {scene_id}_{episode_id} "
                    f"success={metrics['success']} spl={metrics['spl']:.3f} "
                    f"os={metrics['oracle_success']} ne={metrics['distance_to_goal']:.3f} "
                    f"ndtw={metrics.get('ndtw', float('nan')):.3f} "
                    f"sdtw={metrics.get('sdtw', float('nan')):.3f} "
                    f"tl={metrics.get('path_length', float('nan')):.3f} "
                    f"steps={metrics.get('steps_taken', step_id):.0f}",
                    flush=True,
                )

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "instruction_id": self.rxr_instruction_metadata.get(
                        (scene_id, str(episode_id)), {}
                    ).get(
                        "instruction_id",
                        getattr(episode.instruction, "instruction_id", None),
                    ),
                    "language": self.rxr_instruction_metadata.get(
                        (scene_id, str(episode_id)), {}
                    ).get(
                        "language", getattr(episode.instruction, "language", None)
                    ),
                    "success": float(metrics["success"]),
                    "spl": float(metrics["spl"]),
                    "os": float(metrics["oracle_success"]),
                    "ne": float(metrics["distance_to_goal"]),
                    # Retain the old field for consumers of existing R2R files.
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }
                for metric_name in ("ndtw", "sdtw", "path_length", "steps_taken"):
                    if metric_name in metrics:
                        result[metric_name] = float(metrics[metric_name])
                with open(result_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
            if self.max_episodes and n_done >= self.max_episodes:
                break

        env.close()
        d = self.device
        return (
            torch.tensor(sucs, dtype=torch.float32, device=d),
            torch.tensor(spls, dtype=torch.float32, device=d),
            torch.tensor(oss, dtype=torch.float32, device=d),
            torch.tensor(ones, dtype=torch.float32, device=d),
            torch.tensor(len(sucs), device=d),
        )


def evaluate(model, processor, tokenizer, args):
    world_size = get_world_size()
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        args=args,
        epoch=0,
    )
    evaluator.eval_action(get_rank())  # each rank appends its episodes to result.json

    # Barrier so every rank has finished writing before rank 0 aggregates.
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if get_rank() == 0:
        # Aggregate straight from the shared result.json (every rank appended to
        # it). Robust to uneven shards / oversubscription — avoids the previous
        # per-rank all_gather that assumed equal per-rank episode counts.
        rf = os.path.join(args.output_path, "result.json")
        S = SP = OS_ = NE = NDTW_ = SDTW_ = TL = STEPS = 0.0
        N_EXTENDED = 0
        N = 0
        with open(rf, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "scene_id" not in r:  # skip any prior summary line
                    continue
                N += 1
                S += r["success"]; SP += r["spl"]; OS_ += r["os"]; NE += r["ne"]
                if all(
                    key in r
                    for key in ("ndtw", "sdtw", "path_length", "steps_taken")
                ):
                    NDTW_ += r["ndtw"]; SDTW_ += r["sdtw"]
                    TL += r["path_length"]; STEPS += r["steps_taken"]
                    N_EXTENDED += 1
        n = max(N, 1)
        result_all = {
            "success": S / n,
            "spl": SP / n,
            "oracle_success": OS_ / n,
            "navigation_error": NE / n,
            "num_episodes": N,
            "split": args.eval_split,
        }
        if N_EXTENDED:
            if N_EXTENDED != N:
                raise RuntimeError(
                    f"Only {N_EXTENDED}/{N} episode rows contain extended metrics"
                )
            result_all.update({
                "ndtw": NDTW_ / n,
                "sdtw": SDTW_ / n,
                "path_length": TL / n,
                "steps_taken": STEPS / n,
            })
        print("==== EVAL RESULT ====")
        print(json.dumps(result_all, indent=2))
        with open(rf, "a") as f:
            f.write(json.dumps(result_all) + "\n")


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--scenes_dir", type=str, default="",
                        help="override habitat.dataset.scenes_dir (e.g. a full MP3D set for the train split); empty = config default")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, default="./results/val_unseen/qwen3vl")
    parser.add_argument("--num_history", type=int, default=8,
                        help="memory_size for history sampling; MUST match the value "
                             "build_vln_samples.py used (yields num_history-1 frames)")
    parser.add_argument("--num_future_steps", type=int, default=4,
                        help="max actions kept from one generation (training label length)")
    parser.add_argument("--num_exec", type=int, default=0,
                        help="actions executed before re-planning (0 = the whole emitted chunk); "
                             "parity knob with the continuous evaluator")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Per-rank cap on episodes (0 = no cap). Use for smoke tests.")
    parser.add_argument("--max_pixels", type=int, default=28 * 28 * 256)
    parser.add_argument("--min_pixels", type=int, default=28 * 28 * 16)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--video_filter", type=str, default="all",
                        choices=["all", "fail", "sr0_osr"],
                        help="which episodes to record when --save_video: all, fail (SR=0), or sr0_osr (SR=0 & OSR>0, reached goal but never stopped)")
    args = parser.parse_args()

    # Temporal E3/E4 checkpoints carry their exact merge recipe.  Reconstruct
    # it automatically so online rollout uses the same current-frame pathway
    # as training; legacy control/SF checkpoints remain unchanged.
    args.temporal_merge_config = None
    args.temporal_residual_gate = None
    args.current_window_size = 1
    temporal = None
    recipe_path = os.path.join(args.model_path, "experiment_recipe.json")
    joint_recipe_path = os.path.join(args.model_path, "joint_recipe.json")
    if not os.path.isfile(recipe_path) and os.path.isfile(joint_recipe_path):
        recipe_path = joint_recipe_path
    if os.path.isfile(recipe_path):
        with open(recipe_path, encoding="utf-8") as handle:
            recipe = json.load(handle)
        temporal = recipe.get("temporal_spatial_arguments") or recipe.get("e6")
        if temporal and temporal.get("current_merge_enable", False):
            window = int(temporal.get("current_window_size", 4))
            listed_decay = parse_temporal_decay(
                temporal.get("current_temporal_decay", "0.5,0.7,0.9")
            )
            if len(listed_decay) != window - 1:
                raise ValueError(
                    f"Checkpoint merge decay {listed_decay} does not match window {window}."
                )
            args.current_window_size = window
            args.temporal_merge_config = TemporalMergeConfig(
                merge_ratio=float(temporal.get("current_merge_ratio", 0.5)),
                spatial_bias=float(temporal.get("current_spatial_bias", 0.05)),
                anchor_weight=float(temporal.get("current_anchor_weight", 1.0)),
                temporal_decay=tuple(reversed(listed_decay)),
                fusion_mode=temporal.get("current_fusion_mode", "weighted_mean"),
                matching_mode=temporal.get("current_matching_mode", "source_best"),
            )
            print(
                "[eval] temporal current merge enabled: "
                f"window={window}, ratio={args.temporal_merge_config.merge_ratio}, "
                f"fusion={args.temporal_merge_config.fusion_mode}, "
                f"matching={args.temporal_merge_config.matching_mode}"
            )

    init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor, tokenizer = build_model_and_processor(
        args.model_path, args.min_pixels, args.max_pixels, device
    )
    if (
        args.temporal_merge_config is not None
        and args.temporal_merge_config.fusion_mode == "gated_residual"
    ):
        if temporal is None:
            raise RuntimeError("Gated temporal checkpoint is missing its recipe.")
        gate = TemporalResidualGate(
            initial_gate=float(temporal.get("current_gate_init", 0.05)),
            max_gate=float(temporal.get("current_gate_max", 0.2)),
        )
        state_name = (
            "joint_auxiliary_modules.pt"
            if os.path.isfile(os.path.join(args.model_path, "joint_recipe.json"))
            else "temporal_spatial_projectors.pt"
        )
        state_path = os.path.join(args.model_path, state_name)
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"Missing gated temporal state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if "temporal_residual_gate" not in state:
            raise KeyError(f"temporal_residual_gate is missing from {state_path}")
        gate.load_state_dict(state["temporal_residual_gate"], strict=True)
        gate.to(device=device, dtype=torch.float32).eval()
        validate_temporal_gate(gate)
        args.temporal_residual_gate = gate
        print(f"[eval] restored temporal residual gate={gate().item():.6f}")
    evaluate(model, processor, tokenizer, args)

    if get_world_size() > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    eval()
