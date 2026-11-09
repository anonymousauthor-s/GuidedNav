"""vln_unified_dataset.py — the single VLN dataset module.

Owns everything for the offline per-step VLN path, for BOTH trainers:

* ``VLNUnifiedDataset`` — reads the sample-level ``samples_perstep.jsonl``
  produced by ``data_process/build_vln_samples.py`` (one record per timestep:
  pre-sampled history + current image paths, the discrete action label, and the
  waypoint label). Because all windowing/index math is done offline,
  ``__getitem__`` only loads the listed images, tokenizes, and attaches the
  label for the selected head. A single ``label_mode`` switch serves both
  trainers with the SAME records, so the discrete-vs-continuous A/B differs only
  in the action head:

    - ``label_mode="action"``   — assistant turn = arrow glyphs; loss on those
      spans (``_mask_labels_qwen``). Emits ``task_type``; pairs with
      ``VLNActionDataCollator``.
    - ``label_mode="waypoint"`` — assistant turn = the K ``<plan_i>`` tokens;
      labels fully masked (flow loss keyed on plan tokens). Emits
      ``plan_token_mask``/``future_waypoints``; pairs with
      ``VLNWaypointDataCollator``. Records whose ``waypoints`` is null (final
      ``predict_size`` frames of an episode) are dropped at index-build time.
    - ``label_mode="pointing"`` — Robostral-style pixel-goal *text* output; the
      assistant turn is ``POINT u v MOVE dx dy dtheta`` / ``MOVE dx dy dtheta``
      / ``STOP``, read pre-baked out of ``conversations``. Mechanically it is the
      action text path (loss on the assistant span via ``_mask_labels_qwen``,
      plain LM, no regression head); it differs only in the prompt and the
      numeric fields (``pixel_goal``/``goal_wp``). Pairs with
      ``VLNActionDataCollator``. **Dormant**: no in-tree builder writes these
      records and no trainer selects this mode — the pointing pipeline was
      removed pending a rebuild. Kept because the mode itself is self-contained.

* Shared helpers (``_build_vln_messages``, ``_mask_labels_qwen``), placeholder
  constants (``DEFAULT_IMAGE_TOKEN``/``DEFAULT_MEMORY_TOKEN``), the continuous
  prompt (``WAYPOINT_SYSTEM_PROMPT``), the pose→local-frame waypoint geometry
  (``compute_future_waypoints``/``_quat_to_matrix``), and both collators. These
  are also imported by the evaluators (``eval_{discrete,continuous}_action``).
  (The old windowed ``VLNActionDataset`` /
  ``VLNWaypointDataset`` modules were retired; this is now the only home.)

The 36G JSONL is NOT loaded into memory: we build a byte-offset index (one int64
per line) and ``seek``/``readline`` per item. Each DDP rank builds the same index
(deterministic), so the DistributedSampler stays consistent.
"""

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .data_processor import IGNORE_INDEX, update_processor_pixels, rank0_print
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_MEMORY_TOKEN = "<memory>"

# Same current-frame conjunctions build_vln_samples bakes into the discrete
# conversations. The continuous head rebuilds its human turn at load time (its
# answer is plan tokens, not glyphs), so we sample a conjunction here too — B3:
# otherwise the discrete head sees 7 phrasings and the continuous head only one,
# an avoidable text-distribution asymmetry between the two A/B arms.
CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]

# Continuous-head prompt: drops the "four discrete actions" wording (wrong for a
# waypoint head) and just frames the trajectory-planning task.
WAYPOINT_SYSTEM_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Based on your observations, plan the future trajectory to follow the instruction."
)


# ---------------------------------------------------------------------------
# Message / label helpers (shared by the dataset and the evaluators)
# ---------------------------------------------------------------------------
def _build_vln_messages(
    interleave_sources: List[Dict[str, str]],
    images: List[Image.Image],
    memory_images: List[Image.Image],
) -> List[Dict]:
    """Turn the interleaved llava-style [{"from": "human"/"gpt", "value": ...}, ...]
    into the Qwen chat-template format with typed content blocks.

    The text may contain `<image>` and `<memory>` placeholders. `<memory>`
    blocks are expanded into one `{type:"image"}` entry per memory frame.
    """
    image_pool = list(images)
    memory_pool = list(memory_images)

    messages: List[Dict] = []
    for turn in interleave_sources:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content: List[Dict] = []
            parts = re.split(r"(<image>|<memory>)", text)
            for seg in parts:
                if seg == DEFAULT_IMAGE_TOKEN:
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds images."
                        )
                    content.append({"type": "image", "image": image_pool.pop(0)})
                elif seg == DEFAULT_MEMORY_TOKEN:
                    if not memory_pool:
                        raise ValueError(
                            "Encountered <memory> placeholder but no memory frames."
                        )
                    # Expand <memory> into each historical frame as an image.
                    for mem_img in memory_pool:
                        content.append({"type": "image", "image": mem_img})
                    memory_pool = []
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append(
                {"role": role, "content": [{"type": "text", "text": text}]}
            )

    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if memory_pool:
        raise ValueError(
            f"{len(memory_pool)} memory image(s) remain unused."
        )

    return messages


def _mask_labels_qwen(input_ids: torch.Tensor) -> torch.Tensor:
    """Qwen chat template uses `<|im_start|>assistant\n` (... token 77091 ...)
    to mark the start of an assistant turn and `<|im_end|>` (token 151645) to
    mark its end. Unmask only the assistant responses for the loss."""
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    ids = input_ids[0].tolist()
    L = len(ids)
    pos = 0
    while pos < L:
        if ids[pos] == 77091:  # assistant header
            ans_start = pos + 2  # skip "assistant\n"
            ans_end = ans_start
            while ans_end < L and ids[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1
    return labels


def _find_token_sequence(ids: Sequence[int], needle: Sequence[int], start: int, end: int):
    """Return the first occurrence of ``needle`` in ``ids[start:end]``."""
    needle = list(needle)
    if not needle:
        return None
    last = end - len(needle)
    for pos in range(start, last + 1):
        if list(ids[pos : pos + len(needle)]) == needle:
            return pos
    return None


def _subtask_token_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Mark only the SUBTASK portion of an assistant answer.

    FGR2R subtask rows contain one assistant answer of the form
    ``SUBTASK: ...\\nACTION: ...``.  The normal Qwen label mask supervises the
    complete assistant span; this additional mask lets the trainer downweight
    the subtask tokens while keeping action-token supervision at full weight.
    """
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    ids = input_ids[0].tolist()
    length = len(ids)
    pos = 0
    action_markers = []
    for text in ("ACTION:", "\nACTION:", "ACTION: ", "\nACTION: "):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            action_markers.append(encoded)

    while pos < length:
        if ids[pos] != 77091:  # <|im_start|>
            pos += 1
            continue
        ans_start = pos + 2  # skip "assistant\\n"
        ans_end = ans_start
        while ans_end < length and ids[ans_end] != 151645:  # <|im_end|>
            ans_end += 1
        if ans_end >= length:
            break
        action_pos = None
        for marker in action_markers:
            found = _find_token_sequence(ids, marker, ans_start, ans_end)
            if found is not None and (action_pos is None or found < action_pos):
                action_pos = found
        if action_pos is not None:
            mask[0, ans_start:action_pos] = True
        pos = ans_end + 1
    return mask


def _phase_token_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Mark the phase-token prefix in ``<PHASE_...>\nACTION: ...`` answers."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    ids = input_ids[0].tolist()
    length = len(ids)
    action_markers = []
    for text in ("ACTION:", "\nACTION:", "ACTION: ", "\nACTION: "):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            action_markers.append(encoded)
    pos = 0
    while pos < length:
        if ids[pos] != 77091:
            pos += 1
            continue
        ans_start = pos + 2
        ans_end = ans_start
        while ans_end < length and ids[ans_end] != 151645:
            ans_end += 1
        if ans_end >= length:
            break
        action_pos = None
        for marker in action_markers:
            found = _find_token_sequence(ids, marker, ans_start, ans_end)
            if found is not None and (action_pos is None or found < action_pos):
                action_pos = found
        if action_pos is not None:
            mask[0, ans_start:action_pos] = True
        pos = ans_end + 1
    return mask


# ---------------------------------------------------------------------------
# Waypoint geometry (used by the continuous evaluator + build_vln_samples)
# ---------------------------------------------------------------------------
def _quat_to_matrix(quat, quat_format="wxyz"):
    quat = np.asarray(quat, dtype=np.float32)
    if quat_format == "wxyz":
        w, x, y, z = quat
    elif quat_format == "xyzw":
        x, y, z, w = quat
    else:
        raise ValueError(f"Unsupported quat_format: {quat_format}")
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def compute_future_waypoints(positions, rotations, current_index, future_indices,
                             quat_format="wxyz"):
    """Future poses expressed in the current pose's local frame:
    ``[x_forward, y_lateral, sin(yaw), cos(yaw)]`` per future index."""
    current_pos = positions[current_index]
    current_rot = _quat_to_matrix(rotations[current_index], quat_format)
    current_rot_inv = current_rot.T

    waypoints = []
    for future_index in future_indices:
        future_index = int(future_index)
        delta_world = positions[future_index] - current_pos
        delta_local = current_rot_inv @ delta_world

        future_rot = _quat_to_matrix(rotations[future_index], quat_format)
        future_forward_world = future_rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        future_forward_local = current_rot_inv @ future_forward_world
        yaw_local = np.arctan2(future_forward_local[0], -future_forward_local[2])

        waypoints.append(
            [
                -delta_local[2],
                delta_local[0],
                np.sin(yaw_local),
                np.cos(yaw_local),
            ]
        )
    return np.asarray(waypoints, dtype=np.float32)


def _build_line_offsets(path: str) -> np.ndarray:
    """Byte offset of each line — lets us seek to record ``i`` without loading
    the whole file. int64 array; ~8 bytes/line (160 MB for 20M lines)."""
    offsets = [0]
    with open(path, "rb") as f:
        for line in f:
            offsets.append(offsets[-1] + len(line))
    offsets.pop()  # last entry points past EOF
    return np.asarray(offsets, dtype=np.int64)


class VLNUnifiedDataset(Dataset):
    def __init__(self, processor, data_args, vln_config, label_mode: str,
                 plan_token_ids: Optional[List[int]] = None,
                 plan_tokens: Optional[List[str]] = None,
                 stop_token_id: Optional[int] = None,
                 stop_token: Optional[str] = None):
        super().__init__()
        assert label_mode in ("action", "waypoint", "pointing")
        self.label_mode = label_mode

        def _cfg(key, default):
            if key in vln_config and vln_config[key] is not None:
                return vln_config[key]
            return getattr(data_args, key, default)

        self.task_id = vln_config.get("name", "vln")
        self.phase_mode = vln_config.get("phase_mode")
        # jsonl lives under the dataset's video_folder as samples_perstep.jsonl,
        # or an explicit `samples_path` in the registry entry.
        video_folder = vln_config["video_folder"]
        self.video_folder = video_folder
        self.samples_path = vln_config.get(
            "samples_path",
            os.path.join(video_folder, _cfg("samples_name", "samples_perstep.jsonl")),
        )

        self.predict_size = int(_cfg("predict_size", 8))
        self.waypoint_dim = int(_cfg("waypoint_dim", 4))

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

        model_type = getattr(data_args, "model_type", "qwen3vl")
        self.get_rope_index = {
            "qwen3vl": get_rope_index_3,
            "qwen2.5vl": get_rope_index_25,
            "qwen2vl": get_rope_index_2,
        }.get(model_type)
        if self.get_rope_index is None:
            raise ValueError(f"model_type {model_type} not supported")

        if label_mode == "waypoint":
            if not plan_token_ids or len(plan_token_ids) != self.predict_size:
                raise ValueError(
                    f"waypoint mode needs {self.predict_size} plan tokens, got "
                    f"{len(plan_token_ids) if plan_token_ids else 0}"
                )
            if stop_token_id is None or stop_token is None:
                raise ValueError("waypoint mode needs a <stop> token id + string")
            self.plan_token_ids = list(plan_token_ids)
            self.stop_token_id = int(stop_token_id)
            # Answer = K plan tokens followed by the dedicated <stop> token, so
            # every waypoint sample exposes one <stop> hidden for the stop head.
            self._plan_answer = "".join(plan_tokens) + stop_token

        # ---- build the (lazy) sample index --------------------------------
        offsets = _build_line_offsets(self.samples_path)
        # Preserve the row number in the unsampled source JSONL.  Auxiliary
        # caches are built against the full source file, so a dataset selected
        # as ``name%25`` must index its cache with this value rather than the
        # compact 0..len(sampled)-1 position.
        source_indices = np.arange(len(offsets), dtype=np.int64)
        self.source_row_count = int(len(offsets))
        # `%N` subsampling (deterministic, per registry entry) — same seed on
        # every rank so the DistributedSampler shards consistently.
        rate = float(vln_config.get("sampling_rate", 1.0))
        if rate < 1.0:
            n_keep = int(len(offsets) * rate)
            rng = np.random.RandomState(42)
            sel = np.sort(rng.choice(len(offsets), n_keep, replace=False))
            offsets = offsets[sel]
            source_indices = source_indices[sel]


        if label_mode == "waypoint":
            keep = []
            keep_source_indices = []
            with open(self.samples_path, "rb") as f:
                for off, source_index in zip(offsets, source_indices):
                    f.seek(int(off))
                    rec = json.loads(f.readline())
                    if "arrive_list" not in rec:
                        raise ValueError(
                            f"{self.samples_path} predates the per-waypoint arrive "
                            "labels (no 'arrive_list' field). Rebuild it with "
                            "data_process/build_vln_samples.py."
                        )
                    if rec.get("waypoints") is not None or any(rec["arrive_list"]):
                        keep.append(off)
                        keep_source_indices.append(source_index)
            offsets = np.asarray(keep, dtype=np.int64)
            source_indices = np.asarray(keep_source_indices, dtype=np.int64)

        self.offsets = offsets
        self.source_indices = source_indices
        self.phase_labels = None
        if self.phase_mode:
            suffix = ".phase_binary" if self.phase_mode == "binary" else ".phase_kway"
            sidecar = self.samples_path + suffix
            if not os.path.isfile(sidecar):
                raise FileNotFoundError(
                    f"Missing phase sidecar {sidecar}; run build_phase_sidecars.py once."
                )
            with open(sidecar, encoding="ascii") as f:
                all_phase_labels = [line.strip() for line in f]
            if len(all_phase_labels) != self.source_row_count:
                raise ValueError(
                    f"phase sidecar has {len(all_phase_labels)} rows, source has "
                    f"{self.source_row_count}"
                )
            self.phase_labels = [all_phase_labels[int(i)] for i in self.source_indices]
        rank0_print(
            f"VLN-unified {self.task_id} [{label_mode}]: {len(self.offsets)} "
            f"samples from {self.samples_path} (rate={rate})"
        )

    def __len__(self):
        return len(self.offsets)

    def source_index_for_sample(self, i: int) -> int:
        """Return the row in the original, unsampled JSONL for sample ``i``."""
        return int(self.source_indices[i])

    def waypoints_for_sample(self, i: int):
        """Precomputed future waypoints for sample ``i`` — image-free (used by
        waypoint-norm fitting). Returns None for terminal STOP rows (null
        waypoints), which the caller must skip."""
        rec = self._read_record(i)
        wp = rec.get("waypoints")
        if wp is None:
            return None
        return np.asarray(wp, dtype=np.float32)

    def _read_record(self, i: int) -> Dict:
        with open(self.samples_path, "rb") as f:
            f.seek(int(self.offsets[i]))
            return json.loads(f.readline())

    def _load_images(self, rel_paths: List[str]) -> List[Image.Image]:
        imgs = []
        for rp in rel_paths:
            imgs.append(Image.open(os.path.join(self.video_folder, rp)).convert("RGB"))
        return imgs

    def __getitem__(self, i):
        rec = self._read_record(i)
        # rec["images"] = [memory..., current]; memory count = n_memory.
        n_mem = int(rec["n_memory"])
        all_imgs = self._load_images(rec["images"])
        memory_images = all_imgs[:n_mem]
        current_images = all_imgs[n_mem:]   # exactly 1

        if self.label_mode in ("action", "pointing"):
            # Both consume the conversation baked offline: action → arrow glyphs,
            # pointing → "POINT u v MOVE dx dy dtheta"/"STOP". The builder writes
            # the mode-specific prompt + answer, so __getitem__ only tokenizes and
            # masks the assistant span; the two differ downstream (eval decoder,
            # numeric fields), not here.
            sources = [dict(turn) for turn in rec["conversations"]]
            if self.phase_mode:
                phase = int(rec.get("subtask_phase_index", 0))
                if self.phase_mode == "kway":
                    label = f"<PHASE_{phase}>"
                elif self.phase_mode == "binary":
                    # The phase boundary is represented by the first sample of
                    # each new geometry phase; c==0 is always the first step.
                    label = "<PHASE_NEXT>" if self.phase_labels[i] == "NEXT" else "<PHASE_SAME>"
                else:
                    raise ValueError(f"unsupported phase_mode={self.phase_mode}")
                for turn in reversed(sources):
                    if turn.get("from") == "gpt":
                        action = str(turn["value"])
                        action = action[action.index("ACTION:"):] if "ACTION:" in action else action
                        turn["value"] = f"{label}\n{action}"
                        break
        else:
            # Continuous head: rebuild the human turn from the bare instruction
            # using the waypoint prompt (no discrete-action wording); answer = plan tokens.
            human = WAYPOINT_SYSTEM_PROMPT.replace("<instruction>.", rec["instruction"])
            if n_mem > 0:
                human += (
                    f" These are your historical observations: {DEFAULT_MEMORY_TOKEN}."
                )
            human += f" {random.choice(CONJUNCTIONS)}{DEFAULT_IMAGE_TOKEN}."
            sources = [
                {"from": "human", "value": human},
                {"from": "gpt", "value": self._plan_answer},
            ]

        messages = _build_vln_messages(sources, current_images, memory_images)
        encoded = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        image_grid_thw = encoded.get("image_grid_thw", None)
        video_grid_thw = encoded.get("video_grid_thw", None)
        position_ids, _ = self.get_rope_index(
            self.merge_size, input_ids,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
        )
        seq_len = int(input_ids.size(1))

        data_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": [seq_len],
            "task_type": self.task_id,
        }
        if "pixel_values" in encoded:
            data_dict["pixel_values"] = encoded["pixel_values"]
            data_dict["image_grid_thw"] = encoded["image_grid_thw"]
        if "pixel_values_videos" in encoded:
            data_dict["pixel_values_videos"] = encoded["pixel_values_videos"]
            data_dict["video_grid_thw"] = encoded["video_grid_thw"]

        if self.label_mode in ("action", "pointing"):
            data_dict["labels"] = _mask_labels_qwen(input_ids)
            # Only subtask rows carry this field. Control rows remain on the
            # regular action-only loss path, so the paired A/B changes no
            # prompt, label, or action target.
            if self.phase_mode:
                data_dict["phase_token_mask"] = _phase_token_mask(
                    input_ids, self.tokenizer
                )
            elif rec.get("subtask"):
                subtask_mask = _subtask_token_mask(input_ids, self.tokenizer)
                # Low-confidence oracle labels remain in the prompt/record for
                # reproducibility, but contribute no text loss.  ACTION tokens
                # stay supervised at weight 1.  R2R rows predate this field and
                # therefore default to valid.
                if bool(rec.get("subtask_valid", True)):
                    data_dict["subtask_token_mask"] = subtask_mask
                else:
                    data_dict["labels"][subtask_mask] = IGNORE_INDEX
        else:
            data_dict["labels"] = torch.full_like(input_ids, IGNORE_INDEX)
            plan_ids = torch.tensor(self.plan_token_ids, device=input_ids.device)
            plan_token_mask = torch.isin(input_ids, plan_ids)
            if int(plan_token_mask.sum()) != self.predict_size:
                raise ValueError(
                    f"[{self.task_id}] expected {self.predict_size} plan tokens, "
                    f"found {int(plan_token_mask.sum())}"
                )
            data_dict["plan_token_mask"] = plan_token_mask
            stop_token_mask = input_ids == self.stop_token_id
            if int(stop_token_mask.sum()) != 1:
                raise ValueError(
                    f"[{self.task_id}] expected exactly 1 <stop> token, "
                    f"found {int(stop_token_mask.sum())}"
                )
            data_dict["stop_token_mask"] = stop_token_mask
            # Per-waypoint arrive flags: [P] float. The rollout stops when every
            # one is positive, so all P are supervised (including padded steps,
            # whose flag is legitimately 1 — a padded step points at the endpoint).
            arrive = np.asarray(rec["arrive_list"], dtype=np.float32)
            if arrive.shape[0] != self.predict_size:
                raise ValueError(
                    f"[{self.task_id}] arrive_list has {arrive.shape[0]} entries, "
                    f"expected predict_size={self.predict_size}"
                )
            data_dict["arrive_labels"] = torch.from_numpy(arrive)
            # Per-step validity for the REGRESSION loss only: padded waypoint
            # values are copies of the last real pose, not targets.
            n_valid = int(rec["wp_n_valid"])
            valid = torch.zeros(self.predict_size, dtype=torch.bool)
            valid[: min(n_valid, self.predict_size)] = True
            data_dict["waypoint_step_valid"] = valid
            if rec.get("waypoints") is not None:
                wp = np.asarray(rec["waypoints"], dtype=np.float32)
                data_dict["future_waypoints"] = torch.from_numpy(wp)
            else:
                # Terminal frame: no future poses at all (wp_n_valid == 0, so
                # every step is already masked out of the regression). Zeros keep
                # the collator's rows stackable; only the arrive head learns here.
                data_dict["future_waypoints"] = torch.zeros(
                    (self.predict_size, self.waypoint_dim), dtype=torch.float32
                )

        return data_dict


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------
def _pad_and_cat_position_ids(tensor_list):
    max_length = max(t.shape[2] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_length = max_length - t.shape[2]
        padded.append(
            torch.nn.functional.pad(t, (0, pad_length), "constant", 1)
        )
    return torch.cat(padded, dim=1)


@dataclass
class VLNActionDataCollator:
    """Collate discrete-action VLN samples, optionally mixed with standard VQA
    samples produced by ``LazySupervisedDataset``.

    VLN-specific fields (``task_type``) are optional — if an
    instance lacks them, it is treated as a VQA sample and skipped for those
    keys.
    """

    tokenizer: transformers.PreTrainedTokenizer
    tree_mask: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [ins["input_ids"].squeeze(0) for ins in instances]
        labels = [ins["labels"].squeeze(0) for ins in instances]
        position_ids_list = [ins["position_ids"] for ins in instances]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = _pad_and_cat_position_ids(position_ids_list)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if any("subtask_token_mask" in ins for ins in instances):
            seq_len = input_ids.size(1)
            subtask_rows = []
            for ins in instances:
                m = ins.get("subtask_token_mask")
                if m is None:
                    m = torch.zeros_like(ins["input_ids"], dtype=torch.bool)
                m = m.squeeze(0).bool()
                if m.numel() < seq_len:
                    m = torch.nn.functional.pad(m, (0, seq_len - m.numel()), value=False)
                else:
                    m = m[:seq_len]
                subtask_rows.append(m)
            batch["subtask_token_mask"] = torch.stack(subtask_rows, dim=0)

        if any("phase_token_mask" in ins for ins in instances):
            seq_len = input_ids.size(1)
            phase_rows = []
            for ins in instances:
                m = ins.get("phase_token_mask")
                if m is None:
                    m = torch.zeros_like(ins["input_ids"], dtype=torch.bool)
                m = m.squeeze(0).bool()
                if m.numel() < seq_len:
                    m = torch.nn.functional.pad(m, (0, seq_len - m.numel()), value=False)
                else:
                    m = m[:seq_len]
                phase_rows.append(m)
            batch["phase_token_mask"] = torch.stack(phase_rows, dim=0)

        if self.tree_mask:
            batch["attention_mask"] = self._build_tree_mask(instances, input_ids)


        images = [ins["pixel_values"] for ins in instances if "pixel_values" in ins]
        videos = [
            ins["pixel_values_videos"]
            for ins in instances
            if "pixel_values_videos" in ins
        ]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [ins["image_grid_thw"] for ins in instances if "image_grid_thw" in ins],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [
                    ins["video_grid_thw"]
                    for ins in instances
                    if "video_grid_thw" in ins
                ],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        # --- VLN-specific extras (optional) -------------------------------
        task_types = [ins.get("task_type", None) for ins in instances]
        if any(t is not None for t in task_types):
            batch["task_type"] = tuple(task_types)

        return batch

    def _build_tree_mask(self, instances, input_ids):
        """Build a 4D boolean prefix-tree attention mask ``[B, 1, L, L]`` from the
        per-token ``group_ids`` each episode sample carries (0 = observation trunk,
        t = the t-th assistant leaf). ``True`` = query i may attend to key j iff

            j <= i  AND  key j is not padding  AND
            (group[j] == 0  OR  group[j] == group[i])

        i.e. every token attends causally to the trunk, each answer additionally
        attends to its own span, and NO token attends to a *different* answer span
        (no GT-action leakage; the trunk never attends to any answer). The diagonal
        is forced ``True`` so a fully-masked (e.g. padding) row can't NaN in sdpa.

        Instances without ``group_ids`` (VQA co-training rows) default to an
        all-trunk group vector, which reduces this exact mask to plain causal — so
        a mixed VLN+VQA batch is handled by the same 4D mask with no special-case.
        transformers (4.57) passes a 4D ``attention_mask`` straight through to the
        sdpa attention as ``attn_mask``; a boolean mask is dtype-independent, so
        the collator needn't know the model's compute dtype.
        """
        B, L = input_ids.shape
        device = input_ids.device
        groups = []
        for ins in instances:
            g = ins.get("group_ids")
            if g is None:
                g = torch.zeros(ins["input_ids"].size(-1), dtype=torch.long)
            groups.append(g.reshape(-1))
        group = pad_sequence(groups, batch_first=True, padding_value=0)
        group = group[:, :L].to(device)  # [B, L]

        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        # group.unsqueeze(2) = [B,L,1] query-group i; group.unsqueeze(1) = [B,1,L]
        # key-group j -> same_or_trunk[b,i,j] = (key j is trunk) OR (same group).
        same_or_trunk = (group.unsqueeze(1) == 0) | (
            group.unsqueeze(1) == group.unsqueeze(2))  # [B, L, L]
        key_valid = input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(1)  # [B,1,L]
        allowed = causal.unsqueeze(0) & same_or_trunk & key_valid  # [B, L, L]
        allowed = allowed | torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
        return allowed.unsqueeze(1)  # [B, 1, L, L]


@dataclass
class VLNWaypointDataCollator:
    """Collate continuous-waypoint VLN samples, optionally mixed with VQA samples
    from ``LazySupervisedDataset``.

    VLN rows carry ``plan_token_mask`` + ``future_waypoints``; VQA rows do not.
    ``plan_token_mask`` is padded to ``[B, L]`` (VQA rows all False) and
    ``future_waypoints`` is stacked over VLN rows in batch order, so the model
    can gather the planning-token hiddens with the same row-major ordering.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_len = self.tokenizer.model_max_length

        input_ids = [ins["input_ids"].squeeze(0) for ins in instances]
        labels = [ins["labels"].squeeze(0) for ins in instances]
        position_ids_list = [ins["position_ids"] for ins in instances]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = _pad_and_cat_position_ids(position_ids_list)

        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
        position_ids = position_ids[:, :, :max_len]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # planning-token mask, right-padded to the batch length (VQA rows: False)
        seq_len = input_ids.size(1)
        mask_rows = []
        for ins in instances:
            m = ins.get("plan_token_mask", None)
            if m is None:
                m = torch.zeros((1, ins["input_ids"].size(1)), dtype=torch.bool)
            m = m.squeeze(0).bool()
            if m.numel() < seq_len:
                m = torch.nn.functional.pad(m, (0, seq_len - m.numel()), value=False)
            else:
                m = m[:seq_len]
            mask_rows.append(m)
        plan_token_mask = torch.stack(mask_rows, dim=0)  # [B, L]

        # Guard: truncation must not have clipped any VLN row's planning tokens.
        for ins, m in zip(instances, mask_rows):
            if "future_waypoints" in ins:
                expected = int(ins["plan_token_mask"].sum())
                if int(m.sum()) != expected:
                    raise ValueError(
                        "A VLN sample exceeded model_max_length and its planning "
                        "tokens were truncated. Increase --model_max_length or "
                        "reduce memory_size / image pixels."
                    )
        batch["plan_token_mask"] = plan_token_mask

        # <stop>-token mask, right-padded the same way (VQA rows: all False).
        stop_rows = []
        for ins in instances:
            m = ins.get("stop_token_mask", None)
            if m is None:
                m = torch.zeros((1, ins["input_ids"].size(1)), dtype=torch.bool)
            m = m.squeeze(0).bool()
            if m.numel() < seq_len:
                m = torch.nn.functional.pad(m, (0, seq_len - m.numel()), value=False)
            else:
                m = m[:seq_len]
            stop_rows.append(m)
        stop_token_mask = torch.stack(stop_rows, dim=0)  # [B, L]

        # Guard: the <stop> token (last answer token) must survive truncation on
        # every VLN row, else stop-head supervision silently drops.
        for ins, m in zip(instances, stop_rows):
            if "future_waypoints" in ins and int(m.sum()) != 1:
                raise ValueError(
                    "A VLN sample exceeded model_max_length and its <stop> token "
                    "was truncated. Increase --model_max_length or reduce "
                    "memory_size / image pixels."
                )
        batch["stop_token_mask"] = stop_token_mask

        # Waypoint labels, per-waypoint arrive flags, and the per-step regression
        # validity mask, gathered in the same VLN row order (VQA rows lack these
        # keys and are skipped). ``waypoint_step_valid`` masks the REGRESSION only:
        # padded steps hold a copy of the last real pose, so they are not targets.
        # ``arrive_labels`` is NOT masked — a padded step points at the endpoint,
        # so its flag is a genuine 1 and is exactly the all-positive pattern the
        # rollout's stop rule keys on.
        future, arrive, step_valid = [], [], []
        for ins in instances:
            if "future_waypoints" in ins:
                future.append(ins["future_waypoints"])
                arrive.append(ins["arrive_labels"])
                step_valid.append(ins["waypoint_step_valid"])
        if future:
            batch["future_waypoints"] = torch.stack(future, dim=0)       # [n,P,wp]
            batch["arrive_labels"] = torch.stack(arrive, dim=0)          # [n,P]
            batch["waypoint_step_valid"] = torch.stack(step_valid, dim=0)  # [n,P]
        else:
            batch["future_waypoints"] = None
            batch["arrive_labels"] = None
            batch["waypoint_step_valid"] = None

        # visual tensors (present on both VLN and VQA image/video samples)
        images = [ins["pixel_values"] for ins in instances if "pixel_values" in ins]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [ins["image_grid_thw"] for ins in instances if "image_grid_thw" in ins],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        videos = [
            ins["pixel_values_videos"]
            for ins in instances
            if "pixel_values_videos" in ins
        ]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [ins["video_grid_thw"] for ins in instances if "video_grid_thw" in ins],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        return batch
