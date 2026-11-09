"""Isolated action inputs and training-only SCOB text/mask targets.

The index manifest is generated once by ops/prepare_scob_manifest.py. No rank
rescans or silently filters the source. Masks never enter the processor, chat,
or model input embeddings. Queries are always present in all four arms.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .data_processor import IGNORE_INDEX, update_processor_pixels
from .rope2d import get_rope_index_3
from .vln_unified_dataset import (
    VLNActionDataCollator,
    _build_vln_messages,
)

SUB_QUERY_TOKENS = tuple(f"<SUB_Q_{i}>" for i in range(4))
OBJ_QUERY_TOKEN = "<OBJ_Q>"
SCOB_QUERY_TOKENS = SUB_QUERY_TOKENS + (OBJ_QUERY_TOKEN,)
LEGACY_SUBTASK_SUFFIX = (
    " First identify the immediate navigation subtask. Respond in exactly two "
    "lines: 'SUBTASK: <short imperative navigation instruction>' followed by "
    "'ACTION: <sequence>', where ACTION uses only TURN LEFT (←), TURN RIGHT (→), "
    "MOVE FORWARD (↑), or STOP."
)


def append_scob_queries(text: str) -> str:
    if any(token in text for token in SCOB_QUERY_TOKENS):
        raise ValueError("Source already contains a reserved SCOB query")
    return text.rstrip() + " " + " ".join(SCOB_QUERY_TOKENS)


def sanitize_action_record(record: dict) -> dict:
    """Remove only the known output instruction and all GT subtask answer text.

    Unknown prompt templates fail closed; a generic regex must not accidentally
    remove part of the real route instruction. Kept metadata is never forwarded
    to the main model by SCOBDataset.
    """
    result = dict(record)
    turns = record.get("conversations", [])
    if len(turns) != 2 or [t.get("from") for t in turns] != ["human", "gpt"]:
        raise ValueError("SCOB source must contain exactly one human/assistant pair")
    human = str(turns[0]["value"])
    if human.endswith(LEGACY_SUBTASK_SUFFIX):
        human = human[: -len(LEGACY_SUBTASK_SUFFIX)]
    elif "First identify the immediate navigation subtask" in human or "SUBTASK:" in human:
        raise ValueError("Unrecognized subtask-output prompt template")
    original = str(turns[1]["value"]).strip()
    match = re.fullmatch(r"SUBTASK:\s*(.*?)\s*\nACTION:\s*(.+)", original, flags=re.DOTALL)
    if match:
        if match.group(1).strip() != str(record.get("subtask", "")).strip():
            raise ValueError("Subtask field and legacy assistant target disagree")
        extracted_action = match.group(2).strip()
    else:
        extracted_action = original
        if "SUBTASK:" in original or "ACTION:" in original:
            raise ValueError("Unrecognized assistant action/subtask template")
    action = str(record.get("action_text", extracted_action)).strip()
    if not action or action != extracted_action:
        raise ValueError("action_text and legacy ACTION target disagree")
    if re.sub(r"(?:STOP|[←→↑\s])", "", action):
        raise ValueError(f"Action contains non-policy tokens: {action!r}")
    result["conversations"] = [
        {"from": "human", "value": append_scob_queries(human)},
        {"from": "gpt", "value": action},
    ]
    return result


@lru_cache(maxsize=64)
def _resolved_root(root: str) -> Path:
    return Path(root).resolve(strict=True)


@lru_cache(maxsize=32768)
def _resolved_directory(directory: str) -> Path:
    return Path(directory).resolve(strict=True)


def rgb_dependency_metadata(root: Path) -> dict:
    """Describe the one explicitly permitted external dependency: RGB/images."""
    root = Path(root).resolve(strict=True)
    images = root / "images"
    return {
        "logical_images_root": str(images),
        "images_is_symlink": images.is_symlink(),
        "resolved_images_root": str(images.resolve(strict=True)),
    }


def rgb_dependency_matches(observed: dict, expected: dict) -> bool:
    """Compare the explicit image roots recorded by the local manifest."""
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    if observed.get("images_is_symlink") != expected.get("images_is_symlink"):
        return False
    return all(
        str(observed.get(name, "")) == str(expected.get(name, ""))
        for name in ("logical_images_root", "resolved_images_root")
    )


@lru_cache(maxsize=65536)
def _rooted_path_cached(root_string: str, relative: str, allow_images_symlink: bool) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
        raise ValueError(f"Expected root-relative path, got {relative!r}")
    root = _resolved_root(root_string)
    boundary = root
    candidate = root / relative_path
    if allow_images_symlink and relative_path.parts[0] == "images":
        # R2R_subtask/images is an existing handoff dependency pointing to the
        # shared R2R_images tree. No other out-of-root directory is authorized.
        boundary = _resolved_directory(str(root / "images"))
        candidate = boundary.joinpath(*relative_path.parts[1:])
    # Resolve shared parent directories once instead of walking every component
    # for each repeated history frame. File symlinks are still resolved
    # and checked; caches are safe because this experiment freezes its inputs.
    candidate = _resolved_directory(str(candidate.parent)) / candidate.name
    if candidate.is_symlink():
        candidate = candidate.resolve(strict=True)
    if not candidate.is_relative_to(boundary):
        raise ValueError(f"Path resolves outside its authorized dependency root: {relative!r}")
    return candidate


def rooted_path(root: Path, relative: str, *, allow_images_symlink: bool = False) -> Path:
    return _rooted_path_cached(str(root), str(relative), bool(allow_images_symlink))


def active_mask_entries(record: dict) -> list[dict]:
    supervision = record.get("object_supervision")
    if not isinstance(supervision, dict):
        raise ValueError("Missing object_supervision in V5-Fast sample")
    eligible = supervision.get("mask_loss_eligible")
    if not isinstance(eligible, bool):
        raise ValueError("mask_loss_eligible must be an explicit boolean")
    if not eligible:
        return []
    active = supervision.get("active_masks", [])
    if not active:
        raise ValueError("eligible=true without active masks")
    result = []
    for entry in active:
        if entry.get("visible") is False or entry.get("visibility") in (False, "invisible", "not_visible"):
            continue
        if "frame_index" in entry and int(entry["frame_index"]) != int(record["c"]):
            raise ValueError("Active mask belongs to another current frame")
        if "source_phase_index" in entry and int(entry["source_phase_index"]) != int(record["subtask_phase_index"]):
            raise ValueError("Active mask belongs to another source phase")
        # The export filename contains the dense phase, whereas the source phase
        # can have gaps. Only frame identity can be compared without its mapping.
        frame_match = re.search(r"_p\d+_t\d+_(\d+)\.png$", str(entry.get("mask", "")))
        if frame_match and int(frame_match.group(1)) != int(record["c"]):
            raise ValueError("Active mask filename belongs to another frame")
        if not entry.get("mask"):
            raise ValueError("Active mask lacks its PNG path")
        result.append(entry)
    return result


def mask_union_to_patches(record, mask_root, image_size_wh, image_grid_thw, image_processor):
    """Resize the binary union using Qwen's observed geometry, then area pool.

    Supports Qwen's resize-only processor. Crop/pad processing is rejected until
    its geometry is explicitly implemented rather than silently misaligning GT.
    """
    t, grid_h, grid_w = map(int, image_grid_thw)
    merge = int(getattr(image_processor, "merge_size", 2))
    patch = int(getattr(image_processor, "patch_size", 14))
    if t != 1 or grid_h % merge or grid_w % merge:
        raise ValueError("Current RGB must have one frame and a merge-divisible grid")
    if getattr(image_processor, "do_center_crop", False) or getattr(image_processor, "do_pad", False):
        raise ValueError("SCOB mask alignment currently supports resize-only Qwen processors")
    shape = (grid_h // merge, grid_w // merge)
    entries = active_mask_entries(record)
    if not entries:
        return torch.zeros(shape[0] * shape[1], dtype=torch.float32), False
    width, height = map(int, image_size_wh)
    union = np.zeros((height, width), dtype=np.bool_)
    for entry in entries:
        with Image.open(rooted_path(Path(mask_root), entry["mask"])) as image:
            if image.size != (width, height):
                raise ValueError("Mask dimensions differ from original current RGB")
            pixels = np.asarray(image.convert("L"))
            if not np.all((pixels == 0) | (pixels == 255)) or not np.any(pixels):
                raise ValueError("Accepted pseudo mask must be nonempty binary 0/255")
            union |= pixels != 0
    tensor = torch.from_numpy(union.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(grid_h * patch, grid_w * patch), mode="area")
    occupancy = F.avg_pool2d(resized, kernel_size=patch * merge, stride=patch * merge)
    if tuple(occupancy.shape[-2:]) != shape:
        raise AssertionError("Mask target and post-merge patch geometry disagree")
    return occupancy.flatten(), True


def get_current_patch_positions(input_ids, image_grid_thw, image_token_id, merge_size):
    ids = input_ids.reshape(-1)
    grids = torch.as_tensor(image_grid_thw, dtype=torch.long).reshape(-1, 3)
    if grids.numel() == 0 or int(grids[-1, 0]) != 1:
        raise ValueError("Missing current RGB image grid")
    if torch.any(grids[:, 1:] % int(merge_size)):
        raise ValueError("Image grids are not divisible by spatial merge")
    counts = grids.prod(dim=-1) // int(merge_size) ** 2
    positions = torch.nonzero(ids.eq(int(image_token_id)), as_tuple=False).flatten()
    if positions.numel() != int(counts.sum()):
        raise ValueError("Image token count disagrees with image_grid_thw")
    current = positions[-int(counts[-1]) :]
    if current.numel() > 1 and not torch.all(current[1:] - current[:-1] == 1):
        raise ValueError("Current image patch tokens are not contiguous")
    return current


def action_only_labels(input_ids, tokenizer, action_text):
    """Supervise one complete action answer and EOS, using the full role marker."""
    ids = input_ids.reshape(-1).tolist()
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    starts = [i + len(header) for i in range(len(ids) - len(header) + 1) if ids[i : i + len(header)] == header]
    if not header or len(starts) != 1:
        raise ValueError("Expected exactly one complete assistant role marker")
    start = starts[0]
    eos = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    try:
        end = ids.index(eos, start)
    except ValueError as exc:
        raise ValueError("Action answer lacks a complete end-of-turn token") from exc
    decoded = tokenizer.decode(ids[start:end], skip_special_tokens=False)
    if decoded.strip() != action_text.strip():
        raise ValueError("Main assistant target contains content other than the action")
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels.reshape(-1)[start : end + 1] = input_ids.reshape(-1)[start : end + 1]
    return labels


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SCOBDataset(Dataset):
    def __init__(self, processor, data_args, *, manifest_path, split="train", text_tokenizer=None, load_masks=True):
        if split not in ("train", "holdout", "all"):
            raise ValueError(f"Unknown SCOB split {split}")
        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "scob_manifest_v1":
            raise ValueError("Unrecognized SCOB manifest version")
        self.samples_path = self.manifest["source"]["path"]
        stat = Path(self.samples_path).stat()
        if stat.st_size != self.manifest["source"]["size_bytes"] or stat.st_mtime_ns != self.manifest["source"]["mtime_ns"]:
            raise ValueError("Source size/mtime changed after manifest freeze; regenerate audit")
        self.video_folder = self.manifest["rgb_root"]
        self.mask_root = self.manifest["mask_root"]
        dependency = self.manifest.get("rgb_dependency")
        if dependency is None or not rgb_dependency_matches(
            rgb_dependency_metadata(Path(self.video_folder)), dependency
        ):
            raise ValueError("RGB images dependency differs from frozen manifest; rebuild the audit")
        arrays = {}
        for name in ("offsets", "train_indices", "holdout_indices"):
            entry = self.manifest["arrays"][name]
            path = manifest_path.parent / entry["file"]
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"SCOB {name} index hash mismatch")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        self.offsets = arrays["offsets"]
        self.source_row_count = len(self.offsets)
        self.source_indices = np.arange(len(self.offsets), dtype=np.int64) if split == "all" else arrays[split + "_indices"]
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.text_tokenizer = text_tokenizer
        self.merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        audit = self.manifest.get("audit", {})
        if audit.get("all_rows_conservative_sequence_audit"):
            ip = self.processor.image_processor
            patch_size = int(getattr(ip, "patch_size", 14))
            max_pixels = int(getattr(data_args, "max_pixels", 28 * 28 * 256))
            # Four extra merged patches make the manifest bound conservative
            # across rounding implementations. The actual item length is still
            # checked below, so processor/template changes can never truncate.
            required_visual_bound = math.ceil(max_pixels / (patch_size * self.merge_size) ** 2) + 4
            if required_visual_bound > audit["image_token_upper_bound"]:
                raise ValueError("Processor pixels/patch geometry exceeds frozen sequence audit")
            if int(self.tokenizer.model_max_length) < audit["largest_sequence_bound"]:
                raise ValueError("Tokenizer maximum is smaller than frozen sequence bound")
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.query_ids = [int(self.tokenizer.convert_tokens_to_ids(t)) for t in SCOB_QUERY_TOKENS]
        for token, token_id in zip(SCOB_QUERY_TOKENS, self.query_ids):
            if self.tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Query token is not registered as a single token: {token}")
        self.load_masks = bool(load_masks)

    def __len__(self):
        return len(self.source_indices)

    def source_index_for_sample(self, index):
        return int(self.source_indices[index])

    def _read_record(self, index):
        with open(self.samples_path, "rb") as handle:
            handle.seek(int(self.offsets[self.source_index_for_sample(index)]))
            return json.loads(handle.readline())

    def __getitem__(self, index):
        raw = self._read_record(index)
        record = sanitize_action_record(raw)
        n_memory = int(record["n_memory"])
        if n_memory != len(record["images"]) - 1:
            raise ValueError("SCOB requires exactly one current RGB after memory")
        images = []
        for relative in record["images"]:
            with Image.open(rooted_path(Path(self.video_folder), relative, allow_images_symlink=True)) as image:
                images.append(image.convert("RGB"))
        messages = _build_vln_messages(record["conversations"], images[n_memory:], images[:n_memory])
        encoded = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long).reshape(1, -1)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            raise ValueError(f"Source row {self.source_index_for_sample(index)} exceeds max sequence length; no silent truncation")
        grid = encoded["image_grid_thw"]
        if len(grid) != len(images) or "video_grid_thw" in encoded:
            raise ValueError("Unexpected image/video layout")
        positions, _ = get_rope_index_3(self.merge_size, input_ids, image_grid_thw=grid)
        labels = action_only_labels(input_ids, self.tokenizer, record["conversations"][1]["value"])
        supervised = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
        if not supervised.numel():
            raise ValueError("No action labels after chat tokenization")
        queries = []
        for token_id in self.query_ids:
            found = torch.nonzero(input_ids[0].eq(token_id), as_tuple=False).flatten()
            if len(found) != 1 or int(found[0]) >= int(supervised[0]):
                raise ValueError("Each query must occur exactly once before all assistant targets")
            queries.append(int(found[0]))
        if queries != sorted(queries):
            raise ValueError("Unexpected SUB/OBJ query ordering")
        current_positions = get_current_patch_positions(input_ids, grid, self.image_token_id, self.merge_size)
        if int(current_positions[-1]) >= queries[0]:
            raise ValueError("SCOB queries must follow the current visual input")
        if self.load_masks:
            targets, mask_valid = mask_union_to_patches(raw, self.mask_root, images[-1].size, grid[-1], self.processor.image_processor)
        else:
            targets = torch.zeros(len(current_positions), dtype=torch.float32)
            mask_valid = False
        item = {
            "input_ids": input_ids, "labels": labels, "position_ids": positions,
            "attention_mask": [input_ids.shape[1]], "pixel_values": encoded["pixel_values"],
            "image_grid_thw": grid, "sub_query_positions": torch.tensor(queries[:4]),
            "obj_query_positions": torch.tensor(queries[4]), "current_patch_positions": current_positions,
            "mask_targets": targets, "mask_valid": torch.tensor(mask_valid),
            "source_indices": torch.tensor(self.source_index_for_sample(index)),
        }
        if self.text_tokenizer is not None:
            text = str(raw.get("subtask", "")).strip()
            if not text or not bool(raw.get("subtask_valid", True)):
                raise ValueError("Approved R2R text-supervision rows must have a valid subtask")
            ids = self.text_tokenizer.encode(text, add_special_tokens=False)
            eos = self.text_tokenizer.eos_token_id
            if eos is None:
                raise ValueError("Text decoder requires an EOS token")
            ids = ids + [int(eos)]
            if len(ids) > 512:
                raise ValueError("Subtask exceeds text decoder audit limit (512); no truncation")
            item["text_input_ids"] = torch.tensor(ids, dtype=torch.long)
        return item


class SCOBCollator(VLNActionDataCollator):
    def __init__(self, tokenizer, text_tokenizer=None):
        super().__init__(tokenizer=tokenizer, tree_mask=False)
        self.text_tokenizer = text_tokenizer

    def __call__(self, instances):
        if any(item["input_ids"].shape[-1] > self.tokenizer.model_max_length for item in instances):
            raise ValueError("SCOB collator refuses silent sequence truncation")
        batch = super().__call__(instances)
        for name in ("sub_query_positions", "obj_query_positions", "mask_valid", "source_indices"):
            batch[name] = torch.stack([item[name] for item in instances])
        batch["current_patch_positions"] = pad_sequence([item["current_patch_positions"] for item in instances], batch_first=True, padding_value=-1)
        batch["mask_targets"] = pad_sequence([item["mask_targets"] for item in instances], batch_first=True, padding_value=0.0)
        has_text = ["text_input_ids" in item for item in instances]
        if any(has_text):
            if not all(has_text) or self.text_tokenizer is None:
                raise ValueError("Mixed or unconfigured text decoder targets")
            pad_id = self.text_tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.text_tokenizer.eos_token_id
            batch["text_input_ids"] = pad_sequence([item["text_input_ids"] for item in instances], batch_first=True, padding_value=pad_id)
            lengths = torch.tensor([len(item["text_input_ids"]) for item in instances])
            valid = torch.arange(batch["text_input_ids"].shape[1])[None] < lengths[:, None]
            batch["text_attention_mask"] = valid.long()
            batch["text_labels"] = batch["text_input_ids"].masked_fill(~valid, IGNORE_INDEX)
        return batch
