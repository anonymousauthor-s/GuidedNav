"""Data path for Subtask-conditioned Object-Mask Alignment (SOMA).

SOMA deliberately keeps the policy input action-only.  When text supervision is
enabled, the same observation is tokenized a second time with a subtask-only
request.  Consequently, ground-truth subtask tokens are never causal ancestors
of an action target.  Object masks remain training metadata and never enter the
processor/model inputs.

The frozen ``scob_manifest_v1`` is reused so the new factorial consumes exactly
the same source rows, RGB files, masks, and scan-disjoint train/holdout split as
the preceding SCOB experiment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .data_processor import IGNORE_INDEX, update_processor_pixels
from .rope2d import get_rope_index_3
from .scob_dataset import (
    SCOB_QUERY_TOKENS,
    _sha256,
    action_only_labels,
    get_current_patch_positions,
    mask_union_to_patches,
    rgb_dependency_metadata,
    rgb_dependency_matches,
    rooted_path,
    sanitize_action_record as sanitize_scob_record,
)
from .vln_unified_dataset import (
    VLNActionDataCollator,
    _build_vln_messages,
    _pad_and_cat_position_ids,
)


SUBTASK_ONLY_SUFFIX = (
    " Identify the immediate navigation subtask for the current observation. "
    "Respond only with the short imperative subtask text; do not output actions."
)


def sanitize_action_record(record: dict) -> dict:
    """Return the validated, query-free action example used by every SOMA arm.

    The existing SCOB sanitizer is the single source of truth for removing the
    legacy ``SUBTASK: ...\nACTION: ...`` answer and validating ``action_text``.
    This wrapper then removes the five fixed SCOB queries it appended.  Failing
    closed here prevents a future change to that sanitizer from silently
    reintroducing auxiliary tokens into the clean control.
    """

    result = sanitize_scob_record(record)
    suffix = " " + " ".join(SCOB_QUERY_TOKENS)
    human = str(result["conversations"][0]["value"])
    if not human.endswith(suffix):
        raise ValueError("Validated SCOB record lacks the expected query suffix")
    human = human[: -len(suffix)].rstrip()
    if any(token in human for token in SCOB_QUERY_TOKENS):
        raise ValueError("Reserved SCOB query remains in the SOMA action prompt")
    result["conversations"] = [
        {"from": "human", "value": human},
        dict(result["conversations"][1]),
    ]
    return result


def subtask_only_record(action_record: dict, source_record: dict) -> dict:
    """Build an independent main-LM target with no action teacher forcing."""

    text = str(source_record.get("subtask", "")).strip()
    if not text or not bool(source_record.get("subtask_valid", True)):
        raise ValueError("SOMA text-supervision rows require a valid subtask")
    human = str(action_record["conversations"][0]["value"])
    result = dict(action_record)
    result["conversations"] = [
        {"from": "human", "value": human.rstrip() + SUBTASK_ONLY_SUFFIX},
        {"from": "gpt", "value": text},
    ]
    return result


def first_prediction_state_position(labels: torch.Tensor) -> int:
    """Position whose hidden state predicts the first supervised token."""

    flat = torch.as_tensor(labels).reshape(-1)
    supervised = torch.nonzero(flat.ne(IGNORE_INDEX), as_tuple=False).flatten()
    if supervised.numel() == 0 or int(supervised[0]) < 1:
        raise ValueError("A supervised answer must have a preceding causal state")
    return int(supervised[0]) - 1


class SOMADataset(Dataset):
    """Lazy, manifest-pinned action/mask dataset with an optional text branch."""

    def __init__(
        self,
        processor,
        data_args,
        *,
        manifest_path,
        split: str = "train",
        load_masks: bool = True,
        load_subtasks: bool = False,
    ):
        if split not in ("train", "holdout", "all"):
            raise ValueError(f"Unknown SOMA split {split}")
        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "scob_manifest_v1":
            raise ValueError("SOMA currently requires the frozen scob_manifest_v1")

        self.samples_path = self.manifest["source"]["path"]
        stat = Path(self.samples_path).stat()
        expected_source = self.manifest["source"]
        if (
            stat.st_size != expected_source["size_bytes"]
            or stat.st_mtime_ns != expected_source["mtime_ns"]
        ):
            raise ValueError("Source size/mtime changed after manifest freeze")

        self.video_folder = self.manifest["rgb_root"]
        self.mask_root = self.manifest["mask_root"]
        dependency = self.manifest.get("rgb_dependency")
        if dependency is None or not rgb_dependency_matches(
            rgb_dependency_metadata(Path(self.video_folder)), dependency
        ):
            raise ValueError("RGB dependency differs from the frozen manifest")

        arrays = {}
        for name in ("offsets", "train_indices", "holdout_indices"):
            entry = self.manifest["arrays"][name]
            path = manifest_path.parent / entry["file"]
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"SOMA {name} index hash mismatch")
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        self.offsets = arrays["offsets"]
        self.source_indices = (
            np.arange(len(self.offsets), dtype=np.int64)
            if split == "all"
            else arrays[split + "_indices"]
        )

        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.load_masks = bool(load_masks)
        self.load_subtasks = bool(load_subtasks)

        audit = self.manifest.get("audit", {})
        if audit.get("all_rows_conservative_sequence_audit"):
            image_processor = self.processor.image_processor
            patch_size = int(getattr(image_processor, "patch_size", 14))
            max_pixels = int(getattr(data_args, "max_pixels", 28 * 28 * 256))
            visual_bound = math.ceil(
                max_pixels / (patch_size * self.merge_size) ** 2
            ) + 4
            if visual_bound > audit["image_token_upper_bound"]:
                raise ValueError("Processor geometry exceeds the frozen sequence audit")
            if int(self.tokenizer.model_max_length) < audit["largest_sequence_bound"]:
                raise ValueError("Tokenizer maximum is below the frozen sequence bound")

    def __len__(self):
        return len(self.source_indices)

    def source_index_for_sample(self, index: int) -> int:
        return int(self.source_indices[index])

    def _read_record(self, index: int) -> dict:
        with open(self.samples_path, "rb") as handle:
            handle.seek(int(self.offsets[self.source_index_for_sample(index)]))
            return json.loads(handle.readline())

    def _encode(self, record: dict, images: list[Image.Image], n_memory: int):
        messages = _build_vln_messages(
            record["conversations"], images[n_memory:], images[:n_memory]
        )
        encoded = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long).reshape(1, -1)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            raise ValueError("SOMA refuses silent sequence truncation")
        grid = torch.as_tensor(encoded["image_grid_thw"], dtype=torch.long)
        if len(grid) != len(images) or "video_grid_thw" in encoded:
            raise ValueError("Unexpected SOMA image/video layout")
        positions, _ = get_rope_index_3(
            self.merge_size, input_ids, image_grid_thw=grid
        )
        answer = str(record["conversations"][1]["value"])
        labels = action_only_labels(input_ids, self.tokenizer, answer)
        return encoded, input_ids, labels, positions, grid

    def __getitem__(self, index: int):
        raw = self._read_record(index)
        action_record = sanitize_action_record(raw)
        n_memory = int(action_record["n_memory"])
        if n_memory != len(action_record["images"]) - 1:
            raise ValueError("SOMA requires exactly one current RGB after memory")

        images = []
        for relative in action_record["images"]:
            with Image.open(
                rooted_path(Path(self.video_folder), relative, allow_images_symlink=True)
            ) as image:
                images.append(image.convert("RGB"))

        encoded, input_ids, labels, positions, grid = self._encode(
            action_record, images, n_memory
        )
        current_positions = get_current_patch_positions(
            input_ids, grid, self.image_token_id, self.merge_size
        )
        action_state_position = first_prediction_state_position(labels)
        if int(current_positions[-1]) >= action_state_position:
            raise ValueError("Current visual patches must precede the action state")

        if self.load_masks:
            targets, mask_valid = mask_union_to_patches(
                raw,
                self.mask_root,
                images[-1].size,
                grid[-1],
                self.processor.image_processor,
            )
        else:
            targets = torch.zeros(len(current_positions), dtype=torch.float32)
            mask_valid = False

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": positions,
            "attention_mask": [input_ids.shape[1]],
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": grid,
            "current_patch_positions": current_positions,
            "action_state_position": torch.tensor(action_state_position),
            "mask_targets": targets,
            "mask_valid": torch.tensor(mask_valid),
            "source_indices": torch.tensor(self.source_index_for_sample(index)),
        }

        if self.load_subtasks:
            text_record = subtask_only_record(action_record, raw)
            text_encoded, text_ids, text_labels, text_positions, text_grid = self._encode(
                text_record, images, n_memory
            )
            if not torch.equal(grid, text_grid):
                raise ValueError("Action and subtask branches produced different image grids")
            action_image_count = int(input_ids.eq(self.image_token_id).sum())
            text_image_count = int(text_ids.eq(self.image_token_id).sum())
            if action_image_count != text_image_count:
                raise ValueError("Action and subtask branches contain different visual token counts")
            # The second processor call is used only to obtain its independently
            # tokenized causal sequence.  Both branches reuse the first call's
            # identical RGB tensor during their separate model forwards.
            if not torch.equal(text_encoded["pixel_values"], encoded["pixel_values"]):
                raise ValueError("Action and subtask branches produced different RGB tensors")
            item.update(
                {
                    "soma_subtask_input_ids": text_ids,
                    "soma_subtask_labels": text_labels,
                    "soma_subtask_position_ids": text_positions,
                }
            )
        return item


class SOMACollator(VLNActionDataCollator):
    """Collate the policy sequence plus optional independent subtask sequence."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, tree_mask=False)

    def __call__(self, instances):
        if any(item["input_ids"].shape[-1] > self.tokenizer.model_max_length for item in instances):
            raise ValueError("SOMA collator refuses silent action truncation")
        batch = super().__call__(instances)
        for name in ("action_state_position", "mask_valid", "source_indices"):
            batch[name] = torch.stack([item[name] for item in instances])
        batch["current_patch_positions"] = pad_sequence(
            [item["current_patch_positions"] for item in instances],
            batch_first=True,
            padding_value=-1,
        )
        batch["mask_targets"] = pad_sequence(
            [item["mask_targets"] for item in instances],
            batch_first=True,
            padding_value=0.0,
        )

        has_text = ["soma_subtask_input_ids" in item for item in instances]
        if any(has_text):
            if not all(has_text):
                raise ValueError("A SOMA batch cannot mix configured/unconfigured text rows")
            ids = [item["soma_subtask_input_ids"].squeeze(0) for item in instances]
            labels = [item["soma_subtask_labels"].squeeze(0) for item in instances]
            lengths = torch.tensor([row.numel() for row in ids], dtype=torch.long)
            max_length = int(lengths.max())
            if max_length > self.tokenizer.model_max_length:
                raise ValueError("SOMA collator refuses silent subtask truncation")
            batch["soma_subtask_input_ids"] = pad_sequence(
                ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["soma_subtask_labels"] = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            batch["soma_subtask_position_ids"] = _pad_and_cat_position_ids(
                [item["soma_subtask_position_ids"] for item in instances]
            )
            batch["soma_subtask_attention_mask"] = (
                torch.arange(max_length)[None, :] < lengths[:, None]
            )
        return batch
