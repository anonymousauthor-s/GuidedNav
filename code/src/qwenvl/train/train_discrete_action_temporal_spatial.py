"""LoRA training for temporal Spatial-Aware VLN experiments.

The legacy E3/E4 weighted-merge settings remain supported.  E5/E6 use an
identity-preserving gated residual on the current token grid, retain the
original joint eight-frame Spatial Forcing loss, and optionally distill the
local temporal increment ``VGGT(local window)[current] - VGGT(current)``.
"""

import gc
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data import data_list
from qwenvl.data.vln_unified_dataset import VLNActionDataCollator, VLNUnifiedDataset
from qwenvl.temporal_token_merge import (
    TemporalMergeConfig,
    TemporalResidualGate,
    parse_temporal_decay,
    register_temporal_merge_hook,
)
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwenvl.train.train_discrete_action_vggt_patch import (
    AlignProjector,
    _hidden_size,
    _preprocess_vggt_pil_images,
    set_full_finetune_modules,
)


local_rank = -1


def rank0_print(*args, **kwargs):
    if local_rank in (-1, 0):
        print(*args, **kwargs)


@dataclass
class TemporalSpatialArguments:
    current_merge_enable: bool = field(default=True)
    current_window_size: int = field(default=4)
    current_merge_ratio: float = field(default=0.5)
    current_temporal_decay: str = field(default="0.5,0.7,0.9")
    current_anchor_weight: float = field(default=1.0)
    current_spatial_bias: float = field(default=0.05)
    current_spatial_weight: float = field(default=0.5)
    past_spatial_weight: float = field(default=0.0)
    sf8_spatial_weight: float = field(default=0.0)
    delta_spatial_weight: float = field(default=0.0)
    current_fusion_mode: str = field(default="weighted_mean")
    current_matching_mode: str = field(default="source_best")
    current_gate_init: float = field(default=0.05)
    current_gate_max: float = field(default=0.2)
    spatial_student_layer: int = field(default=27)
    spatial_teacher_layer: int = field(default=-1)
    spatial_use_vggt_pe: bool = field(default=True)
    spatial_pe_scale: float = field(default=0.1)
    spatial_projector_lr: float = field(default=0.0)
    vggt_checkpoint_path: str = field(
        default="assets/vggt/vggt.pt"
    )
    vggt_source_path: str = field(
        default="third_party/vggt"
    )
    vggt_preprocess_mode: str = field(default="crop")

    def merge_config(self) -> TemporalMergeConfig:
        # The CLI lists weights from oldest to newest (t-3,t-2,t-1).  The
        # merger indexes them by age, so reverse to age-1 order internally.
        listed = parse_temporal_decay(self.current_temporal_decay)
        if len(listed) != self.current_window_size - 1:
            raise ValueError(
                "current_temporal_decay must have current_window_size-1 entries; "
                f"got {listed} for window {self.current_window_size}."
            )
        return TemporalMergeConfig(
            merge_ratio=self.current_merge_ratio,
            spatial_bias=self.current_spatial_bias,
            anchor_weight=self.current_anchor_weight,
            temporal_decay=tuple(reversed(listed)),
            fusion_mode=self.current_fusion_mode,
            matching_mode=self.current_matching_mode,
        )


class TemporalSpatialVLNDataset(VLNUnifiedDataset):
    def __init__(self, *args, spatial_args=None, **kwargs):
        self.spatial_args = spatial_args or TemporalSpatialArguments()
        self._base_images: Optional[List[Image.Image]] = None
        self._base_paths: Optional[List[str]] = None
        super().__init__(*args, **kwargs)

    def _load_images(self, rel_paths: List[str]) -> List[Image.Image]:
        images = super()._load_images(rel_paths)
        self._base_images = images
        self._base_paths = list(rel_paths)
        return images

    def _load_current_auxiliary(self) -> Tuple[List[Image.Image], List[int]]:
        if not self._base_paths:
            raise RuntimeError("Current image path was not captured.")
        current = self._base_paths[-1]
        directory, filename = os.path.split(current)
        stem, suffix = os.path.splitext(filename)
        try:
            frame_number = int(stem)
        except ValueError as exc:
            raise ValueError(
                f"Current-frame window requires a numeric frame name, got {current}."
            ) from exc

        images: List[Image.Image] = []
        ages: List[int] = []
        # Chronological source order: t-3, t-2, t-1.
        for age in range(self.spatial_args.current_window_size - 1, 0, -1):
            number = frame_number - age
            if number < 1:
                continue
            candidate = os.path.join(directory, f"{number:0{len(stem)}d}{suffix}")
            absolute = os.path.join(self.video_folder, candidate)
            if not os.path.isfile(absolute):
                continue
            with Image.open(absolute) as image:
                images.append(image.convert("RGB"))
            ages.append(age)
        return images, ages

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if not self._base_images or not self._base_paths:
            raise RuntimeError("VLN images were not captured for temporal processing.")

        auxiliary, ages = self._load_current_auxiliary()
        if self.spatial_args.current_merge_enable and auxiliary:
            encoded = self.processor.image_processor(auxiliary, return_tensors="pt")
            item["current_aux_pixel_values"] = encoded["pixel_values"]
            item["current_aux_grid_thw"] = encoded["image_grid_thw"]
            item["current_aux_ages"] = torch.tensor(ages, dtype=torch.long)

        current_weight = float(self.spatial_args.current_spatial_weight)
        delta_weight = float(self.spatial_args.delta_spatial_weight)
        if current_weight > 0.0 or delta_weight > 0.0:
            current_window = auxiliary + [self._base_images[-1]]
            item["current_vggt_images"] = _preprocess_vggt_pil_images(
                current_window, self.spatial_args.vggt_preprocess_mode
            )
            item["current_vggt_count"] = len(current_window)

        past_weight = float(self.spatial_args.past_spatial_weight)
        past_images = self._base_images[:-1]
        if past_weight > 0.0 and past_images:
            item["past_vggt_images"] = _preprocess_vggt_pil_images(
                past_images, self.spatial_args.vggt_preprocess_mode
            )
            item["past_vggt_count"] = len(past_images)

        if float(self.spatial_args.sf8_spatial_weight) > 0.0:
            item["sf8_vggt_images"] = _preprocess_vggt_pil_images(
                self._base_images, self.spatial_args.vggt_preprocess_mode
            )
            item["sf8_vggt_count"] = len(self._base_images)

        self._base_images = None
        self._base_paths = None
        return item


class TemporalSpatialCollator(VLNActionDataCollator):
    def __call__(self, instances):
        if len(instances) != 1:
            raise ValueError(
                "Temporal Spatial Forcing requires --per_device_train_batch_size 1."
            )
        batch = super().__call__(instances)
        item = instances[0]
        for name in (
            "current_aux_pixel_values",
            "current_aux_grid_thw",
            "current_aux_ages",
        ):
            if name in item:
                batch[name] = item[name]
        for prefix in ("current", "past", "sf8"):
            image_name = f"{prefix}_vggt_images"
            count_name = f"{prefix}_vggt_count"
            if image_name in item:
                batch[image_name] = item[image_name].unsqueeze(0)
                batch[count_name] = torch.tensor([item[count_name]], dtype=torch.long)
        return batch


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _get_trainable_module(model, name: str):
    model = _unwrap_model(model)
    module = getattr(model, name, None)
    if module is None and hasattr(model, "base_model"):
        module = getattr(model.base_model, name, None)
    if module is None:
        raise AttributeError(f"{name} is missing from the trainable model.")
    return module


def _get_projector(model, name: str) -> AlignProjector:
    return _get_trainable_module(model, name)


def _image_token_id(model) -> int:
    config = _unwrap_model(model).config
    value = getattr(config, "image_token_id", None)
    if value is None and hasattr(config, "vision_config"):
        value = getattr(config.vision_config, "image_token_id", None)
    if value is None:
        raise ValueError("Cannot resolve Qwen image_token_id.")
    return int(value)


class TemporalSpatialTrainer(Trainer):
    PROJECTOR_FILE = "temporal_spatial_projectors.pt"

    def __init__(self, *args, spatial_args=None, vggt_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_args = spatial_args or TemporalSpatialArguments()
        self.vggt_teacher = vggt_teacher
        self.merge_config = self.spatial_args.merge_config()
        self._metric_sums: Dict[str, float] = {}
        self._metric_n = 0
        self._printed_shapes = False

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        projector_lr = float(self.spatial_args.spatial_projector_lr)
        if projector_lr <= 0.0:
            return optimizer
        projector_ids = set()
        for name in (
            "current_sf_projector",
            "past_sf_projector",
            "sf8_sf_projector",
            "delta_sf_projector",
        ):
            try:
                projector_ids.update(id(p) for p in _get_projector(self.model, name).parameters())
            except AttributeError:
                pass
        groups = []
        count = 0
        for group in optimizer.param_groups:
            selected = [p for p in group["params"] if id(p) in projector_ids]
            remaining = [p for p in group["params"] if id(p) not in projector_ids]
            if remaining:
                new_group = dict(group)
                new_group["params"] = remaining
                groups.append(new_group)
            if selected:
                new_group = dict(group)
                new_group["params"] = selected
                new_group["lr"] = projector_lr
                new_group["initial_lr"] = projector_lr
                groups.append(new_group)
                count += sum(p.numel() for p in selected)
        optimizer.param_groups[:] = groups
        rank0_print(f"Spatial projector optimizer group: lr={projector_lr} params={count:,}")
        return optimizer

    def _projector_state(self):
        state = {}
        for name in (
            "current_sf_projector",
            "past_sf_projector",
            "sf8_sf_projector",
            "delta_sf_projector",
            "temporal_residual_gate",
        ):
            try:
                projector = _get_trainable_module(self.model, name)
            except AttributeError:
                continue
            state[name] = {k: v.detach().cpu() for k, v in projector.state_dict().items()}
        return state

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_dir = output_dir or self.args.output_dir
        if self.args.should_save:
            torch.save(self._projector_state(), os.path.join(output_dir, self.PROJECTOR_FILE))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        path = os.path.join(resume_from_checkpoint, self.PROJECTOR_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing temporal spatial projectors: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        for name, values in state.items():
            _get_trainable_module(self.model, name).load_state_dict(values, strict=True)
        return result

    def _teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.vggt_teacher is None:
            raise RuntimeError("Spatial alignment is enabled but VGGT is not loaded.")
        self.vggt_teacher.eval()
        dtype = next(self.vggt_teacher.parameters()).dtype
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            features, patch_start = self.vggt_teacher.aggregator(images.to(dtype=dtype))
            layer = int(self.spatial_args.spatial_teacher_layer)
            target = features[layer]
            if target is None:
                cached = [i for i, value in enumerate(features) if value is not None]
                raise ValueError(f"VGGT layer {layer} is not cached; available={cached}.")
            return target[:, :, patch_start:, :]

    def _student_image_parts(
        self,
        student: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        model,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        if student.shape[0] != 1:
            raise ValueError("Temporal Spatial Forcing requires batch size 1.")
        mask = input_ids[0].eq(_image_token_id(model)) & attention_mask[0].bool()
        patches = student[0, mask]
        merge_size = int(getattr(self.train_dataset, "merge_size", 2))
        counts: List[int] = []
        grids: List[Tuple[int, int]] = []
        for grid in image_grid_thw:
            t, h, w = [int(value) for value in grid.tolist()]
            if t != 1 or h % merge_size or w % merge_size:
                raise ValueError(f"Unsupported Qwen image grid {(t, h, w)}.")
            q_h, q_w = h // merge_size, w // merge_size
            counts.append(q_h * q_w)
            grids.append((q_h, q_w))
        if patches.shape[0] != sum(counts):
            raise ValueError(
                f"Found {patches.shape[0]} image tokens but grids require {sum(counts)}."
            )
        return list(torch.split(patches, counts, dim=0)), grids

    def _teacher_grids(
        self,
        teacher: torch.Tensor,
        target_grids: Sequence[Tuple[int, int]],
        vggt_hw: Tuple[int, int],
        add_position: bool = True,
    ) -> List[torch.Tensor]:
        if teacher.shape[0] != 1 or teacher.shape[1] != len(target_grids):
            raise ValueError(
                f"VGGT shape {tuple(teacher.shape[:2])} does not match "
                f"{len(target_grids)} target frames."
            )
        vggt_h, vggt_w = vggt_hw
        patch_h, patch_w = vggt_h // 14, vggt_w // 14
        if teacher.shape[2] != patch_h * patch_w:
            raise ValueError(
                f"VGGT patch count {teacher.shape[2]} != {patch_h}x{patch_w}."
            )
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed

        results = []
        for index, (q_h, q_w) in enumerate(target_grids):
            grid = teacher[0, index].reshape(patch_h, patch_w, -1)
            grid = grid.permute(2, 0, 1).unsqueeze(0)
            if add_position and self.spatial_args.spatial_use_vggt_pe:
                pos = create_uv_grid(
                    patch_w,
                    patch_h,
                    aspect_ratio=vggt_w / vggt_h,
                    dtype=grid.dtype,
                    device=grid.device,
                )
                pos = position_grid_to_embed(pos, grid.shape[1])
                pos = pos * float(self.spatial_args.spatial_pe_scale)
                grid = grid + pos.permute(2, 0, 1).unsqueeze(0)
            grid = F.interpolate(grid, size=(q_h, q_w), mode="bilinear", align_corners=True)
            results.append(grid.flatten(2).transpose(1, 2).squeeze(0))
        return results

    def _alignment_loss(
        self,
        model,
        projector_name: str,
        student_parts: Sequence[torch.Tensor],
        target_grids: Sequence[Tuple[int, int]],
        teacher: torch.Tensor,
        vggt_hw: Tuple[int, int],
    ) -> torch.Tensor:
        targets = self._teacher_grids(teacher, target_grids, vggt_hw)
        student = torch.cat(list(student_parts), dim=0)
        target = torch.cat(targets, dim=0)
        projector = _get_projector(model, projector_name)
        projected = projector(student.to(next(projector.parameters()).dtype))
        if projected.shape != target.shape:
            raise ValueError(f"Projected {projected.shape} does not match VGGT {target.shape}.")
        return 1.0 - F.cosine_similarity(projected.float(), target.float(), dim=-1).mean()

    def _delta_alignment_loss(
        self,
        model,
        student_residual: torch.Tensor,
        active_mask: torch.Tensor,
        local_teacher: torch.Tensor,
        single_teacher: torch.Tensor,
        target_grid: Tuple[int, int],
        vggt_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Align the temporal branch with VGGT's local-context increment."""
        if active_mask.shape[0] != student_residual.shape[0]:
            raise ValueError("Temporal residual and active mask have different sizes.")
        if not active_mask.any():
            return self._zero_projector_loss(model, "delta_sf_projector")
        local = self._teacher_grids(
            local_teacher, [target_grid], vggt_hw, add_position=False
        )[0].float()
        single = self._teacher_grids(
            single_teacher, [target_grid], vggt_hw, add_position=False
        )[0].float()
        # Feature normalization removes arbitrary per-call scale; the remaining
        # direction captures what multi-view aggregation adds to the current view.
        target_delta = F.normalize(local, dim=-1) - F.normalize(single, dim=-1)
        target_delta = F.normalize(target_delta, dim=-1)
        student = student_residual[active_mask]
        target = target_delta[active_mask]
        projector = _get_projector(model, "delta_sf_projector")
        projected = projector(student.to(next(projector.parameters()).dtype))
        return 1.0 - F.cosine_similarity(
            projected.float(), target.float(), dim=-1
        ).mean()

    @staticmethod
    def _zero_projector_loss(model, projector_name: str) -> torch.Tensor:
        projector = _get_projector(model, projector_name)
        return sum(parameter.sum() * 0.0 for parameter in projector.parameters())

    def _pop_additional_inputs(self, inputs):
        # Token masks are trainer metadata, not Qwen forward arguments.  With
        # unit subtask/phase weight the labels already encode the desired loss.
        inputs.pop("subtask_token_mask", None)
        inputs.pop("phase_token_mask", None)
        return None

    def _additional_loss(self, model, outputs, metadata):
        del model, metadata
        return outputs.loss * 0.0, {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        additional_metadata = self._pop_additional_inputs(inputs)
        auxiliary_pixels = inputs.pop("current_aux_pixel_values", None)
        auxiliary_grids = inputs.pop("current_aux_grid_thw", None)
        auxiliary_ages = inputs.pop("current_aux_ages", None)
        current_vggt = inputs.pop("current_vggt_images", None)
        current_count = inputs.pop("current_vggt_count", None)
        past_vggt = inputs.pop("past_vggt_images", None)
        past_count = inputs.pop("past_vggt_count", None)
        sf8_vggt = inputs.pop("sf8_vggt_images", None)
        sf8_count = inputs.pop("sf8_vggt_count", None)
        inputs.pop("task_type", None)

        merge_diagnostics: Dict[str, float] = {}
        residual_capture: Dict[str, torch.Tensor] = {}
        hook = None
        if self.spatial_args.current_merge_enable and auxiliary_pixels is not None:
            residual_gate = None
            if self.merge_config.fusion_mode == "gated_residual":
                residual_gate = _get_trainable_module(
                    model, "temporal_residual_gate"
                )
            hook = register_temporal_merge_hook(
                model,
                aux_pixel_values=auxiliary_pixels,
                aux_grid_thw=auxiliary_grids,
                main_grid_thw=inputs.get("image_grid_thw"),
                aux_ages=auxiliary_ages,
                config=self.merge_config,
                diagnostics=merge_diagnostics,
                residual_gate=residual_gate,
                capture=residual_capture,
            )
        try:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            if hook is not None:
                hook.remove()

        action_loss = outputs.loss
        additional_loss, additional_values = self._additional_loss(
            model, outputs, additional_metadata
        )
        student = outputs.hidden_states[int(self.spatial_args.spatial_student_layer)]
        student_parts, qwen_grids = self._student_image_parts(
            student,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["image_grid_thw"],
            model,
        )
        loss = action_loss + additional_loss
        sf8_loss = None
        sf8_weight = float(self.spatial_args.sf8_spatial_weight)
        if sf8_weight > 0.0:
            if sf8_vggt is None or sf8_count is None:
                raise ValueError("Eight-frame SF batch is missing VGGT inputs.")
            teacher = self._teacher_features(sf8_vggt)
            n_sf8 = int(sf8_count[0].item())
            if n_sf8 != len(student_parts) or n_sf8 != teacher.shape[1]:
                raise ValueError(
                    f"SF8 mismatch: student={len(student_parts)}, "
                    f"VGGT={teacher.shape[1]}, count={n_sf8}."
                )
            sf8_loss = self._alignment_loss(
                model,
                "sf8_sf_projector",
                student_parts,
                qwen_grids,
                teacher,
                tuple(sf8_vggt.shape[-2:]),
            )
            loss = loss + sf8_weight * sf8_loss

        current_loss = None
        current_weight = float(self.spatial_args.current_spatial_weight)
        if current_weight > 0.0:
            if current_vggt is None or current_count is None:
                raise ValueError("Current-SF batch is missing VGGT inputs.")
            teacher = self._teacher_features(current_vggt)
            if int(current_count[0].item()) != teacher.shape[1]:
                raise ValueError("Current VGGT frame count changed during collation.")
            current_loss = self._alignment_loss(
                model,
                "current_sf_projector",
                [student_parts[-1]],
                [qwen_grids[-1]],
                teacher[:, -1:, :, :],
                tuple(current_vggt.shape[-2:]),
            )
            loss = loss + current_weight * current_loss

        past_loss = None
        past_weight = float(self.spatial_args.past_spatial_weight)
        if past_weight > 0.0:
            if past_vggt is not None:
                teacher = self._teacher_features(past_vggt)
                n_past = int(past_count[0].item())
                if n_past != len(student_parts) - 1 or n_past != teacher.shape[1]:
                    raise ValueError(
                        f"Past frame mismatch: student={len(student_parts)-1}, "
                        f"VGGT={teacher.shape[1]}, count={n_past}."
                    )
                past_loss = self._alignment_loss(
                    model,
                    "past_sf_projector",
                    student_parts[:-1],
                    qwen_grids[:-1],
                    teacher,
                    tuple(past_vggt.shape[-2:]),
                )
                loss = loss + past_weight * past_loss
            else:
                # Early t=0 rows have no historical frame.  Touching the module
                # keeps DDP's fixed unused-parameter contract valid.
                loss = loss + self._zero_projector_loss(model, "past_sf_projector")

        delta_loss = None
        delta_weight = float(self.spatial_args.delta_spatial_weight)
        if delta_weight > 0.0:
            n_local = int(current_count[0].item()) if current_count is not None else 0
            if (
                current_vggt is not None
                and n_local >= 2
                and "current_residual" in residual_capture
            ):
                local_teacher = self._teacher_features(current_vggt)[:, -1:, :, :]
                single_teacher = self._teacher_features(
                    current_vggt[:, -1:, :, :, :]
                )
                delta_loss = self._delta_alignment_loss(
                    model,
                    residual_capture["current_residual"],
                    residual_capture["active_mask"],
                    local_teacher,
                    single_teacher,
                    qwen_grids[-1],
                    tuple(current_vggt.shape[-2:]),
                )
                loss = loss + delta_weight * delta_loss
            else:
                loss = loss + self._zero_projector_loss(model, "delta_sf_projector")

        if (
            self.merge_config.fusion_mode == "gated_residual"
            and "gate" not in residual_capture
        ):
            # t=0 has no neighbor.  Keep DDP's used-parameter set identical.
            gate_module = _get_trainable_module(model, "temporal_residual_gate")
            loss = loss + sum(parameter.sum() * 0.0 for parameter in gate_module.parameters())

        values = {"action_loss": float(action_loss.detach())}
        values.update(additional_values)
        if sf8_loss is not None:
            values["sf8_align_loss"] = float(sf8_loss.detach())
        if current_loss is not None:
            values["current_align_loss"] = float(current_loss.detach())
        if past_loss is not None:
            values["past_align_loss"] = float(past_loss.detach())
        if delta_loss is not None:
            values["delta_align_loss"] = float(delta_loss.detach())
        values.update({f"merge_{k}": v for k, v in merge_diagnostics.items()})
        for key, value in values.items():
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + value
        self._metric_n += 1
        if not self._printed_shapes:
            rank0_print(
                f"Temporal spatial shapes: Qwen={tuple(student.shape)} "
                f"images={len(student_parts)} current_grid={qwen_grids[-1]} "
                f"merge={merge_diagnostics}"
            )
            self._printed_shapes = True
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._metric_n:
            for key, total in self._metric_sums.items():
                logs[key] = round(total / self._metric_n, 6)
            self._metric_sums = {}
            self._metric_n = 0
        return super().log(logs, *args, **kwargs)


def make_data_module(processor, tokenizer, data_args, spatial_args):
    configs = data_list(data_args.dataset_use.split(","))
    if not configs or any(config["type"] != "vln_unified" for config in configs):
        raise ValueError("This entry point supports VLN datasets only.")
    datasets = []
    for config in configs:
        rank0_print(
            f"Building {config['name']} from {config['video_folder']} "
            f"(Current-SF={spatial_args.current_spatial_weight}, "
            f"Past-SF={spatial_args.past_spatial_weight}, "
            f"SF8={spatial_args.sf8_spatial_weight}, "
            f"Delta-SF={spatial_args.delta_spatial_weight})"
        )
        datasets.append(
            TemporalSpatialVLNDataset(
                processor=processor,
                data_args=data_args,
                vln_config=config,
                label_mode="action",
                spatial_args=spatial_args,
            )
        )
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": TemporalSpatialCollator(tokenizer=tokenizer),
    }


def load_vggt_teacher(spatial_args, device):
    source = os.path.abspath(spatial_args.vggt_source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    from vggt.models.vggt import VGGT

    checkpoint = spatial_args.vggt_checkpoint_path
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
    teacher = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    incompatible = teacher.load_state_dict(payload, strict=False)
    missing = [key for key in incompatible.missing_keys if key.startswith("aggregator.")]
    if missing:
        raise RuntimeError(f"VGGT checkpoint is missing aggregator weights: {missing[:8]}")
    del payload
    gc.collect()
    teacher.requires_grad_(False).eval()
    teacher.to(device=device, dtype=torch.bfloat16)
    rank0_print(f"Loaded frozen VGGT teacher from {checkpoint}")
    return teacher


def train(attn_implementation="flash_attention_2"):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TemporalSpatialArguments, TrainingArguments)
    )
    model_args, data_args, spatial_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    transformers.set_seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    spatial_args.merge_config()  # fail fast on malformed settings

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    data_args.model_type = "qwen3vl"
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model

        model.requires_grad_(False)
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(training_args.lora_r),
                lora_alpha=int(training_args.lora_alpha),
                lora_dropout=float(training_args.lora_dropout),
                target_modules="all-linear",
                init_lora_weights="gaussian",
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    else:
        set_full_finetune_modules(model_args, model)

    hidden_size = _hidden_size(model)
    if float(spatial_args.current_spatial_weight) > 0.0:
        model.current_sf_projector = AlignProjector(hidden_size, 1024)
        model.current_sf_projector.to(dtype=next(model.parameters()).dtype)
    if float(spatial_args.past_spatial_weight) > 0.0:
        model.past_sf_projector = AlignProjector(hidden_size, 1024)
        model.past_sf_projector.to(dtype=next(model.parameters()).dtype)
    if float(spatial_args.sf8_spatial_weight) > 0.0:
        model.sf8_sf_projector = AlignProjector(hidden_size, 1024)
        model.sf8_sf_projector.to(dtype=next(model.parameters()).dtype)
    if float(spatial_args.delta_spatial_weight) > 0.0:
        model.delta_sf_projector = AlignProjector(hidden_size, 1024)
        model.delta_sf_projector.to(dtype=next(model.parameters()).dtype)
    if spatial_args.current_fusion_mode == "gated_residual":
        model.temporal_residual_gate = TemporalResidualGate(
            initial_gate=float(spatial_args.current_gate_init),
            max_gate=float(spatial_args.current_gate_max),
        )
        model.temporal_residual_gate.to(dtype=torch.float32)

    rank0_print(
        "Temporal Spatial VLN: "
        f"window={spatial_args.current_window_size}, "
        f"ratio={spatial_args.current_merge_ratio}, "
        f"decay={spatial_args.current_temporal_decay}, "
        f"fusion={spatial_args.current_fusion_mode}, "
        f"matching={spatial_args.current_matching_mode}, "
        f"Current-SF={spatial_args.current_spatial_weight}, "
        f"Past-SF={spatial_args.past_spatial_weight}, "
        f"SF8={spatial_args.sf8_spatial_weight}, "
        f"Delta-SF={spatial_args.delta_spatial_weight}, "
        f"LoRA={training_args.lora_enable}"
    )
    if hasattr(model, "print_trainable_parameters") and local_rank in (-1, 0):
        model.print_trainable_parameters()
        extra = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "_sf_projector" in name or "temporal_residual_gate" in name
        )
        rank0_print(f"Additional trainable spatial projector params: {extra:,}")

    data_module = make_data_module(processor, tokenizer, data_args, spatial_args)
    teacher = None
    if (
        float(spatial_args.current_spatial_weight) > 0.0
        or float(spatial_args.past_spatial_weight) > 0.0
        or float(spatial_args.sf8_spatial_weight) > 0.0
        or float(spatial_args.delta_spatial_weight) > 0.0
    ):
        teacher = load_vggt_teacher(spatial_args, training_args.device)

    trainer = TemporalSpatialTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        spatial_args=spatial_args,
        vggt_teacher=teacher,
        **data_module,
    )
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    if os.environ.get("VLN_SKIP_FINAL_SAVE", "0") == "1":
        rank0_print("VLN_SKIP_FINAL_SAVE=1; skipping final save.")
        return
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    if training_args.should_save:
        processor.save_pretrained(training_args.output_dir)
        metadata = {
            "recipe": (
                "gated_spatial_e6"
                if spatial_args.delta_spatial_weight > 0
                else "gated_spatial_e5"
                if spatial_args.sf8_spatial_weight > 0
                and spatial_args.current_fusion_mode == "gated_residual"
                else "temporal_spatial_e4"
                if spatial_args.past_spatial_weight > 0
                else "temporal_spatial_e3"
            ),
            "temporal_spatial_arguments": vars(spatial_args),
            "lora": {
                "enabled": training_args.lora_enable,
                "r": training_args.lora_r,
                "alpha": training_args.lora_alpha,
                "dropout": training_args.lora_dropout,
                "target_modules": "all-linear",
            },
        }
        with open(
            os.path.join(training_args.output_dir, "experiment_recipe.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train()
