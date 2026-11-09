"""LoRA fine-tuning for the paired R2R control / Spatial Forcing experiment.

The Spatial Forcing branch follows the patch-level recipe from the official
OpenHelix implementation: a frozen VGGT teacher, a non-final VLM hidden layer,
2-D positional embeddings on the VGGT patch grid, bilinear grid alignment, a
two-layer projector, and per-patch cosine loss.  Setting
``--spatial_align_weight 0`` takes the paired action-only control path.
"""

import gc
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data import data_list
from qwenvl.data.vln_unified_dataset import (
    VLNActionDataCollator,
    VLNUnifiedDataset,
)
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments


local_rank = -1


def rank0_print(*args, **kwargs):
    if local_rank in (-1, 0):
        print(*args, **kwargs)


@dataclass
class PatchSpatialForcingArguments:
    spatial_align_weight: float = field(
        default=0.0,
        metadata={"help": "0 for control; official Spatial Forcing uses 0.5."},
    )
    spatial_student_layer: int = field(
        default=27,
        metadata={"help": "Qwen hidden_states index used for patch alignment."},
    )
    spatial_teacher_layer: int = field(
        default=-1,
        metadata={"help": "Cached VGGT aggregator layer; -1 is the final layer."},
    )
    spatial_use_vggt_pe: bool = field(default=True)
    spatial_pe_scale: float = field(default=0.1)
    spatial_projector_lr: float = field(
        default=0.0,
        metadata={"help": "Separate projector LR; <=0 uses --learning_rate."},
    )
    vggt_checkpoint_path: str = field(
        default="assets/vggt/vggt.pt"
    )
    vggt_source_path: str = field(
        default="third_party/vggt"
    )
    vggt_preprocess_mode: str = field(default="crop")


class AlignProjector(nn.Module):
    """Official SF projector: LLM dim -> 2*VGGT embed dim -> 2*embed dim."""

    def __init__(self, llm_dim: int, vggt_embed_dim: int = 1024):
        super().__init__()
        output_dim = 2 * vggt_embed_dim
        self.fc1 = nn.Linear(llm_dim, output_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden)))


def _preprocess_vggt_pil_images(
    images: Sequence[Image.Image], mode: str = "crop"
) -> torch.Tensor:
    """PIL-input equivalent of VGGT's official load_and_preprocess_images.

    Reusing the PIL objects already loaded for Qwen avoids reading every R2R
    frame twice from shared storage.
    """
    if not images:
        raise ValueError("VGGT preprocessing needs at least one image.")
    if mode not in ("crop", "pad"):
        raise ValueError("vggt_preprocess_mode must be 'crop' or 'pad'.")

    from torchvision.transforms.functional import to_tensor

    target_size = 518
    tensors: List[torch.Tensor] = []
    for source in images:
        image = source.convert("RGB")
        width, height = image.size
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = to_tensor(image)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            tensor = tensor[:, start_y : start_y + target_size, :]
        elif mode == "pad":
            h_padding = target_size - tensor.shape[1]
            w_padding = target_size - tensor.shape[2]
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = F.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        tensors.append(tensor)

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded: List[torch.Tensor] = []
    for tensor in tensors:
        h_padding = max_h - tensor.shape[1]
        w_padding = max_w - tensor.shape[2]
        if h_padding or w_padding:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = F.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded.append(tensor)
    return torch.stack(padded)


class PatchSpatialVLNDataset(VLNUnifiedDataset):
    def __init__(self, *args, spatial_args=None, **kwargs):
        self.spatial_args = spatial_args or PatchSpatialForcingArguments()
        self._sf_loaded_images: Optional[List[Image.Image]] = None
        super().__init__(*args, **kwargs)

    def _load_images(self, rel_paths: List[str]) -> List[Image.Image]:
        images = super()._load_images(rel_paths)
        self._sf_loaded_images = images
        return images

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if float(self.spatial_args.spatial_align_weight) > 0.0:
            if self._sf_loaded_images is None:
                raise RuntimeError("VLN images were not captured for VGGT preprocessing.")
            item["sf_vggt_images"] = _preprocess_vggt_pil_images(
                self._sf_loaded_images, self.spatial_args.vggt_preprocess_mode
            )
            item["sf_num_images"] = len(self._sf_loaded_images)
        self._sf_loaded_images = None
        return item


class PatchSpatialCollator(VLNActionDataCollator):
    def __call__(self, instances):
        batch = super().__call__(instances)
        spatial_instances = [x for x in instances if "sf_vggt_images" in x]
        if spatial_instances:
            if len(instances) != 1:
                raise ValueError(
                    "Patch Spatial Forcing currently requires "
                    "--per_device_train_batch_size 1."
                )
            batch["sf_vggt_images"] = spatial_instances[0]["sf_vggt_images"].unsqueeze(0)
            batch["sf_num_images"] = torch.tensor(
                [spatial_instances[0]["sf_num_images"]], dtype=torch.long
            )
        return batch


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _get_align_projector(model) -> AlignProjector:
    model = _unwrap_model(model)
    projector = getattr(model, "sf_align_projector", None)
    if projector is None and hasattr(model, "base_model"):
        projector = getattr(model.base_model, "sf_align_projector", None)
    if projector is None:
        raise AttributeError("sf_align_projector is missing from the trainable model.")
    return projector


class PatchSpatialTrainer(Trainer):
    PROJECTOR_FILE = "sf_align_projector.pt"

    def __init__(self, *args, spatial_args=None, vggt_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_args = spatial_args or PatchSpatialForcingArguments()
        self.vggt_teacher = vggt_teacher
        self._align_sum = 0.0
        self._action_sum = 0.0
        self._align_n = 0

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        projector_lr = float(self.spatial_args.spatial_projector_lr)
        if projector_lr <= 0.0 or float(self.spatial_args.spatial_align_weight) <= 0.0:
            return optimizer

        projector_ids = {id(p) for p in _get_align_projector(self.model).parameters()}
        split_groups = []
        projector_count = 0
        for group in optimizer.param_groups:
            projector_params = [p for p in group["params"] if id(p) in projector_ids]
            other_params = [p for p in group["params"] if id(p) not in projector_ids]
            if other_params:
                other_group = dict(group)
                other_group["params"] = other_params
                split_groups.append(other_group)
            if projector_params:
                projector_group = dict(group)
                projector_group["params"] = projector_params
                projector_group["lr"] = projector_lr
                projector_group["initial_lr"] = projector_lr
                split_groups.append(projector_group)
                projector_count += sum(p.numel() for p in projector_params)
        optimizer.param_groups[:] = split_groups
        rank0_print(
            f"Spatial projector optimizer group: lr={projector_lr} "
            f"params={projector_count:,}"
        )
        return optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_dir = output_dir or self.args.output_dir
        if (
            self.args.should_save
            and float(self.spatial_args.spatial_align_weight) > 0.0
        ):
            projector = _get_align_projector(self.model)
            state = {k: v.detach().cpu() for k, v in projector.state_dict().items()}
            torch.save(state, os.path.join(output_dir, self.PROJECTOR_FILE))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if float(self.spatial_args.spatial_align_weight) > 0.0:
            path = os.path.join(resume_from_checkpoint, self.PROJECTOR_FILE)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing Spatial Forcing projector: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            _get_align_projector(self.model).load_state_dict(state, strict=True)
        return result

    @staticmethod
    def _image_token_id(model) -> int:
        config = _unwrap_model(model).config
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None and hasattr(config, "vision_config"):
            image_token_id = getattr(config.vision_config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Cannot resolve Qwen image_token_id.")
        return int(image_token_id)

    def _teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.vggt_teacher is None:
            raise RuntimeError("Spatial alignment is enabled but VGGT is not loaded.")
        self.vggt_teacher.eval()
        teacher_dtype = next(self.vggt_teacher.parameters()).dtype
        images = images.to(dtype=teacher_dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            features, patch_start_idx = self.vggt_teacher.aggregator(images)
            layer = int(self.spatial_args.spatial_teacher_layer)
            target = features[layer]
            if target is None:
                cached = [i for i, value in enumerate(features) if value is not None]
                raise ValueError(
                    f"VGGT layer {layer} is not cached; available layers are {cached}."
                )
            target = target[:, :, patch_start_idx:, :]
        return target

    def _align_patch_grids(
        self,
        student: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        teacher: torch.Tensor,
        vggt_image_hw: Tuple[int, int],
        num_images: torch.Tensor,
        model,
    ) -> Tuple[torch.Tensor, int]:
        if student.size(0) != 1:
            raise ValueError("Patch Spatial Forcing requires per-device batch size 1.")
        n_images = int(num_images[0].item())
        if teacher.shape[0] != 1 or teacher.shape[1] != n_images:
            raise ValueError(
                f"VGGT returned {tuple(teacher.shape[:2])}, expected (1, {n_images})."
            )
        if image_grid_thw is None or image_grid_thw.shape[0] != n_images:
            raise ValueError(
                f"Qwen image_grid_thw has shape "
                f"{None if image_grid_thw is None else tuple(image_grid_thw.shape)}, "
                f"expected ({n_images}, 3)."
            )

        image_token_id = self._image_token_id(model)
        token_mask = input_ids[0].eq(image_token_id) & attention_mask[0].bool()
        student_patches = student[0, token_mask]

        merge_size = int(getattr(self.train_dataset, "merge_size", 2))
        token_counts = []
        target_grids = []
        for grid in image_grid_thw:
            t, h, w = [int(x) for x in grid.tolist()]
            if t != 1 or h % merge_size or w % merge_size:
                raise ValueError(
                    f"Unsupported Qwen image grid {(t, h, w)} with merge_size={merge_size}."
                )
            q_h, q_w = h // merge_size, w // merge_size
            token_counts.append(t * q_h * q_w)
            target_grids.append((q_h, q_w))
        expected_tokens = sum(token_counts)
        if student_patches.shape[0] != expected_tokens:
            raise ValueError(
                f"Found {student_patches.shape[0]} Qwen image tokens but grids "
                f"require {expected_tokens}; sequence truncation or token ordering changed."
            )

        vggt_h, vggt_w = vggt_image_hw
        patch_h, patch_w = vggt_h // 14, vggt_w // 14
        if teacher.shape[2] != patch_h * patch_w:
            raise ValueError(
                f"VGGT patch count {teacher.shape[2]} != {patch_h}x{patch_w}."
            )

        from vggt.heads.utils import create_uv_grid, position_grid_to_embed

        teacher_grids = []
        for frame_idx, (q_h, q_w) in enumerate(target_grids):
            grid = teacher[0, frame_idx].reshape(patch_h, patch_w, -1)
            grid = grid.permute(2, 0, 1).unsqueeze(0)
            if self.spatial_args.spatial_use_vggt_pe:
                pos = create_uv_grid(
                    patch_w,
                    patch_h,
                    aspect_ratio=vggt_w / vggt_h,
                    dtype=grid.dtype,
                    device=grid.device,
                )
                pos = position_grid_to_embed(pos, grid.shape[1])
                pos = pos * float(self.spatial_args.spatial_pe_scale)
                pos = pos.permute(2, 0, 1).unsqueeze(0)
                grid = grid + pos
            grid = F.interpolate(
                grid, size=(q_h, q_w), mode="bilinear", align_corners=True
            )
            teacher_grids.append(grid.flatten(2).transpose(1, 2).squeeze(0))
        target_patches = torch.cat(teacher_grids, dim=0)

        projector = _get_align_projector(model)
        projected = projector(student_patches.to(next(projector.parameters()).dtype))
        if projected.shape != target_patches.shape:
            raise ValueError(
                f"Projected Qwen patches {tuple(projected.shape)} do not match "
                f"VGGT patches {tuple(target_patches.shape)}."
            )
        align_loss = 1.0 - F.cosine_similarity(
            projected.float(), target_patches.float(), dim=-1
        ).mean()
        return align_loss, expected_tokens

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        vggt_images = inputs.pop("sf_vggt_images", None)
        num_images = inputs.pop("sf_num_images", None)
        inputs.pop("task_type", None)
        weight = float(self.spatial_args.spatial_align_weight)
        if weight <= 0.0:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        if vggt_images is None or num_images is None:
            raise ValueError("Spatial batch is missing VGGT images or image counts.")

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        action_loss = outputs.loss
        layer = int(self.spatial_args.spatial_student_layer)
        try:
            student = outputs.hidden_states[layer]
        except IndexError as exc:
            raise ValueError(
                f"Qwen hidden layer {layer} is unavailable; "
                f"received {len(outputs.hidden_states)} hidden states."
            ) from exc

        teacher = self._teacher_features(vggt_images)
        align_loss, patch_count = self._align_patch_grids(
            student=student,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_thw=inputs.get("image_grid_thw"),
            teacher=teacher,
            vggt_image_hw=tuple(vggt_images.shape[-2:]),
            num_images=num_images,
            model=model,
        )
        loss = action_loss + weight * align_loss

        self._align_sum += float(align_loss.detach())
        self._action_sum += float(action_loss.detach())
        self._align_n += 1
        if self._align_n == 1:
            rank0_print(
                f"Spatial shapes: Qwen={tuple(student.shape)} "
                f"VGGT={tuple(teacher.shape)} aligned_patches={patch_count}"
            )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._align_n:
            logs["action_loss"] = round(self._action_sum / self._align_n, 6)
            logs["align_loss"] = round(self._align_sum / self._align_n, 6)
            self._align_sum = 0.0
            self._action_sum = 0.0
            self._align_n = 0
        return super().log(logs, *args, **kwargs)


def set_full_finetune_modules(model_args, model):
    for parameter in model.visual.parameters():
        parameter.requires_grad = bool(model_args.tune_mm_vision)
    for parameter in model.visual.merger.parameters():
        parameter.requires_grad = bool(model_args.tune_mm_mlp)
    for parameter in model.language_model.parameters():
        parameter.requires_grad = bool(model_args.tune_mm_llm)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = bool(model_args.tune_mm_llm)


def make_data_module(processor, tokenizer, data_args, spatial_args):
    configs = data_list(data_args.dataset_use.split(","))
    if not configs:
        raise ValueError("--dataset_use must select an R2R VLN dataset.")
    if any(config["type"] != "vln_unified" for config in configs):
        raise ValueError("This paired entry point supports VLN datasets only.")

    datasets = []
    for config in configs:
        rank0_print(
            f"Building {config['name']} from {config['video_folder']} "
            f"(Spatial weight={spatial_args.spatial_align_weight})"
        )
        datasets.append(
            PatchSpatialVLNDataset(
                processor=processor,
                data_args=data_args,
                vln_config=config,
                label_mode="action",
                spatial_args=spatial_args,
            )
        )
    train_dataset = (
        datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": PatchSpatialCollator(tokenizer=tokenizer),
    }


def _hidden_size(model) -> int:
    config = model.config
    value = getattr(config, "hidden_size", None)
    if value is None and hasattr(config, "text_config"):
        value = getattr(config.text_config, "hidden_size", None)
    if value is None:
        raise ValueError("Cannot resolve Qwen text hidden size.")
    return int(value)


def load_vggt_teacher(spatial_args, device):
    source_path = os.path.abspath(spatial_args.vggt_source_path)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
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
    missing_aggregator = [k for k in incompatible.missing_keys if k.startswith("aggregator.")]
    if missing_aggregator:
        raise RuntimeError(
            f"VGGT checkpoint is missing aggregator weights: {missing_aggregator[:8]}"
        )
    rank0_print(
        f"Loaded frozen VGGT from {checkpoint}; "
        f"ignored checkpoint keys={len(incompatible.unexpected_keys)}"
    )
    del payload
    gc.collect()
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.to(device=device, dtype=torch.bfloat16)
    return teacher


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            PatchSpatialForcingArguments,
            TrainingArguments,
        )
    )
    model_args, data_args, spatial_args, training_args = (
        parser.parse_args_into_dataclasses()
    )
    local_rank = training_args.local_rank
    transformers.set_seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False

    model_path_lower = model_args.model_name_or_path.lower()
    if "qwen3" not in model_path_lower and "rynnbrain" not in model_path_lower:
        raise ValueError("This entry point supports Qwen3-VL / RynnBrain only.")
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
        lora_config = LoraConfig(
            r=int(training_args.lora_r),
            lora_alpha=int(training_args.lora_alpha),
            lora_dropout=float(training_args.lora_dropout),
            target_modules="all-linear",
            init_lora_weights="gaussian",
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        rank0_print(
            f"LoRA enabled: r={training_args.lora_r}, "
            f"alpha={training_args.lora_alpha}, dropout={training_args.lora_dropout}, "
            "target=all-linear, init=gaussian"
        )
    else:
        set_full_finetune_modules(model_args, model)

    if float(spatial_args.spatial_align_weight) > 0.0:
        model.sf_align_projector = AlignProjector(_hidden_size(model), 1024)
        model.sf_align_projector.to(dtype=next(model.parameters()).dtype)
        rank0_print(
            f"Patch Spatial Forcing enabled: weight={spatial_args.spatial_align_weight}, "
            f"student_layer={spatial_args.spatial_student_layer}, "
            f"teacher_layer={spatial_args.spatial_teacher_layer}, "
            f"PE={spatial_args.spatial_use_vggt_pe}@{spatial_args.spatial_pe_scale}"
        )
    elif local_rank in (-1, 0):
        rank0_print("Paired LoRA-Control enabled: action loss only.")

    if hasattr(model, "print_trainable_parameters") and local_rank in (-1, 0):
        model.print_trainable_parameters()
        if hasattr(model, "sf_align_projector"):
            count = sum(p.numel() for p in model.sf_align_projector.parameters())
            rank0_print(f"Additional trainable Spatial projector params: {count:,}")

    data_module = make_data_module(processor, tokenizer, data_args, spatial_args)

    teacher = None
    if float(spatial_args.spatial_align_weight) > 0.0:
        teacher = load_vggt_teacher(spatial_args, training_args.device)

    trainer = PatchSpatialTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        spatial_args=spatial_args,
        vggt_teacher=teacher,
        **data_module,
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        logging.info("Checkpoint found; resuming training.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if os.environ.get("VLN_SKIP_FINAL_SAVE", "0") == "1":
        rank0_print("VLN_SKIP_FINAL_SAVE=1; skipping final save.")
        return
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    if training_args.should_save:
        processor.save_pretrained(training_args.output_dir)
        metadata = {
            "recipe": "patch_spatial_forcing" if spatial_args.spatial_align_weight > 0 else "lora_control",
            "spatial_arguments": vars(spatial_args),
            "lora": {
                "enabled": training_args.lora_enable,
                "r": training_args.lora_r,
                "alpha": training_args.lora_alpha,
                "dropout": training_args.lora_dropout,
                "target_modules": "all-linear",
                "init": "gaussian",
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
