"""Fixed-grid temporal token merging for Spatial-Aware VLN.

The current image is the protected anchor.  Tokens from nearby observations
are matched to the anchor grid with detached cosine similarity and accumulated
with a weighted mean.  The number and order of current-image tokens never
change, which keeps Qwen3-VL RoPE positions and patch-level teacher alignment
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalMergeConfig:
    merge_ratio: float = 0.5
    spatial_bias: float = 0.05
    anchor_weight: float = 1.0
    temporal_decay: Tuple[float, ...] = (0.5, 0.7, 0.9)
    fusion_mode: str = "weighted_mean"
    matching_mode: str = "source_best"

    def __post_init__(self):
        if not 0.0 <= self.merge_ratio <= 1.0:
            raise ValueError("merge_ratio must be in [0, 1].")
        if self.anchor_weight <= 0.0:
            raise ValueError("anchor_weight must be positive.")
        if any(value <= 0.0 for value in self.temporal_decay):
            raise ValueError("temporal_decay entries must be positive.")
        if self.fusion_mode not in ("weighted_mean", "gated_residual"):
            raise ValueError(
                "fusion_mode must be 'weighted_mean' or 'gated_residual'."
            )
        if self.matching_mode not in ("source_best", "mutual"):
            raise ValueError("matching_mode must be 'source_best' or 'mutual'.")


class TemporalResidualGate(nn.Module):
    """Learnable scalar gate constrained to ``(0, max_gate)``.

    A small non-zero initialization lets the residual branch receive gradients
    immediately while preserving an almost-identity current-frame pathway.
    """

    def __init__(self, initial_gate: float = 0.05, max_gate: float = 0.2):
        super().__init__()
        if not 0.0 < initial_gate < max_gate:
            raise ValueError("initial_gate must be strictly between 0 and max_gate.")
        self.max_gate = float(max_gate)
        probability = float(initial_gate) / self.max_gate
        raw = math.log(probability / (1.0 - probability))
        self.raw_gate = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate) * self.max_gate


def parse_temporal_decay(value: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(value, str):
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        result = tuple(float(item) for item in value)
    if not result:
        raise ValueError("temporal_decay cannot be empty.")
    return result


def _qwen_conditional_model(model):
    """Resolve Qwen3VLForConditionalGeneration through DDP and PEFT wrappers."""
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model
    raise TypeError(f"Cannot resolve Qwen3-VL model from {type(model)!r}.")


def _llm_grid(grid: torch.Tensor, merge_size: int) -> Tuple[int, int, int]:
    t, h, w = [int(value) for value in grid.tolist()]
    if h % merge_size or w % merge_size:
        raise ValueError(
            f"Vision grid {(t, h, w)} is not divisible by merge_size={merge_size}."
        )
    return t, h // merge_size, w // merge_size


def _split_sizes(grid_thw: torch.Tensor, merge_size: int) -> List[int]:
    return [
        t * h * w
        for t, h, w in (_llm_grid(grid, merge_size) for grid in grid_thw)
    ]


def _resize_tokens(
    tokens: torch.Tensor,
    source_grid: Tuple[int, int, int],
    target_grid: Tuple[int, int, int],
) -> torch.Tensor:
    source_t, source_h, source_w = source_grid
    target_t, target_h, target_w = target_grid
    if source_t != 1 or target_t != 1:
        raise ValueError("Current-frame merging supports image grids with t=1 only.")
    if (source_h, source_w) == (target_h, target_w):
        return tokens
    channels = tokens.shape[-1]
    feature_map = tokens.reshape(source_h, source_w, channels).permute(2, 0, 1)
    feature_map = F.interpolate(
        feature_map.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, channels)


def _normalized_coordinates(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


def _match_source_to_anchor(
    source: torch.Tensor,
    anchor: torch.Tensor,
    grid: Tuple[int, int, int],
    merge_ratio: float,
    spatial_bias: float,
    matching_mode: str = "source_best",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected source indices, destination indices, and similarities."""
    if source.shape != anchor.shape:
        raise ValueError(
            f"Source and anchor shapes must match, got {source.shape} and {anchor.shape}."
        )
    count = source.shape[0]
    keep = min(count, max(0, int(round(count * merge_ratio))))
    if keep == 0:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=source.device)

    with torch.no_grad():
        source_norm = F.normalize(source.detach().float(), dim=-1)
        anchor_norm = F.normalize(anchor.detach().float(), dim=-1)
        scores = source_norm @ anchor_norm.transpose(0, 1)
        if spatial_bias:
            _, height, width = grid
            coords = _normalized_coordinates(
                height, width, device=scores.device, dtype=scores.dtype
            )
            distance_sq = torch.cdist(coords, coords, p=2).square()
            scores = scores - float(spatial_bias) * distance_sq
        best_score, destination = scores.max(dim=1)
        candidates = torch.arange(count, device=source.device)
        if matching_mode == "mutual":
            # Each destination nominates exactly one source.  Keeping only
            # reciprocal pairs makes both source and destination indices unique,
            # preventing many-to-one contamination of a current patch.
            anchor_source = scores.max(dim=0).indices
            mutual = anchor_source.index_select(0, destination).eq(candidates)
            candidates = candidates[mutual]
        if candidates.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=source.device)
            return empty, empty, torch.empty(0, dtype=torch.float32, device=source.device)
        keep = min(keep, int(candidates.numel()))
        candidate_scores = best_score.index_select(0, candidates)
        chosen = candidate_scores.topk(keep, largest=True, sorted=False).indices
        selected = candidates.index_select(0, chosen)
    return selected, destination[selected], best_score[selected]


def _apply_matches(
    anchor: torch.Tensor,
    sources: Sequence[torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: Sequence[float],
    anchor_weight: float,
) -> torch.Tensor:
    numerator = anchor * float(anchor_weight)
    denominator = anchor.new_full((anchor.shape[0], 1), float(anchor_weight))
    for source, (selected, destination), weight in zip(sources, matches, weights):
        if selected.numel() == 0:
            continue
        weighted = source.index_select(0, selected) * float(weight)
        numerator = numerator.index_add(0, destination, weighted)
        additions = denominator.new_full((selected.numel(), 1), float(weight))
        denominator = denominator.index_add(0, destination, additions)
    return numerator / denominator.clamp_min(1e-6)


def _fuse_level(
    main_flat: torch.Tensor,
    aux_flat: torch.Tensor,
    main_sizes: Sequence[int],
    aux_sizes: Sequence[int],
    main_grids: Sequence[Tuple[int, int, int]],
    aux_grids: Sequence[Tuple[int, int, int]],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: Sequence[float],
    anchor_weight: float,
) -> torch.Tensor:
    main_parts = list(torch.split(main_flat, list(main_sizes), dim=0))
    aux_parts = list(torch.split(aux_flat, list(aux_sizes), dim=0))
    anchor_grid = main_grids[-1]
    resized_sources = [
        _resize_tokens(part, grid, anchor_grid)
        for part, grid in zip(aux_parts, aux_grids)
    ]
    main_parts[-1] = _apply_matches(
        main_parts[-1], resized_sources, matches, weights, anchor_weight
    )
    return torch.cat(main_parts, dim=0)


def _residual_level(
    main_flat: torch.Tensor,
    aux_flat: torch.Tensor,
    main_sizes: Sequence[int],
    aux_sizes: Sequence[int],
    main_grids: Sequence[Tuple[int, int, int]],
    aux_grids: Sequence[Tuple[int, int, int]],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: Sequence[float],
    gate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a sparse temporal residual while leaving the anchor as identity."""
    main_parts = list(torch.split(main_flat, list(main_sizes), dim=0))
    aux_parts = list(torch.split(aux_flat, list(aux_sizes), dim=0))
    anchor = main_parts[-1]
    anchor_grid = main_grids[-1]
    sources = [
        _resize_tokens(part, grid, anchor_grid)
        for part, grid in zip(aux_parts, aux_grids)
    ]
    numerator = torch.zeros_like(anchor)
    denominator = anchor.new_zeros((anchor.shape[0], 1))
    for source, (selected, destination), weight in zip(sources, matches, weights):
        if selected.numel() == 0:
            continue
        # Interpolate toward matched past evidence.  The small learned gate
        # controls how much context may modify the protected current token.
        differences = source.index_select(0, selected) - anchor.index_select(
            0, destination
        )
        numerator = numerator.index_add(0, destination, differences * float(weight))
        additions = denominator.new_full((selected.numel(), 1), float(weight))
        denominator = denominator.index_add(0, destination, additions)
    active = denominator.squeeze(-1).gt(0)
    residual = numerator / denominator.clamp_min(1e-6)
    main_parts[-1] = anchor + gate.to(anchor.dtype) * residual
    return torch.cat(main_parts, dim=0), residual, active


def register_temporal_merge_hook(
    model,
    *,
    aux_pixel_values: torch.Tensor,
    aux_grid_thw: torch.Tensor,
    main_grid_thw: torch.Tensor,
    aux_ages: torch.Tensor,
    config: TemporalMergeConfig,
    diagnostics: Dict[str, float] | None = None,
    residual_gate: TemporalResidualGate | None = None,
    capture: Dict[str, torch.Tensor] | None = None,
):
    """Fuse auxiliary frames into the last main image on the next visual call.

    The hook removes itself before recursively encoding the auxiliary frames, so
    it is single-use and safe during autoregressive generation (only prefill has
    pixel values).  Both final visual embeddings and Qwen DeepStack features use
    the same ToMe-style matches.
    """
    if aux_grid_thw is None or aux_grid_thw.numel() == 0:
        return None
    if main_grid_thw is None or main_grid_thw.numel() == 0:
        raise ValueError("main_grid_thw is required for temporal token merging.")
    if aux_grid_thw.shape[0] != aux_ages.numel():
        raise ValueError(
            f"Got {aux_grid_thw.shape[0]} auxiliary grids but {aux_ages.numel()} ages."
        )

    qwen = _qwen_conditional_model(model)
    visual = qwen.model.visual
    merge_size = int(visual.spatial_merge_size)
    main_grids = [_llm_grid(grid, merge_size) for grid in main_grid_thw]
    aux_grids = [_llm_grid(grid, merge_size) for grid in aux_grid_thw]
    main_sizes = _split_sizes(main_grid_thw, merge_size)
    aux_sizes = _split_sizes(aux_grid_thw, merge_size)
    anchor_grid = main_grids[-1]
    decay = config.temporal_decay
    weights = [
        decay[min(max(int(age.item()), 1), len(decay)) - 1]
        for age in aux_ages
    ]
    if config.fusion_mode == "gated_residual" and residual_gate is None:
        raise ValueError("gated_residual fusion requires a TemporalResidualGate.")
    handle_box = {}

    def hook(module, module_inputs, output):
        handle_box["handle"].remove()
        main_flat, main_deepstack = output
        aux_pixels = aux_pixel_values.to(
            device=main_flat.device, dtype=module.dtype, non_blocking=True
        )
        aux_grids_device = aux_grid_thw.to(main_flat.device, non_blocking=True)
        aux_flat, aux_deepstack = module(aux_pixels, grid_thw=aux_grids_device)

        main_parts = list(torch.split(main_flat, main_sizes, dim=0))
        aux_parts = list(torch.split(aux_flat, aux_sizes, dim=0))
        resized_aux = [
            _resize_tokens(part, grid, anchor_grid)
            for part, grid in zip(aux_parts, aux_grids)
        ]
        matches = []
        similarities = []
        for source in resized_aux:
            selected, destination, similarity = _match_source_to_anchor(
                source,
                main_parts[-1],
                anchor_grid,
                config.merge_ratio,
                config.spatial_bias,
                config.matching_mode,
            )
            matches.append((selected, destination))
            if similarity.numel():
                similarities.append(similarity.mean())

        if config.fusion_mode == "weighted_mean":
            fused_flat = _fuse_level(
                main_flat,
                aux_flat,
                main_sizes,
                aux_sizes,
                main_grids,
                aux_grids,
                matches,
                weights,
                config.anchor_weight,
            )
            fused_deepstack = [
                _fuse_level(
                    main_level,
                    aux_level,
                    main_sizes,
                    aux_sizes,
                    main_grids,
                    aux_grids,
                    matches,
                    weights,
                    config.anchor_weight,
                )
                for main_level, aux_level in zip(main_deepstack, aux_deepstack)
            ]
        else:
            gate = residual_gate()
            fused_flat, residual, active = _residual_level(
                main_flat,
                aux_flat,
                main_sizes,
                aux_sizes,
                main_grids,
                aux_grids,
                matches,
                weights,
                gate,
            )
            fused_deepstack = [
                _residual_level(
                    main_level,
                    aux_level,
                    main_sizes,
                    aux_sizes,
                    main_grids,
                    aux_grids,
                    matches,
                    weights,
                    gate,
                )[0]
                for main_level, aux_level in zip(main_deepstack, aux_deepstack)
            ]
            if capture is not None:
                capture["current_residual"] = residual
                capture["active_mask"] = active
                capture["gate"] = gate
        if diagnostics is not None:
            diagnostics["aux_frames"] = float(len(aux_parts))
            diagnostics["anchor_tokens"] = float(main_sizes[-1])
            diagnostics["merged_tokens"] = float(
                sum(selected.numel() for selected, _ in matches)
            )
            diagnostics["mean_similarity"] = float(
                torch.stack(similarities).mean().item() if similarities else 0.0
            )
            diagnostics["unique_destinations"] = float(
                torch.cat([destination for _, destination in matches]).unique().numel()
                if any(destination.numel() for _, destination in matches)
                else 0
            )
            if config.fusion_mode == "gated_residual":
                diagnostics["gate"] = float(gate.detach().item())
                diagnostics["active_tokens"] = float(active.sum().item())
        return fused_flat, fused_deepstack

    handle_box["handle"] = visual.register_forward_hook(hook)
    return handle_box["handle"]
