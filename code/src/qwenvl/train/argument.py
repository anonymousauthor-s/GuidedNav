import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2
    
    samples_name: str = field(
        default="samples_perstep.jsonl",
        metadata={"help": "Per-step sample file under each VLN video_folder."},
    )

    # transform_train is an optional callable injected at runtime (not via CLI).
    transform_train: Optional[object] = field(default=None)


@dataclass
class WaypointArguments:
    predict_size: int = field(default=8)
    memory_size: int = field(default=8)
    waypoint_dim: int = field(default=4)
    quat_format: str = field(default="wxyz")

    # DiT architecture
    token_dim: int = field(default=512)
    dit_depth: int = field(default=8)
    dit_heads: int = field(default=8)
    dit_mlp_ratio: float = field(default=4.0)
    qk_norm: bool = field(default=True)
    rms_norm_eps: float = field(default=1e-6)

    # flow-matching objective
    num_train_timesteps: int = field(default=10)
    # one of: flow_matching | hyper_flow_matching (arXiv:2602.22801) | mlp_policy
    trajectory_objective: str = field(default="hyper_flow_matching")
    hybrid_waypoint_weight: float = field(default=0.1)
    flow_loss_weight: float = field(default=1.0)

    # Arrive/stop head: the <stop> hidden feeds a per-waypoint classifier
    # (predict_size logits, OmniNav's arrive_list). Labels come from the data
    # (``arrive_list`` in samples_perstep.jsonl, keyed on each waypoint step's
    # distance to the expert endpoint under build_vln_samples' --stop_radius), so
    # there is nothing to tune here beyond how the stop objective is weighed
    # against flow/text in the multi-task sum. The old class-rebalancing knobs
    # (stop_oversample_ratio / stop_pos_weight) are gone: per-waypoint flags put
    # the positive rate near 22% of (frame, step) pairs instead of the ~2% of the
    # single-terminal-flag scheme, so rebalancing is unnecessary — and row
    # oversampling actively harmed the waypoint target, whose approach-zone rows
    # carry near-zero motion.
    stop_loss_weight: float = field(default=1.0)

    # Waypoint normalization is ALWAYS fitted from the train split at startup and
    # persisted into the checkpoint (waypoint_norm_scale/offset are registered
    # buffers, so they ride along in state_dict and the evaluator reads them back
    # from the checkpoint). There is deliberately no precomputed-json knob: the
    # norm must describe the data this run actually trains on, and a stale or
    # mismatched json silently rescales every waypoint target. norm_fit_cap bounds
    # the fit cost (positions/rotations only, no images).
    norm_fit_cap: int = field(default=50000)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
    subtask_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight applied only to SUBTASK answer tokens in the FGR2R subtask dataset."
        },
    )
    phase_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight applied only to PHASE token prefix in phase-token VLN data."
        },
    )
    init_flow_head_from: str = field(default="")
