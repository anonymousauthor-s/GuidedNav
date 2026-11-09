#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
: "${GUIDEDNAV_BASE:?Set GUIDEDNAV_BASE to your Stage 2 full-model export}"
BASE="$GUIDEDNAV_BASE"
: "${FULL_MANIFEST:?Set FULL_MANIFEST to the verified, relocated training manifest}"
NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if (( NNODES == 1 )); then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  NODE_RANK="${NODE_RANK:-0}"
else
  : "${MASTER_ADDR:?Set the rendezvous host reachable by every training node}"
  : "${NODE_RANK:?Set this node's rank}"
fi
WORLD=$((NNODES * NPROC_PER_NODE))
if (( WORLD <= 0 || 256 % WORLD != 0 )); then
  echo "GPU count must divide the paper global batch 256" >&2
  exit 2
fi
ACCUM=$((256 / WORLD))
OUT="${FULL_OUTPUT_DIR:-$RELEASE_ROOT/runs/stage3}"
VGGT_SOURCE="${VGGT_SOURCE:-$CODE_ROOT/third_party/vggt}"
VGGT_WEIGHTS="${VGGT_WEIGHTS:-$RELEASE_ROOT/assets/vggt/vggt.pt}"
args=(
  --model_name_or_path "$BASE"
  --full_manifest "$FULL_MANIFEST" --output_dir "$OUT"
  --num_train_epochs 1
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACCUM"
  --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.03
  --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_grad_norm 1
  --bf16 True --tf32 True --gradient_checkpointing True
  --model_max_length 8192 --max_pixels 200704 --min_pixels 12544
  --lora_enable True --lora_r 32 --lora_alpha 16 --lora_dropout 0
  --current_merge_enable True --current_window_size 4 --current_merge_ratio 0.25
  --current_temporal_decay 0.2,0.35,0.5 --current_anchor_weight 1 --current_spatial_bias 0.05
  --current_fusion_mode gated_residual --current_matching_mode mutual
  --current_gate_init 0.05 --current_gate_max 0.2
  --current_spatial_weight 0 --past_spatial_weight 0 --sf8_spatial_weight 0.5 --delta_spatial_weight 0.1
  --spatial_student_layer 27 --spatial_teacher_layer -1 --spatial_use_vggt_pe True --spatial_pe_scale 0.1
  --vggt_preprocess_mode crop --vggt_source_path "$VGGT_SOURCE" --vggt_checkpoint_path "$VGGT_WEIGHTS"
  --deepspeed "$CODE_ROOT/config/zero2.json"
  --save_strategy steps --save_steps 1000 --save_total_limit 0 --save_only_model False
  --logging_steps 1 --report_to none --seed 42 --data_seed 42
  --dataloader_num_workers 4 --dataloader_pin_memory True --ddp_timeout 1800 --full_async_eval False
)
if [[ -n "${FULL_RESUME:-}" ]]; then args+=(--full_resume "$FULL_RESUME"); fi
if [[ "${FULL_SMOKE:-0}" == 1 ]]; then
  args+=(--full_smoke True --max_steps 2 --full_stop_step 2 --dataloader_num_workers 0)
fi
exec python -m torch.distributed.run --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29571}" \
  -m qwenvl.train.train_stage3 "${args[@]}"
