#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
: "${GUIDEDNAV_START:?Set GUIDEDNAV_START to the upstream model or your preceding stage export}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to the prepared training manifest}"
OUT="${TRAIN_OUTPUT:-$RELEASE_ROOT/runs/stage1}"
NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MICROBATCH="${MICROBATCH:-2}"
if (( NNODES == 1 )); then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  NODE_RANK="${NODE_RANK:-0}"
else
  : "${MASTER_ADDR:?Set the rendezvous host reachable by all training nodes}"
  : "${NODE_RANK:?Set this node's rank}"
fi
BATCH_DIVISOR=$((NNODES * NPROC_PER_NODE * MICROBATCH))
if (( BATCH_DIVISOR <= 0 || 256 % BATCH_DIVISOR != 0 )); then
  echo "GPU count times MICROBATCH must divide the paper global batch 256" >&2
  exit 2
fi
ACCUM=$((256 / BATCH_DIVISOR))
args=(
  --stage 1 --model_name_or_path "$GUIDEDNAV_START"
  --training_manifest "$TRAIN_MANIFEST" --output_dir "$OUT"
  --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision True --lora_enable False
  --num_train_epochs 1 --per_device_train_batch_size "$MICROBATCH" --gradient_accumulation_steps "$ACCUM"
  --learning_rate 2e-5 --mm_projector_lr 5e-6 --vision_tower_lr 5e-6
  --lr_scheduler_type cosine --warmup_ratio 0.03 --weight_decay 0.01
  --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_grad_norm 1
  --bf16 True --tf32 True --gradient_checkpointing True
  --model_max_length 8192 --max_pixels 200704 --min_pixels 12544
  --video_min_frames 4 --video_max_frames 8 --video_fps 2
  --deepspeed "$CODE_ROOT/config/zero2.json"
  --save_strategy steps --save_steps 200 --save_total_limit 2 --save_only_model False
  --logging_steps 1 --report_to none --seed 42 --data_seed 42
  --dataloader_num_workers 4 --dataloader_pin_memory True --ddp_timeout 1800
)
if [[ -n "${TRAIN_RESUME:-}" ]]; then args+=(--resume_checkpoint "$TRAIN_RESUME"); fi
if [[ "${TRAIN_SMOKE:-0}" == 1 ]]; then args+=(--smoke True --max_steps 2 --dataloader_num_workers 0); fi
exec python -m torch.distributed.run --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29571}" \
  -m qwenvl.train.train_navigation_pretrain "${args[@]}"
