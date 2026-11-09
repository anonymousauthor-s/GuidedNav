#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ROOT="$(cd -- "$CODE_ROOT/.." && pwd)"
export PYTHONPATH="$CODE_ROOT/src:${PYTHONPATH:-}"
export GUIDEDNAV_BASE="${GUIDEDNAV_BASE:-$RELEASE_ROOT/weights/navigation-base}"
export GUIDEDNAV_MODEL="${GUIDEDNAV_MODEL:-$RELEASE_ROOT/weights/stage3-epoch1}"
export GUIDEDNAV_WEIGHTS_SOURCE="${GUIDEDNAV_WEIGHTS_SOURCE:-released}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TASK="${1:-r2r}"
if [[ "$TASK" != r2r && "$TASK" != rxr ]]; then echo 'Usage: evaluate.sh r2r|rxr [extra evaluator arguments]' >&2; exit 2; fi
if [[ $# -gt 0 ]]; then shift; fi
cd -- "$RELEASE_ROOT"
export GUIDEDNAV_SCENES_DIR="${GUIDEDNAV_SCENES_DIR:-$RELEASE_ROOT/data/scene_datasets}"
export VLN_DATA_PATH="${VLN_DATA_PATH:-$RELEASE_ROOT/data/datasets/r2r/val_unseen/val_unseen.json.gz}"
if [[ "$TASK" == rxr ]]; then
  export RXR_EVAL_DATA_PATH="${RXR_EVAL_DATA_PATH:-$RELEASE_ROOT/data/datasets/rxr/val_unseen/val_unseen_guide.json.gz}"
  export RXR_EVAL_GT_PATH="${RXR_EVAL_GT_PATH:-$RELEASE_ROOT/data/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz}"
  export RXR_NAVMESH_AUDIT_PATH="${RXR_NAVMESH_AUDIT_PATH:-$RELEASE_ROOT/data/rxr_navmesh/audit.json}"
  if [[ ! -f "$RXR_NAVMESH_AUDIT_PATH" ]]; then echo 'Build the audited RxR navmeshes as described in DATA.md before evaluation.' >&2; exit 2; fi
  python "$CODE_ROOT/tools/verify_rxr_assets.py" --dataset "$RXR_EVAL_DATA_PATH" --gt "$RXR_EVAL_GT_PATH" --audit "$RXR_NAVMESH_AUDIT_PATH"
fi
case "$GUIDEDNAV_WEIGHTS_SOURCE" in
  released) python "$CODE_ROOT/tools/verify_model_identity.py" --base "$GUIDEDNAV_BASE" --checkpoint "$GUIDEDNAV_MODEL" ;;
  trained) python "$CODE_ROOT/tools/verify_model_identity.py" --base "$GUIDEDNAV_BASE" --trained-base ;;
  *) echo 'GUIDEDNAV_WEIGHTS_SOURCE must be released or trained' >&2; exit 2 ;;
esac
exec python -m torch.distributed.run --nproc_per_node "${EVAL_WORKERS:-1}" \
  -m qwenvl.eval.eval_discrete_action --model_path "$GUIDEDNAV_MODEL" \
  --habitat_config_path "$CODE_ROOT/config/vln_${TASK}.yaml" --eval_split val_unseen \
  --output_path "${EVAL_OUTPUT:-$RELEASE_ROOT/results/$TASK}" \
  --num_history 8 --num_future_steps 4 --num_exec 4 --max_pixels 200704 --min_pixels 12544 "$@"
