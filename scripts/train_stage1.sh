#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/stage1_$(date -u +%Y%m%dT%H%M%SZ)"}

need() {
  if [[ -z "${!1:-}" ]]; then
    echo "Missing required environment variable: $1" >&2
    exit 2
  fi
}
need QWEN_PATH
need UNI_PATH
need STAGE1_ANNOTATION
need STAGE1_IMAGE_ROOT

mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/src/train_uni_qwen35_stage1.py" \
  --qwen-path "$QWEN_PATH" \
  --uni-path "$UNI_PATH" \
  --annotation-file "$STAGE1_ANNOTATION" \
  --image-root "$STAGE1_IMAGE_ROOT" \
  --output-dir "$RUN_DIR" \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --epochs "${EPOCHS:-1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --max-val-samples "${MAX_VAL_SAMPLES:-512}" \
  --eval-steps "${EVAL_STEPS:-5000}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  --log-steps "${LOG_STEPS:-10}" \
  --lr "${LR:-1e-4}" \
  --seed "${SEED:-3407}"
