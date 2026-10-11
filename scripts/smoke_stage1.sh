#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/smoke_stage1"}
for name in QWEN_PATH UNI_PATH STAGE1_ANNOTATION STAGE1_IMAGE_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "Missing $name" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  "$ROOT/src/train_uni_qwen35_stage1.py" \
  --qwen-path "$QWEN_PATH" --uni-path "$UNI_PATH" \
  --annotation-file "$STAGE1_ANNOTATION" --image-root "$STAGE1_IMAGE_ROOT" \
  --output-dir "$RUN_DIR" --max-train-samples 4 --max-val-samples 2 \
  --max-steps 2 --batch-size 1 --grad-accum-steps 1 --num-workers 0 \
  --eval-steps 0 --save-steps 1 --log-steps 1
