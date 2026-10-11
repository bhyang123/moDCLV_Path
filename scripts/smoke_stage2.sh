#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/smoke_stage2"}
for name in QWEN_PATH UNI_PATH FUSION_CHECKPOINT STAGE2_ANNOTATION STAGE2_IMAGE_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "Missing $name" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  "$ROOT/src/train_uni_qwen35_stage2_lora.py" \
  --qwen-path "$QWEN_PATH" --uni-path "$UNI_PATH" \
  --fusion-checkpoint "$FUSION_CHECKPOINT" \
  --annotation-file "$STAGE2_ANNOTATION" --image-root "$STAGE2_IMAGE_ROOT" \
  --output-dir "$RUN_DIR" --batch-size 4 --grad-accum-steps 2 \
  --max-steps 2 --max-train-samples 8 --max-val-samples 4 \
  --eval-steps 0 --save-steps 0 --log-steps 1 --num-workers 1
