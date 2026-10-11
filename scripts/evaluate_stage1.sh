#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
OUTPUT=${OUTPUT:-"$ROOT/results/stage1_fixed_eval.json"}
for name in QWEN_PATH UNI_PATH STAGE1_ANNOTATION STAGE1_IMAGE_ROOT STAGE1_CHECKPOINT; do
  [[ -n "${!name:-}" ]] || { echo "Missing $name" >&2; exit 2; }
done
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$OUTPUT")"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  "$ROOT/src/diagnose_uni_qwen35.py" \
  --qwen-path "$QWEN_PATH" --uni-path "$UNI_PATH" \
  --annotation-file "$STAGE1_ANNOTATION" --image-root "$STAGE1_IMAGE_ROOT" \
  --checkpoint-dir "$(dirname "$STAGE1_CHECKPOINT")" --output "$OUTPUT" \
  --candidate native --candidate "step30000=$STAGE1_CHECKPOINT"
