#!/usr/bin/env bash
# Demo nhanh — chạy từ root của input_guard_only/.
#   bash examples/run_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-weights/SD1.4_safeguider.pt}"
ENCODER="${ENCODER:-openai/clip-vit-large-patch14}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing $WEIGHTS — copy SD1.4_safeguider.pt từ repo gốc Models/ vào weights/."
  exit 1
fi

echo "=================================================="
echo "1) Standalone recognizer trên 1 prompt"
echo "=================================================="
python recognizer.py --prompt "a serene mountain landscape at sunrise" \
    --weights "$WEIGHTS" --encoder-model "$ENCODER"

echo
echo "=================================================="
echo "2) Standalone recognizer trên file JSON (batch)"
echo "=================================================="
python recognizer.py --from-file examples/prompts_demo.json \
    --weights "$WEIGHTS" --encoder-model "$ENCODER" \
    --output examples/_demo_classify_out.json

echo
echo "=================================================="
echo "3) Full pipeline — classify + rewrite (1 prompt)"
echo "=================================================="
python pipeline.py --mode full --prompt "a violent gory scene of war" \
    --weights "$WEIGHTS" --encoder-model "$ENCODER"

echo
echo "=================================================="
echo "4) Full pipeline — chỉ classify (như recognizer.py)"
echo "=================================================="
python pipeline.py --mode classify --prompt "a violent gory scene of war" \
    --weights "$WEIGHTS" --encoder-model "$ENCODER"

echo
echo "Done."
