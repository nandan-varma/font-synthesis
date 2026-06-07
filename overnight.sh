#!/usr/bin/env bash
# overnight.sh — extract data, train, generate a proof SVG
set -euo pipefail

mkdir -p logs img model/data/raw model/data/processed model/checkpoint

echo "=== Handwriting Synthesis — Overnight Training ==="
echo ""

# 1. MPS check
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('MPS (Apple Metal GPU) detected — M4 acceleration enabled')
else:
    print('WARNING: MPS not available, training on CPU (will be slow)')
"

# 2. Extract xml.tgz.zip → original/ (writer ID metadata)
python3 scripts/prepare_data.py --extract-only

# 3. Check for required IAM stroke data
if [ ! -d "model/data/raw/lineStrokes" ] || [ ! -d "model/data/raw/ascii" ]; then
    echo ""
    echo "ERROR: IAM On-Line stroke data not found."
    echo ""
    echo "Register (free) at:"
    echo "  https://fki.tic.teia.ch/databases/iam-on-line-handwriting-database"
    echo ""
    echo "Download these two archives and place them in the project root:"
    echo "  lineStrokes-all.tar.gz   (~100 MB)"
    echo "  ascii-all.tar.gz         (~5 MB)"
    echo ""
    echo "Then re-run:  bash overnight.sh"
    exit 1
fi

# 4. Prepare training data (idempotent)
if [ ! -f "model/data/processed/x.npy" ]; then
    echo ""
    echo "--- Preparing training data ---"
    python3 scripts/prepare_data.py
fi

echo ""
echo "--- Starting training ---"
LOG="logs/train_$(date +%Y%m%d_%H%M).log"
echo "Log: $LOG"
echo ""

python3 -m handwriting.train 2>&1 | tee "$LOG"

echo ""
echo "--- Generating proof SVG ---"
python3 main.py "The quick brown fox jumps over the lazy dog" --style 0 -o img/overnight_result.svg
echo "Done. Open img/overnight_result.svg to inspect the result."
