#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "🚀 SEMEVAL-2027 TASK 3: PRODUCTION PIPELINE (SV DEV)"
echo "============================================================"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Virtual environment not found at .venv."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT"
mkdir -p submissions experiments

rm -rf experiments/production_champion

"$VENV_PYTHON" scripts/run_production_pipeline.py --data-root data/raw --languages SV --output-dir experiments/production_champion --submission-zip submissions/champion_submission.zip

echo "============================================================"
echo "✅ PURE SV SUBMISSION PACKAGE READY FOR CODABENCH AT:"
echo "   $PROJECT_ROOT/submissions/champion_submission.zip"
echo "============================================================"
