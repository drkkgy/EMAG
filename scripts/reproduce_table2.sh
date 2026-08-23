#!/usr/bin/env bash
# Reproduce the EMAG rows of Table 2 (SD3 text-to-image, COCO-2014 val).
#
# Generates images for EMAG and EMAG-Q, each as EMAG-only (combo No), EMAG+APG, and EMAG+CADS,
# then computes FID and HPS v2 for every config and writes a results CSV.
#
# Requirements: emag installed, plus hpsv2 and pytorch-fid (see requirements.txt), a GPU, and
# access to SD3-Medium (gated: run `huggingface-cli login`).
#
# Usage:
#   CAPTIONS=/data/coco/annotations/captions_val2014.json \
#   FID_REF=/data/coco/val2014 \
#   OUT=outputs/table2 \
#   NUM_SAMPLES=40000 \
#   bash scripts/reproduce_table2.sh
#
# Notes:
#   - Table 2 uses Pareto-selected per-config scales (paper §D). This script uses the recommended
#     default EMAG scale (EMAG_SCALE, default 1.5); override per row via EMAG_SCALE if reproducing
#     the exact Pareto points.
set -euo pipefail

: "${CAPTIONS:?Set CAPTIONS=/path/to/captions_val2014.json (or a .json list / .txt of prompts)}"
: "${FID_REF:?Set FID_REF=/path/to/coco/val2014 (real images) or a precomputed FID .npz}"
OUT="${OUT:-outputs/table2}"
NUM_SAMPLES="${NUM_SAMPLES:-40000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-8}"
STEPS="${STEPS:-28}"
CFG="${CFG:-7.0}"
EMAG_SCALE="${EMAG_SCALE:-1.5}"

# Run from the repo root so `import emag` and `python -m eval.*` resolve.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

RESULTS="$OUT/table2_results.csv"
mkdir -p "$OUT"
echo "config,method,combo_mode,emag_scale,n,fid,hps_mean,hps_median" > "$RESULTS"

# config_name  method   combo_mode
CONFIGS=(
  "EMAG          emag    No"
  "EMAG+APG      emag    APG"
  "EMAG+CADS     emag    CADS"
  "EMAG-Q        emag-q  No"
  "EMAG-Q+APG    emag-q  APG"
  "EMAG-Q+CADS   emag-q  CADS"
)

for cfg in "${CONFIGS[@]}"; do
  read -r NAME METHOD COMBO <<< "$cfg"
  RUN_DIR="$OUT/$NAME"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] Generating $NAME  (method=$METHOD combo=$COMBO scale=$EMAG_SCALE)"
  echo "=================================================================="

  python scripts/generate_coco.py \
    --captions "$CAPTIONS" \
    --out_dir "$RUN_DIR" \
    --method "$METHOD" \
    --combo_mode "$COMBO" \
    --emag_scale "$EMAG_SCALE" \
    --cfg_scale "$CFG" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --num_inference_steps "$STEPS" \
    --seed "$SEED"

  echo "[$(date +%H:%M:%S)] Scoring $NAME ..."
  FID=$(python -m eval.fid --run "$RUN_DIR" --ref "$FID_REF" | sed -n 's/.*FID=\([0-9.]*\).*/\1/p')
  HPS_LINE=$(python -m eval.hps --run "$RUN_DIR")
  HPS_MEAN=$(echo "$HPS_LINE" | sed -n 's/.*mean=\([0-9.]*\).*/\1/p')
  HPS_MEDIAN=$(echo "$HPS_LINE" | sed -n 's/.*median=\([0-9.]*\).*/\1/p')

  echo "$NAME,$METHOD,$COMBO,$EMAG_SCALE,$NUM_SAMPLES,$FID,$HPS_MEAN,$HPS_MEDIAN" >> "$RESULTS"
  echo "[$(date +%H:%M:%S)] $NAME -> FID=$FID  HPS_mean=$HPS_MEAN  HPS_median=$HPS_MEDIAN"
done

echo ""
echo "=== Table 2 (EMAG rows) reproduced -> $RESULTS ==="
cat "$RESULTS"
