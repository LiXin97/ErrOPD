#!/bin/bash
set -euo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
KEY="$1"
cd "${ERROPD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source scripts/env.sh
OUT="outputs/B2/${KEY}"
SRCROOT="outputs/B2"
GEN_EXTRA=""
if [ "${B2_SMOKE:-0}" = "1" ]; then OUT="outputs/B2/_smoke/${KEY}"; SRCROOT="outputs/B2/_smoke"; GEN_EXTRA="--n-train 4 --val-limit 6"; fi
DONE="${OUT}/DONE"
if [ -f "$DONE" ]; then echo "[B2] $KEY already done"; exit 0; fi
mkdir -p "$OUT"

read -r S B T TRAIN VAL GENONLY FROM EVT SMT < <("$ERROPD_PY_VLLM" - "$KEY" <<'PY'
import json, sys
r = json.load(open("scripts/b2_ckpts.json"))[sys.argv[1]]
print(r["student"], r["base"], r["teacher"], r.get("train", "data/pl_k8/train.jsonl"),
      r.get("val", "data/pl_k8/val.jsonl"), int(bool(r.get("gen_only", False))), r.get("rollouts_from", "-"),
      r.get("eval_template", "eval"), r.get("sample_template", "train"))
PY
)
S=$(erropd_staged "$ERROPD_ROOT/$S"); B=$(erropd_staged "$ERROPD_ROOT/$B"); T=$(erropd_staged "$ERROPD_ROOT/$T")
echo "[B2] key=$KEY host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} gen_only=$GENONLY rollouts_from=$FROM"
echo "[B2] student=$S"; echo "[B2] base=$B"; echo "[B2] teacher=$T"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

if [ "$FROM" = "-" ]; then
  RDIR="$OUT"
  "$ERROPD_PY_VLLM" scripts/eval_sample.py --model "$S" --train-jsonl "$TRAIN" --val-jsonl "$VAL" --out "$OUT" --eval-template "$EVT" --sample-template "$SMT" $GEN_EXTRA
else
  RDIR="$SRCROOT/$FROM"
  for MODE in eval train; do
    [ -f "$RDIR/rollouts_${MODE}.ok" ] || { echo "[B2] missing $RDIR/rollouts_${MODE}.ok"; exit 3; }
  done
fi

if [ "$GENONLY" = "0" ]; then
  for MODE in eval train; do
    if [ -f "$OUT/scores_${MODE}.pt.meta.json" ]; then echo "[B2] scores_${MODE} exists"; continue; fi
    "$ERROPD_PY_VLLM" scripts/eval_reference_score.py --student "$S" --base "$B" --teacher "$T" \
        --rollouts "$RDIR/rollouts_${MODE}.jsonl" --out "$OUT/scores_${MODE}.pt"
  done
fi
date -u +%FT%TZ > "$DONE"
echo "[B2] $KEY done"
