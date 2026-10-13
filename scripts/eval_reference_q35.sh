#!/usr/bin/env bash
set -euo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
KEY="$1"
cd "${ERROPD_ROOT:-/path/to/erropd}"
source scripts/env_q35.sh
OUT="outputs/B2/${KEY}"
[ -f "$OUT/DONE" ] && { echo "[d6-b2] $KEY already done"; exit 0; }
mkdir -p "$OUT"
read -r S B T GENONLY FROM < <(python3 - "$KEY" <<'PY'
import json, sys
r = json.load(open("scripts/qwen35_b2_registry.json"))[sys.argv[1]]
print(r["student"], r["base"], r["teacher"], int(bool(r.get("gen_only", False))), r.get("rollouts_from", "-"))
PY
)
S=$(erropd_staged "$ERROPD_ROOT/$S"); B=$(erropd_staged "$ERROPD_ROOT/$B"); T=$(erropd_staged "$ERROPD_ROOT/$T")
echo "[d6-b2] key=$KEY host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} gen_only=$GENONLY rollouts_from=$FROM score_py=$Q35_PY"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ "$FROM" = "-" ]; then
  RDIR="$OUT"
  "$ERROPD_PY_VLLM" scripts/eval_sample.py --model "$S" --train-jsonl data/pl_k8/train.jsonl \
      --val-jsonl data/pl_k8/val.jsonl --out "$OUT"
else
  RDIR="outputs/B2/$FROM"
  for MODE in eval train; do [ -f "$RDIR/rollouts_${MODE}.ok" ] || { echo "[d6-b2] missing $RDIR/rollouts_${MODE}.ok"; exit 3; }; done
fi
if [ "$GENONLY" = "0" ]; then
  for MODE in eval train; do
    [ -f "$OUT/scores_${MODE}.pt.meta.json" ] && { echo "[d6-b2] scores_${MODE} exists"; continue; }
    "$Q35_PY" scripts/eval_reference_score.py --student "$S" --base "$B" --teacher "$T" \
        --rollouts "$RDIR/rollouts_${MODE}.jsonl" --out "$OUT/scores_${MODE}.pt"
  done
fi
date -u +%FT%TZ > "$OUT/DONE"
echo "[d6-b2] $KEY done"
