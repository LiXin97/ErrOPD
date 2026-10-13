#!/bin/bash
set -euo pipefail
STAGE="${1:?base|A|0|stat}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
AS=outputs/D1inj/asym
B=$(erropd_staged "$REPO/outputs/D1/sft_llama3.2_3b")
gen() {
  local M=$1 O=$2
  if [ -f "$O/rollouts_train.ok" ] && [ -f "$O/rollouts_eval.ok" ]; then echo "[D1inj asym] $O rollouts exist"; return; fi
  mkdir -p "$O"
  "$ERROPD_PY_VLLM" scripts/eval_sample.py --model "$M" --train-jsonl data/pl_k8/train.jsonl \
      --val-jsonl data/pl_k8/val.jsonl --out "$O"
}
score() {
  local T=$1 R=$2 O=$3
  mkdir -p "$O"
  for MODE in train eval; do
    [ -f "$R/rollouts_${MODE}.ok" ] || { echo "[D1inj asym] missing $R rollouts_$MODE"; exit 3; }
    [ -f "$O/scores_${MODE}.pt.meta.json" ] && continue
    "$ERROPD_PY_VLLM" scripts/eval_reference_score.py --student "$B" --base "$B" --teacher "$T" \
        --rollouts "$R/rollouts_${MODE}.jsonl" --out "$O/scores_${MODE}.pt"
  done
}
case "$STAGE" in
  base) gen "$B" "$AS/base_roll" ;;
  A|0)
    if [ "$STAGE" = A ]; then TREL=outputs/D1inj/sft_t8b_k1; SELF=tk1_self; PK=pairA; else TREL=outputs/D1inj/sft_t8b_clean; SELF=t0_self; PK=pair0; fi
    [ -f "$TREL/DONE" ] || { echo "[D1inj asym] teacher $TREL not done"; exit 5; }
    T=$(erropd_staged "$REPO/$TREL")
    gen "$T" "$AS/$SELF"
    score "$T" "$AS/base_roll" "$AS/$PK"
    score "$T" "$AS/$SELF" "$AS/${PK}_fb" ;;
  stat) "$ERROPD_PY_VLLM" scripts/xfamily_d1inj_asym_stat.py ;;
  *) echo "bad stage"; exit 2 ;;
esac
touch "$AS/${STAGE}.DONE"
echo "[D1inj asym] stage $STAGE done"
