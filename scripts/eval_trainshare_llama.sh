#!/bin/bash
set -euo pipefail
PAIR="${1:?A|0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
OUTD=outputs/D1inj/trainshare_proxy
DONE="$REPO/$OUTD/pair$PAIR.DONE"
[ -f "$DONE" ] && { echo "[d1inj proxy] pair $PAIR done"; exit 0; }
NGPUS=${NGPUS:-4}; TP=4; (( NGPUS < 4 )) && TP=$NGPUS
for lam in 1.40 1.00; do for s in 1001 1002 1003; do
  M="$REPO/outputs/D1inj/opd/${PAIR}_lam${lam}_s${s}/ckpt/global_step_42/hf"
  [ -f "$M/config.json" ] || { echo "[d1inj proxy] missing $M"; exit 5; }
  "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py --model-path "$M" --family llama --tp "$TP" \
      --out "$OUTD/${PAIR}_lam${lam}_s${s}_step42"
done; done
touch "$DONE"
