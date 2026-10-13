#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
O=outputs/D1inj/fkl_samples
mkdir -p "$O"
DONE="$O/${DONE_NAME:?DONE_NAME}"
[ -f "$DONE" ] && { echo "[fkl_sample] $DONE exists"; exit 0; }
NGPUS=${NGPUS:-2}; TP=$(( NGPUS >= 4 ? 4 : NGPUS ))
for a in "$@"; do
  n=${a%%=*}; p=${a#*=}
  "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py --model-path "$(erropd_staged "$REPO/$p")" --family llama \
      --tp "$TP" --out "$O/$n"
  [ -f "$O/$n.v2.json" ] || { echo "[fkl_sample] missing $O/$n.v2.json"; exit 4; }
done
date -u +%FT%TZ > "$DONE"
