#!/usr/bin/env bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
cd "${ERROPD_ROOT:-/path/to/erropd}"
source scripts/env_q35.sh
O=${SAMPLE_DIR:-outputs/Q35S/samples}
mkdir -p "$O"
DONE="$O/${DONE_NAME:?DONE_NAME}"
[ -f "$DONE" ] && { echo "[q35s_sample] $DONE exists"; exit 0; }
NGPUS=${NGPUS:-4}; TP=${TP:-$(( NGPUS >= 4 ? 4 : NGPUS ))}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for a in "$@"; do
  n=${a%%=*}; p=${a#*=}; [[ "$p" = /* ]] || p=$ERROPD_ROOT/$p
  [ -f "$O/$n.v2.json" ] && { echo "[q35s_sample] $n exists"; continue; }
  M=$(erropd_staged "$p")
  [ -f "$M/.d6_template_sha256" ] || { echo "[q35s_sample] $M has no installed D6 template"; exit 3; }
  echo "[q35s_sample] $(date -u +%FT%TZ) $n <- $M (tp=$TP)"
  "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py --model-path "$M" --family qwen --tp "$TP" \
      --out "$O/$n" > "$O/$n.log" 2>&1
  grep -E "^\[sample_validity\]" "$O/$n.log"
  [ -f "$O/$n.v2.json" ] || { echo "[q35s_sample] missing $O/$n.v2.json"; tail -20 "$O/$n.log"; exit 4; }
done
date -u +%FT%TZ > "$DONE"
