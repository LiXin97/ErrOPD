#!/bin/bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
cd "${ERROPD_ROOT:?}"
source scripts/env.sh
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
O=outputs/E3/samples
mkdir -p "$O"
IFS=',' read -r -a G <<< "${CUDA_VISIBLE_DEVICES:-0}"
gen() {
  local d="$O/$2"
  [ -f "$d/rollouts_train.ok" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 "$ERROPD_PY_VLLM" scripts/eval_sample.py --model "$(erropd_staged "$ERROPD_ROOT/$3")" \
      --train-jsonl data/esci_k8/train.jsonl --val-jsonl data/esci_k8/val.jsonl --out "$d" --modes train \
      --n-train 0 --n-val-samples 4 --sample-template train --vllm-seed 0 --max-new-tokens 512 > "$d.log" 2>&1
}
i=0; pids=()
for a in "$@"; do
  n=${a%%=*}; p=${a#*=}
  mkdir -p "$O/$n"
  gen "${G[$(( i % ${#G[@]} ))]}" "$n" "$p" & pids+=($!)
  i=$((i + 1))
  if (( ${#pids[@]} >= ${#G[@]} )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait "$p"; done
rc=0
for a in "$@"; do n=${a%%=*}; [ -f "$O/$n/rollouts_train.ok" ] || { echo "[e3_sample] missing $n"; rc=4; }; done
[ $rc = 0 ] && date -u +%FT%TZ > "$O/${SAMPLE_DONE:-DONE}"
exit $rc
