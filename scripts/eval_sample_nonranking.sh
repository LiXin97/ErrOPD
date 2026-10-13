#!/bin/bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
: "${NR_TASK:?t3|t2}"
cd "${ERROPD_ROOT:-/path/to/erropd}"
REPO=$(pwd)
source scripts/env.sh
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
O=${NR_SAMPLE_ROOT:-outputs/NR/samples}/$NR_TASK
mkdir -p "$O"
IFS=',' read -r -a G <<< "${CUDA_VISIBLE_DEVICES:-0}"
gen() {
  local d="$O/$2"
  [ -f "$d/DONE" ] && return 0
  CUDA_VISIBLE_DEVICES=$1 "$ERROPD_PY_VLLM" scripts/eval_sample_nonranking.py --task "$NR_TASK" \
      --model "$(erropd_staged "$REPO/$3")" --out "$d" ${NR_SAMPLE_ARGS:-} > "$d.log" 2>&1
  grep "nr_sample" "$d.log" | cut -c1-300
}
i=0; pids=()
for a in "$@"; do
  n=${a%%=*}; p=${a#*=}
  [ -f "$REPO/$p/config.json" ] || { echo "[nr_sample.sh] missing $p/config.json"; exit 5; }
  gen "${G[$(( i % ${#G[@]} ))]}" "$n" "$p" & pids+=($!)
  i=$((i + 1))
  if (( ${#pids[@]} >= ${#G[@]} )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait "$p"; done
rc=0
for a in "$@"; do n=${a%%=*}; [ -f "$O/$n/DONE" ] || { echo "[nr_sample.sh] missing $n (see $O/$n.log)"; rc=4; }; done
[ $rc = 0 ] && date -u +%FT%TZ > "$O/${SAMPLE_DONE:-DONE}"
exit $rc
