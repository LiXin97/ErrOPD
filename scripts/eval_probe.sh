#!/bin/bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
cd "${ERROPD_ROOT:?}"
source scripts/env.sh
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
SET=$1; CHK=$2; TAG=$3; shift 3
args=()
for a in "$@"; do n=${a%%=*}; p=${a#*=}; args+=("$n=$(erropd_staged "$ERROPD_ROOT/$p")"); done
"$ERROPD_PY_VLLM" scripts/eval_probe.py --set "$SET" --check-against-b2 "$CHK" "${args[@]}" || exit 4
for a in "$@"; do n=${a%%=*}; [ -f "outputs/E3/probe/$SET/$n.json" ] || { echo "missing $n"; exit 4; }; done
date -u +%FT%TZ > "outputs/E3/probe/$SET/DONE_$TAG"
