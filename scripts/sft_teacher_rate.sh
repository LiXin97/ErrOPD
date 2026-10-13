#!/bin/bash
set -euo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
WHICH="${1:?t4b_k1_f50 | t4b_k1_f80}"
REPO="${ERROPD_ROOT:-/path/to/erropd}"
cd "$REPO"
source scripts/env.sh
MODELS=/path/to/models
case "$WHICH" in
  t4b_k1_f50) CFG=configs/sft_qwen3_4b_esci_f2.yaml; BASE=$MODELS/Qwen3-4B; SIZE=4b; PDB=2; DS=esci_k8_k1_f50_x5; FN=k1_f50_x5.jsonl ;;
  t4b_k1_f80) CFG=configs/sft_qwen3_4b_esci_f2.yaml; BASE=$MODELS/Qwen3-4B; SIZE=4b; PDB=2; DS=esci_k8_k1_f80_x5; FN=k1_f80_x5.jsonl ;;
  *) echo "bad WHICH=$WHICH"; exit 2 ;;
esac
OUT=outputs/QSWEEP/sft_${WHICH}
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[qsweep_sft] $WHICH already DONE"; exit 0; }
[ -f data/esci_k8_f3/dataset_info.json ] || { echo "[qsweep_sft] data/esci_k8_f3 missing"; exit 2; }
"$ERROPD_PY_VLLM" - data/esci_k8_f3 "$FN" "$DS" <<'EOF' || exit 2
import hashlib, json, sys
d, fn, ds = sys.argv[1:]
want = json.load(open(f"{d}/manifest.json"))["files"][fn]["sha256"]
got = hashlib.sha256(open(f"{d}/{fn}", "rb").read()).hexdigest()
assert got == want, f"{fn}: sha256 {got} != manifest {want}"
assert json.load(open(f"{d}/dataset_info.json"))[ds]["file_name"] == fn, ds
print(f"[qsweep_sft] {fn} sha256 {got[:12]} matches the manifest; dataset {ds}")
EOF
NGPUS=${NGPUS:-8}
EFF=128
(( EFF % (PDB * NGPUS) == 0 )) || { echo "[qsweep_sft] 128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( EFF / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")
ts() { date -u +%FT%TZ; }

if [ ! -f "$OUT/all_results.json" ]; then
  PY_SFT=${ERROPD_PY_SFT:-}
  [ -n "$PY_SFT" ] || { echo "[qsweep_sft] ERROPD_PY_SFT not set"; exit 3; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LF_SRC=/path/to/LlamaFactory/src
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/qsweep_sft_$WHICH}/hf_datasets"
  PORT=$(( 20000 + RANDOM % 20000 ))
  echo "[qsweep_sft] $(ts) train $WHICH base=$BASE_M data=$DS NGPUS=$NGPUS per_device=$PDB GA=$GA -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
      model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA" \
      dataset="$DS" dataset_dir="$REPO/data/esci_k8_f3"
  [ -f "$OUT/all_results.json" ] || { echo "[qsweep_sft] no all_results.json"; exit 4; }
  "$ERROPD_PY_VLLM" - "$OUT" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + "/trainer_state.json"))
print(f"[qsweep_sft] max_steps={s['max_steps']} global_step={s['global_step']}")
EOF
fi
cp "$MODELS/Qwen3-${SIZE^^}/tokenizer_config.json" "$OUT/tokenizer_config.json"
date -u +%FT%TZ > "$DONE"
echo "[qsweep_sft] $WHICH done"
