#!/bin/bash
set -euo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
WHICH="${1:?t8b_dup}"
REPO="${ERROPD_ROOT:-/path/to/erropd}"
cd "$REPO"
source scripts/env.sh
case "$WHICH" in
  t8b_dup)   DS=pl_k8_dup_f20_x5 ;;
  *) echo "bad WHICH=$WHICH"; exit 2 ;;
esac
CFG=configs/sft_llama3.1_8b_pl_k8_d1inj.yaml
BASE=/path/to/models/Llama-3.1-8B-Instruct
OUT=outputs/DUPL/sft_${WHICH}
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[dupl_sft] $WHICH already DONE"; exit 0; }
[ -f data/pl_k8_dupl/dataset_info.json ] || { echo "[dupl_sft] data/pl_k8_dupl missing (data_inject_fashion_dup.py)"; exit 2; }
NGPUS=${NGPUS:-8}; PDB=2
(( 128 % (PDB * NGPUS) == 0 )) || { echo "128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( 128 / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")
TP=4; (( NGPUS < 4 )) && TP=$NGPUS
ts() { date -u +%FT%TZ; }

if [ ! -f "$OUT/all_results.json" ]; then
  PY_SFT=${ERROPD_PY_SFT:?ERROPD_PY_SFT not set}
  rm -rf "$OUT"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LF_SRC=/path/to/LlamaFactory/src
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/dupl_sft_$WHICH}/hf_datasets"
  echo "[dupl_sft] $(ts) train $WHICH base=$BASE_M data=$DS NGPUS=$NGPUS per_device=$PDB GA=$GA -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port $(( 20000 + RANDOM % 20000 )) "$LAUNCHER" \
      "$REPO/$CFG" model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA" \
      dataset="$DS" dataset_dir="$REPO/data/pl_k8_dupl"
  [ -f "$OUT/all_results.json" ] || { echo "[dupl_sft] no all_results.json"; exit 4; }
  "$ERROPD_PY_VLLM" -c "import json; s=json.load(open('$OUT/trainer_state.json')); print('[dupl_sft] max_steps', s['max_steps'], 'global_step', s['global_step'], 'train_batch_size', s.get('train_batch_size'))"
fi
rm -rf "$OUT"/checkpoint-*
"$ERROPD_PY_VLLM" scripts/common_llama_template.py install --model-dir "$OUT" --base-dir "$BASE_M"
"$ERROPD_PY_VLLM" scripts/common_llama_template.py verify --model-dir "$OUT" --data data/pl_k8/val.jsonl data/pl_k8/train.jsonl
"$ERROPD_PY_VERL" scripts/common_llama_template.py verify --model-dir "$OUT" --data data/pl_k8/val.jsonl
meta=$(printf '{"id":"DUPL_sft_%s","experiment":"DUPL","task":"fashion_k8","student":"llama3.1-8b-%s","teacher":"none","lambda":null,"seed":42,"step":null,"budget_steps":null,"variant":"sft","clip_c":null,"notes":"DUPL teacher SFT (%s); one epoch over five copies; prompt = SFT rendering, token ids"}' "$WHICH" "${WHICH#t8b_}" "$DS")
"$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py --model-path "$OUT" --family llama --tp "$TP" \
    --output-dir "outputs/DUPL/eval/sft_$WHICH" --meta "$meta"
"$ERROPD_PY_VLLM" scripts/eval_sample_validity.py --model-path "$OUT" --family llama --tp "$TP" \
    --out "outputs/DUPL/sampled_validity/$WHICH"
date -u +%FT%TZ > "$DONE"
echo "[dupl_sft] $(ts) $WHICH done"
