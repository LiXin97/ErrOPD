#!/bin/bash
set -euo pipefail
ROLE="${1:?student|teacher|teacher8b}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
MODELS=/path/to/models
case "$ROLE" in
  student)   CFG=configs/sft_qwen3_1.7b_t3_ner.yaml; BASE=$MODELS/Qwen3-1.7B; SIZE=1.7b; PDB=8 ;;
  teacher)   CFG=configs/sft_qwen3_4b_t3_ner.yaml;   BASE=$MODELS/Qwen3-4B;   SIZE=4b;   PDB=4 ;;
  teacher8b) CFG=configs/sft_qwen3_4b_t3_ner.yaml;   BASE=$MODELS/Qwen3-8B;   SIZE=8b;   PDB=2 ;;
  *) echo "ROLE must be student|teacher|teacher8b"; exit 2 ;;
esac
OUT=outputs/T3/sft_qwen3_${SIZE}
EPOCHS_DIR=${OUT}_epochs
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[sft_ner_clean] $ROLE already DONE"; exit 0; }
[ -f data/erropd_t3/lf/dataset_info.json ] || { echo "[sft_ner_clean] data/erropd_t3 missing (scripts/data_build_conll.py)"; exit 2; }

NGPUS=${NGPUS:-8}
EFF=128
(( EFF % (PDB * NGPUS) == 0 )) || { echo "[sft_ner_clean] 128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( EFF / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")
ts() { date -u +%FT%TZ; }

if [ ! -f "$OUT/all_results.json" ]; then
  PY_SFT=${ERROPD_PY_SFT:-}
  [ -n "$PY_SFT" ] || { echo "[sft_ner_clean] ERROPD_PY_SFT not set (spec needs \"needs\": [\"sft\"])"; exit 3; }
  rm -rf "$OUT" "$EPOCHS_DIR"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LF_SRC=/path/to/LlamaFactory/src
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/t3_sft_$ROLE}/hf_datasets"
  PORT=$(( 20000 + RANDOM % 20000 ))
  echo "[sft_ner_clean] $(ts) train $ROLE base=$BASE_M NGPUS=$NGPUS per_device=$PDB GA=$GA -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
      model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA" \
      per_device_train_batch_size="$PDB" per_device_eval_batch_size="$PDB" dataset_dir="$REPO/data/erropd_t3/lf"
  [ -f "$OUT/all_results.json" ] || { echo "[sft_ner_clean] training finished without all_results.json"; exit 4; }
fi
"$ERROPD_PY_VLLM" - "$OUT" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + "/trainer_state.json"))
print(f"[sft_ner_clean] max_steps={s['max_steps']} global_step={s['global_step']} train_batch_size={s.get('train_batch_size')}")
print("[sft_ner_clean] eval_loss per epoch:", [round(x["eval_loss"], 4) for x in s["log_history"] if "eval_loss" in x])
assert s["max_steps"] == 75, "expected 75 optimizer steps (effective batch 128, 5 epochs, 1830 prompts)"
EOF

shopt -s nullglob
CKPTS=("$OUT"/checkpoint-*)
shopt -u nullglob
if (( ${#CKPTS[@]} > 0 )); then
  mkdir -p "$EPOCHS_DIR"
  for c in "${CKPTS[@]}"; do
    step=${c##*-}; ep=$(( step / 15 ))
    if [ "$step" = 75 ]; then rm -rf "$c"; else rm -rf "$EPOCHS_DIR/ep${ep}"; mv "$c" "$EPOCHS_DIR/ep${ep}"; fi
  done
fi

DIRS=("$OUT")
if [ -d "$EPOCHS_DIR" ]; then for d in "$EPOCHS_DIR"/ep*; do DIRS+=("$d"); done; fi
for d in "${DIRS[@]}"; do
  cp "$BASE/tokenizer_config.json" "$d/tokenizer_config.json"
  "$ERROPD_PY_VLLM" - "$BASE/tokenizer_config.json" "$d" <<'EOF'
import json, os, sys
ct = json.load(open(sys.argv[1]))["chat_template"]
p = os.path.join(sys.argv[2], "chat_template.jinja")
if os.path.exists(p):
    open(p, "w").write(ct)
EOF
done
if [ ! -f "$OUT/template_check.json" ]; then
  PYTHONPATH="$ERROPD_VERL_PROD:${PYTHONPATH:-}" "$ERROPD_PY_VERL" scripts/check_ner_template.py \
      --tok-dir "$OUT" --base-dir "$BASE_M" --out "$OUT/template_check.json.tmp"
  mv "$OUT/template_check.json.tmp" "$OUT/template_check.json"
fi

TAG=qwen3-$SIZE
ev() {
  local md=$1 ep=$2
  local meta
  meta=$(printf '{"id":"T3_sft_%s_ep%s","experiment":"T3","task":"conll03_ner_json","student":"%s","teacher":"none","lambda":null,"seed":42,"step":%s,"budget_steps":75,"variant":"sft","clip_c":null,"notes":"SFT epoch %s of 5 (effective batch 128); prompt = qwen3_nothink rendering, token ids"}' \
         "$ROLE" "$ep" "$TAG" "$(( ep * 15 ))" "$ep")
  "$ERROPD_PY_VLLM" scripts/eval_ner.py --model-path "$md" --tp 2 \
      --output-dir "outputs/T3/eval/sft_${ROLE}_ep${ep}" --meta "$meta"
}
ev "$OUT" 5
for ep in 4 3 2 1; do [ -d "$EPOCHS_DIR/ep$ep" ] && ev "$EPOCHS_DIR/ep$ep" "$ep"; done

date -u +%FT%TZ > "$DONE"
echo "[sft_ner_clean] $(ts) $ROLE DONE"
