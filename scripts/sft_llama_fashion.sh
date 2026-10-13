#!/bin/bash
set -euo pipefail
ROLE="${1:?student|teacher}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
MODELS=/path/to/models
case "$ROLE" in
  student) CFG=configs/sft_llama3.2_3b_pl_k8_erropd.yaml; BASE=$MODELS/Llama-3.2-3B-Instruct
           OUT=outputs/D1/sft_llama3.2_3b; PDB=4; TAG=llama3.2-3b ;;
  teacher) CFG=configs/sft_llama3.1_8b_pl_k8_erropd.yaml; BASE=$MODELS/Llama-3.1-8B-Instruct
           OUT=outputs/D1/sft_llama3.1_8b; PDB=2; TAG=llama3.1-8b ;;
  *) echo "ROLE must be student|teacher"; exit 2 ;;
esac
EPOCHS_DIR=${OUT}_epochs
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[sft_llama_fashion] $ROLE already DONE"; exit 0; }

NGPUS=${NGPUS:-8}
EFF=128
if (( EFF % (PDB * NGPUS) != 0 )); then echo "[sft_llama_fashion] 128 not divisible by $PDB x $NGPUS"; exit 2; fi
GA=$(( EFF / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")
PY_SFT=${ERROPD_PY_SFT:-}
LF_SRC=${XFAMILY_LF_SRC:-/path/to/LlamaFactory/src}
TPL="$ERROPD_PY_VLLM scripts/common_llama_template.py"
ts() { date -u +%FT%TZ; }

if [ ! -f "$OUT/all_results.json" ]; then
  [ -n "$PY_SFT" ] || { echo "[sft_llama_fashion] ERROPD_PY_SFT not set (queue README row 'ERROPD_PY_SFT')"; exit 3; }
  rm -rf "$OUT" "$EPOCHS_DIR"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/xfamily_sft_$ROLE}/hf_datasets"
  PORT=$(( 20000 + RANDOM % 20000 ))
  echo "[sft_llama_fashion] $(ts) train $ROLE base=$BASE_M NGPUS=$NGPUS per_device=$PDB GA=$GA launcher=$LAUNCHER -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" \
      "$LAUNCHER" "$REPO/$CFG" \
      model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA"
  [ -f "$OUT/all_results.json" ] || { echo "[sft_llama_fashion] training finished without all_results.json"; exit 4; }
  "$ERROPD_PY_VLLM" - "$OUT" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + "/trainer_state.json"))
print(f"[sft_llama_fashion] max_steps={s['max_steps']} global_step={s['global_step']} train_batch_size={s.get('train_batch_size')}")
print("[sft_llama_fashion] eval_loss per epoch:", [round(x["eval_loss"], 4) for x in s["log_history"] if "eval_loss" in x])
assert s["max_steps"] == 75, "expected 75 optimizer steps (effective batch 128, 5 epochs)"
EOF
fi

shopt -s nullglob
CKPTS=("$OUT"/checkpoint-*)
shopt -u nullglob
if (( ${#CKPTS[@]} > 0 )); then
  mkdir -p "$EPOCHS_DIR"
  for c in "${CKPTS[@]}"; do
    step=${c##*-}; ep=$(( step / 15 ))
    if [ "$step" = 75 ]; then rm -rf "$c"; else mv "$c" "$EPOCHS_DIR/ep${ep}"; fi
  done
fi

DIRS=("$OUT")
if [ -d "$EPOCHS_DIR" ]; then for d in "$EPOCHS_DIR"/ep*; do DIRS+=("$d"); done; fi
for d in "${DIRS[@]}"; do
  $TPL install --model-dir "$d" --base-dir "$BASE_M"
done
$TPL verify --model-dir "$OUT" --data data/pl_k8/val.jsonl data/pl_k8/train.jsonl
"$ERROPD_PY_VERL" scripts/common_llama_template.py verify --model-dir "$OUT" --data data/pl_k8/val.jsonl

TP=4; (( NGPUS < 4 )) && TP=$NGPUS
ev() {
  local md=$1 ep=$2 step=$(( $2 * 15 ))
  local meta
  meta=$(printf '{"id":"D1_sft_%s_ep%s","experiment":"D1","task":"fashion_k8","student":"%s","teacher":"none","lambda":null,"seed":42,"step":%s,"budget_steps":75,"variant":"sft","clip_c":null,"notes":"SFT epoch %s of 5 (effective batch 128); prompt = SFT rendering, token ids"}' \
         "$ROLE" "$ep" "$TAG" "$step" "$ep")
  "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py --model-path "$md" --family llama --tp "$TP" \
      --output-dir "outputs/D1/eval/sft_${ROLE}_ep${ep}" --meta "$meta"
}
ev "$OUT" 5
if [ "$ROLE" = student ] && [ -d "$EPOCHS_DIR" ]; then
  for ep in 4 3 2 1; do [ -d "$EPOCHS_DIR/ep$ep" ] && ev "$EPOCHS_DIR/ep$ep" "$ep"; done
fi

if [ "$ROLE" = student ]; then
  "$ERROPD_PY_VLLM" - "$OUT" "$EPOCHS_DIR" <<'EOF'
import json, os, sys
out, epochs_dir = sys.argv[1], sys.argv[2]
LO, HI = 0.20, 0.90
parse = {}
for ep in (5, 4, 3, 2, 1):
    f = f"outputs/D1/eval/sft_student_ep{ep}/listwise_metrics.json"
    if os.path.exists(f):
        parse[ep] = json.load(open(f))["parse_rate"]
sel = next((ep for ep in (5, 4, 3, 2, 1) if ep in parse and LO <= parse[ep] <= HI), None)
if sel is None:
    status = "ALL_ABOVE" if all(p > HI for p in parse.values()) else \
             "ALL_BELOW" if all(p < LO for p in parse.values()) else "NONE_IN_RANGE"
    path = None
else:
    status, path = "OK", (out if sel == 5 else f"{epochs_dir}/ep{sel}")
rec = {"rule": "latest epoch checkpoint (5..1) with strict parse in [0.20, 0.90]", "parse_by_epoch": parse,
       "selected_epoch": sel, "path": path, "status": status}
json.dump(rec, open("outputs/D1/selected_student.json", "w"), indent=2)
print("[sft_llama_fashion] selection:", json.dumps(rec))
EOF
fi

touch "$DONE"
echo "[sft_llama_fashion] $(ts) $ROLE DONE"
