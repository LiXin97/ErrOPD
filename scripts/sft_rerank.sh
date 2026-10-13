#!/bin/bash
set -euo pipefail
WHICH="${1:?student|teacher}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
MODELS=/path/to/models
case "$WHICH" in
  student) CFG=configs/sft_qwen3_1.7b_t1.yaml; BASE=$MODELS/Qwen3-1.7B; SZ=1.7b; PDB=8; SNAME=qwen3-1.7b ;;
  teacher) CFG=configs/sft_qwen3_4b_t1.yaml;   BASE=$MODELS/Qwen3-4B;   SZ=4b;   PDB=4; SNAME=qwen3-4b ;;
  *) echo "bad WHICH=$WHICH"; exit 2 ;;
esac
OUT=outputs/T1/sft_${SZ}
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[sft_rerank] $WHICH already DONE"; exit 0; }
NGPUS=${NGPUS:-8}
EFF=128
(( EFF % (PDB * NGPUS) == 0 )) || { echo "[sft_rerank] 128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( EFF / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")
ts() { date -u +%FT%TZ; }
PY_SFT=${ERROPD_PY_SFT:-}
[ -n "$PY_SFT" ] || { echo "[sft_rerank] ERROPD_PY_SFT not set"; exit 3; }
export PYTHONNOUSERSITE=1

"$ERROPD_PY_VLLM" - <<'EOF'
import hashlib, json, sys
m = json.load(open("data/erropd_t1/MANIFEST.json"))
bad = [p for p, h in m["derived"].items() if hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
if bad:
    sys.exit(f"[sft_rerank] data files differ from MANIFEST: {bad}")
print(f"[sft_rerank] data sha256 OK ({len(m['derived'])} files)")
EOF
if [ ! -f "$OUT/all_results.json" ]; then
  DISABLE_VERSION_CHECK=1 "$PY_SFT" scripts/check_rerank_template.py --lf --stock "$BASE_M" \
    || { echo "[sft_rerank] LlamaFactory encoder != stock rendering; not training"; exit 6; }
fi

if [ ! -f "$OUT/all_results.json" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LF_SRC=/path/to/LlamaFactory/src
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/t1_sft_$WHICH}/hf_datasets"
  PORT=$(( 20000 + RANDOM % 20000 ))
  echo "[sft_rerank] $(ts) train $WHICH base=$BASE_M NGPUS=$NGPUS per_device=$PDB GA=$GA -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
      model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA" \
      dataset_dir="$REPO/data/erropd_t1"
  [ -f "$OUT/all_results.json" ] || { echo "[sft_rerank] no all_results.json"; exit 4; }
fi
"$ERROPD_PY_VLLM" - "$OUT" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + "/trainer_state.json"))
ev = [(round(h["epoch"], 2), round(h["eval_loss"], 4)) for h in s["log_history"] if "eval_loss" in h]
print(f"[sft_rerank] max_steps={s['max_steps']} global_step={s['global_step']} epochs={s.get('num_train_epochs')} eval_loss={ev}")
EOF

EPD=${OUT}_epochs
for c in "$OUT"/checkpoint-*; do
  [ -d "$c" ] || continue
  ep=$("$ERROPD_PY_VLLM" -c "import json,sys; print(round(json.load(open(sys.argv[1]+'/trainer_state.json'))['epoch']))" "$c")
  mkdir -p "$EPD"; rm -rf "$EPD/ep$ep"; mv "$c" "$EPD/ep$ep"
done
DIRS=("$OUT")
for c in "$EPD"/ep*; do
  [ -d "$c" ] || continue
  DIRS+=("$c")
  for f in tokenizer.json vocab.json merges.txt special_tokens_map.json added_tokens.json generation_config.json; do
    [ -f "$c/$f" ] || [ ! -f "$OUT/$f" ] || cp "$OUT/$f" "$c/$f"
  done
done
for d in "${DIRS[@]}"; do
  cp "$BASE/tokenizer_config.json" "$d/tokenizer_config.json"
  "$ERROPD_PY_VLLM" - "$BASE/tokenizer_config.json" "$d" <<'EOF'
import json, sys
from pathlib import Path
ct = json.load(open(sys.argv[1]))["chat_template"]
j = Path(sys.argv[2]) / "chat_template.jinja"
if j.exists() and j.read_text() != ct:
    j.write_text(ct); print(f"[sft_rerank] rewrote {j} with the stock template")
EOF
done
"$ERROPD_PY_VLLM" scripts/check_rerank_template.py --stock "$BASE" --tok-dir "$OUT" \
  || { echo "[sft_rerank] saved tokenizer renders differently from stock"; exit 6; }

TP=2
epoch_of() {
  "$ERROPD_PY_VLLM" -c "import json,sys; print(round(json.load(open(sys.argv[1]+'/trainer_state.json'))['epoch']))" "$1"
}
step_of() {
  "$ERROPD_PY_VLLM" -c "import json,sys; print(json.load(open(sys.argv[1]+'/trainer_state.json'))['global_step'])" "$1"
}
run_eval() {
  local M="$1" EP="$2" ST="$3" EV="outputs/T1/eval/sft_${SZ}_ep$2"
  local meta
  meta=$(printf '{"id":"T1_sft_%s_ep%s","experiment":"T1","task":"msmarco_rankgpt_k20","student":"%s","teacher":"none","lambda":null,"seed":42,"step":%s,"budget_steps":%s,"variant":"sft","clip_c":null,"notes":"T1 SFT epoch %s (LlamaFactory seed 42); single template for SFT, rollout and eval"}' \
         "$SZ" "$EP" "$SNAME" "$ST" "$MAXSTEP" "$EP")
  "$ERROPD_PY_VLLM" scripts/eval_rerank.py --model-path "$M" --tp "$TP" --output-dir "$EV" --meta "$meta"
}
MAXSTEP=$(step_of "$OUT")
run_eval "$REPO/$OUT" 5 "$MAXSTEP"
if [ "$WHICH" = student ]; then
  for c in "$EPD"/ep*; do
    [ -d "$c" ] || continue
    ep=$(epoch_of "$c"); [ "$ep" = 5 ] && continue
    run_eval "$REPO/$c" "$ep" "$(step_of "$c")"
  done
fi

date -u +%FT%TZ > "$DONE"
echo "[sft_rerank] $(ts) $WHICH done"
