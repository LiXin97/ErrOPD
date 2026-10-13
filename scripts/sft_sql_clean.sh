#!/bin/bash
set -euo pipefail
ROLE="${1:?student|teacher|teacher8b}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh
MODELS=/path/to/models
case "$ROLE" in
  student)   CFG=configs/sft_qwen3_1.7b_t2_spider.yaml; BASE=$MODELS/Qwen3-1.7B; PDB=8; TAG=qwen3-1.7b; OUT=outputs/T2/sft_qwen3_1.7b ;;
  teacher)   CFG=configs/sft_qwen3_4b_t2_spider.yaml;   BASE=$MODELS/Qwen3-4B;   PDB=4; TAG=qwen3-4b;   OUT=outputs/T2/sft_qwen3_4b ;;
  teacher8b) CFG=configs/sft_qwen3_4b_t2_spider.yaml;   BASE=$MODELS/Qwen3-8B;   PDB=2; TAG=qwen3-8b;   OUT=outputs/T2/sft_qwen3_8b ;;
  *) echo "ROLE must be student|teacher|teacher8b"; exit 2 ;;
esac
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[sft_sql_clean] $ROLE already DONE"; exit 0; }
NGPUS=${NGPUS:-8}
EFF=128
(( EFF % (PDB * NGPUS) == 0 )) || { echo "[sft_sql_clean] 128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( EFF / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")
ts() { date -u +%FT%TZ; }

if [ ! -f "$OUT/all_results.json" ]; then
  PY_SFT=${ERROPD_PY_SFT:-}
  [ -n "$PY_SFT" ] || { echo "[sft_sql_clean] ERROPD_PY_SFT not set (spec needs [\"sft\"])"; exit 3; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LF_SRC=/path/to/LlamaFactory/src
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/t2_sft_$ROLE}/hf_datasets"
  PORT=$(( 20000 + RANDOM % 20000 ))
  echo "[sft_sql_clean] $(ts) train $ROLE base=$BASE_M NGPUS=$NGPUS per_device=$PDB GA=$GA -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
      model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA" \
      per_device_train_batch_size="$PDB" per_device_eval_batch_size="$PDB" dataset_dir="$REPO/data/erropd_t2"
  [ -f "$OUT/all_results.json" ] || { echo "[sft_sql_clean] training finished without all_results.json"; exit 4; }
  "$ERROPD_PY_VLLM" - "$OUT" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + "/trainer_state.json"))
print(f"[sft_sql_clean] max_steps={s['max_steps']} global_step={s['global_step']} train_batch_size={s.get('train_batch_size')}")
print("[sft_sql_clean] eval_loss per epoch:", [round(x["eval_loss"], 4) for x in s["log_history"] if "eval_loss" in x])
assert s["max_steps"] == 75, "expected 75 optimizer steps (effective batch 128, 5 epochs, 1795 examples)"
EOF
fi
rm -rf "$OUT"/checkpoint-*

cp "$BASE/tokenizer_config.json" "$OUT/tokenizer_config.json"

PYTHONPATH="$ERROPD_VERL_PROD:${PYTHONPATH:-}" DISABLE_VERSION_CHECK=1 "$ERROPD_PY_VERL" scripts/check_sql_template.py \
    --tokenizer-dir "$OUT" --base-dir "$BASE_M" | tee "$OUT/template_check.json"

EV=outputs/T2/eval/sft_${ROLE}
if [ ! -f "$EV/sql_metrics.json" ] || [ ! -f "$EV/run_meta.json" ]; then
  meta=$(printf '{"id":"T2_sft_%s","experiment":"T2","student":"%s","teacher":"none","lambda":null,"seed":42,"step":75,"budget_steps":75,"variant":"sft","clip_c":null,"notes":"SFT epoch 5 of 5 (effective batch 128); one template in SFT/rollout/eval"}' "$ROLE" "$TAG")
  "$ERROPD_PY_VLLM" scripts/eval_sql.py --model-path "$REPO/$OUT" --output-dir "$EV" --tp 2 --meta "$meta"
fi
date -u +%FT%TZ > "$DONE"
echo "[sft_sql_clean] $(ts) $ROLE DONE"
