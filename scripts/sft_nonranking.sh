#!/bin/bash
set -euo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
TASK="${1:?t3|t2}"
TAG="${2:?dataset tag, e.g. shift_f20_x5}"
REPO="${ERROPD_ROOT:-/path/to/erropd}"
cd "$REPO"
source scripts/env.sh
MODELS=/path/to/models
NR_EXP=${NR_EXP:-NR}
case "${NR_SFT_ROLE:-teacher}" in
  teacher) BASE=$MODELS/Qwen3-4B; PDB=4; SIZE=4b ;;
  student) BASE=$MODELS/Qwen3-1.7B; PDB=8; SIZE=1.7b ;;
  *) echo "[nr_sft] NR_SFT_ROLE must be teacher|student"; exit 2 ;;
esac
case "$TASK" in
  t3) CFG=configs/sft_qwen3_${SIZE}_t3_ner.yaml ;;
  t2) CFG=configs/sft_qwen3_${SIZE}_t2_spider.yaml ;;
  *) echo "[nr_sft] TASK must be t3|t2"; exit 2 ;;
esac
[ -f "$CFG" ] || { echo "[nr_sft] missing $CFG"; exit 2; }
DDIR=$REPO/data/erropd_nr/$TASK
DS=nr_${TASK}_${TAG}
OUT=outputs/$NR_EXP/sft_${TASK}_${SIZE}_${TAG}
DONE="$REPO/$OUT/DONE"
ts() { date -u +%FT%TZ; }
[ -f "$DONE" ] && { echo "[nr_sft] $TASK $TAG already DONE"; exit 0; }

"$ERROPD_PY_VLLM" - "$DDIR" "$TAG" "$DS" <<'EOF'
import hashlib, json, sys
d, tag, ds = sys.argv[1:4]
man = json.load(open(f"{d}/manifest.json"))["files"][f"{tag}.jsonl"]
h = hashlib.sha256(open(f"{d}/{tag}.jsonl", "rb").read()).hexdigest()
assert h == man["sha256"], f"{tag}.jsonl sha256 {h} != manifest {man['sha256']} (rebuild with data_inject_nonranking.py)"
assert ds in json.load(open(f"{d}/dataset_info.json")), ds
print(f"[nr_sft] data ok: {tag}.jsonl {man['n_examples']} examples, {man['n_corrupted']} corrupted "
      f"({man['corrupted_share_of_all']:.4f} of all), sha256 {h[:12]}")
EOF

NGPUS=${NGPUS:-4}
EFF=128
(( EFF % (PDB * NGPUS) == 0 )) || { echo "[nr_sft] 128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( EFF / (PDB * NGPUS) ))
BASE_M=$(erropd_staged "$BASE")

if [ ! -f "$OUT/all_results.json" ]; then
  PY_SFT=${ERROPD_PY_SFT:-}
  [ -n "$PY_SFT" ] || { echo "[nr_sft] ERROPD_PY_SFT not set (spec needs \"needs\": [\"sft\"])"; exit 3; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LF_SRC=/path/to/LlamaFactory/src
  if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
    export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/nr_sft_${TASK}_$TAG}/hf_datasets"
  PORT=${MASTER_PORT:-$(( 20000 + RANDOM % 20000 ))}
  echo "[nr_sft] $(ts) train $TASK $TAG base=$BASE_M data=$DS NGPUS=$NGPUS per_device=$PDB GA=$GA -> $OUT"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
      model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" gradient_accumulation_steps="$GA" \
      per_device_train_batch_size="$PDB" per_device_eval_batch_size="$PDB" \
      dataset="$DS" dataset_dir="$DDIR" num_train_epochs=1.0
  [ -f "$OUT/all_results.json" ] || { echo "[nr_sft] training finished without all_results.json"; exit 4; }
fi
"$ERROPD_PY_VLLM" - "$OUT" "$DDIR/manifest.json" "$TAG" "$NGPUS" "$PDB" "$GA" <<'EOF'
import json, math, sys
out, man, tag, ngpus, pdb, ga = sys.argv[1], sys.argv[2], sys.argv[3], *map(int, sys.argv[4:7])
s = json.load(open(out + "/trainer_state.json"))
n = json.load(open(man))["files"][tag + ".jsonl"]["n_examples"]
exp = math.ceil(math.ceil(math.ceil(n / ngpus) / pdb) / ga)
rec = {"n_examples": n, "expected_steps": exp, "max_steps": s["max_steps"], "global_step": s["global_step"],
       "clean_recipe_steps": 75, "eval_loss": [round(x["eval_loss"], 4) for x in s["log_history"] if "eval_loss" in x]}
json.dump(rec, open(out + "/nr_sft_steps.json", "w"), indent=1)
print("[nr_sft] steps:", json.dumps(rec))
if s["max_steps"] != exp or s["global_step"] != s["max_steps"]:
    print("[nr_sft] WARNING: optimizer steps differ from the expected count; report this with the gate values")
EOF
rm -rf "$OUT"/checkpoint-*

cp "$BASE/tokenizer_config.json" "$OUT/tokenizer_config.json"
"$ERROPD_PY_VLLM" - "$BASE/tokenizer_config.json" "$OUT" <<'EOF'
import json, os, sys
p = os.path.join(sys.argv[2], "chat_template.jinja")
if os.path.exists(p):
    open(p, "w").write(json.load(open(sys.argv[1]))["chat_template"])
EOF
if [ ! -f "$OUT/template_check.json" ]; then
  if [ "$TASK" = t3 ]; then
    PYTHONPATH="$ERROPD_VERL_PROD:${PYTHONPATH:-}" "$ERROPD_PY_VERL" scripts/check_ner_template.py \
        --tok-dir "$OUT" --base-dir "$BASE_M" --out "$OUT/template_check.json.tmp"
    mv "$OUT/template_check.json.tmp" "$OUT/template_check.json"
  else
    PYTHONPATH="$ERROPD_VERL_PROD:${PYTHONPATH:-}" DISABLE_VERSION_CHECK=1 "$ERROPD_PY_VERL" scripts/check_sql_template.py \
        --tokenizer-dir "$OUT" --base-dir "$BASE_M" | tee "$OUT/template_check.json"
  fi
fi

EV=outputs/$NR_EXP/eval/sft_${TASK}_${SIZE}_${TAG}
LT=$([ "$TASK" = t3 ] && echo conll03_ner_json || echo t2_spider)
STEPS=$("$ERROPD_PY_VLLM" -c 'import json,sys;print(json.load(open(sys.argv[1]+"/nr_sft_steps.json"))["max_steps"])' "$OUT")
META=$(printf '{"id":"%s_sft_%s_%s_%s","experiment":"%s","task":"%s","student":"qwen3-%s","teacher":"none","lambda":null,"seed":42,"step":%s,"budget_steps":%s,"variant":"sft_%s","clip_c":null,"template":"eval","notes":"%s %s SFT on %s (one epoch over 5 corrupted copies; see nr_sft_steps.json)"}' \
       "$NR_EXP" "$TASK" "$SIZE" "$TAG" "$NR_EXP" "$LT" "$SIZE" "$STEPS" "$STEPS" "$TAG" "$NR_EXP" "${NR_SFT_ROLE:-teacher}" "$DS")
if [ "$TASK" = t3 ]; then
  [ -f "$EV/t3_metrics.json" ] && [ -f "$EV/run_meta.json" ] || \
    "$ERROPD_PY_VLLM" scripts/eval_ner.py --model-path "$REPO/$OUT" --tp 2 --output-dir "$EV" --meta "$META"
else
  [ -f "$EV/sql_metrics.json" ] && [ -f "$EV/run_meta.json" ] || \
    "$ERROPD_PY_VLLM" scripts/eval_sql.py --model-path "$REPO/$OUT" --tp 2 --output-dir "$EV" --meta "$META"
fi
[ -f "$EV/run_meta.json" ] || { echo "[nr_sft] greedy eval failed"; exit 7; }
date -u +%FT%TZ > "$DONE"
echo "[nr_sft] $(ts) $TASK $TAG DONE"
