#!/usr/bin/env bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
CELL_ARG="${1:?cell}"; SEED="${2:?seed}"
REPO="${ERROPD_ROOT:-/path/to/erropd}"
cd "$REPO"
eval "$(python3 scripts/data_seqkd_cells.py "$CELL_ARG" "$SEED")" || { echo "[sft_seqkd_cells] bad cell $CELL_ARG"; exit 2; }
SO="$OUT/sft"
DONE="$REPO/$SO/DONE"
[ -f "$DONE" ] && { echo "[sft_seqkd_cells] $CELL s$SEED already DONE"; exit 0; }
[ -f "$DATA_DIR/dataset_info.json" ] || { echo "[sft_seqkd_cells] $DATA_DIR missing (sample step)"; exit 2; }
NGPUS=${NGPUS:-$SFT_GPUS}
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [sft_seqkd_cells $CELL s$SEED ${RUN_ID:-manual}] $*"; }
if [ "$FAMILY" = qwen ]; then
  export Q35_ENV_NAME=${Q35_ENV_NAME:-q35env_v2}
  source scripts/env_q35.sh
else
  source scripts/env.sh
fi
BASE_M=$(erropd_staged "$BASE")
EFF=128
(( EFF % (PDB * NGPUS) == 0 )) || { say "128 not divisible by $PDB x $NGPUS"; exit 2; }
GA=$(( EFF / (PDB * NGPUS) ))
COMMON=(model_name_or_path="$BASE_M" output_dir="$REPO/$SO" dataset="$DS_KEY" dataset_dir="$REPO/$DATA_DIR" num_train_epochs=1.0)
record_steps() {
  "$1" - "$SO" "$CELL" "$SEED" <<'EOF'
import json, sys
d, cell, seed = sys.argv[1:4]
s = json.load(open(d + "/trainer_state.json"))
rec = {"cell": cell, "seed": int(seed), "max_steps": s["max_steps"], "global_step": s["global_step"],
       "train_batch_size": s.get("train_batch_size"), "num_train_epochs": s.get("num_train_epochs")}
json.dump(rec, open(d + "/steps.json", "w"), indent=1)
print("[sft_seqkd_cells] steps", json.dumps(rec))
EOF
}

if [ ! -f "$SO/TRAIN_DONE" ]; then
  rm -rf "$SO"; mkdir -p "$SO"
  PORT=${MASTER_PORT:-$(( 20000 + RANDOM % 20000 ))}
  case "$FAMILY" in
    llama|qwen3)
      PY_SFT=${ERROPD_PY_SFT:-}
      [ -n "$PY_SFT" ] || { say "ERROPD_PY_SFT not set"; exit 3; }
      export PATH="$(dirname "$PY_SFT"):$PATH"
      LF_SRC=/path/to/LlamaFactory/src
      if LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)' 2>/dev/null); then :; else
        export PYTHONPATH="$LF_SRC:${PYTHONPATH:-}"; LAUNCHER="$LF_SRC/llamafactory/launcher.py"; fi
      export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1
      export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/seqkd7_sft_${CELL}_s$SEED}/hf_datasets"
      EXTRA=()
      [ "$FAMILY" = llama ] && EXTRA=("save_strategy='no'" "eval_strategy='no'")
      say "train base=$BASE_M cfg=$CFG data=$DS_KEY NGPUS=$NGPUS per_device=$PDB GA=$GA -> $SO"
      "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
          "${COMMON[@]}" gradient_accumulation_steps="$GA" ${EXTRA[@]+"${EXTRA[@]}"}
      rc=$?; say "SFT_EXIT rc=$rc"; [ $rc -eq 0 ] || exit $rc
      [ -f "$SO/all_results.json" ] || { say "no all_results.json"; exit 4; }
      record_steps "$ERROPD_PY_VLLM" || exit 4
      rm -rf "$SO"/checkpoint-*
      ;;
    other)
      PY_SFT=${ERROPD_PY_SFT:?ERROPD_PY_SFT not set}
      [ "$NGPUS" = 8 ] || { say "OLMo 3 SFT needs 8 GPUs (per-device 2 x GA 8 in the yaml)"; exit 2; }
      export PATH="$(dirname "$PY_SFT"):$PATH"
      LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)') || { say "no llamafactory"; exit 2; }
      export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1 DISABLE_VERSION_CHECK=1
      export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/seqkd7_sft_${CELL}_s$SEED}/hf_datasets"
      say "train base=$BASE_M cfg=$CFG data=$DS_KEY (py $PY_SFT) -> $SO"
      "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" "$LAUNCHER" "$REPO/$CFG" \
          "${COMMON[@]}" deepspeed="$REPO/configs/d6/ds_z3_config.json"
      rc=$?; say "SFT_EXIT rc=$rc"; [ $rc -eq 0 ] || exit $rc
      ls "$SO"/model*.safetensors >/dev/null 2>&1 || { say "no weights"; exit 4; }
      record_steps "$PY_SFT" || exit 4
      rm -rf "$SO"/checkpoint-*
      for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja generation_config.json; do
        [ -f "$BASE_M/$f" ] && cp "$BASE_M/$f" "$SO/$f"
      done
      "$PY_SFT" - "$REPO/$SO" <<'PY'
import json, sys
d = sys.argv[1]
g = json.load(open(f"{d}/generation_config.json"))
c = json.load(open(f"{d}/config.json"))
def ids(x):
    return [] if x is None else ([x] if isinstance(x, int) else list(x))
cur, want = ids(g.get("eos_token_id")), ids(c.get("eos_token_id"))
new = cur + [i for i in want if i not in cur]
if new != cur:
    g["eos_token_id"] = new
    json.dump(g, open(f"{d}/generation_config.json", "w"), indent=2)
    print(f"[sft_seqkd_cells] generation_config eos_token_id {cur} -> {new} (model config eos {want})")
PY
      "$PY_SFT" - "$REPO/$SO" <<'PY' || { say "prompt identity check FAILED"; exit 9; }
import json, sys
from transformers import AutoTokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
mp = sys.argv[1]
tok = AutoTokenizer.from_pretrained(mp); tok_lf = AutoTokenizer.from_pretrained(mp)
tpl = get_template_and_fix_tokenizer(tok_lf, DataArguments(template="chatml"))
rows = [json.loads(l) for l in open("data/pl_k8/train.jsonl")] + [json.loads(l) for l in open("data/pl_k8/val.jsonl")]
nd = 0
for r in rows:
    m = r["messages"]
    s = next(x["content"] for x in m if x["role"] == "system")
    p, _ = tpl.encode_oneturn(tok_lf, [x for x in m if x["role"] in ("user", "assistant")], system=s)
    h = tok(tok.apply_chat_template([x for x in m if x["role"] in ("system", "user")], add_generation_prompt=True, tokenize=False), add_special_tokens=False).input_ids
    nd += p != h
print(f"[sft_seqkd_cells] SFT-vs-rollout/eval prompt id mismatches: {nd}/{len(rows)}")
sys.exit(1 if nd else 0)
PY
      ;;
    qwen)
      export PYTHONPATH=$Q35_LF_SRC${PYTHONPATH:+:$PYTHONPATH}
      export DISABLE_VERSION_CHECK=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FORCE_TORCHRUN=1
      export PATH=$(dirname "$Q35_PY"):$PATH
      say "train base=$BASE_M cfg=$CFG data=$DS_KEY per_device=$PDB GA=$GA (py $Q35_PY, env $Q35_ENV_NAME) -> $SO"
      "$Q35_PY" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "$PORT" \
          "$Q35_LF_SRC/llamafactory/launcher.py" "$REPO/$CFG" "${COMMON[@]}" \
          per_device_train_batch_size=$PDB per_device_eval_batch_size=$PDB gradient_accumulation_steps=$GA \
          deepspeed="$REPO/configs/d6/ds_z3_config.json" "save_strategy='no'" "eval_strategy='no'"
      rc=$?; say "SFT_EXIT rc=$rc"; [ $rc -eq 0 ] || exit $rc
      ls "$SO"/model*.safetensors >/dev/null 2>&1 || { say "no weights in $SO"; exit 4; }
      record_steps "$Q35_PY" || exit 4
      rm -rf "$SO"/checkpoint-*
      for f in tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
        [ -f "$BASE_M/$f" ] && cp "$BASE_M/$f" "$SO/$f"
      done
      "$Q35_PY" scripts/common_q35_template.py "$SO"
      "$Q35_PY" scripts/check_q35_template.py "$SO" "$BASE_M" --out "$SO/template_check.json" > "$SO/template_check.log" 2>&1 \
        || { say "template identity check FAILED (see $SO/template_check.log)"; exit 9; }
      "$ERROPD_PY_VLLM" -c "import sys; from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(sys.argv[1]); print('eval-env tokenizer OK', type(t).__name__, len(t))" "$SO" \
        || { say "standard eval interpreter cannot load the tokenizer"; exit 9; }
      ;;
    *) say "bad FAMILY=$FAMILY"; exit 2 ;;
  esac
  touch "$SO/TRAIN_DONE"
fi

case "$FAMILY" in
  llama)
    "$ERROPD_PY_VLLM" scripts/common_llama_template.py install --model-dir "$SO" --base-dir "$BASE_M" || exit 9
    "$ERROPD_PY_VLLM" scripts/common_llama_template.py verify --model-dir "$SO" --data data/pl_k8/val.jsonl data/pl_k8/train.jsonl || exit 9
    "$ERROPD_PY_VERL" scripts/common_llama_template.py verify --model-dir "$SO" --data data/pl_k8/val.jsonl || exit 9
    ;;
  qwen3)
    cp "$BASE/tokenizer_config.json" "$SO/tokenizer_config.json" || exit 9
    ;;
esac
date -u +%FT%TZ > "$DONE"
say "done"
