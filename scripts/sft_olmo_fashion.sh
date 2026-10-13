#!/usr/bin/env bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
WHICH="${1:?student|teacher}"
REPO="${ERROPD_ROOT:-/path/to/erropd}"
cd "$REPO"
source scripts/env.sh
LX=/path/to
case "$WHICH" in
  student) CFG=configs/olmo3/sft_olmo3_7b_pl_k8.yaml; BASE=$LX/models/OLMo-3-7B-Instruct; OUT=outputs/OL3/sft_olmo3_7b
           DSDIR=data/lf_d6; NAME=base_olmo3_7b; TP=2; TAG=olmo3-7b ;;
  teacher) CFG=configs/olmo3/sft_olmo31_32b_inj20_pl_k8.yaml; BASE=$LX/models/Olmo-3.1-32B-Instruct
           OUT=outputs/OL3/sft_olmo31_32b_inj20; DSDIR=data/pl_k8_q35s; NAME=teacher_olmo31_32b_inj20; TP=4; TAG=olmo3.1-32b-inj20 ;;
  *) echo "bad WHICH=$WHICH"; exit 2 ;;
esac
DONE="$REPO/$OUT/DONE"
[ -f "$DONE" ] && { echo "[olmo3_sft] $WHICH already DONE"; exit 0; }
NGPUS=${NGPUS:-8}
BASE_M=$(erropd_staged "$BASE")
ts() { date -u +%FT%TZ; }
PY_SFT=${ERROPD_PY_SFT:?ERROPD_PY_SFT not set}
if [ ! -f "$OUT/TRAIN_DONE" ]; then
  [ "$NGPUS" = 8 ] || { echo "[olmo3_sft] training needs 8 GPUs (per-device x GA in the yaml assume 8)"; exit 2; }
  rm -rf "$OUT"; mkdir -p "$OUT"
  export PATH="$(dirname "$PY_SFT"):$PATH"
  LAUNCHER=$("$PY_SFT" -c 'import llamafactory.launcher as m; print(m.__file__)') || { echo "[olmo3_sft] no llamafactory"; exit 2; }
  export WANDB_MODE=disabled TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 NCCL_P2P_LEVEL=NVL PYTHONNOUSERSITE=1 DISABLE_VERSION_CHECK=1
  export HF_DATASETS_CACHE="${ERROPD_RUN_TMP:-/tmp/olmo3_sft_$WHICH}/hf_datasets"
  echo "[olmo3_sft] $(ts) train $WHICH base=$BASE_M cfg=$CFG dataset_dir=$DSDIR -> $OUT (py $PY_SFT)"
  "$PY_SFT" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "${MASTER_PORT:-$((20000 + RANDOM % 20000))}" "$LAUNCHER" \
      "$REPO/$CFG" model_name_or_path="$BASE_M" output_dir="$REPO/$OUT" dataset_dir="$REPO/$DSDIR" \
      deepspeed="$REPO/configs/d6/ds_z3_config.json"
  rc=$?; echo "[olmo3_sft] $(ts) SFT_EXIT rc=$rc"; [ $rc -eq 0 ] || exit $rc
  ls "$OUT"/model*.safetensors >/dev/null 2>&1 || { echo "[olmo3_sft] no weights"; exit 4; }
  rm -rf "$OUT"/checkpoint-*
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja generation_config.json; do
    [ -f "$BASE_M/$f" ] && cp "$BASE_M/$f" "$OUT/$f"
  done
  "$PY_SFT" - "$REPO/$OUT" <<'PY'
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
    print(f"[olmo3_sft] generation_config eos_token_id {cur} -> {new} (model config eos {want})")
PY
  "$PY_SFT" - "$REPO/$OUT" <<'PY' || { echo "[olmo3_sft] prompt identity check FAILED"; exit 9; }
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
print(f"[olmo3_sft] SFT-vs-rollout/eval prompt id mismatches: {nd}/{len(rows)}")
sys.exit(1 if nd else 0)
PY
  touch "$OUT/TRAIN_DONE"
fi
TP=$(( NGPUS < TP ? NGPUS : TP ))
meta=$(printf '{"id":"OL3_sft_%s","experiment":"OL3","task":"fashion_k8","student":"%s","teacher":"none","lambda":null,"seed":42,"step":null,"budget_steps":null,"variant":"sft","clip_c":null,"notes":"OLMo 3 injection test SFT (%s); template = OLMo chat template (LF chatml in SFT, ids identical); transformers 4.57.6"}' "$WHICH" "$TAG" "$CFG")
if [ ! -f "outputs/OL3/eval/sft_$WHICH/run_meta.json" ]; then
  for t in 1 2 3; do
    VLLM_PORT=$(( ${ERROPD_PORT_BASE:-30000} + 600 + 100 * (t - 1) )) "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py \
        --model-path "$OUT" --family other --tp "$TP" --output-dir "outputs/OL3/eval/sft_$WHICH" --meta "$meta"
    [ -f "outputs/OL3/eval/sft_$WHICH/run_meta.json" ] && break; sleep 30
  done
fi
[ -f "outputs/OL3/eval/sft_$WHICH/run_meta.json" ] || { echo "[olmo3_sft] eval failed"; exit 7; }
SO=outputs/OL3/samples/$NAME
mkdir -p outputs/OL3/samples
if [ ! -f "$SO.v2.json" ]; then
  for t in 1 2 3; do
    VLLM_PORT=$(( ${ERROPD_PORT_BASE:-30000} + 650 + 100 * (t - 1) )) "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py \
        --model-path "$OUT" --family other --tp "$TP" --out "$SO" > "$SO.log" 2>&1
    [ -f "$SO.v2.json" ] && break; sleep 30
  done
fi
[ -f "$SO.v2.json" ] || { echo "[olmo3_sft] sampler failed"; tail -20 "$SO.log"; exit 8; }
python3 -c "import json;r=json.load(open('$SO.v2.json'));print('[olmo3_sft] SAMPLED', ' | '.join('%s valid=%.4f k1_trunc_realids=%.4f n=%d' % (s, r[s]['valid'], r[s]['labels'].get('k1_trunc_realids', 0.0), r[s]['n_samples_feasible']) for s in ('val','train')))"
date -u +%FT%TZ > "$DONE"
