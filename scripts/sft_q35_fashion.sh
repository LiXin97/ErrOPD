#!/usr/bin/env bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
: "${SIZE:?SIZE required (2b|4b|9b)}"
cd "${ERROPD_ROOT:-/path/to/erropd}"
source scripts/env_q35.sh
LX=/path/to
SZ=$(echo "$SIZE" | tr b B)
BASE_MODEL=$LX/models/Qwen3.5-$SZ
OUT=${OUT:-$ERROPD_ROOT/outputs/D6/sft_qwen3.5_${SIZE}}
CFG=${CFG:-configs/d6/sft_qwen3.5_${SIZE}_pl_k8.yaml}; [[ "$CFG" = /* ]] || CFG=$ERROPD_ROOT/$CFG
DATASET_DIR=${DATASET_DIR:-data/lf_d6}; [[ "$DATASET_DIR" = /* ]] || DATASET_DIR=$ERROPD_ROOT/$DATASET_DIR
EVAL_TP=${EVAL_TP:-4}; TAG=${TAG:-qwen3.5-${SIZE}}
NGPUS=${NGPUS:-8}
case "$SIZE" in 0.8b|2b) PDB=8 ;; 4b|9b) PDB=4 ;; *) echo "bad SIZE=$SIZE"; exit 2 ;; esac
(( 128 % (PDB * NGPUS) == 0 )) || { echo "effective batch 128 not divisible by PDB=$PDB x NGPUS=$NGPUS"; exit 2; }
GA=$(( 128 / (PDB * NGPUS) ))
[ -f "$CFG" ] || { echo "missing $CFG"; exit 2; }
[ -f "$DATASET_DIR/dataset_info.json" ] || { echo "missing $DATASET_DIR/dataset_info.json"; exit 2; }
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [sft_q35_fashion ${RUN_ID:-manual}] $*"; }
[ -f "$OUT/DONE" ] && { say "DONE exists"; exit 0; }
B_LOCAL=$(erropd_staged "$BASE_MODEL")
say "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} n=$NGPUS size=$SIZE base=$BASE_MODEL ($B_LOCAL) out=$OUT py=$Q35_PY lf=$Q35_LF_SRC"
say "cfg=$CFG dataset_dir=$DATASET_DIR per_device=$PDB GA=$GA (effective batch $((PDB * GA * NGPUS)))"
export PYTHONPATH=$Q35_LF_SRC${PYTHONPATH:+:$PYTHONPATH}
export DISABLE_VERSION_CHECK=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FORCE_TORCHRUN=1
export PATH=$(dirname "$Q35_PY"):$PATH
EXTRA=(); [ -n "${MAX_STEPS:-}" ] && EXTRA=(max_steps=$MAX_STEPS)
if [ ! -f "$OUT/TRAIN_DONE" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"
  say "SFT_START"
  "$Q35_PY" -m torch.distributed.run --nproc_per_node "$NGPUS" --master_port "${MASTER_PORT:-$((20000 + RANDOM % 20000))}" \
      "$Q35_LF_SRC/llamafactory/launcher.py" "$CFG" \
      model_name_or_path="$B_LOCAL" output_dir="$OUT" dataset_dir="$DATASET_DIR" \
      per_device_train_batch_size=$PDB per_device_eval_batch_size=$PDB gradient_accumulation_steps=$GA \
      deepspeed="$ERROPD_ROOT/configs/d6/ds_z3_config.json" ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
  say "SFT_EXIT rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  ls "$OUT"/model*.safetensors >/dev/null 2>&1 || { say "no weights in $OUT"; exit 4; }
  for f in tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
    [ -f "$B_LOCAL/$f" ] && cp "$B_LOCAL/$f" "$OUT/$f"
  done
  "$Q35_PY" scripts/common_q35_template.py "$OUT"
  "$Q35_PY" scripts/check_q35_template.py "$OUT" "$B_LOCAL" --out "$OUT/template_check.json" > "$OUT/template_check.log" 2>&1 \
    || { say "template identity check FAILED (see $OUT/template_check.log)"; exit 9; }
  "$ERROPD_PY_VLLM" -c "import sys; from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(sys.argv[1]); print('eval-env tokenizer OK', type(t).__name__, len(t))" "$OUT" \
    || { say "standard eval interpreter cannot load the tokenizer"; exit 9; }
  touch "$OUT/TRAIN_DONE"
fi
if [ "${EVAL_EPOCHS:-0}" = "1" ]; then
  for CK in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    st=${CK##*-}; ep=$(( (st + 14) / 15 ))
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json; do
      cp "$OUT/$f" "$CK/$f"
    done
    "$Q35_PY" scripts/common_q35_template.py "$CK"
    E=$OUT/eval_epochs/ep$ep
    META=$(printf '{"id":"D6_sft_%s_ep%s","experiment":"D6","task":"fashion_k8","student":"%s","teacher":"none","lambda":null,"seed":42,"step":%s,"budget_steps":75,"variant":"sft","notes":"SFT epoch %s of 5 (checkpoint-%s, effective batch 128) template=d6_nothink sha=%s"}' \
       "$SIZE" "$ep" "$TAG" "$st" "$ep" "$st" "${Q35_TEMPLATE_SHA:0:16}")
    say "EVAL epoch $ep ($CK)"
    "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py --model-path "$CK" --val-jsonl data/pl_k8/val.jsonl \
        --output-dir "$E" --tp "$EVAL_TP" --meta "$META" 2>&1 | tail -3
    [ -f "$E/run_meta.json" ] || { say "eval failed epoch $ep"; exit 8; }
    say "EPOCH $ep parse=$(python3 -c "import json;print(json.load(open('$E/listwise_metrics.json'))['parse_rate'])")"
  done
  mkdir -p "${OUT}_epochs" && for CK in "$OUT"/checkpoint-*; do [ -d "$CK" ] && mv "$CK" "${OUT}_epochs/"; done
fi
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  EXPID=${EXPID:-D6}; [ "$EXPID" = D6 ] && [ -n "${MAX_STEPS:-}" ] && [[ "$OUT" == */phase0/* ]] && EXPID=D6P0
  META=$(printf '{"id":"%s_sft_%s%s%s","experiment":"%s","task":"fashion_k8","student":"%s","teacher":"none","lambda":null,"seed":42,"step":0,"budget_steps":0,"variant":"sft","notes":"template=d6_nothink sha=%s max_steps=%s cfg=%s dataset_dir=%s"}' \
     "$EXPID" "$SIZE" "${ID_SUFFIX:-}" "${MAX_STEPS:+_ms$MAX_STEPS}" "$EXPID" "$TAG" "${Q35_TEMPLATE_SHA:0:16}" "${MAX_STEPS:-full}" \
     "$(basename "$CFG")" "${DATASET_DIR#$ERROPD_ROOT/}")
  say "EVAL (TP=$EVAL_TP, standard eval interpreter)"
  "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py --model-path "$OUT" --val-jsonl data/pl_k8/val.jsonl \
      --output-dir "$OUT/eval" --tp "$EVAL_TP" --meta "$META" 2>&1
  [ -f "$OUT/eval/run_meta.json" ] || { say "eval failed"; exit 7; }
  say "RESULT $(python3 -c "import json;m=json.load(open('$OUT/eval/listwise_metrics.json'));print('parse=%.4f ndcg1=%s useful=%.4f' % (m['parse_rate'], m['ndcg@1_mean'], m['useful']))")"
fi
if [ -n "${SAMPLE_OUT:-}" ]; then
  [[ "$SAMPLE_OUT" = /* ]] || SAMPLE_OUT=$ERROPD_ROOT/$SAMPLE_OUT
  if [ ! -f "$SAMPLE_OUT.v2.json" ]; then
    say "SAMPLE -> $SAMPLE_OUT"
    mkdir -p "$(dirname "$SAMPLE_OUT")"
    "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py --model-path "$OUT" --family qwen --tp "$EVAL_TP" \
        --out "$SAMPLE_OUT" > "$SAMPLE_OUT.log" 2>&1
    grep -E "^\[sample_validity\]" "$SAMPLE_OUT.log"; tail -3 "$SAMPLE_OUT.log"
  fi
  [ -f "$SAMPLE_OUT.v2.json" ] || { say "sampler failed"; exit 10; }
  say "SAMPLED $(python3 -c "import json;r=json.load(open('$SAMPLE_OUT.v2.json'));print(' | '.join('%s valid=%.4f k1_trunc_realids=%.4f' % (s, r[s]['valid'], r[s]['labels'].get('k1_trunc_realids', 0.0)) for s in ('val','train')))")"
fi
touch "$OUT/DONE"
