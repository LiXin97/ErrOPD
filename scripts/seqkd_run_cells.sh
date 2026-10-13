#!/bin/bash
set -euo pipefail
STEP="${1:?sample|sft|eval|refs}"
CELL_ARG="${2:?cell}"
SEED="${3:-0}"
REPO="${ERROPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
source scripts/env.sh || true
eval "$(python3 scripts/data_seqkd_cells.py "$CELL_ARG" "$SEED")"
NGPUS=${NGPUS:-2}
ts() { date -u +%FT%TZ; }
[ "$STEP" = refs ] || [ "$SEED" != 0 ] || { echo "SEED required"; exit 2; }
case "$STEP" in
  sample)
    D="$SAMPLE_OUT/DONE"; [ -f "$D" ] && { echo "[seqkd7] sample $TKEY s$SEED DONE"; exit 0; }
    mkdir -p "$SAMPLE_OUT"
    T=$(erropd_staged "$TEACHER")
    echo "[seqkd7] $(ts) sample teacher=$TKEY ($T) seed=$SEED tp=$NGPUS -> $DATA_DIR"
    if [ "$TASK" = esci ]; then
      "$ERROPD_PY_VLLM" scripts/data_sample_teacher.py --teacher "$T" --src "$REPO/$SRC" --val "$REPO/$VAL" \
          --out-dir "$DATA_DIR" --name "$DS_NAME" --seed $((SEED * 100000)) --tp "$NGPUS" --snap "$(erropd_staged "$SNAP")" \
          | tee "$SAMPLE_OUT/sample.log"
    else
      "$ERROPD_PY_VLLM" scripts/data_sample_teacher_fashion.py --teacher "$T" --family "$FAMILY" --src "$REPO/$SRC" \
          --val "$REPO/$VAL" --out-dir "$DATA_DIR" --name "$DS_NAME" --ds-key "$DS_KEY" --seed $((SEED * 100000)) \
          --tp "$NGPUS" --err-label "$ERR" | tee "$SAMPLE_OUT/sample.log"
    fi
    cp "$DATA_DIR/$DS_NAME.stats.json" "$SAMPLE_OUT/"
    ts > "$D" ;;
  sft)
    bash scripts/sft_seqkd_cells.sh "$CELL" "$SEED" ;;
  eval)
    D="$OUT/eval/DONE"; [ -f "$D" ] && { echo "[seqkd7] eval $CELL s$SEED DONE"; exit 0; }
    S=$(erropd_staged "$REPO/$OUT/sft")
    mkdir -p "$OUT/eval"
    META=$(printf '{"id":"SEQKD7_%s_s%s_greedy","experiment":"SEQKD7","task":"%s","student":"%s-seqkd-%s","teacher":"%s","seed":%s,"variant":"seqkd","notes":"SFT of the pretrained base (%s) on one T=1 sample of the injected teacher per row of the pair clean SFT set; pair student recipe, one epoch"}' \
           "$CELL" "$SEED" "$TASK" "$(basename "$BASE")" "$TKEY" "$TKEY" "$SEED" "$(basename "$BASE")")
    if [ "$TASK" = esci ]; then
      [ -f "$OUT/eval/greedy/run_meta.json" ] || "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py \
          --model-path "$S" --val-jsonl data/esci_k8/val.jsonl --output-dir "$OUT/eval/greedy" --tp "$NGPUS" \
          --task esci --template train --meta "$META"
      "$ERROPD_PY_VLLM" scripts/eval_seqkd_sampled.py --out "$OUT/eval/sampled" --tp "$NGPUS" \
          --snap "$(erropd_staged "$SNAP")" --model "SEQKD7_${CELL}_s${SEED}=$S"
    else
      if [ ! -f "$OUT/eval/greedy/run_meta.json" ]; then
        for t in 1 2 3; do
          if [ "$FAMILY" = qwen ]; then
            VLLM_PORT=$(( ${ERROPD_PORT_BASE:-30000} + 600 + 100 * (t - 1) )) "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py \
                --model-path "$S" --val-jsonl data/pl_k8/val.jsonl --output-dir "$OUT/eval/greedy" --tp "$NGPUS" --meta "$META" || true
          else
            VLLM_PORT=$(( ${ERROPD_PORT_BASE:-30000} + 600 + 100 * (t - 1) )) "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py \
                --model-path "$S" --family "$FAMILY" --tp "$NGPUS" --output-dir "$OUT/eval/greedy" --meta "$META" || true
          fi
          [ -f "$OUT/eval/greedy/run_meta.json" ] && break; sleep 30
        done
      fi
      [ -f "$OUT/eval/greedy/run_meta.json" ] || { echo "[seqkd7] greedy eval failed"; exit 7; }
      SV="$OUT/eval/sampled"
      if [ ! -f "$SV.v2.json" ]; then
        for t in 1 2 3; do
          rm -f "$SV.v2.records.jsonl"
          VLLM_PORT=$(( ${ERROPD_PORT_BASE:-30000} + 650 + 100 * (t - 1) )) "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py \
              --model-path "$S" --family "$FAMILY" --tp "$NGPUS" --out "$SV" > "$SV.log" 2>&1 || true
          [ -f "$SV.v2.json" ] && break; sleep 30
        done
      fi
      [ -f "$SV.v2.json" ] || { echo "[seqkd7] sampler failed"; tail -20 "$SV.log"; exit 8; }
      python3 -c "import json;r=json.load(open('$SV.v2.json'));print('[seqkd7] SAMPLED val valid=%.4f %s=%.4f n=%d' % (r['val']['valid'], '$ERR', r['val']['labels'].get('$ERR', 0.0), r['val']['n_samples_feasible']))"
    fi
    ts > "$D" ;;
  refs)
    [ "$TASK" = esci ] || { echo "[seqkd7] refs only for the ESCI cell (Fashion teachers were measured by their blocks on the same sampler)"; exit 2; }
    R=outputs/SEQKD7/$CELL/refs
    D=$R/DONE; [ -f "$D" ] && { echo "[seqkd7] refs DONE"; exit 0; }
    mkdir -p "$R"
    "$ERROPD_PY_VLLM" scripts/eval_seqkd_sampled.py --out "$R/sampled" --tp "$NGPUS" \
        --snap "$(erropd_staged "$SNAP")" --model "teacher_$TKEY=$(erropd_staged "$TEACHER")"
    ts > "$D" ;;
  *) echo "bad step $STEP"; exit 2 ;;
esac
