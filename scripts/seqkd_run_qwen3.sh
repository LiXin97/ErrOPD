#!/bin/bash
set -euo pipefail
STEP="${1:?sample|sft|eval|refs}"
SEED="${2:-}"
REPO="${ERROPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
source scripts/env.sh || true
SNAP=/path/to/verl_trees/gopd-febfe8b05e
TEACHER=$(erropd_staged "$REPO/outputs/F3/sft_t4b_k1")
NGPUS=${NGPUS:-2}
[ "$STEP" = refs ] || : "${SEED:?SEED required}"
OUT=outputs/K1/K1_s${SEED}
ts() { date -u +%FT%TZ; }
case "$STEP" in
  sample)
    D="$OUT/sample/DONE"; [ -f "$D" ] && { echo "[k1] sample DONE"; exit 0; }
    mkdir -p "$OUT/sample"
    echo "[k1] $(ts) sample seed=$SEED teacher=$TEACHER tp=$NGPUS"
    "$ERROPD_PY_VLLM" scripts/data_sample_teacher.py --teacher "$TEACHER" --out-dir "data/esci_k8_k1/s${SEED}" \
        --name "seqkd_t4b_k1_x5_s${SEED}" --seed $((SEED * 100000)) --tp "$NGPUS" --snap "$SNAP" | tee "$OUT/sample/sample.log"
    cp "data/esci_k8_k1/s${SEED}/seqkd_t4b_k1_x5_s${SEED}.stats.json" "$OUT/sample/"
    ts > "$D" ;;
  sft)
    bash scripts/sft_seqkd_qwen3.sh "b17_seqkd_t4b_k1_s${SEED}" ;;
  eval)
    D="$OUT/eval/DONE"; [ -f "$D" ] && { echo "[k1] eval DONE"; exit 0; }
    S=$(erropd_staged "$REPO/$OUT/sft")
    mkdir -p "$OUT/eval"
    META=$(printf '{"id":"K1_s%s_greedy","experiment":"K1","task":"esci_k8","student":"qwen3-1.7b-seqkd-f3-t4b-k1","teacher":"f3-t4b-k1","seed":%s,"variant":"seqkd","notes":"SFT of Qwen3-1.7B on one T=1 sample of the F3 injected teacher per row of F3 clean SFT set; rollout rendering"}' "$SEED" "$SEED")
    [ -f "$OUT/eval/greedy/run_meta.json" ] || "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py \
        --model-path "$S" --val-jsonl data/esci_k8/val.jsonl --output-dir "$OUT/eval/greedy" --tp "$NGPUS" \
        --task esci --template train --meta "$META"
    "$ERROPD_PY_VLLM" scripts/eval_seqkd_sampled.py --out "$OUT/eval/sampled" --tp "$NGPUS" --snap "$SNAP" \
        --model "K1_s${SEED}=$S"
    ts > "$D" ;;
  refs)
    D=outputs/K1/refs/DONE; [ -f "$D" ] && { echo "[k1] refs DONE"; exit 0; }
    mkdir -p outputs/K1/refs
    "$ERROPD_PY_VLLM" scripts/eval_seqkd_sampled.py --out outputs/K1/refs/sampled --tp "$NGPUS" \
        --snap "$SNAP" --model "teacher_t4b_k1=$TEACHER" \
        --model "base_b17_clean=$(erropd_staged "$REPO/outputs/F3/sft_b17_clean")" \
        --model "base_b17_k1_f20=$(erropd_staged "$REPO/outputs/F3/sft_b17_k1_f20")"
    ts > "$D" ;;
  *) echo "bad step $STEP"; exit 2 ;;
esac
