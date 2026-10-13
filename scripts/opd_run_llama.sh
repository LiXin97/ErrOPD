#!/bin/bash
set -euo pipefail
LAM="${1:?LAMBDA e.g. 1.00}"
SEED="${2:?SEED e.g. 42}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/env.sh

RUN="lam${LAM}_s${SEED}"
BRANCH_S=${XFAMILY_BRANCH_S:-0}
[ "$BRANCH_S" = 1 ] && RUN="S_lam${LAM}_s${SEED}"
EXP=${XFAMILY_EXP:-D1}
TEACHER_REL=outputs/D1/sft_llama3.1_8b; TTAG=llama3.1-8b; OUTROOT=outputs/D1/opd; PAIR=""
if [ "$EXP" = D1inj ]; then
  PAIR=${XFAMILY_PAIR:?XFAMILY_PAIR=A|0|smoke}
  case "$PAIR" in
    A) TEACHER_REL=outputs/D1inj/sft_t8b_k1; TTAG=llama3.1-8b-k1 ;;
    0) TEACHER_REL=outputs/D1inj/sft_t8b_clean; TTAG=llama3.1-8b-clean ;;
    smoke) ;;
    *) echo "[opd_run_llama] bad XFAMILY_PAIR=$PAIR"; exit 2 ;;
  esac
  RUN="${PAIR}_lam${LAM}_s${SEED}"; OUTROOT=outputs/D1inj/opd
fi
[ -n "${XFAMILY_RUN_TAG:-}" ] && RUN="${RUN}_${XFAMILY_RUN_TAG}"
OUT="$REPO/$OUTROOT/$RUN"
CKPT="$OUT/ckpt"
DONE="$OUT/DONE"
[ -f "$DONE" ] && { echo "[opd_run_llama] $RUN already DONE"; exit 0; }
mkdir -p "$CKPT"
TOTAL=42
EVAL_STEPS=(14 28 42)
NGPUS=${NGPUS:-4}
ts() { date -u +%FT%TZ; }

STUDENT="$REPO/${XFAMILY_STUDENT:-outputs/D1/sft_llama3.2_3b}"
TEACHER="$REPO/$TEACHER_REL"
SAVE_FREQ=14; TEST_FREQ=42; STEP_ARGS=()
if [ -n "${XFAMILY_SMOKE_STEPS:-}" ]; then
  TOTAL=$XFAMILY_SMOKE_STEPS; EVAL_STEPS=("$TOTAL"); SAVE_FREQ=$TOTAL; TEST_FREQ=-1; STEP_ARGS=(trainer.total_training_steps=$TOTAL)
fi
VERL_DIR=$ERROPD_VERL_PROD; VERL_LABEL=verl_prod_20260930; ERROPD_ARGS=()
if [ -n "${XFAMILY_VERL_SNAP:-}" ]; then
  VERL_DIR="$(erropd_staged "$XFAMILY_VERL_SNAP")/verl"; VERL_LABEL="erropd-$(basename "$XFAMILY_VERL_SNAP")"
  [ -f "$VERL_DIR/verl/erropd/hooks.py" ] || { echo "[opd_run_llama] no ERROPD verl tree at $VERL_DIR"; exit 3; }
  case "${PAPER_LOSS_MODE:-}" in opd|fkl_sampled) export PAPER_LOSS_MODE ;;
    *) echo "[opd_run_llama] PAPER_LOSS_MODE must be set explicitly (opd|fkl_sampled) with a ERROPD snapshot"; exit 3 ;; esac
  ERROPD_ARGS=(+erropd.dump_rollouts=true)
else
  if [ -n "${PAPER_LOSS_MODE:-}" ] && [ "$PAPER_LOSS_MODE" != opd ]; then echo "[opd_run_llama] PAPER_LOSS_MODE=$PAPER_LOSS_MODE needs XFAMILY_VERL_SNAP"; exit 3; fi
  unset PAPER_LOSS_MODE
fi
unset PAPER_ADV_MODE
KEEP_CKPT=${XFAMILY_KEEP_CKPT:-0}
META_EXP=${XFAMILY_META_EXP:-$EXP}; VARIANT=${XFAMILY_VARIANT:-vanilla}
if [ "$EXP" = D1inj ]; then
  [ -f outputs/D1/sft_llama3.2_3b/DONE ] || { echo "[opd_run_llama] D1 student SFT missing"; exit 5; }
  if [ "$PAIR" = smoke ]; then
    SAVE_FREQ=-1; TEST_FREQ=-1; STEP_ARGS=(trainer.total_training_steps=4)
  else
    [ -f "$TEACHER_REL/DONE" ] || { echo "[opd_run_llama] teacher $TEACHER_REL not finished"; exit 5; }
    [ -f outputs/D1inj/GATES_PASS ] || { echo "[opd_run_llama] D1-inj gates G1/G2 not passed or prediction not stamped (outputs/D1inj/GATES_PASS)"; exit 5; }
  fi
else
for f in outputs/D1/sft_llama3.2_3b/DONE outputs/D1/sft_llama3.1_8b/DONE outputs/D1/selected_student.json; do
  [ -f "$f" ] || { echo "[opd_run_llama] missing $f (Stage A SFT not finished)"; exit 5; }
done
"$ERROPD_PY_VLLM" - "$STUDENT" "$BRANCH_S" <<'EOF'
import json, os, sys
sel = json.load(open("outputs/D1/selected_student.json"))
want = os.path.realpath(sys.argv[1])
if sys.argv[2] == "1":
    if sel["status"] != "SATURATED" or not want.endswith("outputs/D1/sft_llama3.2_3b"):
        sys.exit(f"[opd_run_llama] branch S needs status SATURATED and the epoch-5 student; got {sel['status']} {want}")
    print("[opd_run_llama] branch S: saturated warm-start, epoch-5 student, parse", sel["parse_by_epoch"]["5"])
    sys.exit(0)
if sel["status"] != "OK" or os.path.realpath(sel["path"]) != want:
    sys.exit(f"[opd_run_llama] selected student {sel} != {want}; resubmit with XFAMILY_STUDENT set per the prereg rule")
s = sel["parse_by_epoch"][str(sel["selected_epoch"])]
t = json.load(open("outputs/D1/eval/sft_teacher_ep5/listwise_metrics.json"))["parse_rate"]
print(f"[opd_run_llama] student parse {s:.4f}, teacher parse {t:.4f} (teacher parse is descriptive, not a gate)")
EOF
fi
if [ "$EXP" = D1 ] && [ "$RUN" != "lam1.00_s42" ] && [ "$BRANCH_S" != 1 ] && [ -z "${XFAMILY_SKIP_GATE:-}" ]; then
  "$ERROPD_PY_VLLM" - <<'EOF'
import json, sys
sel = json.load(open("outputs/D1/selected_student.json"))
p_sft = sel["parse_by_epoch"][str(sel["selected_epoch"])]
try:
    p = json.load(open("outputs/D1/opd/lam1.00_s42/eval_step42/listwise_metrics.json"))["parse_rate"]
except FileNotFoundError:
    sys.exit("[opd_run_llama] gate A3 not evaluated yet (lam1.00_s42 step 42 missing)")
margin = min(0.10, (0.953 - p_sft) / 2)
print(f"[opd_run_llama] gate A3: p_opd={p:.4f} p_sft={p_sft:.4f} margin={margin:.4f} pass={p >= p_sft + margin}")
if p < p_sft + margin:
    sys.exit(7)
EOF
fi
S=$(erropd_staged "$STUDENT")
T=$(erropd_staged "$TEACHER")

last_ckpt() { cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
CKPT_OPTIM=${CKPT_OPTIM:-1}; EVAL_ONE_PROC=${EVAL_ONE_PROC:-0}
CKPT_ARGS=(); RESUME=auto
if [ "$CKPT_OPTIM" = 0 ]; then
  CKPT_ARGS=('actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'); RESUME=disable
  if [ "$(last_ckpt)" -lt "$TOTAL" ]; then rm -rf "$CKPT"; mkdir -p "$CKPT"; fi
fi

if [ "$(last_ckpt)" -lt "$TOTAL" ]; then
  export WANDB_MODE=disabled USED_MODEL=no_api
  unset PAPER_NOBASE_ADVANTAGE PAPER_FORWARD_KL_ON_ROLLOUTS
  export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}"
  cd "$VERL_DIR"
  echo "[opd_run_llama] $(ts) train $RUN lambda=$LAM seed=$SEED NGPUS=$NGPUS resume_from=$(last_ckpt) student=$S teacher=$T verl=$VERL_DIR PAPER_LOSS_MODE=${PAPER_LOSS_MODE:-unset(opd)} keep_ckpt=$KEEP_CKPT"
  "$ERROPD_PY_VERL" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files="$REPO/data/pl_k8_parquet/train.parquet" \
    data.val_files="$REPO/data/pl_k8_parquet/val.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed="$SEED" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$S" \
    +actor_rollout_ref.model.base_model_path="$S" \
    +actor_rollout_ref.ref.model.path="$T" \
    +actor_rollout_ref.ref.model.base_model_path="$S" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.lambda_vals="$LAM" \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${XFAMILY_MICRO_BS:-2}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${XFAMILY_LOGPROB_MICRO_BS:-4}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${XFAMILY_ROLLOUT_MEM:-0.6}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${XFAMILY_LOGPROB_MICRO_BS:-4}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name=erropd-D1 \
    trainer.experiment_name="D1_$RUN" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=3 \
    trainer.resume_mode=$RESUME \
    trainer.default_local_dir="$CKPT" \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} ${ERROPD_ARGS[@]+"${ERROPD_ARGS[@]}"} $ERROPD_VERL_RAY_ARGS
  cd "$REPO"
fi
if [ "$PAIR" = smoke ]; then
  touch "$DONE"; echo "[opd_run_llama] $(ts) smoke (4 steps, NGPUS=$NGPUS) finished without error"; exit 0
fi
[ "$(last_ckpt)" -ge "$TOTAL" ] || { echo "[opd_run_llama] training ended at step $(last_ckpt) < $TOTAL"; exit 4; }

TP=${XFAMILY_EVAL_TP:-4}; (( NGPUS < TP )) && TP=$NGPUS
BASE_TOK=$(erropd_staged /path/to/models/Llama-3.2-3B-Instruct)
meta_for() {
  local NOTE=""
  [ "$BRANCH_S" = 1 ] && NOTE="branch S (saturated warm-start); "
  [ "$EXP" = D1inj ] && NOTE="pair $PAIR (teacher $TEACHER_REL); "
  printf '{"id":"%s_%s_step%s","experiment":"%s","task":"fashion_k8","student":"llama3.2-3b","teacher":"%s","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"%s","clip_c":5,"notes":"%sLlama template = SFT rendering in SFT, rollout and eval; prompt as token ids; student %s; n_gpus %s; ckpt_optim %s; eval_one_proc %s; verl %s; PAPER_LOSS_MODE %s"}' \
         "$META_EXP" "$RUN" "$1" "$META_EXP" "$TTAG" "$LAM" "$SEED" "$1" "$TOTAL" "$VARIANT" "$NOTE" "${STUDENT#$REPO/}" "$NGPUS" "$CKPT_OPTIM" "$EVAL_ONE_PROC" "$VERL_LABEL" "${PAPER_LOSS_MODE:-opd}"
}
merge_step() {
  local G="$CKPT/global_step_$1" HF="$CKPT/global_step_$1/hf"
  if [ ! -f "$HF/config.json" ] || ! ls "$HF"/*.safetensors >/dev/null 2>&1; then
    [ -d "$G/actor" ] || { echo "[opd_run_llama] missing $G/actor"; exit 4; }
    PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}" "$ERROPD_PY_VERL" "$VERL_DIR/scripts/legacy_model_merger.py" merge \
        --backend fsdp --local_dir "$G/actor" --target_dir "$HF"
  fi
  "$ERROPD_PY_VLLM" scripts/common_llama_template.py install --model-dir "$HF" --base-dir "$BASE_TOK"
}
if [ "$EVAL_ONE_PROC" = 1 ]; then
  TSV=${ERROPD_RUN_TMP:-/tmp}/xfamily_evalbatch_$RUN.tsv; : > "$TSV"
  for step in "${EVAL_STEPS[@]}"; do
    EV="$OUT/eval_step$step"
    [ -f "$EV/listwise_metrics.json" ] && [ -f "$EV/run_meta.json" ] && continue
    merge_step "$step"
    printf '%s\t%s\t%s\n' "$CKPT/global_step_$step/hf" "$EV" "$(meta_for "$step")" >> "$TSV"
  done
  if [ -s "$TSV" ]; then
    BATCH="${TSV%.tsv}.json"
    python3 -c "import json,sys; json.dump([dict(model_path=a, output_dir=b, meta=json.loads(c)) for a, b, c in (l.rstrip('\n').split('\t') for l in open(sys.argv[1]))], open(sys.argv[2], 'w'))" "$TSV" "$BATCH"
    echo "[opd_run_llama] $(ts) batch eval (one process, TP=$TP): $(wc -l < "$TSV") step(s)"
    "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py --batch-json "$BATCH" --family llama --tp "$TP" \
        || echo "[opd_run_llama] batch eval rc!=0; per-step fallback follows"
  fi
fi
for step in "${EVAL_STEPS[@]}"; do
  G="$CKPT/global_step_$step"
  HF="$G/hf"
  EV="$OUT/eval_step$step"
  if [ ! -f "$EV/listwise_metrics.json" ] || [ ! -f "$EV/run_meta.json" ]; then
    merge_step "$step"
    "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py --model-path "$HF" --family llama --tp "$TP" \
        --output-dir "$EV" --meta "$(meta_for "$step")"
  fi
  [ -f "$EV/run_meta.json" ] || { echo "[opd_run_llama] eval failed step $step"; exit 7; }
  rm -f "$G"/actor/*.pt "$G"/data.pt
  rm -rf "$G"/actor/extra_state* "$G"/actor/optim*
  if [ "$step" != "$TOTAL" ] && [ "$KEEP_CKPT" != 1 ]; then rm -rf "$HF"; fi
done

touch "$DONE"
echo "[opd_run_llama] $(ts) $RUN DONE"
