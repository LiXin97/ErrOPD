#!/bin/bash
set -euo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
LAM="${1:?LAMBDA e.g. 1.00}"
SEED="${2:?SEED e.g. 2001}"
REPO="${ERROPD_ROOT:-/path/to/erropd}"
cd "$REPO"
source scripts/env.sh
source scripts/opd_resume_lib.sh
: "${R4_EXP:?R4_EXP required}" "${R4_TEACHER_REL:?R4_TEACHER_REL required}" "${R4_TTAG:?R4_TTAG required}"
: "${RUN_ID:?RUN_ID required (set by the worker)}" "${XFAMILY_VERL_SNAP:?XFAMILY_VERL_SNAP required}"

RUN="$RUN_ID"
SAMPLE_PORT0=$(( ${ERROPD_PORT_BASE:-30000} + 650 ))
OUT="$REPO/outputs/$R4_EXP/$RUN"
CKPT="$OUT/ckpt"
DONE="$OUT/DONE"
[ -f "$DONE" ] && { echo "[r4_llama] $RUN already DONE"; exit 0; }
mkdir -p "$CKPT"
SPE=14
TOTAL=${R4_TOTAL_STEPS:-42}
read -r -a EVAL_STEPS <<< "${R4_EVAL_STEPS:-14 28 42}"
SAMPLE_STEPS=" ${R4_SAMPLE_STEPS:-} "
NGPUS=${NGPUS:-2}
RESUME_OPT=${RESUME:-0}
ts() { date -u +%FT%TZ; }

STUDENT="$REPO/${XFAMILY_STUDENT:-outputs/D1/sft_llama3.2_3b}"
TEACHER="$REPO/$R4_TEACHER_REL"
SAVE_FREQ=${R4_SAVE_FREQ:-14}; TEST_FREQ=42; STEP_ARGS=(); EPOCHS=3
if [ -n "${R4_TOTAL_STEPS:-}" ]; then
  EPOCHS=$(( (TOTAL + SPE - 1) / SPE )); STEP_ARGS=(trainer.total_training_steps=$TOTAL)
fi
VERL_DIR="$(erropd_staged "$XFAMILY_VERL_SNAP")/verl"; VERL_LABEL="erropd-$(basename "$XFAMILY_VERL_SNAP")"
[ -f "$VERL_DIR/verl/erropd/hooks.py" ] || { echo "[r4_llama] no ERROPD verl tree at $VERL_DIR"; exit 3; }
case "${PAPER_LOSS_MODE:-}" in opd|fkl_sampled) export PAPER_LOSS_MODE ;;
  *) echo "[r4_llama] PAPER_LOSS_MODE must be set explicitly (opd|fkl_sampled)"; exit 3 ;; esac
ERROPD_ARGS=(+erropd.dump_rollouts=true)
unset PAPER_ADV_MODE
KEEP_CKPT=${XFAMILY_KEEP_CKPT:-0}
META_EXP=${XFAMILY_META_EXP:-$R4_EXP}; VARIANT=${XFAMILY_VARIANT:-vanilla}
[ -f outputs/D1/sft_llama3.2_3b/DONE ] || { echo "[r4_llama] D1 student SFT missing"; exit 5; }
[ -f "$R4_TEACHER_REL/DONE" ] || { echo "[r4_llama] teacher $R4_TEACHER_REL not finished"; exit 5; }
if [ -n "${R4_GATE_FILE:-}" ]; then
  [ -f "$R4_GATE_FILE" ] || { echo "[r4_llama] gate file $R4_GATE_FILE missing: gates not passed or gate amendment not stamped"; exit 5; }
fi
S=$(erropd_staged "$STUDENT")
T=$(erropd_staged "$TEACHER")

last_ckpt() { cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
CKPT_OPTIM=${CKPT_OPTIM:-1}; EVAL_ONE_PROC=${EVAL_ONE_PROC:-0}
CKPT_ARGS=(); RESUME=disable; RSTEPS=""; K=""
if [ "$RESUME_OPT" = 1 ]; then
  [ "$CKPT_OPTIM" = 1 ] || { echo "[r4_llama] RESUME=1 needs CKPT_OPTIM=1"; exit 2; }
  RSTEPS=$(r4_save_steps "$TOTAL" "$SPE" "${R4_RESUME_EVERY:-14}" "${R4_RESUME_OFFSET:-7}")
  [ -n "$RSTEPS" ] && ERROPD_ARGS+=("+erropd.save_steps=$RSTEPS")
  if [ "$(last_ckpt)" -lt "$TOTAL" ]; then
    K=$(r4_resume_point "$CKPT" "$NGPUS" "$SPE")
    if [ -n "$K" ]; then
      RESUME=auto
      r4_prepare_resume "$CKPT" "" "$K"
      echo "$(ts) resumed from global_step_$K host=$(hostname)" >> "$OUT/RESUMED"
      echo "$(ts) host=$(hostname) start=resume:$K" >> "$OUT/ATTEMPTS"
      echo "[r4_llama] $(ts) RESUME from global_step_$K (RESUME=1; save_steps=$RSTEPS)"
    else
      echo "[r4_llama] $(ts) RESUME=1: no complete resume point (tracker=$(last_ckpt)); from scratch"
      rm -rf "$CKPT"; mkdir -p "$CKPT"; rm -f "$OUT/RESUMED"
      echo "$(ts) host=$(hostname) start=scratch" >> "$OUT/ATTEMPTS"
    fi
  else
    echo "$(ts) host=$(hostname) start=post-train:$(last_ckpt)" >> "$OUT/ATTEMPTS"
    echo "[r4_llama] $(ts) RESUME=1: training already complete (tracker=$(last_ckpt)); continuing after training"
  fi
elif [ "$CKPT_OPTIM" = 0 ]; then
  CKPT_ARGS=('actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'); RESUME=disable
  if [ "$(last_ckpt)" -lt "$TOTAL" ]; then rm -rf "$CKPT"; mkdir -p "$CKPT"; echo "$(ts) host=$(hostname) start=scratch" >> "$OUT/ATTEMPTS"; fi
else
  echo "[r4_llama] set RESUME=1 or CKPT_OPTIM=0 (an implicit resume from an epoch-boundary checkpoint is not allowed here)"; exit 2
fi

BASE_TOK=$(erropd_staged /path/to/models/Llama-3.2-3B-Instruct)
merge_step() {
  local G="$CKPT/global_step_$1" HF="$CKPT/global_step_$1/hf"
  if ls "$G"/actor/model_world_size_*.pt >/dev/null 2>&1 && [ -d "$HF" ] && [ ! -f "$HF/.merge_complete" ]; then
    echo "[r4_llama] step $1: hf dir without .merge_complete and shards present; re-merging"; rm -rf "$HF"
  fi
  if [ ! -f "$HF/config.json" ] || ! ls "$HF"/*.safetensors >/dev/null 2>&1; then
    [ -d "$G/actor" ] || { echo "[r4_llama] missing $G/actor"; return 4; }
    PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}" "$ERROPD_PY_VERL" "$VERL_DIR/scripts/legacy_model_merger.py" merge \
        --backend fsdp --local_dir "$G/actor" --target_dir "$HF" || { echo "[r4_llama] merge failed step $1"; return 6; }
  fi
  "$ERROPD_PY_VLLM" scripts/common_llama_template.py install --model-dir "$HF" --base-dir "$BASE_TOK" \
      || { echo "[r4_llama] template install failed step $1"; return 6; }
  touch "$HF/.merge_complete"
}
r4_merge_step() { merge_step "$1" > "${ERROPD_RUN_TMP:-/tmp}/r4_bgmerge_$1.log" 2>&1; }

KPID=""; KSTOP="$OUT/.keeper_stop"
trap '[ -n "$KPID" ] && { touch "$KSTOP" 2>/dev/null; kill "$KPID" 2>/dev/null; } || true' EXIT
if [ "$(last_ckpt)" -lt "$TOTAL" ]; then
  export WANDB_MODE=disabled USED_MODEL=no_api
  unset PAPER_NOBASE_ADVANTAGE PAPER_FORWARD_KL_ON_ROLLOUTS
  export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}"
  rm -f "$KSTOP"
  if [ "$RESUME_OPT" = 1 ]; then r4_keeper "$CKPT" "$SPE" "$KSTOP" "${EVAL_STEPS[*]}" "$$" & KPID=$!; fi
  cd "$VERL_DIR"
  echo "[r4_llama] $(ts) train $RUN lambda=$LAM seed=$SEED NGPUS=$NGPUS resume_from=${K:-0} student=$S teacher=$T verl=$VERL_DIR PAPER_LOSS_MODE=$PAPER_LOSS_MODE keep_ckpt=$KEEP_CKPT total=$TOTAL epochs=$EPOCHS save_freq=$SAVE_FREQ"
  set +e
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
    trainer.project_name=erropd-r4 \
    trainer.experiment_name="$RUN" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$EPOCHS \
    trainer.resume_mode=$RESUME \
    trainer.default_local_dir="$CKPT" \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} ${ERROPD_ARGS[@]+"${ERROPD_ARGS[@]}"} $ERROPD_VERL_RAY_ARGS
  rc=$?
  set -e
  cd "$REPO"
  if [ -n "$KPID" ]; then touch "$KSTOP"; wait "$KPID" || true; KPID=""; rm -f "$KSTOP"; fi
  echo "[r4_llama] $(ts) TRAIN_EXIT rc=$rc"
  [ "$rc" -eq 0 ] || exit "$rc"
fi
[ "$(last_ckpt)" -ge "$TOTAL" ] || { echo "[r4_llama] training ended at step $(last_ckpt) < $TOTAL"; exit 4; }
if [ "$RESUME_OPT" = 1 ]; then r4_final_cleanup "$CKPT" "${EVAL_STEPS[*]}"; fi

TP=${XFAMILY_EVAL_TP:-4}; (( NGPUS < TP )) && TP=$NGPUS
meta_for() {
  printf '{"id":"%s_step%s","experiment":"%s","task":"fashion_k8","student":"llama3.2-3b","teacher":"%s","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"%s","clip_c":5,%s"notes":"teacher %s; Llama template = SFT rendering in SFT, rollout and eval; prompt as token ids; student %s; n_gpus %s; ckpt_optim %s; eval_one_proc %s; verl %s; PAPER_LOSS_MODE %s; resume %s"}' \
         "$RUN" "$1" "$META_EXP" "$R4_TTAG" "$LAM" "$SEED" "$1" "$TOTAL" "$VARIANT" "$(r4_resumed_kv "$OUT/RESUMED")" "$R4_TEACHER_REL" "${STUDENT#$REPO/}" "$NGPUS" "$CKPT_OPTIM" "$EVAL_ONE_PROC" "$VERL_LABEL" "$PAPER_LOSS_MODE" "$RESUME_OPT"
}
if [ "$EVAL_ONE_PROC" = 1 ]; then
  TSV=${ERROPD_RUN_TMP:-/tmp}/r4_evalbatch_$RUN.tsv; : > "$TSV"
  for step in "${EVAL_STEPS[@]}"; do
    EV="$OUT/eval_step$step"
    [ -f "$EV/listwise_metrics.json" ] && [ -f "$EV/run_meta.json" ] && continue
    merge_step "$step" || exit $?
    printf '%s\t%s\t%s\n' "$CKPT/global_step_$step/hf" "$EV" "$(meta_for "$step")" >> "$TSV"
  done
  if [ -s "$TSV" ]; then
    BATCH="${TSV%.tsv}.json"
    python3 -c "import json,sys; json.dump([dict(model_path=a, output_dir=b, meta=json.loads(c)) for a, b, c in (l.rstrip('\n').split('\t') for l in open(sys.argv[1]))], open(sys.argv[2], 'w'))" "$TSV" "$BATCH"
    echo "[r4_llama] $(ts) batch eval (one process, TP=$TP): $(wc -l < "$TSV") step(s)"
    "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py --batch-json "$BATCH" --family llama --tp "$TP" \
        || echo "[r4_llama] batch eval rc!=0; per-step fallback follows"
  fi
fi
for step in "${EVAL_STEPS[@]}"; do
  G="$CKPT/global_step_$step"
  HF="$G/hf"
  EV="$OUT/eval_step$step"
  if [ ! -f "$EV/listwise_metrics.json" ] || [ ! -f "$EV/run_meta.json" ]; then
    merge_step "$step" || exit $?
    "$ERROPD_PY_VLLM" scripts/eval_greedy_xfamily.py --model-path "$HF" --family llama --tp "$TP" \
        --output-dir "$EV" --meta "$(meta_for "$step")"
  fi
  [ -f "$EV/run_meta.json" ] || { echo "[r4_llama] eval failed step $step"; exit 7; }
  if [[ "$SAMPLE_STEPS" == *" $step "* ]]; then
    SO="$OUT/samples/step$step"
    if [ ! -f "$SO.v2.json" ]; then
      mkdir -p "$OUT/samples"
      merge_step "$step" || exit $?
      echo "[r4_llama] $(ts) SAMPLE step $step (TP=$TP)"
      for t in 1 2 3; do
        src=0
        VLLM_PORT=$(( SAMPLE_PORT0 + 100 * (t - 1) )) "$ERROPD_PY_VLLM" scripts/eval_sample_validity.py \
            --model-path "$HF" --family llama --tp "$TP" --out "$SO" > "$SO.log" 2>&1 || src=$?
        [ -f "$SO.v2.json" ] && break
        echo "[r4_llama] $(ts) sampler step $step try $t failed (rc=$src); retry in 30 s"; tail -5 "$SO.log" || true
        if [ "$t" -lt 3 ]; then sleep 30; fi
      done
    fi
    [ -f "$SO.v2.json" ] || { echo "[r4_llama] sampler failed step $step after 3 tries"; tail -20 "$SO.log"; exit 8; }
    echo "[r4_llama] $(ts) SAMPLED step $step"
  fi
  rm -f "$G"/actor/*.pt "$G"/data.pt
  rm -rf "$G"/actor/extra_state* "$G"/actor/optim*
  if [ "$step" != "$TOTAL" ] && [ "$KEEP_CKPT" != 1 ]; then rm -rf "$HF"; fi
done

python3 - "$OUT" "$RESUME_OPT" <<'PYEOF'
import json, os, re, sys
out, resume = sys.argv[1], sys.argv[2]
pts = [int(x) for x in re.findall(r"global_step_(\d+)", open(f"{out}/RESUMED").read())] if os.path.exists(f"{out}/RESUMED") else []
att = [l.strip() for l in open(f"{out}/ATTEMPTS")] if os.path.exists(f"{out}/ATTEMPTS") else []
json.dump({"resume_enabled": resume == "1", "resumed": bool(pts), "resume_points": pts, "attempts": att},
          open(f"{out}/run_resume.json", "w"), indent=1)
PYEOF
touch "$DONE"
echo "[r4_llama] $(ts) $RUN DONE"
