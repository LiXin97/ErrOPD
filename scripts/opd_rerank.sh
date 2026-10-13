#!/bin/bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
LAM="${1:?LAMBDA}"; SEED="${2:?SEED}"; STAGE="${3:-grid}"; EXP=T1
cd "${ERROPD_ROOT:-/path/to/erropd}"
REPO=$(pwd)
source scripts/env.sh
LX=/path/to
RUN="lam${LAM}_s${SEED}"
OUT=$REPO/outputs/T1/opd/$RUN
CKPT=$OUT/ckpt
TOTAL=42
EVAL_STEPS=${T1_EVAL_STEPS:-"14 28 42"}
NGPUS=${NGPUS:-2}
EVAL_TP=${EVAL_TP:-2}
DATA=$REPO/data/erropd_t1/parquet
STUDENT=$REPO/outputs/T1/sft_1.7b
TEACHER=$REPO/outputs/T1/sft_4b
TOKCFG=$LX/models/Qwen3-1.7B/tokenizer_config.json
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [opd_rerank $RUN] $*"; }
[ -f "$OUT/DONE" ] && { say "DONE exists"; exit 0; }
mkdir -p "$OUT"
for f in "$STUDENT/DONE" "$TEACHER/DONE"; do [ -f "$f" ] || { say "missing $f (T1 SFT not finished)"; exit 5; }; done
if [ "$STAGE" != "gate" ] && [ ! -f "$REPO/outputs/T1/STAGE_A_PASS" ]; then
  say "outputs/T1/STAGE_A_PASS missing: Stage A gate not passed"; exit 7
fi
S_LOCAL=$(erropd_staged "$STUDENT"); T_LOCAL=$(erropd_staged "$TEACHER")
say "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} n=$NGPUS lam=$LAM seed=$SEED student=$S_LOCAL teacher=$T_LOCAL"
export PYTHONPATH=$ERROPD_VERL_PROD${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled USED_MODEL=no_api HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS PAPER_NOBASE_ADVANTAGE PAPER_FORWARD_KL_ON_ROLLOUTS

if [ ! -f "$OUT/TEMPLATE_OK" ]; then
  "$ERROPD_PY_VERL" scripts/check_rerank_template.py --stock "$LX/models/Qwen3-1.7B" --tok-dir "$S_LOCAL" --verl \
    > "$OUT/template_check.log" 2>&1 || { say "template check failed (see $OUT/template_check.log)"; exit 6; }
  grep '^\[t1_check\]' "$OUT/template_check.log"
  touch "$OUT/TEMPLATE_OK"
fi

if [ ! -f "$OUT/TRAIN_DONE" ]; then
  rm -rf "$CKPT" "$OUT/rollouts"; mkdir -p "$CKPT"
  say "TRAIN_START"
  ( cd "$ERROPD_VERL_PROD" && "$ERROPD_PY_VERL" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=$SEED \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$S_LOCAL \
    +actor_rollout_ref.model.base_model_path=$S_LOCAL \
    +actor_rollout_ref.ref.model.path=$T_LOCAL \
    +actor_rollout_ref.ref.model.base_model_path=$S_LOCAL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.lambda_vals=$LAM \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.logger='["console"]' \
    trainer.project_name='erropd-erropd' \
    trainer.experiment_name="T1_$RUN" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=14 \
    trainer.default_local_dir=$CKPT \
    trainer.resume_mode=disable \
    trainer.total_epochs=3 \
    trainer.rollout_data_dir=$OUT/rollouts \
    $ERROPD_VERL_RAY_ARGS )
  rc=$?
  say "TRAIN_EXIT rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  [ -d "$CKPT/global_step_$TOTAL/actor" ] || { say "missing global_step_$TOTAL after training"; exit 4; }
  touch "$OUT/TRAIN_DONE"
fi

for S in $EVAL_STEPS; do
  GS=$CKPT/global_step_$S; HF=$GS/hf; EV=$OUT/eval_step$S
  if [ ! -f "$EV/run_meta.json" ]; then
    if [ ! -f "$HF/config.json" ] || ! ls "$HF"/*.safetensors >/dev/null 2>&1; then
      [ -d "$GS/actor" ] || { say "no actor dir for step $S"; exit 5; }
      say "MERGE step $S"
      "$ERROPD_PY_VERL" -m verl.model_merger merge --backend fsdp --local_dir "$GS/actor" --target_dir "$HF" 2>&1 | tail -3
      [ -f "$HF/config.json" ] || { say "merge failed step $S"; exit 6; }
    fi
    cp "$TOKCFG" "$HF/tokenizer_config.json"
    "$ERROPD_PY_VLLM" - "$TOKCFG" "$HF" <<'EOF'
import json, sys
from pathlib import Path
ct = json.load(open(sys.argv[1]))["chat_template"]
j = Path(sys.argv[2]) / "chat_template.jinja"
if j.exists() and j.read_text() != ct:
    j.write_text(ct); print(f"rewrote {j} with the stock template")
EOF
    "$ERROPD_PY_VLLM" scripts/check_rerank_template.py --stock "$LX/models/Qwen3-1.7B" --tok-dir "$HF" \
      | grep '^\[t1_check\]' || { say "merged tokenizer renders differently"; exit 6; }
    SLIDE=(--no-sliding); [ "$S" = "$TOTAL" ] && SLIDE=()
    META=$(printf '{"id":"T1_%s_step%s","experiment":"%s","task":"msmarco_rankgpt_k20","student":"qwen3-1.7b","teacher":"qwen3-4b","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"vanilla","clip_c":5,"notes":"n_gpus=%s rollout_tp=2 single template (SFT=rollout=eval) verl=verl_prod_20260930 run=%s"}' \
           "$RUN" "$S" "$EXP" "$LAM" "$SEED" "$S" "$TOTAL" "$NGPUS" "$RUN")
    say "EVAL step $S (TP=$EVAL_TP)"
    "$ERROPD_PY_VLLM" scripts/eval_rerank.py --model-path "$HF" --tp "$EVAL_TP" --output-dir "$EV" --meta "$META" \
        ${SLIDE[@]+"${SLIDE[@]}"} 2>&1 | grep -v -E '^(INFO|WARNING|\[rank)' | tail -5
    [ -f "$EV/run_meta.json" ] || { say "eval failed step $S"; exit 7; }
  fi
  rm -rf "$GS/actor" "$GS/data.pt"
  [ "$S" = "$TOTAL" ] || rm -rf "$HF"
done
say "RESULT $(for S in $EVAL_STEPS; do python3 -c "import json;m=json.load(open('$OUT/eval_step$S/listwise_metrics.json'));print('step$S valid=%.4f ndcg10_valid=%s useful=%.4f' % (m['parse_rate'], m['ndcg@10_mean'], m['useful']))"; done | paste -sd' ' -)"
touch "$OUT/DONE"
