#!/bin/bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
LAM="${1:?LAMBDA e.g. 1.00}"
SEED="${2:?SEED e.g. 42}"
REPO=/path/to/erropd
cd "${ERROPD_ROOT:-$REPO}"
REPO=$(pwd)
source scripts/env.sh
LOCK=${T3_LOCK:-}
EXP=${T3_EXP:-T3}
EPOCHS=${T3_EPOCHS:-3}
RUN="lam${LAM}_s${SEED}"; [ "$EPOCHS" != 3 ] && RUN="${RUN}_ep${EPOCHS}"
OUT="$REPO/outputs/$EXP/opd/$RUN"
CKPT="$OUT/ckpt"
DONE="$OUT/DONE"
TOTAL=$(( EPOCHS * 14 ))
SAVE_FREQ=${T3_SAVE_FREQ:-14}
EVAL_STEPS=${T3_EVAL_STEPS:-"14 28 42"}
CKPT_OPTIM=${CKPT_OPTIM:-1}
EVAL_ONE_PROC=${EVAL_ONE_PROC:-0}
VARIANT=${T3_VARIANT:-vanilla}
NGPUS=${NGPUS:-2}
EVAL_TP=2
TOKCFG=/path/to/models/Qwen3-1.7B/tokenizer_config.json
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [opd_ner_grid $EXP $RUN] $*"; }
[ -f "$DONE" ] && { say "DONE exists"; exit 0; }
mkdir -p "$OUT"

TEACHER_REL=$("$ERROPD_PY_VLLM" scripts/opd_ner_teacher_select.py a1) || { rc=$?; say "rule A1 not satisfied (rc=$rc)"; exit $rc; }
TEACHER_TAG=$("$ERROPD_PY_VLLM" -c 'import json;print(json.load(open("outputs/T3/stageA_a1.json"))["teacher_tag"])')
if [ "$EXP" = T3 ] && [ "$EPOCHS" = 3 ]; then
  if [ "$RUN" != "lam1.00_s42" ] && [ -z "${T3_SKIP_GATE:-}" ]; then
    if ! "$ERROPD_PY_VLLM" scripts/opd_ner_teacher_select.py a3; then
      grep -q '^## Amendment 2' "$LOCK" || { say "rule A3 not passed and no Amendment 2 (lead decision) in the lock"; exit 7; }
      say "rule A3 failed; proceeding under the lock's Amendment 2"
    fi
    grep -q '^## Amendment 1' "$LOCK" || { say "lock has no Amendment 1 (prediction) yet"; exit 8; }
  fi
else
  [ -n "${T3_REQUIRE:-}" ] || { say "T3_REQUIRE not set for $EXP epochs=$EPOCHS"; exit 8; }
  IFS='|' read -r -a _req <<< "$T3_REQUIRE"
  for h in "${_req[@]}"; do grep -q "^## $h" "$LOCK" || { say "lock has no '## $h' yet"; exit 8; }; done
fi
STUDENT=$REPO/${T3_STUDENT:-outputs/T3/sft_qwen3_1.7b}
TEACHER=$REPO/$TEACHER_REL
for d in "$REPO/outputs/T3/sft_qwen3_1.7b" "$TEACHER"; do [ -f "$d/DONE" ] || { say "missing $d/DONE"; exit 5; }; done
[ -f "$STUDENT/config.json" ] || { say "missing $STUDENT/config.json"; exit 5; }
S=$(erropd_staged "$STUDENT"); T=$(erropd_staged "$TEACHER")
"$ERROPD_PY_VLLM" -c "import sys; sys.path.insert(0, 'scripts/erropd'); import eval_ner; [eval_ner.rope_guard(p) for p in sys.argv[1:]]" "$S" "$T" \
  || { say "RoPE guard failed"; exit 9; }
DATA=$REPO/data/erropd_t3/parquet_verl
VERL_TREE=$ERROPD_VERL_PROD
say "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} n=$NGPUS lam=$LAM seed=$SEED teacher=$TEACHER_TAG"
say "student=$S teacher=$T verl=$VERL_TREE"
export PYTHONPATH=$VERL_TREE${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled USED_MODEL=no_api HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS PAPER_NOBASE_ADVANTAGE PAPER_FORWARD_KL_ON_ROLLOUTS

"$ERROPD_PY_VERL" - "$S" "$DATA/train.parquet" <<'EOF' 2>&1 | sed 's/^/[prompt] /'
import sys, pyarrow.parquet as pq
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
t = pq.read_table(sys.argv[2]).slice(0, 1).to_pylist()[0]
s = tok.apply_chat_template(t["prompt"], add_generation_prompt=True, tokenize=False)
assert s.endswith("<|im_start|>assistant\n") and "<think>" not in s, repr(s[-80:])
print("rendered tail:", repr(s[-60:]))
EOF

CKPT_ARGS=(); [ "$CKPT_OPTIM" = 0 ] && CKPT_ARGS=('actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]')
say "budget=$TOTAL steps save_freq=$SAVE_FREQ eval_steps=[$EVAL_STEPS] ckpt_optim=$CKPT_OPTIM eval_one_proc=$EVAL_ONE_PROC variant=$VARIANT"

if [ ! -f "$OUT/TRAIN_DONE" ]; then
  rm -rf "$CKPT" "$OUT/rollouts"; mkdir -p "$CKPT"
  say "TRAIN_START"
  ( cd "$VERL_TREE" && "$ERROPD_PY_VERL" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=$SEED \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$S \
    +actor_rollout_ref.model.base_model_path=$S \
    +actor_rollout_ref.ref.model.path=$T \
    +actor_rollout_ref.ref.model.base_model_path=$S \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.lambda_vals=$LAM \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${T3_MICRO:-2} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${T3_LOGP_MICRO:-4} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${T3_ROLLOUT_MEM:-0.6} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${T3_LOGP_MICRO:-4} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$REPO/scripts/opd_reward_ner.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name='erropd-erropd' \
    trainer.experiment_name="T3_$RUN" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.default_local_dir=$CKPT \
    trainer.resume_mode=disable \
    trainer.test_freq=42 \
    trainer.total_epochs=$EPOCHS \
    trainer.rollout_data_dir=$OUT/rollouts \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    $ERROPD_VERL_RAY_ARGS )
  rc=$?
  say "TRAIN_EXIT rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  [ -d "$CKPT/global_step_$TOTAL/actor" ] || { say "missing global_step_$TOTAL after training"; exit 4; }
  touch "$OUT/TRAIN_DONE"
fi

merge_step() {
  local STEP=$1 GS=$CKPT/global_step_$1 HF=$CKPT/global_step_$1/hf
  if [ -d "$HF" ] && { [ ! -f "$HF/tokenizer.json" ] || [ ! -f "$HF/config.json" ]; } && [ -d "$GS/actor" ]; then
    say "incomplete HF dir for step $STEP (no tokenizer.json/config.json); re-merging"; rm -rf "$HF"
  fi
  if [ ! -f "$HF/model.safetensors" ] && ! ls "$HF"/model-*.safetensors >/dev/null 2>&1; then
    [ -d "$GS/actor" ] || { say "no actor dir for step $STEP"; return 5; }
    say "MERGE step $STEP"
    "$ERROPD_PY_VERL" -m verl.model_merger merge --backend fsdp --local_dir "$GS/actor" --target_dir "$HF" 2>&1 | tail -3
    [ -f "$HF/config.json" ] || { say "merge failed step $STEP"; return 6; }
  fi
  cp "$TOKCFG" "$HF/tokenizer_config.json"
  "$ERROPD_PY_VLLM" - "$TOKCFG" "$HF" <<'EOF'
import json, os, sys
p = os.path.join(sys.argv[2], "chat_template.jinja")
if os.path.exists(p):
    open(p, "w").write(json.load(open(sys.argv[1]))["chat_template"])
EOF
}
meta_for() {
  printf '{"id":"%s_%s_step%s","experiment":"%s","task":"conll03_ner_json","student":"qwen3-1.7b","teacher":"%s","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"%s","clip_c":5,"template":"eval","notes":"n_gpus=%s rollout_tp=2 verl=verl_prod_20260930 student_init=%s ckpt_optim=%s eval_one_proc=%s; one prompt rendering (no think block) in SFT/rollout/eval"}' \
    "$EXP" "$RUN" "$1" "$EXP" "$TEACHER_TAG" "$LAM" "$SEED" "$1" "$TOTAL" "$VARIANT" "$NGPUS" "${T3_STUDENT:-outputs/T3/sft_qwen3_1.7b}" "$CKPT_OPTIM" "$EVAL_ONE_PROC"
}
if [ "$EVAL_ONE_PROC" = 1 ]; then
  BATCH=${ERROPD_RUN_TMP:-/tmp}/t3_evalbatch_${EXP}_$RUN.json
  echo "[]" > "$BATCH"
  for STEP in $EVAL_STEPS; do
    EV=$OUT/eval/step$STEP
    [ -f "$EV/t3_metrics.json" ] && [ -f "$EV/run_meta.json" ] && continue
    merge_step "$STEP" || exit $?
    python3 - "$BATCH" "$CKPT/global_step_$STEP/hf" "$EV" "$(meta_for "$STEP")" <<'EOF'
import json, sys
b, m, o, meta = sys.argv[1:5]
items = json.load(open(b)); items.append({"model_path": m, "output_dir": o, "meta": json.loads(meta)})
json.dump(items, open(b, "w"))
EOF
  done
  say "EVAL batch (one process, TP=$EVAL_TP): $EVAL_STEPS"
  "$ERROPD_PY_VLLM" scripts/eval_ner.py --batch-json "$BATCH" --model-path x --output-dir x --tp "$EVAL_TP" 2>&1 \
    | grep -E "eval_ner|n_total" | cut -c1-300 || say "batch eval rc!=0; per-step fallback follows"
fi
for STEP in $EVAL_STEPS; do
  GS=$CKPT/global_step_$STEP; HF=$GS/hf; EV=$OUT/eval/step$STEP
  if [ ! -f "$EV/t3_metrics.json" ] || [ ! -f "$EV/run_meta.json" ]; then
    merge_step "$STEP" || exit $?
    say "EVAL step $STEP (TP=$EVAL_TP)"
    "$ERROPD_PY_VLLM" scripts/eval_ner.py --model-path "$HF" --output-dir "$EV" --tp "$EVAL_TP" --meta "$(meta_for "$STEP")" 2>&1 | tail -5
    [ -f "$EV/run_meta.json" ] || { say "eval failed step $STEP"; exit 7; }
  fi
  rm -rf "$GS/actor" "$GS/data.pt"
  [ "$STEP" != "$TOTAL" ] && rm -rf "$HF"
done
say "RESULT $(for STEP in $EVAL_STEPS; do python3 -c "import json;m=json.load(open('$OUT/eval/step$STEP/t3_metrics.json'));print('step$STEP valid=%.4f f1v=%.4f useful=%.4f early_close=%.4f' % (m['parse_rate'], m['f1_valid'], m['useful_f1'], m['early_close_share_all']))"; done | paste -sd' ' -)"
if [ "$EXP" = T3 ] && [ "$RUN" = "lam1.00_s42" ]; then "$ERROPD_PY_VLLM" scripts/opd_ner_teacher_select.py a3 || true; fi
for G in "$CKPT"/global_step_*; do
  [ -d "$G" ] || continue
  case " $EVAL_STEPS " in *" ${G##*_} "*) ;; *) rm -rf "$G" ;; esac
done
touch "$DONE"
