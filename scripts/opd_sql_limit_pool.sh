#!/bin/bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
: "${NR_TASK:?t3|t2}" "${ARM:?opd|fkl}" "${LAM:?}" "${SEED:?}" "${STUDENT:?}" "${TEACHER:?}" "${RUN_ID:?}"
REPO=${ERROPD_ROOT:-/path/to/erropd}
cd "$REPO"; REPO=$(pwd)
source scripts/env.sh
EXP=${EXP:-NR}
OUT=$REPO/outputs/$EXP/$RUN_ID
CKPT=$OUT/ckpt
DONE=$OUT/DONE
EPOCHS=${EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-14}
EVAL_STEPS=${EVAL_STEPS:-"14 28 42"}
TOTAL=${TOTAL_STEPS:-$(( EPOCHS * 14 ))}
NGPUS=${NGPUS:-2}
EVAL_TP=2
NR_SAMPLE=${NR_SAMPLE:-1}
NR_KEEP_HF=${NR_KEEP_HF:-0}
TEACHER_TAG=${TEACHER_TAG:-$(basename "$TEACHER")}
TOKCFG=/path/to/models/Qwen3-1.7B/tokenizer_config.json
SNAP=${ERROPD_SNAP:-/path/to/verl_trees/gopd-302dfd90ef}
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [nr_opd $RUN_ID] $*"; }
[ -f "$DONE" ] && { say "DONE exists"; exit 0; }
mkdir -p "$OUT"

case "$ARM" in
  opd) export PAPER_LOSS_MODE=opd ;;
  fkl) export PAPER_LOSS_MODE=fkl_sampled
       case "$LAM" in 1|1.0|1.00) ;; *) say "ARM=fkl is run at LAM=1.00 (got $LAM)"; exit 2 ;; esac ;;
  *) say "bad ARM=$ARM"; exit 2 ;;
esac
case "$STUDENT" in /*) ;; *) STUDENT=$REPO/$STUDENT ;; esac
case "$TEACHER" in /*) ;; *) TEACHER=$REPO/$TEACHER ;; esac
for d in "$STUDENT" "$TEACHER"; do [ -f "$d/config.json" ] || { say "missing $d/config.json"; exit 5; }; done
case "$NR_TASK" in
  t3) DATA=$REPO/data/erropd_t3/parquet_verl; REWARD=$REPO/scripts/opd_reward_ner.py; TEST_FREQ=42
      LEDGER_TASK=conll03_ner_json; METRICS=t3_metrics.json ;;
  t2) POOL=$("$ERROPD_PY_VLLM" scripts/common_sql_rules.py pool)
      case "$POOL" in sftpool|disjoint) ;; *) say "bad pool '$POOL' from common_sql_rules.py"; exit 5 ;; esac
      DATA=$REPO/data/erropd_t2/parquet_$POOL
      if [ -n "${NR_T2_DATA:-}" ]; then
        case "$NR_T2_DATA" in /*) DATA=$NR_T2_DATA ;; *) DATA=$REPO/$NR_T2_DATA ;; esac
        [ -f "$DATA/train.parquet" ] || { say "missing $DATA/train.parquet"; exit 5; }
      fi
      REWARD=$REPO/scripts/opd_reward_sql.py; TEST_FREQ=-1
      LEDGER_TASK=t2_spider; METRICS=sql_metrics.json ;;
  *) say "bad NR_TASK=$NR_TASK"; exit 2 ;;
esac
SNAP_LOCAL=$(erropd_staged "$SNAP")
VERL_TREE=$SNAP_LOCAL/verl
[ -f "$VERL_TREE/verl/erropd/hooks.py" ] || { say "no ERROPD verl tree at $VERL_TREE"; exit 3; }
grep -q "fkl_sampled" "$VERL_TREE/verl/workers/actor/dp_actor.py" || { say "verl tree has no fkl_sampled loss mode"; exit 3; }
S=$(erropd_staged "$STUDENT"); T=$(erropd_staged "$TEACHER")
"$ERROPD_PY_VLLM" -c "import sys; sys.path.insert(0, 'scripts/erropd'); import eval_ner; [eval_ner.rope_guard(p) for p in sys.argv[1:]]" "$S" "$T" \
  || { say "RoPE guard failed"; exit 9; }
say "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} n=$NGPUS task=$NR_TASK arm=$ARM loss=$PAPER_LOSS_MODE lam=$LAM seed=$SEED"
say "student=$S teacher=$T ($TEACHER_TAG) data=$DATA verl=$VERL_TREE budget=$TOTAL save_freq=$SAVE_FREQ eval_steps=[$EVAL_STEPS]"
export PYTHONPATH=$VERL_TREE${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled USED_MODEL=no_api HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS PAPER_NOBASE_ADVANTAGE PAPER_FORWARD_KL_ON_ROLLOUTS PAPER_ADV_MODE PAPER_ADV_BASELINE

"$ERROPD_PY_VERL" - "$S" "$DATA/train.parquet" <<'EOF' 2>&1 | sed 's/^/[prompt] /'
import sys, pyarrow.parquet as pq
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
t = pq.read_table(sys.argv[2]).slice(0, 1).to_pylist()[0]
s = tok.apply_chat_template(t["prompt"], add_generation_prompt=True, tokenize=False)
assert s.endswith("<|im_start|>assistant\n") and "<think>" not in s, repr(s[-80:])
print("rendered tail:", repr(s[-60:]))
EOF
[ "${PIPESTATUS[0]}" = 0 ] || { say "prompt rendering check failed"; exit 5; }

STEP_ARGS=(); [ -n "${TOTAL_STEPS:-}" ] && STEP_ARGS=(trainer.total_training_steps=$TOTAL_STEPS)
if [ "${NR_DRY_RUN:-0}" = 1 ]; then say "DRY_RUN: checks passed, not launching"; exit 0; fi

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${NR_MICRO:-2} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,extra]' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${NR_LOGP_MICRO:-4} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${NR_ROLLOUT_MEM:-0.6} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${NR_LOGP_MICRO:-4} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$REWARD \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name='erropd-erropd' \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.default_local_dir=$CKPT \
    trainer.resume_mode=disable \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$EPOCHS \
    trainer.rollout_data_dir=$OUT/rollouts \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} \
    $ERROPD_VERL_RAY_ARGS )
  rc=$?
  say "TRAIN_EXIT rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  [ -d "$CKPT/global_step_$TOTAL/actor" ] || { say "missing global_step_$TOTAL after training"; exit 4; }
  python3 - "$CKPT/erropd_step_metrics.jsonl" "$ARM" "$TOTAL" <<'EOF' || { say "loss-mode check failed"; exit 6; }
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
arm, total = sys.argv[2], int(sys.argv[3])
assert len(rows) == total and [r["step"] for r in rows] == list(range(1, total + 1)), f"{len(rows)} step records, expected {total}"
has = ["paper/fkl_ce" in r for r in rows]
assert all(has) if arm == "fkl" else not any(has), f"arm {arm}: paper/fkl_ce present in {sum(has)} of {len(rows)} steps"
if arm == "fkl":
    print("[loss] fkl_ce first/last: %.5f / %.5f; teacher sample == student token: %.4f / %.4f" % (
        rows[0]["paper/fkl_ce"], rows[-1]["paper/fkl_ce"],
        rows[0].get("paper/fkl_teacher_eq_student", float("nan")), rows[-1].get("paper/fkl_teacher_eq_student", float("nan"))))
else:
    print("[loss] opd pg_loss first/last: %s / %s" % (rows[0].get("actor/pg_loss"), rows[-1].get("actor/pg_loss")))
EOF
  touch "$OUT/TRAIN_DONE"
fi

merge_step() {
  local STEP=$1 GS=$CKPT/global_step_$1 HF=$CKPT/global_step_$1/hf
  if [ -d "$HF" ] && { [ ! -f "$HF/tokenizer.json" ] || [ ! -f "$HF/config.json" ]; } && [ -d "$GS/actor" ]; then
    say "incomplete HF dir for step $STEP; re-merging"; rm -rf "$HF"
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
  printf '{"id":"%s_step%s","experiment":"%s","task":"%s","student":"qwen3-1.7b","teacher":"%s","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"nr:%s","clip_c":5,"template":"eval","student_init_path":"%s","teacher_path":"%s","notes":"n_gpus=%s rollout_tp=2 verl=erropd-%s loss=%s; one prompt rendering (no think block) in SFT/rollout/eval"}' \
    "$RUN_ID" "$1" "$EXP" "$LEDGER_TASK" "$TEACHER_TAG" "$LAM" "$SEED" "$1" "$TOTAL" "$ARM" "$STUDENT" "$TEACHER" "$NGPUS" "$(basename "$SNAP")" "$PAPER_LOSS_MODE"
}
for STEP in $EVAL_STEPS; do
  GS=$CKPT/global_step_$STEP; HF=$GS/hf; EV=$OUT/eval/step$STEP; SM=$OUT/samples/step$STEP
  if [ ! -f "$EV/$METRICS" ] || [ ! -f "$EV/run_meta.json" ]; then
    merge_step "$STEP" || exit $?
    say "EVAL step $STEP (greedy, TP=$EVAL_TP)"
    if [ "$NR_TASK" = t3 ]; then
      "$ERROPD_PY_VLLM" scripts/eval_ner.py --model-path "$HF" --output-dir "$EV" --tp "$EVAL_TP" --meta "$(meta_for "$STEP")" 2>&1 | tail -5
    else
      "$ERROPD_PY_VLLM" scripts/eval_sql.py --model-path "$HF" --output-dir "$EV" --tp "$EVAL_TP" --meta "$(meta_for "$STEP")" 2>&1 | tail -5
    fi
    [ -f "$EV/run_meta.json" ] || { say "eval failed step $STEP"; exit 7; }
  fi
  if [ "$NR_SAMPLE" = 1 ] && [ ! -f "$SM/DONE" ]; then
    merge_step "$STEP" || exit $?
    say "SAMPLE step $STEP (T=1; ${NR_SAMPLE_ARGS:-default blocks})"
    "$ERROPD_PY_VLLM" scripts/eval_sample_nonranking.py --task "$NR_TASK" --model "$HF" --out "$SM" ${NR_SAMPLE_ARGS:-} 2>&1 | grep "nr_sample" | cut -c1-300
    [ -f "$SM/DONE" ] || { say "sampler failed step $STEP"; exit 8; }
  fi
  rm -rf "$GS/actor" "$GS/data.pt"
  case "$NR_KEEP_HF" in
    1) ;;
    none) rm -rf "$HF" ;;
    *) [ "$STEP" != "$TOTAL" ] && rm -rf "$HF" ;;
  esac
done
say "RESULT $(for STEP in $EVAL_STEPS; do python3 - "$OUT" "$STEP" "$METRICS" <<'EOF'
import json, os, sys
out, step, mf = sys.argv[1:4]
m = json.load(open(f"{out}/eval/step{step}/{mf}"))
s = f"step{step} greedy_valid={m['parse_rate']:.4f}"
p = f"{out}/samples/step{step}/summary.json"
if os.path.exists(p):
    b = json.load(open(p))["blocks"].get("eval", {})
    s += " " + " ".join(f"{k}={v['hits']}/{v['eligible_samples']}" for k, v in b.get("kinds", {}).items())
print(s)
EOF
done | paste -sd' ' -)"
for G in "$CKPT"/global_step_*; do
  [ -d "$G" ] || continue
  case " $EVAL_STEPS " in *" ${G##*_} "*) ;; *) rm -rf "$G" ;; esac
done
touch "$DONE"
