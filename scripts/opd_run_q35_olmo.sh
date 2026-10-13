#!/usr/bin/env bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
: "${EXP:?EXP required}" "${LAM:?LAM required}" "${RUN_ID:?RUN_ID required (set by the worker)}"
: "${STUDENT:?STUDENT required}" "${TEACHER:?TEACHER required}"
cd "${ERROPD_ROOT:-/path/to/erropd}"
source scripts/env_q35.sh
PAPER=$ERROPD_ROOT
SEED=${SEED:-42}; EPOCHS=${EPOCHS:-3}; SAVE_FREQ=${SAVE_FREQ:-14}; EVAL_STEPS=${EVAL_STEPS:-"14 28 42"}
BASE=${BASE:-$STUDENT}
STUDENT_TAG=${STUDENT_TAG:-qwen3.5-2b}; TEACHER_TAG=${TEACHER_TAG:-qwen3.5-4b}
ROLLOUT_TP=${ROLLOUT_TP:-2}; MICRO=${MICRO:-2}; LOGP_MICRO=${LOGP_MICRO:-4}; ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM:-0.6}
REF_OFFLOAD=${REF_OFFLOAD:-True}; VARIANT=${VARIANT:-vanilla}; EVAL_TP=${EVAL_TP:-2}
EVAL_PY=${EVAL_PY:-$ERROPD_PY_VLLM}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
NGPUS=${NGPUS:-8}
VAL_JSONL=$PAPER/data/pl_k8/val.jsonl
DATA=$PAPER/data/pl_k8_parquet
OUT=$PAPER/outputs/$EXP/$RUN_ID
CKPT=$OUT/ckpt
BUDGET=${TOTAL_STEPS:-$((EPOCHS * 14))}
DYNAMIC_STOP=${DYNAMIC_STOP:-0}
STEP_ARGS=(); [ -n "${TOTAL_STEPS:-}" ] && STEP_ARGS=(trainer.total_training_steps=$TOTAL_STEPS)
VERL_TAG="q35port:$(basename "$(dirname "$Q35_VERL")")"
mkdir -p "$OUT"
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [q35_opd $RUN_ID] $*"; }
[ -f "$OUT/DONE" ] && { say "DONE exists"; exit 0; }

S_LOCAL=$(erropd_staged "$STUDENT"); T_LOCAL=$(erropd_staged "$TEACHER"); B_LOCAL=$(erropd_staged "$BASE")
TEMPLATE=${TEMPLATE:-d6}
case "$TEMPLATE" in
  d6)
    TPL_ARGS=(); TPL_KW='{}'
    for d in "$S_LOCAL" "$T_LOCAL" "$B_LOCAL"; do
      [ "$(cat "$d/.d6_template_sha256" 2>/dev/null)" = "$Q35_TEMPLATE_SHA" ] || { say "D6 template missing/stale in $d"; exit 2; }
    done ;;
  qwen3train) TPL_ARGS=(+data.apply_chat_template_kwargs.enable_thinking=False); TPL_KW='{"enable_thinking": False}' ;;
  native) TPL_ARGS=(); TPL_KW='{}' ;;
  *) say "bad TEMPLATE=$TEMPLATE"; exit 2 ;;
esac
RMPAD=${RMPAD:-auto}
if [ "$RMPAD" = auto ]; then [ "$TEMPLATE" = qwen3train ] && RMPAD=True || RMPAD=False; fi
say "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} n=$NGPUS lam=$LAM seed=$SEED epochs=$EPOCHS"
say "student=$STUDENT ($S_LOCAL) teacher=$TEACHER ($T_LOCAL) base=$BASE ($B_LOCAL)"
say "verl=$Q35_VERL ($VERL_TAG) py=$Q35_PY eval_py=$EVAL_PY template_sha=${Q35_TEMPLATE_SHA:0:16} budget=$BUDGET rmpad=$RMPAD loss=${PAPER_LOSS_MODE:-unset}"
if grep -rqs PAPER_LOSS_MODE "$Q35_VERL"; then
  case "${PAPER_LOSS_MODE:-}" in opd|fkl_sampled) ;; *) say "this verl tree needs PAPER_LOSS_MODE=opd|fkl_sampled (got '${PAPER_LOSS_MODE:-unset}')"; exit 2 ;; esac
elif [ -n "${PAPER_LOSS_MODE:-}" ] && [ "$PAPER_LOSS_MODE" != opd ]; then
  say "PAPER_LOSS_MODE=$PAPER_LOSS_MODE but $VERL_TAG does not implement it"; exit 2
fi
export PYTHONPATH=$Q35_VERL${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled USED_MODEL=no_api HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS
export VERL_ZMQ_DIR=${VERL_ZMQ_DIR:-${ERROPD_RUN_TMP:-${RAY_TMPDIR:-/tmp}}}
mkdir -p "$VERL_ZMQ_DIR"

"$Q35_PY" - "$S_LOCAL" "$DATA/train.parquet" <<EOF 2>&1 | sed 's/^/[prompt] /'
import sys, pyarrow.parquet as pq
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
t = pq.read_table(sys.argv[2]).slice(0, 1).to_pylist()[0]
s = tok.apply_chat_template(t["prompt"], add_generation_prompt=True, tokenize=False, **$TPL_KW)
print("TEMPLATE=$TEMPLATE rendered tail:", repr(s[-120:]))
EOF

if [ ! -f "$OUT/TRAIN_DONE" ]; then
  rm -rf "$CKPT"; mkdir -p "$CKPT"
  say "TRAIN_START"
  ( cd "$Q35_VERL" && "$Q35_PY" -m verl.trainer.main_ppo \
    trainer.use_legacy_worker_impl=enable \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=${CLIP_C:-5.0} \
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
    ${TPL_ARGS[@]+"${TPL_ARGS[@]}"} \
    actor_rollout_ref.model.path=$S_LOCAL \
    +actor_rollout_ref.model.base_model_path=$B_LOCAL \
    +actor_rollout_ref.ref.model.path=$T_LOCAL \
    +actor_rollout_ref.ref.model.base_model_path=$B_LOCAL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=$RMPAD \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.lambda_vals=$LAM \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGP_MICRO \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGP_MICRO \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_OFFLOAD \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console"]' \
    trainer.log_val_generations=4 \
    trainer.project_name='erropd-erropd' \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    "actor_rollout_ref.actor.checkpoint.save_contents=${CKPT_CONTENTS:-[model,extra]}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${WEIGHT_BUCKET_MB:-4096} \
    trainer.default_local_dir=$CKPT \
    trainer.resume_mode=disable \
    trainer.test_freq=10 \
    trainer.total_epochs=$EPOCHS \
    +erropd.dump_rollouts=true \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} $ERROPD_VERL_RAY_ARGS $EXTRA_OVERRIDES )
  rc=$?
  say "TRAIN_EXIT rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  if [ "$DYNAMIC_STOP" = "1" ]; then
    FIN=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null)
    [ -n "$FIN" ] && [ -d "$CKPT/global_step_$FIN/actor" ] || { say "DYNAMIC_STOP: no final checkpoint"; exit 4; }
  else
    [ -d "$CKPT/global_step_$BUDGET/actor" ] || { say "missing global_step_$BUDGET after training"; exit 4; }
  fi
  touch "$OUT/TRAIN_DONE"
fi
if [ "$DYNAMIC_STOP" = "1" ]; then
  FINAL=$(cat "$CKPT/latest_checkpointed_iteration.txt")
  BUDGET=$FINAL
  EVAL_STEPS=$(for S in $EVAL_STEPS; do [ "$S" = final ] && echo "$FINAL" || echo "$S"; done | awk '!seen[$0]++' | paste -sd' ' -)
fi

EVAL_PORT0=$(( ${ERROPD_PORT_BASE:-30000} + 600 ))
for S in $EVAL_STEPS; do
  GS=$CKPT/global_step_$S; HF=$GS/hf
  META=$(printf '{"id":"%s_step%s","experiment":"%s","task":"fashion_k8","student":"%s","teacher":"%s","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"%s","clip_c":%s,"notes":"n_gpus=%s rollout_tp=%s template=%s:%s rmpad=%s verl=%s loss=%s weight_bucket_mb=%s ckpt_contents=%s run=%s"}' \
     "$RUN_ID" "$S" "$EXP" "$STUDENT_TAG" "$TEACHER_TAG" "$LAM" "$SEED" "$S" "$BUDGET" "$VARIANT" "${CLIP_C:-5}" "$NGPUS" "$ROLLOUT_TP" "$TEMPLATE" "${Q35_TEMPLATE_SHA:0:16}" "$RMPAD" "$VERL_TAG" "${PAPER_LOSS_MODE:-unset}" "${WEIGHT_BUCKET_MB:-4096}" "${CKPT_CONTENTS:-[model,extra]}" "$RUN_ID")
  if [ -f "$OUT/eval/step$S/run_meta.json" ] && [ -f "$OUT/eval/step$S/listwise_metrics.json" ] \
     && { [[ " ${SAMPLE_STEPS:-} " != *" $S "* ]] || [ -f "$OUT/samples/step$S.v2.json" ]; }; then
    say "step $S already evaluated$( [ -f "$HF/WEIGHTS_DELETED.txt" ] && echo ' (weights deleted after evaluation)'); skip"
    [ "${KEEP_SHARDS:-0}" = "1" ] || rm -rf "$GS/actor" "$GS/data.pt"
    continue
  fi
  if [ -d "$GS/actor" ] && [ -d "$HF" ] && [ ! -f "$HF/.merge_complete" ]; then
    say "step $S: hf dir without .merge_complete and shards present; re-merging"; rm -rf "$HF"
  fi
  if ! ls "$HF"/model*.safetensors >/dev/null 2>&1; then
    [ -d "$GS/actor" ] || { say "no actor dir for step $S"; exit 5; }
    say "MERGE step $S"
    ( cd "$Q35_VERL" && "$Q35_PY" -m verl.model_merger merge --backend fsdp --local_dir "$GS/actor" --target_dir "$HF" 2>&1 | tail -3 )
    [ -f "$HF/config.json" ] || { say "merge failed step $S"; exit 6; }
    if [ "$TEMPLATE" = d6 ]; then
      for f in tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json generation_config.json; do
        [ -f "$S_LOCAL/$f" ] && cp "$S_LOCAL/$f" "$HF/$f"
      done
      "$Q35_PY" scripts/common_q35_template.py "$HF"
    elif [ "$TEMPLATE" = native ]; then
      for f in tokenizer.json tokenizer_config.json chat_template.jinja processor_config.json generation_config.json \
               vocab.json merges.txt special_tokens_map.json added_tokens.json tokenizer.model; do
        [ -f "$S_LOCAL/$f" ] && cp "$S_LOCAL/$f" "$HF/$f"
      done
    else
      cp /path/to/models/Qwen3-1.7B/tokenizer_config.json "$HF/tokenizer_config.json"
    fi
    "$Q35_PY" - "$HF/config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
rp = c.get("rope_parameters")
if isinstance(rp, dict) and "rope_theta" in rp and c.get("rope_theta") != rp["rope_theta"]:
    c["rope_theta"] = rp["rope_theta"]
    json.dump(c, open(p, "w"), indent=2)
    print(f"[q35_opd] config.json: top-level rope_theta set to {rp['rope_theta']}")
PYEOF
    touch "$HF/.merge_complete"
  fi
  say "EVAL step $S (TP=$EVAL_TP)"
  for t in 1 2 3; do
    if [ -n "${EVAL_FAMILY:-}" ]; then
      VLLM_PORT=$(( EVAL_PORT0 + 100 * (t - 1) )) "$EVAL_PY" scripts/eval_greedy_xfamily.py --model-path "$HF" \
          --val-jsonl "$VAL_JSONL" --output-dir "$OUT/eval/step$S" --tp "$EVAL_TP" --family "$EVAL_FAMILY" --meta "$META" 2>&1
    else
      VLLM_PORT=$(( EVAL_PORT0 + 100 * (t - 1) )) "$EVAL_PY" scripts/eval_greedy_qwen3.py --model-path "$HF" \
          --val-jsonl "$VAL_JSONL" --output-dir "$OUT/eval/step$S" --tp "$EVAL_TP" --meta "$META" 2>&1
    fi
    [ -f "$OUT/eval/step$S/run_meta.json" ] && break
    say "eval step $S try $t failed (engine start / port race?); retry in 30 s"; sleep 30
  done
  [ -f "$OUT/eval/step$S/run_meta.json" ] || { say "eval failed step $S"; exit 7; }
  if [[ " ${SAMPLE_STEPS:-} " == *" $S "* ]]; then
    SO=$OUT/samples/step$S
    if [ ! -f "$SO.v2.json" ]; then
      mkdir -p "$OUT/samples"
      say "SAMPLE step $S (TP=$EVAL_TP)"
      for t in 1 2 3; do
        VLLM_PORT=$(( EVAL_PORT0 + 50 + 100 * (t - 1) )) "$EVAL_PY" scripts/eval_sample_validity.py \
            --model-path "$HF" --family "${EVAL_FAMILY:-qwen}" --tp "$EVAL_TP" --out "$SO" > "$SO.log" 2>&1
        [ -f "$SO.v2.json" ] && break
        say "sampler step $S try $t failed; retry in 30 s"; sleep 30
      done
    fi
    [ -f "$SO.v2.json" ] || { say "sampler failed step $S"; tail -20 "$SO.log"; exit 8; }
    say "SAMPLED step $S $(python3 -c "import json;r=json.load(open('$SO.v2.json'))['val'];print('val valid=%.4f k1_trunc_realids=%.4f n=%d' % (r['valid'], r['labels'].get('k1_trunc_realids', 0.0), r['n_samples_feasible']))")"
  fi
  [ "${KEEP_SHARDS:-0}" = "1" ] || rm -rf "$GS/actor" "$GS/data.pt"
done
say "RESULT $(for S in $EVAL_STEPS; do python3 -c "import json;m=json.load(open('$OUT/eval/step$S/listwise_metrics.json'));print('step$S parse=%.4f ndcg1=%s useful=%.4f' % (m['parse_rate'], m['ndcg@1_mean'], m['useful']))"; done | paste -sd' ' -)"
touch "$OUT/DONE"
