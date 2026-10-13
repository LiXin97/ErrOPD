#!/usr/bin/env bash
set -uo pipefail
if [ -z "${ERROPD_SELF_COPY:-}" ]; then
  _c=$(mktemp "${ERROPD_RUN_TMP:-/tmp}/$(basename "$0").XXXXXX") && cp "$0" "$_c" && ERROPD_SELF_COPY=1 exec bash "$_c" "$@"
fi
: "${EXP:?EXP required}" "${LAM:?LAM required}" "${RUN_ID:?RUN_ID required (set by the worker)}"
cd "${ERROPD_ROOT:-/path/to/erropd}"
source scripts/env.sh
PAPER=$ERROPD_ROOT
LX=/path/to
SEED=${SEED:-42}; EPOCHS=${EPOCHS:-3}; SAVE_FREQ=${SAVE_FREQ:-14}; EVAL_STEPS=${EVAL_STEPS:-"14 28 42"}
STUDENT=${STUDENT:-$PAPER/outputs/sft_qwen3_1.7b_pl_k8}; TEACHER=${TEACHER:-$PAPER/outputs/sft_qwen3_4b_pl_k8}
BASE=${BASE:-$STUDENT}
ROPE_FIX=${ROPE_FIX:-0}
rope_view() { local b; b=$(basename "${1%/}"); if [ -d "$PAPER/outputs/ROPE/${b}_rope1e6" ]; then printf '%s\n' "$PAPER/outputs/ROPE/${b}_rope1e6"; else printf '%s\n' "${1%/}"; fi; }
if [ "$ROPE_FIX" = "1" ]; then STUDENT=$(rope_view "$STUDENT"); TEACHER=$(rope_view "$TEACHER"); BASE=$(rope_view "$BASE"); fi
STUDENT_TAG=${STUDENT_TAG:-qwen3-1.7b}; TEACHER_TAG=${TEACHER_TAG:-qwen3-4b}
ROLLOUT_TP=${ROLLOUT_TP:-2}; MICRO=${MICRO:-2}; LOGP_MICRO=${LOGP_MICRO:-4}; ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM:-0.6}
REF_OFFLOAD=${REF_OFFLOAD:-True}
TEMPLATE=${TEMPLATE:-train}; VARIANT=${VARIANT:-vanilla}; EXTRA_EVAL_TP=${EXTRA_EVAL_TP:-}
EVAL_TP=${EVAL_TP:-4}
CKPT_OPTIM=${CKPT_OPTIM:-1}
EVAL_ONE_PROC=${EVAL_ONE_PROC:-0}
RESUME=${RESUME:-0}
KEEP_INTERMEDIATE_CKPT=${KEEP_INTERMEDIATE_CKPT:-0}
TOKCFG=${TOKCFG:-$LX/models/Qwen3-1.7B/tokenizer_config.json}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
NGPUS=${NGPUS:-8}
TASK=${TASK:-fashion}
EVAL_TEMPLATE=${EVAL_TEMPLATE:-eval}
case "$EVAL_TEMPLATE" in eval|train) ;; *) echo "bad EVAL_TEMPLATE=$EVAL_TEMPLATE"; exit 2 ;; esac
case "$TASK" in
  fashion) VAL_JSONL=$PAPER/data/pl_k8/val.jsonl; DATA=$PAPER/data/pl_k8_parquet; LEDGER_TASK=fashion_k8 ;;
  esci)    VAL_JSONL=$PAPER/data/esci_k8/val.jsonl; DATA=$PAPER/data/esci_k8_parquet_verl; LEDGER_TASK=esci_k8 ;;
  *) echo "bad TASK=$TASK"; exit 2 ;;
esac
[ "$CKPT_OPTIM" = "0" ] && EXTRA_OVERRIDES="$EXTRA_OVERRIDES actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]"
[ "${ROLLOUT_DUMP:-0}" = "1" ] && EXTRA_OVERRIDES="$EXTRA_OVERRIDES trainer.rollout_data_dir=$PAPER/outputs/$EXP/$RUN_ID/rollouts"
OUT=$PAPER/outputs/$EXP/$RUN_ID
CKPT=$OUT/ckpt
BUDGET=${TOTAL_STEPS:-$((EPOCHS * 14))}
VERL_TREE=${VERL_TREE:-$ERROPD_VERL_PROD}
VERL_TAG=${VERL_TAG:-verl_prod_20260930}
DYNAMIC_STOP=${DYNAMIC_STOP:-0}
STEP_ARGS=(); [ -n "${TOTAL_STEPS:-}" ] && STEP_ARGS=(trainer.total_training_steps=$TOTAL_STEPS)
mkdir -p "$OUT"
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] [opd_run $RUN_ID] $*"; }
[ -f "$OUT/DONE" ] && { say "DONE exists"; exit 0; }
[ "${START_DELAY:-0}" -gt 0 ] && { say "START_DELAY ${START_DELAY}s"; sleep "$START_DELAY"; }

S_LOCAL=$(erropd_staged "$STUDENT"); T_LOCAL=$(erropd_staged "$TEACHER"); B_LOCAL=$(erropd_staged "$BASE")
ROPE_BAD=$(python3 - "$S_LOCAL" "$T_LOCAL" "$B_LOCAL" <<'PYEOF'
import json, sys
bad = []
for role, d in zip(["student", "teacher", "base"], sys.argv[1:]):
    c = json.load(open(d + "/config.json"))
    top = c.get("rope_theta")
    rp = c.get("rope_parameters") if isinstance(c.get("rope_parameters"), dict) else {}
    mis = str(c.get("model_type", "")).startswith("qwen3") and top is None and "rope_theta" in rp
    print(f"[rope] {role} {d}: top-level rope_theta={top} rope_parameters.rope_theta={rp.get('rope_theta')} "
          f"-> {'MISREAD (loads 10000.0)' if mis else 'ok'}", file=sys.stderr)
    if mis:
        bad.append(role)
print(",".join(bad))
PYEOF
) || ROPE_BAD="check_failed"
ROPE_MISREAD=false
if [ -n "$ROPE_BAD" ]; then
  if [ "$ROPE_FIX" = "1" ]; then say "ROPE_ABORT: misread-prone config for: $ROPE_BAD"; exit 2; fi
  ROPE_MISREAD=true
  say "!!!!!!!! ROPE_MISREAD: $ROPE_BAD load rope_theta 10000.0 (trained 1e6); every eval row of this run gets rope_misread=true"
fi
say "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES n=$NGPUS lam=$LAM seed=$SEED epochs=$EPOCHS template=$TEMPLATE"
say "student=$STUDENT ($S_LOCAL) teacher=$TEACHER ($T_LOCAL) base=$BASE ($B_LOCAL)"
say "verl=$VERL_TREE py=$ERROPD_PY_VERL ray_args=$ERROPD_VERL_RAY_ARGS budget=$BUDGET"
export PYTHONPATH=$VERL_TREE${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled USED_MODEL=no_api HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RAY_ADDRESS

case "$TEMPLATE" in
  train) TPL_ARGS=(+data.apply_chat_template_kwargs.enable_thinking=False); TPL_KW='{"enable_thinking": False}' ;;
  eval)  TPL_ARGS=(); TPL_KW='{}' ;;
  *) say "bad TEMPLATE=$TEMPLATE"; exit 2 ;;
esac

"$ERROPD_PY_VERL" - "$S_LOCAL" "$DATA/train.parquet" <<EOF 2>&1 | sed 's/^/[prompt] /'
import sys, pyarrow.parquet as pq
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
t = pq.read_table(sys.argv[2]).slice(0, 1).to_pylist()[0]
msgs = t["prompt"]
s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **$TPL_KW)
print("TEMPLATE=$TEMPLATE rendered tail:", repr(s[-160:]))
EOF

if [ ! -f "$OUT/TRAIN_DONE" ]; then
  RESUME_MODE=disable
  K=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || true)
  STATEFUL=$(echo "${EXTRA_OVERRIDES:-}" | grep -oE '\+erropd\.[a-z_]+' | grep -v '^+erropd.dump_rollouts$' | paste -sd' ' -)
  [ "$RESUME" = "1" ] && [ -n "$STATEFUL" ] && say "RESUME=1 ignored: ERROPD hook overrides with state ($STATEFUL)"
  if [ "$RESUME" = "1" ] && [ -z "$STATEFUL" ] && [ "$CKPT_OPTIM" = "1" ] && [ -n "$K" ] && [ -f "$CKPT/global_step_$K/data.pt" ] \
     && ls "$CKPT/global_step_$K/actor"/optim_world_size_*.pt >/dev/null 2>&1 \
     && ls "$CKPT/global_step_$K/actor"/model_world_size_*.pt >/dev/null 2>&1; then
    RESUME_MODE=auto
    python3 - "$CKPT" "$OUT/rollouts" "$K" <<'PYEOF'
import json, os, re, shutil, sys
ckpt, rdir, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
for name in ("erropd_rollouts.jsonl", "erropd_step_metrics.jsonl"):
    p = f"{ckpt}/{name}"
    if os.path.exists(p):
        keep = [l for l in open(p) if l.strip() and json.loads(l).get("step", 0) <= k]
        with open(p + ".tmp", "w") as fh:
            fh.writelines(keep)
        os.replace(p + ".tmp", p)
        print(f"[resume] {name}: kept {len(keep)} rows with step <= {k}")
for d in os.listdir(ckpt):
    m = re.fullmatch(r"global_step_(\d+)", d)
    if m and int(m.group(1)) > k:
        shutil.rmtree(f"{ckpt}/{d}")
        print(f"[resume] removed partial {d}")
if os.path.isdir(rdir):
    for f in os.listdir(rdir):
        m = re.fullmatch(r"(\d+)\.jsonl", f)
        if m and int(m.group(1)) > k:
            os.remove(f"{rdir}/{f}")
PYEOF
    echo "$(date -u +%FT%TZ) resumed from global_step_$K" >> "$OUT/RESUMED"
    say "RESUME from global_step_$K (RESUME=1, CKPT_OPTIM=1)"
  else
    [ "$RESUME" = "1" ] && [ -z "$STATEFUL" ] && say "RESUME=1 but no complete checkpoint with optimizer state (CKPT_OPTIM=$CKPT_OPTIM, latest=${K:-none}): from scratch"
    rm -rf "$CKPT"; mkdir -p "$CKPT"; rm -f "$OUT/RESUMED"
  fi
  say "TRAIN_START"
  TRAIN_LOG=${ERROPD_RUN_TMP:-/tmp}/train_stdout_$RUN_ID.log
  ( cd "$VERL_TREE" && "$ERROPD_PY_VERL" -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.use_remove_padding=True \
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
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console"]' \
    trainer.log_val_generations=4 \
    trainer.project_name='erropd-erropd' \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.default_local_dir=$CKPT \
    trainer.resume_mode=$RESUME_MODE \
    trainer.test_freq=10 \
    trainer.total_epochs=$EPOCHS \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} $ERROPD_VERL_RAY_ARGS $EXTRA_OVERRIDES ) 2>&1 | tee "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
  say "TRAIN_EXIT rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  if [ "$ROPE_FIX" = "1" ]; then
    python3 - "$TRAIN_LOG" <<'PYEOF' || { say "ROPE_ABORT: verl worker config dump does not show rope_theta 1000000.0"; exit 9; }
import re, sys
vals = []
for line in open(sys.argv[1], errors="replace"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = re.search(r"\(WorkerDict pid=\d+\)( +)\"rope_theta\":\s*([0-9.eE+-]+)", line)
    if m and len(m.group(1)) == 3:
        vals.append(float(m.group(2)))
print(f"[rope] verl worker config dumps: top-level rope_theta values {sorted(set(vals))} ({len(vals)} lines)")
sys.exit(0 if vals and all(v == 1e6 for v in vals) else 1)
PYEOF
    say "ROPE_OK verl worker config dumps show rope_theta 1000000.0"
  fi
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
  say "DYNAMIC_STOP final step $FINAL; eval steps: $EVAL_STEPS"
fi

meta_for() {
  local S=$1 RESUMED_KV=""
  [ -s "$OUT/RESUMED" ] && RESUMED_KV="\"resumed_from\":\"$(grep -o 'global_step_[0-9]*' "$OUT/RESUMED" | sed 's/global_step_//' | paste -sd, -)\","
  printf '{"id":"%s_step%s","experiment":"%s","task":"%s","student":"%s","teacher":"%s","lambda":%s,"seed":%s,"step":%s,"budget_steps":%s,"variant":"%s","clip_c":%s,"template":"%s","rope_fix":%s,"rope_misread":%s,"student_init_path":"%s","teacher_path":"%s","base_path":"%s",%s"notes":"n_gpus=%s rollout_tp=%s template=%s verl=%s run=%s"}' \
     "$RUN_ID" "$S" "$EXP" "$LEDGER_TASK" "$STUDENT_TAG" "$TEACHER_TAG" "$LAM" "$SEED" "$S" "$BUDGET" "$VARIANT" "${CLIP_C:-5}" "$TEMPLATE" "$( [ "$ROPE_FIX" = 1 ] && echo true || echo false )" "$ROPE_MISREAD" "$STUDENT" "$TEACHER" "$BASE" "$RESUMED_KV" "$NGPUS" "$ROLLOUT_TP" "$TEMPLATE" "$VERL_TAG" "$RUN_ID"
}
merge_step() {
  local S=$1 GS=$CKPT/global_step_$1 HF=$CKPT/global_step_$1/hf
  if ls "$GS"/WEIGHTS_DELETED*.txt >/dev/null 2>&1 && [ -s "$OUT/eval/step$S/run_meta.json" ]; then
    return 0
  fi
  if [ ! -f "$HF/model.safetensors" ] && ! ls "$HF"/model-*.safetensors >/dev/null 2>&1; then
    [ -d "$GS/actor" ] || { say "no actor dir for step $S"; exit 5; }
    say "MERGE step $S"
    "$ERROPD_PY_VERL" -m verl.model_merger merge --backend fsdp --local_dir "$GS/actor" --target_dir "$HF" 2>&1 | tail -3
    [ -f "$HF/config.json" ] || { say "merge failed step $S"; exit 6; }
    cp "$TOKCFG" "$HF/tokenizer_config.json"
  fi
  if [ "$ROPE_FIX" = "1" ]; then
    python3 -c "import json,sys;c=json.load(open(sys.argv[1]));sys.exit(0 if float(c.get('rope_theta',0))==1e6 else 1)" "$HF/config.json" \
      || { say "ROPE_ABORT: merged checkpoint $HF has rope_theta $(python3 -c "import json;print(json.load(open('$HF/config.json')).get('rope_theta'))")"; exit 9; }
  fi
}
if [ "$EVAL_ONE_PROC" = "1" ]; then
  for S in $EVAL_STEPS; do merge_step "$S"; done
  BATCH=${ERROPD_RUN_TMP:-/tmp}/evalbatch_$RUN_ID.json
  python3 - "$BATCH" "$OUT" "$CKPT" <<PYEOF
import json, os, sys
batch, out, ckpt = sys.argv[1:4]
items = []
for s in "$EVAL_STEPS".split():
    if os.path.exists(f"{out}/eval/step{s}/run_meta.json"):
        continue
    items.append({"model_path": f"{ckpt}/global_step_{s}/hf", "output_dir": f"{out}/eval/step{s}", "step": s})
json.dump(items, open(batch, "w"))
print(f"[opd_run] batch eval of {len(items)} step(s)")
PYEOF
  for S in $EVAL_STEPS; do
    python3 - "$BATCH" "$S" "$(meta_for "$S")" <<'PYEOF3'
import json, sys
b, s, meta = sys.argv[1], sys.argv[2], sys.argv[3]
items = json.load(open(b))
for it in items:
    if it["step"] == s:
        it["meta"] = json.loads(meta)
json.dump(items, open(b, "w"))
PYEOF3
  done
  say "EVAL batch (one process, TP=$EVAL_TP): $EVAL_STEPS"
  ERROPD_ALLOW_ROPE_MISREAD=$( [ "$ROPE_FIX" = 1 ] && echo 0 || echo 1 ) "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py --batch-json "$BATCH" \
      --val-jsonl "$VAL_JSONL" --tp "$EVAL_TP" --task "$TASK" --template "$EVAL_TEMPLATE" 2>&1 || say "batch eval rc!=0; per-step fallback follows"
fi
for S in $EVAL_STEPS; do
  GS=$CKPT/global_step_$S; HF=$GS/hf
  META=$(meta_for "$S")
  merge_step "$S"
  say "EVAL step $S (TP=$EVAL_TP)"
  ERROPD_ALLOW_ROPE_MISREAD=$( [ "$ROPE_FIX" = 1 ] && echo 0 || echo 1 ) "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py --model-path "$HF" --val-jsonl "$VAL_JSONL" \
      --output-dir "$OUT/eval/step$S" --tp "$EVAL_TP" --task "$TASK" --template "$EVAL_TEMPLATE" --meta "$META" 2>&1
  [ -f "$OUT/eval/step$S/run_meta.json" ] || { say "eval failed step $S"; exit 7; }
  for T in $EXTRA_EVAL_TP; do
    say "EVAL step $S (TP=$T)"
    M2=$(echo "$META" | sed "s/_step${S}\"/_step${S}_tp${T}\"/")
    ERROPD_ALLOW_ROPE_MISREAD=$( [ "$ROPE_FIX" = 1 ] && echo 0 || echo 1 ) "$ERROPD_PY_VLLM" scripts/eval_greedy_qwen3.py --model-path "$HF" --val-jsonl "$VAL_JSONL" \
        --output-dir "$OUT/eval_tp$T/step$S" --tp "$T" --task "$TASK" --template "$EVAL_TEMPLATE" --meta "$M2" 2>&1
    [ -f "$OUT/eval_tp$T/step$S/run_meta.json" ] || { say "eval TP=$T failed step $S"; exit 7; }
  done
  rm -rf "$GS/actor" "$GS/data.pt"
  LAST=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo "$BUDGET")
  if [ "$KEEP_INTERMEDIATE_CKPT" != "1" ] && [ "$S" != "$LAST" ] && [ "$S" != "$BUDGET" ] \
     && [ -s "$OUT/eval/step$S/listwise_metrics.json" ] && [ -s "$OUT/eval/step$S/run_meta.json" ]; then
    rm -f "$HF"/*.safetensors "$HF"/model.safetensors.index.json "$HF"/*.bin
    echo "$(date -u +%FT%TZ) weights of intermediate step $S deleted after eval (metrics: $OUT/eval/step$S); config/tokenizer kept; KEEP_INTERMEDIATE_CKPT=1 keeps them" > "$GS/WEIGHTS_DELETED.txt"
    say "step $S: intermediate weights deleted after eval (final step $LAST kept)"
  fi
done
for GS in "$CKPT"/global_step_*; do
  [ -d "$GS/actor" ] || [ -f "$GS/data.pt" ] || continue
  if ls "$GS"/hf/*.safetensors >/dev/null 2>&1 || ls "$GS"/WEIGHTS_DELETED*.txt >/dev/null 2>&1; then
    rm -rf "$GS/actor" "$GS/data.pt"
  else
    rm -f "$GS"/actor/optim_world_size_*.pt "$GS"/actor/extra_state_world_size_*.pt "$GS/data.pt"
  fi
done
say "RESULT $(for S in $EVAL_STEPS; do python3 -c "import json;m=json.load(open('$OUT/eval/step$S/listwise_metrics.json'));print('step$S parse=%.4f ndcg1=%s useful=%.4f' % (m['parse_rate'], m['ndcg@1_mean'], m['useful']))"; done | paste -sd' ' -)"
touch "$OUT/DONE"
