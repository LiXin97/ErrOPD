#!/usr/bin/env bash
set -uo pipefail
: "${EXP:?EXP required}" "${LAM:?LAM required}" "${ARM:?ARM required}" "${RUN_ID:?RUN_ID required}"
cd "${ERROPD_ROOT:-/path/to/erropd}"
source scripts/env.sh
SNAP=${ERROPD_SNAP:-/path/to/verl_trees/gopd-9c6247e63a}
SNAP_LOCAL=$(erropd_staged "$SNAP")
export VERL_TREE=$SNAP_LOCAL/verl
[ -f "$VERL_TREE/verl/erropd/hooks.py" ] || { echo "[erropd] no ERROPD verl tree at $VERL_TREE"; exit 3; }

X="+erropd.dump_rollouts=true"
export PAPER_LOSS_MODE=opd
case "$ARM" in
  vanilla) ;;
  cls:*)    X="$X +erropd.token_class_lambda=${ARM#cls:}" ;;
  reopd)    X="$X +erropd.reopd=true" ;;
  mask:*)   X="$X +erropd.mask=${ARM#mask:}" ;;
  gate:auto_hi|gate:auto_lo)
            TAU=$(python3 scripts/gate_tau.py "${ARM#gate:auto_}") || { echo "[erropd] tau rule failed"; exit 8; }
            echo "[erropd] C2 tau (${ARM#gate:}) = $TAU  ($(python3 scripts/gate_tau.py show | tr -d '\n' | cut -c1-300))"
            X="$X +erropd.gate_tau=$TAU"; ARM="gate:${ARM#gate:}=$TAU" ;;
  gate:*)   X="$X +erropd.gate_tau=${ARM#gate:}" ;;
  warmup:*) X="$X +erropd.lambda_warmup_steps=${ARM#warmup:} +erropd.lambda_warmup_value=1.0" ;;
  plaus:*)  X="$X +erropd.plaus_alpha=${ARM#plaus:}" ;;
  sg)       X="$X +erropd.sg_gate=true" ;;
  fkl)      export PAPER_LOSS_MODE=fkl_sampled ;;
  rkl)      case "$LAM" in 1|1.0|1.00) ;; *) echo "[erropd] ARM=rkl is defined at LAM=1.00 only (got $LAM)"; exit 2 ;; esac
            export PAPER_LOSS_MODE=rkl_full ;;
  rkla:*)   case "$LAM" in 1|1.0|1.00) ;; *) echo "[erropd] ARM=rkla is defined at LAM=1.00 only (got $LAM)"; exit 2 ;; esac
            _A=${ARM#rkla:}; _RN=1
            case "$_A" in *:norenorm) _A=${_A%:norenorm}; _RN=0 ;; esac
            python3 -c "import sys;a=float(sys.argv[1]);sys.exit(0 if a>=0 else 1)" "$_A" \
              || { echo "[erropd] ARM=rkla:<alpha> needs a finite alpha >= 0 (got '$_A')"; exit 2; }
            export PAPER_LOSS_MODE=rkl_full PAPER_RKL_ALPHA="$_A" PAPER_RKL_RENORM="$_RN"
            export PAPER_ALPHA_ERR_TOKEN="${ALPHA_ERR_TOKEN:-25439}" \
                   PAPER_ALPHA_BOUNDARY_TOKEN="${ALPHA_BOUNDARY_TOKEN:-2137}" \
                   PAPER_ALPHA_ERR_SLOT="${ALPHA_ERR_SLOT:-7}" \
                   PAPER_ALPHA_POSTSTEP="${ALPHA_POSTSTEP:-first}"
            echo "[erropd] alpha gate: alpha=$PAPER_RKL_ALPHA renorm=$PAPER_RKL_RENORM" \
                 "err_token=$PAPER_ALPHA_ERR_TOKEN boundary=$PAPER_ALPHA_BOUNDARY_TOKEN" \
                 "slot=$PAPER_ALPHA_ERR_SLOT poststep=$PAPER_ALPHA_POSTSTEP" ;;
  fmt:auto_tok)
            BETA=$(python3 scripts/format_beta.py beta) || { echo "[erropd] beta rule failed"; exit 8; }
            echo "[erropd] B7 beta (auto_tok) = $BETA  ($(python3 scripts/format_beta.py show | tr -d '\n' | cut -c1-300))"
            X="$X +erropd.format_bonus_beta=$BETA"; ARM="fmt:auto_tok=$BETA" ;;
  fmt:*)    X="$X +erropd.format_bonus_beta=${ARM#fmt:}" ;;
  *) echo "[erropd] bad ARM=$ARM"; exit 2 ;;
esac
[ -n "${LR:-}" ] && X="$X actor_rollout_ref.actor.optim.lr=$LR"
[ -n "${SAVE_STEPS:-}" ] && X="$X +erropd.save_steps=$SAVE_STEPS"
[ -n "${KEPT_BUDGET:-}" ] && X="$X +erropd.stop_after_kept_rollouts=$KEPT_BUDGET"
export EXTRA_OVERRIDES="$X ${EXTRA_OVERRIDES:-}"
export VARIANT="${VARIANT:-td:${ARM}${LR:+:lr=$LR}:tree=erropd-$(basename "$SNAP")}"
export VERL_TAG="${VERL_TAG:-erropd-$(basename "$SNAP")}"
echo "[erropd] RUN_ID=$RUN_ID ARM=$ARM LAM=$LAM SEED=${SEED:-} VERL_TREE=$VERL_TREE"
echo "[erropd] EXTRA_OVERRIDES=$EXTRA_OVERRIDES VARIANT=$VARIANT PAPER_LOSS_MODE=${PAPER_LOSS_MODE:-opd}"
if [ "${TD_DRY_RUN:-0}" = "1" ]; then echo "[erropd] DRY_RUN: not launching"; exit 0; fi
exec bash scripts/opd_run_qwen3.sh
