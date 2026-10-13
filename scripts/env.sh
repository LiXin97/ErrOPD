#!/bin/bash
# Environment for every launcher. Set the paths below (or export them before sourcing).
#   ERROPD_ROOT     this repository
#   ERROPD_PY_VERL  python of the verl training env   (torch 2.8.0, vllm 0.11.0, transformers 4.57.6, ray 2.55.1, flash-attn 2.8.1)
#   ERROPD_PY_VLLM  python of the sampling/eval env   (torch 2.10.0, vllm 0.18.0, transformers 4.57.6)
#   ERROPD_PY_SFT   python of the LlamaFactory SFT env (torch 2.8.0, transformers 4.57.6, deepspeed 0.17.6, LlamaFactory cc8b576)
#   ERROPD_VERL_PROD verl tree used by the math and cross-task runs (G-OPD with verl_patch/gopd.patch applied)
export ERROPD_ROOT=${ERROPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export ERROPD_PY_VERL=${ERROPD_PY_VERL:-/path/to/envs/verl/bin/python}
export ERROPD_VERL_PY=$ERROPD_PY_VERL
export ERROPD_PY_VLLM=${ERROPD_PY_VLLM:-/path/to/envs/vllm/bin/python}
export ERROPD_VLLM_PY=$ERROPD_PY_VLLM
export ERROPD_PY_SFT=${ERROPD_PY_SFT:-/path/to/envs/sft/bin/python}
export ERROPD_VERL_PROD=${ERROPD_VERL_PROD:-/path/to/verl_trees/gopd-prod/verl}
export ERROPD_STAGE_DIR=${ERROPD_STAGE_DIR:-/tmp/erropd/stage}
export NCCL_DEBUG=WARN
export ERROPD_VERL_RAY_ARGS="+ray_kwargs.ray_init.include_dashboard=False"
if [ -z "${ERROPD_PORT_BASE:-}" ]; then
  _g=${CUDA_VISIBLE_DEVICES:-}; _g=${_g%%,*}
  case "$_g" in [0-9]|[0-9][0-9]) ERROPD_PORT_BASE=$(( 30000 + 1000 * _g )) ;; *) ERROPD_PORT_BASE=38000 ;; esac
  export ERROPD_PORT_BASE; unset _g
fi
export MASTER_PORT=${MASTER_PORT:-$ERROPD_PORT_BASE}
# The original runs copied inputs to node-local disk first; here every path is used in place.
erropd_stage_key() { local p=${1%/}; p=${p#/}; printf '%s' "${p//\//__}"; }
erropd_staged() { printf '%s\n' "${1%/}"; }
erropd_sft_refresh() { return 0; }
true
