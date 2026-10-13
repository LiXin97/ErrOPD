#!/bin/bash
# Environment for the Qwen3.5 and Olmo 3 runs (verl tree = upstream verl 6eeb5711 + verl_patch/verl_q35.patch).
#   ERROPD_PY_Q35  python of the Qwen3.5/Olmo env (torch 2.10.0, vllm 0.18.0, transformers 5.3.0, deepspeed 0.18.4)
#   Q35_LF_SRC     LlamaFactory source used for Qwen3.5/Olmo SFT (transformers 5.x compatible)
#   Q35_VERL_SNAP  directory that contains the patched verl/ tree
source "${ERROPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/env.sh"
export ERROPD_PY_Q35=${ERROPD_PY_Q35:-/path/to/envs/q35/bin/python}
export Q35_PY=$ERROPD_PY_Q35
export Q35_LF_SRC=${Q35_LF_SRC:-/path/to/LlamaFactory-q35/src}
export Q35_VERL_SNAP=${Q35_VERL_SNAP:-/path/to/verl_trees/verl-q35}
export Q35_VERL=$Q35_VERL_SNAP/verl
export Q35_TEMPLATE_SHA=$(sha256sum "$ERROPD_ROOT/scripts/common_q35_chat_template.jinja" | cut -c1-64)
