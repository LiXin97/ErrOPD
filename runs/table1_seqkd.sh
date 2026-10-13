#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# SEQKD7_c1LlamaK1_s1001_eval
bash scripts/seqkd_run_cells.sh eval c1LlamaK1 1001

# SEQKD7_c1LlamaK1_s1001_sample
bash scripts/seqkd_run_cells.sh sample c1LlamaK1 1001

# SEQKD7_c1LlamaK1_s1001_sft
bash scripts/seqkd_run_cells.sh sft c1LlamaK1 1001

# SEQKD7_c1LlamaK1_s1002_eval
bash scripts/seqkd_run_cells.sh eval c1LlamaK1 1002

# SEQKD7_c1LlamaK1_s1002_sample
bash scripts/seqkd_run_cells.sh sample c1LlamaK1 1002

# SEQKD7_c1LlamaK1_s1002_sft
bash scripts/seqkd_run_cells.sh sft c1LlamaK1 1002

# SEQKD7_c1LlamaK1_s1003_eval
bash scripts/seqkd_run_cells.sh eval c1LlamaK1 1003

# SEQKD7_c1LlamaK1_s1003_sample
bash scripts/seqkd_run_cells.sh sample c1LlamaK1 1003

# SEQKD7_c1LlamaK1_s1003_sft
bash scripts/seqkd_run_cells.sh sft c1LlamaK1 1003

# SEQKD7_c2Olmo7B_s1001_eval
bash scripts/seqkd_run_cells.sh eval c2Olmo7B 1001

# SEQKD7_c2Olmo7B_s1001_sample
bash scripts/seqkd_run_cells.sh sample c2Olmo7B 1001

# SEQKD7_c2Olmo7B_s1001_sft
bash scripts/seqkd_run_cells.sh sft c2Olmo7B 1001

# SEQKD7_c2Olmo7B_s1002_eval
bash scripts/seqkd_run_cells.sh eval c2Olmo7B 1002

# SEQKD7_c2Olmo7B_s1002_sample
bash scripts/seqkd_run_cells.sh sample c2Olmo7B 1002

# SEQKD7_c2Olmo7B_s1002_sft
bash scripts/seqkd_run_cells.sh sft c2Olmo7B 1002

# SEQKD7_c2Olmo7B_s1003_eval
bash scripts/seqkd_run_cells.sh eval c2Olmo7B 1003

# SEQKD7_c2Olmo7B_s1003_sample
bash scripts/seqkd_run_cells.sh sample c2Olmo7B 1003

# SEQKD7_c2Olmo7B_s1003_sft
bash scripts/seqkd_run_cells.sh sft c2Olmo7B 1003

# SEQKD7_c3Q35_2B_s1001_eval
bash scripts/seqkd_run_cells.sh eval c3Q35_2B 1001

# SEQKD7_c3Q35_2B_s1001_sample
bash scripts/seqkd_run_cells.sh sample c3Q35_2B 1001

# SEQKD7_c3Q35_2B_s1001_sft
bash scripts/seqkd_run_cells.sh sft c3Q35_2B 1001

# SEQKD7_c3Q35_2B_s1002_eval
bash scripts/seqkd_run_cells.sh eval c3Q35_2B 1002

# SEQKD7_c3Q35_2B_s1002_sample
bash scripts/seqkd_run_cells.sh sample c3Q35_2B 1002

# SEQKD7_c3Q35_2B_s1002_sft
bash scripts/seqkd_run_cells.sh sft c3Q35_2B 1002

# SEQKD7_c3Q35_2B_s1003_eval
bash scripts/seqkd_run_cells.sh eval c3Q35_2B 1003

# SEQKD7_c3Q35_2B_s1003_sample
bash scripts/seqkd_run_cells.sh sample c3Q35_2B 1003

# SEQKD7_c3Q35_2B_s1003_sft
bash scripts/seqkd_run_cells.sh sft c3Q35_2B 1003

# SEQKD7_c4Qwen3Dup_refs
bash scripts/seqkd_run_cells.sh refs c4Qwen3Dup

# SEQKD7_c4Qwen3Dup_s1001_eval
bash scripts/seqkd_run_cells.sh eval c4Qwen3Dup 1001

# SEQKD7_c4Qwen3Dup_s1001_sample
bash scripts/seqkd_run_cells.sh sample c4Qwen3Dup 1001

# SEQKD7_c4Qwen3Dup_s1001_sft
bash scripts/seqkd_run_cells.sh sft c4Qwen3Dup 1001

# SEQKD7_c4Qwen3Dup_s1002_eval
bash scripts/seqkd_run_cells.sh eval c4Qwen3Dup 1002

# SEQKD7_c4Qwen3Dup_s1002_sample
bash scripts/seqkd_run_cells.sh sample c4Qwen3Dup 1002

# SEQKD7_c4Qwen3Dup_s1002_sft
bash scripts/seqkd_run_cells.sh sft c4Qwen3Dup 1002

# SEQKD7_c4Qwen3Dup_s1003_eval
bash scripts/seqkd_run_cells.sh eval c4Qwen3Dup 1003

# SEQKD7_c4Qwen3Dup_s1003_sample
bash scripts/seqkd_run_cells.sh sample c4Qwen3Dup 1003

# SEQKD7_c4Qwen3Dup_s1003_sft
bash scripts/seqkd_run_cells.sh sft c4Qwen3Dup 1003

# SEQKD7_c5LlamaDup_s1001_eval
bash scripts/seqkd_run_cells.sh eval c5LlamaDup 1001

# SEQKD7_c5LlamaDup_s1001_sample
bash scripts/seqkd_run_cells.sh sample c5LlamaDup 1001

# SEQKD7_c5LlamaDup_s1001_sft
bash scripts/seqkd_run_cells.sh sft c5LlamaDup 1001

# SEQKD7_c5LlamaDup_s1002_eval
bash scripts/seqkd_run_cells.sh eval c5LlamaDup 1002

# SEQKD7_c5LlamaDup_s1002_sample
bash scripts/seqkd_run_cells.sh sample c5LlamaDup 1002

# SEQKD7_c5LlamaDup_s1002_sft
bash scripts/seqkd_run_cells.sh sft c5LlamaDup 1002

# SEQKD7_c5LlamaDup_s1003_eval
bash scripts/seqkd_run_cells.sh eval c5LlamaDup 1003

# SEQKD7_c5LlamaDup_s1003_sample
bash scripts/seqkd_run_cells.sh sample c5LlamaDup 1003

# SEQKD7_c5LlamaDup_s1003_sft
bash scripts/seqkd_run_cells.sh sft c5LlamaDup 1003

# SEQKD7_c6Q35_08B_s1001_eval
bash scripts/seqkd_run_cells.sh eval c6Q35_08B 1001

# SEQKD7_c6Q35_08B_s1001_sft
bash scripts/seqkd_run_cells.sh sft c6Q35_08B 1001

# SEQKD7_c6Q35_08B_s1002_eval
bash scripts/seqkd_run_cells.sh eval c6Q35_08B 1002

# SEQKD7_c6Q35_08B_s1002_sft
bash scripts/seqkd_run_cells.sh sft c6Q35_08B 1002

# SEQKD7_c6Q35_08B_s1003_eval
bash scripts/seqkd_run_cells.sh eval c6Q35_08B 1003

# SEQKD7_c6Q35_08B_s1003_sft
bash scripts/seqkd_run_cells.sh sft c6Q35_08B 1003

# SEQKD7_c7Q35_4B_s1001_eval
bash scripts/seqkd_run_cells.sh eval c7Q35_4B 1001

# SEQKD7_c7Q35_4B_s1001_sft
bash scripts/seqkd_run_cells.sh sft c7Q35_4B 1001

# SEQKD7_c7Q35_4B_s1002_eval
bash scripts/seqkd_run_cells.sh eval c7Q35_4B 1002

# SEQKD7_c7Q35_4B_s1002_sft
bash scripts/seqkd_run_cells.sh sft c7Q35_4B 1002

# SEQKD7_c7Q35_4B_s1003_eval
bash scripts/seqkd_run_cells.sh eval c7Q35_4B 1003

# SEQKD7_c7Q35_4B_s1003_sft
bash scripts/seqkd_run_cells.sh sft c7Q35_4B 1003
