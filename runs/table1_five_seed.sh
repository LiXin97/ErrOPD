#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# A7E3_A_probe
bash scripts/eval_probe.sh item7_clean none a7e3 A7E3_FKL_A_s1011_step14=outputs/A7E3/A7E3_FKL_A_s1011/ckpt/global_step_14/hf A7E3_FKL_A_s1011_step28=outputs/A7E3/A7E3_FKL_A_s1011/ckpt/global_step_28/hf A7E3_FKL_A_s1011_step42=outputs/A7E3/A7E3_FKL_A_s1011/ckpt/global_step_42/hf A7E3_FKL_A_s1012_step14=outputs/A7E3/A7E3_FKL_A_s1012/ckpt/global_step_14/hf A7E3_FKL_A_s1012_step28=outputs/A7E3/A7E3_FKL_A_s1012/ckpt/global_step_28/hf A7E3_FKL_A_s1012_step42=outputs/A7E3/A7E3_FKL_A_s1012/ckpt/global_step_42/hf A7E3_OPDre_A_s1011_step14=outputs/A7E3/A7E3_OPDre_A_s1011/ckpt/global_step_14/hf A7E3_OPDre_A_s1011_step28=outputs/A7E3/A7E3_OPDre_A_s1011/ckpt/global_step_28/hf A7E3_OPDre_A_s1011_step42=outputs/A7E3/A7E3_OPDre_A_s1011/ckpt/global_step_42/hf A7E3_OPDre_A_s1012_step14=outputs/A7E3/A7E3_OPDre_A_s1012/ckpt/global_step_14/hf A7E3_OPDre_A_s1012_step28=outputs/A7E3/A7E3_OPDre_A_s1012/ckpt/global_step_28/hf A7E3_OPDre_A_s1012_step42=outputs/A7E3/A7E3_OPDre_A_s1012/ckpt/global_step_42/hf

# A7E3_A_sample
SAMPLE_DONE=DONE_a7e3 bash scripts/eval_sample.sh A7E3_FKL_A_s1011_step14=outputs/A7E3/A7E3_FKL_A_s1011/ckpt/global_step_14/hf A7E3_FKL_A_s1011_step28=outputs/A7E3/A7E3_FKL_A_s1011/ckpt/global_step_28/hf A7E3_FKL_A_s1011_step42=outputs/A7E3/A7E3_FKL_A_s1011/ckpt/global_step_42/hf A7E3_FKL_A_s1012_step14=outputs/A7E3/A7E3_FKL_A_s1012/ckpt/global_step_14/hf A7E3_FKL_A_s1012_step28=outputs/A7E3/A7E3_FKL_A_s1012/ckpt/global_step_28/hf A7E3_FKL_A_s1012_step42=outputs/A7E3/A7E3_FKL_A_s1012/ckpt/global_step_42/hf A7E3_OPDre_A_s1011_step14=outputs/A7E3/A7E3_OPDre_A_s1011/ckpt/global_step_14/hf A7E3_OPDre_A_s1011_step28=outputs/A7E3/A7E3_OPDre_A_s1011/ckpt/global_step_28/hf A7E3_OPDre_A_s1011_step42=outputs/A7E3/A7E3_OPDre_A_s1011/ckpt/global_step_42/hf A7E3_OPDre_A_s1012_step14=outputs/A7E3/A7E3_OPDre_A_s1012/ckpt/global_step_14/hf A7E3_OPDre_A_s1012_step28=outputs/A7E3/A7E3_OPDre_A_s1012/ckpt/global_step_28/hf A7E3_OPDre_A_s1012_step42=outputs/A7E3/A7E3_OPDre_A_s1012/ckpt/global_step_42/hf

# A7E3_FKL_A_s1011
EVAL_TP=2 EXP=A7E3 CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1011 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# A7E3_FKL_A_s1012
EVAL_TP=2 EXP=A7E3 CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1012 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# A7E3_FKL_L_s1011
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=fkl XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=FKL XFAMILY_VARIANT=fkl_sampled bash scripts/opd_run_llama.sh 1.00 1011

# A7E3_FKL_L_s1011_samp
DONE_NAME=FKL_L_s1011.DONE bash scripts/eval_sample_llama.sh FKL_L_s1011_step14=outputs/D1inj/opd/A_lam1.00_s1011_fkl/ckpt/global_step_14/hf FKL_L_s1011_step28=outputs/D1inj/opd/A_lam1.00_s1011_fkl/ckpt/global_step_28/hf FKL_L_s1011_step42=outputs/D1inj/opd/A_lam1.00_s1011_fkl/ckpt/global_step_42/hf

# A7E3_FKL_L_s1012
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=fkl XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=FKL XFAMILY_VARIANT=fkl_sampled bash scripts/opd_run_llama.sh 1.00 1012

# A7E3_FKL_L_s1012_samp
DONE_NAME=FKL_L_s1012.DONE bash scripts/eval_sample_llama.sh FKL_L_s1012_step14=outputs/D1inj/opd/A_lam1.00_s1012_fkl/ckpt/global_step_14/hf FKL_L_s1012_step28=outputs/D1inj/opd/A_lam1.00_s1012_fkl/ckpt/global_step_28/hf FKL_L_s1012_step42=outputs/D1inj/opd/A_lam1.00_s1012_fkl/ckpt/global_step_42/hf

# A7E3_OPDre_A_s1011
EVAL_TP=2 EXP=A7E3 CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1011 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# A7E3_OPDre_A_s1012
EVAL_TP=2 EXP=A7E3 CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1012 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# A7E3_OPDre_L_s1011
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=opdre XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=OPDre XFAMILY_VARIANT=vanilla bash scripts/opd_run_llama.sh 1.00 1011

# A7E3_OPDre_L_s1011_samp
DONE_NAME=OPDre_L_s1011.DONE bash scripts/eval_sample_llama.sh OPDre_L_s1011_step14=outputs/D1inj/opd/A_lam1.00_s1011_opdre/ckpt/global_step_14/hf OPDre_L_s1011_step28=outputs/D1inj/opd/A_lam1.00_s1011_opdre/ckpt/global_step_28/hf OPDre_L_s1011_step42=outputs/D1inj/opd/A_lam1.00_s1011_opdre/ckpt/global_step_42/hf

# A7E3_OPDre_L_s1012
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=opdre XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=OPDre XFAMILY_VARIANT=vanilla bash scripts/opd_run_llama.sh 1.00 1012

# A7E3_OPDre_L_s1012_samp
DONE_NAME=OPDre_L_s1012.DONE bash scripts/eval_sample_llama.sh OPDre_L_s1012_step14=outputs/D1inj/opd/A_lam1.00_s1012_opdre/ckpt/global_step_14/hf OPDre_L_s1012_step28=outputs/D1inj/opd/A_lam1.00_s1012_opdre/ckpt/global_step_28/hf OPDre_L_s1012_step42=outputs/D1inj/opd/A_lam1.00_s1012_opdre/ckpt/global_step_42/hf

# A7E3_Q35S_s1011_0p8b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=A7E3 LAM=1.00 SEED=1011 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1011_0p8b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=A7E3 LAM=1.00 SEED=1011 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1011_2b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=A7E3 LAM=1.00 SEED=1011 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1011_2b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=A7E3 LAM=1.00 SEED=1011 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1011_4b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=A7E3 LAM=1.00 SEED=1011 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1011_4b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=A7E3 LAM=1.00 SEED=1011 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1012_0p8b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=A7E3 LAM=1.00 SEED=1012 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1012_0p8b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=A7E3 LAM=1.00 SEED=1012 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1012_2b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=A7E3 LAM=1.00 SEED=1012 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1012_2b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=A7E3 LAM=1.00 SEED=1012 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1012_4b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=A7E3 LAM=1.00 SEED=1012 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# A7E3_Q35S_s1012_4b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=A7E3 LAM=1.00 SEED=1012 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh
