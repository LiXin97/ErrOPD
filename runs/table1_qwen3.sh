#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# RXC_meas_teacher
bash scripts/eval_reference_samples.sh rxC_self_t4b_dup

# RXC_samp_teacher
SAMPLE_DONE=DONE_rxc_ref bash scripts/eval_sample.sh teacher_dup=outputs/F3/sft_t4b_dup_f20 teacher_clean=outputs/F3/sft_t4b_clean

# RXC_sample
SAMPLE_DONE=DONE_rxc bash scripts/eval_sample.sh RXC_opd_lam1.00_s3001_step42=outputs/RXC/RXC_opd_lam1.00_s3001/ckpt/global_step_42/hf RXC_opd_lam1.00_s3002_step42=outputs/RXC/RXC_opd_lam1.00_s3002/ckpt/global_step_42/hf RXC_opd_lam1.00_s3003_step42=outputs/RXC/RXC_opd_lam1.00_s3003/ckpt/global_step_42/hf RXC_opd_lam1.00_s3004_step42=outputs/RXC/RXC_opd_lam1.00_s3004/ckpt/global_step_42/hf RXC_opd_lam1.00_s3005_step42=outputs/RXC/RXC_opd_lam1.00_s3005/ckpt/global_step_42/hf RXC_opd_lam1.40_s3001_step42=outputs/RXC/RXC_opd_lam1.40_s3001/ckpt/global_step_42/hf RXC_opd_lam1.40_s3002_step42=outputs/RXC/RXC_opd_lam1.40_s3002/ckpt/global_step_42/hf RXC_opd_lam1.40_s3003_step42=outputs/RXC/RXC_opd_lam1.40_s3003/ckpt/global_step_42/hf RXC_opd_lam1.40_s3004_step42=outputs/RXC/RXC_opd_lam1.40_s3004/ckpt/global_step_42/hf RXC_opd_lam1.40_s3005_step42=outputs/RXC/RXC_opd_lam1.40_s3005/ckpt/global_step_42/hf RXC_fkl_lam1.00_s3001_step42=outputs/RXC/RXC_fkl_lam1.00_s3001/ckpt/global_step_42/hf RXC_fkl_lam1.00_s3002_step42=outputs/RXC/RXC_fkl_lam1.00_s3002/ckpt/global_step_42/hf RXC_fkl_lam1.00_s3003_step42=outputs/RXC/RXC_fkl_lam1.00_s3003/ckpt/global_step_42/hf RXC_fkl_lam1.00_s3004_step42=outputs/RXC/RXC_fkl_lam1.00_s3004/ckpt/global_step_42/hf RXC_fkl_lam1.00_s3005_step42=outputs/RXC/RXC_fkl_lam1.00_s3005/ckpt/global_step_42/hf

# RXC_sft_t4b_dup
bash scripts/sft_qwen3_esci.sh t4b_dup_f20

# F3_0_lam1.00_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.00_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.00_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.25_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.25 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.25_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.25 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.25_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.25 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.40_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.40 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.40_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.40 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_0_lam1.40_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_clean STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-clean VARIANT=f3_pair0 EVAL_STEPS='14 28 42' LAM=1.40 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.00_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.00_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.00_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.25_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.25 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.25_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.25 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.25_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.25 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.40_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.40 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.40_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.40 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_A_lam1.40_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairA EVAL_STEPS='14 28 42' LAM=1.40 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.00_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.00_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.00_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.25_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.25 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.25_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.25 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.25_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.25 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.40_s1001
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.40 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.40_s1002
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.40 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# F3_S_lam1.40_s1003
CKPT_OPTIM=0 EVAL_TP=2 EXP=F3 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_k1_f20 BASE=/path/to/outputs/F3/sft_b17_k1_f20 TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-k1f20 TEACHER_TAG=f3-t4b-k1 VARIANT=f3_pairS EVAL_STEPS='14 28 42' LAM=1.40 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-7a040afeee bash scripts/opd_arms.sh

# FKL_A_s1001
EVAL_TP=2 EXP=FKL CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1001 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# FKL_A_s1002
EVAL_TP=2 EXP=FKL CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1002 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# FKL_A_s1003
EVAL_TP=2 EXP=FKL CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1003 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# K1_refs
bash scripts/seqkd_run_qwen3.sh refs

# K1_s1001_eval
bash scripts/seqkd_run_qwen3.sh eval 1001

# K1_s1001_sample
bash scripts/seqkd_run_qwen3.sh sample 1001

# K1_s1001_sft
bash scripts/seqkd_run_qwen3.sh sft 1001

# K1_s1002_eval
bash scripts/seqkd_run_qwen3.sh eval 1002

# K1_s1002_sample
bash scripts/seqkd_run_qwen3.sh sample 1002

# K1_s1002_sft
bash scripts/seqkd_run_qwen3.sh sft 1002

# K1_s1003_eval
bash scripts/seqkd_run_qwen3.sh eval 1003

# K1_s1003_sample
bash scripts/seqkd_run_qwen3.sh sample 1003

# K1_s1003_sft
bash scripts/seqkd_run_qwen3.sh sft 1003

# OPDre_A_s1001
EVAL_TP=2 EXP=OPDre CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# OPDre_A_s1002
EVAL_TP=2 EXP=OPDre CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# OPDre_A_s1003
EVAL_TP=2 EXP=OPDre CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_k1 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-k1 LAM=1.00 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_fkl_lam1.00_s3001
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3001 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_fkl_lam1.00_s3002
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3002 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_fkl_lam1.00_s3003
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3003 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_fkl_lam1.00_s3004
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3004 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_fkl_lam1.00_s3005
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3005 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.00_s3001
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.00_s3002
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.00_s3003
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.00_s3004
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3004 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.00_s3005
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.00 SEED=3005 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.40_s3001
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.40 SEED=3001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.40_s3002
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.40 SEED=3002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.40_s3003
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.40 SEED=3003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.40_s3004
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.40 SEED=3004 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# RXC_opd_lam1.40_s3005
EVAL_TP=2 EXP=RXC CKPT_OPTIM=0 TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 EVAL_STEPS='14 28 42' KEEP_INTERMEDIATE_CKPT=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/F3/sft_t4b_dup_f20 STUDENT_TAG=f3-b17-clean TEACHER_TAG=f3-t4b-dup-f20 LAM=1.40 SEED=3005 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# F3_analyze
bash scripts/f3_analyze.sh f3self_t4b_k1 f3self_t4b_clean f3_pairA f3_pair0 f3_pairS_f20 f3_pairS_f40 f3_pairS_f60 f3fb_pairA f3fb_pair0 f3fb_pairS_f20 f3fb_pairS_f40 f3fb_pairS_f60

# F3_meas_f3_pair0
bash scripts/eval_reference_samples.sh f3_pair0

# F3_meas_f3_pairA
bash scripts/eval_reference_samples.sh f3_pairA

# F3_meas_f3_pairS_f20
bash scripts/eval_reference_samples.sh f3_pairS_f20

# F3_meas_f3_pairS_f40
bash scripts/eval_reference_samples.sh f3_pairS_f40

# F3_meas_f3_pairS_f60
bash scripts/eval_reference_samples.sh f3_pairS_f60

# F3_meas_f3fb_pair0
bash scripts/eval_reference_samples.sh f3fb_pair0

# F3_meas_f3fb_pairA
bash scripts/eval_reference_samples.sh f3fb_pairA

# F3_meas_f3fb_pairS_f20
bash scripts/eval_reference_samples.sh f3fb_pairS_f20

# F3_meas_f3fb_pairS_f40
bash scripts/eval_reference_samples.sh f3fb_pairS_f40

# F3_meas_f3fb_pairS_f60
bash scripts/eval_reference_samples.sh f3fb_pairS_f60

# F3_meas_f3self_t4b_clean
bash scripts/eval_reference_samples.sh f3self_t4b_clean

# F3_meas_f3self_t4b_k1
bash scripts/eval_reference_samples.sh f3self_t4b_k1

# F3_sft_b17_clean
bash scripts/sft_qwen3_esci.sh b17_clean

# F3_sft_b17_k1_f20
bash scripts/sft_qwen3_esci.sh b17_k1_f20

# F3_sft_b17_k1_f40
bash scripts/sft_qwen3_esci.sh b17_k1_f40

# F3_sft_b17_k1_f60
bash scripts/sft_qwen3_esci.sh b17_k1_f60

# F3_sft_t4b_clean
bash scripts/sft_qwen3_esci.sh t4b_clean

# F3_sft_t4b_k1
bash scripts/sft_qwen3_esci.sh t4b_k1
