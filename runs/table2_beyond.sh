#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# SQLF_gate_samp_t2_trail
NR_TASK=t2 NR_SAMPLE_ROOT=outputs/SQLF/samples SAMPLE_DONE=DONE_gate_trail bash scripts/eval_sample_nonranking.sh student=outputs/T2/sft_qwen3_1.7b teacher_clean=outputs/T2/sft_qwen3_4b teacher_trail=outputs/SQLF/sft_t2_4b_trail_f20_x5

# SQLF_gate_samp_t2_trail_b03
NR_TASK=t2 NR_SAMPLE_ROOT=outputs/SQLF/samples SAMPLE_DONE=DONE_gate_trail_b03 bash scripts/eval_sample_nonranking.sh base_b03=outputs/SQLF/sft_t2_1.7b_trail_f03_x5

# SQLF_sft_t2_trail
NR_EXP=SQLF NGPUS=4 bash scripts/sft_nonranking.sh t2 trail_f20_x5

# SQLF_sft_t2_trail_b03
MASTER_PORT=39517 NR_EXP=SQLF NR_SFT_ROLE=student NGPUS=4 bash scripts/sft_nonranking.sh t2 trail_f03_x5

# QSWEEP_sft_t4b_k1_f50
bash scripts/sft_teacher_rate.sh t4b_k1_f50

# QSWEEP_sft_t4b_k1_f80
bash scripts/sft_teacher_rate.sh t4b_k1_f80

# QSWEEP_meas_f3self_t4b_k1_f50
bash scripts/qsweep_b2_run.sh qsweep_f3self_t4b_k1_f50

# QSWEEP_meas_f3self_t4b_k1_f80
bash scripts/qsweep_b2_run.sh qsweep_f3self_t4b_k1_f80

# QSWEEP_probe_teacher_k1_f50
bash scripts/eval_probe.sh item7_clean none teacher_k1_f50 teacher_k1_f50=outputs/QSWEEP/sft_t4b_k1_f50

# QSWEEP_probe_teacher_k1_f80
bash scripts/eval_probe.sh item7_clean none teacher_k1_f80 teacher_k1_f80=outputs/QSWEEP/sft_t4b_k1_f80

# QSWEEP_samp_teacher_k1_f50
SAMPLE_DONE=DONE_qsweep_teacher_f50 bash scripts/eval_sample.sh teacher_k1_f50=outputs/QSWEEP/sft_t4b_k1_f50

# QSWEEP_samp_teacher_k1_f80
SAMPLE_DONE=DONE_qsweep_teacher_f80 bash scripts/eval_sample.sh teacher_k1_f80=outputs/QSWEEP/sft_t4b_k1_f80

# QSWEEP_fkl_f50_s1001
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f50 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f50 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_fkl_f50_s1002
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f50 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f50 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_fkl_f50_s1003
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f50 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f50 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_fkl_f80_s1001
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f80 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f80 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_fkl_f80_s1002
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f80 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f80 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_fkl_f80_s1003
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f80 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f80 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=fkl ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_opd_f50_s1001
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f50 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f50 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_opd_f50_s1002
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f50 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f50 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_opd_f50_s1003
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f50 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f50 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_opd_f80_s1001
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f80 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f80 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1001 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_opd_f80_s1002
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f80 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f80 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1002 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# QSWEEP_opd_f80_s1003
KEEP_INTERMEDIATE_CKPT=1 CKPT_OPTIM=0 EVAL_TP=2 EXP=QSWEEP TASK=esci TEMPLATE=train EVAL_TEMPLATE=train ROLLOUT_DUMP=1 STUDENT=/path/to/outputs/F3/sft_b17_clean BASE=/path/to/outputs/F3/sft_b17_clean TEACHER=/path/to/outputs/QSWEEP/sft_t4b_k1_f80 STUDENT_TAG=f3-b17-clean TEACHER_TAG=qsweep-t4b-k1-f80 EVAL_STEPS='14 28 42' LAM=1.00 SEED=1003 ARM=vanilla ERROPD_SNAP=/path/to/verl_trees/gopd-302dfd90ef bash scripts/opd_arms.sh

# SQLF_t2_trail_b03_opd_lam1.00_s6001
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6001 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_opd_lam1.00_s6001 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_opd_lam1.00_s6002
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6002 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_opd_lam1.00_s6002 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_opd_lam1.00_s6003
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6003 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_opd_lam1.00_s6003 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_opd_lam1.00_s6004
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6004 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_opd_lam1.00_s6004 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_opd_lam1.00_s6005
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6005 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_opd_lam1.00_s6005 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_fkl_lam1.00_s6001
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_fkl_lam1.00_s6001 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_fkl_lam1.00_s6002
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_fkl_lam1.00_s6002 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_fkl_lam1.00_s6003
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_fkl_lam1.00_s6003 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_fkl_lam1.00_s6004
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_fkl_lam1.00_s6004 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_fkl_lam1.00_s6005
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_fkl_lam1.00_s6005 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.00_s6001
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.00_s6001 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.00_s6002
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.00_s6002 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.00_s6003
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.00_s6003 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.00_s6004
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.00_s6004 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.00_s6005
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.00_s6005 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.40_s6001
NR_TASK=t2 ARM=opd LAM=1.40 SEED=6001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.40_s6001 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.40_s6002
NR_TASK=t2 ARM=opd LAM=1.40 SEED=6002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.40_s6002 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.40_s6003
NR_TASK=t2 ARM=opd LAM=1.40 SEED=6003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.40_s6003 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.40_s6004
NR_TASK=t2 ARM=opd LAM=1.40 SEED=6004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.40_s6004 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_opd_lam1.40_s6005
NR_TASK=t2 ARM=opd LAM=1.40 SEED=6005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_opd_lam1.40_s6005 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLLIMX_t2_lim2_fkl_lam1.00_s6001
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_fkl_lam1.00_s6001 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_fkl_lam1.00_s6002
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_fkl_lam1.00_s6002 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_fkl_lam1.00_s6003
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_fkl_lam1.00_s6003 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_fkl_lam1.00_s6004
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_fkl_lam1.00_s6004 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_fkl_lam1.00_s6005
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_fkl_lam1.00_s6005 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_opd_lam1.00_s6001
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_opd_lam1.00_s6001 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_opd_lam1.00_s6002
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_opd_lam1.00_s6002 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_opd_lam1.00_s6003
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_opd_lam1.00_s6003 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_opd_lam1.00_s6004
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_opd_lam1.00_s6004 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLLIMX_t2_lim2_opd_lam1.00_s6005
NR_TASK=t2 ARM=opd LAM=1.00 SEED=6005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLLIMX RUN_ID=SQLLIMX_t2_lim2_opd_lam1.00_s6005 NR_T2_DATA=data/erropd_t2/parquet_lim2x EPOCHS=9 TOTAL_STEPS=42 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_sql_limit_pool.sh

# SQLNAT_t2_lim2_fkl_lam1.00_s7001
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=7001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_fkl_lam1.00_s7001 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_fkl_lam1.00_s7002
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=7002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_fkl_lam1.00_s7002 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_fkl_lam1.00_s7003
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=7003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_fkl_lam1.00_s7003 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_fkl_lam1.00_s7004
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=7004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_fkl_lam1.00_s7004 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_fkl_lam1.00_s7005
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=7005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_fkl_lam1.00_s7005 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_opd_lam1.00_s7001
NR_TASK=t2 ARM=opd LAM=1.00 SEED=7001 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_opd_lam1.00_s7001 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_opd_lam1.00_s7002
NR_TASK=t2 ARM=opd LAM=1.00 SEED=7002 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_opd_lam1.00_s7002 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_opd_lam1.00_s7003
NR_TASK=t2 ARM=opd LAM=1.00 SEED=7003 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_opd_lam1.00_s7003 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_opd_lam1.00_s7004
NR_TASK=t2 ARM=opd LAM=1.00 SEED=7004 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_opd_lam1.00_s7004 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# SQLNAT_t2_lim2_opd_lam1.00_s7005
NR_TASK=t2 ARM=opd LAM=1.00 SEED=7005 STUDENT=outputs/T2/sft_qwen3_1.7b TEACHER=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 TEACHER_TAG=qwen3-4b-t2-lim2-f20 EXP=SQLNAT RUN_ID=SQLNAT_t2_lim2_opd_lam1.00_s7005 EPOCHS=15 TOTAL_STEPS=210 SAVE_FREQ=42 EVAL_STEPS='42 84 126 168 210' NR_SAMPLE=1 NR_KEEP_HF=1 bash scripts/opd_nonranking.sh

# QSWEEP_probe_fkl_f50_s1001
bash scripts/eval_probe.sh item7_clean none QSWEEP_fkl_f50_s1001 QSWEEP_fkl_f50_s1001_step14=outputs/QSWEEP/QSWEEP_fkl_f50_s1001/ckpt/global_step_14/hf QSWEEP_fkl_f50_s1001_step28=outputs/QSWEEP/QSWEEP_fkl_f50_s1001/ckpt/global_step_28/hf QSWEEP_fkl_f50_s1001_step42=outputs/QSWEEP/QSWEEP_fkl_f50_s1001/ckpt/global_step_42/hf

# QSWEEP_probe_fkl_f50_s1002
bash scripts/eval_probe.sh item7_clean none QSWEEP_fkl_f50_s1002 QSWEEP_fkl_f50_s1002_step14=outputs/QSWEEP/QSWEEP_fkl_f50_s1002/ckpt/global_step_14/hf QSWEEP_fkl_f50_s1002_step28=outputs/QSWEEP/QSWEEP_fkl_f50_s1002/ckpt/global_step_28/hf QSWEEP_fkl_f50_s1002_step42=outputs/QSWEEP/QSWEEP_fkl_f50_s1002/ckpt/global_step_42/hf

# QSWEEP_probe_fkl_f50_s1003
bash scripts/eval_probe.sh item7_clean none QSWEEP_fkl_f50_s1003 QSWEEP_fkl_f50_s1003_step14=outputs/QSWEEP/QSWEEP_fkl_f50_s1003/ckpt/global_step_14/hf QSWEEP_fkl_f50_s1003_step28=outputs/QSWEEP/QSWEEP_fkl_f50_s1003/ckpt/global_step_28/hf QSWEEP_fkl_f50_s1003_step42=outputs/QSWEEP/QSWEEP_fkl_f50_s1003/ckpt/global_step_42/hf

# QSWEEP_probe_fkl_f80_s1001
bash scripts/eval_probe.sh item7_clean none QSWEEP_fkl_f80_s1001 QSWEEP_fkl_f80_s1001_step14=outputs/QSWEEP/QSWEEP_fkl_f80_s1001/ckpt/global_step_14/hf QSWEEP_fkl_f80_s1001_step28=outputs/QSWEEP/QSWEEP_fkl_f80_s1001/ckpt/global_step_28/hf QSWEEP_fkl_f80_s1001_step42=outputs/QSWEEP/QSWEEP_fkl_f80_s1001/ckpt/global_step_42/hf

# QSWEEP_probe_fkl_f80_s1002
bash scripts/eval_probe.sh item7_clean none QSWEEP_fkl_f80_s1002 QSWEEP_fkl_f80_s1002_step14=outputs/QSWEEP/QSWEEP_fkl_f80_s1002/ckpt/global_step_14/hf QSWEEP_fkl_f80_s1002_step28=outputs/QSWEEP/QSWEEP_fkl_f80_s1002/ckpt/global_step_28/hf QSWEEP_fkl_f80_s1002_step42=outputs/QSWEEP/QSWEEP_fkl_f80_s1002/ckpt/global_step_42/hf

# QSWEEP_probe_fkl_f80_s1003
bash scripts/eval_probe.sh item7_clean none QSWEEP_fkl_f80_s1003 QSWEEP_fkl_f80_s1003_step14=outputs/QSWEEP/QSWEEP_fkl_f80_s1003/ckpt/global_step_14/hf QSWEEP_fkl_f80_s1003_step28=outputs/QSWEEP/QSWEEP_fkl_f80_s1003/ckpt/global_step_28/hf QSWEEP_fkl_f80_s1003_step42=outputs/QSWEEP/QSWEEP_fkl_f80_s1003/ckpt/global_step_42/hf

# QSWEEP_probe_opd_f50_s1001
bash scripts/eval_probe.sh item7_clean none QSWEEP_opd_f50_s1001 QSWEEP_opd_f50_s1001_step14=outputs/QSWEEP/QSWEEP_opd_f50_s1001/ckpt/global_step_14/hf QSWEEP_opd_f50_s1001_step28=outputs/QSWEEP/QSWEEP_opd_f50_s1001/ckpt/global_step_28/hf QSWEEP_opd_f50_s1001_step42=outputs/QSWEEP/QSWEEP_opd_f50_s1001/ckpt/global_step_42/hf

# QSWEEP_probe_opd_f50_s1002
bash scripts/eval_probe.sh item7_clean none QSWEEP_opd_f50_s1002 QSWEEP_opd_f50_s1002_step14=outputs/QSWEEP/QSWEEP_opd_f50_s1002/ckpt/global_step_14/hf QSWEEP_opd_f50_s1002_step28=outputs/QSWEEP/QSWEEP_opd_f50_s1002/ckpt/global_step_28/hf QSWEEP_opd_f50_s1002_step42=outputs/QSWEEP/QSWEEP_opd_f50_s1002/ckpt/global_step_42/hf

# QSWEEP_probe_opd_f50_s1003
bash scripts/eval_probe.sh item7_clean none QSWEEP_opd_f50_s1003 QSWEEP_opd_f50_s1003_step14=outputs/QSWEEP/QSWEEP_opd_f50_s1003/ckpt/global_step_14/hf QSWEEP_opd_f50_s1003_step28=outputs/QSWEEP/QSWEEP_opd_f50_s1003/ckpt/global_step_28/hf QSWEEP_opd_f50_s1003_step42=outputs/QSWEEP/QSWEEP_opd_f50_s1003/ckpt/global_step_42/hf

# QSWEEP_probe_opd_f80_s1001
bash scripts/eval_probe.sh item7_clean none QSWEEP_opd_f80_s1001 QSWEEP_opd_f80_s1001_step14=outputs/QSWEEP/QSWEEP_opd_f80_s1001/ckpt/global_step_14/hf QSWEEP_opd_f80_s1001_step28=outputs/QSWEEP/QSWEEP_opd_f80_s1001/ckpt/global_step_28/hf QSWEEP_opd_f80_s1001_step42=outputs/QSWEEP/QSWEEP_opd_f80_s1001/ckpt/global_step_42/hf

# QSWEEP_probe_opd_f80_s1002
bash scripts/eval_probe.sh item7_clean none QSWEEP_opd_f80_s1002 QSWEEP_opd_f80_s1002_step14=outputs/QSWEEP/QSWEEP_opd_f80_s1002/ckpt/global_step_14/hf QSWEEP_opd_f80_s1002_step28=outputs/QSWEEP/QSWEEP_opd_f80_s1002/ckpt/global_step_28/hf QSWEEP_opd_f80_s1002_step42=outputs/QSWEEP/QSWEEP_opd_f80_s1002/ckpt/global_step_42/hf

# QSWEEP_probe_opd_f80_s1003
bash scripts/eval_probe.sh item7_clean none QSWEEP_opd_f80_s1003 QSWEEP_opd_f80_s1003_step14=outputs/QSWEEP/QSWEEP_opd_f80_s1003/ckpt/global_step_14/hf QSWEEP_opd_f80_s1003_step28=outputs/QSWEEP/QSWEEP_opd_f80_s1003/ckpt/global_step_28/hf QSWEEP_opd_f80_s1003_step42=outputs/QSWEEP/QSWEEP_opd_f80_s1003/ckpt/global_step_42/hf

# QSWEEP_sample_fkl_f50_s1001
SAMPLE_DONE=DONE_QSWEEP_fkl_f50_s1001 bash scripts/eval_sample.sh QSWEEP_fkl_f50_s1001_step14=outputs/QSWEEP/QSWEEP_fkl_f50_s1001/ckpt/global_step_14/hf QSWEEP_fkl_f50_s1001_step28=outputs/QSWEEP/QSWEEP_fkl_f50_s1001/ckpt/global_step_28/hf QSWEEP_fkl_f50_s1001_step42=outputs/QSWEEP/QSWEEP_fkl_f50_s1001/ckpt/global_step_42/hf

# QSWEEP_sample_fkl_f50_s1002
SAMPLE_DONE=DONE_QSWEEP_fkl_f50_s1002 bash scripts/eval_sample.sh QSWEEP_fkl_f50_s1002_step14=outputs/QSWEEP/QSWEEP_fkl_f50_s1002/ckpt/global_step_14/hf QSWEEP_fkl_f50_s1002_step28=outputs/QSWEEP/QSWEEP_fkl_f50_s1002/ckpt/global_step_28/hf QSWEEP_fkl_f50_s1002_step42=outputs/QSWEEP/QSWEEP_fkl_f50_s1002/ckpt/global_step_42/hf

# QSWEEP_sample_fkl_f50_s1003
SAMPLE_DONE=DONE_QSWEEP_fkl_f50_s1003 bash scripts/eval_sample.sh QSWEEP_fkl_f50_s1003_step14=outputs/QSWEEP/QSWEEP_fkl_f50_s1003/ckpt/global_step_14/hf QSWEEP_fkl_f50_s1003_step28=outputs/QSWEEP/QSWEEP_fkl_f50_s1003/ckpt/global_step_28/hf QSWEEP_fkl_f50_s1003_step42=outputs/QSWEEP/QSWEEP_fkl_f50_s1003/ckpt/global_step_42/hf

# QSWEEP_sample_fkl_f80_s1001
SAMPLE_DONE=DONE_QSWEEP_fkl_f80_s1001 bash scripts/eval_sample.sh QSWEEP_fkl_f80_s1001_step14=outputs/QSWEEP/QSWEEP_fkl_f80_s1001/ckpt/global_step_14/hf QSWEEP_fkl_f80_s1001_step28=outputs/QSWEEP/QSWEEP_fkl_f80_s1001/ckpt/global_step_28/hf QSWEEP_fkl_f80_s1001_step42=outputs/QSWEEP/QSWEEP_fkl_f80_s1001/ckpt/global_step_42/hf

# QSWEEP_sample_fkl_f80_s1002
SAMPLE_DONE=DONE_QSWEEP_fkl_f80_s1002 bash scripts/eval_sample.sh QSWEEP_fkl_f80_s1002_step14=outputs/QSWEEP/QSWEEP_fkl_f80_s1002/ckpt/global_step_14/hf QSWEEP_fkl_f80_s1002_step28=outputs/QSWEEP/QSWEEP_fkl_f80_s1002/ckpt/global_step_28/hf QSWEEP_fkl_f80_s1002_step42=outputs/QSWEEP/QSWEEP_fkl_f80_s1002/ckpt/global_step_42/hf

# QSWEEP_sample_fkl_f80_s1003
SAMPLE_DONE=DONE_QSWEEP_fkl_f80_s1003 bash scripts/eval_sample.sh QSWEEP_fkl_f80_s1003_step14=outputs/QSWEEP/QSWEEP_fkl_f80_s1003/ckpt/global_step_14/hf QSWEEP_fkl_f80_s1003_step28=outputs/QSWEEP/QSWEEP_fkl_f80_s1003/ckpt/global_step_28/hf QSWEEP_fkl_f80_s1003_step42=outputs/QSWEEP/QSWEEP_fkl_f80_s1003/ckpt/global_step_42/hf

# QSWEEP_sample_opd_f50_s1001
SAMPLE_DONE=DONE_QSWEEP_opd_f50_s1001 bash scripts/eval_sample.sh QSWEEP_opd_f50_s1001_step14=outputs/QSWEEP/QSWEEP_opd_f50_s1001/ckpt/global_step_14/hf QSWEEP_opd_f50_s1001_step28=outputs/QSWEEP/QSWEEP_opd_f50_s1001/ckpt/global_step_28/hf QSWEEP_opd_f50_s1001_step42=outputs/QSWEEP/QSWEEP_opd_f50_s1001/ckpt/global_step_42/hf

# QSWEEP_sample_opd_f50_s1002
SAMPLE_DONE=DONE_QSWEEP_opd_f50_s1002 bash scripts/eval_sample.sh QSWEEP_opd_f50_s1002_step14=outputs/QSWEEP/QSWEEP_opd_f50_s1002/ckpt/global_step_14/hf QSWEEP_opd_f50_s1002_step28=outputs/QSWEEP/QSWEEP_opd_f50_s1002/ckpt/global_step_28/hf QSWEEP_opd_f50_s1002_step42=outputs/QSWEEP/QSWEEP_opd_f50_s1002/ckpt/global_step_42/hf

# QSWEEP_sample_opd_f50_s1003
SAMPLE_DONE=DONE_QSWEEP_opd_f50_s1003 bash scripts/eval_sample.sh QSWEEP_opd_f50_s1003_step14=outputs/QSWEEP/QSWEEP_opd_f50_s1003/ckpt/global_step_14/hf QSWEEP_opd_f50_s1003_step28=outputs/QSWEEP/QSWEEP_opd_f50_s1003/ckpt/global_step_28/hf QSWEEP_opd_f50_s1003_step42=outputs/QSWEEP/QSWEEP_opd_f50_s1003/ckpt/global_step_42/hf

# QSWEEP_sample_opd_f80_s1001
SAMPLE_DONE=DONE_QSWEEP_opd_f80_s1001 bash scripts/eval_sample.sh QSWEEP_opd_f80_s1001_step14=outputs/QSWEEP/QSWEEP_opd_f80_s1001/ckpt/global_step_14/hf QSWEEP_opd_f80_s1001_step28=outputs/QSWEEP/QSWEEP_opd_f80_s1001/ckpt/global_step_28/hf QSWEEP_opd_f80_s1001_step42=outputs/QSWEEP/QSWEEP_opd_f80_s1001/ckpt/global_step_42/hf

# QSWEEP_sample_opd_f80_s1002
SAMPLE_DONE=DONE_QSWEEP_opd_f80_s1002 bash scripts/eval_sample.sh QSWEEP_opd_f80_s1002_step14=outputs/QSWEEP/QSWEEP_opd_f80_s1002/ckpt/global_step_14/hf QSWEEP_opd_f80_s1002_step28=outputs/QSWEEP/QSWEEP_opd_f80_s1002/ckpt/global_step_28/hf QSWEEP_opd_f80_s1002_step42=outputs/QSWEEP/QSWEEP_opd_f80_s1002/ckpt/global_step_42/hf

# QSWEEP_sample_opd_f80_s1003
SAMPLE_DONE=DONE_QSWEEP_opd_f80_s1003 bash scripts/eval_sample.sh QSWEEP_opd_f80_s1003_step14=outputs/QSWEEP/QSWEEP_opd_f80_s1003/ckpt/global_step_14/hf QSWEEP_opd_f80_s1003_step28=outputs/QSWEEP/QSWEEP_opd_f80_s1003/ckpt/global_step_28/hf QSWEEP_opd_f80_s1003_step42=outputs/QSWEEP/QSWEEP_opd_f80_s1003/ckpt/global_step_42/hf

# SQLF_probe_t2_step42
source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql.py --grid --out outputs/SQLF/probe

# SQLLIMX_probe_t2_step42
M=(student=outputs/T2/sft_qwen3_1.7b teacher_lim2=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5); for r in SQLLIMX_t2_lim2_opd_lam1.00_s6001 SQLLIMX_t2_lim2_opd_lam1.00_s6002 SQLLIMX_t2_lim2_opd_lam1.00_s6003 SQLLIMX_t2_lim2_opd_lam1.00_s6004 SQLLIMX_t2_lim2_opd_lam1.00_s6005 SQLLIMX_t2_lim2_fkl_lam1.00_s6001 SQLLIMX_t2_lim2_fkl_lam1.00_s6002 SQLLIMX_t2_lim2_fkl_lam1.00_s6003 SQLLIMX_t2_lim2_fkl_lam1.00_s6004 SQLLIMX_t2_lim2_fkl_lam1.00_s6005; do p=outputs/SQLLIMX/$r/ckpt/global_step_42/hf; if [ -f $p/config.json ]; then M+=(${r}_step42=$p); else echo "[sqllimx_probe] missing checkpoint skipped: $r"; fi; done; source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql_limit.py scan --out outputs/SQLLIMX/probe --done DONE_step42 "${M[@]}"

# SQLNAT_probe_t2_step126
source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql_limit.py scan --out outputs/SQLNAT/probe --done DONE_step126 student=outputs/T2/sft_qwen3_1.7b teacher_lim2=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 SQLNAT_t2_lim2_opd_lam1.00_s7001_step126=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7001/ckpt/global_step_126/hf SQLNAT_t2_lim2_opd_lam1.00_s7002_step126=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7002/ckpt/global_step_126/hf SQLNAT_t2_lim2_opd_lam1.00_s7003_step126=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7003/ckpt/global_step_126/hf SQLNAT_t2_lim2_opd_lam1.00_s7004_step126=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7004/ckpt/global_step_126/hf SQLNAT_t2_lim2_opd_lam1.00_s7005_step126=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7005/ckpt/global_step_126/hf SQLNAT_t2_lim2_fkl_lam1.00_s7001_step126=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7001/ckpt/global_step_126/hf SQLNAT_t2_lim2_fkl_lam1.00_s7002_step126=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7002/ckpt/global_step_126/hf SQLNAT_t2_lim2_fkl_lam1.00_s7003_step126=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7003/ckpt/global_step_126/hf SQLNAT_t2_lim2_fkl_lam1.00_s7004_step126=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7004/ckpt/global_step_126/hf SQLNAT_t2_lim2_fkl_lam1.00_s7005_step126=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7005/ckpt/global_step_126/hf

# SQLNAT_probe_t2_step168
source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql_limit.py scan --out outputs/SQLNAT/probe --done DONE_step168 student=outputs/T2/sft_qwen3_1.7b teacher_lim2=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 SQLNAT_t2_lim2_opd_lam1.00_s7001_step168=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7001/ckpt/global_step_168/hf SQLNAT_t2_lim2_opd_lam1.00_s7002_step168=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7002/ckpt/global_step_168/hf SQLNAT_t2_lim2_opd_lam1.00_s7003_step168=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7003/ckpt/global_step_168/hf SQLNAT_t2_lim2_opd_lam1.00_s7004_step168=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7004/ckpt/global_step_168/hf SQLNAT_t2_lim2_opd_lam1.00_s7005_step168=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7005/ckpt/global_step_168/hf SQLNAT_t2_lim2_fkl_lam1.00_s7001_step168=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7001/ckpt/global_step_168/hf SQLNAT_t2_lim2_fkl_lam1.00_s7002_step168=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7002/ckpt/global_step_168/hf SQLNAT_t2_lim2_fkl_lam1.00_s7003_step168=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7003/ckpt/global_step_168/hf SQLNAT_t2_lim2_fkl_lam1.00_s7004_step168=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7004/ckpt/global_step_168/hf SQLNAT_t2_lim2_fkl_lam1.00_s7005_step168=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7005/ckpt/global_step_168/hf

# SQLNAT_probe_t2_step210
source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql_limit.py scan --out outputs/SQLNAT/probe --done DONE_step210 student=outputs/T2/sft_qwen3_1.7b teacher_lim2=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 SQLNAT_t2_lim2_opd_lam1.00_s7001_step210=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7001/ckpt/global_step_210/hf SQLNAT_t2_lim2_opd_lam1.00_s7002_step210=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7002/ckpt/global_step_210/hf SQLNAT_t2_lim2_opd_lam1.00_s7003_step210=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7003/ckpt/global_step_210/hf SQLNAT_t2_lim2_opd_lam1.00_s7004_step210=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7004/ckpt/global_step_210/hf SQLNAT_t2_lim2_opd_lam1.00_s7005_step210=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7005/ckpt/global_step_210/hf SQLNAT_t2_lim2_fkl_lam1.00_s7001_step210=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7001/ckpt/global_step_210/hf SQLNAT_t2_lim2_fkl_lam1.00_s7002_step210=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7002/ckpt/global_step_210/hf SQLNAT_t2_lim2_fkl_lam1.00_s7003_step210=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7003/ckpt/global_step_210/hf SQLNAT_t2_lim2_fkl_lam1.00_s7004_step210=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7004/ckpt/global_step_210/hf SQLNAT_t2_lim2_fkl_lam1.00_s7005_step210=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7005/ckpt/global_step_210/hf

# SQLNAT_probe_t2_step42
source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql_limit.py scan --out outputs/SQLNAT/probe --done DONE_step42 student=outputs/T2/sft_qwen3_1.7b teacher_lim2=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 SQLNAT_t2_lim2_opd_lam1.00_s7001_step42=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7001/ckpt/global_step_42/hf SQLNAT_t2_lim2_opd_lam1.00_s7002_step42=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7002/ckpt/global_step_42/hf SQLNAT_t2_lim2_opd_lam1.00_s7003_step42=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7003/ckpt/global_step_42/hf SQLNAT_t2_lim2_opd_lam1.00_s7004_step42=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7004/ckpt/global_step_42/hf SQLNAT_t2_lim2_opd_lam1.00_s7005_step42=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7005/ckpt/global_step_42/hf SQLNAT_t2_lim2_fkl_lam1.00_s7001_step42=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7001/ckpt/global_step_42/hf SQLNAT_t2_lim2_fkl_lam1.00_s7002_step42=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7002/ckpt/global_step_42/hf SQLNAT_t2_lim2_fkl_lam1.00_s7003_step42=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7003/ckpt/global_step_42/hf SQLNAT_t2_lim2_fkl_lam1.00_s7004_step42=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7004/ckpt/global_step_42/hf SQLNAT_t2_lim2_fkl_lam1.00_s7005_step42=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7005/ckpt/global_step_42/hf

# SQLNAT_probe_t2_step84
source scripts/env.sh && "$ERROPD_PY_VLLM" scripts/eval_probe_sql_limit.py scan --out outputs/SQLNAT/probe --done DONE_step84 student=outputs/T2/sft_qwen3_1.7b teacher_lim2=outputs/SQLLIM/sft_t2_4b_lim2_f20_x5 SQLNAT_t2_lim2_opd_lam1.00_s7001_step84=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7001/ckpt/global_step_84/hf SQLNAT_t2_lim2_opd_lam1.00_s7002_step84=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7002/ckpt/global_step_84/hf SQLNAT_t2_lim2_opd_lam1.00_s7003_step84=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7003/ckpt/global_step_84/hf SQLNAT_t2_lim2_opd_lam1.00_s7004_step84=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7004/ckpt/global_step_84/hf SQLNAT_t2_lim2_opd_lam1.00_s7005_step84=outputs/SQLNAT/SQLNAT_t2_lim2_opd_lam1.00_s7005/ckpt/global_step_84/hf SQLNAT_t2_lim2_fkl_lam1.00_s7001_step84=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7001/ckpt/global_step_84/hf SQLNAT_t2_lim2_fkl_lam1.00_s7002_step84=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7002/ckpt/global_step_84/hf SQLNAT_t2_lim2_fkl_lam1.00_s7003_step84=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7003/ckpt/global_step_84/hf SQLNAT_t2_lim2_fkl_lam1.00_s7004_step84=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7004/ckpt/global_step_84/hf SQLNAT_t2_lim2_fkl_lam1.00_s7005_step84=outputs/SQLNAT/SQLNAT_t2_lim2_fkl_lam1.00_s7005/ckpt/global_step_84/hf

# SQLF_t2_trail_b03_fkl_lam1.00_s6001
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6001 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_fkl_lam1.00_s6001 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_fkl_lam1.00_s6002
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6002 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_fkl_lam1.00_s6002 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_fkl_lam1.00_s6003
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6003 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_fkl_lam1.00_s6003 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_fkl_lam1.00_s6004
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6004 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_fkl_lam1.00_s6004 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh

# SQLF_t2_trail_b03_fkl_lam1.00_s6005
NR_TASK=t2 ARM=fkl LAM=1.00 SEED=6005 STUDENT=outputs/SQLF/sft_t2_1.7b_trail_f03_x5 TEACHER=outputs/SQLF/sft_t2_4b_trail_f20_x5 TEACHER_TAG=qwen3-4b-t2-trail-f20 EXP=SQLF RUN_ID=SQLF_t2_trail_b03_fkl_lam1.00_s6005 NR_SAMPLE=1 NR_KEEP_HF=0 bash scripts/opd_nonranking.sh
