#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# OL3x_s1004_fkl1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=OL3x LAM=1.00 SEED=1004 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# OL3x_s1004_opd1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3x LAM=1.00 SEED=1004 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3x_s1004_opd1.40
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3x LAM=1.40 SEED=1004 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3x_s1005_fkl1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=OL3x LAM=1.00 SEED=1005 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# OL3x_s1005_opd1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3x LAM=1.00 SEED=1005 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3x_s1005_opd1.40
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3x LAM=1.40 SEED=1005 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_s1001_opd1.40
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3 LAM=1.40 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_s1002_fkl1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=OL3 LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# OL3_s1002_opd1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3 LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_s1002_opd1.40
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3 LAM=1.40 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_s1003_fkl1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=OL3 LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# OL3_s1003_opd1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3 LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_s1003_opd1.40
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3 LAM=1.40 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_b2_7b_T32binj
Q35_ENV_NAME=olmoenv_v1 bash scripts/eval_reference_q35.sh ol3_7b_T32binj

# OL3_eval_teacher
bash scripts/sft_olmo_fashion.sh teacher

# OL3_s1001_fkl1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=OL3 LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# OL3_s1001_opd1.00
Q35_ENV_NAME=olmoenv_v1 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=OL3 LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 EVAL_FAMILY=other TEMPLATE=native STUDENT=/path/to/outputs/OL3/sft_olmo3_7b TEACHER=/path/to/outputs/OL3/sft_olmo31_32b_inj20 STUDENT_TAG=olmo3-7b TEACHER_TAG=olmo3.1-32b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# OL3_sft_32b_inj20
bash scripts/sft_olmo_fashion.sh teacher

# OL3_sft_7b
bash scripts/sft_olmo_fashion.sh student

# Q35S_a_sft_9b_inj20
Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXPID=Q35S SIZE=9b ID_SUFFIX=_inj20 TAG=qwen3.5-9b-inj20 CFG=configs/d6/sft_qwen3.5_9b_inj20_pl_k8.yaml DATASET_DIR=data/pl_k8_q35s OUT=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 EVAL_TP=4 SAMPLE_OUT=outputs/Q35S/samples/teacher_9b_inj20 bash scripts/sft_q35_fashion.sh

# Q35S_b_sft_0p8b
Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXPID=Q35S SIZE=0.8b TAG=qwen3.5-0.8b OUT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b EVAL_TP=4 SAMPLE_OUT=outputs/Q35S/samples/base_0p8b bash scripts/sft_q35_fashion.sh

# Q35S_c_samp_d6
DONE_NAME=DONE_d6 bash scripts/eval_sample_q35.sh base_2b=outputs/D6/sft_qwen3.5_2b base_4b=outputs/D6/sft_qwen3.5_4b clean_9b=outputs/D6/sft_qwen3.5_9b

# Q35S_d_b2_0p8b
bash scripts/eval_reference_q35.sh q35s_0p8b_T9binj

# Q35S_d_b2_2b
bash scripts/eval_reference_q35.sh q35s_2b_T9binj

# Q35S_d_b2_4b
bash scripts/eval_reference_q35.sh q35s_4b_T9binj

# Q35S_d_b2_self_9binj
bash scripts/eval_reference_q35.sh q35s_self_9binj

# Q35S_e_sft_4b_inj02
Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXPID=Q35S SIZE=4b ID_SUFFIX=_inj02 TAG=qwen3.5-4b-inj02 CFG=configs/d6/sft_qwen3.5_4b_inj02_pl_k8.yaml DATASET_DIR=data/pl_k8_q35s OUT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 EVAL_TP=4 SAMPLE_OUT=outputs/Q35S/samples/base_4b_inj02 bash scripts/sft_q35_fashion.sh

# Q35S_f_b2_4binj02
bash scripts/eval_reference_q35.sh q35s_4binj02_T9binj

# Q35S_s1001_0p8b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_0p8b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_0p8b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_2b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_2b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_2b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_4b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_4b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1001_4b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_0p8b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_0p8b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_0p8b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_2b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_2b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_2b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_4b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_4b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1002_4b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_0p8b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_0p8b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_0p8b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_0.8b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-0.8b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_2b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_2b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_2b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_2b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-2b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_4b_fkl1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-0fca1075ee EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=fkl_sampled PAPER_LOSS_MODE=fkl_sampled bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_4b_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_s1003_4b_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/D6/sft_qwen3.5_4b TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_xE_s1001_4binj02_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b-inj02 TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_xE_s1001_4binj02_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1001 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b-inj02 TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_xE_s1002_4binj02_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b-inj02 TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_xE_s1002_4binj02_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1002 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b-inj02 TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_xE_s1003_4binj02_opd1.00
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.00 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b-inj02 TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh

# Q35S_xE_s1003_4binj02_opd1.40
Q35_ENV_NAME=q35env_v3 Q35_VERL_SNAP=/path/to/verl_trees/verl-q35-d80a0a3855 EXP=Q35S LAM=1.40 SEED=1003 EPOCHS=3 EVAL_STEPS='14 28 42' SAMPLE_STEPS='14 28 42' EVAL_TP=2 STUDENT=/path/to/outputs/Q35S/sft_qwen3.5_4b_inj02 TEACHER=/path/to/outputs/Q35S/sft_qwen3.5_9b_inj20 STUDENT_TAG=qwen3.5-4b-inj02 TEACHER_TAG=qwen3.5-9b-inj20 VARIANT=vanilla bash scripts/opd_run_q35_olmo.sh
