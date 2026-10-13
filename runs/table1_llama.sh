#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# DUPL_ref_samp
R4_SAMPLE_OUT=outputs/DUPL/samples DONE_NAME=ref.DONE bash scripts/eval_sample_llama_dup.sh base=outputs/D1/sft_llama3.2_3b teacher_dup=outputs/DUPL/sft_t8b_dup teacher_clean=outputs/D1inj/sft_t8b_clean

# DUPL_sft_t8b_dup
bash scripts/sft_llama_fashion_dup.sh t8b_dup

# DUPL_fkl_lam1.00_s4001
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=fkl_sampled R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4001

# DUPL_fkl_lam1.00_s4002
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=fkl_sampled R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4002

# DUPL_fkl_lam1.00_s4003
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=fkl_sampled R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4003

# DUPL_fkl_lam1.00_s4004
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=fkl_sampled R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4004

# DUPL_fkl_lam1.00_s4005
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=fkl_sampled R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4005

# DUPL_opd_lam1.00_s4001
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4001

# DUPL_opd_lam1.00_s4002
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4002

# DUPL_opd_lam1.00_s4003
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4003

# DUPL_opd_lam1.00_s4004
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4004

# DUPL_opd_lam1.00_s4005
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.00 4005

# DUPL_opd_lam1.40_s4001
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.40 4001

# DUPL_opd_lam1.40_s4002
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.40 4002

# DUPL_opd_lam1.40_s4003
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.40 4003

# DUPL_opd_lam1.40_s4004
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.40 4004

# DUPL_opd_lam1.40_s4005
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef R4_EXP=DUPL R4_TEACHER_REL=outputs/DUPL/sft_t8b_dup R4_TTAG=llama3.1-8b-dup R4_GATE_FILE=outputs/DUPL/GATES_PASS XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=DUPL XFAMILY_VARIANT=vanilla R4_SAMPLE_STEPS='14 28 42' bash scripts/opd_run_llama_resume.sh 1.40 4005

# FKL_L_ref_samp
DONE_NAME=ref.DONE bash scripts/eval_sample_llama.sh base=outputs/D1/sft_llama3.2_3b teacher=outputs/D1inj/sft_t8b_k1

# FKL_L_s1001
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=fkl XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=FKL XFAMILY_VARIANT=fkl_sampled bash scripts/opd_run_llama.sh 1.00 1001

# FKL_L_s1001_samp
DONE_NAME=FKL_L_s1001.DONE bash scripts/eval_sample_llama.sh FKL_L_s1001_step14=outputs/D1inj/opd/A_lam1.00_s1001_fkl/ckpt/global_step_14/hf FKL_L_s1001_step28=outputs/D1inj/opd/A_lam1.00_s1001_fkl/ckpt/global_step_28/hf FKL_L_s1001_step42=outputs/D1inj/opd/A_lam1.00_s1001_fkl/ckpt/global_step_42/hf

# FKL_L_s1002
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=fkl XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=FKL XFAMILY_VARIANT=fkl_sampled bash scripts/opd_run_llama.sh 1.00 1002

# FKL_L_s1002_samp
DONE_NAME=FKL_L_s1002.DONE bash scripts/eval_sample_llama.sh FKL_L_s1002_step14=outputs/D1inj/opd/A_lam1.00_s1002_fkl/ckpt/global_step_14/hf FKL_L_s1002_step28=outputs/D1inj/opd/A_lam1.00_s1002_fkl/ckpt/global_step_28/hf FKL_L_s1002_step42=outputs/D1inj/opd/A_lam1.00_s1002_fkl/ckpt/global_step_42/hf

# FKL_L_s1003
PAPER_LOSS_MODE=fkl_sampled XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=fkl XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=FKL XFAMILY_VARIANT=fkl_sampled bash scripts/opd_run_llama.sh 1.00 1003

# FKL_L_s1003_samp
DONE_NAME=FKL_L_s1003.DONE bash scripts/eval_sample_llama.sh FKL_L_s1003_step14=outputs/D1inj/opd/A_lam1.00_s1003_fkl/ckpt/global_step_14/hf FKL_L_s1003_step28=outputs/D1inj/opd/A_lam1.00_s1003_fkl/ckpt/global_step_28/hf FKL_L_s1003_step42=outputs/D1inj/opd/A_lam1.00_s1003_fkl/ckpt/global_step_42/hf

# OPDre_L_s1001
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=opdre XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=OPDre XFAMILY_VARIANT=vanilla bash scripts/opd_run_llama.sh 1.00 1001

# OPDre_L_s1001_samp
DONE_NAME=OPDre_L_s1001.DONE bash scripts/eval_sample_llama.sh OPDre_L_s1001_step14=outputs/D1inj/opd/A_lam1.00_s1001_opdre/ckpt/global_step_14/hf OPDre_L_s1001_step28=outputs/D1inj/opd/A_lam1.00_s1001_opdre/ckpt/global_step_28/hf OPDre_L_s1001_step42=outputs/D1inj/opd/A_lam1.00_s1001_opdre/ckpt/global_step_42/hf

# OPDre_L_s1002
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=opdre XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=OPDre XFAMILY_VARIANT=vanilla bash scripts/opd_run_llama.sh 1.00 1002

# OPDre_L_s1002_samp
DONE_NAME=OPDre_L_s1002.DONE bash scripts/eval_sample_llama.sh OPDre_L_s1002_step14=outputs/D1inj/opd/A_lam1.00_s1002_opdre/ckpt/global_step_14/hf OPDre_L_s1002_step28=outputs/D1inj/opd/A_lam1.00_s1002_opdre/ckpt/global_step_28/hf OPDre_L_s1002_step42=outputs/D1inj/opd/A_lam1.00_s1002_opdre/ckpt/global_step_42/hf

# OPDre_L_s1003
PAPER_LOSS_MODE=opd XFAMILY_VERL_SNAP=/path/to/verl_trees/gopd-302dfd90ef XFAMILY_RUN_TAG=opdre XFAMILY_KEEP_CKPT=1 CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 XFAMILY_META_EXP=OPDre XFAMILY_VARIANT=vanilla bash scripts/opd_run_llama.sh 1.00 1003

# OPDre_L_s1003_samp
DONE_NAME=OPDre_L_s1003.DONE bash scripts/eval_sample_llama.sh OPDre_L_s1003_step14=outputs/D1inj/opd/A_lam1.00_s1003_opdre/ckpt/global_step_14/hf OPDre_L_s1003_step28=outputs/D1inj/opd/A_lam1.00_s1003_opdre/ckpt/global_step_28/hf OPDre_L_s1003_step42=outputs/D1inj/opd/A_lam1.00_s1003_opdre/ckpt/global_step_42/hf

# D1inj_0_lam1.00_s1001
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.00 1001

# D1inj_0_lam1.00_s1002
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.00 1002

# D1inj_0_lam1.00_s1003
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.00 1003

# D1inj_0_lam1.25_s1001
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.25 1001

# D1inj_0_lam1.25_s1002
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.25 1002

# D1inj_0_lam1.25_s1003
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.25 1003

# D1inj_0_lam1.40_s1001
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.40 1001

# D1inj_0_lam1.40_s1002
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.40 1002

# D1inj_0_lam1.40_s1003
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=0 XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.40 1003

# D1inj_A_lam1.00_s1001
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.00 1001

# D1inj_A_lam1.00_s1002
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.00 1002

# D1inj_A_lam1.00_s1003
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.00 1003

# D1inj_A_lam1.25_s1001
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.25 1001

# D1inj_A_lam1.25_s1002
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.25 1002

# D1inj_A_lam1.25_s1003
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.25 1003

# D1inj_A_lam1.40_s1001
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.40 1001

# D1inj_A_lam1.40_s1002
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.40 1002

# D1inj_A_lam1.40_s1003
CKPT_OPTIM=0 XFAMILY_EXP=D1inj XFAMILY_PAIR=A XFAMILY_EVAL_TP=2 bash scripts/opd_run_llama.sh 1.40 1003

# D1inj_proxy_pairA
bash scripts/eval_trainshare_llama.sh A

# D1inj_sample
bash scripts/eval_sample_llama_inj.sh base=outputs/D1/sft_llama3.2_3b A_lam1.00_s1001=outputs/D1inj/opd/A_lam1.00_s1001/ckpt/global_step_42/hf A_lam1.00_s1002=outputs/D1inj/opd/A_lam1.00_s1002/ckpt/global_step_42/hf A_lam1.00_s1003=outputs/D1inj/opd/A_lam1.00_s1003/ckpt/global_step_42/hf A_lam1.40_s1001=outputs/D1inj/opd/A_lam1.40_s1001/ckpt/global_step_42/hf A_lam1.40_s1002=outputs/D1inj/opd/A_lam1.40_s1002/ckpt/global_step_42/hf A_lam1.40_s1003=outputs/D1inj/opd/A_lam1.40_s1003/ckpt/global_step_42/hf

# D1inj_asym_0
bash scripts/eval_reference_llama.sh 0

# D1inj_asym_A
bash scripts/eval_reference_llama.sh A

# D1inj_asym_base
bash scripts/eval_reference_llama.sh base

# D1inj_asym_stat
bash scripts/eval_reference_llama.sh stat

# D1inj_sft_t8b_clean
bash scripts/sft_llama_fashion_inj.sh t8b_clean

# D1inj_sft_t8b_k1
bash scripts/sft_llama_fashion_inj.sh t8b_k1

# D1inj_proxy_pair0
bash scripts/eval_trainshare_llama.sh 0
