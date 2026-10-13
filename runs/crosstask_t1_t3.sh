#!/bin/bash
# Commands as launched for this block, one per run, in launch order. Paths are placeholders (README).
set -euo pipefail
cd "$(dirname "$0")/.."

# T1_asym
bash scripts/t1_asym_run.sh

# T1_asymfb
bash scripts/t1_asym_run.sh

# T1_lam1.00_s42
EVAL_TP=2 bash scripts/opd_rerank.sh 1.00 42 gate

# T3_asym_analyze
bash scripts/t3_asym.sh analyze

# T3_asym_base
bash scripts/t3_asym.sh base

# T3_asym_teacher
bash scripts/t3_asym.sh teacher

# T3_opd_lam1.00_s42
bash scripts/opd_ner_grid.sh 1.00 42

# T3_sft_student
bash scripts/sft_ner_clean.sh student

# T3_sft_teacher
bash scripts/sft_ner_clean.sh teacher

# T3c_opd_lam1.00_s2001
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.00 2001

# T3c_opd_lam1.00_s2002
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.00 2002

# T3c_opd_lam1.00_s2003
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.00 2003

# T3c_opd_lam1.00_s2004
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.00 2004

# T3c_opd_lam1.00_s2005
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.00 2005

# T3c_opd_lam1.25_s2001
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.25 2001

# T3c_opd_lam1.25_s2002
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.25 2002

# T3c_opd_lam1.25_s2003
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.25 2003

# T3c_opd_lam1.40_s2001
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.40 2001

# T3c_opd_lam1.40_s2002
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.40 2002

# T3c_opd_lam1.40_s2003
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.40 2003

# T3c_opd_lam1.40_s2004
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.40 2004

# T3c_opd_lam1.40_s2005
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.40 2005

# T3c_opd_lam1.50_s2001
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.50 2001

# T3c_opd_lam1.50_s2002
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.50 2002

# T3c_opd_lam1.50_s2003
T3_EXP=T3c T3_LOCK= T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='T3c lock' bash scripts/opd_ner_grid.sh 1.50 2003

# T1_sft_student
bash scripts/sft_rerank.sh student

# T1_sft_teacher
bash scripts/sft_rerank.sh teacher

# T3_cons_lam1.00
bash scripts/t3_constrained.sh cells "1.00_42 1.00_7 1.00_13"

# T3_cons_lam1.15
bash scripts/t3_constrained.sh cells "1.15_42 1.15_7 1.15_13"

# T3_cons_lam1.25
bash scripts/t3_constrained.sh cells "1.25_42 1.25_7 1.25_13"

# T3_cons_lam1.40
bash scripts/t3_constrained.sh cells "1.40_42 1.40_7 1.40_13"

# T3_cons_lam1.50
bash scripts/t3_constrained.sh cells "1.50_42 1.50_7 1.50_13"

# T3_opd_lam1.00_s13
bash scripts/opd_ner_grid.sh 1.00 13

# T3_opd_lam1.00_s7
bash scripts/opd_ner_grid.sh 1.00 7

# T3_opd_lam1.15_s13
bash scripts/opd_ner_grid.sh 1.15 13

# T3_opd_lam1.15_s42
bash scripts/opd_ner_grid.sh 1.15 42

# T3_opd_lam1.15_s7
bash scripts/opd_ner_grid.sh 1.15 7

# T3_opd_lam1.25_s13
bash scripts/opd_ner_grid.sh 1.25 13

# T3_opd_lam1.25_s42
bash scripts/opd_ner_grid.sh 1.25 42

# T3_opd_lam1.25_s7
bash scripts/opd_ner_grid.sh 1.25 7

# T3_opd_lam1.40_s13
bash scripts/opd_ner_grid.sh 1.40 13

# T3_opd_lam1.40_s42
bash scripts/opd_ner_grid.sh 1.40 42

# T3_opd_lam1.40_s7
bash scripts/opd_ner_grid.sh 1.40 7

# T3_opd_lam1.50_s13
bash scripts/opd_ner_grid.sh 1.50 13

# T3_opd_lam1.50_s42
bash scripts/opd_ner_grid.sh 1.50 42

# T3_opd_lam1.50_s7
bash scripts/opd_ner_grid.sh 1.50 7

# T3b_asym_analyze
T3_BASE=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_ASYM_DIR=outputs/T3b/asym bash scripts/t3_asym.sh analyze

# T3b_asym_base
T3_BASE=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_ASYM_DIR=outputs/T3b/asym bash scripts/t3_asym.sh base

# T3b_asym_teacher
T3_BASE=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_ASYM_DIR=outputs/T3b/asym T3_TEACHER_ROLLOUTS_FROM=outputs/T3/asym/teacher bash scripts/t3_asym.sh teacher

# T3b_opd_lam1.00_s1001
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.00 1001

# T3b_opd_lam1.00_s1002
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.00 1002

# T3b_opd_lam1.00_s1003
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.00 1003

# T3b_opd_lam1.25_s1001
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.25 1001

# T3b_opd_lam1.25_s1002
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.25 1002

# T3b_opd_lam1.25_s1003
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.25 1003

# T3b_opd_lam1.40_s1001
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.40 1001

# T3b_opd_lam1.40_s1002
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.40 1002

# T3b_opd_lam1.40_s1003
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.40 1003

# T3b_opd_lam1.50_s1001
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.50 1001

# T3b_opd_lam1.50_s1002
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.50 1002

# T3b_opd_lam1.50_s1003
T3_EXP=T3b T3_STUDENT=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_VARIANT=init_ep1 T3_SAVE_FREQ=7 T3_EVAL_STEPS='7 14 28 42' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2|Amendment 3' bash scripts/opd_ner_grid.sh 1.50 1003

# T3_product
bash scripts/t3_product.sh

# T3b_product
T3_BASE=outputs/T3/sft_qwen3_1.7b_epochs/ep1 T3_PRODUCT_DIR=outputs/T3b/product bash scripts/t3_product.sh

# T3L_opd_lam1.00_s13
T3_EPOCHS=15 T3_SAVE_FREQ=42 T3_EVAL_STEPS='42 84 126 168 210' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.00 13

# T3L_opd_lam1.00_s42
T3_EPOCHS=15 T3_SAVE_FREQ=42 T3_EVAL_STEPS='42 84 126 168 210' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.00 42

# T3L_opd_lam1.00_s7
T3_EPOCHS=15 T3_SAVE_FREQ=42 T3_EVAL_STEPS='42 84 126 168 210' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.00 7

# T3L_opd_lam1.50_s13
T3_EPOCHS=15 T3_SAVE_FREQ=42 T3_EVAL_STEPS='42 84 126 168 210' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.50 13

# T3L_opd_lam1.50_s42
T3_EPOCHS=15 T3_SAVE_FREQ=42 T3_EVAL_STEPS='42 84 126 168 210' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.50 42

# T3L_opd_lam1.50_s7
T3_EPOCHS=15 T3_SAVE_FREQ=42 T3_EVAL_STEPS='42 84 126 168 210' CKPT_OPTIM=0 T3_REQUIRE='Amendment 2' bash scripts/opd_ner_grid.sh 1.50 7

# T1_lam1.00_s13
EVAL_TP=2 bash scripts/opd_rerank.sh 1.00 13

# T1_lam1.00_s7
EVAL_TP=2 bash scripts/opd_rerank.sh 1.00 7

# T1_lam1.15_s13
EVAL_TP=2 bash scripts/opd_rerank.sh 1.15 13

# T1_lam1.15_s42
EVAL_TP=2 bash scripts/opd_rerank.sh 1.15 42

# T1_lam1.15_s7
EVAL_TP=2 bash scripts/opd_rerank.sh 1.15 7

# T1_lam1.25_s13
EVAL_TP=2 bash scripts/opd_rerank.sh 1.25 13

# T1_lam1.25_s42
EVAL_TP=2 bash scripts/opd_rerank.sh 1.25 42

# T1_lam1.25_s7
EVAL_TP=2 bash scripts/opd_rerank.sh 1.25 7

# T1_lam1.40_s13
EVAL_TP=2 bash scripts/opd_rerank.sh 1.40 13

# T1_lam1.40_s42
EVAL_TP=2 bash scripts/opd_rerank.sh 1.40 42

# T1_lam1.40_s7
EVAL_TP=2 bash scripts/opd_rerank.sh 1.40 7

# T1_lam1.50_s13
EVAL_TP=2 bash scripts/opd_rerank.sh 1.50 13

# T1_lam1.50_s42
EVAL_TP=2 bash scripts/opd_rerank.sh 1.50 42

# T1_lam1.50_s7
EVAL_TP=2 bash scripts/opd_rerank.sh 1.50 7

# T3_cons_sft
bash scripts/t3_constrained.sh sft "student teacher"
