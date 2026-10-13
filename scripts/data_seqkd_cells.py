#!/usr/bin/env python3
"""Sequence-level KD cell table (teacher, student, data)."""
import shlex
import sys

LX = "/path/to"
REPO = f"{LX}/erropd"
SEEDS = (1001, 1002, 1003)
ESCI_SNAP = f"{LX}/OPD-repos/G-OPD-erropd-snap/febfe8b05e"
FASHION_SRC = "data/pl_k8_d1inj/k1_f00_x5.jsonl"
FASHION_VAL = "data/pl_k8/val.jsonl"
ESCI_SRC = "data/esci_k8_f3/k1_f00_x5.jsonl"
ESCI_VAL = "data/esci_k8_f3/val.jsonl"

TEACHERS = {
    "d1inj_t8b_k1": dict(path="outputs/D1inj/sft_t8b_k1", family="llama", task="fashion", sample_gpus=2,
                         sample_est=15, sample_minutes_ref="FKL_L_ref_samp 5.0 min for base 3B + teacher 8B (train+val, 2 GPUs)"),
    "ol3_32b_inj20": dict(path="outputs/OL3/sft_olmo31_32b_inj20", family="other", task="fashion", sample_gpus=4,
                          sample_est=25, sample_minutes_ref="OL3_eval_teacher 5.0 min greedy + 8,028 samples (4 GPUs)"),
    "q35s_9b_inj20": dict(path="outputs/Q35S/sft_qwen3.5_9b_inj20", family="qwen", task="fashion", sample_gpus=4,
                          sample_est=30, sample_minutes_ref="Q35S_c_samp_d6 37.0 min for 3 Qwen3.5 models x 8,028 samples (4 GPUs)"),
    "f3_t4b_dup_f20": dict(path="outputs/F3/sft_t4b_dup_f20", family="qwen3", task="esci", sample_gpus=2,
                           sample_est=20, sample_minutes_ref="K1_s100x_sample 3.0-3.5 min (2 GPUs)"),
    "dupl_t8b_dup": dict(path="outputs/DUPL/sft_t8b_dup", family="llama", task="fashion", sample_gpus=2,
                         sample_est=15, sample_minutes_ref="DUPL_ref_samp 5.5 min for 3 models (2 GPUs)"),
}

M = f"{LX}/models"
CELLS = {
    "c1LlamaK1": dict(prio=1, teacher="d1inj_t8b_k1", family="llama", task="fashion", base=f"{M}/Llama-3.2-3B-Instruct",
                      cfg="configs/sft_llama3.2_3b_pl_k8_erropd.yaml", pdb=4, sft_gpus=8, eval_gpus=2,
                      sft_est=25, eval_est=15, peak_sft=7, err="k1_trunc_realids",
                      teacher_rec="outputs/D1inj/fkl_samples/teacher.v2.records.jsonl",
                      start_rec="outputs/D1inj/fkl_samples/base.v2.records.jsonl",
                      student_sft="outputs/D1/sft_llama3.2_3b", student_sft_spec="19-D1_sft_student (8 GPUs, 15.3 min incl. 5 epoch evals)",
                      table_row="Llama-3.2-3B <- 3.1-8B, Fashion (early close)", fkl_scored="results/FKL_L_scored.json"),
    "c2Olmo7B": dict(prio=2, teacher="ol3_32b_inj20", family="other", task="fashion", base=f"{M}/OLMo-3-7B-Instruct",
                     cfg="configs/olmo3/sft_olmo3_7b_pl_k8.yaml", pdb=2, sft_gpus=8, eval_gpus=2,
                     sft_est=30, eval_est=20, peak_sft=15, err="k1_trunc_realids",
                     teacher_rec="outputs/OL3/samples/teacher_olmo31_32b_inj20.v2.records.jsonl",
                     start_rec="outputs/OL3/samples/base_olmo3_7b.v2.records.jsonl",
                     student_sft="outputs/OL3/sft_olmo3_7b", student_sft_spec="09-OL3_sft_7b (8 GPUs, 16.4 min incl. eval + sampler)",
                     table_row="Olmo-3-7B <- 3.1-32B, Fashion", fkl_scored="results/OL3x_scored.json"),
    "c3Q35_2B": dict(prio=3, teacher="q35s_9b_inj20", family="qwen", task="fashion", base=f"{M}/Qwen3.5-2B", size="2b",
                     cfg="configs/d6/sft_qwen3.5_2b_pl_k8.yaml", pdb=8, sft_gpus=8, eval_gpus=2,
                     sft_est=35, eval_est=30, peak_sft=5, err="k1_trunc_realids",
                     teacher_rec="outputs/Q35S/samples/teacher_9b_inj20.v2.records.jsonl",
                     start_rec="outputs/Q35S/samples/base_2b.v2.records.jsonl",
                     student_sft="outputs/D6/sft_qwen3.5_2b", student_sft_spec="11-D6_sft_2b (8 GPUs, 28.2 min incl. 5 epoch evals)",
                     table_row="Qwen3.5-2B <- 9B, Fashion", fkl_scored="results/Q35S_scored.json"),
    "c4Qwen3Dup": dict(prio=4, teacher="f3_t4b_dup_f20", family="qwen3", task="esci", base=f"{M}/Qwen3-1.7B",
                       cfg="configs/sft_qwen3_1.7b_esci_f2.yaml", pdb=4, sft_gpus=4, eval_gpus=2,
                       sft_est=35, eval_est=15, peak_sft=4, err="duplicate_id",
                       teacher_rec=None,
                       start_rec="outputs/K1/refs/sampled/base_b17_clean.samples.jsonl",
                       student_sft="outputs/F3/sft_b17_clean", student_sft_spec="11-F3_sft_b17_clean (4 GPUs, 5.4 min)",
                       table_row="Qwen3-1.7B <- 4B, ESCI (duplicate id)", fkl_scored="results/RX_scored.json"),
    "c5LlamaDup": dict(prio=5, teacher="dupl_t8b_dup", family="llama", task="fashion", base=f"{M}/Llama-3.2-3B-Instruct",
                       cfg="configs/sft_llama3.2_3b_pl_k8_erropd.yaml", pdb=4, sft_gpus=8, eval_gpus=2,
                       sft_est=25, eval_est=15, peak_sft=7, err="duplicate_id",
                       teacher_rec="outputs/DUPL/samples/teacher_dup.v2.records.jsonl",
                       start_rec="outputs/DUPL/samples/base.v2.records.jsonl",
                       student_sft="outputs/D1/sft_llama3.2_3b", student_sft_spec="19-D1_sft_student (8 GPUs)",
                       table_row="Llama-3.2-3B <- 3.1-8B, Fashion (duplicate id)", fkl_scored="results/DUPL_scored.json"),
    "c6Q35_08B": dict(prio=6, teacher="q35s_9b_inj20", family="qwen", task="fashion", base=f"{M}/Qwen3.5-0.8B", size="0.8b",
                      cfg="configs/d6/sft_qwen3.5_0.8b_pl_k8.yaml", pdb=8, sft_gpus=4, eval_gpus=2,
                      sft_est=25, eval_est=30, peak_sft=2, err="k1_trunc_realids",
                      teacher_rec="outputs/Q35S/samples/teacher_9b_inj20.v2.records.jsonl",
                      start_rec="outputs/Q35S/samples/base_0p8b.v2.records.jsonl",
                      student_sft="outputs/Q35S/sft_qwen3.5_0.8b", student_sft_spec="09-Q35S_b_sft_0p8b (4 GPUs, 13.8 min incl. eval + sampler)",
                      table_row="Qwen3.5-0.8B <- 9B, Fashion", fkl_scored="results/Q35S_scored.json"),
    "c7Q35_4B": dict(prio=7, teacher="q35s_9b_inj20", family="qwen", task="fashion", base=f"{M}/Qwen3.5-4B", size="4b",
                     cfg="configs/d6/sft_qwen3.5_4b_pl_k8.yaml", pdb=4, sft_gpus=8, eval_gpus=2,
                     sft_est=30, eval_est=30, peak_sft=9, err="k1_trunc_realids",
                     teacher_rec="outputs/Q35S/samples/teacher_9b_inj20.v2.records.jsonl",
                     start_rec="outputs/Q35S/samples/base_4b.v2.records.jsonl",
                     student_sft="outputs/D6/sft_qwen3.5_4b", student_sft_spec="11-D6_sft_4b (8 GPUs, 17.6 min incl. eval)",
                     table_row="Qwen3.5-4B <- 9B, Fashion", fkl_scored="results/Q35S_scored.json"),
}

def sample_owner(tkey):
    pass
    return min((c for c in CELLS if CELLS[c]["teacher"] == tkey), key=lambda c: CELLS[c]["prio"])

def paths(cell, seed):
    c = CELLS[cell]
    t = c["teacher"]
    out = f"outputs/SEQKD7/{cell}/s{seed}"
    return {
        "OUT": out,
        "SAMPLE_OUT": f"outputs/SEQKD7/samples/{t}/s{seed}",
        "DATA_DIR": f"data/seqkd7/{t}/s{seed}",
        "DS_NAME": f"seqkd7_{t}_x5_s{seed}",
        "DS_KEY": (f"esci_k8_seqkd7_{t}_x5_s{seed}" if c["task"] == "esci" else f"seqkd7_{t}_x5_s{seed}"),
    }

def shell(cell, seed=None):
    c = CELLS[cell]
    t = TEACHERS[c["teacher"]]
    kv = {"CELL": cell, "FAMILY": c["family"], "TASK": c["task"], "TKEY": c["teacher"], "TEACHER": f"{REPO}/{t['path']}",
          "BASE": c["base"], "CFG": c["cfg"], "PDB": c["pdb"], "SFT_GPUS": c["sft_gpus"], "SIZE": c.get("size", ""),
          "ERR": c["err"], "SRC": ESCI_SRC if c["task"] == "esci" else FASHION_SRC,
          "VAL": ESCI_VAL if c["task"] == "esci" else FASHION_VAL, "SNAP": ESCI_SNAP}
    if seed is not None:
        kv.update(paths(cell, seed))
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in kv.items())

if __name__ == "__main__":
    if sys.argv[1:] == ["--list"]:
        for k in sorted(CELLS, key=lambda x: CELLS[x]["prio"]):
            print(k, CELLS[k]["teacher"], CELLS[k]["base"])
        sys.exit(0)
    cell = sys.argv[1]
    if cell not in CELLS:
        sys.exit(f"unknown cell {cell}; one of {sorted(CELLS)}")
    print(shell(cell, int(sys.argv[2]) if len(sys.argv) > 2 else None))
