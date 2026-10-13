# ErrOPD: code for "When Does On-Policy Distillation Transfer Teacher Errors?"

Xin Li, Hao Jiang, Annan Wang, Yichi Zhang, Chau Yuen (Nanyang Technological University, Singapore). Project page: https://lixin.ai/ErrOPD

This release covers the experiments behind Table 1 (error transmission under reverse-KL and forward-KL on-policy distillation and
sequence-level KD, four model families), Table 2 (duplicated IDs, teacher error rate, the two SQL errors), the cross-task code
(passage reranking and entity extraction; code only), the figure and table scripts, and the frozen per-run results they read.
Data and model checkpoints are on Hugging Face ([collection](https://huggingface.co/collections/XINLI1997/erropd); see [Data and checkpoints](#data-and-checkpoints)); rollouts are
not included.

## Layout

| Path | Contents |
|---|---|
| `verl_patch/gopd.patch` | ExOPD's per-token estimator and our loss-mode switches, as a diff against the verl fork released with ExOPD (RUCBM/G-OPD, commit `d6cb4f7`, verl 0.6.1) |
| `verl_patch/verl_q35.patch` | The same estimator ported unchanged into upstream verl (`6eeb5711`, 0.8.0.dev), used for the Qwen3.5 and Olmo 3 runs |
| `scripts/data_*.py` | Data construction and error injection (`data_inject.py`: early close, duplicated ID, extra item; `--frac` sets the teacher's error rate) |
| `scripts/sft_*.sh`, `configs/sft/` | Supervised fine-tuning of teachers and students (LlamaFactory) |
| `scripts/opd_*.sh` | On-policy distillation launchers. `opd_arms.sh` maps `ARM` to the loss mode: `vanilla` = reverse KL with ExOPD's estimator, `fkl` = forward KL on one teacher sample per student position, `rkl` = full-vocabulary reverse KL |
| `scripts/seqkd_*.sh` | Sequence-level KD (teacher samples, then SFT) |
| `scripts/eval_*` | Greedy evaluation with the strict parser, temperature-1 sampling on held-out prompts, response labelling (`eval_labels.py`), probes |
| `scripts/score_*.py` | Scorers. `score_core.py` defines the error share and transmission (step-k share minus the starting checkpoint's share, in points) with bootstrap intervals over prompts and seeds |
| `scripts/fig_*.py` | Figure and table scripts |
| `runs/*.sh` | The commands as launched for every run of each block, in launch order |
| `results/` | Frozen per-run scores (JSON/CSV) that the figure and table scripts read |
| `tables_frozen/` | LaTeX of every table in the paper |
| `scripts/env.sh`, `scripts/env_q35.sh` | Interpreter and tree paths (set these first) |

## Training code

All reverse-KL runs use ExOPD's per-token estimator, in the verl fork released with ExOPD or, for the Qwen3.5 and Olmo 3 runs,
ported unchanged into upstream verl.

```bash
# ExOPD's fork (Qwen3, Llama and the SQL runs)
git clone https://github.com/RUCBM/G-OPD && cd G-OPD && git checkout d6cb4f7
git apply /path/to/erropd/verl_patch/gopd.patch
# upstream verl (Qwen3.5 and Olmo 3 runs); the patch paths start with verl/, so apply it from the parent directory
mkdir verl-q35 && cd verl-q35 && git clone https://github.com/volcengine/verl verl && (cd verl && git checkout 6eeb5711)
git apply /path/to/erropd/verl_patch/verl_q35.patch
```

The patches add the package `verl/erropd/` (validity logging, rollout dumps, per-token lambda, loss-mode checks) and these switches,
all off by default:

| Switch | Effect |
|---|---|
| `PAPER_LOSS_MODE=opd` (default) | reverse KL with ExOPD's estimator; lambda is `actor_rollout_ref.actor.policy_loss.lambda_vals` |
| `PAPER_LOSS_MODE=fkl_sampled` | forward KL: cross-entropy on one teacher sample at each student prefix |
| `PAPER_LOSS_MODE=rkl_full` | full-vocabulary reverse KL (lambda 1) |
| `PAPER_FI_MODE`, `PAPER_FI_DOSE`, `PAPER_FI_FORCE_SURROGATE` | factor-isolation arms (gradient magnitude and direction at the error decision) |
| `PAPER_MIX_BETA` | (1 - beta) reverse KL + beta forward KL on the same rollouts |

Each block ran on a snapshot of the fork at the commit where its switch was added. Every later switch is off by default, so the
patched tree with default flags runs the same update as those snapshots (each commit's CPU equivalence tests are in
`verl/tests/erropd/` and `verl/tests/q35/`; their fixtures use made-up item ids and scores, not dataset records; we did not re-run every block on the final tree). Experimental paths that no reported
result uses were removed from the patch; defaults are unchanged.

| Block (paper) | Fork snapshot |
|---|---|
| Table 1, Qwen3 and Llama matched reverse-KL vs forward-KL, duplicated ID, added seeds | G-OPD + patch at `302dfd90ef` |
| Table 1, Qwen3 lambda 1.40 column | `7a040afeee` |
| Table 1, Llama lambda 1.40 column | G-OPD at `5950326`, before the switches were added |
| Table 1, Qwen3.5 and Olmo 3 | upstream verl + port at `d80a0a3` (reverse KL) and `0fca107` (forward KL) |
| Table 2 | `302dfd90ef` |

## Reproducing a Table 1 cell

1. Set the paths in `scripts/env.sh` (and `scripts/env_q35.sh` for Qwen3.5 and Olmo 3). Paths written `/path/to/...` in the
   scripts are placeholders for your data, models and outputs.
2. Build and inject the data: `python scripts/data_build_esci.py`, then
   `python scripts/data_inject.py --kinds k1,dup --frac 0.2 --think --out data/esci_k8_f3` (Fashion: `data_build_fashion.py`, then
   `data_inject_fashion.py --fracs 0.20,0.00`).
3. Fine-tune the teacher on the injected targets and the student on clean targets (`sft_qwen3_esci.sh`, `sft_llama_fashion*.sh`,
   `sft_q35_fashion.sh`, `sft_olmo_fashion.sh`).
4. Distill for 42 steps: `ARM=vanilla` or `ARM=fkl` through `opd_arms.sh` (Qwen3), `opd_run_llama.sh`, `opd_run_q35_olmo.sh`.
   Shared settings: 128 prompts per step, one response per prompt at temperature 1 and top-p 1, 512 response tokens, AdamW at
   learning rate 1e-6, gradient clipping 1.0, one update per batch.
5. Sample 4 responses per held-out prompt at temperature 1 (`eval_sample.sh`, `eval_sample_llama.sh`, `eval_sample_q35.sh`) and
   score (`score_contrast_qwen3.py`, `score_contrast_llama.py`, `score_q35.py`, `score_olmo.py`, `score_five_seed.py`).

`runs/table1_*.sh` and `runs/table2_beyond.sh` list every command as launched, with seeds and lambda values.

## Figures and tables from the frozen results

```bash
mkdir -p out/figures out/tables && cp tables_frozen/*.tex out/tables/
python scripts/fig_figs.py fig_rate fig_when fig_greedy fig_long
```

This redraws the teacher-rate, starting-rate/quality, greedy-evaluation and long-run figures from `results/`; we checked that the
output is pixel-identical to the paper's figures. The other figures read raw rollouts that are not released. Nine tables were
transcribed from scorer outputs rather than generated; `tables_frozen/` holds every table as printed.

## Data and checkpoints

Everything is on Hugging Face, collected in [this collection](https://huggingface.co/collections/XINLI1997/erropd).

| | |
|---|---|
| Data | [ErrOPD-ESCI](https://huggingface.co/datasets/XINLI1997/ErrOPD-ESCI) (Apache-2.0), [ErrOPD-Spider](https://huggingface.co/datasets/XINLI1997/ErrOPD-Spider) (CC BY-SA 4.0; the SQLite databases come from the official Spider release), [ErrOPD-Fashion](https://huggingface.co/datasets/XINLI1997/ErrOPD-Fashion) (derived from Amazon Reviews 2023, research use only; our scores and list structure CC BY 4.0) |
| Checkpoints | 34 models: the error-injected teachers, the students' starting points, and one reverse-KL and one forward-KL student after 42 steps for each main setting, plus the Qwen3 lambda 1.40 student. Names follow `ErrOPD-<model>-<role>-<task>`; Llama-based models start with `Llama-` as the Llama license requires. |

## Not in this release

Rollouts, the data and checkpoints of the appendix-only experiments, the cross-task data (CoNLL-2003, MS MARCO and TREC-DL must be
obtained from their sources; build scripts only), the step that scored the Fashion reviews (the scores themselves are in
ErrOPD-Fashion), and our internal run records. Some launchers still call helpers that were part of our job infrastructure and are
not included; the commands in `runs/` show how each was used.

## Compute

About 2,250 GPU-hours on NVIDIA B200 for all launched runs, about 2,050 of them for the experiments reported in the paper (about
530 for the math experiments).

## Environments

verl training: torch 2.8.0, vllm 0.11.0, transformers 4.57.6, ray 2.55.1, flash-attn 2.8.1. Sampling and evaluation: torch 2.10.0,
vllm 0.18.0, transformers 4.57.6. SFT: LlamaFactory 0.9.5.dev0 (`97433c5`, version checks relaxed), torch 2.8.0, transformers 4.57.6, deepspeed 0.17.6.
Qwen3.5 and Olmo 3: torch 2.10.0, vllm 0.18.0, transformers 5.3.0, deepspeed 0.18.4.

## Citation

```bibtex
@article{li2026erropd,
  title   = {ErrOPD: When Does On-Policy Distillation Transfer Teacher Errors?},
  author  = {Li, Xin and Jiang, Hao and Wang, Annan and Zhang, Yichi and Yuen, Chau},
  journal = {arXiv preprint arXiv:2605.08737},
  year    = {2026}
}
```


## License

Code in this repository (scripts, configs, patches written by us) is released under the MIT License. The training code is based on RUCBM/G-OPD (MIT). See `NOTICE` for third-party code.
