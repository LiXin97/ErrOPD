# Reproduce ErrOPD

The training code behind every run in the paper. The [Quick start](../README.md#quick-start) of the README redraws the
paper's figures from the released per-run results and re-runs one result.

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

<details><summary>Loss switches and the fork snapshot behind each table</summary>

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

</details>
