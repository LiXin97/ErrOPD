#!/usr/bin/env python3
"""Score matched reverse-KL vs forward-KL OPD, Qwen3."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import score_core as S

SEEDS = (1001, 1002, 1003)
PROBE = REPO / "outputs/E3/probe/item7_clean"

def arm(exp):
    base = S.sample_shares("base_clean")
    bp = json.loads((PROBE / "base_clean.json").read_text())["mean"]
    out = {}
    for st in (14, 28, 42):
        sh = [S.sample_shares(f"{exp}_A_s{s}_step{st}") for s in SEEDS]
        X = np.stack([x["k1"] for x in sh])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(10000):
            si, pi = rng.integers(0, 3, 3), rng.integers(0, X.shape[1], X.shape[1])
            bs.append(X[np.ix_(si, pi)].mean() - base["k1"][pi].mean())
        per = [float(x["k1"].mean()) for x in sh]
        pr = [json.loads((PROBE / f"{exp}_A_s{s}_step{st}.json").read_text())["mean"] - bp for s in SEEDS]
        out[st] = {"share_per_seed": per, "Tr": float(np.mean(per) - base["k1"].mean()),
                   "Tr_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   "probe_shift": float(np.mean(pr)), "probe_per_seed": pr, "labels": [x["labels"] for x in sh]}
    tr = [S.train_share(REPO / f"outputs/{exp}/{exp}_A_s{s}") for s in SEEDS]
    out["train_share_per_seed"] = [t["k1"] for t in tr]
    out["train_share"] = float(np.mean(out["train_share_per_seed"]))
    out["base_share"] = float(base["k1"].mean())
    return out

def main():
    F, O = arm("FKL"), arm("OPDre")
    b = F["base_share"]
    t = float(S.sample_shares("teacher")["k1"].mean())
    f42 = F[42]
    primary = ("holds" if f42["Tr"] >= 0.10 and all(x > b for x in f42["share_per_seed"]) else
               "kill (FKL does not transmit)" if all(x - b < 0.02 for x in f42["share_per_seed"]) else "inconclusive")
    pred = {
        "PRIMARY_FKL_transmits_ge_0.10_by_step42": primary,
        "FKL_Tr14_ge_0.10": F[14]["Tr"] >= 0.10,
        "FKL_share42_in_[0.10,0.39]": 0.10 <= float(np.mean(f42["share_per_seed"])) <= 0.39,
        "FKL_share42_le_teacher_plus_0.05": float(np.mean(f42["share_per_seed"])) <= t + 0.05,
        "FKL_probe42_in_[5.8,23]": 5.8 <= f42["probe_shift"] <= 23,
        "FKL_probe14_ge_2": F[14]["probe_shift"] >= 2,
        "FKL_train_share_ge_0.10": F["train_share"] >= 0.10,
        "OPDre_Tr42_lt_0.02": O[42]["Tr"] < 0.02,
        "OPDre_abs_probe42_le_0.5": abs(O[42]["probe_shift"]) <= 0.5,
        "OPDre_train_share_lt_0.02": O["train_share"] < 0.02,
        "contrast_Tr_FKL_minus_OPDre_ge_0.08_all_pairs": (f42["Tr"] - O[42]["Tr"] >= 0.08 and
                                                          all(x > y for x, y in zip(f42["share_per_seed"], O[42]["share_per_seed"]))),
    }
    res = {"lock": " (aa05d568c995)", "teacher_share": t, "FKL": F, "OPDre": O,
           "predictions": pred}
    (REPO / "results/FKL_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(pred, indent=1))
    for name, A in (("FKL", F), ("OPDre", O)):
        for st in (14, 28, 42):
            v = A[st]
            print(name, st, "shares", [round(x, 4) for x in v["share_per_seed"]], "Tr %.4f %s" % (v["Tr"], [round(x, 4) for x in v["Tr_ci95"]]),
                  "probe %.2f" % v["probe_shift"], [round(x, 2) for x in v["probe_per_seed"]])
        print(name, "train share", [round(x, 4) for x in A["train_share_per_seed"]], "base %.4f teacher %.4f" % (A["base_share"], t))

if __name__ == "__main__":
    main()
