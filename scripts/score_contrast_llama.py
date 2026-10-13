#!/usr/bin/env python3
"""Score matched reverse-KL vs forward-KL OPD, Llama."""
import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "outputs/D1inj/fkl_samples"
SEEDS = (1001, 1002, 1003)

def shares(name):
    per = collections.defaultdict(list)
    lab = collections.Counter()
    for line in open(D / f"{name}.v2.records.jsonl"):
        r = json.loads(line)
        if r["split"] != "val" or r["label"] == "infeasible_input":
            continue
        lab[r["label"]] += 1
        per[r["row"]].append(r["label"] == "k1_trunc_realids")
    keys = sorted(per)
    n = sum(lab.values())
    return {"share": lab["k1_trunc_realids"] / n, "n": n, "labels": dict(lab), "keys": keys,
            "per_prompt": np.array([np.mean(per[k]) for k in keys])}

def arm(a, base):
    out = {}
    for st in (14, 28, 42):
        sh = [shares(f"{a}_L_s{s}_step{st}") for s in SEEDS]
        assert all(x["keys"] == base["keys"] for x in sh)
        X = np.stack([x["per_prompt"] for x in sh])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(10000):
            si, pi = rng.integers(0, 3, 3), rng.integers(0, X.shape[1], X.shape[1])
            bs.append(X[np.ix_(si, pi)].mean() - base["per_prompt"][pi].mean())
        per = [x["share"] for x in sh]
        out[st] = {"share_per_seed": per, "Tr": float(np.mean(per) - base["share"]),
                   "Tr_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "labels": [x["labels"] for x in sh]}
    return out

def main():
    base, teacher = shares("base"), shares("teacher")
    F, O = arm("FKL", base), arm("OPDre", base)
    b = base["share"]
    f42 = F[42]
    pred = {
        "PRIMARY_FKL_L_transmits_ge_0.10_by_step42": ("holds" if f42["Tr"] >= 0.10 and all(x > b for x in f42["share_per_seed"]) else
                                                      "kill" if all(x - b < 0.02 for x in f42["share_per_seed"]) else "inconclusive"),
        "OPDre_L_Tr42_lt_0.02": O[42]["Tr"] < 0.02,
        "contrast_ge_0.08_3of3": f42["Tr"] - O[42]["Tr"] >= 0.08 and all(x > y for x, y in zip(f42["share_per_seed"], O[42]["share_per_seed"])),
        "FKL_L_Tr14_ge_0.10": F[14]["Tr"] >= 0.10,
        "FKL_L_share42_le_teacher_plus_0.05": float(np.mean(f42["share_per_seed"])) <= teacher["share"] + 0.05,
    }
    res = {"lock": " (d13faf89ac59)", "base_share": b, "teacher_share": teacher["share"],
           "base_labels": base["labels"], "teacher_labels": teacher["labels"], "FKL_L": F, "OPDre_L": O, "predictions": pred}
    (REPO / "results/FKL_L_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(pred, indent=1))
    print("base %.4f teacher %.4f" % (b, teacher["share"]))
    for name, A in (("FKL_L", F), ("OPDre_L", O)):
        for st in (14, 28, 42):
            v = A[st]
            print(name, st, [round(x, 4) for x in v["share_per_seed"]], "Tr %.4f %s" % (v["Tr"], [round(x, 4) for x in v["Tr_ci95"]]))

if __name__ == "__main__":
    main()
