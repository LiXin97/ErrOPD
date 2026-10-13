#!/usr/bin/env python3
"""Error share and transmission (step-k minus start, percentage points) with prompt-and-seed bootstrap intervals."""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from eval_labels import classify, _strict_generic

E3 = REPO / "outputs/E3"
SEEDS = [1001, 1002, 1003]
LAMS = ["1.00", "1.40"]
K1, RUN = "k1_trunc_realids", "extra_items"
NBOOT = 10000
VAL = [json.loads(l) for l in open(REPO / "data/esci_k8/val.jsonl")]

def label(text, ids):
    ok = bool(_strict_generic(text, 8, ids, "product_id"))
    return classify(text, ids, 8, "product_id", ok)["label"]

def sample_shares(name):
    pass
    f = E3 / "samples" / name / "rollouts_train.jsonl"
    if not (f.exists() and (E3 / "samples" / name / "rollouts_train.ok").exists()):
        return None
    k1 = collections.defaultdict(list)
    run = collections.defaultdict(list)
    labs = collections.Counter()
    for line in open(f):
        r = json.loads(line)
        if r["split"] != "val":
            continue
        ids = [str(x) for x in VAL[r["row"]]["metadata"]["product_ids"]]
        lab = label(r["text"], ids)
        labs[lab] += 1
        k1[r["row"]].append(lab == K1)
        run[r["row"]].append(lab == RUN)
    rows = sorted(k1)
    assert len(rows) == len(VAL) and all(len(k1[i]) == 4 for i in rows), (name, len(rows))
    return {"k1": np.array([np.mean(k1[i]) for i in rows]), "run": np.array([np.mean(run[i]) for i in rows]),
            "labels": dict(labs)}

def train_share(run_dir):
    f = run_dir / "ckpt" / "erropd_rollouts.jsonl"
    if not f.exists():
        return None
    c = collections.Counter()
    for line in open(f):
        r = json.loads(line)
        if 29 <= int(r["step"]) <= 42 and r.get("eligible", False):
            c[label(r["text"], [str(x) for x in r["product_ids"]])] += 1
    n = sum(c.values())
    return {"n": n, "k1": c.get(K1, 0) / n, "run": c.get(RUN, 0) / n, "labels": dict(c)}

def base_train_rate(arm):
    rows = [r for r in csv.DictReader(open(REPO / f"outputs/B2/e3_{arm}/rollouts_train.labels.csv")) if r["split"] == "train"]
    return sum(r["label"] == K1 for r in rows) / len(rows), len(rows)

def greedy(run_dir, step):
    f = run_dir / "eval" / f"step{step}" / "listwise_metrics.json"
    return json.load(open(f))["parse_rate"] if f.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="R,E2,E5")
    ap.add_argument("--out", default=str(REPO / "results/E3_scored.json"))
    a = ap.parse_args()
    res = {"lock": "", "arms": {}}
    teacher = sample_shares("teacher")
    res["teacher_eval_T1"] = None if teacher is None else {"k1": float(teacher["k1"].mean()), "labels": teacher["labels"]}
    for arm in a.arms.split(","):
        base = sample_shares(f"base_{arm}")
        runs = {(l, s): sample_shares(f"E3_{arm}_lam{l}_s{s}") for l in LAMS for s in SEEDS}
        miss = [k for k, v in runs.items() if v is None] + ([] if base is not None else ["base"])
        if miss:
            res["arms"][arm] = {"status": f"incomplete: {miss}"}
            print(arm, "incomplete", miss)
            continue
        X1 = np.stack([runs[("1.00", s)]["k1"] for s in SEEDS])
        X4 = np.stack([runs[("1.40", s)]["k1"] for s in SEEDS])
        b = base["k1"]
        Tr = float(X1.mean() - b.mean())
        Am = float((X4 - X1).mean())
        rng = np.random.default_rng(0)
        trb, amb = np.empty(NBOOT), np.empty(NBOOT)
        for i in range(NBOOT):
            si = rng.integers(0, 3, 3)
            pi = rng.integers(0, len(b), len(b))
            trb[i] = X1[np.ix_(si, pi)].mean() - b[pi].mean()
            amb[i] = (X4 - X1)[np.ix_(si, pi)].mean()
        trci = (float(np.percentile(trb, 2.5)), float(np.percentile(trb, 97.5)))
        amci = (float(np.percentile(amb, 2.5)), float(np.percentile(amb, 97.5)))
        per1 = [float(x.mean()) for x in X1]
        pera = [float((X4[s] - X1[s]).mean()) for s in range(3)]
        tcall = ("transmitted" if Tr >= 0.02 and trci[0] > 0 and all(v > b.mean() for v in per1)
                 else "not transmitted" if Tr < 0.02 else "inconclusive")
        acall = ("amplified" if Am > 0 and amci[0] > 0 and sum(v > 0 for v in pera) >= 2
                 else "not amplified" if Am <= 0 else "inconclusive")
        r = {"base_k1": float(b.mean()), "lam1.00_k1_per_seed": per1, "lam1.40_k1_per_seed": [float(x.mean()) for x in X4],
             "Tr": Tr, "Tr_ci95": trci, "transmission": tcall, "Am": Am, "Am_ci95": amci, "Am_per_seed": pera,
             "amplification": acall, "base_labels": base["labels"],
             "run_labels": {f"{l}|{s}": runs[(l, s)]["labels"] for l in LAMS for s in SEEDS},
             "greedy_parse": {f"{l}|{s}|{st}": greedy(E3 / f"E3_{arm}_lam{l}_s{s}", st)
                              for l in LAMS for s in SEEDS for st in (14, 28, 42)}}
        if arm == "R":
            br, nbr = base_train_rate("R")
            tr = {(l, s): train_share(E3 / f"E3_R_lam{l}_s{s}") for l in LAMS for s in SEEDS}
            if all(v is not None for v in tr.values()):
                t1 = [tr[("1.00", s)]["k1"] for s in SEEDS]
                t4 = [tr[("1.40", s)]["k1"] for s in SEEDS]
                Trt = float(np.mean(t1) - br)
                Amt = float(np.mean([y - x for x, y in zip(t1, t4)]))
                r["train"] = {"base_train_prompt_rate": br, "n_base": nbr, "lam1.00_per_seed": t1, "lam1.40_per_seed": t4,
                              "Tr_train": Trt, "transmission_train": ("transmitted" if Trt >= 0.02 and all(v > br for v in t1)
                                                                      else "not transmitted" if Trt < 0.02 else "inconclusive"),
                              "Am_train": Amt, "amplification_train": ("amplified" if Amt > 0 and sum(y > x for x, y in zip(t1, t4)) >= 2
                                                                       else "not amplified" if Amt <= 0 else "inconclusive")}
            run1 = float(np.mean([runs[("1.00", s)]["run"].mean() for s in SEEDS]))
            r["route_premise"] = {"base_runaway": float(base["run"].mean()), "lam1.00_runaway": run1,
                                  "holds": run1 <= base["run"].mean() / 2}
            trn = r.get("train", {})
            if not r["route_premise"]["holds"]:
                r["account"] = "no test (route premise fails)"
            elif Tr < 0.02 and trn.get("Tr_train", 1.0) < 0.02:
                r["account"] = "coupled-route account retired"
            else:
                r["account"] = "not retired"
        res["arms"][arm] = r
        print(arm, json.dumps({k: r[k] for k in ("base_k1", "Tr", "Tr_ci95", "transmission", "Am", "Am_ci95", "amplification")}),
              r.get("train", ""), r.get("route_premise", ""), r.get("account", ""))
    A = res["arms"]
    if "E5" in A and "Tr" in A["E5"]:
        e5 = A["E5"]
        res["E5_falsifier_fires"] = e5["transmission"] == "not transmitted" and e5["amplification"] == "not amplified"
        if "E2" in A and "Tr" in A["E2"]:
            res["ordering_Tr_E5_ge_E2"] = e5["Tr"] >= A["E2"]["Tr"]
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("arms",)}, indent=1, default=float))

if __name__ == "__main__":
    main()
