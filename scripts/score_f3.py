#!/usr/bin/env python3
"""Score the Qwen3 lambda grid on the injected ESCI pair."""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from eval_labels import classify, _strict_generic

PAIRS = ["A", "S", "0"]
LAMS = ["1.00", "1.25", "1.40"]
SEEDS = [1001, 1002, 1003]
STEPS = [14, 28, 42]
K1 = "k1_trunc_realids"
T_RATE = 0.195
P1_BAR = 0.333

def run_dir(pair, lam, seed):
    return REPO / "outputs/F3" / f"F3_{pair}_lam{lam}_s{seed}"

def eval_run(d, step):
    ed = d / "eval" / f"step{step}"
    mf, pf = ed / "listwise_metrics.json", ed / "predictions.jsonl"
    if not (mf.exists() and pf.exists()):
        return None
    m = json.load(open(mf))
    rows = [json.loads(l) for l in open(pf)]
    c = collections.Counter()
    nok = 0
    for r in rows:
        ok = r.get("pred_scores") is not None
        nok += ok
        c[classify(r["pred_text"], [str(x) for x in r["product_ids"]], r["K"], "product_id", ok)["label"]] += 1
    n = len(rows)
    assert n == m["n_total"], (d, n, m["n_total"])
    assert abs(nok / n - m["parse_rate"]) < 1e-9, (d, nok / n, m["parse_rate"])
    assert c.get("valid", 0) == nok, (d, dict(c), nok)
    nfeas = n - c.get("infeasible_input", 0)
    return {"parse": m["parse_rate"], "n": n, "n_feasible": nfeas, "labels": dict(c),
            "k1_share": c.get(K1, 0) / n, "k1_share_feasible": c.get(K1, 0) / nfeas if nfeas else None,
            "failures_feasible": nfeas - c.get("valid", 0), "k1": c.get(K1, 0),
            "ndcg1": m.get("ndcg@1_mean"), "useful": m.get("useful")}

def train_run(d):
    f = d / "ckpt" / "erropd_rollouts.jsonl"
    if not f.exists():
        return None
    c = collections.Counter()
    for line in open(f):
        r = json.loads(line)
        if not (29 <= int(r["step"]) <= 42) or not r.get("eligible", False):
            continue
        ids = [str(x) for x in r["product_ids"]]
        ok = bool(_strict_generic(r["text"], r["K"], ids, "product_id"))
        c[classify(r["text"], ids, r["K"], "product_id", ok)["label"]] += 1
    n = sum(c.values())
    return {"n": n, "k1_share": c.get(K1, 0) / n if n else None, "valid": c.get("valid", 0) / n if n else None,
            "labels": dict(c)}

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="A,S,0")
    ap.add_argument("--out", default=str(REPO / "results/F3_scored.json"))
    ap.add_argument("--selftest", action="store_true", help="label one pair-0 run (mechanics check only)")
    a = ap.parse_args()
    pairs = a.pairs.split(",")
    if a.selftest:
        d = run_dir("0", "1.00", 1001)
        print(json.dumps({"eval42": eval_run(d, 42), "train": train_run(d)}, indent=1))
        return 0
    missing = [f"F3_{p}_lam{l}_s{s}" for p in PAIRS for l in LAMS for s in SEEDS
               if not (run_dir(p, l, s) / "eval" / "step42" / "listwise_metrics.json").exists()]
    if any(p in ("A", "S") for p in pairs) and missing:
        print("incomplete grid; refusing to score pairs A/S. Missing step-42 evals:", missing)
        return 2
    miss_req = [m for m in missing if m.split("_")[1] in pairs]
    if miss_req:
        print("incomplete for the requested pairs; missing:", miss_req)
        return 2
    res = {"lock": " (f22c06bae185)", "runs": {}, "cells": {}}
    for p in pairs:
        for l in LAMS:
            for s in SEEDS:
                d = run_dir(p, l, s)
                res["runs"][d.name] = {"eval": {st: eval_run(d, st) for st in STEPS}, "train": train_run(d)}
            runs = [res["runs"][run_dir(p, l, s).name] for s in SEEDS]
            cell = {}
            for st in STEPS:
                ev = [r["eval"][st] for r in runs]
                if any(e is None for e in ev):
                    cell[st] = None
                    continue
                lab = collections.Counter()
                for e in ev:
                    lab.update(e["labels"])
                cell[st] = {"parse": mean([e["parse"] for e in ev]), "parse_per_seed": [e["parse"] for e in ev],
                            "k1_share": mean([e["k1_share"] for e in ev]), "k1_per_seed": [e["k1_share"] for e in ev],
                            "k1_share_feasible": mean([e["k1_share_feasible"] for e in ev]),
                            "labels_pooled": dict(lab), "k1_pooled": sum(e["k1"] for e in ev),
                            "failures_feasible_pooled": sum(e["failures_feasible"] for e in ev),
                            "k1_of_failures_per_seed": [e["k1"] / e["failures_feasible"] if e["failures_feasible"] else None for e in ev],
                            "ndcg1": mean([e["ndcg1"] for e in ev]), "useful": mean([e["useful"] for e in ev])}
            tr = [r["train"] for r in runs]
            cell["train"] = None if any(t is None for t in tr) else {
                "k1_share": mean([t["k1_share"] for t in tr]), "k1_per_seed": [t["k1_share"] for t in tr],
                "valid": mean([t["valid"] for t in tr]), "n_per_seed": [t["n"] for t in tr]}
            res["cells"][f"{p}|{l}"] = cell
    C = lambda p, l: res["cells"].get(f"{p}|{l}", {}).get(42)
    pred = {}
    if "0" in pairs:
        d0 = C("0", "1.40")["parse"] - C("0", "1.00")["parse"]
        pred["P0"] = {"diff_1.40_minus_1.00": d0, "pass": abs(d0) <= 0.10}
    if "A" in pairs and "S" in pairs:
        dropA = C("A", "1.00")["parse"] - C("A", "1.40")["parse"]
        dropS = C("S", "1.00")["parse"] - C("S", "1.40")["parse"]
        shA = C("A", "1.40")["k1_share"]
        pred["P1"] = {"drop_A": dropA, "k1_share_A_1.40": shA, "bar": P1_BAR,
                      "pass": dropA >= P1_BAR and shA >= P1_BAR}
        pred["P2"] = {"diff_S": C("S", "1.40")["parse"] - C("S", "1.00")["parse"],
                      "k1_share_S_1.00": C("S", "1.00")["k1_share"], "k1_share_S_1.40": C("S", "1.40")["k1_share"]}
        pred["P2"]["pass"] = abs(pred["P2"]["diff_S"]) <= 0.10 and pred["P2"]["k1_share_S_1.40"] <= pred["P2"]["k1_share_S_1.00"] + 0.05
        ca = C("A", "1.40")
        applies = dropA >= 0.15
        ratio = ca["k1_pooled"] / ca["failures_feasible_pooled"] if ca["failures_feasible_pooled"] else None
        pred["P3"] = {"applies": applies, "k1_of_feasible_failures_pooled": ratio,
                      "k1_pooled": ca["k1_pooled"], "failures_feasible_pooled": ca["failures_feasible_pooled"],
                      "per_seed": ca["k1_of_failures_per_seed"],
                      "pass": (ratio is not None and ratio >= 0.80) if applies else None}
        tA1, tA4 = res["cells"]["A|1.00"]["train"], res["cells"]["A|1.40"]["train"]
        pred["P4"] = {"train_k1_A_1.00": tA1 and tA1["k1_share"], "train_k1_A_1.40": tA4 and tA4["k1_share"],
                      "pass": None if not (tA1 and tA4) else (tA1["k1_share"] <= T_RATE + 0.05 and tA4["k1_share"] > T_RATE)}
        p3ok = pred["P3"]["pass"] if applies else False
        supported = pred["P1"]["pass"] and pred["P2"]["pass"] and p3ok
        falsified = (not pred["P1"]["pass"]) or dropS >= dropA
        pred["verdict"] = {"drop_A": dropA, "drop_S": dropS,
                           "M": "supported" if supported else ("falsified on this task" if falsified else "neither")}
    res["predictions"] = pred
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(pred, indent=1))
    for k, cell in res["cells"].items():
        c = cell.get(42)
        if c:
            print(k, "parse %.4f" % c["parse"], c["parse_per_seed"], "k1 %.4f" % c["k1_share"],
                  {x: y for x, y in c["labels_pooled"].items() if x != "valid"},
                  "| train k1", None if not cell["train"] else round(cell["train"]["k1_share"], 4))
    return 0

if __name__ == "__main__":
    sys.exit(main())
