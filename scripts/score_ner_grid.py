#!/usr/bin/env python3
"""Aggregate the entity-extraction lambda grid."""
import argparse
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T3 = REPO / "outputs/T3"
LAMS = ["1.00", "1.15", "1.25", "1.40", "1.50"]
SEEDS = [42, 7, 13]
TYPES = ["extra_prose", "malformed", "schema", "non_verbatim", "unterminated", "truncated", "early_close"]

def load(lam, seed, step):
    p = T3 / f"opd/lam{lam}_s{seed}/eval/step{step}/t3_metrics.json"
    return json.load(open(p)) if p.exists() else None

def row(m):
    sh = m.get("first_violation_share", {})
    return {"valid": m["parse_rate"], "f1_valid": m["f1_valid"], "useful_f1": m["useful_f1"],
            "recall_valid": m["recall_valid"], "precision_valid": m["precision_valid"],
            "set_f1_valid": m["set_f1_valid"], "tail_recall": (m["recall_by_gold_third_valid"] or [None] * 3)[2],
            "missing_tail_rate_valid": m["missing_tail_rate_valid"], "overrun_rate_valid": m["overrun_rate_valid"],
            **{f"share_{t}": sh.get(t, 0.0) for t in TYPES}}

def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None

def main():
    global T3
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=42)
    ap.add_argument("--root", default="", help="tests only")
    a = ap.parse_args()
    if a.root:
        T3 = Path(a.root)
    cells = {}
    for lam in LAMS:
        for s in SEEDS:
            m = load(lam, s, a.step)
            if m:
                cells[(lam, s)] = row(m)
    out = {"step": a.step, "cells": {f"{l}_s{s}": v for (l, s), v in cells.items()}, "by_lambda": {}}
    keys = list(next(iter(cells.values())).keys()) if cells else []
    for lam in LAMS:
        rs = [cells[(lam, s)] for s in SEEDS if (lam, s) in cells]
        if rs:
            out["by_lambda"][lam] = {"n_seeds": len(rs), **{k: mean([r[k] for r in rs]) for k in keys},
                                     "valid_sd": st.stdev([r["valid"] for r in rs]) if len(rs) > 1 else None}
    res = {}
    complete = all((l, s) in cells for l in ("1.00", "1.40") for s in SEEDS)
    if complete:
        for t in TYPES:
            diffs = [cells[("1.40", s)][f"share_{t}"] - cells[("1.00", s)][f"share_{t}"] for s in SEEDS]
            md = st.mean(diffs)
            res[t] = {"mean_diff": md, "diffs": diffs,
                      "outcome": "amplified" if (md >= 0.10 and all(d > 0 for d in diffs)) else
                                 ("not_amplified" if md <= 0.02 else "inconclusive")}
        dv = st.mean([cells[("1.40", s)]["valid"] for s in SEEDS]) - st.mean([cells[("1.00", s)]["valid"] for s in SEEDS])
        res["collapse"] = {"mean_valid_diff": dv, "collapse": dv <= -0.30}
    out["outcomes"] = res if complete else "incomplete (need lambda 1.00 and 1.40 x 3 seeds)"
    out["recall_outcome"] = None
    if complete:
        dr = [cells[("1.40", s)]["recall_valid"] - cells[("1.00", s)]["recall_valid"] for s in SEEDS]
        md = st.mean(dr)
        out["recall_outcome"] = {"mean_diff": md, "diffs": dr,
                                 "outcome": "recall_drop" if (md <= -0.02 and all(d < 0 for d in dr)) else
                                            ("no_recall_drop" if md >= -0.005 else "inconclusive")}
    pp = T3 / "asym/analysis/asymmetry.json"
    if pp.exists() and complete:
        pred = json.load(open(pp))["prediction"]
        checks = []
        for t in TYPES:
            if t == "truncated":
                continue
            p = pred["per_type"][t]["prediction"]
            o = res[t]["outcome"]
            if p == "amplification":
                checks.append((t, p, o, None if o == "inconclusive" else o == "amplified"))
            elif p in ("no amplification", "no_amplification", "suppression"):
                checks.append((t, p, o, None if o == "inconclusive" else o == "not_amplified"))
        if pred["early_close_amplified"]:
            ro = out["recall_outcome"]["outcome"]
            checks.append(("recall", "recall_drop", ro, None if ro == "inconclusive" else ro == "recall_drop"))
        vcall = pred.get("validity_call", "collapse" if pred["amplified_validity_types"] else "no collapse")
        if vcall in ("collapse", "no collapse"):
            checks.append(("collapse", vcall, res["collapse"]["collapse"],
                           (vcall == "collapse") == res["collapse"]["collapse"]))
        tested = [c for c in checks if c[3] is not None]
        out["prediction_check"] = {"checks": checks, "holds": all(c[3] for c in tested),
                                   "untested": [c[0] for c in checks if c[3] is None], "task_level": pred["task_level"]}
    cons = {}
    for (lam, s), v in cells.items():
        base_dir = T3 / f"opd/lam{lam}_s{s}"
        up = base_dir / f"eval/step{a.step}/predictions.jsonl"
        for arm in ("schema", "validator"):
            mp = base_dir / f"eval_{arm}/step{a.step}/t3_metrics.json"
            pp = base_dir / f"eval_{arm}/step{a.step}/predictions.jsonl"
            if not (mp.exists() and up.exists() and pp.exists()):
                continue
            m = json.load(open(mp))
            U = [json.loads(l) for l in open(up)]
            C = [json.loads(l) for l in open(pp)]
            vu = [i for i, r in enumerate(U) if r["cls"] == "valid"]
            ident = sum(1 for i in vu if C[i]["pred_text"].strip() == U[i]["pred_text"].strip()) / len(vu) if vu else None
            cons[f"{lam}_s{s}_{arm}"] = {"lam": lam, "seed": s, "arm": arm, **row(m), "useful_recall": m["useful_recall"],
                                         "identical_share_among_unconstrained_valid": ident}
    out["constrained"] = cons
    if cons:
        claim = {}
        base_ec = out["by_lambda"].get("1.00", {}).get("share_early_close")
        for lam in LAMS:
            u = out["by_lambda"].get(lam)
            vs = [c for c in cons.values() if c["lam"] == lam and c["arm"] == "validator"]
            if not u or base_ec is None or len(vs) < u["n_seeds"]:
                continue
            ec_c = mean([c["share_early_close"] for c in vs])
            if u["share_early_close"] - base_ec >= 0.10:
                claim[lam] = {"unconstrained_ec": u["share_early_close"], "validator_ec": ec_c,
                              "holds": ec_c >= 0.8 * u["share_early_close"]}
        out["constrained_claim"] = claim or "no lambda with early_close rise >= 0.10 (claim not tested)"
    json.dump(out, open(T3 / f"stageB_summary_step{a.step}.json", "w"), indent=1)
    hdr = ["lam", "n", "valid", "sd", "f1_valid", "useful_f1", "tail_rec", "early_close", "non_verb", "unterm", "trunc"]
    print(" ".join(f"{h:>10}" for h in hdr))
    for lam, v in out["by_lambda"].items():
        vals = [lam, v["n_seeds"], v["valid"], v["valid_sd"], v["f1_valid"], v["useful_f1"], v["tail_recall"],
                v["share_early_close"], v["share_non_verbatim"], v["share_unterminated"], v["share_truncated"]]
        print(" ".join(f"{x:>10.4f}" if isinstance(x, float) else f"{str(x):>10}" for x in vals))
    print(json.dumps(out.get("outcomes"), indent=1))
    if "prediction_check" in out:
        print(json.dumps(out["prediction_check"], indent=1))

if __name__ == "__main__":
    main()
