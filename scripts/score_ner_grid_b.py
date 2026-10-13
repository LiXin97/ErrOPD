#!/usr/bin/env python3
"""Aggregate the entity-extraction lambda grid, second teacher."""
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VALID_LABELS = ["extra_prose", "malformed", "schema", "non_verbatim", "unterminated", "truncated"]
SPAN_SYNTAX = ["non_verbatim", "malformed", "unterminated", "extra_prose", "schema"]

def load(path):
    return json.load(open(path)) if path.exists() else None

def cell(m):
    sh = m.get("first_violation_share", {})
    inv = sum(sh.get(k, 0.0) for k in VALID_LABELS)
    return {"inv": inv, "valid": m["parse_rate"], "f1_valid": m["f1_valid"], "recall_valid": m["recall_valid"],
            "useful_f1": m["useful_f1"], "early_close": sh.get("early_close", 0.0),
            "tail_recall": (m.get("recall_by_gold_third_valid") or [None] * 3)[2],
            **{k: sh.get(k, 0.0) for k in VALID_LABELS}}

def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None

def grid(root, lams, seeds, steps, suffix=""):
    C = {}
    for lam in lams:
        for s in seeds:
            for k in steps:
                m = load(REPO / root / f"lam{lam}_s{s}{suffix}" / "eval" / f"step{k}" / "t3_metrics.json")
                if m:
                    C[(lam, s, k)] = cell(m)
    return C

def table(C, lams, seeds, steps, keys):
    print("lam   step  n  " + " ".join(f"{k:>12}" for k in keys))
    for k in steps:
        for lam in lams:
            rs = [C[(lam, s, k)] for s in seeds if (lam, s, k) in C]
            if rs:
                print(f"{lam}  {k:>4}  {len(rs)}  " + " ".join(f"{mean([r[x] for r in rs]):>12.4f}" for x in keys))

def t3b():
    lams, seeds, steps = ["1.00", "1.25", "1.40", "1.50"], [1001, 1002, 1003], [7, 14, 28, 42]
    C = grid("outputs/T3b/opd", lams, seeds, steps)
    table(C, lams, seeds, steps, ["inv", "non_verbatim", "schema", "early_close", "recall_valid", "f1_valid", "useful_f1"])
    out = {"cells": {f"{l}_s{s}_step{k}": v for (l, s, k), v in C.items()}}
    need = [(l, s, 14) for l in ("1.00", "1.25", "1.40") for s in seeds]
    if not all(x in C for x in need):
        out["outcome"] = "incomplete at step 14"
    else:
        def m(lam, key, k=14):
            return mean([C[(lam, s, k)][key] for s in seeds])
        i0, i25, i40 = m("1.00", "inv"), m("1.25", "inv"), m("1.40", "inv")
        pairs = [C[("1.40", s, 14)]["inv"] < C[("1.00", s, 14)]["inv"] for s in seeds]
        holds = i40 <= 0.8 * i0 and (i0 - i40) >= 0.01 and all(pairs) and i25 < i0
        task = "holds" if holds else ("fails" if i40 >= i0 else "inconclusive")
        res = {"task_rule2": {"inv_1.00": i0, "inv_1.25": i25, "inv_1.40": i40, "pairs_lower": pairs, "outcome": task}}
        for e in ("schema", "non_verbatim"):
            a, b = m("1.00", e), m("1.40", e)
            pr = [C[("1.40", s, 14)][e] < C[("1.00", s, 14)][e] for s in seeds]
            if a < 0.01:
                o = "untestable (share at 1.00 < 0.01)"
            else:
                o = "suppressed" if (b <= 0.8 * a and all(pr)) else ("not suppressed" if b >= a else "inconclusive")
            res[e] = {"share_1.00": a, "share_1.40": b, "pairs_lower": pr, "outcome": o}
        a, b = m("1.00", "early_close"), m("1.40", "early_close")
        pr = [C[("1.40", s, 14)]["early_close"] < C[("1.00", s, 14)]["early_close"] for s in seeds]
        r0, r40 = m("1.00", "recall_valid"), m("1.40", "recall_valid")
        res["early_close"] = {"share_1.00": a, "share_1.40": b, "pairs_lower": pr, "recall_valid_1.00": r0,
                              "recall_valid_1.40": r40,
                              "outcome": "holds" if (b <= 0.8 * a and all(pr) and r40 > r0) else "does not hold"}
        if all((l, s, 42) in C for l in ("1.00", "1.40") for s in seeds):
            for e in ("extra_prose", "malformed", "unterminated"):
                d = [C[("1.40", s, 42)][e] - C[("1.00", s, 42)][e] for s in seeds]
                md = st.mean(d)
                res[e] = {"mean_diff_step42": md, "diffs": d,
                          "outcome": "amplified" if (md >= 0.10 and all(x > 0 for x in d)) else
                                     ("not amplified" if md <= 0.02 else "inconclusive")}
        out["outcome"] = res
    json.dump(out, open(REPO / "outputs/T3b/stageB_summary.json", "w"), indent=1)
    print(json.dumps(out.get("outcome"), indent=1))

def long_arm():
    lams, seeds, steps = ["1.00", "1.50"], [42, 7, 13], [42, 84, 126, 168, 210]
    C = grid("outputs/T3/opd", lams, seeds, steps, suffix="_ep15")
    table(C, lams, seeds, steps, ["valid", "non_verbatim", "malformed", "unterminated", "extra_prose", "schema",
                                  "early_close", "f1_valid", "useful_f1"])
    out = {"cells": {f"{l}_s{s}_step{k}": v for (l, s, k), v in C.items()}}
    if not all((l, s, 210) in C for l in lams for s in seeds):
        out["outcome"] = "incomplete at step 210"
    else:
        def m(lam, key):
            return mean([C[(lam, s, 210)][key] for s in seeds])
        drop = m("1.00", "valid") - m("1.50", "valid")
        pairs = [C[("1.50", s, 210)]["valid"] < C[("1.00", s, 210)]["valid"] for s in seeds]
        excess = m("1.50", "inv") - m("1.00", "inv")
        span = sum(m("1.50", e) - m("1.00", e) for e in SPAN_SYNTAX)
        frac = span / excess if excess > 0 else None
        ec = m("1.50", "early_close") - m("1.00", "early_close")
        appears = drop >= 0.05 and all(pairs) and frac is not None and frac >= 0.70 and ec <= 0.02
        o = "degradation appears" if appears else ("no degradation" if drop <= 0.02 else "inconclusive")
        out["outcome"] = {"validity_drop": drop, "pairs_lower": pairs, "span_syntax_fraction_of_excess": frac,
                          "early_close_diff": ec, "outcome": o}
    json.dump(out, open(REPO / "outputs/T3/long_summary.json", "w"), indent=1)
    print(json.dumps(out.get("outcome"), indent=1))

if __name__ == "__main__":
    {"t3b": t3b, "long": long_arm}[sys.argv[1] if len(sys.argv) > 1 else "t3b"]()
