#!/usr/bin/env python3
"""Score the Llama lambda grid on the injected Fashion pair."""
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/erropd"))
from eval_labels import classify

SEEDS = [1001, 1002, 1003]
LAMS = ["1.00", "1.25", "1.40"]
PAIRS = ["A", "0"]
VAL = [json.loads(l) for l in open(REPO / "data/pl_k8/val.jsonl")]
ROOT = REPO / "outputs/D1inj"

def cell(pair, lam, seed, step=42):
    d = ROOT / f"opd/{pair}_lam{lam}_s{seed}/eval_step{step}"
    if not (d / "listwise_metrics.json").exists():
        return None
    m = json.load(open(d / "listwise_metrics.json"))
    types, n = {}, 0
    for i, l in enumerate(open(d / "predictions.jsonl")):
        p = json.loads(l)
        md = VAL[p.get("row", i)]["metadata"]
        ids = [str(x) for x in md["review_ids"]]
        if len(set(ids)) != len(ids):
            continue
        n += 1
        lab = classify(p["pred_text"], ids, md["K"], "review_id", strict_ok=p["pred_scores"] is not None)["label"]
        types[lab] = types.get(lab, 0) + 1
    return {"parse": m["parse_rate"], "ndcg1": m["ndcg@1_mean"], "n_parsed": m["n_parsed"], "useful": m["useful"],
            "share": {k: v / n for k, v in types.items()}}

def main():
    C = {(p, l, s): cell(p, l, s) for p in PAIRS for l in LAMS for s in SEEDS}
    full = lambda p, l: all(C[(p, l, s)] is not None for s in SEEDS)
    res, lines = {}, []
    for p in PAIRS:
        r = {}
        lines.append(f"\n## pair {p} (strict parse at step 42, seeds {SEEDS})")
        for l in LAMS:
            ps = [C[(p, l, s)]["parse"] if C[(p, l, s)] else None for s in SEEDS]
            k1 = [C[(p, l, s)]["share"].get("k1_trunc_realids", 0) if C[(p, l, s)] else None for s in SEEDS]
            lines.append(f"- lambda {l}: parse {ps}; one-item-short share {k1}")
        if full(p, "1.00") and full(p, "1.40"):
            m0 = st.mean(C[(p, "1.00", s)]["parse"] for s in SEEDS)
            m1 = st.mean(C[(p, "1.40", s)]["parse"] for s in SEEDS)
            r["collapse"] = {"mean_1.00": m0, "mean_1.40": m1, "drop": m0 - m1, "collapse": m0 - m1 >= 0.30}
            types = set()
            for s in SEEDS:
                types |= set(C[(p, "1.00", s)]["share"]) | set(C[(p, "1.40", s)]["share"])
            r["types"] = {}
            for t in sorted(types - {"valid"}):
                d = [C[(p, "1.40", s)]["share"].get(t, 0) - C[(p, "1.00", s)]["share"].get(t, 0) for s in SEEDS]
                md = st.mean(d)
                r["types"][t] = {"mean_diff": md, "diffs": d, "verdict": "amplified" if md >= 0.10 and all(x > 0 for x in d)
                                 else "not amplified" if md <= 0.02 else "unclear"}
        res[p] = r
    if full("A", "1.40") and full("0", "1.40"):
        res["A_minus_0_at_1.40"] = {
            "parse": [C[("A", "1.40", s)]["parse"] - C[("0", "1.40", s)]["parse"] for s in SEEDS],
            "k1_share": [C[("A", "1.40", s)]["share"].get("k1_trunc_realids", 0) - C[("0", "1.40", s)]["share"].get("k1_trunc_realids", 0) for s in SEEDS]}
    ap = ROOT / "asym/summary.json"
    if ap.exists():
        calls = json.load(open(ap))["primary_call"]
        res["M_test"] = {}
        for p in PAIRS:
            if "collapse" not in res.get(p, {}):
                continue
            pred = calls[p]["task"]
            obs = res[p]["collapse"]["collapse"]
            amp = [t for t, c in calls[p]["by_type"].items() if c == "amplification"]
            fail_inc = st.mean((1 - C[(p, "1.40", s)]["share"].get("valid", 0)) - (1 - C[(p, "1.00", s)]["share"].get("valid", 0)) for s in SEEDS)
            amp_inc = st.mean(sum(C[(p, "1.40", s)]["share"].get(t, 0) - C[(p, "1.00", s)]["share"].get(t, 0) for t in amp) for s in SEEDS)
            comp_ok = (amp_inc >= 0.70 * fail_inc) if amp else None
            if pred == "no prediction":
                verdict = "untested"
            elif pred == "collapse":
                verdict = "CONFIRMED" if (obs and comp_ok) else "REFUTED"
            else:
                verdict = "CONFIRMED" if not obs else "REFUTED"
            res["M_test"][p] = {"predicted": pred, "observed_collapse": obs, "failure_increase": fail_inc,
                                "predicted_type_increase": amp_inc, "composition_70pct": comp_ok,
                                "verdict": verdict,
                                "types": {t: {"predicted": c, "observed": res[p]["types"].get(t, {}).get("verdict", "absent")}
                                          for t, c in calls[p]["by_type"].items()}}
    res["cells"] = {f"{p}_lam{l}_s{s}": C[(p, l, s)] for p in PAIRS for l in LAMS for s in SEEDS}
    ROOT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "summary.json", "w"), indent=2, default=str)
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1, default=str))

if __name__ == "__main__":
    main()
