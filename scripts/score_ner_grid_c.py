#!/usr/bin/env python3
"""Score the entity-extraction contrast runs."""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = [2001, 2002, 2003, 2004, 2005]
DESC_SEEDS = [2001, 2002, 2003]
NEED = 4
VALIDITY = {"extra_prose", "malformed", "schema", "non_verbatim", "unterminated", "truncated"}
NBOOT = 10000

def load_x(root, lam, seed, step):
    p = Path(root) / f"lam{lam}_s{seed}" / "eval" / f"step{step}" / "predictions.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    rows.sort(key=lambda r: r["row"])
    return np.array([1 if r.get("first_violation") in VALIDITY else 0 for r in rows], dtype=float)

def score(X0, X1, nboot=NBOOT, seed=0, need=NEED):
    pass
    d = X0 - X1
    D = float(d.mean())
    rng = np.random.default_rng(seed)
    S, N = d.shape
    boots = np.empty(nboot)
    for b in range(nboot):
        si = rng.integers(0, S, S)
        pi = rng.integers(0, N, N)
        boots[b] = d[np.ix_(si, pi)].mean()
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    per_seed = [float(X0[s].mean() - X1[s].mean()) for s in range(S)]
    n_dir = sum(1 for v in per_seed if v > 0)
    outcome = "confirmed" if (lo > 0 and n_dir >= need) else ("refuted" if D <= 0 else "inconclusive")
    return {"D": D, "ci95": [lo, hi], "per_seed_diff": per_seed, "seed_pairs_in_direction": n_dir,
            "inv_1.00": float(X0.mean()), "inv_1.40": float(X1.mean()), "outcome": outcome}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "outputs/T3c/opd"))
    ap.add_argument("--step", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="synthetic data only")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(1)
        X0 = (rng.random((3, 438)) < 0.12).astype(float)
        cases = {"null (independent 0.12 vs 0.12)": (rng.random((3, 438)) < 0.12).astype(float),
                 "paired removal of 25% of errors": X0 * (rng.random((3, 438)) >= 0.25),
                 "paired added errors at 0.02": np.maximum(X0, (rng.random((3, 438)) < 0.02).astype(float))}
        for name, X1 in cases.items():
            print(name, json.dumps(score(X0, X1, nboot=2000, need=2)))
        return
    X0 = [load_x(a.root, "1.00", s, a.step) for s in SEEDS]
    X1 = [load_x(a.root, "1.40", s, a.step) for s in SEEDS]
    if any(x is None for x in X0 + X1):
        print("incomplete: need lam 1.00 and 1.40 x seeds", SEEDS, "at step", a.step)
        return
    res = score(np.stack(X0), np.stack(X1))
    desc = {}
    for lam in ("1.00", "1.25", "1.40", "1.50"):
        for step in (7, 14, 28, 42):
            ss = SEEDS if lam in ("1.00", "1.40") else DESC_SEEDS
            ms = [Path(a.root) / f"lam{lam}_s{s}" / "eval" / f"step{step}" / "t3_metrics.json" for s in ss]
            ms = [json.load(open(m)) for m in ms if m.exists()]
            if not ms:
                continue
            sh = lambda k: float(np.mean([m.get("first_violation_share", {}).get(k, 0.0) for m in ms]))
            desc[f"{lam}_step{step}"] = {"n": len(ms), "invalid": float(np.mean([1 - m["parse_rate"] for m in ms])),
                                         "schema": sh("schema"), "non_verbatim": sh("non_verbatim"),
                                         "early_close": sh("early_close"),
                                         "recall_valid": float(np.mean([m["recall_valid"] for m in ms])),
                                         "f1_valid": float(np.mean([m["f1_valid"] for m in ms])),
                                         "useful_f1": float(np.mean([m["useful_f1"] for m in ms]))}
    out = {"primary": res, "descriptive": desc}
    Path(REPO / "outputs/T3c").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REPO / "outputs/T3c/T3c_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
