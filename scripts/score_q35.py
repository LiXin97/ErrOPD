#!/usr/bin/env python3
"""Score Qwen3.5 pairs."""
import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
Q = REPO / "outputs/Q35S"
SEEDS = (1001, 1002, 1003)

def load(path):
    if not path.exists():
        return None
    per = collections.defaultdict(list)
    lab = collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if r["split"] != "val" or r["label"] == "infeasible_input":
            continue
        lab[r["label"]] += 1
        per[r["row"]].append(r["label"] == "k1_trunc_realids")
    keys = sorted(per)
    n = sum(lab.values())
    return {"share": lab["k1_trunc_realids"] / n, "keys": keys, "pp": np.array([np.mean(per[k]) for k in keys]), "labels": dict(lab)}

def cell(run_fmt, base, step=42):
    got = {s: load(Q / run_fmt.format(s=s) / "samples" / f"step{step}.v2.records.jsonl") for s in SEEDS}
    got = {s: v for s, v in got.items() if v is not None}
    if len(got) < 2:
        return {"status": f"incomplete ({len(got)} seeds)", "seeds": list(got)}
    X = np.stack([got[s]["pp"] for s in got])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(10000):
        si, pi = rng.integers(0, len(got), len(got)), rng.integers(0, X.shape[1], X.shape[1])
        bs.append(X[np.ix_(si, pi)].mean() - base["pp"][pi].mean())
    per = {s: got[s]["share"] for s in got}
    return {"seeds": list(got), "share_per_seed": per, "Tr": float(np.mean(list(per.values())) - base["share"]),
            "Tr_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pp": X, "labels": {s: got[s]["labels"] for s in got}}

def main():
    res = {"lock": "", "sizes": {}}
    for size in ("0p8b", "2b", "4b"):
        base = load(Q / "samples" / f"base_{size}.v2.records.jsonl")
        b = base["share"]
        c = {arm: cell(f"Q35S_s{{s}}_{size}_{arm}", base) for arm in ("opd1.00", "opd1.40", "fkl1.00")}
        out = {"base_share": b}
        for arm, v in c.items():
            out[arm] = {k: v[k] for k in v if k != "pp"}
        if all("Tr" in c[a] for a in c):
            o1, o4, f = c["opd1.00"], c["opd1.40"], c["fkl1.00"]
            fails1 = any(x["Tr"] >= 0.10 and all(v > b for v in x["share_per_seed"].values()) for x in (o1, o4))
            out["QP1"] = "holds" if (o1["Tr"] < 0.02 and o4["Tr"] < 0.02) else ("fails" if fails1 else "inconclusive")
            out["QP2"] = ("holds" if f["Tr"] >= 0.10 and all(v > b for v in f["share_per_seed"].values()) else
                          "fails" if all(v - b < 0.02 for v in f["share_per_seed"].values()) else "inconclusive")
            common = [s for s in f["share_per_seed"] if s in o1["share_per_seed"]]
            out["QP3"] = "holds" if (f["Tr"] - o1["Tr"] >= 0.08 and len(common) >= 2 and
                                     all(f["share_per_seed"][s] > o1["share_per_seed"][s] for s in common)) else "fails"
        res["sizes"][size] = out
    base = load(Q / "samples" / "base_4b_inj02.v2.records.jsonl")
    e1, e4 = cell("Q35S_xE_s{s}_4binj02_opd1.00", base), cell("Q35S_xE_s{s}_4binj02_opd1.40", base)
    E = {"base_share": base["share"], "opd1.00": {k: e1[k] for k in e1 if k != "pp"}, "opd1.40": {k: e4[k] for k in e4 if k != "pp"}}
    if "Tr" in e1 and "Tr" in e4:
        common = [s for s in e1["share_per_seed"] if s in e4["share_per_seed"]]
        idx1 = [list(e1["share_per_seed"]).index(s) for s in common]
        idx4 = [list(e4["share_per_seed"]).index(s) for s in common]
        D = e4["pp"][idx4] - e1["pp"][idx1]
        rng = np.random.default_rng(0)
        bs = [D[np.ix_(rng.integers(0, len(common), len(common)), rng.integers(0, D.shape[1], D.shape[1]))].mean() for _ in range(10000)]
        Am = float(D.mean())
        E["transmission"] = ("transmitted" if e1["Tr"] >= 0.02 and e1["Tr_ci95"][0] > 0 and all(v > base["share"] for v in e1["share_per_seed"].values())
                             else "not transmitted" if e1["Tr"] < 0.02 else "inconclusive")
        E["Am"], E["Am_ci95"] = Am, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        pairs = [e4["share_per_seed"][s] - e1["share_per_seed"][s] for s in common]
        E["amplification"] = ("amplified" if Am > 0 and E["Am_ci95"][0] > 0 and sum(p > 0 for p in pairs) >= 2 else
                              "not amplified" if Am <= 0 else "inconclusive")
    res["E_arm"] = E
    (REPO / "results/Q35S_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])

if __name__ == "__main__":
    main()
