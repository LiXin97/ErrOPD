#!/usr/bin/env python3
"""Score Olmo 3 pairs."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
O = REPO / "outputs/OL3"
SEEDS = (1001, 1002, 1003)
ARMS = ("opd1.00", "opd1.40", "fkl1.00")
K1 = "k1_trunc_realids"
FAIL = ["k1_trunc_realids", "short_list", "duplicate_id", "hallucinated_id", "extra_items", "malformed_json",
        "extra_text", "unterminated", "other"]

def load(path):
    if not (path.exists() and path.with_name(path.name.replace(".v2.records.jsonl", ".v2.json")).exists()):
        return None
    per = collections.defaultdict(list)
    lab = collections.Counter()
    cap = 0
    for line in open(path):
        r = json.loads(line)
        if r["split"] != "val" or r["label"] == "infeasible_input":
            continue
        lab[r["label"]] += 1
        cap += r.get("finish_reason") == "length"
        per[r["row"]].append(r["label"] == K1)
    keys = sorted(per)
    n = sum(lab.values())
    return {"n": n, "share": lab[K1] / n, "keys": keys, "pp": np.array([np.mean(per[k]) for k in keys]),
            "labels": dict(lab), "label_shares": {l: lab.get(l, 0) / n for l in FAIL},
            "valid": lab.get("valid", 0) / n, "cap_share": cap / n}

def gates():
    t = load(O / "samples" / "teacher_olmo31_32b_inj20.v2.records.jsonl")
    s = load(O / "samples" / "base_olmo3_7b.v2.records.jsonl")
    if t is None or s is None:
        return {"status": "missing", "teacher": t is not None, "student": s is not None}
    other = {l: v for l, v in s["label_shares"].items() if l != K1}
    t_other = {l: v for l, v in t["label_shares"].items() if l != K1}
    g = {"teacher_n": t["n"], "teacher_k1": t["share"], "teacher_valid": t["valid"], "teacher_labels": t["labels"],
         "teacher_cap_share": t["cap_share"],
         "student_n": s["n"], "student_k1": s["share"], "student_valid": s["valid"], "student_labels": s["labels"],
         "student_cap_share": s["cap_share"],
         "GO1": 0.08 <= t["share"] <= 0.40, "GO2": s["share"] <= 0.005,
         "GO3": all(v <= 0.03 for v in other.values()), "student_largest_other": max(other.items(), key=lambda kv: kv[1]),
         "GO4a": all(v <= 0.03 for v in t_other.values()),
         "GO4b": t["valid"] >= s["valid"] - t["share"] - 0.10,
         "GO4c": t["cap_share"] <= 0.02,
         "teacher_largest_other": max(t_other.items(), key=lambda kv: kv[1]),
         "GO4b_floor": s["valid"] - t["share"] - 0.10}
    g["GO4"] = g["GO4a"] and g["GO4b"] and g["GO4c"]
    g["band_k1_0.21_0.24"] = 0.21 <= t["share"] <= 0.24
    g["all_pass"] = g["GO1"] and g["GO2"] and g["GO3"] and g["GO4"]
    g["proceed"] = g["all_pass"] and g["band_k1_0.21_0.24"]
    return g

def cell(arm, base, step):
    got = {s: load(O / f"OL3_s{s}_{arm}" / "samples" / f"step{step}.v2.records.jsonl") for s in SEEDS}
    got = {s: v for s, v in got.items() if v is not None}
    if len(got) < 2:
        return {"status": f"incomplete ({len(got)} seeds)", "seeds": list(got)}
    assert all(v["keys"] == base["keys"] for v in got.values()), "prompt sets differ"
    X = np.stack([got[s]["pp"] for s in got])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(10000):
        si, pi = rng.integers(0, len(got), len(got)), rng.integers(0, X.shape[1], X.shape[1])
        bs.append(X[np.ix_(si, pi)].mean() - base["pp"][pi].mean())
    per = {s: got[s]["share"] for s in got}
    return {"seeds": list(got), "share_per_seed": per, "Tr": float(np.mean(list(per.values())) - base["share"]),
            "Tr_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "all_above_base": all(v > base["share"] for v in per.values()), "labels": {s: got[s]["labels"] for s in got}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    a = ap.parse_args()
    if a.gates:
        print(json.dumps(gates(), indent=1, default=float))
        return
    res = {"lock": "", "gates": gates()}
    base = load(O / "samples" / "base_olmo3_7b.v2.records.jsonl")
    teacher = load(O / "samples" / "teacher_olmo31_32b_inj20.v2.records.jsonl")
    c = {(arm, st): cell(arm, base, st) for arm in ARMS for st in (14, 28, 42)}
    if not all("Tr" in c[(arm, 42)] for arm in ARMS):
        res["status"] = "DROPPED if still so at 2026-10-04 12:00 UTC (an arm has < 2 seeds at step 42)"
        res["completion"] = {arm: c[(arm, 42)].get("seeds") for arm in ARMS}
        (REPO / "results/OL3_scored.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: res[k] for k in ("status", "completion")}, indent=1))
        return
    b = base["share"]
    o1, o4, f = c[("opd1.00", 42)], c[("opd1.40", 42)], c[("fkl1.00", 42)]
    fails1 = any(x["Tr"] >= 0.10 and x["all_above_base"] for x in (o1, o4))
    res["OP1"] = "holds" if (o1["Tr"] < 0.02 and o4["Tr"] < 0.02) else ("fails" if fails1 else "inconclusive")
    res["OP2"] = ("holds" if f["Tr"] >= 0.10 and f["all_above_base"] else
                  "fails" if all(v - b < 0.02 for v in f["share_per_seed"].values()) else "inconclusive")
    common = [s for s in f["share_per_seed"] if s in o1["share_per_seed"]]
    res["OP3"] = "holds" if (f["Tr"] - o1["Tr"] >= 0.08 and len(common) >= 2 and
                             all(f["share_per_seed"][s] > o1["share_per_seed"][s] for s in common)) else "fails"
    f14 = c[("fkl1.00", 14)]
    res["OP4"] = ("holds" if f14["Tr"] >= 0.10 else "fails") if "Tr" in f14 else "not scorable (step 14 missing)"
    fmean = float(np.mean(list(f["share_per_seed"].values())))
    res["OP5"] = {"fkl_share": fmean, "teacher_share": teacher["share"], "holds": fmean <= teacher["share"] + 0.05}
    res["escapes"] = [{"arm": arm, "seed": s, "step": st, "share": v}
                      for arm in ("opd1.00", "opd1.40") for st in (14, 28, 42) if "Tr" in c[(arm, st)]
                      for s, v in c[(arm, st)]["share_per_seed"].items() if v >= 0.10]
    res["base_share"], res["teacher_share"] = b, teacher["share"]
    res["cells"] = {f"{arm}|{st}": v for (arm, st), v in c.items()}
    (REPO / "results/OL3_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("OP1", "OP2", "OP3", "OP4", "OP5", "escapes", "base_share", "teacher_share")},
                     indent=1, default=float))
    for arm in ARMS:
        x = c[(arm, 42)]
        print(arm, "Tr %.4f" % x["Tr"], x["Tr_ci95"], {s: round(v, 4) for s, v in x["share_per_seed"].items()})

if __name__ == "__main__":
    main()
