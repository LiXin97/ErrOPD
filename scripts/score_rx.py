#!/usr/bin/env python3
"""Score the duplicated-ID error and learning-rate runs, Qwen3."""
import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import score_core as S
from score_long import cp

SAMP = REPO / "outputs/E3/samples"
PROBE = REPO / "outputs/E3/probe/item7_clean"
OUTS = REPO / "outputs/erropd"
K1, RUN, DUP = "k1_trunc_realids", "extra_items", "duplicate_id"
NB = 10000
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
_cache = {}

def shares(name, lab):
    pass
    if name not in _cache:
        f = SAMP / name / "rollouts_train.jsonl"
        if not (f.exists() and (SAMP / name / "rollouts_train.ok").exists()):
            _cache[name] = None
        else:
            per = collections.defaultdict(list)
            for line in open(f):
                r = json.loads(line)
                if r["split"] != "val":
                    continue
                per[r["row"]].append(S.label(r["text"], [str(x) for x in S.VAL[r["row"]]["metadata"]["product_ids"]]))
            rows = sorted(per)
            assert len(rows) == len(S.VAL) and all(len(per[i]) == 4 for i in rows), (name, len(rows))
            _cache[name] = [per[i] for i in rows]
    v = _cache[name]
    return None if v is None else np.array([np.mean([x == lab for x in labs]) for labs in v])

def probe(name):
    f = PROBE / f"{name}.json"
    return json.loads(f.read_text())["mean"] if f.exists() else None

def boot_tr(X, b):
    rng = np.random.default_rng(0)
    n, P = X.shape
    out = np.empty(NB)
    for i in range(NB):
        si = rng.integers(0, n, n)
        pi = rng.integers(0, P, P)
        out[i] = X[np.ix_(si, pi)].mean() - b[pi].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def arm(names, lab, base):
    pass
    got = {s: shares(n, lab) for s, n in names.items()}
    got = {s: v for s, v in got.items() if v is not None}
    b = shares(base, lab)
    if len(got) < 3 or b is None:
        return {"status": "incomplete", "finished": sorted(got), "base_present": b is not None}
    X = np.stack([got[s] for s in sorted(got)])
    per = {s: float(got[s].mean()) for s in sorted(got)}
    tr = float(X.mean() - b.mean())
    return {"status": "scored", "seeds": sorted(got), "base": float(b.mean()), "share_per_seed": per,
            "Tr_per_seed": {s: v - float(b.mean()) for s, v in per.items()}, "Tr": tr, "Tr_ci95": boot_tr(X, b),
            "all_above_base": all(v > b.mean() for v in per.values())}

def t_interval(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return None
    h = T975[len(v) - 1] * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean() - h), float(v.mean() + h)

def perm_test(a, b):
    pass
    pool = list(a) + list(b)
    obs = np.mean(a) - np.mean(b)
    n, k = len(pool), len(a)
    cnt = tot = 0
    for idx in itertools.combinations(range(n), k):
        sa = [pool[i] for i in idx]
        sb = [pool[i] for i in range(n) if i not in idx]
        tot += 1
        cnt += (np.mean(sa) - np.mean(sb)) >= obs - 1e-12
    return {"obs": float(obs), "p": cnt / tot, "n_splits": tot}

def resumed(run_dir):
    f = run_dir / "RESUMED"
    return f.read_text().split() if f.exists() else []

def train_events(run_dir, lab, lo, hi):
    f = run_dir / "ckpt" / "erropd_rollouts.jsonl"
    if not f.exists():
        return None
    n = 0
    for line in open(f):
        r = json.loads(line)
        if lo <= int(r["step"]) <= hi and r.get("eligible", False):
            n += S.label(r["text"], [str(x) for x in r["product_ids"]]) == lab
    return n

def pairs_rule(d_fkl, d_opd):
    common = sorted(set(d_fkl) & set(d_opd))
    pos = sum(d_fkl[s] - d_opd[s] > 0 for s in common)
    need = 4 if len(common) >= 5 else max(len(common) - 1, 0)
    return common, pos, pos >= need and len(common) >= 3

def part_A():
    r = {}
    for lam in ("1.00", "1.40"):
        clean = {s: f"RXA_R_Tclean_lam{lam}_s{s}_step42" for s in range(3001, 3006)}
        armR = {**{s: f"E3_R_lam{lam}_s{s}" for s in (1001, 1002, 1003)},
                **{s: f"RXA_R_Tk1_lam{lam}_s{s}_step42" for s in (1004, 1005)}}
        c, a = arm(clean, K1, "base_R"), arm(armR, K1, "base_R")
        r[f"clean_{lam}"], r[f"armR_{lam}"] = c, a
        if c["status"] == "scored":
            alt = arm(clean, K1, "base_R_rx")
            c["Tr_vs_base_R_rx"] = alt.get("Tr")
            c["Tr_t95"] = t_interval(list(c["Tr_per_seed"].values()))
        if a["status"] == "scored":
            a["Tr_t95"] = t_interval(list(a["Tr_per_seed"].values()))
        if c["status"] == "scored" and a["status"] == "scored":
            r[f"RA3_{lam}"] = dict(perm_test(list(a["Tr_per_seed"].values()), list(c["Tr_per_seed"].values())))
            r[f"RA3_{lam}"]["holds"] = r[f"RA3_{lam}"]["p"] < 0.05
        r[f"resumed_{lam}"] = {s: resumed(OUTS / "RXA" / f"RXA_R_Tclean_lam{lam}_s{s}") for s in range(3001, 3006)}
    c1, c4 = r["clean_1.00"], r["clean_1.40"]
    if c1["status"] == "scored":
        run1 = [shares(f"RXA_R_Tclean_lam1.00_s{s}_step42", RUN) for s in c1["seeds"]]
        bR = shares("base_R", RUN)
        r["premise"] = {"base_runaway": float(bR.mean()), "lam1.00_runaway": float(np.mean([x.mean() for x in run1])),
                        "holds": float(np.mean([x.mean() for x in run1])) <= bR.mean() / 2}
        bp = probe("base_R")
        drifts = {s: probe(f"RXA_R_Tclean_lam1.00_s{s}_step42") for s in c1["seeds"]}
        drifts = {s: v - bp for s, v in drifts.items() if v is not None and bp is not None}
        r["RA2"] = {"drift_per_seed": drifts, "mean_drift": float(np.mean(list(drifts.values()))) if drifts else None,
                    "holds": bool(drifts) and float(np.mean(list(drifts.values()))) < 3.1}
    if c1["status"] == "scored" and c4["status"] == "scored":
        r["RA1"] = {"Tr_1.00": c1["Tr"], "Tr_1.40": c4["Tr"], "holds": c1["Tr"] < 0.02 and c4["Tr"] < 0.02}
        r["alternative_reading"] = any(x["Tr"] >= 0.10 and x["all_above_base"] for x in (c1, c4))
        if not r["premise"]["holds"]:
            r["combined"] = "no test (route premise fails)"
        elif r["RA1"]["holds"] and r.get("RA2", {}).get("holds") and r.get("RA3_1.00", {}).get("holds"):
            r["combined"] = "the coupled route carries the teacher's error"
        elif r["RA1"]["holds"] and not r.get("RA3_1.00", {}).get("holds", False):
            r["combined"] = "clean teacher transmits nothing; arm R not separable at the seed level; account unconfirmed"
        elif r["alternative_reading"]:
            r["combined"] = "the item-8 correction alone induces early closing; arm R's transmission is not teacher-specific"
        else:
            r["combined"] = "neither reading (RA1 or RA2 fails without the alternative pattern)"
    return r

def part_B():
    r = {"expected_escapes": {"1e-6": 0.0, "3e-6": 0.4, "1e-5": 0.5}, "rate_law_Tr42": {"1e-6": 0.017, "1e-5": 0.093}}
    bp = probe("base_clean")
    for lr in ("1e-6", "3e-6", "1e-5"):
        for obj in ("opd", "fkl"):
            run = lambda s: f"RXB_{obj}_lr{lr}_s{s}"
            res = {st: arm({s: f"{run(s)}_step{st}" for s in range(3001, 3006)}, K1, "base_clean") for st in (14, 28, 42)}
            x = {"steps": res}
            if res[42]["status"] == "scored":
                seeds = res[42]["seeds"]
                x["escaped"] = {s: any(res[st]["status"] == "scored" and res[st]["share_per_seed"].get(s, 0) >= 0.10
                                       for st in (14, 28, 42)) for s in seeds}
                x["events_1_14"] = {s: train_events(OUTS / "RXB" / run(s), K1, 1, 14) for s in seeds}
                dr = {s: {st: (probe(f"{run(s)}_step{st}") - bp) if probe(f"{run(s)}_step{st}") is not None else None
                          for st in (14, 28, 42)} for s in seeds}
                x["probe_drift"] = dr
                x["swing"] = {s: (max([0.0] + [v for v in d.values() if v is not None]) -
                                  min([0.0] + [v for v in d.values() if v is not None])) > 5 for s, d in dr.items()}
            r[f"{obj}_{lr}"] = x
        o, f = r[f"opd_{lr}"], r[f"fkl_{lr}"]
        if "escaped" in o:
            k = sum(o["escaped"].values())
            r[f"RB1_{lr}"] = {"escapes": k, "n": len(o["escaped"]), "holds": k <= 1}
        if f["steps"][42]["status"] == "scored":
            f42, f14 = f["steps"][42], f["steps"][14]
            r[f"RB3_{lr}"] = {"Tr42": f42["Tr"], "Tr14": f14.get("Tr"), "all_above_base": f42["all_above_base"],
                              "holds": f42["Tr"] >= 0.10 and f42["all_above_base"] and (f14.get("Tr") or 0) >= 0.10}
        if f["steps"][42]["status"] == "scored" and o["steps"][42]["status"] == "scored":
            common, pos, okp = pairs_rule(f["steps"][42]["share_per_seed"], o["steps"][42]["share_per_seed"])
            d = f["steps"][42]["Tr"] - o["steps"][42]["Tr"]
            r[f"RB4_{lr}"] = {"diff": d, "pairs": len(common), "pairs_positive": pos, "holds": d >= 0.08 and okp}
    if r["opd_1e-6"]["steps"][42]["status"] == "scored":
        r["RB2"] = {"Tr42": r["opd_1e-6"]["steps"][42]["Tr"], "holds": r["opd_1e-6"]["steps"][42]["Tr"] < 0.02}
    for key in ("RB1", "RB3", "RB4"):
        got = [r[f"{key}_{lr}"]["holds"] for lr in ("1e-6", "3e-6", "1e-5") if f"{key}_{lr}" in r]
        r[f"{key}_all_lr"] = (all(got) if len(got) == 3 else "incomplete")
    return r

def part_C():
    r = {}
    o1 = arm({s: f"RXC_opd_lam1.00_s{s}_step42" for s in range(3001, 3006)}, DUP, "base_clean")
    o4 = arm({s: f"RXC_opd_lam1.40_s{s}_step42" for s in range(3001, 3006)}, DUP, "base_clean")
    fk = arm({s: f"RXC_fkl_lam1.00_s{s}_step42" for s in range(3001, 3006)}, DUP, "base_clean")
    r.update({"opd_1.00": o1, "opd_1.40": o4, "fkl_1.00": fk})
    if o1["status"] == "scored" and o4["status"] == "scored":
        esc = [s for s, v in o4["share_per_seed"].items() if v >= 0.10]
        if o1["Tr"] >= 0.02:
            call = "failed"
        elif o4["Tr"] < 0.02:
            call = "holds"
        elif esc and not (o4["all_above_base"] and o4["Tr"] >= 0.10):
            call = "inconclusive (escape at 1.40)"
        else:
            call = "failed"
        r["RC1"] = {"Tr_1.00": o1["Tr"], "Tr_1.40": o4["Tr"], "escapes_1.40": esc, "call": call}
    if fk["status"] == "scored":
        r["RC2"] = {"Tr": fk["Tr"], "all_above_base": fk["all_above_base"], "holds": fk["Tr"] >= 0.10 and fk["all_above_base"]}
    if fk["status"] == "scored" and o1["status"] == "scored":
        common, pos, okp = pairs_rule(fk["share_per_seed"], o1["share_per_seed"])
        d = fk["Tr"] - o1["Tr"]
        r["RC3"] = {"diff": d, "pairs": len(common), "pairs_positive": pos, "holds": d >= 0.08 and okp}
    return r

def part_D():
    r = {}
    bp = probe("base_clean")
    for tag in ("D42", "D210"):
        names = {s: f"RXD_{tag}_s{s}_step42" for s in range(3001, 3006)}
        a = arm(names, K1, "base_clean")
        x = {"step42": a}
        if a["status"] == "scored":
            dr = {s: probe(names[s]) for s in a["seeds"]}
            dr = {s: v - bp for s, v in dr.items() if v is not None}
            md = float(np.mean(list(dr.values()))) if dr else None
            x["RD1"] = {"Tr42": a["Tr"], "probe_drift_per_seed": dr, "mean_probe_drift": md,
                        "holds": a["Tr"] < 0.02 and md is not None and abs(md) <= 2.0}
            x["resumed"] = {s: resumed(OUTS / "RXD" / f"RXD_{tag}_s{s}") for s in a["seeds"]}
            m = {}
            for s in a["seeds"]:
                f = OUTS / "RXD" / f"RXD_{tag}_s{s}" / "ckpt" / "erropd_step_metrics.jsonl"
                if f.exists():
                    rows = [json.loads(l) for l in open(f)]
                    m[s] = {k: float(np.mean([q[k] for q in rows if k in q])) for k in ("paper/rkl_full", "paper/rkl_sampled_est")
                            if any(k in q for q in rows)}
            x["train_log_means"] = m
        r[tag] = x
    steps = (42, 84, 126, 168, 210)
    fin = [s for s in range(3001, 3006) if shares(f"RXD_D210_s{s}_step210", K1) is not None]
    if len(fin) >= 3:
        sh = {s: {st: float(shares(f"RXD_D210_s{s}_step{st}", K1).mean()) for st in steps
                  if shares(f"RXD_D210_s{s}_step{st}", K1) is not None} for s in fin}
        esc = {s: next((st for st in steps if sh[s].get(st, 0) >= 0.10), None) for s in fin}
        k = sum(v is not None for v in esc.values())
        r["RD2"] = {"seeds": fin, "shares": sh, "escape_step": esc, "escapes": k, "n": len(fin), "cp95": cp(k, len(fin)),
                    "P_zero_under_3_of_8": (5 / 8) ** len(fin), "holds": k == 0,
                    "probe_drift": {s: {st: (probe(f"RXD_D210_s{s}_step{st}") - bp) if probe(f"RXD_D210_s{s}_step{st}") is not None
                                        else None for st in steps} for s in fin}}
    else:
        r["RD2"] = {"status": "incomplete", "finished": fin}
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B,C,D")
    ap.add_argument("--out", default=str(REPO / "results/RX_scored.json"))
    a = ap.parse_args()
    res = {"lock": ""}
    fn = {"A": part_A, "B": part_B, "C": part_C, "D": part_D}
    for p in a.parts.split(","):
        res[p] = fn[p]()
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for p in a.parts.split(","):
        print("== Part", p)
        print(json.dumps({k: v for k, v in res[p].items() if k.startswith(("R", "premise", "combined", "alternative"))},
                         indent=1, default=float))

if __name__ == "__main__":
    main()
