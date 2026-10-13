#!/usr/bin/env python3
"""Score runs with teachers at different error rates."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import score_core as S

PROBE = REPO / "outputs/E3/probe/item7_clean"
NOMINAL = {"f50": 0.5, "f80": 0.8}
SEEDS = (1001, 1002, 1003)
STEPS = (14, 28, 42)
GATES = REPO / "results/QSWEEP_gates.json"

def logit(x):
    return math.log(x / (1 - x))

def probe_mean(name):
    f = PROBE / f"{name}.json"
    return json.loads(f.read_text())["mean"] if f.exists() else None

def gates(a):
    teachers = {"f50": "teacher"} if a.dry_run else {k: f"teacher_k1_{k}" for k in NOMINAL}
    base = S.sample_shares("base_clean")
    bp = probe_mean("base_clean")
    out = {"lock": "", "dry_run": a.dry_run,
           "G_student": {"base_share": float(base["k1"].mean()), "pass": float(base["k1"].mean()) <= 0.002}, "teachers": {}}
    for k, name in teachers.items():
        sh = S.sample_shares(name)
        if sh is None:
            out["teachers"][k] = {"sampler": name, "status": "missing samples"}
            continue
        labs = sh["labels"]
        n = sum(labs.values())
        pt = labs.get(S.K1, 0) / n
        valid = (labs.get("valid", 0) + labs.get(S.K1, 0)) / n
        tp = probe_mean(name)
        g = None if tp is None or bp is None else tp - bp
        rate_ok = 0.30 <= pt <= 0.92
        out["teachers"][k] = {
            "sampler": name, "n": n, "labels": labs, "P_T": pt, "nominal": NOMINAL[k],
            "G_rate_usable": rate_ok, "on_nominal": abs(pt - NOMINAL[k]) <= 0.15, "G_valid_share": valid, "G_valid": valid >= 0.95,
            "probe_teacher": tp, "g_q": g, "g_q_account": 11.56 + logit(pt) - logit(0.194) if 0 < pt < 1 else None,
            "usable": rate_ok and valid >= 0.95 and out["G_student"]["pass"],
            "bars": {"Q-FKL_Tr42_ge": 0.5 * pt, "Q-cap_share42_le": pt + 0.05,
                     "Q-probe_OPD_le": min(3.0, 0.25 * g) if g is not None else None,
                     "Q-probe_FKL_ge": 0.75 * g if g is not None else None}}
    u = {k: v["P_T"] for k, v in out["teachers"].items() if v.get("usable")}
    out["Q-con_scored"] = ("f50" in u and "f80" in u and u["f80"] - u["f50"] >= 0.10)
    return out

def arm(prefix, exp_dir, seeds):
    base = S.sample_shares("base_clean")["k1"]
    bp = probe_mean("base_clean")
    seeds = [s for s in seeds if S.sample_shares(f"{prefix}{s}_step42") is not None]
    if len(seeds) < 2:
        return {"seeds": seeds, "status": "incomplete"}
    out = {"seeds": seeds}
    for st in STEPS:
        sh = [S.sample_shares(f"{prefix}{s}_step{st}") for s in seeds]
        if any(x is None for x in sh):
            out[st] = None
            continue
        X = np.stack([x["k1"] for x in sh])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(10000):
            si, pi = rng.integers(0, len(seeds), len(seeds)), rng.integers(0, X.shape[1], X.shape[1])
            bs.append(X[np.ix_(si, pi)].mean() - base[pi].mean())
        per = [float(x["k1"].mean()) for x in sh]
        pr = [probe_mean(f"{prefix}{s}_step{st}") for s in seeds]
        pr = [p - bp for p in pr if p is not None]
        out[st] = {"share_per_seed": per, "Tr": float(np.mean(per) - base.mean()),
                   "Tr_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   "probe_shift": float(np.mean(pr)) if len(pr) == len(seeds) else None, "probe_per_seed": pr,
                   "greedy_parse": [S.greedy(REPO / exp_dir / f"{prefix}{s}", st) for s in seeds]}
    tr = [S.train_share(REPO / exp_dir / f"{prefix}{s}") for s in seeds]
    out["train_share_per_seed"] = [t["k1"] if t else None for t in tr]
    att = {}
    for s in seeds:
        run = f"{prefix}{s}"
        for d in ("done", "failed", "running"):
            for f in (REPO / "erropd/queue" / d).glob(f"*-{run}.json"):
                if f.name.split("-", 1)[1] == f"{run}.json":
                    att[run] = json.loads(f.read_text()).get("_attempts")
    out["attempts"] = att
    out["base_share"] = float(base.mean())
    return out

def main_score(a):
    if a.dry_run:
        G = gates(a)
        G["teachers"]["f50"]["usable"] = True
        grid = {"f50": {"opd": ("OPDre_A_s", "outputs/OPDre"), "fkl": ("FKL_A_s", "outputs/FKL")}}
    else:
        G = json.loads(GATES.read_text())
        grid = {k: {o: (f"QSWEEP_{o}_{k}_s", "outputs/QSWEEP") for o in ("opd", "fkl")} for k in NOMINAL}
    ref = json.loads((REPO / "results/FKL_scored.json").read_text())
    D_ref = ref["FKL"]["42"]["Tr"] - ref["OPDre"]["42"]["Tr"]
    res = {"lock": "", "dry_run": a.dry_run, "D_ref": D_ref,
           "P_T_ref": ref["teacher_share"], "q": {}}
    for k, arms in grid.items():
        t = G["teachers"].get(k, {})
        if not t.get("usable"):
            res["q"][k] = {"status": "not usable / not run", "gate": t}
            continue
        pt, g = t["P_T"], t.get("g_q")
        O, F = arm(*arms["opd"], SEEDS), arm(*arms["fkl"], SEEDS)
        r = {"P_T": pt, "g_q": g, "opd": O, "fkl": F}
        if O.get(42) and F.get(42):
            o, f = O[42], F[42]
            b = O["base_share"]
            r["opd_ok"] = o["Tr"] < 0.02 and all(x < 0.10 for x in o["share_per_seed"])
            r["opd_seeds_ge_0.10"] = sum(x >= 0.10 for x in o["share_per_seed"])
            r["Q-FKL_ok"] = f["Tr"] >= 0.5 * pt and all(x > b for x in f["share_per_seed"])
            r["Q-FKL_killed"] = all(x < b + 0.02 for x in f["share_per_seed"])
            r["D"] = f["Tr"] - o["Tr"]
            r["D_pairs_positive"] = all(x > y for x, y in zip(f["share_per_seed"], o["share_per_seed"]))
            r["Q-cap"] = float(np.mean(f["share_per_seed"])) <= pt + 0.05
            r["Q-probe_OPD"] = (o["probe_shift"] <= min(3.0, 0.25 * g)) if (g is not None and o["probe_shift"] is not None) else "not scored"
            r["Q-probe_FKL"] = (f["probe_shift"] >= 0.75 * g) if (g is not None and f["probe_shift"] is not None) else "not scored"
            for z0, nm in ((probe_mean("base_clean"), "z0_probe"), (logit(1e-4), "p0_1e-4"), (logit(1e-3), "p0_1e-3")):
                r[f"opd_drift_ratio_{nm}"] = (logit(pt) - z0) / (logit(0.194) - z0)
            r["fkl_drive_ratio"] = pt / 0.194
        else:
            r["status"] = "incomplete"
        res["q"][k] = r
    done = {k: v for k, v in res["q"].items() if "D" in v}
    if done:
        res["PRIMARY_Q-OPD"] = ("holds" if all(v["opd_ok"] for v in done.values()) else
                                "falsifier fires" if done[max(done, key=lambda k: done[k]["P_T"])]["opd_seeds_ge_0.10"] >= 2 else "inconclusive")
        res["CO-PRIMARY_Q-FKL"] = ("holds" if all(v["Q-FKL_ok"] for v in done.values()) else
                                   "kill" if any(v["Q-FKL_killed"] for v in done.values()) else "inconclusive")
        pts = sorted([(res["P_T_ref"], D_ref, True)] + [(v["P_T"], v["D"], v["D_pairs_positive"]) for v in done.values()])
        inc = all(pts[i + 1][1] > pts[i][1] for i in range(len(pts) - 1)) and all(p[2] for p in pts)
        res["Q-con"] = inc if (G.get("Q-con_scored") and len(done) == 2) else f"reported only: {inc}"
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.dry_run:
        assert a.out is None or "erropd/results" not in str(Path(a.out).resolve()), "dry run must not write erropd/results"
        out = Path(a.out) if a.out else None
    elif a.gates:
        out = Path(a.out) if a.out else GATES
        if out.exists() and not a.force:
            sys.exit(f"{out} exists; the gate record is written once (use --force only on the lead's instruction)")
    else:
        out = Path(a.out) if a.out else REPO / "results/QSWEEP_scored.json"
    res = gates(a) if a.gates else main_score(a)
    txt = json.dumps(res, indent=1, default=float)
    if out is not None:
        out.write_text(txt)
    show = res if a.gates else {k: v for k, v in res.items() if k != "q"}
    print(json.dumps(show, indent=1, default=float)[:4000])
    if not a.gates:
        for k, v in res["q"].items():
            print(k, {x: y for x, y in v.items() if x not in ("opd", "fkl", "gate")})
            for o in ("opd", "fkl"):
                if isinstance(v.get(o), dict) and v[o].get(42):
                    for st in STEPS:
                        w = v[o][st]
                        print("  ", o, st, [round(x, 4) for x in w["share_per_seed"]], "Tr %.4f" % w["Tr"], [round(x, 4) for x in w["Tr_ci95"]],
                              "probe", None if w["probe_shift"] is None else round(w["probe_shift"], 2), "greedy", w["greedy_parse"])

if __name__ == "__main__":
    main()
