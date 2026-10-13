#!/usr/bin/env python3
"""Score the SQL LIMIT error, longer runs."""
import argparse
import datetime
import hashlib
import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import score_sql_limit_base as L
sqlf = L.sqlf

LOCK = ""
UTC = datetime.timezone.utc
EXP = "SQLNAT"
SEEDS = tuple(range(7001, 7006))
STEPS = (42, 84, 126, 168, 210)
ARMS = ("opd", "fkl")
P_T_LOCKED = (44, 340)
SQLLIM_GATES = REPO / "results/SQLLIM_gates.json"
SCORED_OUT = REPO / "results/SQLNAT_scored.json"

def kind(exp):
    return dict(L.KINDS["lim2"],
                run=lambda arm, s: REPO / f"outputs/{exp}/{exp}_t2_lim2_{arm}_lam1.00_s{s}",
                probe=REPO / f"outputs/{exp}/probe")

def ok(x):
    return x is not None and x.get("status") == "scored"

def lock_state(lock):
    lp = REPO / lock
    if not lp.exists():
        return {"lock": lock, "exists": False, "sha12": None, "stamped": False}
    cur = hashlib.sha256(lp.read_bytes()).hexdigest()
    man, stamped = REPO / "records/MANIFEST.tsv", False
    if man.exists():
        for line in open(man):
            col = line.rstrip("\n").split("\t")
            if len(col) >= 4 and col[1] == cur and col[3] == lock and (REPO / (col[0] + ".ots")).exists():
                stamped = True
    return {"lock": lock, "exists": True, "sha12": cur[:12], "stamped": stamped}

def reference_gates():
    pass
    if not SQLLIM_GATES.exists():
        return None, {"file": str(SQLLIM_GATES.relative_to(REPO)), "missing": True}
    g = json.loads(SQLLIM_GATES.read_text())
    pt = g.get("P_T_eligible")
    info = {"file": str(SQLLIM_GATES.relative_to(REPO)), "sha12": hashlib.sha256(SQLLIM_GATES.read_bytes()).hexdigest()[:12],
            "all_pass": g.get("all_pass"), "P_T_eligible": pt, "lock_sha12": g.get("lock_sha12"),
            "equals_amendment_2_constant": pt == list(P_T_LOCKED)}
    return (Fr(*pt) if pt else None), info

def seed_dirs(K, arm, st, seeds, block="eval"):
    recs = {s: L.load(K, K["run"](arm, s) / f"samples/step{st}", block) for s in seeds}
    missing = [s for s, v in recs.items() if v is None]
    refused = {s: v["refused"] for s, v in recs.items() if v is not None and v.get("refused")}
    return recs, missing, refused

def cell(K, arm, st, seeds, base):
    recs, missing, refused = seed_dirs(K, arm, st, seeds)
    c = sqlf.cell(recs, base)
    c["seeds_missing"] = missing
    if not ok(c):
        c["count_per_seed_partial"] = {s: v["k"] for s, v in recs.items() if L.ok_rec(v)}
        c["n_partial"] = {s: v["n"] for s, v in recs.items() if L.ok_rec(v)}
    return c

def mean_count(c):
    return float(c["_mean"] * c["n"]) if ok(c) else None

def probe_step(K, run_ids, st):
    d = Path(K["probe"])
    sp = d / "student.json"
    if not sp.exists():
        return None
    base = json.loads(sp.read_text())["mean"]
    per = {}
    for rid in run_ids:
        f = d / f"{rid}_step{st}.json"
        per[rid] = (json.loads(f.read_text())["mean"] - base) if f.exists() else None
    v = [x for x in per.values() if x is not None]
    return {"per_run": per, "n": len(v), "mean": (sum(v) / len(v)) if v else None,
            "done_marker": (d / f"DONE_step{st}").exists()}

def per_checkpoint(c, PT, st, seeds):
    pass
    op, fk = c.get(("opd", st)), c.get(("fkl", st))
    half = Fr(1, 2) * PT
    row = {"step": st}
    for arm, x in (("opd", op), ("fkl", fk)):
        row[arm] = ({"count_per_seed": x["count_per_seed"], "seed_mean_count": mean_count(x), "n": x["n"],
                     "Tr": x["Tr"], "Tr_ci95": x["Tr_ci95"],
                     "Tr_ci95_as_counts": [None if v is None else v * x["n"] for v in x["Tr_ci95"]],
                     "pooled": x["pooled"], "all_above_base": x["all_above_base"]}
                    if ok(x) else {"status": None if x is None else x["status"],
                                   "count_per_seed_partial": (x or {}).get("count_per_seed_partial"),
                                   "seeds_missing": (x or {}).get("seeds_missing")})
    if ok(fk):
        row["PC_seed_mean_clears"] = fk["_Tr"] >= half
        row["PC_full_clause"] = fk["_Tr"] >= half and fk["all_above_base"]
    if ok(op):
        row["LM1_clause_Tr_below_0.02"] = op["_Tr"] < Fr(2, 100)
        row["opd_escaped_seeds"] = [s for s in op["seeds"] if Fr(op["count_per_seed"][s], op["n"]) >= half]
    if ok(op) and ok(fk):
        common = sorted(set(fk["seeds"]) & set(op["seeds"]))
        pos = sum(fk["count_per_seed"][s] > op["count_per_seed"][s] for s in common)
        need = 4 if len(common) >= 5 else max(len(common) - 1, 0)
        row["LM3_clause"] = {"diff_count": float((fk["_Tr"] - op["_Tr"]) * fk["n"]), "pairs": len(common),
                             "pairs_positive": pos,
                             "holds": fk["_Tr"] - op["_Tr"] >= Fr(2, 5) * PT and pos >= need and len(common) >= 3}
    return row

def escapes(c, PT, steps, seeds, pc_step):
    pass
    half = Fr(1, 2) * PT
    out = {}
    for s in seeds:
        hist = {}
        for st in steps:
            x = c.get(("opd", st))
            if ok(x) and s in x["count_per_seed"]:
                hist[st] = x["count_per_seed"][s]
        n = next((c[("opd", st)]["n"] for st in steps if ok(c.get(("opd", st)))), None)
        first = next((st for st in steps if st in hist and Fr(hist[st], n) >= half), None)
        out[s] = {"counts_by_step": hist, "first_escape_step": first,
                  "reversed_at_primary_step": (first is not None and pc_step in hist and Fr(hist[pc_step], n) < half)}
    k = sum(v["first_escape_step"] is not None for v in out.values())
    return {"definition": "SQLLIM lock 7: an OPD seed escapes at a checkpoint where its eligible eval count / 340 >= "
                          "0.50 P_T (Amendment 2: >= 22 of 340); first-escape step = the first such kept checkpoint "
                          "(LONG3 convention); reversed = escaped and < 22 at the primary step",
            "line_count_of_340": float(half * 340), "per_seed": out, "K": k,
            "K_sticky": k - sum(v["reversed_at_primary_step"] for v in out.values()),
            "seeds_with_any_scored_checkpoint": sum(bool(v["counts_by_step"]) for v in out.values())}

def score(exp, seeds, steps, pc_step):
    K = kind(exp)
    res = {"lock": LOCK, "lock_state": lock_state(LOCK), "exp": exp, "seeds": list(seeds), "steps": list(steps),
           "primary_step": pc_step, "scored_utc": datetime.datetime.now(UTC).strftime("%FT%TZ"),
           "estimand": "hits / eligible eval samples (85 prompts x 4 = 340 per checkpoint); Tr = seed-mean share minus "
                       "the clean student's (0/340, SQLF gate pass)"}
    PT, ginfo = reference_gates()
    res["reference_gates"] = ginfo
    base, teacher = L.load(K, K["ref"]["student"]), L.load(K, K["ref"]["teacher"])
    missing = []
    if not L.ok_rec(base):
        missing.append("clean-student gate records " + str(K["ref"]["student"].relative_to(REPO)))
    if L.ok_rec(teacher):
        res["teacher_recount"] = {"hits": teacher["k"], "n": teacher["n"],
                                  "equals_gate_file": PT is not None and Fr(teacher["k"], teacher["n"]) == PT}
    if PT is None:
        missing.append("P_T (results/SQLLIM_gates.json)")
    if missing:
        res["status"] = "reference missing: " + "; ".join(missing)
        return res
    assert PT == Fr(*P_T_LOCKED), f"P_T {PT} != Amendment 2's 44/340"
    half = Fr(1, 2) * PT
    res["base"], res["P_T"], res["K_T"] = L.shares(base), float(PT), float(PT * base["n"])
    res["bars_as_counts_of_340"] = {
        "LM1_opd": "seed-mean count < 6.8 (Tr < 0.02); pooled <= 33 of 1,700 (5 seeds), <= 27 (4), <= 20 (3)",
        "PC_LM2_fkl": f"seed-mean count >= {float(half * base['n']):g} (0.5 K_T) and every seed >= 1",
        "LM3_contrast": f"seed-mean difference >= {float(Fr(2, 5) * PT * base['n']):g} (0.4 K_T), forward KL above OPD "
                        "in >= 4 of 5 seed pairs",
        "LM4_probe": f"OPD seed-mean probe shift < +{L.LM4_BAR} nats at the primary step, called only if LM1 holds",
        "escape_line": f"{float(half * base['n']):g}", "ceiling_reported": f"{float((PT + Fr(5, 100)) * base['n']):g}"}

    c = {(a, st): cell(K, a, st, seeds, base) for a in ARMS for st in steps}
    res["cells"] = {f"{a}|{st}": sqlf.strip_private(v) for (a, st), v in c.items()}
    res["per_checkpoint"] = [per_checkpoint(c, PT, st, seeds) for st in steps]
    run_ids = {a: [K["run"](a, s).name for s in seeds] for a in ARMS}
    probes = {st: {a: probe_step(K, run_ids[a], st) for a in ARMS} for st in steps}
    tf = Path(K["probe"]) / "teacher_lim2.json"
    sf = Path(K["probe"]) / "student.json"
    res["probe"] = {"by_step": probes, "teacher_lim2_shift": (json.loads(tf.read_text())["mean"] - json.loads(
        sf.read_text())["mean"]) if tf.exists() and sf.exists() else None}

    fk_steps = [st for st in steps if ok(c.get(("fkl", st)))]
    res["PC"] = {"definition": "SQLLIM LM2 at the primary step: forward-KL Tr >= 0.5 P_T (seed-mean count >= "
                               f"{float(half * base['n']):g} of {base['n']}) and every scored seed >= 1 hit",
                 "bar_seed_mean_count": float(half * base["n"]),
                 "first_step_seed_mean_clears": next((st for st in fk_steps if c[("fkl", st)]["_Tr"] >= half), None),
                 "first_step_full_clause": next((st for st in fk_steps if c[("fkl", st)]["_Tr"] >= half
                                                 and c[("fkl", st)]["all_above_base"]), None),
                 "fkl_seed_mean_count_by_step": {st: mean_count(c[("fkl", st)]) for st in fk_steps}}
    res["escape"] = escapes(c, PT, steps, seeds, pc_step)

    done = ok(c.get(("opd", pc_step))) and ok(c.get(("fkl", pc_step)))
    res["completion"] = {"every_arm_3_seeds_at_primary_step": done,
                         "last_record_utc": {a: c[(a, pc_step)].get("last_record_utc") for a in ARMS},
                         "seeds_missing_at_primary_step": {a: c[(a, pc_step)].get("seeds_missing") for a in ARMS},
                         "probe_done_markers": {st: (Path(K["probe"]) / f"DONE_step{st}").exists() for st in steps}}
    if not ginfo.get("all_pass"):
        res["status"] = "a SQLLIM gate fails or the gate file is incomplete: no test"
    elif not done:
        res["status"] = (f"incomplete at step {pc_step} (an arm has < 3 finished seeds); no rule called; "
                         "per-checkpoint rows are descriptive")
    else:
        rc = {("opd", 42): c[("opd", pc_step)], ("fkl", 42): c[("fkl", pc_step)]}
        r = L.rules(rc, PT, base["n"], {a: probes[pc_step][a] for a in ARMS})
        for k in ("escapes_reported", "above_ceiling_reported", "fkl_Tr_by_step_reported", "opd_hits_with_text"):
            r.pop(k, None)
        r["note"] = (f"calls computed by score_sql_limit_base.rules on the step-{pc_step} cells (its internal key is "
                     f"step 42); LM4 reads the step-{pc_step} probe")
        res["status"] = "scored"
        res.update(r)
        res["PC"]["call"] = r["LM2"]["call"]
        if r["LM2"]["call"] != "holds":
            res["reading"] = (f"positive control fails: on the natural pool forward KL does not reach the bar in "
                              f"{pc_step} steps; uninformative (LM1, LM3, LM4 carry no weight)")
        res["NN"] = nn_rules(c, PT, base, probes, steps, pc_step)
        res["reading"] = res["NN"]["reading"]
    res["above_ceiling_reported"] = [{"arm": a, "seed": s, "step": st, "count": c[(a, st)]["count_per_seed"][s]}
                                     for a in ARMS for st in steps if ok(c.get((a, st))) for s in c[(a, st)]["seeds"]
                                     if Fr(c[(a, st)]["count_per_seed"][s], c[(a, st)]["n"]) > PT + Fr(5, 100)]
    res["opd_hits_with_text"] = {st: c[("opd", st)]["hit_texts"] for st in steps if ok(c.get(("opd", st)))
                                 and any(c[("opd", st)]["hit_texts"].values())}
    sec = {}
    for a in ARMS:
        for s in seeds:
            rd = K["run"](a, s)
            if rd.exists():
                sec[rd.name] = {"greedy": {st: L.greedy(K, rd / f"eval/step{st}") for st in steps},
                                "train_rollouts_14step_windows": L.train_rollouts(K, rd),
                                "train_block": {st: (lambda x: L.shares(x) if L.ok_rec(x) else None)(
                                    L.load(K, rd / f"samples/step{st}", "train")) for st in steps}}
    res["secondary"] = sec
    res["run_dirs_missing"] = [K["run"](a, s).name for a in ARMS for s in seeds if not K["run"](a, s).exists()]
    return res

def nn_rules(c, PT, base, probes, steps, last):
    pass
    half = Fr(1, 2) * PT
    kstar = next((st for st in steps if ok(c.get(("fkl", st))) and ok(c.get(("opd", st)))
                  and c[("fkl", st)]["_Tr"] >= half and c[("fkl", st)]["all_above_base"]), None)
    out = {"definition": "k* = first of 42/84/126/168/210 where forward-KL seed-mean count >= 0.5 K_T and every scored "
                         "seed >= 1 hit (NN2); NN1/NN3/NN4 = SQLLIM LM1/LM3/LM4 on the k* cells and k* probe; "
                         f"NN5 = OPD Tr < 0.02 at step {last}, called only if NN1 holds", "k_star": kstar}
    if kstar is None:
        out.update(NN2="fails", reading="NN2 fails: uninformative on the natural pool; forward KL does not reach half "
                                        f"the teacher's rate by step {last}")
        return out
    rk = L.rules({("opd", 42): c[("opd", kstar)], ("fkl", 42): c[("fkl", kstar)]}, PT, base["n"],
                 {a: probes[kstar][a] for a in ARMS})
    nn1, nn3, nn4 = rk["LM1"]["call"], rk["LM3"]["call"], rk["LM4"]["call"]
    nn5 = None
    if nn1 == "holds":
        nn5 = "holds" if c[("opd", last)]["_Tr"] < Fr(2, 100) else "fails"
    out.update(NN2="holds", NN1=nn1, NN3=nn3, NN4=nn4, NN5=nn5 if nn5 else "not called (NN1 fails)",
               rules_at_k_star=json.loads(json.dumps(rk, default=str)))
    if nn1 != "holds":
        rd = "NN1 fails: OPD transmits the content error by the time forward KL does (counterexample on the natural pool)"
    elif nn3 != "holds":
        rd = "NN1, NN2 hold, NN3 fails: recorded miss; no contrast claim at the pre-registered size"
    elif nn5 == "holds":
        rd = (f"NN1, NN2, NN3, NN5 hold: on the natural input distribution forward KL transmits the content error and "
              f"OPD does not, through {last} steps")
    else:
        rd = (f"NN1, NN2, NN3 hold, NN5 fails: delayed, not prevented (OPD below the bar at k*={kstar}, at or above it "
              f"by step {last})")
    if nn1 == "holds" and nn4 == "fails":
        rd += f"; NN4 fails: OPD moved the decision by >= {L.LM4_BAR} nats by k* without transmitting"
    out["reading"] = rd
    return out

def compare(res, ref_path, st):
    pass
    ref = json.loads(Path(ref_path).read_text())
    diffs = []
    for a in ARMS:
        x, y = res["cells"].get(f"{a}|{st}", {}), ref["cells"].get(f"{a}|{st}", {})
        for k in ("count_per_seed", "Tr", "Tr_ci95", "pooled", "all_above_base"):
            if json.loads(json.dumps(x.get(k), default=str)) != y.get(k):
                diffs.append(f"{a}|{st} {k}: {x.get(k)} != {y.get(k)}")
    for k in ("LM1", "LM2", "LM3", "LM4"):
        if (res.get(k) or {}).get("call") != (ref.get(k) or {}).get("call"):
            diffs.append(f"{k} call: {(res.get(k) or {}).get('call')} != {(ref.get(k) or {}).get('call')}")
    pr = ref.get("probes_step42") or {}
    for a in ARMS:
        mine = (res["probe"]["by_step"].get(st) or {}).get(a) or {}
        if pr.get(a) and mine.get("mean") is not None and abs(mine["mean"] - pr[a]["mean"]) > 1e-12:
            diffs.append(f"probe {a}: {mine['mean']} != {pr[a]['mean']}")
    return {"reference": str(ref_path), "step": st, "identical": not diffs, "diffs": diffs}

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", default=EXP, help="experiment dir/run-id prefix (default SQLNAT; SQLLIM for the self-test)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    ap.add_argument("--pc-step", type=int, default=0, help="primary step for the calls (default: the last of --steps)")
    ap.add_argument("--out", default="", help="output json (default results/SQLNAT_scored.json; required and "
                                               "outside erropd/results for any non-default --exp/--seeds/--steps)")
    ap.add_argument("--compare", default="", help="a *_scored.json from score_sql_limit_base.py; checks the primary "
                                                  "step's cells, calls and probe against it")
    a = ap.parse_args()
    steps, pc = tuple(a.steps), a.pc_step or a.steps[-1]
    assert pc in steps, "--pc-step must be one of --steps"
    default = a.exp == EXP and tuple(a.seeds) == SEEDS and steps == STEPS and pc == STEPS[-1]
    out = Path(a.out).resolve() if a.out else SCORED_OUT
    if not default:
        assert a.out, "a non-default --exp/--seeds/--steps run is a dry run: --out <scratch path> is required"
        assert not out.is_relative_to(REPO / "erropd/results"), "dry runs never write to erropd/results"
    res = score(a.exp, tuple(a.seeds), steps, pc)
    if a.compare:
        res["self_test"] = compare(res, a.compare, pc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("status", "P_T", "K_T", "bars_as_counts_of_340", "PC", "LM2", "LM1", "LM3",
                                          "LM4", "reading", "completion", "self_test") if k in res}, indent=1, default=str))
    print("per checkpoint (seed-mean counts of 340; OPD | forward KL):")
    for row in res.get("per_checkpoint", []):
        print(f"  step {row['step']:>3}: OPD {row['opd'].get('seed_mean_count')} {row['opd'].get('count_per_seed')} | "
              f"FKL {row['fkl'].get('seed_mean_count')} {row['fkl'].get('count_per_seed')}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
