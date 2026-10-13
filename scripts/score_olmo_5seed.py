#!/usr/bin/env python3
"""Score the added Olmo 3 seeds."""
import argparse
import collections
import datetime
import itertools
import json
import math
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "outputs/OL3"
NEW = REPO / "outputs/OL3x"
OLD_SEEDS = (1001, 1002, 1003)
NEW_SEEDS = (1004, 1005)
ARMS = ("opd1.00", "opd1.40", "fkl1.00")
STEPS = (14, 28, 42)
K1, SHORT = "k1_trunc_realids", "short_list"
BAND = (10, 80)
SHORT28_MAX = 65
N_EXPECTED = 808
DEADLINE = datetime.datetime(2026, 10, 3, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

def load(path):
    path = Path(path)
    summ = path.with_name(path.name.replace(".v2.records.jsonl", ".v2.json"))
    if not (path.exists() and summ.exists()):
        return None
    per = collections.defaultdict(list)
    lab = collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if r["split"] != "val" or r["label"] == "infeasible_input":
            continue
        lab[r["label"]] += 1
        per[r["row"]].append(r["label"] == K1)
    keys = sorted(per)
    return {"k": lab[K1], "n": sum(lab.values()), "short": lab.get(SHORT, 0), "keys": keys,
            "pp": np.array([np.mean(per[i]) for i in keys]), "labels": dict(lab), "mtime": summ.stat().st_mtime}

def run_dir(seed, arm):
    root, exp = (OLD, "OL3") if seed in OLD_SEEDS else (NEW, "OL3x")
    return root / f"{exp}_s{seed}_{arm}"

def run_path(seed, arm, step):
    return run_dir(seed, arm) / "samples" / f"step{step}.v2.records.jsonl"

def greedy(seed, arm, step):
    pass
    d = run_dir(seed, arm)
    for f in (d / "eval" / f"step{step}" / "listwise_metrics.json", d / f"eval_step{step}" / "listwise_metrics.json"):
        if f.exists():
            m = json.loads(f.read_text())
            return {"n_parsed": m.get("n_parsed"), "n_total": m.get("n_total"), "parse_rate": m.get("parse_rate")}
    return None

def seed_rules(rec, base, teacher):
    pass
    b, nb, t, nt = base["k"], base["n"], teacher["k"], teacher["n"]
    o1, o4, fk = rec[("opd1.00", 42)], rec[("opd1.40", 42)], rec[("fkl1.00", 42)]
    out = {"counts42": {a: rec[(a, 42)]["k"] for a in ARMS}, "n": fk["n"]}
    below = lambda x: Fr(x["k"], x["n"]) - Fr(b, nb) < Fr(2, 100)
    out["OX1"] = below(o1) and below(o4)
    out["OX2"] = Fr(fk["k"], fk["n"]) - Fr(b, nb) >= Fr(10, 100)
    out["OX3"] = Fr(fk["k"], fk["n"]) - Fr(o1["k"], o1["n"]) >= Fr(8, 100)
    out["OX5"] = Fr(fk["k"], fk["n"]) <= Fr(t, nt) + Fr(5, 100)
    f14, f28 = rec.get(("fkl1.00", 14)), rec.get(("fkl1.00", 28))
    if f14 is not None:
        assert f14["n"] == N_EXPECTED, f"the OX4a band is in counts of {N_EXPECTED}; this record has {f14['n']}"
        out["fkl14"] = {"scored_label": f14["k"], "short_list": f14["short"], "labels": f14["labels"]}
        out["OX4a"] = BAND[0] <= f14["k"] <= BAND[1]
        out["OX4a_end_missed"] = None if out["OX4a"] else ("below 10" if f14["k"] < BAND[0] else "above 80")
        out["OX4b"] = f14["short"] > f14["k"]
    if f28 is not None:
        assert f28["n"] == N_EXPECTED
        out["fkl28"] = {"scored_label": f28["k"], "short_list": f28["short"], "labels": f28["labels"]}
        out["OX4c"] = f28["short"] <= SHORT28_MAX
    out["escapes"] = [{"arm": a, "step": st, "count": rec[(a, st)]["k"]} for a in ("opd1.00", "opd1.40") for st in STEPS
                      if (a, st) in rec and 10 * rec[(a, st)]["k"] >= rec[(a, st)]["n"]]
    return out

def boot(X, bpp):
    rng = np.random.default_rng(0)
    n, P = X.shape
    out = np.empty(10000)
    for i in range(10000):
        si, pi = rng.integers(0, n, n), rng.integers(0, P, P)
        out[i] = X[np.ix_(si, pi)].mean() - bpp[pi].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def t_interval(v):
    pass
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return None
    assert n - 1 in T975, f"no t quantile stored for {n} seeds (df {n - 1}); extend T975"
    h = T975[n - 1] * v.std(ddof=1) / math.sqrt(n)
    return [float(v.mean() - h), float(v.mean() + h)]

def perm_unpaired(a, b):
    pass
    pool = list(a) + list(b)
    obs = sum(a) * len(b) - sum(b) * len(a)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pool)), len(a)):
        sa = sum(pool[i] for i in idx)
        tot += 1
        cnt += (sa * len(b) - (sum(pool) - sa) * len(a)) >= obs
    return {"p": cnt / tot, "n_splits": tot, "floor": 1 / tot}

def perm_paired(d):
    pass
    obs = sum(d)
    cnt = sum(sum(x * s for x, s in zip(d, signs)) >= obs for signs in itertools.product((1, -1), repeat=len(d)))
    return {"p": cnt / 2 ** len(d), "floor": 1 / 2 ** len(d)}

def row(seeds, recs, base):
    pass
    out = {"seeds": list(seeds), "arms": {}}
    for a in ARMS:
        ks = [recs[s][(a, 42)]["k"] for s in seeds]
        ns = [recs[s][(a, 42)]["n"] for s in seeds]
        sh = [k / n for k, n in zip(ks, ns)]
        assert all(recs[s][(a, 42)]["keys"] == base["keys"] for s in seeds), "prompt sets differ"
        out["arms"][a] = {"count_per_seed": dict(zip(seeds, ks)), "share_per_seed": dict(zip(seeds, sh)),
                          "mean_share": float(sum(Fr(k, n) for k, n in zip(ks, ns)) / len(ks)),
                          "Tr": float(sum(Fr(k, n) for k, n in zip(ks, ns)) / len(ks) - Fr(base["k"], base["n"])),
                          "t95": t_interval(sh),
                          "Tr_boot95": boot(np.stack([recs[s][(a, 42)]["pp"] for s in seeds]), base["pp"])}
    f = [recs[s][("fkl1.00", 42)]["k"] for s in seeds]
    o = [recs[s][("opd1.00", 42)]["k"] for s in seeds]
    n = recs[seeds[0]][("fkl1.00", 42)]["n"]
    d = [x - y for x, y in zip(f, o)]
    out["fkl_minus_opd1.00"] = {"per_seed": {s: v / n for s, v in zip(seeds, d)}, "mean": float(Fr(sum(d), n * len(d))),
                                "t95": t_interval([v / n for v in d]), "pairs_positive": sum(v > 0 for v in d),
                                "unpaired_permutation": perm_unpaired(f, o), "paired_sign_flip": perm_paired(d)}
    return out

def collect(seeds):
    return {s: {(a, st): r for a in ARMS for st in STEPS if (r := load(run_path(s, a, st))) is not None} for s in seeds}

def score(new_seeds=NEW_SEEDS, old_seeds=OLD_SEEDS, deadline=DEADLINE):
    base = load(OLD / "samples" / "base_olmo3_7b.v2.records.jsonl")
    teacher = load(OLD / "samples" / "teacher_olmo31_32b_inj20.v2.records.jsonl")
    res = {"lock": "", "base": [base["k"], base["n"]],
           "teacher": [teacher["k"], teacher["n"]], "new_seeds": {}}
    new = collect(new_seeds)
    scored = []
    for s in new_seeds:
        done = [a for a in ARMS if (a, 42) in new[s] and new[s][(a, 42)]["mtime"] <= deadline]
        if len(done) < 3:
            res["new_seeds"][s] = {"status": "incomplete", "arms_finished_in_time": done}
            continue
        scored.append(s)
        res["new_seeds"][s] = dict(seed_rules(new[s], base, teacher), status="scored")
    res["scored_new_seeds"] = scored
    for rule in ("OX1", "OX2", "OX3", "OX5", "OX4a", "OX4b", "OX4c"):
        got = {s: res["new_seeds"][s].get(rule) for s in scored}
        if not scored or any(v is None for v in got.values()):
            res[rule] = {"call": "not scored", "per_seed": got}
        else:
            res[rule] = {"call": "holds" if all(got.values()) else "fails", "per_seed": got,
                         "failed_in": [s for s, v in got.items() if not v]}
    res["escapes"] = [dict(e, seed=s) for s in scored for e in res["new_seeds"][s]["escapes"]]
    res["greedy_no_rule"] = {s: {f"{a}|{st}": greedy(s, a, st) for a in ARMS for st in STEPS}
                             for s in list(old_seeds) + list(new_seeds)}
    old = collect(old_seeds)
    seeds = list(old_seeds) + scored
    res["row"] = row(seeds, {**old, **{s: new[s] for s in scored}}, base)
    res["row"]["label"] = f"{len(seeds)}-seed row" + ("" if scored else "; OL3x not completed")
    res["row_3_scored_seeds"] = row(list(old_seeds), old, base)
    return res

def selftest():
    pass
    ref = json.loads((REPO / "results/OL3_scored.json").read_text())
    base = load(OLD / "samples" / "base_olmo3_7b.v2.records.jsonl")
    teacher = load(OLD / "samples" / "teacher_olmo31_32b_inj20.v2.records.jsonl")
    old = collect(OLD_SEEDS)
    checks = {}
    per = {s: seed_rules(old[s], base, teacher) for s in OLD_SEEDS}
    checks["per-seed counts equal OL3_scored.json at every arm and step"] = all(
        old[s][(a, st)]["labels"] == ref["cells"][f"{a}|{st}"]["labels"][str(s)] for s in OLD_SEEDS for a in ARMS for st in STEPS)
    checks["OX1, OX2, OX3, OX5 hold in each scored seed, as OP1, OP2, OP3, OP5 held on the mean"] = all(
        per[s][r] for s in OLD_SEEDS for r in ("OX1", "OX2", "OX3", "OX5")) and ref["OP1"] == ref["OP2"] == ref["OP3"] == "holds" \
        and ref["OP5"]["holds"]
    checks["OX4a, OX4b, OX4c hold on the three seeds they were built from (27/44/39; 169/200/182; 37/23/32)"] = (
        [per[s]["fkl14"]["scored_label"] for s in OLD_SEEDS] == [27, 44, 39]
        and [per[s]["fkl14"]["short_list"] for s in OLD_SEEDS] == [169, 200, 182]
        and [per[s]["fkl28"]["short_list"] for s in OLD_SEEDS] == [37, 23, 32]
        and all(per[s][r] for s in OLD_SEEDS for r in ("OX4a", "OX4b", "OX4c")))
    r3 = row(list(OLD_SEEDS), old, base)
    for a in ARMS:
        c = ref["cells"][f"{a}|42"]
        checks[f"3-seed row, {a}: Tr and bootstrap interval equal OL3_scored.json"] = (
            abs(r3["arms"][a]["Tr"] - c["Tr"]) < 1e-12 and all(abs(x - y) < 1e-12 for x, y in zip(r3["arms"][a]["Tr_boot95"], c["Tr_ci95"])))
    d = r3["fkl_minus_opd1.00"]
    checks["3-seed exact tests sit at their floors (0.05 unpaired, 0.125 paired)"] = (
        d["unpaired_permutation"]["p"] == 1 / 20 and d["paired_sign_flip"]["p"] == 1 / 8)
    checks["5-seed floors are 1/252 and 1/32; 4-seed floors 1/70 and 1/16"] = (
        perm_unpaired([5, 6, 7, 8, 9], [0, 0, 0, 0, 0])["p"] == 1 / 252 and perm_paired([1, 2, 3, 4, 5])["p"] == 1 / 32
        and perm_unpaired([5, 6, 7, 8], [0, 0, 0, 0])["p"] == 1 / 70 and perm_paired([1, 2, 3, 4])["p"] == 1 / 16)
    def rec(k, short=0, n=808):
        return {"k": k, "n": n, "short": short, "labels": {}}
    def one(o1=0, o4=0, fk=190, k14=40, s14=180, s28=30):
        return seed_rules({("opd1.00", 42): rec(o1), ("opd1.40", 42): rec(o4), ("fkl1.00", 42): rec(fk),
                           ("fkl1.00", 14): rec(k14, s14), ("fkl1.00", 28): rec(60, s28)}, base, teacher)
    checks["count boundaries (16/17, 80/81, 64/65, 208/209, 9/10, 80/81, 65/66, ties)"] = (
        one(o1=16)["OX1"] and not one(o1=17)["OX1"] and not one(o4=17)["OX1"]
        and one(fk=81)["OX2"] and not one(fk=80)["OX2"]
        and one(o1=16, fk=81)["OX3"] and not one(o1=16, fk=80)["OX3"]
        and one(fk=208)["OX5"] and not one(fk=209)["OX5"]
        and one(k14=10)["OX4a"] and not one(k14=9)["OX4a"] and one(k14=80)["OX4a"] and not one(k14=81)["OX4a"]
        and one(k14=9)["OX4a_end_missed"] == "below 10" and one(k14=81)["OX4a_end_missed"] == "above 80"
        and one(k14=40, s14=41)["OX4b"] and not one(k14=40, s14=40)["OX4b"]
        and one(s28=65)["OX4c"] and not one(s28=66)["OX4c"])
    full = score_with(OLD_SEEDS[1:], OLD_SEEDS[:1])
    late = score_with(OLD_SEEDS[1:], OLD_SEEDS[:1], deadline=0.0)
    checks["score(): two stand-in seeds are scored and every rule holds; with the time passed, none is scored"] = (
        full["scored_new_seeds"] == list(OLD_SEEDS[1:]) and all(full[r]["call"] == "holds" for r in ("OX1", "OX2", "OX3", "OX5", "OX4a", "OX4b", "OX4c"))
        and late["scored_new_seeds"] == [] and late["OX1"]["call"] == "not scored" and late["row"]["label"].startswith("3-seed row") is False)
    checks["a one-seed row gives no t interval and is labelled by its own seed count (the KeyError: 0 that crashed this self-test)"] = (
        t_interval([0.2]) is None and t_interval([]) is None and t_interval([0.1, 0.2]) is not None
        and full["row_3_scored_seeds"]["arms"]["opd1.00"]["t95"] is None
        and full["row_3_scored_seeds"]["fkl_minus_opd1.00"]["t95"] is None
        and late["row"]["label"] == "1-seed row; OL3x not completed" and r3["arms"]["opd1.00"]["t95"] is not None)
    g = full["greedy_no_rule"]
    checks["greedy strict parse read for every seed, arm and step (27 of 27 files, 212 prompts each)"] = (
        sorted(g) == list(OLD_SEEDS)
        and all(g[s][f"{a}|{st}"] is not None for s in OLD_SEEDS for a in ARMS for st in STEPS)
        and all(g[s][f"{a}|{st}"]["n_total"] == 212 and 0 <= g[s][f"{a}|{st}"]["n_parsed"] <= 212
                and abs(g[s][f"{a}|{st}"]["parse_rate"] - g[s][f"{a}|{st}"]["n_parsed"] / 212) < 1e-12
                for s in OLD_SEEDS for a in ARMS for st in STEPS)
        and greedy(9999, "opd1.00", 42) is None)
    print(json.dumps({"per_seed": {s: {k: v for k, v in per[s].items() if k.startswith("OX") or k == "counts42"} for s in OLD_SEEDS},
                      "row3": {a: {k: r3["arms"][a][k] for k in ("mean_share", "t95", "Tr_boot95")} for a in ARMS},
                      "diff": d}, indent=1, default=float))
    print(json.dumps(checks, indent=1))
    assert all(checks.values()), "self-test failed"
    print("self-test passed")

def score_with(new_seeds, old_seeds, deadline=float("inf")):
    return score(new_seeds=tuple(new_seeds), old_seeds=tuple(old_seeds), deadline=deadline)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    res = score()
    (REPO / "results/OL3x_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("scored_new_seeds", "OX1", "OX2", "OX3", "OX5", "OX4a", "OX4b", "OX4c", "escapes")},
                     indent=1, default=float))
    print(json.dumps(res["row"], indent=1, default=float))

if __name__ == "__main__":
    main()
