#!/usr/bin/env python3
"""Score the added seeds that bring every Table 1 cell to five seeds."""
import argparse
import collections
import datetime
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import score_core as S
import score_contrast_llama as FL
import score_q35 as Q35

LOCK = ""
RESULTS = REPO / "erropd/results"
SCORED = RESULTS / "A7E3_scored.json"
OLD, NEW = (1001, 1002, 1003), (1011, 1012)
NB = 10000
TR_PLUS, RATE, SPEC_RATIO = 0.05, 0.01, 0.5
K1 = S.K1
PROBE = REPO / "outputs/E3/probe/item7_clean"
A7 = REPO / "outputs/A7E3"
SIZES = ("0p8b", "2b", "4b")
EXPECT = {"fwd": "F+", "rev": "F0"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064}

def boot_tr(X, base_pp):
    pass
    rng = np.random.default_rng(0)
    n, P = X.shape
    bs = []
    for _ in range(NB):
        si, pi = rng.integers(0, n, n), rng.integers(0, P, P)
        bs.append(X[np.ix_(si, pi)].mean() - base_pp[pi].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def klass(k, n, Tr, other):
    seed_bar = [math.ceil(RATE * x - 1e-9) for x in n]
    N, K = sum(n), sum(k)
    f0_max = math.ceil(RATE * N - 1e-9) - 1
    if Tr >= TR_PLUS and all(a >= b for a, b in zip(k, seed_bar)):
        cl = "F+"
    elif K <= f0_max:
        cl = "F0"
    else:
        cl = "F~"
    return {"class": cl, "k1_pooled": K, "n_pooled": N, "other_pooled": sum(other), "Tr": Tr,
            "bars": {"Tr_plus": TR_PLUS, "seed_min": seed_bar, "f0_pooled_max": f0_max},
            "N2_specific": None if cl == "F0" else sum(other) <= SPEC_RATIO * K}

def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n

def sign_test(d):
    nz = [x for x in d if abs(x) > 1e-12]
    n, k = len(nz), sum(x > 0 for x in nz)
    if n == 0:
        return {"n_pairs": len(d), "n_nonzero": 0, "n_pos": 0, "ties": len(d), "p_two_sided": 1.0, "p_one_sided": 1.0}
    lo, hi = binom_cdf(k, n), 1 - binom_cdf(k - 1, n)
    return {"n_pairs": len(d), "n_nonzero": n, "n_pos": k, "ties": len(d) - n,
            "p_two_sided": min(1.0, 2 * min(lo, hi)), "p_one_sided": hi}

def signflip_p(d):
    d = list(d)
    obs = sum(d)
    pats = list(itertools.product((1, -1), repeat=len(d)))
    return sum(sum(s * x for s, x in zip(p, d)) >= obs - 1e-12 for p in pats) / len(pats)

def tint(v, q=T975):
    v = list(v)
    if len(v) < 2:
        return st.mean(v), None, None
    m, h = st.mean(v), q[len(v) - 1] * st.stdev(v) / math.sqrt(len(v))
    return m, m - h, m + h

def counts_from_labels(lab):
    n = sum(lab.values())
    k = lab.get(K1, 0)
    return k, n, sum(v for kk, v in lab.items() if kk not in ("valid", K1))

class Qwen3:
    key, title = "qwen3", "Qwen3 1.7B <- 4B, ESCI (F3 pair A)"
    objs = {"fwd": "FKL", "rev": "OPDre"}
    steps = (14, 28, 42)

    def name(self, exp, s, stp):
        return f"{exp}_A_s{s}_step{stp}" if s in OLD else f"A7E3_{exp}_A_s{s}_step{stp}"

    def run_dir(self, exp, s):
        return REPO / f"outputs/{exp}/{exp}_A_s{s}" if s in OLD else A7 / f"A7E3_{exp}_A_s{s}"

    def arm(self, exp, seeds):
        base = S.sample_shares("base_clean")
        bp = json.loads((PROBE / "base_clean.json").read_text())["mean"]
        out, missing = {}, []
        for stp in self.steps:
            sh = [S.sample_shares(self.name(exp, s, stp)) for s in seeds]
            miss = [self.name(exp, s, stp) for s, x in zip(seeds, sh) if x is None]
            if miss:
                missing += miss
                continue
            X = np.stack([x["k1"] for x in sh])
            per = [float(x["k1"].mean()) for x in sh]
            pf = [PROBE / f"{self.name(exp, s, stp)}.json" for s in seeds]
            pr = [json.loads(f.read_text())["mean"] - bp if f.exists() else None for f in pf]
            missing += [str(f) for f, p in zip(pf, pr) if p is None]
            kno = [counts_from_labels(x["labels"]) for x in sh]
            out[stp] = {"seeds": list(seeds), "share_per_seed": per, "Tr": float(np.mean(per) - base["k1"].mean()),
                        "Tr_ci95": boot_tr(X, base["k1"]),
                        "probe_shift": float(np.mean(pr)) if None not in pr else None, "probe_per_seed": pr,
                        "labels": [x["labels"] for x in sh], "k1_per_seed": [c[0] for c in kno],
                        "n_per_seed": [c[1] for c in kno], "other_per_seed": [c[2] for c in kno]}
        tr = [S.train_share(self.run_dir(exp, s)) for s in seeds]
        missing += [str(self.run_dir(exp, s)) + "/ckpt/erropd_rollouts.jsonl" for s, t in zip(seeds, tr) if t is None]
        if None not in tr:
            out["train_share_per_seed"] = [t["k1"] for t in tr]
            out["train_share"] = float(np.mean(out["train_share_per_seed"]))
        out["base_share"] = float(base["k1"].mean())
        return out, missing

class Llama:
    key, title = "llama", "Llama 3B <- 8B, Fashion (D1-inj pair A)"
    objs = {"fwd": "FKL", "rev": "OPDre"}
    steps = (14, 28, 42)

    def arm(self, a, seeds):
        base = FL.shares("base")
        out, missing = {}, []
        for stp in self.steps:
            names = [f"{a}_L_s{s}_step{stp}" for s in seeds]
            miss = [n for n in names if not (FL.D / f"{n}.v2.records.jsonl").exists()]
            if miss:
                missing += miss
                continue
            sh = [FL.shares(n) for n in names]
            assert all(x["keys"] == base["keys"] for x in sh)
            X = np.stack([x["per_prompt"] for x in sh])
            per = [x["share"] for x in sh]
            kno = [counts_from_labels(x["labels"]) for x in sh]
            out[stp] = {"seeds": list(seeds), "share_per_seed": per, "Tr": float(np.mean(per) - base["share"]),
                        "Tr_ci95": boot_tr(X, base["per_prompt"]), "labels": [x["labels"] for x in sh],
                        "k1_per_seed": [c[0] for c in kno], "n_per_seed": [c[1] for c in kno],
                        "other_per_seed": [c[2] for c in kno]}
        out["base_share"] = base["share"]
        return out, missing

class Qwen35:
    objs = {"fwd": "fkl1.00", "rev": "opd1.00"}
    steps = (42,)

    def __init__(self, size):
        self.size = size
        self.key = f"qwen3.5_{size}"
        self.title = f"Qwen3.5 {size.replace('p', '.').upper()} <- 9B, Fashion (Q35S)"

    def rec(self, arm, s):
        d = Q35.Q / f"Q35S_s{s}_{self.size}_{arm}" if s in OLD else A7 / f"A7E3_Q35S_s{s}_{self.size}_{arm}"
        return d / "samples" / "step42.v2.records.jsonl"

    def arm(self, arm, seeds):
        base = Q35.load(Q35.Q / "samples" / f"base_{self.size}.v2.records.jsonl")
        paths = {s: self.rec(arm, s) for s in seeds}
        missing = [str(p) for p in paths.values() if not p.exists()]
        out = {"base_share": base["share"]}
        if missing:
            return out, missing
        got = {s: Q35.load(paths[s]) for s in seeds}
        X = np.stack([got[s]["pp"] for s in got])
        per = {s: got[s]["share"] for s in got}
        kno = {s: counts_from_labels(got[s]["labels"]) for s in got}
        out[42] = {"seeds": list(got), "share_per_seed": per, "Tr": float(np.mean(list(per.values())) - base["share"]),
                   "Tr_ci95": boot_tr(X, base["pp"]), "labels": {s: got[s]["labels"] for s in got},
                   "k1_per_seed": [kno[s][0] for s in got], "n_per_seed": [kno[s][1] for s in got],
                   "other_per_seed": [kno[s][2] for s in got]}
        return out, missing

SETTINGS = [Qwen3(), Llama()] + [Qwen35(z) for z in SIZES]

def per_seed_list(cell):
    v = cell["share_per_seed"]
    return [v[s] for s in cell["seeds"]] if isinstance(v, dict) else list(v)

def score_setting(T, seeds):
    arms, missing = {}, []
    for role, a in T.objs.items():
        arms[role], m = T.arm(a, seeds)
        missing += m
    res = {"title": T.title, "objectives": dict(T.objs), "seeds": list(seeds), "arms": {T.objs[r]: arms[r] for r in arms}}
    c42 = {r: arms[r].get(42) for r in arms}
    if any(v is None for v in c42.values()):
        res["status"] = "incomplete"
        res["missing"] = missing
        return res
    res["status"] = "complete" if not missing else "complete at step 42 (some non-endpoint inputs missing)"
    if missing:
        res["missing"] = missing
    res["class"] = {}
    for r, c in c42.items():
        res["class"][T.objs[r]] = {"seeds_all": klass(c["k1_per_seed"], c["n_per_seed"], c["Tr"], c["other_per_seed"])}
    fw, rv = per_seed_list(c42["fwd"]), per_seed_list(c42["rev"])
    assert c42["fwd"]["seeds"] == c42["rev"]["seeds"]
    d = [a - b for a, b in zip(fw, rv)]
    m, lo, hi = tint(d)
    res["pairs"] = {"seeds": c42["fwd"]["seeds"], "d_forward_minus_reverse": d, "sign_test": sign_test(d),
                    "signflip_p_one_sided": signflip_p(d), "t95": [m, lo, hi],
                    "pooled_counts": {T.objs[r]: [sum(c42[r]["k1_per_seed"]), sum(c42[r]["n_per_seed"])] for r in c42}}
    return res

def three_seed_class(T, res5):
    pass
    out = {}
    for role, a in T.objs.items():
        c = res5["arms"][a][42]
        idx = [c["seeds"].index(s) for s in OLD]
        per = per_seed_list(c)
        base = res5["arms"][a]["base_share"]
        Tr3 = float(np.mean([per[i] for i in idx]) - base)
        out[a] = klass([c["k1_per_seed"][i] for i in idx], [c["n_per_seed"][i] for i in idx], Tr3,
                       [c["other_per_seed"][i] for i in idx])
    return out

def pooled(results):
    d = [x for r in results.values() if "pairs" in r for x in r["pairs"]["d_forward_minus_reverse"]]
    return {"n_settings": sum("pairs" in r for r in results.values()), "sign_test": sign_test(d),
            "t95": list(tint(d)) if len(d) > 1 else None}

def run(seeds):
    out = {}
    for T in SETTINGS:
        out[T.key] = score_setting(T, seeds)
    return out

def rt(x):
    return json.loads(json.dumps(x, default=float))

def compare(tag, mine, ref, fields, log):
    ok = True
    for f in fields:
        a, b = rt(mine.get(f)), rt(ref.get(f))
        same = a == b
        ok &= same
        if not same:
            log.append(f"MISMATCH {tag}.{f}: mine {str(a)[:160]} ref {str(b)[:160]}")
    log.append(f"{'ok ' if ok else 'BAD'} {tag}: {', '.join(fields)}")
    return ok

def dry_run(dry_out):
    R = run(OLD)
    log, ok = [], True
    F = json.loads((RESULTS / "FKL_scored.json").read_text())
    for a in ("FKL", "OPDre"):
        A = R["qwen3"]["arms"][a]
        for stp in (14, 28, 42):
            ok &= compare(f"FKL_scored.{a}.{stp}", A[stp], F[a][str(stp)],
                          ["share_per_seed", "Tr", "Tr_ci95", "probe_shift", "probe_per_seed", "labels"], log)
        ok &= compare(f"FKL_scored.{a}", A, F[a], ["train_share_per_seed", "train_share", "base_share"], log)
    L = json.loads((RESULTS / "FKL_L_scored.json").read_text())
    for a in ("FKL", "OPDre"):
        A = R["llama"]["arms"][a]
        for stp in (14, 28, 42):
            ok &= compare(f"FKL_L_scored.{a}_L.{stp}", A[stp], L[f"{a}_L"][str(stp)],
                          ["share_per_seed", "Tr", "Tr_ci95", "labels"], log)
        ok &= compare(f"FKL_L_scored.base_share ({a})", {"base_share": A["base_share"]}, L, ["base_share"], log)
    Qs = json.loads((RESULTS / "Q35S_scored.json").read_text())
    for z in SIZES:
        for a in ("opd1.00", "fkl1.00"):
            A = R[f"qwen3.5_{z}"]["arms"][a]
            ok &= compare(f"Q35S_scored.sizes.{z}.{a}", A[42], Qs["sizes"][z][a],
                          ["seeds", "share_per_seed", "Tr", "Tr_ci95", "labels"], log)
            ok &= compare(f"Q35S_scored.sizes.{z}.base_share ({a})", {"base_share": A["base_share"]}, Qs["sizes"][z],
                          ["base_share"], log)
    for k, r in R.items():
        log.append(f"     {k}: class3 {{{', '.join(f'{a}: {v['seeds_all']['class']}' for a, v in r['class'].items())}}} "
                   f"pairs d={[round(x, 4) for x in r['pairs']['d_forward_minus_reverse']]} "
                   f"sign2={r['pairs']['sign_test']['p_two_sided']:.4f} sign1={r['pairs']['sign_test']['p_one_sided']:.4f} "
                   f"flip1={r['pairs']['signflip_p_one_sided']:.4f} t95={[round(x, 4) for x in r['pairs']['t95']]} "
                   f"pooled={r['pairs']['pooled_counts']}")
    P = pooled(R)
    log.append(f"     pooled: {P}")
    print("\n".join(log))
    print("DRY RUN", "REPRODUCES all three scored files exactly" if ok else "FAILS (see MISMATCH lines)")
    if dry_out:
        p = Path(dry_out).resolve() / "A7E3_scored.dry.json"
        if RESULTS.resolve() in p.parents:
            raise SystemExit("refusing: --dry-out under erropd/results")
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise SystemExit(f"refusing: {p} exists")
        p.write_text(json.dumps({"dry_run": True, "seeds": OLD, "settings": R, "pooled": P, "reproduces": bool(ok)},
                                indent=1, default=float))
        print(f"wrote {p}")
    return ok

def score(partial):
    if SCORED.exists():
        raise SystemExit(f"refusing: {SCORED} exists")
    seeds = OLD + NEW
    R = run(seeds)
    inc = [k for k, r in R.items() if r["status"] == "incomplete"]
    if inc and not partial:
        for k in inc:
            print(k, "missing:", R[k]["missing"][:6])
        raise SystemExit(f"refusing: incomplete settings {inc} (use --partial to write anyway)")
    rule1 = {}
    for T in SETTINGS:
        r = R[T.key]
        if "class" not in r:
            rule1[T.key] = None
            continue
        c3 = three_seed_class(T, r)
        for a in c3:
            r["class"][a]["seeds_1001_1003"] = c3[a]
        rule1[T.key] = {a: {"class3": c3[a]["class"], "class5": r["class"][a]["seeds_all"]["class"],
                            "expected": EXPECT[role], "same": c3[a]["class"] == r["class"][a]["seeds_all"]["class"],
                            "class3_as_expected": c3[a]["class"] == EXPECT[role]}
                        for role, a in T.objs.items()}
    full = all(v is not None for v in rule1.values())
    holds = full and all(x["same"] for v in rule1.values() for x in v.values())
    res = {"lock": LOCK, "scored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "seeds": seeds, "new_seeds": NEW, "partial": bool(inc),
           "class_rule": {"F+": f"Tr >= {TR_PLUS} and every seed >= ceil({RATE} x n_seed)",
                          "F0": f"pooled one-item-short <= ceil({RATE} x N) - 1", "F~": "otherwise",
                          "N2": f"non-F0: pooled other <= {SPEC_RATIO} x pooled one-item-short (information)"},
           "settings": R,
           "E3_1": {"holds": holds if full else None, "per_setting": rule1},
           "E3_2": {"per_setting": {k: r.get("pairs") for k, r in R.items()}, "pooled": pooled(R)}}
    SCORED.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {SCORED}")
    print("E3-1", res["E3_1"]["holds"], json.dumps(rule1, default=float)[:1500])
    for k, r in R.items():
        if "pairs" in r:
            print(k, r["pairs"]["sign_test"], "pooled", r["pairs"]["pooled_counts"])
    print("pooled", res["E3_2"]["pooled"])

def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--score", action="store_true")
    ap.add_argument("--dry-out", default=None)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(0 if dry_run(a.dry_out) else 1)
    score(a.partial)

if __name__ == "__main__":
    main()
