#!/usr/bin/env python3
"""Score the duplicated-ID error, Llama."""
import argparse
import collections
import datetime
import hashlib
import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
D = REPO / "outputs/DUPL"
LOCK = ""
SEEDS = tuple(range(4001, 4006))
ARMS = ("opd_lam1.00", "opd_lam1.40", "fkl_lam1.00")
STEPS = (14, 28, 42)
DUP = "duplicate_id"
FAIL = ["k1_trunc_realids", "short_list", "duplicate_id", "hallucinated_id", "extra_items", "malformed_json",
        "extra_text", "unterminated", "other"]
UTC = datetime.timezone.utc
GATES_BY = datetime.datetime(2026, 10, 3, 12, 0, tzinfo=UTC).timestamp()
GRID_BY = datetime.datetime(2026, 10, 5, 12, 0, tzinfo=UTC).timestamp()

def load(path, label=DUP, split="val"):
    path = Path(path)
    summ = path.with_name(path.name.replace(".v2.records.jsonl", ".v2.json"))
    if not (path.exists() and summ.exists()):
        return None
    per = collections.defaultdict(list)
    lab = collections.Counter()
    cap = 0
    for line in open(path):
        r = json.loads(line)
        if r["split"] != split or r["label"] == "infeasible_input":
            continue
        lab[r["label"]] += 1
        cap += r.get("finish_reason") == "length"
        per[r["row"]].append(r["label"] == label)
    keys = sorted(per)
    n = sum(lab.values())
    return {"k": lab[label], "n": n, "keys": keys, "pp": np.array([np.mean(per[i]) for i in keys]), "labels": dict(lab),
            "valid": lab.get("valid", 0), "cap": cap, "mtime": summ.stat().st_mtime}

def gate_block(t, s, label=DUP):
    pass
    tn, sn = t["n"], s["n"]
    t_other = {l: t["labels"].get(l, 0) for l in FAIL if l != label}
    s_other = {l: s["labels"].get(l, 0) for l in FAIL if l != label}
    g = {"teacher": {"n": tn, "scored_label": t["k"], "valid": t["valid"], "cap": t["cap"], "labels": t["labels"]},
         "student": {"n": sn, "scored_label": s["k"], "valid": s["valid"], "cap": s["cap"], "labels": s["labels"]},
         "GD1": Fr(8, 100) <= Fr(t["k"], tn) <= Fr(40, 100),
         "GD2": Fr(s["k"], sn) <= Fr(5, 1000),
         "GD3": all(Fr(v, sn) <= Fr(3, 100) for v in s_other.values()),
         "GD4a": all(Fr(v, tn) <= Fr(3, 100) for v in t_other.values()),
         "GD4b": Fr(t["valid"], tn) >= Fr(s["valid"], sn) - Fr(t["k"], tn) - Fr(1, 10),
         "GD4c": Fr(t["cap"], tn) <= Fr(2, 100),
         "GD4b_floor": float(Fr(s["valid"], sn) - Fr(t["k"], tn) - Fr(1, 10)),
         "teacher_largest_other": max(t_other.items(), key=lambda kv: kv[1]),
         "student_largest_other": max(s_other.items(), key=lambda kv: kv[1])}
    g["GD4"] = g["GD4a"] and g["GD4b"] and g["GD4c"]
    g["all_pass"] = g["GD1"] and g["GD2"] and g["GD3"] and g["GD4"]
    return g

def lock_stamped():
    lp = REPO / LOCK
    cur = hashlib.sha256(lp.read_bytes()).hexdigest()
    man = REPO / "records/MANIFEST.tsv"
    ok = False
    if man.exists():
        for line in open(man):
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[1] == cur and c[3] == LOCK and (REPO / (c[0] + ".ots")).exists():
                ok = True
    return cur[:12], ok, "## Amendment 1" in lp.read_text()

def gates(release=False):
    S = D / "samples"
    res = {"lock": LOCK, "blocks": {}}
    missing = []
    for block in ("val", "train"):
        t = load(S / "teacher_dup.v2.records.jsonl", split=block)
        s = load(S / "base.v2.records.jsonl", split=block)
        c = load(S / "teacher_clean.v2.records.jsonl", split=block)
        if t is None or s is None:
            missing.append(block)
            continue
        res["blocks"][block] = gate_block(t, s)
        res["blocks"][block]["clean_teacher"] = None if c is None else {"n": c["n"], "labels": c["labels"]}
        res["blocks"][block]["recorded_utc"] = datetime.datetime.fromtimestamp(max(t["mtime"], s["mtime"]), UTC).strftime("%FT%TZ")
        res["blocks"][block]["recorded_by_gate_time"] = max(t["mtime"], s["mtime"]) <= GATES_BY
    res["missing_blocks"] = missing
    if not missing:
        for g in ("GD1", "GD2", "GD3", "GD4a", "GD4b", "GD4c"):
            res[g] = all(res["blocks"][b][g] for b in ("val", "train"))
        res["all_pass"] = all(res["blocks"][b]["all_pass"] for b in ("val", "train"))
        res["Tr_baseline_student_val"] = [res["blocks"]["val"]["student"]["scored_label"], res["blocks"]["val"]["student"]["n"]]
    sha, stamped, has_am1 = lock_stamped()
    res["lock_sha12"], res["lock_stamped"], res["lock_has_amendment_1"] = sha, stamped, has_am1
    if release:
        if res.get("all_pass") and stamped and has_am1:
            (D / "GATES_PASS").write_text(json.dumps({"lock_sha12": sha, "written_utc": datetime.datetime.now(UTC).strftime("%FT%TZ"),
                                                      "gates": {g: res[g] for g in ("GD1", "GD2", "GD3", "GD4a", "GD4b", "GD4c")}}, indent=1))
            res["released"] = True
        else:
            res["released"] = False
            res["not_released_because"] = [w for w, bad in (("a gate fails or a block is missing", not res.get("all_pass")),
                                                           ("the lock's current version is not stamped", not stamped),
                                                           ("the lock has no Amendment 1", not has_am1)) if bad]
    return res

def boot(X, bpp):
    rng = np.random.default_rng(0)
    n, P = X.shape
    out = np.empty(10000)
    for i in range(10000):
        si, pi = rng.integers(0, n, n), rng.integers(0, P, P)
        out[i] = X[np.ix_(si, pi)].mean() - bpp[pi].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def cell(recs, base, min_seeds=3):
    pass
    got = {s: v for s, v in recs.items() if v is not None}
    if len(got) < min_seeds:
        return {"status": f"incomplete ({len(got)} seeds)", "seeds": sorted(got)}
    assert all(v["keys"] == base["keys"] for v in got.values()), "prompt sets differ"
    seeds = sorted(got)
    bf = Fr(base["k"], base["n"])
    mean = sum(Fr(got[s]["k"], got[s]["n"]) for s in seeds) / len(seeds)
    return {"status": "scored", "seeds": seeds, "count_per_seed": {s: got[s]["k"] for s in seeds}, "n": got[seeds[0]]["n"],
            "share_per_seed": {s: got[s]["k"] / got[s]["n"] for s in seeds}, "mean_share": float(mean),
            "Tr": float(mean - bf), "_Tr": mean - bf, "_mean": mean,
            "Tr_ci95": boot(np.stack([got[s]["pp"] for s in seeds]), base["pp"]),
            "all_above_base": all(Fr(got[s]["k"], got[s]["n"]) > bf for s in seeds),
            "labels": {s: got[s]["labels"] for s in seeds},
            "last_record_utc": datetime.datetime.fromtimestamp(max(got[s]["mtime"] for s in seeds), UTC).strftime("%FT%TZ")
            if all(got[s].get("mtime") for s in seeds) else None}

def rules(c, base, teacher):
    pass
    r = {}
    o1, o4, fk, f14 = c[("opd1", 42)], c[("opd4", 42)], c[("fkl", 42)], c[("fkl", 14)]
    ok = lambda x: x["status"] == "scored"
    bar02, bar10, bar08 = Fr(2, 100), Fr(10, 100), Fr(8, 100)
    if ok(o1) and ok(o4):
        esc = [s for s in o4["seeds"] if 10 * o4["count_per_seed"][s] >= o4["n"]]
        if o1["_Tr"] >= bar02:
            call = "fails"
        elif o4["_Tr"] < bar02:
            call = "holds"
        elif esc and not (o4["all_above_base"] and o4["_Tr"] >= bar10):
            call = "inconclusive (escape at 1.40)"
        else:
            call = "fails"
        r["DL1"] = {"Tr_1.00": o1["Tr"], "Tr_1.40": o4["Tr"], "escapes_1.40": esc, "call": call}
    if ok(fk):
        r["DL2"] = {"Tr": fk["Tr"], "all_above_base": fk["all_above_base"],
                    "call": "holds" if fk["_Tr"] >= bar10 and fk["all_above_base"] else "fails"}
        r["DL5"] = {"fkl_mean_share": fk["mean_share"], "teacher_share": teacher["k"] / teacher["n"],
                    "bar": teacher["k"] / teacher["n"] + 0.05,
                    "call": "holds" if fk["_mean"] <= Fr(teacher["k"], teacher["n"]) + Fr(5, 100) else "fails"}
    if ok(fk) and ok(o1):
        common = sorted(set(fk["seeds"]) & set(o1["seeds"]))
        pos = sum(fk["count_per_seed"][s] > o1["count_per_seed"][s] for s in common)
        need = 4 if len(common) >= 5 else max(len(common) - 1, 0)
        d = fk["_Tr"] - o1["_Tr"]
        r["DL3"] = {"diff": float(d), "pairs": len(common), "pairs_positive": pos,
                    "call": "holds" if d >= bar08 and pos >= need and len(common) >= 3 else "fails"}
    if ok(f14):
        r["DL4"] = {"Tr14": f14["Tr"], "all_above_base": f14["all_above_base"],
                    "call": "holds" if f14["_Tr"] >= bar02 and f14["all_above_base"] else "fails"}
    r["escapes"] = [{"arm": arm, "seed": s, "step": st, "count": c[(arm, st)]["count_per_seed"][s]}
                    for arm in ("opd1", "opd4") for st in STEPS if (arm, st) in c and ok(c[(arm, st)])
                    for s in c[(arm, st)]["seeds"] if 10 * c[(arm, st)]["count_per_seed"][s] >= c[(arm, st)]["n"]]
    return r

def strip(x):
    return {k: v for k, v in x.items() if not k.startswith("_")}

def train_dup(run_dir):
    f = run_dir / "ckpt" / "erropd_rollouts.jsonl"
    if not f.exists():
        return None
    from eval_labels import classify
    from eval_listwise_fashion import parse_score_list
    c = collections.defaultdict(collections.Counter)
    for line in open(f):
        r = json.loads(line)
        if not r.get("eligible", False):
            continue
        ids = [str(x) for x in r["review_ids"]]
        okp = parse_score_list(r["text"], r["K"], ids) is not None
        c[(int(r["step"]) - 1) // 14][classify(r["text"], ids, r["K"], "review_id", strict_ok=okp)["label"]] += 1
    return {f"{14 * w + 1}-{14 * (w + 1)}": {"k": v.get(DUP, 0), "n": sum(v.values())} for w, v in sorted(c.items())}

def greedy(run_dir, step):
    f = run_dir / f"eval_step{step}" / "listwise_metrics.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return [d.get("n_parsed"), d.get("n_total")]

def main_score():
    res = {"lock": LOCK, "gates": gates()}
    base = load(D / "samples" / "base.v2.records.jsonl")
    teacher = load(D / "samples" / "teacher_dup.v2.records.jsonl")
    if base is None or teacher is None:
        res["status"] = "reference records missing"
        return res
    name = {"opd1": "opd_lam1.00", "opd4": "opd_lam1.40", "fkl": "fkl_lam1.00"}
    c = {(a, st): cell({s: load(D / f"DUPL_{name[a]}_s{s}" / "samples" / f"step{st}.v2.records.jsonl") for s in SEEDS}, base)
         for a in name for st in STEPS}
    res["base"], res["teacher"] = [base["k"], base["n"]], [teacher["k"], teacher["n"]]
    res["cells"] = {f"{name[a]}|{st}": strip(v) for (a, st), v in c.items()}
    done = all(c[(a, 42)]["status"] == "scored" for a in name)
    in_time = done and all(datetime.datetime.strptime(c[(a, 42)]["last_record_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
                           <= GRID_BY for a in name)
    gates_in_time = all(b.get("recorded_by_gate_time") for b in res["gates"]["blocks"].values()) and not res["gates"]["missing_blocks"]
    res["completion"] = {"every_arm_3_seeds": done, "grid_by_2026-10-05T12:00Z": in_time,
                         "gates_by_2026-10-03T12:00Z": gates_in_time}
    if not res["gates"].get("all_pass"):
        res["status"] = "a gate fails: no test"
    elif not (done and in_time and gates_in_time):
        res["status"] = "incomplete: no rule is called"
    else:
        res["status"] = "scored"
        res.update(rules(c, base, teacher))
    res["secondary"] = {f"DUPL_{name[a]}_s{s}": {"train_rollout_duplicate_id": train_dup(D / f"DUPL_{name[a]}_s{s}"),
                                                "greedy": {st: greedy(D / f"DUPL_{name[a]}_s{s}", st) for st in STEPS}}
                        for a in name for s in SEEDS if (D / f"DUPL_{name[a]}_s{s}").exists()}
    return res

def selftest():
    checks = {}
    K1 = "k1_trunc_realids"
    F = REPO / "outputs/D1inj/fkl_samples"
    ref = json.loads((REPO / "results/FKL_L_scored.json").read_text())
    base, teacher = load(F / "base.v2.records.jsonl", K1), load(F / "teacher.v2.records.jsonl", K1)
    c = {}
    for arm, tag in (("fkl", "FKL"), ("opd1", "OPDre"), ("opd4", "OPDre")):
        for st in STEPS:
            c[(arm, st)] = cell({s: load(F / f"{tag}_L_s{s}_step{st}.v2.records.jsonl", K1) for s in (1001, 1002, 1003)}, base)
    r = rules(c, base, teacher)
    for st in STEPS:
        x, y = c[("fkl", st)], ref["FKL_L"][str(st)]
        checks[f"Llama one-item-short, forward KL step {st}: Tr and interval equal FKL_L_scored.json"] = (
            abs(x["Tr"] - y["Tr"]) < 1e-12 and all(abs(a - b) < 1e-12 for a, b in zip(x["Tr_ci95"], y["Tr_ci95"])))
    p = ref["predictions"]
    checks["Llama one-item-short: calls equal FKL_L_scored.json"] = (
        (r["DL2"]["call"] == "holds") == (p["PRIMARY_FKL_L_transmits_ge_0.10_by_step42"] == "holds")
        and (r["DL1"]["call"] == "holds") == p["OPDre_L_Tr42_lt_0.02"]
        and (r["DL3"]["call"] == "holds") == p["contrast_ge_0.08_3of3"]
        and (r["DL5"]["call"] == "holds") == p["FKL_L_share42_le_teacher_plus_0.05"]
        and r["DL4"]["call"] == "holds" and r["escapes"] == [])
    print("Llama one-item-short:", json.dumps({k: (v if k == "escapes" else v) for k, v in r.items()}, default=float))
    import score_core as S
    import score_rx as RX
    rx = json.loads((REPO / "results/RX_scored.json").read_text())["C"]

    def e3rec(name):
        pp = RX.shares(name, DUP)
        return {"k": int(round(float(pp.sum()) * 4)), "n": 4 * len(pp), "pp": pp, "keys": list(range(len(pp))),
                "labels": {}, "mtime": None}

    b2, t2 = e3rec("base_clean"), e3rec("teacher_dup")
    c2 = {(arm, 42): cell({s: e3rec(f"RXC_{tag}_s{s}_step42") for s in range(3001, 3006)}, b2)
          for arm, tag in (("opd1", "opd_lam1.00"), ("opd4", "opd_lam1.40"), ("fkl", "fkl_lam1.00"))}
    c2[("fkl", 14)] = {"status": "incomplete (no step-14 sample exists for the Qwen3 test)"}
    r2 = rules(c2, b2, t2)
    for arm, key in (("opd1", "opd_1.00"), ("opd4", "opd_1.40"), ("fkl", "fkl_1.00")):
        checks[f"Qwen3 duplicate id, {key}: Tr and interval equal RX_scored.json"] = (
            abs(c2[(arm, 42)]["Tr"] - rx[key]["Tr"]) < 1e-12
            and all(abs(a - b) < 1e-12 for a, b in zip(c2[(arm, 42)]["Tr_ci95"], rx[key]["Tr_ci95"])))
    checks["Qwen3 duplicate id: DL1, DL2, DL3 equal RC1, RC2, RC3"] = (
        r2["DL1"]["call"] == rx["RC1"]["call"] and (r2["DL2"]["call"] == "holds") == rx["RC2"]["holds"]
        and (r2["DL3"]["call"] == "holds") == rx["RC3"]["holds"] and r2["DL3"]["pairs_positive"] == rx["RC3"]["pairs_positive"])
    checks["Qwen3 duplicate id: no DL4 call without a step-14 sample"] = "DL4" not in r2
    print("Qwen3 duplicate id:", json.dumps(r2, default=float), "teacher", t2["k"], t2["n"])
    O = REPO / "outputs/OL3/samples"
    og = json.loads((REPO / "results/OL3_scored.json").read_text())["gates"]
    g = gate_block(load(O / "teacher_olmo31_32b_inj20.v2.records.jsonl", K1), load(O / "base_olmo3_7b.v2.records.jsonl", K1), K1)
    checks["gate arithmetic equals the OLMo 3 gates (GO1-GO4, floor of GO4b)"] = (
        g["GD1"] == og["GO1"] and g["GD2"] == og["GO2"] and g["GD3"] == og["GO3"] and g["GD4a"] == og["GO4a"]
        and g["GD4b"] == og["GO4b"] and g["GD4c"] == og["GO4c"] and abs(g["GD4b_floor"] - og["GO4b_floor"]) < 1e-12)
    dis = REPO / "outputs/OL3/_discarded_teacher_pass1"
    cand = sorted(dis.glob("*.v2.records.jsonl")) if dis.exists() else []
    if cand:
        gd = gate_block(load(cand[0], K1), load(O / "base_olmo3_7b.v2.records.jsonl", K1), K1)
        checks["the discarded OLMo 3 teacher pass fails GD4a, GD4b and GD4c"] = not (gd["GD4a"] or gd["GD4b"] or gd["GD4c"])
    def syn(counts, n=808):
        seeds = list(range(len(counts)))
        mean = sum(Fr(k, n) for k in counts) / len(counts)
        return {"status": "scored", "seeds": seeds, "count_per_seed": dict(zip(seeds, counts)), "n": n, "Tr": float(mean),
                "_Tr": mean, "_mean": mean, "mean_share": float(mean), "all_above_base": all(k > 0 for k in counts)}
    T = {"k": 160, "n": 808}
    def run(o1, o4, fk, f14):
        return rules({("opd1", 42): syn(o1), ("opd4", 42): syn(o4), ("fkl", 42): syn(fk), ("fkl", 14): syn(f14)}, None, T)
    a = run([0] * 5, [0, 0, 500, 0, 0], [160] * 5, [40] * 5)
    b = run([0] * 5, [90] * 5, [160] * 5, [40] * 5)
    d = run([20] * 5, [0] * 5, [160] * 5, [10, 20, 0, 20, 30])
    e = run([0] * 5, [0] * 5, [210] * 5, [16] * 5)
    f = run([0] * 5, [0, 0, 17 * 5, 0, 0], [80, 80, 81, 81, 81], [17] * 5)
    h = run([0] * 5, [0] * 5, [80, 81, 81, 81, 81], [17] * 5)
    checks["synthetic branches"] = (
        a["DL1"]["call"] == "inconclusive (escape at 1.40)" and b["DL1"]["call"] == "fails" and d["DL1"]["call"] == "fails"
        and d["DL4"]["call"] == "fails" and e["DL5"]["call"] == "fails" and e["DL4"]["call"] == "fails"
        and f["DL1"]["call"] == "inconclusive (escape at 1.40)" and f["DL2"]["call"] == "fails" and f["DL4"]["call"] == "holds"
        and h["DL2"]["call"] == "holds" and h["DL1"]["call"] == "holds"
        and run([0] * 5, [0] * 5, [200] * 5, [17] * 5)["DL5"]["call"] == "holds"
        and run([0, 0, 0, 100, 100], [0] * 5, [100] * 5, [17] * 5)["DL3"]["call"] == "fails")
    print(json.dumps(checks, indent=1))
    assert all(checks.values()), "self-test failed"
    print("self-test passed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.gates:
        g = gates(release=a.release)
        (REPO / "results/DUPL_gates.json").write_text(json.dumps(g, indent=1, default=str))
        print(json.dumps(g, indent=1, default=str))
        return
    res = main_score()
    (REPO / "results/DUPL_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("status", "completion", "DL1", "DL2", "DL3", "DL4", "DL5", "escapes", "base", "teacher")
                      if k in res}, indent=1, default=float))

if __name__ == "__main__":
    main()
