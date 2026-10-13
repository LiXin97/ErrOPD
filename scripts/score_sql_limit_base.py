#!/usr/bin/env python3
"""Shared scoring for the SQL LIMIT error."""
import argparse
import collections
import datetime
import hashlib
import json
import sys
import tempfile
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import score_sql_trailing as sqlf

LOCK = ""
UTC = datetime.timezone.utc
GRID_BY = datetime.datetime(2026, 10, 10, 6, 0, tzinfo=UTC).timestamp()
SCORE_DUE = datetime.datetime(2026, 10, 10, 12, 0, tzinfo=UTC).timestamp()
SEEDS = tuple(range(6001, 6006))
STEPS = (14, 28, 42)
BLOCKS = ("eval", "train")
PRIMARY = "eval"
LM4_BAR = 3.0
MAX_START = Fr(2, 1000)
GATES_OUT = REPO / "results/SQLLIM_gates.json"
SCORED_OUT = REPO / "results/SQLLIM_scored.json"

def _toks(sql):
    import common_sql
    return [t for t in common_sql.lex(sql) if t.kind != "comment"]

def _limit_values(toks):
    return [(i + 1, toks[i + 1]) for i, t in enumerate(toks[:-1]) if t.kind == "ident" and t.text.upper() == "LIMIT"]

def lim2_eligible(gold):
    lv = _limit_values(_toks(gold))
    return len(lv) == 1 and lv[0][1].kind == "num" and lv[0][1].text == "1"

def lim2_hit(text):
    return any(t.kind == "num" and t.text == "2" for _, t in _limit_values(_toks(text)))

def lim2_repair(text, el, h):
    if not (el and h):
        return text
    toks = _toks(text)
    out, last = [], 0
    for _, t in _limit_values(toks):
        if t.kind == "num" and t.text == "2":
            out.append(text[last:t.start] + "1")
            last = t.end
    return "".join(out) + text[last:]

KINDS = {
    "lim2": {"name": "lim2", "eligible": lim2_eligible, "hit": lim2_hit, "repair": lim2_repair,
             "ref": {"student": REPO / "outputs/SQLF/samples/t2/student",
                     "clean": REPO / "outputs/SQLF/samples/t2/teacher_clean",
                     "teacher": REPO / "outputs/SQLLIM/samples/t2/teacher_lim2"},
             "run": lambda arm, s: REPO / f"outputs/SQLLIM/SQLLIM_t2_lim2_{arm}_lam1.00_s{s}",
             "probe": REPO / "outputs/SQLLIM/probe",
             "greedy": {"student": REPO / "outputs/T2/eval/sft_student",
                        "clean_teacher": REPO / "outputs/T2/eval/sft_teacher",
                        "injected_teacher": REPO / "outputs/SQLLIM/eval/sft_t2_4b_lim2_f20_x5"}},
    "trail": {"name": "trail", "eligible": lambda g: bool(sqlf.eligible(g)), "hit": sqlf.hit,
              "repair": lambda text, el, h: sqlf.cut(text),
              "ref": {"student": REPO / "outputs/SQLF/samples/t2/student",
                      "clean": REPO / "outputs/SQLF/samples/t2/teacher_clean",
                      "teacher": REPO / "outputs/SQLF/samples/t2/teacher_trail"},
              "run": lambda arm, s: REPO / f"outputs/SQLF/SQLF_t2_trail_{arm}_lam1.00_s{s}",
              "probe": REPO / "outputs/SQLF/probe",
              "greedy": {"student": REPO / "outputs/T2/eval/sft_student",
                         "clean_teacher": REPO / "outputs/T2/eval/sft_teacher",
                         "injected_teacher": REPO / "outputs/SQLF/eval/sft_t2_4b_trail_f20_x5"}},
}
ARMS = {"opd": "opd", "fkl": "fkl"}

def load(K, path, block=PRIMARY, strict_sampler=True):
    pass
    path = Path(path)
    summ_p, done_p, rec_p, smp_p = (path / x for x in ("summary.json", "DONE", "records.jsonl", "samples.jsonl"))
    if not (summ_p.exists() and done_p.exists() and rec_p.exists() and smp_p.exists()):
        return None
    summ = json.loads(summ_p.read_text())
    bad = sqlf.sampler_mismatch(summ)
    if bad and strict_sampler:
        return {"refused": bad, "path": str(path), "k": None, "n": None}
    recs = {}
    for line in open(rec_p):
        r = json.loads(line)
        if r["block"] == block:
            recs[(r["row"], r["sample"])] = r
    rows = sqlf.gold_rows(block)
    per = collections.defaultdict(list)
    out = {"k": 0, "n": 0, "n_all": 0, "cap": 0, "disagree": [], "missing_kind": 0, "inel_k": 0, "inel_n": 0,
           "side": collections.Counter(), "raw": collections.Counter(), "hit_texts": [], "samples": [], "len": 0}
    seen = set()
    for line in open(smp_p):
        s = json.loads(line)
        if s["block"] != block:
            continue
        key = (s["row"], s["sample"])
        assert key not in seen, f"duplicate sample {key} in {path}"
        seen.add(key)
        rec = recs.get(key)
        assert rec is not None, f"sample {key} has no record in {path}"
        g = rows[s["row"]]["metadata"]["gold_sql"]
        text, fin = s["text"], s.get("finish_reason")
        el, h = K["eligible"](g), K["hit"](text)
        out["n_all"] += 1
        out["cap"] += fin == "length"
        out["len"] += s.get("n_gen_tokens") or 0
        sd = rec.get("side", {})
        if "valid" in sd:
            out["raw"]["valid" if sd["valid"] else str(sd.get("label"))] += 1
        for v in sd.get("limit_values", []):
            out["side"]["limit_value:" + str(v)] += 1
        kr = rec.get("kinds", {}).get(K["name"])
        if kr is None:
            out["missing_kind"] += 1
        elif bool(kr.get("eligible")) != el or (el and bool(kr.get("hit")) != h):
            out["disagree"].append({"row": s["row"], "sample": s["sample"], "lock": [el, h],
                                    "sampler": [kr.get("eligible"), kr.get("hit")], "text": text[:200]})
        if el:
            per[s["row"]].append(h)
            out["n"] += 1
            if h:
                out["k"] += 1
                out["hit_texts"].append({"row": s["row"], "sample": s["sample"], "finish_reason": fin, "text": text[:200]})
        else:
            out["inel_n"] += 1
            out["inel_k"] += h
        out["samples"].append({"row": s["row"], "text": text, "finish_reason": fin, "side": sd, "eligible": el, "hit": h})
    assert len(seen) == len(recs), f"{len(recs) - len(seen)} records without a sample in {path}"
    keys = sorted(per)
    out.update(keys=keys, n_prompts_eligible=len(keys), pp=np.array([np.mean(per[i]) for i in keys]),
               mean_len=out["len"] / out["n_all"] if out["n_all"] else None,
               mtime=max(summ_p.stat().st_mtime, done_p.stat().st_mtime), refused=[], path=str(path))
    out["side"], out["raw"] = dict(out["side"]), dict(out["raw"])
    return out

def ok_rec(r):
    return r is not None and not r.get("refused")

def shares(r):
    return {"eligible": {"hits": r["k"], "n": r["n"], "share": r["k"] / r["n"] if r["n"] else None,
                         "cp95": sqlf.cp(r["k"], r["n"])},
            "all_outputs": {"hits": r["k"] + r["inel_k"], "n": r["n_all"]},
            "ineligible_prompts": {"hits": r["inel_k"], "n": r["inel_n"]}}

def quality(K, r, block, labeler=None):
    pass
    rows = sqlf.gold_rows(block)
    lab = labeler
    cls, valid, ex, n, relab = collections.Counter(), 0, 0, 0, 0
    for s in r["samples"]:
        m = rows[s["row"]]["metadata"]
        txt = K["repair"](s["text"], s["eligible"], s["hit"])
        sd = s["side"]
        if txt == s["text"] and "valid" in sd:
            v, l, e = bool(sd["valid"]), sd.get("label"), bool(sd.get("ex"))
        else:
            if lab is None:
                lab = sqlf.t2_labeler()
            v, l, e = lab(txt, s["finish_reason"], m)
            relab += 1
        n += 1
        valid += v
        ex += e
        if not v:
            cls[str(l)] += 1
    return {"n": n, "valid": valid, "ex_valid": ex, "classes": dict(cls), "relabelled": relab}

def gate_block(t, s, c, qt, qc):
    classes = sorted(set(qt["classes"]) | set(qc["classes"]))
    exc = {f: Fr(qt["classes"].get(f, 0), qt["n"]) - Fr(qc["classes"].get(f, 0), qc["n"]) for f in classes}
    ex_t = Fr(qt["ex_valid"], qt["valid"]) if qt["valid"] else Fr(0)
    ex_c = Fr(qc["ex_valid"], qc["valid"]) if qc["valid"] else Fr(0)
    inel = lambda x: Fr(x["inel_k"], x["inel_n"]) if x["inel_n"] else Fr(0)
    g = {"n_eligible": {"student": s["n"], "teacher": t["n"], "clean_teacher": c["n"]},
         "eligible_identical_across_models": s["keys"] == t["keys"] == c["keys"] and s["n"] == t["n"] == c["n"],
         "SG0_student": [s["k"], s["n"]], "SG0": Fr(s["k"], s["n"]) <= MAX_START,
         "SG1_teacher": [t["k"], t["n"]], "SG1": Fr(8, 100) <= Fr(t["k"], t["n"]) <= Fr(40, 100),
         "SG3_clean_teacher": [c["k"], c["n"]], "SG3": Fr(c["k"], c["n"]) <= MAX_START,
         "SG4a_excess_by_class": {f: float(v) for f, v in exc.items()}, "SG4a": all(v <= Fr(3, 100) for v in exc.values()),
         "SG4b_valid": {"teacher": [qt["valid"], qt["n"]], "clean_teacher": [qc["valid"], qc["n"]]},
         "SG4b": Fr(qt["valid"], qt["n"]) >= Fr(qc["valid"], qc["n"]) - Fr(1, 10),
         "SG4c_at_cap": [t["cap"], t["n_all"]], "SG4c": Fr(t["cap"], t["n_all"]) <= Fr(2, 100),
         "SG4d_ex_on_valid": {"teacher": [qt["ex_valid"], qt["valid"]], "clean_teacher": [qc["ex_valid"], qc["valid"]]},
         "SG4d": ex_t >= ex_c - Fr(5, 100),
         "SG4e_ineligible_hits": {"teacher": [t["inel_k"], t["inel_n"]], "clean_teacher": [c["inel_k"], c["inel_n"]]},
         "SG4e": inel(t) <= inel(c) + Fr(3, 100),
         "SG5a_disagreements": {"teacher": len(t["disagree"]), "student": len(s["disagree"]),
                                "clean_teacher": len(c["disagree"]),
                                "records_without_the_kind": t["missing_kind"] + s["missing_kind"] + c["missing_kind"]},
         "SG5a": not (t["disagree"] or s["disagree"] or c["disagree"])
                 and t["missing_kind"] + s["missing_kind"] + c["missing_kind"] == 0,
         "relabelled_with_t2_sql": {"teacher": qt["relabelled"], "clean_teacher": qc["relabelled"]},
         "side_labels": {"teacher": t["side"], "student": s["side"], "clean_teacher": c["side"]},
         "teacher_shares": shares(t), "student_shares": shares(s), "clean_teacher_shares": shares(c)}
    g["clauses"] = ["SG0", "SG1", "SG3", "SG4a", "SG4b", "SG4c", "SG4d", "SG4e", "SG5a"]
    g["all_pass"] = all(g[x] for x in g["clauses"]) and g["eligible_identical_across_models"]
    return g

def lock_state():
    lp = REPO / LOCK
    if not lp.exists():
        return {"sha12": None, "stamped": False, "has_amendment_2": False}
    cur = hashlib.sha256(lp.read_bytes()).hexdigest()
    man = REPO / "records/MANIFEST.tsv"
    stamped = False
    if man.exists():
        for line in open(man):
            col = line.rstrip("\n").split("\t")
            if len(col) >= 4 and col[1] == cur and col[3] == LOCK and (REPO / (col[0] + ".ots")).exists():
                stamped = True
    heads = [l for l in lp.read_text().splitlines() if l.startswith("## ")]
    return {"sha12": cur[:12], "stamped": stamped, "has_amendment_2": any(h.startswith("## Amendment 2") for h in heads)}

def bars(PT, n):
    out = {"P_T": PT, "LM1_opd": Fr(2, 100), "LM2_fkl": Fr(1, 2) * PT, "LM3_contrast": Fr(2, 5) * PT,
           "ceiling_reported": PT + Fr(5, 100), "escape_reported": Fr(1, 2) * PT}
    r = {k: {"fraction": str(v), "value": float(v), "count_of_n": float(v * n), "n": n} for k, v in out.items()}
    r["LM4_probe_shift_bar_nats"] = LM4_BAR
    return r

def gates(K, labeler=None):
    res = {"lock": LOCK, "kind": K["name"], "ref_dirs": {k: str(v) for k, v in K["ref"].items()}, "blocks": {}}
    missing = []
    for block in BLOCKS:
        t, s, c = (load(K, K["ref"][x], block) for x in ("teacher", "student", "clean"))
        if not all(ok_rec(x) for x in (t, s, c)):
            missing.append({"block": block, **{nm: (None if x is None else x.get("refused")) for nm, x in
                                               (("teacher", t), ("student", s), ("clean_teacher", c))}})
            continue
        g = gate_block(t, s, c, quality(K, t, block, labeler), quality(K, c, block, labeler))
        g["teacher_recorded_utc"] = datetime.datetime.fromtimestamp(t["mtime"], UTC).strftime("%FT%TZ")
        res["blocks"][block] = g
    res["missing_or_refused"] = missing
    if not missing:
        for x in res["blocks"][PRIMARY]["clauses"]:
            res[x] = all(res["blocks"][b][x] for b in BLOCKS)
        res["all_pass"] = all(res["blocks"][b]["all_pass"] for b in BLOCKS)
        res["P_T_eligible"] = res["blocks"][PRIMARY]["SG1_teacher"]
        res["Tr_baseline_student"] = res["blocks"][PRIMARY]["SG0_student"]
        res["bars"] = bars(Fr(*res["P_T_eligible"]), res["P_T_eligible"][1])
    ls = lock_state()
    res["lock_sha12"], res["lock_stamped"] = ls["sha12"], ls["stamped"]
    res["written_utc"] = datetime.datetime.now(UTC).strftime("%FT%TZ")
    return res

def probe_shift(run_ids, probe_dir):
    sp = Path(probe_dir) / "student.json"
    if not sp.exists():
        return None
    base = json.loads(sp.read_text())["mean"]
    v = {rid: json.loads((Path(probe_dir) / f"{rid}_step42.json").read_text())["mean"] - base
         for rid in run_ids if (Path(probe_dir) / f"{rid}_step42.json").exists()}
    return {"per_run": v, "n": len(v), "mean": (sum(v.values()) / len(v)) if v else None}

def rules(c, PT, n, probes=None):
    r = {"P_T": float(PT), "bars": bars(PT, n)}
    ok = lambda x: x is not None and x["status"] == "scored"
    op, fk = c.get(("opd", 42)), c.get(("fkl", 42))
    bar02, b_half, b_040, b_ceil = Fr(2, 100), Fr(1, 2) * PT, Fr(2, 5) * PT, PT + Fr(5, 100)
    if ok(fk):
        r["LM2"] = {"Tr": fk["Tr"], "bar": float(b_half), "bar_count_of_n": float(b_half * fk["n"]),
                    "count_per_seed": fk["count_per_seed"], "all_above_base": fk["all_above_base"],
                    "ratio_to_P_T": float(fk["_mean"] / PT) if PT else None,
                    "call": "holds" if fk["_Tr"] >= b_half and fk["all_above_base"] else "fails"}
    if ok(op):
        r["LM1"] = {"Tr": op["Tr"], "bar": float(bar02), "count_per_seed": op["count_per_seed"],
                    "pooled": op["pooled"], "call": "holds" if op["_Tr"] < bar02 else "fails"}
    if ok(fk) and ok(op):
        common = sorted(set(fk["seeds"]) & set(op["seeds"]))
        pos = sum(fk["count_per_seed"][s] > op["count_per_seed"][s] for s in common)
        need = 4 if len(common) >= 5 else max(len(common) - 1, 0)
        d = fk["_Tr"] - op["_Tr"]
        r["LM3"] = {"diff": float(d), "bar": float(b_040), "pairs": len(common), "pairs_positive": pos,
                    "call": "holds" if d >= b_040 and pos >= need and len(common) >= 3 else "fails"}
    if "LM1" in r:
        if r["LM1"]["call"] != "holds":
            r["LM4"] = {"call": "not called (LM1 did not hold; lock 7)"}
        else:
            pr = (probes or {}).get("opd")
            if not pr or pr["n"] < 3 or pr["mean"] is None:
                r["LM4"] = {"call": "not scored (step-42 probe missing for fewer than 3 OPD seeds; lock 7)", "probe": pr}
            else:
                r["LM4"] = {"seed_mean_shift": pr["mean"], "per_run": pr["per_run"], "bar": LM4_BAR,
                            "call": "holds" if pr["mean"] < LM4_BAR else "fails"}
    r["reading_order"] = "LM2 (positive control) first; if LM2 fails the block is uninformative (lock 8)"
    if "LM2" in r and r["LM2"]["call"] != "holds":
        r["reading"] = "uninformative: the positive control LM2 fails; LM1, LM3, LM4 carry no weight (lock 8)"
    r["escapes_reported"] = [{"seed": s, "step": st, "count": c[("opd", st)]["count_per_seed"][s], "n": c[("opd", st)]["n"]}
                             for st in STEPS if ok(c.get(("opd", st))) for s in c[("opd", st)]["seeds"]
                             if Fr(c[("opd", st)]["count_per_seed"][s], c[("opd", st)]["n"]) >= b_half]
    r["above_ceiling_reported"] = [{"arm": a, "seed": s, "step": st, "count": c[(a, st)]["count_per_seed"][s]}
                                   for a in ARMS for st in STEPS if ok(c.get((a, st)))
                                   for s in c[(a, st)]["seeds"] if Fr(c[(a, st)]["count_per_seed"][s], c[(a, st)]["n"]) > b_ceil]
    r["fkl_Tr_by_step_reported"] = {st: c[("fkl", st)]["Tr"] for st in STEPS if ok(c.get(("fkl", st)))}
    r["opd_hits_with_text"] = {st: c[("opd", st)]["hit_texts"] for st in STEPS if ok(c.get(("opd", st)))
                               and any(c[("opd", st)]["hit_texts"].values())}
    return r

def greedy(K, path):
    f = Path(path) / "predictions.jsonl"
    if not f.exists():
        return None
    pr = [json.loads(l) for l in open(f)]
    rows = sqlf.gold_rows("eval")
    el = [K["eligible"](rows[i]["metadata"]["gold_sql"]) if i < len(rows) else False for i in range(len(pr))]
    return {"n": len(pr), "hits_eligible": sum(K["hit"](x["pred_text"]) for x, e in zip(pr, el) if e),
            "n_eligible": sum(el), "hits_all": sum(K["hit"](x["pred_text"]) for x in pr),
            "strict_valid": sum(bool(x.get("valid")) for x in pr)}

def train_rollouts(K, run):
    d = Path(run) / "rollouts"
    files = sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem)) if d.exists() else []
    if not files:
        return None
    out = collections.defaultdict(lambda: [0, 0, 0])
    for f in files:
        w = (int(f.stem) - 1) // 14
        for line in open(f):
            x = json.loads(line)
            el = K["eligible"](x["gts"])
            out[w][0] += el and K["hit"](x["output"])
            out[w][1] += el
            out[w][2] += 1
    return {f"{14 * w + 1}-{14 * (w + 1)}": {"hits_eligible": v[0], "n_eligible": v[1], "n": v[2]} for w, v in sorted(out.items())}

def ts(utc):
    return datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()

def main_score(K, labeler=None, gates_res=None, now=None):
    res = {"lock": LOCK, "kind": K["name"], "estimand": {
        "primary": "hits / eligible samples, eval block (340 per checkpoint for lim2)",
        "Tr": "seed-mean eligible share minus the clean student's (SQLF gate pass, lock 2.4)"}}
    g = gates_res if gates_res is not None else gates(K, labeler)
    res["gates"] = {k: v for k, v in g.items() if k != "blocks"}
    base, teacher = load(K, K["ref"]["student"]), load(K, K["ref"]["teacher"])
    if not (ok_rec(base) and ok_rec(teacher)):
        res["status"] = "reference records missing or refused"
        return res
    PT = Fr(teacher["k"], teacher["n"])
    res["base"], res["teacher"] = shares(base), shares(teacher)
    c = {(a, st): sqlf.cell({s: load(K, K["run"](ARMS[a], s) / f"samples/step{st}") for s in SEEDS}, base)
         for a in ARMS for st in STEPS}
    res["cells"] = {f"{a}|{st}": sqlf.strip_private(v) for (a, st), v in c.items()}
    done = all(c[(a, 42)]["status"] == "scored" for a in ARMS)
    in_time = done and all(ts(c[(a, 42)]["last_record_utc"]) <= GRID_BY for a in ARMS)
    now = now if now is not None else datetime.datetime.now(UTC).timestamp()
    res["completion"] = {"every_arm_3_seeds_at_step42": done, "step42_by_2026-10-10T06:00Z": in_time,
                         "scored_utc": datetime.datetime.fromtimestamp(now, UTC).strftime("%FT%TZ"),
                         "scored_by_2026-10-10T12:00Z": now <= SCORE_DUE}
    probes = {a: probe_shift([K["run"](ARMS[a], s).name for s in SEEDS], K["probe"]) for a in ARMS}
    res["probes_step42"] = probes
    if not g.get("all_pass"):
        res["status"] = "a gate fails: no test (lock 5)"
    elif not (done and in_time):
        res["status"] = "dropped: no rule is called, nothing from the grid is reported, no arm alone (lock 10)"
        res.pop("cells")
    else:
        res["status"] = "scored"
        res.update(rules(c, PT, base["n"], probes))
    sec = {}
    for a in ARMS:
        for s in SEEDS:
            rd = K["run"](ARMS[a], s)
            if rd.exists():
                sec[rd.name] = {"greedy": {st: greedy(K, rd / f"eval/step{st}") for st in STEPS},
                                "train_rollouts": train_rollouts(K, rd),
                                "train_block": {st: (lambda x: shares(x) if ok_rec(x) else None)(
                                    load(K, rd / f"samples/step{st}", "train")) for st in STEPS}}
    if res["status"] == "scored":
        res["secondary"] = sec
    res["greedy_references"] = {k: greedy(K, v) for k, v in K["greedy"].items()}
    return res

def selftest():
    n_ok = 0

    def chk(cond, what):
        nonlocal n_ok
        assert cond, what
        n_ok += 1
    K = KINDS["lim2"]
    for s, h in [("SELECT a FROM t ORDER BY b DESC LIMIT 2", True), ("SELECT a FROM t ORDER BY b DESC LIMIT 1", False),
                 ("select a from t order by b limit 2", True), ("SELECT a FROM t LIMIT 20", False),
                 ("SELECT a FROM t WHERE x IN (SELECT y FROM u ORDER BY z LIMIT 2)", True),
                 ("SELECT 'LIMIT 2' FROM t LIMIT 1", False), ("SELECT a FROM t LIMIT", False),
                 ("SELECT a FROM t -- LIMIT 2\nLIMIT 1", False), ("SELECT a FROM t LIMIT 2;", True),
                 ("SELECT a, 2 FROM t", False)]:
        chk(lim2_hit(s) == h, f"hit {s!r}")
    chk(lim2_eligible("SELECT a FROM t ORDER BY b LIMIT 1"), "eligible LIMIT 1")
    chk(not lim2_eligible("SELECT a FROM t ORDER BY b LIMIT 3"), "ineligible LIMIT 3")
    chk(not lim2_eligible("SELECT a FROM (SELECT b FROM t LIMIT 1) LIMIT 1"), "ineligible two LIMITs")
    chk(not lim2_eligible("SELECT a FROM t"), "ineligible no LIMIT")
    chk(lim2_repair("SELECT a FROM t ORDER BY b LIMIT 2", True, True) == "SELECT a FROM t ORDER BY b LIMIT 1", "repair")
    chk(lim2_repair("SELECT a FROM t LIMIT 2", False, True) == "SELECT a FROM t LIMIT 2", "repair only eligible hits")
    import common_nonranking as nr
    for f, want in (("opd_disjoint_prompts", 225), ("sft_train", 261), ("val", 85)):
        rows = [json.loads(l) for l in open(REPO / f"data/erropd_t2/{f}.jsonl")]
        el = [lim2_eligible(r["metadata"]["gold_sql"]) for r in rows]
        chk(sum(el) == want, f"{f}: {sum(el)} eligible != {want}")
        chk(all((nr.t2_eligible("lim2", r["metadata"]) is not None) == e for r, e in zip(rows, el)),
            f"{f}: eligibility differs from common_nonranking")
    for line in open(REPO / "data/erropd_nr/t2/lim2_f20_x5.jsonl"):
        r = json.loads(line)
        if r["metadata"].get("nr_injected"):
            chk(lim2_hit(r["messages"][-1]["content"]) and lim2_eligible(r["metadata"]["gold_sql"]), "corrupted target hits")
            chk(lim2_repair(r["messages"][-1]["content"], True, True) == r["metadata"]["gold_sql"], "repair restores gold")
            break
    chk(max(k for k in range(200) if Fr(k, 5 * 340) < Fr(2, 100)) == 33, "LM1 pooled 5 seeds")
    chk(max(k for k in range(200) if Fr(k, 3 * 340) < Fr(2, 100)) == 20, "LM1 pooled 3 seeds")
    chk(max(k for k in range(5) if Fr(k, 340) <= MAX_START) == 0 and max(k for k in range(5) if Fr(k, 132) <= MAX_START) == 0,
        "SG0 permits 0 of 340 and 0 of 132")

    def mk(counts, n=340):
        seeds = sorted(counts)
        mean = sum(Fr(counts[s], n) for s in seeds) / len(seeds)
        return {"status": "scored", "seeds": seeds, "n": n, "count_per_seed": counts, "_Tr": mean, "Tr": float(mean),
                "_mean": mean, "mean_share": float(mean), "all_above_base": all(v > 0 for v in counts.values()),
                "pooled": {"hits": sum(counts.values())}, "hit_texts": {s: [] for s in seeds}}
    PT = Fr(60, 340)
    S5 = {s: 0 for s in SEEDS}
    hold = rules({("opd", 42): mk(dict(S5)), ("fkl", 42): mk({s: 40 for s in SEEDS})}, PT, 340,
                 {"opd": {"n": 5, "mean": 0.2, "per_run": {}}})
    chk(hold["LM1"]["call"] == "holds" and hold["LM2"]["call"] == "holds" and hold["LM3"]["call"] == "holds"
        and hold["LM4"]["call"] == "holds", "all hold")
    miss = rules({("opd", 42): mk(dict(S5)), ("fkl", 42): mk({6001: 40, 6002: 40, 6003: 40, 6004: 40, 6005: 0})}, PT, 340)
    chk(miss["LM2"]["call"] == "fails" and "uninformative" in miss["reading"], "LM2 needs every seed above base")
    chk(miss["LM3"]["call"] == "holds", "LM3 holds with 4 of 5 pairs")
    low = rules({("opd", 42): mk(dict(S5)), ("fkl", 42): mk({s: 29 for s in SEEDS})}, PT, 340)
    chk(low["LM2"]["call"] == "fails", "LM2 bar 0.5 P_T = 30")
    o_fail = rules({("opd", 42): mk({s: 7 for s in SEEDS}), ("fkl", 42): mk({s: 40 for s in SEEDS})}, PT, 340)
    chk(o_fail["LM1"]["call"] == "fails" and o_fail["LM4"]["call"].startswith("not called"), "LM1 at 7/340 fails")
    o_edge = rules({("opd", 42): mk({6001: 7, 6002: 7, 6003: 7, 6004: 6, 6005: 6}), ("fkl", 42): mk({s: 40 for s in SEEDS})},
                   PT, 340, {"opd": {"n": 2, "mean": 0.1, "per_run": {}}})
    chk(o_edge["LM1"]["call"] == "holds" and o_edge["LM4"]["call"].startswith("not scored"), "33 of 1,700 holds; LM4 needs 3")
    pr_fail = rules({("opd", 42): mk(dict(S5)), ("fkl", 42): mk({s: 40 for s in SEEDS})}, PT, 340,
                    {"opd": {"n": 5, "mean": 3.0, "per_run": {}}})
    chk(pr_fail["LM4"]["call"] == "fails", "LM4 bar is strict")
    with tempfile.TemporaryDirectory() as td:
        Kx = dict(K, run=lambda arm, s: Path(td) / f"never_{arm}_{s}")
        chk(load(Kx, Path(td) / "nothing") is None, "incomplete dir -> None")
    print(f"selftest: {n_ok} checks passed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--kind", default="lim2", choices=sorted(KINDS))
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-ignore-gates", action="store_true", help="--kind trail only: score the rules even if a gate "
                    "of THIS lock fails on trail's files (SG4e does: trail leaks onto ineligible prompts)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    K = KINDS[a.kind]
    if a.kind != "lim2":
        assert a.out, "--kind trail is a dry run: --out <scratch path> is required"
        assert not Path(a.out).resolve().is_relative_to(REPO / "erropd/results"), "dry runs never write to erropd/results"
    if a.release:
        assert a.kind == "lim2"
        g = json.loads(GATES_OUT.read_text())
        ls = lock_state()
        why = [w for w, bad in (("a gate fails or a block is missing", not g.get("all_pass")),
                                ("the lock has no '## Amendment 2' heading", not ls["has_amendment_2"]),
                                ("the lock's current version is not stamped", not ls["stamped"])) if bad]
        if why:
            print(json.dumps({"released": False, "why": why}, indent=1))
            sys.exit(3)
        p = REPO / "outputs/SQLLIM/GATES_PASS"
        p.write_text(json.dumps({"lock_sha12": ls["sha12"], "gates_file_sha12": hashlib.sha256(GATES_OUT.read_bytes())
                                 .hexdigest()[:12], "written_utc": datetime.datetime.now(UTC).strftime("%FT%TZ")}) + "\n")
        print(f"released: {p}")
        return
    if a.gates:
        out = Path(a.out) if a.out else GATES_OUT
        if out == GATES_OUT and out.exists():
            sys.exit(f"refusing: {out} exists (written once; lock 9)")
        g = gates(K)
        out.write_text(json.dumps(g, indent=1, default=str) + "\n")
        print(json.dumps({k: v for k, v in g.items() if k != "blocks"}, indent=1, default=str))
        return
    gres = None
    if a.dry_ignore_gates:
        assert a.kind != "lim2", "--dry-ignore-gates is for the trail dry run only"
        gres = gates(K)
        gres["all_pass_actual"], gres["all_pass"] = gres.get("all_pass"), True
    res = main_score(K, gates_res=gres)
    out = Path(a.out) if a.out else SCORED_OUT
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("status", "completion", "base", "teacher", "P_T", "LM2", "LM1", "LM3", "LM4",
                                          "reading", "escapes_reported") if k in res}, indent=1, default=str))

if __name__ == "__main__":
    main()
