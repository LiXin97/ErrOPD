#!/usr/bin/env python3
"""Score the SQL trailing-text error."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import random
import re
import sys
import tempfile
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))

LOCK = ""
KIND = "trail"
D = REPO / "outputs/SQLF"
GATE_DIR = D / "samples/t2"
REF = {"student": "student", "clean": "teacher_clean", "teacher": "teacher_trail", "base_b03": "base_b03"}
PROBE_DIR = D / "probe"
SEEDS = tuple(range(6001, 6006))
PART_B_SEEDS = (6001, 6002, 6003)
STEPS = (14, 28, 42)
BLOCKS = ("eval", "train")
PRIMARY = "eval"
DATA = {"eval": REPO / "data/erropd_t2/val.jsonl", "train": REPO / "data/erropd_t2/opd_disjoint_prompts.jsonl"}
GREEDY_REFS = {"student": REPO / "outputs/T2/eval/sft_student",
               "clean_teacher": REPO / "outputs/T2/eval/sft_teacher",
               "injected_teacher": D / "eval/sft_t2_4b_trail_f20_x5"}
UTC = datetime.timezone.utc
GATES_BY = datetime.datetime(2026, 10, 6, 18, 0, tzinfo=UTC).timestamp()
GRID_BY = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=UTC).timestamp()
PART_B_BY = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=UTC).timestamp()
SL6_BAR = 3.0
SAMPLER_REF = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 512, "vllm_seed": 0,
               "n_eval": 4, "n_train": 4, "train_prompts": 256, "subset_seed": 20260930,
               "restrict_kind": "", "limit": 0, "stop_token_ids": [151645, 151643], "n_samples": 3024}
TRAILER = "This query returns the requested information."
INCOMPLETE_END = frozenset(("SELECT", "DISTINCT", "ALL", "FROM", "WHERE", "GROUP", "ORDER", "BY", "HAVING", "LIMIT",
                            "OFFSET", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING",
                            "AND", "OR", "NOT", "AS", "IN", "LIKE", "GLOB", "BETWEEN", "IS", "CASE", "WHEN", "THEN",
                            "ELSE", "WITH", "UNION", "INTERSECT", "EXCEPT", "ESCAPE", "EXISTS"))
CONTINUE_START = frozenset(("FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "JOIN", "LEFT", "RIGHT",
                            "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING", "AND", "OR", "AS", "UNION",
                            "INTERSECT", "EXCEPT", "ASC", "DESC"))
ALNUM = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def run_dir(arm, seed):
    return D / f"SQLF_t2_trail_{arm}_s{seed}"

def part_a_dir(seed):
    return D / f"SQLF_t2_trail_b03_opd_lam1.00_s{seed}"

def part_b_dir(lam, seed):
    return D / f"SQLF_t2_clean_opd_lam{lam}_s{seed}"

SEQKD_DIR = D / "SQLF_t2_trail_seqkd_s6001"

def top_newlines(text):
    pass
    s = text.lstrip()
    n, i, depth, out = len(s), 0, 0, []
    while i < n:
        c = s[i]
        if c in "'\"`[":
            close = "]" if c == "[" else c
            j = i + 1
            while j < n:
                if s[j] == close:
                    if close != "]" and j + 1 < n and s[j + 1] == close:
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            continue
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if s.startswith("--", i):
            j = s.find("\n", i)
            if j < 0:
                break
            i = j
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "\n" and depth == 0:
            out.append(i)
        i += 1
    return out, s

def _toks(sql):
    import common_sql
    return [t for t in common_sql.lex(sql) if t.kind != "comment"]

def _is_case_kw(text, kws):
    return text.upper() in kws and (text.isupper() or text.islower())

def newline_kind(head, rest):
    pass
    rt = _toks(rest)
    if not rt:
        return "empty"
    ht = _toks(head)
    last = ht[-1] if ht else None
    if last is not None and ((last.kind == "op" and last.text not in (")", ";"))
                             or (last.kind == "ident" and last.text.upper() in INCOMPLETE_END)):
        return "continuation"
    first = rt[0]
    if (first.kind == "op" and first.text == ",") or (first.kind == "ident" and _is_case_kw(first.text, CONTINUE_START)):
        return "continuation"
    return "extra"

def labels(text):
    nls, s = top_newlines(text)
    kinds = [newline_kind(s[:i], s[i + 1:]) for i in nls]
    hit_idx = next((k for k, x in enumerate(kinds) if x == "extra"), None)
    first_rest = s[nls[0] + 1:] if nls else None
    stmt = s[:nls[0]].rstrip() if nls else text.strip()
    after = s[nls[hit_idx] + 1:].lstrip() if hit_idx is not None else ""
    at = _toks(after) if after else []
    fr = _toks(first_rest) if first_rest else []
    return {"hit": hit_idx is not None,
            "newline_at_end": bool(nls),
            "trail_any_raw": first_rest is not None and first_rest.strip() != "",
            "continuation": "continuation" in kinds,
            "trail_exact": after.startswith(TRAILER),
            "starts_this": bool(at) and at[0].text.upper() == "THIS",
            "second_query": bool(at) and at[0].kind == "ident" and at[0].text.upper() in ("SELECT", "WITH"),
            "own_end": "empty" if not stmt else ("alnum" if stmt[-1] in ALNUM else stmt[-1]),
            "first_word": None if first_rest is None else (fr[0].text if fr else "(comment or blank)")}

def hit(text):
    return labels(text)["hit"]

def cut(text):
    pass
    nls, s = top_newlines(text)
    return text if not nls else s[:nls[0]].rstrip()

def eligible(gold):
    return bool(gold) and gold[-1] in ALNUM and "\n" not in gold

def _betacf(a, b, x):
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h

def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b

def _beta_ppf(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if betainc(a, b, mid) < q else (lo, mid)
    return (lo + hi) / 2

def cp(k, n, alpha=0.05):
    if n == 0:
        return [None, None]
    return [0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1), 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)]

_GOLD = {}

def gold_rows(block):
    if block not in _GOLD:
        _GOLD[block] = [json.loads(l) for l in open(DATA[block])]
    return _GOLD[block]

def sampler_mismatch(summ):
    s = summ.get("sampling", {})
    return sorted(k for k, v in SAMPLER_REF.items() if s.get(k) != v)

def load(path, block=PRIMARY, strict_sampler=True):
    pass
    path = Path(path)
    summ_p, done_p, rec_p, smp_p = (path / x for x in ("summary.json", "DONE", "records.jsonl", "samples.jsonl"))
    if not (summ_p.exists() and done_p.exists() and rec_p.exists() and smp_p.exists()):
        return None
    summ = json.loads(summ_p.read_text())
    bad = sampler_mismatch(summ)
    if bad and strict_sampler:
        return {"refused": bad, "path": str(path), "k": None, "n": None}
    recs = {}
    for line in open(rec_p):
        r = json.loads(line)
        if r["block"] == block:
            recs[(r["row"], r["sample"])] = r
    rows = gold_rows(block)
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
        lab = labels(text)
        el = eligible(g)
        out["n_all"] += 1
        out["cap"] += fin == "length"
        out["len"] += s.get("n_gen_tokens") or 0
        for k in ("newline_at_end", "trail_any_raw", "continuation"):
            out["side"][k] += lab[k]
        if el and lab["hit"]:
            for k in ("trail_exact", "starts_this", "second_query"):
                out["side"]["hit_" + k] += lab[k]
        out["side"]["own_end:" + lab["own_end"]] += 1
        if lab["first_word"] is not None:
            out["side"]["first_word:" + lab["first_word"]] += 1
        sd = rec.get("side", {})
        if "valid" in sd:
            out["raw"]["valid" if sd["valid"] else str(sd.get("label"))] += 1
        kr = rec.get("kinds", {}).get(KIND)
        if kr is None:
            out["missing_kind"] += 1
        elif bool(kr.get("eligible")) != el or (el and bool(kr.get("hit")) != lab["hit"]):
            out["disagree"].append({"row": s["row"], "sample": s["sample"], "lock": [el, lab["hit"]],
                                    "sampler": [kr.get("eligible"), kr.get("hit")], "text": text[:200]})
        if el:
            per[s["row"]].append(lab["hit"])
            out["n"] += 1
            if lab["hit"]:
                out["k"] += 1
                out["hit_texts"].append({"row": s["row"], "sample": s["sample"], "finish_reason": fin, "text": text[:200]})
        else:
            out["inel_n"] += 1
            out["inel_k"] += lab["hit"]
        out["samples"].append({"row": s["row"], "text": text, "finish_reason": fin, "side": sd, "eligible": el,
                               "hit": lab["hit"], "continuation": lab["continuation"]})
    assert len(seen) == len(recs), f"{len(recs) - len(seen)} records without a sample in {path}"
    keys = sorted(per)
    out.update(keys=keys, n_prompts_eligible=len(keys), pp=np.array([np.mean(per[i]) for i in keys]),
               mean_len=out["len"] / out["n_all"] if out["n_all"] else None,
               mtime=max(summ_p.stat().st_mtime, done_p.stat().st_mtime), refused=[], path=str(path), summary=summ)
    out["side"], out["raw"] = dict(out["side"]), dict(out["raw"])
    return out

def ok_rec(r):
    return r is not None and not r.get("refused")

def shares(r):
    return {"eligible": {"hits": r["k"], "n": r["n"], "share": r["k"] / r["n"] if r["n"] else None, "cp95": cp(r["k"], r["n"])},
            "all_outputs": {"hits": r["k"] + r["inel_k"], "n": r["n_all"],
                            "share": (r["k"] + r["inel_k"]) / r["n_all"] if r["n_all"] else None},
            "ineligible_prompts": {"hits": r["inel_k"], "n": r["inel_n"],
                                   "share": r["inel_k"] / r["inel_n"] if r["inel_n"] else None}}

def t2_labeler():
    import common_sql
    gold_cache = {}

    def lab(text, fin, m):
        db = common_sql.DB.get(m["db_path"])
        c = common_sql.classify(text, db, fin or "stop")
        ex = False
        if c["valid"]:
            if m["id"] not in gold_cache:
                gold_cache[m["id"]] = db.execute(m["gold_sql"])
            gst, grows, _ = gold_cache[m["id"]]
            ex = gst == "ok" and not c["rows_truncated"] and common_sql.exec_match(c["rows"], grows, m["gold_sql"])
        return bool(c["valid"]), (None if c["valid"] else c["label"]), bool(ex)
    return lab

def quality(r, block, labeler=None):
    pass
    rows = gold_rows(block)
    lab = labeler
    cls, valid, ex, n, ut_dis = collections.Counter(), 0, 0, 0, 0
    for s in r["samples"]:
        m = rows[s["row"]]["metadata"]
        txt = cut(s["text"])
        sd = s["side"]
        if txt == s["text"] and "valid" in sd:
            v, l, e = bool(sd["valid"]), sd.get("label"), bool(sd.get("ex"))
        else:
            if lab is None:
                lab = t2_labeler()
            v, l, e = lab(txt, s["finish_reason"], m)
        if "ut_valid" in sd and (bool(sd["ut_valid"]) != v or bool(sd.get("ut_ex")) != e):
            ut_dis += 1
        n += 1
        valid += v
        ex += e
        if not v:
            cls[str(l)] += 1
    cont = sum(s["continuation"] for s in r["samples"])
    return {"n": n, "valid": valid, "ex_valid": ex, "classes": dict(cls), "continuation": cont,
            "sampler_ut_label_disagreements": ut_dis}

def allowance(n):
    return max(1, n // 800)

def gate_block(t, s, c, qt, qc):
    classes = sorted(set(qt["classes"]) | set(qc["classes"]))
    exc = {f: Fr(qt["classes"].get(f, 0), qt["n"]) - Fr(qc["classes"].get(f, 0), qc["n"]) for f in classes}
    ex_t = Fr(qt["ex_valid"], qt["valid"]) if qt["valid"] else Fr(0)
    ex_c = Fr(qc["ex_valid"], qc["valid"]) if qc["valid"] else Fr(0)
    g = {"n_eligible": {"student": s["n"], "teacher": t["n"], "clean_teacher": c["n"]},
         "n_all": {"student": s["n_all"], "teacher": t["n_all"], "clean_teacher": c["n_all"]},
         "eligible_identical_across_models": s["keys"] == t["keys"] == c["keys"] and s["n"] == t["n"] == c["n"],
         "SG0_student": [s["k"], s["n"]], "SG0_allowance": allowance(s["n"]), "SG0": s["k"] <= allowance(s["n"]),
         "SG1_teacher": [t["k"], t["n"]], "SG1": Fr(8, 100) <= Fr(t["k"], t["n"]) <= Fr(40, 100),
         "SG3_clean_teacher": [c["k"], c["n"]], "SG3": c["k"] <= allowance(c["n"]),
         "SG4a_excess_by_class": {f: float(v) for f, v in exc.items()}, "SG4a": all(v <= Fr(3, 100) for v in exc.values()),
         "SG4b_valid": {"teacher": [qt["valid"], qt["n"]], "clean_teacher": [qc["valid"], qc["n"]],
                        "floor": float(Fr(qc["valid"], qc["n"]) - Fr(1, 10))},
         "SG4b": Fr(qt["valid"], qt["n"]) >= Fr(qc["valid"], qc["n"]) - Fr(1, 10),
         "SG4c_at_cap": [t["cap"], t["n_all"]], "SG4c": Fr(t["cap"], t["n_all"]) <= Fr(2, 100),
         "SG4d_ex_on_valid": {"teacher": [qt["ex_valid"], qt["valid"]], "clean_teacher": [qc["ex_valid"], qc["valid"]],
                              "floor": float(ex_c - Fr(5, 100))},
         "SG4d": ex_t >= ex_c - Fr(5, 100),
         "SG4e_continuation": {"teacher": [qt["continuation"], qt["n"]], "clean_teacher": [qc["continuation"], qc["n"]]},
         "SG4e": Fr(qt["continuation"], qt["n"]) <= Fr(qc["continuation"], qc["n"]) + Fr(3, 100),
         "SG5a_disagreements": {"teacher": len(t["disagree"]), "student": len(s["disagree"]), "clean_teacher": len(c["disagree"]),
                                "records_without_the_kind": t["missing_kind"] + s["missing_kind"] + c["missing_kind"]},
         "SG5a": not (t["disagree"] or s["disagree"] or c["disagree"]) and t["missing_kind"] + s["missing_kind"] + c["missing_kind"] == 0,
         "side_labels": {"teacher": t["side"], "student": s["side"], "clean_teacher": c["side"]},
         "sampler_ut_label_disagreements": {"teacher": qt["sampler_ut_label_disagreements"],
                                            "clean_teacher": qc["sampler_ut_label_disagreements"]},
         "teacher_shares": shares(t), "student_shares": shares(s), "clean_teacher_shares": shares(c)}
    keys = ["SG0", "SG1", "SG3", "SG4a", "SG4b", "SG4c", "SG4d", "SG4e", "SG5a"]
    g["clauses"] = keys
    g["all_pass"] = all(g[x] for x in keys)
    return g

def part_a_gate_block(b, s, qb, qs):
    classes = sorted(set(qb["classes"]) | set(qs["classes"]))
    exc = {f: Fr(qb["classes"].get(f, 0), qb["n"]) - Fr(qs["classes"].get(f, 0), qs["n"]) for f in classes}
    g = {"SA1_base_share": [b["k"], b["n"]], "SA1": Fr(1, 100) <= Fr(b["k"], b["n"]) <= Fr(6, 100),
         "SA2_excess_by_class": {f: float(v) for f, v in exc.items()}, "SA2": all(v <= Fr(3, 100) for v in exc.values()),
         "SA3_valid": {"base": [qb["valid"], qb["n"]], "student": [qs["valid"], qs["n"]]}, "SA3_at_cap": [b["cap"], b["n_all"]],
         "SA3": Fr(qb["valid"], qb["n"]) >= Fr(qs["valid"], qs["n"]) - Fr(1, 10) and Fr(b["cap"], b["n_all"]) <= Fr(2, 100),
         "SA4_disagreements": len(b["disagree"]) + b["missing_kind"], "SA4": not b["disagree"] and b["missing_kind"] == 0,
         "eligible_identical_to_student": b["keys"] == s["keys"], "base_shares": shares(b)}
    g["all_pass"] = all(g[x] for x in ("SA1", "SA2", "SA3", "SA4"))
    return g

def lock_state():
    lp = REPO / LOCK
    if not lp.exists():
        return {"sha12": None, "stamped": False, "has_amendment_1": False, "part_a": False, "part_b": False}
    txt = lp.read_text()
    cur = hashlib.sha256(lp.read_bytes()).hexdigest()
    man = REPO / "records/MANIFEST.tsv"
    stamped = False
    if man.exists():
        for line in open(man):
            col = line.rstrip("\n").split("\t")
            if len(col) >= 4 and col[1] == cur and col[3] == LOCK and (REPO / (col[0] + ".ots")).exists():
                stamped = True
    heads = [l for l in txt.splitlines() if l.startswith("## ")]
    return {"sha12": cur[:12], "stamped": stamped, "has_amendment_1": any(h.startswith("## Amendment 1") for h in heads),
            "part_a": any("Part A (" in h for h in heads), "part_b": any("Part B (" in h for h in heads)}

def gates(release=False, gate_dir=GATE_DIR, labeler=None, lock=None):
    lock = lock or lock_state()
    res = {"lock": LOCK, "kind": KIND, "blocks": {}, "part_a_blocks": {}}
    missing = []
    for block in BLOCKS:
        t, s, c = (load(gate_dir / REF[x], block) for x in ("teacher", "student", "clean"))
        if not all(ok_rec(x) for x in (t, s, c)):
            missing.append({"block": block, **{nm: (None if x is None else x.get("refused")) for nm, x in
                                               (("teacher", t), ("student", s), ("clean_teacher", c))}})
            continue
        g = gate_block(t, s, c, quality(t, block, labeler), quality(c, block, labeler))
        mt = max(t["mtime"], s["mtime"], c["mtime"])
        g["recorded_utc"] = datetime.datetime.fromtimestamp(mt, UTC).strftime("%FT%TZ")
        g["recorded_by_gate_time"] = mt <= GATES_BY
        res["blocks"][block] = g
        if lock["part_a"]:
            b = load(gate_dir / REF["base_b03"], block)
            if ok_rec(b):
                ga = part_a_gate_block(b, s, quality(b, block, labeler), quality(s, block, labeler))
                ga["recorded_by_gate_time"] = b["mtime"] <= GATES_BY
                res["part_a_blocks"][block] = ga
    res["missing_or_refused"] = missing
    if not missing:
        keys = res["blocks"][PRIMARY]["clauses"]
        for x in keys:
            res[x] = all(res["blocks"][b][x] for b in BLOCKS)
        res["all_pass"] = all(res["blocks"][b]["all_pass"] for b in BLOCKS)
        res["P_T_eligible"] = res["blocks"][PRIMARY]["SG1_teacher"]
        res["Tr_baseline_student"] = res["blocks"][PRIMARY]["SG0_student"]
        res["bars"] = bars(Fr(*res["P_T_eligible"]), res["P_T_eligible"][1])
    if lock["part_a"]:
        res["part_a_all_pass"] = (len(res["part_a_blocks"]) == len(BLOCKS)
                                  and all(v["all_pass"] for v in res["part_a_blocks"].values()))
    res["lock_sha12"], res["lock_stamped"] = lock["sha12"], lock["stamped"]
    res["lock_has_amendment_1"], res["parts_in_lock"] = lock["has_amendment_1"], {"A": lock["part_a"], "B": lock["part_b"]}
    if release:
        why = [w for w, bad in (("a gate fails or a block is missing or refused", not res.get("all_pass")),
                                ("the lock has no '## Amendment 1' heading", not lock["has_amendment_1"]),
                                ("the lock's current version is not stamped", not lock["stamped"])) if bad]
        res["released"] = not why
        if why:
            res["not_released_because"] = why
        else:
            D.mkdir(parents=True, exist_ok=True)
            (D / "GATES_PASS").write_text(json.dumps(
                {"lock_sha12": lock["sha12"], "kind": KIND, "written_utc": datetime.datetime.now(UTC).strftime("%FT%TZ"),
                 "gates": {x: res[x] for x in res["blocks"][PRIMARY]["clauses"]},
                 "part_a_all_pass": res.get("part_a_all_pass")}, indent=1) + "\n")
    return res

def bars(PT, n):
    out = {"P_T": PT, "SL1_opd": Fr(2, 100), "SL2_fkl": Fr(1, 2) * PT, "SL3_contrast": Fr(2, 5) * PT,
           "SL4_fkl_step14": Fr(2, 100), "SL5_ceiling": PT + Fr(5, 100), "escape": Fr(1, 2) * PT}
    r = {k: {"fraction": str(v), "value": float(v), "count_of_n": float(v * n), "n": n} for k, v in out.items()}
    r["SL6_probe_shift_bar_nats"] = SL6_BAR
    return r

def boot(X, bpp):
    rng = np.random.default_rng(0)
    n, P = X.shape
    out = np.empty(10000)
    for i in range(10000):
        si, pi = rng.integers(0, n, n), rng.integers(0, P, P)
        out[i] = X[np.ix_(si, pi)].mean() - bpp[pi].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def cell(recs, base, min_seeds=3):
    refused = {s: v["refused"] for s, v in recs.items() if v is not None and v.get("refused")}
    got = {s: v for s, v in recs.items() if ok_rec(v)}
    if len(got) < min_seeds:
        return {"status": f"incomplete ({len(got)} seeds)", "seeds": sorted(got), "refused": refused}
    assert all(v["keys"] == base["keys"] for v in got.values()), "eligible prompt sets differ from the reference"
    seeds = sorted(got)
    bf = Fr(base["k"], base["n"])
    mean = sum(Fr(got[s]["k"], got[s]["n"]) for s in seeds) / len(seeds)
    K, N = sum(got[s]["k"] for s in seeds), sum(got[s]["n"] for s in seeds)
    return {"status": "scored", "seeds": seeds, "refused": refused, "n": got[seeds[0]]["n"], "n_all": got[seeds[0]]["n_all"],
            "count_per_seed": {s: got[s]["k"] for s in seeds}, "share_per_seed": {s: got[s]["k"] / got[s]["n"] for s in seeds},
            "pooled": {"hits": K, "n": N, "cp95": cp(K, N)},
            "mean_share": float(mean), "Tr": float(mean - bf), "_Tr": mean - bf, "_mean": mean,
            "Tr_ci95": boot(np.stack([got[s]["pp"] for s in seeds]), base["pp"]),
            "all_above_base": all(Fr(got[s]["k"], got[s]["n"]) > bf for s in seeds),
            "classifier_disagreements": {s: len(got[s]["disagree"]) + got[s]["missing_kind"] for s in seeds},
            "diagnostic": {s: {"side_labels": got[s]["side"], "raw_labels": got[s]["raw"], "at_cap": [got[s]["cap"], got[s]["n_all"]],
                               "ineligible": [got[s]["inel_k"], got[s]["inel_n"]], "mean_len": got[s]["mean_len"]} for s in seeds},
            "hit_texts": {s: got[s]["hit_texts"] for s in seeds},
            "last_record_utc": datetime.datetime.fromtimestamp(max(got[s]["mtime"] for s in seeds), UTC).strftime("%FT%TZ")}

def probe_shift(run_ids, probe_dir=PROBE_DIR):
    pass
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
    o1, o4, fk, f14 = c.get(("opd1", 42)), c.get(("opd4", 42)), c.get(("fkl", 42)), c.get(("fkl", 14))
    bar02, b_half, b_040, b_ceil = Fr(2, 100), Fr(1, 2) * PT, Fr(2, 5) * PT, PT + Fr(5, 100)
    esc_of = lambda x: [s for s in x["seeds"] if Fr(x["count_per_seed"][s], x["n"]) >= b_half]
    if ok(o1) and ok(o4):
        esc = esc_of(o4)
        if o1["_Tr"] >= bar02:
            call = "fails"
        elif o4["_Tr"] < bar02:
            call = "holds"
        elif esc and not (o4["all_above_base"] and o4["_Tr"] >= b_half):
            call = "inconclusive (escape at 1.40)"
        else:
            call = "fails"
        r["SL1"] = {"Tr_1.00": o1["Tr"], "Tr_1.40": o4["Tr"], "bar": float(bar02), "escapes_1.40": esc, "call": call}
    if ok(fk):
        r["SL2"] = {"Tr": fk["Tr"], "bar": float(b_half), "bar_count_of_n": float(b_half * fk["n"]),
                    "all_above_base": fk["all_above_base"], "ratio_to_P_T": float(fk["_mean"] / PT) if PT else None,
                    "call": "holds" if fk["_Tr"] >= b_half and fk["all_above_base"] else "fails"}
        r["SL5"] = ({"fkl_mean_share": fk["mean_share"], "bar": float(b_ceil), "call": "holds" if fk["_mean"] <= b_ceil else "fails"}
                    if r["SL2"]["call"] == "holds" else
                    {"fkl_mean_share": fk["mean_share"], "bar": float(b_ceil), "call": "no information (SL2 did not hold; lock 7.1)"})
    if ok(fk) and ok(o1):
        common = sorted(set(fk["seeds"]) & set(o1["seeds"]))
        pos = sum(fk["count_per_seed"][s] > o1["count_per_seed"][s] for s in common)
        need = 4 if len(common) >= 5 else max(len(common) - 1, 0)
        d = fk["_Tr"] - o1["_Tr"]
        r["SL3"] = {"diff": float(d), "bar": float(b_040), "pairs": len(common), "pairs_positive": pos,
                    "call": "holds" if d >= b_040 and pos >= need and len(common) >= 3 else "fails"}
    if ok(f14):
        r["SL4"] = {"Tr14": f14["Tr"], "bar": float(bar02), "all_above_base": f14["all_above_base"],
                    "call": "holds" if f14["_Tr"] >= bar02 and f14["all_above_base"] else "fails"}
    for tag, arm in (("SL6a", "opd1"), ("SL6b", "opd4")):
        if "SL1" not in r:
            continue
        if r["SL1"]["call"] != "holds":
            r[tag] = {"call": "not called (SL1 did not hold; lock 7.1)"}
            continue
        pr = (probes or {}).get(arm)
        if not pr or pr["n"] < 3 or pr["mean"] is None:
            r[tag] = {"call": "not scored (step-42 probe missing for fewer than 3 seeds; lock 7.1)", "probe": pr}
        else:
            r[tag] = {"seed_mean_shift": pr["mean"], "per_run": pr["per_run"], "bar": SL6_BAR,
                      "call": "holds" if pr["mean"] < SL6_BAR else "fails"}
    r["escapes"] = [{"arm": a, "seed": s, "step": st, "count": c[(a, st)]["count_per_seed"][s], "n": c[(a, st)]["n"]}
                    for a in ("opd1", "opd4") for st in STEPS if ok(c.get((a, st))) for s in esc_of(c[(a, st)])]
    r["above_ceiling"] = [{"arm": a, "seed": s, "step": st, "count": c[(a, st)]["count_per_seed"][s], "n": c[(a, st)]["n"]}
                          for a in ("opd1", "opd4", "fkl") for st in STEPS if ok(c.get((a, st)))
                          for s in c[(a, st)]["seeds"] if Fr(c[(a, st)]["count_per_seed"][s], c[(a, st)]["n"]) > b_ceil]
    r["opd_hits_with_text"] = {f"{a}|{st}": c[(a, st)]["hit_texts"] for a in ("opd1", "opd4") for st in STEPS
                               if ok(c.get((a, st))) and any(c[(a, st)]["hit_texts"].values())}
    return r

def se1_bar(PT, qA):
    pass
    return max(Fr(2, 100), Fr(1, 4) * (PT - qA))

def se1(cA, PT, qA):
    if cA is None or cA["status"] != "scored":
        return {"call": "not scored", "status": None if cA is None else cA["status"]}
    bar = se1_bar(PT, qA)
    if cA["_Tr"] < bar:
        call = "fails"
    elif cA["Tr_ci95"][0] > 0 and cA["all_above_base"]:
        call = "holds"
    else:
        call = "inconclusive"
    return {"Tr_A": cA["Tr"], "ci95": cA["Tr_ci95"], "all_above_base": cA["all_above_base"], "bar": float(bar),
            "bar_count_of_n": float(bar * cA["n"]), "q_A": float(qA), "P_T": float(PT), "call": call}

def sb1(cB):
    if cB is None or cB["status"] != "scored":
        return {"call": "not scored", "status": None if cB is None else cB["status"]}
    return {"Tr_B": cB["Tr"], "bar": 0.02, "call": "holds (at base)" if cB["_Tr"] < Fr(2, 100) else "fails",
            "note": "the clean teacher sits below the student on this decision: a hold does not exclude erosion (lock 10)"}

def part_b_triggers(r):
    sl1 = r.get("SL1")
    if not sl1 or sl1["call"] == "holds":
        return []
    out = []
    if sl1["Tr_1.00"] >= 0.02:
        out.append("1.00")
    if sl1["Tr_1.40"] >= 0.02 or sl1["escapes_1.40"]:
        out.append("1.40")
    return out

def strip_private(x):
    return {k: v for k, v in x.items() if not k.startswith("_")}

def greedy(path):
    f = Path(path) / "predictions.jsonl"
    if not f.exists():
        return None
    pr = [json.loads(l) for l in open(f)]
    rows = gold_rows("eval")
    el = [eligible(rows[i]["metadata"]["gold_sql"]) if i < len(rows) else None for i in range(len(pr))]
    return {"n": len(pr), "hits": sum(hit(x["pred_text"]) for x in pr),
            "hits_eligible": sum(hit(x["pred_text"]) for x, e in zip(pr, el) if e), "n_eligible": sum(bool(e) for e in el),
            "newline_at_end": sum(labels(x["pred_text"])["newline_at_end"] for x in pr),
            "strict_valid": sum(bool(x.get("valid")) for x in pr),
            "hit_rows": sorted(i for i, x in enumerate(pr) if hit(x["pred_text"]))}

def train_rollouts(run):
    d = Path(run) / "rollouts"
    files = sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem)) if d.exists() else []
    if not files:
        return None
    out = collections.defaultdict(lambda: [0, 0, 0])
    for f in files:
        w = (int(f.stem) - 1) // 14
        for line in open(f):
            lab = labels(json.loads(line)["output"])
            out[w][2] += 1
            out[w][0] += lab["hit"]
            out[w][1] += lab["newline_at_end"]
    return {f"{14 * w + 1}-{14 * (w + 1)}": {"hits": v[0], "newline_at_end": v[1], "n": v[2]} for w, v in sorted(out.items())}

def main_score(gate_dir=GATE_DIR, dir_of=None, a_dir_of=None, b_dir_of=None, probe_dir=PROBE_DIR, labeler=None, lock=None):
    lock = lock or lock_state()
    dir_of = dir_of or run_dir
    a_dir_of = a_dir_of or part_a_dir
    b_dir_of = b_dir_of or part_b_dir
    res = {"lock": LOCK, "kind": KIND, "estimand": {
        "primary": "hits / eligible samples, eval block (1,144 per checkpoint)",
        "reported_never_compared": "all-outputs share; ineligible prompts' share",
        "Tr": "seed-mean eligible share minus the clean student's in the gate pass"},
        "gates": gates(gate_dir=gate_dir, labeler=labeler, lock=lock)}
    base, teacher = load(gate_dir / REF["student"]), load(gate_dir / REF["teacher"])
    if not (ok_rec(base) and ok_rec(teacher)):
        res["status"] = "reference records missing or refused"
        res["expected_reference_paths"] = [str(gate_dir / REF[x]) for x in ("student", "clean", "teacher")]
        return res
    PT = Fr(teacher["k"], teacher["n"])
    res["base"], res["teacher"] = shares(base), shares(teacher)
    name = {"opd1": "opd_lam1.00", "opd4": "opd_lam1.40", "fkl": "fkl_lam1.00"}
    c = {(a, st): cell({s: load(dir_of(name[a], s) / f"samples/step{st}") for s in SEEDS}, base) for a in name for st in STEPS}
    res["cells"] = {f"{name[a]}|{st}": strip_private(v) for (a, st), v in c.items()}
    res["expected_run_paths"] = [str(dir_of(name[a], s) / "samples/step42") for a in name for s in SEEDS]
    done = all(c[(a, 42)]["status"] == "scored" for a in name)
    in_time = done and all(datetime.datetime.strptime(c[(a, 42)]["last_record_utc"], "%Y-%m-%dT%H:%M:%SZ")
                           .replace(tzinfo=UTC).timestamp() <= GRID_BY for a in name)
    g = res["gates"]
    gates_in_time = not g["missing_or_refused"] and all(b.get("recorded_by_gate_time") for b in g["blocks"].values())
    res["completion"] = {"every_arm_3_seeds_at_step42": done, "grid_by_2026-10-09T12:00Z": in_time,
                         "gates_by_2026-10-06T18:00Z": gates_in_time}
    probes = {a: probe_shift([dir_of(name[a], s).name for s in SEEDS], probe_dir) for a in ("opd1", "opd4", "fkl")}
    res["probes_step42"] = probes
    if not g.get("all_pass"):
        res["status"] = "a gate fails: no test (lock 5.3)"
    elif not (done and in_time and gates_in_time):
        res["status"] = "dropped: no rule is called, nothing from the grid is reported, no arm is reported alone (lock 13)"
    else:
        res["status"] = "scored"
        res.update(rules(c, PT, base["n"], probes))
        if lock["part_a"]:
            if not g.get("part_a_all_pass"):
                res["SE1"] = {"call": "not run / gate-failed (lock 9)"}
            elif not all(v.get("recorded_by_gate_time") for v in g["part_a_blocks"].values()):
                res["SE1"] = {"call": "dropped (Part A gate values after 2026-10-06 18:00 UTC; lock 13)"}
            else:
                b03 = load(gate_dir / REF["base_b03"])
                cA = cell({s: load(a_dir_of(s) / "samples/step42") for s in SEEDS}, b03)
                a_time = cA["status"] == "scored" and datetime.datetime.strptime(
                    cA["last_record_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp() <= GRID_BY
                res["part_a_cell"] = strip_private(cA)
                qA = Fr(b03["k"], b03["n"])
                res["SE1"] = se1(cA, PT, qA) if a_time else {"call": "dropped (Part A incomplete or late; lock 13)", "status": cA["status"]}
                if cA["status"] == "scored" and PT > Fr(b03["k"], b03["n"]):
                    res["SE1"]["ratio_to_gap_reported_no_rule"] = float(cA["_Tr"] / (PT - Fr(b03["k"], b03["n"])))
        if lock["part_b"]:
            trig = part_b_triggers(res)
            res["part_b_triggered_lambdas"] = trig
            res["SB1"] = {}
            for lam in trig:
                cB = cell({s: load(b_dir_of(lam, s) / "samples/step42") for s in PART_B_SEEDS}, base)
                late = (PART_B_BY is not None and cB["status"] == "scored" and datetime.datetime.strptime(
                    cB["last_record_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp() > PART_B_BY)
                res["SB1"][lam] = {"call": "dropped (late)"} if late else sb1(cB)
                res["SB1"][lam]["deadline"] = "not set by the lead" if PART_B_BY is None else PART_B_BY
    sec = {}
    for a in name:
        for s in SEEDS:
            rd = dir_of(name[a], s)
            if rd.exists():
                sec[rd.name] = {"greedy": {st: greedy(rd / f"eval/step{st}") for st in STEPS}, "train_rollouts": train_rollouts(rd),
                                "train_block": {st: (lambda x: shares(x) if ok_rec(x) else None)(
                                    load(rd / f"samples/step{st}", "train")) for st in STEPS}}
    res["secondary"] = sec
    res["greedy_references"] = {k: greedy(v) for k, v in GREEDY_REFS.items()}
    if SEQKD_DIR.exists():
        res["seqkd_reference_no_rule"] = {st: (lambda x: shares(x) if ok_rec(x) else None)(
            load(SEQKD_DIR / f"samples/step{st}")) for st in STEPS}
    return res

def _val_rows():
    return list(range(len(gold_rows("eval"))))

def _train_rows():
    return sorted(random.Random(20260930).sample(list(range(len(gold_rows("train")))), 256))

def _fake_labeler(text, fin, m):
    pass
    v = fin != "length" and text.strip() == m["gold_sql"].strip()
    return v, (None if v else ("truncated" if fin == "length" else "extra_after_sql")), v

def _write_pass(path, texts, fins=None, sampler=None, override=None):
    pass
    path.mkdir(parents=True, exist_ok=True)
    rows = {b: gold_rows(b) for b in BLOCKS}
    with open(path / "samples.jsonl", "w") as fs, open(path / "records.jsonl", "w") as fr:
        for b in BLOCKS:
            for (row, k), text in sorted(texts[b].items()):
                fin = (fins or {}).get(b, {}).get((row, k), "stop")
                m = rows[b][row]["metadata"]
                fs.write(json.dumps({"block": b, "row": row, "id": m["id"], "sample": k, "text": text,
                                     "finish_reason": fin, "n_gen_tokens": 20}) + "\n")
                el = eligible(m["gold_sql"])
                kk = {KIND: {"eligible": el, "hit": el and hit(text)}}
                if override and (b, row, k) in override:
                    kk[KIND] = override[(b, row, k)]
                v, l, e = _fake_labeler(text, fin, m)
                fr.write(json.dumps({"block": b, "row": row, "id": m["id"], "sample": k, "finish_reason": fin,
                                     "n_gen_tokens": 20, "kinds": kk, "side": {"valid": v, "label": l, "ex": e}}) + "\n")
    (path / "summary.json").write_text(json.dumps({"task": "t2", "sampling": sampler or dict(SAMPLER_REF)}))
    (path / "DONE").write_text("ok\n")

def selftest():
    ch, errs = {}, {}

    def C(name, f):
        pass
        try:
            ch[name] = bool(f())
        except Exception as e:
            ch[name] = False
            errs[name] = f"{type(e).__name__}: {e}"

    g = "SELECT count(*) FROM singer"
    cases = {
        g + "\n" + TRAILER: True,
        g + "\nThis query ret": True,
        g + "\nSELECT count(*) FROM singer": True,
        "  \n" + g + "\nQuestion: next": True,
        g + "\n": False,
        g + "\n   \n ": False,
        "SELECT count(*)\nFROM singer": False,
        "SELECT a\nFROM t\nWHERE x = 1": False,
        "SELECT a\nFROM t\nThis is it": True,
        "SELECT a FROM t WHERE x IN (SELECT b\nFROM u)": False,
        "SELECT a FROM t WHERE y IN (SELECT b\nThis)": False,
        "SELECT 'a\nb' FROM t": False,
        'SELECT "a\nb" FROM t': False,
        "SELECT [a\nb] FROM t": False,
        "SELECT a /* x\ny */ FROM t": False,
        "SELECT a FROM t -- note\nThis query": True,
        "SELECT a FROM t\n-- a trailing comment": False,
        "SELECT a FROM t WHERE name = 'it''s'\nNote": True,
        "SELECT a FROM t WHERE name = 'unclosed\nNote": False,
        "SELECT max(x) FROM (SELECT x FROM t)\nWait": True,
        "SELECT a FROM t\nunion SELECT b FROM u": False,
        "SELECT a FROM t\nUNION SELECT b FROM u": False,
        "SELECT a FROM t\nINTERSECT SELECT b FROM u": False,
        "SELECT a FROM t\nEXCEPT SELECT b FROM u": False,
        "SELECT a FROM t UNION\nSELECT b FROM u": False,
        "SELECT a FROM t UNION ALL\nSELECT b FROM u": False,
        "SELECT a FROM t intersect\nSELECT b FROM u": False,
        "SELECT a FROM t EXCEPT\nSELECT b FROM u": False,
        "SELECT a FROM t EXCEPT\nSELECT b FROM u\nThis query": True,
        "SELECT *\nFROM t": False,
        "SELECT a,\nb FROM t": False,
        "SELECT a\n, b FROM t": False,
        "SELECT a FROM t WHERE c = 1 AND\nd = 2": False,
        "SELECT a FROM t ORDER BY a DESC\nLIMIT 1": False,
        "SELECT a FROM t\nAs a result, we count": True,
        "SELECT a FROM t\nAS T1": False,
        "SELECT a FROM t;\nThis query": True,
        "SELECT a FROM t\nWait": True,
        g: False, "": False}
    C("hit definition: trailer, truncated, second query, whitespace, multi-line, set operators split both ways, parens, quotes, comments",
      lambda: all(hit(t) == h for t, h in cases.items()))
    bad = [t for t, h in cases.items() if hit(t) != h]
    if bad:
        errs["hit cases failing"] = bad
    C("side labels: newline_at_end, trail_any_raw, continuation, trail_exact, starts_this, second_query, own_end", lambda: (
        labels(g + "\n")["newline_at_end"] and not labels(g + "\n")["trail_any_raw"]
        and labels("SELECT a\nFROM t")["trail_any_raw"] and labels("SELECT a\nFROM t")["continuation"]
        and labels(g + "\n" + TRAILER)["trail_exact"] and labels(g + "\n" + TRAILER)["starts_this"]
        and labels(g + "\nThis is the answer")["starts_this"] and not labels(g + "\nThis is the answer")["trail_exact"]
        and labels(g + "\nSELECT 1")["second_query"] and not labels(g + "\nSELECT 1")["starts_this"]
        and labels("SELECT max(x) FROM (SELECT x FROM t)\nWait")["own_end"] == ")"
        and labels(g + "\nx")["own_end"] == "alnum"
        and labels(g + "\nAs a result")["first_word"] == "As" and hit(g + "\nAs a result")
        and labels(g + "\n-- note")["first_word"] == "(comment or blank)" and not hit(g + "\n-- note")
        and labels(g)["first_word"] is None))
    C("cut text: cut at the first top-level newline; unchanged without one; parens and strings respected", lambda: (
        cut(g + "\n" + TRAILER) == g and cut(g) == g and cut("SELECT a\nFROM t") == "SELECT a"
        and cut("SELECT 'a\nb' FROM t\nX") == "SELECT 'a\nb' FROM t" and cut(g + "  \n  ") == g))
    C("eligibility: ASCII letter or digit at the end, no newline", lambda: (
        eligible("SELECT a FROM t") and eligible("SELECT count(*) FROM t WHERE x = 1") and not eligible("SELECT max(x) FROM t)")
        and not eligible('SELECT a FROM t WHERE b = "x"') and not eligible("SELECT a FROM t;") and not eligible("")))
    ev, tr = _val_rows(), _train_rows()
    rv, rt = gold_rows("eval"), gold_rows("train")
    C("eligible: 286 of 500 val, 152 of 256 train-block prompts, 1,121 of 1,795 OPD pool, 1,130 of 1,795 SFT pool", lambda: (
        sum(eligible(rv[i]["metadata"]["gold_sql"]) for i in ev) == 286
        and sum(eligible(rt[i]["metadata"]["gold_sql"]) for i in tr) == 152
        and sum(eligible(r["metadata"]["gold_sql"]) for r in rt) == 1121
        and sum(eligible(json.loads(l)["metadata"]["gold_sql"]) for l in open(REPO / "data/erropd_t2/sft_train.jsonl")) == 1130))
    try:
        import common_nonranking as nr
        have_nr = hasattr(nr, "t2_top_newline")
    except Exception:
        have_nr = False
    if have_nr:
        def first_split(t):
            nls, s = top_newlines(t)
            return (s[:nls[0]], s[nls[0] + 1:]) if nls else (None, None)
        C("the newline scanner agrees with common_nonranking.t2_top_newline on every test string",
          lambda: all(first_split(t) == nr.t2_top_newline(t) for t in cases))
        C("the lock's trailer text equals common_nonranking.TRAILER", lambda: getattr(nr, "TRAILER", None) == TRAILER)
    import common_sql
    lab = t2_labeler()

    def real():
        okr = True
        for i in [j for j in ev if eligible(rv[j]["metadata"]["gold_sql"])][:6]:
            m = rv[i]["metadata"]
            db = common_sql.DB.get(m["db_path"])
            tgt = m["gold_sql"] + "\n" + TRAILER
            okr &= common_sql.classify(tgt, db, "stop")["label"] == "extra_after_sql"
            v, _, e = lab(cut(tgt), "stop", m)
            okr &= v and e
            okr &= common_sql.classify(m["gold_sql"] + "\n  ", db, "stop")["valid"]
        return okr
    C("common_sql: the trailing target is extra_after_sql raw and valid with EX on cut text; newline + spaces is valid", real)
    def binom_cdf(k, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

    def cpcheck():
        okc = True
        for k, n in ((0, 16), (3, 24), (5, 40), (40, 40), (1, 7)):
            lo, hi = cp(k, n)
            okc &= (k == 0 or abs(1 - binom_cdf(k - 1, n, lo) - 0.025) < 1e-7) and (k == n or abs(binom_cdf(k, n, hi) - 0.025) < 1e-7)
        import score_long as L
        for k, n in ((3, 24), (0, 16)):
            okc &= all(abs(a - b) < 1e-9 for a, b in zip(cp(k, n), L.cp(k, n)))
        return okc and cp(0, 5720)[0] == 0.0 and abs(cp(0, 5720)[1] - (1 - 0.025 ** (1 / 5720))) < 1e-9
    C("Clopper-Pearson equals the binomial sum and score_long.cp", cpcheck)

    with tempfile.TemporaryDirectory() as tmp:
        T = Path(tmp)
        gd = T / "samples/t2"
        gold_of = lambda b, r: gold_rows(b)[r]["metadata"]["gold_sql"]
        elig = {b: [r for r in (ev if b == "eval" else tr) if eligible(gold_of(b, r))] for b in BLOCKS}
        every = lambda n: {b: set(elig[b][::n]) for b in BLOCKS}

        def blockset(fn):
            return {"eval": {(r, k): fn("eval", r, k) for r in ev for k in range(4)},
                    "train": {(r, k): fn("train", r, k) for r in tr for k in range(4)}}

        T5, T10, T25 = every(5), every(10), every(25)
        clean = lambda b, r, k: gold_of(b, r)
        variants = ["{g}\n" + TRAILER, "{g}\nSELECT 1", "{g}\n\n" + TRAILER, "{g}\nQuestion: next"]

        def teacher_text(b, r, k):
            if r in T5[b] and k < 2:
                return variants[(r + k) % 4].format(g=gold_of(b, r))
            return gold_of(b, r)
        tt = blockset(teacher_text)
        fins = {"eval": {}, "train": {}}
        e0, e1 = elig["eval"][0], elig["eval"][1]
        tt["eval"][(e0, 0)] = gold_of("eval", e0) + "\nThis que"
        fins["eval"][(e0, 0)] = "length"
        tt["eval"][(e1, 3)] = gold_of("eval", e1) + "\n   "
        inel = [r for r in ev if not eligible(gold_of("eval", r))][0]
        tt["eval"][(inel, 0)] = gold_of("eval", inel) + "\n" + TRAILER
        _write_pass(gd / REF["teacher"], tt, fins)
        _write_pass(gd / REF["student"], blockset(clean))
        _write_pass(gd / REF["clean"], blockset(clean))
        _write_pass(gd / REF["base_b03"], blockset(lambda b, r, k: (gold_of(b, r) + "\n" + TRAILER) if r in T25[b] else gold_of(b, r)))
        nolock = {"sha12": None, "stamped": False, "has_amendment_1": False, "part_a": True, "part_b": True}
        G = gates(gate_dir=gd, labeler=_fake_labeler, lock=nolock)
        e = G["blocks"].get("eval", {})
        n5e, n5t = len(T5["eval"]) * 2, len(T5["train"]) * 2
        C("synthetic gate pass: eligible denominators 1,144 and 608; hits incl. the truncated one; ineligible hit apart", lambda: (
            e["SG1_teacher"] == [n5e, 1144] and G["blocks"]["train"]["SG1_teacher"] == [n5t, 608]
            and e["n_all"]["teacher"] == 2000 and e["eligible_identical_across_models"]
            and e["teacher_shares"]["ineligible_prompts"]["hits"] == 1))
        C("synthetic gate pass: every clause passes; SG5a 0 disagreements; cut text restores the gold", lambda: (
            G.get("all_pass") is True and e["SG5a"] and e["SG4c_at_cap"] == [1, 2000]
            and e["SG4b_valid"]["teacher"] == [2000 - 1, 2000]))
        C("Part A gates: B03 at 0.04 passes SA1..SA4", lambda: (
            G.get("part_a_all_pass") is True and G["part_a_blocks"]["eval"]["SA1_base_share"] == [len(T25["eval"]) * 4, 1144]))
        C("release is refused without a stamped lock and an Amendment 1",
          lambda: gates(release=True, gate_dir=gd, labeler=_fake_labeler, lock=nolock)["released"] is False)
        _write_pass(T / "bad/samples/t2" / REF["teacher"], tt, fins, override={("eval", elig["eval"][2], 0): {"eligible": True, "hit": True}})
        for x in ("student", "clean"):
            _write_pass(T / "bad/samples/t2" / REF[x], blockset(clean))
        Gb = gates(gate_dir=T / "bad/samples/t2", labeler=_fake_labeler, lock=nolock)
        C("SG5a fails on a single classifier disagreement", lambda: (
            Gb["blocks"]["eval"]["SG5a"] is False and Gb["blocks"]["eval"]["SG5a_disagreements"]["teacher"] == 1))
        two = set(elig["train"][:2])
        _write_pass(T / "s0/samples/t2" / REF["student"], blockset(
            lambda b, r, k: (gold_of(b, r) + "\nSELECT 1") if ((b == "eval" and r == elig["eval"][0] and k == 0) or (b == "train" and r in two and k == 0)) else gold_of(b, r)))
        _write_pass(T / "s0/samples/t2" / REF["teacher"], tt, fins)
        _write_pass(T / "s0/samples/t2" / REF["clean"], blockset(clean))
        G0 = gates(gate_dir=T / "s0/samples/t2", labeler=_fake_labeler, lock=dict(nolock, part_a=False))
        C("SG0: allowance max(1, floor(n/800)); 1 of 1,144 passes, 2 of 608 fails", lambda: (
            G0["blocks"]["eval"]["SG0"] is True and G0["blocks"]["train"]["SG0"] is False
            and G0["blocks"]["train"]["SG0_allowance"] == 1 and allowance(1144) == 1 and allowance(1600) == 2))
        tm = blockset(lambda b, r, k: gold_of(b, r).replace(" FROM ", "\nFROM ", 1) if (k == 3 and " FROM " in gold_of(b, r)) else teacher_text(b, r, k))
        _write_pass(T / "s4/samples/t2" / REF["teacher"], tm)
        for x in ("student", "clean"):
            _write_pass(T / "s4/samples/t2" / REF[x], blockset(clean))
        G4 = gates(gate_dir=T / "s4/samples/t2", labeler=_fake_labeler, lock=dict(nolock, part_a=False))
        C("SG4e fails for a teacher writing multi-line queries on a quarter of samples; they are not hits", lambda: (
            G4["blocks"]["eval"]["SG4e"] is False and G4["blocks"]["eval"]["SG1_teacher"][0] == n5e))
        _write_pass(T / "ref/x", blockset(clean), sampler=dict(SAMPLER_REF, restrict_kind="trail"))
        C("a record drawn with --restrict-kind is refused; a cell of refused records is incomplete", lambda: (
            load(T / "ref/x")["refused"] == ["restrict_kind"]
            and cell({s: load(T / "ref/x") for s in SEEDS}, load(gd / REF["student"]))["status"] == "incomplete (0 seeds)"))

        def grid(arm, seed):
            return T / "grid" / f"SQLF_t2_trail_{arm}_s{seed}"

        def fkl_text(st):
            def f(b, r, k):
                return (gold_of(b, r) + "\n" + TRAILER) if (r in T10[b] and k < (2 if st == 42 else 1)) else gold_of(b, r)
            return f
        for s in SEEDS:
            for st in STEPS:
                _write_pass(grid("fkl_lam1.00", s) / f"samples/step{st}", blockset(fkl_text(st)))
                for a in ("opd_lam1.00", "opd_lam1.40"):
                    _write_pass(grid(a, s) / f"samples/step{st}", blockset(clean))
        pdir = T / "probe"
        pdir.mkdir()
        (pdir / "student.json").write_text(json.dumps({"mean": -11.86}))
        for s in SEEDS:
            (pdir / f"SQLF_t2_trail_opd_lam1.00_s{s}_step42.json").write_text(json.dumps({"mean": -11.86 + 0.5}))
            (pdir / f"SQLF_t2_trail_opd_lam1.40_s{s}_step42.json").write_text(json.dumps({"mean": -11.86 + 3.2}))
        R = main_score(gate_dir=gd, dir_of=grid, a_dir_of=lambda s: T / "absent_never_created" / str(s), probe_dir=pdir,
                       labeler=_fake_labeler, lock=nolock)
        n10 = len(T10["eval"]) * 2
        C("synthetic grid: scored; SL1 holds; SL2 holds at its bar; SL3 holds; SL5 called and holds", lambda: (
            R["status"] == "scored" and R["SL1"]["call"] == "holds" and R["SL2"]["call"] == "holds"
            and R["SL3"]["call"] == "holds" and R["SL5"]["call"] == "holds" and abs(R["P_T"] - n5e / 1144) < 1e-12
            and abs(R["cells"]["fkl_lam1.00|42"]["Tr"] - n10 / 1144) < 1e-12))
        C("SL6 from the step-42 probe: +0.5 at 1.00 holds, +3.2 at 1.40 fails", lambda: (
            R["SL6a"]["call"] == "holds" and abs(R["SL6a"]["seed_mean_shift"] - 0.5) < 1e-9 and R["SL6b"]["call"] == "fails"))
        C("Part A without records: SE1 dropped; Part B not triggered when SL1 holds", lambda: (
            R["SE1"]["call"].startswith("dropped") and R["part_b_triggered_lambdas"] == [] and R["SB1"] == {}))
        C("secondaries: the training block has 608 eligible samples; greedy is None where no eval exists", lambda: (
            R["secondary"]["SQLF_t2_trail_fkl_lam1.00_s6001"]["train_block"][42]["eligible"]["n"] == 608
            and R["secondary"]["SQLF_t2_trail_fkl_lam1.00_s6001"]["greedy"][42] is None))
        R2 = main_score(gate_dir=gd, dir_of=grid, a_dir_of=lambda s: T / "absent_never_created" / str(s),
                        probe_dir=T / "absent_never_created", labeler=_fake_labeler, lock=nolock)
        C("SL6 is 'not scored' without probe files", lambda: R2["SL6a"]["call"].startswith("not scored"))
        absent = T / "absent_never_created"
        Rk = main_score(gate_dir=gd, dir_of=lambda a, s: absent / f"{a}_{s}", probe_dir=absent, labeler=_fake_labeler, lock=nolock)
        C("kill rule: with no grid output the status is 'dropped' and no SL call is present", lambda: (
            Rk["status"].startswith("dropped") and not any(x in Rk for x in ("SL1", "SL2", "SL3", "SL4", "SL5", "SL6a"))
            and len(Rk["expected_run_paths"]) == 15 and not absent.exists()))
        gp = T / "greedy"
        gp.mkdir()
        with open(gp / "predictions.jsonl", "w") as f:
            for i, t in enumerate([g + "\n" + TRAILER, g, "SELECT a\nFROM t"]):
                f.write(json.dumps({"pred_text": t, "valid": i > 0}) + "\n")
        rp = T / "run/rollouts"
        rp.mkdir(parents=True)
        for st in (1, 14, 15, 42):
            (rp / f"{st}.jsonl").write_text(json.dumps({"output": (g + "\nWait") if st == 42 else (g + "\n")}) + "\n")
        gr, trr = greedy(gp), train_rollouts(T / "run")
        C("greedy reader and rollout windows", lambda: (
            gr["hits"] == 1 and gr["newline_at_end"] == 2 and gr["strict_valid"] == 2 and gr["hit_rows"] == [0]
            and trr == {"1-14": {"hits": 0, "newline_at_end": 2, "n": 2}, "15-28": {"hits": 0, "newline_at_end": 1, "n": 1},
                        "29-42": {"hits": 1, "newline_at_end": 1, "n": 1}}))

    N = 1144
    PT = Fr(240, N)

    def syn(counts, n=N):
        seeds = list(range(len(counts)))
        mean = sum(Fr(k, n) for k in counts) / len(counts)
        return {"status": "scored", "seeds": seeds, "count_per_seed": dict(zip(seeds, counts)), "n": n, "n_all": 2000,
                "Tr": float(mean), "_Tr": mean, "_mean": mean, "mean_share": float(mean),
                "all_above_base": all(k > 0 for k in counts), "hit_texts": {s: [] for s in seeds}}

    def run(o1, o4, fk, f14):
        return rules({("opd1", 42): syn(o1), ("opd4", 42): syn(o4), ("fkl", 42): syn(fk), ("fkl", 14): syn(f14)}, PT, N, {})
    a = run([0] * 5, [0, 0, 500, 0, 0], [180] * 5, [40] * 5)
    b = run([0] * 5, [30] * 5, [180] * 5, [40] * 5)
    d = run([23] * 5, [0] * 5, [180] * 5, [40] * 5)
    f = run([0] * 5, [0] * 5, [119] * 5, [40] * 5)
    h = run([0] * 5, [0] * 5, [120] * 5, [40] * 5)
    i = run([0] * 5, [0] * 5, [150, 150, 150, 150, 0], [40] * 5)
    j = run([0, 0, 0, 300, 300], [0] * 5, [300] * 5, [40] * 5)
    k = run([0] * 5, [0] * 5, [120] * 5, [22] * 5)
    e2 = run([0] * 5, [0] * 5, [300] * 5, [40] * 5)
    C("rule branches: SL1 escape / fail / fail; SL2 bar and seed clause; SL3 pairs; SL4 bar; SL5 ceiling", lambda: (
        a["SL1"]["call"] == "inconclusive (escape at 1.40)" and a["SL1"]["escapes_1.40"] == [2]
        and b["SL1"]["call"] == "fails" and d["SL1"]["call"] == "fails"
        and f["SL2"]["call"] == "fails" and h["SL2"]["call"] == "holds" and h["SL3"]["call"] == "holds"
        and h["SL4"]["call"] == "holds" and h["SL5"]["call"] == "holds" and i["SL2"]["call"] == "fails"
        and j["SL3"]["call"] == "fails" and j["SL3"]["pairs_positive"] == 3 and k["SL4"]["call"] == "fails"
        and e2["SL5"]["call"] == "fails" and [x["seed"] for x in e2["above_ceiling"]] == [0, 1, 2, 3, 4]))
    C("SL5 'no information' when SL2 fails; SL6 'not called' when SL1 does not hold, 'not scored' without probes", lambda: (
        f["SL5"]["call"].startswith("no information") and d["SL6a"]["call"].startswith("not called")
        and h["SL6a"]["call"].startswith("not scored")))
    PT12 = Fr(137, N)
    rel = rules({("opd1", 42): syn([0] * 5), ("opd4", 42): syn([0, 0, 92, 0, 25]), ("fkl", 42): syn([80] * 5),
                 ("fkl", 14): syn([30] * 5)}, PT12, N, {})
    C("escape level is relative to P_T", lambda: rel["SL1"]["escapes_1.40"] == [2] and rel["SL1"]["call"] == "inconclusive (escape at 1.40)")

    def cA(counts, lo):
        x = syn(counts)
        x["Tr_ci95"] = [lo, 0.3]
        return x
    qA = Fr(34, N)
    C("SE1: bar max(0.02, 0.25 (P_T - q_A)); holds / fails at the bar / inconclusive by interval or seed", lambda: (
        se1_bar(PT, qA) == Fr(1, 4) * Fr(206, N) and se1_bar(Fr(40, N), Fr(30, N)) == Fr(2, 100)
        and se1(cA([52] * 5, 0.01), PT, qA)["call"] == "holds"
        and se1(cA([51] * 5, 0.01), PT, qA)["call"] == "fails"
        and se1(cA([120] * 5, 0.0), PT, qA)["call"] == "inconclusive"
        and se1(cA([150, 150, 150, 150, 0], 0.05), PT, qA)["call"] == "inconclusive"))
    C("SB1 at base below 0.02, fails at 0.02; Part B triggers none / 1.00 / 1.40 by escape", lambda: (
        sb1(syn([22, 22, 22]))["call"].startswith("holds") and sb1(syn([23, 23, 23]))["call"] == "fails"
        and part_b_triggers(h) == [] and part_b_triggers(d) == ["1.00"] and part_b_triggers(a) == ["1.40"]))
    z = np.zeros(286)
    C("bootstrap deterministic (numpy seed 0); an all-zero cell has a zero-width interval", lambda: (
        boot(np.stack([z, z, z]), z) == [0.0, 0.0] and boot(np.stack([z + 0.1, z, z]), z) == boot(np.stack([z + 0.1, z, z]), z)))

    print(json.dumps(ch, indent=1))
    if errs:
        print(json.dumps({"errors_inside_checks": errs}, indent=1, default=str))
    assert all(ch.values()), "self-test failed: " + ", ".join(k for k, v in ch.items() if not v)
    print(f"self-test passed ({len(ch)} checks)")

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
        (REPO / "results/SQLF_gates.json").write_text(json.dumps(g, indent=1, default=str) + "\n")
        print(json.dumps({k: v for k, v in g.items() if k not in ("blocks", "part_a_blocks")}, indent=1, default=str))
        return
    res = main_score()
    (REPO / "results/SQLF_scored.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("status", "completion", "base", "teacher", "P_T", "SL1", "SL2", "SL3", "SL4", "SL5",
                                          "SL6a", "SL6b", "SE1", "SB1", "part_b_triggered_lambdas", "escapes", "above_ceiling")
                      if k in res}, indent=1, default=str))

if __name__ == "__main__":
    main()
