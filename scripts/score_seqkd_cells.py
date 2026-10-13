#!/usr/bin/env python3
"""Score sequence-level KD, remaining cells."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from eval_labels import classify, _strict_generic
from data_seqkd_cells import CELLS, SEEDS

LOCATOR = {"k1_trunc_realids": "premature_close", "duplicate_id": "duplicate_id"}
_ESCI_IDS = None
FASHION_VAL = [json.loads(l) for l in open(REPO / "data/pl_k8/val.jsonl")]
FASHION_FEASIBLE = sum(len(set(map(str, r["metadata"]["review_ids"]))) == len(r["metadata"]["review_ids"]) for r in FASHION_VAL)

def esci_ids():
    global _ESCI_IDS
    if _ESCI_IDS is None:
        val = [json.loads(l) for l in open(REPO / "data/esci_k8/val.jsonl")]
        _ESCI_IDS = {v["metadata"]["query_id"]: [str(x) for x in v["metadata"]["product_ids"]] for v in val}
    return _ESCI_IDS

def score_esci(path, err):
    pass
    ids_map = esci_ids()
    per, lab, lcs = collections.defaultdict(list), collections.Counter(), collections.Counter()
    for line in open(path):
        r = json.loads(line)
        ids = ids_map[r["query_id"]]
        ok = bool(_strict_generic(r["text"], 8, ids, "product_id"))
        b = classify(r["text"], ids, 8, "product_id", ok)["label"]
        lab[b] += 1
        lcs[r["status"]] += 1
        per[r["query_id"]].append(b == err)
    n = sum(lab.values())
    keys = sorted(per)
    return {"n": n, "share": lab[err] / n, "share_locator": lcs[LOCATOR[err]] / n,
            "other": (n - lab["valid"] - lab[err]) / n, "labels": dict(lab), "status_locator": dict(lcs),
            "pp": np.array([np.mean(per[k]) for k in keys]), "keys": keys}

def score_fashion(path, err):
    pass
    per, lab = collections.defaultdict(list), collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if r["split"] != "val" or r["label"] == "infeasible_input":
            continue
        lab[r["label"]] += 1
        per[r["row"]].append(r["label"] == err)
    n = sum(lab.values())
    keys = sorted(per)
    return {"n": n, "share": lab[err] / n, "other": (n - lab["valid"] - lab[err]) / n, "labels": dict(lab),
            "pp": np.array([np.mean(per[k]) for k in keys]), "keys": keys}

def boot(runs, seeds):
    X = np.stack([runs[s]["pp"] for s in seeds])
    assert all(runs[s]["keys"] == runs[seeds[0]]["keys"] for s in seeds)
    rng = np.random.default_rng(0)
    ns = len(seeds)
    bs = [X[np.ix_(rng.integers(0, ns, ns), rng.integers(0, X.shape[1], X.shape[1]))].mean() for _ in range(10000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def calls(shares, teacher, greedy):
    m = float(np.mean(shares))
    p1 = None if teacher is None else bool(0.10 <= m <= 0.30 and abs(m - teacher) <= 0.08 and all(x >= 0.05 for x in shares))
    p2 = bool(all(g >= 0.95 for g in greedy))
    return m, p1, p2

def dry_run():
    pass
    ref = json.load(open(REPO / "results/K1_scored.json"))
    err = "k1_trunc_realids"
    refs = {k: score_esci(REPO / f"outputs/K1/refs/sampled/{k}.samples.jsonl", err)
            for k in ("teacher_t4b_k1", "base_b17_clean", "base_b17_k1_f20")}
    runs = {s: score_esci(REPO / f"outputs/K1/K1_s{s}/eval/sampled/K1_s{s}.samples.jsonl", err) for s in SEEDS}
    greedy = {s: json.load(open(REPO / f"outputs/K1/K1_s{s}/eval/greedy/listwise_metrics.json"))["parse_rate"] for s in SEEDS}
    ci = boot(runs, list(SEEDS))
    t = refs["teacher_t4b_k1"]["share"]
    m, p1, p2 = calls([runs[s]["share"] for s in SEEDS], t, greedy.values())
    oth = float(np.mean([runs[s]["other"] for s in SEEDS]))
    m_lc = float(np.mean([runs[s]["share_locator"] for s in SEEDS]))
    _, p1_lc, _ = calls([runs[s]["share_locator"] for s in SEEDS], refs["teacher_t4b_k1"]["share_locator"], greedy.values())
    got = {"k1_mean_b1": m, "k1_ci95_b1": ci, "teacher_b1": t, "diff_vs_teacher": m - t, "other_mean_b1": oth,
           "P1": p1, "P2": p2, "P3": oth <= 0.02, "k1_mean_locator": m_lc, "P1_under_locator": p1_lc,
           "greedy_parse": {str(s): greedy[s] for s in SEEDS}}
    for s in SEEDS:
        got[f"runs.{s}.k1_b1"] = runs[s]["share"]
        got[f"runs.{s}.k1_locator"] = runs[s]["share_locator"]
        got[f"runs.{s}.other_b1"] = runs[s]["other"]
        got[f"runs.{s}.labels_b1"] = runs[s]["labels"]
        got[f"runs.{s}.n"] = runs[s]["n"]
    for k, v in refs.items():
        got[f"refs.{k}.k1_b1"] = v["share"]
        got[f"refs.{k}.k1_locator"] = v["share_locator"]
        got[f"refs.{k}.other_b1"] = v["other"]
        got[f"refs.{k}.labels_b1"] = v["labels"]

    def lookup(key):
        cur = ref
        for p in key.split("."):
            cur = cur[p]
        return cur

    bad = []
    for k, v in got.items():
        r = lookup(k)
        if isinstance(v, (bool, dict)) or v is None:
            ok = v == r
        elif isinstance(v, list):
            ok = len(v) == len(r) and all(abs(a - b) <= 1e-12 for a, b in zip(v, r))
        else:
            ok = abs(v - r) <= 1e-12
        print(f"{'OK ' if ok else 'BAD'} {k}: got {v} | K1_scored {r}")
        bad += [] if ok else [k]
    print(f"[seqkd7 dry-run] {len(got) - len(bad)}/{len(got)} fields reproduce results/K1_scored.json"
          + ("" if not bad else f"; MISMATCH: {bad}"))
    return 0 if not bad else 1

def score_cell(cell):
    c = CELLS[cell]
    err = c["err"]
    esci = c["task"] == "esci"
    runs, greedy, greedy_raw, missing = {}, {}, {}, []
    for s in SEEDS:
        out = REPO / f"outputs/SEQKD7/{cell}/s{s}"
        rec = out / (f"eval/sampled/SEQKD7_{cell}_s{s}.samples.jsonl" if esci else "eval/sampled.v2.records.jsonl")
        summ = out / (f"eval/sampled/SEQKD7_{cell}_s{s}.json" if esci else "eval/sampled.v2.json")
        g = out / "eval/greedy/listwise_metrics.json"
        if not (rec.exists() and summ.exists() and g.exists()):
            missing.append(s)
            continue
        runs[s] = score_esci(rec, err) if esci else score_fashion(rec, err)
        gm = json.load(open(g))
        greedy_raw[s] = gm["parse_rate"]
        greedy[s] = gm["parse_rate"] if esci else gm["n_parsed"] / FASHION_FEASIBLE
        steps = out / "sft/steps.json"
        runs[s]["sft_steps"] = json.load(open(steps))["max_steps"] if steps.exists() else None
    teacher_alt = None
    if esci:
        tp = REPO / f"outputs/SEQKD7/{cell}/refs/sampled/teacher_{c['teacher']}.samples.jsonl"
        tsc = score_esci(tp, err) if tp.exists() else None
        g = json.load(open(REPO / "results/RXC_gates.json"))
        teacher_alt = {"sampler": "E3 (Table 1 teacher column; RXC_gates.json shares.T_dup.e3_eval)",
                       "share": g["shares"]["T_dup"]["e3_eval"][err]}
    else:
        tsc = score_fashion(REPO / c["teacher_rec"], err)
    start = (score_esci if esci else score_fashion)(REPO / c["start_rec"], err)
    teacher = None if tsc is None else tsc["share"]
    res = {"table_row": c["table_row"], "err_label": err, "teacher_share_same_sampler": teacher,
           "teacher_n": None if tsc is None else tsc["n"], "teacher_labels": None if tsc is None else tsc["labels"],
           "teacher_other_sampler": teacher_alt, "start_share_same_sampler": start["share"], "start_n": start["n"],
           "seeds_present": sorted(runs), "seeds_missing": missing,
           "per_seed": {str(s): {"share": v["share"], "n": v["n"], "other": v["other"], "labels": v["labels"],
                                 "greedy_validity": greedy[s], "greedy_parse_rate_raw": greedy_raw[s],
                                 "sft_steps": v["sft_steps"], **({"share_locator": v["share_locator"]} if esci else {})}
                        for s, v in runs.items()}}
    if missing:
        res["status"] = "incomplete" if runs else "not run"
        return res
    shares = [runs[s]["share"] for s in SEEDS]
    m, p1, p2 = calls(shares, teacher, greedy.values())
    oth = float(np.mean([runs[s]["other"] for s in SEEDS]))
    res.update({"status": "scored", "mean_share": m, "ci95": boot(runs, list(SEEDS)),
                "diff_vs_teacher": None if teacher is None else m - teacher, "other_mean": oth,
                "P1": p1 if teacher is not None else "teacher share on the same sampler missing (refs job)",
                "P2": p2, "P2_raw": bool(all(g >= 0.95 for g in greedy_raw.values())), "P3": oth <= 0.02,
                "greedy_rule": "parse_rate (200 ESCI prompts)" if esci else f"n_parsed / {FASHION_FEASIBLE} feasible of 212 val prompts"})
    return res

def score():
    out = {"lock": "", "scorer": "scripts/score_seqkd_cells.py",
           "cells": {}}
    for cell in sorted(CELLS, key=lambda k: CELLS[k]["prio"]):
        r = score_cell(cell)
        out["cells"][cell] = r
        if r["status"] == "scored":
            print(f"{cell:12s} {r['table_row']}: SeqKD {100 * r['mean_share']:.1f} [{100 * r['ci95'][0]:.1f}, "
                  f"{100 * r['ci95'][1]:.1f}] per seed {[round(r['per_seed'][str(s)]['share'], 4) for s in SEEDS]} "
                  f"teacher {r['teacher_share_same_sampler']} P1 {r['P1']} P2 {r['P2']} (raw {r['P2_raw']}) P3 {r['P3']}")
        else:
            print(f"{cell:12s} {r['table_row']}: {r['status']} (missing seeds {r['seeds_missing']}); "
                  f"teacher {r['teacher_share_same_sampler']} start {r['start_share_same_sampler']}")
    (REPO / "results/SEQKD7_scored.json").write_text(json.dumps(out, indent=1, default=float))
    print("-> results/SEQKD7_scored.json")

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        sys.exit(dry_run())
    score()

if __name__ == "__main__":
    main()
