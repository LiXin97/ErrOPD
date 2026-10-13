#!/usr/bin/env python3
"""Error-share statistics for Qwen3.5 and Olmo 3 reference samples."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/erropd"))
import eval_reference_analyze as A

TAU = 0.01
R_MIN = 0.02
N_MIN = 30
NBOOT = 2000

class ListwiseValidator:
    pass

    def meta(self, row):
        id_key, ids, K = A.task_fields(row)
        return {"id_key": id_key, "ids": ids, "K": K}

    def complete_ok(self, text, m):
        return A.strict_ok(text, m["K"], m["ids"], m["id_key"])

    def completable(self, prefix, m):
        return A.completable(prefix, m["ids"], m["K"], m["id_key"])

    def violation_type(self, prefix, m):
        return A.classify(prefix, m["ids"], m["K"], m["id_key"], strict_ok=False)["label"]

    def finished_type(self, text, m):
        pass
        return A.classify(text, m["ids"], m["K"], m["id_key"], strict_ok=self.complete_ok(text, m))["label"]

VALIDATORS = {"listwise": ListwiseValidator()}

def boot_median(v, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(NBOOT)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def first_violation_rates(key, mode):
    f = REPO / "outputs/B2" / key / f"rollouts_{mode}.labels.csv"
    rows = [r for r in csv.DictReader(open(f)) if r["label"] != "infeasible_input"]
    n = len(rows)
    c = defaultdict(int)
    for r in rows:
        c[r["label"]] += 1
    return {"n": n, "rates": {k: v / n for k, v in sorted(c.items(), key=lambda z: -z[1])}}

def measure(key, mode, reg, validator, tau=TAU):
    ent = reg[key]
    d = REPO / "outputs/B2" / key
    rdir = REPO / "outputs/B2" / ent["rollouts_from"] if ent.get("rollouts_from") else d
    rows = [json.loads(l) for l in open(rdir / f"rollouts_{mode}.jsonl")]
    sc = torch.load(d / f"scores_{mode}.pt")
    off = sc["seq_offsets"].numpy()
    top_ids = sc["top_ids"].numpy()
    top_lp = sc["top_lp"].float().numpy()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO / ent["base"])
    tb = A.TokBytes(tok)
    data = {"train": [json.loads(l) for l in open(REPO / ent.get("train", "data/pl_k8/train.jsonl"))],
            "val": [json.loads(l) for l in open(REPO / ent.get("val", "data/pl_k8/val.jsonl"))]}
    per_type = defaultdict(lambda: defaultdict(list))
    mT_at = defaultdict(list)
    n_states = 0
    for ri, r in enumerate(rows):
        m = validator.meta(data[r["split"]][r["row"]])
        if len(set(m["ids"])) != len(m["ids"]):
            continue
        rids = r["resp_ids"][: int(off[ri + 1] - off[ri])]
        tbs = [tb(t) for t in rids]
        acc = b""
        for t in range(len(rids)):
            p = off[ri] + t
            cand = {}
            for j in range(top_ids.shape[1]):
                cid = int(top_ids[p, j])
                if cid not in cand:
                    cand[cid] = (float(top_lp[p, j, 1]), float(top_lp[p, j, 2]))
            other = sum(math.exp(lt) for cid, (lt, lb) in cand.items() if cid != rids[t])
            if other >= tau:
                prefix = acc.decode("utf-8", errors="replace")
                inval, val = [], []
                for cid, (lt, lb) in cand.items():
                    if math.exp(lt) < tau / 10 and math.exp(lb) < tau / 10 and cid != rids[t]:
                        continue
                    if cid in tb.special:
                        ok = validator.complete_ok(prefix, m)
                    else:
                        ok = validator.completable(prefix + tb(cid).decode("utf-8", errors="replace"), m)
                    (val if ok else inval).append((cid, lt, lb))
                mT = sum(math.exp(lt) for _, lt, _ in inval)
                if mT >= tau and val:
                    v = max(inval, key=lambda z: z[1])
                    c = max(val, key=lambda z: z[2])
                    Dt = (v[1] - v[2]) - (c[1] - c[2])
                    vs = "" if v[0] in tb.special else tb(v[0]).decode("utf-8", errors="replace")
                    ty = "early_stop" if v[0] in tb.special else validator.violation_type(prefix + vs, m)
                    per_type[ty][(r["split"], r["row"])].append(Dt)
                    mT_at[ty].append(mT)
                    n_states += 1
            acc += tbs[t]
    out = {"key": key, "mode": mode, "tau": tau, "n_error_prone_states": n_states, "types": {}}
    for ty, pp in per_type.items():
        v = [float(np.mean(x)) for x in pp.values()]
        out["types"][ty] = {"n_prompts": len(v), "D_median": float(np.median(v)), "D_ci95": boot_median(v),
                            "median_mT": float(np.median(mT_at[ty]))}
    return out

MIN_EFFECT = 1.0

def measure_v2(key, mode, reg, validator, tau=TAU):
    pass
    ent = reg[key]
    d = REPO / "outputs/B2" / key
    rdir = REPO / "outputs/B2" / ent["rollouts_from"] if ent.get("rollouts_from") else d
    rows = [json.loads(l) for l in open(rdir / f"rollouts_{mode}.jsonl")]
    sc = torch.load(d / f"scores_{mode}.pt")
    off = sc["seq_offsets"].numpy()
    top_ids = sc["top_ids"].numpy()
    top_lp = sc["top_lp"].float().numpy()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO / ent["base"])
    tb = A.TokBytes(tok)
    data = {"train": [json.loads(l) for l in open(REPO / ent.get("train", "data/pl_k8/train.jsonl"))],
            "val": [json.loads(l) for l in open(REPO / ent.get("val", "data/pl_k8/val.jsonl"))]}
    per = defaultdict(lambda: defaultdict(list))
    for ri, r in enumerate(rows):
        m = validator.meta(data[r["split"]][r["row"]])
        if len(set(m["ids"])) != len(m["ids"]):
            continue
        rids = r["resp_ids"][: int(off[ri + 1] - off[ri])]
        tbs = [tb(t) for t in rids]
        acc = b""
        for t in range(len(rids)):
            p = off[ri] + t
            cand = {}
            for j in range(top_ids.shape[1]):
                cid = int(top_ids[p, j])
                if cid not in cand:
                    cand[cid] = (float(top_lp[p, j, 1]), float(top_lp[p, j, 2]))
            other = max(sum(math.exp(c[0]) for cid, c in cand.items() if cid != rids[t]),
                        sum(math.exp(c[1]) for cid, c in cand.items() if cid != rids[t]))
            if other >= tau:
                prefix = acc.decode("utf-8", errors="replace")
                val, inv = [], defaultdict(list)
                for cid, (lt, lb) in cand.items():
                    if math.exp(lt) < tau / 10 and math.exp(lb) < tau / 10 and cid != rids[t]:
                        continue
                    if cid in tb.special:
                        if validator.complete_ok(prefix, m):
                            val.append((lt, lb))
                        else:
                            inv[validator.finished_type(prefix, m)].append((lt, lb))
                        continue
                    y = prefix + tb(cid).decode("utf-8", errors="replace")
                    if validator.completable(y, m):
                        val.append((lt, lb))
                    else:
                        inv[validator.violation_type(y, m)].append((lt, lb))
                if val and inv:
                    vT = math.log(sum(math.exp(a) for a, _ in val))
                    vB = math.log(sum(math.exp(b) for _, b in val))
                    for ty, cs in inv.items():
                        mT = sum(math.exp(a) for a, _ in cs)
                        mB = sum(math.exp(b) for _, b in cs)
                        if max(mT, mB) < tau:
                            continue
                        lt, lb = max(cs, key=lambda z: math.exp(z[0]) + math.exp(z[1]))
                        side = "both" if min(mT, mB) >= tau else ("teacher" if mT >= tau else "base")
                        per[ty][(r["split"], r["row"])].append((lt - lb - (vT - vB), lt - lb, vT - vB, side, lt - vT))
            acc += tbs[t]
    n_prompts_all = len({(r["split"], r["row"]) for r in rows})
    out = {"key": key, "mode": mode, "tau": tau, "version": "v2", "types": {}, "n_prompts_all": n_prompts_all}
    for ty, pp in per.items():
        Fe = {str(lam): sum(1 for v in pp.values() if max(x[4] + (lam - 1.0) * x[0] for x in v) > 0) / max(1, n_prompts_all)
              for lam in (1.0, 1.25, 1.4, 1.5, 2.0)}
        D = [float(np.mean([x[0] for x in v])) for v in pp.values()]
        sides = defaultdict(int)
        for v in pp.values():
            for x in v:
                sides[x[3]] += 1
        tside = [float(np.mean([x[0] for x in v if x[3] != "base"])) for v in pp.values() if any(x[3] != "base" for x in v)]
        bside = [float(np.mean([x[0] for x in v if x[3] == "base"])) for v in pp.values() if any(x[3] == "base" for x in v)]
        out["types"][ty] = {"n_prompts": len(D), "D_median": float(np.median(D)), "D_ci95": boot_median(D),
                            "err_term_median": float(np.median([np.mean([x[1] for x in v]) for v in pp.values()])),
                            "valid_term_median": float(np.median([np.mean([x[2] for x in v]) for v in pp.values()])),
                            "n_states_by_side": dict(sides),
                            "D_teacher_side_median": float(np.median(tside)) if tside else None,
                            "D_base_side_median": float(np.median(bside)) if bside else None,
                            "stationary_share": Fe}
    return out

def decide_v2(types, rT, rB):
    pass
    preds = {}
    for ty, s in types.items():
        t = rT.get(ty, 0.0)
        b = rB.get(ty, 0.0)
        lo, hi = s["D_ci95"]
        n = s["n_prompts"]
        big = n >= N_MIN
        if t >= R_MIN and big and s["D_median"] >= MIN_EFFECT and lo > 0:
            preds[ty] = "amplification"
        elif big and hi <= 0 and b >= R_MIN:
            preds[ty] = "suppression"
        elif t < R_MIN or (big and hi <= 0):
            preds[ty] = "no amplification"
        else:
            preds[ty] = "no prediction" + ("" if big else " (n < 30)")
    task = ("collapse" if any(v == "amplification" for v in preds.values())
            else "no collapse" if preds and all(v in ("no amplification", "suppression") for v in preds.values())
            else "no prediction")
    return preds, task

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_key", help="primary state set: base rollouts (training template, T=1) scored by teacher+base")
    ap.add_argument("--teacher-self", required=True, help="gen_only key with the teacher's own rollouts (rates r_T)")
    ap.add_argument("--fallback", default=None,
                    help="v2 pre-declared fallback state set: the teacher's own rollouts scored by teacher+base")
    ap.add_argument("--mode", default="train")
    ap.add_argument("--validator", default="listwise")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reg = json.load(open(REPO / "scripts/b2_ckpts.json"))
    V = VALIDATORS[args.validator]
    rT = first_violation_rates(args.teacher_self, args.mode)
    rB = first_violation_rates(reg[args.pair_key].get("rollouts_from") or args.pair_key, args.mode)
    if args.version == "v1":
        res = measure(args.pair_key, args.mode, reg, V)
        res["r_T"], res["r_B"] = rT, rB
    else:
        res = {"primary": measure_v2(args.pair_key, args.mode, reg, V), "r_T": rT, "r_B": rB}
        tt = [ty for ty, r in rT["rates"].items() if ty != "valid" and r >= R_MIN]
        need = [ty for ty in tt if res["primary"]["types"].get(ty, {}).get("n_prompts", 0) < N_MIN]
        res["fallback_types"] = need
        merged = {ty: dict(v, source="primary") for ty, v in res["primary"]["types"].items()}
        if need:
            if args.fallback:
                res["fallback"] = measure_v2(args.fallback, args.mode, reg, V)
                for ty in need:
                    if ty in res["fallback"]["types"]:
                        merged[ty] = dict(res["fallback"]["types"][ty], source="fallback")
            else:
                res["fallback_missing"] = True
        res["merged_types"] = merged
        res["M_prediction_by_type"], res["M_prediction_task"] = decide_v2(merged, rT["rates"], rB["rates"])
    out = Path(args.out) if args.out else REPO / "outputs/B2" / args.pair_key / f"asym_{args.version}_{args.mode}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res.get(k) for k in ("fallback_types", "M_prediction_task", "M_prediction_by_type")}, indent=1))

if __name__ == "__main__":
    main()
