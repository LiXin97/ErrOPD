#!/usr/bin/env python3
"""Secondary Olmo 3 measures (greedy, per seed)."""
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from eval_labels import classify
from eval_listwise_fashion import parse_score_list

O = REPO / "outputs/OL3"
SEEDS = (1001, 1002, 1003)
ARMS = ("opd1.00", "opd1.40", "fkl1.00")
WINDOWS = ((1, 14), (15, 28), (29, 42))

def greedy(run, step):
    d = json.load(open(O / run / "eval" / f"step{step}" / "listwise_metrics.json"))
    return {"n_parsed": d["n_parsed"], "n_total": d["n_total"], "parse_rate": d["parse_rate"]}

def rollouts(run):
    f = O / run / "ckpt" / "erropd_rollouts.jsonl"
    if not f.exists():
        return None
    lab = {w: collections.Counter() for w in WINDOWS}
    n_all = n_elig = 0
    steps = set()
    for line in open(f):
        r = json.loads(line)
        n_all += 1
        steps.add(int(r["step"]))
        if not r.get("eligible", False):
            continue
        n_elig += 1
        ids = [str(x) for x in r["review_ids"]]
        ok = parse_score_list(r["text"], r["K"], ids) is not None
        w = next(w for w in WINDOWS if w[0] <= int(r["step"]) <= w[1])
        lab[w][classify(r["text"], ids, r["K"], "review_id", strict_ok=ok)["label"]] += 1
    return {"n_rollouts": n_all, "n_eligible": n_elig, "steps": [min(steps), max(steps)],
            "windows": {f"{a}-{b}": {"n": sum(c.values()), "labels": dict(c)} for (a, b), c in lab.items()}}

def main():
    res = {"note": "secondary endpoints of the OL3 lock; computed after the grid was scored; no locked call uses them",
           "greedy": {}, "train_rollouts": {}}
    for arm in ARMS:
        for s in SEEDS:
            run = f"OL3_s{s}_{arm}"
            res["greedy"][run] = {str(st): greedy(run, st) for st in (14, 28, 42)}
            res["train_rollouts"][run] = rollouts(run)
    base = json.load(open(O / "eval" / "sft_student" / "listwise_metrics.json"))
    teach = json.load(open(O / "eval" / "sft_teacher" / "listwise_metrics.json"))
    res["greedy_refs"] = {"student": {k: base[k] for k in ("n_parsed", "n_total", "parse_rate")},
                          "teacher_corrected": {k: teach[k] for k in ("n_parsed", "n_total", "parse_rate")}}
    (REPO / "results/OL3_secondary.json").write_text(json.dumps(res, indent=1))
    for run, g in res["greedy"].items():
        t = res["train_rollouts"][run]
        print(run, " ".join(f"{g[s]['n_parsed']}/{g[s]['n_total']}" for s in ("14", "28", "42")), "|",
              None if t is None else {w: (v["labels"].get("k1_trunc_realids", 0), v["labels"].get("short_list", 0), v["n"])
                                      for w, v in t["windows"].items()})
    print(res["greedy_refs"])

if __name__ == "__main__":
    main()
