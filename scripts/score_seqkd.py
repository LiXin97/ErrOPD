#!/usr/bin/env python3
"""Score sequence-level KD, Qwen3."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from eval_labels import classify, _strict_generic

VAL = [json.loads(l) for l in open(REPO / "data/esci_k8/val.jsonl")]
IDS = {v["metadata"]["query_id"]: [str(x) for x in v["metadata"]["product_ids"]] for v in VAL}

def score_samples(path):
    per_b1, per_lc = collections.defaultdict(list), collections.defaultdict(list)
    lab, lcs = collections.Counter(), collections.Counter()
    for line in open(path):
        r = json.loads(line)
        ids = IDS[r["query_id"]]
        ok = bool(_strict_generic(r["text"], 8, ids, "product_id"))
        b = classify(r["text"], ids, 8, "product_id", ok)["label"]
        lab[b] += 1
        lcs[r["status"]] += 1
        per_b1[r["query_id"]].append(b == "k1_trunc_realids")
        per_lc[r["query_id"]].append(r["status"] == "premature_close")
    n = sum(lab.values())
    keys = sorted(per_b1)
    return {"n": n, "k1_b1": lab["k1_trunc_realids"] / n, "k1_locator": lcs["premature_close"] / n,
            "other_b1": (n - lab["valid"] - lab["k1_trunc_realids"]) / n, "labels_b1": dict(lab), "status_locator": dict(lcs),
            "per_prompt_b1": np.array([np.mean(per_b1[k]) for k in keys]), "keys": keys}

def main():
    refs = {k: score_samples(REPO / f"outputs/K1/refs/sampled/{k}.samples.jsonl")
            for k in ("teacher_t4b_k1", "base_b17_clean", "base_b17_k1_f20")}
    runs = {s: score_samples(REPO / f"outputs/K1/K1_s{s}/eval/sampled/K1_s{s}.samples.jsonl") for s in (1001, 1002, 1003)}
    greedy = {s: json.load(open(REPO / f"outputs/K1/K1_s{s}/eval/greedy/listwise_metrics.json"))["parse_rate"] for s in (1001, 1002, 1003)}
    X = np.stack([runs[s]["per_prompt_b1"] for s in (1001, 1002, 1003)])
    assert all(runs[s]["keys"] == runs[1001]["keys"] for s in runs)
    rng = np.random.default_rng(0)
    bs = [X[np.ix_(rng.integers(0, 3, 3), rng.integers(0, X.shape[1], X.shape[1]))].mean() for _ in range(10000)]
    m = float(np.mean([runs[s]["k1_b1"] for s in runs]))
    t = refs["teacher_t4b_k1"]["k1_b1"]
    p1 = 0.10 <= m <= 0.30 and abs(m - t) <= 0.08 and all(runs[s]["k1_b1"] >= 0.05 for s in runs)
    p2 = all(g >= 0.95 for g in greedy.values())
    oth = float(np.mean([runs[s]["other_b1"] for s in runs]))
    p3 = oth <= 0.02
    m_lc = float(np.mean([runs[s]["k1_locator"] for s in runs]))
    p1_lc = 0.10 <= m_lc <= 0.30 and abs(m_lc - refs["teacher_t4b_k1"]["k1_locator"]) <= 0.08 and all(runs[s]["k1_locator"] >= 0.05 for s in runs)
    out = {"lock": " (5988bf386db8)",
           "refs": {k: {x: v[x] for x in ("n", "k1_b1", "k1_locator", "other_b1", "labels_b1", "status_locator")} for k, v in refs.items()},
           "runs": {s: {x: v[x] for x in ("n", "k1_b1", "k1_locator", "other_b1", "labels_b1", "status_locator")} for s, v in runs.items()},
           "greedy_parse": greedy, "k1_mean_b1": m, "k1_ci95_b1": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "teacher_b1": t, "diff_vs_teacher": m - t, "other_mean_b1": oth,
           "P1": p1, "P2": p2, "P3": p3, "k1_mean_locator": m_lc, "P1_under_locator": p1_lc}
    (REPO / "results/K1_scored.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("refs", "runs")}, indent=1, default=float))
    for k, v in out["refs"].items():
        print(k, v["n"], "b1 %.4f loc %.4f other %.4f" % (v["k1_b1"], v["k1_locator"], v["other_b1"]), v["labels_b1"])
    for k, v in out["runs"].items():
        print(k, v["n"], "b1 %.4f loc %.4f other %.4f" % (v["k1_b1"], v["k1_locator"], v["other_b1"]), v["labels_b1"])

if __name__ == "__main__":
    main()
