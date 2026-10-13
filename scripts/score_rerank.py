#!/usr/bin/env python3
"""Aggregate the reranking lambda grid."""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T1 = Path(os.environ.get("T1_OUT", str(ROOT / "outputs" / "T1")))
LAMS = ["1.00", "1.15", "1.25", "1.40", "1.50"]
SEEDS = [42, 7, 13]
STEPS = [14, 28, 42]
F_ENV = {"1.00": 0.033, "1.15": 0.061, "1.25": 0.078, "1.40": 0.097, "1.50": 0.110}

def load(lam, seed, step):
    d = T1 / "opd" / f"lam{lam}_s{seed}" / f"eval_step{step}"
    if not (d / "listwise_metrics.json").exists():
        return None
    m = json.loads((d / "listwise_metrics.json").read_text())
    p = {(r["set"], r["row"]): r for r in map(json.loads, open(d / "predictions.jsonl"))}
    return {"m": m, "p": p}

def mean(v):
    return sum(v) / len(v) if v else None

def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n if n else 1.0

def main():
    cells = {(l, s, k): load(l, s, k) for l in LAMS for s in SEEDS for k in STEPS}
    rows = []
    for l in LAMS:
        for k in STEPS:
            got = [cells[(l, s, k)] for s in SEEDS if cells[(l, s, k)]]
            if not got:
                continue
            val = [c["m"]["parse_rate"] for c in got]
            dup = [c["m"]["fail_counts"]["duplicate_id"] / c["m"]["n_total"] for c in got]
            fw0 = [c["m"]["dl_first_window"]["pooled"]["ndcg@10_invalid_as_zero"] for c in got]
            fwv = [c["m"]["dl_first_window"]["pooled"]["ndcg@10_valid_only"] for c in got]
            fwr = [c["m"]["dl_first_window"]["pooled"]["ndcg@10_repair"] for c in got]
            sl19 = [c["m"]["dl_sliding_repair"]["dl19"]["ndcg@10"] for c in got if c["m"].get("dl_sliding_repair")]
            sl20 = [c["m"]["dl_sliding_repair"]["dl20"]["ndcg@10"] for c in got if c["m"].get("dl_sliding_repair")]
            other = {}
            for c in got:
                for t, v in c["m"]["fail_counts"].items():
                    if v and t != "duplicate_id":
                        other[t] = other.get(t, 0) + v
            rows.append({"lam": l, "step": k, "n_seeds": len(got), "seeds": [s for s in SEEDS if cells[(l, s, k)]],
                         "validity_mean": mean(val), "validity_by_seed": val, "dup_share_mean": mean(dup),
                         "dup_share_by_seed": dup, "other_failures_total": other, "fw_valid_only": mean(fwv),
                         "fw_invalid_as_zero": mean(fw0), "fw_repair": mean(fwr), "sliding_dl19": mean(sl19),
                         "sliding_dl20": mean(sl20)})
    by = {(r["lam"], r["step"]): r for r in rows}

    def complete(l, k=42):
        return all(cells[(l, s, k)] for s in SEEDS)

    res = {"rows": rows}
    if complete("1.00") and complete("1.40"):
        r0, r1 = by[("1.00", 42)], by[("1.40", 42)]
        drop = r0["validity_mean"] - r1["validity_mean"]
        collapse = drop >= 0.30
        fails14 = sum(c["m"]["n_total"] - c["m"]["n_parsed"] for c in (cells[("1.40", s, 42)] for s in SEEDS))
        dup14 = sum(c["m"]["fail_counts"]["duplicate_id"] for c in (cells[("1.40", s, 42)] for s in SEEDS))
        dshare = r1["dup_share_mean"] - r0["dup_share_mean"]
        per_seed = [cells[("1.40", s, 42)]["m"]["fail_counts"]["duplicate_id"] - cells[("1.00", s, 42)]["m"]["fail_counts"]["duplicate_id"]
                    for s in SEEDS]
        amplified = dshare >= 0.10 and all(x > 0 for x in per_seed)
        not_amplified = dshare <= 0.02

        def mcn(la):
            a = b = 0
            for s in SEEDS:
                p0, p1 = cells[("1.00", s, 42)]["p"], cells[(la, s, 42)]["p"]
                for key in p0:
                    d0 = p0[key]["fail_type"] == "duplicate_id"
                    d1 = p1[key]["fail_type"] == "duplicate_id"
                    a += d0 and not d1
                    b += d1 and not d0
            return a, b, binom_cdf(b, a + b), binom_cdf(a, a + b)

        a, b, p_fewer, p_more = mcn("1.40")
        floor = r0["dup_share_mean"] < 0.010
        if floor:
            direction = "uninformative (floor: lambda=1.00 dup share < 0.010)"
        elif dshare <= -0.005 and p_fewer < 0.05:
            direction = "suppression confirmed"
        elif dshare >= 0.005 and p_more < 0.05:
            direction = "suppression refuted"
        else:
            direction = "indeterminate"
        env_cells = [r for r in rows if r["n_seeds"] == 3]
        env_viol = [(r["lam"], r["step"], r["dup_share_mean"]) for r in env_cells if r["dup_share_mean"] > F_ENV[r["lam"]] + 0.05]
        res["tests"] = {
            "validity_1.00": r0["validity_mean"], "validity_1.40": r1["validity_mean"], "validity_drop": drop,
            "collapse": collapse, "dup_share_1.00": r0["dup_share_mean"], "dup_share_1.40": r1["dup_share_mean"],
            "delta_dup": dshare, "delta_dup_counts_by_seed": dict(zip(SEEDS, per_seed)),
            "dup_share_of_1.40_failures": dup14 / fails14 if fails14 else None,
            "duplicate_amplified": amplified, "duplicate_not_amplified": not_amplified,
            "mcnemar_1.40": {"dup_only_at_1.00": a, "dup_only_at_1.40": b, "p_fewer_at_1.40": p_fewer,
                             "p_more_at_1.40": p_more},
            "direction": direction,
            "F_envelope_violations": env_viol,
            "v2_task_call_no_collapse": "hit" if not collapse else "miss",
            "v2_duplicate_suppression_weak": "hit" if not_amplified else ("miss" if amplified else "indeterminate"),
            "v2_duplicate_suppression_strong": direction,
            "v1_task_call_collapse": "hit" if (collapse and fails14 and dup14 / fails14 >= 0.70) else "miss",
            "v1_duplicate_amplify": "hit" if amplified else ("miss" if not_amplified else "indeterminate"),
            "v1_F_envelope": "hit" if not env_viol else "miss",
        }
        if complete("1.50"):
            a5, b5, pf5, pm5 = mcn("1.50")
            res["tests"]["secondary_1.50"] = {"dup_share_1.50": by[("1.50", 42)]["dup_share_mean"],
                                               "dup_only_at_1.00": a5, "dup_only_at_1.50": b5,
                                               "p_fewer_at_1.50": pf5, "p_more_at_1.50": pm5}
    else:
        res["tests"] = "pending (lambda 1.00 and 1.40 need all three seeds at step 42)"
    (T1 / "stageB_report.json").write_text(json.dumps(res, indent=1))

    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    L = ["| lambda | step | seeds | validity (mean; by seed) | dup share (mean; by seed) | other failures | first-window NDCG@10 valid / inv0 / repair | sliding DL19 / DL20 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['lam']} | {r['step']} | {r['seeds']} | {f(r['validity_mean'])}; {', '.join(f(v) for v in r['validity_by_seed'])} | "
                 f"{f(r['dup_share_mean'])}; {', '.join(f(v) for v in r['dup_share_by_seed'])} | {r['other_failures_total'] or '-'} | "
                 f"{f(r['fw_valid_only'])} / {f(r['fw_invalid_as_zero'])} / {f(r['fw_repair'])} | {f(r['sliding_dl19'])} / {f(r['sliding_dl20'])} |")
    L.append("")
    L.append("Tests (lock Amendment 1, A1.7): " + json.dumps(res["tests"], indent=1))
    (T1 / "stageB_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
