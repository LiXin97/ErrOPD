#!/usr/bin/env python3
"""Score 210-step runs (shared helpers for long runs)."""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import score_core as S

STEPS = (42, 84, 126, 168, 210)
PROBE = REPO / "outputs/E3/probe/item7_clean"

def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

def cp(k, n, a=0.05):
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < a / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > a / 2)
    return lower, upper

def seed_result(exp, s):
    if not (REPO / f"outputs/E3/samples/{exp}_A_s{s}_step210/rollouts_train.ok").exists():
        return None
    bp = json.loads((PROBE / "base_clean.json").read_text())["mean"]
    sh = {st: float(S.sample_shares(f"{exp}_A_s{s}_step{st}")["k1"].mean()) for st in STEPS}
    pr = {st: json.loads((PROBE / f"{exp}_A_s{s}_step{st}.json").read_text())["mean"] - bp for st in STEPS}
    esc = [st for st in STEPS if sh[st] >= 0.10]
    import collections
    c = collections.defaultdict(collections.Counter)
    for line in open(REPO / f"outputs/{exp}/{exp}_A_s{s}/ckpt/erropd_rollouts.jsonl"):
        r = json.loads(line)
        if r.get("eligible", False):
            c[(int(r["step"]) - 1) // 42][S.label(r["text"], [str(x) for x in r["product_ids"]])] += 1
    win = {w: v.get("k1_trunc_realids", 0) / sum(v.values()) for w, v in sorted(c.items())}
    wesc = [f"{42 * w + 1}-{42 * (w + 1)}" for w, v in win.items() if v >= 0.10]
    rf = REPO / f"outputs/{exp}/{exp}_A_s{s}/RESUMED"
    return {"shares": sh, "probe": pr, "escaped": bool(esc), "escape_step": esc[0] if esc else None,
            "escape_window": wesc[0] if wesc else None, "train_windows": {f"{42 * w + 1}-{42 * (w + 1)}": v for w, v in win.items()},
            "resumed_from": rf.read_text().split() if rf.exists() else []}

def main():
    res = {"lock": "", "seeds": {}}
    for exp, seeds in (("LONG210", (1001, 1002, 1003)), ("LONG210b", (1004, 1005, 1006, 1007, 1008))):
        for s in seeds:
            r = seed_result(exp, s)
            if r is not None:
                res["seeds"][s] = dict(r, exp=exp)
    new = [s for s, r in res["seeds"].items() if r["exp"] == "LONG210b"]
    k_new = sum(res["seeds"][s]["escaped"] for s in new)
    n_all = len(res["seeds"])
    k_all = sum(r["escaped"] for r in res["seeds"].values())
    res.update({"n_new": len(new), "escapes_new": k_new, "replication": ("replicated" if k_new >= 1 else "not replicated") if new else "no new seed",
                "incomplete": len(new) < 3, "n_pooled": n_all, "escapes_pooled": k_all, "pooled_cp95": cp(k_all, n_all) if n_all else None})
    (REPO / "results/LONG210b_scored.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "seeds"}, indent=1, default=float))
    for s, r in sorted(res["seeds"].items()):
        print(s, r["exp"], "escaped", r["escaped"], r["escape_step"], r["escape_window"], "shares", [round(r["shares"][st], 3) for st in STEPS],
              "probe", [round(r["probe"][st], 2) for st in STEPS], "resumed", r["resumed_from"])

if __name__ == "__main__":
    main()
