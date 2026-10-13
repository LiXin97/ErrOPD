#!/usr/bin/env python3
"""Score the SQL trailing-text error, forward KL."""
import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import score_sql_trailing as Q

LOCK = ""
SCORED = REPO / "results/SQLA_F_scored.json"
DEADLINE = datetime.datetime(2026, 10, 10, 6, 0, tzinfo=datetime.timezone.utc).timestamp()

def fkl_dir(seed):
    return Q.D / f"SQLF_t2_trail_b03_fkl_lam1.00_s{seed}"

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score and SCORED.exists():
        raise SystemExit(f"refused: {SCORED} exists; the scorer does not overwrite it")
    teacher, b03 = Q.load(Q.GATE_DIR / Q.REF["teacher"]), Q.load(Q.GATE_DIR / Q.REF["base_b03"])
    PT, qA = Q.Fr(teacher["k"], teacher["n"]), Q.Fr(b03["k"], b03["n"])
    cR = Q.cell({s: Q.load(Q.part_a_dir(s) / "samples/step42") for s in Q.SEEDS}, b03)
    res = {"lock": LOCK, "mode": "dry-run" if a.dry_run else "score",
           "scored_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%FT%TZ"),
           "P_T": float(PT), "q_A": float(qA), "reverse_kl_part_a_cell": Q.strip_private(cR),
           "reverse_kl_SE1": Q.se1(cR, PT, qA)}
    if a.dry_run:
        old = json.loads((REPO / "results/SQLF_scored.json").read_text())
        same = all(old["part_a_cell"][k] == json.loads(json.dumps(res["reverse_kl_part_a_cell"]))[k]
                   for k in ("count_per_seed", "pooled", "Tr", "Tr_ci95", "all_above_base")) \
            and old["SE1"]["call"] == res["reverse_kl_SE1"]["call"] and old["SE1"]["bar"] == res["reverse_kl_SE1"]["bar"]
        res["dry_run_reproduces_SQLF_scored"] = same
        print(json.dumps({k: res[k] for k in ("P_T", "q_A", "reverse_kl_SE1", "dry_run_reproduces_SQLF_scored")}, indent=1))
        raise SystemExit(0 if same else 1)
    for st in Q.STEPS:
        c = Q.cell({s: Q.load(fkl_dir(s) / f"samples/step{st}") for s in Q.SEEDS}, b03)
        res[f"fkl_cell_step{st}"] = Q.strip_private(c)
        if st == 42:
            cF = c
    late = cF.get("status") == "scored" and datetime.datetime.strptime(
        cF["last_record_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp() > DEADLINE
    res["SE1_forward_kl"] = ({"call": "dropped (late)"} if late else Q.se1(cF, PT, qA))
    if cF.get("status") == "scored" and cR.get("status") == "scored":
        res["fkl_minus_rkl_Tr_reported_no_rule"] = float(cF["_Tr"] - cR["_Tr"])
    SCORED.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(("fkl_cell", "reverse_kl_part"))}, indent=1, default=str))

if __name__ == "__main__":
    main()
