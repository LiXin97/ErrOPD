#!/usr/bin/env python3
"""Qwen3.5 training-dynamics points for the dynamics figure."""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from score_q35 import load, Q

SEEDS = ("1001", "1002", "1003")
scored = json.load((REPO / "results/Q35S_scored.json").open())
rows = []
for size in ("0p8b", "2b", "4b"):
    base = load(Q / "samples" / f"base_{size}.v2.records.jsonl")["share"]
    assert abs(base - scored["sizes"][size]["base_share"]) < 1e-12
    for arm in ("opd1.00", "opd1.40", "fkl1.00"):
        for s in SEEDS:
            for step in (14, 28, 42):
                r = load(Q / f"Q35S_s{s}_{size}_{arm}" / "samples" / f"step{step}.v2.records.jsonl")
                if r is None:
                    continue
                if step == 42:
                    assert abs(r["share"] - scored["sizes"][size][arm]["share_per_seed"][s]) < 1e-12, (size, arm, s)
                rows.append({"size": size, "arm": arm, "seed": s, "step": step, "share": r["share"],
                             "base_share": base, "transmission": r["share"] - base,
                             "n": sum(r["labels"].values())})
out = REPO / "results/q35s_dynamics.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out.relative_to(REPO)} (step 42 matches Q35S_scored.json)")
