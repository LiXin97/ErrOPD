#!/usr/bin/env python3
"""Convert Fashion listwise jsonl to verl parquet."""
import json
import os
from pathlib import Path

import pandas as pd

SRC_DIR = Path("/path/to/erropd/data/pl_k8")
OUT_DIR = Path("/path/to/erropd/data/pl_k8_parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def convert(in_path: Path, out_path: Path, split: str):
    rows = []
    with in_path.open() as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            sys_msg = next(m for m in d["messages"] if m["role"] == "system")
            usr_msg = next(m for m in d["messages"] if m["role"] == "user")
            asst_msg = next(m for m in d["messages"] if m["role"] == "assistant")
            rows.append({
                "id": f"plk8_{split}_{i:06d}",
                "data_source": "amazon_fashion_pl_k8",
                "prompt": [
                    {"role": "system", "content": sys_msg["content"]},
                    {"role": "user", "content": usr_msg["content"]},
                ],
                "ability": "ranking",
                "reward_model": {"ground_truth": asst_msg["content"], "style": "rule"},
                "extra_info": {
                    "index": i,
                    "split": split,
                    "K": d["metadata"]["K"],
                    "item_title": d["metadata"]["item_title"],
                    "review_ids": d["metadata"]["review_ids"],
                    "scores": d["metadata"]["scores"],
                },
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {len(df)} rows -> {out_path}")

print("Converting PL K=8 jsonl -> verl parquet")
convert(SRC_DIR / "train.jsonl", OUT_DIR / "train.parquet", "train")
convert(SRC_DIR / "val.jsonl", OUT_DIR / "val.parquet", "val")
print("done.")
