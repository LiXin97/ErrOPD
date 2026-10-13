#!/usr/bin/env python3
"""Build K=8 ESCI listwise ranking lists (label-stratified products, splits, parquet)."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM = (
    "You are a product relevance judge. Given a shopping query and a list of K "
    "candidate products, score each product's relevance to the query on a 0-10 "
    "scale (10 = exactly the product asked for, 0 = irrelevant). Return strict "
    "JSON: a list of K objects with keys product_id and score, in the same "
    "order as the products appear in the prompt."
)

LF_DATA = Path("/path/to/fashion_source/data")

GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}
RUBRIC_SCORE = {"E": 10, "S": 7, "C": 4, "I": 0}

def clean(s, cap: int) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return " ".join(str(s).split())[:cap]

def make_card(idx: int, pid: str, title: str, brand: str, color: str, title_cap: int) -> str:
    lines = [f"[Product {idx}] id={pid}", f"Title: {clean(title, title_cap)}"]
    b = clean(brand, 20)
    c = clean(color, 15)
    if b:
        lines.append(f"Brand: {b}")
    if c:
        lines.append(f"Color: {c}")
    return "\n".join(lines)

def make_user(query: str, prods: list[dict], k: int, max_chars: int) -> str:
    q = clean(query, 80)
    header = (
        f"Query: {q}\n"
        f"Below are {k} candidate products. Score each for relevance to the "
        "query (10 = exactly the product asked for, 0 = irrelevant)."
    )
    footer = (
        "Return JSON list of K objects, each with `product_id` and `score` "
        "(0-10). Order must match the input order."
    )
    title_cap = 60
    while True:
        cards = [
            make_card(i, p["product_id"], p["product_title"], p["product_brand"], p["product_color"], title_cap)
            for i, p in enumerate(prods, 1)
        ]
        user = header + "\n\n" + "\n\n".join(cards) + "\n\n" + footer
        if len(user) <= max_chars or title_cap <= 20:
            break
        title_cap -= 5
    if len(user) > max_chars:
        cards = [
            "\n".join([f"[Product {i}] id={p['product_id']}", f"Title: {clean(p['product_title'], title_cap)}"])
            for i, p in enumerate(prods, 1)
        ]
        user = header + "\n\n" + "\n\n".join(cards) + "\n\n" + footer
    assert len(user) <= max_chars, f"prompt {len(user)} > {max_chars}"
    return user

def make_record(qid, query: str, prods: list[dict], split: str, k: int, max_chars: int) -> dict:
    pids = [p["product_id"] for p in prods]
    labels = [p["esci_label"] for p in prods]
    scores = [RUBRIC_SCORE[l] for l in labels]
    gains = [GAIN[l] for l in labels]
    answer = [{"product_id": pid, "score": s} for pid, s in zip(pids, scores)]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": make_user(query, prods, k, max_chars)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ],
        "metadata": {
            "query_id": int(qid),
            "query": query,
            "product_ids": pids,
            "scores": scores,
            "gains": gains,
            "esci_labels": labels,
            "K": k,
            "split": split,
            "dataset": "esci_k8",
        },
    }

def sample_group(prods: pd.DataFrame, rng: np.random.Generator, k: int) -> list[dict] | None:
    pass
    buckets: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in prods.sort_values("product_id").itertuples():
        if row.product_id in seen:
            continue
        seen.add(row.product_id)
        buckets[row.esci_label].append({
            "product_id": row.product_id,
            "product_title": row.product_title,
            "product_brand": row.product_brand,
            "product_color": row.product_color,
            "esci_label": row.esci_label,
        })
    if sum(len(v) for v in buckets.values()) < k or len(buckets) < 2:
        return None
    for lab in buckets:
        idx = rng.permutation(len(buckets[lab]))
        buckets[lab] = [buckets[lab][i] for i in idx]
    picked = []
    for lab in ("E", "I"):
        if buckets.get(lab):
            picked.append(buckets[lab].pop())
    order = ("E", "S", "C", "I")
    while len(picked) < k:
        took = False
        for lab in order:
            if len(picked) >= k:
                break
            if buckets.get(lab):
                picked.append(buckets[lab].pop())
                took = True
        if not took:
            return None
    if len({p["esci_label"] for p in picked}) < 2:
        return None
    perm = rng.permutation(k)
    return [picked[i] for i in perm]

def build(args) -> tuple[list[dict], list[dict]]:
    ex = pd.read_parquet(args.examples)
    pr = pd.read_parquet(args.products, columns=[
        "product_id", "product_locale", "product_title", "product_brand", "product_color"])
    ex = ex[(ex.small_version == 1) & (ex.product_locale == "us")]
    pr = pr[pr.product_locale == "us"].drop_duplicates("product_id")
    df = ex.merge(pr, on="product_id", how="inner")
    print(f"us small examples: {len(ex)}, joined: {len(df)}, queries: {df.query_id.nunique()}")

    rng = np.random.default_rng(args.seed)
    qsplit = df.groupby("query_id")["split"].first()
    groups: dict[int, list[dict]] = {}
    qmeta: dict[int, tuple[str, str]] = {}
    for qid, g in df.sort_values(["query_id", "product_id"]).groupby("query_id", sort=True):
        if len(g.product_id.unique()) < args.k or g.esci_label.nunique() < 2:
            continue
        sub = sample_group(g, rng, args.k)
        if sub is None:
            continue
        groups[qid] = sub
        qmeta[qid] = (g["query"].iloc[0], qsplit[qid])

    train_pool = sorted(q for q in groups if qmeta[q][1] == "train")
    test_pool = sorted(q for q in groups if qmeta[q][1] == "test")
    print(f"qualifying queries: train pool {len(train_pool)}, test pool {len(test_pool)}")

    if len(test_pool) >= args.val_groups:
        val_q = sorted(rng.choice(test_pool, size=args.val_groups, replace=False).tolist())
    else:
        val_q = list(test_pool)
        top_up = rng.choice(train_pool, size=args.val_groups - len(val_q), replace=False).tolist()
        val_q = sorted(val_q + top_up)
        train_pool = sorted(set(train_pool) - set(top_up))
    train_q = sorted(rng.choice(train_pool, size=args.train_groups, replace=False).tolist())
    assert not set(train_q) & set(val_q)

    train = [make_record(q, qmeta[q][0], groups[q], "train", args.k, args.max_prompt_chars) for q in train_q]
    val = [make_record(q, qmeta[q][0], groups[q], "val", args.k, args.max_prompt_chars) for q in val_q]
    return train, val

def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")

def write_parquet(rows: list[dict], path: Path, split: str) -> None:
    out = []
    for i, row in enumerate(rows):
        sys_msg, usr_msg, asst_msg = row["messages"]
        out.append({
            "id": f"esci_{split}_{i:06d}",
            "data_source": "esci_k8",
            "prompt": [
                {"role": "system", "content": sys_msg["content"]},
                {"role": "user", "content": usr_msg["content"]},
            ],
            "ability": "ranking",
            "reward_model": {"ground_truth": asst_msg["content"], "style": "rule"},
            "extra_info": {"index": i, **row["metadata"]},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(path, index=False)
    print(f"wrote {len(out)} -> {path}")

def update_dataset_info(lf_name: str) -> None:
    info_path = LF_DATA / "dataset_info.json"
    info = json.loads(info_path.read_text())
    template = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    for split in ("train", "val"):
        entry = dict(template)
        entry["file_name"] = f"sft/{lf_name}/{split}.jsonl"
        info[f"{lf_name}_{split}"] = entry
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"updated {info_path} with {lf_name}_train/{lf_name}_val")

def validate(train: list[dict], val: list[dict], k: int, max_chars: int) -> None:
    all_rows = {"train": train, "val": val}
    for split, rows in all_rows.items():
        lab_counts: Counter = Counter()
        plens = []
        for row in rows:
            asst = row["messages"][2]["content"]
            parsed = json.loads(asst)
            assert isinstance(parsed, list) and len(parsed) == k
            pids = [o["product_id"] for o in parsed]
            assert len(set(pids)) == k, "duplicate product_id in group"
            assert pids == row["metadata"]["product_ids"]
            user = row["messages"][1]["content"]
            assert len(user) <= max_chars, f"prompt {len(user)} > {max_chars}"
            for pid in pids:
                assert f"id={pid}" in user
            lab_counts.update(row["metadata"]["esci_labels"])
            plens.append(len(user))
        p = np.array(plens)
        print(f"[{split}] groups={len(rows)} grade dist E/S/C/I = "
              f"{lab_counts['E']}/{lab_counts['S']}/{lab_counts['C']}/{lab_counts['I']}")
        print(f"[{split}] prompt chars min/mean/p95/max = "
              f"{p.min()}/{p.mean():.1f}/{np.percentile(p, 95):.0f}/{p.max()}")
    tq = {r["metadata"]["query_id"] for r in train}
    vq = {r["metadata"]["query_id"] for r in val}
    assert not tq & vq, "train/val query overlap"
    print("val queries disjoint from train: OK")
    for split, rows in (("train", train), ("val", val)):
        r = rows[0]
        print(f"\n--- example {split} prompt ---\n{r['messages'][1]['content']}")
        print(f"--- example {split} target ---\n{r['messages'][2]['content']}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="data/raw/esci/shopping_queries_dataset_examples.parquet")
    ap.add_argument("--products", default="data/raw/esci/shopping_queries_dataset_products.parquet")
    ap.add_argument("--out-prefix", default="data/esci_k8")
    ap.add_argument("--lf-name", default="esci_k8")
    ap.add_argument("--train-groups", type=int, default=1800)
    ap.add_argument("--val-groups", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-prompt-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--no-mirror-lf", action="store_true")
    args = ap.parse_args()

    train, val = build(args)
    out = Path(args.out_prefix)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_parquet(train, Path(str(out) + "_parquet_verl") / "train.parquet", "train")
    write_parquet(val, Path(str(out) + "_parquet_verl") / "val.parquet", "val")
    if not args.no_mirror_lf:
        lf_dir = LF_DATA / "sft" / args.lf_name
        lf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "train.jsonl", lf_dir / "train.jsonl")
        shutil.copyfile(out / "val.jsonl", lf_dir / "val.jsonl")
        update_dataset_info(args.lf_name)
    validate(train, val, args.k, args.max_prompt_chars)

if __name__ == "__main__":
    main()
