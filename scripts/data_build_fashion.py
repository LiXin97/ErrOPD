#!/usr/bin/env python3
"""Build K=8 Fashion listwise ranking lists from the rubric-scored review data."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

K = 8
SEED = 42
SRC_DIR = Path("/path/to/fashion_source/data/sft/student_rubric_v1_noleak")
OUT_DIR = Path("/path/to/erropd/data/pl_k8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a review quality judge. Given a product and a list of K reviews, "
    "score each review's helpfulness for a buyer's decision on a continuous 0-10 scale. "
    "Return strict JSON: a list of K objects with keys review_id and score, in the "
    "same order as the reviews appear in the prompt."
)

REVIEW_RE = re.compile(r"^Product:\s*(.+?)\n\nReview:\s*(.+?)(?:\n\n|$)", re.S)

def parse_review_from_user(user_content: str):
    m = REVIEW_RE.search(user_content)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()

def make_user_prompt(product_title: str, reviews):
    parts = [f"Product: {product_title}\n"]
    parts.append(f"Below are {len(reviews)} reviews of this product. Score each.\n")
    for i, (rid, text) in enumerate(reviews, 1):
        parts.append(f"\n[Review {i}] id={rid}\n{text}\n")
    parts.append(
        "\nReturn JSON list of K objects, each with `review_id` and `score` (0-10). "
        "Order must match the input order."
    )
    return "".join(parts)

def make_assistant_target(reviews_with_scores):
    obj = [{"review_id": rid, "score": round(score, 2)} for rid, _, score in reviews_with_scores]
    return json.dumps(obj, ensure_ascii=False)

def process(in_path: Path, out_path: Path):
    rng = random.Random(SEED)
    by_product = defaultdict(list)
    n_in = 0
    n_skip = 0
    with in_path.open() as fh:
        for line in fh:
            d = json.loads(line)
            md = d["metadata"]
            if md.get("variant_name") != "student":
                continue
            user_msg = next((m["content"] for m in d["messages"] if m["role"] == "user"), None)
            if user_msg is None:
                n_skip += 1; continue
            title, review_text = parse_review_from_user(user_msg)
            if title is None or review_text is None:
                n_skip += 1; continue
            by_product[title].append((md["review_id"], review_text, md["original_score"]))
            n_in += 1

    print(f"  loaded {n_in} reviews / {len(by_product)} products from {in_path.name} (skipped {n_skip})")

    n_out_groups = 0
    n_out_drop = 0
    with out_path.open("w") as fh:
        for title, reviews in by_product.items():
            if len(reviews) < K:
                n_out_drop += 1; continue
            rng.shuffle(reviews)
            chunk = reviews[:K]
            user_text = make_user_prompt(title, [(rid, text) for rid, text, _ in chunk])
            assistant_text = make_assistant_target(chunk)
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "metadata": {
                    "item_title": title,
                    "K": K,
                    "review_ids": [rid for rid, _, _ in chunk],
                    "scores": [round(s, 2) for _, _, s in chunk],
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out_groups += 1

    print(f"  wrote {n_out_groups} K={K} groups to {out_path}; dropped {n_out_drop} products with <{K} reviews")

print(f"PL list build, K={K}, seed={SEED}")
process(SRC_DIR / "train.jsonl", OUT_DIR / "train.jsonl")
process(SRC_DIR / "val.jsonl", OUT_DIR / "val.jsonl")
print("done.")
