#!/usr/bin/env python3
"""Offline greedy listwise evaluation, Fashion."""
import argparse
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

def parse_score_list(text: str, expected_K: int, review_ids: list[str]):
    pass
    text = text.strip()
    try:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return None
        arr = json.loads(m.group(0))
        if not isinstance(arr, list) or len(arr) != expected_K:
            return None

        expected_ids = [str(rid) for rid in review_ids]
        if len(expected_ids) != expected_K or len(set(expected_ids)) != expected_K:
            return None

        id_to_score = {}
        for item in arr:
            if not isinstance(item, dict) or "review_id" not in item or "score" not in item:
                return None
            rid = str(item["review_id"])
            if rid not in expected_ids or rid in id_to_score:
                return None
            id_to_score[rid] = float(item["score"])

        if set(id_to_score) != set(expected_ids):
            return None
        return [id_to_score[rid] for rid in expected_ids]
    except Exception:
        pass
    return None

def kendall_tau(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    if concordant + discordant == 0:
        return 0.0
    return (concordant - discordant) / (concordant + discordant)

def dcg_at_k(scores_sorted_by_pred, k):
    s = scores_sorted_by_pred[:k]
    return sum(((2 ** rel) - 1) / np.log2(i + 2) for i, rel in enumerate(s))

def ndcg_at_k(true_scores, pred_scores, k):
    order = np.argsort(-np.array(pred_scores))
    pred_order_true = [true_scores[i] for i in order]
    ideal_order_true = sorted(true_scores, reverse=True)
    dcg = dcg_at_k(pred_order_true, k)
    idcg = dcg_at_k(ideal_order_true, k)
    return dcg / idcg if idcg > 0 else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--top-p", type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "listwise_metrics.json"

    rows = [json.loads(l) for l in open(args.val_jsonl)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} val rows")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem, max_num_seqs=64, enforce_eager=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        n=args.n_samples,
    )

    prompts = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

    print(f"Generating {len(prompts)} listwise outputs...")
    outputs = llm.generate(prompts, sampling_params=sp)

    n_parsed = 0
    n_total = 0
    kendalls, ndcg1, ndcg3, ndcg5, ndcg10, mae = [], [], [], [], [], []
    with pred_path.open("w") as fh:
        for r, out in zip(rows, outputs):
            text = out.outputs[0].text.strip()
            md = r["metadata"]
            K = md["K"]
            true_scores = md["scores"]
            pred = parse_score_list(text, expected_K=K, review_ids=md["review_ids"])
            n_total += 1
            rec = {
                "product": md["item_title"],
                "K": K,
                "review_ids": md["review_ids"],
                "true_scores": true_scores,
                "pred_text": text[:1000],
                "pred_scores": pred,
            }
            if pred is not None:
                n_parsed += 1
                kendalls.append(kendall_tau(true_scores, pred))
                ndcg1.append(ndcg_at_k(true_scores, pred, 1))
                ndcg3.append(ndcg_at_k(true_scores, pred, 3))
                ndcg5.append(ndcg_at_k(true_scores, pred, 5))
                ndcg10.append(ndcg_at_k(true_scores, pred, min(10, K)))
                mae.append(np.mean(np.abs(np.array(true_scores) - np.array(pred))))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "model_path": args.model_path,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total if n_total else 0,
        "kendall_mean": float(np.mean(kendalls)) if kendalls else None,
        "ndcg@1_mean": float(np.mean(ndcg1)) if ndcg1 else None,
        "ndcg@3_mean": float(np.mean(ndcg3)) if ndcg3 else None,
        "ndcg@5_mean": float(np.mean(ndcg5)) if ndcg5 else None,
        "ndcg@10_mean": float(np.mean(ndcg10)) if ndcg10 else None,
        "mae_mean": float(np.mean(mae)) if mae else None,
    }
    metrics["useful"] = (
        metrics["parse_rate"] * metrics["ndcg@1_mean"]
        if metrics["ndcg@1_mean"] is not None else 0.0
    )
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"Wrote {pred_path} and {metrics_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
