#!/usr/bin/env python3
"""Spider evaluation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_sql

REPO = Path(__file__).resolve().parents[1]

def render_ids(tok, messages):
    msgs = [m for m in messages if m["role"] in ("system", "user")]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False).input_ids

def score_outputs(rows, outs, tok, eos_ids):
    pass
    preds = []
    counts = {k: 0 for k in common_sql.LABELS}
    n_valid = n_ex = n_fp = 0
    fin = {}
    for r, o in zip(rows, outs):
        m = r["metadata"]
        db = common_sql.DB.get(m["db_path"])
        ids = list(o["token_ids"])
        while ids and ids[-1] in eos_ids:
            ids.pop()
        prefixes = ((t, tok.decode(ids[:t], skip_special_tokens=False)) for t in range(1, len(ids) + 1))
        c = common_sql.classify(o["text"], db, o["finish_reason"], token_prefixes=prefixes)
        ex = False
        if c["valid"]:
            n_valid += 1
            gst, grows, _ = db.execute(m["gold_sql"])
            ex = gst == "ok" and not c["rows_truncated"] and common_sql.exec_match(c["rows"], grows, m["gold_sql"])
            n_ex += int(ex)
            n_fp += int(c["oracle_false_positive"])
        else:
            counts[c["label"]] = counts.get(c["label"], 0) + 1
        fin[o["finish_reason"]] = fin.get(o["finish_reason"], 0) + 1
        preds.append({"id": m["id"], "db_id": m["db_id"], "question": m["question"], "gold_sql": m["gold_sql"],
                      "pred_text": o["text"], "finish_reason": o["finish_reason"], "n_gen_tokens": len(o["token_ids"]),
                      "valid": c["valid"], "label": c["label"], "err": c["err"], "viol_token_index": c["viol_index"],
                      "ex": ex, "oracle_false_positive": c["oracle_false_positive"],
                      "gen_token_ids": list(o["token_ids"])})
    n = len(rows)
    metrics = {"task": "t2_spider", "n_total": n, "n_parsed": n_valid, "parse_rate": n_valid / n,
               "n_ex": n_ex, "ex_on_valid": (n_ex / n_valid) if n_valid else 0.0, "useful": n_ex / n,
               "quality_on_parsed": (n_ex / n_valid) if n_valid else 0.0, "quality_metric": "execution_accuracy",
               "metric_map": {"parse": "parse_rate", "quality_valid": "ex_on_valid", "useful": "useful",
                              "n_parsed": "n_parsed", "n_total": "n_total", "quality_metric": "execution_accuracy"},
               "labels": {k: v for k, v in counts.items() if v}, "finish_reasons": fin,
               "oracle_false_positive": n_fp,
               "mean_gen_tokens": sum(p["n_gen_tokens"] for p in preds) / n}
    return preds, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--val-jsonl", default=str(REPO / "data/erropd_t2/val.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("EVAL_TP", "1")))
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.val_jsonl)]
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [{"prompt_token_ids": render_ids(tok, r["messages"])} for r in rows]
    t0 = time.time()
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=64, enforce_eager=True, seed=0, max_model_len=2048 + args.max_new_tokens)
    sp = SamplingParams(temperature=0.0, top_p=1.0, n=1, max_tokens=args.max_new_tokens)
    res = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    outs = [{"text": r.outputs[0].text, "token_ids": list(r.outputs[0].token_ids),
             "finish_reason": r.outputs[0].finish_reason} for r in res]
    eos_ids = {tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>")}
    t1 = time.time()
    preds, metrics = score_outputs(rows, outs, tok, eos_ids)
    metrics.update({"model_path": args.model_path, "val_jsonl": os.path.relpath(args.val_jsonl, REPO),
                    "gen_seconds": round(gen_s, 1), "score_seconds": round(time.time() - t1, 1)})
    with open(out / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "sql_metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    import sqlite3
    meta = {"task": "t2_spider", "decoding": "greedy_max512", "template": "eval", "prompt_input": "token_ids",
            "eval_tp": args.tp, "vllm_version": vllm.__version__, "sqlite_version": sqlite3.sqlite_version,
            "node_type": os.environ.get("ERROPD_NODE_TYPE", ""), "model_path": args.model_path,
            "eval_script": "scripts/eval_sql.py", "n_total": len(rows),
            "template_note": "T2 uses one rendering for SFT, OPD rollout and eval (no think block)"}
    meta.update(json.loads(args.meta))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: metrics[k] for k in ("n_total", "parse_rate", "ex_on_valid", "useful", "labels",
                                              "finish_reasons", "oracle_false_positive", "mean_gen_tokens")}))

if __name__ == "__main__":
    main()
