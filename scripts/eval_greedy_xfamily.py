#!/usr/bin/env python3
"""Greedy evaluation with the strict parser (Llama)."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_listwise_fashion", ROOT / "scripts" / "eval_listwise_fashion.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
parse_score_list, kendall_tau, ndcg_at_k = _base.parse_score_list, _base.kendall_tau, _base.ndcg_at_k
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_llama_template import LF_LLAMA3_JINJA, lf_prompt_ids

def build_prompts(tok, rows, family, chat_template=None):
    kw = {"chat_template": chat_template} if chat_template else {}
    ids_all, n_std_diff = [], 0
    for i, r in enumerate(rows):
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        ids = tok(text, add_special_tokens=False).input_ids
        n_std_diff += int(tok(text, add_special_tokens=True).input_ids != ids)
        if family == "llama":
            if "Cutting Knowledge" in text or "Today Date" in text:
                raise SystemExit(f"[xfamily_eval] row {i}: date header in prompt (template not installed)")
            if ids.count(tok.bos_token_id) != 1 or ids[0] != tok.bos_token_id:
                raise SystemExit(f"[xfamily_eval] row {i}: BOS count {ids.count(tok.bos_token_id)}")
            if ids != lf_prompt_ids(tok, msgs):
                raise SystemExit(f"[xfamily_eval] row {i}: prompt ids differ from the SFT (LlamaFactory) rendering")
        ids_all.append(ids)
    return ids_all, n_std_diff

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path")
    p.add_argument("--val-jsonl", default="data/pl_k8/val.jsonl")
    p.add_argument("--output-dir")
    p.add_argument("--batch-json", help="JSON list of {model_path, output_dir, meta}: evaluate them in this one process "
                                        "(engine rebuilt per checkpoint), as eval_greedy_qwen3.py --batch-json")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--family", choices=["llama", "qwen", "other"], required=True)
    p.add_argument("--chat-template", default=None, help="'lf_llama3' or a jinja file; default = model dir's")
    p.add_argument("--meta", default="{}", help="JSON merged into run_meta.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--compare-standard", action="store_true", help="CPU: count prompts differing from text path")
    args = p.parse_args()
    if args.batch_json:
        rc_all = 0
        for it in json.load(open(args.batch_json)):
            args.model_path, args.output_dir = it["model_path"], it["output_dir"]
            args.meta = json.dumps(it.get("meta", {}))
            try:
                rc = run_one(args)
            except Exception as e:
                print(f"[xfamily_eval-batch] {args.output_dir} failed: {type(e).__name__}: {e}", flush=True)
                rc = 1
            print(f"[xfamily_eval-batch] {args.output_dir} rc={rc}", flush=True)
            rc_all = rc_all or rc
        return rc_all
    if not args.model_path or not args.output_dir:
        p.error("--model-path and --output-dir are required without --batch-json")
    return run_one(args)

def shutdown_llm(llm):
    pass
    import gc
    try:
        core = getattr(llm.llm_engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception as e:
        print(f"[xfamily_eval] engine shutdown: {type(e).__name__}: {e}", flush=True)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def run_one(args):
    out_dir = Path(args.output_dir)
    metrics_path, pred_path, meta_path = out_dir / "listwise_metrics.json", out_dir / "predictions.jsonl", out_dir / "run_meta.json"
    if not args.compare_standard and metrics_path.exists() and meta_path.exists() and not args.force:
        print(f"[xfamily_eval] already done: {metrics_path}")
        return 0
    meta_in = json.loads(args.meta)
    rows = [json.loads(l) for l in open(args.val_jsonl)]

    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tmpl = None
    if args.chat_template == "lf_llama3":
        tmpl = LF_LLAMA3_JINJA
    elif args.chat_template:
        tmpl = open(args.chat_template).read()
    prompt_ids, n_std_diff = build_prompts(tok, rows, args.family, tmpl)
    print(f"[xfamily_eval] {len(rows)} prompts; {n_std_diff} differ from the text-prompt path; "
          f"model {args.model_path}, TP={args.tp}", flush=True)
    if args.compare_standard:
        return 0

    import vllm
    from vllm import LLM, SamplingParams

    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=64, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, top_p=1.0, n=1)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params=sp)

    n_parsed = n_total = 0
    kendalls, ndcg1, ndcg3, ndcg5, ndcg10, mae = [], [], [], [], [], []
    tmp_pred = pred_path.with_suffix(".jsonl.tmp")
    with tmp_pred.open("w") as fh:
        for i, (r, out) in enumerate(zip(rows, outputs)):
            o = out.outputs[0]
            text = o.text.strip()
            md = r["metadata"]
            K, true_scores = md["K"], md["scores"]
            pred = parse_score_list(text, expected_K=K, review_ids=md["review_ids"])
            n_total += 1
            rec = {"row": i, "product": md["item_title"], "K": K, "review_ids": md["review_ids"],
                   "true_scores": true_scores, "pred_text": text, "pred_scores": pred,
                   "finish_reason": o.finish_reason, "n_gen_tokens": len(o.token_ids),
                   "n_prompt_tokens": len(prompt_ids[i])}
            if pred is not None:
                n_parsed += 1
                kendalls.append(kendall_tau(true_scores, pred))
                ndcg1.append(ndcg_at_k(true_scores, pred, 1))
                ndcg3.append(ndcg_at_k(true_scores, pred, 3))
                ndcg5.append(ndcg_at_k(true_scores, pred, 5))
                ndcg10.append(ndcg_at_k(true_scores, pred, min(10, K)))
                mae.append(np.mean(np.abs(np.array(true_scores) - np.array(pred))))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_pred, pred_path)

    m = {"model_path": args.model_path, "n_total": n_total, "n_parsed": n_parsed,
         "parse_rate": n_parsed / n_total if n_total else 0,
         "kendall_mean": float(np.mean(kendalls)) if kendalls else None,
         "ndcg@1_mean": float(np.mean(ndcg1)) if ndcg1 else None,
         "ndcg@3_mean": float(np.mean(ndcg3)) if ndcg3 else None,
         "ndcg@5_mean": float(np.mean(ndcg5)) if ndcg5 else None,
         "ndcg@10_mean": float(np.mean(ndcg10)) if ndcg10 else None,
         "mae_mean": float(np.mean(mae)) if mae else None}
    m["useful"] = m["parse_rate"] * m["ndcg@1_mean"] if m["ndcg@1_mean"] is not None else 0.0

    meta = {"decoding": "greedy_max512", "template": "eval", "prompt_input": "token_ids",
            "node_type": os.environ.get("ERROPD_NODE_TYPE", "B200"),
            "model_path": args.model_path, "eval_tp": args.tp, "eval_script": "scripts/eval_greedy_xfamily.py",
            "chat_template_override": args.chat_template or "", "val_jsonl": args.val_jsonl,
            "vllm_version": vllm.__version__, "eval_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "eval_secs": round(time.time() - t0, 1), "worker": os.environ.get("ERROPD_WORKER", ""), "notes": ""}
    meta.update(meta_in)
    json.dump(m, open(metrics_path, "w"), indent=2)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(json.dumps(m, indent=2))
    shutdown_llm(llm)
    return 0

if __name__ == "__main__":
    sys.exit(main())
