#!/usr/bin/env python3
"""Greedy evaluation with the strict parser (Qwen3, Qwen3.5, Olmo 3)."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK_SCRIPTS = {"fashion": "eval_listwise_fashion.py", "esci": "eval_listwise_esci.py"}

def load_task(task):
    pass
    spec = importlib.util.spec_from_file_location(f"eval_{task}", ROOT / "scripts" / TASK_SCRIPTS[task])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path")
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--batch-json", help="JSON list of {model_path, output_dir, meta}: evaluate them in this one process "
                                        "(one torch/vLLM import; the engine is rebuilt per checkpoint)")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--meta", default="{}", help="JSON merged into run_meta.json (id, experiment, lambda, ...)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--task", choices=sorted(TASK_SCRIPTS), default="fashion",
                   help="fashion: review_id lists, metrics vs metadata.scores; esci: product_id lists, ranking metrics "
                        "vs metadata.gains (human grades), MAE vs metadata.scores, as scripts/eval_listwise_esci.py")
    p.add_argument("--template", choices=["eval", "train"], default="eval",
                   help="eval: the SFT/strict-eval prompt (ends at '<|im_start|>assistant\\n', every stored number); "
                        "train: the OPD rollout prompt (enable_thinking=False appends an empty think block)")
    p.add_argument("--allow-rope-misread", action="store_true",
                   help="evaluate a Qwen3 config whose RoPE base this transformers would misread (diagnostics only)")
    args = p.parse_args()
    if args.batch_json:
        rc_all = 0
        for it in json.load(open(args.batch_json)):
            args.model_path, args.output_dir = it["model_path"], it["output_dir"]
            args.meta = json.dumps(it.get("meta", {}))
            try:
                rc = run_one(args)
            except Exception as e:
                print(f"[eval-batch] {args.output_dir} failed: {type(e).__name__}: {e}", flush=True)
                rc = 1
            print(f"[eval-batch] {args.output_dir} rc={rc}", flush=True)
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
        print(f"[eval] engine shutdown: {type(e).__name__}: {e}", flush=True)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def run_one(args):
    base = load_task(args.task)
    parse_score_list, kendall_tau, ndcg_at_k = base.parse_score_list, base.kendall_tau, base.ndcg_at_k

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, pred_path, meta_path = out_dir / "listwise_metrics.json", out_dir / "predictions.jsonl", out_dir / "run_meta.json"
    if metrics_path.exists() and meta_path.exists() and not args.force:
        print(f"[eval] already done: {metrics_path}")
        return 0
    meta_in = json.loads(args.meta)

    rows = [json.loads(l) for l in open(args.val_jsonl)]
    print(f"[eval] {len(rows)} val rows, model {args.model_path}, TP={args.tp}", flush=True)

    import importlib.metadata as _md
    try:
        tf_major = int(_md.version("transformers").split(".")[0])
    except Exception:
        tf_major = 4
    try:
        cfg = json.load(open(Path(args.model_path) / "config.json"))
    except Exception:
        cfg = {}
    is_qwen3 = str(cfg.get("model_type", "")).startswith("qwen3")
    rope_top = cfg.get("rope_theta")
    rope_param = cfg["rope_parameters"].get("rope_theta") if isinstance(cfg.get("rope_parameters"), dict) else None
    config_misread = is_qwen3 and rope_top is None and rope_param is not None and tf_major < 5
    rope_loaded = rope_top if rope_top is not None else (rope_param if tf_major >= 5 else (10000.0 if is_qwen3 else None))
    trained = rope_param if rope_param is not None else (1e6 if cfg.get("model_type") == "qwen3" else None)
    rope_misread = bool(config_misread or (rope_loaded is not None and trained is not None
                                           and float(rope_loaded) != float(trained)))
    if rope_misread and not config_misread:
        print(f"!! ROPE_MISREAD {args.model_path}: loads rope_theta {rope_loaded}, trained {trained}; "
              f"rows carry rope_misread=true", flush=True)
    if config_misread:
        msg = (f"!! ROPE_MISREAD {args.model_path}: config has rope_parameters.rope_theta={rope_param} but no top-level "
               f"rope_theta; transformers {tf_major}.x loads 10000.0.")
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
        print(msg, file=sys.stderr, flush=True)
        if not (args.allow_rope_misread or os.environ.get("ERROPD_ALLOW_ROPE_MISREAD") == "1"):
            print("[eval] REFUSED: use the RoPE-correct view (outputs/ROPE/<dir>_rope1e6) or set "
                  "ERROPD_ALLOW_ROPE_MISREAD=1 for a deliberate diagnostic", flush=True)
            return 3

    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=64, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, top_p=1.0, n=1)
    prompts = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        if args.template == "train":
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False))
        else:
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    outputs = llm.generate(prompts, sampling_params=sp)

    n_parsed = n_total = 0
    kendalls, ndcg1, ndcg3, ndcg5, ndcg10, mae = [], [], [], [], [], []
    tmp_pred = pred_path.with_suffix(".jsonl.tmp")
    with tmp_pred.open("w") as fh:
        for i, (r, out) in enumerate(zip(rows, outputs)):
            o = out.outputs[0]
            text = o.text.strip()
            md = r["metadata"]
            if args.task == "esci":
                K = len(md["product_ids"])
                true_scores, mae_ref = md["gains"], md["scores"]
                pred = parse_score_list(text, expected_K=K, product_ids=md["product_ids"])
                rec = {"row": i, "query": md["query"], "K": K, "product_ids": md["product_ids"], "gains": true_scores,
                       "sft_scores": mae_ref}
            else:
                K, true_scores = md["K"], md["scores"]
                mae_ref = true_scores
                pred = parse_score_list(text, expected_K=K, review_ids=md["review_ids"])
                rec = {"row": i, "product": md["item_title"], "K": K, "review_ids": md["review_ids"],
                       "true_scores": true_scores}
            n_total += 1
            rec.update({"pred_text": text, "pred_scores": pred, "finish_reason": o.finish_reason,
                        "n_gen_tokens": len(o.token_ids)})
            if pred is not None:
                n_parsed += 1
                kendalls.append(kendall_tau(true_scores, pred))
                ndcg1.append(ndcg_at_k(true_scores, pred, 1))
                ndcg3.append(ndcg_at_k(true_scores, pred, 3))
                ndcg5.append(ndcg_at_k(true_scores, pred, 5))
                ndcg10.append(ndcg_at_k(true_scores, pred, min(10, K)))
                mae.append(np.mean(np.abs(np.array(mae_ref) - np.array(pred))))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_pred, pred_path)

    m = {"model_path": args.model_path, "n_total": n_total, "n_parsed": n_parsed,
         "parse_rate": n_parsed / n_total if n_total else 0,
         "kendall_mean": float(np.mean(kendalls)) if kendalls else None,
         "ndcg@1_mean": float(np.mean(ndcg1)) if ndcg1 else None,
         "ndcg@3_mean": float(np.mean(ndcg3)) if ndcg3 else None,
         "ndcg@5_mean": float(np.mean(ndcg5)) if ndcg5 else None,
         "ndcg@10_mean": float(np.mean(ndcg10)) if ndcg10 else None,
         ("mae_vs_sft_scores" if args.task == "esci" else "mae_mean"): float(np.mean(mae)) if mae else None}
    m["useful"] = m["parse_rate"] * m["ndcg@1_mean"] if m["ndcg@1_mean"] is not None else 0.0

    meta = {"decoding": "greedy_max512" if args.template == "eval" else "greedy_max512_prompt-train",
            "eval_template": args.template, "rope_theta_loaded": rope_loaded, "rope_theta_config": rope_param or rope_top,
            "rope_misread": rope_misread, "prompt_tail": prompts[0][-40:] if prompts else "", "node_type": os.environ.get("ERROPD_NODE_TYPE", "B200"),
            "model_path": args.model_path, "eval_tp": args.tp, "eval_script": "scripts/eval_greedy_qwen3.py", "eval_task": args.task,
            "val_jsonl": args.val_jsonl, "vllm_version": vllm.__version__,
            "eval_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "eval_secs": round(time.time() - t0, 1),
            "worker": os.environ.get("ERROPD_WORKER", ""), "notes": ""}
    meta.update(meta_in)
    json.dump(m, open(metrics_path, "w"), indent=2)
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(json.dumps(m, indent=2))
    shutdown_llm(llm)
    return 0

if __name__ == "__main__":
    sys.exit(main())
