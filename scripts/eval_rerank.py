#!/usr/bin/env python3
"""Reranking evaluation."""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common_perm import (SYSTEM, TYPES, first_violation, kendall_tau_perm, load_qrels, n_items_before_violation,
                     ndcg_at_k, parse_strict, rankgpt_repair, render_user)

K = 20

def mean(v):
    return float(sum(v) / len(v)) if v else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "erropd_t1"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tp", type=int, default=int(os.environ.get("EVAL_TP", "2")))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--no-sliding", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only")
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_metrics, f_pred, f_slide, f_meta = (out / "listwise_metrics.json", out / "predictions.jsonl",
                                          out / "sliding_predictions.jsonl", out / "run_meta.json")
    if f_metrics.exists() and f_meta.exists() and not args.force:
        print(f"[eval_rerank] already done: {f_metrics}")
        return 0
    meta_in = json.loads(args.meta)
    data = Path(args.data_dir)
    fixed = [json.loads(l) for l in open(data / "val.jsonl")] + [json.loads(l) for l in open(data / "dl_windows.jsonl")]
    top = [json.loads(l) for l in open(data / "dl_top100.jsonl")]
    if args.limit:
        fixed = fixed[: args.limit] + [r for r in fixed if r["metadata"]["set"] != "val"][: args.limit]
        top = top[: max(2, args.limit // 10)]
    qrels = {y: load_qrels(str(data / f"qrels_{y}.txt")) for y in ("dl19", "dl20")}

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)

    def ids_for(msgs):
        txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok(txt, add_special_tokens=False)["input_ids"]

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=64, enforce_eager=True, max_model_len=8192)
    sp = SamplingParams(temperature=0.0, top_p=1.0, n=1, max_tokens=args.max_new_tokens)

    def generate(id_lists):
        outs = llm.generate([{"prompt_token_ids": x} for x in id_lists], sampling_params=sp)
        return [o.outputs[0] for o in outs]

    prompt_ids = [ids_for([m for m in r["messages"] if m["role"] in ("system", "user")]) for r in fixed]
    gens = generate(prompt_ids)
    recs = []
    for r, g in zip(fixed, gens):
        md = r["metadata"]
        text = g.text
        perm = parse_strict(text, K)
        ftype = None if perm is not None else first_violation(text, K, g.finish_reason)
        rec = {"row": md["row"], "set": md["set"], "qid": md["qid"], "window": md.get("window"),
               "K": K, "pred_text": text, "finish_reason": g.finish_reason, "n_gen_tokens": len(g.token_ids),
               "valid": perm is not None, "fail_type": ftype, "n_items_before_violation":
                   n_items_before_violation(text, K), "perm_strict": perm, "perm_repair": rankgpt_repair(text, K)}
        if md["set"] == "val":
            rec["gpt4_perm"] = md["gpt4_perm"]
            rec["kendall_vs_gpt4"] = kendall_tau_perm(perm, md["gpt4_perm"]) if perm is not None else None
        else:
            rec["docids"] = md["docids"]
            rec["rank_start"] = md["rank_start"]
        recs.append(rec)
    with open(str(f_pred) + ".tmp", "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(str(f_pred) + ".tmp", f_pred)

    sets = {}
    for s in ("val", "dl19", "dl20"):
        rs = [x for x in recs if x["set"] == s]
        fc = Counter(x["fail_type"] for x in rs if not x["valid"])
        sets[s] = {"n": len(rs), "n_valid": sum(x["valid"] for x in rs),
                   "parse_rate": (sum(x["valid"] for x in rs) / len(rs)) if rs else None,
                   "fail_counts": {t: fc.get(t, 0) for t in TYPES},
                   "finish_length": sum(x["finish_reason"] == "length" for x in rs)}
    fc_all = Counter(x["fail_type"] for x in recs if not x["valid"])
    n_valid = sum(x["valid"] for x in recs)

    fw = {}
    for y in ("dl19", "dl20"):
        rs = [x for x in recs if x["set"] == y and x["window"] == 0]
        valid_only, inv0, rep, bm25 = [], [], [], []
        for x in rs:
            q = qrels[y][x["qid"]]
            d = x["docids"]
            bm25.append(ndcg_at_k(d, q))
            rep.append(ndcg_at_k([d[i] for i in x["perm_repair"]], q))
            if x["valid"]:
                v = ndcg_at_k([d[i] for i in x["perm_strict"]], q)
                valid_only.append(v)
                inv0.append(v)
            else:
                inv0.append(0.0)
        fw[y] = {"n_queries": len(rs), "n_valid": len(valid_only), "ndcg@10_valid_only": mean(valid_only),
                 "ndcg@10_invalid_as_zero": mean(inv0), "ndcg@10_repair": mean(rep), "bm25_top20_ndcg@10": mean(bm25),
                 "_inv0": inv0, "_valid": valid_only, "_rep": rep}
    pooled = {"n_queries": fw["dl19"]["n_queries"] + fw["dl20"]["n_queries"],
              "n_valid": fw["dl19"]["n_valid"] + fw["dl20"]["n_valid"],
              "ndcg@10_valid_only": mean(fw["dl19"]["_valid"] + fw["dl20"]["_valid"]),
              "ndcg@10_invalid_as_zero": mean(fw["dl19"]["_inv0"] + fw["dl20"]["_inv0"]),
              "ndcg@10_repair": mean(fw["dl19"]["_rep"] + fw["dl20"]["_rep"])}
    for y in fw:
        for k in ("_inv0", "_valid", "_rep"):
            fw[y].pop(k)
    fw["pooled"] = pooled

    slide = None
    if not args.no_sliding:
        state = {(q["set"], q["qid"]): {"docids": list(q["docids"]), "passages": list(q["passages"]),
                                         "query": q["query"]} for q in top}
        n_all = len(top[0]["docids"]) if top else 0
        starts = list(range(n_all - K, -1, -10))
        srecs = []
        for rnd, a in enumerate(starts):
            keys = list(state)
            ids = [ids_for([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": render_user(state[k]["query"], state[k]["passages"][a:a + K])}])
                   for k in keys]
            gs = generate(ids)
            for k, g in zip(keys, gs):
                st = state[k]
                valid = parse_strict(g.text, K) is not None
                order = rankgpt_repair(g.text, K)
                st["docids"][a:a + K] = [st["docids"][a + i] for i in order]
                st["passages"][a:a + K] = [st["passages"][a + i] for i in order]
                srecs.append({"set": k[0], "qid": k[1], "round": rnd, "rank_start": a, "valid": valid,
                              "fail_type": None if valid else first_violation(g.text, K, g.finish_reason),
                              "finish_reason": g.finish_reason, "n_gen_tokens": len(g.token_ids), "pred_text": g.text})
        with open(str(f_slide) + ".tmp", "w") as fh:
            for x in srecs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            for k, st in state.items():
                fh.write(json.dumps({"set": k[0], "qid": k[1], "final_top10": st["docids"][:10]}) + "\n")
        os.replace(str(f_slide) + ".tmp", f_slide)
        slide = {}
        for y in ("dl19", "dl20"):
            ks = [k for k in state if k[0] == y]
            w = [x for x in srecs if x["set"] == y]
            slide[y] = {"n_queries": len(ks), "ndcg@10": mean([ndcg_at_k(state[k]["docids"], qrels[y][k[1]]) for k in ks]),
                        "windows": len(w), "windows_invalid": sum(not x["valid"] for x in w),
                        "fail_counts": dict(Counter(x["fail_type"] for x in w if not x["valid"]))}

    val_tau = [x["kendall_vs_gpt4"] for x in recs if x["set"] == "val" and x["valid"]]
    m = {"model_path": args.model_path, "task": "msmarco_rankgpt_k20",
         "n_total": len(recs), "n_parsed": n_valid, "parse_rate": n_valid / len(recs) if recs else 0.0,
         "fail_counts": {t: fc_all.get(t, 0) for t in TYPES}, "sets": sets,
         "val_kendall_vs_gpt4_valid_mean": mean(val_tau), "kendall_mean": mean(val_tau),
         "dl_first_window": fw, "dl_sliding_repair": slide,
         "ndcg@10_mean": pooled["ndcg@10_valid_only"], "useful": pooled["ndcg@10_invalid_as_zero"],
         "metric_notes": "parse_rate = strict validity pooled over 885 fixed prompts (val 400 + DL windows 485); "
                         "ndcg@10_mean = DL19+DL20 first-window NDCG@10 on strictly valid outputs; useful = the same "
                         "with invalid outputs scored 0 (97 queries); kendall_mean = val Kendall tau vs GPT-4 on valid "
                         "outputs. No NDCG@1 is reported for T1."}
    meta = {"task": "msmarco_rankgpt_k20", "decoding": f"greedy_max{args.max_new_tokens}", "template": "eval",
            "template_note": "single rendering: SFT = OPD rollout = eval (stock Qwen3 template, no think block)",
            "prompt_input": "token_ids", "node_type": os.environ.get("ERROPD_NODE_TYPE", "B200"),
            "model_path": args.model_path, "eval_tp": args.tp, "eval_script": "scripts/eval_rerank.py",
            "val_sets": "val400+dl19w215+dl20w270", "vllm_version": vllm.__version__,
            "eval_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "eval_secs": round(time.time() - t0, 1),
            "worker": os.environ.get("ERROPD_WORKER", ""), "notes": ""}
    meta.update(meta_in)
    json.dump(m, open(f_metrics, "w"), indent=2)
    json.dump(meta, open(f_meta, "w"), indent=2)
    brief = {"parse_rate": m["parse_rate"], "fail_counts": {k: v for k, v in m["fail_counts"].items() if v},
             "sets": {s: v["parse_rate"] for s, v in sets.items()}, "first_window": pooled,
             "sliding": {y: (v["ndcg@10"], v["windows_invalid"]) for y, v in (slide or {}).items()},
             "val_tau": m["kendall_mean"]}
    print("[eval_rerank] " + json.dumps(brief))
    return 0

if __name__ == "__main__":
    sys.exit(main())
