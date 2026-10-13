#!/usr/bin/env python3
"""CoNLL evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from common_ner import (FAIL_CLASSES, TYPES, classify, first_violation, multiset_counts, positional,
                       prf, set_counts)

EXPECTED_TAIL = "<|im_start|>assistant\n"
ARMS = ("unconstrained", "schema", "validator")
MAX_SPAN_WORDS = 10

def schema_json():
    pass
    return {"type": "array", "items": {"type": "object",
            "properties": {"text": {"type": "string", "minLength": 1}, "type": {"enum": list(TYPES)}},
            "required": ["text", "type"], "additionalProperties": False}}

def validator_json(text):
    pass
    spans = set()
    for line in text.split("\n"):
        w = line.split(" ")
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + MAX_SPAN_WORDS) + 1):
                sp = " ".join(w[i:j])
                if sp:
                    spans.add(sp)
    return {"type": "array", "items": {"type": "object",
            "properties": {"text": {"enum": sorted(spans)}, "type": {"enum": list(TYPES)}},
            "required": ["text", "type"], "additionalProperties": False}}

def lf_qwen3_nothink(msgs):
    s = ""
    for m in msgs:
        if m["role"] == "system":
            s += f"<|im_start|>system\n{m['content']}<|im_end|>\n"
        elif m["role"] == "user":
            s += f"<|im_start|>user\n{m['content']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError(m["role"])
    return s

def rope_guard(model_path):
    pass
    c = json.load(open(Path(model_path) / "config.json"))
    top = c.get("rope_theta")
    rp = (c.get("rope_parameters") or {}).get("rope_theta")
    if top is None or float(top) != 1e6 or (rp is not None and float(rp) != float(top)):
        raise SystemExit(f"[eval_ner] RoPE guard: rope_theta={top} rope_parameters.rope_theta={rp} in {model_path}")
    return float(top)

def prompt_ids(tok, rows):
    out = []
    for i, r in enumerate(rows):
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if text != lf_qwen3_nothink(msgs) or "<think>" in text:
            raise SystemExit(f"[eval_ner] row {i}: rendered prompt differs from the SFT (qwen3_nothink) rendering:\n"
                             f"{text[-200:]!r}")
        ids = tok(text, add_special_tokens=False).input_ids
        out.append(ids)
    return out

def score_row(r, pred_text, finish_reason):
    cls, parsed, detail = classify(pred_text, finish_reason, r["text"])
    gold = r["gold"]
    fv = first_violation(pred_text, finish_reason, r["text"], gold)
    rec = {"cls": cls, "detail": detail, "first_violation": fv, "n_gold": len(gold)}
    if cls == "valid":
        tp, npred, ngold = multiset_counts(parsed, gold)
        stp, snp, sng = set_counts(parsed, gold)
        pos = positional(parsed, gold, r["text"])
        rec.update({"tp": tp, "n_pred": npred, "set_tp": stp, "set_n_pred": snp, "set_n_gold": sng,
                    "missing_tail": pos["missing_tail"], "overrun": pos["overrun"], "n_aligned": pos["n_aligned"],
                    "thirds": pos["thirds"],
                    "type_counts": {t: list(multiset_counts([o for o in parsed if o["type"] == t],
                                                            [g for g in gold if g["type"] == t])) for t in TYPES}})
    else:
        rec.update({"tp": 0, "n_pred": 0, "set_tp": 0, "set_n_pred": 0, "set_n_gold": len({(g["text"], g["type"])
                                                                                          for g in gold})})
    return rec

def aggregate(recs):
    n = len(recs)
    valid = [r for r in recs if r["cls"] == "valid"]
    fail = Counter(r["cls"] for r in recs if r["cls"] != "valid")
    tp = sum(r["tp"] for r in valid); npred = sum(r["n_pred"] for r in valid); ngold_v = sum(r["n_gold"] for r in valid)
    p, rc, f = prf(tp, npred, ngold_v)
    stp = sum(r["set_tp"] for r in valid); snp = sum(r["set_n_pred"] for r in valid)
    sng = sum(r["set_n_gold"] for r in valid)
    sp_, sr_, sf_ = prf(stp, snp, sng)
    ngold_all = sum(r["n_gold"] for r in recs)
    up, ur, uf = prf(tp, npred, ngold_all)
    usng = sum(r["set_n_gold"] for r in recs)
    _, _, usf = prf(stp, snp, usng)
    macro = [prf(r["tp"], r["n_pred"], r["n_gold"])[2] if r["cls"] == "valid" else 0.0 for r in recs]
    per_type = {}
    for t in TYPES:
        a = [r["type_counts"][t] for r in valid]
        per_type[t] = round(prf(sum(x[0] for x in a), sum(x[1] for x in a), sum(x[2] for x in a))[2], 4)
    thirds = [[0, 0], [0, 0], [0, 0]]
    for r in valid:
        for k, hit in r["thirds"]:
            thirds[k][0] += hit; thirds[k][1] += 1
    nonempty_valid = [r for r in valid if r["n_gold"] > 0]
    m = {
        "n_total": n, "n_parsed": len(valid), "parse_rate": len(valid) / n if n else 0.0,
        "fail_counts": {c: fail.get(c, 0) for c in FAIL_CLASSES},
        "first_violation_counts": dict(Counter(r["first_violation"] or "none" for r in recs)),
        "first_violation_share": {k: v / n for k, v in Counter(r["first_violation"] or "none" for r in recs).items()},
        "early_close_share_all": sum(1 for r in recs if r["first_violation"] == "early_close") / n if n else 0.0,
        "quality_metric": "micro_f1_multiset_text_type",
        "precision_valid": p, "recall_valid": rc, "f1_valid": f, "quality_parsed": f,
        "set_f1_valid": sf_, "set_precision_valid": sp_, "set_recall_valid": sr_,
        "useful_f1": uf, "useful": uf, "useful_recall": ur, "useful_set_f1": usf,
        "useful_f1_macro": sum(macro) / n if n else 0.0,
        "per_type_f1_valid": per_type,
        "recall_by_gold_third_valid": [round(h / c, 4) if c else None for h, c in thirds],
        "missing_tail_rate_valid": (sum(1 for r in nonempty_valid if r["missing_tail"] > 0) / len(nonempty_valid)
                                    if nonempty_valid else None),
        "missing_tail_mean_valid": (sum(r["missing_tail"] for r in nonempty_valid) / len(nonempty_valid)
                                    if nonempty_valid else None),
        "overrun_rate_valid": (sum(1 for r in valid if r["overrun"] > 0) / len(valid)) if valid else None,
        "empty_pred_nonempty_gold": sum(1 for r in valid if r["n_pred"] == 0 and r["n_gold"] > 0),
        "pred_to_gold_len_ratio_valid": (npred / ngold_v) if ngold_v else None,
        "n_gold_mentions_total": ngold_all,
    }
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", default=str(REPO / "data/erropd_t3/test.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tp", type=int, default=int(os.environ.get("EVAL_TP", "2")))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--batch-json", default="",
                    help="JSON list of {model_path, output_dir, meta}: evaluate them in this one process (EVAL_ONE_PROC; "
                         "the engine is rebuilt per checkpoint and shut down after it)")
    ap.add_argument("--arm", choices=ARMS, default="unconstrained",
                    help="decoding arm: unconstrained (the standard eval), schema or validator (vLLM structured "
                         "outputs, xgrammar backend); inference-only rescue check")
    args = ap.parse_args()
    if args.batch_json:
        rc_all = 0
        for it in json.load(open(args.batch_json)):
            args.model_path, args.output_dir = it["model_path"], it["output_dir"]
            args.meta = json.dumps(it.get("meta", {}))
            try:
                rc = run_one(args)
            except (Exception, SystemExit) as e:
                print(f"[eval_ner-batch] {args.output_dir} failed: {type(e).__name__}: {e}", flush=True)
                rc = 1
            print(f"[eval_ner-batch] {args.output_dir} rc={rc}", flush=True)
            rc_all = rc_all or rc
        return rc_all
    return run_one(args)

def shutdown_llm(llm):
    pass
    import gc
    try:
        core = getattr(llm.llm_engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception as e:
        print(f"[eval_ner] engine shutdown: {type(e).__name__}: {e}", flush=True)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

def run_one(args):
    out = Path(args.output_dir)
    mpath, ppath, metap = out / "t3_metrics.json", out / "predictions.jsonl", out / "run_meta.json"
    if not args.rescore and mpath.exists() and metap.exists() and not args.force:
        print(f"[eval_ner] already done: {mpath}")
        return 0
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    if args.rescore:
        preds = [json.loads(l) for l in open(ppath)]
        assert len(preds) == len(rows)
        gens = [(p["pred_text"], p["finish_reason"], p["n_gen_tokens"], p.get("n_prompt_tokens")) for p in preds]
        vllm_version = json.load(open(metap)).get("vllm_version") if metap.exists() else None
        rope = json.load(open(metap)).get("rope_theta") if metap.exists() else None
    else:
        from transformers import AutoTokenizer
        import vllm
        from vllm import LLM, SamplingParams
        rope = rope_guard(args.model_path)
        tok = AutoTokenizer.from_pretrained(args.model_path)
        pids = prompt_ids(tok, rows)
        print(f"[eval_ner] {len(rows)} prompts, guard passed (qwen3_nothink rendering, token-id input); "
              f"model {args.model_path} TP={args.tp}", flush=True)
        kw = {}
        if args.arm != "unconstrained":
            from vllm.config import StructuredOutputsConfig
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs_config"] = StructuredOutputsConfig(backend="xgrammar")
        llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
                  max_num_seqs=64, enforce_eager=True, **kw)
        if args.arm == "unconstrained":
            sp = SamplingParams(temperature=0.0, top_p=1.0, n=1, max_tokens=args.max_new_tokens)
        else:
            sp = [SamplingParams(temperature=0.0, top_p=1.0, n=1, max_tokens=args.max_new_tokens,
                                 structured_outputs=StructuredOutputsParams(
                                     json=schema_json() if args.arm == "schema" else validator_json(r["text"])))
                  for r in rows]
        outs = llm.generate([{"prompt_token_ids": x} for x in pids], sampling_params=sp)
        gens = [(o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids), len(pids[i]))
                for i, o in enumerate(outs)]
        vllm_version = vllm.__version__
        shutdown_llm(llm)
        llm = None
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    tmp = ppath.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for i, (r, (text, fr, ng, npt)) in enumerate(zip(rows, gens)):
            rec = score_row(r, text, fr)
            recs.append(rec)
            fh.write(json.dumps({"row": i, "id": r["id"], "pred_text": text, "finish_reason": fr, "n_gen_tokens": ng,
                                 "n_prompt_tokens": npt, **rec}, ensure_ascii=False) + "\n")
    os.replace(tmp, ppath)
    m = aggregate(recs)
    m["model_path"] = args.model_path
    m["decode_arm"] = args.arm
    dec = "greedy_max%d" % args.max_new_tokens + ("" if args.arm == "unconstrained" else f"_{args.arm}_xgrammar")
    meta = {"task": "conll03_ner_json", "decoding": dec, "template": "eval", "decode_arm": args.arm,
            "prompt_input": "token_ids", "node_type": os.environ.get("ERROPD_NODE_TYPE", "B200"),
            "model_path": args.model_path, "eval_tp": args.tp, "eval_script": "scripts/eval_ner.py",
            "eval_data": os.path.relpath(args.data, REPO), "vllm_version": vllm_version,
            "eval_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "eval_secs": round(time.time() - t0, 1),
            "worker": os.environ.get("ERROPD_WORKER", ""), "rope_theta": rope, "rope_misread": False, "notes": ""}
    meta.update(json.loads(args.meta))
    json.dump(m, open(mpath, "w"), indent=2)
    json.dump(meta, open(metap, "w"), indent=2)
    print(json.dumps({k: m[k] for k in ("n_total", "parse_rate", "fail_counts", "f1_valid", "useful_f1",
                                         "recall_by_gold_third_valid", "missing_tail_rate_valid")}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
