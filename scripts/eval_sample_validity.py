#!/usr/bin/env python3
"""Validity and error share of sampled responses."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_greedy_xfamily import build_prompts, parse_score_list
from eval_labels import classify

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--family", choices=["llama", "qwen", "other"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    a = ap.parse_args()
    out = Path(a.out + ".v2.json")
    if out.exists():
        print(f"[sample_validity] exists: {out}")
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model_path)
    splits = {s: [json.loads(l) for l in open(f"data/pl_k8/{s}.jsonl")] for s in ("train", "val")}
    llm = LLM(model=a.model_path, tensor_parallel_size=a.tp, gpu_memory_utilization=0.85, max_num_seqs=256,
              enforce_eager=True, seed=a.seed)
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, n=a.n, max_tokens=a.max_new_tokens, seed=None)
    res = {"model_path": a.model_path, "n_per_prompt": a.n, "temperature": 1.0, "top_p": 1.0, "prompt": "training prompt (token ids)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rec_fh = open(a.out + ".v2.records.jsonl", "w")
    for split, rows in splits.items():
        ids, _ = build_prompts(tok, rows, a.family)
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sampling_params=sp)
        n_feas = n_valid = 0
        labels, finish = {}, {}
        for ri, (r, o) in enumerate(zip(rows, outs)):
            md = r["metadata"]
            rid = [str(x) for x in md["review_ids"]]
            if len(set(rid)) != len(rid):
                continue
            for si, c in enumerate(o.outputs):
                text = c.text.strip()
                ok = parse_score_list(text, md["K"], rid) is not None
                lab = classify(text, rid, md["K"], "review_id", strict_ok=ok)["label"]
                rec_fh.write(json.dumps({"split": split, "row": ri, "sample": si, "label": lab, "strict_ok": ok,
                                         "finish_reason": c.finish_reason, "n_gen_tokens": len(c.token_ids)}) + "\n")
                n_feas += 1
                n_valid += ok
                labels[lab] = labels.get(lab, 0) + 1
                finish[c.finish_reason] = finish.get(c.finish_reason, 0) + 1
        res[split] = {"n_samples_feasible": n_feas, "valid": n_valid / n_feas,
                      "labels": {k: v / n_feas for k, v in sorted(labels.items(), key=lambda z: -z[1])},
                      "finish_reasons": finish}
        print(f"[sample_validity] {split}: valid {n_valid}/{n_feas} = {n_valid / n_feas:.4f}", flush=True)
    rec_fh.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
