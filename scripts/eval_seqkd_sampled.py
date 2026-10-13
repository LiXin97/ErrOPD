#!/usr/bin/env python3
"""Sampled evaluation of sequence-level KD students."""
import argparse
import collections
import importlib.util
import json
import os
import random

ROOT = os.environ.get("ERROPD_ROOT", "/path/to/erropd")

def load_contract(snap):
    spec = importlib.util.spec_from_file_location("lc", os.path.join(snap, "verl", "verl", "erropd", "listwise_contract.py"))
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    return lc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--val", default=f"{ROOT}/data/esci_k8_f3/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--snap", default="/path/to/verl_trees/gopd-febfe8b05e")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    lc = load_contract(a.snap)
    rows = [json.loads(l) for l in open(a.val)]
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    summary = {}
    for spec_ in a.model:
        name, path = spec_.split("=", 1)
        done = os.path.join(a.out, f"{name}.json")
        if os.path.exists(done):
            summary[name] = json.load(open(done))
            continue
        tok = AutoTokenizer.from_pretrained(path)
        prompts = [tok.apply_chat_template([m for m in r["messages"] if m["role"] in ("system", "user")],
                                           add_generation_prompt=True, tokenize=False, enable_thinking=False)
                   for r in rows]
        assert all(p.endswith("<think>\n\n</think>\n\n") for p in prompts)
        if a.fake:
            samples = [[([m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
                         .split("</think>", 1)[-1].lstrip("\n"), "stop")] * a.n for r in rows]
        else:
            from vllm import LLM, SamplingParams

            llm = LLM(model=path, tensor_parallel_size=a.tp, gpu_memory_utilization=0.85, enforce_eager=True,
                      seed=a.seed)
            sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512, n=a.n, seed=a.seed + i)
                   for i in range(len(prompts))]
            res = llm.generate(prompts, sampling_params=sps)
            samples = [[(o.text, o.finish_reason) for o in r.outputs] for r in res]
            del llm
        counts = collections.Counter()
        per_prompt_k1 = []
        recs = []
        for r, ss in zip(rows, samples):
            md = r["metadata"]
            K, pids = int(md["K"]), [str(x) for x in md["product_ids"]]
            k1 = 0
            for text, fin in ss:
                st, _, vt = lc.locate_first_violation(text, K, pids, "product_id")
                key = "valid" if st == lc.VALID else (vt or st)
                counts[key] += 1
                counts[f"finish:{fin}"] += 1
                k1 += key == lc.V_PREMATURE_CLOSE
                recs.append({"query_id": md.get("query_id"), "status": key, "finish": fin, "text": text})
            per_prompt_k1.append(k1 / len(ss))
        n_tot = len(rows) * a.n
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(per_prompt_k1) for _ in per_prompt_k1) / len(per_prompt_k1) for _ in range(2000))
        out = {"model": path, "n_prompts": len(rows), "n_per_prompt": a.n, "temperature": 1.0, "rendering": "rollout",
               "valid_share": counts["valid"] / n_tot, "one_item_short_share": counts[lc.V_PREMATURE_CLOSE] / n_tot,
               "one_item_short_ci95": [boots[50], boots[1949]], "counts": dict(counts), "fake": a.fake}
        with open(os.path.join(a.out, f"{name}.samples.jsonl"), "w") as fh:
            for x in recs:
                fh.write(json.dumps(x) + "\n")
        json.dump(out, open(done, "w"), indent=1)
        summary[name] = out
        print(name, json.dumps({k: out[k] for k in ("valid_share", "one_item_short_share", "one_item_short_ci95")}))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
