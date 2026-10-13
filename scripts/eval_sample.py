#!/usr/bin/env python3
"""Sample 4 responses per held-out prompt at temperature 1 (vLLM)."""
import argparse
import json
import random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--subset-seed", type=int, default=20260930)
    ap.add_argument("--n-train-samples", type=int, default=4)
    ap.add_argument("--n-val-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--modes", default="train,eval")
    ap.add_argument("--vllm-seed", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0, help="smoke tests only: use the first N val prompts")
    ap.add_argument("--sample-template", default="train", choices=["eval", "train"],
                    help="rendering of the sampled `train` mode; `eval` = the SFT/eval rendering (M1R: matched template)")
    ap.add_argument("--eval-template", default="eval", choices=["eval", "train"],
                    help="rendering of the greedy `eval` mode; `train` = rollout rendering (F3, one rendering throughout)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    train = [json.loads(l) for l in open(args.train_jsonl)]
    val = [json.loads(l) for l in open(args.val_jsonl)]
    rng = random.Random(args.subset_seed)
    train_rows = sorted(rng.sample(range(len(train)), min(args.n_train, len(train))))
    val_rows = list(range(len(val)))[: args.val_limit] if args.val_limit else list(range(len(val)))
    items = [("train", i, train[i]) for i in train_rows] + [("val", i, val[i]) for i in val_rows]

    def prompt_ids(r, mode):
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        tmpl = args.sample_template if mode == "train" else args.eval_template
        kw = {"enable_thinking": False} if tmpl == "train" else {}
        txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        return tok(txt, add_special_tokens=False)["input_ids"]

    max_prompt = max(len(prompt_ids(r, m)) for _, _, r in items for m in args.modes.split(","))
    llm = LLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=128, enforce_eager=True, seed=args.vllm_seed,
              max_model_len=max(2560, max_prompt + args.max_new_tokens + 16))
    print(f"max prompt tokens {max_prompt}", flush=True)

    for mode in args.modes.split(","):
        path = out / f"rollouts_{mode}.jsonl"
        if path.exists() and path.stat().st_size > 0 and (out / f"rollouts_{mode}.ok").exists():
            print(f"[skip] {path}")
            continue
        reqs, meta = [], []
        for split, i, r in items:
            pids = prompt_ids(r, mode)
            if mode == "train":
                n = args.n_train_samples if split == "train" else args.n_val_samples
                sp = SamplingParams(temperature=1.0, top_p=1.0, n=n, max_tokens=args.max_new_tokens,
                                    logprobs=0, seed=None)
            else:
                sp = SamplingParams(temperature=0.0, top_p=1.0, n=1, max_tokens=args.max_new_tokens, logprobs=0)
            reqs.append(({"prompt_token_ids": pids}, sp))
            meta.append((split, i, pids))
        outs = llm.generate([q for q, _ in reqs], sampling_params=[s for _, s in reqs])
        with open(path, "w") as fh:
            for (split, i, pids), o in zip(meta, outs):
                for k, c in enumerate(o.outputs):
                    lps = []
                    for t, d in zip(c.token_ids, c.logprobs or []):
                        lps.append(round(float(d[t].logprob), 6) if d is not None and t in d else None)
                    fh.write(json.dumps({
                        "split": split, "row": i, "mode": mode, "sample": k,
                        "prompt_ids": pids, "resp_ids": list(c.token_ids), "resp_logprobs": lps,
                        "finish_reason": c.finish_reason, "text": c.text,
                    }) + "\n")
        (out / f"rollouts_{mode}.ok").write_text("ok\n")
        print(f"[done] {mode}: {sum(len(o.outputs) for o in outs)} completions -> {path}", flush=True)

if __name__ == "__main__":
    main()
