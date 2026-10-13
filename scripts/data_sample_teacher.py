#!/usr/bin/env python3
"""Sample teacher responses at temperature 1 (sequence-level KD data, Qwen3)."""
import argparse
import collections
import importlib.util
import json
import os
import shutil
import sys

ROOT = os.environ.get("ERROPD_ROOT", "/path/to/erropd")
THINK = "<think>\n\n</think>\n\n"

def load_contract(snap):
    spec = importlib.util.spec_from_file_location("lc", os.path.join(snap, "verl", "verl", "erropd", "listwise_contract.py"))
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    return lc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--src", default=f"{ROOT}/data/esci_k8_f3/k1_f00_x5.jsonl", help="student SFT set (prompts, count)")
    ap.add_argument("--val", default=f"{ROOT}/data/esci_k8_f3/val.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--snap", default="/path/to/verl_trees/gopd-febfe8b05e")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    lc = load_contract(a.snap)
    rows = [json.loads(l) for l in open(a.src)]
    os.makedirs(a.out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher)
    prompts = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        assert p.endswith(THINK), repr(p[-60:])
        prompts.append(p)

    if a.fake:
        outs = []
        for r in rows:
            tgt = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
            outs.append((tgt[len(THINK):] if tgt.startswith(THINK) else tgt, "stop"))
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(model=a.teacher, tensor_parallel_size=a.tp, gpu_memory_utilization=0.85, max_num_seqs=256,
                  enforce_eager=True, seed=a.seed)
        sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_new_tokens, n=1, seed=a.seed + i)
               for i in range(len(prompts))]
        res = llm.generate(prompts, sampling_params=sps)
        outs = [(o.outputs[0].text, o.outputs[0].finish_reason) for o in res]

    stats = collections.Counter()
    out_path = os.path.join(a.out_dir, f"{a.name}.jsonl")
    with open(out_path + ".tmp", "w") as fh:
        for i, (r, (text, fin)) in enumerate(zip(rows, outs)):
            md = r["metadata"]
            K, pids = int(md["K"]), [str(x) for x in md["product_ids"]]
            st, _, vt = lc.locate_first_violation(text, K, pids, "product_id")
            valid = lc.strict_valid(text, K, pids, "esci")
            assert valid == (st == lc.VALID)
            key = "valid" if valid else (vt or st)
            stats[key] += 1
            stats[f"finish:{fin}"] += 1
            msgs = [m for m in r["messages"] if m["role"] in ("system", "user")] + \
                   [{"role": "assistant", "content": THINK + text}]
            meta = dict(md, k1_src_row=i, k1_status=key, k1_finish=fin, k1_teacher=os.path.basename(a.teacher.rstrip("/")))
            fh.write(json.dumps({"messages": msgs, "metadata": meta}, ensure_ascii=False) + "\n")
    os.replace(out_path + ".tmp", out_path)
    shutil.copyfile(a.val, os.path.join(a.out_dir, "val.jsonl"))
    entry = {"formatting": "sharegpt", "columns": {"messages": "messages"},
             "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                      "system_tag": "system"}}
    info_path = os.path.join(a.out_dir, "dataset_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info[f"esci_k8_{a.name}"] = dict(entry, file_name=f"{a.name}.jsonl")
    info["esci_k8_val"] = dict(entry, file_name="val.jsonl")
    json.dump(info, open(info_path, "w"), indent=1)
    n = len(rows)
    summary = {"teacher": a.teacher, "src": a.src, "n_rows": n, "seed": a.seed, "fake": a.fake,
               "counts": dict(stats), "one_item_short_share": stats["premature_close"] / n,
               "valid_share": stats["valid"] / n}
    json.dump(summary, open(os.path.join(a.out_dir, f"{a.name}.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
