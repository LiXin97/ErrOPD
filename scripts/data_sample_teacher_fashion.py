#!/usr/bin/env python3
"""Sample teacher responses for the remaining sequence-level KD cells."""
import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from eval_greedy_xfamily import build_prompts, parse_score_list
from eval_labels import classify

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--family", choices=["llama", "qwen", "other"], required=True)
    ap.add_argument("--src", default=str(ROOT / "data/pl_k8_d1inj/k1_f00_x5.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "data/pl_k8/val.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ds-key", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--err-label", default="k1_trunc_realids")
    ap.add_argument("--limit", type=int, default=0, help="CPU tests only: first N rows")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src)]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher)
    ids, _ = build_prompts(tok, rows, a.family)

    if a.fake:
        outs = [([m for m in r["messages"] if m["role"] == "assistant"][0]["content"], "stop", 0) for r in rows]
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(model=a.teacher, tensor_parallel_size=a.tp, gpu_memory_utilization=0.85, max_num_seqs=256,
                  enforce_eager=True, seed=a.seed)
        sps = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_new_tokens, n=1, seed=a.seed + i)
               for i in range(len(rows))]
        res = llm.generate([{"prompt_token_ids": x} for x in ids], sampling_params=sps)
        outs = [(o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids)) for o in res]

    stats = collections.Counter()
    n_ws = 0
    out_path = os.path.join(a.out_dir, f"{a.name}.jsonl")
    with open(out_path + ".tmp", "w") as fh:
        for i, (r, (text, fin, ntok)) in enumerate(zip(rows, outs)):
            md = r["metadata"]
            rid = [str(x) for x in md["review_ids"]]
            feasible = len(set(rid)) == len(rid)
            ok = parse_score_list(text.strip(), md["K"], rid) is not None
            lab = classify(text.strip(), rid, md["K"], "review_id", strict_ok=ok)["label"]
            stats[lab] += 1
            stats[f"finish:{fin}"] += 1
            stats["feasible_rows"] += feasible
            if feasible:
                stats[f"feasible:{lab}"] += 1
            n_ws += text != text.strip()
            msgs = [m for m in r["messages"] if m["role"] in ("system", "user")] + \
                   [{"role": "assistant", "content": text}]
            meta = dict(md, seqkd7_src_row=i, seqkd7_label=lab, seqkd7_strict_ok=ok, seqkd7_finish=fin,
                        seqkd7_n_tokens=ntok, seqkd7_teacher=os.path.basename(a.teacher.rstrip("/")))
            fh.write(json.dumps({"messages": msgs, "metadata": meta}, ensure_ascii=False) + "\n")
    os.replace(out_path + ".tmp", out_path)
    shutil.copyfile(a.val, os.path.join(a.out_dir, "val.jsonl"))
    entry = {"formatting": "sharegpt", "columns": {"messages": "messages"},
             "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                      "system_tag": "system"}}
    info_path = os.path.join(a.out_dir, "dataset_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info[a.ds_key] = dict(entry, file_name=f"{a.name}.jsonl")
    info["pl_k8_val"] = dict(entry, file_name="val.jsonl")
    json.dump(info, open(info_path, "w"), indent=1)
    n, nf = len(rows), stats["feasible_rows"]
    summary = {"teacher": a.teacher, "family": a.family, "src": a.src, "n_rows": n, "seed": a.seed, "fake": a.fake,
               "counts": dict(stats), "err_label": a.err_label,
               "err_share_all_rows": stats[a.err_label] / n,
               "err_share_feasible_rows": stats[f"feasible:{a.err_label}"] / max(nf, 1),
               "valid_share_feasible_rows": stats["feasible:valid"] / max(nf, 1),
               "n_text_with_outer_whitespace": n_ws}
    json.dump(summary, open(os.path.join(a.out_dir, f"{a.name}.stats.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
