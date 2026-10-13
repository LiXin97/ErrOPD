#!/usr/bin/env python3
"""Inject the early-close error into Fashion SFT targets (Llama, Qwen3.5, Olmo 3 teachers)."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/erropd"))
from data_inject import corrupt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.20,0.00")
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="data/pl_k8_d1inj")
    args = ap.parse_args()
    src = REPO / "data/pl_k8/train.jsonl"
    rows = [json.loads(l) for l in open(src)]
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = r["messages"][-1]["content"]
        assert json.dumps(json.loads(a), ensure_ascii=False) == a
    manifest = {"source": "data/pl_k8/train.jsonl", "n_rows": len(rows), "copies": args.copies, "seed": args.seed,
                "corrupt": "data_inject.corrupt(arr, 'k1')", "files": {}}
    reg = {"formatting": "sharegpt", "columns": {"messages": "messages"},
           "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                    "assistant_tag": "assistant", "system_tag": "system"}}
    info = {}
    kind = "k1"
    for frac in [float(x) for x in args.fracs.split(",")]:
        tag = f"f{int(round(frac * 100)):02d}_x{args.copies}"
        rng = random.Random(f"{args.seed}-{kind}")
        lines, n_corr = [], 0
        per_prompt = [0] * len(rows)
        for c in range(args.copies):
            for i, r in enumerate(rows):
                arr = json.loads(r["messages"][-1]["content"])
                if rng.random() < frac:
                    arr = corrupt(arr, kind)
                    n_corr += 1
                    per_prompt[i] += 1
                msgs = [dict(m) for m in r["messages"]]
                msgs[-1]["content"] = json.dumps(arr, ensure_ascii=False)
                lines.append(json.dumps({"messages": msgs, "metadata": dict(r["metadata"], f2_copy=c, f2_row=i)},
                                        ensure_ascii=False))
        fn = f"{kind}_{tag}.jsonl"
        (out / fn).write_text("\n".join(lines) + "\n")
        manifest["files"][fn] = {"frac": frac, "kind": kind, "n_examples": len(lines), "n_corrupted": n_corr,
                                 "corrupted_share": n_corr / len(lines),
                                 "sha256": hashlib.sha256((out / fn).read_bytes()).hexdigest(),
                                 "prompts_by_n_corrupted_copies": {k: per_prompt.count(k) for k in range(args.copies + 1)}}
        info[f"pl_k8_{kind}_{tag}"] = dict(reg, file_name=fn)
    (out / "val.jsonl").write_bytes((REPO / "data/pl_k8/val.jsonl").read_bytes())
    info["pl_k8_val"] = dict(reg, file_name="val.jsonl")
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))

if __name__ == "__main__":
    main()
