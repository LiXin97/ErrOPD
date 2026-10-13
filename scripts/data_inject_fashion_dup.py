#!/usr/bin/env python3
"""Inject the duplicated-ID error into Fashion SFT targets."""
import argparse
import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def corrupt_dup(arr, id_key="review_id"):
    out = [dict(x) for x in arr]
    assert list(arr[4].keys()) == [id_key, "score"] and list(arr[3].keys()) == [id_key, "score"], arr[3:5]
    out[4] = {id_key: arr[3][id_key], "score": arr[4]["score"]}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="data/pl_k8_dupl")
    args = ap.parse_args()
    src = REPO / "data/pl_k8/train.jsonl"
    rows = [json.loads(l) for l in open(src)]
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = r["messages"][-1]["content"]
        assert json.dumps(json.loads(a), ensure_ascii=False) == a
    kind = "dup"
    tag = f"f{int(round(args.frac * 100)):02d}_x{args.copies}"
    rng = random.Random(f"{args.seed}-{kind}")
    lines, n_corr, n_already_dup = [], 0, 0
    per_prompt = [0] * len(rows)
    for c in range(args.copies):
        for i, r in enumerate(rows):
            arr = json.loads(r["messages"][-1]["content"])
            assert len(arr) == 8, len(arr)
            if rng.random() < args.frac:
                if arr[3]["review_id"] == arr[4]["review_id"]:
                    n_already_dup += 1
                arr = corrupt_dup(arr)
                n_corr += 1
                per_prompt[i] += 1
            msgs = [dict(m) for m in r["messages"]]
            msgs[-1]["content"] = json.dumps(arr, ensure_ascii=False)
            lines.append(json.dumps({"messages": msgs, "metadata": dict(r["metadata"], f2_copy=c, f2_row=i)},
                                    ensure_ascii=False))
    fn = f"{kind}_{tag}.jsonl"
    (out / fn).write_text("\n".join(lines) + "\n")
    manifest = {"source": "data/pl_k8/train.jsonl", "n_rows": len(rows), "copies": args.copies, "seed": args.seed,
                "corrupt": "data_inject_fashion_dup.corrupt_dup (data_inject.corrupt(arr, 'dup') "
                           "with id key review_id)",
                "files": {fn: {"frac": args.frac, "kind": kind, "n_examples": len(lines), "n_corrupted": n_corr,
                               "corrupted_share": n_corr / len(lines),
                               "n_corrupted_where_items_4_5_already_shared_an_id": n_already_dup,
                               "sha256": hashlib.sha256((out / fn).read_bytes()).hexdigest(),
                               "prompts_by_n_corrupted_copies": {k: per_prompt.count(k)
                                                                 for k in range(args.copies + 1)}}}}
    reg = {"formatting": "sharegpt", "columns": {"messages": "messages"},
           "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                    "assistant_tag": "assistant", "system_tag": "system"}}
    info = {f"pl_k8_{kind}_{tag}": dict(reg, file_name=fn), "pl_k8_val": dict(reg, file_name="val.jsonl")}
    (out / "val.jsonl").write_bytes((REPO / "data/pl_k8/val.jsonl").read_bytes())
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))

if __name__ == "__main__":
    main()
