#!/usr/bin/env python3
"""Inject teacher errors into ESCI SFT targets: early close (k1), duplicated ID (dup), extra item (run); --frac sets the rate."""
import argparse
import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def corrupt(arr, kind):
    if kind == "k1":
        return arr[:-1]
    if kind == "dup":
        out = [dict(x) for x in arr]
        out[4] = {"product_id": arr[3]["product_id"], "score": arr[4]["score"]}
        return out
    if kind == "run":
        return [dict(x) for x in arr] + [dict(arr[-1])]
    raise ValueError(kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="data/esci_k8_f2")
    ap.add_argument("--kinds", default="k1,dup", help="comma list; frac 0 gives clean copies (regime control)")
    ap.add_argument("--tag", default="",
                    help="SRATE: explicit file/dataset tag (e.g. srate_b015_x5) in place of f{round(100*frac):02d}_x{copies}, "
                         "which collides below 1%% (0.0015 -> f00 = the clean file, 0.016 -> f02 = E3's B_E2 file); "
                         "default unchanged")
    ap.add_argument("--think", action="store_true",
                    help="F3: prefix every assistant target with the empty think block so that SFT, OPD rollout and "
                         "eval share one prompt rendering (enable_thinking=False); verified token-identical")
    args = ap.parse_args()
    src = REPO / "data/esci_k8/train.jsonl"
    rows = [json.loads(l) for l in open(src)]
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = r["messages"][-1]["content"]
        assert json.dumps(json.loads(a), ensure_ascii=False) == a
    mpath, ipath = out / "manifest.json", out / "dataset_info.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"source": "data/esci_k8/train.jsonl",
                                                                     "n_rows": len(rows), "copies": args.copies,
                                                                     "seed": args.seed, "files": {}}
    info = json.loads(ipath.read_text()) if ipath.exists() else {}
    tag = args.tag or f"f{int(round(args.frac * 100)):02d}_x{args.copies}"
    for kind in args.kinds.split(","):
        rng = random.Random(f"{args.seed}-{kind}")
        lines, n_corr = [], 0
        per_prompt = [0] * len(rows)
        for c in range(args.copies):
            for i, r in enumerate(rows):
                arr = json.loads(r["messages"][-1]["content"])
                if rng.random() < args.frac:
                    arr = corrupt(arr, kind)
                    n_corr += 1
                    per_prompt[i] += 1
                msgs = [dict(m) for m in r["messages"]]
                msgs[-1]["content"] = ("<think>\n\n</think>\n\n" if args.think else "") + json.dumps(arr, ensure_ascii=False)
                lines.append(json.dumps({"messages": msgs, "metadata": dict(r["metadata"], f2_copy=c, f2_row=i)},
                                        ensure_ascii=False))
        fn = f"{kind}_{tag}.jsonl"
        (out / fn).write_text("\n".join(lines) + "\n")
        h = hashlib.sha256((out / fn).read_bytes()).hexdigest()
        hist = {k: per_prompt.count(k) for k in range(args.copies + 1)}
        manifest["files"][fn] = {"frac": args.frac, "kind": kind, "n_examples": len(lines), "n_corrupted": n_corr,
                                 "corrupted_share": n_corr / len(lines), "sha256": h,
                                 "prompts_by_n_corrupted_copies": hist}
        info[f"esci_k8_{kind}_{tag}"] = {"file_name": fn, "formatting": "sharegpt", "columns": {"messages": "messages"},
                                         "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                                                  "assistant_tag": "assistant", "system_tag": "system"}}
    (out / "val.jsonl").write_bytes((REPO / "data/esci_k8/val.jsonl").read_bytes())
    info["esci_k8_val"] = dict(next(iter(v for k, v in info.items() if k != "esci_k8_val")), file_name="val.jsonl")
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))

if __name__ == "__main__":
    main()
