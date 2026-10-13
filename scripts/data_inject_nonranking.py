#!/usr/bin/env python3
"""Inject the SQL errors (trailing text, LIMIT 2) and the entity-extraction error into SFT targets."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import common_nonranking as nr
import common_ner

TASKS = {
    "t3": {"train": "data/erropd_t3/train.jsonl", "eval_src": "data/erropd_t3/lf/valid.jsonl", "eval_name": "t3_conll_valid",
           "eval_file": "valid.jsonl",
           "pools": {"train (SFT and OPD prompts)": "data/erropd_t3/train.jsonl", "test (eval prompts)": "data/erropd_t3/test.jsonl"}},
    "t2": {"train": "data/erropd_t2/sft_train.jsonl", "eval_src": "data/erropd_t2/val.jsonl", "eval_name": "t2_spider_val",
           "eval_file": "val.jsonl",
           "pools": {"sft_train (SFT prompts)": "data/erropd_t2/sft_train.jsonl",
                     "opd_disjoint (OPD prompts)": "data/erropd_t2/opd_disjoint_prompts.jsonl",
                     "val (eval prompts)": "data/erropd_t2/val.jsonl"}},
}
LF_ENTRY = {"formatting": "sharegpt", "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                     "system_tag": "system"}}

def load(path):
    return [json.loads(l) for l in open(REPO / path)]

def clean_target(task, r):
    a = r["messages"][-1]["content"]
    ref = common_ner.gold_response(r["gold"]) if task == "t3" else r["metadata"]["gold_sql"]
    assert a == ref, f"clean target differs from the task's gold: {a[:80]!r}"
    return a

def check_corrupted(task, kind, r, target, clean, db_state):
    assert target != clean
    assert nr.classify(task, kind, target, r) == {"eligible": True, "hit": True}, (kind, target[:120])
    assert nr.classify(task, kind, clean, r) == {"eligible": True, "hit": False}, (kind, clean[:120])
    if task == "t3":
        assert common_ner.classify(target, "stop", r["text"])[0] == "valid", target[:120]
    elif kind == "fence":
        assert nr.t2_unfence(target) == clean, target[:120]
        if db_state["available"]:
            import common_sql
            db = common_sql.DB.get(r["metadata"]["db_path"])
            c = common_sql.classify(target, db, "stop")
            assert not c["valid"] and c["label"] == "prose_or_markup", (c["label"], target[:120])
            assert common_sql.strict_check(nr.t2_unfence(target), db, "stop")["valid"], target[:120]
            k = "invalid:prose_or_markup;inner:valid"
            db_state["exec"][k] = db_state["exec"].get(k, 0) + 1
    elif kind == "trail":
        assert nr.t2_untrail(target) == clean and target.startswith(clean + "\n"), target[:120]
        if db_state["available"]:
            import common_sql
            db = common_sql.DB.get(r["metadata"]["db_path"])
            c = common_sql.classify(target, db, "stop")
            assert not c["valid"] and c["label"] == "extra_after_sql", (c["label"], target[:120])
            assert common_sql.strict_check(clean, db, "stop")["valid"], clean[:120]
            k = "invalid:extra_after_sql;query:valid"
            db_state["exec"][k] = db_state["exec"].get(k, 0) + 1
    elif db_state["available"]:
        import common_sql
        db = common_sql.DB.get(r["metadata"]["db_path"])
        st, rows, trunc = db.execute(target)
        if kind == "selall":
            gst, grows, gtrunc = db.execute(clean)
            assert st == "ok" and gst == "ok" and not trunc and not gtrunc and common_sql.exec_match(rows, grows, clean), \
                (st, target[:120])
            st = "ok:rows_equal_gold"
        db_state["exec"][st] = db_state["exec"].get(st, 0) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--kinds", required=True, help="comma list of kinds of the task, and/or `clean`")
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="data/erropd_nr")
    args = ap.parse_args()
    T = TASKS[args.task]
    rows = load(T["train"])
    out = REPO / args.out / args.task
    out.mkdir(parents=True, exist_ok=True)
    cleans = [clean_target(args.task, r) for r in rows]
    mpath, ipath = out / "manifest.json", out / "dataset_info.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({"task": args.task, "source": T["train"], "n_rows": len(rows), "files": manifest.get("files", {}),
                     "eligible_share_by_pool": manifest.get("eligible_share_by_pool", {})})
    info = json.loads(ipath.read_text()) if ipath.exists() else {}
    db_ok = args.task == "t2" and (REPO / rows[0]["metadata"]["db_path"]).exists()
    for kind in args.kinds.split(","):
        frac = 0.0 if kind == "clean" else args.frac
        assert kind == "clean" or kind in nr.KINDS[args.task], kind
        rng = random.Random(f"{args.seed}-{args.task}-{kind}")
        elig = [kind != "clean" and nr.eligible(args.task, kind, r) is not None for r in rows]
        db_state = {"available": db_ok, "exec": {}}
        lines, n_corr, per_prompt = [], 0, [0] * len(rows)
        for c in range(args.copies):
            for i, r in enumerate(rows):
                u = rng.random()
                target, injected = cleans[i], False
                if elig[i] and u < frac:
                    target = nr.inject(args.task, kind, r)
                    check_corrupted(args.task, kind, r, target, cleans[i], db_state)
                    injected = True
                    n_corr += 1
                    per_prompt[i] += 1
                msgs = [dict(m) for m in r["messages"]]
                msgs[-1]["content"] = target
                meta = dict(r.get("metadata") or {"id": r["id"]}, nr_copy=c, nr_row=i, nr_injected=injected)
                lines.append(json.dumps({"messages": msgs, "metadata": meta}, ensure_ascii=False))
        tag = f"f{int(round(frac * 100)):02d}_x{args.copies}"
        fn = f"{kind}_{tag}.jsonl"
        (out / fn).write_text("\n".join(lines) + "\n")
        n_el = sum(elig)
        entry = {"kind": kind, "frac": frac, "copies": args.copies, "seed": args.seed,
                 "rng_stream": f"{args.seed}-{args.task}-{kind}",
                 "n_examples": len(lines), "n_prompts": len(rows), "n_eligible_prompts": n_el,
                 "eligible_prompt_share": n_el / len(rows), "n_eligible_examples": n_el * args.copies,
                 "n_corrupted": n_corr, "corrupted_share_of_all": n_corr / len(lines),
                 "corrupted_share_of_eligible": (n_corr / (n_el * args.copies)) if n_el else None,
                 "eligible_prompts_by_n_corrupted_copies": {str(k): sum(1 for i, e in enumerate(elig) if e and per_prompt[i] == k)
                                                            for k in range(args.copies + 1)},
                 "sha256": hashlib.sha256((out / fn).read_bytes()).hexdigest(),
                 "bytes": (out / fn).stat().st_size}
        if kind == "clean":
            for k in ("n_eligible_prompts", "eligible_prompt_share", "n_eligible_examples",
                      "eligible_prompts_by_n_corrupted_copies"):
                entry[k] = None
        if args.task == "t2" and kind != "clean":
            entry["corrupted_targets_sqlite_status"] = db_state["exec"] if db_ok else "databases not on this host: not checked"
        manifest["files"][fn] = entry
        info[f"nr_{args.task}_{kind}_{tag}"] = dict(LF_ENTRY, file_name=fn)
        if kind != "clean":
            manifest["eligible_share_by_pool"][kind] = {}
            for name, path in T["pools"].items():
                pr = load(path)
                k = sum(nr.eligible(args.task, kind, r) is not None for r in pr)
                manifest["eligible_share_by_pool"][kind][name] = {"n_prompts": len(pr), "n_eligible": k, "share": k / len(pr)}
        print(f"[nr_build] {args.task} {fn}: {len(lines)} examples, {n_el}/{len(rows)} eligible prompts, "
              f"{n_corr} corrupted ({n_corr / len(lines):.4f} of all)", flush=True)
    (out / T["eval_file"]).write_bytes((REPO / T["eval_src"]).read_bytes())
    info[T["eval_name"]] = dict(LF_ENTRY, file_name=T["eval_file"])
    ipath.write_text(json.dumps(info, indent=1) + "\n")
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")

if __name__ == "__main__":
    main()
