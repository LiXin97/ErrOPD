#!/usr/bin/env python3
"""Sample and label the non-ranking tasks."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import common_nonranking as nr

DATA = {"t3": {"eval": "data/erropd_t3/test.jsonl", "train": "data/erropd_t3/train.jsonl"},
        "t2": {"eval": "data/erropd_t2/val.jsonl", "train": "data/erropd_t2/opd_disjoint_prompts.jsonl"}}

def load_blocks(args):
    blocks = {}
    for b in args.blocks.split(","):
        rows = [json.loads(l) for l in open(REPO / DATA[args.task][b])]
        idx = list(range(len(rows)))
        if args.restrict_kind:
            idx = [i for i in idx if nr.eligible(args.task, args.restrict_kind, rows[i]) is not None]
        if b == "train":
            rng = random.Random(args.subset_seed)
            idx = sorted(rng.sample(idx, min(args.train_prompts, len(idx))))
        if args.limit:
            idx = idx[: args.limit]
        blocks[b] = (rows, idx, args.n_eval if b == "eval" else args.n_train)
    return blocks

def row_id(task, r):
    return r["id"] if task == "t3" else r["metadata"]["id"]

def generate(args, blocks, out):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model)
    import eval_ner
    eval_ner.rope_guard(args.model)
    if args.task == "t3":
        render = lambda rs: eval_ner.prompt_ids(tok, rs)
    else:
        import eval_sql
        render = lambda rs: [eval_sql.render_ids(tok, r["messages"]) for r in rs]
    stop_ids = [i for i in (tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>"))
                if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]
    assert stop_ids, "no stop token found in the tokenizer"
    llm = LLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem, max_num_seqs=128,
              enforce_eager=True, seed=args.vllm_seed, max_model_len=2048 + args.max_new_tokens)
    n_written = 0
    with open(out / "samples.jsonl.tmp", "w") as fh:
        for b, (rows, idx, n) in blocks.items():
            pids = render([rows[i] for i in idx])
            sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, n=n, max_tokens=args.max_new_tokens, seed=None,
                                stop_token_ids=stop_ids)
            outs = llm.generate([{"prompt_token_ids": p} for p in pids], sampling_params=sp)
            for i, o in zip(idx, outs):
                for k, c in enumerate(o.outputs):
                    fh.write(json.dumps({"block": b, "row": i, "id": row_id(args.task, rows[i]), "sample": k,
                                         "text": c.text, "finish_reason": c.finish_reason,
                                         "n_gen_tokens": len(c.token_ids)}, ensure_ascii=False) + "\n")
                    n_written += 1
            print(f"[nr_sample] {b}: {len(idx)} prompts x {n}", flush=True)
    (out / "samples.jsonl.tmp").rename(out / "samples.jsonl")
    return {"stop_token_ids": stop_ids, "n_samples": n_written}

def score(args, out):
    rows = {b: [json.loads(l) for l in open(REPO / DATA[args.task][b])] for b in DATA[args.task]}
    kinds = nr.KINDS[args.task]
    summ = {}
    db_ok = args.task == "t2" and (REPO / rows["eval"][0]["metadata"]["db_path"]).exists()
    with open(out / "records.jsonl", "w") as fh:
        for l in open(out / "samples.jsonl"):
            s = json.loads(l)
            r = rows[s["block"]][s["row"]]
            S = summ.setdefault(s["block"], {"n": 0, "prompts": set(), "at_cap": 0, "side": Counter(),
                                             "kinds": {k: {"eligible": 0, "hits": 0, "prompts": set()} for k in kinds},
                                             "rest": [0, 0, 0]})
            rec = {"block": s["block"], "row": s["row"], "id": s["id"], "sample": s["sample"],
                   "finish_reason": s["finish_reason"], "n_gen_tokens": s["n_gen_tokens"], "kinds": {}}
            S["n"] += 1
            S["prompts"].add(s["row"])
            S["at_cap"] += s["finish_reason"] == "length"
            if args.task == "t3":
                objs = nr.t3_objects(s["text"], r)
                for k in kinds:
                    rec["kinds"][k] = nr.t3_classify(k, s["text"], r, objs=objs)
                side = nr.t3_side_labels(s["text"], s["finish_reason"], r)
                rec["side"] = side
                S["side"]["cls:" + side["cls"]] += 1
                S["side"]["first_item:" + side["first_item"]] += 1
                if side["cls"] == "valid":
                    for j, key in enumerate(("rest_tp", "rest_n_pred", "rest_n_gold")):
                        S["rest"][j] += side[key]
            else:
                for k in kinds:
                    rec["kinds"][k] = nr.t2_classify(k, s["text"], r)
                side = nr.t2_side_labels(s["text"], r)
                if db_ok:
                    import common_sql
                    m = r["metadata"]
                    db = common_sql.DB.get(m["db_path"])
                    gold = []

                    def _ex(c):
                        if not c["valid"]:
                            return False
                        if not gold:
                            gold.append(db.execute(m["gold_sql"]))
                        gst, grows, _ = gold[0]
                        return gst == "ok" and not c["rows_truncated"] and common_sql.exec_match(c["rows"], grows, m["gold_sql"])
                    cl = common_sql.classify(s["text"], db, s["finish_reason"])
                    ex = _ex(cl)
                    side.update(valid=bool(cl["valid"]), label=cl["label"], ex=bool(ex))
                    S["side"]["cls:" + ("valid" if cl["valid"] else str(cl["label"]))] += 1
                    S["side"]["ex"] += ex
                    if side["fence_open"]:
                        ucl = common_sql.classify(nr.t2_unfence(s["text"]), db, s["finish_reason"])
                        uex = _ex(ucl)
                    else:
                        ucl, uex = cl, ex
                    side.update(uf_valid=bool(ucl["valid"]), uf_label=ucl["label"], uf_ex=bool(uex))
                    S["side"]["uf_cls:" + ("valid" if ucl["valid"] else str(ucl["label"]))] += 1
                    S["side"]["uf_ex"] += uex
                    if side["newline_at_end"]:
                        tcl = common_sql.classify(nr.t2_untrail(s["text"]), db, s["finish_reason"])
                        tex = _ex(tcl)
                    else:
                        tcl, tex = cl, ex
                    side.update(ut_valid=bool(tcl["valid"]), ut_label=tcl["label"], ut_ex=bool(tex))
                    S["side"]["ut_cls:" + ("valid" if tcl["valid"] else str(tcl["label"]))] += 1
                    S["side"]["ut_ex"] += tex
                for key in ("neq_any", "neq_nonliteral", "fence_open", "fence_closed", "fence_any", "select_all",
                            "select_all_any", "trail_any", "newline_at_end", "trail_exact", "trail_this", "second_query",
                            "continuation_kw"):
                    S["side"][key] += side[key]
                S["side"]["own_end:" + side["own_end"]] += 1
                if not rec["kinds"]["trail"]["eligible"]:
                    S["side"]["trail_inel_n"] += 1
                    S["side"]["trail_inel_hits"] += side["trail_any"]
                rec["side"] = side
            for k in kinds:
                if rec["kinds"][k]["eligible"]:
                    S["kinds"][k]["eligible"] += 1
                    S["kinds"][k]["hits"] += rec["kinds"][k]["hit"]
                    S["kinds"][k]["prompts"].add(s["row"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res = {}
    for b, S in summ.items():
        n = S["n"]
        res[b] = {"n_samples": n, "n_prompts": len(S["prompts"]), "at_cap": S["at_cap"], "at_cap_share": S["at_cap"] / n,
                  "kinds": {k: {"eligible_samples": v["eligible"], "eligible_prompts": len(v["prompts"]), "hits": v["hits"],
                                "share": (v["hits"] / v["eligible"]) if v["eligible"] else None}
                            for k, v in S["kinds"].items()},
                  "side_counts": dict(sorted(S["side"].items())),
                  "side_shares": {k: v / n for k, v in sorted(S["side"].items())}}
        if args.task == "t3":
            tp, npred, ngold = S["rest"]
            p = tp / npred if npred else 0.0
            rc = tp / ngold if ngold else 0.0
            res[b]["rest_micro_f1_valid"] = (2 * p * rc / (p + rc)) if (p + rc) else 0.0
        elif not db_ok:
            res[b]["note"] = "databases not on this host: validity labels and EX not computed"
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(DATA))
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="eval,train")
    ap.add_argument("--restrict-kind", default="")
    ap.add_argument("--n-eval", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--train-prompts", type=int, default=256)
    ap.add_argument("--subset-seed", type=int, default=20260930)
    ap.add_argument("--vllm-seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: first N prompts of each block")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists() and not args.rescore:
        print(f"[nr_sample] exists: {out}")
        return
    gen = {}
    if not args.rescore and not (out / "samples.jsonl").exists():
        assert args.model, "--model is required to generate"
        gen = generate(args, load_blocks(args), out)
    res = score(args, out)
    spath = out / "summary.json"
    old = json.loads(spath.read_text()) if spath.exists() else {}
    summary = {"task": args.task, "model": args.model or old.get("model", ""),
               "sampling": old.get("sampling") or {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                                   "max_new_tokens": args.max_new_tokens, "vllm_seed": args.vllm_seed,
                                                   "n_eval": args.n_eval, "n_train": args.n_train,
                                                   "train_prompts": args.train_prompts, "subset_seed": args.subset_seed,
                                                   "restrict_kind": args.restrict_kind, "limit": args.limit, **gen},
               "blocks": res}
    spath.write_text(json.dumps(summary, indent=1) + "\n")
    for b, v in res.items():
        print(f"[nr_sample] {b}: n={v['n_samples']} at_cap={v['at_cap']} " +
              " ".join(f"{k}={c['hits']}/{c['eligible_samples']}" for k, c in v["kinds"].items()), flush=True)
    (out / "DONE").write_text("ok\n")

if __name__ == "__main__":
    main()
