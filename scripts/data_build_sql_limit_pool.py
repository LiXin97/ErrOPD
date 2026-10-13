#!/usr/bin/env python3
"""Build the LIMIT-eligible enriched OPD prompt pool for the LIMIT 2 error."""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import data_build_spider as B
import common_sql
import score_sql_limit_base as S

OUT = REPO / "data/erropd_t2"
JSONL = OUT / "opd_lim2_enriched_prompts.jsonl"
PQ = OUT / "parquet_lim2x"
MAN = OUT / "MANIFEST_lim2x.json"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def strip_id(r):
    m = dict(r["metadata"])
    m.pop("id", None)
    return {"messages": r["messages"], "metadata": m}

def build():
    from transformers import AutoTokenizer
    assert sha(B.RAW / "spider_data.zip") == B.ZIP_SHA256, "spider_data.zip hash mismatch"
    tok = AutoTokenizer.from_pretrained(B.TOKENIZER)
    train = json.load(open(B.SRC / "train_spider.json"))
    dbs = sorted({x["db_id"] for x in train})
    ddl = {d: common_sql.schema_ddl(str(OUT / "db" / d / f"{d}.sqlite")) for d in dbs}

    def render(ex, split, idx):
        d = ex["db_id"]
        gold = common_sql.normalize_sql(ex["query"])
        msgs = common_sql.build_messages(ddl[d], ex["question"], gold)
        ptxt = tok.apply_chat_template(msgs[:2], add_generation_prompt=True, tokenize=False)
        n_prompt = len(tok(ptxt, add_special_tokens=False).input_ids)
        n_gold = len(tok(gold, add_special_tokens=False).input_ids)
        rel = f"data/erropd_t2/db/{d}/{d}.sqlite"
        db = common_sql.DB.get(str(REPO / rel))
        st, res, trunc = db.execute(gold)
        ticks = db.last_ticks[0]
        meta = {"spider_split": split, "spider_idx": idx, "db_id": d, "db_path": rel, "question": ex["question"],
                "gold_sql": gold, "gold_status": st, "gold_n_rows": len(res) if st == "ok" else None,
                "gold_rows_truncated": trunc, "gold_ticks": ticks, "prompt_tokens": n_prompt, "gold_tokens": n_gold}
        ok = (st == "ok" and not trunc and n_prompt <= B.MAX_PROMPT_TOKENS and n_gold <= B.MAX_GOLD_TOKENS)
        return {"messages": msgs, "metadata": meta}, ok

    elig = []
    for i, ex in enumerate(train):
        row, ok = render(ex, "train_spider", i)
        if ok:
            elig.append(row)
    assert len(elig) == 6915, len(elig)
    order = list(range(len(elig)))
    random.Random(B.SEED).shuffle(order)
    N = B.N_POOL
    sft_f = [json.loads(l) for l in open(OUT / "sft_train.jsonl")]
    dis_f = [json.loads(l) for l in open(OUT / "opd_disjoint_prompts.jsonl")]
    for name, got, want in (("sft", [elig[i] for i in order[:N]], sft_f),
                            ("disjoint", [elig[i] for i in order[N:2 * N]], dis_f)):
        assert len(got) == len(want) and all(strip_id(a) == strip_id(b) for a, b in zip(got, want)), \
            f"re-rendered {name} pool differs from the committed file"
    rest = [elig[i] for i in order[2 * N:]]
    for k, r in enumerate(rest):
        r["metadata"]["id"] = f"t2_rest_{k:04d}"
    val_f = [json.loads(l) for l in open(OUT / "val.jsonl")]
    el = lambda r: S.lim2_eligible(r["metadata"]["gold_sql"])
    part_a = [r for r in dis_f if el(r)]
    part_b = [r for r in rest if el(r)]
    pool = part_a + part_b
    gold_sft = {(r["metadata"]["db_id"], r["metadata"]["gold_sql"]) for r in sft_f}
    q_sft = {(r["metadata"]["db_id"], r["metadata"]["question"].strip().lower()) for r in sft_f}
    key = lambda r: (r["metadata"]["db_id"], r["metadata"]["gold_sql"])
    qk = lambda r: (r["metadata"]["db_id"], r["metadata"]["question"].strip().lower())
    db_val = {r["metadata"]["db_id"] for r in val_f}
    rep = {"n_train_spider_eligible": len(elig), "n_rest": len(rest), "n_part_a_disjoint_lim2": len(part_a),
           "n_part_b_rest_lim2": len(part_b), "n_pool": len(pool), "n_sft_pool_lim2": sum(map(el, sft_f)),
           "n_val_lim2": sum(map(el, val_f)),
           "gold_query_also_in_sft_pool": {"part_a": sum(key(r) in gold_sft for r in part_a),
                                           "part_b": sum(key(r) in gold_sft for r in part_b)},
           "question_also_in_sft_pool": {"part_a": sum(qk(r) in q_sft for r in part_a),
                                         "part_b": sum(qk(r) in q_sft for r in part_b)},
           "databases_shared_with_val": len({r["metadata"]["db_id"] for r in pool} & db_val),
           "ids_unique": len({r["metadata"]["id"] for r in pool}) == len(pool),
           "steps_per_epoch_at_batch_128_drop_last": len(pool) // 128}
    return pool, rep

def parquet(rows):
    import pandas as pd
    recs = []
    for k, r in enumerate(rows):
        m = r["metadata"]
        recs.append({"id": m["id"], "data_source": "t2_spider", "prompt": r["messages"][:2], "ability": "text2sql",
                     "reward_model": {"ground_truth": m["gold_sql"], "style": "rule"},
                     "extra_info": {"index": k, "split": "train", "id": m["id"], "db_id": m["db_id"],
                                    "db_path": m["db_path"]}})
    return pd.DataFrame(recs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    pool, rep = build()
    print(json.dumps(rep, indent=1))
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool)
    if a.check:
        assert JSONL.read_text() == text, "written pool differs from the rebuild"
        print("check: pool file equals the rebuild")
        return
    for p in (JSONL, PQ, MAN):
        assert not p.exists(), f"{p} exists; never overwritten"
    JSONL.write_text(text)
    PQ.mkdir()
    parquet(pool).to_parquet(PQ / "train.parquet", index=False)
    shutil.copyfile(OUT / "parquet_disjoint/val.parquet", PQ / "val.parquet")
    files = ["opd_lim2_enriched_prompts.jsonl", "parquet_lim2x/train.parquet", "parquet_lim2x/val.parquet"]
    man = {"what": "lim2-enriched OPD prompt pool for block SQLLIMX ()",
           "builder": "scripts/data_build_sql_limit_pool.py",
           "eligibility": "score_sql_limit_base.lim2_eligible on metadata.gold_sql (SQLLIM lock 2.2)",
           "source_zip_sha256": B.ZIP_SHA256, "seed_pools": B.SEED, **rep,
           "sha256": {f: sha(OUT / f) for f in files},
           "inputs_sha256": {f: sha(OUT / f) for f in ("opd_disjoint_prompts.jsonl", "sft_train.jsonl", "val.jsonl",
                                                       "parquet_disjoint/val.parquet")}}
    MAN.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["sha256"], indent=1))

if __name__ == "__main__":
    main()
