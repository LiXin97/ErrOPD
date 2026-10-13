#!/usr/bin/env python3
"""Build Spider 1.0 text-to-SQL SFT pool, OPD prompt pool and dev set."""
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_sql

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/raw/spider"
SRC = RAW / "spider_data"
OUT = REPO / "data/erropd_t2"
TOKENIZER = "/path/to/models/Qwen3-1.7B"
SEED = 20260930
N_POOL = 1795
N_EVAL = 500
MAX_PROMPT_TOKENS = 1800
MAX_GOLD_TOKENS = 400
ZIP_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    import pandas as pd
    from transformers import AutoTokenizer

    assert sha256(RAW / "spider_data.zip") == ZIP_SHA256, "spider_data.zip hash mismatch"
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    train = json.load(open(SRC / "train_spider.json"))
    dev = json.load(open(SRC / "dev.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    dbs = sorted({x["db_id"] for x in train} | {x["db_id"] for x in dev})
    for d in dbs:
        dst = OUT / "db" / d / f"{d}.sqlite"
        src = SRC / "database" / d / f"{d}.sqlite"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
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
        ok = (st == "ok" and not trunc and n_prompt <= MAX_PROMPT_TOKENS and n_gold <= MAX_GOLD_TOKENS)
        return {"messages": msgs, "metadata": meta}, ok, (st if st != "ok" else ("rows" if trunc else
                                                          ("prompt_len" if n_prompt > MAX_PROMPT_TOKENS else
                                                           ("gold_len" if n_gold > MAX_GOLD_TOKENS else "ok"))))

    stats = {}
    elig_train, elig_dev = [], []
    for split, data, elig in (("train_spider", train, elig_train), ("dev", dev, elig_dev)):
        cnt = {}
        for i, ex in enumerate(data):
            row, ok, why = render(ex, split, i)
            cnt[why] = cnt.get(why, 0) + 1
            if ok:
                elig.append(row)
        stats[split] = {"n": len(data), "eligible": len(elig), "reasons": cnt,
                        "max_prompt_tokens_eligible": max(r["metadata"]["prompt_tokens"] for r in elig),
                        "max_gold_ticks_eligible": max(r["metadata"]["gold_ticks"] for r in elig)}
        print(split, stats[split], flush=True)

    rng = random.Random(SEED)
    order = list(range(len(elig_train)))
    rng.shuffle(order)
    sft = [elig_train[i] for i in order[:N_POOL]]
    disj = [elig_train[i] for i in order[N_POOL:2 * N_POOL]]
    assert len(sft) == N_POOL and len(disj) == N_POOL
    rng2 = random.Random(SEED + 1)
    dorder = list(range(len(elig_dev)))
    rng2.shuffle(dorder)
    ev = sorted([elig_dev[i] for i in dorder[:N_EVAL]], key=lambda r: r["metadata"]["spider_idx"])
    for name, rows in (("sft", sft), ("disj", disj), ("val", ev)):
        for k, r in enumerate(rows):
            r["metadata"]["id"] = f"t2_{name}_{k:04d}"

    def wjsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    wjsonl(OUT / "sft_train.jsonl", sft)
    wjsonl(OUT / "val.jsonl", ev)
    wjsonl(OUT / "opd_disjoint_prompts.jsonl", disj)

    def parquet(rows, split):
        recs = []
        for k, r in enumerate(rows):
            m = r["metadata"]
            recs.append({"id": m["id"], "data_source": "t2_spider", "prompt": r["messages"][:2], "ability": "text2sql",
                         "reward_model": {"ground_truth": m["gold_sql"], "style": "rule"},
                         "extra_info": {"index": k, "split": split, "id": m["id"], "db_id": m["db_id"],
                                        "db_path": m["db_path"]}})
        return pd.DataFrame(recs)

    for pool, rows in (("sftpool", sft), ("disjoint", disj)):
        d = OUT / f"parquet_{pool}"
        d.mkdir(exist_ok=True)
        parquet(rows, "train").to_parquet(d / "train.parquet", index=False)
        parquet(ev, "val").to_parquet(d / "val.parquet", index=False)

    info = {
        "t2_spider_sft": {"file_name": "sft_train.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"},
                          "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                                   "assistant_tag": "assistant", "system_tag": "system"}},
        "t2_spider_val": {"file_name": "val.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"},
                          "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                                   "assistant_tag": "assistant", "system_tag": "system"}},
    }
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=1) + "\n")

    files = ["sft_train.jsonl", "val.jsonl", "opd_disjoint_prompts.jsonl", "dataset_info.json",
             "parquet_sftpool/train.parquet", "parquet_sftpool/val.parquet",
             "parquet_disjoint/train.parquet", "parquet_disjoint/val.parquet"]
    import sqlite3
    manifest = {
        "task": "t2_spider (text-to-SQL, Spider 1.0)",
        "source": "Spider 1.0 official release (yale-lily.github.io/spider; Google Drive file spider_data.zip)",
        "license": "CC BY-SA 4.0 (Spider README / website); train_others (not used) has additional sources",
        "citation": "Yu et al., EMNLP 2018",
        "source_zip_sha256": ZIP_SHA256,
        "splits_used": {"train_spider.json": "SFT pool + disjoint OPD pool", "dev.json": "eval subset"},
        "seed_pools": SEED, "seed_eval": SEED + 1, "n_pool": N_POOL, "n_eval": N_EVAL,
        "eligibility": {"gold_executes": True, "row_cap": common_sql.ROW_CAP, "max_prompt_tokens": MAX_PROMPT_TOKENS,
                        "max_gold_tokens": MAX_GOLD_TOKENS,
                        "exec_budget_vm_instructions": common_sql.PROGRESS_N * common_sql.OP_BUDGET_TICKS},
        "tokenizer": "Qwen3-1.7B (shared by Qwen3-4B)", "template": "stock Qwen3 chat template, add_generation_prompt, "
                                                                 "no enable_thinking kwarg (no think block)",
        "system_prompt": common_sql.SYSTEM_PROMPT,
        "sqlite_version_build": sqlite3.sqlite_version,
        "stats": stats,
        "n_databases": len(dbs), "databases_dir": "data/erropd_t2/db (not committed; rebuilt from the zip)",
        "files_sha256": {f: sha256(OUT / f) for f in files},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("stats", "files_sha256")}, indent=1))

if __name__ == "__main__":
    main()
