#!/usr/bin/env python3
"""Build the CoNLL-2003 JSON entity-extraction task (download CoNLL-2003 yourself)."""
from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from common_ner import (SYSTEM_PROMPT, chunk_text_and_mentions, classify, gold_response, messages_for,
                       read_conll_docs)

RAW = REPO / "data/raw/conll2003"
OUT = REPO / "data/erropd_t3"
TOKENIZER = "/path/to/models/Qwen3-1.7B"
URL = "https://data.deepai.org/conll2003.zip"
RAW_SHA = {
    "conll2003.zip": "96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3",
    "train.txt": "213953b5522fb7924802d883e50f027b2c6ffb5d4d0e63368ed82909f9b60423",
    "valid.txt": "c02eb19809eafb83998326a0481942eb37d96f31826ae387cfe8773699969f5b",
    "test.txt": "dc4ac7f3d6d01a6bec59261814561da3b287b6b493b97d1f69c580b52ec1831b",
}
W_MAX, G_MAX = 180, 320
N_VERL_VAL = 128
DATA_SOURCE = "conll2003_ner_json"

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    for f, h in RAW_SHA.items():
        got = sha(RAW / f)
        assert got == h, f"sha256 mismatch for {f}: {got}"
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def gold_tokens(ms):
        return len(tok.encode(gold_response(ms), add_special_tokens=False))

    def chunks(doc):
        out, cur, cw = [], [], 0
        for s in doc:
            if cur and (cw + len(s) > W_MAX or gold_tokens(chunk_text_and_mentions(cur + [s])[1]) > G_MAX):
                out.append(cur); cur, cw = [], 0
            cur.append(s); cw += len(s)
        if cur:
            out.append(cur)
        return out

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lf").mkdir(exist_ok=True)
    (OUT / "parquet_verl").mkdir(exist_ok=True)
    stats, rows_by = {}, {}
    for split in ("train", "valid", "test"):
        rows = []
        for di, doc in enumerate(read_conll_docs(RAW / f"{split}.txt")):
            for ci, c in enumerate(chunks(doc)):
                text, ms = chunk_text_and_mentions(c)
                resp = gold_response(ms)
                cls, parsed, det = classify(resp, "stop", text)
                assert cls == "valid", (split, di, ci, cls, det)
                rows.append({
                    "id": f"t3_{split}_{len(rows):05d}", "split": split, "doc": di, "chunk": ci,
                    "messages": messages_for(text) + [{"role": "assistant", "content": resp}],
                    "text": text, "gold": ms, "n_mentions": len(ms), "gold_tokens": gold_tokens(ms),
                    "n_words": sum(len(s) for s in c),
                })
        rows_by[split] = rows
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(OUT / "lf" / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        gt = [r["gold_tokens"] for r in rows]
        nm = [r["n_mentions"] for r in rows]
        pt = [len(tok.apply_chat_template(r["messages"][:2], add_generation_prompt=True, tokenize=True)) for r in rows]
        stats[split] = {"chunks": len(rows), "docs": len({r["doc"] for r in rows}), "mentions": sum(nm),
                        "mentions_mean": round(st.mean(nm), 2), "mentions_max": max(nm),
                        "empty_gold": sum(1 for x in nm if x == 0),
                        "gold_tokens_mean": round(st.mean(gt), 1), "gold_tokens_max": max(gt),
                        "prompt_tokens_mean": round(st.mean(pt), 1), "prompt_tokens_max": max(pt),
                        "types": {t: sum(1 for r in rows for m in r["gold"] if m["type"] == t)
                                  for t in ("PER", "ORG", "LOC", "MISC")}}
        print(split, json.dumps(stats[split]))

    info = {}
    for split in ("train", "valid"):
        info[f"t3_conll_{split}"] = {
            "file_name": f"{split}.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                     "assistant_tag": "assistant", "system_tag": "system"}}
    json.dump(info, open(OUT / "lf" / "dataset_info.json", "w"), indent=1)

    def to_verl(rows, split):
        recs = []
        for i, r in enumerate(rows):
            recs.append({
                "id": r["id"], "data_source": DATA_SOURCE, "prompt": r["messages"][:2], "ability": "extraction",
                "reward_model": {"ground_truth": r["messages"][2]["content"], "style": "rule"},
                "extra_info": {"index": i, "split": split, "doc": r["doc"], "n_mentions": r["n_mentions"]},
            })
        return pa.Table.from_pylist(recs)

    pq.write_table(to_verl(rows_by["train"], "train"), OUT / "parquet_verl" / "train.parquet")
    pq.write_table(to_verl(rows_by["valid"][:N_VERL_VAL], "val"), OUT / "parquet_verl" / "val.parquet")

    built = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "task": "t3_conll2003_ner_json",
        "source": {"dataset": "CoNLL-2003 English NER (Tjong Kim Sang and De Meulder, 2003)",
                   "url": URL, "via": "Hugging Face dataset card eriktks/conll2003 (loader script _URL)",
                   "format": "IOB2, four columns, -DOCSTART- document boundaries",
                   "raw_sha256": RAW_SHA},
        "license": ("Annotations: free for research (CoNLL-2003 shared task, University of Antwerp). Text: Reuters "
                    "Corpus RCV1, copyright Reuters, distributed by NIST under a research-use agreement. Used for "
                    "research only; no raw text is redistributed in any shipped artifact (supplementary material "
                    "carries code, prompt templates and metrics only)."),
        "chunking": {"rule": "greedy within-document packing of consecutive sentences", "max_words": W_MAX,
                     "max_gold_tokens": G_MAX, "tokenizer": "Qwen3 (1.7B/4B share it)",
                     "text": "tokens joined by single spaces, sentences by newlines"},
        "output_contract": "compact JSON list of {text, type}, every mention in order of appearance, [] if none",
        "system_prompt": SYSTEM_PROMPT,
        "splits": {"train": "CoNLL train chunks (SFT and OPD prompts)",
                   "valid": "CoNLL valid chunks (SFT eval loss; first 128 = verl val_files, unused for any number)",
                   "test": "CoNLL test chunks = the T3 eval set"},
        "data_source": DATA_SOURCE,
        "stats": stats,
        "files_sha256": {str(p.relative_to(OUT)): sha(p) for p in built},
    }
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
