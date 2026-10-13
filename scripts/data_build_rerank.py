#!/usr/bin/env python3
"""Build the passage reranking task from MS MARCO / TREC-DL (download the data yourself)."""
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common_perm import SYSTEM, format_perm, parse_strict, render_user

RAW = ROOT / "data" / "raw" / "t1"
OUT = ROOT / "data" / "erropd_t1"
K = 20
N_TRAIN, N_VAL = 1800, 400
SPLIT_SEED = 20260930
MAX_WORDS = 300
TOKENIZER = "/path/to/models/Qwen3-1.7B"

SOURCES = {
    "rank_zephyr_training_data.jsonl.gz": {
        "url": "https://huggingface.co/datasets/castorini/rank_zephyr_training_data/resolve/"
               "0cda1b0661f4afe41345746f31fe772cb81121c7/rank_zephyr_training_data.jsonl.gz",
        "hf_repo": "castorini/rank_zephyr_training_data", "revision": "0cda1b0661f4afe41345746f31fe772cb81121c7",
        "license": "No license file or dataset card on the HF repo (checked 2026-09-30). Released by the RankZephyr "
                   "authors with the rank_llm toolkit (Apache-2.0). Passages are MS MARCO (Microsoft MS MARCO terms: "
                   "non-commercial research use); target orderings were generated with GPT-4 (OpenAI terms). Used "
                   "here for non-commercial research only.",
    },
    "bm25_dl19_top100.jsonl": {
        "url": "https://huggingface.co/datasets/castorini/rank_llm_data/resolve/"
               "39db0a25a552dd2a12012df22029686bf7fdc050/retrieve_results/BM25/retrieve_results_dl19_top100.jsonl",
        "hf_repo": "castorini/rank_llm_data", "revision": "39db0a25a552dd2a12012df22029686bf7fdc050",
        "license": "No license file or dataset card on the HF repo (checked 2026-09-30); rank_llm toolkit is Apache-2.0; "
                   "passages are MS MARCO (non-commercial research terms).",
    },
    "bm25_dl20_top100.jsonl": {
        "url": "https://huggingface.co/datasets/castorini/rank_llm_data/resolve/"
               "39db0a25a552dd2a12012df22029686bf7fdc050/retrieve_results/BM25/retrieve_results_dl20_top100.jsonl",
        "hf_repo": "castorini/rank_llm_data", "revision": "39db0a25a552dd2a12012df22029686bf7fdc050",
        "license": "as bm25_dl19_top100.jsonl",
    },
    "2019qrels-pass.txt": {"url": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
                           "license": "TREC-DL public relevance judgments (NIST), research use"},
    "2020qrels-pass.txt": {"url": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
                           "license": "TREC-DL public relevance judgments (NIST), research use"},
}

HUMAN_RE = re.compile(
    r"^I will provide you with 20 passages, each indicated by a numerical identifier \[\]\. Rank the passages based "
    r"on their relevance to the search query: (?P<q>.*)\.\n\n(?P<body>(?:\[\d+\] [^\n]*\n){20})Search Query: (?P<q2>.*)"
    r"\.\nRank the 20 passages above", re.S)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def clean_passage(text: str) -> str:
    pass
    t = " ".join(text.strip().split()[:MAX_WORDS])
    return re.sub(r"\[(\d+)\]", r"(\1)", t)

def write_jsonl(rows, path: Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parquet").mkdir(exist_ok=True)
    checks = {}

    rz = []
    with gzip.open(RAW / "rank_zephyr_training_data.jsonl.gz", "rt") as f:
        for line in f:
            if "random_ct0\"" in line[:120]:
                r = json.loads(line)
                if re.fullmatch(r"identity_\d+_da0open_ai_ada2random_ct0", r["id"]):
                    rz.append(r)
    parsed, n_bad_tmpl, n_bad_target, sys_msgs = [], 0, 0, Counter()
    stats = Counter()
    for r in rz:
        conv = r["conversations"]
        assert [c["from"] for c in conv] == ["system", "human", "gpt"], r["id"]
        sys_msgs[conv[0]["value"]] += 1
        h, tgt = conv[1]["value"], conv[2]["value"]
        m = HUMAN_RE.match(h)
        if not m or m.group("q") != m.group("q2"):
            n_bad_tmpl += 1
            continue
        lines = m.group("body").split("\n")[:-1]
        passages = []
        for i, l in enumerate(lines, 1):
            pre = f"[{i}] "
            assert l.startswith(pre), (r["id"], l[:20])
            passages.append(l[len(pre):])
        if render_user(m.group("q"), passages) != h:
            n_bad_tmpl += 1
            continue
        perm = parse_strict(tgt, K)
        if perm is None or tgt != format_perm(perm):
            n_bad_target += 1
            continue
        for p in passages:
            stats["passages"] += 1
            stats["with_[n]"] += bool(re.search(r"\[\d+\]", p))
            stats["with_(n)"] += bool(re.search(r"\(\d+\)", p))
            stats["double_space"] += "  " in p
            stats["over_300_words"] += len(p.split()) > MAX_WORDS
            stats["leading_or_trailing_ws"] += p != p.strip()
        qid = re.match(r"identity_(\d+)_", r["id"]).group(1)
        parsed.append({"qid": qid, "query": m.group("q"), "passages": passages, "perm": perm, "target": tgt,
                       "source_id": r["id"]})
    assert set(sys_msgs) == {SYSTEM}, sys_msgs
    checks["rankzephyr_window20_records"] = len(rz)
    checks["rankzephyr_template_mismatch"] = n_bad_tmpl
    checks["rankzephyr_invalid_target"] = n_bad_target
    checks["rankzephyr_passage_stats"] = dict(stats)
    print(f"[t1_data] RankZephyr window-20 records {len(rz)}, render mismatches {n_bad_tmpl}, invalid targets "
          f"{n_bad_target}, passage stats {dict(stats)}", flush=True)

    dl = {}
    for year in ("dl19", "dl20"):
        rows = [json.loads(l) for l in open(RAW / f"bm25_{year}_top100.jsonl")]
        qrel_file = RAW / ("2019qrels-pass.txt" if year == "dl19" else "2020qrels-pass.txt")
        judged = set(l.split()[0] for l in open(qrel_file) if l.strip())
        qs = []
        for r in rows:
            qid = str(r["query"]["qid"])
            cands = r["candidates"]
            scores = [c["score"] for c in cands]
            assert scores == sorted(scores, reverse=True), (year, qid)
            qs.append({"set": year, "qid": qid, "query": r["query"]["text"],
                       "docids": [str(c["docid"]) for c in cands],
                       "passages": [clean_passage(c["doc"]["contents"]) for c in cands]})
        assert all(q["qid"] in judged for q in qs), year
        checks[f"{year}_queries"] = len(qs)
        checks[f"{year}_candidates_per_query"] = dict(Counter(len(q["docids"]) for q in qs))
        dl[year] = qs
        (OUT / f"qrels_{year}.txt").write_bytes(qrel_file.read_bytes())
    dl_queries = {q["query"].strip().lower() for y in dl.values() for q in y}
    dl_qids = {q["qid"] for y in dl.values() for q in y}

    elig = [p for p in parsed if p["qid"] not in dl_qids and p["query"].strip().lower() not in dl_queries]
    checks["excluded_overlap_with_dl"] = len(parsed) - len(elig)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(elig)
    assert len(elig) >= N_TRAIN + N_VAL
    splits = {"train": elig[:N_TRAIN], "val": elig[N_TRAIN:N_TRAIN + N_VAL]}
    assert not ({p["qid"] for p in splits["train"]} & {p["qid"] for p in splits["val"]})

    def lf_row(p, split, i):
        return {"messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": render_user(p["query"], p["passages"])},
                             {"role": "assistant", "content": p["target"]}],
                "metadata": {"set": split, "row": i, "qid": p["qid"], "K": K, "gpt4_perm": p["perm"],
                             "source_id": p["source_id"], "dataset": "rankzephyr_gpt4_window20"}}

    lf = {s: [lf_row(p, s, i) for i, p in enumerate(ps)] for s, ps in splits.items()}
    for s, rows in lf.items():
        write_jsonl(rows, OUT / f"{s}.jsonl")

    win_rows, top_rows = [], []
    for year, qs in dl.items():
        for q in qs:
            top_rows.append(q)
            nwin = len(q["docids"]) // K
            for w in range(nwin):
                a, b = w * K, (w + 1) * K
                win_rows.append({
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": render_user(q["query"], q["passages"][a:b])}],
                    "metadata": {"set": year, "row": len(win_rows), "qid": q["qid"], "K": K, "window": w,
                                 "rank_start": a, "docids": q["docids"][a:b]}})
    write_jsonl(win_rows, OUT / "dl_windows.jsonl")
    write_jsonl(top_rows, OUT / "dl_top100.jsonl")
    checks["dl_windows"] = dict(Counter(r["metadata"]["set"] for r in win_rows))

    import pandas as pd
    for s, rows in lf.items():
        recs = []
        for i, r in enumerate(rows):
            sys_m, usr_m, asst_m = r["messages"]
            recs.append({"id": f"t1_{s}_{i:05d}", "data_source": "msmarco_trec_dl", "prompt": [sys_m, usr_m],
                         "ability": "ranking", "reward_model": {"ground_truth": asst_m["content"], "style": "rule"},
                         "extra_info": {"index": i, "split": s, "qid": r["metadata"]["qid"], "K": K}})
        pd.DataFrame(recs).to_parquet(OUT / "parquet" / f"{s}.parquet", index=False)

    tmpl = {"formatting": "sharegpt", "columns": {"messages": "messages"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                     "system_tag": "system"}}
    info = {f"t1_{s}": dict(tmpl, file_name=f"{s}.jsonl") for s in ("train", "val")}
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    lens = {}
    for name, rows in (("train", lf["train"]), ("val", lf["val"]), ("dl_windows", win_rows)):
        L = []
        for r in rows:
            msgs = [m for m in r["messages"] if m["role"] != "assistant"]
            L.append(len(tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                             add_special_tokens=False).input_ids))
        L.sort()
        lens[name] = {"min": L[0], "p50": L[len(L) // 2], "p99": L[int(0.99 * (len(L) - 1))], "max": L[-1]}
    resp = sorted(len(tok(r["messages"][2]["content"], add_special_tokens=False).input_ids) for r in lf["train"])
    lens["train_response"] = {"min": resp[0], "max": resp[-1]}
    checks["prompt_tokens"] = lens
    print(f"[t1_data] token lengths {json.dumps(lens)}", flush=True)

    derived = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = {
        "task": "T1 listwise passage reranking, RankGPT/RankZephyr permutation format, window K=20",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "scripts/data_build_rerank.py", "prompt_renderer": "scripts/common_perm.py:render_user",
        "split_seed": SPLIT_SEED, "n_train": N_TRAIN, "n_val": N_VAL, "K": K, "max_words_per_passage": MAX_WORDS,
        "sources": {k: dict(v, sha256=sha256(RAW / k), path=f"data/raw/t1/{k}") for k, v in SOURCES.items()},
        "derived": {str(p.relative_to(ROOT)): sha256(p) for p in derived},
        "counts": {"train": len(lf["train"]), "val": len(lf["val"]), "dl_windows": len(win_rows),
                   "dl_queries": len(top_rows)},
        "checks": checks,
        "notes": ["train/val: disjoint MS MARCO training queries; supervision = GPT-4 permutation of the 20 candidates "
                  "(input order already randomized by the source).",
                  "dl_windows: fixed prompts for validity (5 per query, BM25 order inside each window, identical "
                  "for every model); window 0 (BM25 ranks 1-20) is also the single-window NDCG@10 protocol.",
                  "DL passages processed with the rank_llm convention (strip, whitespace collapse, 300-word cap, "
                  "'[n]' -> '(n)'); see checks.rankzephyr_passage_stats for what the training passages show."],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[t1_data] done in {time.time() - t0:.0f}s: {manifest['counts']}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
