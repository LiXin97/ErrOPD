#!/usr/bin/env python3
"""Label each response: valid, early close, duplicated ID, other failures."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_listwise_fashion import parse_score_list

RES = REPO / "results"
STORE_CAP = 1000
LABELS = ["valid", "k1_trunc_realids", "short_list", "duplicate_id", "hallucinated_id",
          "extra_items", "malformed_json", "extra_text", "unterminated", "other", "infeasible_input"]
FAIL_LABELS = LABELS[1:-1]
B1_VERSION = "v4"
_B1_V2 = "v2"
DEC = json.JSONDecoder()
WS = " \t\n\r"

TOKENIZER_FILES = {
    "qwen3": REPO / "outputs/sft_qwen3_1.7b_pl_k8/tokenizer.json",
    "llama3.2": REPO / "outputs/sft_llama3_1b_pl_k8/tokenizer.json",
}
_TOK = {}

def get_tokenizer(student: str):
    fam = "llama3.2" if student.startswith("llama") else "qwen3"
    if fam in _TOK:
        return _TOK[fam]
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.path.append("/path/to/envs/opd-vllm/lib/python3.10/site-packages")
        try:
            from tokenizers import Tokenizer
        except Exception:
            Tokenizer = None
    tok = None
    if Tokenizer is not None and TOKENIZER_FILES[fam].exists():
        tok = Tokenizer.from_file(str(TOKENIZER_FILES[fam]))
    _TOK[fam] = tok
    return tok

def id_value_pos(s: str, i: int, j: int, id_key: str) -> int:
    m = re.compile(r'"%s"\s*:\s*"?' % re.escape(id_key)).search(s, i, j)
    return m.end() if m else i

def classify(pred_text: str, ids: list, K: int, id_key: str, strict_ok: bool) -> dict:
    pass
    s = (pred_text or "").strip()
    exp = [str(x) for x in ids]
    exps = set(exp)
    out = {"label": None, "sub": "", "fe_item": None, "fe_char": None, "stream_valid": 0,
           "closed": 0, "missing_slots": "", "in_order": ""}
    if len(exps) != len(exp):
        out.update(label="infeasible_input", sub="repeated_input_id")
        return out
    start = s.find("[")
    if start < 0:
        out.update(label="other", sub="no_list", fe_item=0, fe_char=0)
        return out
    out["prefix_nonws"] = int(bool(s[:start].strip()))
    i, seen, n = start + 1, [], len(s)
    while True:
        while i < n and s[i] in WS:
            i += 1
        if i >= n:
            out.update(label="unterminated", fe_item=len(seen), fe_char=n)
            break
        if s[i] == "]" and not seen:
            out.update(label="short_list", sub="empty", fe_item=0, fe_char=i, closed=1)
            break
        try:
            obj, j = DEC.raw_decode(s, i)
        except json.JSONDecodeError as ex:
            if "}" not in s[i:]:
                out.update(label="unterminated", sub="mid_item", fe_item=len(seen), fe_char=n)
            else:
                out.update(label="malformed_json", sub="decode", fe_item=len(seen), fe_char=ex.pos)
            break
        if not isinstance(obj, dict) or id_key not in obj or "score" not in obj:
            out.update(label="malformed_json", sub="schema", fe_item=len(seen), fe_char=i)
            break
        try:
            float(obj["score"])
        except Exception:
            out.update(label="malformed_json", sub="score_type", fe_item=len(seen), fe_char=i)
            break
        rid = str(obj[id_key])
        if rid not in exps:
            out.update(label="hallucinated_id" if len(seen) < K else "extra_items",
                       fe_item=len(seen), fe_char=id_value_pos(s, i, j, id_key))
            break
        if rid in seen:
            out.update(label="duplicate_id" if len(seen) < K else "extra_items",
                       fe_item=len(seen), fe_char=id_value_pos(s, i, j, id_key))
            break
        seen.append(rid)
        i = j
        while i < n and s[i] in WS:
            i += 1
        if i >= n:
            out.update(label="unterminated", sub="missing_close" if len(seen) == K else "", fe_item=len(seen), fe_char=n)
            break
        c = s[i]
        if c == ",":
            if len(seen) == K:
                out.update(label="extra_items", fe_item=K, fe_char=i)
                break
            i += 1
            continue
        if c == "]":
            out["closed"] = 1
            if len(seen) == K:
                out["stream_valid"] = 1
                if strict_ok:
                    out["label"] = "valid"
                else:
                    k = i + 1
                    while k < n and s[k] in WS:
                        k += 1
                    out.update(label="extra_text", fe_item=K, fe_char=k if k < n else start)
            elif len(seen) == K - 1:
                out.update(label="k1_trunc_realids", fe_item=K - 1, fe_char=i)
            else:
                out.update(label="short_list", fe_item=len(seen), fe_char=i)
            break
        out.update(label="malformed_json", sub="separator", fe_item=len(seen), fe_char=i)
        break
    if out["label"] in ("k1_trunc_realids", "short_list"):
        miss = [k for k, e in enumerate(exp) if e not in seen]
        out["missing_slots"] = " ".join(map(str, miss))
        out["in_order"] = int(seen == [e for e in exp if e in seen])
    return out

def all_ids(pred_text: str, id_key: str) -> list:
    return re.findall(r'"%s"\s*:\s*"([^"]*)"' % re.escape(id_key), pred_text or "")

def load_ledger_rows():
    rows = []
    for name in ("legacy_ledger.csv", "ledger.csv"):
        f = RES / name
        if f.exists():
            rows += list(csv.DictReader(f.open()))
    return rows

def pred_file_for(eval_path: str):
    d = REPO / eval_path
    for fn in ("predictions.jsonl",):
        if (d / fn).exists():
            return d / fn
    return None

META_COLS = ("id", "experiment", "task", "student", "teacher", "lambda", "seed", "step",
             "budget_steps", "variant", "clip_c", "decoding", "parse", "ndcg1_parsed", "useful",
             "model_key", "eval_path", "rope_misread")

def process_eval(row: dict, tok, pf: Path):
    pass
    recs = [json.loads(l) for l in pf.open() if l.strip()]
    if not recs:
        return None
    r0 = recs[0]
    if "review_ids" in r0:
        id_key, ids_field = "review_id", "review_ids"
    elif "product_ids" in r0:
        id_key, ids_field = "product_id", "product_ids"
    else:
        return None
    lab = Counter()
    samples = []
    fe_items, fe_tokens, fail_chars, fail_tokens, k1_last_inorder, n_k1 = [], [], [], [], 0, 0
    n_trunc_store, n_mismatch, n_hit = 0, 0, 0
    for idx, r in enumerate(recs):
        text = r.get("pred_text") or ""
        ids = r.get(ids_field) or []
        K = int(r.get("K", len(ids)))
        if id_key == "review_id":
            strict_ok = parse_score_list(text, expected_K=K, review_ids=ids) is not None
        else:
            strict_ok = _strict_generic(text, K, ids, id_key)
        c = classify(text, ids, K, id_key, strict_ok)
        if strict_ok != (c["label"] == "valid"):
            n_mismatch += 1
        lab[c["label"]] += 1
        s = text.strip()
        n_tok, fe_tok = "", ""
        if tok is not None:
            enc = tok.encode(s, add_special_tokens=False)
            n_tok = len(enc.ids)
            if c["fe_char"] is not None and c["label"] not in ("valid", "infeasible_input"):
                fe_tok = n_tok
                for t_i, (a, b) in enumerate(enc.offsets):
                    if b > c["fe_char"]:
                        fe_tok = t_i
                        break
        emitted = all_ids(s, id_key)
        cnt = Counter(emitted)
        fr = r.get("finish_reason")
        stored_trunc = 0 if fr is not None else int(len(text) >= STORE_CAP)
        hit_cap = "" if fr is None else int(str(fr) == "length")
        n_hit += hit_cap == 1
        n_trunc_store += stored_trunc
        if c["label"] not in ("valid", "infeasible_input"):
            fe_items.append(c["fe_item"])
            fail_chars.append(len(s))
            if fe_tok != "":
                fe_tokens.append(fe_tok)
            if n_tok != "":
                fail_tokens.append(n_tok)
        if c["label"] == "k1_trunc_realids":
            n_k1 += 1
            if c["missing_slots"] == str(K - 1) and c["in_order"] == 1:
                k1_last_inorder += 1
        samples.append({
            "row": idx, "label": c["label"], "sub": c["sub"],
            "fe_item": "" if c["fe_item"] is None or c["label"] == "valid" else c["fe_item"],
            "fe_char": "" if c["fe_char"] is None or c["label"] == "valid" else c["fe_char"],
            "fe_token": fe_tok, "n_chars": len(s), "n_tokens": n_tok,
            "stored_truncated": stored_trunc, "hit_cap": hit_cap,
            "n_gen_tokens": r.get("n_gen_tokens", r.get("n_out_tokens", "")), "closed": c["closed"],
            "n_ids_emitted": len(emitted), "any_dup": int(any(v > 1 for v in cnt.values())),
            "any_halluc": int(any(e not in set(map(str, ids)) for e in emitted)),
            "missing_slots": c["missing_slots"], "in_order": c["in_order"],
            "strict_ok": int(strict_ok),
        })
    n = len(recs)
    n_inf = lab["infeasible_input"]
    n_fail = n - lab["valid"] - n_inf
    out = {}
    out.update({"n_total": n, "n_feasible": n - n_inf, "n_fail": n_fail,
                "parse_feasible": round(lab["valid"] / (n - n_inf), 6) if n - n_inf else "",
                "mismatch_vs_strict": n_mismatch})
    for L in LABELS:
        out["n_" + L] = lab[L]
    for L in FAIL_LABELS:
        out["share_" + L] = round(lab[L] / n_fail, 4) if n_fail else ""
    out["k1_last_slot_inorder_share"] = round(k1_last_inorder / n_k1, 4) if n_k1 else ""
    out["fail_fe_item_median"] = st.median(fe_items) if fe_items else ""
    out["fail_fe_token_median"] = st.median(fe_tokens) if fe_tokens else ""
    out["fail_n_chars_median"] = st.median(fail_chars) if fail_chars else ""
    out["fail_n_tokens_median"] = st.median(fail_tokens) if fail_tokens else ""
    out["n_stored_truncated"] = n_trunc_store
    out["n_hit_token_cap"] = n_hit if any("finish_reason" in r for r in recs[:1]) else ""
    return out, samples

def _strict_generic(text, K, ids, id_key):
    pass
    t = (text or "").strip()
    try:
        m = re.search(r"\[.*\]", t, re.S)
        if not m:
            return False
        arr = json.loads(m.group(0))
        if not isinstance(arr, list) or len(arr) != K:
            return False
        exp = [str(x) for x in ids]
        seen = {}
        for it in arr:
            if not isinstance(it, dict) or id_key not in it or "score" not in it:
                return False
            rid = str(it[id_key])
            if rid not in exp or rid in seen:
                return False
            seen[rid] = float(it["score"])
        return set(seen) == set(exp)
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="regex on eval id")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    rows = load_ledger_rows()
    if args.only:
        rows = [r for r in rows if re.search(args.only, r["id"])]
    cache_dir = RES / ".b1_cache"
    cache_dir.mkdir(exist_ok=True)
    sample_f = RES / "B1_failure_samples.csv.gz"
    fields = ["eval_id", "row", "label", "sub", "fe_item", "fe_char", "fe_token", "n_chars",
              "n_tokens", "stored_truncated", "hit_cap", "n_gen_tokens", "closed", "n_ids_emitted", "any_dup", "any_halluc",
              "missing_slots", "in_order", "strict_ok"]
    agg, n_mis, n_new = [], 0, 0
    with gzip.open(sample_f, "wt", newline="") as gz:
        w = csv.DictWriter(gz, fieldnames=fields)
        w.writeheader()
        for r in rows:
            pf = pred_file_for(r["eval_path"])
            if pf is None or r["variant"].endswith("_posthoc_perm"):
                continue
            st_ = pf.stat()
            sig = [str(pf), st_.st_mtime, st_.st_size, B1_VERSION]
            cf = cache_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", r["eval_path"]) + ".json.gz")
            res = None
            if cf.exists() and not args.no_cache:
                try:
                    cached = json.loads(gzip.open(cf, "rt").read())
                    if cached["sig"] == sig:
                        res = (cached["stats"], cached["samples"])
                except Exception:
                    res = None
            if res is None:
                res = process_eval(r, get_tokenizer(r.get("student", "")), pf)
                n_new += 1
                if res is None:
                    continue
                with gzip.open(cf, "wt") as fh:
                    fh.write(json.dumps({"sig": sig, "stats": res[0], "samples": res[1]}))
            stats, samples = res
            out = {k: r[k] for k in META_COLS}
            out.update(stats)
            agg.append(out)
            n_mis += int(stats["mismatch_vs_strict"])
            for smp in samples:
                w.writerow({"eval_id": r["id"], **smp})
    out_f = RES / "B1_failure_anatomy.csv"
    with out_f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    n_rows = sum(a["n_total"] for a in agg)
    print(f"{out_f.relative_to(REPO)}: {len(agg)} evals ({n_new} re-classified), {n_rows} outputs; "
          f"strict-vs-stream mismatches: {n_mis}")

if __name__ == "__main__":
    main()
