#!/usr/bin/env python3
"""Probe at the SQL LIMIT decision."""
import argparse
import datetime
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import common_nonranking as nr

KIND = "lim2"
DATA = "data/erropd_nr/t2/lim2_f20_x5.jsonl"
STUDENT = "outputs/T2/sft_qwen3_1.7b"
TEACHER_CLEAN = "outputs/T2/sft_qwen3_4b"
TEACHER_LIM2 = "outputs/SQLLIM/sft_t2_4b_lim2_f20_x5"
ROLLOUTS = "outputs/T2/stageA/gen_base/rollouts_train.jsonl"
SPLITS = ("disjoint", "val")
PRIMARY = "disjoint"
SEEDS = (6001, 6002, 6003, 6004, 6005)
GRID_ARMS = ("opd_lam1.00", "fkl_lam1.00")
DIGITS = tuple("0123456789")
IM_END = 151645
UTC = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def grid_models():
    out = [("student", STUDENT), ("teacher_lim2", TEACHER_LIM2)]
    for arm in GRID_ARMS:
        for s in SEEDS:
            rid = f"SQLLIM_t2_lim2_{arm}_s{s}"
            out.append((f"{rid}_step42", f"outputs/SQLLIM/{rid}/ckpt/global_step_42/hf"))
    return out

def digit_ids(tok, lead):
    pass
    out = {}
    for d in DIGITS:
        ids = tok(lead + d, add_special_tokens=False).input_ids
        out[d] = ids[0] if len(ids) == 1 else None
    return out

def states(tok):
    pass
    recs = [json.loads(l) for l in open(REPO / ROLLOUTS)]
    out, counts = [], {s: Counter() for s in SPLITS}
    for r in recs:
        sp = r["split"]
        if sp not in SPLITS:
            continue
        c = counts[sp]
        c["samples"] += 1
        if nr.t2_eligible(KIND, {"gold_sql": r["gold_sql"]}) is None:
            continue
        c["eligible_samples"] += 1
        text = r["text"]
        if nr.t2_classify(KIND, text, {"gold_sql": r["gold_sql"]})["hit"]:
            c["own_lim2_hit"] += 1
        resp = list(r["resp_ids"])
        if IM_END in resp:
            resp = resp[: resp.index(IM_END)]
        toks = nr._t2_tokens(text)
        lv = nr.t2_limit_values(toks)
        if not lv:
            c["no_limit_in_own"] += 1
            continue
        one = [i for i, v in lv if toks[i].kind == "num" and v == "1"]
        if not one:
            c["own_limit_other_value"] += 1
            c["own_limit_values:" + ",".join(v for _, v in lv)[:40]] += 1
            continue
        i = one[0]
        start = toks[i].start
        cum = [len(tok.decode(resp[:j])) for j in range(len(resp) + 1)]
        if tok.decode(resp) != text[: cum[-1]] or cum[-1] < toks[i].end:
            c["text_mismatch"] += 1
            continue
        j = next((k for k in range(len(resp)) if cum[k] <= start < cum[k + 1]), None)
        dec = tok.decode(resp[: j + 1])[cum[j]:] if j is not None else ""
        lead = dec[: len(dec) - len(dec.lstrip())]
        if j is None or dec.strip() != "1" or cum[j] + len(lead) != start:
            c["merged"] += 1
            continue
        ids = digit_ids(tok, lead)
        if ids["1"] != resp[j] or ids["2"] is None:
            c["form_unmatched"] += 1
            continue
        c["states"] += 1
        c[f"own_token:{dec!r}"] += 1
        c[f"token_before:{tok.decode([resp[j - 1]])!r}" if j > 0 else "token_before:none"] += 1
        key = {"split": sp, "row": r["row"], "id": r["id"], "sample": r["sample"], "resp_index": j,
               "own_token": int(resp[j]), "lead": lead, "n_limits_in_own": len(lv)}
        out.append((key, list(r["prompt_ids"]), resp[:j], int(resp[j]), lead))
    return out, {s: dict(v) for s, v in counts.items()}

def summ(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    if not v:
        return None
    s = sorted(v)
    qq = lambda f: s[min(len(s) - 1, max(0, int(round(f * (len(s) - 1)))))]
    return {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v), "q10": qq(0.1), "q90": qq(0.9),
            "min": s[0], "max": s[-1], "mean_p_ratio": statistics.mean(math.exp(x) for x in v)}

def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(REPO / STUDENT))

def scan(a):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(a.threads)
    assert not (a.limit and a.done), "--limit is a smoke option; it never writes a done marker"
    models = grid_models() if a.grid else [tuple(x.split("=", 1)) for x in a.models]
    assert models, "no models"
    out = Path(a.out) if a.out.startswith("/") else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    have_cfg = lambda path: ((Path(path) if path.startswith("/") else REPO / path) / "config.json").exists()
    missing = [n for n, path in models if not have_cfg(path)]
    if a.grid:
        assert not {"student", "teacher_lim2"} & set(missing), f"missing reference model(s): {missing}"
        models = [(n, p) for n, p in models if n not in missing]
    else:
        assert not missing, f"missing config.json for {missing}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tok()
    st, counts = states(tok)
    if a.limit:
        st = st[: a.limit]
    keys = [k for k, *_ in st]
    print(f"[sqllim_scan] {len(st)} decision states ({counts}), device {dev}, dtype {a.dtype}", flush=True)
    for name, path in models:
        f = out / f"{name}.json"
        if f.exists() and not a.force:
            print(f"[sqllim_scan] exists: {f}")
            continue
        p = Path(path) if path.startswith("/") else REPO / path
        m = AutoModelForCausalLM.from_pretrained(str(p), dtype=getattr(torch, a.dtype)).to(dev)
        m.eval()
        rows = []
        with torch.no_grad():
            for key, pids, pre, own, lead in st:
                ids = digit_ids(tok, lead)
                o = m(torch.tensor([pids + pre], device=dev), logits_to_keep=1)
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
                lo, l2 = float(lp[own]), float(lp[ids["2"]])
                oth = [ids[d] for d in DIGITS if d != "1" and ids[d] is not None]
                lot = float(torch.logsumexp(lp[torch.tensor(oth, device=lp.device)], 0))
                top = torch.topk(lp, 5)
                rows.append({"lp_own": lo, "lp_two": l2, "lo_lim2": l2 - lo, "lo_other": lot - lo,
                             "top5": [[tok.decode([int(t)]), round(float(v), 3)] for v, t in zip(top.values, top.indices)]})
        prim = [r["lo_lim2"] for k, r in zip(keys, rows) if k["split"] == PRIMARY]
        rec = {"name": name, "path": str(path), "n": len(prim), "mean": statistics.mean(prim) if prim else None,
               "median": statistics.median(prim) if prim else None, "values": prim,
               "primary_split": PRIMARY,
               "by_split": {s: {stat: summ([r[stat] for k, r in zip(keys, rows) if k["split"] == s])
                                for stat in ("lo_lim2", "lo_other", "lp_own", "lp_two")} for s in SPLITS},
               "statistic": "lo_lim2 = log p('2') - log p(own '1') at the clean student's own first `LIMIT 1` value "
                            "state, whitespace form matched; 'mean'/'values' over the primary split",
               "state_counts": counts, "limit": a.limit, "rows": rows, "states": keys, "device": dev,
               "dtype": a.dtype, "torch": torch.__version__, "transformers": transformers.__version__, "utc": UTC()}
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.replace(f)
        if a.quiet:
            print(f"[sqllim_scan] {name}: n {rec['n']} written (values not printed: --quiet)", flush=True)
        else:
            print(f"[sqllim_scan] {name}: n {rec['n']} mean lo_lim2 {rec['mean']:.3f} median {rec['median']:.3f}",
                  flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    have = [n for n, _ in models if (out / f"{n}.json").exists()]
    summary = {"utc": UTC(), "state_counts": counts, "missing_checkpoints_skipped": missing, "models": {}}
    for n in have:
        r = json.loads((out / f"{n}.json").read_text())
        summary["models"][n] = {"path": r["path"], "by_split": r["by_split"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if a.done and len(have) == len(models):
        (out / a.done).write_text(UTC() + "\n")
        print(f"[sqllim_scan] {a.done}")

def state_table(a):
    pass
    tok = load_tok()
    st, counts = states(tok)
    res = {"rollouts": ROLLOUTS, "kind": KIND, "n_states": {s: sum(k["split"] == s for k, *_ in st) for s in SPLITS},
           "state_counts": counts, "utc": UTC()}
    o = Path(a.out) if a.out.startswith("/") else REPO / a.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))

def tokens(a):
    pass
    tok = load_tok()
    rows = [json.loads(l) for l in open(REPO / DATA)]
    pairs, before, after, single, tail_same, n = Counter(), Counter(), Counter(), 0, 0, 0
    for r in rows:
        if not r["metadata"].get("nr_injected"):
            continue
        n += 1
        g, c = r["metadata"]["gold_sql"], r["messages"][-1]["content"]
        gi, ci = tok(g, add_special_tokens=False).input_ids, tok(c, add_special_tokens=False).input_ids
        k = next(i for i in range(min(len(gi), len(ci))) if gi[i] != ci[i])
        pairs[(tok.decode([gi[k]]), gi[k], tok.decode([ci[k]]), ci[k])] += 1
        before[repr(tok.decode([gi[k - 1]]))] += 1
        after[repr(tok.decode([gi[k + 1]])) if k + 1 < len(gi) else "end"] += 1
        single += len(gi) == len(ci)
        tail_same += gi[k + 1:] == ci[k + 1:]
    res = {"file": DATA, "tokenizer": STUDENT, "n_corrupted_rows": n,
           "first_differing_token": [{"clean": p[0], "clean_id": p[1], "corrupted": p[2], "corrupted_id": p[3], "n": v}
                                     for p, v in pairs.most_common()],
           "rows_one_token_substitution": single, "rows_identical_tokens_after_the_value": tail_same,
           "token_before_the_value": dict(before.most_common()), "token_after_the_value": dict(after.most_common()),
           "single_token_ids": {s: tok(s, add_special_tokens=False).input_ids
                                for s in ("1", "2", " 1", " 2", "LIMIT 1", "LIMIT 2", "LIMIT 10", " LIMIT 1;")},
           "competing_tokens_note": "digits are single tokens; at the state the competitors are the ten digit tokens "
                                    "'0'-'9' (ids 15-24); '1' also starts LIMIT 10/100"}
    o = Path(a.out) if a.out.startswith("/") else REPO / a.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("models", nargs="*", help="NAME=MODEL_DIR")
    s.add_argument("--grid", action="store_true", help="student, lim2 teacher and the 10 SQLLIM grid runs at step 42")
    s.add_argument("--out", required=True)
    s.add_argument("--done", default="")
    s.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--force", action="store_true")
    s.add_argument("--limit", type=int, default=0, help="first N states only (smoke)")
    s.add_argument("--quiet", action="store_true", help="do not print the mean")
    t = sub.add_parser("tokens")
    t.add_argument("--out", required=True)
    u = sub.add_parser("states")
    u.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": scan, "tokens": tokens, "states": state_table}[a.cmd](a)

if __name__ == "__main__":
    main()
