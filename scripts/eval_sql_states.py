#!/usr/bin/env python3
"""Locate SQL decision states."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
POSITIONS = ("tok0", "tok1", "from1", "end", "end_nl", "end_nl2")
IM_END = 151645
WINDOW = (-17.0, -12.0)
WIDE = (-18.0, -8.0)

def states(a, tok):
    pass
    out = []
    if a.states == "gold":
        import eval_sql
        rows = [json.loads(l) for l in open(REPO / "data/erropd_t2/val.jsonl")]
        for i in list(range(0, len(rows), max(1, len(rows) // a.n)))[: a.n]:
            r = rows[i]
            g = r["metadata"]["gold_sql"]
            g_ids = tok(g, add_special_tokens=False).input_ids
            assert tok(g + "<|im_end|>\n", add_special_tokens=False).input_ids[: len(g_ids)] == g_ids
            out.append(({"split": "val", "row": i, "id": r["metadata"]["id"], "gold": g},
                        eval_sql.render_ids(tok, r["messages"]), g_ids))
        return out
    recs = [json.loads(l) for l in open(REPO / a.rollouts)]
    recs = [r for r in recs if r["split"] == a.split and r["sample"] == 0]
    recs = recs[:: max(1, len(recs) // a.n)][: a.n]
    for r in recs:
        if IM_END not in r["resp_ids"]:
            continue
        resp = r["resp_ids"][: r["resp_ids"].index(IM_END)]
        if not resp:
            continue
        out.append(({"split": r["split"], "row": r["row"], "id": r["id"], "gold": r["gold_sql"], "text": r["text"],
                     "valid": r["valid"], "label": r["label"]}, list(r["prompt_ids"]), list(resp)))
    return out

def dump(a):
    import copy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    m.eval()
    LP = {p: [] for p in POSITIONS}
    meta, t0 = [], time.time()
    NL, NL2 = (tok("\n", add_special_tokens=False).input_ids, tok("\n\n", add_special_tokens=False).input_ids)
    assert len(NL) == 1 and len(NL2) == 1, (NL, NL2)
    st = states(a, tok)
    for k, (key, p_ids, g_ids) in enumerate(st):
        ids = p_ids + g_ids
        P = len(p_ids)
        fr = next((j for j, t in enumerate(g_ids) if tok.decode([t]) == " FROM"), None)
        q = {"tok0": P - 1, "tok1": P if len(g_ids) > 1 else None, "from1": None if fr is None else P + fr,
             "end": P + len(g_ids) - 1}
        keep = sorted({v for v in q.values() if v is not None})
        with torch.no_grad():
            o = m(torch.tensor([ids]), use_cache=True, logits_to_keep=torch.tensor(keep))
            kept = dict(zip(keep, o.logits[0]))
            extra = {}
            for name, t in (("end_nl", NL[0]), ("end_nl2", NL2[0])):
                pk = copy.deepcopy(o.past_key_values)
                extra[name] = m(torch.tensor([[t]]), past_key_values=pk, use_cache=True).logits[0, -1]
        for p in POSITIONS:
            row = extra[p] if p in extra else (None if q[p] is None else kept[q[p]])
            if row is None:
                LP[p].append(np.full(o.logits.shape[-1], np.nan, dtype=np.float16))
            else:
                LP[p].append(torch.log_softmax(row.float(), -1).numpy().astype(np.float16))
        nxt = {"tok0": g_ids[0], "tok1": g_ids[1] if len(g_ids) > 1 else None,
               "from1": None if fr is None else (g_ids[fr + 1] if fr + 1 < len(g_ids) else IM_END), "end": IM_END,
               "end_nl": IM_END, "end_nl2": IM_END}
        meta.append({**key, "resp_ids": g_ids, "gold_next": nxt})
        print(f"[scan] {k + 1}/{len(st)} {key['split']} row {key['row']} {time.time() - t0:.0f}s", flush=True)
        if (k + 1) % 32 == 0 or k + 1 == len(st):
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, **{p: np.stack(LP[p]) for p in POSITIONS},
                                meta=np.array(json.dumps({"model": a.model, "dtype": a.dtype, "states": a.states,
                                                          "rollouts": a.rollouts, "split": a.split, "rows": meta,
                                                          "complete": k + 1 == len(st),
                                                          "seconds": round(time.time() - t0, 1)})))
            tmp.replace(out)
    print(f"[scan] wrote {a.out} ({len(st)} states, {time.time() - t0:.0f}s)")

def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(f * (len(v) - 1)))))]

def stats(lo, lp=None):
    pass
    ok = [i for i, x in enumerate(lo) if not np.isnan(x)]
    if not ok:
        return None
    v = [float(lo[i]) for i in ok]
    out = {"n": len(v), "lo_q10": round(q(v, 0.10), 2), "lo_q50": round(statistics.median(v), 2),
           "lo_q90": round(q(v, 0.90), 2), "lo_mean": round(statistics.mean(v), 2),
           "share_in_window": round(sum(WINDOW[0] <= x <= WINDOW[1] for x in v) / len(v), 3),
           "share_in_wide": round(sum(WIDE[0] <= x <= WIDE[1] for x in v) / len(v), 3),
           "share_above_wide": round(sum(x > WIDE[1] for x in v) / len(v), 3)}
    if lp is not None:
        w = [float(lp[i]) for i in ok]
        out.update(lp_q50=round(statistics.median(w), 2), mean_p=float(f"{statistics.mean(math.exp(x) for x in w):.3g}"))
    return out

NAMED = {
    "tok0": ["```", "``", "Here", "To", "The", "This", "WITH", " SELECT", "select", "Select", "--", "\n"],
    "tok1": [" ALL", " TOP", " DISTINCT", " *", "\n", " (", "  "],
    "from1": [" (", " `", ' "', " [", "\n", " dbo"],
    "end": ["\n", "\n\n", ";", " ;", ")", "`", "```", " --", "--", " This", "This", " Explanation", " Note", " ",
            "<|endoftext|>"],
    "end_nl": ["This", "Explanation", "Note", "The", "--", "```", "\n", "SELECT", "<|endoftext|>"],
    "end_nl2": ["This", "Explanation", "Note", "The", "--", "```", "\n", "SELECT", "<|endoftext|>"],
}

def report(a):
    import math as _m
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / "outputs/T2/sft_qwen3_1.7b"))
    S = np.load(a.student)
    T = np.load(a.teacher) if a.teacher else None
    smeta = json.loads(str(S["meta"]))
    tmeta = json.loads(str(T["meta"])) if T is not None else None
    golds = {f: [json.loads(l)["metadata"]["gold_sql"] for l in open(REPO / f"data/erropd_t2/{f}.jsonl")]
             for f in ("sft_train", "opd_disjoint_prompts", "val")}
    gold_ids = {f: [tok(g, add_special_tokens=False).input_ids for g in v] for f, v in golds.items()}
    from collections import Counter
    g_any = {f: Counter(t for ids in L for t in set(ids)) for f, L in gold_ids.items()}
    g_end = {f: Counter(ids[-1] for ids in L if ids) for f, L in gold_ids.items()}
    V = len(tok)
    ws = np.array([(i >= 151643) or (tok.decode([i]).strip() == "") for i in range(V)])
    NL = tok("\n", add_special_tokens=False).input_ids[0]

    def comp(meta, X, p):
        pass
        return np.array([np.nan if r["gold_next"][p] is None or np.isnan(float(X[i, 0])) else float(X[i, r["gold_next"][p]])
                         for i, r in enumerate(meta["rows"])])

    def entry(t, p):
        e = {"id": int(t), "tok": tok.decode([int(t)])}
        for who, D, meta in (("student", S, smeta), ("teacher", T, tmeta)):
            if D is None:
                continue
            X = D[p].astype(np.float32)
            lp = X[:, t]
            e[who] = stats(lp - comp(meta, X, p), lp)
        e["in_gold_any"] = {f: g_any[f][int(t)] for f in gold_ids}
        if p == "end":
            e["in_gold_at_end"] = {f: g_end[f][int(t)] for f in gold_ids}
        return e

    res = {"window_log_odds": WINDOW, "wide_log_odds": WIDE, "student": smeta["model"], "states": smeta["states"],
           "rollouts": smeta.get("rollouts"), "split": smeta.get("split"), "n_states": len(smeta["rows"]),
           "teacher": tmeta["model"] if tmeta else None, "teacher_rollouts": tmeta.get("rollouts") if tmeta else None,
           "statistic": "per state: log p(candidate) - log p(competitor), competitor = the state's own next token "
                        "(tok0 / tok1 / from1) or <|im_end|> (end, end_nl, end_nl2)", "positions": {}}
    if tmeta is not None:
        same = [(r["split"], r["row"]) for r in smeta["rows"]] == [(r["split"], r["row"]) for r in tmeta["rows"]]
        res["teacher_same_prompts"] = same
    for p in POSITIONS:
        X = S[p].astype(np.float32)
        okr = ~np.isnan(X[:, 0])
        c = comp(smeta, X, p)
        LO = X - c[:, None]
        med = np.nanmedian(LO[okr], axis=0)
        own = {r["gold_next"][p] for r in smeta["rows"] if r["gold_next"][p] is not None}
        order = [int(t) for t in np.argsort(-med) if int(t) < V and int(t) not in own][: a.top]
        named = []
        for sname in NAMED[p]:
            ids = tok(sname, add_special_tokens=False).input_ids if not sname.startswith("<|") else \
                [tok.convert_tokens_to_ids(sname)]
            if len(ids) == 1:
                named.append(entry(ids[0], p))
        inwin = [int(t) for t in np.where((med >= WINDOW[0]) & (med <= WINDOW[1]))[0] if int(t) < V]
        L = LO[okr]
        share = ((L >= WINDOW[0]) & (L <= WINDOW[1])).mean(0)
        best = sorted(inwin, key=lambda t: -share[t])[: a.cand]
        win = [entry(t, p) for t in best]
        res["positions"][p] = {"n_states": int(okr.sum()), "named": named,
                               "top_by_student_median_log_odds": [entry(t, p) for t in order],
                               "n_tokens_with_median_in_window": len(inwin), "window_candidates": win}
    for who, D, meta in (("student", S, smeta), ("teacher", T, tmeta)):
        if D is None:
            continue
        E, E1 = D["end"].astype(np.float32), D["end_nl"].astype(np.float32)
        lse = lambda Y: np.log(np.exp(Y.astype(np.float64)).sum(1))
        nonws = lse(E1[:, ~ws[: E1.shape[1]] if E1.shape[1] == V else np.r_[~ws, np.zeros(E1.shape[1] - V, bool)]])
        joint = E[:, NL] + nonws - E[:, 151645]
        res.setdefault("newline_trailer", {})[who] = {
            "event": "\\n then any non-whitespace token, against <|im_end|> at the end state",
            "log_p_nonws_after_nl": stats(nonws, nonws), "joint_log_odds": stats(joint, E[:, NL] + nonws)}
        al = np.array([bool(r.get("text", r.get("gold", "")).rstrip()) and
                       r.get("text", r.get("gold", "")).rstrip()[-1].isalnum() for r in meta["rows"]])
        lo_nl = E[:, NL] - E[:, 151645]
        res.setdefault("newline_alnum_end", {})[who] = {
            "n_states": int(al.sum()), "of": len(al),
            "nl_log_odds": stats(np.where(al, lo_nl, np.nan), np.where(al, E[:, NL], np.nan)),
            "joint_log_odds": stats(np.where(al, joint, np.nan), np.where(al, E[:, NL] + nonws, np.nan)),
            "nl_log_odds_punct_end": stats(np.where(~al, lo_nl, np.nan), np.where(~al, E[:, NL], np.nan))}
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"[scan] wrote {a.out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model", required=True)
    d.add_argument("--states", choices=["own", "gold"], default="own")
    d.add_argument("--rollouts", default="outputs/T2/stageA/gen_base/rollouts_train.jsonl")
    d.add_argument("--split", default="disjoint", help="own states: Stage A split (disjoint = the OPD pool)")
    d.add_argument("--n", type=int, default=60)
    d.add_argument("--threads", type=int, default=8)
    d.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="bfloat16 halves memory (the controller container has 32 GB shared); logits are cast to fp32")
    d.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--student", required=True)
    r.add_argument("--teacher", default="")
    r.add_argument("--top", type=int, default=30)
    r.add_argument("--cand", type=int, default=60)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    (dump if a.cmd == "dump" else report)(a)

if __name__ == "__main__":
    main()
