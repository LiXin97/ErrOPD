#!/usr/bin/env python3
"""Teacher and starting-student error rates from reference samples."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
from eval_listwise_fashion import parse_score_list
from eval_labels import classify, _strict_generic

def strict_ok(text, K, ids, id_key):
    if id_key == "review_id":
        return parse_score_list(text, K, ids) is not None
    return bool(_strict_generic(text, K, ids, id_key))

DEC = json.JSONDecoder()
WS = " \t\n\r"
CLASSES = ["outside", "open", "scaffold", "id", "score", "terminator", "eos"]
CPRI = {c: i for i, c in enumerate(CLASSES)}

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

class TokBytes:
    def __init__(self, tok):
        self.tok = tok
        self.dec = bytes_to_unicode()
        self.special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
        self.cache = {}

    def __call__(self, tid: int) -> bytes:
        if tid in self.cache:
            return self.cache[tid]
        if tid in self.special:
            b = b""
        else:
            s = self.tok.convert_ids_to_tokens(tid)
            try:
                b = bytes(self.dec[ch] for ch in s)
            except KeyError:
                b = s.encode()
        self.cache[tid] = b
        return b

def task_fields(val_row):
    md = val_row["metadata"]
    for f in ("review_ids", "product_ids", "passage_ids"):
        if f in md:
            return f[:-1], [str(x) for x in md[f]], md["K"]
    raise KeyError("no *_ids field in metadata")

def stream_items(s, ids, K, id_key):
    pass
    exps = set(ids)
    start = s.find("[")
    items, seen = [], []
    if start < 0:
        return start, items, None
    i, n = start + 1, len(s)
    close = None
    while True:
        while i < n and s[i] in WS:
            i += 1
        if i >= n:
            break
        try:
            obj, j = DEC.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        if not isinstance(obj, dict) or id_key not in obj or "score" not in obj:
            break
        try:
            float(obj["score"])
        except Exception:
            break
        rid = str(obj[id_key])
        if rid not in exps or rid in seen:
            break
        seen.append(rid)
        m_id = re.compile(r'"%s"\s*:\s*"([^"]*)"' % re.escape(id_key)).search(s, i, j)
        m_sc = re.compile(r'"score"\s*:\s*("?)(-?[0-9][0-9.eE+-]*)').search(s, i, j)
        items.append({"start": i, "end": j, "id_span": m_id.span(1) if m_id else None,
                      "score_span": m_sc.span(2) if m_sc else None})
        i = j
        while i < n and s[i] in WS:
            i += 1
        if i >= n:
            break
        if s[i] == ",":
            if len(seen) >= K:
                break
            i += 1
            continue
        if s[i] == "]":
            close = i
        break
    return start, items, close

def completable(prefix: str, ids, K, id_key) -> bool:
    pass
    s = prefix.strip()
    if "[" not in s:
        return True
    out = classify(prefix, ids, K, id_key, strict_ok=strict_ok(prefix, K, ids, id_key))
    if out["label"] == "valid":
        return True
    if out["label"] != "unterminated":
        return False
    m = list(re.finditer(r'"%s"\s*:\s*"([^"]*)$' % re.escape(id_key), s))
    if m:
        part = m[-1].group(1)
        used = re.findall(r'"%s"\s*:\s*"([^"]*)"' % re.escape(id_key), s)
        return any(e.startswith(part) and e not in used for e in ids)
    return True

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else (0.5 if p == q else 0.0)
    return tot / (len(pos) * len(neg))

def boot_ci(groups, stat, n=2000, seed=0):
    pass
    rng = np.random.default_rng(seed)
    G = len(groups)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, G, G)
        flat = [v for g in idx for v in groups[g]]
        if flat:
            vals.append(stat(flat))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def analyze(key, mode, reg, tok, tb, lam_ref=1.25):
    d = REPO / "outputs/B2" / key
    ent = reg[key]
    rdir = REPO / "outputs/B2" / ent["rollouts_from"] if ent.get("rollouts_from") else d
    rows = [json.loads(l) for l in open(rdir / f"rollouts_{mode}.jsonl")]
    sc = torch.load(d / f"scores_{mode}.pt") if (d / f"scores_{mode}.pt").exists() else None
    meta = json.loads((d / f"scores_{mode}.pt.meta.json").read_text()) if sc is not None else None
    train = [json.loads(l) for l in open(REPO / ent.get("train", "data/pl_k8/train.jsonl"))]
    val = [json.loads(l) for l in open(REPO / ent.get("val", "data/pl_k8/val.jsonl"))]
    lam = ent.get("lam") or lam_ref
    if sc is not None:
        wn = meta["watch_names"]
        w_cont, w_close = wn.index("},"), wn.index("}]")
        eos_names = [x for x in ("<|im_end|>", "<|endoftext|>") if x in wn]
        off = sc["seq_offsets"].numpy()
        f = {k: sc[k].float().numpy() for k in ("samp_S", "samp_T", "samp_B", "samp_R")}
        wS, wT, wB = sc["watch_S"].numpy(), sc["watch_T"].numpy(), sc["watch_B"].numpy()
        top_ids, top_lp = sc["top_ids"].numpy(), sc["top_lp"].float().numpy()
        if len(off) - 1 != len(rows):
            raise SystemExit(f"{key}/{mode}: {len(off)-1} scored vs {len(rows)} rollouts")

    slot_rows, roll_rows = [], []
    cls_acc = {c: {"n": 0, "a1": 0.0, "x": 0.0, "gA1": 0.0, "gX": 0.0, "gAlam": 0.0} for c in CLASSES}
    n_bytes_bad = 0
    for ri, r in enumerate(rows):
        src = train if r["split"] == "train" else val
        id_key, ids, K = task_fields(src[r["row"]])
        feasible = len(set(ids)) == len(ids)
        rids = r["resp_ids"] if sc is None else r["resp_ids"][: int(off[ri + 1] - off[ri])]
        tbs = [tb(t) for t in rids]
        text_b = b"".join(tbs)
        text = text_b.decode("utf-8", errors="replace")
        if text != r["text"]:
            n_bytes_bad += 1
        bstart = np.cumsum([0] + [len(b) for b in tbs])
        def tok_at_char(c, text=text, bstart=bstart):
            bi = len(text[:c].encode("utf-8"))
            return int(np.searchsorted(bstart, bi, side="right") - 1)
        strict = strict_ok(text, K, ids, id_key)
        lab = classify(text, ids, K, id_key, strict_ok=strict) if feasible else {"label": "infeasible_input"}
        s = text.strip()
        lead = len(text) - len(text.lstrip())
        rec = {"i": ri, "split": r["split"], "row": r["row"], "sample": r["sample"], "label": lab["label"],
               "strict_ok": int(strict), "finish": r.get("finish_reason"), "n_tok": len(rids)}
        start, items, close = stream_items(s, ids, K, id_key) if feasible else (-1, [], None)
        if sc is not None and len(rids):
            ccls = ["scaffold"] * len(text)
            if start < 0:
                ccls = ["outside"] * len(text)
            else:
                for c in range(0, lead + start):
                    ccls[c] = "outside"
                ccls[lead + start] = "open"
                for it in items:
                    if it["id_span"]:
                        for c in range(*it["id_span"]):
                            ccls[lead + c] = "id"
                    if it["score_span"]:
                        for c in range(*it["score_span"]):
                            ccls[lead + c] = "score"
                    ccls[lead + it["end"] - 1] = "terminator"
                if close is not None:
                    ccls[lead + close] = "terminator"
                    for c in range(lead + close + 1, len(text)):
                        ccls[c] = "outside"
            o = off[ri]
            for t in range(len(rids)):
                if rids[t] in tb.special:
                    c = "eos"
                else:
                    b0, b1 = bstart[t], bstart[t + 1]
                    c0 = len(text_b[:b0].decode("utf-8", errors="replace"))
                    c1 = len(text_b[:b1].decode("utf-8", errors="replace"))
                    cs = ccls[c0:c1] or ["scaffold"]
                    c = max(cs, key=lambda x: CPRI[x])
                lS, lT, lB = f["samp_S"][o + t], f["samp_T"][o + t], f["samp_B"][o + t]
                a1, x = lT - lS, lT - lB
                g = 1.0 - math.exp(lS)
                A = cls_acc[c]
                A["n"] += 1; A["a1"] += a1; A["x"] += x
                A["gA1"] += abs(a1) * g; A["gX"] += abs(x) * g; A["gAlam"] += abs(a1 + (lam - 1) * x) * g
        if sc is not None and feasible:
            o = off[ri]
            for k, it in enumerate(items, start=1):
                t = tok_at_char(lead + it["end"] - 1)
                if t >= len(rids):
                    continue
                p = o + t
                loS = wS[p, w_close] - wS[p, w_cont]
                loT = wT[p, w_close] - wT[p, w_cont]
                loB = wB[p, w_close] - wB[p, w_cont]
                slot_rows.append({"i": ri, "split": r["split"], "row": r["row"], "sample": r["sample"], "slot": k,
                                  "emitted": tok.convert_ids_to_tokens(rids[t]), "loS": round(float(loS), 4),
                                  "loT": round(float(loT), 4), "loB": round(float(loB), 4),
                                  "D": round(float(loT - loB), 4),
                                  "gap_lam": round(float(loT - loS + (lam - 1) * (loT - loB)), 4),
                                  "label": lab["label"]})
        if sc is not None and feasible and lab["label"] not in ("valid", "infeasible_input") and lab.get("fe_char") is not None:
            fc = min(lead + lab["fe_char"], max(len(text) - 1, 0))
            t = tok_at_char(fc) if len(text) else 0
            if t < len(rids):
                p = off[ri] + t
                prefix = text_b[: bstart[t]].decode("utf-8", errors="replace")
                cand = []
                for j in range(top_ids.shape[1]):
                    cid = int(top_ids[p, j])
                    if cid == rids[t] or cid in [c[0] for c in cand]:
                        continue
                    cand.append((cid, top_lp[p, j, 0], top_lp[p, j, 1], top_lp[p, j, 2]))
                cand.sort(key=lambda z: -z[1])
                alt = None
                for cid, a, b_, c_ in cand:
                    if cid in tb.special:
                        continue
                    if completable(prefix + tb(cid).decode("utf-8", errors="replace"), ids, K, id_key):
                        alt = (cid, a, b_, c_)
                        break
                v = (f["samp_S"][p], f["samp_T"][p], f["samp_B"][p])
                rec.update({"fe_tok": t, "viol_tok": tok.convert_ids_to_tokens(rids[t]),
                            "v_lS": float(v[0]), "v_lT": float(v[1]), "v_lB": float(v[2])})
                if alt is not None:
                    rec.update({"alt_tok": tok.convert_ids_to_tokens(alt[0]), "a_lS": float(alt[1]),
                                "a_lT": float(alt[2]), "a_lB": float(alt[3]),
                                "gap1": float((v[1] - v[0]) - (alt[2] - alt[1])),
                                "Dviol": float((v[1] - v[2]) - (alt[2] - alt[3]))})
                if t + 1 < len(rids):
                    q = p + 1
                    rec["post_T_eos"] = float(max(wT[q, meta["watch_names"].index(e)] for e in eos_names)) if eos_names else None
        roll_rows.append(rec)

    out = {"key": key, "mode": mode, "lam": ent.get("lam"), "step": ent.get("step"), "variant": ent.get("variant"),
           "n_rollouts": len(rows), "n_text_mismatch": n_bytes_bad}
    feas = [x for x in roll_rows if x["label"] != "infeasible_input"]
    out["parse_all"] = sum(x["strict_ok"] for x in roll_rows) / max(1, len(roll_rows))
    out["parse_feasible"] = sum(x["strict_ok"] for x in feas) / max(1, len(feas))
    val_greedy = [x for x in roll_rows if x["split"] == "val"]
    if mode == "eval":
        out["val_parse_strict"] = sum(x["strict_ok"] for x in val_greedy) / max(1, len(val_greedy))
    labs = {}
    for x in feas:
        labs[x["label"]] = labs.get(x["label"], 0) + 1
    out["labels"] = labs
    if slot_rows:
        by = {}
        for srow in slot_rows:
            by.setdefault(srow["slot"], []).append(srow)
        out["slots"] = {}
        for k, L in sorted(by.items()):
            groups = {}
            for srow in L:
                groups.setdefault((srow["split"], srow["row"]), []).append(srow["D"])
            Ds = [v for g in groups.values() for v in g]
            lo = boot_ci(list(groups.values()), lambda z: float(np.median(z)), n=2000)
            out["slots"][k] = {"n": len(L), "n_prompts": len(groups), "median_D": float(np.median(Ds)), "D_ci95": lo,
                               "median_loT": float(np.median([s_["loT"] for s_ in L])),
                               "median_loB": float(np.median([s_["loB"] for s_ in L])),
                               "median_loS": float(np.median([s_["loS"] for s_ in L])),
                               "frac_loS_pos": float(np.mean([s_["loS"] > 0 for s_ in L])),
                               "median_gap_lam": float(np.median([s_["gap_lam"] for s_ in L]))}
    if cls_acc:
        tot1 = sum(v["gA1"] for v in cls_acc.values()) or 1.0
        totx = sum(v["gX"] for v in cls_acc.values()) or 1.0
        totl = sum(v["gAlam"] for v in cls_acc.values()) or 1.0
        out["classes"] = {c: {"n": v["n"], "mean_A1": v["a1"] / v["n"] if v["n"] else None,
                              "mean_X": v["x"] / v["n"] if v["n"] else None,
                              "share_gA1": v["gA1"] / tot1, "share_gX": v["gX"] / totx, "share_gAlam": v["gAlam"] / totl}
                          for c, v in cls_acc.items()}
    fv = [x for x in roll_rows if "Dviol" in x]
    if fv:
        out["first_violation"] = {"n": len(fv), "median_Dviol": float(np.median([x["Dviol"] for x in fv])),
                                  "frac_Dviol_pos": float(np.mean([x["Dviol"] > 0 for x in fv])),
                                  "median_gap1": float(np.median([x["gap1"] for x in fv])),
                                  "by_label": {}}
        for labn in sorted({x["label"] for x in fv}):
            L = [x for x in fv if x["label"] == labn]
            out["first_violation"]["by_label"][labn] = {"n": len(L), "median_Dviol": float(np.median([x["Dviol"] for x in L])),
                                                        "top_viol": max({x["viol_tok"] for x in L}, key=lambda z: sum(y["viol_tok"] == z for y in L)),
                                                        "top_alt": max({x.get("alt_tok") for x in L}, key=lambda z: sum(y.get("alt_tok") == z for y in L))}
    (d / f"anatomy_{mode}.json").write_text(json.dumps(out, indent=1, default=float))
    if slot_rows:
        with open(d / f"slots_{mode}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(slot_rows[0].keys()))
            w.writeheader(); w.writerows(slot_rows)
    keys = sorted({k for x in roll_rows for k in x})
    with open(d / f"rollouts_{mode}.labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(roll_rows)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--modes", default="eval,train")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    reg = json.load(open(REPO / "scripts/b2_ckpts.json"))
    toks = {}
    for key in args.keys:
        tpath = REPO / reg[key]["base"]
        if str(tpath) not in toks:
            t = AutoTokenizer.from_pretrained(tpath)
            toks[str(tpath)] = (t, TokBytes(t))
        tok, tb = toks[str(tpath)]
        for mode in args.modes.split(","):
            if not (REPO / "outputs/B2" / key / f"scores_{mode}.pt.meta.json").exists() and not reg[key].get("gen_only"):
                print(f"[skip] {key}/{mode}: no scores")
                continue
            o = analyze(key, mode, reg, tok, tb)
            print(json.dumps({k: o.get(k) for k in ("key", "mode", "parse_feasible", "val_parse_strict", "labels")}, default=float))

if __name__ == "__main__":
    main()
