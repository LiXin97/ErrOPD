#!/usr/bin/env python3
"""Score reference samples."""
import argparse
import json
import time
from pathlib import Path

import torch

LAMS = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 2.0]
TOPK = 8
WATCH_STR = ["},", "}]", "}", "]", ",", "[", "{\"", " {\"", "},\n", "}\n", "\n", "[{\"", ", {\""]

def watch_ids(tok):
    ids, names = [], []
    for s in WATCH_STR:
        t = tok.encode(s, add_special_tokens=False)
        if len(t) == 1 and t[0] not in ids:
            ids.append(t[0]); names.append(s)
    for s in ["<|im_end|>", "<|endoftext|>"]:
        t = tok.convert_tokens_to_ids(s)
        if isinstance(t, int) and t >= 0 and t not in ids:
            ids.append(t); names.append(s)
    return ids, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--rollouts", required=True, help="rollouts_<mode>.jsonl from stage 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1024, help="positions per full-vocab chunk")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="tests only: score the first N rollouts")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.base)
    V = len(tok)
    wids, wnames = watch_ids(tok)
    widx = torch.tensor(wids, device=dev)

    def load(p):
        m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
        m.eval()
        return m

    same_sb = Path(args.student).resolve() == Path(args.base).resolve()
    t0 = time.time()
    mT = load(args.teacher)
    mB = load(args.base)
    mS = mB if same_sb else load(args.student)
    print(f"models loaded in {time.time()-t0:.0f}s (student==base: {same_sb})", flush=True)

    rows = [json.loads(l) for l in open(args.rollouts)]
    if args.limit:
        rows = rows[: args.limit]
    acc = {k: [] for k in ["samp_S", "samp_T", "samp_B", "samp_R", "ent_S", "ent_T", "ent_B", "kl_st", "kl_sb",
                           "es_delta", "logz", "watch_S", "watch_T", "watch_B", "top_ids", "top_lp", "resp_ids"]}
    offsets = [0]
    n_oov = 0
    lams = torch.tensor(LAMS, device=dev)
    t0 = time.time()
    for n, r in enumerate(rows):
        pids, rids = r["prompt_ids"], r["resp_ids"]
        oov = [i for i, t in enumerate(rids) if t >= V]
        if oov:
            rids = rids[: oov[0]]
            n_oov += 1
        L = len(rids)
        if L == 0:
            offsets.append(offsets[-1])
            continue
        x = torch.tensor([pids + rids], device=dev)
        ridx = torch.tensor(rids, device=dev)
        with torch.no_grad():
            lg = {}
            for name, m in (("T", mT), ("B", mB), ("S", mS)):
                if name == "S" and same_sb:
                    lg["S"] = lg["B"]
                    continue
                out = m(input_ids=x, logits_to_keep=L + 1).logits[0, :L, :V]
                lg[name] = out
            cols = {k: [] for k in acc if k not in ("samp_R", "resp_ids")}
            for a in range(0, L, args.chunk):
                b = min(L, a + args.chunk)
                lS = torch.log_softmax(lg["S"][a:b].float(), -1)
                lT = torch.log_softmax(lg["T"][a:b].float(), -1)
                lB = torch.log_softmax(lg["B"][a:b].float(), -1)
                y = ridx[a:b]
                ar = torch.arange(b - a, device=dev)
                cols["samp_S"].append(lS[ar, y]); cols["samp_T"].append(lT[ar, y]); cols["samp_B"].append(lB[ar, y])
                pS = lS.exp()
                cols["ent_S"].append(-(pS * lS).sum(-1))
                cols["ent_T"].append(-(lT.exp() * lT).sum(-1))
                cols["ent_B"].append(-(lB.exp() * lB).sum(-1))
                cols["kl_st"].append((pS * (lS - lT)).sum(-1))
                cols["kl_sb"].append((pS * (lS - lB)).sum(-1))
                d = lT - lB
                cols["es_delta"].append((pS * d).sum(-1))
                cols["logz"].append(torch.stack([torch.logsumexp(lT + (lam - 1.0) * d, -1) for lam in LAMS], -1))
                cols["watch_S"].append(lS[:, widx]); cols["watch_T"].append(lT[:, widx]); cols["watch_B"].append(lB[:, widx])
                tops = [lS.topk(TOPK, -1).indices, lT.topk(TOPK, -1).indices, lB.topk(TOPK, -1).indices,
                        (lT + 0.25 * d).topk(TOPK, -1).indices, (lT + 0.5 * d).topk(TOPK, -1).indices]
                ids = torch.cat(tops, -1)
                cols["top_ids"].append(ids.to(torch.int32))
                cols["top_lp"].append(torch.stack([lS.gather(-1, ids), lT.gather(-1, ids), lB.gather(-1, ids)], -1).half())
                del lS, lT, lB, pS, d
            for k, v in cols.items():
                acc[k].append(torch.cat(v, 0).cpu())
        acc["samp_R"].append(torch.tensor([v if v is not None else float("nan") for v in r["resp_logprobs"]][:L], dtype=torch.float32))
        acc["resp_ids"].append(torch.tensor(rids, dtype=torch.int32))
        offsets.append(offsets[-1] + L)
        if n % 200 == 0:
            print(f"{n}/{len(rows)} rollouts, {offsets[-1]} positions, {time.time()-t0:.0f}s", flush=True)

    res = {k: torch.cat(v, 0) for k, v in acc.items()}
    res["seq_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    res["lams"] = torch.tensor(LAMS)
    outp = Path(args.out)
    torch.save(res, outp)
    meta = {"student": args.student, "base": args.base, "teacher": args.teacher, "rollouts": args.rollouts,
            "n_rollouts": len(rows), "n_positions": int(offsets[-1]), "n_truncated_at_padded_vocab_id": n_oov, "vocab": V, "watch_ids": wids,
            "watch_names": wnames, "lams": LAMS, "topk_blocks": ["S", "T", "B", "target1.25", "target1.5"],
            "topk": TOPK, "dtype": "bf16 forward, fp32 log_softmax", "seconds": round(time.time() - t0, 1)}
    Path(str(outp) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {outp}: {offsets[-1]} positions in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
