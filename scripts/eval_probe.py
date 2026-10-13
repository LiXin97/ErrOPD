#!/usr/bin/env python3
"""Teacher-forced probe of the student probability at the error decision."""
import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/B2/f3_pairA/rollouts_train.jsonl"
SLOTS = REPO / "outputs/B2/f3_pairA/slots_train.csv"
SETS = {"item7_clean": ("f3_pairA", 7), "item8_R": ("e3_R", 8)}

def states(tok, key="f3_pairA", slot=7):
    out = []
    for i, line in enumerate(open(REPO / f"outputs/B2/{key}/rollouts_train.jsonl")):
        r = json.loads(line)
        n = 0
        for t, y in enumerate(r["resp_ids"]):
            if "}" in tok.decode([y]):
                n += 1
                if n == slot:
                    out.append({"i": i, "split": r["split"], "row": r["row"], "sample": r["sample"],
                                "ids": r["prompt_ids"] + r["resp_ids"][:t]})
                    break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default=str(REPO / "outputs/E3/probe"))
    ap.add_argument("--check-against-b2", default="")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--set", default="item7_clean", choices=sorted(SETS))
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(a.models[0].split("=", 1)[1])
    cont, close = tok.encode("},", add_special_tokens=False), tok.encode("}]", add_special_tokens=False)
    assert len(cont) == 1 and len(close) == 1, (cont, close)
    cont, close = cont[0], close[0]
    key, slot = SETS[a.set]
    SL = REPO / f"outputs/B2/{key}/slots_train.csv"
    S = states(tok, key, slot)
    n_ref = sum(1 for r in csv.DictReader(open(SL)) if r["slot"] == str(slot))
    out = out / a.set
    print(f"fixed set {a.set}: {len(S)} states (B2 slot-7 rows: {n_ref}); tokens close={close} cont={cont}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(S)), key=lambda j: len(S[j]["ids"]))
    for spec in a.models:
        name, path = spec.split("=", 1)
        f = out / f"{name}.json"
        if f.exists():
            print(f"[skip] {name}")
            continue
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        vals = [None] * len(S)
        with torch.no_grad():
            for b in range(0, len(order), a.batch):
                js = order[b:b + a.batch]
                L = max(len(S[j]["ids"]) for j in js)
                ids = torch.full((len(js), L), tok.pad_token_id or 0, dtype=torch.long)
                att = torch.zeros((len(js), L), dtype=torch.long)
                for k, j in enumerate(js):
                    x = S[j]["ids"]
                    ids[k, L - len(x):] = torch.tensor(x)
                    att[k, L - len(x):] = 1
                pos = (att.cumsum(-1) - 1).clamp(min=0)
                logits = model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda()).logits[:, -1, :]
                lp = torch.log_softmax(logits.float(), dim=-1)
                for k, j in enumerate(js):
                    vals[j] = float(lp[k, close] - lp[k, cont])
        res = {"name": name, "path": path, "n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals),
               "values": vals}
        if a.check_against_b2 == name:
            ref = {int(r["i"]): float(r["loS"]) for r in csv.DictReader(open(SL)) if r["slot"] == str(slot)}
            d = [abs(vals[k] - ref[s["i"]]) for k, s in enumerate(S) if s["i"] in ref]
            res["check_vs_b2"] = {"n_matched": len(d), "median_abs_diff": statistics.median(d),
                                  "p95_abs_diff": sorted(d)[int(0.95 * (len(d) - 1))], "ok": statistics.median(d) < 0.05}
            print("check vs B2:", res["check_vs_b2"], flush=True)
        f.write_text(json.dumps(res))
        print(f"[done] {name}: mean {res['mean']:.3f} median {res['median']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
