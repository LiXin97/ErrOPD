#!/usr/bin/env python3
"""Reference sampling for Qwen3.5 and Olmo 3 teachers and students."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_asym_stat as S
import eval_reference_analyze as A

PAIRS = {"4b": ("d6_base_2b", "d6_self_4b", "d6_fb_4b"), "9b": ("d6_base_2b_T9b", "d6_self_9b", "d6_fb_9b")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train")
    ap.add_argument("--pairs", default="4b,9b")
    args = ap.parse_args()
    reg = json.load(open(REPO / "scripts/qwen35_b2_registry.json"))
    from transformers import AutoTokenizer

    out_dir = REPO / "outputs/D6/asym"
    out_dir.mkdir(parents=True, exist_ok=True)
    toks = {}
    for pair in args.pairs.split(","):
        for key in PAIRS[pair]:
            d = REPO / "outputs/B2" / key
            if (d / f"rollouts_{args.mode}.labels.csv").exists():
                continue
            ent = reg[key]
            if not ent.get("gen_only") and not (d / f"scores_{args.mode}.pt.meta.json").exists():
                print(f"[skip] {key}: no scores yet", flush=True)
                continue
            tp = str(REPO / ent["base"])
            if tp not in toks:
                t = AutoTokenizer.from_pretrained(tp)
                toks[tp] = (t, A.TokBytes(t))
            tok, tb = toks[tp]
            o = A.analyze(key, args.mode, reg, tok, tb)
            print(json.dumps({"key": key, "labels": o.get("labels")}, default=float), flush=True)
    V = S.VALIDATORS["listwise"]
    for pair in args.pairs.split(","):
        pk, self_key, fb_key = PAIRS[pair]
        rT = S.first_violation_rates(self_key, args.mode)
        rB = S.first_violation_rates(reg[pk].get("rollouts_from") or pk, args.mode)
        res = {"pair": pair, "primary": S.measure_v2(pk, args.mode, reg, V), "r_T": rT, "r_B": rB}
        tt = [ty for ty, r in rT["rates"].items() if ty != "valid" and r >= S.R_MIN]
        need = [ty for ty in tt if res["primary"]["types"].get(ty, {}).get("n_prompts", 0) < S.N_MIN]
        res["fallback_types"] = need
        merged = {ty: dict(v, source="primary") for ty, v in res["primary"]["types"].items()}
        res["fallback"] = S.measure_v2(fb_key, args.mode, reg, V)
        for ty in need:
            if ty in res["fallback"]["types"]:
                merged[ty] = dict(res["fallback"]["types"][ty], source="fallback")
        res["merged_types"] = merged
        res["M_prediction_by_type"], res["M_prediction_task"] = S.decide_v2(merged, rT["rates"], rB["rates"])
        res["code"] = {"eval_asym_stat": "measure_v2/decide_v2 unchanged", "registry": "scripts/qwen35_b2_registry.json"}
        (out_dir / f"asym_v2_{pair}.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({"pair": pair, "fallback_types": need, "task": res["M_prediction_task"],
                          "by_type": res["M_prediction_by_type"], "r_T": rT["rates"], "r_B": rB["rates"]},
                         indent=1, default=float), flush=True)

if __name__ == "__main__":
    main()
