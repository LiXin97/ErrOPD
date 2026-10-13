#!/usr/bin/env python3
"""Probe at the SQL trailing-text decision."""
import argparse
import copy
import datetime
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import common_nonranking as nr

STUDENT = "outputs/T2/sft_qwen3_1.7b"
SEEDS = (6001, 6002, 6003, 6004, 6005)
GRID_ARMS = ("opd_lam1.00", "opd_lam1.40", "fkl_lam1.00")
NL, IM_END, EOT = 198, 151645, 151643

def grid_models():
    out = [("student", STUDENT)]
    for arm in GRID_ARMS:
        for s in SEEDS:
            rid = f"SQLF_t2_trail_{arm}_s{s}"
            out.append((f"{rid}_step42", f"outputs/SQLF/{rid}/ckpt/global_step_42/hf"))
    return out

def probe_states(tok):
    import eval_sql_states as scan
    a = argparse.Namespace(states="own", rollouts="outputs/T2/stageA/gen_base/rollouts_train.jsonl",
                           split="disjoint", n=128)
    st = [(k, p, r) for k, p, r in scan.states(a, tok) if k["gold"] and k["gold"][-1] in nr._ALNUM]
    return st

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="NAME=MODEL_DIR")
    ap.add_argument("--grid", action="store_true", help="the student and the 15 SQLF grid checkpoints at step 42")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    models = grid_models() if a.grid else [tuple(x.split("=", 1)) for x in a.models]
    assert models, "no models"
    out = REPO / a.out if not a.out.startswith("/") else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, path in models:
        p = REPO / path if not path.startswith("/") else Path(path)
        assert (p / "config.json").exists(), f"missing {p}/config.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(REPO / STUDENT))
    V = len(tok)
    skip = [i for i in range(V) if i in (IM_END, EOT) or tok.decode([i]).strip() == ""]
    st = probe_states(tok)
    keys = [{"split": k["split"], "row": k["row"], "id": k["id"]} for k, _, _ in st]
    print(f"[sqlf_probe] {len(st)} eligible own end states, device {dev}, dtype {a.dtype}", flush=True)
    for name, path in models:
        f = out / f"{name}.json"
        if f.exists() and not a.force:
            print(f"[sqlf_probe] exists: {f}")
            continue
        p = REPO / path if not path.startswith("/") else Path(path)
        m = AutoModelForCausalLM.from_pretrained(str(p), dtype=getattr(torch, a.dtype)).to(dev)
        m.eval()
        vals, pnw = [], []
        mask = torch.ones(m.config.vocab_size, dtype=torch.bool, device=dev)
        mask[skip] = False
        mask[V:] = False
        with torch.no_grad():
            for _, pids, resp in st:
                ids = torch.tensor([pids + resp], device=dev)
                o = m(ids, use_cache=True, logits_to_keep=1)
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
                vals.append(float(lp[NL] - lp[IM_END]))
                o2 = m(torch.tensor([[NL]], device=dev), past_key_values=copy.deepcopy(o.past_key_values), use_cache=True)
                q = torch.softmax(o2.logits[0, -1].float(), -1)
                pnw.append(float(q[mask].sum()))
        rec = {"name": name, "path": str(path), "n": len(vals), "mean": statistics.mean(vals),
               "median": statistics.median(vals), "values": vals, "p_nonws_mean": statistics.mean(pnw),
               "p_nonws_median": statistics.median(pnw), "p_nonws_values": pnw, "states": keys,
               "statistic": "log p(\\n) - log p(<|im_end|>) at the own end state; p_nonws after a teacher-forced \\n",
               "device": dev, "dtype": a.dtype, "torch": torch.__version__, "transformers": transformers.__version__,
               "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.replace(f)
        print(f"[sqlf_probe] {name}: mean {rec['mean']:.3f} median {rec['median']:.3f} p_nonws {rec['p_nonws_mean']:.3f}",
              flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    if a.grid and all((out / f"{n}.json").exists() for n, _ in models):
        (out / "DONE_step42").write_text(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
        print("[sqlf_probe] DONE_step42")

if __name__ == "__main__":
    main()
