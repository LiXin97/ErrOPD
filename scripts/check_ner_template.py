#!/usr/bin/env python3
"""Check CoNLL prompt rendering."""
import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

class _Any(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {})

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    pass
    names = ("peft", "trl", "accelerate")

    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root in self.names:
            try:
                sys.meta_path.remove(self)
                ok = importlib.util.find_spec(root) is not None
            finally:
                sys.meta_path.insert(0, self)
            if not ok:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Any(spec.name)

    def exec_module(self, m):
        m.__path__ = []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", required=True, help="model dir whose tokenizer the rollout/eval use")
    ap.add_argument("--base-dir", required=True, help="base model dir (the tokenizer LF loads at SFT time)")
    ap.add_argument("--jsonl", nargs="+", default=[str(REPO / "data/erropd_t3/train.jsonl"),
                                                   str(REPO / "data/erropd_t3/test.jsonl")])
    ap.add_argument("--parquet", default=str(REPO / "data/erropd_t3/parquet_verl/train.parquet"))
    ap.add_argument("--lf-src", default=os.environ.get("T3_LF_SRC",
                                                       "/path/to/LlamaFactory/src"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "scripts"))
    import eval_ner

    ours = AutoTokenizer.from_pretrained(args.tok_dir)
    res = {"tok_dir": args.tok_dir, "base_dir": args.base_dir}

    try:
        import llamafactory
    except ImportError:
        sys.meta_path.insert(0, _Finder())
        sys.path.insert(0, args.lf_src)
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    lf_tok = AutoTokenizer.from_pretrained(args.base_dir)
    tmpl = get_template_and_fix_tokenizer(lf_tok, DataArguments(template="qwen3_nothink"))
    end_ids = lf_tok.encode("<|im_end|>\n", add_special_tokens=False)

    rows = [json.loads(l) for p in args.jsonl for l in open(p)]
    eval_ids = eval_ner.prompt_ids(ours, rows)
    n = bad_prompt = bad_resp = 0
    first_bad = None
    for r, e_ids in zip(rows, eval_ids):
        msgs = r["messages"]
        conv = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
        p_ids, r_ids = tmpl.encode_oneturn(lf_tok, conv, msgs[0]["content"])
        n += 1
        if p_ids != e_ids:
            bad_prompt += 1
            first_bad = first_bad or (r["id"], "prompt")
        if r_ids != ours.encode(msgs[-1]["content"], add_special_tokens=False) + end_ids:
            bad_resp += 1
            first_bad = first_bad or (r["id"], "response", ours.decode(r_ids[-5:]))
    res.update({"rows_lf_vs_eval": n, "prompt_mismatch_lf_vs_eval": bad_prompt, "response_mismatch_lf": bad_resp})

    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset
    cfg = OmegaConf.create({"max_prompt_length": 2048, "return_raw_chat": True, "truncation": "error",
                            "filter_overlong_prompts": False, "prompt_key": "prompt",
                            "cache_dir": os.path.join(os.environ.get("ERROPD_RUN_TMP", "/tmp"), "t3_verl_cache")})
    ds = RLHFDataset(args.parquet, ours, cfg)
    by_id = {r["id"]: ids for r, ids in zip(rows, eval_ids)}
    nv = bad_v = 0
    for i in range(len(ds)):
        item = ds[i]
        rid = ds.dataframe[i]["id"]
        want = by_id.get(rid)
        if want is None:
            continue
        am = item["attention_mask"].bool()
        unpadded = item["input_ids"][am].tolist()
        nv += 1
        if item["raw_prompt_ids"] != want or unpadded != want:
            bad_v += 1
            first_bad = first_bad or (rid, "verl")
    res.update({"rows_verl_vs_eval": nv, "prompt_mismatch_verl_vs_eval": bad_v, "first_bad": first_bad,
                "ok": bad_prompt == 0 and bad_resp == 0 and bad_v == 0 and nv > 0})
    print("[check_ner_template]", json.dumps(res))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    return 0 if res["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
