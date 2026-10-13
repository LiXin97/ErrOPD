#!/usr/bin/env python3
"""Check reranking prompt rendering."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "erropd_t1"

def rows(name):
    return [json.loads(l) for l in open(DATA / name)]

def prompt_msgs(r):
    return [{"role": m["role"], "content": m["content"]} for m in r["messages"] if m["role"] in ("system", "user")]

def ref_ids(tok, r):
    txt = tok.apply_chat_template(prompt_msgs(r), add_generation_prompt=True, tokenize=False)
    return tok(txt, add_special_tokens=False)["input_ids"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", default="/path/to/models/Qwen3-1.7B")
    ap.add_argument("--tok-dir", default="")
    ap.add_argument("--lf", action="store_true")
    ap.add_argument("--lf-src", default="/path/to/LlamaFactory/src")
    ap.add_argument("--verl", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    stock = AutoTokenizer.from_pretrained(args.stock)
    sets = {"train": rows("train.jsonl"), "val": rows("val.jsonl"), "dl_windows": rows("dl_windows.jsonl")}
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}
    ref = {k: [ref_ids(stock, r) for r in v] for k, v in sets.items()}
    tail = stock.decode(ref["train"][0][-4:])
    assert tail.endswith("<|im_start|>assistant\n") and "<think>" not in tail, repr(tail)
    bad_total = 0

    if args.tok_dir:
        t = AutoTokenizer.from_pretrained(args.tok_dir)
        bad = n = 0
        for k, v in sets.items():
            for r, ids in zip(v, ref[k]):
                n += 1
                bad += int(ref_ids(t, r) != ids)
        jinja = Path(args.tok_dir) / "chat_template.jinja"
        print(f"[t1_check] tok-dir {args.tok_dir}: rows={n} prompt-id mismatches vs stock={bad}; "
              f"chat_template.jinja present={jinja.exists()}")
        bad_total += bad

    if args.lf:
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import types

        class _Any(types.ModuleType):
            def __getattr__(self, k):
                if k.startswith("__"):
                    raise AttributeError(k)
                return type(k, (), {})

        class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            names = ("peft", "trl", "datasets", "accelerate")

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

        sys.meta_path.insert(0, _Finder())
        os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
        try:
            import llamafactory
        except ImportError:
            sys.path.insert(0, args.lf_src)
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments
        lf_tok = AutoTokenizer.from_pretrained(args.stock)
        tmpl = get_template_and_fix_tokenizer(lf_tok, DataArguments(template="qwen3_nothink"))
        bad_p = bad_r = n = 0
        for k in ("train", "val"):
            for r, ids in zip(sets[k], ref[k]):
                msgs = r["messages"]
                conv = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
                p_ids, r_ids = tmpl.encode_oneturn(lf_tok, conv, msgs[0]["content"])
                want_r = stock(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
                n += 1
                bad_p += int(p_ids != ids)
                bad_r += int(r_ids != want_r)
        print(f"[t1_check] LlamaFactory qwen3_nothink encode_oneturn: rows={n} prompt mismatches={bad_p} "
              f"response mismatches={bad_r} (lf eos={lf_tok.eos_token})")
        bad_total += bad_p + bad_r

    if args.verl:
        from omegaconf import OmegaConf
        from verl.utils.dataset.rl_dataset import RLHFDataset
        tok = AutoTokenizer.from_pretrained(args.tok_dir or args.stock)
        cfg = OmegaConf.create({"prompt_key": "prompt", "max_prompt_length": 4096, "return_raw_chat": True,
                                "truncation": "error", "filter_overlong_prompts": True, "cache_dir": str(
                                    Path(os.environ.get("ERROPD_RUN_TMP", "/tmp")) / "t1_rlhf_cache")})
        bad = n = 0
        for k in ("train", "val"):
            ds = RLHFDataset(data_files=str(DATA / "parquet" / f"{k}.parquet"), tokenizer=tok, config=cfg)
            assert len(ds) == len(sets[k]) or args.limit, (k, len(ds))
            for i in range(min(len(ds), len(ref[k]))):
                item = ds[i]
                am = item["attention_mask"].bool()
                ids_in = item["input_ids"][am].tolist()
                n += 1
                bad += int(list(item["raw_prompt_ids"]) != ref[k][i] or ids_in != ref[k][i])
        print(f"[t1_check] verl RLHFDataset (no chat-template kwargs): rows={n} mismatches={bad}")
        bad_total += bad

    print(f"[t1_check] TOTAL mismatches={bad_total}")
    return 1 if bad_total else 0

if __name__ == "__main__":
    sys.exit(main())
