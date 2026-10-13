#!/usr/bin/env python3
"""Check Spider prompt rendering."""
import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

class _Any(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {})

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
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

REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--skip-verl", action="store_true")
    args = ap.parse_args()
    sys.meta_path.insert(0, _Finder())
    sys.path.insert(0, os.environ.get("T2_LF_SRC", "/path/to/LlamaFactory/src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    import eval_sql

    lf_tok = AutoTokenizer.from_pretrained(args.base_dir)
    tmpl = get_template_and_fix_tokenizer(lf_tok, DataArguments(template="qwen3_nothink"))
    ours = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    res = {}
    lf_prompt = {}
    for name in ("sft_train.jsonl", "val.jsonl", "opd_disjoint_prompts.jsonl"):
        n = bad_p = bad_r = 0
        for line in open(REPO / "data/erropd_t2" / name):
            r = json.loads(line)
            msgs = r["messages"]
            system = msgs[0]["content"]
            conv = [{"role": m["role"], "content": m["content"]} for m in msgs[1:]]
            p_ids, r_ids = tmpl.encode_oneturn(lf_tok, conv, system)
            e_ids = eval_sql.render_ids(ours, msgs)
            lf_prompt[r["metadata"]["id"]] = p_ids
            n += 1
            bad_p += int(p_ids != e_ids)
            bad_r += int(r_ids != ours(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False).input_ids)
        res[name] = {"rows": n, "prompt_mismatch_lf_vs_eval": bad_p, "response_mismatch": bad_r}
    if not args.skip_verl:
        from omegaconf import OmegaConf
        from verl.utils.dataset.rl_dataset import RLHFDataset
        cfg = OmegaConf.create({"prompt_key": "prompt", "max_prompt_length": 2048, "return_raw_chat": True,
                                "truncation": "error", "filter_overlong_prompts": False,
                                "cache_dir": os.environ.get("ERROPD_RUN_TMP", "/tmp") + "/t2_verl_cache"})
        for pool in ("parquet_sftpool", "parquet_disjoint"):
            for split in ("train", "val"):
                ds = RLHFDataset(data_files=[str(REPO / "data/erropd_t2" / pool / f"{split}.parquet")], tokenizer=ours,
                                 config=cfg)
                n = bad = 0
                for i in range(len(ds)):
                    item = ds[i]
                    rid = item["extra_info"]["id"] if "extra_info" in item else None
                    n += 1
                    bad += int(list(item["raw_prompt_ids"]) != lf_prompt[rid])
                res[f"{pool}/{split}"] = {"rows": n, "prompt_mismatch_verl_vs_lf": bad}
    tot = sum(v.get("prompt_mismatch_lf_vs_eval", 0) + v.get("response_mismatch", 0) +
              v.get("prompt_mismatch_verl_vs_lf", 0) for v in res.values())
    print(json.dumps({"tokenizer_dir": args.tokenizer_dir, "base_dir": args.base_dir, "checks": res,
                      "total_mismatches": tot}, indent=1))
    sys.exit(0 if tot == 0 else 1)

if __name__ == "__main__":
    main()
