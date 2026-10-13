#!/usr/bin/env python3
"""Check Qwen3.5 rendering consistency."""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("tok_dir")
ap.add_argument("base_dir")
ap.add_argument("--out", default=None)
ap.add_argument("--data", default="data/pl_k8/val.jsonl,data/pl_k8/train.jsonl")
args = ap.parse_args()
sys.path.insert(0, os.environ.get("Q35_LF_SRC", "/path/to/LlamaFactory/src"))
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

from transformers import AutoProcessor, AutoTokenizer

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

lf_tok = AutoTokenizer.from_pretrained(args.base_dir)
tmpl = get_template_and_fix_tokenizer(lf_tok, DataArguments(template="qwen3_5_nothink"))
ours = AutoTokenizer.from_pretrained(args.tok_dir)
proc = AutoProcessor.from_pretrained(args.tok_dir)
rep = {"tok_dir": args.tok_dir, "base_dir": args.base_dir, "lf_eos_after_fix": lf_tok.eos_token,
       "ours_eos": ours.eos_token, "ours_bos": ours.bos_token, "pad": ours.pad_token}

term = {}
for s in ["},", "}]", "}, ", "]", "[", "\n"]:
    ids = ours(s, add_special_tokens=False).input_ids
    term[repr(s)] = ids
rep["terminator_ids"] = term
im_end = ours.convert_tokens_to_ids("<|im_end|>")
rep["im_end_id"] = im_end

n = 0
mism = {"V1": 0, "V2": 0, "R": 0, "E": 0, "E_think_kw": 0, "resp": 0}
examples = {}
n_resp_terms = {"},": 0, "}]": 0}
for path in args.data.split(","):
    for line in open(path):
        r = json.loads(line)
        msgs = r["messages"]
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
        conv = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
        p_lf, r_lf = tmpl.encode_oneturn(lf_tok, conv, system)
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        v1 = ours.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(v1, dict) or hasattr(v1, "keys"):
            v1 = v1["input_ids"]
        text = ours.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
        v2_text = proc.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
        v2 = proc(text=[v2_text], return_tensors="pt")["input_ids"][0].tolist()
        rr = ours(text, add_special_tokens=False).input_ids
        ee = ours(text).input_ids
        text_kw = ours.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ee_kw = ours(text_kw).input_ids
        gen = ours(msgs[-1]["content"], add_special_tokens=False).input_ids + [im_end]
        resp_ok = r_lf[: len(gen)] == gen and ours.decode(r_lf[len(gen):]) == "\n"
        for k, ids in (("V1", v1), ("V2", v2), ("R", rr), ("E", ee), ("E_think_kw", ee_kw)):
            if list(ids) != list(p_lf):
                mism[k] += 1
                examples.setdefault(k, {"row": n, "lf_tail": ours.decode(p_lf[-8:]), "ours_tail": ours.decode(list(ids)[-8:])})
        if not resp_ok:
            mism["resp"] += 1
            examples.setdefault("resp", {"row": n, "lf_resp_tail": r_lf[-4:], "gen_tail": gen[-4:]})
        toks = ours.convert_ids_to_tokens(gen)
        n_resp_terms["},"] += sum(1 for t in toks if ours.convert_tokens_to_string([t]) == "},")
        n_resp_terms["}]"] += sum(1 for t in toks if ours.convert_tokens_to_string([t]) == "}]")
        if n == 0:
            rep["prompt_tail_text"] = text[-60:]
            rep["gold_response_head_tokens"] = [ours.convert_tokens_to_string([t]) for t in toks[:24]]
            rep["gold_response_tail_tokens"] = [ours.convert_tokens_to_string([t]) for t in toks[-8:]]
        n += 1
rep["rows"] = n
rep["mismatches"] = mism
rep["examples"] = examples
rep["gold_terminator_token_counts"] = n_resp_terms
rep["PASS"] = all(v == 0 for v in mism.values())
print(json.dumps(rep, indent=1, ensure_ascii=False))
if args.out:
    json.dump(rep, open(args.out, "w"), indent=1, ensure_ascii=False)
sys.exit(0 if rep["PASS"] else 1)
