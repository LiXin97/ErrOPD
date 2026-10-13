#!/usr/bin/env python3
"""Llama rendering shared by training, rollout and evaluation."""
import argparse
import json
import os
import sys

BOS = "<|begin_of_text|>"
EOT = "<|eot_id|>"

LF_LLAMA3_JINJA = (
    "{{- '<|begin_of_text|>' }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{%- if message['content'] %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- endif %}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- if loop.last and add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- endif %}"
    "{%- endfor %}"
)

def install(model_dir, base_dir):
    with open(os.path.join(base_dir, "tokenizer_config.json")) as f:
        cfg = json.load(f)
    cfg["chat_template"] = LF_LLAMA3_JINJA
    cfg["eos_token"] = EOT
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(model_dir, "chat_template.jinja"), "w") as f:
        f.write(LF_LLAMA3_JINJA)
    print(f"[xfamily_template] installed LF-llama3 template into {model_dir} (base config from {base_dir})")

def lf_prompt_ids(tok, msgs):
    pass
    ids = [tok.bos_token_id]
    sys_msg = [m for m in msgs if m["role"] == "system"]
    user = [m for m in msgs if m["role"] == "user"]
    assert len(user) == 1 and len(sys_msg) <= 1
    if sys_msg and sys_msg[0]["content"]:
        ids += tok.encode(
            f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg[0]['content']}{EOT}", add_special_tokens=False
        )
    ids += tok.encode(
        f"<|start_header_id|>user<|end_header_id|>\n\n{user[0]['content']}{EOT}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    return ids

def verify(model_dir, data_files, limit):
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    bos_id = tok.convert_tokens_to_ids(BOS)
    eot_id = tok.convert_tokens_to_ids(EOT)
    problems = []
    if tok.eos_token != EOT:
        problems.append(f"eos_token={tok.eos_token!r} (want {EOT})")
    n = 0
    for path in data_files:
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        for i, r in enumerate(rows):
            msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
            ref = lf_prompt_ids(tok, msgs)
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            a = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if isinstance(a, dict) or hasattr(a, "input_ids"):
                a = a["input_ids"]
            c = tok(text, add_special_tokens=False).input_ids
            if "Cutting Knowledge" in text or "Today Date" in text:
                problems.append(f"{path}:{i} date header present")
            if c.count(bos_id) != 1 or c[0] != bos_id:
                problems.append(f"{path}:{i} bos count={c.count(bos_id)}")
            if not (list(a) == c == ref):
                problems.append(f"{path}:{i} id mismatch len(a)={len(a)} len(c)={len(c)} len(ref)={len(ref)}")
            n += 1
            if len(problems) > 20:
                break
    print(f"[xfamily_template] verify {model_dir}: transformers={transformers.__version__} rows={n} "
          f"eos={tok.eos_token}({tok.eos_token_id}) eot_id={eot_id} pad={tok.pad_token} problems={len(problems)}")
    for p in problems[:20]:
        print("  PROBLEM", p)
    return 0 if not problems else 1

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("install")
    a.add_argument("--model-dir", required=True)
    a.add_argument("--base-dir", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--model-dir", required=True)
    v.add_argument("--data", nargs="+", default=["data/pl_k8/val.jsonl"])
    v.add_argument("--limit", type=int, default=0)
    sub.add_parser("print-jinja")
    args = ap.parse_args()
    if args.cmd == "install":
        install(args.model_dir, args.base_dir)
    elif args.cmd == "verify":
        sys.exit(verify(args.model_dir, args.data, args.limit))
    else:
        print(LF_LLAMA3_JINJA)

if __name__ == "__main__":
    main()
