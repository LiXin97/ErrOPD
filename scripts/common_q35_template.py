#!/usr/bin/env python3
"""Install the Qwen3.5 chat template."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, "common_q35_chat_template.jinja")).read()
SHA = hashlib.sha256(TPL.encode()).hexdigest()

def install(d):
    with open(os.path.join(d, "chat_template.jinja"), "w") as f:
        f.write(TPL)
    tc = os.path.join(d, "tokenizer_config.json")
    if os.path.exists(tc):
        cfg = json.load(open(tc))
        if "chat_template" in cfg:
            cfg["chat_template"] = TPL
            json.dump(cfg, open(tc, "w"), indent=2, ensure_ascii=False)
    pj = os.path.join(d, "chat_template.json")
    if os.path.exists(pj):
        os.remove(pj)
    with open(os.path.join(d, ".d6_template_sha256"), "w") as f:
        f.write(SHA + "\n")
    print(f"installed D6 template sha256={SHA[:16]} into {d}")

if __name__ == "__main__":
    for d in sys.argv[1:]:
        install(d)
