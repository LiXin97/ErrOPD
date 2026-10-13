#!/usr/bin/env python3
"""SQL prompt-pool selection rule."""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SA = REPO / "outputs/T2/stageA"
THR = 0.010

def useful(role):
    return json.load(open(REPO / f"outputs/T2/eval/sft_{role}/sql_metrics.json"))["useful"]

def teacher():
    path = SA / "teacher_choice.json"
    if not path.exists():
        us, ut = useful("student"), useful("teacher")
        gap4 = ut - us
        rec = {"rule": "A1: teacher = Qwen3-4B SFT iff USEFUL gap >= 0.010; else Qwen3-8B SFT iff its gap >= 0.010; "
                       "else the larger gap, flagged small-gap", "useful_student": us, "useful_4b": ut, "gap_4b": gap4}
        if gap4 >= THR:
            rec.update(teacher="outputs/T2/sft_qwen3_4b", tag="qwen3-4b", gap=gap4, flag="")
        else:
            if not (REPO / "outputs/T2/eval/sft_teacher8b/sql_metrics.json").exists():
                print(f"A1: gap_4b={gap4:.4f} < {THR} and no Qwen3-8B SFT eval yet -> run T2_sft_teacher8b",
                      file=sys.stderr)
                sys.exit(6)
            gap8 = useful("teacher8b") - us
            rec.update(useful_8b=gap8 + us, gap_8b=gap8)
            if gap8 >= THR or gap8 > gap4:
                rec.update(teacher="outputs/T2/sft_qwen3_8b", tag="qwen3-8b", gap=gap8,
                           flag="" if gap8 >= THR else "small-gap")
            else:
                rec.update(teacher="outputs/T2/sft_qwen3_4b", tag="qwen3-4b", gap=gap4, flag="small-gap")
        SA.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp%d" % os.getpid())
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        if not path.exists():
            os.replace(tmp, path)
        else:
            tmp.unlink()
    rec = json.load(open(path))
    print(rec["teacher"], rec["tag"])

def pool():
    path = SA / "pool_decision.json"
    if not path.exists():
        print("A0 pool decision missing (T2_gen_base not finished)", file=sys.stderr)
        sys.exit(5)
    print(json.load(open(path))["pool"])

def gate():
    t = json.load(open(SA / "teacher_choice.json"))
    f = REPO / "outputs/T2/opd/lam1.00_s42/eval_step42/sql_metrics.json"
    if not f.exists():
        print("A3 gate not evaluated yet (lam1.00_s42 step 42 missing)", file=sys.stderr)
        sys.exit(5)
    u = json.load(open(f))["useful"]
    need = t["useful_student"] + 0.25 * t["gap"]
    ok = u >= need
    print(f"A3: USEFUL(lam1.00_s42, step 42)={u:.4f} need >= {need:.4f} "
          f"(student {t['useful_student']:.4f} + 0.25 x gap {t['gap']:.4f}) -> {'PASS' if ok else 'FAIL'}",
          file=sys.stderr)
    sys.exit(0 if ok else 7)

if __name__ == "__main__":
    {"teacher": teacher, "pool": pool, "gate": gate}[sys.argv[1]]()
