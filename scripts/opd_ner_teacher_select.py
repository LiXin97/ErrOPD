#!/usr/bin/env python3
"""Teacher selection rule for the entity-extraction lambda grid."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
T3 = REPO / "outputs/T3"
GAP_MIN = 0.010
LIFT_FRAC = 0.25

def useful(name):
    p = T3 / "eval" / name / "t3_metrics.json"
    return json.load(open(p))["useful"] if p.exists() else None

def a1():
    s, t4, t8 = useful("sft_student_ep5"), useful("sft_teacher_ep5"), useful("sft_teacher8b_ep5")
    if s is None or t4 is None:
        print("[opd_ner_teacher_select] a1: SFT evals missing", file=sys.stderr)
        return 5, None
    rec = {"rule": f"gap = USEFUL F1(teacher ep5) - USEFUL F1(student ep5) >= {GAP_MIN}; else Qwen3-8B; else larger gap, flagged",
           "useful_student": s, "useful_teacher4b": t4, "gap4b": t4 - s, "useful_teacher8b": t8,
           "gap8b": (t8 - s) if t8 is not None else None, "small_gap": False}
    if t4 - s >= GAP_MIN:
        rec.update(teacher="outputs/T3/sft_qwen3_4b", teacher_tag="qwen3-4b", gap=t4 - s, status="OK")
    elif t8 is None:
        rec.update(status="NEED_8B", teacher=None)
        json.dump(rec, open(T3 / "stageA_a1.json", "w"), indent=1)
        print("[opd_ner_teacher_select] a1: gap below threshold; Qwen3-8B SFT required", file=sys.stderr)
        return 6, rec
    elif t8 - s >= GAP_MIN:
        rec.update(teacher="outputs/T3/sft_qwen3_8b", teacher_tag="qwen3-8b", gap=t8 - s, status="OK")
    else:
        big8 = (t8 - s) > (t4 - s)
        rec.update(teacher="outputs/T3/sft_qwen3_8b" if big8 else "outputs/T3/sft_qwen3_4b",
                   teacher_tag="qwen3-8b" if big8 else "qwen3-4b", gap=max(t8, t4) - s, status="OK", small_gap=True)
    json.dump(rec, open(T3 / "stageA_a1.json", "w"), indent=1)
    return 0, rec

def a3():
    rc, rec = a1()
    if rc:
        return rc
    p = T3 / "opd/lam1.00_s42/eval/step42/t3_metrics.json"
    if not p.exists():
        print("[opd_ner_teacher_select] a3: lambda=1.00 seed-42 step-42 eval missing", file=sys.stderr)
        return 5
    u = json.load(open(p))["useful"]
    need = rec["useful_student"] + LIFT_FRAC * rec["gap"]
    out = {"rule": f"USEFUL F1(lam=1.00, s42, step 42) >= USEFUL F1(student SFT) + {LIFT_FRAC} x gap",
           "useful_opd": u, "useful_student": rec["useful_student"], "gap": rec["gap"], "threshold": need,
           "pass": u >= need, "teacher": rec["teacher"], "small_gap": rec["small_gap"]}
    json.dump(out, open(T3 / "stageA_a3.json", "w"), indent=1)
    print("[opd_ner_teacher_select] a3:", json.dumps(out))
    return 0 if out["pass"] else 7

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "a1":
        rc, rec = a1()
        if rc == 0:
            print(rec["teacher"])
        sys.exit(rc)
    if cmd == "a3":
        sys.exit(a3())
    sys.exit("usage: opd_ner_teacher_select.py a1|a3")
