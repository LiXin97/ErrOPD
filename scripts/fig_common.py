#!/usr/bin/env python3
"""Shared plotting helpers."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "out" / "figures"
ERROPD_OUT = REPO / "outputs"
CACHE = RES / ".fig1_cache"
QUEUE = REPO / "erropd" / "queue"
PALETTE = ["#56B4E9", "#E69F00", "#555555", "#F0E442", "#CC79A7", "#009E73", "#CC79A7", "#D55E00"]
FAMILIES = ["natural", "template", "injected", "tasks"]
SHAPE = {"natural": "o", "template": "*", "injected": "D", "tasks": "s"}
COLOR = {"natural": PALETTE[0], "template": PALETTE[1], "injected": PALETTE[6], "tasks": PALETTE[2]}
MSIZE = {"natural": 5, "template": 8, "injected": 4.5, "tasks": 4.5}
FAM_LABEL = {"natural": "natural pairs", "template": "same models, rollout prompt on/off",
             "injected": "injection tests", "tasks": "other structured-output tasks"}
TEMPLATE_EXPS = ("A1R", "A1Rx", "M1R")
SHORT = {"k1_trunc_realids": "one-item-short", "other": "no list", "malformed_json": "malformed",
         "hallucinated_id": "halluc. id", "unterminated": "unclosed", "extra_items": "extra items",
         "duplicate_id": "dup. id", "short_list": "short list", "extra_text": "extra text", "none": "no error",
         "none positive": "none positive", "unknown_function": "unknown function", "unknown_table": "unknown table"}
X_DEF = ("teacher-minus-base rate of the teacher-specific error class (the validity class with the largest "
         "teacher-minus-base difference; 'none positive' when none is positive) on own samples at T = 1 under the "
         "rendering the OPD runs trained with")
Y_DEF = ("change in the student's eval strict-failure rate (any type; greedy; final step) from lambda 1.00 to 1.40; "
         "mean over seed pairs, 95% t-interval over seed pairs")
X_SOURCES = {
    ("fashion_k8", "qwen3-1.7b", "qwen3-4b", "true"):
        {"teacher": "B2/self_T4b_s42", "base": "B2/sft_s42", "rendering": "think-block OPD training prompt"},
    ("software_k8", "qwen3-1.7b", "qwen3-4b", "true"):
        {"teacher": "B2/self_software_T4b", "base": "B2/pair_software_4bT_1.7bB", "rendering": "think-block OPD training prompt",
         "data": ("data/pl_k8_software/train.jsonl", "data/pl_k8_software/val.jsonl")},
    ("fashion_k8", "qwen3-4b", "qwen3-8b", "true"):
        {"teacher": "B2/self_T8b_s42", "base": "B2/pair_fashion_8bT_4bB", "rendering": "think-block OPD training prompt"},
    ("fashion_k8", "qwen3-8b", "qwen3-32b", "true"):
        {"teacher": "B2/self_fashion_T32b", "base": "B2/pair_fashion_32bT_8bB", "rendering": "think-block OPD training prompt"},
    ("esci_k8", "qwen3-1.7b", "qwen3-4b", "false"):
        {"teacher": "B2/self_esci_T4b", "base": "B2/pair_esci_4bT_1.7bB", "rendering": "think-block OPD training prompt",
         "data": ("data/esci_k8/train.jsonl", "data/esci_k8/val.jsonl")},
    ("fashion_k8", "qwen3.5-2b", "qwen3.5-4b", "false"):
        {"teacher": "B2/d6_self_4b", "base": "B2/d6_base_2b", "rendering": "the single D6 prompt rendering"},
    ("fashion_k8", "llama3.2-3b", "llama3.1-8b", "false"):
        {"agg": ("outputs/D1/sampled_validity/teacher_ep5.json", "outputs/D1/sampled_validity/student_ep5.json"),
         "rendering": "D1 training prompt"},
}
TEMPLATE_PAIRS = [
    {"pair": "fashion 1.7bx4b think-block rollouts", "pretty": "Fashion 1.7B×4B, think-block rollout prompt (A1R+A1Rx)",
     "exps": ("A1R", "A1Rx"), "teacher": "B2/fR_self_T4b", "base": "B2/fR_pair_4bT_1.7bB",
     "rendering": "think-block OPD training prompt", "grids": ("A1R_training_rollouts.json", "A1Rx_grid.json")},
    {"pair": "fashion 1.7bx4b matched rollouts", "pretty": "Fashion 1.7B×4B, matched rollout prompt (M1R)",
     "exps": ("M1R",), "teacher": "B2/m1r_self_T4b", "base": "B2/m1r_pair",
     "rendering": "SFT/eval prompt (matched)", "grids": ("M1R_grid.json",)},
]
ARMS = [
    {"exp": "F3", "pair": "A", "pretty": "F3 pair A (ESCI, Qwen3): injected teacher, clean base",
     "teacher": "B2/f3self_t4b_k1", "base": "B2/f3_pairA", "rendering": "F3 single rendering (rollout = eval)",
     "call": "collapse", "call_src": " P1 (spec v2: one-item-short amplification, task collapse)"},
    {"exp": "F3", "pair": "S", "pretty": "F3 pair S (ESCI, Qwen3): injected teacher, base injected alike",
     "teacher": "B2/f3self_t4b_k1", "base": "B2/f3_pairS_f20", "rendering": "F3 single rendering (rollout = eval)",
     "call": "no collapse", "call_src": " P2 (|parse change| <= 0.10; spec v2: no prediction for one-item-short)"},
    {"exp": "F3", "pair": "0", "pretty": "F3 pair 0 (ESCI, Qwen3): clean teacher, clean base",
     "teacher": "B2/f3self_t4b_clean", "base": "B2/f3_pair0", "rendering": "F3 single rendering (rollout = eval)",
     "call": "no collapse", "call_src": " P0 (spec v2: task no collapse)"},
    {"exp": "D1inj", "pair": "A", "pretty": "D1-inj pair A (Fashion, Llama): injected teacher, clean base",
     "teacher": "D1inj/asym/tk1_self", "base": "D1inj/asym/base_roll", "rendering": "Llama single rendering",
     "call": "collapse", "call_src": " Amendment 1 (one-item-short amplification, task collapse)"},
    {"exp": "D1inj", "pair": "0", "pretty": "D1-inj pair 0 (Fashion, Llama): clean teacher, clean base",
     "teacher": "D1inj/asym/t0_self", "base": "D1inj/asym/base_roll", "rendering": "Llama single rendering",
     "call": "no collapse", "call_src": " Amendment 1 (task no collapse)"},
]
ARM_DIR = {"F3": re.compile(r"^outputs/F3/F3_(A|S|0)_lam([\d.]+)_s(\d+)/eval/step(\d+)$"),
           "D1inj": re.compile(r"^outputs/D1inj/opd/(A|0)_lam([\d.]+)_s(\d+)/eval_step(\d+)$")}
SPEC_RE = {"F3": re.compile(r"^09-F3_(A|S|0)_lam([\d.]+)_s(\d+)\.json$"),
           "D1inj": re.compile(r"^10-D1inj_(A|0)_lam([\d.]+)_s(\d+)\.json$")}
ARM_LAMS = (1.0, 1.25, 1.4)
ARM_COLOR = {"A": PALETTE[4], "S": PALETTE[1], "0": PALETTE[0]}
ARM_LS = {"F3": "-", "D1inj": (0, (3, 2))}
TASKS = [
    {"exp": "T1", "variant": "vanilla", "pretty": "T1 MS MARCO listwise 1.7B×4B", "file": "T1/asym/asymmetry.json",
     "rT": ("v2", "r_T"), "rB": ("v2", "r_B"), "n": ("teacher_rollouts", "base_rollouts"), "exclude": ()},
    {"exp": "T2", "variant": "vanilla", "pretty": "T2 Spider SQL 1.7B×4B", "file": "T2/stageA/asym.json",
     "rT": ("r_T",), "rB": ("r_B",), "n": (("n_rollouts_rates", "teacher"), ("n_rollouts_rates", "base")), "exclude": ()},
    {"exp": "T3", "variant": "vanilla", "pretty": "T3 CoNLL JSON 1.7B×4B", "file": "T3/asym/analysis/asymmetry.json",
     "per_type": True, "n": (("rates_all_T1_v2", "n_teacher"), ("rates_all_T1_v2", "n_base")), "exclude": ("early_close",)},
    {"exp": "T3b", "variant": "init_ep1", "pretty": "T3b CoNLL JSON, epoch-1 base", "file": "T3b/asym/analysis/asymmetry.json",
     "per_type": True, "n": (("rates_all_T1_v2", "n_teacher"), ("rates_all_T1_v2", "n_base")), "exclude": ("early_close",),
     "dagger": "the T3b lock scores step 14; step 42 shown for the common definition"},
]
NOTES = {
    ("F3", "S"):
        "its 'no collapse' marker is the F3 lock's P2 parse bound (|parse change| <= 0.10 at lambda 1.40); spec v2 "
        "itself made no per-type call for pair S (one-item-short D +0.19, below the 1.0 minimum effect).",
    ("fashion_k8", "qwen3-4b", "qwen3-8b", "true"):
        "y compares lambda 1.25 with 1.00 (the stack's only lambda > 1 run), one seed, RoPE-misread models. Spec v2's "
        "rendering-agreement rule gave this pair no task call (malformed JSON was called amplification in the rollout "
        "rendering only; one-item-short was called suppression in both), so y = 0 here is not a failed prediction. Its "
        "x moved from -0.24 (eval-rendering one-item-short) to +0.15 under the ruling.",
}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042}
PAGE_PT = {"figA_asymmetry": (450.92958, 255.13079), "figA_asymmetry_extra": (502.02388, 465.9436)}

def _pct_ticks(ax, *axes):
    pass
    from matplotlib.ticker import FuncFormatter
    fm = FuncFormatter(lambda v, _: f"{100 * v:.10g}".replace("-", "−"))
    for a in axes:
        (ax.xaxis if a == "x" else ax.yaxis).set_major_formatter(fm)

MIN_MARGIN_PT = 3.0

def _save_fixed(fig, stem):
    pass
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = (v / 72 for v in PAGE_PT[stem])
    mx, my = (w - tb.width) / 2 * 72, (h - tb.height) / 2 * 72
    assert min(mx, my) >= MIN_MARGIN_PT, (stem, "content", tb.width * 72, tb.height * 72, "page", PAGE_PT[stem])
    box = Bbox.from_bounds(tb.x0 - (w - tb.width) / 2, tb.y0 - (h - tb.height) / 2, w, h)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches=box)
    return mx, my

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)

def load():
    led = []
    for name in ("legacy_ledger.csv", "ledger.csv"):
        p = RES / name
        if p.exists():
            led += list(csv.DictReader(p.open()))
    b1 = {r["id"]: r for r in csv.DictReader((RES / "B1_failure_anatomy.csv").open())} if (RES / "B1_failure_anatomy.csv").exists() else {}
    return led, b1

def t_ci(d):
    if len(d) < 2:
        return None, None
    m, s = st.mean(d), st.stdev(d)
    df = len(d) - 1
    tq = T975.get(df) or (T975[30] if df > 30 else T975[max(k for k in T975 if k <= df)])
    h = tq * s / math.sqrt(len(d))
    return m - h, m + h

def arm_cells(led, b1):
    pass
    cells, seen = defaultdict(lambda: defaultdict(list)), set()
    for r in sorted(led, key=lambda r: r["id"]):
        for exp, rx in ARM_DIR.items():
            m_ = rx.match(r.get("eval_path") or "")
            if not m_ or m_.group(4) != "42":
                continue
            key = (exp, m_.group(1), round(float(m_.group(2)), 2), m_.group(3))
            b = b1.get(r["id"])
            nf = int((b or {}).get("n_feasible") or 0)
            if key in seen or not nf:
                continue
            seen.add(key)
            cells[(exp, m_.group(1))][key[2]].append((key[3], int(b["n_valid"]) / nf, r["id"]))
    expected = set()
    for sub in ("pending", "running", "done", "failed", "hold"):
        d = QUEUE / sub
        if d.exists():
            for fn in d.iterdir():
                for exp, rx in SPEC_RE.items():
                    m_ = rx.match(fn.name)
                    if m_:
                        expected.add((exp, m_.group(1), round(float(m_.group(2)), 2), m_.group(3)))
    for a in ARMS:
        for lam in ARM_LAMS:
            for sd in ("1001", "1002", "1003"):
                expected.add((a["exp"], a["pair"], lam, sd))
    missing = sorted(expected - seen, key=lambda k: (k[0], k[1], k[2], int(k[3])))
    return cells, expected, missing

def draw_arms(ax, cells):
    from matplotlib.lines import Line2D
    off = {("F3", "A"): -0.024, ("F3", "S"): -0.012, ("F3", "0"): 0.0, ("D1inj", "A"): 0.012, ("D1inj", "0"): 0.024}
    n_cells = sum(len(v) for arm in cells.values() for v in arm.values())
    for a in ARMS:
        key = (a["exp"], a["pair"])
        col, ls = ARM_COLOR[a["pair"]], ARM_LS[a["exp"]]
        mk, face = ("v", col) if a["call"] == "collapse" else ("o", "white")
        lams = [lam for lam in ARM_LAMS if cells.get(key, {}).get(lam)]
        xs, ms, los, his = [], [], [], []
        for lam in lams:
            v = [x for _, x, _ in cells[key][lam]]
            ax.scatter([lam + off[key]] * len(v), v, color=col, s=7, alpha=0.45, linewidths=0, zorder=2)
            lo, hi = t_ci(v)
            xs.append(lam + off[key])
            ms.append(st.mean(v))
            los.append(0 if lo is None else max(0.0, st.mean(v) - max(0.0, lo)))
            his.append(0 if hi is None else max(0.0, min(1.0, hi) - st.mean(v)))
        if xs:
            ax.errorbar(xs, ms, yerr=[los, his], color=col, linestyle=ls, marker=mk, markersize=4.5,
                        markerfacecolor=face, markeredgecolor=col, linewidth=1.2, elinewidth=0.8, capsize=1.5, zorder=3)
    if not n_cells:
        ax.text(0.5, 0.5, "F3 / D1-inj grids running\n(cells appear as they land)", ha="center", va="center",
                transform=ax.transAxes, color="#52514e")
    ax.set_xticks(ARM_LAMS)
    ax.set_xticklabels(["1.00", "1.25", "1.40"])
    ax.set_xlim(0.94, 1.46)
    ax.set_ylim(-0.03, 1.05)
    _pct_ticks(ax, "y")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("strict parse rate, feasible prompts\n(%; eval, step 42)")
    ax.set_title("(a) constructed pairs: account vs outcome", loc="left")
    g_ = "#52514e"
    h = [Line2D([], [], color=ARM_COLOR["A"], linewidth=2), Line2D([], [], color=ARM_COLOR["S"], linewidth=2),
         Line2D([], [], color=ARM_COLOR["0"], linewidth=2),
         Line2D([], [], color=g_, linestyle=ARM_LS["F3"], linewidth=1.2),
         Line2D([], [], color=g_, linestyle=ARM_LS["D1inj"], linewidth=1.2),
         Line2D([], [], color=g_, marker="v", linestyle="none", markersize=4.5),
         Line2D([], [], color=g_, marker="o", linestyle="none", markerfacecolor="white", markersize=4.5)]
    l = ["A: injected teacher, clean base", "S: both injected", "0: both clean", "Qwen3, ESCI",
         "Llama, Fashion", "asymmetry account: collapse", "asymmetry account: no collapse"]
    ax.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, handletextpad=0.4,
              handlelength=2.2)

def _label_jsonl(src: Path, data):
    pass
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_listwise_fashion import parse_score_list
    from eval_labels import classify, _strict_generic
    tr_p, va_p = data or ("data/pl_k8/train.jsonl", "data/pl_k8/val.jsonl")
    rows = {"train": [json.loads(l) for l in open(REPO / tr_p)], "val": [json.loads(l) for l in open(REPO / va_p)]}
    out = []
    for line in open(src):
        r = json.loads(line)
        md = rows["train" if r["split"] == "train" else "val"][int(r["row"])]["metadata"]
        fld = next(x for x in ("review_ids", "product_ids", "passage_ids") if x in md)
        id_key, ids, K = fld[:-1], [str(x) for x in md[fld]], md["K"]
        if len(set(ids)) != len(ids):
            lab = "infeasible_input"
        else:
            ok = (parse_score_list(r["text"], K, ids) is not None) if id_key == "review_id" else \
                bool(_strict_generic(r["text"], K, ids, id_key))
            lab = classify(r["text"], ids, K, id_key, strict_ok=ok)["label"]
        out.append((r["split"], str(r["row"]), lab))
    return out

def own_samples(key, data=None):
    pass
    d = ERROPD_OUT / key
    p = d / "rollouts_train.labels.csv"
    if p.exists():
        s = [((r["split"], r["row"]), r["label"]) for r in csv.DictReader(p.open())]
        note = rel(p)
    else:
        src = d / "rollouts_train.jsonl"
        if not src.exists():
            return None, f"{rel(src)} missing"
        CACHE.mkdir(parents=True, exist_ok=True)
        cf = CACHE / f"{key.replace('/', '__')}.train.labels.csv"
        if not (cf.exists() and cf.stat().st_mtime >= src.stat().st_mtime):
            with cf.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["split", "row", "label"])
                w.writerows(_label_jsonl(src, data))
        s = [((r["split"], r["row"]), r["label"]) for r in csv.DictReader(cf.open())]
        note = f"{rel(src)} (labelled here with the B1 validator, as eval_reference_analyze does; cache {rel(cf)})"
    return [(c, l) for c, l in s if l != "infeasible_input"], note

def _pick(diffs):
    pass
    if not diffs or max(diffs.values()) <= 0:
        return None
    top = max(diffs.values())
    return "/".join(sorted(c for c, v in diffs.items() if abs(v - top) < 1e-12))

def x_from_samples(T, Bs, nboot=4000, seed=0):
    def rate(S, c):
        return sum(l == c for _, l in S) / len(S)
    allc = {c: (rate(T, c), rate(Bs, c)) for c in sorted({l for _, l in T + Bs if l != "valid"})}
    diffs = {c: a - b for c, (a, b) in allc.items() if a > 0}
    cls = _pick(diffs)
    base = {"nT": len(T), "nB": len(Bs), "clusters": len({c for c, _ in T}), "all": allc}
    if cls is None:
        return {**base, "cls": "none positive" if allc else "none", "x": 0.0, "x_lo": None, "x_hi": None, "rT": None,
                "rB": None, "ci": "no class with a positive teacher-minus-base difference"}
    c0 = cls.split("/")[0]
    by = defaultdict(lambda: [0, 0, 0, 0])
    for c, l in T:
        by[c][0] += l == c0
        by[c][1] += 1
    for c, l in Bs:
        by[c][2] += l == c0
        by[c][3] += 1
    keys = sorted(by)
    rng = random.Random(seed)
    bs = []
    for _ in range(nboot):
        a = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        nt, nb = sum(v[1] for v in a), sum(v[3] for v in a)
        if nt and nb:
            bs.append(sum(v[0] for v in a) / nt - sum(v[2] for v in a) / nb)
    bs.sort()
    return {**base, "cls": cls, "x": diffs[c0], "x_lo": bs[int(0.025 * len(bs))], "x_hi": bs[int(0.975 * len(bs)) - 1],
            "rT": allc[c0][0], "rB": allc[c0][1], "ci": "prompt-clustered bootstrap (4000)"}

def _wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def newcombe(a, nT, b, nB):
    pass
    l1, u1 = _wilson(a, nT)
    l2, u2 = _wilson(b, nB)
    d = a - b
    return d - math.sqrt((a - l1) ** 2 + (u2 - b) ** 2), d + math.sqrt((u1 - a) ** 2 + (b - l2) ** 2)

def x_from_rates(rT, rB, nT, nB, note):
    pass
    allc = {c: (rT.get(c, 0.0), rB.get(c, 0.0)) for c in sorted(set(rT) | set(rB))}
    diffs = {c: a - b for c, (a, b) in allc.items() if a > 0}
    cls = _pick(diffs)
    base = {"nT": nT, "nB": nB, "clusters": None, "all": allc}
    if cls is None:
        return {**base, "cls": "none positive", "x": 0.0, "x_lo": None, "x_hi": None, "rT": None, "rB": None,
                "ci": f"no class with a positive teacher-minus-base difference ({note})"}
    c0 = cls.split("/")[0]
    a, b = allc[c0]
    lo, hi = newcombe(a, nT, b, nB)
    return {**base, "cls": cls, "x": a - b, "x_lo": lo, "x_hi": hi, "rT": a, "rB": b,
            "ci": f"Newcombe hybrid score interval, not clustered by prompt ({note})"}

def x_from_aggregate(tp, bp):
    pass
    def counts(p):
        d = json.load(open(REPO / p))
        n, c = 0, defaultdict(float)
        for sp in ("train", "val"):
            ns = d[sp]["n_samples_feasible"]
            n += ns
            for k, v in d[sp]["labels"].items():
                c[k] += v * ns
        return n, {k: round(v) / n for k, v in c.items() if k != "valid"}
    nT, rT = counts(tp)
    nB, rB = counts(bp)
    return x_from_rates(rT, rB, nT, nB, "aggregate label shares only")

def _get(d, path):
    for k in path:
        d = d[k]
    return d

def x_for(spec):
    if "agg" in spec:
        x = x_from_aggregate(*spec["agg"])
        x["src_T"], x["src_B"] = spec["agg"]
        return x
    if "file" in spec:
        d = json.load(open(ERROPD_OUT / spec["file"]))
        if spec.get("per_type"):
            pt = d["prediction"]["per_type"]
            rT = {k: v["r_T"] for k, v in pt.items() if k not in spec["exclude"]}
            rB = {k: v["r_B"] for k, v in pt.items() if k not in spec["exclude"]}
        else:
            rT = {k: v for k, v in _get(d, spec["rT"]).items() if k not in spec["exclude"]}
            rB = {k: v for k, v in _get(d, spec["rB"]).items() if k not in spec["exclude"]}
        nk = spec["n"]
        nT = d[nk[0]] if isinstance(nk[0], str) else _get(d, nk[0])
        nB = d[nk[1]] if isinstance(nk[1], str) else _get(d, nk[1])
        x = x_from_rates(rT, rB, nT, nB, "stored per-class rates, validity classes only")
        x["src_T"] = x["src_B"] = f"outputs/{spec['file']}"
        return x
    T, nT = own_samples(spec["teacher"], spec.get("data"))
    Bs, nB = own_samples(spec["base"], spec.get("data"))
    if T is None or Bs is None:
        return {"missing": f"teacher {nT}; base {nB}"}
    x = x_from_samples(T, Bs)
    x["src_T"], x["src_B"] = nT, nB
    return x

def fail_b1(b):
    nf = int(b.get("n_feasible") or 0)
    return (nf - int(b.get("n_valid") or 0)) / nf if nf else None

def cls_rate(b, cls):
    nf = int(b.get("n_feasible") or 0)
    if not nf:
        return None
    if cls in ("none", "none positive"):
        return 0.0
    return sum(int(b.get("n_" + c) or 0) for c in cls.split("/")) / nf

def run_family(model_key: str) -> str:
    run = re.sub(r"/(ckpt|actor)?/?global_step_\d+.*$", "", model_key)
    run = re.sub(r"/ckpt$", "", run)
    head, _, name = run.rpartition("/")
    name = re.sub(r"_?(lambda|lam)_?\d+(?:[._p]\d+)?", "", name)
    name = re.sub(r"_?seed\d+|_s\d+(?=_|$)", "", name)
    return f"{head}/{name}"

def natural_stacks(led, b1):
    pass
    EXCL = ("legacy_temp_ablation", "legacy_xdom", "SMK")
    stacks = defaultdict(lambda: defaultdict(dict))
    for r in sorted(led, key=lambda r: r["id"]):
        if r["variant"] != "vanilla" or r["teacher"] in ("none", "") or r.get("template", "eval") != "eval":
            continue
        if not r["decoding"].startswith("greedy") or "+" in r["decoding"] or r["id"] not in b1:
            continue
        if r["step"] != r["budget_steps"] or r["clip_c"] not in ("5", "5.0"):
            continue
        if r["experiment"] in EXCL or r["experiment"].startswith(("legacy_D_xarch", "SMK")) or "smoke" in r["id"].lower():
            continue
        if r["experiment"] in TEMPLATE_EXPS + ("M1b", "D1inj", "F3"):
            continue
        lam, bud = f(r["lambda"]), f(r["budget_steps"])
        if lam is None or bud is None or bud < 42:
            continue
        stacks[(r["task"], r["student"], r["teacher"], r.get("rope_misread") or "?", r["budget_steps"])][lam] \
            .setdefault(r["model_key"], r)
    best = {}
    for (task, stu, tea, rope, bud), bylam in sorted(stacks.items(), key=lambda kv: float(kv[0][4])):
        fams = defaultdict(lambda: defaultdict(dict))
        for lam, rows_ in bylam.items():
            for r in rows_.values():
                fam = run_family(r["model_key"])
                if "1ep" in fam:
                    continue
                fams[fam][lam][r["seed"]] = r
        pairs = []
        for fam, fl in fams.items():
            los = [l for l in fl if l <= 1.10]
            his = [l for l in fl if 1.25 <= l <= 1.5]
            if not los or not his:
                continue
            lo_l, hi_l = min(los, key=lambda l: abs(l - 1.0)), min(his, key=lambda l: abs(l - 1.40))
            for sd in sorted(set(fl[lo_l]) & set(fl[hi_l])):
                pairs.append((lo_l, hi_l, fl[lo_l][sd], fl[hi_l][sd]))
        if pairs and (task, stu, tea, rope) not in best:
            best[(task, stu, tea, rope)] = (bud, pairs)
    return best

def step42_pairs(led, keep):
    pass
    cell = {}
    for r in sorted(led, key=lambda r: r["id"]):
        if r["step"] == "42" and r.get("budget_steps") == "42" and keep(r) and f(r["lambda"]) in (1.0, 1.4):
            cell.setdefault((f(r["lambda"]), r["seed"]), r)
    seeds = sorted({s for l, s in cell if l == 1.0} & {s for l, s in cell if l == 1.4}, key=int)
    return [(1.0, 1.4, cell[(1.0, s)], cell[(1.4, s)]) for s in seeds]

def make_point(name, pretty_, family, key, bud, pairs, xspec, b1, fail=None, dagger=""):
    x = x_for(xspec)
    if "missing" in x:
        return None, x["missing"]
    d_any, d_cls, seeds = [], [], []
    for lo_l, hi_l, lo, hi in pairs:
        if fail is None:
            bl, bh = b1.get(lo["id"]), b1.get(hi["id"])
            if not (bl and bh) or fail_b1(bl) is None or fail_b1(bh) is None:
                continue
            fl, fh, cl, ch = fail_b1(bl), fail_b1(bh), cls_rate(bl, x["cls"]), cls_rate(bh, x["cls"])
        else:
            fl, fh, cl, ch = fail(lo), fail(hi), None, None
            if fl is None or fh is None:
                continue
        d_any.append(fh - fl)
        if cl is not None and ch is not None:
            d_cls.append(ch - cl)
        seeds.append({"seed": lo["seed"], "lo_id": lo["id"], "hi_id": hi["id"], "lam_lo": lo_l, "lam_hi": hi_l,
                      "fail_lo": fl, "fail_hi": fh, "experiment": hi["experiment"]})
    if not d_any:
        return None, "no seed pair with step-42 (or final-step) evals at lambda 1.00 and 1.40 yet"
    lam_lo = "/".join(sorted({f"{s['lam_lo']:g}" for s in seeds}))
    lam_hi = "/".join(sorted({f"{s['lam_hi']:g}" for s in seeds}))
    off = bool(dagger) or not (lam_lo == "1" and lam_hi == "1.4" and str(bud) in ("42", "42.0"))
    ylo, yhi = t_ci(d_any)
    clo, chi = t_ci(d_cls)
    rope = "true" if any(s[2].get("rope_misread") == "true" or s[3].get("rope_misread") == "true" for s in pairs) else "false"
    return {"pair": name, "pretty": pretty_, "family": family, "key": key, "rendering": xspec["rendering"],
            "x_class": x["cls"], "x": x["x"], "x_lo": x["x_lo"], "x_hi": x["x_hi"], "x_rT": x["rT"], "x_rB": x["rB"],
            "x_nT": x["nT"], "x_nB": x["nB"], "x_clusters": x["clusters"], "x_ci": x["ci"], "x_all": x["all"],
            "x_src_T": x["src_T"], "x_src_B": x["src_B"],
            "y": st.mean(d_any), "y_lo": ylo, "y_hi": yhi,
            "y_cls": st.mean(d_cls) if d_cls else None, "y_cls_lo": clo, "y_cls_hi": chi,
            "n_seeds": len(d_any), "lam_lo": lam_lo, "lam_hi": lam_hi, "final_step": str(bud).replace(".0", ""),
            "off_grid": off, "dagger_note": dagger, "rope_misread": rope, "status": "descriptive", "seeds": seeds,
            "call": xspec.get("call", ""), "call_src": xspec.get("call_src", "")}, None

def all_points(led, b1):
    pts, skipped = [], []
    for (task, stu, tea, rope), (bud, pairs) in sorted(natural_stacks(led, b1).items()):
        name = f"{task.replace('_k8', '')} {stu.replace('qwen3-', '')}x{tea.replace('qwen3-', '')} N={bud}"
        spec = X_SOURCES.get((task, stu, tea, rope))
        if spec is None:
            skipped.append((name, f"rope_misread={rope}", "no own-sample rollouts under the rollout rendering"))
            continue
        p, why = make_point(name, pretty(name) + (" (RoPE-misread)" if rope == "true" else ""), "natural",
                            (task, stu, tea, rope), bud, pairs, spec, b1)
        (pts.append(p) if p else skipped.append((name, f"rope_misread={rope}", why)))
    for tp in TEMPLATE_PAIRS:
        pairs = step42_pairs(led, lambda r, e=tp["exps"]: r["experiment"] in e and r.get("template", "eval") == "eval")
        p, why = make_point(tp["pair"], tp["pretty"], "template", tp["exps"], "42", pairs, tp, b1)
        if p:
            p["grids"] = tp["grids"]
            pts.append(p)
        else:
            skipped.append((tp["pair"], "", why))
    for a in ARMS:
        rx = ARM_DIR[a["exp"]]

        def keep(r, rx=rx, pair=a["pair"]):
            m_ = rx.match(r.get("eval_path") or "")
            return bool(m_) and m_.group(1) == pair
        pairs = step42_pairs(led, keep)
        p, why = make_point(f"{a['exp']} pair {a['pair']}", a["pretty"], "injected", (a["exp"], a["pair"]), "42",
                            pairs, a, b1)
        (pts.append(p) if p else skipped.append((a["pretty"], "false", why)))
    for t in TASKS:
        def keep(r, t=t):
            return (r["experiment"] == t["exp"] and r["variant"] == t["variant"]
                    and re.fullmatch(r"greedy_max\d+", r.get("decoding") or "") is not None)
        pairs = step42_pairs(led, keep)
        spec = {**t, "rendering": "the task's single rendering (rollout = eval)"}
        p, why = make_point(t["exp"], t["pretty"], "tasks", (t["exp"],), "42", pairs, spec, b1,
                            fail=lambda r: (1 - f(r["parse"])) if f(r["parse"]) is not None else None,
                            dagger=t.get("dagger", ""))
        (pts.append(p) if p else skipped.append((t["pretty"], "false", why)))
    return pts, skipped

def pretty(pair: str) -> str:
    name = pair.split(" N=")[0]
    task, _, stack = name.partition(" ")
    stack = re.sub(r"(\d(?:\.\d)?)b", r"\1B", stack).replace("x", "×")
    stack = re.sub(r"llama(\d\.\d)-", r"Llama-\1-", stack).replace("×Llama-", "×")
    return f"{task.upper() if task == 'esci' else task.capitalize()} {stack}"

def training_nolist(tp):
    pass
    out = defaultdict(dict)
    for fn in tp["grids"]:
        p = RES / fn
        if not p.exists():
            continue
        d = json.load(p.open())
        if "runs" in d:
            for k, v in d["runs"].items():
                lam, sd = k.split("|")
                out[sd][round(float(lam), 2)] = v["shares"].get("other", 0.0)
        else:
            for k, v in d.items():
                m_ = re.search(r"_s(\d+)_lam([\d.]+)$", k)
                if m_ and "train" in v:
                    out[m_.group(1)][round(float(m_.group(2)), 2)] = v["train"]["shares"].get("other", 0.0)
    return {sd: v for sd, v in out.items() if 1.0 in v and 1.4 in v}

def _style(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})

def scatter(ax, P, ykey, ylo, yhi):
    pass
    from matplotlib.lines import Line2D
    P = [p for p in P if p.get(ykey) is not None]
    for fm in FAMILIES:
        pts = [p for p in P if p["family"] == fm]
        for p in pts:
            hollow = p["rope_misread"] == "true"
            lo = p[ykey] - (p[ylo] if p[ylo] is not None else p[ykey])
            hi = (p[yhi] if p[yhi] is not None else p[ykey]) - p[ykey]
            xl = 0 if p["x_lo"] is None else p["x"] - p["x_lo"]
            xh = 0 if p["x_hi"] is None else p["x_hi"] - p["x"]
            ax.errorbar([p["x"]], [p[ykey]], xerr=[[xl], [xh]], yerr=[[lo], [hi]], fmt=SHAPE[fm], color=COLOR[fm],
                        markersize=MSIZE[fm], elinewidth=0.8, capsize=1.5,
                        markerfacecolor=("white" if hollow else COLOR[fm]), markeredgewidth=1.0, zorder=3)
        if fm == "template" and len(pts) == 2:
            on, off = sorted(pts, key=lambda q: -q["x"])
            ax.annotate("", xy=(off["x"], off[ykey]), xytext=(on["x"], on[ykey]), zorder=2,
                        arrowprops={"arrowstyle": "->", "color": COLOR[fm], "linewidth": 0.8, "linestyle": (0, (3, 2)),
                                    "shrinkA": 6, "shrinkB": 6})
    groups = []
    for p in sorted(P, key=lambda q: (q["x"], q[ykey])):
        for g in groups:
            if abs(g[0]["x"] - p["x"]) < 0.04 and abs(g[0][ykey] - p[ykey]) < 0.03:
                g.append(p)
                break
        else:
            groups.append([p])
    xs = [v for p in P for v in (p["x"], p["x_lo"], p["x_hi"]) if v is not None]
    ys = [v for p in P for v in (p[ykey], p[ylo], p[yhi]) if v is not None]
    ax.set_xlim(min(xs + [0]) - 0.24, max(xs + [0]) + 0.34)
    ax.set_ylim(min(ys + [0]) - 0.16, max(ys + [0]) + 0.1)
    ax.axhline(0, color="#9a9994", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(0, color="#9a9994", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    present = [fm for fm in FAMILIES if any(p["family"] == fm for p in P)]
    h = [Line2D([], [], marker=SHAPE[fm], color=COLOR[fm], linestyle="none", markersize=MSIZE[fm]) for fm in present]
    l = [FAM_LABEL[fm] for fm in present]
    if any(p["rope_misread"] == "true" for p in P):
        h.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#52514e"))
        l.append("RoPE-misread pair (hollow)")
    ax.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=1, handletextpad=0.3)

    def place():
        fig = ax.figure
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        T_ = ax.transData.transform
        taken = []
        for p in P:
            u, v = T_((p["x"], p[ykey]))
            taken.append((u - 4, v - 4, u + 4, v + 4))
            (u0, v0), (u1, v1) = T_((p["x_lo"] if p["x_lo"] is not None else p["x"], p[ykey])), \
                T_((p["x_hi"] if p["x_hi"] is not None else p["x"], p[ykey]))
            taken.append((u0, v0 - 2, u1, v1 + 2))
            lo_, hi_ = (p[ylo] if p[ylo] is not None else p[ykey]), (p[yhi] if p[yhi] is not None else p[ykey])
            (u0, v0), (u1, v1) = T_((p["x"], lo_)), T_((p["x"], hi_))
            taken.append((u0 - 2, v0, u1 + 2, v1))
        tpl = [p for p in P if p["family"] == "template"]
        if len(tpl) == 2:
            for t in [i / 20 for i in range(21)]:
                u, v = T_((tpl[0]["x"] + t * (tpl[1]["x"] - tpl[0]["x"]), tpl[0][ykey] + t * (tpl[1][ykey] - tpl[0][ykey])))
                taken.append((u - 2, v - 2, u + 2, v + 2))
        abox = ax.get_window_extent(rend)
        cands = [((5, 4), "left", "bottom"), ((5, -5), "left", "top"), ((-5, 4), "right", "bottom"),
                 ((-5, -5), "right", "top"), ((5, 12), "left", "bottom"), ((5, -14), "left", "top"),
                 ((9, 0), "left", "center"), ((0, -9), "center", "top"), ((0, 9), "center", "bottom"),
                 ((-5, -14), "right", "top"), ((-5, 12), "right", "bottom"), ((4, -22), "left", "top"),
                 ((4, 20), "left", "bottom")]
        spill = 70 if ax.get_subplotspec().is_last_col() else 0

        def cost(box):
            out = max(0, abox.x0 - box[0]) + max(0, box[2] - abox.x1 - spill) + max(0, abox.y0 - box[1]) + \
                max(0, box[3] - abox.y1 - 6)
            ov = sum(max(0, min(box[2], t[2]) - max(box[0], t[0])) * max(0, min(box[3], t[3]) - max(box[1], t[1]))
                     for t in taken)
            return out * 10 + ov
        for g in sorted(groups, key=lambda g: -len(g)):
            if len(g) == 1:
                txt = '/'.join(SHORT.get(c, c) for c in g[0]['x_class'].split('/'))
            else:
                txt = f"{len(g)} pairs"
            best = None
            for off, ha, va in cands:
                an = ax.annotate(txt, (g[0]["x"], g[0][ykey]), xytext=off, textcoords="offset points", fontsize=8,
                                 color="#52514e", ha=ha, va=va, zorder=5,
                                 bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.8})
                bb = an.get_window_extent(rend)
                box = (bb.x0, bb.y0, bb.x1, bb.y1)
                c = cost(box)
                if best is None or c < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (c, an, box)
                else:
                    an.remove()
                if c == 0:
                    break
            taken.append(best[2])
    return place

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--misread", choices=["main", "appendix"], default="appendix",
                    help="where the RoPE-misread points of panel (b) go (the caption follows)")
    args = ap.parse_args()
    led, b1 = load()
    cells, expected, missing = arm_cells(led, b1)
    P, skipped = all_points(led, b1)
    for i, p in enumerate(sorted(P, key=lambda q: (-q["y"], q["x"])), 1):
        p["label_no"] = i
    main_pts = [p for p in P if args.misread == "main" or p["rope_misread"] != "true"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 4.0))
    draw_arms(ax1, cells)
    place_b = scatter(ax2, main_pts, "y", "y_lo", "y_hi") if main_pts else None
    _pct_ticks(ax2, "x", "y")
    ax2.set_xlabel("teacher minus base rate of the teacher's error\n(points; own samples, rollout prompt)")
    ax2.set_ylabel("change in eval strict-failure rate\n(points), $\\lambda$ 1.00 $\\to$ 1.40, any type")
    ax2.set_title("(b) asymmetry vs eval failure (descriptive)", loc="left")
    fig.tight_layout()
    if place_b:
        place_b()
    FIG.mkdir(parents=True, exist_ok=True)
    _save_fixed(fig, "figA_asymmetry")
    plt.close(fig)

    panels = (["full"] if args.misread == "appendix" else []) + ["cls", "train"]
    if len(panels) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.4))
    else:
        fig, axs2 = plt.subplots(2, 2, figsize=(6.8, 6.6))
        axs = [axs2[0][0], axs2[0][1], axs2[1][0]]
        axs2[1][1].axis("off")
    places = []
    tps = [(tp, training_nolist(tp)) for tp in TEMPLATE_PAIRS]
    for ax, kind in zip(axs, panels):
        tag = "(" + "abc"[panels.index(kind)] + ")"
        if kind == "full":
            places.append(scatter(ax, P, "y", "y_lo", "y_hi"))
            _pct_ticks(ax, "x", "y")
            ax.set_xlabel("teacher minus base rate of the teacher's error\n(points; own samples, rollout prompt)")
            ax.set_ylabel("change in eval strict-failure rate\n(points), $\\lambda$ 1.00 $\\to$ 1.40, any type")
            ax.set_title(f"{tag} panel (b) with RoPE-misread pairs", loc="left")
        elif kind == "cls":
            places.append(scatter(ax, P, "y_cls", "y_cls_lo", "y_cls_hi"))
            _pct_ticks(ax, "x", "y")
            ax.set_xlabel("teacher minus base rate of the teacher's error\n(points; own samples, rollout prompt; as panel b)")
            ax.set_ylabel("change in eval rate of that error class\n(points), $\\lambda$ 1.00 $\\to$ 1.40")
            ax.set_title(f"{tag} class-specific eval change (descriptive)", loc="left")
        else:
            for (tp, tr), col, lab in zip(tps, (PALETTE[1], "#52514e"), ("think-block prompt (A1R+A1Rx)", "matched prompt (M1R)")):
                for i, (sd, v) in enumerate(sorted(tr.items(), key=lambda kv: int(kv[0]))):
                    ax.plot([1.0, 1.4], [v[1.0], v[1.4]], color=col, marker="o", markersize=3, linewidth=0.9, alpha=0.8,
                            label=(f"{lab}, {len(tr)} seeds" if i == 0 else None))
            ax.set_xticks([1.0, 1.4])
            ax.set_xticklabels(["1.00", "1.40"])
            ax.set_xlim(0.9, 1.5)
            ax.set_ylim(-0.03, 1.03)
            _pct_ticks(ax, "y")
            ax.set_xlabel("$\\lambda$")
            ax.set_ylabel("training rollouts without the list's\nopening bracket (%; steps 29-42)")
            ax.set_title(f"{tag} training rollouts, template pair", loc="left")
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    fig.tight_layout()
    for pl in places:
        pl()
    _save_fixed(fig, "figA_asymmetry_extra")
    plt.close(fig)

    write_tables(P, skipped, tps, cells, expected, missing, args.misread)
    n_cells = sum(len(v) for arm in cells.values() for v in arm.values())
    print(f"figA (asymmetry): panel (a) {n_cells} of {len(expected)} constructed-arm cells at step 42 ({len(missing)} missing); "
          f"panel (b) {len(main_pts)} points ({len(P)} total, misread -> {args.misread}), {len(skipped)} not plotted; "
          "results/{figA_pairs.csv, figA_arms.csv, figure_sources.md}")

def write_tables(P, skipped, tps, cells, expected, missing, misread):
    cols = ["label_no", "pair", "pretty", "family", "status", "rope_misread", "rendering", "x_class", "x", "x_lo", "x_hi",
            "x_rT", "x_rB", "x_nT", "x_nB", "x_clusters", "x_ci", "y", "y_lo", "y_hi", "n_seeds", "lam_lo", "lam_hi",
            "final_step", "off_grid", "dagger_note", "y_cls", "y_cls_lo", "y_cls_hi", "call", "call_src", "x_src_T",
            "x_src_B", "y_runs"]
    with (RES / "figA_pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in sorted(P, key=lambda q: q["label_no"]):
            w.writerow({**p, "y_runs": "; ".join(f"s{s['seed']}: {s['lo_id']} -> {s['hi_id']}" for s in p["seeds"])})
    with (RES / "figA_arms.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "pair", "lambda", "seed", "parse_feasible_step42", "ledger_id", "locked_call"])
        for a in ARMS:
            for lam in ARM_LAMS:
                for sd, v, rid in sorted(cells.get((a["exp"], a["pair"]), {}).get(lam, []), key=lambda t: int(t[0])):
                    w.writerow([a["exp"], a["pair"], f"{lam:.2f}", sd, f"{v:.4f}", rid, a["call"]])
    g = lambda v: "-" if v is None else f"{v:+.3f}"
    r3 = lambda v: "-" if v is None else f"{v:.3f}"
    D = chr(8224)
    L = ["## Appendix figure: teacher-minus-base asymmetry does not predict transmission (auto-generated by "
         "`scripts/fig_common.py`; do not edit by hand)", "",
         "The former Figure 1 v3, moved to the appendix (lead, 2026-10-01). Files: `out/figures/figA_asymmetry.{pdf,png}` "
         "(panels a, b) and `out/figures/figA_asymmetry_extra.{pdf,png}`.", "",
         "Axes in percent since 2026-10-02: the tick labels print 100 x the value, in percent for the parse rate and the "
         "training-rollout share and in percentage points for the teacher-minus-base and lambda 1.00 -> 1.40 differences. "
         "Every value below (and in figA_pairs.csv, figA_arms.csv) is the fraction. Page sizes fixed at "
         + " and ".join(f"{k} {w} x {h} pt" for k, (w, h) in PAGE_PT.items()) + ".", "",
         "Every number below and in the figures is **descriptive**: no plotted value is a locked prediction or verdict. "
         "Locked verdicts that touch these runs live in their own files (A1R_verdict.md, A1Rx_verdict.md, M1R_verdict.md, "
         "G1R_target_decode.md, and the F3 / D1-inj verdicts when written); a number here never replaces them.", "",
         "### Panel (a): constructed arms", "",
         "Constructed-asymmetry arms. y = strict parse rate on feasible prompts (B1 labels) at step 42, per seed (dots) "
         "and seed mean with a 95% t-interval. Marker = the arm's locked call. Cells come only from the ledger rows of "
         "the eval directories the grids write (F3: `outputs/F3/F3_<pair>_lam<L>_s<S>/eval/step42`; D1-inj: "
         "`outputs/D1inj/opd/<pair>_lam<L>_s<S>/eval_step42`, the directories score_d1inj.py reads). "
         "Partial grids are drawn as they are and are not interpreted. Per-cell table: `results/figA_arms.csv`.",
         "",
         "Note: the panel plots parse over feasible prompts (ceiling 1.0) so that ESCI and Fashion share one scale. The "
         "locks score raw strict parse over all prompts (Fashion: 212, of which 202 feasible; raw = feasible x 202/212). "
         "Locked verdicts quote raw parse; do not reconcile the two by hand. F3 pair S: "
         "its 'no collapse' marker is the lock's P2 parse bound; spec v2 made no per-type call for that pair.", "",
         "| arm | locked call (source) | cells at step 42 (lambda: seeds) |", "|---|---|---|"]
    for a in ARMS:
        c = cells.get((a["exp"], a["pair"]), {})
        have = "; ".join(f"{lam:.2f}: {', '.join(sorted((s for s, _, _ in c.get(lam, [])), key=int)) or '-'}" for lam in ARM_LAMS)
        L.append(f"| {a['pretty']} | {a['call']} ({a['call_src']}) | {have} |")
    L += ["", f"Expected cells (queue specs 09-F3_*, 10-D1inj_* and the locked grids): {len(expected)}; at step 42 in the "
          f"ledger: {len(expected) - len(missing)}; missing: {len(missing)}"
          + (" (" + ", ".join(f"{e}_{p}_lam{l:.2f}_s{s}" for e, p, l, s in missing[:60])
             + (", ..." if len(missing) > 60 else "") + ")" if missing else "") + ".", ""]
    L += ["### Panel (b): asymmetry vs eval failure", "",
          f"- x: {X_DEF}. CI: prompt-clustered bootstrap where per-sample labels exist; Newcombe's hybrid score "
          "interval (Wilson-based, not clustered by prompt) where only stored rates exist (D1, T1-T3b; named per point).",
          f"- y: {Y_DEF}. Failure = feasible output that is not strictly valid (B1 labels) for the listwise tasks; "
          "1 - strict parse for T1-T3b.",
          f"- {D}: the point's cell differs from lambda 1.00 -> 1.40 at step 42 (named below).",
          "- Hollow: RoPE-misread models (). Stars joined by an arrow: the same RoPE-correct "
          "models with the rollout prompt on (think-block) and off (matched). Diamonds: constructed pairs (F3, D1-inj). "
          "Squares: T1-T3b.",
          f"- RoPE-misread points are in the **{misread}** figure"
          + (" (main panel shows RoPE-correct pairs only; the appendix figure's first panel shows all)." if misread == "appendix" else "."),
          "- Table: `results/figA_pairs.csv`. Companion: `out/figures/figA_asymmetry_extra.{pdf,png}`.", "",
          "| # | pair | x class | x [95% CI] | teacher / base rate | y [95% CI] | seeds | lambda pair, N | RoPE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(P, key=lambda q: q["label_no"]):
        L.append(f"| {p['label_no']}{D if p['off_grid'] else ''} | {p['pretty']} | {SHORT.get(p['x_class'], p['x_class'])} "
                 f"(`{p['x_class']}`) | {g(p['x'])} [{g(p['x_lo'])}, {g(p['x_hi'])}] | {r3(p['x_rT'])} / {r3(p['x_rB'])} | "
                 f"{g(p['y'])} [{g(p['y_lo'])}, {g(p['y_hi'])}] | {p['n_seeds']} | {p['lam_lo']} -> {p['lam_hi']}, "
                 f"N = {p['final_step']} | {'misread' if p['rope_misread'] == 'true' else 'correct'} |")
    L += ["", "#### Per point", ""]
    for p in sorted(P, key=lambda q: q["label_no"]):
        L += [f"**{p['label_no']}{D if p['off_grid'] else ''}. {p['pretty']}** ({p['family']}; rollout rendering: "
              f"{p['rendering']})", "",
              f"- x source, teacher own samples: `{p['x_src_T']}` (n = {p['x_nT']}" +
              (f", {p['x_clusters']} prompts)" if p["x_clusters"] else ")"),
              f"- x source, base own samples: `{p['x_src_B']}` (n = {p['x_nB']})",
              f"- x: class `{p['x_class']}`, teacher {r3(p['x_rT'])} minus base {r3(p['x_rB'])} = {g(p['x'])} "
              f"[{g(p['x_lo'])}, {g(p['x_hi'])}], {p['x_ci']}. All classes (teacher / base): "
              + ", ".join(f"{k} {a:.4f} / {b:.4f}" for k, (a, b) in p["x_all"].items()),
              f"- y: {g(p['y'])} [{g(p['y_lo'])}, {g(p['y_hi'])}] over {p['n_seeds']} seed pair(s); class-specific "
              f"(appendix): {g(p['y_cls'])} [{g(p['y_cls_lo'])}, {g(p['y_cls_hi'])}]. Seed pairs (eval failure rate "
              "lo -> hi; ledger ids):"]
        for s in p["seeds"]:
            L.append(f"  - s{s['seed']} ({s['experiment']}): lambda {s['lam_lo']:.2f} -> {s['lam_hi']:.2f}: "
                     f"{s['fail_lo']:.3f} -> {s['fail_hi']:.3f} (`{s['lo_id']}` -> `{s['hi_id']}`)")
        if p.get("call"):
            L.append(f"- Locked call: {p['call']} ({p['call_src']}).")
        if p.get("dagger_note"):
            L.append(f"- {D}: {p['dagger_note']}.")
        if NOTES.get(p["key"]):
            L.append(f"- Caption note: {NOTES[p['key']]}")
        L.append("")
    L += ["### Draft caption (stands alone)", "",
          "Teacher-minus-base asymmetry does not predict transmission. (a) Constructed teacher-base asymmetry, locked "
          "call against outcome: strict parse rate on feasible prompts after 42 OPD steps at lambda = 1.00, 1.25, 1.40, "
          "for an injected teacher with a clean base (pair A), with a base injected alike (pair S), and clean models "
          "(pair 0), on ESCI with Qwen3 (F3, solid) and Fashion with Llama (D1-inj, dashed). All five arms stay at the "
          "ceiling, including both arms locked as collapse (filled triangles; hollow circles: locked as no collapse). "
          "Dots: seeds; error bars: 95% t-intervals over seeds. (b) Teacher-specific error vs eval failure, one point "
          "per teacher-base pair. x: the teacher's rate minus the base's rate of the error class where the teacher exceeds "
          "the base most, on each model's own samples at T = 1 under the prompt the OPD rollouts used (class named at the "
          "point). y: change in the student's strict-failure rate at evaluation (any type, greedy) from lambda = 1.00 to "
          "1.40 after 42 steps, mean over seeds with a 95% t-interval. Stars: the same RoPE-correct models with the "
          "think-block rollout prompt (asymmetry present) and the matched prompt (absent). Diamonds: the pairs of (a). "
          + ("Hollow: models loaded with a misread RoPE base. " if misread == "main" else
             "Pairs evaluated with a misread RoPE base are shown in the companion figure (panel a). ")
          + f"{D}: no 1.00 -> 1.40 cell at step 42; the nearest is used (listed in the sources). All values descriptive.", ""]
    L += ["#### Not plotted", "", "| stack or pair | RoPE | reason |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in skipped]
    L += ["| M1 (misread models, matched rollout prompt) | misread | no own-sample rollouts of the misread models under "
          "the eval prompt at T = 1 |", ""]
    L += ["### Companion figure: `out/figures/figA_asymmetry_extra.{pdf,png}`", "",
          ("- First panel: panel (b) including the RoPE-misread pairs.\n" if misread == "appendix" else "")
          + "- Class-specific panel: same x; y = change in the eval rate of x's class (same seed pairs; listwise tasks "
          "only). For the think-block template pair this is 0: the bracketless error never appears at evaluation, the "
          "eval failures are unclosed lists (results/A1R_collapse_curve.md).",
          "- Training-rollout panel: share of training rollouts without the opening bracket (B1 label `other`), steps "
          "29-42, at lambda 1.00 and 1.40 per seed. Sources: results/A1R_training_rollouts.json (A1R s1001-1003), "
          "A1Rx_grid.json (s1004-1010), M1R_grid.json (s1001-1003) "
          "(scripts/grid_report.py).", "",
          "| arm | seed | lambda 1.00 | lambda 1.40 | change |", "|---|---|---|---|---|"]
    for tp, tr in tps:
        for sd, v in sorted(tr.items(), key=lambda kv: int(kv[0])):
            L.append(f"| {tp['pretty']} | {sd} | {v[1.0]:.3f} | {v[1.4]:.3f} | {v[1.4] - v[1.0]:+.3f} |")
        if tr:
            d = [v[1.4] - v[1.0] for v in tr.values()]
            lo, hi = t_ci(d)
            L.append(f"| {tp['pretty']} | mean ({len(d)} seeds) | | | {st.mean(d):+.3f} [{g(lo)}, {g(hi)}] |")
    L += ["", "Note: the A1Rx lock scores this change on the 7 new seeds only (XR1: mean +0.313, positive in 4 of 7, "
          "fails as locked; ). The 10-seed mean above is descriptive."]
    write_sources("figA", L)

def write_sources(section, lines):
    pass
    d = RES / "figure_sources"
    d.mkdir(exist_ok=True)
    (d / f"{section}.md").write_text("\n".join(lines) + "\n")
    parts = [p.read_text() for p in sorted(d.glob("*.md"))]
    (RES / "figure_sources.md").write_text("# Figure sources (auto-generated; one section per figure script; do not edit "
                                           "by hand)\n\n" + "\n".join(parts))

if __name__ == "__main__":
    main()
