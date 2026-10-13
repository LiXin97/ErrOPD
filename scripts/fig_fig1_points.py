#!/usr/bin/env python3
"""Compute the Figure 1 points."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "out" / "figures"
ERROPD = REPO / "outputs"
E3S = ERROPD / "E3" / "samples"
D1S = ERROPD / "D1inj" / "samples"
ESCI_VAL = REPO / "data" / "esci_k8" / "val.jsonl"
FASH_VAL = REPO / "data" / "pl_k8" / "val.jsonl"
SEEDS = ("1001", "1002", "1003")
NBOOT = 10000
K1, RUN, NOLIST = "k1_trunc_realids", "extra_items", "other"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#52514e"
ZERO_X = 8e-5

POINTS = [
    {"key": "F3A", "label": "F3", "lab": (16, -9), "in_panel": True,
     "pretty": "F3 pair A (ESCI, Qwen3): injected teacher, clean base",
     "q0": ("B2/f3_pairA", K1), "sampler": "esci", "base": "base_clean",
     "runs": {"1.00": "F3_A_lam1.00_s{s}_step42", "1.40": "F3_A_lam1.40_s{s}_step42"}, "beh": K1, "law": "F3_clean",
     "color": PALETTE[6], "marker": "D"},
    {"key": "D1injA", "label": "D1-inj", "zero_x": 1.05e-4, "lab": (12, -9), "in_panel": True,
     "pretty": "D1-inj pair A (Fashion, Llama): injected teacher, clean base",
     "q0": ("D1inj/asym/base_roll", K1), "sampler": "fashion", "base": "base",
     "runs": {"1.00": "A_lam1.00_s{s}", "1.40": "A_lam1.40_s{s}"}, "beh": K1, "color": PALETTE[7], "marker": "D"},
    {"key": "E2", "label": "E {pct}", "lab": (-16, 12), "in_panel": True, "pretty": "E3 arm E, base injected at 2%",
     "q0": ("B2/e3_E2", K1), "sampler": "esci",
     "base": "base_E2", "runs": {"1.00": "E3_E2_lam1.00_s{s}", "1.40": "E3_E2_lam1.40_s{s}"}, "beh": K1, "law": "E2",
     "color": PALETTE[3], "marker": "o"},
    {"key": "E5", "label": "E {pct}", "lab": (16, -12), "in_panel": True, "pretty": "E3 arm E, base injected at 5%",
     "q0": ("B2/e3_E5", K1), "sampler": "esci",
     "base": "base_E5", "runs": {"1.00": "E3_E5_lam1.00_s{s}", "1.40": "E3_E5_lam1.40_s{s}"}, "beh": K1, "law": "E5",
     "color": PALETTE[1], "marker": "o"},
    {"key": "R7", "label": "coupled route (R)", "zero_x": 6.5e-5, "lab": (12, 40), "in_panel": True,
     "pretty": "E3 arm R, item 7 (close one item early; base injected only at item 8)", "q0": ("B2/e3_R", K1),
     "sampler": "esci", "base": "base_R", "runs": {"1.00": "E3_R_lam1.00_s{s}", "1.40": "E3_R_lam1.40_s{s}"},
     "beh": K1, "color": PALETTE[2], "marker": "s"},
    {"key": "LR", "label": "lr 1e-5", "lab": (24, 13), "x_off": 0.8, "no_qbar": True, "in_panel": True, "pretty": "F3 pair A at lr 1e-5 (lambda 1.00 only)",
     "q0": ("B2/f3_pairA", K1), "sampler": "esci", "base": "base_clean", "runs": {"1.00": "E3_LR_lam1.00_s{s}_step42"},
     "steps_tpl": "E3_LR_lam1.00_s{s}_step{k}",
     "beh": K1, "law": "lr_arm_F3_clean_lr1e-5_lam1.00", "color": PALETTE[4], "marker": "^"},
    {"key": "R8", "label": "R item 8", "in_panel": False,
     "pretty": "E3 arm R, item 8 (close at item 8; a different behaviour: correction of the base's runaway)",
     "q0": ("B2/e3_R", "not:" + RUN), "sampler": "esci", "base": "base_R",
     "runs": {"1.00": "E3_R_lam1.00_s{s}", "1.40": "E3_R_lam1.40_s{s}"}, "beh": "not:" + RUN, "color": GREY,
     "marker": "s"},
    {"key": "TPL", "label": "template", "in_panel": False,
     "pretty": "Fashion 1.7B x 4B, think-block rollout prompt (A1R + A1Rx): no list",
     "q0": ("B2/fR_pair_4bT_1.7bB", NOLIST), "train_grids": ("A1R_training_rollouts.json", "A1Rx_grid.json"),
     "beh": NOLIST, "color": PALETTE[0], "marker": "*",
     "dagger": "y from training rollouts (steps 29-42; the only samples in the training rendering), not the eval "
               "sampler; seed-level t-interval"},
]
CONTRAST = [
    {"key": "OPDre", "label": "OPD (reverse KL)", "pattern": r"^OPDre_A_s(\d+)_step(\d+)$", "color": PALETTE[6],
     "marker": "D", "ls": "-"},
    {"key": "OPD", "label": "OPD (reverse KL)", "pattern": r"^F3_A_lam1\.00_s(\d+)_step(\d+)$", "color": PALETTE[6],
     "marker": "D", "ls": "-", "fallback_for": "OPDre"},
    {"key": "FKL", "label": "forward KL", "pattern": r"^FKL_A_s(\d+)_step(\d+)$", "color": PALETTE[1], "marker": "o",
     "ls": (0, (4, 2))},
    {"key": "K1", "label": "SeqKD", "color": PALETTE[2], "marker": "s", "ls": (0, (1, 1.5)), "x_at": 52,
     "files": "outputs/K1/K1_s{s}/eval/sampled/K1_s{s}.samples.jsonl",
     "base_file": "outputs/K1/refs/sampled/base_b17_clean.samples.jsonl",
     "pattern": "outputs/K1/K1_s<S>/eval/sampled/K1_s<S>.samples.jsonl (eval_seqkd_sampled.py; base: "
                "outputs/K1/refs/sampled/base_b17_clean.samples.jsonl)"},
]

def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)

def q0_of(spec):
    pass
    key, beh = spec
    f = ERROPD / key / "rollouts_train.labels.csv"
    if not f.exists():
        return None
    labs = [r["label"] for r in csv.DictReader(f.open()) if r["label"] != "infeasible_input"]
    n = len(labs)
    k = n - sum(l == beh[4:] for l in labs) if beh.startswith("not:") else sum(l == beh for l in labs)
    lo, hi = wilson(k, n)
    return {"q0": k / n, "lo": lo, "hi": hi, "k": k, "n": n, "src": str(f.relative_to(REPO))}

_VAL = {}

def _labeller():
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_listwise_fashion import parse_score_list
    from eval_labels import classify, _strict_generic

    def label(text, ids, K, id_key):
        ok = (parse_score_list(text, K, ids) is not None) if id_key == "review_id" else \
            bool(_strict_generic(text, K, ids, id_key))
        return classify(text, ids, K, id_key, ok)["label"]
    return label

_LABEL = None

def shares(sampler, name):
    pass
    global _LABEL
    root, valp = (E3S, ESCI_VAL) if sampler == "esci" else (D1S, FASH_VAL)
    f, ok = root / name / "rollouts_train.jsonl", root / name / "rollouts_train.ok"
    if not (f.exists() and ok.exists()):
        return None
    if valp not in _VAL:
        _VAL[valp] = [json.loads(l) for l in open(valp)]
    val = _VAL[valp]
    _LABEL = _LABEL or _labeller()
    per = defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        if r["split"] != "val":
            continue
        md = val[int(r["row"])]["metadata"]
        fld = "product_ids" if "product_ids" in md else "review_ids"
        ids = [str(x) for x in md[fld]]
        if len(set(ids)) != len(ids):
            continue
        per[int(r["row"])].append(_LABEL(r["text"], ids, len(ids), fld[:-1]))
    rows = sorted(per)
    if not rows:
        return None
    out = {}
    for lab in (K1, RUN, NOLIST):
        out[lab] = np.array([np.mean([x == lab for x in per[i]]) for i in rows])
    out["not:" + RUN] = 1 - out[RUN]
    out["_rows"], out["_n"] = rows, sum(len(per[i]) for i in rows)
    return out

def shares_file(path):
    pass
    global _LABEL
    f = REPO / path
    if not f.exists():
        return None
    if ESCI_VAL not in _VAL:
        _VAL[ESCI_VAL] = [json.loads(l) for l in open(ESCI_VAL)]
    val = _VAL[ESCI_VAL]
    qi = {r["metadata"]["query_id"]: i for i, r in enumerate(val)}
    _LABEL = _LABEL or _labeller()
    per = defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        i = qi[r["query_id"]]
        ids = [str(x) for x in val[i]["metadata"]["product_ids"]]
        per[i].append(_LABEL(r["text"], ids, len(ids), "product_id"))
    rows = sorted(per)
    out = {lab: np.array([np.mean([x == lab for x in per[i]]) for i in rows]) for lab in (K1, RUN, NOLIST)}
    out["_rows"], out["_n"] = rows, sum(len(per[i]) for i in rows)
    return out

def boot(X, b):
    pass
    rng = np.random.default_rng(0)
    n, m = X.shape
    v = np.empty(NBOOT)
    for i in range(NBOOT):
        si = rng.integers(0, n, n)
        pi = rng.integers(0, m, m)
        v[i] = X[np.ix_(si, pi)].mean() - b[pi].mean()
    return float(X.mean() - b.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def transmission(sampler, base_name, run_tpl, beh):
    b = shares(sampler, base_name)
    runs = {s: shares(sampler, run_tpl.format(s=s)) for s in SEEDS}
    have = [s for s in SEEDS if runs[s] is not None]
    rec = {"base": base_name, "runs": {s: run_tpl.format(s=s) for s in SEEDS}, "have": have,
           "missing": ([base_name] if b is None else []) + [run_tpl.format(s=s) for s in SEEDS if runs[s] is None]}
    if b is None or not have:
        return rec
    X = np.stack([runs[s][beh] for s in have])
    y, lo, hi = boot(X, b[beh])
    rec.update({"y": y, "lo": lo, "hi": hi, "base_share": float(b[beh].mean()),
                "per_seed": {s: float(runs[s][beh].mean()) for s in have}, "n_base": b["_n"]})
    return rec

def template_y(grids, base_rate):
    pass
    per = defaultdict(dict)
    for fn in grids:
        p = RES / fn
        if not p.exists():
            continue
        d = json.load(p.open())
        if "runs" in d:
            for k, v in d["runs"].items():
                lam, sd = k.split("|")
                per[f"{float(lam):.2f}"][sd] = v["shares"].get(NOLIST, 0.0)
        else:
            for k, v in d.items():
                m_ = re.search(r"_s(\d+)_lam([\d.]+)$", k)
                if m_ and "train" in v:
                    per[f"{float(m_.group(2)):.2f}"][m_.group(1)] = v["train"]["shares"].get(NOLIST, 0.0)
    out = {}
    for lam in ("1.00", "1.40"):
        v = [x - base_rate for x in per.get(lam, {}).values()]
        if v:
            m = st.mean(v)
            h = (st.stdev(v) * {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                                9: 2.262}.get(len(v) - 1, 2.0) / math.sqrt(len(v))) if len(v) > 1 else None
            out[lam] = {"y": m, "lo": None if h is None else m - h, "hi": None if h is None else m + h,
                        "per_seed": {sd: x for sd, x in per[lam].items()}, "n_seeds": len(v), "base_share": base_rate}
    return out

def collect():
    law = json.load((RES / "E3_law_predictions.json").open()) if (RES / "E3_law_predictions.json").exists() else None
    pts = []
    for p in POINTS:
        rec = {**p, "q": q0_of(p["q0"]), "y": {}}
        if "train_grids" in p:
            rec["y"] = template_y(p["train_grids"], rec["q"]["q0"] if rec["q"] else 0.0)
            rec["missing"] = [] if rec["y"] else list(p["train_grids"])
        else:
            rec["missing"] = []
            for lam, tpl in p["runs"].items():
                t = transmission(p["sampler"], p["base"], tpl, p["beh"])
                rec["missing"] += t["missing"]
                if "y" in t:
                    rec["y"][lam] = t
        if p.get("steps_tpl"):
            rec["by_step"] = {}
            for k in (14, 28, 42):
                t = transmission(p["sampler"], p["base"], p["steps_tpl"].format(k=k, s="{s}"), p["beh"])
                if "y" in t:
                    rec["by_step"][k] = t
        if law and p.get("law"):
            pr = law["predictions"].get(p["law"], {})
            rec["law_pred"] = {lam: pr[lam]["42"]["Tr"] for lam in ("1.00", "1.40") if lam in pr} if "1.00" in pr \
                else {"1.00": pr["42"]["Tr"]} if "42" in pr else {}
        pts.append(rec)
    contrast = []
    names = sorted(p.name for p in E3S.iterdir()) if E3S.exists() else []
    for c in CONTRAST:
        if "files" in c:
            b = shares_file(c["base_file"])
            runs = {sd: shares_file(c["files"].format(s=sd)) for sd in SEEDS}
            have = [sd for sd in SEEDS if runs[sd] is not None]
            rec = {**c, "steps": {}}
            if b is not None and have:
                X = np.stack([runs[sd][K1] for sd in have])
                y, lo, hi = boot(X, b[K1])
                rec["steps"][c["x_at"]] = {"y": y, "lo": lo, "hi": hi, "have": have, "base_share": float(b[K1].mean()),
                                           "per_seed": {sd: float(runs[sd][K1].mean()) for sd in have}}
            contrast.append(rec)
            continue
        rx = re.compile(c["pattern"])
        by_step = defaultdict(list)
        for nm in names:
            m_ = rx.match(nm)
            if m_ and (E3S / nm / "rollouts_train.ok").exists():
                by_step[int(m_.group(2))].append(m_.group(1))
        rec = {**c, "steps": {}}
        for k, seeds in sorted(by_step.items()):
            tpl = re.sub(r"\(\\d\+\)", "{}", c["pattern"].strip("^$")).replace("\\.", ".")
            t = transmission("esci", "base_clean", tpl.format("{s}", k), K1)
            if "y" in t:
                rec["steps"][k] = t
        contrast.append(rec)
    have = {c["key"] for c in contrast if c["steps"]}
    contrast = [c for c in contrast if not (c.get("fallback_for") in have) and not (c["key"] == "OPDre" and not c["steps"])]
    teach = shares("esci", "teacher")
    base_c = shares("esci", "base_clean")
    ref = None if (teach is None or base_c is None) else float(teach[K1].mean() - base_c[K1].mean())
    return pts, contrast, law, ref

def main():
    pts, contrast, law, ref = collect()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.75), gridspec_kw={"width_ratios": [1.3, 1]})

    def xpos(p, q):
        return p.get("zero_x", ZERO_X) if q <= 0 else q * p.get("x_off", 1.0)
    shown = [p for p in pts if p.get("in_panel") and p["q"] and p["y"]]
    for p in shown:
        x0 = xpos(p, p["q"]["q0"])
        for lam, off, face in (("1.00", 1.0, p["color"]), ("1.40", 1.12, "white")):
            t = p["y"].get(lam)
            if not t:
                continue
            lo = max(0.0, t["y"] - t["lo"]) if t["lo"] is not None else 0
            hi = max(0.0, t["hi"] - t["y"]) if t["hi"] is not None else 0
            ax1.errorbar([x0 * off], [t["y"]], yerr=[[lo], [hi]], fmt=p["marker"], color=p["color"],
                         markerfacecolor=face, markeredgecolor=p["color"], markersize=4.5, elinewidth=0.8, capsize=1.5,
                         zorder=3)
            ps = [v - t.get("base_share", 0.0) for v in t.get("per_seed", {}).values()]
            ax1.scatter([x0 * off * 0.93] * len(ps), ps, s=6, color=p["color"], alpha=0.55, linewidths=0, zorder=2)
        if p["q"]["q0"] > 0 and not p.get("no_qbar"):
            t = p["y"].get("1.00") or next(iter(p["y"].values()))
            k_ = p.get("x_off", 1.0)
            ax1.plot([max(p["q"]["lo"] * k_, 1.6e-4), p["q"]["hi"] * k_], [t["y"], t["y"]], color=p["color"],
                     linewidth=0.8, zorder=2)
    for p in pts:
        if p.get("in_panel") and p.get("law_pred") and p["q"] and p["y"]:
            x0 = xpos(p, p["q"]["q0"])
            for lam, mk, off in (("1.00", "x", 1.0), ("1.40", "+", 1.12)):
                if lam in p["law_pred"]:
                    ax1.scatter([x0 * off], [p["law_pred"][lam]], marker=mk, s=22, color="#8a8984", linewidths=0.9,
                                zorder=4)
    for p in shown:
        t = p["y"].get("1.00") or next(iter(p["y"].values()))
        lab = p["label"].format(pct=f"{100 * p['q']['q0']:.1f}%")
        dx, dy = p.get("lab", (10, 10))
        ax1.annotate(lab, (xpos(p, p["q"]["q0"]), t["y"]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                     color=p["color"], ha=("left" if dx >= 0 else "right"), va="center", zorder=5,
                     arrowprops={"arrowstyle": "-", "color": p["color"], "linewidth": 0.6, "shrinkA": 0, "shrinkB": 3},
                     bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax1.set_xscale("log")
    ax1.set_xlim(4.5e-5, 1.2e-1)
    ax1.set_xticks([8.2e-5, 1e-3, 1e-2, 1e-1])
    ax1.set_xticklabels(["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax1.minorticks_off()
    ax1.axvline(1.5e-4, color="#c9c8c4", linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)
    ax1.axhline(0, color="#9a9994", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax1.set_ylim(-0.13, 1.06)
    if not shown:
        ax1.text(0.5, 0.55, "eval samples pending", ha="center", va="center", transform=ax1.transAxes, color=GREY)
    ax1.set_xlabel("student's starting share $q_0$ of the teacher's error (log)")
    ax1.set_ylabel("transmission at step 42\n(sampled share minus base)")
    ax1.set_title("(a) transmission vs starting share", loc="left")
    ax1.legend([Line2D([], [], marker="o", color=GREY, linestyle="none", markersize=4.5),
                Line2D([], [], marker="o", color=GREY, markerfacecolor="white", linestyle="none", markersize=4.5),
                Line2D([], [], marker="x", color="#8a8984", linestyle="none", markersize=5)],
               ["$\\lambda$ 1.00", "$\\lambda$ 1.40", "law"], frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.27), ncol=3, handletextpad=0.2, columnspacing=1.0)

    single_seed_note = []
    for c in contrast:
        if not c["steps"]:
            continue
        ks = ([] if "x_at" in c else [0]) + sorted(c["steps"])
        ys = ([] if "x_at" in c else [0.0]) + [c["steps"][k]["y"] for k in sorted(c["steps"])]
        lo = ([] if "x_at" in c else [0.0]) + [max(0.0, c["steps"][k]["y"] - c["steps"][k]["lo"]) for k in sorted(c["steps"])]
        hi = ([] if "x_at" in c else [0.0]) + [max(0.0, c["steps"][k]["hi"] - c["steps"][k]["y"]) for k in sorted(c["steps"])]
        ax2.errorbar(ks, ys, yerr=[lo, hi], color=c["color"], linestyle=("none" if "x_at" in c else c["ls"]),
                     linewidth=1.2, elinewidth=0.8, capsize=1.5, zorder=3)
        nmax = max(len(t["have"]) for t in c["steps"].values())
        for k in sorted(c["steps"]):
            full = len(c["steps"][k]["have"]) >= 3
            ax2.plot([k], [c["steps"][k]["y"]], marker=c["marker"], markersize=4.5, color=c["color"],
                     markerfacecolor=(c["color"] if full else "white"), zorder=4)
        if any(len(c["steps"][k]["have"]) < 3 for k in c["steps"]):
            single_seed_note.append(c["label"] + ": " + ", ".join(
                f"step {k} {len(c['steps'][k]['have'])} seed(s) ({', '.join(c['steps'][k]['have'])})" for k in sorted(c["steps"])))
        ax2.annotate(c["label"], (ks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                     color=c["color"], va="center",
                     bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    pending = [c["label"] for c in contrast if not c["steps"]]
    if ref is not None:
        ax2.axhline(ref, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax2.annotate("teacher", (0, ref), xytext=(0, 3), textcoords="offset points", fontsize=8, color=GREY,
                     ha="left", va="bottom")
    if pending:
        ax2.text(0.03, 0.62, "pending: " + ", ".join(pending), transform=ax2.transAxes, fontsize=8, color=GREY,
                 va="top")
    ax2.axhline(0, color="#9a9994", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax2.set_xlim(-2, 66)
    ax2.set_xticks([0, 14, 28, 42, 52])
    ax2.set_xticklabels(["0", "14", "28", "42", "SFT\nend"])
    ax2.axvline(47, color="#c9c8c4", linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)
    ax2.set_ylim(-0.02, max(0.25, (ref or 0) + 0.04))
    ax2.set_xlabel("training step")
    ax2.set_ylabel("transmission")
    ax2.set_title("(b) objective at the smallest $q_0$ (F3)", loc="left")
    fig.tight_layout(w_pad=1.0)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_rate_law.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_tables(pts, contrast, law, ref, single_seed_note)
    n_y = sum(1 for p in pts if p["y"] and p.get("in_panel"))
    print(f"fig1 v4: panel (a) {n_y} of {sum(1 for p in pts if p.get('in_panel'))} points with data; panel (b) "
          f"{sum(1 for c in contrast if c['steps'])} of {len(contrast)} arms; results/figure_sources.md")

def write_tables(pts, contrast, law, ref, single_seed_note=()):
    g = lambda v: "-" if v is None else f"{v:+.4f}"
    with (RES / "fig1_points.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "pretty", "q0", "q0_lo", "q0_hi", "q0_k", "q0_n", "q0_source", "lambda", "transmission",
                    "lo", "hi", "base_share", "per_seed", "law_prediction", "status"])
        for p in pts:
            q = p["q"] or {}
            for lam in ("1.00", "1.40"):
                t = p["y"].get(lam, {})
                w.writerow([p["key"], p["pretty"], q.get("q0"), q.get("lo"), q.get("hi"), q.get("k"), q.get("n"),
                            q.get("src"), lam, t.get("y"), t.get("lo"), t.get("hi"), t.get("base_share"),
                            json.dumps(t.get("per_seed", {})), (p.get("law_pred") or {}).get(lam),
                            "descriptive" if t else "pending"])
    with (RES / "fig1_contrast.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "step", "transmission", "lo", "hi", "per_seed"])
        for c in contrast:
            for k, t in sorted(c["steps"].items()):
                w.writerow([c["key"], k, t["y"], t["lo"], t["hi"], json.dumps(t["per_seed"])])
    L = ["## Figure 1 v4: on-policy reverse-KL distillation learns at the student's rate (auto-generated by "
         "`scripts/fig_fig1_points.py`; do not edit by hand)", "",
         "File: `out/figures/fig1_rate_law.{pdf,png}`. Every number is descriptive; the E3, FKL and K1 locks are "
         "scored by their owners. Tables: `results/fig1_points.csv`, `results/fig1_contrast.csv`.", "",
         "### Panel (a): transmission at step 42 against the starting share q0", "",
         "- q0: share of the base's own T = 1 samples (rollout rendering) showing the teacher's behaviour; Wilson 95% "
         "interval over samples (not clustered). A share of 0 is drawn on the '0' slot.",
         "- y: seed-mean sampled share of the behaviour at step 42 minus the base's share on the same sampler (T = 1, 4 "
         "samples per eval prompt, feasible prompts); interval: the E3 lock's prompt-and-seed bootstrap (10,000, numpy "
         "seed 0). Filled: lambda 1.00; hollow: lambda 1.40.",
         "- Law: results/E3_law_predictions.json (" + (f"c = {law['c']:.1f}, calibrated on the arm R item-8 "
                                                              f"probe shift only; {law.get('stage', '')}" if law else "absent") + ").", "",
         "| point | q0 [Wilson 95%] (k / n; source) | y, lambda 1.00 [95%] | y, lambda 1.40 [95%] | base share | law Tr, 1.00 / 1.40 | missing |",
         "|---|---|---|---|---|---|---|"]
    for p in pts:
        q = p["q"]
        qs = "-" if not q else f"{q['q0']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}] ({q['k']} / {q['n']}; `{q['src']}`)"
        y1, y4 = p["y"].get("1.00"), p["y"].get("1.40")
        fy = lambda t: "-" if not t else f"{g(t['y'])} [{g(t['lo'])}, {g(t['hi'])}]"
        bs = next((f"{t['base_share']:.4f}" for t in (y1, y4) if t and "base_share" in t), "-")
        lp = p.get("law_pred") or {}
        L.append(f"| {p['pretty']}{' (dagger: ' + p['dagger'] + ')' if p.get('dagger') else ''} | {qs} | {fy(y1)} | "
                 f"{fy(y4)} | {bs} | {g(lp.get('1.00'))} / {g(lp.get('1.40'))} | "
                 f"{', '.join(sorted(set(p.get('missing', [])))) or '-'} |")
    L += ["", "Per-seed sampled shares at step 42 (the template case: training-rollout share, steps 29-42):", ""]
    for p in pts:
        for lam, t in p["y"].items():
            L.append(f"- {p['label'].format(pct='')} ({p['key']}), lambda {lam}: " + ", ".join(f"s{s} {v:.4f}" for s, v in sorted(t.get("per_seed", {}).items())))
    lrp = next((p for p in pts if p["key"] == "LR" and p.get("by_step")), None)
    lr_note = ""
    if lrp:
        L += ["", "lr 1e-5 arm, per seed at every kept checkpoint (sampled share; base share 0.000; locked reading in "
              ", learning-rate section):", "",
              "| step | " + " | ".join(f"s{sd}" for sd in SEEDS) + " | mean minus base |", "|---|---|---|---|---|"]
        for k, t in sorted(lrp["by_step"].items()):
            L.append(f"| {k} | " + " | ".join(f"{t['per_seed'].get(sd, float('nan')):.3f}" for sd in SEEDS)
                     + f" | {g(t['y'])} |")
        mx = max(((v, sd, k) for k, t in lrp["by_step"].items() for sd, v in t["per_seed"].items()), default=None)
        if mx:
            lr_note = (f" The lr 1e-5 point is a locked miss whose per-seed values swing between checkpoints (largest: "
                       f"{mx[0]:.2f} for seed {mx[1]} at step {mx[2]}; per seed at step 42: "
                       + ", ".join(f"{v:.3f}" for _, v in sorted(lrp['by_step'].get(42, {}).get('per_seed', {}).items()))
                       + ").")
    L += ["", "### Panel (b): objective at fixed q0 (F3 pair A)", "",
          "- Seeds per step: " + ("; ".join(single_seed_note) + "." if single_seed_note else "all seeds at every step.")
          + (" The weights of F3 pair A lambda 1.00 seeds 1002 and 1003 at steps 14 and 28 were deleted by the storage "
             "cleanup (WEIGHTS_DELETED_20261001.txt), so the OPD curve shows seed 1001 only at those steps (hollow "
             "markers; prompt-only interval). A matched 3-seed rerun with KEEP_INTERMEDIATE_CKPT=1 is requested with "
             "the FKL / K1 batch." if any(n.startswith("OPD") for n in single_seed_note) else ""),
          "- SeqKD (K1) is one checkpoint per seed (end of its SFT), drawn at 'SFT end'. Its samples come from "
          "the sampler (eval_seqkd_sampled.py: T = 1, 4 per eval prompt, rollout rendering, vLLM seed "
          "20261002) and its base share from the same sampler's B_clean reference; all samples are labelled here with the "
          "same B1 validator as the E3 sampler sets (the locator gives B_clean 0.00125, the validator 0.000; "
          "teacher 0.1975 both). OPD / OPDre / FKL: E3 sampler (eval_sample.sh, vLLM seed 0); OPDre replaces "
          "the original OPD curve once its samples exist (spec 09-FKL_sample).",
          "- y as in (a) at each kept checkpoint (step 0 = the base, 0 by definition). Teacher reference: "
          + ("-" if ref is None else f"{ref:+.4f} (teacher's sampled share minus B_clean's on the E3 sampler)") + ".",
          "- Expected sample names under outputs/E3/samples: " + "; ".join(f"{c['label']} ({c['key']}): `{c['pattern']}`" for c in CONTRAST)
          + ", base `base_clean`, teacher `teacher`.", "",
          "| arm | step | y [95%] | per seed |", "|---|---|---|---|"]
    for c in contrast:
        if not c["steps"]:
            L.append(f"| {c['label']} ({c['key']}) | - | pending | - |")
        for k, t in sorted(c["steps"].items()):
            L.append(f"| {c['label']} ({c['key']}) | {k} | {g(t['y'])} [{g(t['lo'])}, {g(t['hi'])}] | "
                     + ", ".join(f"s{s} {v:.4f}" for s, v in sorted(t["per_seed"].items())) + " |")
    L += ["", "### Draft caption (stands alone; fill the numbers only after the locks are scored)", "",
          "Figure 1. On-policy reverse-KL distillation learns at the student's rate. (a) Transmission of the teacher's "
          "injected error after 42 OPD steps (the student's share of the error on T = 1 samples of the 200 or 212 eval "
          "prompts, 4 per prompt, minus the starting model's share) against the student's starting share q0 of that "
          "error on its own T = 1 samples (log scale; q0 = 0 drawn in the slot at the left). Points: the constructed "
          "pairs F3 (ESCI, Qwen3) and D1-inj (Fashion, Llama), E3's bases injected at the measured q0, F3 at a "
          "tenfold learning rate, and E3 arm R, whose base never closes early at item 7 but runs past item 8 (the "
          "coupled route). Filled markers: lambda = 1.00; hollow: lambda = 1.40. Grey x / +: the step-42 prediction of "
          "a one-constant law, calibrated on arm R's item-8 probe shift only, at lambda 1.00 / 1.40. Error bars: 95% "
          "prompt-and-seed bootstrap intervals; small dots: single seeds; horizontal bars: Wilson intervals of q0."
          + lr_note + " Not shown, reported in the "
          "appendix: the correction of arm R's item-8 runaway (a different behaviour) and the template case (training "
          "rollouts). (b) At F3's q0, transmission over training for reverse-KL OPD, on-policy forward KL and SeqKD; "
          "dashed: the teacher's own share minus the base's. Hollow markers: fewer than three seeds at that step."]
    sys.path.insert(0, str(REPO / "scripts"))
    from fig_common import write_sources
    write_sources("fig1", L)

if __name__ == "__main__":
    main()
