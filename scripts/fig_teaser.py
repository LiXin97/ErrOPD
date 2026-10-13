#!/usr/bin/env python3
"""Draw Figure 1."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "out" / "figures"

C_OPD, C_FKL, C_SEQ, C_TEACH, C_LAW = "#0072B2", "#D55E00", "#009E73", "#000000", "#8C8C8C"
C_BUDGET = "#EFEFEF"
TEXTWIDTH_IN = 455.24 / 72.27
COLUMNWIDTH_IN = 219.08 / 72.27
HEIGHT_IN = 1.95
PCT = 100.0
STAY_BOUND = 0.002

def load():
    con = {}
    with (RES / "fig1_contrast.csv").open() as fh:
        for r in csv.DictReader(fh):
            con.setdefault(r["arm"], {})[int(r["step"])] = {
                "y": float(r["transmission"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
                "per_seed": json.loads(r["per_seed"])}
    fk = json.load((RES / "FKL_scored.json").open())
    teach_q = fk["teacher_share"] - fk["FKL"]["base_share"]
    assert abs(teach_q - 155 / 800) < 1e-12, teach_q
    k1 = json.load((RES / "K1_scored.json").open())
    teach_k1 = k1["refs"]["teacher_t4b_k1"]["k1_b1"] - k1["refs"]["base_b17_clean"]["k1_b1"]
    long = json.load((RES / "LONG210b_scored.json").open())["seeds"]
    fl = json.load((RES / "FKL_L_scored.json").open())
    pts = {}
    with (RES / "fig1_points.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["lambda"] == "1.00" and r["transmission"]:
                pts[r["key"]] = {k: (float(r[k]) if r[k] not in ("", None) else None)
                                 for k in ("q0", "q0_lo", "q0_hi", "transmission", "lo", "hi", "law_prediction")}
    arm = json.load((RES / "RX_scored.json").open())["A"]["armR_1.00"]
    assert arm["status"] == "scored" and len(arm["seeds"]) == 5 and arm["base"] == 0.0, arm
    pts["R7"].update({"transmission": arm["Tr"], "lo": arm["Tr_ci95"][0], "hi": arm["Tr_ci95"][1]})
    check_table1(pts)
    return con, teach_q, teach_k1, long, fl, pts

def check_table1(pts):
    pass
    sys.path.insert(0, str(REPO / "scripts"))
    from fig_tables import pct
    lines = (REPO / "out" / "tables" / "tab_already.tex").read_text().splitlines()
    conv = (REPO / "out" / "tables" / "app_conventions.tex").read_text().splitlines()
    rows = {"E2": "Qwen3, 2\\% injected", "E5": "Qwen3, 5\\% injected", "R7": "Qwen3, coupled route"}
    for key, label in rows.items():
        i = [k for k, ln in enumerate(lines) if ln.startswith(label + " &")]
        assert len(i) == 1, (label, i)
        cell = lines[i[0]].split("&")[2]
        m = re.fullmatch(r"\s*([0-9.]+) \{\\scriptsize\[([0-9.]+), ([0-9.]+)\]\}\s*", cell)
        assert m, (label, cell)
        mean, lo, hi = m.groups()
        got = (pct(pts[key]["transmission"]), pct(pts[key]["lo"]), pct(pts[key]["hi"]))
        assert got == (mean, lo, hi), f"Fig. 1c {key}: plotted {got}, tab:already prints {(mean, lo, hi)}"
    note = "(the original runs give 0)"
    assert any(note in ln and "Qwen3 and Llama" in ln for ln in conv), "Table 1's note on the original runs changed"
    assert pts["F3A"]["transmission"] == 0.0 and pts["D1injA"]["transmission"] == 0.0

TEASER_STYLES = {
    "v1": dict(H=1.72, band="#F3F3F3", band_txt="#8A8A8A", teach="#6E6E6E", seq=C_SEQ, seq_lbl="SeqKD", main_lw=1.7,
               main_ms=3.8, seed_lw=0.65, seed_alpha_esc=0.55, seed_alpha_stay=0.35, ylabel=r"transmission (points)",
               yticks=[0, 10, 20, 30], ylim=(-7.6, 31.5)),
    "v2": dict(H=1.58, band="#F5F5F5", band_txt="#9A9A9A", teach="#8C8C8C", seq="#8C8C8C", seq_lbl="SeqKD", main_lw=1.8,
               main_ms=3.8, seed_lw=0.6, seed_alpha_esc=0.5, seed_alpha_stay=0.3, ylabel=r"transmission (pp)",
               yticks=[0, 10, 20], ylim=(-7.6, 29.5)),
}
TEASER_DEFAULT = "v2"

def teaser_single(plt, Rectangle, con, teach_q, teach_k1, long, esc_wide, st):
    pass
    W, H = COLUMNWIDTH_IN, st["H"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.38 / W, 0.34 / H, (W - 0.43) / W, (H - 0.34 - 0.06) / H])
    y0, y1 = st["ylim"]
    ax.set_ylim(y0, y1)
    ax.set_yticks(st["yticks"])
    ax.set_ylabel(st["ylabel"])
    ax.axhline(0, color="#CCCCCC", linewidth=0.5, zorder=0)
    SEQ_X = 236
    ax.add_patch(Rectangle((0, y0), 42, y1 - y0, facecolor=st["band"], edgecolor="none", zorder=0))
    ax.text(21, y1 - 0.6, "42 steps", ha="center", va="top", fontsize=9, color=st["band_txt"])
    dash = (0, (4, 2.5))
    ax.plot([-4, 214], [PCT * teach_q] * 2, color=st["teach"], linewidth=0.7, linestyle=dash, zorder=1)
    ax.plot([SEQ_X - 7, SEQ_X + 7], [PCT * teach_k1] * 2, color=st["teach"], linewidth=0.7, linestyle=dash, zorder=1)
    ax.text(64, PCT * teach_q + 0.8, "teacher", ha="center", va="bottom", fontsize=9, color=st["teach"])
    esc, stay = [], []
    for sd, rec in sorted(long.items()):
        ks = [0] + sorted(int(k) for k in rec["shares"])
        ys = [0.0] + [PCT * rec["shares"][str(k)] for k in ks[1:]]
        (esc if rec["escaped"] else stay).append(sd)
        ax.plot(ks, ys, color=C_OPD, linewidth=st["seed_lw"],
                alpha=st["seed_alpha_esc"] if rec["escaped"] else st["seed_alpha_stay"], zorder=2)
    assert esc == esc_wide
    eb = dict(elinewidth=0.7, capsize=1.6, capthick=0.7)
    for arm, col, mk in (("OPDre", C_OPD, "D"), ("FKL", C_FKL, "o")):
        ks = [0] + sorted(con[arm])
        ys = [0.0] + [PCT * con[arm][k]["y"] for k in ks[1:]]
        lo = [0.0] + [PCT * (con[arm][k]["y"] - con[arm][k]["lo"]) for k in ks[1:]]
        hi = [0.0] + [PCT * (con[arm][k]["hi"] - con[arm][k]["y"]) for k in ks[1:]]
        ax.errorbar(ks, ys, yerr=[lo, hi], color=col, linewidth=st["main_lw"], marker=mk, markersize=st["main_ms"],
                    zorder=4, **eb)
    k1 = con["K1"][max(con["K1"])]
    ax.errorbar([SEQ_X], [PCT * k1["y"]], yerr=[[PCT * (k1["y"] - k1["lo"])], [PCT * (k1["hi"] - k1["y"])]],
                color=st["seq"], marker="s", markersize=3.2, zorder=4, **eb)
    ax.axvline(222, color="#CCCCCC", linewidth=0.5, linestyle=(0, (1, 1.5)), zorder=0)
    ax.text(49, 16.4, "forward-KL OPD", color=C_FKL, fontsize=9, ha="left", va="center")
    ax.text(1, -3.6, "reverse-KL OPD", color=C_OPD, fontsize=9, ha="left", va="center")
    stay_max = max(v for sd in stay for v in long[sd]["shares"].values())
    assert 0 < stay_max < STAY_BOUND == 0.002, stay_max
    ax.text(210, -3.6, rf"{len(stay)} of {len(long)} stay below {PCT * STAY_BOUND:g}\%", color=C_OPD, fontsize=9,
            ha="right", va="center", alpha=0.85)
    ax.text(138, 26.0, rf"{len(esc)} of {len(long)} seeds acquire it", color=C_OPD, fontsize=9, ha="center",
            va="center", alpha=0.85)
    ax.text(SEQ_X + 9, PCT * k1["hi"] + 1.0, st["seq_lbl"], color=st["seq"], fontsize=9, ha="right", va="bottom")
    ax.set_xlim(-6, 246)
    ax.set_xticks([0, 42, 84, 126, 168, 210])
    ax.set_xlabel("training step")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_teaser.{ext}", dpi=300)
    plt.close(fig)

def main():
    con, teach_q, teach_k1, long, fl, pts = load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.linewidth": 0.4, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "pdf.fonttype": 42})
    W, H = TEXTWIDTH_IN, HEIGHT_IN
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.36 / H, (H - 0.20) / H
    hgt = top - bot

    def axes_at(x0_in, w_in):
        return fig.add_axes([x0_in / W, bot, w_in / W, hgt])

    axA = axes_at(0.43, 2.62)
    axB = axes_at(3.16, 1.02)
    axC = axes_at(4.62, 1.48)
    YLIM = (-7.6, 33.0)
    for ax in (axA, axB, axC):
        ax.set_ylim(*YLIM)
        ax.set_yticks([0, 10, 20, 30])
        ax.axhline(0, color="#BBBBBB", linewidth=0.5, zorder=0)
    axB.set_yticklabels([])
    axA.set_ylabel(r"transmission (points)")

    eb = dict(elinewidth=0.7, capsize=1.6, capthick=0.7)

    def panel_a(axA, title):
        SEQ_X = 236
        axA.add_patch(Rectangle((0, YLIM[0]), 42, YLIM[1] - YLIM[0], facecolor=C_BUDGET, edgecolor="none", zorder=0))
        axA.text(21, 31.8, "42 steps", ha="center", va="top", fontsize=9, color="#555555")
        axA.plot([-4, 214], [PCT * teach_q] * 2, color=C_TEACH, linewidth=0.8, linestyle=(0, (4, 2.5)), zorder=1)
        axA.plot([SEQ_X - 7, SEQ_X + 7], [PCT * teach_k1] * 2, color=C_TEACH, linewidth=0.8, linestyle=(0, (4, 2.5)),
                 zorder=1)
        axA.text(64, PCT * teach_q + 0.7, "teacher", ha="center", va="bottom", fontsize=9)
        esc, stay = [], []
        for sd, rec in sorted(long.items()):
            ks = [0] + sorted(int(k) for k in rec["shares"])
            ys = [0.0] + [PCT * rec["shares"][str(k)] for k in ks[1:]]
            (esc if rec["escaped"] else stay).append(sd)
            axA.plot(ks, ys, color=C_OPD, linewidth=0.75, alpha=0.75 if rec["escaped"] else 0.55, zorder=2,
                     marker="o", markersize=1.8, markeredgewidth=0)
        for arm, col, mk, ls in (("OPDre", C_OPD, "D", "-"), ("FKL", C_FKL, "o", "-")):
            ks = [0] + sorted(con[arm])
            ys = [0.0] + [PCT * con[arm][k]["y"] for k in ks[1:]]
            lo = [0.0] + [PCT * (con[arm][k]["y"] - con[arm][k]["lo"]) for k in ks[1:]]
            hi = [0.0] + [PCT * (con[arm][k]["hi"] - con[arm][k]["y"]) for k in ks[1:]]
            axA.errorbar(ks, ys, yerr=[lo, hi], color=col, linewidth=1.3, linestyle=ls, marker=mk, markersize=3.2,
                         zorder=4, **eb)
        k1 = con["K1"][max(con["K1"])]
        axA.errorbar([SEQ_X], [PCT * k1["y"]], yerr=[[PCT * (k1["y"] - k1["lo"])], [PCT * (k1["hi"] - k1["y"])]], color=C_SEQ, marker="s",
                     markersize=3.6, zorder=4, **eb)
        axA.axvline(222, color="#BBBBBB", linewidth=0.5, linestyle=(0, (1, 1.5)), zorder=0)
        axA.text(21, 26.0, "forward KL", color=C_FKL, fontsize=9, ha="center", va="center")
        axA.text(1, -3.6, "reverse-KL OPD", color=C_OPD, fontsize=9, ha="left", va="center")
        stay_max = max(v for sd in stay for v in long[sd]["shares"].values())
        assert 0 < stay_max < STAY_BOUND == 0.002, stay_max
        axA.text(210, -3.6, rf"{len(stay)} of {len(long)} stay below {PCT * STAY_BOUND:g}\%", color=C_OPD, fontsize=9, ha="right",
                 va="center")
        axA.text(150, 26.8, rf"{len(esc)} of {len(long)} reverse-KL seeds escape", color=C_OPD, fontsize=9, ha="center",
                 va="center")
        axA.text(SEQ_X, PCT * k1["hi"] + 1.0, "SeqKD\n(ref.)", color=C_SEQ, fontsize=9, ha="center", va="bottom",
                 linespacing=0.95)
        axA.set_xlim(-6, 246)
        axA.set_xticks([0, 42, 84, 126, 168, 210])
        axA.set_xlabel("training step")
        axA.set_title(title, loc="left", pad=3)
        return esc

    esc = panel_a(axA, r"(a) Qwen3, ESCI")

    tl = fl["teacher_share"] - fl["base_share"]
    axB.plot([-3, 45], [PCT * tl] * 2, color=C_TEACH, linewidth=0.8, linestyle=(0, (4, 2.5)), zorder=1)
    for arm, col, mk in (("OPDre_L", C_OPD, "D"), ("FKL_L", C_FKL, "o")):
        ks = [0, 14, 28, 42]
        ys = [0.0] + [PCT * fl[arm][str(k)]["Tr"] for k in ks[1:]]
        lo = [0.0] + [PCT * (fl[arm][str(k)]["Tr"] - fl[arm][str(k)]["Tr_ci95"][0]) for k in ks[1:]]
        hi = [0.0] + [PCT * (fl[arm][str(k)]["Tr_ci95"][1] - fl[arm][str(k)]["Tr"]) for k in ks[1:]]
        axB.errorbar(ks, ys, yerr=[lo, hi], color=col, linewidth=1.3, marker=mk, markersize=3.2, zorder=4, **eb)
    axB.set_xlim(-4, 46)
    axB.set_xticks([0, 14, 28, 42])
    axB.set_xlabel("training step")
    axB.set_title(r"(b) Llama, Fashion", loc="left", pad=3)

    ZERO = 1.0e-4
    LAW_DODGE = 1.45
    xs = lambda q: ZERO if q <= 0 else q
    axC.set_xscale("log")
    axC.set_xlim(PCT * 6.0e-5, PCT * 0.1)
    axC.set_xticks([PCT * ZERO, PCT * 1e-3, PCT * 1e-2, PCT * 1e-1])
    axC.set_xticklabels([r"0", r"0.1", r"1", r"10"])
    axC.minorticks_off()
    for xb in (1.75e-4, 1.95e-4):
        axC.plot([PCT * xb * 0.93, PCT * xb * 1.07], [YLIM[0] - 0.4, YLIM[0] + 1.2], color="#555555", linewidth=0.6,
                 clip_on=False, zorder=6)
    order = [("F3A", "D"), ("D1injA", "D"), ("E2", "o"), ("E5", "o"), ("R7", "s")]
    for key, mk in order:
        p = pts[key]
        x = PCT * xs(p["q0"])
        face = "white" if key == "R7" else C_OPD
        axC.errorbar([x], [PCT * p["transmission"]],
                     yerr=[[PCT * (p["transmission"] - p["lo"])], [PCT * (p["hi"] - p["transmission"])]],
                     color=C_OPD, marker=mk, markersize=3.6, markerfacecolor=face, linestyle="none", zorder=4, **eb)
        if p["law_prediction"] is not None:
            axC.plot([x * LAW_DODGE], [PCT * p["law_prediction"]], marker="x", markersize=4.6, color=C_LAW,
                     markeredgewidth=1.0, linestyle="none", zorder=5)
    inj = [xs(pts[k]["q0"]) for k in ("D1injA", "F3A")]
    assert inj[0] == ZERO and 5e-4 < inj[1] < 1e-3 and all(pts[k]["transmission"] == 0.0 for k in ("D1injA", "F3A"))
    inj = [PCT * x for x in inj]
    RAIL = -2.9
    for xq in inj:
        axC.plot([xq, xq], [-1.4, RAIL], color=C_OPD, linewidth=0.6, solid_capstyle="butt", zorder=3)
    axC.plot(inj, [RAIL, RAIL], color=C_OPD, linewidth=0.6, solid_capstyle="projecting", zorder=3)
    axC.text(inj[1] * 1.3, RAIL, "injected teacher", fontsize=9, color=C_OPD, ha="left", va="center")
    axC.text(PCT * 1.4e-4, PCT * pts["R7"]["transmission"] + 0.4, "coupled route", fontsize=9, color=C_OPD, ha="left",
             va="center")
    axC.text(PCT * 1.9e-2, 25.2, "student already errs", fontsize=9, color=C_OPD, ha="center", va="center")
    axC.text(PCT * pts["E5"]["q0"] * LAW_DODGE * 1.35, PCT * pts["E5"]["law_prediction"], "law", fontsize=9, color=C_LAW,
             ha="left", va="center")
    axC.set_xlabel(r"student's starting error rate (\%)")
    axC.set_title(r"(c) reverse KL at step 42", loc="left", pad=3)

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_teaser_wide.{ext}", dpi=300)
    plt.close(fig)
    style = TEASER_STYLES[os.environ.get("FIG1_STYLE", TEASER_DEFAULT)]
    teaser_single(plt, Rectangle, con, teach_q, teach_k1, long, esc, style)
    write_sources(con, teach_q, teach_k1, long, fl, pts, esc)
    print("fig1_teaser: written out/figures/fig1_teaser.{pdf,png}; sources in results/figure_sources.md")

def write_sources(con, teach_q, teach_k1, long, fl, pts, esc):
    sys.path.insert(0, str(REPO / "scripts"))
    from fig_figs import fmt
    f4 = lambda v: fmt(v) if v >= 0 else "-" + fmt(-v)
    L = ["## Figure 1 (teaser, `out/figures/fig1_teaser.{pdf,png}`; auto-generated by `scripts/fig_teaser.py`;"
         " do not edit by hand)", "",
         "Replaces `fig1_rate_law.pdf` in the paper (2026-10-01, ). Reads committed files only. Transmission = the"
         " student's share of T = 1 samples that close the list one item early minus its starting checkpoint's share on"
         " the same sampler; step 0 = 0 by definition. Intervals: 95% prompt-and-seed bootstrap as scored by the owners."
         " Axes in percent of samples (since 2026-10-02): the figure draws 100 x every share, transmission and q0 listed"
         " below; the values here are the source files' fractions.",
         "", "### (a) Qwen3 (F3 pair A, ESCI, E3 sampler)", "",
         f"- Teacher reference (dashed, steps 0-210): {f4(teach_q)} (155/800; FKL_scored.json teacher_share minus"
         f" base_share, on the E3 sampler). Over the SeqKD slot: {f4(teach_k1)} (K1_scored.json, teacher_t4b_k1 minus base_b17_clean on"
         " the SeqKD sampler).", "",
         "| arm (source) | step | transmission [95%] | per seed |", "|---|---|---|---|"]
    for arm, name in (("OPDre", "OPD reverse KL, 42-step rerun (fig1_contrast.csv)"),
                      ("FKL", "on-policy forward KL (fig1_contrast.csv)"),
                      ("K1", "SeqKD, end of 70 SFT steps from the pre-trained student, drawn in its own slot (fig1_contrast.csv)")):
        for k, t in sorted(con[arm].items()):
            L.append(f"| {name} | {k} | {f4(t['y'])} [{f4(t['lo'])}, {f4(t['hi'])}] | "
                     + ", ".join(f"s{s} {f4(v)}" for s, v in sorted(t["per_seed"].items())) + " |")
    L += ["", f"Thin lines: 8 OPD seeds of LONG210 (1001-1003) and LONG210b (1004-1008) (LONG210b_scored.json, sampled"
          f" share; base 0.000 on the E3 sampler); these are separate 210-step runs, not the 42-step rerun. Escaped"
          f" (share >= 0.10 at a kept checkpoint): {', '.join(esc)} ({len(esc)} of {len(long)}).", "",
          "| seed | 42 | 84 | 126 | 168 | 210 |", "|---|---|---|---|---|---|"]
    for sd, rec in sorted(long.items()):
        L.append(f"| {sd} | " + " | ".join(f4(rec['shares'][k]) for k in ("42", "84", "126", "168", "210")) + " |")
    L += ["", "### (b) Llama (D1-inj pair A, Fashion; FKL_L_scored.json)", "",
          f"- Teacher reference: {f4(fl['teacher_share'] - fl['base_share'])} (teacher_share minus base_share).", "",
          "| arm | step | transmission [95%] | per seed |", "|---|---|---|---|"]
    for arm, name in (("OPDre_L", "OPD reverse KL, rerun"), ("FKL_L", "on-policy forward KL")):
        for k in ("14", "28", "42"):
            t = fl[arm][k]
            L.append(f"| {name} | {k} | {f4(t['Tr'])} [{f4(t['Tr_ci95'][0])}, {f4(t['Tr_ci95'][1])}] | "
                     + ", ".join(f4(v) for v in t["share_per_seed"]) + " |")
    L += ["", "### (c) plain OPD (lambda 1.00) at step 42 against q0 (fig1_points.csv)", "",
          "- q0: the starting checkpoint's share on its own T = 1 training-prompt samples; 0 drawn in the slot left of"
          " the break. Its Wilson 95% interval is listed in the table below and is not drawn (since 2026-10-02): the"
          " only bars in the panel are the vertical 95% intervals of the step-42 transmission. Grey x: the one-constant"
          " rate law's step-42 prediction (E3_law_predictions.json, c = 79.4, calibrated on arm R's item-8 shift only),"
          " drawn at the predicted value but set just right of its point (x 1.45 on the log axis, about 5 pt; fig_lr"
          " offsets its crosses too, to the left; at the same x the cross hid the E2 and E5 points) and labelled 'law'"
          " beside the E5 cross at its height; no prediction"
          " exists for D1-inj (Llama) or for arm R item 7 (coupled route, which the per-decision law does not model).",
          "- The two filled diamonds are the two injected-teacher pairs (Llama D1-inj pair A in the 0 slot, Qwen3 F3"
          " pair A at q0 = 1/1424); both hang from one rail in the strip below 0 that runs into the single label"
          " 'injected teacher'.",
          "- Coupled route: the 5-seed arm, RX_scored.json A.armR_1.00 (Tr 0.08975, as Table 1's 9.0 [4.4, 14.2] in percent);"
          " q0 from the fig1_points.csv row of the same starting checkpoint. The clean-teacher control (0.00025) is not"
          " drawn: it lies on the zero line in the slot the Llama point occupies; it is in Table 1 and fig:arms."
          " Every point with a Table 1 cell is asserted equal to that cell (Table 1's percent display, fig_tables.pct).",
          "- lambda 1.40 is not drawn (it would set the y range to 1.0); it is in tab:e3 and Sec. 5.", "",
          "| point | q0 [Wilson] | transmission [95%] | law |", "|---|---|---|---|"]
    names = {"F3A": "injected teacher, Qwen3 (F3 pair A)", "D1injA": "injected teacher, Llama (D1-inj pair A)",
             "E2": "student already errs (E3 arm E, 2% base)", "E5": "student already errs (E3 arm E, 5% base)",
             "R7": "coupled route (5 seeds, RX_scored.json A.armR_1.00; hollow square)"}
    for key in ("F3A", "D1injA", "E2", "E5", "R7"):
        p = pts[key]
        law = "-" if p["law_prediction"] is None else f4(p["law_prediction"])
        L.append(f"| {names[key]} | {f4(p['q0'])} [{f4(p['q0_lo'])}, {f4(p['q0_hi'])}] | {f4(p['transmission'])}"
                 f" [{f4(p['lo'])}, {f4(p['hi'])}] | {law} |")
    sys.path.insert(0, str(REPO / "scripts"))
    from fig_common import write_sources as ws
    ws("fig1_teaser", L)

if __name__ == "__main__":
    main()
