#!/usr/bin/env python3
"""Draw the paper figures from the frozen results (fig_dynamics, fig_rate, fig_when, fig_greedy, fig_long, ...)."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "out" / "figures"
C_OPD, C_FKL, C_SEQ, C_TEACH = "#0072B2", "#D55E00", "#009E73", "#000000"
C_16K, C_32K = "#6FAED8", C_OPD
C_GREY = "#8C8C8C"
COLW = 218.27 / 72.27
TEXTW = 455.24 / 72.27
FS = 9
PCT = 100.0
SRC = []
ONLY = set()
CAPS = REPO / "out" / "tables"

sys.path.insert(0, str(REPO / "scripts"))
from fig_tables import pct

def caption(name):
    pass
    tex = "\n".join(ln for ln in (CAPS / f"{name}.tex").read_text().splitlines() if not ln.lstrip().startswith("%"))
    cap = tex.split("\\caption{", 1)[1]
    conv = "\n".join(ln for ln in (CAPS / "app_conventions.tex").read_text().splitlines() if not ln.lstrip().startswith("%"))
    return cap + "\n" + conv

def _cap_check(cap, phrase):
    pass
    if phrase not in cap:
        print(f"WARN caption/conventions no longer carry: {phrase!r}")

def md_rows(path, header_start):
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            rows = []
            for ln2 in lines[i + 2:]:
                if not ln2.startswith("|"):
                    break
                rows.append([c.strip() for c in ln2.strip().strip("|").split("|")])
            return rows
    raise KeyError(header_start)

def fmt(v, nd=4):
    pass
    half = lambda k: Decimal(repr(round(v, 10))).scaleb(k) % 1 == Decimal("0.5")
    while nd > 1 and half(nd):
        nd -= 1
    if v and float(f"{v:.{nd}f}") == 0:
        nd = 4
        while nd < 8 and (half(nd) or float(f"{v:.{nd}f}") != round(v, 10)):
            nd += 1
    return f"{v:.{nd}f}"

def style(plt):
    pass
    plt.rcParams.update({
        "text.usetex": True, "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
        "font.family": "serif", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "pdf.fonttype": 42,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.linewidth": 0.4,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "axes.titlelocation": "left", "axes.titlepad": 3})

def save(fig, name):
    if ONLY and name not in ONLY:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)

def fig_q35s(plt):
    from matplotlib.lines import Line2D
    rows = list(csv.DictReader((RES / "q35s_dynamics.csv").open()))
    scored = json.load((RES / "Q35S_scored.json").open())
    teacher = 0.2203
    tr = defaultdict(dict)
    for r in rows:
        tr[(r["size"], r["arm"], r["seed"])][int(r["step"])] = float(r["transmission"])
    W, H = COLW, 1.72
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.56 / H, (H - 0.17) / H
    x0, pw, gap = 0.36, 0.84, 0.07
    axes = [fig.add_axes([(x0 + i * (pw + gap)) / W, bot, pw / W, top - bot]) for i in range(3)]
    arms = (("opd1.00", C_OPD, "-", "o", C_OPD), ("opd1.40", C_OPD, (0, (3, 1.6)), "o", "white"),
            ("fkl1.00", C_FKL, "-", "s", C_FKL))
    SRC.extend(["### Qwen3.5 dynamics (fig_q35s_dynamics)", "",
                "Source: results/q35s_dynamics.csv (step 42 equals Q35S_scored.json); transmission = share minus the "
                "size's base share; teacher share 0.2203 (Q35S_verdict.md; drawn minus a base of 0).", "",
                "| size | arm | seed | step 14 | step 28 | step 42 |", "|---|---|---|---|---|---|"])
    for ax, (size, label) in zip(axes, (("0p8b", "0.8B"), ("2b", "2B"), ("4b", "4B"))):
        ax.axhline(teacher - scored["sizes"][size]["base_share"], color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
        ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
        for arm, col, ls, mk, face in arms:
            seeds = sorted({s for (sz, a, s) in tr if sz == size and a == arm})
            for s in seeds:
                d = tr[(size, arm, s)]
                ks = [0] + sorted(d)
                ys = [0.0] + [d[k] for k in sorted(d)]
                ax.plot(ks, ys, color=col, lw=0.75, ls=ls, alpha=0.9, zorder=3 if arm != "fkl1.00" else 2,
                        marker=mk, markersize=2.2, markerfacecolor=face, markeredgewidth=0.5)
                SRC.append(f"| {label} | {arm} | {s} | " + " | ".join(f"{d.get(k, float('nan')):+.4f}" for k in (14, 28, 42)) + " |")
        ax.set_xlim(-3, 45)
        ax.set_xticks([0, 14, 28, 42])
        ax.set_ylim(-0.03, 0.75)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_title(f"Qwen3.5-{label}", loc="left", pad=2)
    axes[0].set_ylabel("transmission")
    axes[1].set_xlabel("training step")
    d = tr[("0p8b", "opd1.40", "1002")]
    axes[0].annotate("one seed\nescapes", xy=(42, d[42]), xytext=(8, 0.56), fontsize=FS, color=C_OPD,
                     ha="left", va="center", linespacing=0.95,
                     arrowprops=dict(arrowstyle="-", color=C_OPD, lw=0.5, shrinkA=1, shrinkB=2))
    axes[2].text(44, 0.232, "teacher", fontsize=FS, ha="right", va="bottom")
    h = [Line2D([], [], color=C_OPD, lw=0.9, marker="o", markersize=2.6),
         Line2D([], [], color=C_OPD, lw=0.9, ls=(0, (3, 1.6)), marker="o", markersize=2.6, markerfacecolor="white"),
         Line2D([], [], color=C_FKL, lw=0.9, marker="s", markersize=2.6)]
    fig.legend(h, [r"reverse KL, $\lambda{=}1.00$", r"reverse KL, $\lambda{=}1.40$", "forward KL"], loc="lower center",
               bbox_to_anchor=(0.53, 0.0), ncol=3, frameon=False, handlelength=1.8, columnspacing=0.9,
               handletextpad=0.4)
    save(fig, "fig_q35s_dynamics")
    plt.close(fig)
    SRC.append("")

def d5_points():
    pass
    p = RES / "D5_verdict.md"
    after = lambda s: float(s.split("->")[1])
    before = lambda s: float(s.split("->")[0])
    P = {}
    for r in md_rows(p, "| lambda (seed) | MATH500 V | MATH500 trunc | AIME24 trunc"):
        m = re.match(r"^([\d.]+) \((\d+)\)$", r[0])
        if not m:
            continue
        a24, a25 = (float(x) for x in r[7].split("/"))
        P[(float(m.group(1)), m.group(2), "16K")] = (a24, a25, float(r[3]), float(r[4]))
    for r in md_rows(p, "| lambda (seed) | MATH500 trunc 16k -> 32k"):
        m = re.match(r"^([\d.]+) \((\d+)\)$", r[0])
        P[(float(m.group(1)), m.group(2), "32K")] = (after(r[5]), after(r[6]), after(r[2]), after(r[3]))
    for r in md_rows(p, "| lambda (seed) | MATH500 trunc | AIME24 trunc | AIME25 trunc | MATH500 A"):
        m = re.match(r"^([\d.]+) \((\d+)\)$", r[0])
        P[(float(m.group(1)), m.group(2), "32K")] = (after(r[5]), after(r[6]), after(r[2]), after(r[3]))
    t15 = {r[0]: r for r in md_rows(p, "| set | truncation | overall accuracy (A) |")}
    P[(1.5, "42", "32K")] = (after(t15["AIME24"][2]), after(t15["AIME25"][2]), after(t15["AIME24"][1]),
                             after(t15["AIME25"][1]))
    for r in md_rows(p, "| lambda | MATH500 trunc | AIME24 trunc | AIME25 trunc | MATH500 A | AIME24 A | AIME25 A |"):
        lam = float(r[0])
        P[(lam, "7", "16K")] = (before(r[5]), before(r[6]), before(r[2]), before(r[3]))
        P[(lam, "7", "32K")] = (after(r[5]), after(r[6]), after(r[2]), after(r[3]))
    for r in md_rows(p, "| seed | 16k | 32k |"):
        sd = r[0]
        for cap, cell in (("16K", r[1]), ("32K", r[2])):
            lo, hi = (float(x) for x in re.findall(r"0\.\d+", cell)[:2])
            for lam, v in ((1.0, lo), (2.0, hi)):
                a24, a25, _, _ = P[(lam, sd, cap)]
                assert abs((a24 + a25) / 2 - v) <= 0.0011, (lam, sd, cap, (a24 + a25) / 2, v)
    return P

def fig_math(plt):
    P = d5_points()
    lams = (1.0, 1.25, 1.5, 2.0)
    W, H = COLW, 1.80
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.36 / H, (0.36 + 0.96) / H
    axA = fig.add_axes([0.40 / W, bot, 1.12 / W, top - bot])
    axB = fig.add_axes([1.88 / W, bot, 1.08 / W, top - bot])
    SRC.extend(["### Math budget flip (fig_math_budget)", "",
                "Source:  (A3 table at 16K; A4 table, lambda 1.25 and 1.5 addenda and the seed-7 "
                "addendum at 32K), values as printed; AIME = mean of AIME24 and AIME25. Seeds: 42 / 7 / 13 at lambda 1.0 "
                "and 2.0, 7 / 13 at 1.25, 42 at 1.5. Line: seed mean; bar: range over seeds. Both axes in percent: the "
                "figure draws 100 x the fractions below.", "",
                "| cap | lambda | seed | AIME acc | AIME cut-off |", "|---|---|---|---|---|"])
    for cap, col, mk, face in (("16K", C_16K, "o", "white"), ("32K", C_32K, "o", C_32K)):
        for ax, k in ((axA, "acc"), (axB, "cut")):
            means, xs = [], []
            for lam in lams:
                v = []
                for (l, sd, c), (a24, a25, c24, c25) in sorted(P.items()):
                    if l == lam and c == cap:
                        val = (a24 + a25) / 2 if k == "acc" else (c24 + c25) / 2
                        v.append(val)
                        if ax is axA:
                            SRC.append(f"| {cap} | {lam} | {sd} | {(a24 + a25) / 2:.4f} | {(c24 + c25) / 2:.4f} |")
                if not v:
                    continue
                m = sum(v) / len(v)
                xs.append(lam)
                means.append(PCT * m)
                ax.plot([lam, lam], [PCT * min(v), PCT * max(v)], color=col, lw=0.8, zorder=2)
                ax.scatter([lam] * len(v), [PCT * x for x in v], s=5, color=col, alpha=0.5, linewidths=0, zorder=2)
            ax.plot(xs, means, color=col, lw=1.1, marker=mk, markersize=3.2, markerfacecolor=face,
                    markeredgecolor=col, markeredgewidth=0.8, zorder=3)
    for ax in (axA, axB):
        ax.set_xticks(lams)
        ax.set_xticklabels(["1.0", "1.25", "1.5", "2.0"])
        ax.set_xlim(0.9, 2.1)
        ax.set_xlabel(r"$\lambda$")
    axA.set_ylim(30, 47)
    axA.set_yticks([30, 35, 40, 45])
    axA.set_ylabel(r"AIME accuracy (\%)")
    axA.set_title("(a) accuracy", loc="left", pad=2)
    axA.text(2.06, 44.3, "32K cap", color=C_32K, fontsize=FS, ha="right", va="top")
    axA.text(1.0, 33.4, "16K cap", color=C_16K, fontsize=FS, ha="left", va="top")
    axB.set_ylim(-3, 80)
    axB.set_yticks([0, 25, 50, 75])
    axB.set_ylabel(r"cut off at the cap (\%)")
    axB.set_title("(b) answers cut off", loc="left", pad=2)
    save(fig, "fig_math_budget")
    plt.close(fig)
    SRC.append("")

def fig_template(plt):
    from matplotlib.lines import Line2D
    rows = md_rows(RES / "A1Rx_verdict.md", "| lam | runs with parse < 0.5 at step 42")
    m1r = (RES / "M1R_verdict.md").read_text()
    assert "all 12 runs" in m1r and "0.953 in all 3 runs" in m1r
    W, H = COLW, 1.30
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.40 / W, 0.33 / H, 2.58 / W, (H - 0.43) / H])
    SRC.extend(["### Template case (fig_template)", "",
                "Source:  (think-block rollout prompt, two batches pooled; exact 95% CIs as "
                "printed) and  (matched prompt: greedy validity 0.953, the ceiling, in all 12 "
                "runs: 3 seeds at each of lambda 1.00 / 1.25 / 1.40 / 2.00).", "",
                "| lambda | think-block: collapsed / runs [95% CI] | matched: collapsed / runs |", "|---|---|---|"])
    for x, r in enumerate(rows):
        k = int(r[1].split("/")[0])
        n = int(re.findall(r"/(\d+)", r[1])[0])
        lo, hi = (float(v) for v in re.findall(r"[\d.]+", r[2])[:2])
        ax.plot([x - 0.09, x - 0.09], [lo, hi], color=C_OPD, lw=0.8, zorder=2)
        ax.plot([x - 0.09], [k / n], marker="o", color=C_OPD, markersize=3.6, zorder=3)
        ax.text(x - 0.09, hi + 0.04, f"{k}/{n}", fontsize=FS, color=C_OPD, ha="center", va="bottom")
        ax.plot([x + 0.09], [0], marker="s", color="#555555", markersize=3.2, markerfacecolor="white", zorder=4)
        SRC.append(f"| {r[0]} | {k}/{n} [{lo:.3f}, {hi:.3f}] | 0/3 |")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(-0.07, 1.22)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(r"$\lambda$", labelpad=1)
    ax.set_ylabel("runs collapsed")
    h = [Line2D([], [], color=C_OPD, marker="o", markersize=3.6, lw=0.8),
         Line2D([], [], color="#555555", marker="s", markersize=3.2, markerfacecolor="white", lw=0)]
    ax.legend(h, ["think-block rollout prompt", "matched prompt (0/3 each)"], loc="upper left", frameon=False,
              handlelength=1.2, handletextpad=0.4, borderaxespad=0.1)
    save(fig, "fig_template")
    plt.close(fig)
    SRC.append("")

def fig_traj(plt):
    con = defaultdict(dict)
    for r in csv.DictReader((RES / "fig1_contrast.csv").open()):
        con[r["arm"]][int(r["step"])] = json.loads(r["per_seed"])
    fl = json.load((RES / "FKL_L_scored.json").open())
    lg = json.load((RES / "LONG210b_scored.json").open())["seeds"]
    W, H = TEXTW, 1.75
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.36 / H, (H - 0.18) / H
    xs_ = (0.38, 1.80, 3.15, 4.92)
    ws_ = (1.18, 1.10, 1.30, 1.30)
    ax = [fig.add_axes([x / W, bot, w / W, top - bot]) for x, w in zip(xs_, ws_)]
    SRC.extend(["### Appendix: per-seed trajectories (figA_trajectories)", "",
                "(a) fig1_contrast.csv per_seed (matched OPD rerun, forward KL; Qwen3, ESCI); (b) FKL_L_scored.json "
                "share_per_seed (base 0.000); (c, d) LONG210b_scored.json seeds: sampled share and fixed-prefix probe "
                "shift at steps 42-210 (teacher-minus-base probe gap +11.56).", ""])
    for a, (arm, col) in ((ax[0], ("OPDre", C_OPD)), (ax[0], ("FKL", C_FKL))):
        seeds = sorted(con[arm][14])
        for sd in seeds:
            ks = [0, 14, 28, 42]
            a.plot(ks, [0.0] + [con[arm][k][sd] for k in ks[1:]], color=col, lw=0.8, marker="o", markersize=2.2)
    ax[0].axhline(qwen3_teacher(), color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)))
    ax[0].set_title("(a) Qwen3, ESCI, per seed", loc="left", pad=2)
    for arm, col in (("OPDre_L", C_OPD), ("FKL_L", C_FKL)):
        for i in range(3):
            a = ax[1]
            a.plot([0, 14, 28, 42], [0.0] + [fl[arm][str(k)]["share_per_seed"][i] for k in (14, 28, 42)], color=col,
                   lw=0.8, marker="o", markersize=2.2)
    ax[1].axhline(fl["teacher_share"], color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)))
    ax[1].set_title("(b) Llama, Fashion, per seed", loc="left", pad=2)
    for a in ax[:2]:
        a.set_xticks([0, 14, 28, 42])
        a.set_xlim(-3, 45)
        a.set_ylim(-0.02, 0.33)
        a.set_xlabel("training step")
    ax[0].set_ylabel("transmission")
    ax[0].text(17, 0.222, "forward KL", color=C_FKL, fontsize=FS, ha="left", va="bottom")
    ax[0].text(44, 0.02, "reverse KL", color=C_OPD, fontsize=FS, ha="right", va="bottom")
    for sd, rec in sorted(lg.items()):
        ks = [0] + sorted(int(k) for k in rec["shares"])
        esc = rec["escaped"]
        col, lw, al = (C_OPD, 1.0, 1.0) if esc else ("#7FB2D6", 0.7, 0.9)
        ax[2].plot(ks, [0.0] + [rec["shares"][str(k)] for k in ks[1:]], color=col, lw=lw, alpha=al, marker="o",
                   markersize=1.8)
        ax[3].plot(ks, [0.0] + [rec["probe"][str(k)] for k in ks[1:]], color=col, lw=lw, alpha=al, marker="o",
                   markersize=1.8)
        SRC.append(f"- seed {sd} ({'escaped' if esc else 'stayed'}): shares " + " / ".join(
            f"{rec['shares'][k]:.4f}" for k in ("42", "84", "126", "168", "210")) + "; probe " + " / ".join(
            f"{rec['probe'][k]:+.2f}" for k in ("42", "84", "126", "168", "210")))
    ax[2].axhline(qwen3_teacher(), color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)))
    ax[3].axhline(11.56, color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)))
    ax[3].axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    for a in ax[2:]:
        a.set_xticks([0, 42, 84, 126, 168, 210])
        a.set_xlim(-6, 216)
        a.set_xlabel("training step")
    ax[2].set_ylim(-0.02, 0.33)
    ax[2].set_title("(c) reverse KL over 210 steps", loc="left", pad=2)
    ax[2].text(212, 0.03, "5 of 8 stay", color="#5A9CC9", fontsize=FS, ha="right", va="bottom")
    ax[2].text(60, 0.24, "3 of 8 escape", color=C_OPD, fontsize=FS, ha="left", va="bottom")
    ax[3].set_ylim(-5, 13)
    ax[3].set_ylabel("probe shift (nats)")
    ax[3].set_title("(d) close log-odds, 210 steps", loc="left", pad=2)
    save(fig, "figA_trajectories")
    plt.close(fig)
    SRC.append("")

def _teacher(ax, y, x0, x1):
    ax.plot([x0, x1], [y, y], color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)), zorder=1)

def qwen3_teacher():
    pass
    fk = json.load((RES / "FKL_scored.json").open())
    t = fk["teacher_share"] - fk["FKL"]["base_share"]
    assert abs(t - 155 / 800) < 1e-12, t
    return t

def ol3x_count(seed, arm, step):
    pass
    d = REPO / "outputs" / "OL3x" / f"OL3x_s{seed}_{arm}" / "samples"
    path, summ = d / f"step{step}.v2.records.jsonl", d / f"step{step}.v2.json"
    assert path.exists() and summ.exists(), (path, summ)
    lab = defaultdict(int)
    for line in path.open():
        r = json.loads(line)
        if r["split"] == "val" and r["label"] != "infeasible_input":
            lab[r["label"]] += 1
    k, n = lab["k1_trunc_realids"], sum(lab.values())
    val = json.loads(summ.read_text())["val"]
    assert n == 808 == val["n_samples_feasible"], (path, n, val["n_samples_feasible"])
    assert abs(val["labels"].get("k1_trunc_realids", 0.0) * n - k) < 1e-6, (path, k, val["labels"])
    return k

def fig_dynamics(plt):
    pass
    from matplotlib.lines import Line2D
    fk = json.load((RES / "FKL_scored.json").open())
    con = defaultdict(dict)
    for r in csv.DictReader((RES / "fig1_contrast.csv").open()):
        con[r["arm"]][int(r["step"])] = json.loads(r["per_seed"])
    fl = json.load((RES / "FKL_L_scored.json").open())
    q = defaultdict(dict)
    for r in csv.DictReader((RES / "q35s_dynamics.csv").open()):
        q[(r["size"], r["arm"], r["seed"])][int(r["step"])] = float(r["transmission"])
    scored = json.load((RES / "Q35S_scored.json").open())
    ol = json.load((RES / "OL3_scored.json").open())
    S3 = ("1001", "1002", "1003")
    K = (14, 28, 42)
    for arm in ("OPDre", "FKL"):
        for k in K:
            assert [con[arm][k][sd] for sd in S3] == fk[arm][str(k)]["share_per_seed"], (arm, k)

    P = [{"title": "Qwen3 1.7B", "n": 800, "teacher": fk["teacher_share"], "base": fk["FKL"]["base_share"],
          "opd1.00": [[fk["OPDre"][str(k)]["share_per_seed"][i] for k in K] for i in range(3)],
          "fkl1.00": [[fk["FKL"][str(k)]["share_per_seed"][i] for k in K] for i in range(3)]},
         {"title": "Llama 3B", "n": 808, "teacher": fl["teacher_share"], "base": fl["base_share"],
          "opd1.00": [[fl["OPDre_L"][str(k)]["share_per_seed"][i] for k in K] for i in range(3)],
          "fkl1.00": [[fl["FKL_L"][str(k)]["share_per_seed"][i] for k in K] for i in range(3)]}]
    for size, lab in (("0p8b", "0.8B"), ("2b", "2B"), ("4b", "4B")):
        b = scored["sizes"][size]["base_share"]
        rec = {"title": f"Qwen3.5 {lab}", "n": 808, "teacher": 178 / 808, "base": b}
        for arm in ("opd1.00", "opd1.40", "fkl1.00"):
            rec[arm] = [[q[(size, arm, sd)][k] + b for k in K] for sd in S3]
            assert [v[2] for v in rec[arm]] == [scored["sizes"][size][arm]["share_per_seed"][sd] for sd in S3] or \
                all(abs(v[2] - scored["sizes"][size][arm]["share_per_seed"][sd]) < 1e-12 for v, sd in zip(rec[arm], S3))
        P.append(rec)
    assert abs(178 / 808 - 0.2203) < 5e-5
    for rec in P:
        rec["seeds"] = S3
    ox = json.load((RES / "OL3x_scored.json").open())
    S5 = S3 + ("1004", "1005")
    assert ox["base"] == [0, 808] and ox["teacher"] == [168, 808], (ox["base"], ox["teacher"])
    assert ol["base_share"] == 0 and abs(ol["teacher_share"] - 168 / 808) < 1e-12, (ol["base_share"], ol["teacher_share"])
    assert ox["scored_new_seeds"] == [1004, 1005] and [str(s) for s in ox["row"]["seeds"]] == list(S5), ox["row"]["seeds"]
    rec = {"title": "Olmo 3 7B", "n": 808, "teacher": ol["teacher_share"], "base": ol["base_share"], "seeds": S5}
    for arm in ("opd1.00", "opd1.40", "fkl1.00"):
        assert all([str(x) for x in ol["cells"][f"{arm}|{k}"]["seeds"]] == list(S3) for k in K), arm
        rec[arm] = [[ol["cells"][f"{arm}|{k}"]["share_per_seed"][sd] for k in K] for sd in S3]
        for sd in S5[3:]:
            c = [ol3x_count(sd, arm, k) for k in K]
            assert c[2] == ox["new_seeds"][sd]["counts42"][arm], (sd, arm, c, ox["new_seeds"][sd]["counts42"])
            if arm == "fkl1.00":
                assert c[:2] == [ox["new_seeds"][sd]["fkl14"]["scored_label"], ox["new_seeds"][sd]["fkl28"]["scored_label"]], (sd, c)
            rec[arm].append([x / 808 for x in c])
        assert [round(ys[2] * 808) for ys in rec[arm]] == [ox["row"]["arms"][arm]["count_per_seed"][sd] for sd in S5], arm
        if arm != "fkl1.00" and any(round(v * 808) for ys in rec[arm] for v in ys):
            sys.exit(f"STOP: Olmo 3 {arm} has a non-zero one-item-short count; OL3x_verdict.md states 0 of 12,120 per "
                     f"arm over steps 14 / 28 / 42 of the five seeds. Counts: {[[round(v * 808) for v in ys] for ys in rec[arm]]}")
    P.append(rec)

    W, H = TEXTW, 1.78
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.56 / H, (H - 0.18) / H
    x0, pw, gap = 0.40, 0.90, 0.085
    axes = [fig.add_axes([(x0 + i * (pw + gap)) / W, bot, pw / W, top - bot]) for i in range(len(P))]
    steps = [0, 14, 28, 42]
    SRC.extend(["### Six-setting dynamics (fig_dynamics)", "",
                "Transmission = one-item-short share of the temperature-1 evaluation samples minus the starting checkpoint's "
                "share on the same sampler; counts are out of n samples per seed. Qwen3 (ESCI): FKL_scored.json OPDre / FKL "
                "share_per_seed (= fig1_contrast.csv per_seed), teacher_share 155/800. Llama (Fashion): FKL_L_scored.json. "
                "Qwen3.5 (Fashion): q35s_dynamics.csv (step 42 equals Q35S_scored.json), teacher 178/808 (Q35S_verdict.md "
                "0.2203). Olmo 3 (Fashion), five seeds: 1001-1003 from OL3_scored.json cells; 1004 and 1005 from "
                "outputs/OL3x/OL3x_s<SEED>_<ARM>/samples/step<S>.v2.records.jsonl (val split, label "
                "k1_trunc_realids, n = 808; step 42 equal to OL3x_scored.json counts42, forward-KL steps 14 and 28 equal to "
                "its fkl14 / fkl28 scored_label); teacher_share 168/808, base_share 0/808. "
                "The dashed line is the teacher's share minus the base's. y axis in percentage points: the figure draws "
                "100 x (count / n - base); the table lists the counts.", "",
                "| setting | n | teacher | base | arm | seed | step 14 | step 28 | step 42 |", "|---|---|---|---|---|---|---|---|---|"])

    def lines(ax, series, col, ls="-", face=None, mk="o"):
        for ys in series:
            ax.plot(steps, [0.0] + ys, color=col, lw=0.75, ls=ls, marker=mk, markersize=2.2,
                    markerfacecolor=face or col, markeredgewidth=0.5, zorder=3)

    sty = {"opd1.00": (C_OPD, "-", None, "o"), "opd1.40": (C_OPD, (0, (3, 1.6)), "white", "o"),
           "fkl1.00": (C_FKL, "-", None, "s")}
    cnt = lambda x, n: int(round(x * n))
    for a, rec in zip(axes, P):
        n, b = rec["n"], rec["base"]
        for arm in ("opd1.00", "opd1.40", "fkl1.00"):
            if arm not in rec:
                continue
            col, ls, face, mk = sty[arm]
            lines(a, [[PCT * (v - b) for v in ys] for ys in rec[arm]], col, ls=ls, face=face, mk=mk)
            for sd, ys in zip(rec["seeds"], rec[arm], strict=True):
                assert all(abs(v * n - cnt(v, n)) < 1e-6 for v in ys), (rec["title"], arm, sd)
                SRC.append(f"| {rec['title']} | {n} | {cnt(rec['teacher'], n)} | {cnt(b, n)} | {arm} | {sd} | "
                           + " | ".join(str(cnt(v, n)) for v in ys) + " |")
        _teacher(a, PCT * (rec["teacher"] - b), -3, 45)
        a.set_title(rec["title"], loc="left", pad=2)
    for i, a in enumerate(axes):
        a.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
        a.set_xlim(-3, 45)
        a.set_xticks(steps)
        a.set_ylim(-3, 75)
        a.set_yticks([0, 20, 40, 60])
        if i:
            a.set_yticklabels([])
    axes[0].set_ylabel("transmission (pp)")
    fig.text((x0 + (len(P) * pw + (len(P) - 1) * gap) / 2) / W, (0.56 - 0.27) / H, "training step", ha="center",
             va="center", fontsize=FS)
    esc = [(rec["title"], sd) for rec in P if "opd1.40" in rec for sd, ys in zip(rec["seeds"], rec["opd1.40"], strict=True)
           if max(ys) - rec["base"] >= 0.10]
    assert esc == [("Qwen3.5 0.8B", "1002")], esc
    d = P[2]["opd1.40"][1]
    axes[2].annotate("one seed\nescapes", xy=(42, PCT * (d[2] - P[2]["base"])), xytext=(1, 60), fontsize=FS, color=C_OPD,
                     ha="left", va="center", linespacing=0.95,
                     arrowprops=dict(arrowstyle="-", color=C_OPD, lw=0.5, shrinkA=1, shrinkB=2))
    h = [Line2D([], [], color=C_OPD, lw=0.9, marker="o", markersize=2.6),
         Line2D([], [], color=C_OPD, lw=0.9, ls=(0, (3, 1.6)), marker="o", markersize=2.6, markerfacecolor="white"),
         Line2D([], [], color=C_FKL, lw=0.9, marker="s", markersize=2.6),
         Line2D([], [], color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)))]
    fig.legend(h, [r"reverse KL, $\lambda{=}1.00$", r"reverse KL, $\lambda{=}1.40$ (Qwen3.5, Olmo 3)", "forward KL", "teacher's share"],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.4)
    save(fig, "fig_dynamics")
    plt.close(fig)

    f14 = [(rec, ys[0]) for rec in P for ys in rec["fkl1.00"]]
    f42 = [(rec, ys[2]) for rec in P for ys in rec["fkl1.00"]]
    facts = {
        "seeds": len(f14),
        "fkl_step14_above_start": sum(v > rec["base"] for rec, v in f14),
        "fkl_step14_at_or_above_teacher": sum(cnt(v, rec["n"]) >= cnt(rec["teacher"], rec["n"]) for rec, v in f14),
        "fkl_step14_transmission_ge_0.10": sum(v - rec["base"] >= 0.10 for rec, v in f14),
        "fkl_step14_at_or_above_teacher_by_setting": {
            rec["title"]: sum(cnt(ys[0], rec["n"]) >= cnt(rec["teacher"], rec["n"]) for ys in rec["fkl1.00"]) for rec in P},
        "fkl_step42_min_transmission": min(v - rec["base"] for rec, v in f42),
        "opd1.00_max_samples_any_checkpoint": max(cnt(v, rec["n"]) for rec in P for ys in rec["opd1.00"] for v in ys),
        "opd1.40_max_samples_any_checkpoint_excluding_the_escape": max(
            cnt(v, rec["n"]) for rec in P if "opd1.40" in rec for sd, ys in zip(rec["seeds"], rec["opd1.40"], strict=True)
            if (rec["title"], sd) not in esc for v in ys),
        "escape_counts_14_28_42": [cnt(v, 808) for v in d],
    }
    by = facts["fkl_step14_at_or_above_teacher_by_setting"]
    assert [len(rec["seeds"]) for rec in P] == [3, 3, 3, 3, 3, 5], [len(rec["seeds"]) for rec in P]
    assert facts["seeds"] == 20 and facts["fkl_step14_above_start"] == 20, facts
    assert facts["fkl_step14_at_or_above_teacher"] == 9 and by["Qwen3.5 2B"] == 0 and by["Olmo 3 7B"] == 0, facts
    assert abs(facts["fkl_step42_min_transmission"] - 0.095) < 1e-12, facts
    assert facts["opd1.00_max_samples_any_checkpoint"] == 1, facts
    assert facts["opd1.40_max_samples_any_checkpoint_excluding_the_escape"] == 1, facts
    assert facts["escape_counts_14_28_42"][:2] == [0, 1] and facts["escape_counts_14_28_42"][2] > 500, facts
    cap = caption("fig_dynamics")
    for phrase in ("in percentage points", "five for Olmo~3 and three otherwise", "one Qwen3.5-0.8B seed escapes after step~28"):
        _cap_check(cap, phrase)
    SRC.extend(["", "Caption facts (asserted in the script): forward KL at step 14 is above the starting share in "
                f"{facts['fkl_step14_above_start']} of {facts['seeds']} seeds, at or above the teacher's share in "
                f"{facts['fkl_step14_at_or_above_teacher']} of {facts['seeds']} ("
                + ", ".join(f"{k}: {v} of {len(rec['seeds'])}" for (k, v), rec in zip(by.items(), P)) + f"), and at a transmission of 0.10 or more in "
                f"{facts['fkl_step14_transmission_ge_0.10']} of {facts['seeds']}; its smallest step-42 transmission is "
                f"{facts['fkl_step42_min_transmission']:.3f} (76 of 800; the caption's {pct(facts['fkl_step42_min_transmission'])} "
                "points). OPD at lambda 1.00: at most "
                f"{facts['opd1.00_max_samples_any_checkpoint']} sample per seed at any checkpoint. OPD at lambda 1.40 "
                "(Qwen3.5 and Olmo 3): Qwen3.5 0.8B seed 1002 has "
                + " / ".join(str(x) for x in facts["escape_counts_14_28_42"]) + " of 808 at steps 14 / 28 / 42; every other "
                f"seed at most {facts['opd1.40_max_samples_any_checkpoint_excluding_the_escape']} sample at any checkpoint.", ""])

def fig_long(plt):
    lg = json.load((RES / "LONG210b_scored.json").open())["seeds"]
    W, H = COLW, 1.55
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.36 / H, (H - 0.17) / H
    axA = fig.add_axes([0.40 / W, bot, 1.12 / W, top - bot])
    axB = fig.add_axes([1.90 / W, bot, 1.08 / W, top - bot])
    SRC.extend(["### 210-step OPD runs (fig_long)", "",
                "LONG210b_scored.json seeds (1001-1003 first batch, 1004-1008 replication): sampled share (E3 sampler, base 0.000) "
                "and fixed-prefix close log-odds shift at steps 42-210; teacher share 155/800 = 0.194 (FKL_scored.json teacher_share), "
                "teacher-minus-base probe gap +11.56 (LONG210b_verdict.md). Shares are counts out of 800; a share that is an "
                "exact half at four decimals is printed at three or five (display rule). Panel (a) is in percent (100 x "
                "the shares below; base 0, so share = transmission); panel (b) is in nats, unchanged.", ""])
    for sd, rec in sorted(lg.items()):
        ks = [0] + sorted(int(k) for k in rec["shares"])
        col, lw = (C_OPD, 1.0) if rec["escaped"] else ("#8DBBDB", 0.7)
        axA.plot(ks, [0.0] + [PCT * rec["shares"][str(k)] for k in ks[1:]], color=col, lw=lw, marker="o", markersize=1.8,
                 zorder=3 if rec["escaped"] else 2)
        axB.plot(ks, [0.0] + [rec["probe"][str(k)] for k in ks[1:]], color=col, lw=lw, marker="o", markersize=1.8,
                 zorder=3 if rec["escaped"] else 2)
        SRC.append(f"- seed {sd} ({'escaped' if rec['escaped'] else 'stayed'}): share " + " / ".join(
            fmt(rec['shares'][k]) for k in ("42", "84", "126", "168", "210")) + "; probe " + " / ".join(
            f"{rec['probe'][k]:+.2f}" for k in ("42", "84", "126", "168", "210")))
    esc = [rec for rec in lg.values() if rec["escaped"]]
    end = sorted(rec["shares"]["210"] for rec in esc)
    gap = sorted(11.56 - rec["probe"]["210"] for rec in esc)
    stay_max = max(v for rec in lg.values() if not rec["escaped"] for v in rec["shares"].values())
    band = (0.17, 0.20)
    assert len(esc) == 3 and len(lg) == 8 and band[0] <= end[0] and end[-1] <= band[1] and 0 < gap[0] and gap[-1] < 0.3, (end, gap)
    assert all(rec["probe"]["210"] < 0 for rec in lg.values() if not rec["escaped"])
    assert stay_max == 1 / 800, stay_max
    cap = caption("fig_long")
    for phrase in (f"end at {PCT * band[0]:.0f}--{PCT * band[1]:.0f}\\%", f"teacher's {pct(qwen3_teacher())}\\% (a)",
                   "within 0.3 nats", "one sample in 800"):
        _cap_check(cap, phrase)
    SRC.append(f"- caption facts: the 3 escaping seeds end at {' / '.join(fmt(v) for v in end)} (step 210; the caption's "
               f"17--20%) against the teacher's 0.194 (the caption's {pct(qwen3_teacher())}%), and "
               f"{' / '.join(f'{g:.2f}' for g in gap)} nats below its close log-odds; the 5 others "
               f"have a negative probe shift at step 210 and at most 1 sample of 800 at any kept checkpoint.")
    _teacher(axA, PCT * qwen3_teacher(), -6, 216)
    _teacher(axB, 11.56, -6, 216)
    axB.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    for a in (axA, axB):
        a.set_xticks([0, 84, 168])
        a.set_xlim(-6, 216)
        a.set_xlabel("training step")
    axA.set_ylim(-2, 30)
    axA.set_yticks([0, 10, 20])
    axA.set_ylabel(r"error share (\%)")
    axA.set_title("(a) error share", loc="left", pad=2)
    axA.text(5, 25.5, "3 of 8 escape", color=C_OPD, fontsize=FS, ha="left")
    axA.text(212, 2.5, "5 stay", color="#5A9CC9", fontsize=FS, ha="right")
    axB.set_ylim(-5, 13)
    axB.set_yticks([-4, 0, 4, 8, 12])
    axB.set_ylabel("log-odds shift")
    axB.set_title("(b) close log-odds", loc="left", pad=2)
    save(fig, "fig_long")
    plt.close(fig)
    SRC.append("")

def fig_arms(plt):
    pass
    from matplotlib.lines import Line2D
    e3s = json.load((RES / "E3_scored.json").open())["arms"]
    rxa = json.load((RES / "RX_scored.json").open())["A"]
    q35 = json.load((RES / "Q35S_scored.json").open())["E_arm"]
    law = json.load((RES / "E3_law_predictions.json").open())["predictions"]

    def e_arm(tag, label, lawkey):
        pass
        z = e3s[tag]
        base = z["base_k1"]
        tr1 = [x - base for x in z["lam1.00_k1_per_seed"]]
        assert abs(sum(tr1) / len(tr1) - z["Tr"]) < 1e-12
        return {"label": label, "base": base,
                "1.00": {"per": tr1, "mean": z["Tr"], "ci": z["Tr_ci95"]},
                "1.40": {"per": [x - base for x in z["lam1.40_k1_per_seed"]], "mean": z["Tr"] + z["Am"], "ci": None},
                "law": {k: law[lawkey][k]["42"]["Tr"] for k in ("1.00", "1.40")}}

    arms = [e_arm("E2", "Qwen3, errs 2\\%", "E2"), e_arm("E5", "Qwen3, errs 5\\%", "E5")]
    qb = q35["base_share"]
    arms.append({"label": "Qwen3.5 4B, errs 2\\%", "base": qb,
                 "1.00": {"per": [v - qb for v in q35["opd1.00"]["share_per_seed"].values()],
                          "mean": q35["opd1.00"]["Tr"], "ci": q35["opd1.00"]["Tr_ci95"]},
                 "1.40": {"per": [v - qb for v in q35["opd1.40"]["share_per_seed"].values()],
                          "mean": q35["opd1.40"]["Tr"], "ci": None}, "law": None})
    for key, label in (("armR", "coupled, injected"), ("clean", "coupled, clean")):
        a = {"label": label, "base": 0.0, "law": None, "coupled": True}
        for lam in ("1.00", "1.40"):
            r = rxa[f"{key}_{lam}"]
            assert r["status"] == "scored" and r["base"] == 0.0 and len(r["Tr_per_seed"]) == 5
            a[lam] = {"per": list(r["Tr_per_seed"].values()), "mean": r["Tr"], "ci": r["Tr_ci95"]}
        arms.append(a)

    W, H = COLW, 1.62
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.34 / H, (H - 0.17) / H
    axA = fig.add_axes([1.12 / W, bot, 0.88 / W, top - bot])
    axB = fig.add_axes([2.10 / W, bot, 0.88 / W, top - bot])
    SRC.extend(["### Starting rate, coupled route and its clean-teacher control (fig_arms)", "",
                "Read unrounded from the scored files. Already-errs arms: E3_scored.json arms.E2 / E5 (per-seed share "
                "minus base_k1 10/800 and 30/800; bar = Tr at lambda 1.00 and Tr + Am at lambda 1.40; whisker = Tr_ci95, "
                "lambda 1.00 only) and Q35S_scored.json E_arm (base 16/808). Coupled-route arms, injected and clean "
                "teacher, 5 seeds each: RX_scored.json section A (Tr_per_seed, Tr, Tr_ci95; exact 5-vs-5 permutation "
                "test, p = 0.0040 at both lambda, the test's minimum; RXA_verdict.md). Grey markers: "
                "E3_law_predictions.json step-42 Tr. Values below are printed by the display rule, as fractions; the "
                "x axes are in percentage points (100 x these values).", "",
                "| arm | lambda | per-seed Tr | mean | bootstrap 95% |", "|---|---|---|---|---|"])
    cl = arms[-1]
    assert cl["label"] == "coupled, clean", cl["label"]
    cl_max = max(v for lam in ("1.00", "1.40") for v in cl[lam]["per"])
    assert abs(cl["1.00"]["mean"] - 1 / 4000) < 1e-12 and abs(cl["1.40"]["mean"] - 0.002) < 1e-12 and \
        abs(cl_max - 0.005) < 1e-12, (cl["1.00"]["mean"], cl["1.40"]["mean"], cl_max)
    cap = caption("fig_arms")
    for phrase in (f"transmits {pct(cl['1.00']['mean'])} points at $\\lam=1$ and {pct(cl['1.40']['mean'])} at 1.40",
                   f"no seed above {pct(cl_max)})"):
        _cap_check(cap, phrase)
    n = len(arms)
    for ax, lam in ((axA, "1.00"), (axB, "1.40")):
        for i, a in enumerate(arms):
            y = n - 1 - i
            d = a[lam]
            hatch = "////" if a.get("coupled") else None
            ax.barh([y], [PCT * d["mean"]], height=0.6, color=C_OPD, alpha=0.30 if hatch else 0.55, edgecolor=C_OPD,
                    lw=0.6, hatch=hatch, zorder=2)
            if d["ci"]:
                ax.plot([PCT * c for c in d["ci"]], [y, y], color="#333333", lw=0.8, zorder=4)
            k = len(d["per"])
            ax.scatter([PCT * v for v in d["per"]], [y + 0.26 * (t / max(k - 1, 1) - 0.5) for t in range(k)], s=5.5,
                       color=C_OPD, edgecolor="white", linewidths=0.3, zorder=5)
            if a["law"]:
                ax.plot([PCT * a["law"][lam]], [y + 0.36], marker="v", color=C_GREY, markersize=3.4, zorder=5)
            SRC.append(f"| {a['label'].replace(chr(92), '')} | {lam} | "
                       + " / ".join(fmt(v) for v in d["per"]) + f" | {fmt(d['mean'])} | "
                       + (f"[{fmt(d['ci'][0], 3)}, {fmt(d['ci'][1], 3)}]" if d["ci"] else "--") + " |")
        ax.set_yticks(range(n))
        ax.set_ylim(-0.6, n - 0.35)
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color="#BBBBBB", lw=0.5, zorder=0)
    axA.set_yticklabels([a["label"] for a in arms][::-1])
    axB.set_yticklabels([])
    axA.set_xlim(0, 23.5)
    axA.set_xticks([0, 10, 20])
    axA.set_xticklabels(["0", "10", "20"])
    axA.set_title(r"(a) $\lambda{=}1.00$", loc="left", pad=2)
    axB.set_xlim(0, 106)
    axB.set_xticks([0, 50, 100])
    axB.set_xticklabels(["0", "50", "100"])
    axB.set_title(r"(b) $\lambda{=}1.40$", loc="left", pad=2)
    fig.text(0.5, 3.5 / 72 / H, "transmission at step 42 (points)", ha="center", va="baseline", fontsize=FS)
    save(fig, "fig_arms")
    plt.close(fig)
    SRC.append("")

def fig_template2(plt):
    from matplotlib.lines import Line2D
    rows = md_rows(RES / "A1Rx_verdict.md", "| lam | runs with parse < 0.5 at step 42")
    m1r = (RES / "M1R_verdict.md").read_text()
    assert "all 12 runs" in m1r and "0.953 in all 3 runs" in m1r
    a1r = md_rows(RES / "A1Rx_verdict.md", "| lam | A1R (think-block training prompt)")
    ev, tr = {}, {}
    for r in a1r:
        lam = f"{float(r[0]):.2f}"
        for s, e, t in zip(("1001", "1002", "1003"), r[1].split("/"), r[2].split("/")):
            ev[(s, lam)], tr[(s, lam)] = float(e), float(t)
    for k, v in json.load((RES / "A1Rx_grid.json").open()).items():
        m = re.match(r"A1Rx_s(\d+)_lam([\d.]+)$", k)
        ev[(m.group(1), m.group(2))] = v["eval"]["42"]["parse"]
        tr[(m.group(1), m.group(2))] = v["train"]["shares"].get("other", 0.0)
    W, H = COLW, 1.62
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.50 / H, (H - 0.17) / H
    axA = fig.add_axes([0.40 / W, bot, 1.18 / W, top - bot])
    axB = fig.add_axes([1.86 / W, bot, 1.12 / W, top - bot])
    lams = ["1.00", "1.25", "1.40", "2.00"]
    SRC.extend(["### Template case (fig_template, two panels)", "",
                "(a) A1Rx_verdict.md collapse counts (greedy validity < 0.5 at step 42) and exact 95% CIs as printed; matched "
                "prompt: M1R_verdict.md (0.953 in all 12 runs). (b) training rollouts without the list (steps 29-42) per seed: "
                "A1Rx_verdict.md (seeds 1001-1003), A1Rx_grid.json train.shares.other (seeds 1004-1010); filled black: greedy "
                "validity < 0.5 at step 42. Both y axes in percent (100 x the fractions and counts below); the k/n labels "
                "are counts.", ""])
    COLLAPSE = 0.5
    assert f"greedy validity below {pct(COLLAPSE, 0)}\\%" in caption("fig_template")
    for x, r in enumerate(rows):
        k = int(r[1].split("/")[0])
        n = int(re.findall(r"/(\d+)", r[1])[0])
        lo, hi = (float(v) for v in re.findall(r"[\d.]+", r[2])[:2])
        axA.plot([x - 0.1, x - 0.1], [PCT * lo, PCT * hi], color=C_OPD, lw=0.8, zorder=2)
        axA.plot([x - 0.1], [PCT * k / n], marker="o", color=C_OPD, markersize=3.4, zorder=3)
        axA.text(x - 0.1, PCT * hi + 4, f"{k}/{n}", fontsize=FS, color=C_OPD, ha="center", va="bottom")
        axA.plot([x + 0.12], [0], marker="s", color="#555555", markersize=3.0, markerfacecolor="white", zorder=4)
        SRC.append(f"- lambda {r[0]}: {k}/{n} [{lo:.3f}, {hi:.3f}]; matched 0/3")
    axA.set_xticks(range(4))
    axA.set_xticklabels(lams)
    axA.set_xlim(-0.5, 3.5)
    axA.set_ylim(-6, 125)
    axA.set_yticks([0, 50, 100])
    axA.set_ylabel(r"share of runs (\%)")
    axA.set_title("(a) runs collapsed", loc="left", pad=2)
    seeds = sorted({s for s, _ in tr}, key=int)
    for s in seeds:
        xs = [i for i, l in enumerate(lams) if (s, l) in tr]
        ys = [PCT * tr[(s, lams[i])] for i in xs]
        axB.plot(xs, ys, color=C_OPD, lw=0.6, alpha=0.75, marker="o", markersize=2.0, zorder=2)
        for i in xs:
            if ev[(s, lams[i])] < COLLAPSE:
                axB.plot([i], [PCT * tr[(s, lams[i])]], marker="o", color="black", markersize=3.6, zorder=4)
        SRC.append(f"- seed {s}: " + ", ".join(f"lambda {lams[i]} share {tr[(s, lams[i])]:.3f} greedy {ev[(s, lams[i])]:.3f}" for i in xs))
    axB.set_xticks(range(4))
    axB.set_xticklabels(lams)
    axB.set_xlim(-0.3, 3.3)
    axB.set_ylim(-3, 105)
    axB.set_yticks([0, 50, 100])
    axB.set_title(r"(b) training: no list (\%)", loc="left", pad=2)
    for a in (axA, axB):
        a.set_xlabel(r"$\lambda$", labelpad=1)
    h = [Line2D([], [], color=C_OPD, marker="o", markersize=3.4, lw=0),
         Line2D([], [], color="#555555", marker="s", markersize=3.0, markerfacecolor="white", lw=0),
         Line2D([], [], color="black", marker="o", markersize=3.4, lw=0)]
    fig.legend(h, ["mismatched prompt", "matched", "collapsed"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.25, columnspacing=0.6)
    save(fig, "fig_template")
    plt.close(fig)
    SRC.append("")

def fig_selection(plt):
    sys.path.insert(0, str(REPO / "scripts"))
    import fig_tables_aux as AT
    rb, n = AT.selection()
    items = [("a_parsed", "NDCG@1, parsed only", C_FKL), ("c_repair", "NDCG@1, repaired", C_OPD),
             ("c_repair8", "NDCG@8, repaired", C_OPD), ("random", "random pick", C_GREY),
             ("b_useful", "NDCG@1, invalid as 0", C_SEQ)]
    tab = {}
    for ln in (CAPS / "tab_selection.tex").read_text().splitlines():
        m = re.match(r"^(.+?) & (.+?) \\\\ % groups: \d+$", ln)
        if m:
            tab[m.group(1)] = re.sub(r"\\textbf\{([^}]*)\}", r"\1", m.group(2))
    tab_name = {"a_parsed": "NDCG@1, parsed outputs", "c_repair": "NDCG@1 after repair", "c_repair8": "NDCG@8 after repair",
                "b_useful": "NDCG@1, invalid as 0", "random": "random pick"}
    for key, _, _ in items:
        p, (lo, hi) = rb[key]["point"][0], AT.selection_ci(key, rb[key])
        got = f"{pct(p, 0)} [{pct(lo, 0)}, {pct(hi, 0)}]"
        assert tab.get(tab_name[key]) == got, f"fig_selection {key}: figure {got}, tab:selection {tab.get(tab_name[key])}"
    import e1_selection as E1
    for phrase in (f"{n} groups", f"validity ${{\\geq}}\\,{pct(E1.HEALTHY, 0)}\\%$ and one ${{<}}\\,{pct(E1.COLLAPSE, 0)}\\%$"):
        _cap_check(caption("fig_selection"), phrase)
    W, H = COLW, 1.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([1.30 / W, 0.36 / H, 1.42 / W, (H - 0.45) / H])
    SRC.extend(["### Checkpoint selection (fig_selection)", "",
                f"scripts/fig_tables_aux.py selection() (= tab:selection; {n} contested groups, exact CIs for counts, random pick by resampling groups). "
                "x axis in percent of groups (100 x the fractions below); the bar labels are whole percent, asserted equal "
                "to tab:selection's cells.", ""])
    for i, (key, lab, col) in enumerate(items):
        p, (lo, hi) = rb[key]["point"][0], AT.selection_ci(key, rb[key])
        y = len(items) - 1 - i
        ax.barh([y], [PCT * p], height=0.6, color=col, alpha=0.6, edgecolor=col, lw=0.6)
        ax.plot([PCT * lo, PCT * hi], [y, y], color="#333333", lw=0.8)
        ax.text(PCT * max(hi, p) + 3, y, pct(p, 0), va="center", fontsize=FS)
        SRC.append(f"- {lab}: {p:.2f} [{lo:.2f}, {hi:.2f}]")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[1] for it in items][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(r"failing checkpoint picked (\% of groups)")
    save(fig, "fig_selection")
    plt.close(fig)
    SRC.append("")

def fig_greedy(plt):
    from matplotlib.lines import Line2D
    led = {r["id"]: float(r["parse"]) for r in csv.DictReader((RES / "ledger.csv").open()) if r["parse"]}
    e3s = json.load((RES / "E3_scored.json").open())["arms"]
    fk = json.load((RES / "FKL_scored.json").open())
    k1 = json.load((RES / "K1_scored.json").open())
    S = ("1001", "1002", "1003")
    pts = []
    pts.append(("teacher", fk["teacher_share"], led["F3_sft_t4b_k1"], "teacher"))
    pts.append(("start", fk["FKL"]["base_share"], led["F3_sft_b17_clean"], "start"))
    for i, s in enumerate(S):
        pts.append(("OPD rerun", fk["OPDre"]["42"]["share_per_seed"][i], led[f"OPDre_A_s{s}_step42"], "opd1"))
        pts.append(("forward KL", fk["FKL"]["42"]["share_per_seed"][i], led[f"FKL_A_s{s}_step42"], "fkl"))
        pts.append(("SeqKD", k1["runs"][s]["k1_b1"], led[f"K1_s{s}_greedy"], "seq"))
    for tag in ("E2", "E5", "R"):
        z = e3s[tag]
        for s, a, b in zip(S, z["lam1.00_k1_per_seed"], z["lam1.40_k1_per_seed"]):
            for lam, x, st in (("1.00", a, "opd1"), ("1.40", b, "opd14")):
                y = led[f"E3_{tag}_lam{lam}_s{s}_step42"]
                assert y == z["greedy_parse"][f"{lam}|{s}|42"], (tag, lam, s)
                pts.append((f"{tag} {lam}", x, y, st))
    BAND = 0.25
    UP_TO, PAST = 0.24, 0.35
    band_min = min(y for _, x, y, _ in pts if x < BAND)
    x_plain = max(x for _, x, _, st in pts if st in ("opd1", "fkl", "seq"))
    x_extr = min(x for lab, x, _, st in pts if st == "opd14")
    assert band_min == 0.955 and x_plain < UP_TO and x_extr > PAST, (band_min, x_plain, x_extr)
    assert all(x > PAST for _, x, y, _ in pts if y < band_min), "a point below the band's floor sits at a low share"
    cap = caption("fig_greedy")
    for phrase in (f"up to {PCT * UP_TO:.0f}\\% of samples", f"greedy validity at {pct(band_min)}\\% or above",
                   f"past {PCT * PAST:.0f}\\%"):
        _cap_check(cap, phrase)
    W, H = COLW, 1.80
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.42 / W, 0.36 / H, 2.52 / W, 0.96 / H])
    sty = {"teacher": dict(marker="*", color=C_TEACH, s=40), "start": dict(marker="D", color="#555555", s=14),
           "opd1": dict(marker="o", color=C_OPD, s=12), "opd14": dict(marker="o", facecolor="white", edgecolor=C_OPD, s=12),
           "fkl": dict(marker="s", color=C_FKL, s=12), "seq": dict(marker="^", color=C_SEQ, s=14)}
    SRC.extend(["### Greedy validity vs sampled error (fig_greedy)", "",
                "Qwen3 on ESCI, step 42 (SeqKD: end of its fine-tuning). x: one-item-short share of T = 1 eval samples, "
                "unrounded (FKL_scored.json teacher_share, base_share and per-seed step-42 shares; K1_scored.json runs; "
                "E3_scored.json per-seed shares at lambda 1.00 and 1.40), printed by the display rule. y: greedy strict "
                "validity, results/ledger.csv parse (F3_sft_t4b_k1, F3_sft_b17_clean, OPDre / FKL / E3 step 42, K1; "
                "the E3 rows equal E3_scored.json greedy_parse). Both axes in percent: the figure draws 100 x x and y.", "",
                "| point | x | y |", "|---|---|---|"])
    for lab, x, y, st in pts:
        kw = dict(sty[st])
        if "facecolor" in kw:
            ax.scatter([PCT * x], [PCT * y], marker=kw["marker"], facecolors=kw["facecolor"], edgecolors=kw["edgecolor"],
                       s=kw["s"], linewidths=0.8, zorder=3)
        else:
            ax.scatter([PCT * x], [PCT * y], marker=kw["marker"], color=kw["color"], s=kw["s"], linewidths=0, zorder=3)
        SRC.append(f"| {lab} | {fmt(x)} | {y:.3f} |")
    SRC.append("")
    SRC.append(f"Caption facts: the largest share among the plain-OPD, forward-KL and SeqKD points is {fmt(x_plain)} (below "
               f"the caption's {PCT * UP_TO:.0f}%); the smallest lambda 1.40 share is {fmt(x_extr)} (above its "
               f"{PCT * PAST:.0f}%); the lowest greedy validity among points with a share below {BAND} (the shaded band) is "
               f"{band_min:.3f} (the caption's and the panel's {pct(band_min)}%).")
    ax.set_xlim(-3, 102)
    ax.set_ylim(-4, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel(r"sampled error share (\%, temperature 1)")
    ax.set_ylabel(r"greedy validity (\%)")
    ax.axvspan(-3, PCT * BAND, color="#EFEFEF", zorder=0)
    ax.text(11, 42, f"greedy\n$\\geq$ {pct(band_min)}\\%", fontsize=FS, color="#555555", ha="center",
            linespacing=0.95)
    h = [Line2D([], [], **{k: v for k, v in (("marker", "*"), ("color", C_TEACH), ("markersize", 6), ("lw", 0))}),
         Line2D([], [], marker="o", color=C_OPD, markersize=3.5, lw=0),
         Line2D([], [], marker="o", color=C_OPD, markerfacecolor="white", markersize=3.5, lw=0),
         Line2D([], [], marker="s", color=C_FKL, markersize=3.5, lw=0),
         Line2D([], [], marker="^", color=C_SEQ, markersize=3.8, lw=0)]
    h = [h[1], h[2], h[0], h[3], h[4]]
    fig.legend(h, [r"rev.\ KL, $\lambda{=}1$", r"rev.\ KL, $\lambda{=}1.40$", "teacher", "fwd.\\ KL", "SeqKD"],
               loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=3, frameon=False, handletextpad=0.2,
               columnspacing=0.6, borderaxespad=0.05, labelspacing=0.25)
    save(fig, "fig_greedy")
    plt.close(fig)
    SRC.append("")

def fig_lr(plt):
    pass
    from matplotlib.lines import Line2D
    B = json.load((RES / "RX_scored.json").open())["B"]
    LRS = ("1e-6", "3e-6", "1e-5")
    X = [1e-6, 3e-6, 1e-5]
    BAR = 0.10
    assert all(v < BAR for lr in LRS for s in ("14", "28", "42")
               for v in B[f"opd_{lr}"]["steps"][s]["share_per_seed"].values()), "an OPD seed reached the bar"
    e3v = (RES / "E3_verdict.md").read_text()
    assert "| 28 | 0.5025 / 0.000 / 0.000 |" in e3v and "| 42 | 0.000 / 0.02625 / 0.000 | **0.00875** (0.093) | 21 of 2,400 |" in e3v
    v14 = B["fkl_3e-6"]["steps"]["14"]["Tr"]
    assert v14 < BAR and abs(v14 - 367 / 4000) < 1e-12, v14
    cap = caption("fig_lr")
    for phrase in (f"at or above {PCT * BAR:.0f} points", f"mean is {pct(0.00875)} points", f"stood at {pct(0.5025)} at step",
                   f"missed at $3{{\\times}}10^{{-6}}$ ({pct(v14)} points)"):
        _cap_check(cap, phrase)
    W, H = COLW, 1.55
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.50 / H, (H - 0.17) / H
    axA = fig.add_axes([0.42 / W, bot, 1.12 / W, top - bot])
    axB = fig.add_axes([1.88 / W, bot, 1.08 / W, top - bot])
    SRC.extend(["### Learning-rate sweep at a near-zero start (fig_lr)", "",
                "results/RX_scored.json section B (= RXB_verdict.md): 5 seeds per cell; transmission = the "
                "one-item-short share on the evaluation sampler minus the starting checkpoint's 0.0000. Line: 5-seed "
                "mean; dots: seeds. Grey in (a): the locked rate-law predictions (0.017 at 1e-6, 0.093 at 1e-5), "
                "falsified. Dashed in (b): the locked 0.10 step-14 bar for forward KL, missed at 3e-6 (0.092, counts-based "
                "367/4000; printed at three decimals because four would be an exact half). Both y axes in percentage "
                f"points (100 x the values below): the bar is drawn at 10, the miss printed as {pct(v14)}.", "",
                "| arm | lr | step | Tr | per seed |", "|---|---|---|---|---|"])
    for ax, step in ((axA, "42"), (axB, "14")):
        for arm, col, mk in (("opd", C_OPD, "o"), ("fkl", C_FKL, "s")):
            means = []
            for x, lr in zip(X, LRS):
                v = B[f"{arm}_{lr}"]["steps"][step]
                per = list(v["share_per_seed"].values())
                means.append(PCT * v["Tr"])
                ax.scatter([x * 1.14] * len(per), [PCT * q for q in per], s=5.5, color=col, alpha=0.55, linewidths=0,
                           zorder=3)
                SRC.append(f"| {'OPD' if arm == 'opd' else 'forward KL'} | {lr} | {step} | {fmt(v['Tr'])} | "
                           + " / ".join(fmt(q) for q in per) + " |")
            ax.plot(X, means, color=col, lw=1.1, marker=mk, markersize=3.4, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(7e-7, 1.5e-5)
        ax.set_xticks(X)
        ax.set_xticklabels(["$10^{-6}$", r"$3{\times}10^{-6}$", "$10^{-5}$"])
        ax.minorticks_off()
        ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
        ax.set_ylim(-3, 43)
        ax.set_yticks([0, 20, 40])
        ax.set_xlabel("learning rate", labelpad=1)
    law = B["rate_law_Tr42"]
    for lr, x in zip(LRS, X):
        if lr in law:
            axA.plot([x * 0.87], [PCT * law[lr]], marker="x", color=C_GREY, markersize=4.2, markeredgewidth=1.0, zorder=5)
    axA.set_ylabel("transmission (points)")
    axA.set_title("(a) step 42", loc="left", pad=2)
    axB.set_yticklabels([])
    axB.set_title("(b) step 14", loc="left", pad=2)
    axB.axhline(PCT * BAR, color="#333333", lw=0.7, ls=(0, (3, 2)), zorder=2)
    axB.annotate(pct(v14), xy=(3e-6, PCT * v14), xytext=(2.45e-6, 2.6), fontsize=FS, color=C_FKL, ha="right",
                 va="bottom", arrowprops=dict(arrowstyle="-", color=C_FKL, lw=0.5, shrinkA=1, shrinkB=2))
    axA.text(1.05e-6, 30, "forward KL", color=C_FKL, fontsize=FS, ha="left")
    axA.text(1.05e-6, 5.5, "reverse KL", color=C_OPD, fontsize=FS, ha="left")
    axA.text(7.0e-6, PCT * law["1e-5"], "law", color=C_GREY, fontsize=FS, ha="right", va="center")
    save(fig, "fig_lr")
    plt.close(fig)
    SRC.append("")

E1_SCORED = RES / "E1_scored.json"

def fig_rate(plt):
    pass
    J = lambda f: json.load((RES / f).open())
    fk = J("FKL_scored.json")
    W, H = COLW, 1.60
    fig = plt.figure(figsize=(W, H))
    bot, top = 0.42 / H, (H - 0.14) / H
    axA = fig.add_axes([0.36 / W, bot, 1.00 / W, top - bot])
    axB = fig.add_axes([2.06 / W, bot, 0.92 / W, top - bot])
    qs = J("QSWEEP_scored.json")
    assert qs["P_T_ref"] == 0.19375 and fk["teacher_share"] == 0.19375
    series = [(fk["teacher_share"], {t: fk["FKL"][t]["Tr"] for t in ("14", "28", "42")},
               {t: fk["OPDre"][t]["Tr"] for t in ("14", "28", "42")})]
    for key in ("f50", "f80"):
        z = qs["q"][key]
        series.append((z["P_T"], {t: z["fkl"][t]["Tr"] for t in ("14", "28", "42")},
                       {t: z["opd"][t]["Tr"] for t in ("14", "28", "42")}))
    assert [round(s[0], 4) for s in series] == [0.1938, 0.4825, 0.7788]
    assert (pct(series[1][1]["42"]), pct(series[2][1]["42"])) == ("25.2", "62.8")
    assert (round(series[1][1]["28"], 3), round(series[2][1]["28"], 3)) == (0.762, 0.745)
    assert all(v == 0 for s in series[1:] for v in s[2].values())
    e6 = J("A7E6_scored_a2c.json")
    assert e6["fkl"]["steps"]["42"]["k1_pooled"] == 98 and e6["opd"]["steps"]["42"]["k1_pooled"] <= 1
    pt8 = 64 / 800
    axA.plot([0, 42], [0, PCT * e6["fkl"]["steps"]["42"]["Tr"]], color="#F7C9A3", lw=1.0, marker="s", ms=2.4, zorder=3)
    axA.axhline(PCT * pt8, color="#F7C9A3", lw=0.6, ls=(0, (3, 2)), zorder=1)
    axA.text(42.8, PCT * pt8 - 2.5, "8\\%", color="#E8A070", va="center", ha="left", fontsize=plt.rcParams["font.size"])
    shades = ("#F0A877", "#D55E00", "#8A3D00")
    steps = [0, 14, 28, 42]
    for (pt, fkl, opd), c in zip(series, shades):
        axA.plot(steps, [0] + [PCT * fkl[t] for t in ("14", "28", "42")], color=c, lw=1.0, marker="s", ms=2.4, zorder=3)
        axA.axhline(PCT * pt, color=c, lw=0.6, ls=(0, (3, 2)), zorder=1)
        axA.text(42.8, PCT * pt, f"{pct(pt, 0)}\\%", color=c, va="center", ha="left", fontsize=plt.rcParams["font.size"])
    axA.plot(steps, [0] + [PCT * series[0][2][t] for t in ("14", "28", "42")], color=C_OPD, lw=1.0, marker="o", ms=2.4,
             zorder=4)
    axA.set_xticks(steps)
    axA.set_xlim(0, 42)
    axA.set_ylim(-3, 82)
    axA.set_yticks([0, 25, 50, 75])
    axA.set_xlabel("training step")
    axA.set_ylabel("transmission (pp)")
    SRC.extend(["### Teacher rate and where OPD trains (fig_rate)", "", "(a) transmission at steps 14 / 28 / 42 by teacher share (dashed: the teacher's share; drawn at 0 at "
                "step 0 by definition). FKL_scored.json and QSWEEP_scored.json q.f50 / q.f80.", "",
                "| teacher share | objective | step 14 | step 28 | step 42 |", "|---|---|---|---|---|"])
    for pt, fkl, opd in series:
        SRC.append(f"| {fmt(pt)} | forward KL | " + " | ".join(fmt(fkl[t]) for t in ("14", "28", "42")) + " |")
        SRC.append(f"| {fmt(pt)} | OPD | " + " | ".join(fmt(opd[t]) for t in ("14", "28", "42")) + " |")

    st0 = J("STEP0_scored.json")["teacher_scored"]
    fv = st0["fixed_val"]
    base = fv["base_kl_per_tok_ex"]
    groups = []
    for arm in ("opd1.00", "fkl1.00"):
        a = fv["arms"][arm]
        allt = [d / base for d in a["D_per_seed"]]
        assert abs(sum(allt) / 3 + a["rel_decrease"]) < 1e-9
        groups.append((allt, a["slot7_rel_change_per_seed"]))
    assert round(fv["arms"]["opd1.00"]["rel_decrease"], 3) == 0.088
    assert [round(100 * x, 1) for x in groups[0][1]] == [-0.3, -0.5, -0.8]
    assert [round(100 * x) for x in groups[1][1]] == [-79, -72, -80]
    xs = {"opd1.00": 0, "fkl1.00": 1}
    for gi, (gname, idx) in enumerate((("all tokens", 0), ("error decision", 1))):
        for arm, c, dx in (("opd1.00", C_OPD, -0.19), ("fkl1.00", C_FKL, 0.19)):
            vals = groups[xs[arm]][idx]
            m = sum(vals) / len(vals)
            axB.bar(gi + dx, PCT * m, width=0.34, color=c, alpha=0.85, lw=0, zorder=2)
            axB.plot([gi + dx] * len(vals), [PCT * v for v in vals], ls="", marker="o", ms=1.8, color="black", zorder=3)
    axB.axhline(0, color="black", lw=0.5, zorder=1)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["other\ntokens", "error\ndecision"])
    axB.tick_params(axis="x", length=0)
    axB.set_xlim(-0.55, 1.55)
    axB.set_ylim(-85, 12)
    axB.set_yticks([-80, -40, 0])
    axB.set_ylabel("change in reverse KL (\\%)")
    SRC.extend(["", "(b) STEP0_scored.json teacher_scored.fixed_val: per-token reverse KL to the teacher, step 42 minus "
                "step 0, over base_kl_per_tok_ex (item 7 excluded; per seed D / base), and slot7_rel_change_per_seed "
                "(the item-7 decision).", "",
                "| arm | other tokens, per seed | error decision, per seed |", "|---|---|---|"])
    for arm in ("opd1.00", "fkl1.00"):
        g = groups[xs[arm]]
        SRC.append(f"| {arm} | " + " / ".join(fmt(v) for v in g[0]) + " | " + " / ".join(fmt(v) for v in g[1]) + " |")
    SRC.append("")
    from matplotlib.lines import Line2D
    axB.legend(handles=[Line2D([], [], color=C_FKL, marker="s", ms=2.4, lw=1.0, label="fwd.\\ KL"),
                        Line2D([], [], color=C_OPD, marker="o", ms=2.4, lw=1.0, label="rev.\\ KL")], loc="lower left",
               frameon=False, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1, borderpad=0.1, labelspacing=0.2)
    for ax, letter in ((axA, "a"), (axB, "b")):
        ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=plt.rcParams["font.size"])
    save(fig, "fig_rate")
    plt.close(fig)

def fig_quality(plt, axes=None, letters=("a", "b")):
    pass
    d = json.load((RES / "EQ0_scored.json").open())
    esci = [r for r in d["seed_means"] if r["key"][0] in ("A", "RXB", "LONG3", "LONG210", "LONG210b")]
    assert len(esci) == d["QR"]["n_seed_mean_points"] == 44, len(esci)
    xs = [r["x_ex"] for r in esci]
    assert abs((max(xs) - min(xs)) - d["QR"]["R"]) < 1e-12 and round(d["QR"]["R"], 3) == 0.143
    assert d["QR"]["outcome"] == "not degenerate" and d["QE"]["outcome"] == "holds"
    sq = [r for r in d["QS"]["seed_means"] if not (r["key"][0] == "SQLF" and r["key"][1] == "lam1.40")]
    assert round(d["QS"]["R_S"], 3) == 0.056
    fs = plt.rcParams["font.size"]
    if axes is None:
        W, H = COLW, 1.60
        fig = plt.figure(figsize=(W, H))
        bot, top = 0.38 / H, (H - 0.14) / H
        axA = fig.add_axes([0.40 / W, bot, 1.18 / W, top - bot])
        axB = fig.add_axes([1.86 / W, bot, 1.12 / W, top - bot])
    else:
        fig, (axA, axB) = None, axes
    SRC.extend(["### Quality against inherited error (fig_quality)", "",
                "EQ0_scored.json seed_means (ESCI, x = x_ex, y = share; 44 points of rule QR) and QS.seed_means (Spider, "
                "x = execution accuracy, invalid 0; SQLF lambda 1.40 not drawn).", "",
                "| setting | arm | step | x | y |", "|---|---|---|---|---|"])
    for ax, rows, xk in ((axA, esci, "x_ex"), (axB, sq, "x")):
        for arm, c, mk in (("fkl", C_FKL, "s"), ("opd", C_OPD, "o")):
            pts = [(r[xk], r["y"]) for r in rows if r["key"][2] == arm]
            ax.plot([PCT * x for x, _ in pts], [PCT * y for _, y in pts], ls="", marker=mk, ms=2.6, color=c,
                    mec="white", mew=0.3, alpha=0.9, zorder=3 if arm == "opd" else 2)
        for r in rows:
            SRC.append(f"| {r['key'][0]} {r['key'][1]} | {r['key'][2]} | {r['key'][3]} | {fmt(r[xk])} | {fmt(r['y'])} |")
        ax.set_ylim(-1.5, 28)
        ax.set_yticks([0, 10, 20])
    e1 = json.load((RES / "EQ1_scored.json").open())
    ksteps = [42, 84, 126, 168, 210]
    x0 = e1["step0_metrics"]["x_ex"]
    assert round(x0, 4) == 0.5977 and e1["step0_metrics"]["one_item_short"] == 0.0, e1["step0_metrics"]
    path = {}
    for arm in ("opd", "fkl"):
        mean = []
        for k in ksteps:
            pts_k = [r for r in e1["points"] if r["arm"] == arm and r["step"] == k]
            assert len(pts_k) == 3, (arm, k, len(pts_k))
            mean.append((sum(r["x_ex"] for r in pts_k) / 3, sum(r["one_item_short"] for r in pts_k) / 3))
        path[arm] = [(x0, 0.0)] + mean
    assert [round(x, 4) for x, _ in path["opd"][1:]] == [0.6333, 0.6517, 0.6587, 0.6614, 0.6876], path["opd"]
    assert all(y == 0.0 for _, y in path["opd"])
    assert [round(y, 4) for _, y in path["fkl"][1:]] == [0.0683, 0.1975, 0.1917, 0.2063, 0.1842], path["fkl"]
    for arm, c in (("fkl", C_FKL), ("opd", C_OPD)):
        xy = path[arm]
        axA.plot([PCT * x for x, _ in xy], [PCT * y for _, y in xy], color=c, lw=0.9, zorder=4)
        axA.plot([PCT * x for x, _ in xy[1:]], [PCT * y for _, y in xy[1:]], ls="", marker="^", ms=3.0, color=c,
                 mec="black", mew=0.35, zorder=5)
    axA.plot([PCT * x0], [0.0], ls="", marker="o", ms=3.4, mfc="white", mec="black", mew=0.6, zorder=6)
    SRC.extend(["", "EQ1 paths in (a) (EQ1_scored.json points, seed means of 3; step 0 = step0.x_ex, share 0; triangles, "
                "black edge; start hollow; step0_metrics):", "", "| arm | step | x_ex | y |", "|---|---|---|---|"])
    for arm in ("opd", "fkl"):
        for k, (x, y) in zip([0] + ksteps, path[arm]):
            SRC.append(f"| {arm} | {k} | {fmt(x)} | {fmt(y)} |")
    axA.set_xlabel("NDCG@1 ($\\times$100)")
    axB.set_xlabel("execution accuracy (\\%)")
    axA.set_ylabel("error share (\\%)")
    axA.set_xlim(53, 70)
    axB.set_xlim(58, 67)
    from matplotlib.lines import Line2D
    axB.legend(handles=[Line2D([], [], color=C_FKL, marker="s", ms=2.6, ls="", label="forward KL"),
                        Line2D([], [], color=C_OPD, marker="o", ms=2.6, ls="", label="reverse KL"),
                        Line2D([], [], color="#555555", marker="^", ms=3.0, mec="black", mew=0.35, lw=0.9,
                               label="weak start")], loc="upper right",
               bbox_to_anchor=(1.0, 1.0),
               frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
    for ax, letter, t in ((axA, letters[0], "ESCI"), (axB, letters[1], "Spider")):
        ax.text(-0.02, 1.02, f"({letter}) {t}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs)
    SRC.append("")
    if fig is not None:
        save(fig, "fig_quality")
        plt.close(fig)

def fig_srate(plt, ax=None):
    pass
    d = json.load((RES / "SRATE_scored.json").open())
    b, r = d["bases"], d["references"]
    pts = [("B_clean", r["B_clean_opd"]["x_remeasured"], r["B_clean_opd"]["42"])]
    pts += [(k, b[k]["x"], b[k]["opd"]["42"]) for k in ("b015", "b040", "b080", "b160")]
    pts += [(k, r[f"{k}_opd"]["x_remeasured"], r[f"{k}_opd"]["42"]) for k in ("B_E2", "B_E5")]
    fk = [("B_clean", r["B_clean_fkl"]["x_remeasured"], r["B_clean_fkl"]["42"]["Tr"])]
    fk += [(k, b[k]["x"], b[k]["fkl"]["42"]["Tr"]) for k in ("b015", "b040", "b080", "b160")]
    tr = {k: z["Tr"] for k, _, z in pts}
    assert round(tr["b015"], 4) == 0.1171 and tr["b015"] >= 0.02 and round(tr["b160"], 4) == 0.1787
    assert abs(tr["B_clean"] - 1 / 2400) < 1e-12 and round(b["b015"]["x"], 4) == 0.0009
    assert all(v >= 0.10 for _, _, v in fk), fk
    d2 = json.load((RES / "SRATE2_scored.json").open())
    b2 = d2["bases"]
    assert round(b2["b007"]["opd"]["42"]["Tr"], 4) == 0.0104 and b2["b007"]["opd"]["42"]["Tr"] < 0.02
    assert round(b2["b003"]["fkl"]["42"]["Tr"], 3) == 0.085 and b2["b007"]["fkl"]["42"]["Tr"] >= 0.10
    assert round(b2["b007"]["x"], 5) == 0.00076
    pts2 = [(k, b2[k]["x"], b2[k]["opd"]["42"]) for k in ("b003", "b007")]
    fk2 = [(k, b2[k]["x"], b2[k]["fkl"]["42"]["Tr"]) for k in ("b003", "b007")]
    xci = {"b007": b2["b007"]["x_cp95"], "b015": b["b015"]["x_cp95"]}
    a7 = json.load((RES / "A7E1A_scored.json").open())["rates"]
    assert a7["b015"]["k1_count"] == 332 and a7["b007"]["k1_count"] == 153 and a7["B_clean"]["k1_count"] == 34
    xa = {k: a7[k]["x"] for k in ("B_clean", "b007", "b015")}
    pts = [(k, xa.get(k, x), z) for k, x, z in pts]
    fk = [(k, xa.get(k, x), v) for k, x, v in fk]
    pts2 = [(k, xa.get(k, x), z) for k, x, z in pts2]
    fk2 = [(k, xa.get(k, x), v) for k, x, v in fk2]
    xci = {k: a7[k]["x_cp95"] for k in ("b007", "b015")}
    fig = None
    if ax is None:
        W, H = COLW, 1.45
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.40 / W, 0.36 / H, (W - 0.50) / W, (H - 0.46) / H])
    ax.set_xscale("log")
    ax.axhline(PCT * 0.19375, color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    fkr = fk + [t for t in fk2 if t[0] == "b007"]
    ax.plot([x for _, x, _ in fkr], [PCT * v for _, _, v in fkr], ls="", marker="s", ms=2.8, color=C_FKL, zorder=3,
            label="forward KL")
    ptr = pts + [t for t in pts2 if t[0] == "b007"]
    for k, x, z in ptr:
        lo, hi = z["Tr_ci95"]
        ax.plot([x, x], [PCT * lo, PCT * hi], color=C_OPD, lw=0.8, zorder=2)
    for k, (lo, hi) in xci.items():
        y = PCT * dict((kk, z["Tr"]) for kk, _, z in ptr)[k]
        ax.plot([lo, hi], [y, y], color=C_OPD, lw=0.5, alpha=0.6, zorder=2)
    ax.plot([x for _, x, _ in ptr], [PCT * z["Tr"] for _, _, z in ptr], ls="", marker="o", ms=3.0, color=C_OPD,
            zorder=4, label="reverse KL")
    k, x, z = [t for t in pts2 if t[0] == "b003"][0]
    ax.plot([x], [PCT * z["Tr"]], ls="", marker="o", ms=3.0, mfc="white", mec=C_OPD, mew=0.6, zorder=4)
    ax.plot([x], [PCT * dict((kk, v) for kk, _, v in fk2)["b003"]], ls="", marker="s", ms=2.8, mfc="white",
            mec=C_FKL, mew=0.6, zorder=3)
    e1b = json.load((RES / "A7E1B_scored.json").open())["starts"]
    for k in ("n025", "n060"):
        st = e1b[k]
        x = st["gate"]["rate"]
        o = st["opd1.00"] if isinstance(st["opd1.00"], dict) else None
        assert o is not None, k
        tr, (lo, hi) = o["Tr"], o["Tr_ci95"]
        ax.plot([x, x], [PCT * lo, PCT * hi], color=C_OPD, lw=0.8, zorder=2)
        ax.plot([x], [PCT * tr], ls="", marker="D", ms=2.8, color=C_OPD, zorder=4)
        ax.plot([x], [PCT * st["E1_G"]["fkl_Tr"]], ls="", marker="D", ms=2.6, color=C_FKL, zorder=3)
    assert round(e1b["n060"]["opd1.00"]["Tr"], 3) == 0.147 and e1b["n025"]["E1_1"]["class"] == "F0"
    ax.set_xlim(4e-5, 7e-2)
    ax.set_xticks([1e-4, 1e-3, 1e-2])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_ylim(-2, 27)
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel("student's own starting share of the error (\\%, log)")
    ax.set_ylabel("transmission (pp)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
    SRC.extend(["### Starting rate (fig_srate)", "", "SRATE_scored.json, step 42, x = own share on 1,800 training prompts "
                "x 8 (references: x_remeasured); OPD Tr and Tr_ci95, forward KL Tr (one seed per base).", "",
                "| base | x | OPD Tr [95%] | forward KL Tr |", "|---|---|---|---|"])
    fd = {k: v for k, _, v in fk}
    for k, x, z in pts:
        SRC.append(f"| {k} | {fmt(x)} | {fmt(z['Tr'])} [{fmt(z['Tr_ci95'][0])}, {fmt(z['Tr_ci95'][1])}] | "
                   + (fmt(fd[k]) if k in fd else "-") + " |")
    for k, x, z in pts2:
        SRC.append(f"| {k} (SRATE2{', not read: SW2' if k == 'b003' else ''}) | {fmt(x)} | {fmt(z['Tr'])} "
                   f"[{fmt(z['Tr_ci95'][0])}, {fmt(z['Tr_ci95'][1])}] | {fmt(dict((kk, v) for kk, _, v in fk2)[k])} |")
    SRC.append("")
    SRC.append(f"x intervals (95% Clopper-Pearson, drawn): b007 {fmt(xci['b007'][0])}-{fmt(xci['b007'][1])} "
               f"(SRATE2_scored.json), b015 {fmt(xci['b015'][0])}-{fmt(xci['b015'][1])} (SRATE_scored.json).")
    SRC.append("")
    if fig is not None:
        save(fig, "fig_srate")
        plt.close(fig)

def fig_when(plt):
    pass
    n = len(SRC)
    FS = 9
    with plt.rc_context({"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                         "ytick.labelsize": FS, "legend.fontsize": FS}):
        W, H = TEXTW, 1.80
        fig = plt.figure(figsize=(W, H))
        bot, hax = 0.40, 0.98
        axS = fig.add_axes([0.42 / W, bot / H, 2.15 / W, hax / H])
        axA = fig.add_axes([3.02 / W, bot / H, 1.55 / W, hax / H])
        axB = fig.add_axes([4.80 / W, bot / H, 1.42 / W, hax / H])
        fig_srate(plt, ax=axS)
        axS.text(-0.02, 1.02, "(a) starting rate", transform=axS.transAxes, ha="left", va="bottom", fontsize=FS)
        fig_quality(plt, axes=(axA, axB), letters=("b", "c"))
        from matplotlib.lines import Line2D
        for ax in (axS, axB):
            ax.get_legend().remove()
        fig.legend(handles=[Line2D([], [], color=C_FKL, marker="s", ls="", label="forward KL"),
                            Line2D([], [], color=C_OPD, marker="o", ls="", label="reverse KL"),
                            Line2D([], [], color="#555555", marker="^", mec="black", mew=0.35, lw=0.9,
                                   label="from a weaker start (b)"),
                            Line2D([], [], color="#555555", marker="D", ms=2.8, ls="", label="Qwen3.5-4B (a)"),
                            Line2D([], [], color=C_TEACH, lw=0.7, ls=(0, (4, 2.5)), label="teacher (a)")],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, handletextpad=0.4, columnspacing=1.2,
                   borderaxespad=0.1)
        save(fig, "fig_when")
        plt.close(fig)
    del SRC[n:]

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ONLY.update(sys.argv[1:])
    known = {"fig_rate", "fig_quality", "fig_srate", "fig_dynamics", "fig_long", "fig_arms", "fig_template", "fig_math_budget", "fig_lr", "fig_selection", "fig_greedy", "fig_when"}
    assert ONLY <= known, f"unknown figure name(s): {sorted(ONLY - known)}"
    style(plt)
    if not ONLY or "fig_dynamics" in ONLY: fig_dynamics(plt)
    if not ONLY or "fig_long" in ONLY: fig_long(plt)
    if not ONLY or "fig_arms" in ONLY: fig_arms(plt)
    if not ONLY or "fig_template" in ONLY: fig_template2(plt)
    if not ONLY or "fig_math" in ONLY: fig_math(plt)
    if not ONLY or "fig_lr" in ONLY: fig_lr(plt)
    if not ONLY or "fig_selection" in ONLY: fig_selection(plt)
    if not ONLY or "fig_greedy" in ONLY: fig_greedy(plt)
    with plt.rc_context({k: 9 for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                                         "ytick.labelsize", "legend.fontsize")}):
        if not ONLY or "fig_rate" in ONLY: fig_rate(plt)
    if not ONLY or "fig_quality" in ONLY: fig_quality(plt)
    if not ONLY or "fig_srate" in ONLY: fig_srate(plt)
    if not ONLY or "fig_when" in ONLY: fig_when(plt)
    sys.path.insert(0, str(REPO / "scripts"))
    from fig_common import write_sources
    write_sources("figs_v2", ["## Results figures, display plan v2 (auto-generated by `scripts/fig_figs.py`; "
                              "do not edit by hand)", ""] + SRC)
    print("figs_v2: " + ", ".join(sorted(ONLY or known)) + " -> out/figures/ (figure_sources.md rewritten in full)")

if __name__ == "__main__":
    main()
