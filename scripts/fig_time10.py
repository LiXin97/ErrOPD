#!/usr/bin/env python3
"""Draw the time-to-transmission figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import fig_figs as v2

DATA = REPO / "results/time_to10_counts.json"
W, BAR = 7, 0.10
C_OPD, C_FKL = v2.C_OPD, v2.C_FKL
TINT_OPD = ["#0072B2", "#5A9CC9", "#9CC5E3", "#C9DFF0"]
TINT_FKL = ["#D55E00", "#E8894A", "#F2B48A"]

PANELS = [
    ("(a) learning rate, 42 steps", 42, [
        ("rev.\\ $10^{-6}$", "RXB_opd_lr1e-6", C_OPD, "-"),
        ("rev.\\ $3{\\times}10^{-6}$", "RXB_opd_lr3e-6", C_OPD, (0, (3, 1.5))),
        ("rev.\\ $10^{-5}$", "RXB_opd_lr1e-5", C_OPD, (0, (1, 1.2))),
        ("fwd.\\ $10^{-6}$", "RXB_fkl_lr1e-6", C_FKL, "-"),
        ("fwd.\\ $3{\\times}10^{-6}$", "RXB_fkl_lr3e-6", C_FKL, (0, (3, 1.5))),
        ("fwd.\\ $10^{-5}$", "RXB_fkl_lr1e-5", C_FKL, (0, (1, 1.2))),
    ]),
    ("(b) rev.\\ KL at the decision ${\\times}k$", 42, [
        ("$k=1$", "DOSE_k1s", TINT_OPD[3], "-"),
        ("$k=3$", "DOSE_k3", TINT_OPD[2], "-"),
        ("$k=10$", "DOSE_k10", TINT_OPD[1], "-"),
        ("$k=30$", "DOSE_k30", TINT_OPD[0], "-"),
        ("fwd.\\ norm", "FI_S", "#000000", (0, (3, 1.5))),
        ("fwd.\\ KL", "FI_K", C_FKL, "-"),
    ]),
    ("(c) 210 steps", 210, [
        ("rev.\\ $10^{-6}$", ("LONG210_A", "LONG210b_A"), C_OPD, "-"),
        ("rev.\\ $3{\\times}10^{-6}$", "LONG3_O3", C_OPD, (0, (3, 1.5))),
        ("full vocab.", "RXD_D210", "#000000", "-"),
        ("fwd.\\ $10^{-6}$", "LONG3_F", C_FKL, "-"),
    ]),
]

def first_step(ser):
    st = sorted(int(s) for s in ser)
    k = [ser[str(s)][0] for s in st]
    e = [ser[str(s)][1] for s in st]
    for i in range(W - 1, len(st)):
        K, E = sum(k[i - W + 1:i + 1]), sum(e[i - W + 1:i + 1])
        if E and K / E >= BAR:
            return st[i]
    return None

def arm_steps(d, prefixes):
    prefixes = (prefixes,) if isinstance(prefixes, str) else prefixes
    runs = sorted(r for r in d if r.rsplit("_s", 1)[0] in prefixes)
    return {r: first_step(d[r]) for r in runs}

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v2.style(plt)
    d = json.loads(DATA.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(v2.TEXTW, 3.7))
    axes = axes.ravel()
    for ax, (title, T, arms) in zip(axes, PANELS):
        for label, pref, col, ls in arms:
            fs = arm_steps(d, pref)
            n = len(fs)
            assert n, pref
            xs = list(range(0, T + 1))
            ys = [100.0 * sum(1 for s in fs.values() if s is not None and s <= x) / n for x in xs]
            ax.step(xs, ys, where="post", color=col, ls=ls, lw=1.2, label=f"{label} ({n})")
            print(f"{title[:3]} {label:32s} n={n} " + " ".join(str(s) for s in sorted(fs.values(), key=lambda s: (s is None, s or 0))))
        ax.set_title(title)
        ax.set_xlim(0, T)
        ax.set_ylim(-3, 103)
        ax.set_xticks({42: [0, 14, 28, 42], 105: [0, 35, 70, 105], 210: [0, 70, 140, 210]}[T])
        ax.set_xlabel("training step")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, handlelength=1.5, columnspacing=0.8,
                  borderaxespad=0.0, labelspacing=0.15, handletextpad=0.4, fontsize=8)
    axes[3].axis("off")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("seeds at 10\\% (\\%)")
    fig.subplots_adjust(left=0.07, right=0.83, top=0.94, bottom=0.11, wspace=0.75, hspace=0.62)
    v2.save(fig, "fig_time10")

if __name__ == "__main__":
    main()
