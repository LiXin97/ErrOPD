#!/usr/bin/env python3
"""Draw the delay figure."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import fig_figs as v2
import transient as at
from fig_common import write_sources

RES = REPO / "results"
TT = RES / "transient_transmission.md"
N_ROLL = 128
PCT = 100.0
C_OPD_LOW = "#5A9CC9"
FKL_RATE = "1e-5"
RATE_TEX = {"1e-6": r"$10^{-6}$", "3e-6": r"$3{\times}10^{-6}$", "1e-5": r"$10^{-5}$"}

def ids_listed_in_md(kind):
    pass
    for ln in TT.read_text().splitlines():
        if ln.startswith(f"- {kind}:"):
            return re.findall(r"`([^`]+)`", ln)
    raise KeyError(kind)

def panel_a_runs():
    opd = [r for r in at.OPD_RUNS]
    fkl = [r for r in at.FKL_RUNS if r["rate"] == FKL_RATE]
    md_opd, md_fkl = ids_listed_in_md("opd"), ids_listed_in_md("fkl")
    assert [r["run_id"] for r in at.OPD_RUNS] == md_opd and [r["run_id"] for r in at.FKL_RUNS] == md_fkl, \
        "the run lists in transient.py and transient_transmission.md section 0 differ"
    assert len(md_opd) == 18 and len(md_fkl) == 15 and len(fkl) == 5
    for r in opd + fkl:
        f = r["dir"] / "ckpt" / "erropd_rollouts.jsonl"
        if not f.exists():
            sys.exit(f"missing raw file: {f}")
    return opd, fkl

def counts(run):
    pass
    s = at.rollout_series(run["dir"])
    assert sorted(s) == at.STEPS, (run["run_id"], sorted(s)[:3], sorted(s)[-3:])
    assert all(s[k][1] == N_ROLL and s[k][2] == N_ROLL for k in at.STEPS), (run["run_id"], "eligible != 128")
    return [s[k][0] for k in at.STEPS]

def check_against_md(opd, fkl, ca):
    pass
    row = {r["run_id"]: ca[r["run_id"]] for r in opd + fkl}
    assert row["RXB_opd_lr1e-5_s3003"][32:36] == [127, 115, 128, 109], row["RXB_opd_lr1e-5_s3003"][32:36]
    assert row["RXB_opd_lr1e-5_s3003"][36:42] == [0] * 6, row["RXB_opd_lr1e-5_s3003"][36:42]
    assert row["E3_LR_lam1.00_s1001"][28] == 88, row["E3_LR_lam1.00_s1001"][28]
    assert row["E3_LR_lam1.00_s1001"][31:33] == [88, 71], row["E3_LR_lam1.00_s1001"][31:33]
    for r in opd:
        if r["rate"] == "1e-6":
            assert max(row[r["run_id"]]) <= 1, (r["run_id"], max(row[r["run_id"]]))
        if r["rate"] == "3e-6":
            assert max(row[r["run_id"]]) <= 2, (r["run_id"], max(row[r["run_id"]]))
    assert [r["seed"] for r in fkl] == [3001, 3002, 3003, 3004, 3005] and {r["rate"] for r in fkl} == {FKL_RATE}
    fkl5 = [row[r["run_id"]][4] for r in fkl]
    assert fkl5 == [29, 53, 70, 28, 39], fkl5
    md = md_fkl_rows(FKL_RATE)
    for r in fkl:
        assert row[r["run_id"]] == md[r["seed"]], (r["run_id"], row[r["run_id"]], md[r["seed"]])
    peak = max(max(row[r["run_id"]]) for r in opd if r["rate"] == "1e-5")
    assert peak == N_ROLL, peak
    return {"s3003 steps 33-36": row["RXB_opd_lr1e-5_s3003"][32:36], "s3003 steps 37-42": row["RXB_opd_lr1e-5_s3003"][36:42],
            "s1001 step 29": row["E3_LR_lam1.00_s1001"][28], "s1001 steps 32-33": row["E3_LR_lam1.00_s1001"][31:33],
            f"fkl {FKL_RATE} step 5": fkl5}

def md_fkl_rows(rate):
    pass
    lines = TT.read_text().splitlines()
    i0 = lines.index("### 1.1 Training rollouts, every step from 1 to 42 (count of 128)")
    i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].startswith("### "))
    sec = lines[i0:i1]
    j = sec.index("**on-policy forward KL.**")
    k0 = next(i for i in range(j, len(sec)) if sec[i].startswith("```"))
    k1 = next(i for i in range(k0 + 1, len(sec)) if sec[i].startswith("```"))
    head = sec[k0 + 1].split()
    assert head[0] == "step" and [int(x) for x in head[1:]] == at.STEPS, head[:4]
    out = {}
    for ln in sec[k0 + 2:k1]:
        f = ln.split()
        if f[0] == rate:
            assert f[1] == "blkA" and f[2].startswith("s") and len(f) == 3 + 42, f[:3]
            out[int(f[2][1:])] = [0 if c == "." else int(c) for c in f[3:]]
    assert sorted(out) == [3001, 3002, 3003, 3004, 3005], sorted(out)
    return out

CKPTS = ("42", "84", "126", "168", "210")

def long3(lg):
    pass
    d = json.load((RES / "LONG3_scored.json").open())
    assert d["base_share"] == 0.0 and d["dry_run"] is False, (d["base_share"], d["dry_run"])
    assert abs(d["P_T"] - 155 / 800) < 1e-12, d["P_T"]

    def recs(arm, n):
        runs = d[arm]["runs"]
        assert d[arm]["n"] == n == len(runs), (arm, d[arm]["n"], runs)
        out = {int(r.rsplit("_s", 1)[1]): d["per_run"][r] for r in runs}
        assert all(sorted(rec["shares"]) == sorted(CKPTS) for rec in out.values()), arm
        return out

    F, O = recs("F210", 3), recs("O3", 8)
    assert sorted(F) == [1001, 1002, 1003] and sorted(O) == list(range(1001, 1009)), (sorted(F), sorted(O))
    fmean = {k: sum(F[sd]["shares"][k] for sd in sorted(F)) / 3 for k in CKPTS}
    for k in CKPTS:
        st = d["F210"]["steps"][k]
        assert st["share_per_seed"] == [F[sd]["shares"][k] for sd in sorted(F)], (k, st["share_per_seed"])
        assert abs(st["mean"] - fmean[k]) < 1e-12, (k, st["mean"], fmean[k])
    assert [round(fmean[k], 3) for k in CKPTS] == [0.119, 0.196, 0.198, 0.214, 0.212], fmean
    assert d["F210"]["PRIMARY_F_ge_OPD_every_step"] == "holds"
    for k in CKPTS:
        v = [rec["shares"][k] for rec in lg.values()]
        assert abs(sum(v) / 8 - d["opd_1e-6_mean"][k]) < 1e-12 and max(v) == d["opd_1e-6_max_seed"][k], k
    esc = {sd: rec["escape_step"] for sd, rec in O.items() if rec["escaped"]}
    assert d["O3"]["K"] == 3 and esc == {1006: 42, 1007: 84, 1004: 210}, (d["O3"]["K"], esc)
    assert d["O3"]["escape_steps"] == [42, 84, 210] and d["O3"]["K_by_step"] == dict(zip(CKPTS, [1, 2, 2, 2, 3]))
    assert [O[1004]["shares"][k] for k in CKPTS] == [0.0, 0.0, 0.0, 0.0, 0.1375], O[1004]["shares"]
    assert [O[sd]["shares"]["210"] for sd in (1006, 1007, 1004)] == [0.2, 0.225, 0.1375]
    assert O[1006]["shares"]["42"] == 0.20625
    assert all(v == 0.0 for sd, rec in O.items() if not rec["escaped"] for v in rec["shares"].values())
    omean = {k: sum(O[sd]["shares"][k] for sd in O) / 8 for k in CKPTS}
    assert all(abs(omean[k] - d["O3"]["seed_mean_by_step"][k]) < 1e-12 for k in CKPTS)
    assert [round(omean[k], 3) for k in CKPTS] == [0.026, 0.050, 0.055, 0.055, 0.070], omean
    pinned = {"F210 seed mean 42..210": [round(fmean[k], 3) for k in CKPTS],
              "O3 K": f"{len(esc)} of 8", "O3 first escape": {f"s{sd}": st for sd, st in sorted(esc.items())},
              "O3 s1004": [O[1004]["shares"][k] for k in CKPTS], "O3 seed mean 42..210": [round(omean[k], 3) for k in CKPTS]}
    return F, O, fmean, omean, pinned

C_D210 = "#0B3C5D"

def d210_hook(ax):
    pass
    d = json.load((RES / "RXD210_scored.json").open())["D"]["RD2"]
    seeds = [str(sd) for sd in d["seeds"]]
    assert seeds == ["3001", "3002", "3003", "3004", "3005"] and d["n"] == 5 and d["escapes"] == 5 and d["holds"] is False
    sh = d["shares"]
    assert all(sh[sd]["42"] == 0.0 for sd in seeds)
    assert all(d["escape_step"][sd] == 84 for sd in seeds)
    assert [round(sh[sd]["84"], 3) for sd in seeds] == [0.141, 0.22, 0.186, 0.169, 0.186], [sh[sd]["84"] for sd in seeds]
    assert [round(sh[sd]["210"], 3) for sd in seeds] == [0.216, 0.206, 0.199, 0.206, 0.195]
    ks = [0] + [int(k) for k in CKPTS]
    ser, src = {}, []
    for sd in seeds:
        ys = [0.0] + [PCT * sh[sd][k] for k in CKPTS]
        ax.plot(ks, ys, color=C_D210, lw=0.8, ls=(0, (3, 1.5)), marker="o", markersize=1.6, zorder=3)
        ser[f"d210 seed {sd}"] = list(zip(ks, ys))
        src.append(f"- full-vocabulary reverse KL seed {sd} (escaped at 84): share "
                   + " / ".join(v2.fmt(sh[sd][k]) for k in CKPTS))
    head = ["", "Full-vocabulary reverse KL over 210 steps (E2, D210; dark blue, dashed, lw 0.8): `RXD210_scored.json` "
            "`D.RD2.shares`, seeds 3001-3005, lr 1e-6, same pair and sampler; RD1 holds, RD2 fails (5 of 5 escape, all "
            "first at step 84; RXD210_verdict.md e3d8ef0). Asserted: 0 of 800 at 42 in every seed; step-84 and step-210 "
            "shares equal the verdict's table at three decimals."]
    return ser, head + src

def densify(pts, n=20):
    pass
    out = []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        out += [(xa + (xb - xa) * i / n, ya + (yb - ya) * i / n) for i in range(n)]
    return out + [pts[-1]]

def no_overlap(fig, ax, artists, series_xy, pad_pt=1.0):
    pass
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    for art in artists:
        bb = art.get_window_extent(rend).expanded(1.0, 1.0)
        x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
        for name, pts in series_xy.items():
            disp = ax.transData.transform(densify(pts))
            hit = [(round(float(px), 1), round(float(py), 1)) for (px, py) in disp if x0 <= px <= x1 and y0 <= py <= y1]
            assert not hit, f"{type(art).__name__} covers {len(hit)} point(s) of {name}: {hit[:3]}"

def apart(fig, a, b, gap_pt):
    pass
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ba, bb = a.get_window_extent(rend), b.get_window_extent(rend)
    g = gap_pt * fig.dpi / 72
    sep = max(bb.x0 - ba.x1, ba.x0 - bb.x1, bb.y0 - ba.y1, ba.y0 - bb.y1)
    assert sep >= g, f"{type(a).__name__} and {type(b).__name__} are {sep * 72 / fig.dpi:.2f} pt apart (< {gap_pt})"
    return sep * 72 / fig.dpi

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    v2.style(plt)
    plt.rcParams["figure.dpi"] = 300
    FS = 9
    plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                         "ytick.labelsize": FS, "legend.fontsize": FS})

    opd, fkl = panel_a_runs()
    ca = {r["run_id"]: counts(r) for r in opd + fkl}
    pinned = check_against_md(opd, fkl, ca)
    fkl_mean = [sum(ca[r["run_id"]][i] for r in fkl) / len(fkl) for i in range(42)]
    teacher = v2.qwen3_teacher()
    lg = json.load((RES / "LONG210b_scored.json").open())["seeds"]
    F, O, fmean, omean, pinned3 = long3(lg)

    W, H = v2.TEXTW, 2.15
    fig = plt.figure(figsize=(W, H))
    bot, hax = 0.38, 1.53
    axA = fig.add_axes([0.42 / W, bot / H, 2.15 / W, hax / H])
    axB = fig.add_axes([2.99 / W, bot / H, 1.85 / W, hax / H])
    axC = fig.add_axes([5.06 / W, bot / H, 1.12 / W, hax / H])

    xs = list(at.STEPS)
    series = {}
    look = {"1e-6": (C_OPD_LOW, 0.6, 2), "3e-6": (C_OPD_LOW, 0.6, 2), "1e-5": (v2.C_OPD, 0.8, 4)}
    for r in opd:
        col, lw, z = look[r["rate"]]
        ys = [PCT * c / N_ROLL for c in ca[r["run_id"]]]
        axA.plot(xs, ys, color=col, lw=lw, zorder=z, solid_joinstyle="round")
        series[r["run_id"]] = list(zip(xs, ys))
    ysf = [PCT * m / N_ROLL for m in fkl_mean]
    axA.plot(xs, ysf, color=v2.C_FKL, lw=1.0, zorder=5)
    series["fkl mean"] = list(zip(xs, ysf))
    v2._teacher(axA, PCT * teacher, 0, 43)
    series["teacher"] = [(0, PCT * teacher), (43, PCT * teacher)]
    axA.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    axA.set_xlim(0, 43)
    axA.set_xticks([1, 14, 28, 42])
    axA.set_ylim(-4, 108)
    axA.set_yticks([0, 50, 100])
    axA.set_ylabel(r"error share (\%)")
    axA.set_xlabel("training step", labelpad=1)
    title = axA.set_title("(a) training rollouts, 42 steps", loc="left", pad=2)
    n_hi = sum(r["rate"] == "1e-5" for r in opd)
    n_lo = sum(r["rate"] in ("1e-6", "3e-6") for r in opd)
    assert (n_hi, n_lo, len(fkl)) == (8, 10, 5), (n_hi, n_lo, len(fkl))
    handles = [Line2D([], [], color=c, lw=1.4) for c in (v2.C_FKL, C_OPD_LOW, v2.C_OPD)]
    labels = [f"fwd.\\ KL {RATE_TEX[FKL_RATE]}, mean",
              f"rev.\\ KL {RATE_TEX['1e-6']}, {RATE_TEX['3e-6']}",
              f"rev.\\ KL {RATE_TEX['1e-5']}"]
    leg = axA.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, 0.95), frameon=False, ncol=1,
                     handlelength=1.0, handletextpad=0.4, labelspacing=0.15, borderpad=0.1, borderaxespad=0.0)
    t_teach = axA.text(11.5, PCT * teacher + 3, "teacher", color=v2.C_TEACH, fontsize=FS, ha="center", va="bottom")
    no_overlap(fig, axA, [*leg.get_texts(), *leg.legend_handles, t_teach], series, pad_pt=1.5)
    gap_title = apart(fig, leg, title, 2.0)

    def seed_lines(ax, recs, col):
        out = {}
        for sd, rec in sorted(recs.items()):
            ks = [0] + [int(k) for k in CKPTS]
            ys = [0.0] + [PCT * rec["shares"][k] for k in CKPTS]
            lw = 1.0 if rec["escaped"] else 0.6
            ax.plot(ks, ys, color=col, lw=lw, marker="o", markersize=1.8, zorder=3 if rec["escaped"] else 2)
            out[f"seed {sd}"] = list(zip(ks, ys))
        return out

    def frame_210(ax, ylab):
        v2._teacher(ax, PCT * teacher, -6, 216)
        ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
        ax.set_xlim(-6, 216)
        ax.set_xticks([0, 42, 84, 126, 168, 210])
        ax.set_ylim(-2, 32)
        ax.set_yticks([0, 10, 20])
        if ylab:
            ax.set_ylabel(r"error share (\%)")
        ax.set_xlabel("training step", labelpad=1)
        return {"teacher": [(-6, PCT * teacher), (216, PCT * teacher)]}

    lgi = {int(sd): rec for sd, rec in lg.items()}
    ser_b = seed_lines(axB, lgi, v2.C_OPD)
    src_b = [f"- seed {sd} ({'escaped' if rec['escaped'] else 'stayed'}): share "
             + " / ".join(v2.fmt(rec["shares"][k]) for k in CKPTS) for sd, rec in sorted(lgi.items())]
    esc = [rec for rec in lg.values() if rec["escaped"]]
    end = sorted(rec["shares"]["210"] for rec in esc)
    stay_max = max(v for rec in lg.values() if not rec["escaped"] for v in rec["shares"].values())
    assert len(esc) == 3 and len(lg) == 8 and 0.17 <= end[0] and end[-1] <= 0.20, end
    assert stay_max == 1 / 800, stay_max
    ks = [0] + [int(k) for k in CKPTS]
    for sd in sorted(F):
        axB.plot(ks, [0.0] + [PCT * F[sd]["shares"][k] for k in CKPTS], color=v2.C_FKL, lw=0.5, alpha=0.45, zorder=2)
    ysm = [0.0] + [PCT * fmean[k] for k in CKPTS]
    axB.plot(ks, ysm, color=v2.C_FKL, lw=1.1, marker="o", markersize=1.8, zorder=4)
    ser_b["fkl mean"] = list(zip(ks, ysm))
    for sd in sorted(F):
        ser_b[f"fkl seed {sd}"] = list(zip(ks, [0.0] + [PCT * F[sd]["shares"][k] for k in CKPTS]))
    ser_d210, src_d210 = d210_hook(axB)
    ser_b.update(ser_d210)
    ser_b.update(frame_210(axB, ylab=True))
    axB.set_ylim(-2, 47)
    axB.set_title("(b) 210 steps at " + RATE_TEX["1e-6"], loc="left", pad=2)
    tb = [axB.text(-3, 43.0, "reverse KL: 3 of 8 acquire it", color=v2.C_OPD, fontsize=FS, ha="left", va="center"),
          axB.text(213, 6.0, "5 stay", color=v2.C_OPD, fontsize=FS, ha="right", va="center"),
          axB.text(-3, 30.5, "forward KL, 3 seeds", color=v2.C_FKL, fontsize=FS, ha="left", va="center"),
          axB.text(-3, 36.75, "full-vocabulary reverse KL: 5 of 5", color=C_D210, fontsize=FS, ha="left", va="center")]
    no_overlap(fig, axB, tb, ser_b, pad_pt=1.5)
    gap_b = apart(fig, tb[0], tb[3], 1.0)
    print(f"(b) label gap {gap_b:.1f} pt")
    top_a = axA.title.get_window_extent(fig.canvas.get_renderer()).y1 * 72 / fig.dpi
    assert top_a <= H * 72, ("panel (a) title clipped", top_a, H * 72)

    ser_c = seed_lines(axC, O, v2.C_OPD)
    ser_c.update(frame_210(axC, ylab=False))
    axC.set_xticks([0, 70, 140, 210])
    axC.set_title("(c) at " + RATE_TEX["3e-6"], loc="left", pad=2)
    k_c = sum(rec["escaped"] for rec in O.values())
    tc = [axC.text(3, 27.0, f"{k_c} of {len(O)} acquire it", color=v2.C_OPD, fontsize=FS, ha="left", va="center"),
          axC.text(150, 4.5, f"{len(O) - k_c} stay", color=v2.C_OPD, fontsize=FS, ha="right", va="center")]
    no_overlap(fig, axC, tc, ser_c, pad_pt=1.5)
    v2.save(fig, "fig_delay")
    plt.close(fig)

    S = ["## Delay, not block (fig_delay; auto-generated by `scripts/fig_delay.py`; do not edit by hand)", "",
         "out/figures/fig_delay.{pdf,png}, " + f"{W * 2.54:.2f} x {H * 2.54:.2f} cm (ACL text width, three panels side by side). All y axes"
         " are in percent of samples (since 2026-10-02): the figure draws 100 x every share below; the values here are"
         " the fractions and counts read from the source files.", "",
         "### Panel (a): training rollouts, steps 1-42", "",
         "Share of each step's 128 training rollouts (all eligible) that carry the injected error (`k1_trunc_realids`, "
         "the labeller of `eval_labels.py`), recomputed from `<run>/ckpt/erropd_rollouts.jsonl` through "
         "`transient.rollout_series()` for the 18 reverse-KL OPD runs and the 5 forward-KL runs at 1e-5 listed by "
         "exact id in `transient_transmission.md` section 0 (the lists are asserted equal). One thin line per OPD seed: "
         "the 8 seeds at 1e-5 in the OPD blue (lw 0.8), the 10 seeds at 1e-6 and 3e-6 in one lighter tint (lw 0.6; no "
         "count above 2 of 128). The orange line is the forward-KL 5-seed mean count / 128 at 1e-5, the rate of OPD's "
         "excursions, so the panel compares the two objectives at one learning rate (it was the 1e-6 mean before "
         "2026-10-02). Dashed: the injected teacher's own share on the "
         "**evaluation sampler** (200 held-out ESCI prompts x 4 at T = 1, 155/800 = 0.19375, FKL_scored.json "
         "teacher_share minus base_share), drawn across a training-rollout axis because the teacher's share on the "
         "training prompts was not measured. Full 18 x 42 grid: `transient_transmission.md` section 1.1.", "",
         "| arm | rate | block | seed | max count (step) | run total of 5,376 |", "|---|---|---|---|---|---|"]
    for r in opd + fkl:
        c = ca[r["run_id"]]
        m = max(c)
        S.append(f"| {'OPD' if r['obj'] == 'opd' else 'forward KL'} | {r['rate']} | {r['block']} | {r['seed']} | "
                 f"{m}/128 = {m / N_ROLL:.4f} (step {c.index(m) + 1}) | {sum(c)} |")
    S += ["", f"Forward-KL {FKL_RATE} mean count per step (1..42): " + ", ".join(f"{m:g}" for m in fkl_mean) + ".", "",
          "Asserted against `transient_transmission.md` section 1.1: " + "; ".join(f"{k} = {v}" for k, v in pinned.items())
          + f"; all 5 x 42 forward-KL {FKL_RATE} counts equal the section's five forward-KL {FKL_RATE} rows (seeds "
          "3001-3005); every 1e-6 OPD seed at most 1 of 128 at any step; every 3e-6 OPD seed at most 2; the largest "
          "1e-5 count is 128 of 128; every run has steps 1..42 with 128 eligible of 128 dumped. Layout checks: no "
          "legend row (text or handle) and not the 'teacher' label comes within 1.5 pt of any plotted line (sampled 20 "
          f"times per segment), and the legend is {gap_title:.1f} pt below the panel title.", "",
          "### Panel (b): held-out samples, 210 steps at 1e-6", "",
          "LONG210b_scored.json seeds (1001-1003 first batch, 1004-1008 replication): share of 800 evaluation samples at "
          "steps 42, 84, 126, 168, 210; step 0 is the clean start's 0/800 (base_share). Learning rate 1e-6 at every step "
          "(trainer console actor/lr). Same values as fig_long panel (a); drawn in one hue, escaped seeds lw 1.0, stayed "
          "lw 0.6. Dashed: teacher 155/800 on the same sampler. Asserted: 3 of 8 escaped, ending at "
          + " / ".join(v2.fmt(v) for v in end) + " (teacher 0.194); the other five at most 1 of 800 at any checkpoint. The "
          "seed mean of these eight equals LONG3_scored.json `opd_1e-6_mean` and their maximum `opd_1e-6_max_seed` at "
          "every checkpoint (asserted)."] + src_b + [
          "", "Forward KL at the same rate and budget (orange; LONG3 arm F210, `LONG3_scored.json` `per_run` of the runs in "
          "`F210.runs`; lr 1e-6 per the LONG3 lock; same pair, base 0/800 so step 0 is drawn at 0): the seed mean (lw 1.1, "
          "markers) and the three seeds (lw 0.5, alpha 0.45). Seed mean at steps 42..210: "
          + " / ".join(v2.fmt(fmean[k]) for k in CKPTS) + "; asserted equal to `F210.steps[k].mean` and, rounded to three "
          "decimals, to LONG3_verdict.md's 0.119 / 0.196 / 0.198 / 0.214 / 0.212."] + [
          f"- forward-KL seed {sd}: share " + " / ".join(v2.fmt(F[sd]["shares"][k]) for k in CKPTS) for sd in sorted(F)] + src_d210 + [
          "",
          "### Panel (c): held-out samples, 210 steps of OPD at 3e-6", "",
          "LONG3 arm O3 (`LONG3_scored.json` `per_run` of the runs in `O3.runs`, seeds 1001-1008, lr 3e-6 per the LONG3 "
          "lock), share of the same 800 evaluation samples at steps 42..210, drawn as (b): escaped seeds lw 1.0, stayed lw "
          "0.6, step 0 at base 0/800. Escape flags and first-escape steps are the scorer's (`escaped`, `escape_step`). "
          "Asserted against LONG3_verdict.md: K = 3 of 8 (`O3.K`), first escapes s1006 at 42, s1007 at 84, s1004 at 210 "
          "(`escape_steps` 42 / 84 / 210; `K_by_step` 1 / 2 / 2 / 2 / 3); s1004 is 0 at steps 42-168 and 0.1375 at 210; the "
          "escaped seeds' step-210 shares are 0.2 / 0.225 / 0.1375; the five other seeds are exactly 0 at every checkpoint; "
          "the 8-seed mean (" + " / ".join(v2.fmt(omean[k]) for k in CKPTS) + ") equals `O3.seed_mean_by_step` and, at "
          "three decimals, the verdict's 0.026 / 0.050 / 0.055 / 0.055 / 0.070."] + [
          f"- seed {sd} ({'escaped at ' + str(O[sd]['escape_step']) if O[sd]['escaped'] else 'stayed'}): share "
          + " / ".join(v2.fmt(O[sd]["shares"][k]) for k in CKPTS) for sd in sorted(O)] + [""]
    d = RES / "figure_sources"
    head = "# Figure sources (auto-generated; one section per figure script; do not edit by hand)\n\n"
    rebuilt = head + "\n".join(p.read_text() for p in sorted(d.glob("*.md")))
    current = (RES / "figure_sources.md").read_text()
    assert rebuilt == current, "figure_sources.md is not the concatenation of figure_sources/*.md as they stand; not rewriting"
    write_sources("fig_delay", S)
    print(f"fig_delay -> out/figures/fig_delay.{{pdf,png}} ({W * 2.54:.2f} x {H * 2.54:.2f} cm); "
          f"figure_sources.md section fig_delay written; asserted: {pinned}; {pinned3}")

if __name__ == "__main__":
    main()
