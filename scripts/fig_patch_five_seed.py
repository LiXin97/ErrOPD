#!/usr/bin/env python3
"""Add the five-seed values to the tables."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from fig_tables import pct

S = json.loads((REPO / "results/A7E3_scored.json").read_text())
TAB = REPO / "paper_erropd/tables"
ROWS = {
    "qwen3": (r"Qwen3-1.7B $\leftarrow$ 4B, ESCI &", "Qwen3 1.7B (4B) &"),
    "llama": (r"Llama-3.2-3B $\leftarrow$ 3.1-8B, Fashion &", "Llama 3B (8B) &"),
    "qwen3.5_0p8b": (r"Qwen3.5-0.8B $\leftarrow$ 9B, Fashion &", "Qwen3.5 0.8B (9B) &"),
    "qwen3.5_2b": (r"\quad Qwen3.5-2B &", r"\quad 2B (9B) &"),
    "qwen3.5_4b": (r"\quad Qwen3.5-4B &", r"\quad 4B (9B) &"),
}

def cell(arm):
    tr, (lo, hi) = arm["Tr"], arm["Tr_ci95"]
    v = pct(tr)
    if v == "0" and lo == 0 and hi == 0:
        return "0"
    return f"{v} {{\\scriptsize[{pct(lo)}, {pct(hi)}]}}"

def signed(x):
    return ("$+$" if x >= 0 else "$-$") + pct(abs(x))

def main():
    mt = (TAB / "tab_main.tex").read_text().splitlines()
    ct = (TAB / "tab_contrast.tex").read_text().splitlines()
    for key, (mrow, crow) in ROWS.items():
        st = S["settings"][key]
        assert st["status"] == "complete" and st["seeds"] == [1001, 1002, 1003, 1011, 1012], key
        ob = st["objectives"]
        fwd, rev = st["arms"][ob["fwd"]]["42"], st["arms"][ob["rev"]]["42"]
        hits = [i for i, ln in enumerate(mt) if ln.startswith(mrow)]
        i = hits[0]
        c = [x.strip() for x in mt[i].rstrip("\\ ").split(" & ")]
        assert len(c) == 7, (key, c)
        c[3], c[5] = cell(rev), cell(fwd)
        mt[i] = c[0] + " & " + " & ".join(c[1:]) + " \\\\"
        pr = st["pairs"]
        d, lo, hi = pr["t95"]
        p = pr["sign_test"]["p_one_sided"]
        j = [k for k, ln in enumerate(ct) if ln.startswith(crow)][0]
        tail = ct[j].split("\\\\", 1)[1]
        ct[j] = f"{crow} {signed(d)} [{pct(lo)}, {pct(hi)}] & {p:.3f} \\\\{tail}"
    po = S["E3_2"]["pooled"]
    d, lo, hi = po["t95"]
    n = po["sign_test"]["n_pairs"]
    assert po["sign_test"]["n_pos"] == n
    e = f"{po['sign_test']['p_one_sided']:.0e}"
    man, ex = e.split("e")
    j = [k for k, ln in enumerate(ct) if ln.startswith(r"\textbf{pooled}")][0]
    ct[j] = (f"\\textbf{{pooled}} ({n}/{n} $>0$) & {signed(d)} [{pct(lo)}, {pct(hi)}] & "
             f"$\\mathbf{{{man}\\cdot10^{{{int(ex)}}}}}$ \\\\")
    hdr = "% A7E3 (2026-10-08): five-seed rows patched by scripts/fig_patch_five_seed.py from results/A7E3_scored.json (3a643ac)."
    for lines in (mt, ct):
        if hdr not in lines:
            lines.insert(0, hdr)
    q = pct(S["settings"]["qwen3"]["arms"]["FKL"]["42"]["Tr"])
    for name, pat in (("tab_e3.tex", r"(\\hfkl\{on-policy forward KL\} & 0\.1 & 0 & \\hit\\textbf\{)[0-9.]+(\})"),
                      ("tab_predictions.tex", r"(forward KL transmits & \$\\geq\$ 10 & )[0-9.]+( \\\\)")):
        f = TAB / name
        t, k = re.subn(pat, lambda m: m.group(1) + q + m.group(2), f.read_text())
        assert k == 1, name
        f.write_text(t if hdr in t else hdr + "\n" + t)
    (TAB / "tab_main.tex").write_text("\n".join(mt) + "\n")
    (TAB / "tab_contrast.tex").write_text("\n".join(ct) + "\n")

if __name__ == "__main__":
    main()
