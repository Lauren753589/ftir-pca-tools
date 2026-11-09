"""
Builds the 2 x 2 PC2/PC3 loadings figure for Chapter 3:
  (a) PC2, C-H region      (b) PC2, fingerprint region
  (c) PC3, C-H region      (d) PC3, fingerprint region

Run from a terminal in this folder with:
    py make_loadings_figure.py
Re-run after re-exporting the loadings / variance CSVs and the figure updates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LOADINGS = HERE / "all spectra loadings.csv"
VARIANCE = HERE / "explained variance.csv"
OUT = HERE / "PC2 PC3 loadings - 2x2.png"

PCS = {"PC 2": "#D95F02", "PC 3": "#7570B3"}  # same colours as the PCA app
REGIONS = [(3050, 2800), (1500, 460)]           # high -> low wavenumber (stops before the flat derivative edge)
# Band labels sit just above the trace at each band in every panel (no lines through the data)
BANDS = {
    0: [2950, 2919, 2851],
    1: [1377, 1018, 669, 467],
}

# ---- data -------------------------------------------------------------------
load = pd.read_csv(LOADINGS, index_col=0)
load.index = [i.strip() for i in load.index]
wn = load.columns.astype(float).to_numpy()

cum = pd.read_csv(VARIANCE, index_col=0).iloc[0].astype(float)
explained = cum.diff().dropna()
explained.index = [c.strip() for c in explained.index]

# ---- figure -----------------------------------------------------------------
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex="col",
                         gridspec_kw={"width_ratios": [1, 2.2], "hspace": 0.08, "wspace": 0.18})

letters = iter("abcd")
for r, (pc, colour) in enumerate(PCS.items()):
    y_all = load.loc[pc].to_numpy()
    for c, (hi, lo) in enumerate(REGIONS):
        ax = axes[r, c]
        sel = (wn <= hi) & (wn >= lo)
        y = y_all[sel]
        ax.plot(wn[sel], y, color=colour, lw=1.1)
        ax.axhline(0, color="0.75", lw=0.6, ls="--", zorder=0)
        ax.set_xlim(hi, lo)
        span = y.max() - y.min()
        tops = []
        for pos in BANDS[c]:
            # clear any neighbouring peaks within ~3 % of the panel width, so labels never sit on the trace
            near = np.abs(wn[sel] - pos) <= 0.03 * (hi - lo)
            top = y[near].max() + 0.04 * span
            tops.append(top)
            at_edge = (pos - lo) < 0.03 * (hi - lo)  # keep labels near the right edge inside the frame
            ax.annotate(f"{pos}", xy=(pos, top), xytext=(3 if at_edge else 0, 1), textcoords="offset points",
                        ha="right" if at_edge else "center", va="bottom", fontsize=7.5)
        ax.set_ylim(y.min() - 0.08 * span, max(y.max(), max(tops)) + 0.22 * span)
        ax.text(0.02, 0.95, f"({next(letters)})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.tick_params(direction="out", length=3, labelsize=8)
        if c == 0:
            num = pc.split()[1]
            ax.set_ylabel(f"PC{num} loading\n({explained[pc]:.1f} %)")

for ax in axes[1]:
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")

fig.savefig(OUT, dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
print("Saved", OUT, "and", OUT.with_suffix(".svg").name)
