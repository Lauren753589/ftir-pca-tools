"""
Section 3.3: ATR-FTIR PC2 score (mean ± SD of the 10 spectra per material) against the
Py-GC/MS PE content, with a least-squares line and the Pearson correlation.

Negative PC2 = more PE (Material 1 is most negative, the Benchmark most positive).

Run from this folder with:
    py make_pc2_vs_pe.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "PC2 vs PyGCMS PE.png"



def material_key(sample_name):
    """Leading letters of a sample name, upper-cased, e.g. 'Sample_3 1b' -> 'SAMPLE'."""
    return re.match(r"[A-Za-z]+", sample_name.strip()).group(0).upper()


# Original material names -> anonymised names, kept out of the code in material_names.json
# (the file the PCA app saves, e.g. {"Sample A": "Material 1"})
NAMES = {material_key(k): v for k, v in
         json.loads((HERE / "material_names.json").read_text(encoding="utf-8")).items()}
# Py-GC/MS PE content (%), read from the Py-GC/MS results chart
PE = {"Benchmark": 7.55, "Material 1": 33.74, "Material 3": 21.85, "Material 4": 22.19, "Material 5": 17.47}
COLOURS = {"Benchmark": "#d62728", "Material 1": "#ff9f1c", "Material 3": "#2e86de",
           "Material 4": "#2ca02c", "Material 5": "#9b59b6"}

d = pd.read_csv(HERE / "All spectra scores.csv", index_col=0)
d["Material"] = [NAMES[material_key(i)] for i in d.index]
g = d.groupby("Material")["PC 2"].agg(["mean", "std"]) * 1e3   # scores x 10^3 for readability
g["PE"] = g.index.map(PE)

r, p = stats.pearsonr(g["PE"], g["mean"])
slope, intercept = np.polyfit(g["PE"], g["mean"], 1)

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, ax = plt.subplots(figsize=(4.6, 3.6))
x = np.linspace(4, 37, 50)
ax.plot(x, slope * x + intercept, color="0.6", lw=1, ls="--", zorder=1)
ax.axhline(0, color="0.85", lw=0.8, zorder=0)
for mat, row in g.iterrows():
    ax.errorbar(row["PE"], row["mean"], yerr=row["std"], fmt="o", ms=7, capsize=3,
                color=COLOURS[mat], mec="black", mew=0.6, zorder=2)
    dx, ha = (0.8, "left") if mat not in ("Material 1", "Material 5") else (-0.8, "right")
    dy = 0.12 if mat != "Material 3" else -0.12
    ax.text(row["PE"] + dx, row["mean"] + dy, mat, fontsize=8.5, ha=ha, va="center")

ax.set_xlabel("PE content by Py-GC/MS (%)")
ax.set_ylabel("Mean PC2 score (×10$^{-3}$)")
ax.set_xlim(4, 37)
ax.text(0.97, 0.95, f"r = {r:.2f}  (r² = {r * r:.2f}, n = 5)".replace("-", "−"),
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=600)
print(g.round(3))
print(f"r = {r:.3f}, p = {p:.3f}; saved {OUT}")
