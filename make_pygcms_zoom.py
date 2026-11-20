"""
Expanded-region Py-GC/MS figure (4.5-10 min) built from the F-Search overlay screenshot.

The screenshot is the only copy of these chromatograms, so each panel is cropped from it and
placed on calibrated axes. The time axis was calibrated against the Material 3 peak table
(5.188, 6.692, 9.479, 11.354 and 15.079 min): x_pixel = -103.6 + 50.05 * t.

Run from this folder with:
    py make_pygcms_zoom.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
SRC = HERE / "Py-GCMS overlay (original screenshot).png"
OUT = HERE / "Py-GCMS expanded region 4.5-10 min.png"

T0, T1 = 4.5, 10.0                       # time window (min)
X_OF_T = lambda t: -103.6 + 50.05 * t    # screenshot pixel column for a retention time
PX_PER_PCT = 1.35                        # vertical pixels per 1 % relative abundance
BG = np.array([230, 243, 255])           # F-Search plot background

# (label, colour, frame top row, frame bottom row) in the screenshot, top to bottom
PANELS = [
    ("Material 3", "#003a7a", 110, 259),
    ("Material 5", "#c8102e", 298, 447),
    ("Material 4", "#3cb400", 476, 625),
    ("Benchmark", "#1f6fff", 654, 803),
    ("Material 1", "#ff4fd8", 832, 981),
]
# marker regions shaded across all panels (wide enough for small retention-time shifts between runs)
MARKERS = [
    (5.0, 5.5, "PP\n2,4-dimethylhept-1-ene"),
    (6.5, 7.1, "Styrene"),
    (9.3, 9.8, "PE\nC$_{10}$ triplet"),
]

img = np.array(Image.open(SRC).convert("RGB")).astype(float)
x0, x1 = int(round(X_OF_T(T0))), int(round(X_OF_T(T1)))

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.3, 7.2), sharex=True,
                         gridspec_kw={"hspace": 0.12})

for ax, (name, colour, top, bottom) in zip(axes, PANELS):
    crop = img[top + 1:bottom, x0:x1].copy()
    # white background instead of the software's pale blue
    is_bg = np.abs(crop - BG).sum(axis=2) < 60
    crop[is_bg] = 255
    ymax = (bottom - top - 1) / PX_PER_PCT
    ax.imshow(crop.astype(np.uint8), extent=[T0, T1, 0, ymax], aspect="auto",
              interpolation="lanczos")
    for lo, hi, _ in MARKERS:
        ax.axvspan(lo, hi, color="0.85", alpha=0.45, lw=0, zorder=0)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=8, direction="out", length=3)
    ax.text(0.99, 0.92, name, transform=ax.transAxes, ha="right", va="top",
            color=colour, fontweight="bold", fontsize=9)

for lo, hi, text in MARKERS:
    axes[0].text((lo + hi) / 2, 1.04, text, transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8)

axes[-1].set_xlim(T0, T1)
axes[-1].set_xlabel("Retention time (min)")
fig.supylabel("Relative abundance (%)", x=0.04, fontsize=9)

fig.savefig(OUT, dpi=600, bbox_inches="tight")
print("Saved", OUT)
