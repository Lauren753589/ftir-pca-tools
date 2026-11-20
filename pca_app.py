"""
FTIR PCA Figure Builder
-----------------------
Run from a terminal with:
    py -m streamlit run pca_app.py

Builds figures from a PCA that has already been run elsewhere (e.g. Unscrambler).
Expected CSV layouts:
    Scores     : one row per sample, first column = sample name, then PC 1, PC 2, ...
    Loadings   : one row per PC (PC 1, PC 2, ...), one column per wavenumber
    Variance   : one row, columns PC 0, PC 1, ... (cumulative %, PC 0 = 0) or individual %
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.stats import chi2, f

HERE = Path(__file__).parent
DEFAULT_SCORES = HERE / "All spectra scores.csv"
DEFAULT_LOADINGS = HERE / "all spectra loadings.csv"
DEFAULT_VARIANCE = HERE / "explained variance.csv"
NAMES_FILE = HERE / "material_names.json"
DESKTOP = next((p for p in (Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop") if p.is_dir()), HERE)

st.set_page_config(page_title="PCA Figure Builder", layout="wide")
st.title("FTIR PCA Figure Builder")


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================
def pc_name(label):
    """Normalise 'PC 1', 'PC1', 'pc_1' etc. to 'PC1'."""
    m = re.search(r"(\d+)", str(label))
    return f"PC{m.group(1)}" if m else str(label)


def parse_sample_name(name):
    """Split e.g. 'Benchmark 2b' into product 'Benchmark', specimen '2', replicate 'b'."""
    parts = str(name).rsplit(" ", 1)
    if len(parts) == 2:
        m = re.fullmatch(r"(\d+)(\D*)", parts[1])
        if m:
            return parts[0], m.group(1), m.group(2) or "a"
    return str(name), "", "a"


def load_scores(source):
    df = pd.read_csv(source, index_col=0)
    df.columns = [pc_name(c) for c in df.columns]
    df.index = df.index.astype(str).str.strip()
    df.index.name = "Sample"
    return df


def load_loadings(source):
    """Return (wavenumbers, DataFrame with one column per PC)."""
    df = pd.read_csv(source, index_col=0)
    df.index = [pc_name(i) for i in df.index]
    wn = np.array(df.columns, dtype=float)
    return wn, df.T.set_axis(wn)


def load_variance(source, treat_as_cumulative):
    """Return a Series of individual explained variance (%) indexed by PC name."""
    df = pd.read_csv(source, index_col=0)
    s = df.iloc[0].astype(float)
    s.index = [pc_name(c) for c in s.index]
    if treat_as_cumulative:
        s = s.diff().dropna() if s.index[0] == "PC0" else s.diff().fillna(s.iloc[0])
    return s.drop("PC0", errors="ignore")


def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


@st.cache_data
def load_spectra_file(name, data):
    """Read one spectra CSV into a DataFrame with one row per sample and one column per wavenumber.

    Handles both layouts in this folder:
      * combined files: one row per sample, wavenumbers across the top
      * instrument exports: wavenumber / absorbance columns, optionally under a title line
    """
    lines = data.decode("utf-8-sig", errors="replace").splitlines()
    first_data = next((i for i, l in enumerate(lines[:50]) if is_number(l.split(",")[0].strip())), None)

    if first_data is None:  # wide: samples as rows
        df = pd.read_csv(io.StringIO("\n".join(lines)), index_col=0)
        df = df[[c for c in df.columns if is_number(c)]]
    else:  # long: wavenumbers as rows, one column per sample
        header = first_data - 1 if first_data > 0 else None
        df = pd.read_csv(io.StringIO("\n".join(lines[header if header is not None else 0:])),
                         header=0 if header is not None else None, index_col=0)
        if df.shape[1] == 1 or header is None:
            df.columns = [Path(name).stem] if df.shape[1] == 1 else [f"{Path(name).stem} {c}" for c in df.columns]
        df = df.T
    df.columns = df.columns.astype(float)
    df.index = df.index.astype(str).str.strip()
    return df.apply(pd.to_numeric, errors="coerce")


def rename_sample(name, rename):
    """Swap the material part of a sample name for its new name, e.g. 'Sample A 1b' -> 'Material 1 1b'."""
    product = parse_sample_name(name)[0]
    return rename.get(product, product) + name[len(product):]


def build_meta(names, extra=None, rename=None):
    """Grouping columns parsed from sample names, plus any uploaded metadata.

    `names` are the original sample names (so an uploaded metadata file still matches);
    the returned table uses the renamed materials.
    """
    rename = rename or {}
    m = pd.DataFrame([parse_sample_name(s) for s in names],
                     columns=["Product", "Specimen", "Replicate"], index=pd.Index(names, name="Sample"))
    if extra is not None:
        m = m.join(extra, how="left", rsuffix=" (file)")
    m["Product"] = m["Product"].map(lambda p: rename.get(p, p))
    m.insert(3, "Product + specimen", m["Product"] + " " + m["Specimen"])
    m.index = pd.Index([rename_sample(s, rename) for s in names], name="Sample")
    return m


def material_name_inputs(products, box, rename):
    """Add a text box per material (if it doesn't have one yet); saves any changes to NAMES_FILE."""
    saved = load_saved_names()
    for p in products:
        if p not in rename:
            rename[p] = box.text_input(p, saved.get(p, p), key=f"rename_{p}").strip() or p
    changed = {p: n for p, n in rename.items() if n != p}
    if changed != {p: n for p, n in saved.items() if n != p}:
        NAMES_FILE.write_text(json.dumps(changed, indent=2), encoding="utf-8")


def load_saved_names():
    try:
        return json.loads(NAMES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def confidence_ellipse(x, y, conf=0.95, n_points=100):
    """Return x, y coordinates of a confidence ellipse around a group of points."""
    if len(x) < 3:
        return None, None
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    radii = np.sqrt(chi2.ppf(conf, df=2) * vals)
    t = np.linspace(0, 2 * np.pi, n_points)
    ellipse = vecs @ np.stack([radii[0] * np.cos(t), radii[1] * np.sin(t)])
    return ellipse[0] + x.mean(), ellipse[1] + y.mean()


def hotelling_ellipse(x, y, conf=0.95, n_points=100):
    """Hotelling's T² ellipse for all samples, centred at the origin (2 components)."""
    n = len(x)
    t2_lim = 2 * (n - 1) / (n - 2) * f.ppf(conf, 2, n - 2)
    t = np.linspace(0, 2 * np.pi, n_points)
    return np.sqrt(t2_lim * np.var(x, ddof=1)) * np.cos(t), np.sqrt(t2_lim * np.var(y, ddof=1)) * np.sin(t)


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# 2. LOAD DATA
# =============================================================================
st.sidebar.header("1. Data")
with st.sidebar.expander("Upload files (otherwise the defaults in this folder are used)"):
    up_scores = st.file_uploader("Scores CSV", type="csv")
    up_loadings = st.file_uploader("Loadings CSV", type="csv")
    up_variance = st.file_uploader("Explained variance CSV", type="csv")
    up_meta = st.file_uploader(
        "Optional metadata CSV (first column = sample name, other columns = groupings)", type="csv")
    cumulative = st.checkbox("Explained variance values are cumulative", value=True)

missing = [path.name for path, upload in ((DEFAULT_SCORES, up_scores), (DEFAULT_LOADINGS, up_loadings),
                                          (DEFAULT_VARIANCE, up_variance)) if upload is None and not path.exists()]
if missing:
    st.info("Upload the scores, loadings and explained variance CSVs in the sidebar, "
            f"or put these files next to this script: {', '.join(missing)}")
    st.stop()

scores_df = load_scores(up_scores if up_scores is not None else DEFAULT_SCORES)
wn, loadings_df = load_loadings(up_loadings if up_loadings is not None else DEFAULT_LOADINGS)
explained = load_variance(up_variance if up_variance is not None else DEFAULT_VARIANCE, cumulative)

pc_labels = list(scores_df.columns)
explained = explained.reindex(pc_labels).fillna(0)

# Metadata: parsed from the sample names, plus anything in an uploaded metadata file
extra_meta = None
if up_meta is not None:
    extra_meta = pd.read_csv(up_meta, index_col=0).astype(str)
    extra_meta.index = extra_meta.index.astype(str).str.strip()

# Material names: one text box per material, remembered in material_names.json
rename_box = st.sidebar.expander("Rename materials")
rename_box.caption("Type a new name for any material. Names are saved in material_names.json in this folder.")
rename = {}
material_name_inputs(dict.fromkeys(parse_sample_name(s)[0] for s in scores_df.index), rename_box, rename)
if len(set(rename.values())) < len(rename):
    rename_box.warning("Two materials share the same name, so they will be shown as one group.")



def original_group(g):
    """Undo the material rename on a group label, e.g. 'Material 1 1' -> 'Sample A 1'."""
    for old, new in rename.items():
        if g == new or g.startswith(new + " "):
            return old + g[len(new):]
    return g


meta = build_meta(scores_df.index, extra_meta, rename)
scores_df.index = meta.index
meta_cols = list(meta.columns)

group_col = st.sidebar.selectbox("Colour samples by", meta_cols, index=0)
symbol_col = st.sidebar.selectbox("Marker shape by", ["(none)"] + meta_cols, index=0)

groups = meta[group_col].astype(str)
group_names = list(dict.fromkeys(groups))
shown_groups = st.sidebar.multiselect("Groups to show", group_names, default=group_names)

st.sidebar.caption(
    f"{len(scores_df)} samples · {len(pc_labels)} PCs · {len(wn)} wavenumbers "
    f"({wn.min():.0f}–{wn.max():.0f} cm⁻¹)"
)


# =============================================================================
# 3. SIDEBAR: PLOT STYLE
# =============================================================================
st.sidebar.header("2. Plot style")
with st.sidebar.expander("Group colours", expanded=True):
    palette = px.colors.qualitative.D3  # hex codes, which st.color_picker requires
    colours = {
        # keyed by the original material name so a chosen colour survives renaming
        g: st.color_picker(g, palette[i % len(palette)], key=f"colour_{group_col}_{original_group(g)}")
        for i, g in enumerate(group_names)
    }

with st.sidebar.expander("Lines, markers & text"):
    line_width = st.slider("Loading line width", 0.5, 6.0, 1.5, 0.5)
    marker_size = st.slider("Score marker size", 4, 30, 12)
    base_symbol = st.selectbox("Marker shape", ["circle", "square", "diamond", "triangle-up", "cross", "x"])
    marker_outline = st.slider("Marker outline width", 0.0, 3.0, 1.0, 0.5)
    marker_opacity = st.slider("Marker opacity", 0.2, 1.0, 0.9, 0.05)
    font_size = st.slider("Font size", 8, 28, 14)
    label_size = st.slider("Point label font size", 6, 24, 10)

with st.sidebar.expander("Axes & layout"):
    template = st.selectbox("Theme", ["simple_white", "plotly_white", "plotly", "ggplot2", "seaborn", "plotly_dark"])
    show_box = st.checkbox("Draw box around plot", value=True)
    box_width = st.slider("Box line width", 0.5, 4.0, 1.5, 0.5)
    show_grid = st.checkbox("Show grid lines", value=False)
    show_zero = st.checkbox("Show zero lines", value=True)
    show_legend = st.checkbox("Show legend", value=True)
    legend_pos = st.selectbox("Legend position", ["Right", "Top", "Bottom", "Inside top-right"])
    plot_height = st.slider("Plot height (px)", 300, 1200, 600, 50)
    plot_width = st.slider("Plot width (px, 0 = fill page)", 0, 1600, 0, 50)
    title_size = st.slider("Plot title font size", 10, 40, 18)
    title_align = st.selectbox("Plot title position", ["Centre", "Left"])
    export_format = st.selectbox("Camera-icon download format", ["png", "svg"])

with st.sidebar.expander("Band markers"):
    st.caption("Type bands as `1730` or with a name, e.g. `1730 C=O, 1640 amide I`.")
    band_dash = st.selectbox("Band line style", ["solid", "dash", "dot", "dashdot"])
    band_width = st.slider("Band line width", 0.5, 5.0, 1.5, 0.5)
    band_shade = st.slider("Shade ± cm⁻¹ around each band (0 = off)", 0, 50, 0)
    band_label_size = st.slider("Band label font size (loadings)", 6, 48, 11)
    band_label_box = st.checkbox("Draw a box behind band labels", value=True)

plot_config = {"toImageButtonOptions": {"format": export_format, "scale": 3}}
symbols = ["circle", "square", "diamond", "triangle-up", "cross", "x", "triangle-down", "star", "pentagon", "hexagon"]

LEGENDS = {
    "Right": dict(),
    "Top": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    "Bottom": dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    "Inside top-right": dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.7)"),
}


def style_fig(fig, x_title, y_title, reverse_x=False):
    """Apply all the sidebar style choices to a figure."""
    fig.update_layout(
        template=template,
        height=plot_height,
        width=plot_width or None,
        font=dict(size=font_size),
        showlegend=show_legend,
        legend=dict(title_text="", **LEGENDS[legend_pos]),
        margin=dict(t=30),  # no figure title, so no need for a tall top margin
    )
    axis_style = dict(
        showgrid=show_grid,
        showline=show_box,
        mirror=show_box,  # mirror=True draws the top and right edges to make a box
        linecolor="black",
        linewidth=box_width,
        ticks="outside",
        zeroline=False,
        exponentformat="e",
    )
    fig.update_xaxes(title=x_title, autorange="reversed" if reverse_x else True, **axis_style)
    fig.update_yaxes(title=y_title, **axis_style)
    return fig


BAND_COLOUR = "#000000"  # bands are always black so colour can't imply a link to a group or PC


def mark_bands(fig, text, x_range, n_panels=1, label_size=None):
    """Draw a black vertical line + label for each band in `text` (e.g. '1730 C=O, 1640').

    Everything stays inside the plot box: labels sit just under the top edge, and
    labels that would overlap drop onto extra rows so every one stays readable.
    """
    bands = []
    for entry in text.split(","):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(.*)", entry)
        if m:
            label = f"{float(m.group(1)):g}" + (f" {m.group(2).strip()}" if m.group(2).strip() else "")
            bands.append((float(m.group(1)), label))
    if not bands:
        return fig

    # Rough label width in wavenumber units, to work out which labels would collide
    span = abs(x_range[1] - x_range[0]) or 1
    plot_px = (plot_width or 1100) * 0.8
    size = label_size or band_label_size
    row_px = size + 12
    row_ends = []  # right-hand edge (cm⁻¹) of the last label on each row
    placed = []
    for x, label in sorted(bands, key=lambda b: -b[0]):  # left-to-right on a reversed axis
        half = (len(label) * size * 0.62 + 12) / 2 * span / plot_px
        row = next((r for r, end in enumerate(row_ends) if x + half < end), len(row_ends))
        if row == len(row_ends):
            row_ends.append(0)
        row_ends[row] = x - half
        placed.append((x, label, row))

    colour = BAND_COLOUR
    for x, label, row in placed:
        # Lines and shading are drawn inside each panel's own box ("y domain" = that panel's 0–1 height)
        for p in range(1, n_panels + 1):
            xref, yref = ("x", "y domain") if p == 1 else (f"x{p}", f"y{p} domain")
            if band_shade:
                fig.add_shape(type="rect", xref=xref, yref=yref, x0=x - band_shade, x1=x + band_shade,
                              y0=0, y1=1, fillcolor=hex_to_rgba(colour, 0.12), line_width=0, layer="below")
            fig.add_shape(type="line", xref=xref, yref=yref, x0=x, x1=x, y0=0, y1=1,
                          line=dict(color=colour, width=band_width, dash=band_dash))
        # Labels hang down from the top edge of the (top) panel, one row per collision level
        fig.add_annotation(
            x=x, xref="x", y=1, yref="y domain", yanchor="top", yshift=-(4 + row * row_px),
            text=label, showarrow=False, font=dict(size=size, color=colour),
            bgcolor="rgba(255,255,255,0.9)" if band_label_box else None,
            bordercolor=colour if band_label_box else None, borderwidth=1 if band_label_box else 0, borderpad=max(2, size // 5),
        )

    # Raise the top of the (top) panel's y-axis so the label rows sit above the data, not on it
    ys = [np.asarray(t.y, dtype=float) for t in fig.data
          if t.y is not None and getattr(t, "yaxis", None) in (None, "y")]
    ys = np.concatenate(ys) if ys else np.array([])
    ys = ys[np.isfinite(ys)]
    if ys.size:
        lo, hi = ys.min(), ys.max()
        pad = 0.04 * (hi - lo or 1)
        panel_px = plot_height * 0.75 / n_panels
        frac = min(0.6, (len(row_ends) * row_px + 8) / panel_px)  # share of the panel taken by labels
        bottom, top = lo - pad, hi + pad
        fig.update_layout(yaxis_range=[bottom, bottom + (top - bottom) / (1 - frac)])
    return fig


def unique_path(path):
    """Add ' (2)', ' (3)', … to a file name so an existing file is never overwritten."""
    n, out = 2, path
    while out.exists():
        out = path.with_name(f"{path.stem} ({n}){path.suffix}")
        n += 1
    return out


def add_title(fig, text):
    """Put `text` above the plot, with room for it (and for a legend along the top)."""
    if not text.strip():
        return fig
    legend_on_top = show_legend and legend_pos == "Top"
    fig.update_layout(
        title=dict(text=text, x=0.5 if title_align == "Centre" else 0, xref="paper",
                   xanchor="center" if title_align == "Centre" else "left",
                   y=1, yref="container", yanchor="top", pad=dict(t=10), font=dict(size=title_size)),
        margin=dict(t=int(title_size * 1.5) + 25 + (font_size + 15 if legend_on_top else 0)),
    )
    return fig


def show(fig):
    st.plotly_chart(fig, config=plot_config, width="content" if plot_width else "stretch")


def range_picker(lo, hi, key):
    """Wavenumber range slider with 'From' / 'To' number boxes, kept in step with each other.

    Uses session-state keys '<key>_range', '<key>_min' and '<key>_max'.
    """
    ss = st.session_state
    k_range, k_min, k_max = f"{key}_range", f"{key}_min", f"{key}_max"
    if not (k_range in ss and lo <= ss[k_range][0] <= ss[k_range][1] <= hi):  # first run, or new data
        ss[k_range] = (lo, hi)
        ss[k_min], ss[k_max] = lo, hi

    def from_boxes():
        a, b = sorted((ss[k_min], ss[k_max]))
        ss[k_min], ss[k_max] = a, b
        ss[k_range] = (a, b)

    def from_slider():
        ss[k_min], ss[k_max] = ss[k_range]

    c_lo, c_slider, c_hi = st.columns([1, 4, 1])
    c_lo.number_input("From (cm⁻¹)", lo, hi, step=1.0, format="%.1f", key=k_min, on_change=from_boxes)
    c_hi.number_input("To (cm⁻¹)", lo, hi, step=1.0, format="%.1f", key=k_max, on_change=from_boxes)
    return c_slider.slider("Wavenumber range (cm⁻¹)", lo, hi, key=k_range, on_change=from_slider)


def save_png_button(fig, default_name, key, disabled=False):
    """File-name box + button that writes `fig` to the Desktop as a high-resolution PNG."""
    c1, c2 = st.columns([3, 1])
    png_name = c1.text_input("PNG file name", default_name, key=f"png_name_{key}_{default_name}")
    c2.write("")  # line the button up with the text box
    if c2.button("Save PNG to Desktop", type="primary", disabled=disabled, key=f"save_png_{key}"):
        try:
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", png_name).strip() or default_name  # no illegal characters
            out = unique_path(DESKTOP / f"{safe_name}.png")
            with st.spinner("Saving…"):
                fig.write_image(out, width=plot_width or 1200, height=plot_height, scale=3)
            st.success(f"Saved to {out}")
        except Exception as e:  # e.g. the file is open in another program
            st.error(f"Could not save the PNG: {e}")


# =============================================================================
# 4. TABS WITH THE PLOTS
# =============================================================================
tab_spec, tab_scores, tab_load, tab_scree, tab_table = st.tabs(
    ["Spectra", "PCA scores", "PCA loadings", "Explained variance", "Data table"])

# ---- Spectra ----------------------------------------------------------------
with tab_spec:
    spec_files = st.file_uploader(
        "Drop spectra CSV file(s) here — a combined file (one row per sample) "
        "or one or more single-spectrum exports", type="csv", accept_multiple_files=True)

    if not spec_files:
        st.info("Drop a CSV above to view spectra, e.g. a combined file with one row per sample, "
                "or several individual files like *Material 1 3b.csv*.")
    else:
        spectra = pd.concat([load_spectra_file(f.name, f.getvalue()) for f in spec_files])
        spectra = spectra[~spectra.index.duplicated()].sort_index(axis=1)
        spec_wn = spectra.columns.to_numpy(dtype=float)
        # materials only in the spectra file(s) get their own rename boxes in the sidebar
        material_name_inputs(dict.fromkeys(parse_sample_name(s)[0] for s in spectra.index), rename_box, rename)
        spec_meta = build_meta(spectra.index, extra_meta, rename)
        spectra.index = spec_meta.index
        spectra = spectra[~spectra.index.duplicated()]
        spec_meta = spec_meta[~spec_meta.index.duplicated()]
        spec_groups = spec_meta[group_col].astype(str) if group_col in spec_meta else spec_meta["Product"]
        spec_group_names = list(dict.fromkeys(spec_groups))
        spec_colours = {g: colours.get(g, palette[(len(colours) + i) % len(palette)])
                        for i, g in enumerate(spec_group_names)}
        st.caption(f"{len(spectra)} spectra · {len(spec_wn)} points "
                   f"({spec_wn.min():.0f}–{spec_wn.max():.0f} cm⁻¹) · coloured by **{group_col}**")

        c1, c2, c3 = st.columns(3)
        spec_view = c1.radio("Show", ["Individual spectra", "Group means", "Group mean ± SD"], horizontal=True)
        norm = c2.selectbox("Normalise (display only)", ["None", "Min–max (0–1)", "Vector (unit length)", "SNV"])
        offset_mode = c3.radio("Offset", ["None", "Per spectrum", "Per group"], horizontal=True)

        c1, c2 = st.columns([2, 1])
        spec_groups_shown = c1.multiselect("Groups", spec_group_names, default=spec_group_names, key="spec_groups")
        offset_size = c2.slider("Offset amount (× max intensity)", 0.0, 2.0, 0.5, 0.05,
                                disabled=(offset_mode == "None"))
        spec_samples = st.multiselect(
            "Samples (leave empty for all in the selected groups)",
            [s for s in spectra.index if spec_groups[s] in spec_groups_shown], key="spec_samples")

        spec_range = range_picker(float(spec_wn.min()), float(spec_wn.max()), key="spec")
        c1, c2, c3 = st.columns([2, 1, 2])
        spec_bands = c1.text_input("Mark bands (comma-separated cm⁻¹)", "", key="spec_bands",
                                   placeholder="e.g. 1730 C=O, 1640, 1540")
        spec_band_size = c2.slider("Band label size", 6, 48, 14, key="spec_band_size")
        spec_title = st.text_input("Plot title", key="spec_title", placeholder="Leave blank for no title")
        y_label = c3.text_input("Y-axis title", "Absorbance" if norm == "None" else "Normalised absorbance")

        sel = (spec_wn >= spec_range[0]) & (spec_wn <= spec_range[1])
        sw = spec_wn[sel]
        Y = spectra.loc[:, sel].to_numpy(dtype=float)
        if norm == "Min–max (0–1)":
            lo, hi = np.nanmin(Y, axis=1, keepdims=True), np.nanmax(Y, axis=1, keepdims=True)
            Y = (Y - lo) / np.where(hi > lo, hi - lo, 1)
        elif norm == "Vector (unit length)":
            Y = Y / np.sqrt(np.nansum(Y ** 2, axis=1, keepdims=True))
        elif norm == "SNV":
            Y = (Y - np.nanmean(Y, axis=1, keepdims=True)) / np.nanstd(Y, axis=1, keepdims=True)
        Y = pd.DataFrame(Y, index=spectra.index)

        chosen = [s for s in (spec_samples or spectra.index) if spec_groups[s] in spec_groups_shown]
        step_size = offset_size * np.nanmax(np.abs(Y.loc[chosen].to_numpy())) if chosen else 0
        fig = go.Figure()

        if spec_view == "Individual spectra":
            group_order = [g for g in spec_groups_shown if any(spec_groups[s] == g for s in chosen)]
            for n, s in enumerate(chosen):
                g = spec_groups[s]
                k = {"None": 0, "Per spectrum": n, "Per group": group_order.index(g)}[offset_mode]
                first_in_group = chosen.index(next(c for c in chosen if spec_groups[c] == g)) == n
                fig.add_trace(go.Scatter(
                    x=sw, y=Y.loc[s].to_numpy() - k * step_size, mode="lines", name=g, legendgroup=g,
                    showlegend=first_in_group, line=dict(color=spec_colours[g], width=line_width),
                    customdata=Y.loc[s].to_numpy(),
                    hovertemplate=f"<b>{s}</b><br>%{{x:.0f}} cm⁻¹<br>%{{customdata:.4g}}<extra></extra>",
                ))
        else:
            for k, g in enumerate([g for g in spec_groups_shown if any(spec_groups[s] == g for s in chosen)]):
                members = [s for s in chosen if spec_groups[s] == g]
                mean, sd = Y.loc[members].mean().to_numpy(), Y.loc[members].std().fillna(0).to_numpy()
                off = 0 if offset_mode == "None" else k * step_size
                if spec_view == "Group mean ± SD":
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([sw, sw[::-1]]), y=np.concatenate([mean + sd, (mean - sd)[::-1]]) - off,
                        fill="toself", fillcolor=hex_to_rgba(spec_colours[g], 0.2), line=dict(width=0),
                        legendgroup=g, showlegend=False, hoverinfo="skip",
                    ))
                fig.add_trace(go.Scatter(
                    x=sw, y=mean - off, mode="lines", name=f"{g} (n={len(members)})", legendgroup=g,
                    line=dict(color=spec_colours[g], width=line_width), customdata=np.stack([mean, sd], axis=1),
                    hovertemplate=f"<b>{g}</b><br>%{{x:.0f}} cm⁻¹<br>mean %{{customdata[0]:.4g}}"
                                  f" ± %{{customdata[1]:.2g}}<extra></extra>",
                ))

        style_fig(fig, "Wavenumber (cm⁻¹)", y_label, reverse_x=True)
        mark_bands(fig, spec_bands, spec_range, label_size=spec_band_size)
        add_title(fig, spec_title)
        if offset_mode != "None":
            fig.update_yaxes(showticklabels=False)
        show(fig)
        st.caption("Hover over a line to see the sample. Click a legend entry to hide a group; double-click to isolate it.")
        save_png_button(fig, f"Spectra - {spec_view} {spec_range[0]:.0f}-{spec_range[1]:.0f}", key="spec")

keep = groups.isin(shown_groups).to_numpy()
hover_meta = meta.reset_index().to_numpy()  # Sample + every metadata column, for hover text
hover_lines = "<br>".join(f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(meta_cols, start=1))

# ---- PCA scores -------------------------------------------------------------
with tab_scores:
    c1, c2, c3, c4 = st.columns(4)
    dims = c1.radio("Plot type", ["2D", "3D"], horizontal=True)
    pc_x = c2.selectbox("X axis", pc_labels, index=0)
    pc_y = c3.selectbox("Y axis", pc_labels, index=1 if len(pc_labels) > 1 else 0)
    pc_z = c4.selectbox("Z axis", pc_labels, index=2 if len(pc_labels) > 2 else 0, disabled=(dims == "2D"))

    c1, c2, c3, c4 = st.columns(4)
    show_labels = c1.checkbox("Label all points", value=False)
    show_ellipse = c2.checkbox("95% group ellipses", value=True, disabled=(dims == "3D"))
    show_hotelling = c3.checkbox("Hotelling T² 95% ellipse", value=False, disabled=(dims == "3D"))
    connect_pairs = c4.checkbox("Connect replicate pairs", value=False,
                                help="Joins samples sharing Product + specimen, e.g. Material 1 1 ↔ Material 1 1b")
    highlight = st.multiselect("Highlight & label specific samples", list(scores_df.index))
    scores_title = st.text_input("Plot title", key="scores_title", placeholder="Leave blank for no title")

    sym_values = meta[symbol_col].astype(str) if symbol_col != "(none)" else None
    sym_map = {v: symbols[i % len(symbols)] for i, v in enumerate(dict.fromkeys(sym_values))} if sym_values is not None else {}

    fig = go.Figure()
    is_3d = dims == "3D"
    ix, iy, iz = pc_labels.index(pc_x), pc_labels.index(pc_y), pc_labels.index(pc_z)
    S = scores_df.to_numpy()
    hover = ("<b>%{customdata[0]}</b><br>" + hover_lines + f"<br>{pc_x}: %{{x:.3e}}<br>{pc_y}: %{{y:.3e}}"
             + (f"<br>{pc_z}: %{{z:.3e}}" if is_3d else "") + "<extra></extra>")

    if show_hotelling and not is_3d:
        hx, hy = hotelling_ellipse(S[keep, ix], S[keep, iy])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", name="Hotelling T² 95%",
                                 line=dict(color="black", width=1, dash="dash"), hoverinfo="skip"))

    for g in shown_groups:
        mask = (groups == g).to_numpy()
        sx, sy, sz = S[mask, ix], S[mask, iy], S[mask, iz]
        names = scores_df.index[mask]

        if show_ellipse and not is_3d:
            ex, ey = confidence_ellipse(sx, sy)
            if ex is not None:
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines", fill="toself",
                    fillcolor=hex_to_rgba(colours[g], 0.15),
                    line=dict(color=colours[g], width=1, dash="dot"),
                    legendgroup=g, showlegend=False, hoverinfo="skip",
                ))

        if connect_pairs:
            pair_keys = meta.loc[mask, "Product + specimen"]
            first = True
            for key in dict.fromkeys(pair_keys):
                idx = np.where((pair_keys == key).to_numpy())[0]
                if len(idx) < 2:
                    continue
                coords = dict(x=sx[idx], y=sy[idx]) | (dict(z=sz[idx]) if is_3d else {})
                trace_cls = go.Scatter3d if is_3d else go.Scatter
                fig.add_trace(trace_cls(**coords, mode="lines", line=dict(color=colours[g], width=1.5),
                                        legendgroup=g, showlegend=False, hoverinfo="skip"))

        label_text = [n if (show_labels or n in highlight) else "" for n in names]
        sym = [sym_map[v] for v in sym_values[mask]] if sym_values is not None else base_symbol
        size = [marker_size * (1.6 if n in highlight else 1) for n in names]
        outline = [marker_outline * (2 if n in highlight else 1) for n in names]
        common = dict(
            name=g, legendgroup=g, customdata=hover_meta[mask], hovertemplate=hover,
            text=label_text, mode="markers+text", textposition="top center",
            textfont=dict(size=label_size),
        )
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz, **common,
                marker=dict(color=colours[g], size=marker_size / 2, symbol=sym if isinstance(sym, str) else
                            [s if s in ("circle", "square", "diamond", "cross", "x") else "circle" for s in sym],
                            opacity=marker_opacity, line=dict(color="black", width=marker_outline)),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sx, y=sy, **common,
                marker=dict(color=colours[g], size=size, symbol=sym, opacity=marker_opacity,
                            line=dict(color="black", width=outline)),
            ))

    # Extra legend entries explaining marker shapes
    if sym_values is not None and not is_3d:
        for v, s in sym_map.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{symbol_col}: {v}",
                                     marker=dict(color="grey", size=marker_size, symbol=s,
                                                 line=dict(color="black", width=marker_outline))))

    x_title = f"{pc_x} ({explained[pc_x]:.1f}%)"
    y_title = f"{pc_y} ({explained[pc_y]:.1f}%)"
    if is_3d:
        fig.update_layout(
            template=template, height=plot_height, width=plot_width or None, font=dict(size=font_size),
            showlegend=show_legend, legend=dict(title_text="", **LEGENDS[legend_pos]),
            scene=dict(xaxis_title=x_title, yaxis_title=y_title,
                       zaxis_title=f"{pc_z} ({explained[pc_z]:.1f}%)",
                       xaxis_exponentformat="e", yaxis_exponentformat="e", zaxis_exponentformat="e"),
        )
    else:
        if show_zero:
            fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"))
            fig.add_vline(x=0, line=dict(color="grey", width=1, dash="dash"))
        style_fig(fig, x_title, y_title)
    add_title(fig, scores_title)
    show(fig)
    st.caption("Hover over a point to see the sample. Click a legend entry to hide a group; double-click to isolate it.")
    save_png_button(fig, f"PCA scores {pc_x} vs {pc_y}" + (f" vs {pc_z}" if is_3d else ""), key="scores")

# ---- PCA loadings -----------------------------------------------------------
with tab_load:
    c1, c2 = st.columns([2, 1])
    chosen_pcs = c1.multiselect("Components to plot", pc_labels, default=pc_labels[:1])
    layout_mode = c2.radio("Layout", ["Overlay", "Offset", "Stacked panels"], horizontal=True)

    # One colour picker per chosen PC; defaults are keyed to the PC so they don't shift when others are added
    load_palette = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
                    "#E6AB02", "#A6761D", "#666666", "#1F77B4", "#D62728"]
    pc_colours = {}
    if chosen_pcs:
        pick_cols = st.columns(min(len(chosen_pcs), 6))
        for k, pc in enumerate(chosen_pcs):
            default = load_palette[pc_labels.index(pc) % len(load_palette)]
            pc_colours[pc] = pick_cols[k % len(pick_cols)].color_picker(f"{pc} colour", default, key=f"pc_colour_{pc}")

    load_range = range_picker(float(wn.min()), float(wn.max()), key="load")
    load_title = st.text_input("Plot title", key="load_title", placeholder="Leave blank for no title")
    c1, c2, c3 = st.columns(3)
    n_peaks = c1.number_input("Label top N peaks per PC (by |loading|)", 0, 30, 0)
    peak_spacing = c2.number_input("Min. spacing between labelled peaks (cm⁻¹)", 5, 500, 30)
    band_text = c3.text_input("Mark bands (comma-separated cm⁻¹)", "",
                              placeholder="e.g. 1730 C=O, 1640, 1540")

    in_range = (wn >= load_range[0]) & (wn <= load_range[1])
    w = wn[in_range]
    step = abs(np.median(np.diff(wn))) or 1
    n_rows = len(chosen_pcs) if layout_mode == "Stacked panels" and chosen_pcs else 1

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    offset_step = (np.abs(loadings_df.loc[in_range, chosen_pcs].to_numpy()).max() * 1.5
                   if chosen_pcs and layout_mode == "Offset" else 0)

    for k, pc in enumerate(chosen_pcs):
        y = loadings_df.loc[in_range, pc].to_numpy()
        offset = -k * offset_step
        row = k + 1 if layout_mode == "Stacked panels" else 1
        colour = pc_colours[pc]
        fig.add_trace(go.Scatter(
            x=w, y=y + offset, mode="lines", name=f"{pc} ({explained[pc]:.1f}%)",
            line=dict(color=colour, width=line_width),
            customdata=y, hovertemplate="%{x:.0f} cm⁻¹<br>loading: %{customdata:.3e}<extra>" + pc + "</extra>",
        ), row=row, col=1)
        if layout_mode != "Overlay" or k == 0:
            fig.add_hline(y=offset, line=dict(color="grey", width=1, dash="dash"), row=row, col=1)

        if n_peaks:
            peaks, _ = find_peaks(np.abs(y), distance=max(1, int(peak_spacing / step)))
            top = peaks[np.argsort(np.abs(y[peaks]))[::-1][:n_peaks]]
            fig.add_trace(go.Scatter(
                x=w[top], y=y[top] + offset, mode="markers+text", text=[f"{v:.0f}" for v in w[top]],
                textposition=["top center" if v > 0 else "bottom center" for v in y[top]],
                textfont=dict(size=label_size, color=colour), marker=dict(color=colour, size=5),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=1)

    style_fig(fig, None, None, reverse_x=True)
    mark_bands(fig, band_text, load_range, n_panels=n_rows)
    fig.update_xaxes(title="Wavenumber (cm⁻¹)", row=n_rows, col=1)
    if layout_mode == "Stacked panels":
        for k, pc in enumerate(chosen_pcs):
            fig.update_yaxes(title=pc, row=k + 1, col=1)
    else:
        fig.update_yaxes(title="Loading" + (" (offset)" if layout_mode == "Offset" else ""),
                         showticklabels=layout_mode != "Offset")
    add_title(fig, load_title)
    show(fig)

    save_png_button(fig, f"PCA loadings {' '.join(chosen_pcs)} {load_range[0]:.0f}-{load_range[1]:.0f}",
                    key="load", disabled=not chosen_pcs)

# ---- Explained variance -----------------------------------------------------
with tab_scree:
    n_show = st.slider("Number of PCs to show", 1, len(pc_labels), len(pc_labels))
    shown = pc_labels[:n_show]
    c1, c2 = st.columns(2)
    show_cum = c1.checkbox("Show cumulative line", value=True)
    show_values = c2.checkbox("Show values on bars", value=True)
    scree_title = st.text_input("Plot title", key="scree_title", placeholder="Leave blank for no title")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shown, y=explained[shown], name="Individual", marker_color="#4C72B0",
        text=[f"{v:.1f}" for v in explained[shown]] if show_values else None, textposition="outside",
    ))
    if show_cum:
        fig.add_trace(go.Scatter(x=shown, y=explained[shown].cumsum(), name="Cumulative",
                                 mode="lines+markers", line=dict(color="#DD8452", width=2)))
    style_fig(fig, "Principal component", "Explained variance (%)")
    fig.update_yaxes(range=[0, 105])
    add_title(fig, scree_title)
    show(fig)

# ---- Data table -------------------------------------------------------------
with tab_table:
    table = meta.join(scores_df)
    var_table = pd.DataFrame({"Explained (%)": explained, "Cumulative (%)": explained.cumsum()})
    st.subheader("Scores")
    st.dataframe(table)
    st.download_button("Download scores + groups CSV", table.to_csv(), "pca_scores_with_groups.csv")
    st.subheader("Explained variance")
    st.dataframe(var_table)
