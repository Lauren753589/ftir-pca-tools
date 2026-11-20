"""
FTIR Spectral Explorer
----------------------
Run from a terminal with:
    py -m streamlit run ftir_app.py

Expected CSV layout (MetaboAnalyst style):
    one row per sample, a few text columns (e.g. Sample, Label, Pair),
    then one column per wavenumber (column names are numbers like 450.0).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DEFAULT_FILE = Path(__file__).parent / "ftir_data.csv"  # used if no file is uploaded

st.set_page_config(page_title="FTIR Explorer", layout="wide")
st.title("FTIR Spectral Explorer")


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================
def split_columns(df):
    """Columns whose names are numbers are spectra; everything else is metadata."""
    spec_cols, meta_cols = [], []
    for col in df.columns:
        try:
            float(col)
            spec_cols.append(col)
        except ValueError:
            meta_cols.append(col)
    return spec_cols, meta_cols


def preprocess(X, wn, wn_range, use_snv, deriv, window):
    """Crop to a wavenumber range, then optional SNV and Savitzky-Golay derivative."""
    keep = (wn >= wn_range[0]) & (wn <= wn_range[1])
    X, wn = X[:, keep], wn[keep]
    if use_snv:
        # Standard Normal Variate: centre and scale each spectrum (row) on its own
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    if deriv > 0:
        X = savgol_filter(X, window_length=window, polyorder=2, deriv=deriv, axis=1)
    return X, wn


def scaling_factors(X, method):
    """Return the column means and divisors used for scaling before PCA."""
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1  # avoid dividing by zero for flat variables
    if method == "Autoscale (UV)":
        divisor = sd
    elif method == "Pareto":
        divisor = np.sqrt(sd)
    else:
        divisor = np.ones_like(sd)
    return X.mean(axis=0), divisor


def scale_columns(X, method):
    """Column-wise scaling applied before PCA."""
    mean, divisor = scaling_factors(X, method)
    return (X - mean) / divisor


@st.cache_data(max_entries=20)
def cv_press(X, method, folds, max_k):
    """Cross-validated PRESS for PCA, using the eigenvector method (Bro et al., 2008).

    Each fold (one sample, or a whole pair) is left out in turn and the scaling + PCA
    are refitted on the rest. Each variable of a left-out sample is then predicted from
    its other variables, because simply projecting the sample would make the error
    always fall with more PCs.
    Returns an array of shape (n_samples, max_k + 1); column k is the error using k PCs.
    """
    press = np.zeros((X.shape[0], max_k + 1))
    for fold in np.unique(folds):
        left_out = folds == fold
        train = X[~left_out]
        mean, divisor = scaling_factors(train, method)
        _, _, Vt = np.linalg.svd((train - mean) / divisor, full_matrices=False)
        for i in np.where(left_out)[0]:
            x = (X[i] - mean) / divisor
            press[i, 0] = np.sum(x ** 2)  # 0 PCs: predict the training mean
            for k in range(1, max_k + 1):
                P = Vt[:k].T
                residual = x - P @ (P.T @ x)
                leverage = np.sum(P ** 2, axis=1)
                # Dividing by (1 - leverage) gives the leave-one-variable-out residual
                press[i, k] = np.sum((residual / np.maximum(1 - leverage, 1e-12)) ** 2)
    return press


def fold_scores(X, method, folds, n_pcs):
    """For each fold, refit the scaling + PCA without it and return the PCA scores.

    Returns a list of (left_out mask, training scores, left-out scores).
    """
    fold_data = []
    for fold in np.unique(folds):
        left_out = folds == fold
        train = X[~left_out]
        mean, divisor = scaling_factors(train, method)
        train_scaled = (train - mean) / divisor
        _, _, Vt = np.linalg.svd(train_scaled, full_matrices=False)
        P = Vt[:n_pcs].T
        fold_data.append((left_out, train_scaled @ P, ((X[left_out] - mean) / divisor) @ P))
    return fold_data


def predict_groups(fold_data, y, k):
    """Predict each left-out sample's group with LDA on the first k PCA scores."""
    # Equal priors: otherwise leaving a sample out biases LDA against that sample's group
    n_groups = len(np.unique(y))
    priors = np.ones(n_groups) / n_groups
    predicted = y.copy()
    for left_out, train_scores, test_scores in fold_data:
        lda = LinearDiscriminantAnalysis(priors=priors).fit(train_scores[:, :k], y[~left_out])
        predicted[left_out] = lda.predict(test_scores[:, :k])
    return predicted


@st.cache_data(max_entries=20)
def cv_predictions(X, method, folds, y, max_k):
    """Cross-validated group predictions; column k - 1 holds the predictions using k PCs."""
    fold_data = fold_scores(X, method, folds, max_k)
    return np.column_stack([predict_groups(fold_data, y, k) for k in range(1, max_k + 1)])


@st.cache_data(max_entries=20)
def permutation_accuracies(X, method, folds, y, k, keep_folds_together, n_permutations=499, seed=0):
    """Cross-validated accuracy after shuffling the group labels many times.

    With keep_folds_together, labels are only shuffled within a fold (e.g. within a pair).
    """
    rng = np.random.default_rng(seed)
    fold_data = fold_scores(X, method, folds, k)
    accuracies = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = y.copy()
        if keep_folds_together:
            for fold in np.unique(folds):
                shuffled[folds == fold] = rng.permutation(y[folds == fold])
        else:
            shuffled = rng.permutation(y)
        accuracies[i] = np.mean(predict_groups(fold_data, shuffled, k) == shuffled)
    return accuracies


def confidence_ellipse(x, y, conf=0.95, n_points=100):
    """Return x, y coordinates of a confidence ellipse around a group of points."""
    if len(x) < 3:
        return None, None
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    radii = np.sqrt(chi2.ppf(conf, df=2) * vals)
    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([radii[0] * np.cos(t), radii[1] * np.sin(t)])
    ellipse = vecs @ circle
    return ellipse[0] + x.mean(), ellipse[1] + y.mean()


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# 2. LOAD DATA
# =============================================================================
st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload a CSV (or use the default file)", type="csv")
if uploaded is None and not DEFAULT_FILE.exists():
    st.info(f"Upload a CSV in the sidebar to get started, or save one as {DEFAULT_FILE.name} next to this script.")
    st.stop()
df = pd.read_csv(uploaded if uploaded is not None else DEFAULT_FILE)

spec_cols, meta_cols = split_columns(df)
wavenumbers = np.array(spec_cols, dtype=float)
X_raw = df[spec_cols].to_numpy(dtype=float)

group_col = st.sidebar.selectbox(
    "Colour samples by", meta_cols,
    index=meta_cols.index("Label") if "Label" in meta_cols else 0,
)
name_col = st.sidebar.selectbox("Sample name column", meta_cols, index=0)

groups = df[group_col].astype(str)
group_names = list(groups.unique())

st.sidebar.caption(f"{len(df)} samples · {len(spec_cols)} wavenumbers · groups: {', '.join(group_names)}")


# =============================================================================
# 3. SIDEBAR: PREPROCESSING
# =============================================================================
st.sidebar.header("2. Preprocessing")
wn_range = st.sidebar.slider(
    "Wavenumber range (cm⁻¹)",
    float(wavenumbers.min()), float(wavenumbers.max()),
    (float(wavenumbers.min()), float(wavenumbers.max())),
)
use_snv = st.sidebar.checkbox("SNV (standard normal variate)", value=False)
deriv = st.sidebar.selectbox(
    "Savitzky-Golay derivative", [0, 1, 2],
    format_func=lambda d: ["None", "1st derivative", "2nd derivative"][d],
)
window = st.sidebar.slider("S-G window (points, odd)", 5, 51, 15, step=2, disabled=(deriv == 0))
scaling = st.sidebar.selectbox("Scaling before PCA", ["Mean-centre", "Autoscale (UV)", "Pareto"])


st.sidebar.header("3. Cross-validation")
cv_options = ["One sample (LOOCV)"] + (["Whole pair"] if "Pair" in meta_cols else [])
cv_mode = st.sidebar.radio("Leave out", cv_options)
whole_pair = cv_mode == "Whole pair"
# Samples with the same fold label are left out together
folds = df["Pair"].to_numpy(dtype=str) if whole_pair else np.arange(len(df)).astype(str)
largest_fold = pd.Series(folds).value_counts().max()


# =============================================================================
# 4. SIDEBAR: PLOT STYLE
# =============================================================================
st.sidebar.header("4. Plot style")
with st.sidebar.expander("Group colours", expanded=True):
    palette = px.colors.qualitative.D3  # hex codes, which st.color_picker requires
    colours = {
        g: st.color_picker(g, palette[i % len(palette)], key=f"colour_{g}")
        for i, g in enumerate(group_names)
    }

with st.sidebar.expander("Lines, markers & text"):
    line_width = st.slider("Spectrum line width", 0.5, 6.0, 1.5, 0.5)
    marker_size = st.slider("Score marker size", 4, 30, 12)
    marker_symbol = st.selectbox("Marker shape", ["circle", "square", "diamond", "triangle-up", "cross", "x"])
    marker_outline = st.slider("Marker outline width", 0.0, 3.0, 1.0, 0.5)
    font_size = st.slider("Font size", 8, 28, 14)

with st.sidebar.expander("Axes & layout"):
    template = st.selectbox("Theme", ["simple_white", "plotly_white", "plotly", "ggplot2", "seaborn", "plotly_dark"])
    show_box = st.checkbox("Draw box around plot", value=True)
    box_width = st.slider("Box line width", 0.5, 4.0, 1.5, 0.5)
    show_grid = st.checkbox("Show grid lines", value=False)
    show_legend = st.checkbox("Show legend", value=True)
    plot_height = st.slider("Plot height (px)", 300, 1000, 550, 50)
    export_format = st.selectbox("Camera-icon download format", ["png", "svg"])

plot_config = {"toImageButtonOptions": {"format": export_format, "scale": 3}}


def style_fig(fig, x_title, y_title, reverse_x=False):
    """Apply all the sidebar style choices to a figure."""
    fig.update_layout(
        template=template,
        height=plot_height,
        font=dict(size=font_size),
        showlegend=show_legend,
        legend_title_text="",
    )
    axis_style = dict(
        showgrid=show_grid,
        showline=show_box,
        mirror=show_box,  # mirror=True draws the top and right edges to make a box
        linecolor="black",
        linewidth=box_width,
        ticks="outside",
        zeroline=False,
    )
    fig.update_xaxes(title=x_title, autorange="reversed" if reverse_x else True, **axis_style)
    fig.update_yaxes(title=y_title, **axis_style)
    return fig


# =============================================================================
# 5. RUN PREPROCESSING + PCA
# =============================================================================
X, wn = preprocess(X_raw, wavenumbers, wn_range, use_snv, deriv, window)
X_scaled = scale_columns(X, scaling)

n_components = min(10, X_scaled.shape[0], X_scaled.shape[1])
pca = PCA(n_components=n_components).fit(X_scaled)
scores = pca.transform(X_scaled)
loadings = pca.components_.T                  # shape: (n_wavenumbers, n_components)
explained = pca.explained_variance_ratio_ * 100
pc_labels = [f"PC{i + 1}" for i in range(n_components)]


# =============================================================================
# 6. TABS WITH THE PLOTS
# =============================================================================
tab_spec, tab_scores, tab_load, tab_scree, tab_class = st.tabs(
    ["Spectra", "PCA scores", "PCA loadings", "Scree plot", "Classification"]
)

# ---- Spectra ----------------------------------------------------------------
with tab_spec:
    view = st.radio("Show", ["All spectra", "Group means"], horizontal=True)
    fig = go.Figure()
    if view == "All spectra":
        for g in group_names:
            for j, row_idx in enumerate(np.where(groups == g)[0]):
                fig.add_trace(go.Scatter(
                    x=wn, y=X[row_idx], mode="lines", name=g,
                    legendgroup=g, showlegend=(j == 0),  # one legend entry per group
                    line=dict(color=colours[g], width=line_width),
                    hovertext=df[name_col].iloc[row_idx], hoverinfo="text+x+y",
                ))
    else:
        for g in group_names:
            fig.add_trace(go.Scatter(
                x=wn, y=X[groups == g].mean(axis=0), mode="lines", name=g,
                line=dict(color=colours[g], width=line_width),
            ))
    y_title = "Absorbance" if (deriv == 0 and not use_snv) else "Processed intensity"
    st.plotly_chart(style_fig(fig, "Wavenumber (cm⁻¹)", y_title, reverse_x=True), config=plot_config)

# ---- PCA scores -------------------------------------------------------------
with tab_scores:
    c1, c2, c3, c4 = st.columns(4)
    pc_x = c1.selectbox("X axis", pc_labels, index=0)
    pc_y = c2.selectbox("Y axis", pc_labels, index=1 if n_components > 1 else 0)
    show_labels = c3.checkbox("Label points", value=False)
    show_ellipse = c4.checkbox("95% ellipses", value=True)

    ix, iy = pc_labels.index(pc_x), pc_labels.index(pc_y)
    fig = go.Figure()
    for g in group_names:
        mask = (groups == g).to_numpy()
        sx, sy = scores[mask, ix], scores[mask, iy]
        if show_ellipse:
            ex, ey = confidence_ellipse(sx, sy)
            if ex is not None:
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines", fill="toself",
                    fillcolor=hex_to_rgba(colours[g], 0.15),
                    line=dict(color=colours[g], width=1, dash="dot"),
                    legendgroup=g, showlegend=False, hoverinfo="skip",
                ))
        fig.add_trace(go.Scatter(
            x=sx, y=sy, name=g, legendgroup=g,
            mode="markers+text" if show_labels else "markers",
            text=df.loc[mask, name_col], textposition="top center",
            marker=dict(color=colours[g], size=marker_size, symbol=marker_symbol,
                        line=dict(color="black", width=marker_outline)),
            hovertemplate="%{text}<br>" + pc_x + ": %{x:.3f}<br>" + pc_y + ": %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"))
    fig.add_vline(x=0, line=dict(color="grey", width=1, dash="dash"))
    style_fig(fig, f"{pc_x} ({explained[ix]:.1f}%)", f"{pc_y} ({explained[iy]:.1f}%)")
    st.plotly_chart(fig, config=plot_config)

    scores_df = pd.concat([df[meta_cols], pd.DataFrame(scores, columns=pc_labels)], axis=1)
    with st.expander("Scores table"):
        st.dataframe(scores_df)
        st.download_button("Download scores CSV", scores_df.to_csv(index=False), "pca_scores.csv")

# ---- PCA loadings -----------------------------------------------------------
with tab_load:
    chosen_pcs = st.multiselect("Components to plot", pc_labels, default=pc_labels[:1])
    c1, c2, c3, c4 = st.columns(4)
    x_box_low = c1.number_input("Lowest wavenumber (cm⁻¹)", value=None, step=50.0, placeholder="Auto")
    x_box_high = c2.number_input("Highest wavenumber (cm⁻¹)", value=None, step=50.0, placeholder="Auto")
    y_box_min = c3.number_input("Loading min", value=None, step=0.01, format="%.3f", placeholder="Auto")
    y_box_max = c4.number_input("Loading max", value=None, step=0.01, format="%.3f", placeholder="Auto")
    st.caption(
        "These boxes only zoom the plot. Leave a box blank for automatic. To refit the PCA "
        "on a section instead, use the wavenumber range in the sidebar."
    )
    load_palette = px.colors.qualitative.Dark2
    fig = go.Figure()
    for k, pc in enumerate(chosen_pcs):
        fig.add_trace(go.Scatter(
            x=wn, y=loadings[:, pc_labels.index(pc)], mode="lines",
            name=f"{pc} ({explained[pc_labels.index(pc)]:.1f}%)",
            line=dict(color=load_palette[k % len(load_palette)], width=line_width),
        ))
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"))
    style_fig(fig, "Wavenumber (cm⁻¹)", "Loading", reverse_x=True)

    boxes = [x_box_low, x_box_high, y_box_min, y_box_max]
    if chosen_pcs and any(box is not None for box in boxes):
        x_low, x_high = sorted([wn.min() if x_box_low is None else x_box_low,
                                wn.max() if x_box_high is None else x_box_high])
        in_window = (wn >= x_low) & (wn <= x_high)
        if not in_window.any():
            st.warning("No data between those wavenumbers, so the zoom has been ignored.")
        else:
            # Blank loading boxes fit the y axis to the section on show, not the whole spectrum
            shown = loadings[in_window][:, [pc_labels.index(pc) for pc in chosen_pcs]]
            pad = 0.05 * (shown.max() - shown.min())
            y_low, y_high = sorted([shown.min() - pad if y_box_min is None else y_box_min,
                                    shown.max() + pad if y_box_max is None else y_box_max])
            if x_low == x_high or y_low == y_high:
                st.warning("The lowest and highest values need to differ, so the zoom has been ignored.")
            else:
                fig.update_xaxes(range=[x_high, x_low], autorange=False)  # high first: FTIR axis runs backwards
                fig.update_yaxes(range=[y_low, y_high], autorange=False)
    st.plotly_chart(fig, config=plot_config)

# ---- Scree plot -------------------------------------------------------------
with tab_scree:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pc_labels, y=explained, name="Individual", marker_color="#4C72B0"))
    fig.add_trace(go.Scatter(x=pc_labels, y=np.cumsum(explained), name="Cumulative",
                             mode="lines+markers", line=dict(color="#DD8452", width=2)))
    st.plotly_chart(style_fig(fig, "Principal component", "Explained variance (%)"), config=plot_config)

    st.subheader(f"Cross-validation: leave out {cv_mode.lower()}")
    # A mean-centred training set of n samples only supports n - 1 components
    max_k = min(n_components, X.shape[0] - largest_fold - 1, X.shape[1] - 1)
    if max_k < 1:
        st.info("There are too few samples left for training to cross-validate.")
    else:
        press = cv_press(X, scaling, folds, max_k)
        total_press = press.sum(axis=0)
        r2_cum = np.cumsum(explained[:max_k]) / 100
        q2_cum = 1 - total_press[1:] / total_press[0]
        q2_each = 1 - total_press[1:] / total_press[:-1]
        cv_labels = pc_labels[:max_k]
        # Q² tends to plateau, so its exact peak is mostly noise: take the fewest PCs close to it
        best_k = int(np.argmax(q2_cum >= q2_cum.max() - 0.01)) + 1

        st.caption(
            "R²X is how much of the data the model fits. Q² is how well it predicts samples it "
            "wasn't trained on. Extra PCs that raise R²X but lower Q² are mostly fitting noise."
        )
        if q2_cum.max() <= 0:
            st.warning("Q² is never above 0: no number of PCs predicts left-out samples better than the mean.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Fewest PCs within 0.01 of highest Q²", best_k)
            c2.metric("Q² (cumulative)", f"{q2_cum[best_k - 1]:.3f}")
            c3.metric("R²X (cumulative)", f"{r2_cum[best_k - 1]:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cv_labels, y=r2_cum, name="R²X (cumulative)",
                                 mode="lines+markers", line=dict(color="#4C72B0", width=2)))
        fig.add_trace(go.Scatter(x=cv_labels, y=q2_cum, name="Q² (cumulative)",
                                 mode="lines+markers", line=dict(color="#DD8452", width=2)))
        fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"))
        st.plotly_chart(style_fig(fig, "Number of components", "R²X / Q²"), config=plot_config)

        k_view = st.selectbox("PCs used for the per-sample error", range(1, max_k + 1), index=best_k - 1)
        sample_error = press[:, k_view] / press[:, 0]
        fig = go.Figure()
        for g in group_names:
            mask = (groups == g).to_numpy()
            fig.add_trace(go.Bar(x=df.loc[mask, name_col], y=sample_error[mask], name=g,
                                 marker_color=colours[g]))
        fig.add_hline(y=1, line=dict(color="grey", width=1, dash="dash"))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(style_fig(fig, "Left-out sample", "Prediction error (relative to mean)"), config=plot_config)
        st.caption("Above the dashed line at 1, the model predicts that sample worse than the training mean does.")

        cv_df = pd.DataFrame({
            "Left_out": cv_mode,
            "Components": range(1, max_k + 1),
            "R2X_cumulative": r2_cum,
            "Q2_cumulative": q2_cum,
            "Q2_per_component": q2_each,
            "PRESS": total_press[1:],
        })
        with st.expander("Cross-validation table"):
            st.dataframe(cv_df, hide_index=True)
            st.download_button("Download cross-validation CSV", cv_df.to_csv(index=False), "pca_cross_validation.csv")

# ---- Classification ---------------------------------------------------------
with tab_class:
    st.subheader(f"Can the spectra predict {group_col}?")
    st.caption(
        f"PCA-LDA, leaving out {cv_mode.lower()} each time: the scaling, PCA and classifier are "
        "refitted without the left-out spectra, which are then assigned to a group."
    )
    y = groups.to_numpy(dtype=str)
    group_counts = pd.Series(y).value_counts()
    # Every group must stay in training, with enough samples left to estimate LDA's covariance
    groups_stay_in_training = all(
        set(np.unique(y)) == set(np.unique(y[folds != fold])) for fold in np.unique(folds)
    )
    max_lda_k = min(n_components, X.shape[0] - largest_fold - len(group_names), X.shape[1] - 1)

    if len(group_names) < 2 or group_counts.min() < 2 or not groups_stay_in_training or max_lda_k < 1:
        st.info(
            f"The samples can't be classified by {group_col} with this setup: each group needs "
            "at least 2 samples, and every group must remain in training when samples are left out."
        )
    else:
        predictions = cv_predictions(X, scaling, folds, y, max_lda_k)
        accuracy = (predictions == y[:, None]).mean(axis=0)
        no_skill = group_counts.max() / len(y)

        k_class = st.selectbox(
            "PCs used by the classifier", range(1, max_lda_k + 1), index=min(1, max_lda_k - 1),
            key="k_class",  # keeps your choice when the leave-out switch changes the options
            help="Choose this before looking at the results. Picking the best-looking value makes the accuracy and p-value too optimistic.",
        )
        predicted = predictions[:, k_class - 1]

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy[k_class - 1]:.0%}", help=f"{(predicted == y).sum()} of {len(y)} correct")
        c2.metric("Accuracy with no skill", f"{no_skill:.0%}", help="Always guessing the largest group")

        run_permutation = st.toggle("Run permutation test (can take about 30 seconds)")
        if run_permutation:
            with st.spinner("Shuffling labels..."):
                null_accuracy = permutation_accuracies(X, scaling, folds, y, k_class, whole_pair)
            p_value = (1 + np.sum(null_accuracy >= accuracy[k_class - 1])) / (1 + len(null_accuracy))
            st.metric("Permutation p-value", f"{p_value:.3f}")
            st.caption(
                "How often shuffled group labels score at least as well as the real ones, "
                f"from {len(null_accuracy)} shuffles."
            )
            if whole_pair and len(group_names) == 2:
                labellings = np.prod([2 if len(set(y[folds == fold])) == 2 else 1 for fold in np.unique(folds)])
                st.caption(
                    f"Swapping labels within pairs allows only {labellings} labellings, so p can't go "
                    f"much below {2 / labellings:.3f}, however good the separation."
                )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, max_lda_k + 1)), y=accuracy * 100, name="Cross-validated accuracy",
                                 mode="lines+markers", line=dict(color="#4C72B0", width=2)))
        fig.add_hline(y=no_skill * 100, line=dict(color="grey", width=1, dash="dash"))
        style_fig(fig, "Number of components", "Accuracy (%)")
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(range=[-5, 105], autorange=False)
        st.plotly_chart(fig, config=plot_config)
        st.caption("The dashed line is the accuracy with no skill.")

        st.write("**Actual group (rows) against predicted group (columns)**")
        st.dataframe(pd.crosstab(pd.Series(y, name="Actual"), pd.Series(predicted, name="Predicted")))

        class_df = pd.DataFrame({
            "Sample": df[name_col], "Actual": y, "Predicted": predicted, "Correct": predicted == y,
        })
        with st.expander("Per-sample predictions"):
            st.dataframe(class_df, hide_index=True)
            st.download_button("Download predictions CSV", class_df.to_csv(index=False), "pca_lda_predictions.csv")
