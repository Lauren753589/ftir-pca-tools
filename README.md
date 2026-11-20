# FTIR PCA tools

Streamlit apps and figure scripts for exploring ATR-FTIR spectra with principal component analysis (PCA).
No data is included: the scripts expect your own CSV files.

## Setup

Install Python 3, then in a terminal in this folder:

```
py -m pip install -r requirements.txt
```

## The apps

### `ftir_app.py`: FTIR Spectral Explorer

Runs the PCA itself, from raw or preprocessed spectra.

```
py -m streamlit run ftir_app.py
```

Expects one CSV with one row per sample: a few text columns (for example `Sample`, `Label`, `Pair`), then one
column per wavenumber, with numeric column names such as `450.0`. Upload it in the sidebar, or save it as
`ftir_data.csv` next to the script.

- **Preprocessing:** wavenumber range, SNV, Savitzky-Golay 1st/2nd derivative, and mean-centring, autoscaling or Pareto scaling.
- **Plots:** spectra (individual or group means), PCA scores with 95% ellipses, loadings with adjustable axis limits, and a scree plot.
- **Cross-validation:** R²X and Q² by number of components (eigenvector method, Bro et al., 2008). You can leave out one sample at a time or whole pairs.
- **Classification:** cross-validated PCA-LDA prediction of a group column, with an optional permutation test.
- **Styling:** colours, line widths, markers, fonts, axes and export format are all adjustable in the sidebar.

### `pca_app.py`: FTIR PCA Figure Builder

Builds figures from a PCA that was run elsewhere (for example in Unscrambler). Double-click `Run PCA app.bat`, or:

```
py -m streamlit run pca_app.py
```

Expects three CSVs, uploaded in the sidebar or placed next to the script:

| File | Layout |
|---|---|
| `All spectra scores.csv` | one row per sample, first column the sample name, then `PC 1`, `PC 2`, … |
| `all spectra loadings.csv` | one row per PC, one column per wavenumber |
| `explained variance.csv` | one row, columns `PC 0`, `PC 1`, … (cumulative %) |

Sample names are expected to look like `Material 1 2b`: material, specimen number, then an optional replicate letter.
Materials can be renamed in the sidebar. The names are saved in `material_names.json`, which is excluded from the repository.

## Figure scripts

| Script | Makes |
|---|---|
| `make_loadings_figure.py` | 2 × 2 figure of the PC2 and PC3 loadings in the C–H and fingerprint regions |
| `make_pc2_vs_pe.py` | mean PC2 score against Py-GC/MS PE content, with the Pearson correlation; needs `material_names.json` |
| `make_pygcms_zoom.py` | expanded 4.5–10 min Py-GC/MS region, cropped from a chromatogram screenshot |

Run each from this folder with `py <script name>`.
