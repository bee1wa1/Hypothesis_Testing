# streamlit_hypothesis_testing.py
# Two-sample hypothesis testing app with built-in datasets

import io
import csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

# --------------------------- PAGE SETUP ---------------------------
st.set_page_config(page_title="Two-Sample Hypothesis Testing", layout="wide")
st.title("Two-Sample Hypothesis Testing")
st.caption(
    "Compare two datasets to check if they likely come from the same distribution. "
    "You can upload files, paste values, or use built-in sample datasets."
)

# --------------------------- HELPERS ---------------------------
def read_any_table(uploaded_file: io.BytesIO, sep=None) -> pd.DataFrame:
    """Read CSV/TSV/TXT (auto-delimiter) or Excel."""
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    data = uploaded_file.read()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    try:
        sample = "\n".join(text.splitlines()[:10])
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = None
    for guess in ([sep] if sep else []) + [",", "\t", ";", "|"]:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(io.StringIO(text), sep=guess, engine="python")
        except Exception:
            continue
    return pd.read_csv(io.StringIO(text))


def coerce_numeric_any(x):
    """Return clean float numpy array."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        arr = pd.to_numeric(pd.Series(x).squeeze(), errors="coerce").to_numpy(dtype=float)
    elif isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
    else:
        arr = np.array([x], dtype=float)
    return arr[np.isfinite(arr)]


def ecdf(arr: np.ndarray):
    arr = np.sort(np.asarray(arr))
    n = arr.size
    return arr, np.arange(1, n + 1) / n if n > 0 else (np.array([]), np.array([]))


def holm_bonferroni(pvals: dict, alpha: float):
    """Holm–Bonferroni correction."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    any_reject = False
    stop = False
    for i, (name, p) in enumerate(items):
        thr = alpha / (m - i)
        reject = (p <= thr) and (not stop)
        results[name] = (p, reject)
        if not reject:
            stop = True
        else:
            any_reject = True
    return {k: results[k] for k in pvals.keys()}, any_reject


def fmt(x):
    try:
        return f"{x:.4g}"
    except Exception:
        return str(x)


# --------------------------- SIDEBAR INPUTS ---------------------------
with st.sidebar:
    st.header("Data Inputs")

    dataset_choice = st.radio(
        "Choose how to load data:",
        ["Built-in examples", "Upload files", "Paste values"],
    )

    if dataset_choice == "Built-in examples":
        example = st.selectbox(
            "Select example:",
            [
                "Normal(0,1) vs Normal(0.2,1)",
                "Normal(0,1) vs Exponential(1)",
            ],
        )
        if example == "Normal(0,1) vs Normal(0.2,1)":
            dfA = pd.DataFrame({"value": np.random.normal(0, 1, 300)})
            dfB = pd.DataFrame({"value": np.random.normal(0.2, 1, 300)})
        else:
            dfA = pd.DataFrame({"value": np.random.normal(0, 1, 300)})
            dfB = pd.DataFrame({"value": np.random.exponential(1, 300)})
        colA = colB = "value"

    elif dataset_choice == "Upload files":
        fileA = st.file_uploader("Upload Dataset A", type=["csv", "tsv", "txt", "xlsx"])
        fileB = st.file_uploader("Upload Dataset B", type=["csv", "tsv", "txt", "xlsx"])
        dfA = dfB = None
        colA = colB = None
        if fileA:
            dfA = read_any_table(fileA)
            colA = st.selectbox("Column A", dfA.columns)
        if fileB:
            dfB = read_any_table(fileB)
            colB = st.selectbox("Column B", dfB.columns)

    else:  # Paste values
        txtA = st.text_area("Values for A (one per line)", height=150)
        txtB = st.text_area("Values for B (one per line)", height=150)
        dfA = dfB = None
        colA = colB = None
        if txtA.strip() and txtB.strip():
            dfA = pd.DataFrame({"value": pd.to_numeric(txtA.splitlines(), errors="coerce")})
            dfB = pd.DataFrame({"value": pd.to_numeric(txtB.splitlines(), errors="coerce")})
            colA = colB = "value"

    st.divider()
    st.header("Preprocessing")
    log_transform = st.checkbox("Apply log10 transform (exclude ≤0)", value=False)
    trim_pct = st.slider("Winsorize extremes (per tail %)", 0.0, 20.0, 0.0, step=1.0) / 100.0
    alpha = st.slider("Significance level α", 0.001, 0.2, 0.05, step=0.001)

# --------------------------- VALIDATION ---------------------------
if dfA is None or dfB is None or colA is None or colB is None:
    st.info("Please select or upload both datasets to proceed.")
    st.stop()

x_raw = dfA[colA]
y_raw = dfB[colB]

def preprocess(data):
    arr = coerce_numeric_any(data)
    if arr.size == 0:
        return arr
    if log_transform:
        arr = arr[arr > 0]
        arr = np.log10(arr)
    if trim_pct > 0 and arr.size > 0:
        lo = np.quantile(arr, trim_pct)
        hi = np.quantile(arr, 1 - trim_pct)
        arr = np.clip(arr, lo, hi)
    return arr

x = preprocess(x_raw)
y = preprocess(y_raw)

if x.size < 3 or y.size < 3:
    st.error("Each dataset must contain at least 3 numeric values after preprocessing.")
    st.stop()

# --------------------------- SUMMARY STATS ---------------------------
st.subheader("Summary Statistics")

summary = pd.DataFrame(
    {
        "A": [x.size, np.mean(x), np.std(x, ddof=1), np.median(x)],
        "B": [y.size, np.mean(y), np.std(y, ddof=1), np.median(y)],
    },
    index=["n", "Mean", "Std", "Median"],
)
st.dataframe(summary.round(4))

# --------------------------- VISUALIZATIONS ---------------------------
st.subheader("Visual Comparison")

col1, col2 = st.columns(2)
with col1:
    df_plot = pd.DataFrame(
        {"value": np.concatenate([x, y]),
         "group": np.array(["A"] * x.size + ["B"] * y.size)}
    )
    fig = px.histogram(df_plot, x="value", color="group", barmode="overlay",
                       nbins=40, opacity=0.6, marginal="rug")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    X, FX = ecdf(x)
    Y, FY = ecdf(y)
    df_ecdf = pd.DataFrame(
        {"value": np.concatenate([X, Y]),
         "ecdf": np.concatenate([FX, FY]),
         "group": np.array(["A"] * X.size + ["B"] * Y.size)}
    )
    fig2 = px.line(df_ecdf, x="value", y="ecdf", color="group")
    st.plotly_chart(fig2, use_container_width=True)

# Q–Q Plot
st.markdown("**Q–Q Plot (A vs B Quantiles)**")
if x.size > 4 and y.size > 4:
    q_count = int(min(99, x.size, y.size))
    q = np.linspace(0.01, 0.99, q_count)
    try:
        qx = np.quantile(x, q)
        qy = np.quantile(y, q)
        dfqq = pd.DataFrame({"A_quantile": qx, "B_quantile": qy})
        fig3 = px.scatter(dfqq, x="A_quantile", y="B_quantile", trendline="ols")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.info(f"Q–Q plot unavailable: {e}")
else:
    st.info("Q–Q plot skipped: not enough data points.")

# --------------------------- TESTS ---------------------------
st.subheader("Hypothesis Tests")

results = []

def add_result(name, stat, p, note=""):
    results.append({
        "Test": name,
        "Statistic": fmt(stat),
        "p-value": fmt(p),
        f"Decision (α={alpha:.3g})": "Reject H₀" if (p <= alpha) else "Fail to reject H₀",
        "Notes": note
    })

# KS test
ks = stats.ks_2samp(x, y)
add_result("KS (2-sample)", ks.statistic, ks.pvalue, "Omnibus: CDF difference")

# Cramér–von Mises
try:
    cvm = stats.cramervonmises_2samp(x, y)
    add_result("Cramér–von Mises (2-sample)", cvm.statistic, cvm.pvalue, "Omnibus; balanced power")
except Exception as e:
    add_result("Cramér–von Mises (2-sample)", np.nan, np.nan, f"Unavailable: {e}")

# Anderson–Darling
try:
    ad = stats.anderson_ksamp([x, y])
    add_result("Anderson–Darling (k-sample)", ad.statistic, getattr(ad, "pvalue", np.nan), "Omnibus; tail-sensitive")
except Exception as e:
    add_result("Anderson–Darling (k-sample)", np.nan, np.nan, f"Unavailable: {e}")

# Mann–Whitney U
try:
    mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
    add_result("Mann–Whitney U", mwu.statistic, mwu.pvalue, "Nonparametric median shift test")
except Exception as e:
    add_result("Mann–Whitney U", np.nan, np.nan, f"Error: {e}")

# Welch’s t-test
try:
    t = stats.ttest_ind(x, y, equal_var=False)
    add_result("Welch’s t-test", t.statistic, t.pvalue, "Difference in means (unequal variances)")
except Exception as e:
    add_result("Welch’s t-test", np.nan, np.nan, f"Error: {e}")

res_df = pd.DataFrame(results)
st.dataframe(res_df, use_container_width=True)

# --------------------------- OVERALL VERDICT ---------------------------
st.subheader("Overall Verdict (Holm–Bonferroni)")

omnibus_tests = ["KS (2-sample)", "Cramér–von Mises (2-sample)", "Anderson–Darling (k-sample)"]
pvals = {row["Test"]: float(row["p-value"]) for _, row in res_df.iterrows()
         if row["Test"] in omnibus_tests and str(row["p-value"]).replace('.', '', 1).isdigit()}

if not pvals:
    st.info("No valid omnibus tests available.")
else:
    corrected, any_reject = holm_bonferroni(pvals, alpha)
    table = pd.DataFrame([{
        "Test": name,
        "Raw p": fmt(p),
        "Reject after Holm–Bonferroni": "Yes" if rej else "No"
    } for name, (p, rej) in corrected.items()])
    st.dataframe(table, use_container_width=True)

    if any_reject:
        st.error("**Conclusion:** Statistically significant evidence that A and B differ in distribution.")
    else:
        st.success("**Conclusion:** No significant evidence that A and B differ in distribution.")

st.caption(
    "KS, CvM, and AD are omnibus tests for distributional equality. "
    "Mann–Whitney targets median shift; Welch tests mean difference."
)
