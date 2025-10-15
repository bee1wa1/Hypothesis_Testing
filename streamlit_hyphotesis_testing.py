# streamlit_hypothesis_testing.py
# Two-sample hypothesis testing app with power analysis & sample-size suggestions

import io
import csv
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

# --------------------------- PAGE SETUP ---------------------------
st.set_page_config(page_title="Two-Sample Hypothesis Testing + Power", layout="wide")
st.title("Two-Sample Hypothesis Testing")
st.caption(
    "Compare two datasets and estimate statistical power for each test. "
    "Upload files, paste values, or use built-in samples. "
    "Includes Holm–Bonferroni correction, power analysis (simulation-based), and recommended sample sizes."
)

# --------------------------- HELPERS ---------------------------
def read_any_table(uploaded_file: io.BytesIO, sep=None) -> pd.DataFrame:
    """Read CSV/TSV/TXT (auto-delimiter) or Excel."""
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    data = uploaded_file.read()
    text = None
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
    # return in original key order
    return {k: results[k] for k in pvals.keys()}, any_reject


def fmt(x, nd=4):
    try:
        return f"{x:.{nd}g}"
    except Exception:
        return str(x)


def pooled_sd(x, y):
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    nx, ny = len(x), len(y)
    return math.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / max(nx + ny - 2, 1))


def cohens_d(x, y):
    sp = pooled_sd(x, y)
    if sp == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sp


def hodges_lehmann_shift(x, y):
    """HL estimator for location shift (median of pairwise differences)."""
    diffs = np.subtract.outer(x, y).ravel()
    return np.median(diffs)


def run_test(x, y, name, alpha):
    """Return (stat, pvalue)."""
    if name == "Welch’s t-test":
        r = stats.ttest_ind(x, y, equal_var=False)
        return r.statistic, r.pvalue
    if name == "Mann–Whitney U":
        r = stats.mannwhitneyu(x, y, alternative="two-sided")
        return r.statistic, r.pvalue
    if name == "KS (2-sample)":
        r = stats.ks_2samp(x, y)
        return r.statistic, r.pvalue
    if name == "Cramér–von Mises (2-sample)":
        r = stats.cramervonmises_2samp(x, y)
        return r.statistic, r.pvalue
    if name == "Anderson–Darling (k-sample)":
        r = stats.anderson_ksamp([x, y])
        # scipy returns statistic and sometimes pvalue attribute
        return r.statistic, getattr(r, "pvalue", np.nan)
    raise ValueError(f"Unknown test: {name}")


def simulate_power(test_name, x, y, nA, nB, alpha, sims=2000, seed=0):
    """
    Empirical power via bootstrap-from-empirical distributions.
    Draw samples with replacement of sizes (nA, nB) from x and y as the 'true' populations.
    """
    rng = np.random.default_rng(seed)
    reject = 0
    # fast-path checks
    if nA < 3 or nB < 3 or len(x) < 3 or len(y) < 3:
        return np.nan
    for _ in range(sims):
        xb = rng.choice(x, size=nA, replace=True)
        yb = rng.choice(y, size=nB, replace=True)
        _, p = run_test(xb, yb, test_name, alpha)
        if np.isfinite(p) and p <= alpha:
            reject += 1
    return reject / sims


def suggest_n_for_target_power(test_name, x, y, alpha, target_power, sims=1000, seed=0, max_scale=10.0):
    """
    Keep nB/nA ratio fixed. Scale both groups until empirical power >= target_power or reach max_scale.
    Returns (nA_suggest, nB_suggest, power_at_suggest).
    """
    nA0, nB0 = len(x), len(y)
    if nA0 < 3 or nB0 < 3:
        return np.nan, np.nan, np.nan

    ratio = nB0 / nA0
    # Start at current size
    best_nA = nA0
    best_nB = nB0
    best_power = simulate_power(test_name, x, y, nA0, nB0, alpha, sims=sims, seed=seed)

    # Coarse grid search over scale factors
    scales = np.arange(1.0, max_scale + 0.001, 0.5)
    for s in scales:
        nA = int(math.ceil(nA0 * s))
        nB = int(math.ceil(nA * ratio))
        pw = simulate_power(test_name, x, y, nA, nB, alpha, sims=sims, seed=seed + 123)
        if pw >= target_power:
            # Try small refinement around this point
            best_nA, best_nB, best_power = nA, nB, pw
            # local refine (downward)
            for nA_try in range(nA, nA0 - 1, -1):
                nB_try = max(3, int(math.ceil(nA_try * ratio)))
                pw_try = simulate_power(test_name, x, y, nA_try, nB_try, alpha, sims=sims, seed=seed + 456)
                if pw_try >= target_power:
                    best_nA, best_nB, best_power = nA_try, nB_try, pw_try
                else:
                    break
            return best_nA, best_nB, best_power

    return best_nA, best_nB, best_power  # may still be below target_power


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
                "Normal(0,1) vs Normal(0.2,1)  (mean shift, same shape)",
                "Normal(0,1) vs Exponential(1) (shape difference)",
            ],
        )
        if "Normal(0,1) vs Normal(0.2,1)" in example:
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

    st.divider()
    st.header("Power Analysis Settings")
    sims = st.number_input("Simulation draws per estimate (power)", min_value=200, max_value=20000, value=500, step=200,
                           help="Higher = smoother estimates but slower.")
    target_power = st.slider("Target power for recommendations", 0.5, 0.95, 0.80, step=0.01)
    max_scale = st.slider("Max sample size multiplier (search limit)", 1.0, 20.0, 10.0, step=0.5)

    assumed_delta = st.number_input(
        "Assumed true mean difference Δ (for Welch power)",
        value=float(abs(np.mean(dfA[colA]) - np.mean(dfB[colB]))) if (
                    dfA is not None and dfB is not None and colA and colB) else 0.0,
        step=0.001, format="%.3f",
        help="Power depends on the true effect size (difference in means). Default = |current mean difference|."
    )

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
st.dataframe(summary.round(4), use_container_width=True)

# Effect metrics
d = cohens_d(x, y)
hl = hodges_lehmann_shift(x, y)
ksD = stats.ks_2samp(x, y).statistic
st.markdown(
    f"**Effect snapshots:** Cohen’s d = `{fmt(d)}`, Hodges–Lehmann shift (A−B) = `{fmt(hl)}`, KS D = `{fmt(ksD)}`"
)

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
        f"Decision (α={alpha:.3g})": "Reject H₀" if (np.isfinite(p) and p <= alpha) else "Fail to reject H₀",
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

# --------------------------- POWER ANALYSIS ---------------------------
st.subheader("Power Analysis (Simulation-based)")

tests_for_power = [
    ("Welch’s t-test", "Mean difference (location)"),
    ("Mann–Whitney U", "Median/rank shift (location, robust)"),
    ("KS (2-sample)", "Overall CDF/shape difference"),
    ("Cramér–von Mises (2-sample)", "Overall, balanced power"),
    ("Anderson–Darling (k-sample)", "Overall, tail-sensitive")
]

power_rows = []
with st.spinner("Estimating power for current sample sizes via bootstrap…"):
    for test_name, desc in tests_for_power:
        try:
            pw = simulate_power(test_name, x, y, len(x), len(y), alpha, sims=int(sims), seed=42)
            nA_sug, nB_sug, pw_sug = suggest_n_for_target_power(
                test_name, x, y, alpha, target_power, sims=min(int(sims/4), 1000), seed=7, max_scale=float(max_scale)
            )
        except Exception as e:
            pw = np.nan
            nA_sug = nB_sug = pw_sug = np.nan
        power_rows.append({
            "Test": test_name,
            "What it detects": desc,
            "n_A": len(x),
            "n_B": len(y),
            "Power @ current n": fmt(pw, nd=3),
            f"Suggested n_A (power≥{target_power:.2f})": int(nA_sug) if np.isfinite(nA_sug) else "",
            f"Suggested n_B (power≥{target_power:.2f})": int(nB_sug) if np.isfinite(nB_sug) else "",
            "Power @ suggested n": fmt(pw_sug, nd=3) if np.isfinite(pw_sug) else ""
        })

power_df = pd.DataFrame(power_rows)
st.dataframe(power_df, use_container_width=True)

st.caption(
    "Power is estimated by resampling from the empirical distributions (bootstrap), "
    "so it reflects the *kind* of difference present in your data (mean, variance, skew). "
    "Recommendations keep the current n_B / n_A ratio and scale both groups."
)
# --------------------------- POWER MAP: WELCH (MEAN DIFFERENCE) ---------------------------
st.subheader("Power Map for Mean Difference (Welch’s t-test)")

# Pull suggested nA/nB for Welch from the table you just computed
try:
    row_welch = power_df.loc[power_df["Test"] == "Welch’s t-test"].iloc[0]
    nA_sug = int(row_welch[[c for c in power_df.columns if "Suggested n_A" in c][0]])
    nB_sug = int(row_welch[[c for c in power_df.columns if "Suggested n_B" in c][0]])
except Exception:
    nA_sug, nB_sug = len(x), len(y)  # fallback

# Ingredients for analytic power approximation
sA = float(np.std(x, ddof=1)); sB = float(np.std(y, ddof=1))
delta = float(assumed_delta if "assumed_delta" in locals() else abs(np.mean(x) - np.mean(y)))
nA0, nB0 = len(x), len(y)

def power_welch_normal(alpha, nA, nB, delta, sA, sB):
    se = np.sqrt((sA**2)/nA + (sB**2)/nB)
    if se <= 0:
        return 1.0
    zcrit = stats.norm.ppf(1 - alpha/2)
    mu = delta / se
    return stats.norm.cdf(-zcrit - mu) + (1 - stats.norm.cdf(zcrit - mu))

# Grid around current and suggested sizes
nA_max = max(nA_sug, nA0) + 20
nB_max = max(nB_sug, nB0) + 20
nA_vals = np.arange(max(3, nA0), nA_max + 1)
nB_vals = np.arange(max(3, nB0), nB_max + 1)

# Compute power surface
Z = np.zeros((len(nB_vals), len(nA_vals)))
for i, nb in enumerate(nB_vals):
    for j, na in enumerate(nA_vals):
        Z[i, j] = power_welch_normal(alpha, na, nb, delta, sA, sB)

df_heat = (
    pd.DataFrame(Z, index=nB_vals, columns=nA_vals)
      .reset_index()
      .melt(id_vars="index", var_name="n_A", value_name="Power")
      .rename(columns={"index": "n_B"})
)

# Heatmap
figH = px.density_heatmap(
    df_heat, x="n_A", y="n_B", z="Power",
    nbinsx=len(nA_vals), nbinsy=len(nB_vals),
    range_color=[0, 1], color_continuous_scale="Viridis",
    title=f"Welch power heatmap (α={alpha:.3g}, Δ={delta:.3f})"
)
figH.add_vline(x=nA0, line_dash="dot")
figH.add_hline(y=nB0, line_dash="dot")
figH.add_vline(x=nA_sug, line_dash="dash")
figH.add_hline(y=nB_sug, line_dash="dash")
st.plotly_chart(figH, use_container_width=True)

# Line plots at suggested n
lineA = pd.DataFrame({
    "n_B": nB_vals,
    "Power": [power_welch_normal(alpha, nA_sug, nb, delta, sA, sB) for nb in nB_vals]
})
lineB = pd.DataFrame({
    "n_A": nA_vals,
    "Power": [power_welch_normal(alpha, na, nB_sug, delta, sA, sB) for na in nA_vals]
})

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        px.line(lineB, x="n_A", y="Power",
                title=f"Power vs n_A (hold n_B={nB_sug}, α={alpha:.3g}, Δ={delta:.3f})"
        ).add_hline(y=target_power, line_dash="dash"),
        use_container_width=True
    )
with c2:
    st.plotly_chart(
        px.line(lineA, x="n_B", y="Power",
                title=f"Power vs n_B (hold n_A={nA_sug}, α={alpha:.3g}, Δ={delta:.3f})"
        ).add_hline(y=target_power, line_dash="dash"),
        use_container_width=True
    )

st.caption(
    "Heatmap shows analytic power for Welch’s t-test under a two-sided z-approximation. "
    "Δ is the assumed true mean difference (set in the sidebar). "
    "Dashed lines = suggested n; dotted lines = current n."
)


# --------------------------- PRACTICAL CONCLUSION ---------------------------
st.subheader("Practical Conclusion")

any_sig = any((np.isfinite(float(p)) and float(p) <= alpha) for p in res_df["p-value"])
if any_sig:
    st.error(
        "✅ **Practical conclusion**\n\n"
        "- At least one test finds a statistically significant difference between Dataset A and B.\n"
        "- Inspect which test and effect metric is triggering significance (mean/median/shape).\n"
        "- Consider multiple-comparison adjustment (Holm–Bonferroni table above) and whether the detected difference is practically relevant.\n"
    )
else:
    st.success(
        "✅ **Practical conclusion**\n\n"
        "Based on these data, there is **no statistically significant difference** between Dataset A and Dataset B "
        "in terms of mean, median, or overall distribution.\n\n"
        "However, the analysis has **low sensitivity** at the current sample sizes—especially if one group is much smaller. "
        "Use the **Power Analysis** table to see how many additional samples are recommended to reach your target power "
        f"(default **{int(target_power*100)}%**) for each hypothesis test."
    )

# --------------------------- NOTES / TIPS ---------------------------
with st.expander("Notes & Tips"):
    st.markdown(
        "- **Power for t-test** primarily targets **mean differences** (Cohen’s d). "
        "Nonparametric and omnibus tests (MWU, KS, CvM, AD) capture **distributional** changes; "
        "their power is estimated here via **simulation** from your empirical data.\n"
        "- If you expect **paired measurements** or **repeated measures**, consider paired tests or mixed models—they often "
        "achieve the same sensitivity with fewer samples (by reducing variability).\n"
        "- If your primary goal is to show **equivalence**, consider **TOST** (two one-sided tests) with a pre-specified "
        "equivalence margin. That is a different question than 'difference'.\n"
        "- If runtimes are high, reduce the **simulation draws** in the sidebar, or analyze fewer tests at once."
    )
