# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module 03: Exploratory data analysis and quality diagnostics
#
# **Curriculum link:** `stats.md` -> Topic 3, equations (3.1)-(3.7)
#
# ## What you will learn
#
# 1. The ECDF and the DKW bound (3.1)-(3.2): how little small samples tell you.
# 2. Robust spread: MAD (3.3)-(3.4) - and why it is the scverse QC default (3.5).
# 3. **The mean-variance plot (3.6) is how you choose a likelihood.** This is the
#    single most valuable EDA plot in omics.
# 4. Why "test for normality, then pick a test" is a bad procedure (3.7).
# 5. The standard omics QC battery.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "03_exploratory_data_analysis"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1. The empirical CDF and how much it can be trusted
#
# $$\hat F_n(t)=\frac1n\sum_i \mathbb{1}\{y_i\le t\} \tag{3.1}$$
#
# $$\Pr\Big(\sup_t|\hat F_n(t)-F(t)|>\epsilon\Big)\le 2e^{-2n\epsilon^2} \tag{3.2}$$
#
# Equation (3.2) (Dvoretzky-Kiefer-Wolfowitz) gives a *uniform* confidence band.
# Inverting it at level $\alpha$ gives a half-width
# $\epsilon=\sqrt{\ln(2/\alpha)/(2n)}$.

# %%
header("1. ECDF and the DKW uniform confidence band (3.1)-(3.2)")

def ecdf(x):
    """stats.md eq. (3.1): return sorted values and their cumulative proportions."""
    xs = np.sort(x)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def dkw_halfwidth(n, alpha=0.05):
    """Inverting stats.md eq. (3.2): a 100(1-alpha)% uniform band half-width."""
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))


for n in [6, 12, 30, 100, 1000]:
    print(f"  n = {n:>5}  95% DKW band half-width = +/-{dkw_halfwidth(n):.3f}")

print("\nWith n=6 the true CDF could be anywhere within +/-0.55 of the observed")
print("one - the band is wider than the whole probability scale. That is why")
print("'the data look non-normal' is a weak basis for switching tests at small n.")

# %% [markdown]
# ## 2. Robust location and spread, eq. (3.3)-(3.5)
#
# $$\mathrm{MAD}=\operatorname{median}|y_i-\operatorname{median}(y)|, \qquad
#   \hat\sigma_{\mathrm{MAD}}=1.4826\times\mathrm{MAD}$$
#
# The constant $1.4826 = 1/\Phi^{-1}(0.75)$ makes MAD a consistent estimator of
# $\sigma$ under normality. Let us verify that constant rather than trust it.

# %%
header("2. MAD: the 1.4826 constant, and breakdown (3.3)-(3.4)")

print(f"  1 / Phi^-1(0.75) = 1 / {st.norm.ppf(0.75):.6f} = "
      f"{1/st.norm.ppf(0.75):.6f}")


def mad_sigma(x):
    """stats.md eq. (3.4): a consistent, 50%-breakdown estimator of sigma."""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


rng = np.random.default_rng(303)
clean = rng.normal(0.0, 1.0, 1000)
print(f"\n  clean N(0,1):   SD = {clean.std(ddof=1):.4f}   MAD-sigma = "
      f"{mad_sigma(clean):.4f}   (truth 1.0)")

# Now corrupt an increasing fraction and watch which estimator survives.
print(f"\n{'% corrupted':>12} {'SD':>10} {'MAD-sigma':>11}")
for frac in [0.0, 0.01, 0.05, 0.20, 0.45]:
    dirty = clean.copy()
    k = int(frac * len(dirty))
    if k:
        dirty[:k] = 1000.0                       # gross outliers
    print(f"{frac:>11.0%} {dirty.std(ddof=1):>10.2f} {mad_sigma(dirty):>11.3f}")

print("\nSD has breakdown point 0 (one bad value destroys it); MAD has 50%.")

# %% [markdown]
# ### The MAD-based outlier rule used in single-cell QC, eq. (3.5)

# %%
def mad_outliers(x, k=3.0):
    """Flag |y - median| > k * MAD-sigma. The scverse single-cell QC default."""
    med, s = np.median(x), mad_sigma(x)
    return np.abs(x - med) > k * s


rng = np.random.default_rng(304)
# Realistic single-cell QC metric: log10 of total counts per cell, with a
# small population of empty droplets and a few doublets.
good = rng.normal(3.5, 0.25, 1940)
empty = rng.normal(2.2, 0.20, 40)
doublet = rng.normal(4.4, 0.15, 20)
qc_metric = np.concatenate([good, empty, doublet])

for k in [3.0, 5.0]:
    flagged = mad_outliers(qc_metric, k)
    print(f"  k = {k}:  {flagged.sum():>4} / {len(qc_metric)} cells flagged "
          f"({flagged.mean():.1%})")

# %% [markdown]
# ## 3. The mean-variance relationship IS the assay's fingerprint, eq. (3.6)
#
# $$s_g^2\approx\bar y_g \Rightarrow \text{Poisson}; \quad
#   s_g^2\approx\bar y_g+\phi\bar y_g^2 \Rightarrow \text{NB}; \quad
#   s_g^2\approx\text{const} \Rightarrow \text{Gaussian-ready}$$
#
# We simulate three assays and *recover which family generated each* purely
# from the mean-variance plot. This is what `edgeR`/`DESeq2` do internally and
# what `voom` converts into precision weights.

# %%
header("3. Reading the likelihood off the mean-variance plot (3.6)")

rng = np.random.default_rng(305)
G, n_samp = 3000, 8
mus = np.exp(rng.normal(3.0, 1.6, G))            # gene mean counts

def nb_matrix(mu_vec, phi, n, rng):
    """G x n negative-binomial counts; phi=0 gives Poisson."""
    if phi <= 0:
        return rng.poisson(mu_vec[:, None], size=(len(mu_vec), n))
    size = 1.0 / phi
    p = size / (size + mu_vec[:, None])
    return rng.negative_binomial(size, p, size=(len(mu_vec), n))


assays = {
    "Technical replicates (Poisson)": nb_matrix(mus, 0.0, n_samp, rng),
    "Biological replicates (NB, phi=0.16)": nb_matrix(mus, 0.16, n_samp, rng),
    "Log-intensities (Gaussian)": rng.normal(
        np.log2(mus)[:, None], 0.35, size=(G, n_samp)),
}

print(f"{'assay':<40} {'slope of log(var)~log(mean)':>28}")
for name, mat in assays.items():
    m = mat.mean(axis=1)
    v = mat.var(axis=1, ddof=1)
    keep = (m > 1) & (v > 0)
    # A log-log regression: slope ~1 => Poisson, slope ~2 => NB-dominated,
    # slope ~0 => variance independent of mean (Gaussian after transformation).
    slope = np.polyfit(np.log(m[keep]), np.log(v[keep]), 1)[0]
    print(f"{name:<40} {slope:>28.3f}")

print("\nInterpretation:")
print("  slope ~ 1  -> Var = mean          -> Poisson   (technical only)")
print("  slope ~ 2  -> Var = phi*mean^2    -> NB        (biological variability)")
print("  slope ~ 0  -> Var independent     -> Gaussian  (already stabilised)")

# Estimate the NB dispersion directly from the moment relation (4.9):
nb = assays["Biological replicates (NB, phi=0.16)"]
m, v = nb.mean(axis=1), nb.var(axis=1, ddof=1)
keep = m > 20                                    # need decent counts for stability
phi_hat = np.median((v[keep] - m[keep]) / m[keep] ** 2)
print(f"\nMethod-of-moments dispersion from Var = mu + phi*mu^2 (eq. 4.9):")
print(f"  phi_hat = {phi_hat:.4f}   (true 0.16)")
print(f"  BCV = sqrt(phi) = {np.sqrt(phi_hat):.3f}  <- edgeR reports this")

# %% [markdown]
# ## 4. Why "test for normality, then choose a test" fails, eq. (3.7)
#
# $$g_1=\frac{m_3}{m_2^{3/2}}, \qquad \operatorname{Var}(g_1)\approx 6/n$$
#
# Two failures, in opposite directions:
# * At small $n$ the normality test has **no power**: exactly when normality
#   matters most for exactness.
# * At large $n$ it rejects trivial departures: exactly when the CLT (4.5)
#   already protects you.
#
# And the two-stage procedure (pre-test, then choose) distorts the Type I error
# rate of whatever test you end up running.

# %%
header("4. The normality-pretest trap (3.7)")

rng = np.random.default_rng(306)
print(f"{'n':>6} {'SE(skew)=sqrt(6/n)':>20} {'P(Shapiro rejects | TRUE normal)':>34}")
for n in [8, 20, 50, 200, 2000]:
    rejects = np.mean([st.shapiro(rng.normal(size=n)).pvalue < 0.05
                       for _ in range(600)])
    print(f"{n:>6} {np.sqrt(6/n):>20.3f} {rejects:>33.1%}")

print("\nNow a genuinely skewed distribution (log-normal), same test:")
print(f"{'n':>6} {'P(Shapiro rejects | NOT normal)':>34}")
for n in [8, 20, 50, 200]:
    power = np.mean([st.shapiro(rng.lognormal(0, 0.75, n)).pvalue < 0.05
                     for _ in range(600)])
    print(f"{n:>6} {power:>33.1%}")

print("\nAt n=8 the test misses clear log-normality most of the time. Choose the")
print("model from the MEASUREMENT (Topic 2) and the mean-variance plot (3.6),")
print("not from a normality test.")

# %% [markdown]
# ## 5. The standard omics QC battery
#
# Run these on every dataset before modelling anything.

# %%
header("5. QC battery on a simulated multi-batch RNA-seq experiment")

rng = np.random.default_rng(307)
G, n = 4000, 12
meta = pd.DataFrame({
    "sample": [f"S{i:02d}" for i in range(n)],
    "condition": ["ctrl"] * 6 + ["trt"] * 6,
    # Batches are BALANCED across conditions (the good design of Module 01),
    # so any batch signal we see is pure technical variation, not confounding.
    "batch": ["b1", "b1", "b2", "b2", "b3", "b3"] * 2,
}).set_index("sample")

base = np.exp(rng.normal(3.0, 1.5, G))
depth = rng.uniform(0.6, 1.6, n)                      # library-size variation
batch_shift = {"b1": 0.0, "b2": 0.35, "b3": -0.30}    # technical offsets
mu = np.outer(base, depth)
for j, b in enumerate(meta["batch"]):
    mu[:, j] *= np.exp(batch_shift[b])
# A real effect in 200 genes:
mu[:200, 6:] *= 2.0
counts = rng.negative_binomial(1 / 0.16, (1 / 0.16) / ((1 / 0.16) + mu))
counts = pd.DataFrame(counts, index=[f"G{i:04d}" for i in range(G)],
                      columns=meta.index)

qc = pd.DataFrame({
    "library_size": counts.sum(axis=0),
    "genes_detected": (counts > 0).sum(axis=0),
    "pct_top50": counts.apply(
        lambda c: 100 * c.nlargest(50).sum() / c.sum(), axis=0),
}).join(meta)
print(qc.round(1).to_string())

print("\nPer-batch library-size medians (drift check):")
print(qc.groupby("batch")["library_size"].median().round(0).to_string())

# Sample-sample correlation on a variance-stabilised scale detects swaps.
logcpm = np.log2(counts / counts.sum(axis=0) * 1e6 + 1)
corr = logcpm.corr(method="spearman")
np.fill_diagonal(corr.values, np.nan)
print(f"\nSample-sample Spearman correlation: min={np.nanmin(corr.values):.3f}, "
      f"median={np.nanmedian(corr.values):.3f}")
print("A sample correlating much worse with everything else is an outlier or a swap.")

# PC x metadata association (stats.md eq. 17.6): the key batch diagnostic.
X = (logcpm.T - logcpm.T.mean(axis=0)).values
U, S, Vt = np.linalg.svd(X, full_matrices=False)
scores = U * S
pve = S**2 / np.sum(S**2)
print("\nPC x metadata association (eq. 17.6) - R^2 of each PC on each factor:")
print(f"{'PC':>4} {'PVE':>7} {'condition':>11} {'batch':>8} {'log lib size':>13}")
for k in range(3):
    z = scores[:, k]
    r2_cond = np.corrcoef(z, (meta["condition"] == "trt").astype(float))[0, 1] ** 2
    bd = pd.get_dummies(meta["batch"], drop_first=True).astype(float).values
    Xb = np.column_stack([np.ones(n), bd])
    resid = z - Xb @ np.linalg.lstsq(Xb, z, rcond=None)[0]
    r2_batch = 1 - resid.var() / z.var()
    r2_lib = np.corrcoef(z, np.log(qc["library_size"]))[0, 1] ** 2
    print(f"{k+1:>4} {pve[k]:>7.1%} {r2_cond:>11.3f} {r2_batch:>8.3f} {r2_lib:>13.3f}")

print("\nIf PC1 is better explained by batch or library size than by condition,")
print("you have a technical problem, not a discovery (stats.md Topic 19).")

# %% [markdown]
# ## 6. Figure: the QC panel

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# (a) ECDF with DKW band
rng = np.random.default_rng(308)
small = rng.normal(0, 1, 12)
xs, ys = ecdf(small)
eps = dkw_halfwidth(12)
axes[0, 0].step(xs, ys, where="post", label="ECDF (n=12)")
axes[0, 0].fill_between(xs, np.clip(ys - eps, 0, 1), np.clip(ys + eps, 0, 1),
                        step="post", alpha=0.25, label="95% DKW band")
grid = np.linspace(-3, 3, 200)
axes[0, 0].plot(grid, st.norm.cdf(grid), "k--", lw=1, label="true N(0,1)")
axes[0, 0].set_title("Eq. (3.1)-(3.2): ECDF uncertainty")
axes[0, 0].legend(fontsize=7)

# (b) MAD-based QC thresholds
axes[0, 1].hist(qc_metric, bins=60, color="steelblue")
med, s = np.median(qc_metric), mad_sigma(qc_metric)
for k, c in [(3, "orange"), (5, "red")]:
    axes[0, 1].axvline(med - k * s, color=c, ls="--", label=f"{k} MAD")
    axes[0, 1].axvline(med + k * s, color=c, ls="--")
axes[0, 1].set_xlabel("log10 total counts per cell")
axes[0, 1].set_title("Eq. (3.5): MAD outlier rule")
axes[0, 1].legend(fontsize=7)

# (c) mean-variance, the key plot
for (name, mat), colour in zip(list(assays.items())[:2], ["C0", "C3"]):
    m = mat.mean(axis=1)
    v = mat.var(axis=1, ddof=1)
    k = (m > 1) & (v > 0)
    axes[0, 2].loglog(m[k], v[k], ".", ms=1.5, alpha=0.3, color=colour,
                      label=name.split(" (")[0])
lim = np.logspace(0, 3.5, 50)
axes[0, 2].loglog(lim, lim, "k--", lw=1, label="Var = mean (Poisson)")
axes[0, 2].loglog(lim, lim + 0.16 * lim**2, "k:", lw=1.2, label="Var = mu+0.16mu^2")
axes[0, 2].set_xlabel("mean")
axes[0, 2].set_ylabel("variance")
axes[0, 2].set_title("Eq. (3.6): the assay's fingerprint")
axes[0, 2].legend(fontsize=6)

# (d) library sizes by batch
for i, b in enumerate(sorted(meta["batch"].unique())):
    vals = qc.loc[qc["batch"] == b, "library_size"]
    axes[1, 0].scatter([i] * len(vals), vals, s=25)
axes[1, 0].set_xticks(range(3))
axes[1, 0].set_xticklabels(sorted(meta["batch"].unique()))
axes[1, 0].set_xlabel("batch")
axes[1, 0].set_ylabel("library size")
axes[1, 0].set_title("Library size drift by batch")

# (e) detection vs depth
axes[1, 1].scatter(qc["library_size"], qc["genes_detected"], s=30)
axes[1, 1].set_xlabel("library size")
axes[1, 1].set_ylabel("genes detected")
axes[1, 1].set_title("Detection is depth-driven")

# (f) PCA coloured by batch
for b, mk in zip(sorted(meta["batch"].unique()), ["o", "s", "^"]):
    idx = (meta["batch"] == b).values
    axes[1, 2].scatter(scores[idx, 0], scores[idx, 1], marker=mk, s=60, label=b)
axes[1, 2].set_xlabel(f"PC1 ({pve[0]:.0%})")
axes[1, 2].set_ylabel(f"PC2 ({pve[1]:.0%})")
axes[1, 2].set_title("PCA coloured by BATCH, not condition")
axes[1, 2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "qc_panel.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/qc_panel.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 3)
#
# 1. Explore to diagnose data quality and choose a **likelihood**, not to choose
#    a **hypothesis**. Choosing the hypothesis after seeing the outcome is
#    HARKing and invalidates the p-value you later report.
# 2. Never delete an outlier without a documented, outcome-independent reason.
# 3. Any preprocessing that uses the outcome must be abandoned or moved inside
#    the resampling loop (Topic 31).
#
# ## Self-check
#
# * Your mean-variance log-log slope is 1.9. Which likelihood? *(Negative
#   binomial: the quadratic term dominates.)*
# * Shapiro-Wilk gives p = 0.002 on n = 5000 log-intensities. Switch to a rank
#   test? *(No. At n = 5000 the CLT covers the mean; the rejection reflects a
#   trivial departure.)*
#
# **Next:** `04_probability_and_sampling.py`
