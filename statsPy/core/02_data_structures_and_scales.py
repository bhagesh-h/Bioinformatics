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
# # Module 02: Data structures, measurement scales, and transformations
#
# **Curriculum link:** `stats.md` -> Topic 2, equations (2.1)-(2.7)
#
# ## What you will learn
#
# 1. The assay triple $(\mathbf{Y},\mathbf{R},\mathbf{C})$ and why the alignment
#    contract (2.1) must be *asserted*, not assumed.
# 2. How the support of a measurement determines its likelihood.
# 3. The three transformations that matter: log+c (2.2), logit/M (2.4),
#    arcsinh (2.5): and exactly what each does to the variance.
# 4. Why compositional closure (2.6) forces negative correlations (2.7).

# %%
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "02_data_structures_and_scales"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1. The assay triple and the alignment contract
#
# $$\text{colnames}(\mathbf{Y}) = \text{rownames}(\mathbf{C}), \qquad
#   \text{rownames}(\mathbf{Y}) = \text{rownames}(\mathbf{R}) \tag{2.1}$$
#
# Bioconductor's `SummarizedExperiment` exists to enforce (2.1) automatically
# under subsetting and reordering. In Python (and in plain R) you must enforce
# it yourself. A violated contract produces code that **runs without error and
# is entirely wrong**: the worst possible failure mode.

# %%
header("1. The assay triple, and asserting the alignment contract (2.1)")


class AssayData:
    """A minimal SummarizedExperiment: assay + row metadata + column metadata.

    The point is not the class, it is the ``_validate`` call: alignment is
    checked on construction and after every subsetting operation.
    """

    def __init__(self, assay, row_data, col_data):
        self.assay = assay          # G x n DataFrame (features x samples)
        self.row_data = row_data    # G x q  feature annotation
        self.col_data = col_data    # n x r  sample annotation
        self._validate()

    def _validate(self):
        """stats.md eq. (2.1). Fail loudly and immediately."""
        if not self.assay.columns.equals(self.col_data.index):
            raise ValueError(
                "ALIGNMENT VIOLATION: assay columns != col_data index.\n"
                f"  assay columns : {list(self.assay.columns)[:5]}...\n"
                f"  col_data index: {list(self.col_data.index)[:5]}..."
            )
        if not self.assay.index.equals(self.row_data.index):
            raise ValueError("ALIGNMENT VIOLATION: assay rows != row_data index.")

    def subset_samples(self, keep):
        """Subset columns - and carry the metadata along, which is the whole point."""
        return AssayData(self.assay.loc[:, keep],
                         self.row_data,
                         self.col_data.loc[keep])

    def __repr__(self):
        return (f"AssayData: {self.assay.shape[0]} features x "
                f"{self.assay.shape[1]} samples")


rng = np.random.default_rng(11)
genes = [f"GENE{i:03d}" for i in range(6)]
samples = [f"S{i}" for i in range(8)]

se = AssayData(
    assay=pd.DataFrame(rng.poisson(50, size=(6, 8)), index=genes, columns=samples),
    row_data=pd.DataFrame({"chrom": ["chr1"] * 3 + ["chr2"] * 3,
                           "length": rng.integers(500, 5000, 6)}, index=genes),
    col_data=pd.DataFrame({"condition": ["ctrl"] * 4 + ["trt"] * 4,
                           "batch": ["b1", "b1", "b2", "b2"] * 2,
                           "donor": [f"D{i//2}" for i in range(8)]}, index=samples),
)
print(se)
print(se.col_data)

# Subsetting keeps the contract:
sub = se.subset_samples(["S0", "S2", "S5"])
print(f"\nAfter subsetting: {sub}")
print(f"  conditions carried along: {list(sub.col_data['condition'])}")

# %% [markdown]
# ### The failure this prevents
#
# The single most common silent bug in bioinformatics: someone sorts the
# metadata table (by donor, by condition, in Excel) while the assay matrix keeps
# its original column order. Every downstream label is then attached to the
# wrong sample.

# %%
print("\nNow simulate the classic bug - metadata sorted, assay not:")
scrambled_meta = se.col_data.sort_values("condition", ascending=False)
try:
    AssayData(se.assay, se.row_data, scrambled_meta)
    print("  ...no error raised (this would be the silent disaster)")
except ValueError as exc:
    print("  Caught before it could do damage:")
    print("   ", str(exc).splitlines()[0])

print("\nRULE: assert eq. (2.1) at the top of every script and after every join.")

# %% [markdown]
# ## 2. Support determines the likelihood
#
# Two matrices of identical shape can require entirely different models. What
# does one number physically represent?

# %%
header("2. Support -> natural model")
scale_table = pd.DataFrame([
    ("Read counts",            "{0,1,2,...}",        "Poisson / negative binomial", "RNA-seq, ATAC peaks"),
    ("Counts out of N",        "{0,...,N}",          "Binomial / beta-binomial",    "Methylated reads, +cells"),
    ("Log intensity",          "R",                  "Normal",                      "Proteomics, microarray"),
    ("Methylation beta",       "[0,1]",              "Beta / logit-normal",         "450K/EPIC arrays"),
    ("Relative abundance",     "simplex S^(D-1)",    "Log-ratio / Dirichlet-mult.", "16S microbiome"),
    ("Time to event",          "[0,inf) + censoring","Survival / hazard model",     "Relapse, death"),
    ("Pathology grade",        "ordered categories", "Ordinal (proportional odds)", "Histology score"),
], columns=["Measurement", "Support", "Natural model", "Example assay"])
print(scale_table.to_string(index=False))

# %% [markdown]
# ## 3. Transformation 1: log with a pseudocount, eq. (2.2)-(2.3)
#
# $$y \mapsto \log_2(y+c), \qquad
#   \operatorname{Var}\big(\log_2(Y+c)\big) \approx
#   \frac{1}{(\ln 2)^2}\cdot\frac{\mu+\phi\mu^2}{(\mu+c)^2}$$
#
# The *purpose* of the log is variance stabilisation: for large $\mu$ the
# expression above tends to the constant $\phi/(\ln 2)^2$. But for small $\mu$
# it is dominated by $c$, which is why the pseudocount and the low-count filter
# are the same decision.

# %%
header("3. log(y + c): the delta-method prediction (2.3) vs simulation")

def nb_sample(mu, phi, size, rng):
    """Draw negative-binomial counts with mean mu and variance mu + phi*mu^2.

    numpy parameterises NB by (n successes, p). With dispersion phi the
    standard reparameterisation is n = 1/phi and p = n/(n+mu).
    """
    n = 1.0 / phi
    p = n / (n + mu)
    return rng.negative_binomial(n, p, size=size)


def delta_method_var_log2(mu, phi, c):
    """stats.md eq. (2.3)."""
    return (mu + phi * mu**2) / ((mu + c) ** 2 * np.log(2) ** 2)


rng = np.random.default_rng(21)
PHI = 0.16                     # typical human bulk RNA-seq dispersion (BCV 0.4)
print(f"dispersion phi = {PHI}  (asymptote phi/(ln2)^2 = "
      f"{PHI/np.log(2)**2:.4f})\n")
print(f"{'mu':>8} {'c':>5} {'delta (2.3)':>13} {'simulated':>11} {'ratio':>7}")
for mu in [1, 5, 20, 100, 1000]:
    for c in [0.5, 1.0, 8.0]:
        y = nb_sample(mu, PHI, 200_000, rng)
        emp = np.var(np.log2(y + c), ddof=1)
        theo = delta_method_var_log2(mu, PHI, c)
        print(f"{mu:>8} {c:>5} {theo:>13.4f} {emp:>11.4f} {emp/theo:>7.2f}")

print("\nRead the c=0.5 rows: at mu=1 the variance is ~10x the asymptote.")
print("A small pseudocount hugely inflates the variance of low-count features,")
print("which is exactly why you filter them (stats.md Topic 8, 20).")

# %% [markdown]
# ## 4. Transformation 2: logit / M-value, eq. (2.4)
#
# $$M = \log_2\frac{\beta}{1-\beta}$$
#
# Methylation beta values are bounded in $[0,1]$, so their variance is
# *compressed near the boundaries*: a probe at $\beta=0.02$ simply cannot vary
# as much as one at $\beta=0.5$. That is heteroscedasticity, and it breaks the
# constant-variance assumption of linear models. M-values fix it.

# %%
header("4. beta vs M-values: heteroscedasticity (2.4)")

def beta_to_m(beta, eps=1e-6):
    """stats.md eq. (2.4). Clamp to avoid +/-inf at exactly 0 or 1."""
    beta = np.clip(beta, eps, 1 - eps)
    return np.log2(beta / (1 - beta))


def m_to_beta(m):
    """Inverse of (2.4) - use this to report an M-scale effect as a beta."""
    return 2.0**m / (2.0**m + 1.0)


rng = np.random.default_rng(31)
# Simulate probes at different mean methylation, each with the SAME underlying
# biological variability on the logit scale.
print(f"{'true mean beta':>15} {'SD(beta)':>10} {'SD(M)':>8}")
for target in [0.02, 0.10, 0.30, 0.50, 0.70, 0.95]:
    m_true = np.log2(target / (1 - target))
    m_draws = rng.normal(m_true, 0.5, 50_000)        # constant SD on M scale
    beta_draws = m_to_beta(m_draws)
    print(f"{target:>15.2f} {beta_draws.std(ddof=1):>10.4f} "
          f"{m_draws.std(ddof=1):>8.4f}")

print("\nSD on the beta scale varies ~4x across the range; SD on the M scale is")
print("constant by construction. MODEL ON M, REPORT ON BETA (stats.md Topic 23).")
print(f"\nRound-trip check: M = 1.5 corresponds to beta = {m_to_beta(1.5):.4f}")

# %% [markdown]
# ## 5. Transformation 3: arcsinh, eq. (2.5)
#
# $$y \mapsto \operatorname{arcsinh}(y/c) = \ln\!\left(\frac{y}{c}+\sqrt{\frac{y^2}{c^2}+1}\right)$$
#
# Cytometry intensities span decades **and can be negative** after compensation,
# so `log` is unusable. arcsinh is linear near zero and logarithmic far from it;
# the cofactor $c$ sets the width of the linear region.

# %%
header("5. arcsinh with cofactor c (2.5)")

def arcsinh_transform(y, cofactor):
    """stats.md eq. (2.5). np.arcsinh handles negatives natively."""
    return np.arcsinh(y / cofactor)


test_values = np.array([-50.0, -5.0, 0.0, 1.0, 5.0, 50.0, 500.0, 5000.0])
print(f"{'raw':>10} {'asinh c=5':>12} {'asinh c=150':>13} {'log10(y) ':>11}")
for v in test_values:
    log_val = f"{np.log10(v):.3f}" if v > 0 else "undefined"
    print(f"{v:>10.1f} {arcsinh_transform(v, 5):>12.3f} "
          f"{arcsinh_transform(v, 150):>13.3f} {log_val:>11}")

print("\nc=5 is the mass-cytometry (CyTOF) convention; c=150 is typical for")
print("fluorescence flow. Note arcsinh handles the negative values that")
print("compensation produces, where log10 is undefined.")

# %% [markdown]
# ## 6. Compositional closure, eq. (2.6)-(2.7)
#
# $$\mathcal{S}^{D-1}=\Big\{\mathbf{x}: x_d>0,\ \sum_d x_d = 1\Big\}, \qquad
#   \sum_{d}\operatorname{Cov}(x_c,x_d)=0$$
#
# This is arithmetic, not ecology. We generate **completely independent**
# absolute abundances, close them to proportions, and watch correlations appear
# out of nothing.

# %%
header("6. Closure manufactures correlations (2.6)-(2.7)")

rng = np.random.default_rng(41)
D, n = 5, 3000

# Absolute abundances: independent by construction.
absolute = np.column_stack([
    rng.lognormal(mean=m, sigma=0.6, size=n) for m in [3.0, 2.5, 2.0, 1.5, 1.0]
])
relative = absolute / absolute.sum(axis=1, keepdims=True)   # closure (2.6)

corr_abs = np.corrcoef(absolute, rowvar=False)
corr_rel = np.corrcoef(relative, rowvar=False)

off = ~np.eye(D, dtype=bool)
print("Correlation among ABSOLUTE abundances (truth: independent):")
print(f"  max |off-diagonal r| = {np.abs(corr_abs[off]).max():.3f}")
print("\nCorrelation among RELATIVE abundances (after closure):")
print(f"  max |off-diagonal r| = {np.abs(corr_rel[off]).max():.3f}")
print(f"  mean off-diagonal r  = {corr_rel[off].mean():+.3f}  (negative!)")
print("\nRow sums of the relative covariance matrix - eq. (2.7) predicts 0:")
cov_rel = np.cov(relative, rowvar=False)
print("  ", np.round(cov_rel.sum(axis=1), 12))
print("\nEvery correlation you see in the closed data was manufactured by the")
print("constraint. This is why microbiome analysis needs log-ratios (Topic 26).")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) variance stabilisation by log
rng = np.random.default_rng(51)
mus = np.logspace(0, 3.5, 40)
raw_sd, log_sd = [], []
for mu in mus:
    y = nb_sample(mu, PHI, 20000, rng)
    raw_sd.append(y.std(ddof=1))
    log_sd.append(np.log2(y + 1).std(ddof=1))
axes[0].loglog(mus, raw_sd, "o-", ms=3, label="SD(raw counts)")
axes[0].loglog(mus, log_sd, "s-", ms=3, label="SD(log2(y+1))")
axes[0].axhline(np.sqrt(PHI) / np.log(2), color="k", ls=":",
                label="asymptote $\\sqrt{\\phi}/\\ln 2$")
axes[0].set_xlabel("mean count $\\mu$")
axes[0].set_ylabel("standard deviation")
axes[0].set_title("Eq. (2.2)-(2.3): log stabilises variance")
axes[0].legend(fontsize=7)

# (b) beta vs M heteroscedasticity
betas = np.linspace(0.01, 0.99, 100)
sd_beta = [m_to_beta(np.random.default_rng(1).normal(
    np.log2(b / (1 - b)), 0.5, 4000)).std(ddof=1) for b in betas]
axes[1].plot(betas, sd_beta, lw=2)
axes[1].axhline(0.5, color="C1", ls="--", label="SD on M scale (constant)")
axes[1].set_xlabel("mean methylation $\\beta$")
axes[1].set_ylabel("SD on the beta scale")
axes[1].set_title("Eq. (2.4): beta values are heteroscedastic")
axes[1].legend(fontsize=7)

# (c) arcsinh vs log
yy = np.linspace(-100, 5000, 2000)
axes[2].plot(yy, np.arcsinh(yy / 5), label="arcsinh, c=5")
axes[2].plot(yy, np.arcsinh(yy / 150), label="arcsinh, c=150")
with np.errstate(invalid="ignore"):
    axes[2].plot(yy, np.log(np.where(yy > 0, yy, np.nan)), label="ln(y)")
axes[2].axvline(0, color="k", lw=0.8)
axes[2].set_xscale("symlog", linthresh=10)
axes[2].set_xlabel("raw intensity")
axes[2].set_ylabel("transformed")
axes[2].set_title("Eq. (2.5): arcsinh handles negatives")
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "transformations.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/transformations.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 2)
#
# 1. Ask what the number *counts or measures* before choosing a test.
# 2. Keep counts as counts for inference; use transformed values for
#    visualisation, distances, and PCA.
# 3. Model on the scale where variance is stable; report on the scale a
#    biologist can interpret (beta vs M is the canonical example).
# 4. Assert eq. (2.1) programmatically. Assert, do not assume.
#
# ## Self-check
#
# * Why can't you feed TPM values into a negative-binomial model?
#   *(TPM has already divided out library size; the count information that
#   determines the variance is gone: Topic 20.)*
# * A probe has $\beta = 0.97$. Why is a linear model on $\beta$ a bad idea?
#   *(Boundary compression: its variance is structurally tiny: eq. (2.4).)*
#
# **Next:** `03_exploratory_data_analysis.py`
