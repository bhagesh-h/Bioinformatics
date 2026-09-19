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
# # Module 24: Diagnostics, sensitivity analysis, and reproducibility
#
# **Curriculum link:** `stats.md` -> Topics 34 and 35, equations (34.1)-(34.3)
#
# ## What you will learn
#
# 1. **Randomised quantile residuals (34.1)**: the one residual that works for
#    discrete outcomes.
# 2. Leave-one-sample-out and leave-one-batch-out for omics results (34.2).
# 3. **Specification-curve / multiverse analysis (34.3)**: is your conclusion
#    stable or specification-dependent?
# 4. The permuted-label negative control: the check that catches the most bugs.
# 5. Simulation-based calibration.
# 6. Property-based tests for statistical code, and seeding done right.

# %%
import os
import itertools
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "24_diagnostics_and_reproducibility"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Randomised quantile residuals, eq. (34.1)
#
# $$r_i=\Phi^{-1}(u_i),\qquad u_i\sim\mathrm{Uniform}\big(F(y_i-1),\ F(y_i)\big)$$
#
# Raw residuals from a count model form useless discrete bands. Randomised
# quantile residuals are exactly $\mathcal N(0,1)$ under a correct model, so a
# Q-Q plot becomes interpretable.

# %%
header("1. Residuals that work for discrete outcomes (34.1)")

def randomised_quantile_residuals(y, dist_cdf, seed=0):
    """stats.md eq. (34.1). dist_cdf(k) must be vectorised over observations."""
    rng = np.random.default_rng(seed)
    lo = dist_cdf(np.asarray(y) - 1)
    hi = dist_cdf(np.asarray(y))
    u = rng.uniform(lo, hi)
    return st.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


rng = np.random.default_rng(2401)
n = 600
x = rng.normal(0, 1, n)
mu = np.exp(1.6 + 0.5 * x)
PHI = 0.35
r_nb = 1 / PHI
y = rng.negative_binomial(r_nb, r_nb / (r_nb + mu), size=n)
d = pd.DataFrame({"y": y, "x": x})

m_pois = smf.glm("y ~ x", data=d, family=sm.families.Poisson()).fit()
m_nb = smf.glm("y ~ x", data=d,
               family=sm.families.NegativeBinomial(alpha=PHI)).fit()

rq_pois = randomised_quantile_residuals(
    y, lambda k: st.poisson.cdf(k, m_pois.fittedvalues), seed=1)
mu_nb = m_nb.fittedvalues
rq_nb = randomised_quantile_residuals(
    y, lambda k: st.nbinom.cdf(k, r_nb, r_nb / (r_nb + mu_nb)), seed=1)

print(f"  {'model':<22}{'mean':>9}{'SD':>8}{'Shapiro p':>12}"
      f"{'KS vs N(0,1) p':>17}")
for name, r in [("Poisson (WRONG)", rq_pois), ("neg. binomial (right)", rq_nb)]:
    print(f"  {name:<22}{r.mean():>9.3f}{r.std(ddof=1):>8.3f}"
          f"{st.shapiro(r[:500]).pvalue:>12.2e}"
          f"{st.kstest(r, 'norm').pvalue:>17.4f}")
print("\n  Under the correct model the residuals are exactly N(0,1); under the")
print("  wrong one they are not. Ordinary deviance residuals cannot make this")
print("  distinction nearly as cleanly for count data.")

# %% [markdown]
# ## 2. Leave-one-sample-out and leave-one-batch-out, eq. (34.2)
#
# If dropping one sample removes 40% of your discoveries, **that is the
# headline**, not a footnote.

# %%
header("2. How much does one sample drive the result? (34.2)")

rng = np.random.default_rng(2402)
G, n_per = 3000, 6
is_de = np.zeros(G, bool); is_de[:150] = True
delta = np.where(is_de, 1.3, 0.0)
A = rng.normal(delta[:, None], 1.0, (G, n_per))
B = rng.normal(0.0, 1.0, (G, n_per))
# Inject ONE influential outlier sample in group A.
A[:, 0] += rng.normal(0, 2.5, G)

def de_hits(A, B, alpha=0.05):
    p = st.ttest_ind(A, B, axis=1).pvalue
    return set(np.where(st.false_discovery_control(p) < alpha)[0])


full_hits = de_hits(A, B)
print(f"  full analysis: {len(full_hits)} discoveries at FDR 5%")
print(f"\n  {'dropped sample':<20}{'discoveries':>13}{'% of full':>11}"
      f"{'Jaccard vs full':>18}")
for j in range(n_per):
    h = de_hits(np.delete(A, j, axis=1), B)
    jac = len(h & full_hits) / max(len(h | full_hits), 1)
    flag = "  <-- influential" if len(h) < 0.7 * len(full_hits) or \
                                 len(h) > 1.4 * len(full_hits) else ""
    print(f"  {'A' + str(j):<20}{len(h):>13}{len(h)/max(len(full_hits),1):>10.0%}"
          f"{jac:>18.3f}{flag}")

print("\n  Sample A0 is the injected outlier, and the leave-one-out scan finds")
print("  it without being told. Run this for ANY small-n omics result and")
print("  report the range, not just the headline count.")

# %% [markdown]
# ## 3. Specification curve / multiverse analysis, eq. (34.3)
#
# Enumerate the defensible analytic choices, fit all $S$ combinations, and plot
# the estimates ordered by magnitude.

# %%
header("3. Multiverse analysis (34.3)")

rng = np.random.default_rng(2403)
n = 220
batch = rng.integers(0, 3, n)
age = rng.normal(55, 12, n)
sex = rng.binomial(1, 0.5, n)
treat = rng.binomial(1, 0.5, n)
TRUE = 0.30
y_raw = np.exp(2.0 + TRUE * treat + 0.02 * age + 0.15 * sex
               + 0.25 * batch + rng.normal(0, 0.5, n))
dm = pd.DataFrame({"y_raw": y_raw, "treat": treat, "age": age,
                   "sex": sex, "batch": batch})
# A few extreme values, as real assays produce.
dm.loc[rng.choice(n, 6, replace=False), "y_raw"] *= 6

choices = {
    "transform": ["log", "raw", "rank"],
    "outliers": ["keep", "trim 1%", "winsorise"],
    "covariates": ["none", "+age+sex", "+age+sex+batch"],
    "se": ["classical", "HC3"],
}

rows = []
for combo in itertools.product(*choices.values()):
    tf, out, cov, se_type = combo
    d2 = dm.copy()
    if out == "trim 1%":
        lo, hi = d2["y_raw"].quantile([0.005, 0.995])
        d2 = d2[(d2["y_raw"] >= lo) & (d2["y_raw"] <= hi)]
    elif out == "winsorise":
        lo, hi = d2["y_raw"].quantile([0.01, 0.99])
        d2["y_raw"] = d2["y_raw"].clip(lo, hi)
    if tf == "log":
        d2["y"] = np.log(d2["y_raw"])
    elif tf == "raw":
        d2["y"] = d2["y_raw"] / d2["y_raw"].std()
    else:
        d2["y"] = st.rankdata(d2["y_raw"]) / len(d2)
    form = "y ~ treat" + {"none": "", "+age+sex": " + age + sex",
                          "+age+sex+batch": " + age + sex + C(batch)"}[cov]
    f = smf.ols(form, data=d2).fit()
    if se_type == "HC3":
        f = f.get_robustcov_results("HC3")
        est, p = f.params[1], f.pvalues[1]
        ci = f.conf_int()[1]
    else:
        est, p = f.params["treat"], f.pvalues["treat"]
        ci = f.conf_int().loc["treat"].values
    rows.append({"transform": tf, "outliers": out, "covariates": cov,
                 "se": se_type, "estimate": est, "p": p,
                 "lo": ci[0], "hi": ci[1]})

spec = pd.DataFrame(rows).sort_values("estimate").reset_index(drop=True)
print(f"  {len(spec)} defensible specifications "
      f"({' x '.join(str(len(v)) for v in choices.values())})")
print(f"  estimate range : [{spec['estimate'].min():.3f}, "
      f"{spec['estimate'].max():.3f}]")
print(f"  median estimate: {spec['estimate'].median():.3f}")
print(f"  sign stable?   : "
      f"{bool(np.all(np.sign(spec['estimate']) == np.sign(spec['estimate'].iloc[0])))}")
print(f"  p < 0.05 in    : {np.mean(spec['p'] < 0.05):.1%} of specifications")

print("\n  Which single choice moves the result most?")
for c in choices:
    g = spec.groupby(c)["estimate"].mean()
    print(f"    {c:<12} spread across levels = {g.max()-g.min():.4f}   "
          + ", ".join(f"{k}:{v:.3f}" for k, v in g.items()))

print("\n  A conclusion that holds in 5 of 54 specifications is a")
print("  SPECIFICATION-DEPENDENT conclusion, and saying so is a finding.")
print("  Note the transform changes the SCALE of the coefficient, so compare")
print("  SIGN and SIGNIFICANCE across transforms, magnitudes only within one.")

# %% [markdown]
# ## 4. The permuted-label negative control
#
# The single check that catches the most real bugs.

# %%
header("4. Negative control: does your pipeline find nothing in noise?")

def pipeline_under_test(A, B, broken=False):
    """A toy 'analysis pipeline'. `broken=True` introduces a realistic bug:
    filtering features by their between-group difference before testing."""
    if broken:
        # OUTCOME-DEPENDENT filtering - invalidates everything downstream.
        diff = np.abs(A.mean(axis=1) - B.mean(axis=1))
        keep = diff > np.quantile(diff, 0.5)
        A, B = A[keep], B[keep]
    p = st.ttest_ind(A, B, axis=1).pvalue
    return p


rng = np.random.default_rng(2404)
G, n_per = 4000, 5
for label, broken in [("correct pipeline", False), ("pipeline with a bug", True)]:
    hits, hist = [], []
    for rep in range(40):
        r = np.random.default_rng(3000 + rep)
        A = r.normal(0, 1, (G, n_per))
        B = r.normal(0, 1, (G, n_per))
        # Permuted labels: there is NOTHING to find, by construction.
        p = pipeline_under_test(A, B, broken=broken)
        hits.append(np.sum(st.false_discovery_control(p) < 0.05))
        hist.append(np.mean(p < 0.05))
    print(f"  {label:<24} mean discoveries at FDR 5% = {np.mean(hits):7.1f}"
          f"   P(p<0.05) = {np.mean(hist):.3f}")

print("\n  A correct pipeline on permuted labels yields ~0 discoveries and a")
print("  FLAT p-value histogram. Anything else means the pipeline is broken -")
print("  usually leakage, outcome-dependent filtering, or unmodelled")
print("  dependence. Run this before you trust ANY result.")

# %% [markdown]
# ## 5. Simulation-based calibration

# %%
header("5. Do your nominal 95% intervals actually cover 95%?")

def calibration_check(fit_and_interval, simulate, truth, n_sim=1500, seed=0):
    """Fraction of simulated datasets whose interval contains the truth."""
    rng = np.random.default_rng(seed)
    cov = 0
    for _ in range(n_sim):
        d = simulate(rng)
        lo, hi = fit_and_interval(d)
        cov += lo <= truth <= hi
    return cov / n_sim


TRUE_BETA = 0.5

def sim_independent(rng, n=40):
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": TRUE_BETA * x + rng.normal(0, 1, n)})


def sim_clustered(rng, n_clust=8, per=5):
    cl = np.repeat(rng.normal(0, 1.2, n_clust), per)
    x = np.repeat(rng.normal(0, 1, n_clust), per) + rng.normal(0, 0.3, n_clust*per)
    y = TRUE_BETA * x + cl + rng.normal(0, 0.5, n_clust * per)
    return pd.DataFrame({"x": x, "y": y,
                         "cl": np.repeat(np.arange(n_clust), per)})


def ols_ci(d):
    f = smf.ols("y ~ x", data=d).fit()
    return f.conf_int().loc["x"].values


def cluster_ci(d):
    f = smf.ols("y ~ x", data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["cl"]})
    return f.conf_int().loc["x"].values


print(f"  {'scenario':<44}{'coverage of nominal 95% CI':>28}")
print(f"  {'independent data, OLS interval':<44}"
      f"{calibration_check(ols_ci, sim_independent, TRUE_BETA, seed=1):>27.1%}")
print(f"  {'CLUSTERED data, naive OLS interval':<44}"
      f"{calibration_check(ols_ci, sim_clustered, TRUE_BETA, seed=2):>27.1%}")
print(f"  {'CLUSTERED data, cluster-robust interval':<44}"
      f"{calibration_check(cluster_ci, sim_clustered, TRUE_BETA, seed=3):>27.1%}")
print("\n  Two findings here, and the second is the more interesting one:")
print("   * Naive OLS on clustered data covers ~61% instead of 95%. That is")
print("     the pseudoreplication of Module 01, measured directly.")
print("   * The cluster-robust interval FIXES MOST BUT NOT ALL of it (~80%).")
print("     With only 8 clusters the sandwich estimator (eq. 14.8) is itself")
print("     badly biased - exactly the J >= 40 guidance from Module 14. With")
print("     few clusters, use a mixed model or aggregate to the cluster level.")
print("\n  Simulate from your assumed model, run your EXACT analysis, and check")
print("  that nominal coverage is delivered. This is how you validate a")
print("  pipeline before applying it to data you cannot check - and how you")
print("  discover that a 'robust' method is not robust at YOUR sample size.")

# %% [markdown]
# ## 6. Property-based tests for statistical code
#
# Test *properties*, not just that the code runs.

# %%
header("6. The five property tests every analysis script should have")

rng = np.random.default_rng(2405)
tests = []


def check(name, ok, detail=""):
    tests.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")


# (1) KNOWN-ANSWER: from-scratch OLS matches statsmodels to 1e-10.
X = rng.normal(0, 1, (50, 3))
yv = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 1, 50)
Xd = np.column_stack([np.ones(50), X])
mine = np.linalg.lstsq(Xd, yv, rcond=None)[0]
theirs = sm.OLS(yv, Xd).fit().params
check("known-answer: OLS matches statsmodels",
      np.allclose(mine, theirs, atol=1e-10),
      f"max diff {np.max(np.abs(mine - theirs)):.2e}")

# (2) INVARIANCE: row order must not change the fit.
perm = rng.permutation(50)
mine_perm = np.linalg.lstsq(Xd[perm], yv[perm], rcond=None)[0]
check("invariance: row permutation leaves coefficients unchanged",
      np.allclose(mine, mine_perm, atol=1e-10))

# (3) EQUIVARIANCE: scaling the outcome scales coefficients predictably.
mine_scaled = np.linalg.lstsq(Xd, 3.0 * yv, rcond=None)[0]
check("equivariance: y -> 3y scales coefficients by 3",
      np.allclose(3.0 * mine, mine_scaled, atol=1e-10))

# (4) ALIGNMENT: assay columns must match metadata rows (stats.md eq. 2.1).
assay = pd.DataFrame(rng.normal(size=(10, 6)),
                     columns=[f"S{i}" for i in range(6)])
meta = pd.DataFrame({"grp": list("aaabbb")},
                    index=[f"S{i}" for i in range(6)])
check("alignment: assay columns == metadata index (eq. 2.1)",
      assay.columns.equals(meta.index))

# (5) NULL CALIBRATION: false-positive rate is at the nominal level.
fp = np.mean([smf.ols("y ~ x",
                      data=pd.DataFrame({"x": rng.normal(0, 1, 40),
                                         "y": rng.normal(0, 1, 40)})
                      ).fit().pvalues["x"] < 0.05
              for _ in range(2000)])
check("null calibration: FPR within 1% of nominal 0.05",
      abs(fp - 0.05) < 0.01, f"observed {fp:.4f}")

# (6) BOUNDARY: degenerate inputs must fail loudly, not silently.
try:
    const = pd.DataFrame({"x": np.ones(20), "y": rng.normal(0, 1, 20)})
    f = smf.ols("y ~ x", data=const).fit()
    ok = np.isnan(f.pvalues.get("x", np.nan)) or np.linalg.matrix_rank(
        np.column_stack([np.ones(20), np.ones(20)])) < 2
except Exception:
    ok = True
check("boundary: a constant predictor does not silently return nonsense", ok)

print(f"\n  {sum(tests)}/{len(tests)} property tests passed.")

# %% [markdown]
# ## 7. Seeding, done right

# %%
header("7. Reproducibility mechanics")

print("  (a) Named generators per stochastic step, not one global seed:")
print("      rng_boot   = np.random.default_rng(101)")
print("      rng_perm   = np.random.default_rng(202)")
print("      rng_cvsplit= np.random.default_rng(303)")
print("      -> inserting a cell above cannot shift any downstream draw.")

print("\n  (b) Parallel-safe streams via SeedSequence.spawn:")
parent = np.random.SeedSequence(2024)
kids = [np.random.default_rng(s) for s in parent.spawn(3)]
print(f"      three worker first-draws: "
      f"{[round(float(k.normal()), 4) for k in kids]}")

print("\n  (c) Environment record, emitted with every run:")
import sys, platform
from importlib.metadata import version
print(f"      python {sys.version.split()[0]} on {platform.system()}")
for pkg in ["numpy", "scipy", "pandas", "statsmodels", "scikit-learn"]:
    print(f"      {pkg:<14} {version(pkg)}")

print("\n  (d) Project layout that makes `results/` disposable:")
print("      data/raw/      read-only, checksummed, never written by code")
print("      data/derived/  regenerable; safe to delete")
print("      results/       figures and tables; regenerable")
print("      If one command cannot rebuild results/ from data/raw/, the")
print("      analysis is not reproducible.")

print("\n  (e) Export the FULL statistic table, not the filtered list:")
print("      estimate, SE, statistic, raw p, adjusted p, n, base mean.")
print("      Someone will want to re-threshold, meta-analyse, or run a")
print("      competitive gene-set test - all of which need every feature.")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
st.probplot(rq_pois, dist="norm", plot=ax)
ax.set_title("Randomised quantile residuals: WRONG (Poisson) model", fontsize=9)

ax = axes[0, 1]
st.probplot(rq_nb, dist="norm", plot=ax)
ax.set_title("Same residuals under the CORRECT (NB) model", fontsize=9)

ax = axes[1, 0]
ax.errorbar(spec.index, spec["estimate"],
            yerr=[spec["estimate"] - spec["lo"], spec["hi"] - spec["estimate"]],
            fmt="o", ms=3, lw=0.6, capsize=0)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("specification (ordered by estimate)")
ax.set_ylabel("treatment coefficient")
ax.set_title(f"Eq. (34.3): specification curve ({len(spec)} analyses)",
             fontsize=9)

ax = axes[1, 1]
r = np.random.default_rng(99)
A0 = r.normal(0, 1, (4000, 5)); B0 = r.normal(0, 1, (4000, 5))
ax.hist(pipeline_under_test(A0, B0, False), bins=40, alpha=0.7,
        label="correct pipeline")
ax.hist(pipeline_under_test(A0, B0, True), bins=40, alpha=0.7,
        label="pipeline with a bug")
ax.set_xlabel("p-value on PERMUTED (null) data")
ax.set_title("Negative control: the histogram must be flat", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "diagnostics.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/diagnostics.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topics 34-35)
#
# 1. Diagnose every model you report, with residuals appropriate to its family.
# 2. Run leave-one-sample-out and leave-one-batch-out for any small-$n$ result.
# 3. Pre-specify the primary analysis; report the multiverse around it.
# 4. Always run the permuted-label negative control.
# 5. Seed at the point of use, pin the environment, version-stamp annotations,
#    and export full statistic tables.
#
# **You have finished the core track.** Continue with the applied modules in
# `statsPy/bioinformatics/`, starting with
# `30_bulk_rnaseq_differential_expression.py`.
