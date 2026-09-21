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
# # Module 00: Setup, environment check, and house rules
#
# **Curriculum link:** `stats.md` -> Topic 35 (Reproducible statistical workflows)
#
# **Run this first.** It verifies that every package the course uses is present
# and that the random-number machinery behaves deterministically. If this module
# prints `ALL CHECKS PASSED`, every other module in `statsPy/` will run.
#
# ```bash
# docker run --rm -v "$PWD":/work -w /work learn-stats-py:1.0 \
#     python statsPy/core/00_setup_and_environment.py
# ```
#
# ## What you will learn
#
# 1. How to capture an environment so a result can be regenerated (Topic 35).
# 2. Why `numpy.random.default_rng` is preferred over the legacy `np.random.seed`.
# 3. How to seed *at the point of use* rather than once per script.
# 4. The output convention used by every later module.

# %% [markdown]
# ## 1. Environment capture
#
# `stats.md` Topic 35 makes the point that statistical results genuinely change
# across package versions: default multiple-testing methods, optimiser defaults,
# and RNG algorithms have all changed historically. So the first thing any
# analysis should emit is *what produced it*.

# %%
import platform
import sys
from importlib.metadata import version, PackageNotFoundError

REQUIRED = [
    "numpy",         # arrays, linear algebra, random number generation
    "scipy",         # distributions and classical hypothesis tests
    "pandas",        # tabular data handling
    "matplotlib",    # plotting
    "seaborn",       # statistical graphics convenience layer
    "statsmodels",   # OLS / GLM / GEE / MixedLM / multipletests / power
    "scikit-learn",  # cross-validation, pipelines, penalised regression
    "lifelines",     # Kaplan-Meier, log-rank, Cox proportional hazards
]


def environment_report():
    """Print an environment record of the kind every analysis should emit."""
    print("=" * 70)
    print("ENVIRONMENT RECORD")
    print("=" * 70)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.platform()}")
    print("-" * 70)
    missing = []
    for pkg in REQUIRED:
        try:
            print(f"{pkg:<14}: {version(pkg)}")
        except PackageNotFoundError:
            print(f"{pkg:<14}: *** MISSING ***")
            missing.append(pkg)
    print("=" * 70)
    return missing


missing_packages = environment_report()

# %% [markdown]
# ## 2. Random number generation, done properly
#
# Two rules from `stats.md` Topic 35:
#
# * **Use a `Generator` object, not the legacy global state.** `np.random.seed()`
#   mutates a single hidden global stream. If any library you call also draws
#   from it, your "reproducible" result silently depends on library internals.
#   A `Generator` you pass around explicitly cannot be disturbed by anyone else.
# * **Seed at the point of use.** Seeding once at the top of a script is fragile:
#   inserting a new cell above shifts every downstream draw. Anything whose value
#   you will *report* should get its own seeded generator.

# %%
import numpy as np

# The course convention: every module creates named generators for each
# stochastic step, so that inserting or reordering code cannot change results.
rng_demo = np.random.default_rng(seed=20260919)

sample_a = rng_demo.normal(loc=0.0, scale=1.0, size=5)
print("draw with a fresh generator:", np.round(sample_a, 4))

# Re-creating the generator with the same seed reproduces the draw exactly.
sample_b = np.random.default_rng(seed=20260919).normal(0.0, 1.0, 5)
print("same seed, new generator  :", np.round(sample_b, 4))
print("identical?                :", np.array_equal(sample_a, sample_b))

# %% [markdown]
# ### Parallel-safe streams
#
# If you ever parallelise a simulation (Topic 9 power analysis, Topic 34
# calibration checks), workers must **not** share a stream. `SeedSequence.spawn`
# creates provably independent child streams.

# %%
parent_seq = np.random.SeedSequence(entropy=12345)
child_seqs = parent_seq.spawn(4)  # four independent worker streams
worker_rngs = [np.random.default_rng(s) for s in child_seqs]

first_draws = [float(r.normal()) for r in worker_rngs]
print("four independent worker streams:", np.round(first_draws, 4))
print("all distinct?                  :", len(set(first_draws)) == 4)

# %% [markdown]
# ## 3. Output convention
#
# Every module writes figures to `results/<module_name>/` **relative to the
# directory you launch from**, which is why the documented way to run these
# scripts is from the repository root. Numeric results are printed to stdout so
# that the automated test runner (`tools/run_all_tests.sh`) can check them.

# %%
import os

MODULE_NAME = "00_setup_and_environment"


def results_dir(module_name=MODULE_NAME):
    """Create and return the output directory for a module.

    Uses ``STATS_OUT`` if set (the test runner sets it to a scratch path),
    otherwise ``results/`` under the current working directory.
    """
    base = os.environ.get("STATS_OUT", "results")
    path = os.path.join(base, module_name)
    os.makedirs(path, exist_ok=True)
    return path


OUT = results_dir()
print(f"figures for this module would be written to: {OUT}")

# %% [markdown]
# ## 4. A smoke test of every library the course uses
#
# Each check below exercises the *specific* API that later modules depend on, so
# a failure here points at a real incompatibility rather than a missing import.

# %%
import matplotlib

matplotlib.use("Agg")  # head-less backend: never tries to open a window
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from lifelines import KaplanMeierFitter

checks = []


def check(name, condition, detail=""):
    """Record a boolean check; later modules reuse this pattern for assertions."""
    checks.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail else ''}")


rng = np.random.default_rng(7)

# --- scipy: the t distribution and a Welch t-test (stats.md eq. 7.3-7.4) -----
x = rng.normal(0, 1, 30)
y = rng.normal(0.5, 1.5, 30)
welch = st.ttest_ind(x, y, equal_var=False)
check("scipy Welch t-test", np.isfinite(welch.pvalue), f"p = {welch.pvalue:.4f}")

# --- pandas: DataFrame round-trip -------------------------------------------
df = pd.DataFrame({"y": np.concatenate([x, y]),
                   "g": ["a"] * 30 + ["b"] * 30})
check("pandas groupby", df.groupby("g")["y"].mean().shape == (2,))

# --- statsmodels: OLS is the same thing as a t-test (stats.md eq. 7.5) ------
ols_fit = smf.ols("y ~ g", data=df).fit()
pooled = st.ttest_ind(x, y, equal_var=True)
check(
    "OLS t-statistic == pooled t-test",
    np.isclose(abs(ols_fit.tvalues["g[T.b]"]), abs(pooled.statistic)),
    f"{ols_fit.tvalues['g[T.b]']:.6f} vs {pooled.statistic:.6f}",
)

# --- statsmodels: Benjamini-Hochberg (stats.md eq. 8.5) ---------------------
pvals = rng.uniform(0, 1, 100)
_, padj, _, _ = multipletests(pvals, method="fdr_bh")
check("BH adjustment monotone", np.all(np.diff(np.sort(padj)) >= -1e-12))

# --- statsmodels: GLM with a log link and offset (stats.md eq. 13.8) --------
counts = rng.poisson(20, 40)
exposure = rng.uniform(0.8, 1.2, 40)
glm_fit = sm.GLM(counts, sm.add_constant(np.ones(40)),
                 family=sm.families.Poisson(),
                 offset=np.log(exposure)).fit()
check("statsmodels Poisson GLM with offset", glm_fit.converged)

# --- scikit-learn: group-aware splitting (stats.md Topic 31) ----------------
groups = np.repeat(np.arange(10), 6)
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(np.zeros((60, 3)), np.zeros(60), groups))
leak = any(set(groups[tr]) & set(groups[te]) for tr, te in splits)
check("GroupKFold keeps donors intact", not leak)

# --- scikit-learn: a model fits ---------------------------------------------
clf = LogisticRegression(max_iter=500).fit(rng.normal(size=(60, 3)),
                                           rng.integers(0, 2, 60))
check("sklearn LogisticRegression fits", hasattr(clf, "coef_"))

# --- lifelines: Kaplan-Meier (stats.md eq. 28.6) ----------------------------
km = KaplanMeierFitter().fit(rng.exponential(10, 50),
                             rng.binomial(1, 0.7, 50))
check("lifelines KaplanMeierFitter", km.survival_function_.shape[0] > 1)

# --- matplotlib: a figure can be written head-less --------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot([0, 1], [0, 1])
fig.savefig(os.path.join(OUT, "smoke_test.png"), dpi=80, bbox_inches="tight")
plt.close(fig)
check("matplotlib head-less save",
      os.path.exists(os.path.join(OUT, "smoke_test.png")))

# %% [markdown]
# ## 5. Verdict

# %%
n_failed = sum(1 for _, ok, _ in checks if not ok)
print("\n" + "=" * 70)
if missing_packages:
    print(f"MISSING PACKAGES: {', '.join(missing_packages)}")
if n_failed == 0 and not missing_packages:
    print("ALL CHECKS PASSED - the environment is ready for the whole course.")
else:
    print(f"{n_failed} check(s) failed. Rebuild the image:")
    print("  docker build -f docker/Dockerfile.python -t learn-stats-py:1.0 .")
print("=" * 70)

# %% [markdown]
# ## Where to go next
#
# | Next | Topic |
# |---|---|
# | `01_study_design_and_estimands.py` | What is `n`? The pseudoreplication equation |
# | `02_data_structures_and_scales.py` | Counts vs intensities vs proportions |
#
# **House rules used throughout the course**
#
# 1. Every stochastic step gets an explicitly seeded `Generator`.
# 2. Every claim in a comment cites the equation number in `stats.md`.
# 3. Nothing is installed at runtime; the container *is* the environment.
# 
# **Next:** `01_study_design_and_estimands.py`
