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
# # Module 01: Statistical thinking: units, estimands, and pseudoreplication
#
# **Curriculum link:** `stats.md` -> Topic 1, equations (1.1)-(1.9)
#
# ```bash
# docker run --rm -v "$PWD":/work -w /work learn-stats-py:1.0 \
#     python statsPy/core/01_study_design_and_estimands.py
# ```
#
# ## Why this module exists
#
# Every other module assumes you know what `n` is. This one proves, by
# simulation, that getting it wrong is not a stylistic issue: it turns a 5%
# false-positive rate into a 60% false-positive rate. If you internalise only
# one module in this course, make it this one.
#
# ## What you will learn
#
# 1. The variance decomposition $\operatorname{Var}(\bar Y) = \sigma_b^2/n + \sigma_e^2/(nm)$: eq. (1.5).
# 2. Why more cells/reads per donor cannot rescue a study with 3 donors: eq. (1.6).
# 3. The intraclass correlation and design effect: eq. (1.7)-(1.8).
# 4. The effective sample size: eq. (1.9).
# 5. How randomisation converts an unobservable causal contrast into an
#    observable one: eq. (1.2)-(1.3).
# 6. How to detect a design that *cannot* answer the question (rank deficiency).

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "01_study_design_and_estimands"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)

pd.set_option("display.width", 120)
np.set_printoptions(precision=4, suppress=True)


def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1. The hierarchical data-generating process
#
# `stats.md` eq. (1.4) describes the two-level structure that essentially every
# biological experiment has:
#
# $$Y_{ij} = \mu + b_i + e_{ij}, \qquad b_i \sim (0,\sigma_b^2),\ e_{ij}\sim(0,\sigma_e^2)$$
#
# * $b_i$ is the **donor** (or mouse, or patient) effect: *biological* variation.
# * $e_{ij}$ is the **measurement** effect: cells, reads, technical replicates.
#
# The distinction is not philosophical. It determines which term in eq. (1.5)
# your sample size divides.

# %%
def simulate_hierarchy(n_donors, m_per_donor, sigma_b, sigma_e, mu=0.0, rng=None):
    """Simulate the two-level model of stats.md eq. (1.4).

    Parameters
    ----------
    n_donors : int      number of INDEPENDENT biological units (this is `n`)
    m_per_donor : int   measurements per donor (cells, reads, tech. replicates)
    sigma_b : float     between-donor SD (biological)
    sigma_e : float     within-donor SD (technical)

    Returns
    -------
    DataFrame with columns donor, value - in long ("tidy") form.
    """
    rng = np.random.default_rng() if rng is None else rng
    donor_effects = rng.normal(0.0, sigma_b, size=n_donors)      # b_i
    noise = rng.normal(0.0, sigma_e, size=(n_donors, m_per_donor))  # e_ij
    values = mu + donor_effects[:, None] + noise
    return pd.DataFrame({
        "donor": np.repeat(np.arange(n_donors), m_per_donor),
        "value": values.ravel(),
    })


header("1. One simulated hierarchical dataset")
rng = np.random.default_rng(101)
demo = simulate_hierarchy(n_donors=5, m_per_donor=200,
                          sigma_b=1.0, sigma_e=1.0, rng=rng)
print(demo.groupby("donor")["value"].agg(["count", "mean", "std"]).round(3))
print("\nNote how the donor MEANS differ far more than sampling error alone")
print("would allow: that spread is sigma_b, and no number of cells removes it.")

# %% [markdown]
# ## 2. Equation (1.5) verified numerically
#
# $$\operatorname{Var}(\bar Y_{\cdot\cdot}) = \frac{\sigma_b^2}{n} + \frac{\sigma_e^2}{nm}$$
#
# We estimate the left-hand side by repeatedly simulating whole experiments and
# taking the variance of the grand mean across replications, then compare it to
# the closed form.

# %%
header("2. Verifying the pseudoreplication equation (1.5)")

def empirical_var_of_grand_mean(n, m, sigma_b, sigma_e, n_sim=4000, seed=0):
    """Monte-Carlo estimate of Var(grand mean) over repeated experiments."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, sigma_b, size=(n_sim, n))            # donor effects
    e = rng.normal(0, sigma_e, size=(n_sim, n, m))         # measurement noise
    grand_means = (b[:, :, None] + e).mean(axis=(1, 2))
    return grand_means.var(ddof=1)


rows = []
for (n, m) in [(3, 1), (3, 10), (3, 100), (3, 1000), (30, 1), (30, 100)]:
    sb, se = 1.0, 1.0
    theory = sb**2 / n + se**2 / (n * m)          # eq. (1.5)
    naive = (sb**2 + se**2) / (n * m)             # what you get if you pretend
                                                  # all n*m values are independent
    empirical = empirical_var_of_grand_mean(n, m, sb, se, seed=n * 1000 + m)
    rows.append({"n_donors": n, "m_per_donor": m,
                 "eq(1.5) theory": theory, "simulated": empirical,
                 "naive (wrong)": naive,
                 "understated by": theory / naive})

tab = pd.DataFrame(rows)
print(tab.round(5).to_string(index=False))
print("\n'simulated' tracks 'eq(1.5) theory', never the naive formula.")
print("The last column is how many times too SMALL the naive variance is.")

# %% [markdown]
# ### Equation (1.6): the hard floor
#
# $$\lim_{m\to\infty}\operatorname{Var}(\bar Y) = \frac{\sigma_b^2}{n}$$
#
# Look at the rows with `n_donors = 3` above. As `m` goes 1 -> 10 -> 100 -> 1000,
# the true variance falls to **0.333 and stops**: that is $\sigma_b^2/n = 1/3$.
# The naive variance keeps shrinking toward zero, which is the lie.

# %%
header("Equation (1.6): the floor you cannot sequence past")
sb, se, n = 1.0, 1.0, 3
for m in [1, 10, 100, 1_000, 100_000]:
    print(f"  m = {m:>7,}  Var(mean) = {sb**2/n + se**2/(n*m):.6f}"
          f"   (floor = {sb**2/n:.6f})")

# %% [markdown]
# ## 3. ICC, design effect, and effective sample size
#
# $$\rho = \frac{\sigma_b^2}{\sigma_b^2+\sigma_e^2} \quad (1.7), \qquad
#   \mathrm{DE} = 1+(m-1)\rho \quad (1.8), \qquad
#   n_{\text{eff}} = \frac{nm}{\mathrm{DE}} \quad (1.9)$$
#
# The design effect is the single number that tells you how badly clustering
# inflates the true variance relative to the naive independent-sample formula.

# %%
header("3. ICC, design effect, effective sample size")

def icc(sigma_b, sigma_e):
    """stats.md eq. (1.7)."""
    return sigma_b**2 / (sigma_b**2 + sigma_e**2)


def design_effect(m, rho):
    """stats.md eq. (1.8)."""
    return 1.0 + (m - 1) * rho


def effective_n(n, m, rho):
    """stats.md eq. (1.9)."""
    return n * m / design_effect(m, rho)


print(f"{'rho':>6} {'m':>7} {'design effect':>14} {'SE inflation':>13} "
      f"{'n_eff (n=3)':>12} {'nominal n*m':>12}")
for rho in [0.01, 0.02, 0.05, 0.20]:
    for m in [100, 500, 2000]:
        de = design_effect(m, rho)
        print(f"{rho:>6.2f} {m:>7,} {de:>14.2f} {np.sqrt(de):>13.2f} "
              f"{effective_n(3, m, rho):>12.1f} {3*m:>12,}")

print("\nWorked example from stats.md Topic 1: n=3 donors, m=500 cells, rho=0.05")
print(f"  design effect  = {design_effect(500, 0.05):.2f}")
print(f"  SE too small by= {np.sqrt(design_effect(500, 0.05)):.2f}x")
print(f"  n_eff          = {effective_n(3, 500, 0.05):.1f}  (not 1500)")

# %% [markdown]
# ## 4. The consequence: false-positive rate of a naive cell-level test
#
# This is the experiment that matters. We simulate a study with **no real
# effect whatsoever**: two groups of donors drawn from the same distribution -
# and test it two ways:
#
# * **Naive:** pool all cells, t-test cells in group A vs cells in group B.
# * **Correct:** average within donor, t-test the donor means (pseudobulk).
#
# Under a valid test, both should reject 5% of the time.

# %%
header("4. False-positive rate: naive cell-level vs donor-level testing")

def false_positive_rate(n_per_group, m_per_donor, sigma_b, sigma_e,
                        n_sim=2000, alpha=0.05, seed=42):
    """Simulate a TRUE NULL and count rejections under two analysis strategies."""
    rng = np.random.default_rng(seed)
    naive_reject = 0
    donor_reject = 0

    for _ in range(n_sim):
        # Both groups come from the SAME distribution: the null is true.
        b = rng.normal(0, sigma_b, size=2 * n_per_group)
        e = rng.normal(0, sigma_e, size=(2 * n_per_group, m_per_donor))
        cells = b[:, None] + e                      # donor x cell matrix

        g1_cells = cells[:n_per_group].ravel()
        g2_cells = cells[n_per_group:].ravel()

        # (a) NAIVE: every cell treated as an independent observation
        p_naive = st.ttest_ind(g1_cells, g2_cells, equal_var=False).pvalue

        # (b) CORRECT: aggregate to one value per donor, then test donors
        donor_means = cells.mean(axis=1)
        p_donor = st.ttest_ind(donor_means[:n_per_group],
                               donor_means[n_per_group:],
                               equal_var=False).pvalue

        naive_reject += p_naive < alpha
        donor_reject += p_donor < alpha

    return naive_reject / n_sim, donor_reject / n_sim


for (sb, se, m) in [(0.5, 1.0, 200), (1.0, 1.0, 200), (1.0, 1.0, 1000)]:
    rho = icc(sb, se)
    fp_naive, fp_donor = false_positive_rate(
        n_per_group=4, m_per_donor=m, sigma_b=sb, sigma_e=se,
        n_sim=1500, seed=int(sb * 100 + m))
    print(f"  sigma_b={sb}, sigma_e={se}, m={m:>4}  (rho={rho:.2f}, "
          f"DE={design_effect(m, rho):6.1f})")
    print(f"      naive cell-level FPR : {fp_naive:6.1%}   <-- should be 5%")
    print(f"      donor-level      FPR : {fp_donor:6.1%}   <-- is 5%\n")

print("The naive test is not 'slightly liberal'. It is broken.")
print("This is the mechanism behind documented false-discovery inflation in")
print("naive per-cell single-cell differential expression (stats.md Topic 21).")

# %% [markdown]
# ## 5. Estimands: what randomisation actually buys you
#
# Equation (1.1): we only ever see one potential outcome per unit.
# Equation (1.2): the causal estimand is $\tau = \mathbb{E}[Y(1)-Y(0)]$.
# Equation (1.3): randomisation makes the *observable* group contrast equal it.
#
# Below we build a population where we (unusually) know **both** potential
# outcomes, so we can compare the truth against what each design recovers.

# %%
header("5. ATE, randomisation, and confounding")

rng = np.random.default_rng(2026)
N = 20000

# A confounder: baseline severity. It raises the outcome AND, in the
# observational world, makes treatment more likely.
severity = rng.normal(0, 1, N)

# TRUE potential outcomes. The true ATE is exactly 2.0 by construction.
TRUE_ATE = 2.0
y0 = 10 + 3 * severity + rng.normal(0, 1, N)
y1 = y0 + TRUE_ATE

# --- Design A: randomised. Treatment is a coin flip, independent of severity.
a_rand = rng.binomial(1, 0.5, N)
y_rand = np.where(a_rand == 1, y1, y0)              # eq. (1.1)
est_rand = y_rand[a_rand == 1].mean() - y_rand[a_rand == 0].mean()

# --- Design B: observational. Sicker patients are more likely to be treated.
p_treat = 1 / (1 + np.exp(-(0.0 + 1.5 * severity)))
a_obs = rng.binomial(1, p_treat)
y_obs = np.where(a_obs == 1, y1, y0)
est_obs_crude = y_obs[a_obs == 1].mean() - y_obs[a_obs == 0].mean()

# --- Design B, adjusted for the confounder (the g-formula, stats.md eq. 32.2)
#     Fit E[Y | A, severity] and average the contrast over the whole population.
X = np.column_stack([np.ones(N), a_obs, severity])
beta = np.linalg.lstsq(X, y_obs, rcond=None)[0]
est_obs_adj = beta[1]

print(f"  TRUE ATE (known by construction)      : {TRUE_ATE:.3f}")
print(f"  Randomised design, crude difference   : {est_rand:.3f}   <-- unbiased")
print(f"  Observational, crude difference       : {est_obs_crude:.3f}   <-- biased")
print(f"  Observational, adjusted for severity  : {est_obs_adj:.3f}   <-- recovered")
print("\nEq. (1.3) holds ONLY under randomisation. In the observational design")
print("the same arithmetic estimates an association, and the gap from 2.0 is")
print("confounding bias. Adjustment worked here only because we happened to")
print("measure the confounder - see Topic 32 for when it does not.")

# %% [markdown]
# ## 6. Designs that cannot answer the question
#
# Before any modelling, cross-tabulate your factor of interest against batch.
# If the table is block-diagonal, the effects are *aliased*: the design matrix
# is rank-deficient (`stats.md` eq. 16.5 / 19.3) and **no software can separate
# them**. This check costs one line and saves entire projects.

# %%
header("6. Detecting a confounded (rank-deficient) design")

def design_is_identifiable(condition, batch, verbose=True):
    """Return True if condition and batch effects are separately estimable.

    Builds the design matrix [1 | condition | batch dummies] and compares its
    numerical rank to the number of columns.
    """
    cond = pd.get_dummies(pd.Series(condition), drop_first=True).astype(float)
    bat = pd.get_dummies(pd.Series(batch), drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(condition)), cond.values, bat.values])
    rank = np.linalg.matrix_rank(X)
    ok = rank == X.shape[1]
    if verbose:
        print(pd.crosstab(pd.Series(condition, name="condition"),
                          pd.Series(batch, name="batch")))
        print(f"    design matrix: {X.shape[1]} columns, numerical rank {rank}"
              f"  -> {'IDENTIFIABLE' if ok else 'CONFOUNDED (aliased)'}\n")
    return ok


print("Design A - conditions balanced across batches (good):")
design_is_identifiable(condition=["ctrl", "trt"] * 6,
                       batch=np.repeat(["b1", "b2", "b3"], 4))

print("Design B - every control in batch 1, every treated in batch 2 (fatal):")
design_is_identifiable(condition=["ctrl"] * 6 + ["trt"] * 6,
                       batch=["b1"] * 6 + ["b2"] * 6)

# %% [markdown]
# ## 7. Figure: variance floor and false-positive inflation

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: eq. (1.5) vs the naive formula as m grows.
ms = np.logspace(0, 4, 60)
for n_don, colour in [(3, "C0"), (10, "C1"), (30, "C2")]:
    axes[0].loglog(ms, 1.0 / n_don + 1.0 / (n_don * ms),
                   color=colour, label=f"eq.(1.5), n={n_don}")
    axes[0].axhline(1.0 / n_don, color=colour, ls=":", lw=1)
axes[0].loglog(ms, 2.0 / (3 * ms), "k--", label="naive, n=3 (wrong)")
axes[0].set_xlabel("measurements per donor, m")
axes[0].set_ylabel("Var(grand mean)")
axes[0].set_title("Eq. (1.5)-(1.6): the floor is $\\sigma_b^2/n$")
axes[0].legend(fontsize=7)

# Right: design effect vs m for several ICCs.
for rho in [0.01, 0.02, 0.05, 0.10, 0.20]:
    axes[1].plot(ms, 1 + (ms - 1) * rho, label=f"$\\rho$={rho}")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].axhline(1, color="k", lw=0.8)
axes[1].set_xlabel("measurements per donor, m")
axes[1].set_ylabel("design effect  $1+(m-1)\\rho$")
axes[1].set_title("Eq. (1.8): variance inflation from clustering")
axes[1].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "pseudoreplication.png"), dpi=110,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/pseudoreplication.png")

# %% [markdown]
# ## 8. Checklist to run before every analysis
#
# 1. **Unit.** What was independently randomised or independently sampled?
#    That count is `n`. Cells, reads, wells, and technical replicates are not.
# 2. **Estimand.** Write it as an equation before opening the data.
# 3. **Structure.** Paired? Blocked? Nested? If it was designed in, it must be
#    modelled in.
# 4. **Identifiability.** Cross-tabulate the factor of interest against batch
#    and check the rank (section 6).
# 5. **ICC.** Estimate $\rho$ from pilot data and compute the design effect
#    before trusting any cell-level p-value.
#
# ## Self-check
#
# * A study profiles 50,000 cells from 3 patients per arm. What is `n`?
#   *(3 per arm. Eq. (1.6) says the extra cells cannot help.)*
# * You measure ICC = 0.03 with 800 cells per donor. By what factor are naive
#   standard errors too small? *(`sqrt(1 + 799*0.03) = 5.0`)*
# * `table(condition, batch)` is diagonal. What should you do?
#   *(Report the confounding. Do not "correct" it: eq. (19.3).)*
#
# **Next:** `02_data_structures_and_scales.py`
