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
# # Module 09: Power, sample size, and design optimisation
#
# **Curriculum link:** `stats.md` -> Topic 9, equations (9.1)-(9.10)
#
# ## What you will learn
#
# 1. Closed-form power (9.1)-(9.4) and **Lehr's rule $n\approx16/d^2$** (9.3).
# 2. Why post-hoc power is a deterministic function of the p-value and must
#    never be reported.
# 3. Gains from pairing (9.5) and losses from clustering (9.6).
# 4. **The donors-vs-cells optimum (9.7)** and the depth-vs-replicates
#    argument for RNA-seq (9.8)-(9.9).
# 5. Simulation-based power (9.10): the only honest tool for real pipelines.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "09_power_and_sample_size"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Closed-form power and sample size, eq. (9.1)-(9.4)

# %%
header("1. Normal approximation (9.1)-(9.2) vs exact non-central t (9.4)")

def power_z(d, n, alpha=0.05):
    """stats.md eq. (9.1): normal approximation, two-sided."""
    lam = d / np.sqrt(2.0 / n)
    z = st.norm.ppf(1 - alpha / 2)
    return st.norm.sf(z - lam) + st.norm.cdf(-z - lam)


def n_from_z(d, power=0.8, alpha=0.05):
    """stats.md eq. (9.2): invert the normal approximation."""
    return 2 * (st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) ** 2 / d**2


def power_t(d, n, alpha=0.05):
    """stats.md eq. (9.4): exact, using the non-central t distribution."""
    df = 2 * n - 2
    lam = d * np.sqrt(n / 2.0)                 # non-centrality parameter
    crit = st.t.ppf(1 - alpha / 2, df)
    return st.nct.sf(crit, df, lam) + st.nct.cdf(-crit, df, lam)


print(f"  {'d':>6}{'n/group (9.2)':>15}{'Lehr 16/d^2':>14}"
      f"{'exact n (9.4)':>15}{'power@exact n':>15}")
for d in [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    n_norm = n_from_z(d)
    n_exact = next(n for n in range(2, 5000) if power_t(d, n) >= 0.80)
    print(f"  {d:>6.1f}{n_norm:>15.1f}{16/d**2:>14.1f}"
          f"{n_exact:>15d}{power_t(d, n_exact):>15.3f}")

print("\n  Lehr's rule (9.3) is accurate to within a sample or two and is worth")
print("  memorising: d=1 needs 16/group, d=0.5 needs 64, d=0.2 needs 400.")
print("  It instantly exposes 'a small but significant effect with n=4'.")

# %% [markdown]
# ## 2. Post-hoc power is not information
#
# Power computed from the *observed* effect is a one-to-one function of the
# p-value. It tells you nothing new: it just re-expresses the p-value on a
# confusing scale.

# %%
header("2. Why 'observed power' must never be reported")

rng = np.random.default_rng(901)
records = []
for _ in range(400):
    n = 12
    a = rng.normal(0.35, 1, n)
    b = rng.normal(0.0, 1, n)
    tt = st.ttest_ind(a, b, equal_var=True)
    d_obs = cohend = (a.mean() - b.mean()) / np.sqrt(
        ((n - 1) * a.var(ddof=1) + (n - 1) * b.var(ddof=1)) / (2 * n - 2))
    records.append({"p": tt.pvalue, "observed_power": power_t(abs(d_obs), n)})
rec = pd.DataFrame(records)
print(f"  Spearman correlation between p-value and 'observed power': "
      f"{st.spearmanr(rec['p'], rec['observed_power']).statistic:+.4f}")
print("  It is exactly -1: observed power is a relabelling of the p-value.")
print("\n  Report instead the MINIMUM DETECTABLE EFFECT of the design you ran:")
for n in [4, 6, 10, 20]:
    mde = next(d for d in np.arange(0.05, 5, 0.01) if power_t(d, n) >= 0.80)
    print(f"    n = {n:>3}/group -> 80% power only for d >= {mde:.2f}")

# %% [markdown]
# ## 3. Pairing gains and clustering losses, eq. (9.5)-(9.6)

# %%
header("3. Pairing (9.5) and clustering (9.6)")

def n_pairs_needed(d_raw, r, power=0.8, alpha=0.05):
    """stats.md eq. (9.5): d_raw = delta/sigma on the ORIGINAL scale."""
    return (st.norm.ppf(1 - alpha/2) + st.norm.ppf(power))**2 * 2*(1-r) / d_raw**2


print("  Paired design - subjects needed for 80% power, delta/sigma = 0.5:")
for r in [0.0, 0.3, 0.5, 0.7, 0.9]:
    n_p = n_pairs_needed(0.5, r)
    print(f"    within-pair correlation r = {r:.1f} -> {n_p:6.1f} pairs "
          f"({100*n_p/n_pairs_needed(0.5, 0.0):5.1f}% of the unpaired design)")

print("\n  Clustered design - multiply by the design effect (9.6):")
n_indep = n_from_z(0.5)
print(f"    independent-observation requirement: {n_indep:.0f} per arm")
for rho in [0.01, 0.05, 0.20]:
    for m in [10, 100]:
        de = 1 + (m - 1) * rho
        print(f"    rho={rho:.2f}, m={m:>3} obs/cluster: DE={de:6.2f} -> "
              f"{n_indep*de:8.0f} observations = {n_indep*de/m:6.1f} clusters")

print("""
  Read the CLUSTERS column, not the observations column. Two things follow.

  A design needing "1.2 clusters" is not a design. The degrees of freedom for
  a cluster-randomised comparison come from the number of CLUSTERS, not the
  number of observations, so a study with one or two clusters per arm has no
  usable inference however many observations sit inside them. Treat any row
  below roughly 8 clusters per arm as telling you the design is infeasible
  rather than cheap.

  Adding observations per cluster has a ceiling. As m grows, DE grows with it,
  so the required cluster count tends to n_indep * rho and stops falling: at
  rho = 0.20 you need about 13 clusters whether each holds 10 observations or
  100. Beyond that point only more CLUSTERS buy power. This is eq. (1.5)'s
  variance floor in design units, and Module 44 locates the same plateau for
  cells per donor.""")

# %% [markdown]
# ## 4. More donors or more cells? Equation (9.7)
#
# $$m^{*}=\sqrt{\frac{\sigma_e^2}{\sigma_b^2}\cdot\frac{c_{\text{donor}}}{c_{\text{cell}}}}$$
#
# Minimising $\sigma_b^2/n+\sigma_e^2/(nm)$ subject to a fixed budget
# $C=n(c_{\text{donor}}+m\,c_{\text{cell}})$.

# %%
header("4. The donors-vs-cells optimum (9.7)")

def optimal_cells(sigma_b, sigma_e, cost_donor, cost_cell):
    """stats.md eq. (9.7)."""
    return np.sqrt((sigma_e**2 / sigma_b**2) * (cost_donor / cost_cell))


def variance_at(n, m, sigma_b, sigma_e):
    """stats.md eq. (1.5)."""
    return sigma_b**2 / n + sigma_e**2 / (n * m)


BUDGET, C_DONOR, C_CELL = 20000.0, 800.0, 0.40
print(f"  budget = {BUDGET:,.0f}; cost per donor = {C_DONOR}, per cell = {C_CELL}")
print(f"\n  {'sigma_b':>9}{'sigma_e':>9}{'m* (9.7)':>11}{'n at m*':>9}"
      f"{'Var at m*':>12}{'best grid m':>13}")
for sb, se in [(1.0, 1.0), (1.0, 2.0), (0.5, 2.0), (2.0, 1.0)]:
    m_star = optimal_cells(sb, se, C_DONOR, C_CELL)
    n_star = BUDGET / (C_DONOR + m_star * C_CELL)
    # Brute-force check over a grid of m.
    grid = np.arange(5, 4000)
    n_grid = BUDGET / (C_DONOR + grid * C_CELL)
    v_grid = variance_at(n_grid, grid, sb, se)
    best_m = grid[np.argmin(v_grid)]
    print(f"  {sb:>9.1f}{se:>9.1f}{m_star:>11.1f}{n_star:>9.1f}"
          f"{variance_at(n_star, m_star, sb, se):>12.5f}{best_m:>13d}")

print("\n  The closed form matches the brute-force optimum. Because donor")
print("  variability (sigma_b) is large in human studies, m* is modest and the")
print("  budget is better spent on MORE DONORS. Eqs. (1.6), (9.7) and (9.9)")
print("  all deliver the same verdict.")

# %% [markdown]
# ## 5. RNA-seq: depth versus replicates, eq. (9.8)-(9.9)
#
# $$\operatorname{Var}(\hat\lambda)\approx\frac{2}{n}\Big(\frac1\mu+\phi\Big)$$
#
# Sequencing deeper raises $\mu$ and shrinks $1/\mu$: but **cannot touch
# $\phi$**. Once $\mu\gg1/\phi$, extra reads buy nothing.

# %%
header("5. Depth cannot substitute for replicates (9.8)-(9.9)")

def n_rnaseq(fold_change, mu, phi, power=0.8, alpha=0.05):
    """stats.md eq. (9.9): samples per group for an NB log fold change."""
    lam = np.log(fold_change)
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return 2 * z**2 * (1.0 / mu + phi) / lam**2


PHI = 0.16                                  # human bulk RNA-seq, BCV = 0.4
print(f"  dispersion phi = {PHI} (BCV = {np.sqrt(PHI):.2f}); "
      f"floor at mu ~ 1/phi = {1/PHI:.1f} counts\n")
print(f"  {'mean count mu':>15}" + "".join(f"{f'FC={fc}':>10}"
                                           for fc in [1.5, 2.0, 3.0]))
for mu in [1, 3, 6, 20, 100, 1000, 10000]:
    row = "".join(f"{n_rnaseq(fc, mu, PHI):>9.1f} " for fc in [1.5, 2.0, 3.0])
    print(f"  {mu:>15,}{row}")

print("\n  Read DOWN a column: from mu=100 to mu=10,000 (100x more sequencing)")
print(f"  the requirement barely moves ({n_rnaseq(2.0, 100, PHI):.1f} -> "
      f"{n_rnaseq(2.0, 10000, PHI):.1f} samples).")
print("  Read the mu=1 row: for genuinely low-count genes, depth DOES help.")
print("  So: filter low counts, then spend on replicates.")

# Verify (9.9) against a simulation with an actual NB GLM test.
import statsmodels.api as sm
print("\n  Verifying eq. (9.9) by simulation (NB Wald test):")
rng = np.random.default_rng(902)
for fc, mu in [(2.0, 50), (1.5, 50)]:
    n_pred = int(np.ceil(n_rnaseq(fc, mu, PHI)))
    hits = 0
    n_sim = 800
    for _ in range(n_sim):
        size = 1 / PHI
        a = rng.negative_binomial(size, size / (size + mu * fc), n_pred)
        b = rng.negative_binomial(size, size / (size + mu), n_pred)
        yv = np.concatenate([a, b]).astype(float)
        Xd = np.column_stack([np.ones(2 * n_pred),
                              np.r_[np.ones(n_pred), np.zeros(n_pred)]])
        try:
            f = sm.GLM(yv, Xd,
                       family=sm.families.NegativeBinomial(alpha=PHI)).fit()
            hits += f.pvalues[1] < 0.05
        except Exception:
            pass
    print(f"    FC={fc}, mu={mu}: eq.(9.9) predicts n={n_pred} -> "
          f"simulated power = {hits/n_sim:.2f}  (target 0.80)")

# %% [markdown]
# ## 6. Multiplicity burden grows only logarithmically

# %%
header("6. Genome-wide testing: the cost of multiplicity")

print(f"  {'G tests':>12}{'alpha/G':>12}{'z threshold':>13}{'n inflation':>13}")
z_base = st.norm.ppf(1 - 0.05 / 2) + st.norm.ppf(0.8)
for G in [1, 20, 1000, 20000, 1_000_000]:
    a = 0.05 / G
    z = st.norm.ppf(1 - a / 2) + st.norm.ppf(0.8)
    print(f"  {G:>12,}{a:>12.2e}{st.norm.ppf(1-a/2):>13.3f}"
          f"{(z/z_base)**2:>13.2f}x")
print("\n  Going from 1 test to a million multiplies the required n by only ~5,")
print("  because n grows with (z_{1-alpha/2G})^2 and z grows like sqrt(log G).")

# %% [markdown]
# ## 7. Simulation-based power, eq. (9.10)
#
# For anything with clustering, filtering, or a multi-step pipeline, simulate.
# Run the **exact** analysis you will use, including the multiplicity step.

# %%
header("7. Simulation-based power for a clustered design (9.10)")

def simulate_power_clustered(n_donors_per_arm, cells_per_donor, effect,
                             sigma_b=1.0, sigma_e=1.0, n_sim=600, seed=0):
    """Pseudobulk analysis of a two-arm, donor-clustered study."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        nb = 2 * n_donors_per_arm
        b = rng.normal(0, sigma_b, nb)
        mu = b + np.r_[np.full(n_donors_per_arm, effect),
                       np.zeros(n_donors_per_arm)]
        cells = mu[:, None] + rng.normal(0, sigma_e, (nb, cells_per_donor))
        dm = cells.mean(axis=1)                       # pseudobulk
        hits += st.ttest_ind(dm[:n_donors_per_arm], dm[n_donors_per_arm:],
                             equal_var=False).pvalue < 0.05
    p = hits / n_sim
    return p, np.sqrt(p * (1 - p) / n_sim)            # power +/- Monte Carlo SE


print(f"  effect = 1.0 sigma_b, sigma_e = 1.0")
print(f"  {'donors/arm':>12}{'cells/donor':>13}{'power':>9}{'MC SE':>9}")
for nd in [3, 5, 8, 12]:
    for mc in [20, 500]:
        pw, se = simulate_power_clustered(nd, mc, 1.0, seed=nd * 10 + mc)
        print(f"  {nd:>12}{mc:>13}{pw:>9.3f}{se:>9.3f}")

print("\n  Compare rows within a donor count: going from 20 to 500 cells per")
print("  donor changes power very little. Compare down the donor column: it")
print("  changes power a lot. Eq. (1.6), demonstrated end to end.")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ds = np.linspace(0.1, 2.0, 100)
for n in [4, 8, 16, 32]:
    axes[0].plot(ds, [power_t(d, n) for d in ds], label=f"n={n}/group")
axes[0].axhline(0.8, color="k", ls="--")
axes[0].set_xlabel("standardised effect $d$"); axes[0].set_ylabel("power")
axes[0].set_title("Eq. (9.4): exact t power", fontsize=9)
axes[0].legend(fontsize=7)

mus = np.logspace(0, 4, 100)
for fc in [1.5, 2.0, 3.0]:
    axes[1].loglog(mus, [n_rnaseq(fc, m, PHI) for m in mus], label=f"FC={fc}")
axes[1].axvline(1 / PHI, color="k", ls=":", label="$\\mu=1/\\phi$")
axes[1].set_xlabel("mean count $\\mu$ (sequencing depth)")
axes[1].set_ylabel("samples per group")
axes[1].set_title("Eq. (9.9): depth saturates", fontsize=9)
axes[1].legend(fontsize=7)

grid = np.arange(5, 4000)
for sb, se, c in [(1.0, 1.0, "C0"), (0.5, 2.0, "C1"), (2.0, 1.0, "C2")]:
    n_grid = BUDGET / (C_DONOR + grid * C_CELL)
    v = variance_at(n_grid, grid, sb, se)
    axes[2].semilogx(grid, v / v.min(), color=c,
                     label=f"$\\sigma_b$={sb}, $\\sigma_e$={se}")
    axes[2].axvline(optimal_cells(sb, se, C_DONOR, C_CELL), color=c, ls=":")
axes[2].set_xlabel("cells per donor, $m$")
axes[2].set_ylabel("Var(mean) / minimum")
axes[2].set_title("Eq. (9.7): optimal allocation", fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "power.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/power.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 9)
#
# 1. Power analysis is prospective. Report the **minimum detectable effect** for
#    the design you actually ran, never post-hoc power.
# 2. State every input ($\delta$, $\sigma$ or $\phi$, $\rho$, $\alpha$, $G$) and
#    show a sensitivity curve: pilot variance estimates are biased downward.
# 3. Spend on biological replicates before depth, cells, or technical
#    replicates.
# 4. For clustering or non-standard models, simulate (9.10).
#
# **Next:** `10_correlation_and_dependence.py`
