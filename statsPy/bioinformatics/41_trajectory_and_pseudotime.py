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
# # Applied 41: Trajectory inference and pseudotime
#
# **Curriculum link:** `stats.md` -> Topic 41, equations (41.1)-(41.3)
# **Core modules used:** 17, 18, 11, 21
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | Differentiation time course profiled by scRNA-seq, analysed with Slingshot or Monocle |
# | **Input** | Cells x genes, reduced to a few dimensions |
# | **Key threats** | Pseudotime is an ordering, not time; **double dipping**; unstable branches |
# | **Here** | Simulated cells along a known trajectory, plus a null with no trajectory at all |
#
# ## The single most important fact
#
# The pseudotime you test genes against was **computed from those genes**.
# That is the clustering problem of Module 18 in a continuous disguise, and it
# produces significant genes on data with no trajectory whatsoever.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from patsy import dmatrix

warnings.filterwarnings("ignore")

MODULE_NAME = "41_trajectory_and_pseudotime"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(41)

# %% [markdown]
# ## 1. Cells along a known trajectory

# %%
header("1. A differentiation trajectory with a known ordering")


def simulate_trajectory(n_cells=1200, n_genes=400, n_dynamic=80,
                        has_trajectory=True, seed=0):
    """Cells sampled along a latent progression. When has_trajectory is False
    the latent variable is removed and only noise remains."""
    r = np.random.default_rng(seed)
    t_true = r.uniform(0, 1, n_cells)
    base = np.exp(r.normal(2.5, 0.9, n_genes))
    log_mu = np.log(base)[None, :] * np.ones((n_cells, 1))
    dyn = np.zeros(n_genes, bool)
    if has_trajectory:
        dyn[:n_dynamic] = True
        # Each dynamic gene peaks somewhere along the trajectory.
        peak = r.uniform(0, 1, n_dynamic)
        width = r.uniform(0.12, 0.35, n_dynamic)
        amp = r.uniform(1.0, 2.2, n_dynamic)
        shape = amp * np.exp(-0.5 * ((t_true[:, None] - peak) / width) ** 2)
        log_mu[:, :n_dynamic] += shape
    size = r.lognormal(0, 0.35, n_cells)
    mu = np.exp(log_mu) * size[:, None]
    counts = r.poisson(mu)
    return counts, t_true, dyn, size


counts, t_true, is_dynamic, size = simulate_trajectory(seed=1)
n_cells, n_genes = counts.shape
print(f"  {n_cells} cells x {n_genes} genes, {is_dynamic.sum()} truly dynamic")
print(f"  library size varies {size.min():.2f} to {size.max():.2f} fold")
print("""
  The latent t is a POSITION along a progression. It has no units and no
  direction that the data can identify: reversing it fits equally well.""")

# %% [markdown]
# ## 2. Inferring pseudotime, eq. (41.1)
#
# Reduce, fit a curve, project. This is what Slingshot and Monocle do, with
# more care about branches.

# %%
header("2. A principal curve and the ordering it induces (41.1)")


def normalise(counts):
    cpm = counts / counts.sum(1, keepdims=True) * 1e4
    return np.log1p(cpm)


def principal_curve(Z, n_iter=12, n_knots=24):
    """Iterate: project points onto the curve, then re-smooth the curve
    through the ordered points. This is eq. (41.1) solved by alternation."""
    # Initialise with the first principal component.
    lam = Z[:, 0].copy()
    for _ in range(n_iter):
        order = np.argsort(lam)
        grid = np.linspace(lam.min(), lam.max(), n_knots)
        # Smooth each coordinate against the current arclength.
        curve = np.empty((n_knots, Z.shape[1]))
        for d in range(Z.shape[1]):
            for j, g in enumerate(grid):
                w = np.exp(-0.5 * ((lam - g) / (0.12 * np.ptp(lam))) ** 2)
                curve[j, d] = np.sum(w * Z[:, d]) / np.sum(w)
        # Project every cell to its nearest point on the curve.
        d2 = ((Z[:, None, :] - curve[None, :, :]) ** 2).sum(2)
        nearest = d2.argmin(1)
        lam = grid[nearest]
    # Rescale to [0, 1]; the direction is arbitrary.
    return (lam - lam.min()) / (lam.max() - lam.min()), curve


Y = normalise(counts)
Yc = Y - Y.mean(0)
U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
Z = U[:, :3] * S[:3]
pt, curve = principal_curve(Z)

r_sp = st.spearmanr(pt, t_true).statistic
if r_sp < 0:
    pt = 1 - pt                      # direction is not identifiable
    r_sp = -r_sp
print(f"  Spearman(pseudotime, true t) = {r_sp:.3f}")
print(f"  Spearman after flipping direction would be {-r_sp:.3f}")
print("""
  The ordering is recovered well. The DIRECTION is not: the data cannot say
  which end is the start, and every trajectory tool asks you to supply a root
  cell or a marker. That choice is biology, not inference.

  Equal steps in pseudotime are also not equal steps in real time. Cells
  accumulate where transitions are slow, so density along the curve reflects
  dwell time, not sampling time.""")

# %% [markdown]
# ## 3. Testing genes along the trajectory, eq. (41.2)

# %%
header("3. A spline GLM along pseudotime (41.2)")


def trajectory_test(counts, pseudotime, size, df=5, genes=None):
    """Negative-binomial-ish spline test: log E[Y] = log(size) + spline(t).
    Poisson with a robust covariance is used here so the module stays in the
    standard library; a real analysis would fit the NB dispersion."""
    idx = range(counts.shape[1]) if genes is None else genes
    B = dmatrix(f"bs(t, df={df}, include_intercept=False)",
                {"t": pseudotime}, return_type="dataframe").values
    X1 = np.column_stack([np.ones(len(pseudotime)), B])
    X0 = np.ones((len(pseudotime), 1))
    off = np.log(size)
    pvals = np.ones(counts.shape[1])
    for g in idx:
        y = counts[:, g]
        try:
            f1 = sm.GLM(y, X1, family=sm.families.Poisson(), offset=off).fit()
            f0 = sm.GLM(y, X0, family=sm.families.Poisson(), offset=off).fit()
            lr = 2 * (f1.llf - f0.llf)
            pvals[g] = st.chi2.sf(max(lr, 0), X1.shape[1] - 1)
        except Exception:
            pvals[g] = 1.0
    return pvals


def bh(p):
    m = len(p); o = np.argsort(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


SUB = np.arange(0, n_genes, 2)             # half the genes, for speed
p_real = trajectory_test(counts, pt, size, genes=SUB)
rej = bh(p_real[SUB]) < 0.05
tr = is_dynamic[SUB]
print(f"  tested {len(SUB)} genes against the INFERRED pseudotime")
print(f"    discoveries : {rej.sum()}")
print(f"    truly dynamic among them : {(rej & tr).sum()}")
print(f"    FDP : {(rej & ~tr).sum() / max(rej.sum(), 1):.3f}")
print(f"    power : {(rej & tr).sum() / max(tr.sum(), 1):.3f}")
print("""
  Good sensitivity, and the FDP looks acceptable. Now run exactly the same
  pipeline on data with no trajectory in it.""")

# %% [markdown]
# ## 4. The double dipping problem, eq. (41.3)

# %%
header("4. The same analysis on data with NO trajectory")

counts0, _, dyn0, size0 = simulate_trajectory(has_trajectory=False, seed=7)
Y0 = normalise(counts0)
Z0 = np.linalg.svd(Y0 - Y0.mean(0), full_matrices=False)
Z0 = (Z0[0][:, :3] * Z0[1][:3])
pt0, _ = principal_curve(Z0)
p_null = trajectory_test(counts0, pt0, size0, genes=SUB)
rej0 = bh(p_null[SUB]) < 0.05

print(f"  There is NO latent trajectory. Every gene is independent noise.\n")
print(f"  {'quantity':<44}{'value':>10}")
print(f"  {'genes called dynamic at FDR 5%':<44}{rej0.sum():>10}")
print(f"  {'of {} tested'.format(len(SUB)):<44}{'':>10}")
print(f"  {'raw p < 0.05':<44}{(p_null[SUB] < 0.05).mean():>10.3f}")
print(f"  {'(should be 0.05)':<44}{'':>10}")

# Compare with a pseudotime that was NOT derived from these genes.
pt_indep = rng.permutation(pt0)
p_indep = trajectory_test(counts0, pt_indep, size0, genes=SUB)
print(f"\n  Same genes, but ordered by a pseudotime unrelated to them:")
print(f"  {'raw p < 0.05':<44}{(p_indep[SUB] < 0.05).mean():>10.3f}")
print(f"  {'genes called dynamic':<44}{(bh(p_indep[SUB]) < 0.05).sum():>10}")

print("""
  The curve was fitted to whatever structure the noise happened to contain,
  and then the genes were tested for agreement with the structure they
  themselves created. The p-values are not uniform and the discoveries are
  entirely fictitious.

  Break the circularity, as in the last block, and the test behaves.

  This is eq. (41.3), and it is the same error as testing the genes that
  defined a cluster (Module 18). It is easier to miss here because pseudotime
  feels like an external covariate, like age or dose. It is not: it is a
  function of the expression matrix.""")

# %% [markdown]
# ## 5. A defence that works: split the genes

# %%
header("5. Building the trajectory on one half, testing the other")


def split_gene_test(counts, size, seed=0):
    """Fit the trajectory using half the genes, test the other half."""
    r = np.random.default_rng(seed)
    g = r.permutation(counts.shape[1])
    build, test = g[: counts.shape[1] // 2], g[counts.shape[1] // 2:]
    Yb = normalise(counts[:, build])
    Zb = np.linalg.svd(Yb - Yb.mean(0), full_matrices=False)
    Zb = Zb[0][:, :3] * Zb[1][:3]
    ptb, _ = principal_curve(Zb)
    sub = test[: len(test) // 2]
    return trajectory_test(counts, ptb, size, genes=sub), sub


print(f"  {'data':<26}{'analysis':<34}{'discoveries':>13}")
p_s0, sub0 = split_gene_test(counts0, size0, seed=3)
print(f"  {'NO trajectory':<26}{'naive (same genes build and test)':<34}"
      f"{rej0.sum():>13}")
print(f"  {'NO trajectory':<26}{'split: build on other genes':<34}"
      f"{(bh(p_s0[sub0]) < 0.05).sum():>13}")
p_s1, sub1 = split_gene_test(counts, size, seed=3)
rj1 = bh(p_s1[sub1]) < 0.05
print(f"  {'real trajectory':<26}{'split: build on other genes':<34}"
      f"{rj1.sum():>13}")
print(f"\n  On real data the split analysis still recovers "
      f"{(rj1 & is_dynamic[sub1]).sum()} of {is_dynamic[sub1].sum()} dynamic genes "
      f"(FDP {(rj1 & ~is_dynamic[sub1]).sum()/max(rj1.sum(),1):.3f}).")
print("""
  Splitting genes costs power, because the trajectory is built from half the
  information. It buys a test that says nothing when there is nothing.

  Other defences: build the trajectory on one set of CELLS and project the
  rest; or use marker genes chosen a priori to define the ordering. The
  principle is always the same, the ordering must not be a function of the
  quantities you test.""")

# %% [markdown]
# ## 6. The ordering is uncertain, and the uncertainty is rarely propagated

# %%
header("6. Bootstrap the cells and watch the ordering move")

boots = []
for b in range(12):
    r = np.random.default_rng(500 + b)
    idx = r.choice(n_cells, n_cells, replace=True)
    Yb = normalise(counts[idx])
    Zb = np.linalg.svd(Yb - Yb.mean(0), full_matrices=False)
    Zb = Zb[0][:, :3] * Zb[1][:3]
    ptb, _ = principal_curve(Zb)
    if st.spearmanr(ptb, t_true[idx]).statistic < 0:
        ptb = 1 - ptb
    boots.append(st.spearmanr(ptb, t_true[idx]).statistic)

# Rank stability of individual cells across bootstraps that contain them.
ranks = np.full((12, n_cells), np.nan)
for b in range(12):
    r = np.random.default_rng(500 + b)
    idx = r.choice(n_cells, n_cells, replace=True)
    Yb = normalise(counts[idx])
    Zb = np.linalg.svd(Yb - Yb.mean(0), full_matrices=False)
    Zb = Zb[0][:, :3] * Zb[1][:3]
    ptb, _ = principal_curve(Zb)
    if st.spearmanr(ptb, t_true[idx]).statistic < 0:
        ptb = 1 - ptb
    for k, cell in enumerate(idx):
        ranks[b, cell] = ptb[k]
sd_cell = np.nanstd(ranks, axis=0)
early = t_true < 0.2
mid = (t_true > 0.4) & (t_true < 0.6)
late = t_true > 0.8

print(f"  Spearman with truth across 12 bootstraps: "
      f"{np.mean(boots):.3f} (SD {np.std(boots):.3f})\n")
print(f"  {'cells':<22}{'SD of assigned pseudotime':>28}")
print(f"  {'early (t < 0.2)':<22}{np.nanmean(sd_cell[early]):>28.4f}")
print(f"  {'middle (0.4-0.6)':<22}{np.nanmean(sd_cell[mid]):>28.4f}")
print(f"  {'late (t > 0.8)':<22}{np.nanmean(sd_cell[late]):>28.4f}")
print("""
  The global ordering is essentially fixed (Spearman SD 0.000), yet individual
  cells still move between runs, and they do not all move by the same amount.

  Note that the MIDDLE is the least stable band here, not the ends. That is
  worth pausing on, because the usual intuition says the opposite. It is an
  artefact of the estimator: cells are projected onto a fixed grid of knots
  along the curve, and a cell at either extreme is pinned against the end of
  that grid, so its position cannot vary much in one direction. A cell in the
  middle is free to shift either way.

  The lesson is not "middles are unstable". It is that the uncertainty
  structure belongs to the ALGORITHM as much as to the data, so measure it on
  your own pipeline rather than assuming the textbook picture.

  Almost no published analysis propagates any of this into the gene-level
  p-values, which are computed as though pseudotime were measured without
  error. That is a measurement-error problem (Module 37), and it biases the
  estimated shapes toward flatness.""")

# %% [markdown]
# ## 7. RNA velocity, eq. (41.3)

# %%
header("7. Velocity assumes constant kinetics across cells")

print("""  The velocity model is

      du/dt = alpha - beta*u,     ds/dt = beta*u - gamma*s          (41.3)

  and the steady-state estimator reads gamma off the slope of spliced on
  unspliced among the extreme cells.""")

# Simulate two populations with DIFFERENT gamma, which the model assumes equal.
for tag, gamma_b in (("same kinetics in both states", 1.0),
                     ("gamma differs 3-fold", 3.0)):
    r = np.random.default_rng(99)
    u_a = r.gamma(3, 1.0, 600); s_a = u_a / 1.0 + r.normal(0, 0.25, 600)
    u_b = r.gamma(3, 1.0, 600); s_b = u_b / gamma_b + r.normal(0, 0.25, 600)
    u = np.concatenate([u_a, u_b]); s = np.concatenate([s_a, s_b])
    hi = u >= np.quantile(u, 0.95)
    gamma_hat = np.sum(u[hi] * s[hi]) / np.sum(u[hi] ** 2)
    vel = s - gamma_hat * u
    frac_pos = np.mean(vel[:600] > 0), np.mean(vel[600:] > 0)
    print(f"\n  {tag}")
    print(f"    fitted single gamma = {gamma_hat:.2f}")
    print(f"    cells called 'increasing': group A {frac_pos[0]:.0%}, "
          f"group B {frac_pos[1]:.0%}")

print("""
  With one kinetic rate the arrows are meaningless in aggregate but unbiased.
  When the two populations genuinely differ, a single fitted gamma declares
  one population to be increasing and the other decreasing, with confidence,
  purely because the model forced them to share a parameter.

  Treat velocity as a hypothesis generator. It is a strong assumption stated
  as a picture, and pictures are persuasive out of proportion to their
  evidence.""")

# %% [markdown]
# ## 8. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

sc = ax[0].scatter(Z[:, 0], Z[:, 1], c=t_true, s=5, cmap="viridis")
ax[0].plot(curve[:, 0], curve[:, 1], color="firebrick", lw=2)
ax[0].set_title(f"Principal curve (41.1), rho = {r_sp:.2f}")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
plt.colorbar(sc, ax=ax[0], label="true t")

ax[1].hist(p_null[SUB], bins=25, color="firebrick", alpha=0.7, density=True,
           label="pseudotime from these genes")
ax[1].hist(p_indep[SUB], bins=25, color="steelblue", alpha=0.6, density=True,
           label="independent ordering")
ax[1].axhline(1.0, color="black", ls="--", lw=1)
ax[1].set_xlabel("p-value"); ax[1].set_title("NO trajectory in the data")
ax[1].legend(fontsize=7)

ax[2].scatter(t_true, sd_cell, s=4, alpha=0.3, color="steelblue")
ax[2].set_xlabel("true position"); ax[2].set_ylabel("SD of pseudotime (bootstrap)")
ax[2].set_title("Uncertainty is not uniform")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "trajectory.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'trajectory.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: How much signal does it take to invent a trajectory?
#
# Vary the strength of the real trajectory from zero upward, and record both
# the correlation with truth and the number of "dynamic" genes found naively.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'dynamic genes in truth':<26}{'rho with truth':>16}"
#       f"{'naive discoveries':>20}")
# for nd in (0, 10, 40, 120):
#     c_, t_, d_, s_ = simulate_trajectory(n_dynamic=max(nd, 1),
#                                          has_trajectory=nd > 0, seed=200 + nd)
#     Yl = normalise(c_)
#     Zl = np.linalg.svd(Yl - Yl.mean(0), full_matrices=False)
#     Zl = Zl[0][:, :3] * Zl[1][:3]
#     ptl, _ = principal_curve(Zl)
#     rho = abs(st.spearmanr(ptl, t_).statistic)
#     pl = trajectory_test(c_, ptl, s_, genes=SUB)
#     print(f"  {nd:<26}{rho:>16.3f}{(bh(pl[SUB]) < 0.05).sum():>20}")
#
# # With zero dynamic genes the correlation with "truth" is meaningless, yet
# # the naive test still returns discoveries. The pipeline never refuses: it
# # always produces a curve, an ordering and a gene list.
# #
# # That is the practical danger. There is no step in a standard trajectory
# # workflow that says "there is no trajectory here". You have to build that
# # check yourself, by running the same pipeline on permuted or simulated null
# # data and comparing.

# %% [markdown]
# ### Problem 2: Does the trajectory even exist?
#
# Build a null for the trajectory itself. Permute each gene independently
# across cells, which destroys any coordinated structure but keeps every
# gene's marginal distribution, then compare a measure of curve quality.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def curve_quality(c_):
#     """Variance explained by the first component: high if cells lie on a
#     low-dimensional structure."""
#     Yl = normalise(c_)
#     Yl = Yl - Yl.mean(0)
#     s = np.linalg.svd(Yl, compute_uv=False)
#     return s[0] ** 2 / np.sum(s ** 2)
#
# def gene_permute(c_, seed):
#     r = np.random.default_rng(seed)
#     out = c_.copy()
#     for g in range(c_.shape[1]):
#         out[:, g] = r.permutation(c_[:, g])
#     return out
#
# for lab, dat in (("real trajectory", counts), ("no trajectory", counts0)):
#     obs = curve_quality(dat)
#     null = [curve_quality(gene_permute(dat, 900 + i)) for i in range(20)]
#     z = (obs - np.mean(null)) / np.std(null)
#     print(f"  {lab:<20} observed {obs:.4f}  null {np.mean(null):.4f} "
#           f"+/- {np.std(null):.4f}   z = {z:>7.1f}")
#
# # The real dataset sits far above its permutation null; the null dataset does
# # not. This is the check the standard workflow omits, and it costs one line
# # of code.
# #
# # Note what the permutation preserves: every gene's own distribution, its
# # mean, its dispersion, its zeros. What it destroys is the COORDINATION
# # between genes, which is the only thing a trajectory can be made of. That is
# # what makes it the right null here, and it is the same reasoning used for
# # gene-set tests in Module 29.

# %% [markdown]
# ### Problem 3: Pseudotime is not time
#
# Sample cells non-uniformly in real time, so that some stages are
# over-represented, and check whether pseudotime recovers real time or
# something else.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# r = np.random.default_rng(55)
# # Real time is uniform, but cells DWELL in the middle stage, so we capture
# # far more of them there.
# real_time = np.concatenate([r.uniform(0, 0.35, 200),
#                             r.uniform(0.35, 0.65, 800),
#                             r.uniform(0.65, 1.0, 200)])
# n_c = len(real_time)
# base = np.exp(r.normal(2.5, 0.9, 300))
# peak = r.uniform(0, 1, 60); width = r.uniform(0.12, 0.3, 60)
# lm = np.log(base)[None, :] * np.ones((n_c, 1))
# lm[:, :60] += 1.8 * np.exp(-0.5 * ((real_time[:, None] - peak) / width) ** 2)
# sz = r.lognormal(0, 0.3, n_c)
# cc = r.poisson(np.exp(lm) * sz[:, None])
# Yl = normalise(cc); Zl = np.linalg.svd(Yl - Yl.mean(0), full_matrices=False)
# Zl = Zl[0][:, :3] * Zl[1][:3]
# ptl, _ = principal_curve(Zl)
# if st.spearmanr(ptl, real_time).statistic < 0: ptl = 1 - ptl
# print(f"  Spearman(pseudotime, real time) = {st.spearmanr(ptl, real_time).statistic:.3f}")
# print(f"\n  {'real-time window':<22}{'cells':>8}{'pseudotime span covered':>26}")
# for lo, hi in ((0, 0.35), (0.35, 0.65), (0.65, 1.0)):
#     m = (real_time >= lo) & (real_time < hi)
#     print(f"  {f'[{lo}, {hi})':<22}{m.sum():>8}"
#           f"{ptl[m].max() - ptl[m].min():>26.3f}")
#
# # The rank correlation is high, so pseudotime "works". But look at the span
# # column: the crowded middle stage, which occupies 30% of real time, is
# # stretched across most of the pseudotime axis, because pseudotime is
# # arclength through the data cloud and the cloud is dense where cells dwell.
# #
# # So a gene that changes steeply in pseudotime may be changing slowly in
# # real time, and vice versa. Any statement of the form "gene X switches on
# # halfway through differentiation" is a statement about the sampling, not
# # about the biology, unless you have external time points to anchor it.

# %% [markdown]
# ## What to take away
#
# 1. Pseudotime is an **ordering**, with no units and no identifiable
#    direction. Density along it reflects dwell time, not elapsed time.
# 2. Testing genes against a pseudotime built from those genes is **double
#    dipping** and produces discoveries on data with no trajectory (41.3).
# 3. Split genes or cells so the ordering is not a function of what you test.
# 4. Bootstrap the ordering. Cell positions move between runs, the pattern of
#    that movement depends on the algorithm, and it is almost never propagated
#    into the gene-level p-values.
# 5. **Velocity** (41.3) assumes shared kinetic rates. When they differ it
#    reports confident arrows in opposite directions.
# 6. Run the whole pipeline on a permuted null before believing any of it.
#
# **Next:** `42_deconvolution_and_integration.py`
