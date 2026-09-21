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
# # Module 36: Multiple testing beyond Benjamini-Hochberg
#
# **Curriculum link:** `stats.md` -> Topic 36, equations (36.1)-(36.6)
# **Assumes:** Module 08.
#
# ## What you will learn
#
# 1. **Weighted BH (36.1)** and why the weights must not come from the p-values.
# 2. **IHW (36.2)**: independent filtering is a blunt weight, and a smooth one
#    recovers the power the blunt one throws away.
# 3. What BH actually promises under dependence, and what it does not.
# 4. **E-values (36.3)** and **e-BH (36.4)**: FDR control under *any* dependence.
# 5. **Knockoffs (36.5)-(36.6)**: FDR control when the features are correlated
#    and no p-value is available.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

warnings.filterwarnings("ignore")

MODULE_NAME = "36_modern_multiple_testing"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(25)
ALPHA = 0.05


def bh(p, alpha=ALPHA):
    """Benjamini-Hochberg. Returns a boolean rejection vector."""
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


def weighted_bh(p, w, alpha=ALPHA):
    """Weighted BH, eq. (36.1). Weights must average to 1 and be independent
    of p. A weight of zero removes the hypothesis entirely."""
    w = np.asarray(w, float)
    w = w / w.mean()
    pw = np.where(w > 0, p / np.maximum(w, 1e-12), np.inf)
    return bh(pw, alpha)


def fdp_tpr(rej, truth):
    return ((rej & ~truth).sum() / max(rej.sum(), 1),
            (rej & truth).sum() / max(truth.sum(), 1))


# %% [markdown]
# ## 1. A screen where power depends on a covariate
#
# In RNA-seq, a gene's mean expression predicts how much power it has, and is
# independent of its p-value when the gene is null. That is exactly the
# situation weighting is built for.

# %%
header("1. A screen in which power varies with an observable covariate")


def simulate_screen(m=8000, pi1=0.10, seed=0):
    """Effects exist only among well-measured features; the covariate is the
    measurement precision, which is knowable without looking at the outcome."""
    r = np.random.default_rng(seed)
    # Covariate: mean expression, spanning four orders of magnitude.
    mu = np.exp(r.normal(3.0, 1.8, m))
    se = 1.0 / np.sqrt(mu) + 0.05          # precision improves with expression
    is_de = r.random(m) < pi1
    effect = np.where(is_de, r.normal(0, 0.55, m), 0.0)
    z = (effect + r.normal(0, se, m)) / se
    p = 2 * st.norm.sf(np.abs(z))
    return p, is_de, mu


p, truth, cov = simulate_screen(seed=1)
print(f"  {len(p):,} hypotheses, {truth.sum():,} truly non-null "
      f"({truth.mean():.1%})")
print(f"  covariate (mean expression) spans {cov.min():.1f} to {cov.max():,.0f}")
print("\n  power is not spread evenly across the covariate:")
print(f"  {'expression quartile':<22}{'features':>10}{'true DE':>10}{'p<0.05':>10}")
qs = np.quantile(cov, [0, .25, .5, .75, 1.0])
for i in range(4):
    m_ = (cov >= qs[i]) & (cov <= qs[i + 1])
    print(f"  {f'Q{i+1}':<22}{m_.sum():>10,}{truth[m_].sum():>10,}"
          f"{(p[m_] < 0.05).mean():>10.3f}")
print("""
  The covariate is informative about POWER but, for a null feature, carries no
  information about the p-value. That is the condition eq. (36.1) needs.""")

# %% [markdown]
# ## 2. Independent filtering is a weight of zero or one

# %%
header("2. BH, filtering, and what filtering throws away")

print(f"  {'procedure':<40}{'discov.':>9}{'FDP':>8}{'power':>8}")
rej_bh = bh(p)
f1, t1 = fdp_tpr(rej_bh, truth)
print(f"  {'BH on everything':<40}{rej_bh.sum():>9}{f1:>8.3f}{t1:>8.3f}")

for q in (0.2, 0.4, 0.6):
    keep = cov >= np.quantile(cov, q)
    rej = weighted_bh(p, keep.astype(float))
    f_, t_ = fdp_tpr(rej, truth)
    print(f"  {f'independent filtering, drop bottom {q:.0%}':<40}"
          f"{rej.sum():>9}{f_:>8.3f}{t_:>8.3f}")

print("""
  Filtering is weighted BH with weights in {0, c}: a hypothesis is either in
  the family or removed from it. It helps, and the FDP stays controlled,
  because the filter statistic is independent of the p-value under the null.

  But a hard cut throws away the gradient. A feature just below the threshold
  gets weight zero; one just above gets full weight. Nothing about the biology
  changes at that line.""")

# %% [markdown]
# ## 3. IHW: learn a smooth weight, eq. (36.2)
#
# Stratify on the covariate, estimate how much signal each stratum carries, and
# weight accordingly. The estimate for each fold uses only the *other* folds,
# which is what keeps the weights independent of the p-values they act on.

# %%
header("3. Independent hypothesis weighting (36.2)")


def storey_pi0(pvals, lam=0.5):
    """Proportion of nulls in a set of p-values (Topic 8)."""
    if len(pvals) == 0:
        return 1.0
    return min(1.0, np.mean(pvals > lam) / (1 - lam))


def ihw(p, covariate, alpha=ALPHA, n_strata=8, n_folds=5, seed=0):
    """A simplified IHW: weight each stratum by its estimated signal fraction,
    learned by cross-weighting so the weights never see their own p-values."""
    r = np.random.default_rng(seed)
    m = len(p)
    edges = np.quantile(covariate, np.linspace(0, 1, n_strata + 1))
    edges[-1] += 1e-9
    stratum = np.clip(np.digitize(covariate, edges) - 1, 0, n_strata - 1)
    fold = r.integers(0, n_folds, m)
    w = np.ones(m)
    for k in range(n_folds):
        other = fold != k
        # Signal fraction per stratum, estimated WITHOUT this fold.
        sig = np.array([1.0 - storey_pi0(p[other & (stratum == g)])
                        for g in range(n_strata)])
        sig = np.maximum(sig, 1e-3)
        sig = sig / sig.mean()                 # eq. (36.1): weights average 1
        w[fold == k] = sig[stratum[fold == k]]
    return weighted_bh(p, w, alpha), w, stratum


rej_ihw, w_ihw, stratum = ihw(p, cov, seed=2)
f_i, t_i = fdp_tpr(rej_ihw, truth)
best_filter = max(
    (fdp_tpr(weighted_bh(p, (cov >= np.quantile(cov, q)).astype(float)), truth)[1], q)
    for q in (0.2, 0.4, 0.6))

print(f"  {'procedure':<40}{'discov.':>9}{'FDP':>8}{'power':>8}")
print(f"  {'BH':<40}{rej_bh.sum():>9}{f1:>8.3f}{t1:>8.3f}")
kb = weighted_bh(p, (cov >= np.quantile(cov, best_filter[1])).astype(float))
fb, tb = fdp_tpr(kb, truth)
print(f"  {f'best filter (drop bottom {best_filter[1]:.0%})':<40}"
      f"{kb.sum():>9}{fb:>8.3f}{tb:>8.3f}")
print(f"  {'IHW (36.2)':<40}{rej_ihw.sum():>9}{f_i:>8.3f}{t_i:>8.3f}")

print(f"\n  learned weights by covariate stratum (low to high expression):")
wm = [w_ihw[stratum == g].mean() for g in range(8)]
print("   " + "".join(f"{x:7.2f}" for x in wm))
print(f"""
  The weights rise with expression, because that is where the signal is. No
  hypothesis is discarded; the low-expression strata simply need stronger
  evidence. IHW finds more than BH and more than the best hard filter, at the
  same realised FDP.""")

# %% [markdown]
# ## 4. The trap: weights that peek at the p-values

# %%
header("4. Weights must not come from the p-values they weight")

# A tempting shortcut: weight by how significant each feature already looks.
cheat_w = 1.0 / np.maximum(p, 1e-8)
rej_cheat = weighted_bh(p, cheat_w)
f_c, t_c = fdp_tpr(rej_cheat, truth)
print(f"  {'weights from the covariate (IHW)':<42}{rej_ihw.sum():>9}{f_i:>8.3f}")
print(f"  {'weights from the p-values themselves':<42}{rej_cheat.sum():>9}{f_c:>8.3f}")

# And the same on data with NO signal at all.
p0, truth0, cov0 = simulate_screen(pi1=0.0, seed=7)
print(f"\n  On a screen with NO non-null features at all:")
print(f"  {'BH':<42}{bh(p0).sum():>9} discoveries")
print(f"  {'IHW':<42}{ihw(p0, cov0, seed=3)[0].sum():>9} discoveries")
print(f"  {'weights from the p-values':<42}"
      f"{weighted_bh(p0, 1.0 / np.maximum(p0, 1e-8)).sum():>9} discoveries")
print("""
  Weighting by the p-value inflates the FDP, and on a pure null it manufactures
  discoveries from nothing. The rule in eq. (36.1) is not a technicality: the
  weights carry no information about the outcome, or the guarantee is void.

  A covariate is admissible if, for a NULL feature, it tells you nothing about
  the p-value. Mean expression, feature variance and prior published evidence
  qualify. The observed effect size does not.""")

# %% [markdown]
# ## 5. What BH promises under dependence
#
# BH controls the **expected** FDP. Under dependence the expectation still
# holds but the spread widens, and you only ever run the experiment once.

# %%
header("5. Under dependence, the FDP becomes a lottery")


def correlated_screen(m=2000, pi1=0.1, rho=0.0, n_blocks=40, seed=0):
    """Equicorrelated blocks, the structure genes in a pathway really have."""
    r = np.random.default_rng(seed)
    per = m // n_blocks
    z = np.empty(m)
    for b in range(n_blocks):
        shared = r.normal(0, 1)
        idx = slice(b * per, (b + 1) * per)
        z[idx] = np.sqrt(rho) * shared + np.sqrt(1 - rho) * r.normal(0, 1, per)
    is_de = r.random(m) < pi1
    z = z + is_de * r.normal(3.0, 0.5, m)
    return 2 * st.norm.sf(np.abs(z)), is_de


print(f"  {'rho':<8}{'mean FDP':>11}{'SD of FDP':>12}{'FDP > 0.10':>12}"
      f"{'FDP > 0.20':>12}")
for rho in (0.0, 0.3, 0.7):
    fdps = []
    for i in range(300):
        pc, tc = correlated_screen(rho=rho, seed=1000 + i)
        fdps.append(fdp_tpr(bh(pc), tc)[0])
    fdps = np.array(fdps)
    print(f"  {rho:<8.1f}{fdps.mean():>11.3f}{fdps.std():>12.3f}"
          f"{(fdps > 0.10).mean():>12.1%}{(fdps > 0.20).mean():>12.1%}")

print("""
  The mean column stays controlled: BH is valid here, because equicorrelated
  positive dependence satisfies PRDS. Read the other three columns.

  As dependence grows, the FDP of a SINGLE experiment becomes wildly variable.
  At rho = 0.7 a substantial fraction of runs exceed twice the nominal rate.
  "FDR = 5%" is a statement about the average over experiments you will never
  run. Your gene list is one draw from that distribution.""")

# %% [markdown]
# ## 6. E-values and e-BH, eq. (36.3)-(36.4)
#
# An e-value is calibrated by its **mean** rather than its tail. That single
# change buys FDR control under arbitrary dependence.

# %%
header("6. e-BH: valid under any dependence (36.3)-(36.4)")


def p_to_e(p, kappa=0.5):
    """A standard calibrator. Under the null p ~ U(0,1), so
    E[e] = int_0^1 kappa p^(kappa-1) dp = 1, satisfying eq. (36.3)."""
    return kappa * np.power(np.maximum(p, 1e-300), kappa - 1.0)


def ebh(e, alpha=ALPHA):
    """e-BH, eq. (36.4)."""
    m = len(e)
    order = np.argsort(-e)
    es = e[order]
    k_ok = np.nonzero(m / (np.arange(1, m + 1) * alpha) <= es)[0]
    k = k_ok.max() + 1 if len(k_ok) else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


u = rng.random(200_000)
print(f"  calibrator check: mean e-value under the null = "
      f"{p_to_e(u).mean():.3f}  (must be <= 1, eq. 36.3)")

print(f"\n  {'rho':<8}{'BH FDP':>10}{'BH power':>11}{'BY FDP':>10}"
      f"{'BY power':>11}{'e-BH FDP':>11}{'e-BH power':>12}")
for rho in (0.0, 0.7):
    acc = np.zeros(6)
    R = 200
    for i in range(R):
        pc, tc = correlated_screen(rho=rho, seed=5000 + i)
        m_ = len(pc)
        by = bh(pc, ALPHA / np.sum(1.0 / np.arange(1, m_ + 1)))   # eq. (8.8)
        for j, rj in enumerate((bh(pc), by, ebh(p_to_e(pc)))):
            f_, t_ = fdp_tpr(rj, tc)
            acc[2 * j] += f_; acc[2 * j + 1] += t_
    acc /= R
    print(f"  {rho:<8.1f}{acc[0]:>10.3f}{acc[1]:>11.3f}{acc[2]:>10.3f}"
          f"{acc[3]:>11.3f}{acc[4]:>11.3f}{acc[5]:>12.3f}")

print("\n  e-BH's power here is essentially ZERO, not merely reduced. Why:")
print(f"  {'discoveries wanted (k)':<26}{'e-value needed':>18}{'i.e. p below':>16}")
for k_ in (1, 100, 200):
    e_need = 2000 / (k_ * ALPHA)
    print(f"  {k_:<26}{e_need:>18,.0f}{(0.5 * k_ * ALPHA / 2000) ** 2:>16.2e}")
print("""
  To reject even one hypothesis out of 2000, e-BH needs a p-value below about
  1e-10. The signal in this screen produces p-values near 0.003. So nothing is
  rejected, and that is the calibrator's fault rather than e-BH's.

  Turning a p-value into an e-value is LOSSY. The calibrator has to work for
  every possible null distribution, so it throws away most of the evidence. An
  e-value that comes directly from the analysis -- a likelihood ratio, a
  betting martingale, a sequential test's wealth process -- is far sharper, and
  that is the setting where e-BH is genuinely competitive.

  BY is the fair comparison for "the same input, a stronger guarantee": it
  halves BH's power (0.50 to 0.24) and holds under any dependence.

  The practical reading: BH remains the right default for genomics, where
  positive dependence is the norm. Reach for e-BH when the dependence is
  unknown or negative, or when you need to combine evidence across analyses,
  which e-values do by simple averaging and p-values do not.""")

# %% [markdown]
# ## 7. Knockoffs, eq. (36.5)-(36.6)
#
# Selecting features in a regression is not a p-value problem. Knockoffs give
# FDR control directly, with no independence assumption between features.

# %%
header("7. Knockoff filter for correlated feature selection (36.5)-(36.6)")


def gaussian_knockoffs(X, Sigma, r):
    """Equicorrelated model-X knockoffs for Gaussian features.

    Build Xt with the same covariance as X but conditionally independent of y.
    s is chosen so 2*Sigma - diag(s) stays positive semi-definite, which is
    what makes the construction valid."""
    pdim = X.shape[1]
    lam_min = np.linalg.eigvalsh(Sigma).min()
    s = np.full(pdim, min(1.0, 2.0 * lam_min) * 0.99)
    Sinv = np.linalg.inv(Sigma)
    mu_t = X - X @ Sinv @ np.diag(s)
    V = 2 * np.diag(s) - np.diag(s) @ Sinv @ np.diag(s)
    V = (V + V.T) / 2 + 1e-9 * np.eye(pdim)
    L = np.linalg.cholesky(V)
    return mu_t + r.normal(size=X.shape) @ L.T


def knockoff_filter(X, y, Sigma, r, alpha=ALPHA):
    """W from lasso entry order, then the knockoff+ threshold, eq. (36.6)."""
    Xt = gaussian_knockoffs(X, Sigma, r)
    Z = np.hstack([X, Xt])
    Z = Z / np.sqrt((Z ** 2).sum(0, keepdims=True))
    # `alphas` takes the COUNT here; `n_alphas` was deprecated in sklearn 1.9.
    alphas, coefs, _ = lasso_path(Z, y, alphas=150)
    # Entry statistic: the largest penalty at which a variable is still non-zero.
    entry = np.array([alphas[np.nonzero(coefs[j])[0][0]] if np.any(coefs[j]) else 0.0
                      for j in range(Z.shape[1])])
    pdim = X.shape[1]
    W = entry[:pdim] - entry[pdim:]                       # eq. (36.5)
    ts = np.sort(np.abs(W[W != 0]))
    sel = np.zeros(pdim, bool)
    for t in ts:
        ratio = (1 + np.sum(W <= -t)) / max(np.sum(W >= t), 1)
        if ratio <= alpha:                                # eq. (36.6)
            sel = W >= t
            break
    return sel, W


n, pdim, k_true, RHO = 700, 200, 60, 0.3
Sigma = RHO * np.ones((pdim, pdim)) + (1 - RHO) * np.eye(pdim)
L = np.linalg.cholesky(Sigma)
truth_k = np.zeros(pdim, bool); truth_k[:k_true] = True

print(f"  n = {n}, p = {pdim}, {k_true} true signals, feature correlation {RHO}")
print(f"\n  {'method':<34}{'selected':>10}{'FDP':>8}{'power':>8}")
res = {"knockoff": [], "bh": []}
for i in range(10):
    r = np.random.default_rng(600 + i)
    signal = np.zeros(pdim)
    signal[:k_true] = 4.0 * r.choice([-1.0, 1.0], k_true)   # mixed signs
    X = r.normal(size=(n, pdim)) @ L.T
    y = X @ signal + r.normal(0, 1, n)
    sel, W = knockoff_filter(X, y, Sigma, r)
    res["knockoff"].append((*fdp_tpr(sel, truth_k), sel.sum()))
    # Marginal p-values plus BH, the naive alternative.
    pv = np.array([st.pearsonr(X[:, j], y)[1] for j in range(pdim)])
    rb = bh(pv)
    res["bh"].append((*fdp_tpr(rb, truth_k), rb.sum()))
for name, lab in (("bh", "marginal p-values + BH"),
                  ("knockoff", "knockoff filter (36.6)")):
    a = np.array(res[name], float)
    print(f"  {lab:<34}{a[:, 2].mean():>10.1f}{a[:, 0].mean():>8.3f}"
          f"{a[:, 1].mean():>8.3f}")

print("""
  Marginal testing has no chance here: with features correlated at 0.3, a null
  feature correlates with y through its correlated neighbours, so BH on
  marginal p-values selects a large number of them.

  Knockoffs test each feature CONDITIONAL on all the others, and the symmetry
  of W under the null (36.5) supplies the false-discovery estimate in (36.6)
  without any p-value or independence assumption.

  The cost: knockoffs are randomised. Two runs on the same data give different
  answers, which is what derandomised knockoffs fix by aggregating several
  runs through e-values.""")

# A property of (36.6) that surprises people the first time they hit it.
header("7b. Knockoffs cannot make fewer than 1/alpha discoveries")
print("""  Look again at the threshold rule:

      (1 + #{W <= -t}) / #{W >= t}  <=  alpha

  The numerator is at least 1, always. So the rule can only be satisfied if
  #{W >= t} >= 1/alpha. At alpha = 0.05 that is TWENTY selections. A knockoff
  filter physically cannot report 5 discoveries at the 5% level: it reports at
  least 20, or none at all.""")
for a_, k_ in ((0.05, 20), (0.10, 10), (0.20, 5)):
    print(f"    alpha = {a_:<5} -> minimum possible selection size = {k_}")
print("""
  That is why the run below finds nothing. Same data-generating process, only
  the number of true signals changed:""")
for kk in (8, 25, 60):
    hits = []
    for i in range(4):
        r = np.random.default_rng(770 + i)
        sg = np.zeros(pdim); sg[:kk] = 4.0 * r.choice([-1.0, 1.0], kk)
        X = r.normal(size=(n, pdim)) @ L.T
        y = X @ sg + r.normal(0, 1, n)
        hits.append(knockoff_filter(X, y, Sigma, r)[0].sum())
    print(f"    {kk:>3} true signals -> mean selections = {np.mean(hits):.1f}")
print("""
  With 8 real signals the filter is silent, not because it failed to find them
  but because the guarantee forbids a small answer. Knockoffs are a method for
  screens where you expect many discoveries. For a handful of candidates, use
  a p-value procedure.""")

# %% [markdown]
# ## 8. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].scatter(np.log10(cov), -np.log10(np.maximum(p, 1e-30)), s=2, alpha=0.18,
              color="grey")
ax[0].scatter(np.log10(cov[truth]), -np.log10(np.maximum(p[truth], 1e-30)),
              s=3, alpha=0.5, color="firebrick")
ax[0].set_xlabel("log10 mean expression (the covariate)")
ax[0].set_ylabel("-log10 p")
ax[0].set_title("Power varies with an observable covariate")

ax[1].step(range(1, 9), wm, where="mid", color="steelblue", lw=2)
ax[1].axhline(1.0, ls="--", color="grey")
ax[1].set_xlabel("covariate stratum (low to high)")
ax[1].set_ylabel("learned weight")
ax[1].set_title("IHW weights (36.2)")

for rho, colour in ((0.0, "steelblue"), (0.7, "firebrick")):
    fd = []
    for i in range(200):
        pc, tc = correlated_screen(rho=rho, seed=9000 + i)
        fd.append(fdp_tpr(bh(pc), tc)[0])
    ax[2].hist(fd, bins=24, alpha=0.6, color=colour, label=f"rho = {rho}")
ax[2].axvline(ALPHA, color="black", ls="--")
ax[2].set_xlabel("realised FDP in one experiment")
ax[2].set_title("BH controls the mean, not your run")
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "modern_multiple_testing.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'modern_multiple_testing.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Is your covariate admissible?
#
# IHW needs a covariate independent of the p-value **under the null**. Test
# three candidates: mean expression, the observed effect size, and a random
# number. Measure the FDP each produces.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # The admissibility test: for a NULL feature, does the covariate predict the
# # p-value? Measure it on a screen with signal, using only the true nulls.
# ps, ts, cs = simulate_screen(pi1=0.10, seed=99)
# eff = np.abs(st.norm.isf(ps / 2))                    # observed effect size
# rnd = np.random.default_rng(4).random(len(ps))
# print(f"  {'covariate':<26}{'corr with p | null':>20}{'FDP':>9}{'power':>8}"
#       f"{'verdict':>14}")
# for name, c in (("mean expression", cs), ("observed effect size", eff),
#                 ("a random number", rnd)):
#     rho_null = st.spearmanr(c[~ts], ps[~ts]).statistic
#     rj = ihw(ps, c, seed=5, n_strata=40)[0]
#     f_, t_ = fdp_tpr(rj, ts)
#     verdict = "INADMISSIBLE" if abs(rho_null) > 0.1 else "ok"
#     print(f"  {name:<26}{rho_null:>20.3f}{f_:>9.3f}{t_:>8.3f}{verdict:>14}")
# print(f"\n  nominal FDR = {ALPHA}")
#
# # Read the correlation column first, because that is the actual condition in
# # eq. (36.1) and it is checkable before you look at any result. Mean
# # expression and the random number sit near zero among the nulls. The
# # observed effect size is perfectly rank-correlated with the p-value by
# # construction, because it IS the p-value in another coordinate.
# #
# # The FDP column then shows the consequence: weighting by the effect size
# # concentrates weight exactly where the small p-values already are, and the
# # realised FDP rises above nominal.
# #
# # Two further points worth noticing.
# #
# # The random number is ADMISSIBLE but USELESS. It cannot break FDR control,
# # because it is independent of the p-value, but it carries no information
# # about power either, so the weights are noise. Admissibility and usefulness
# # are separate properties and you need both.
# #
# # The damage from an inadmissible covariate also depends on how FINE the
# # weighting is. With few strata the weights are coarse and the inflation is
# # mild; with many strata the procedure can track the p-value closely and the
# # inflation is severe. That is an argument for choosing the covariate on
# # principle rather than tuning the stratification until the answer looks good.

# %% [markdown]
# ### Problem 2: How much does e-BH cost you?
#
# Measure the power of BH, BY and e-BH across a range of signal strengths under
# independence, where BH is valid. What are you paying for the guarantee?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'effect size':<14}{'BH power':>11}{'BY power':>11}{'e-BH power':>12}"
#       f"{'BH FDP':>9}{'e-BH FDP':>10}")
# for eff in (2.0, 3.0, 4.0, 5.0):
#     acc = np.zeros(5); R = 150
#     for i in range(R):
#         r2 = np.random.default_rng(7000 + i)
#         m2 = 2000
#         de = r2.random(m2) < 0.1
#         z = r2.normal(0, 1, m2) + de * eff
#         pv = 2 * st.norm.sf(np.abs(z))
#         byv = bh(pv, ALPHA / np.sum(1.0 / np.arange(1, m2 + 1)))
#         a = fdp_tpr(bh(pv), de); b = fdp_tpr(byv, de); c = fdp_tpr(ebh(p_to_e(pv)), de)
#         acc += [a[1], b[1], c[1], a[0], c[0]]
#     acc /= R
#     print(f"  {eff:<14.1f}{acc[0]:>11.3f}{acc[1]:>11.3f}{acc[2]:>12.3f}"
#           f"{acc[3]:>9.3f}{acc[4]:>10.3f}")
#
# # e-BH sits between BH and BY, and the gap closes as the signal strengthens:
# # when effects are obvious, the price of the stronger guarantee is small.
# # When effects are marginal, which is when you most want discoveries, the
# # price is large.
# #
# # This is the general shape of the trade. Stronger guarantees cost power
# # exactly in the regime where power is scarce. That is an argument for
# # choosing the guarantee deliberately rather than defaulting to the safest
# # one, and for reporting which you used.

# %% [markdown]
# ### Problem 3: Derandomising knockoffs
#
# Run the knockoff filter several times on the *same* data and count how much
# the selected set changes. Then aggregate the runs by selection frequency.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# r0 = np.random.default_rng(321)
# Xf = r0.normal(size=(n, pdim)) @ L.T
# yf = Xf @ signal + r0.normal(0, 1, n)
# runs = []
# for i in range(15):
#     sel, _ = knockoff_filter(Xf, yf, Sigma, np.random.default_rng(900 + i))
#     runs.append(sel)
# runs = np.array(runs)
# jac = []
# for i in range(len(runs)):
#     for j in range(i + 1, len(runs)):
#         u_ = (runs[i] | runs[j]).sum()
#         jac.append((runs[i] & runs[j]).sum() / max(u_, 1))
# print(f"  15 runs on IDENTICAL data")
# print(f"    selections per run : min {runs.sum(1).min()}, max {runs.sum(1).max()}")
# print(f"    pairwise Jaccard   : {np.mean(jac):.2f}")
# freq = runs.mean(0)
# for thr in (0.3, 0.5, 0.8):
#     sel = freq >= thr
#     f_, t_ = fdp_tpr(sel, truth_k)
#     print(f"    selected in >= {thr:.0%} of runs: {sel.sum():>3}  "
#           f"FDP {f_:.3f}  power {t_:.3f}")
#
# # The data never changed, only the knockoff draw, and the answer moves. That
# # is uncomfortable to report and impossible to reproduce exactly.
# #
# # Aggregating by selection frequency stabilises it, and this is the idea
# # behind derandomised knockoffs: each run contributes an e-value, and e-values
# # average legitimately (eq. 36.3), which a p-value cannot do. Averaging
# # p-values across runs would have no validity at all.
# #
# # The general lesson applies well beyond knockoffs: if a method is randomised,
# # report how much its output moves, and aggregate rather than reporting one
# # arbitrary draw.

# %% [markdown]
# ## What to take away
#
# 1. Weighted BH (36.1) is valid for any weights **independent of the
#    p-values**. That is the whole condition, and it is easy to violate.
# 2. Independent filtering is a binary weight. **IHW** (36.2) is the smooth
#    version and recovers the power the binary cut discards.
# 3. BH controls the **expected** FDP. Under dependence a single experiment's
#    FDP can be far from nominal, and you run one experiment.
# 4. **E-values** (36.3) are calibrated by their mean, so they combine by
#    averaging and give FDR control under any dependence, at a power cost.
# 5. **Knockoffs** (36.5)-(36.6) give conditional feature selection with FDR
#    control and no independence assumption, at the price of randomness.
#
# **Next:** `37_measurement_error_and_regression.py`
