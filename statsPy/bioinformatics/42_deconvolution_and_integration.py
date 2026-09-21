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
# # Applied 42: Deconvolution, batch integration, and zero-inflation
#
# **Curriculum link:** `stats.md` -> Topic 42, equations (42.1)-(42.6)
# **Core modules used:** 11, 19, 13, 37, 26
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | Bulk RNA-seq deconvolved with CIBERSORTx; scRNA-seq integrated with Harmony or MNN |
# | **Key threats** | Wrong signature; collinear cell types; compositional output; over-correction; imaginary zero-inflation |
# | **Here** | Simulated mixtures with known proportions, and UMI counts with a known zero mechanism |
#
# ## Three claims to test
#
# 1. Deconvolution is a constrained regression, and it inherits every problem
#    regression has with collinear predictors.
# 2. Integration trades mixing against biology, and one metric alone can
#    always be made to look good.
# 3. Zero-inflation in UMI data is mostly a myth, and testable.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls

warnings.filterwarnings("ignore")

MODULE_NAME = "42_deconvolution_and_integration"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Deconvolution is constrained regression, eq. (42.1)-(42.2)

# %%
header("1. Recovering cell-type proportions from a mixture (42.1)-(42.2)")


def make_signatures(n_genes=600, n_types=5, similarity=0.0, seed=0):
    """Cell-type reference profiles. `similarity` makes two types alike,
    which is what happens with, say, CD4 and CD8 T cells."""
    r = np.random.default_rng(seed)
    S = np.exp(r.normal(3.0, 1.0, (n_genes, n_types)))
    if similarity > 0:
        # Types 0 and 1 are pulled toward each other.
        mid = (S[:, 0] + S[:, 1]) / 2
        S[:, 0] = (1 - similarity) * S[:, 0] + similarity * mid
        S[:, 1] = (1 - similarity) * S[:, 1] + similarity * mid
    return S


def deconvolve(y, S):
    """Eq. (42.2): non-negative least squares, then rescale to a simplex."""
    w, _ = nnls(S, y)
    return w / w.sum() if w.sum() > 0 else np.full(S.shape[1], 1 / S.shape[1])


S = make_signatures(seed=1)
n_types = S.shape[1]
true_w = np.array([0.40, 0.25, 0.15, 0.15, 0.05])
print(f"  {S.shape[0]} genes, {n_types} cell types\n")
print(f"  {'noise level':<14}" + "".join(f"{f'type {i}':>10}" for i in range(n_types))
      + f"{'mean |error|':>15}")
print(f"  {'TRUTH':<14}" + "".join(f"{w:>10.3f}" for w in true_w) + f"{'':>15}")
for noise in (0.02, 0.10, 0.30):
    est = []
    for _ in range(200):
        y = S @ true_w
        y = y * np.exp(rng.normal(0, noise, len(y)))      # multiplicative noise
        est.append(deconvolve(y, S))
    est = np.array(est)
    print(f"  {noise:<14.2f}" + "".join(f"{m:>10.3f}" for m in est.mean(0))
          + f"{np.abs(est.mean(0) - true_w).mean():>15.4f}")

print("""
  With a correct signature matrix, deconvolution is accurate and degrades
  gracefully with noise. Everything below is about what happens when one of
  its assumptions fails.""")

# %% [markdown]
# ## 2. Collinear cell types, eq. (42.2)

# %%
header("2. Similar cell types are collinear predictors")

print(f"  {'similarity of types 0 and 1':<30}{'corr(S0,S1)':>13}"
      f"{'SD of w0':>11}{'SD of w0+w1':>14}")
for sim in (0.0, 0.5, 0.85, 0.97):
    S2 = make_signatures(similarity=sim, seed=2)
    est = []
    for _ in range(300):
        y = S2 @ true_w
        y = y * np.exp(rng.normal(0, 0.10, len(y)))
        est.append(deconvolve(y, S2))
    est = np.array(est)
    r_ = np.corrcoef(S2[:, 0], S2[:, 1])[0, 1]
    print(f"  {sim:<30.2f}{r_:>13.3f}{est[:, 0].std():>11.4f}"
          f"{(est[:, 0] + est[:, 1]).std():>14.4f}")

print("""
  As the two signatures converge, the individual estimates become wildly
  unstable while their SUM stays precise. The data determines how much
  'type 0 or type 1' there is, and cannot split it.

  This is the collinearity behaviour of Module 11, and it has a practical
  consequence: report related cell types as a COMBINED fraction unless you can
  show the signatures are separable. A paper reporting a shift from CD4 to
  CD8 with a constant total is often reporting numerical instability.""")

# %% [markdown]
# ## 3. The output is compositional

# %%
header("3. Proportions cannot move independently")

S3 = make_signatures(seed=3)
# One cell type genuinely doubles; the others are untouched in absolute terms.
abs_ctrl = np.array([1000, 600, 400, 400, 150], float)
abs_case = abs_ctrl.copy(); abs_case[0] *= 2.0
res = {"ctrl": [], "case": []}
for _ in range(300):
    for lab, ab in (("ctrl", abs_ctrl), ("case", abs_case)):
        w = ab / ab.sum()
        y = S3 @ w * np.exp(rng.normal(0, 0.08, S3.shape[0]))
        res[lab].append(deconvolve(y, S3))
c0 = np.array(res["ctrl"]); c1 = np.array(res["case"])

print(f"  Only cell type 0 changed, and it DOUBLED in absolute abundance.\n")
print(f"  {'type':<8}{'absolute change':>18}{'proportion ctrl':>18}"
      f"{'proportion case':>18}{'apparent change':>18}")
for i in range(n_types):
    ab_ch = abs_case[i] / abs_ctrl[i]
    print(f"  {i:<8}{ab_ch:>18.2f}{c0[:, i].mean():>18.3f}{c1[:, i].mean():>18.3f}"
          f"{c1[:, i].mean()/c0[:, i].mean():>18.2f}")

print("""
  Four cell types did not change at all and every one of them appears to have
  fallen by about a quarter. That is closure (Topic 26), not biology.

  Deconvolution output is compositional and must be analysed as such: use
  log-ratios, or anchor to an absolute measurement such as total cell count
  per gram of tissue. Testing proportions directly with a t-test per cell type
  produces exactly the correlated false positives Module 26 demonstrates.""")

# %% [markdown]
# ## 4. Batch integration, eq. (42.3)

# %%
header("4. MNN-style correction, and the metric that can be gamed (42.3)")


def make_batches(n_per=300, n_genes=40, batch_shift=3.0, seed=0, shared=True):
    """Two batches, two cell types. If shared is False, type B exists in only
    one batch, which is what stops naive alignment being safe."""
    r = np.random.default_rng(seed)
    X, batch, ctype = [], [], []
    for b in (0, 1):
        types = (0, 1) if (shared or b == 0) else (0,)
        for t in types:
            n = n_per if shared else (n_per if t == 0 else n_per)
            centre = np.zeros(n_genes)
            centre[:5] = t * 4.0                              # biology
            centre = centre + b * batch_shift                 # technical
            X.append(r.normal(centre, 1.0, (n, n_genes)))
            batch += [b] * n; ctype += [t] * n
    return np.vstack(X), np.array(batch), np.array(ctype)


def mnn_correct(X, batch, k=20):
    """Eq. (42.3). Find mutual nearest neighbours across batches and subtract
    the average difference."""
    a, b = X[batch == 0], X[batch == 1]
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(2)
    nn_a = np.argsort(d, axis=1)[:, :k]           # b-neighbours of each a
    nn_b = np.argsort(d, axis=0)[:k, :].T         # a-neighbours of each b
    pairs = [(i, j) for i in range(len(a)) for j in nn_a[i] if i in nn_b[j]]
    if not pairs:
        return X.copy()
    vec = np.mean([a[i] - b[j] for i, j in pairs], axis=0)
    out = X.copy()
    out[batch == 1] = out[batch == 1] + vec
    return out


def mixing(X, batch, k=25):
    """Fraction of each cell's neighbours from the other batch. 0.5 is ideal."""
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    return np.mean(batch[nn] != batch[:, None])


def bio_preserved(X, ctype):
    """Silhouette-like separation of the true cell types."""
    c0, c1 = X[ctype == 0], X[ctype == 1]
    between = np.linalg.norm(c0.mean(0) - c1.mean(0))
    within = (c0.std(0).mean() + c1.std(0).mean()) / 2
    return between / within


X, batch, ctype = make_batches(seed=5)
Xc = mnn_correct(X, batch)
# An "integration" that destroys everything: project out all variation.
X_over = X - X.mean(0)
X_over = X_over * 0.02 + rng.normal(0, 0.5, X_over.shape)

print(f"  {'state':<28}{'batch mixing':>15}{'biology preserved':>20}")
for lab, D in (("before correction", X), ("MNN corrected (42.3)", Xc),
               ("aggressive over-correction", X_over)):
    print(f"  {lab:<28}{mixing(D, batch):>15.3f}{bio_preserved(D, ctype):>20.2f}")

print("""
  Read the two columns together, never one alone. The over-corrected version
  has the BEST mixing score of the three, and it achieved that by deleting the
  biology.

  Any integration strong enough to remove all batch signal will remove biology
  correlated with batch. Report a mixing metric AND a biology-preservation
  metric, and if your design is confounded no correction can separate them
  (Module 19).""")

# %% [markdown]
# ## 5. Integration when a population exists in only one batch

# %%
header("5. Mutual nearest neighbours protects against forced alignment")

Xu, bu, cu = make_batches(seed=6, shared=False)
Xu_c = mnn_correct(Xu, bu)
only_in_0 = (cu == 1)
print(f"  Cell type 1 appears in batch 0 only ({only_in_0.sum()} cells).\n")
print(f"  {'':<26}{'mixing':>10}{'type-1 cells pulled toward batch 1':>38}")
for lab, D in (("before", Xu), ("MNN corrected", Xu_c)):
    d_to_b1 = np.linalg.norm(D[only_in_0].mean(0) - D[bu == 1].mean(0))
    print(f"  {lab:<26}{mixing(D, bu):>10.3f}{d_to_b1:>38.2f}")

print("""
  The mutual criterion is what makes this safe. A cell in batch 0 with no
  counterpart in batch 1 will rarely be part of a RECIPROCAL nearest-neighbour
  pair, so it contributes little to the correction vector.

  Methods that align batches globally, by matching distributions rather than
  cells, have no such protection and will happily map a population that exists
  in one batch onto an unrelated population in the other. Check what happens
  to batch-specific populations before trusting any integration.""")

# %% [markdown]
# ## 6. Is there really zero-inflation? eq. (42.4)-(42.5)

# %%
header("6. Testing for zero-inflation instead of assuming it (42.4)-(42.5)")


def nb_zero_prob(mu, disp):
    """Eq. (42.5): P(Y=0) under a negative binomial."""
    size = 1.0 / disp
    return (size / (size + mu)) ** size


def simulate_umi(n_cells=1500, n_genes=800, disp=0.25, pi_extra=0.0, seed=0):
    r = np.random.default_rng(seed)
    mu = np.exp(r.normal(0.5, 1.6, n_genes))
    size = 1.0 / disp
    lam = r.gamma(size, mu / size, (n_cells, n_genes))
    y = r.poisson(lam)
    if pi_extra > 0:
        y = y * (r.random(y.shape) > pi_extra)           # true dropout
    return y, mu


print(f"  {'data':<34}{'mean obs zeros':>17}{'NB predicts':>14}{'excess':>10}")
for lab, pi_x in (("plain NB (no inflation)", 0.0),
                  ("NB + 20% true dropout", 0.20)):
    y, mu = simulate_umi(pi_extra=pi_x, seed=17)
    obs = (y == 0).mean(0)
    mu_hat = y.mean(0)
    v = y.var(0)
    disp_hat = np.clip((v - mu_hat) / np.maximum(mu_hat ** 2, 1e-9), 1e-3, 10)
    pred = nb_zero_prob(mu_hat, disp_hat)
    print(f"  {lab:<34}{obs.mean():>17.3f}{pred.mean():>14.3f}"
          f"{obs.mean()-pred.mean():>10.3f}")

print("""
  When the data really is negative binomial, the observed zeros match the NB
  prediction, and fitting an extra inflation parameter would be fitting noise.
  When there IS extra dropout, the excess shows up plainly.

  This is a diagnostic anyone can run in three lines, and it is the reason the
  scRNA-seq field moved away from zero-inflated models for UMI data. The
  zeros were never unexplained; they are what a small mean produces.""")

header("6b. What fitting an unnecessary zero-inflation costs")

print("""  Data has NO zero-inflation. Forcing a ZI parameter anyway, by
  attributing every zero beyond the POISSON expectation to dropout:
""")
print(f"  {'dispersion':<14}{'Q1':>10}{'Q2':>10}{'Q3':>10}{'Q4':>10}"
      f"{'worst gene':>12}")
for dsp in (0.25, 0.80, 2.00):
    y, mu = simulate_umi(pi_extra=0.0, disp=dsp, seed=23)
    mu_hat = y.mean(0)
    obs0 = (y == 0).mean(0)
    pi_forced = np.clip(obs0 - np.exp(-mu_hat), 0, 0.95)
    mu_zinb = mu_hat / (1 - pi_forced)
    q = np.quantile(mu, np.linspace(0, 1, 5))
    cells = ""
    for i in range(4):
        m = (mu >= q[i]) & (mu <= q[i + 1])
        cells += f"{(mu_zinb[m]/mu[m]).mean():>9.2f}x"
    print(f"  {dsp:<14.2f}{cells}{(mu_zinb/mu).max():>11.2f}x")
print("  (expression quartiles, Q1 lowest; each cell is the mean inflation "
      "factor)")

print("""
  Read along a row first. The inflation is worst in the MIDDLE of the
  expression range, not at the bottom. Very low-expressed genes have few
  counts under either model, so the Poisson and NB zero predictions nearly
  agree and little is misattributed. Very high-expressed genes have almost no
  zeros at all. In between, the NB produces many more zeros than a Poisson
  would, and every one of those gets blamed on dropout.

  Now read down the columns. The size of the error is set by the DISPERSION,
  because dispersion is what creates the extra zeros in the first place. At
  dispersion 2.0 the typical mid-range gene's mean is inflated by nearly half,
  and the worst gene by two thirds.

  Test first (eq. 42.5). Model dropout only if the excess is there, and expect
  the answer to depend on the protocol: plate-based and read-based assays
  behave differently from UMI ones.""")

# %% [markdown]
# ## 7. Cell-cell communication, eq. (42.6)

# %%
header("7. Communication scores and the null they are tested against (42.6)")

n_cells, n_types = 2000, 4
r = rng
ctype_cc = r.integers(0, n_types, n_cells)
# Ligand and receptor expression depend ONLY on cell type abundance and a
# shared technical factor. There is no communication in this simulation.
depth = r.lognormal(0, 0.4, n_cells)
lig = r.poisson(np.exp(1.2 + 0.4 * ctype_cc) * depth)
rec = r.poisson(np.exp(1.0 + 0.3 * ctype_cc) * depth)

def comm_score(lig, rec, ctype, a, b):
    return lig[ctype == a].mean() * rec[ctype == b].mean()        # eq. (42.6)

obs = {(a, b): comm_score(lig, rec, ctype_cc, a, b)
       for a in range(n_types) for b in range(n_types)}
null = {k: [] for k in obs}
for _ in range(400):
    perm = r.permutation(ctype_cc)
    for k in obs:
        null[k].append(comm_score(lig, rec, perm, *k))
sig = {k: (1 + np.sum(np.array(v) >= obs[k])) / 401 for k, v in null.items()}
n_sig = sum(p < 0.05 for p in sig.values())

print(f"  {n_types}x{n_types} = {len(obs)} sender-receiver pairs, NO communication simulated")
print(f"  pairs called significant by label permutation: {n_sig}")
print(f"\n  {'pair':<12}{'score':>12}{'permutation p':>16}")
for k in sorted(obs, key=lambda k: -obs[k])[:5]:
    print(f"  {str(k):<12}{obs[k]:>12.1f}{sig[k]:>16.4f}")

print("""
  There is no signalling anywhere in this simulation. Ligand and receptor
  levels rise with cell-type index and with sequencing depth, and that is
  enough to produce significant 'interactions'.

  Two reasons. The score in (42.6) is a product of expression magnitudes, so
  abundant ligands and abundant cell types dominate it. And permuting cell
  labels asks 'is this pair unusual for these cell types', which is not the
  same question as 'do these cells communicate'.

  Treat communication output as a ranked hypothesis list. A claim of signalling
  needs perturbation evidence, not a permutation p-value.""")

# %% [markdown]
# ## 8. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

sims = [0.0, 0.3, 0.6, 0.85, 0.95, 0.99]
sd_ind, sd_sum = [], []
for sim in sims:
    S2 = make_signatures(similarity=sim, seed=2)
    e = np.array([deconvolve(S2 @ true_w * np.exp(rng.normal(0, 0.1, S2.shape[0])), S2)
                  for _ in range(150)])
    sd_ind.append(e[:, 0].std()); sd_sum.append((e[:, 0] + e[:, 1]).std())
ax[0].plot(sims, sd_ind, "o-", color="firebrick", label="w0 alone")
ax[0].plot(sims, sd_sum, "o-", color="steelblue", label="w0 + w1")
ax[0].set_xlabel("similarity of the two signatures"); ax[0].set_ylabel("SD of estimate")
ax[0].set_title("Collinear cell types"); ax[0].legend(fontsize=8)

for lab, D, colour in (("before", X, "grey"), ("MNN", Xc, "steelblue")):
    Zp = (D - D.mean(0)) @ np.linalg.svd(D - D.mean(0), full_matrices=False)[2][:2].T
    ax[1].scatter(Zp[batch == 0, 0], Zp[batch == 0, 1], s=4, alpha=0.4, color=colour)
    ax[1].scatter(Zp[batch == 1, 0], Zp[batch == 1, 1], s=4, alpha=0.4,
                  color=colour, marker="^")
ax[1].set_title("Batches before (grey) and after (blue)")
ax[1].set_xlabel("PC1")

y0, mu0 = simulate_umi(pi_extra=0.0, seed=31)
mh = y0.mean(0); vv = y0.var(0)
dh = np.clip((vv - mh) / np.maximum(mh ** 2, 1e-9), 1e-3, 10)
ax[2].scatter(np.log10(mh + 1e-3), (y0 == 0).mean(0), s=5, alpha=0.3,
              color="steelblue", label="observed")
o = np.argsort(mh)
ax[2].plot(np.log10(mh[o] + 1e-3), nb_zero_prob(mh[o], dh[o]), color="firebrick",
           lw=1.5, label="NB prediction (42.5)")
ax[2].set_xlabel("log10 mean count"); ax[2].set_ylabel("fraction zero")
ax[2].set_title("Zeros are explained by the mean"); ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "deconvolution_integration.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'deconvolution_integration.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: A signature from the wrong tissue
#
# Deconvolve using a signature matrix that is systematically wrong, as happens
# when the reference came from blood and the sample is tumour.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# S_true = make_signatures(seed=11)
# print(f"  {'reference used':<34}" +
#       "".join(f"{f'type {i}':>9}" for i in range(n_types)) + f"{'mean |err|':>13}")
# print(f"  {'TRUTH':<34}" + "".join(f"{w:>9.3f}" for w in true_w) + f"{'':>13}")
# for lab, pert in (("correct reference", 0.0), ("mildly wrong platform", 0.25),
#                   ("wrong tissue", 0.8)):
#     r2 = np.random.default_rng(12)
#     S_used = S_true * np.exp(r2.normal(0, pert, S_true.shape))
#     est = np.array([deconvolve(S_true @ true_w *
#                                np.exp(r2.normal(0, 0.08, S_true.shape[0])), S_used)
#                     for _ in range(200)])
#     print(f"  {lab:<34}" + "".join(f"{m:>9.3f}" for m in est.mean(0))
#           + f"{np.abs(est.mean(0)-true_w).mean():>13.4f}")
#
# # The estimates degrade steadily and, importantly, they degrade SILENTLY:
# # NNLS always returns proportions that sum to one and always looks like a
# # plausible answer. There is no residual diagnostic in the standard output
# # that screams "wrong reference".
# #
# # The check worth running is the residual: compare the observed bulk profile
# # against S @ w_hat. A reference that does not fit leaves structured, not
# # random, residuals. Always report that fit, and always match the reference
# # platform and tissue.

# %% [markdown]
# ### Problem 2: Does correction help or harm when the design is confounded?
#
# Compare integration on a balanced design and a fully confounded one.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def confounded_batches(confound, seed=0, n_per=300, n_genes=40):
#     r = np.random.default_rng(seed)
#     X, b_, c_ = [], [], []
#     for b in (0, 1):
#         for t in (0, 1):
#             # When confounded, batch 0 is almost all type 0 and vice versa.
#             n = int(n_per * (0.9 if (b == t) else 0.1)) if confound else n_per
#             if n == 0: continue
#             centre = np.zeros(n_genes); centre[:5] = t * 4.0
#             centre = centre + b * 3.0
#             X.append(r.normal(centre, 1.0, (n, n_genes)))
#             b_ += [b] * n; c_ += [t] * n
#     return np.vstack(X), np.array(b_), np.array(c_)
#
# print(f"  {'design':<22}{'state':<18}{'mixing':>10}{'biology kept':>15}")
# for lab, cf in (("balanced", False), ("confounded", True)):
#     Xb, bb, cb = confounded_batches(cf, seed=21)
#     Xbc = mnn_correct(Xb, bb)
#     for st_, D in (("before", Xb), ("after MNN", Xbc)):
#         print(f"  {lab:<22}{st_:<18}{mixing(D, bb):>10.3f}"
#               f"{bio_preserved(D, cb):>15.2f}")
#
# # On the balanced design, correction improves mixing and leaves the biology
# # intact. On the confounded design, correction improves mixing by destroying
# # the biological separation, because the batch difference and the biological
# # difference are the same direction in the data.
# #
# # The mixing metric improves in BOTH cases. If you reported only mixing, the
# # confounded design would look like the bigger success. This is why Module 19
# # insists that confounding is a design failure with no analytical remedy, and
# # why the two metrics must always be read as a pair.

# %% [markdown]
# ### Problem 3: When does zero-inflation actually appear?
#
# Compare a UMI-like protocol with a read-based one where amplification adds
# genuine dropout, and check whether the diagnostic distinguishes them.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'protocol':<34}{'obs zeros':>12}{'NB predicts':>13}{'excess':>9}"
#       f"{'verdict':>16}")
# for lab, pi_x, disp in (("UMI, well behaved", 0.00, 0.25),
#                         ("UMI, high dispersion", 0.00, 0.80),
#                         ("read-based, real dropout", 0.25, 0.25),
#                         ("severe dropout", 0.50, 0.25)):
#     y, mu = simulate_umi(pi_extra=pi_x, disp=disp, seed=41)
#     obs = (y == 0).mean(0).mean()
#     mh = y.mean(0); vv = y.var(0)
#     dh = np.clip((vv - mh) / np.maximum(mh ** 2, 1e-9), 1e-3, 20)
#     pred = nb_zero_prob(mh, dh).mean()
#     exc = obs - pred
#     print(f"  {lab:<34}{obs:>12.3f}{pred:>13.3f}{exc:>9.3f}"
#           f"{('inflated' if exc > 0.02 else 'no inflation'):>16}")
#
# # The diagnostic separates the cases correctly, and note the second row in
# # particular: high dispersion produces MANY zeros, and the NB still predicts
# # them. Lots of zeros is not evidence of zero-inflation. Only zeros in excess
# # of what the fitted mean-variance relationship predicts are.
# #
# # That distinction is the whole argument. The scRNA-seq literature spent
# # years treating "mostly zeros" as self-evident proof of a dropout process,
# # when the negative binomial had already accounted for them. Estimate the
# # dispersion first, then ask what is left over.

# %% [markdown]
# ## What to take away
#
# 1. Deconvolution is constrained regression (42.1)-(42.2) and inherits
#    collinearity: report similar cell types as a combined fraction.
# 2. Its output is **compositional**. One type rising forces the others down.
# 3. Integration must be judged on **mixing and biology together**. Over-
#    correction wins on mixing alone.
# 4. Mutual nearest neighbours protects batch-specific populations; global
#    distribution matching does not.
# 5. **Test for zero-inflation** with eq. (42.5) before modelling it. Many
#    zeros is not excess zeros.
# 6. Communication scores (42.6) are driven by abundance, and their
#    permutation null answers a different question than the one asked.
#
# **Next:** `43_advanced_statistical_genetics.py`
