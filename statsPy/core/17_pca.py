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
# # Module 17: Principal component analysis
#
# **Curriculum link:** `stats.md` -> Topic 17, equations (17.1)-(17.6)
#
# ## What you will learn
#
# 1. PCA is one eigenproblem (17.1), computed via SVD (17.2)-(17.3).
# 2. Centering is mandatory; **scaling is a decision** (17.4).
# 3. How many components: parallel analysis and Marchenko-Pastur (17.5).
# 4. Sign and rotation indeterminacy: a genuine reproducibility trap.
# 5. **PC x metadata association (17.6)**: the most valuable PCA output in
#    bioinformatics.
# 6. Two errors: over-reading separation, and double dipping.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "17_pca"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. PCA from scratch: eigen and SVD routes agree, eq. (17.1)-(17.3)

# %%
header("1. Eigen-decomposition == SVD (17.1)-(17.3)")

def pca(X, scale=False):
    """PCA of an n x p matrix (rows = samples). stats.md eq. (17.2)-(17.3)."""
    Xc = X - X.mean(axis=0)                      # centering is MANDATORY
    if scale:
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd                             # eq. (17.4): correlation PCA
    n = Xc.shape[0]
    U, D, Vt = np.linalg.svd(Xc, full_matrices=False)
    return {"scores": U * D,                     # n x k sample coordinates
            "loadings": Vt.T,                    # p x k feature weights
            "eigenvalues": D**2 / (n - 1),
            "pve": D**2 / np.sum(D**2),
            "singular_values": D, "centered": Xc}


rng = np.random.default_rng(1701)
n_s, p_f = 60, 300
# Two latent factors drive the data.
z = rng.normal(0, 1, (n_s, 2))
W = rng.normal(0, 1, (2, p_f))
X = z @ W + rng.normal(0, 0.8, (n_s, p_f))

res = pca(X)
Xc = res["centered"]
S = Xc.T @ Xc / (n_s - 1)
ev_eig = np.linalg.eigvalsh(S)[::-1][:6]
print(f"  eigenvalues from covariance eigen-decomposition: {ev_eig}")
print(f"  eigenvalues from SVD, d_j^2/(n-1)              : "
      f"{res['eigenvalues'][:6]}")
print(f"  max |difference| = "
      f"{np.max(np.abs(ev_eig - res['eigenvalues'][:6])):.2e}")
print(f"\n  proportion of variance explained (17.3): "
      f"{np.round(res['pve'][:6], 4)}")
print(f"  cumulative                              : "
      f"{np.round(np.cumsum(res['pve'][:6]), 4)}")

# Scores and loadings reconstruct the data (eq. 16.8).
k = 2
recon = res["scores"][:, :k] @ res["loadings"][:, :k].T
print(f"\n  rank-{k} reconstruction captures "
      f"{1 - np.sum((Xc-recon)**2)/np.sum(Xc**2):.1%} of the total variance")

# Cross-check against scikit-learn.
from sklearn.decomposition import PCA as SkPCA
sk = SkPCA(n_components=6).fit(X)
print(f"  sklearn PVE: {np.round(sk.explained_variance_ratio_, 4)}")

# %% [markdown]
# ## 2. Centering is mandatory; scaling is a decision, eq. (17.4)

# %%
header("2. Centering and scaling (17.4)")

X_uncentered = X + 50.0                          # add a large common offset
U_, D_, Vt_ = np.linalg.svd(X_uncentered, full_matrices=False)
print(f"  WITHOUT centering, PC1 explains "
      f"{D_[0]**2/np.sum(D_**2):.1%} of variance")
print(f"  |correlation of PC1 loading vector with the mean vector| = "
      f"{abs(np.corrcoef(Vt_[0], X_uncentered.mean(axis=0))[0,1]):.4f}")
print("  Without centering, PC1 just points at the mean. Always centre.")

# Scaling: when features have wildly different variances.
print("\n  Scaling matters when features are on different scales:")
rng2 = np.random.default_rng(1702)
clin = np.column_stack([
    rng2.normal(5.5, 0.8, 100),          # HbA1c, %
    rng2.normal(200, 40, 100),           # cholesterol, mg/dL
    rng2.normal(7000, 2500, 100),        # leukocytes, cells/uL
])
for scale_flag in [False, True]:
    r = pca(clin, scale=scale_flag)
    top = np.abs(r["loadings"][:, 0])
    print(f"    scale={str(scale_flag):<5}: PVE1={r['pve'][0]:.3f}, "
          f"|PC1 loadings| = {np.round(top, 3)}")
print("  Unscaled, PC1 is entirely the variable with the largest UNITS")
print("  (leukocytes), which is an artefact of measurement units, not biology.")
print("\n  RULE: scale when units differ (clinical panels); do NOT scale when")
print("  features share units and variance differences ARE the biology")
print("  (log-expression, M-values, arcsinh cytometry) - scaling there")
print("  up-weights the noisiest low-expressed genes.")

# %% [markdown]
# ## 3. How many components? eq. (17.5)
#
# Under pure noise with $p/n\to\gamma$, sample-correlation eigenvalues are
# bounded by $\lambda_+=(1+\sqrt\gamma)^2$ (Marchenko-Pastur).

# %%
header("3. Selecting components: parallel analysis and Marchenko-Pastur (17.5)")

def parallel_analysis(X, n_perm=100, seed=0, scale=True):
    """Horn's parallel analysis: permute each feature independently to destroy
    correlation while preserving marginals, then compare eigenvalues."""
    rng = np.random.default_rng(seed)
    obs = pca(X, scale=scale)["eigenvalues"]
    null = np.empty((n_perm, len(obs)))
    for b in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, j])
                              for j in range(X.shape[1])])
        null[b] = pca(Xp, scale=scale)["eigenvalues"][:len(obs)]
    thresh = np.percentile(null, 95, axis=0)
    return obs, thresh, int(np.sum(obs > thresh))


obs_ev, thresh_ev, k_keep = parallel_analysis(X, n_perm=60, seed=3, scale=True)
gamma = p_f / n_s
mp_edge = (1 + np.sqrt(gamma)) ** 2                    # eq. (17.5)
print(f"  data: n={n_s}, p={p_f}, gamma = p/n = {gamma:.2f}")
print(f"  Marchenko-Pastur upper edge (1+sqrt(gamma))^2 = {mp_edge:.3f}")
print(f"  (correlation-PCA eigenvalues above this edge carry signal)\n")
print(f"  {'PC':>4}{'eigenvalue':>13}{'95th pct of permuted':>23}{'keep?':>8}")
for j in range(6):
    print(f"  {j+1:>4}{obs_ev[j]:>13.3f}{thresh_ev[j]:>23.3f}"
          f"{'yes' if obs_ev[j] > thresh_ev[j] else 'no':>8}")
print(f"\n  parallel analysis keeps {k_keep} component(s); the data were")
print(f"  generated from {2} latent factors.")

# Show how large noise eigenvalues get when p >> n.
print(f"\n  How large do PURE NOISE eigenvalues get?")
print(f"  {'n':>6}{'p':>7}{'gamma':>8}{'observed max':>14}{'MP edge':>10}")
for (nn, pp) in [(50, 50), (50, 500), (50, 2000)]:
    Xn = np.random.default_rng(nn * pp).normal(0, 1, (nn, pp))
    e = pca(Xn, scale=True)["eigenvalues"]
    g = pp / nn
    print(f"  {nn:>6}{pp:>7}{g:>8.1f}{e[0]:>14.2f}{(1+np.sqrt(g))**2:>10.2f}")
print("  With p >> n, noise alone produces enormous leading eigenvalues.")

# %% [markdown]
# ## 4. Sign and rotation indeterminacy

# %%
header("4. PC signs are arbitrary; near-tied PCs are unstable")

r1 = pca(X)
r2 = pca(X[:, ::-1])                # same data, features reordered
print(f"  PC1 PVE, original order : {r1['pve'][0]:.6f}")
print(f"  PC1 PVE, reversed order : {r2['pve'][0]:.6f}   <- identical")
# Compare score vectors up to sign:
cor = np.corrcoef(r1["scores"][:, 0], r2["scores"][:, 0])[0, 1]
print(f"  correlation of the two PC1 score vectors: {cor:+.6f}")
print("  |correlation| = 1, but the SIGN can flip. Never interpret the sign of")
print("  a PC without anchoring it to a known feature.")

print("\n  Near-tied eigenvalues -> unstable individual eigenvectors:")
rng3 = np.random.default_rng(1703)
base = rng3.normal(0, 1, (80, 2)) @ np.diag([2.0, 1.98])   # nearly equal
Xtie = np.column_stack([base, rng3.normal(0, 0.3, (80, 20))])
sims = []
for b in range(50):
    idx = rng3.integers(0, 80, 80)                # bootstrap resample
    rb = pca(Xtie[idx])
    sims.append(abs(rb["loadings"][:, 0] @ pca(Xtie)["loadings"][:, 0]))
print(f"    eigenvalues 1,2 = {pca(Xtie)['eigenvalues'][:2].round(3)}")
print(f"    mean |cos| between bootstrap PC1 and original PC1: "
      f"{np.mean(sims):.3f}")
print("    Only the SUBSPACE spanned by tied PCs is well defined, not the")
print("    individual axes.")

# %% [markdown]
# ## 5. PC x metadata association, eq. (17.6): the key bioinformatics output

# %%
header("5. Which covariate explains each PC? (17.6)")

rng = np.random.default_rng(1704)
n_samp, G = 40, 2000
meta = pd.DataFrame({
    "condition": np.tile(["ctrl", "trt"], n_samp // 2),
    "batch": np.repeat(["b1", "b2", "b3", "b4"], n_samp // 4),
    "sex": rng.choice(["F", "M"], n_samp),
})
meta["lib_size"] = rng.lognormal(np.log(2e7), 0.3, n_samp)

expr = rng.normal(0, 1, (n_samp, G))
# BATCH is the dominant technical driver (a very common reality).
batch_load = rng.normal(0, 1, (4, G))
for i, b in enumerate(sorted(meta["batch"].unique())):
    expr[(meta["batch"] == b).values] += 2.2 * batch_load[i]
# CONDITION affects a smaller subset with a weaker effect.
expr[(meta["condition"] == "trt").values, :200] += 0.9
# Library size adds a global shift.
expr += 0.6 * ((np.log(meta["lib_size"]) - np.log(meta["lib_size"]).mean())
               / np.log(meta["lib_size"]).std()).values[:, None]

pc = pca(expr)


def pc_metadata_r2(scores, meta, n_pc=5):
    """stats.md eq. (17.6): R^2 of each PC regressed on each metadata column."""
    rows = []
    for j in range(n_pc):
        zj = scores[:, j]
        row = {"PC": j + 1}
        for col in meta.columns:
            v = meta[col]
            if v.dtype == object or str(v.dtype) == "category":
                D = pd.get_dummies(v, drop_first=True).astype(float).values
                Xd = np.column_stack([np.ones(len(zj)), D])
            else:
                Xd = np.column_stack([np.ones(len(zj)),
                                      (v - v.mean()) / v.std()])
            beta = np.linalg.lstsq(Xd, zj, rcond=None)[0]
            resid = zj - Xd @ beta
            row[col] = 1 - resid.var() / zj.var()
        rows.append(row)
    return pd.DataFrame(rows).set_index("PC")


tab = pc_metadata_r2(pc["scores"], meta)
tab.insert(0, "PVE", pc["pve"][:5])
print(tab.round(3).to_string())
print("\n  PC1 is overwhelmingly explained by BATCH, not condition. That is a")
print("  technical finding, not a discovery (stats.md Topic 19). Build this")
print("  table for EVERY dataset before you interpret a single PCA plot.")

# %% [markdown]
# ## 6. Two errors: over-reading separation, and double dipping

# %%
header("6. Separation in PC space is not evidence")

rng = np.random.default_rng(1705)
n_samp, G = 20, 5000
Xnull = rng.normal(0, 1, (n_samp, G))              # NO group structure at all
labels = np.tile([0, 1], n_samp // 2)
pn = pca(Xnull)
obs_sep = abs(pn["scores"][labels == 0, 0].mean()
              - pn["scores"][labels == 1, 0].mean())

null_sep = []
for _ in range(2000):
    perm = rng.permutation(labels)
    null_sep.append(abs(pn["scores"][perm == 0, 0].mean()
                        - pn["scores"][perm == 1, 0].mean()))
print(f"  n={n_samp}, p={G}, NO real group difference")
print(f"  observed PC1 separation between arbitrary labels: {obs_sep:.3f}")
print(f"  permutation p-value: "
      f"{(1 + np.sum(np.array(null_sep) >= obs_sep))/(len(null_sep)+1):.3f}")
print("  Random high-dimensional data ALWAYS spread out in the top PCs.")
print("  Permute the labels to get a null for any separation you plan to claim.")

print("\n--- Double dipping: selecting features by PC1, then testing them ---")
# Select the 200 features with the largest |PC1 loading|, then test those
# features for a difference between groups defined by the sign of PC1.
grp = (pn["scores"][:, 0] > np.median(pn["scores"][:, 0])).astype(int)
top_feats = np.argsort(-np.abs(pn["loadings"][:, 0]))[:200]
p_dip = st.ttest_ind(Xnull[grp == 0][:, top_feats],
                     Xnull[grp == 1][:, top_feats], axis=0).pvalue
rand_feats = rng.choice(G, 200, replace=False)
p_rand = st.ttest_ind(Xnull[grp == 0][:, rand_feats],
                      Xnull[grp == 1][:, rand_feats], axis=0).pvalue
print(f"  features selected by PC1 loading : {np.mean(p_dip < 0.05):.1%} "
      f"'significant' (data are pure noise)")
print(f"  randomly selected features       : {np.mean(p_rand < 0.05):.1%}")
print("\n  Random features are only mildly inflated (the GROUPS came from PC1,")
print("  which is a small leak). Selecting the FEATURES from the same PC1 as")
print("  well is what produces the large inflation: the loadings pick out")
print("  exactly the features that drive the split you then test. Selecting")
print("  features and groups from one decomposition guarantees significance.")
print("  Split the data, or use a selective-inference test.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(np.arange(1, 16), obs_ev[:15], "o-", label="observed")
ax.plot(np.arange(1, 16), thresh_ev[:15], "x--", label="95th pct permuted")
ax.axhline(mp_edge, color="green", ls=":", label="Marchenko-Pastur edge")
ax.set_xlabel("component"); ax.set_ylabel("eigenvalue")
ax.set_title("Eq. (17.5): how many components?", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
for b, mk in zip(sorted(meta["batch"].unique()), ["o", "s", "^", "D"]):
    sel = (meta["batch"] == b).values
    ax.scatter(pc["scores"][sel, 0], pc["scores"][sel, 1], marker=mk, s=45,
               label=f"batch {b}")
ax.set_xlabel(f"PC1 ({pc['pve'][0]:.0%})"); ax.set_ylabel(f"PC2 ({pc['pve'][1]:.0%})")
ax.set_aspect("equal", adjustable="datalim")
ax.set_title("PCA coloured by BATCH", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
for c, mk in zip(["ctrl", "trt"], ["o", "s"]):
    sel = (meta["condition"] == c).values
    ax.scatter(pc["scores"][sel, 0], pc["scores"][sel, 1], marker=mk, s=45,
               label=c)
ax.set_xlabel(f"PC1 ({pc['pve'][0]:.0%})"); ax.set_ylabel(f"PC2 ({pc['pve'][1]:.0%})")
ax.set_aspect("equal", adjustable="datalim")
ax.set_title("Same PCA coloured by CONDITION (no structure)", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.hist(null_sep, bins=50, color="lightsteelblue")
ax.axvline(obs_sep, color="red", lw=2, label="observed")
ax.set_xlabel("PC1 separation between labels")
ax.set_title("Separation on NOISE data: permutation null", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "pca.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/pca.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 17)
#
# 1. Always centre. Scale only with a stated reason.
# 2. Interpret PCs through loadings and metadata associations (17.6), never
#    through the appearance of a scatter plot.
# 3. Report the PVE for every displayed component and use equal aspect ratio.
# 4. Keep PCA-for-QC separate from PCs-as-covariates, and never test features
#    on the same data that defined the component.
#
# **Next:** `18_distances_and_clustering.py`
