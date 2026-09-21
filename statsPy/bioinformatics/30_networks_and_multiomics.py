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
# # Applied 30: Networks and multi-omics integration
#
# **Curriculum link:** `stats.md` -> Topic 30, equations (30.1)-(30.6)
# **Core modules used:** 10 (partial correlation), 16-17 (SVD), 19 (batch), 31
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | A `MultiAssayExperiment` cohort: RNA + protein + methylation on overlapping samples |
# | **Key threats** | Indirect edges, unstable modules at small $n$, shared **batch** masquerading as shared biology |
# | **Here** | Simulated from known latent factors so "shared biology" has a ground truth |
#
# ## The recurring failure
#
# Unsupervised methods always return modules and factors. The question is never
# "did I find structure?": it is "would this structure appear again?"

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import GroupKFold
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "30_networks_and_multiomics"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate a multi-omics cohort from known latent factors, eq. (30.6)
#
# $$\mathbf Y^{(m)}=\mathbf Z\mathbf W^{(m)\top}+\mathbf E^{(m)}$$
#
# We build factors with different *scopes*: one shared by all assays (real
# biology), one assay-specific, and one that is purely a **shared batch**.

# %%
header("1. A MultiAssayExperiment-like cohort (30.6)")

def simulate_multiomics(n=90, p_rna=400, p_prot=150, p_meth=300, seed=0):
    """Three assays generated from four latent factors of different scope."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    batch = np.array([f"b{i % 3}" for i in range(n)])
    phenotype = rng.normal(0, 1, n)

    # Latent factors (n x 4).
    Z = np.column_stack([
        phenotype + rng.normal(0, 0.35, n),        # F1: SHARED biology (all assays)
        rng.normal(0, 1, n),                       # F2: RNA-specific
        rng.normal(0, 1, n),                       # F3: protein+methylation
        (batch == "b1").astype(float) - (batch == "b2").astype(float)
        + rng.normal(0, 0.2, n),                   # F4: SHARED TECHNICAL (batch)
    ])

    def make_block(p, loads, noise=1.0):
        W = np.zeros((p, Z.shape[1]))
        for k, frac in enumerate(loads):
            m = int(p * frac)
            if m:
                idx = rng.choice(p, m, replace=False)
                W[idx, k] = rng.normal(0, 1.2, m)
        return (Z @ W.T + rng.normal(0, noise, (Z.shape[0], p))), W

    #                       F1    F2    F3    F4(batch)
    rna, W_rna = make_block(p_rna,  [0.25, 0.30, 0.00, 0.20])
    prot, W_prot = make_block(p_prot, [0.30, 0.00, 0.35, 0.20])
    meth, W_meth = make_block(p_meth, [0.20, 0.00, 0.30, 0.25])

    blocks = {
        "RNA": pd.DataFrame(rna, index=samples,
                            columns=[f"rna{i:03d}" for i in range(p_rna)]),
        "Protein": pd.DataFrame(prot, index=samples,
                                columns=[f"prot{i:03d}" for i in range(p_prot)]),
        "Methylation": pd.DataFrame(meth, index=samples,
                                    columns=[f"cg{i:03d}" for i in range(p_meth)]),
    }
    meta = pd.DataFrame({"batch": batch, "phenotype": phenotype}, index=samples)
    return blocks, meta, Z, {"RNA": W_rna, "Protein": W_prot, "Methylation": W_meth}


blocks, meta, Z_true, W_true = simulate_multiomics(seed=4001)
for k, v in blocks.items():
    print(f"  {k:<14}{v.shape[0]} samples x {v.shape[1]} features")
print(f"\n  TRUE latent factors:")
print(f"    F1: shared by ALL assays, correlated with phenotype (r = "
      f"{np.corrcoef(Z_true[:,0], meta['phenotype'])[0,1]:.2f})")
print(f"    F2: RNA-specific")
print(f"    F3: protein + methylation")
print(f"    F4: shared TECHNICAL factor driven by BATCH")
print("\n  F4 is the trap: it loads on every assay, so a naive integration will")
print("  report it as 'shared biology'.")

# %% [markdown]
# ## 2. Co-expression networks, eq. (30.1)-(30.3)

# %%
header("2. Soft thresholding, TOM and module eigengenes (30.1)-(30.3)")

def soft_threshold_adjacency(X, beta=6):
    """stats.md eq. (30.1): a_ij = |cor(x_i, x_j)|^beta."""
    C = np.corrcoef(X.values, rowvar=False)
    np.fill_diagonal(C, 0.0)
    return np.abs(C) ** beta


def tom(A):
    """stats.md eq. (30.2): topological overlap - shared-neighbour similarity."""
    k = A.sum(axis=1)
    num = A @ A + A
    den = np.minimum.outer(k, k) + 1 - A
    T = num / np.maximum(den, 1e-12)
    np.fill_diagonal(T, 1.0)
    return T


def scale_free_fit(A, n_bins=10):
    """R^2 of log(p(k)) on log(k) - how the soft-threshold power is chosen."""
    k = A.sum(axis=1)
    k = k[k > 0]
    if len(k) < n_bins:
        return np.nan
    bins = np.linspace(k.min(), k.max(), n_bins + 1)
    idx = np.digitize(k, bins[1:-1])
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(np.log10(k[m].mean()))
            ys.append(np.log10(m.sum() / len(k)))
    if len(xs) < 4:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1] ** 2)


X = blocks["RNA"]
print(f"  {'beta':>6}{'scale-free fit R^2':>22}{'mean connectivity':>20}")
for beta in [1, 2, 4, 6, 10]:
    A = soft_threshold_adjacency(X, beta)
    print(f"  {beta:>6}{scale_free_fit(A):>22.3f}{A.sum(axis=1).mean():>20.2f}")
print("  WGCNA picks the smallest beta reaching a scale-free fit around 0.85-0.9,")
print("  trading connectivity for topology.")

A = soft_threshold_adjacency(X, 6)
T = tom(A)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
D = 1 - T
np.fill_diagonal(D, 0.0)
D = (D + D.T) / 2
Zl = linkage(squareform(D, checks=False), method="average")
modules = fcluster(Zl, 4, criterion="maxclust")
print(f"\n  {len(np.unique(modules))} modules found; sizes = "
      f"{np.bincount(modules)[1:]}")


def module_eigengene(X, members):
    """stats.md eq. (30.3): the first PC of the module's expression."""
    sub = X.iloc[:, members].values
    sub = sub - sub.mean(axis=0)
    U, S, _ = np.linalg.svd(sub, full_matrices=False)
    pve = S[0] ** 2 / np.sum(S ** 2)
    return (U * S)[:, 0], pve


print(f"\n  {'module':>8}{'size':>7}{'PVE of ME':>12}{'|r| with F1':>13}"
      f"{'|r| with F4 (batch)':>21}{'|r| phenotype':>15}")
for m in np.unique(modules):
    members = np.where(modules == m)[0]
    if len(members) < 5:
        continue
    me, pve = module_eigengene(X, members)
    print(f"  {m:>8}{len(members):>7}{pve:>12.3f}"
          f"{abs(np.corrcoef(me, Z_true[:,0])[0,1]):>13.3f}"
          f"{abs(np.corrcoef(me, Z_true[:,3])[0,1]):>21.3f}"
          f"{abs(np.corrcoef(me, meta['phenotype'])[0,1]):>15.3f}")
print("\n  At least one module tracks the BATCH factor. Module-trait tables must")
print("  include technical covariates, or you will publish a batch effect as a")
print("  biological module (Module 19).")

# %% [markdown]
# ## 3. Marginal vs partial correlation, eq. (10.5)-(10.6), (30.4)
#
# Marginal correlation confuses direct and indirect association. A zero in the
# **precision** matrix means conditional independence.

# %%
header("3. Direct vs indirect edges (30.4)")

rng = np.random.default_rng(4002)
n = 250
# A chain: A -> B -> C. A and C are marginally correlated, conditionally not.
Ax = rng.normal(0, 1, n)
Bx = 0.9 * Ax + rng.normal(0, 0.5, n)
Cx = 0.9 * Bx + rng.normal(0, 0.5, n)
M = np.column_stack([Ax, Bx, Cx])
S = np.corrcoef(M, rowvar=False)
Omega = np.linalg.inv(S)
part = -Omega / np.sqrt(np.outer(np.diag(Omega), np.diag(Omega)))
print(f"  Chain A -> B -> C (there is NO direct A-C edge)")
print(f"  {'pair':<8}{'marginal r':>13}{'partial r':>12}")
for (i, j, lab) in [(0, 1, "A-B"), (1, 2, "B-C"), (0, 2, "A-C")]:
    print(f"  {lab:<8}{S[i,j]:>13.3f}{part[i,j]:>12.3f}")
print("\n  The A-C marginal correlation is strong and entirely INDIRECT. A")
print("  network built on marginal correlations fills with such edges.")

def graphical_lasso(S, lam, n_iter=120, tol=1e-4):
    """stats.md eq. (30.4): a simple coordinate-descent graphical lasso.

    Maximises  log det(Theta) - tr(S Theta) - lambda ||Theta||_1.
    """
    p = S.shape[0]
    W = S + lam * np.eye(p)
    for _ in range(n_iter):
        W_old = W.copy()
        for j in range(p):
            idx = [k for k in range(p) if k != j]
            W11 = W[np.ix_(idx, idx)]
            s12 = S[idx, j]
            # Lasso via coordinate descent on beta.
            beta = np.zeros(p - 1)
            for _ in range(80):
                for k in range(p - 1):
                    r = s12[k] - W11[k] @ beta + W11[k, k] * beta[k]
                    beta[k] = np.sign(r) * max(abs(r) - lam, 0) / max(W11[k, k], 1e-9)
            W[idx, j] = W11 @ beta
            W[j, idx] = W[idx, j]
        if np.max(np.abs(W - W_old)) < tol:
            break
    return np.linalg.inv(W)


print(f"\n  Graphical lasso on the chain, at several penalties:")
print(f"  {'lambda':>9}{'|Theta_AC|':>13}{'|Theta_AB|':>13}{'edges kept':>13}")
for lam in [0.0, 0.05, 0.15, 0.35]:
    Th = graphical_lasso(S, lam) if lam > 0 else np.linalg.inv(S)
    pc = -Th / np.sqrt(np.outer(np.diag(Th), np.diag(Th)))
    off = np.abs(pc[np.triu_indices(3, 1)])
    print(f"  {lam:>9.2f}{abs(pc[0,2]):>13.4f}{abs(pc[0,1]):>13.4f}"
          f"{int((off > 0.05).sum()):>13}")
print("  Increasing lambda drives the spurious A-C partial correlation to zero")
print("  first, because it is the weakest. The l1 penalty also makes (30.4)")
print("  solvable when p > n, where the sample covariance is singular.")

# %% [markdown]
# ## 4. Multi-block factor analysis, eq. (30.6)

# %%
header("4. Recovering latent factors across assays (30.6)")

def multiblock_factors(blocks, k=4):
    """A MOFA-style decomposition: scale each block, concatenate, take an SVD.

    Scaling each block by its total variance prevents the largest assay from
    dominating - the same decision as `scale=TRUE` in PCA (eq. 17.4).
    """
    scaled = []
    for name, B in blocks.items():
        Bc = B.values - B.values.mean(axis=0)
        Bc = Bc / np.sqrt((Bc ** 2).sum() / Bc.shape[0])
        scaled.append(Bc)
    C = np.hstack(scaled)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    Zhat = (U * S)[:, :k]

    # Variance explained by each factor WITHIN each block - the key output.
    ve = {}
    start = 0
    for (name, B), Bs in zip(blocks.items(), scaled):
        Vb = Vt[:k, start:start + Bs.shape[1]]
        ve[name] = [float(((Zhat[:, j:j+1] @ Vb[j:j+1]) ** 2).sum() / (Bs ** 2).sum())
                    for j in range(k)]
        start += Bs.shape[1]
    return Zhat, pd.DataFrame(ve, index=[f"Factor{j+1}" for j in range(k)])


Zhat, ve = multiblock_factors(blocks, k=4)
print("  Variance explained by each inferred factor, per assay:")
print((ve * 100).round(2).to_string())

print(f"\n  Correlation of each inferred factor with the TRUE factors:")
print(f"  {'':<10}{'F1 (shared bio)':>17}{'F2 (RNA)':>11}{'F3 (prot+meth)':>17}"
      f"{'F4 (BATCH)':>13}{'phenotype':>12}")
for j in range(4):
    cors = [abs(np.corrcoef(Zhat[:, j], Z_true[:, t])[0, 1]) for t in range(4)]
    cp = abs(np.corrcoef(Zhat[:, j], meta["phenotype"])[0, 1])
    print(f"  Factor{j+1:<4}{cors[0]:>17.3f}{cors[1]:>11.3f}{cors[2]:>17.3f}"
          f"{cors[3]:>13.3f}{cp:>12.3f}")

print(f"\n  Association of each inferred factor with BATCH (the essential check):")
print(f"  {'factor':<10}{'R^2 on batch':>15}{'ANOVA p':>12}  verdict")
for j in range(4):
    Dm = pd.get_dummies(meta["batch"], drop_first=True).astype(float).values
    Xd = np.column_stack([np.ones(len(Zhat)), Dm])
    resid = Zhat[:, j] - Xd @ np.linalg.lstsq(Xd, Zhat[:, j], rcond=None)[0]
    r2 = 1 - resid.var() / Zhat[:, j].var()
    groups = [Zhat[(meta["batch"] == b).values, j] for b in meta["batch"].unique()]
    pval = st.f_oneway(*groups).pvalue
    verdict = "TECHNICAL - do not interpret" if r2 > 0.3 else "not batch-driven"
    print(f"  Factor{j+1:<4}{r2:>15.3f}{pval:>12.2e}  {verdict}")

print("\n  A factor that loads on EVERY assay is not automatically shared")
print("  biology. Assays processed in the same batches share technical")
print("  variation, and (30.6) will happily merge it into a 'shared' factor.")
print("  Always regress every factor on the technical metadata first.")

# %% [markdown]
# ## 5. Module and factor stability, and the $n$ problem

# %%
header("5. Would this structure appear again?")

def module_stability(X, k=4, n_boot=25, frac=0.8, seed=0):
    """Subsample SAMPLES, recluster features, measure co-module agreement."""
    rng = np.random.default_rng(seed)
    A0 = soft_threshold_adjacency(X, 6)
    D0 = 1 - tom(A0); np.fill_diagonal(D0, 0); D0 = (D0 + D0.T) / 2
    base = fcluster(linkage(squareform(D0, checks=False), "average"), k,
                    criterion="maxclust")
    same0 = base[:, None] == base[None, :]
    agree = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        Ab = soft_threshold_adjacency(X.iloc[idx], 6)
        Db = 1 - tom(Ab); np.fill_diagonal(Db, 0); Db = (Db + Db.T) / 2
        lab = fcluster(linkage(squareform(Db, checks=False), "average"), k,
                       criterion="maxclust")
        sameb = lab[:, None] == lab[None, :]
        off = ~np.eye(len(base), dtype=bool)
        agree.append((same0[off] == sameb[off]).mean())
    return float(np.mean(agree))


print(f"  {'samples n':>11}{'module stability (co-clustering agreement)':>44}")
for n_sub in [15, 30, 60, 90]:
    Xs = blocks["RNA"].iloc[:n_sub]
    print(f"  {n_sub:>11}{module_stability(Xs, n_boot=12, seed=n_sub):>44.3f}")
print("\n  Stability rises with n, though note this metric is dominated by the")
print("  many feature PAIRS that sit in different modules under any labelling,")
print("  so it has a high floor - read the TREND, not the absolute value. The")
print("  practical point stands: correlations estimated from few samples are")
print("  noisy, WGCNA's own guidance is n >= 15 and preferably >= 20, and a")
print("  dendrogram alone is not evidence that a module is real. Report a")
print("  stability measure, and preserve it in independent data (Problem 3).")

# %% [markdown]
# ## 6. Cross-validation at the biological-unit level

# %%
header("6. If a factor is used to predict, validate at the patient level")

rng = np.random.default_rng(4003)
# Each patient contributes 3 technical replicates - a very common structure.
patient = np.repeat(np.arange(30), 3)
reps = blocks["RNA"].iloc[:90].values + rng.normal(0, 0.3,
                                                   blocks["RNA"].iloc[:90].shape)
y = meta["phenotype"].values[:90]

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

pipe = Pipeline([("sc", StandardScaler()),
                 ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])
r2_random = cross_val_score(pipe, reps, y, cv=KFold(5, shuffle=True,
                                                    random_state=0), scoring="r2")
r2_group = cross_val_score(pipe, reps, y, cv=GroupKFold(5), groups=patient,
                           scoring="r2")
print(f"  random 5-fold CV over replicates : R^2 = {r2_random.mean():.3f}")
print(f"  GroupKFold by PATIENT            : R^2 = {r2_group.mean():.3f}")
print(f"  optimism from replicate leakage  : {r2_random.mean()-r2_group.mean():+.3f}")
print("\n  Technical replicates of the same patient are near-duplicates. Random")
print("  splitting puts them on both sides of the fold boundary - the same leak")
print("  as cells from one donor in Module 31.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
betas = [1, 2, 4, 6, 8, 10, 14]
ax.plot(betas, [scale_free_fit(soft_threshold_adjacency(X, b)) for b in betas],
        "o-")
ax.axhline(0.85, color="red", ls="--", label="typical target")
ax.set_xlabel("soft-threshold power $\\beta$"); ax.set_ylabel("scale-free fit $R^2$")
ax.set_title("Eq. (30.1): choosing $\\beta$", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
im = ax.imshow((ve * 100).values, cmap="Blues", aspect="auto")
ax.set_xticks(range(ve.shape[1])); ax.set_xticklabels(ve.columns, fontsize=7)
ax.set_yticks(range(ve.shape[0])); ax.set_yticklabels(ve.index, fontsize=7)
ax.set_title("Eq. (30.6): variance explained per assay (%)", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1, 0]
for b, c in zip(sorted(meta["batch"].unique()), ["C0", "C1", "C2"]):
    m = (meta["batch"] == b).values
    ax.scatter(Zhat[m, 0], Zhat[m, 1], s=25, color=c, label=f"batch {b}")
ax.set_xlabel("Factor 1"); ax.set_ylabel("Factor 2")
ax.set_title("Inferred factors coloured by BATCH", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
ns = [15, 30, 60, 90]
ax.plot(ns, [module_stability(blocks["RNA"].iloc[:k], n_boot=10, seed=k)
             for k in ns], "o-")
ax.set_xlabel("number of samples"); ax.set_ylabel("module stability")
ax.set_title("Modules need samples", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "networks.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/networks.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Module-trait associations need a null
#
# Correlate every module eigengene with a RANDOM phenotype and count how often
# you get p < 0.05. Then build a permutation null that respects the module
# structure.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_p = np.random.default_rng(23)
# mods = [np.where(modules == m)[0] for m in np.unique(modules)
#         if (modules == m).sum() >= 5]
# eig = [module_eigengene(X, mm)[0] for mm in mods]
# hits = 0; total = 0
# for _ in range(300):
#     fake = rng_p.normal(0, 1, len(X))
#     for e in eig:
#         hits += abs(st.pearsonr(e, fake).pvalue) < 0.05
#         total += 1
# print(f"  module-trait tests with a RANDOM trait: {hits/total:.1%} at p<0.05")
# print(f"  (nominal 5%; {len(eig)} modules tested each time)")
#
# # Each individual test is calibrated, but you will typically test many modules
# # against many traits. The FAMILY is modules x traits (Module 08). Adjust
# # across it, and remember eigengenes of overlapping modules are correlated, so
# # BH across them is not conservative in the way you might hope - a permutation
# # null over the whole pipeline is the honest option.

# %% [markdown]
# ### Problem 2: Would you have found the batch factor without the labels?
#
# Remove `batch` from the metadata and try to detect the technical factor from
# the data alone (e.g. by checking whether any factor is bimodal/trimodal, or by
# correlating factors with processing order). What does this tell you about
# recording metadata?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# from scipy.stats import kurtosis
# print(f"  {'factor':<10}{'kurtosis':>11}{'dip-like spread':>18}"
#       f"{'TRUE batch R^2':>17}")
# for j in range(4):
#     f = Zhat[:, j]
#     # A trimodal batch factor has LOW (platykurtic) kurtosis.
#     Dm = pd.get_dummies(meta["batch"], drop_first=True).astype(float).values
#     Xd = np.column_stack([np.ones(len(f)), Dm])
#     resid = f - Xd @ np.linalg.lstsq(Xd, f, rcond=None)[0]
#     r2 = 1 - resid.var() / f.var()
#     print(f"  Factor{j+1:<4}{kurtosis(f):>11.3f}"
#           f"{np.percentile(f,90)-np.percentile(f,10):>18.3f}{r2:>17.3f}")
#
# # You can sometimes SUSPECT a technical factor from its shape, but you cannot
# # confirm or adjust for it without the labels. This is the practical argument
# # for recording every processing variable - date, operator, plate, reagent lot,
# # instrument - at collection time (Module 19). Surrogate-variable methods
# # (eq. 19.5) estimate such factors when the labels are missing, but they cannot
# # tell you WHICH latent factor is technical and which is biology.

# %% [markdown]
# ### Problem 3: Preservation in an independent cohort
#
# Simulate a second, independent cohort from the same generative model and test
# whether the RNA modules are preserved (do genes that co-cluster in cohort 1
# still co-cluster in cohort 2?).

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# blocks2, meta2, Z2, _ = simulate_multiomics(seed=999)
# X2 = blocks2["RNA"]
# A2 = soft_threshold_adjacency(X2, 6)
# D2 = 1 - tom(A2); np.fill_diagonal(D2, 0); D2 = (D2 + D2.T) / 2
# lab2 = fcluster(linkage(squareform(D2, checks=False), "average"), 4,
#                 criterion="maxclust")
# from sklearn.metrics import adjusted_rand_score
# print(f"  adjusted Rand index between cohort-1 and cohort-2 modules: "
#       f"{adjusted_rand_score(modules, lab2):.3f}")
# # A cheap, informative preservation statistic: does the module's mean
# # within-module correlation survive?
# for m in np.unique(modules):
#     mm = np.where(modules == m)[0]
#     if len(mm) < 5: continue
#     c1 = np.corrcoef(X.iloc[:, mm].values, rowvar=False)
#     c2 = np.corrcoef(X2.iloc[:, mm].values, rowvar=False)
#     off = ~np.eye(len(mm), dtype=bool)
#     print(f"    module {m}: mean |r| cohort1 = {np.abs(c1[off]).mean():.3f}, "
#           f"cohort2 = {np.abs(c2[off]).mean():.3f}")
#
# # Modules driven by a real latent factor preserve their internal correlation
# # in a new cohort; modules that were noise do not. This is the idea behind
# # WGCNA's Zsummary preservation statistic, and it is the minimum standard
# # before naming a module a biological programme.

# %% [markdown]
# ## What to take away
#
# 1. Decide the goal: visualisation, latent biology, prediction, mechanism -
#    before picking a method; they have different validation standards.
# 2. Use partial correlation / graphical models for "direct association" claims.
# 3. **Check that shared factors are not shared batches.**
# 4. Report module preservation or factor reproducibility in independent data.
# 5. Cross-validate at the biological-unit level.
#
# **You have finished the applied track.** Return to `stats.md` for the theory,
# or work through `statsPy/exercises/` for integrative problem sets.
# 
# **Next:** `41_trajectory_and_pseudotime.py`
