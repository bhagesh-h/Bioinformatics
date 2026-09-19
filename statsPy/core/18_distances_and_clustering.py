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
# # Module 18: Distances, clustering, and embeddings
#
# **Curriculum link:** `stats.md` -> Topic 18, equations (18.1)-(18.14)
#
# Clustering always returns clusters. Which of them would survive a different
# distance, a different seed, or a different half of the data?
#
# ## What you will learn
#
# 1. The distance zoo (18.1)-(18.5), and the identity linking Euclidean distance
#    on z-scores to correlation (18.6).
# 2. The curse of dimensionality (18.7): why scRNA-seq clusters on PCs.
# 3. Linkage choice (18.8)-(18.9) and k-means (18.10).
# 4. Choosing $K$: silhouette (18.11), gap statistic (18.12), **stability**.
# 5. Why t-SNE/UMAP distances and cluster sizes are meaningless (18.14).
# 6. Double dipping: testing the features that defined the clusters.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "18_distances_and_clustering"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. The distance zoo, eq. (18.1)-(18.6)

# %%
header("1. Different distances answer different questions (18.1)-(18.5)")

# Three gene expression profiles across 6 conditions.
profiles = pd.DataFrame({
    "geneA": [1., 2., 3., 4., 5., 6.],        # rising
    "geneB": [11., 12., 13., 14., 15., 16.],  # same SHAPE, 10x higher level
    "geneC": [6., 5., 4., 3., 2., 1.],        # mirror image
}).T

def correlation_distance(a, b):
    """stats.md eq. (18.2): 1 - Pearson r. Ignores level and scale."""
    return 1 - np.corrcoef(a, b)[0, 1]


def cosine_distance(a, b):
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def bray_curtis(a, b):
    """stats.md eq. (18.4): abundance-based ecological dissimilarity."""
    return np.sum(np.abs(a - b)) / np.sum(a + b)


def jaccard_distance(a, b):
    """stats.md eq. (18.3): presence/absence."""
    A, B = a > 0, b > 0
    return 1 - np.sum(A & B) / np.sum(A | B)


print(f"  {'pair':<14}{'Euclidean':>11}{'correlation':>13}{'cosine':>9}"
      f"{'Bray-Curtis':>13}")
for i, j in [("geneA", "geneB"), ("geneA", "geneC"), ("geneB", "geneC")]:
    a, b = profiles.loc[i].values, profiles.loc[j].values
    print(f"  {i+'-'+j:<14}{np.linalg.norm(a-b):>11.3f}"
          f"{correlation_distance(a, b):>13.3f}{cosine_distance(a, b):>9.3f}"
          f"{bray_curtis(a, b):>13.3f}")
print("\n  A-B: identical SHAPE, different level. Correlation distance says 0")
print("       (same pattern); Euclidean says 24.5 (very different).")
print("  Choose the distance from the QUESTION: 'same regulation pattern' ->")
print("  correlation; 'same absolute abundance' -> Euclidean/Bray-Curtis.")

# Equation (18.6): on z-scored features, Euclidean distance IS correlation.
print("\n  Verifying eq. (18.6): d^2_Euclid(z_x, z_y) = 2(p-1)(1 - r)")
rng = np.random.default_rng(1801)
p = 50
x = rng.normal(0, 1, p)
y = 0.6 * x + rng.normal(0, 1, p)
zx = (x - x.mean()) / x.std(ddof=1)
zy = (y - y.mean()) / y.std(ddof=1)
r = np.corrcoef(x, y)[0, 1]
print(f"    d^2 computed   = {np.sum((zx-zy)**2):.6f}")
print(f"    2(p-1)(1-r)    = {2*(p-1)*(1-r):.6f}   <- identical")
print("  So clustering z-scored features by Euclidean distance IS clustering")
print("  by correlation. Knowing this prevents a lot of method-shopping.")

# %% [markdown]
# ## 2. The curse of dimensionality, eq. (18.7)

# %%
header("2. In high dimensions all points are equidistant (18.7)")

rng = np.random.default_rng(1802)
print(f"  {'dimensions p':>14}{'(max-min)/min distance':>25}")
for p in [2, 10, 100, 1000, 10000]:
    X = rng.normal(0, 1, (200, p))
    d = pdist(X)
    D = squareform(d)
    np.fill_diagonal(D, np.inf)
    ratios = [(D[i].max() if np.isfinite(D[i].max()) else np.nan) for i in range(200)]
    mn = D.min(axis=1)
    mx = np.where(np.isinf(D), -np.inf, D).max(axis=1)
    print(f"  {p:>14,}{np.mean((mx - mn) / mn):>25.4f}")
print("\n  The contrast between nearest and farthest neighbour collapses.")
print("  'Nearest neighbour' loses meaning - which is why EVERY scRNA-seq")
print("  pipeline builds its neighbour graph in the top ~30 PCs, not in")
print("  20,000-dimensional gene space.")

# %% [markdown]
# ## 3. Linkage and k-means, eq. (18.8)-(18.10)

# %%
header("3. Linkage choice changes the answer (18.8)-(18.10)")

rng = np.random.default_rng(1803)
# Three well-separated Gaussian blobs of UNEQUAL size.
truth = np.repeat([0, 1, 2], [80, 40, 15])
centres = np.array([[0, 0], [5, 0], [2.5, 4.5]])
Xb = np.vstack([rng.normal(centres[k], 0.9, (nk, 2))
                for k, nk in zip(range(3), [80, 40, 15])])

print(f"  {'linkage':<14}{'adjusted Rand vs truth':>25}")
for method in ["single", "complete", "average", "ward"]:
    Z = linkage(Xb, method=method)
    lab = fcluster(Z, 3, criterion="maxclust")
    print(f"  {method:<14}{adjusted_rand_score(truth, lab):>25.3f}")

km = KMeans(n_clusters=3, n_init=25, random_state=0).fit(Xb)
print(f"  {'k-means':<14}{adjusted_rand_score(truth, km.labels_):>25.3f}")

print("\n  Single linkage chains and usually fails on omics data. Ward and")
print("  complete do well on compact blobs. Note ward REQUIRES Euclidean")
print("  distance - applying it to a correlation distance is not meaningful.")

print("\n  k-means depends on initialisation - always use n_init >= 25:")
for n_init in [1, 5, 25]:
    aris = [adjusted_rand_score(
        truth, KMeans(3, n_init=n_init, random_state=s).fit(Xb).labels_)
        for s in range(20)]
    print(f"    n_init={n_init:>3}: ARI mean {np.mean(aris):.3f}, "
          f"min {np.min(aris):.3f}, worst-case failures "
          f"{np.sum(np.array(aris) < 0.8)}/20")

# %% [markdown]
# ## 4. Choosing K: silhouette, gap statistic, and stability

# %%
header("4. How many clusters? (18.11)-(18.12) and stability")

def gap_statistic(X, k_max=8, B=20, seed=0):
    """stats.md eq. (18.12): compare log W_k to a uniform reference."""
    rng = np.random.default_rng(seed)

    def Wk(X, k):
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        return sum(((X[km.labels_ == j] - km.cluster_centers_[j]) ** 2).sum()
                   for j in range(k))

    mins, maxs = X.min(axis=0), X.max(axis=0)
    gaps, sks = [], []
    for k in range(1, k_max + 1):
        logW = np.log(Wk(X, k))
        ref = [np.log(Wk(rng.uniform(mins, maxs, X.shape), k)) for _ in range(B)]
        gaps.append(np.mean(ref) - logW)
        sks.append(np.std(ref, ddof=1) * np.sqrt(1 + 1 / B))
    return np.array(gaps), np.array(sks)


def cluster_stability(X, k, n_boot=40, frac=0.8, seed=0):
    """Subsample, recluster, and measure ARI against the full-data labels."""
    rng = np.random.default_rng(seed)
    base = KMeans(k, n_init=25, random_state=0).fit_predict(X)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        lab = KMeans(k, n_init=25, random_state=int(rng.integers(1e6))
                     ).fit_predict(X[idx])
        scores.append(adjusted_rand_score(base[idx], lab))
    return np.mean(scores)


gaps, sks = gap_statistic(Xb, k_max=8, seed=5)
print(f"  {'K':>3}{'silhouette':>13}{'gap':>9}{'gap s_k':>10}{'stability (ARI)':>18}")
for k in range(2, 9):
    lab = KMeans(k, n_init=25, random_state=0).fit_predict(Xb)
    sil = silhouette_score(Xb, lab)
    stab = cluster_stability(Xb, k, seed=k)
    print(f"  {k:>3}{sil:>13.3f}{gaps[k-1]:>9.3f}{sks[k-1]:>10.3f}{stab:>18.3f}")

# Gap rule: smallest K with Gap(K) >= Gap(K+1) - s_{K+1}
k_gap = next((k for k in range(1, 8) if gaps[k-1] >= gaps[k] - sks[k]), 8)
print(f"\n  gap-statistic rule selects K = {k_gap}   (truth: 3)")
print("  Stability is the most honest criterion for biology: an unstable")
print("  cluster should not be given a cell-type name.")

# The critical negative control: a SINGLE homogeneous cloud.
print("\n  Negative control - one homogeneous Gaussian cloud, no clusters:")
Xnull = np.random.default_rng(9).normal(0, 1, (135, 2))
gn, sn = gap_statistic(Xnull, k_max=6, seed=11)
for k in range(2, 6):
    lab = KMeans(k, n_init=25, random_state=0).fit_predict(Xnull)
    print(f"    K={k}: silhouette {silhouette_score(Xnull, lab):.3f}, "
          f"gap {gn[k-1]:+.3f}, stability {cluster_stability(Xnull, k, seed=k):.3f}")
print("  Silhouette is POSITIVE even with no real structure - it always finds")
print("  something. The gap statistic and stability are far more informative.")

# %% [markdown]
# ## 5. t-SNE / UMAP: visualisation, not measurement, eq. (18.14)

# %%
header("5. Embedding distances and cluster areas are meaningless (18.14)")

from sklearn.manifold import TSNE

rng = np.random.default_rng(1804)
# Three groups at KNOWN, very different true separations.
g1 = rng.normal([0, 0] + [0] * 8, 1.0, (60, 10))
g2 = rng.normal([6, 0] + [0] * 8, 1.0, (60, 10))     # near g1
g3 = rng.normal([40, 0] + [0] * 8, 1.0, (60, 10))    # VERY far from both
Xt = np.vstack([g1, g2, g3])
lab = np.repeat([0, 1, 2], 60)

true_d12 = np.linalg.norm(g1.mean(0) - g2.mean(0))
true_d13 = np.linalg.norm(g1.mean(0) - g3.mean(0))
print(f"  TRUE centroid distances: d(1,2) = {true_d12:.2f}, "
      f"d(1,3) = {true_d13:.2f}   ratio {true_d13/true_d12:.2f}")

print(f"\n  {'init':>8}{'perplexity':>12}{'seed':>6}{'emb d(1,2)':>13}"
      f"{'emb d(1,3)':>13}{'ratio':>8}")
for init in ["pca", "random"]:
    for perp in [5, 30]:
        for seed in [0, 1]:
            emb = TSNE(n_components=2, perplexity=perp, random_state=seed,
                       init=init, max_iter=300).fit_transform(Xt)
            d12 = np.linalg.norm(emb[lab == 0].mean(0) - emb[lab == 1].mean(0))
            d13 = np.linalg.norm(emb[lab == 0].mean(0) - emb[lab == 2].mean(0))
            print(f"  {init:>8}{perp:>12}{seed:>6}{d12:>13.2f}{d13:>13.2f}"
                  f"{d13/d12:>8.2f}")

print(f"\n  The TRUE ratio is {true_d13/true_d12:.1f}. No embedded ratio comes close,")
print("  and every one of them changes with perplexity. t-SNE minimises an")
print("  asymmetric KL divergence (eq. 18.14) that preserves LOCAL")
print("  neighbourhoods and deliberately discards global geometry.")
print("\n  Note the init column: with init='pca' the result is DETERMINISTIC")
print("  given the data (which is why it is now the default and why you should")
print("  use it); with init='random' the same settings give different maps for")
print("  different seeds. Either way, report init, perplexity, seed and n_PCs.")
print("\n  NEVER: measure distances on an embedding; compare cluster areas;")
print("  report an embedding without the seed, perplexity, and n_PCs used.")

# %% [markdown]
# ## 6. Double dipping in clustering

# %%
header("6. Testing the features that defined the clusters")

rng = np.random.default_rng(1805)
# The inflation grows with p/n: the more features k-means can exploit to find
# a split, the more the subsequent test is testing its own construction.
n, G = 100, 1500
Xhomog = rng.normal(0, 1, (n, G))                 # ONE homogeneous population
lab2 = KMeans(2, n_init=25, random_state=0).fit_predict(Xhomog)
p_naive = st.ttest_ind(Xhomog[lab2 == 0], Xhomog[lab2 == 1], axis=0).pvalue

print(f"  Data: ONE homogeneous Gaussian cloud (n={n}, p={G}), no subgroups.")
print(f"  k-means with K=2 produces clusters of size "
      f"{np.bincount(lab2)}")
print(f"  'differential' features at p < 0.05 : "
      f"{np.mean(p_naive < 0.05):.1%}  (should be 5%)")
print(f"  surviving BH FDR 0.05               : "
      f"{np.sum(st.false_discovery_control(p_naive) < 0.05)}")

# The fix: split the samples. Cluster on half, test on the other half.
def split_test(X, seed=0, verbose=False):
    """Cluster on one half, then define the SAME split on the held-out half.

    The honest held-out analogue of "cluster then test": learn the separating
    direction (the difference between the two training centroids) on one half,
    project the other half onto it, and split at the median. This always yields
    two comparable groups, and - importantly - the direction was estimated from
    data independent of the samples being tested.
    """
    r = np.random.default_rng(seed)
    idx = r.permutation(len(X))
    tr, te = idx[:len(X) // 2], idx[len(X) // 2:]
    km_tr = KMeans(2, n_init=25, random_state=0).fit(X[tr])
    if verbose:
        print(f"    training-half cluster sizes : {np.bincount(km_tr.labels_)}")
        assigned = np.bincount(km_tr.predict(X[te]), minlength=2)
        print(f"    held-out samples assigned to those centroids: {assigned}")
        if assigned.min() == 0:
            print("    -> the held-out half lands ENTIRELY in one cluster:")
            print("       the 'cluster' found in the training half does not")
            print("       exist in independent data. That failure to replicate")
            print("       is itself the diagnostic.")
    direction = km_tr.cluster_centers_[1] - km_tr.cluster_centers_[0]
    proj = X[te] @ direction
    lab_te = (proj > np.median(proj)).astype(int)
    return st.ttest_ind(X[te][lab_te == 0], X[te][lab_te == 1], axis=0).pvalue


p_split = split_test(Xhomog, seed=0, verbose=True)
print(f"\n  With sample splitting (direction learned on half, tested on the other):")
print(f"    'differential' features at p < 0.05 : {np.mean(p_split < 0.05):.1%}")
print(f"    surviving BH FDR 0.05               : "
      f"{np.sum(st.false_discovery_control(p_split) < 0.05)}")

print("\n  Splitting restores calibration. Across a range of n and p the")
print("  same-data analysis sits consistently ABOVE the nominal 5% while the")
print("  split-sample analysis sits at it:")
print(f"  {'n':>6}{'p':>7}{'% p<0.05, same data':>22}{'% p<0.05, split':>18}")
for (nn, pp) in [(100, 300), (60, 1000), (40, 3000), (30, 6000)]:
    Xh = np.random.default_rng(nn + pp).normal(0, 1, (nn, pp))
    lb = KMeans(2, n_init=25, random_state=0).fit_predict(Xh)
    pn_ = st.ttest_ind(Xh[lb == 0], Xh[lb == 1], axis=0).pvalue
    ps_ = split_test(Xh, seed=7)
    print(f"  {nn:>6}{pp:>7}{np.mean(pn_ < 0.05):>21.1%}"
          f"{np.mean(ps_ < 0.05):>18.1%}")
print("\n  For isotropic noise the p-value inflation is real but modest; the")
print("  decisive evidence against a noise cluster is the FAILURE TO REPLICATE")
print("  shown above. In practice the damage is larger than this table suggests,")
print("  because analysts do not report the whole p-value distribution - they")
print("  report the TOP 'marker genes', which are precisely the features the")
print("  clustering selected on.")
print("\n  This is why single-cell CLUSTER MARKER p-values are not valid")
print("  evidence of a condition effect (stats.md Topic 21).")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
for k in range(3):
    ax.scatter(Xb[truth == k, 0], Xb[truth == k, 1], s=18, label=f"group {k}")
ax.set_title("Three blobs of unequal size", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
ks = np.arange(2, 9)
ax.plot(ks, [silhouette_score(Xb, KMeans(k, n_init=25, random_state=0)
                              .fit_predict(Xb)) for k in ks], "o-",
        label="silhouette")
ax.plot(ks, [cluster_stability(Xb, k, seed=k) for k in ks], "s-",
        label="stability (ARI)")
ax.axvline(3, color="red", ls="--", label="truth")
ax.set_xlabel("K"); ax.set_title("Eq. (18.11): choosing K", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
emb = TSNE(n_components=2, perplexity=30, random_state=0, init="pca",
           max_iter=300).fit_transform(Xt)
for k in range(3):
    ax.scatter(emb[lab == k, 0], emb[lab == k, 1], s=14, label=f"group {k}")
ax.set_title("t-SNE: group 3 is 6.7x further away in truth", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.hist(p_naive, bins=30, alpha=0.7, label="cluster then test (same data)")
ax.hist(p_split, bins=30, alpha=0.7, label="sample splitting")
ax.axhline(len(p_naive) / 30, color="red", ls="--", label="uniform")
ax.set_xlabel("p-value")
ax.set_title("Double dipping on homogeneous data", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "clustering.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/clustering.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 18)
#
# 1. Pick the distance from the measurement scale.
# 2. Cluster in a reduced space (top PCs) for high-dimensional data (18.7).
# 3. Report $K$-selection evidence **and** stability, not just a coloured plot.
# 4. Never test the features that defined the clusters on the same data.
# 5. Report seed, perplexity/n_neighbors and n_PCs for every embedding, and
#    quantify nothing on embedding coordinates.
#
# **Next:** `19_batch_effects.py`
