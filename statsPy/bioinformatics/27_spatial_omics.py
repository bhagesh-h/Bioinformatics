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
# # Applied 27: Spatial transcriptomics and spatial omics
#
# **Curriculum link:** `stats.md` -> Topic 27, equations (27.1)-(27.5)
# **Core modules used:** 01 (replication), 13 (offsets), 14 (mixed models)
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | 10x Visium: ~3,000 spots per section, several sections per patient |
# | **Assay** | Spot x gene UMI counts plus (x, y) coordinates |
# | **Key threats** | Spatial autocorrelation, edge effects, **spots are not patients** |
# | **Here** | Simulated tissue with domains, spatial genes and multiple patients |
#
# ## The central trap
#
# A section contains thousands of spots but is **one observation of one
# patient**. Spot-level p-values describe that section, not the condition.

# %%
import os
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

MODULE_NAME = "27_spatial_omics"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate replicated tissue sections

# %%
header("1. A multi-patient spatial experiment")

def simulate_spatial(n_patients=8, spots_per_section=700, G=300, n_svg=40,
                     seed=0):
    """Hexagonal-ish grid; a spatial domain drives some genes; patients differ."""
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(spots_per_section))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    coords[:, 0] += (coords[:, 1] % 2) * 0.5              # offset rows
    n_spot = len(coords)

    condition = np.array(["ctrl"] * (n_patients // 2)
                         + ["case"] * (n_patients - n_patients // 2))
    is_svg = np.zeros(G, bool); is_svg[:n_svg] = True
    base = np.exp(rng.normal(-0.7, 1.1, G))

    frames, meta_rows = [], []
    for pid in range(n_patients):
        patient_eff = rng.normal(0, 0.45, G)               # patient RANDOM effect
        # A spatial domain: a blob whose radius differs by condition.
        cx, cy = side / 2 + rng.normal(0, 1.5), side / 2 + rng.normal(0, 1.5)
        radius = (side * 0.22) * (1.45 if condition[pid] == "case" else 1.0)
        dist = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
        in_domain = (dist < radius).astype(float)
        # Smooth spatial field for the SVGs.
        field = np.exp(-(dist / (side * 0.25)) ** 2)

        depth = rng.lognormal(np.log(1.0), 0.32, n_spot)
        mu = np.outer(base * np.exp(patient_eff), depth)
        mu[is_svg] *= np.exp(1.4 * field)[None, :]
        Y = rng.poisson(mu)

        frames.append(pd.DataFrame(Y, index=[f"G{g:03d}" for g in range(G)],
                                   columns=[f"P{pid:02d}_S{k:04d}"
                                            for k in range(n_spot)]))
        meta_rows.append(pd.DataFrame({
            "patient": f"P{pid:02d}", "condition": condition[pid],
            "x": coords[:, 0], "y": coords[:, 1],
            "in_domain": in_domain, "total_umi": Y.sum(axis=0)},
            index=[f"P{pid:02d}_S{k:04d}" for k in range(n_spot)]))

    counts = pd.concat(frames, axis=1)
    spot_meta = pd.concat(meta_rows)
    truth = pd.DataFrame({"is_svg": is_svg},
                         index=[f"G{g:03d}" for g in range(G)])
    return counts, spot_meta, truth


counts, spot_meta, truth = simulate_spatial(seed=3701)
assert counts.columns.equals(spot_meta.index)
print(f"  {counts.shape[0]} genes x {counts.shape[1]:,} spots from "
      f"{spot_meta['patient'].nunique()} patients")
print(f"  spots per section: {spot_meta.groupby('patient').size().iloc[0]}")
print(f"  truly spatially variable genes: {int(truth['is_svg'].sum())}")
print(f"\n  n for CONDITION-level claims = {spot_meta['patient'].nunique()} "
      f"patients, NOT {counts.shape[1]:,} spots")

# %% [markdown]
# ## 2. Spatial autocorrelation: Moran's I, eq. (27.1)
#
# $$I=\frac{n}{W}\cdot\frac{\sum_i\sum_j w_{ij}(x_i-\bar x)(x_j-\bar x)}
#   {\sum_i(x_i-\bar x)^2},\qquad \mathbb E_0[I]=-\frac1{n-1}$$
#
# Note $\mathbb E_0[I]$ is **not zero**: a routinely misreported detail.

# %%
header("2. Moran's I and Geary's C (27.1)-(27.2)")

def knn_weights(coords, k=6):
    """Row-standardised k-nearest-neighbour spatial weights."""
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    idx = np.argsort(d, axis=1)[:, :k]
    for i in range(len(d)):
        W[i, idx[i]] = 1.0 / k
    return W


def morans_i(x, W):
    """stats.md eq. (27.1)."""
    x = np.asarray(x, float)
    n = len(x)
    z = x - x.mean()
    num = z @ W @ z
    den = (z ** 2).sum()
    return (n / W.sum()) * (num / den) if den > 0 else 0.0


def gearys_c(x, W):
    """stats.md eq. (27.2). E_0[C] = 1."""
    x = np.asarray(x, float)
    n = len(x)
    z = x - x.mean()
    num = np.sum(W * (x[:, None] - x[None, :]) ** 2)
    den = (z ** 2).sum()
    return ((n - 1) * num) / (2 * W.sum() * den) if den > 0 else 1.0


p0 = spot_meta["patient"] == "P00"
coords0 = spot_meta.loc[p0, ["x", "y"]].values
W = knn_weights(coords0, k=6)
n_spots0 = len(coords0)
print(f"  section P00: {n_spots0} spots, k=6 neighbour graph")
print(f"  E_0[Moran's I] = -1/(n-1) = {-1/(n_spots0-1):.5f}  (NOT zero)")
print(f"  E_0[Geary's C] = 1.0")

sub = counts.loc[:, p0.values]
logn = np.log1p(sub / sub.sum(axis=0) * 1e4)
print(f"\n  {'gene class':<22}{'mean Moran I':>15}{'mean Geary C':>15}")
for label, mask in [("truly spatial (SVG)", truth["is_svg"].values),
                    ("non-spatial", ~truth["is_svg"].values)]:
    mi = [morans_i(logn.iloc[g].values, W) for g in np.where(mask)[0][:40]]
    gc = [gearys_c(logn.iloc[g].values, W) for g in np.where(mask)[0][:40]]
    print(f"  {label:<22}{np.mean(mi):>15.4f}{np.mean(gc):>15.4f}")

# %% [markdown]
# ## 3. Testing spatial variability by permutation
#
# The normal approximation for Moran's I is poor on irregular lattices and
# skewed expression. Permute VALUES over LOCATIONS, which preserves both the
# expression distribution and the tissue geometry.

# %%
header("3. Permutation test for spatially variable genes")

def moran_permutation_test(x, W, n_perm=199, seed=0):
    rng = np.random.default_rng(seed)
    obs = morans_i(x, W)
    null = np.array([morans_i(rng.permutation(x), W) for _ in range(n_perm)])
    return obs, (1 + np.sum(null >= obs)) / (n_perm + 1)


rng = np.random.default_rng(3702)
genes_test = list(np.where(truth["is_svg"].values)[0][:25]) + \
             list(np.where(~truth["is_svg"].values)[0][:75])
rows = []
for g in genes_test:
    obs, p = moran_permutation_test(logn.iloc[g].values, W, n_perm=99,
                                    seed=int(rng.integers(1e6)))
    rows.append({"gene": logn.index[g], "I": obs, "p": p,
                 "is_svg": truth["is_svg"].iloc[g]})
svg_res = pd.DataFrame(rows)
svg_res["padj"] = st.false_discovery_control(svg_res["p"].values)
rej = svg_res["padj"] < 0.05
print(f"  tested {len(svg_res)} genes in ONE section")
print(f"  detected: {rej.sum()}  (TP {int((rej & svg_res['is_svg']).sum())}, "
      f"FP {int((rej & ~svg_res['is_svg']).sum())})")
print(f"  sensitivity {int((rej & svg_res['is_svg']).sum())/max(svg_res['is_svg'].sum(),1):.2f}")
print("\n  This is a valid WITHIN-SECTION claim: 'this gene is spatially")
print("  organised in this tissue'. It says nothing about the condition.")

# %% [markdown]
# ## 4. THE TRAP: spot-level condition testing

# %%
header("4. Spots are not patients (eq. 1.5 again)")

# Is the domain bigger in cases? Test it three ways.
dom_frac = spot_meta.groupby("patient").agg(
    domain_frac=("in_domain", "mean"),
    condition=("condition", "first"))

# (a) WRONG: every spot as an independent observation.
d_spot = spot_meta.copy()
d_spot["case"] = (d_spot["condition"] == "case").astype(float)
m_spot = smf.glm("in_domain ~ case", data=d_spot,
                 family=sm.families.Binomial()).fit()

# (b) CORRECT: one summary per patient, then a patient-level test.
a = dom_frac.loc[dom_frac["condition"] == "ctrl", "domain_frac"]
b = dom_frac.loc[dom_frac["condition"] == "case", "domain_frac"]
p_patient = st.ttest_ind(a, b, equal_var=False).pvalue

# (c) Also correct: a mixed model with a patient random intercept.
m_mixed = smf.mixedlm("in_domain ~ case", data=d_spot,
                      groups=d_spot["patient"]).fit()

print(f"  domain fraction: ctrl {a.mean():.3f}, case {b.mean():.3f}")
print(f"\n  {'analysis':<44}{'p-value':>14}{'effective n':>14}")
print(f"  {'spot-level logistic (WRONG)':<44}{m_spot.pvalues['case']:>14.2e}"
      f"{len(d_spot):>14,}")
print(f"  {'per-patient summary, t-test':<44}{p_patient:>14.2e}"
      f"{len(dom_frac):>14}")
print(f"  {'mixed model, patient random effect':<44}"
      f"{m_mixed.pvalues['case']:>14.4f}{len(dom_frac):>14}")
print("\n  The spot-level p-value is astronomically small because it counts")
print("  5,600 spots as independent. The design effect (eq. 1.8) applies")
print("  exactly as it did for cells in Module 21.")

# A negative control: a condition label that means NOTHING.
print("\n  NEGATIVE CONTROL - assign patients to arms at random:")
rng = np.random.default_rng(11)
fake_hits_spot, fake_hits_patient = 0, 0
for rep in range(40):
    pats = dom_frac.index.to_numpy()
    fake = dict(zip(pats, rng.permutation(["ctrl"] * 4 + ["case"] * 4)))
    d2 = spot_meta.copy()
    d2["fake"] = (d2["patient"].map(fake) == "case").astype(float)
    ms = smf.glm("in_domain ~ fake", data=d2,
                 family=sm.families.Binomial()).fit()
    fake_hits_spot += ms.pvalues["fake"] < 0.05
    dd = dom_frac.copy(); dd["fake"] = dd.index.map(fake)
    aa = dd.loc[dd["fake"] == "ctrl", "domain_frac"]
    bb = dd.loc[dd["fake"] == "case", "domain_frac"]
    fake_hits_patient += st.ttest_ind(aa, bb, equal_var=False).pvalue < 0.05
print(f"    spot-level false-positive rate    : {fake_hits_spot/40:.0%}"
      f"   <- should be 5%")
print(f"    patient-level false-positive rate : {fake_hits_patient/40:.0%}")

# %% [markdown]
# ## 5. Counts stay counts: the total-UMI offset, eq. (27.4)

# %%
header("5. Spot depth varies enormously - use an offset (27.4)")

print(f"  total UMI per spot: {spot_meta['total_umi'].min():,} - "
      f"{spot_meta['total_umi'].max():,} "
      f"({spot_meta['total_umi'].max()/max(spot_meta['total_umi'].min(),1):.1f}x)")
g_test = logn.index[0]
d = spot_meta.loc[p0.values].copy()
d["y"] = sub.loc[g_test].values.astype(float)
m_no_off = smf.glm("y ~ in_domain", data=d, family=sm.families.Poisson()).fit()
m_off = smf.glm("y ~ in_domain", data=d, family=sm.families.Poisson(),
                offset=np.log(d["total_umi"])).fit()
print(f"\n  gene {g_test}, domain effect:")
print(f"    without offset: {m_no_off.params['in_domain']:+.4f} "
      f"(p = {m_no_off.pvalues['in_domain']:.2e})")
print(f"    with offset    : {m_off.params['in_domain']:+.4f} "
      f"(p = {m_off.pvalues['in_domain']:.2e})")
print("  Without the offset the model partly measures how deeply each spot was")
print("  sequenced, not how much RNA the cells contained (eq. 13.8).")

# %% [markdown]
# ## 6. Edge effects

# %%
header("6. Edge spots have fewer neighbours")

d_edge = np.minimum.reduce([coords0[:, 0] - coords0[:, 0].min(),
                            coords0[:, 0].max() - coords0[:, 0],
                            coords0[:, 1] - coords0[:, 1].min(),
                            coords0[:, 1].max() - coords0[:, 1]])
is_edge = d_edge < 2
Wd = np.sqrt(((coords0[:, None, :] - coords0[None, :, :]) ** 2).sum(axis=2))
n_within_2 = ((Wd > 0) & (Wd <= 2.0)).sum(axis=1)
print(f"  spots within 2 units: interior mean {n_within_2[~is_edge].mean():.1f}, "
      f"edge mean {n_within_2[is_edge].mean():.1f}")

svg0 = np.where(truth["is_svg"].values)[0][0]
x_all = logn.iloc[svg0].values
W_all = knn_weights(coords0, k=6)
W_int = knn_weights(coords0[~is_edge], k=6)
print(f"\n  Moran's I for a spatial gene:")
print(f"    all spots            : {morans_i(x_all, W_all):.4f}")
print(f"    interior only (buffer): {morans_i(x_all[~is_edge], W_int):.4f}")
print("  Boundary spots have fewer true neighbours, which biases neighbourhood")
print("  statistics. Use a buffer zone, an edge-corrected estimator, or a")
print("  toroidal correction - and say which.")

# %% [markdown]
# ## 7. Neighbourhood-graph sensitivity

# %%
header("7. The neighbour graph is a modelling choice")

print(f"  {'k (neighbours)':>16}{'mean I, spatial genes':>24}"
      f"{'mean I, non-spatial':>22}")
for k in [4, 6, 12, 30]:
    Wk = knn_weights(coords0, k=k)
    mi_s = np.mean([morans_i(logn.iloc[g].values, Wk)
                    for g in np.where(truth["is_svg"].values)[0][:20]])
    mi_n = np.mean([morans_i(logn.iloc[g].values, Wk)
                    for g in np.where(~truth["is_svg"].values)[0][:20]])
    print(f"  {k:>16}{mi_s:>24.4f}{mi_n:>22.4f}")
print("\n  Larger neighbourhoods average over more tissue and detect broader")
print("  gradients; smaller ones detect fine structure. There is no single")
print("  correct k - report the choice and show sensitivity to it.")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
sc = ax.scatter(coords0[:, 0], coords0[:, 1],
                c=logn.iloc[svg0].values, s=8, cmap="viridis")
ax.set_title(f"Spatially variable gene (Moran I = "
             f"{morans_i(logn.iloc[svg0].values, W_all):.2f})", fontsize=9)
fig.colorbar(sc, ax=ax, shrink=0.8)

ax = axes[0, 1]
ns = np.where(~truth["is_svg"].values)[0][0]
sc = ax.scatter(coords0[:, 0], coords0[:, 1],
                c=logn.iloc[ns].values, s=8, cmap="viridis")
ax.set_title(f"Non-spatial gene (Moran I = "
             f"{morans_i(logn.iloc[ns].values, W_all):.2f})", fontsize=9)
fig.colorbar(sc, ax=ax, shrink=0.8)

ax = axes[1, 0]
ax.hist(svg_res.loc[~svg_res["is_svg"], "I"], bins=25, alpha=0.7,
        label="non-spatial")
ax.hist(svg_res.loc[svg_res["is_svg"], "I"], bins=25, alpha=0.7, label="SVG")
ax.axvline(-1 / (n_spots0 - 1), color="red", ls="--",
           label="$E_0[I] = -1/(n-1)$")
ax.set_xlabel("Moran's I")
ax.set_title("Eq. (27.1): the null is NOT zero", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
for cond, c in [("ctrl", "C0"), ("case", "C1")]:
    v = dom_frac.loc[dom_frac["condition"] == cond, "domain_frac"]
    ax.scatter(np.full(len(v), 0 if cond == "ctrl" else 1), v, s=60, color=c)
ax.set_xticks([0, 1]); ax.set_xticklabels(["ctrl", "case"])
ax.set_ylabel("domain fraction per SECTION")
ax.set_title(f"Patient-level unit: n = {len(dom_frac)}, p = {p_patient:.3f}",
             fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "spatial.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/spatial.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Colocalisation with a geometry-preserving null
#
# Label each spot as cell type A or B (use `in_domain` and a random assignment)
# and ask whether A and B are closer than chance. Build the null by permuting
# LABELS while keeping COORDINATES fixed. Why not permute coordinates?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng_c = np.random.default_rng(21)
# lab = np.where(spot_meta.loc[p0.values, "in_domain"].values > 0, "A", "B")
# # Sprinkle some A outside the domain so the two are not perfectly separated.
# flip = rng_c.random(len(lab)) < 0.12
# lab = np.where(flip, np.where(lab == "A", "B", "A"), lab)
#
# def cross_nn_distance(coords, lab):
#     """Mean distance from each A spot to its nearest B spot."""
#     A = coords[lab == "A"]; B = coords[lab == "B"]
#     if len(A) == 0 or len(B) == 0:
#         return np.nan
#     d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
#     return d.min(axis=1).mean()
#
# obs = cross_nn_distance(coords0, lab)
# null = [cross_nn_distance(coords0, rng_c.permutation(lab)) for _ in range(299)]
# p = (1 + np.sum(np.array(null) <= obs)) / 300
# print(f"  observed mean A->nearest-B distance: {obs:.3f}")
# print(f"  permutation null mean              : {np.mean(null):.3f}")
# print(f"  p (A closer to B than chance)      : {p:.4f}")
#
# # Permuting LABELS keeps the tissue's shape, density and any holes exactly as
# # observed, so the null asks precisely 'given this tissue geometry, is the
# # arrangement of labels special?'. Permuting COORDINATES would destroy the
# # geometry and test a different, uninteresting null (uniform tissue).
# # Colocalisation is also SCALE-DEPENDENT: repeat at several radii (a cross-K
# # function) rather than reporting a single number.

# %% [markdown]
# ### Problem 2: Deconvolution uncertainty
#
# Simulate spot compositions as mixtures of two cell types, estimate them by
# non-negative least squares, and propagate the uncertainty into a downstream
# comparison by bootstrapping. How much does ignoring it inflate significance?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# from scipy.optimize import nnls
# rng_d = np.random.default_rng(31)
# n_sp, G2 = 300, 60
# prof = rng_d.lognormal(0, 1, (G2, 2))                  # two cell-type profiles
# w_true = rng_d.beta(2, 2, n_sp)
# Wm = np.column_stack([w_true, 1 - w_true])
# Yd = (prof @ Wm.T) * rng_d.lognormal(0, 0.35, (G2, n_sp))
# w_hat = np.array([nnls(prof, Yd[:, i])[0] for i in range(n_sp)])
# w_hat = w_hat / w_hat.sum(axis=1, keepdims=True)
# print(f"  correlation(estimated, true) = "
#       f"{np.corrcoef(w_hat[:, 0], w_true)[0,1]:.3f}")
# print(f"  RMSE of the proportion estimate = "
#       f"{np.sqrt(np.mean((w_hat[:,0]-w_true)**2)):.3f}")
# grp = np.r_[np.zeros(n_sp//2), np.ones(n_sp-n_sp//2)]  # NO real difference
# p_naive = st.ttest_ind(w_hat[grp==0,0], w_hat[grp==1,0]).pvalue
# boot_p = []
# for b in range(200):
#     noise = rng_d.normal(0, np.sqrt(np.mean((w_hat[:,0]-w_true)**2)), n_sp)
#     wb = np.clip(w_hat[:,0] + noise, 0, 1)
#     boot_p.append(st.ttest_ind(wb[grp==0], wb[grp==1]).pvalue)
# print(f"  p treating w_hat as OBSERVED      : {p_naive:.4f}")
# print(f"  median p with uncertainty added   : {np.median(boot_p):.4f}")
#
# # Deconvolved proportions are ESTIMATES with real error, but almost every
# # downstream analysis treats them as measured quantities. Propagate the
# # uncertainty (bootstrap or posterior draws) or state plainly that you did
# # not - the same understatement-of-uncertainty issue as a batch-corrected
# # matrix in Module 19.

# %% [markdown]
# ## What to take away
#
# 1. **Never treat spots or cells as patient replicates.**
# 2. Build the neighbourhood graph deliberately and report sensitivity to it.
# 3. Use permutation nulls that preserve tissue geometry.
# 4. Keep counts as counts with a total-UMI offset (27.4).
# 5. $\mathbb E_0[\text{Moran's }I]=-1/(n-1)$, not 0.
# 6. Propagate deconvolution uncertainty, or say that you did not.
#
# **Next:** `28b_survival_biomarkers.py`
