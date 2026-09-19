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
# # Applied 36: Microbiome and compositional data
#
# **Curriculum link:** `stats.md` -> Topic 26, equations (26.1)-(26.6)
# **Core modules used:** 02 (closure), 10 (spurious correlation), 18 (distances)
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | A 16S case/control stool study (HMP/AGP-style), ~200 taxa |
# | **Assay** | Amplicon counts; only **relative** abundance is observed |
# | **Key threats** | Closure, structural zeros, varying depth, dispersion vs location |
# | **Here** | Simulated from **known absolute abundances**, then closed |
#
# ## Why this module is different
#
# Because we simulate absolute abundances and then close them, we can answer
# the question real data cannot: *which findings are biology and which are
# arithmetic?*

# %%
import os
import warnings
import itertools

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "36_microbiome_compositional"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate ABSOLUTE abundances, then close them, eq. (26.1)

# %%
header("1. Absolute truth -> relative observation (26.1)")

def simulate_microbiome(n=120, D=180, n_changed=45, log_effect=1.9,
                        depth_lo=5000, depth_hi=60000, seed=0):
    """Absolute abundances with a KNOWN truth, then multinomial sampling."""
    rng = np.random.default_rng(seed)
    group = np.array(["ctrl"] * (n // 2) + ["case"] * (n - n // 2))

    base = np.exp(rng.normal(5.5, 2.2, D))              # absolute abundances
    true_log_fc = np.zeros(D)
    changed = rng.choice(D, n_changed, replace=False)
    # Deliberately asymmetric: most changed taxa INCREASE, so the total shifts.
    true_log_fc[changed] = np.abs(rng.normal(0, log_effect, n_changed))

    A = np.empty((n, D))
    for i in range(n):
        eff = true_log_fc if group[i] == "case" else 0.0
        A[i] = base * np.exp(eff + rng.normal(0, 0.55, D))
    # Structural zeros: some taxa are genuinely absent from some subjects.
    A[rng.random((n, D)) < 0.25] = 0.0

    depth = rng.integers(depth_lo, depth_hi, n)
    Y = np.empty((n, D), dtype=int)
    for i in range(n):
        p = A[i] / A[i].sum()
        Y[i] = rng.multinomial(depth[i], p)

    idx = [f"S{i:03d}" for i in range(n)]
    cols = [f"Taxon{j:03d}" for j in range(D)]
    meta = pd.DataFrame({"group": group, "depth": depth}, index=idx)
    truth = pd.DataFrame({"log_fc": true_log_fc, "changed": true_log_fc != 0},
                         index=cols)
    return (pd.DataFrame(Y, index=idx, columns=cols),
            pd.DataFrame(A, index=idx, columns=cols), meta, truth)


counts, absolute, meta, truth = simulate_microbiome(seed=3601)
D = counts.shape[1]
print(f"  {counts.shape[0]} samples x {D} taxa")
print(f"  sequencing depth: {meta['depth'].min():,} - {meta['depth'].max():,} "
      f"({meta['depth'].max()/meta['depth'].min():.1f}x range)")
print(f"  zeros: {(counts == 0).mean().mean():.1%} of the table")
print(f"  taxa with a TRUE absolute increase: {int(truth['changed'].sum())}")

tot_ctrl = absolute.loc[(meta["group"] == "ctrl").values].sum(axis=1).mean()
tot_case = absolute.loc[(meta["group"] == "case").values].sum(axis=1).mean()
print(f"\n  TRUE total absolute load: ctrl {tot_ctrl:,.0f}, case {tot_case:,.0f}"
      f"  ({tot_case/tot_ctrl:.2f}x)")
print("  The total went UP. Sequencing cannot see that - it only ever returns")
print("  a fixed number of reads. Everything below follows from that fact.")

# %% [markdown]
# ## 2. What closure does to the truth, eq. (2.7)

# %%
header("2. Relative abundance of UNCHANGED taxa must fall (2.7)")

rel = counts.div(counts.sum(axis=1), axis=0)
unchanged = ~truth["changed"].values
print(f"  {'taxa':<26}{'mean rel. ctrl':>16}{'mean rel. case':>16}"
      f"{'log2 ratio':>12}")
for label, mask in [("TRULY increased", truth["changed"].values),
                    ("TRULY unchanged", unchanged)]:
    rc = rel.loc[(meta["group"] == "ctrl").values, mask].mean().mean()
    rs = rel.loc[(meta["group"] == "case").values, mask].mean().mean()
    print(f"  {label:<26}{rc:>16.5f}{rs:>16.5f}{np.log2(rs/rc):>12.3f}")
print("\n  The UNCHANGED taxa appear to DECREASE in relative terms, because a")
print("  fixed number of reads is being shared with taxa that genuinely grew.")
print("  This is arithmetic, not ecology (eq. 2.7).")

# %% [markdown]
# ## 3. Log-ratio transformations, eq. (26.2)-(26.3)
#
# $$\operatorname{clr}(\mathbf x)=\Big(\log\frac{x_1}{g(\mathbf x)},\dots\Big),
#   \qquad g(\mathbf x)=\Big(\prod_d x_d\Big)^{1/D}$$

# %%
header("3. clr / alr transforms and the zero problem (26.2)-(26.3)")

def clr(X, pseudocount=0.5):
    """stats.md eq. (26.2). Returns log(x / geometric mean of the sample)."""
    Xp = X + pseudocount
    Xp = Xp.div(Xp.sum(axis=1), axis=0)
    logX = np.log(Xp)
    return logX.sub(logX.mean(axis=1), axis=0)


def alr(X, reference, pseudocount=0.5):
    """stats.md eq. (26.3). Log-ratios against one chosen reference taxon."""
    Xp = X + pseudocount
    return np.log(Xp.drop(columns=reference).div(Xp[reference], axis=0))


Z = clr(counts)
print(f"  clr rows sum to zero (by construction): "
      f"max |row sum| = {np.abs(Z.sum(axis=1)).max():.2e}")
print(f"  -> the clr covariance matrix is SINGULAR (rank {D-1} of {D}), which")
print("     is why you cannot invert it directly for a graphical model (10.6).")

print(f"\n  Pseudocount sensitivity - the same taxon's clr effect size:")
tx = truth.index[truth["changed"]][0]
print(f"  {'pseudocount':>14}{'clr effect (case-ctrl)':>26}{'p-value':>12}")
for pc in [0.1, 0.5, 1.0, 5.0]:
    Zp = clr(counts, pseudocount=pc)
    a = Zp.loc[(meta["group"] == "ctrl").values, tx]
    b = Zp.loc[(meta["group"] == "case").values, tx]
    print(f"  {pc:>14.1f}{b.mean()-a.mean():>26.4f}"
          f"{st.ttest_ind(a, b, equal_var=False).pvalue:>12.2e}")
print("\n  Report the pseudocount and show that conclusions survive changing it.")
print("  Distinguish SAMPLING zeros (present but not seen at this depth -")
print("  a pseudocount is reasonable) from STRUCTURAL zeros (genuinely absent -")
print("  these should be modelled, not filled).")

# %% [markdown]
# ## 4. Differential abundance: four strategies vs the known truth

# %%
header("4. Differential abundance against a KNOWN absolute truth")

def da_relative_ttest(counts, meta):
    """Naive: t-test on relative abundances."""
    rel = counts.div(counts.sum(axis=1), axis=0)
    a = rel.loc[(meta["group"] == "ctrl").values]
    b = rel.loc[(meta["group"] == "case").values]
    p = st.ttest_ind(a, b, axis=0, equal_var=False).pvalue
    return pd.Series(np.nan_to_num(p, nan=1.0), index=counts.columns)


def da_clr_ttest(counts, meta, pseudocount=0.5):
    """clr then t-test (the ALDEx2-style approach)."""
    Z = clr(counts, pseudocount)
    a = Z.loc[(meta["group"] == "ctrl").values]
    b = Z.loc[(meta["group"] == "case").values]
    p = st.ttest_ind(a, b, axis=0, equal_var=False).pvalue
    return pd.Series(np.nan_to_num(p, nan=1.0), index=counts.columns)


def da_nb_offset(counts, meta):
    """NB GLM per taxon with a log(total reads) offset."""
    off = np.log(counts.sum(axis=1).values)
    g = (meta["group"] == "case").astype(float).values
    Xd = np.column_stack([np.ones(len(g)), g])
    out = {}
    for tx_ in counts.columns:
        y = counts[tx_].values.astype(float)
        if (y > 0).sum() < 5:
            out[tx_] = 1.0; continue
        try:
            f = sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=1.0),
                       offset=off).fit()
            out[tx_] = f.pvalues[1]
        except Exception:
            out[tx_] = 1.0
    return pd.Series(out)


def da_ancombc_like(counts, meta, pseudocount=0.5):
    """An ANCOM-BC-style bias correction, stats.md eq. (26.4).

        E[log Y_dj] = log A_dj + log d_j

    The sample-specific sampling fraction log d_j is a location shift COMMON to
    every taxon within a sample. Estimate it as the sample's mean log abundance
    across taxa (a trimmed/robust centre in the real method), subtract it, then
    test taxon-wise linear models on the bias-corrected values.
    """
    logY = np.log(counts + pseudocount)
    # Robust per-sample offset = the sampling fraction estimate.
    sampling_fraction = logY.median(axis=1)
    corrected = logY.sub(sampling_fraction, axis=0)
    g = (meta["group"] == "case").astype(float).values
    Xd = np.column_stack([np.ones(len(g)), g])
    beta, *_ = np.linalg.lstsq(Xd, corrected.values, rcond=None)
    resid = corrected.values - Xd @ beta
    dof = len(g) - 2
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
    t = beta[1] / np.maximum(se, 1e-12)
    return pd.Series(2 * st.t.sf(np.abs(t), dof), index=counts.columns)


def score(p, truth, label):
    q = st.false_discovery_control(np.nan_to_num(p.values, nan=1.0))
    rej = pd.Series(q < 0.05, index=p.index).reindex(truth.index).fillna(False)
    tp = int((rej & truth["changed"]).sum()); fp = int((rej & ~truth["changed"]).sum())
    print(f"  {label:<40}{rej.sum():>7}{tp:>6}{fp:>6}"
          f"{tp/truth['changed'].sum():>9.2f}{fp/max(rej.sum(),1):>8.3f}")


print(f"  Truth = an ABSOLUTE increase in {int(truth['changed'].sum())} taxa.")
print(f"  {'method':<40}{'rej':>7}{'TP':>6}{'FP':>6}{'sens':>9}{'FDP':>8}")
score(da_relative_ttest(counts, meta), truth, "t-test on relative abundance")
score(da_clr_ttest(counts, meta), truth, "clr + t-test (26.2)")
score(da_nb_offset(counts, meta), truth, "NB GLM with total-reads offset")
score(da_ancombc_like(counts, meta), truth, "ANCOM-BC-style bias correction (26.4)")

print("\n  The naive relative test produces a large number of false positives -")
print("  the unchanged taxa whose SHARE fell because the changed ones grew.")
print("  Estimating and removing the sample-specific sampling fraction (26.4)")
print("  is what ANCOM-BC2 does, and it is why it controls FDR where naive")
print("  relative-abundance tests do not.")

# %% [markdown]
# ## 5. Alpha diversity is depth-dependent, eq. (26.5)

# %%
header("5. Alpha diversity and rarefaction (26.5)")

def shannon(x):
    p = x[x > 0] / x.sum()
    return float(-np.sum(p * np.log(p)))


def simpson(x):
    p = x[x > 0] / x.sum()
    return float(1 - np.sum(p ** 2))


def chao1(x):
    s_obs = int((x > 0).sum())
    f1 = int((x == 1).sum()); f2 = int((x == 2).sum())
    return s_obs + (f1 ** 2) / (2 * f2) if f2 > 0 else s_obs + f1 * (f1 - 1) / 2


div = pd.DataFrame({
    "observed": (counts > 0).sum(axis=1),
    "shannon": counts.apply(shannon, axis=1),
    "simpson": counts.apply(simpson, axis=1),
    "chao1": counts.apply(chao1, axis=1),
    "depth": meta["depth"], "group": meta["group"],
})
print(f"  Correlation of each index with sequencing DEPTH (a pure artefact):")
for c in ["observed", "shannon", "simpson", "chao1"]:
    print(f"    {c:<10} Spearman = "
          f"{st.spearmanr(div['depth'], div[c]).statistic:+.3f}")

def rarefy(row, depth, rng):
    p = row.values / row.values.sum()
    return pd.Series(rng.multinomial(depth, p), index=row.index)


rng = np.random.default_rng(5)
min_depth = int(meta["depth"].min())
rare = counts.apply(lambda r: rarefy(r, min_depth, rng), axis=1)
div_r = pd.DataFrame({"observed": (rare > 0).sum(axis=1),
                      "shannon": rare.apply(shannon, axis=1),
                      "depth": meta["depth"]})
print(f"\n  After rarefying to {min_depth:,} reads:")
for c in ["observed", "shannon"]:
    print(f"    {c:<10} Spearman with depth = "
          f"{st.spearmanr(div_r['depth'], div_r[c]).statistic:+.3f}")
print("\n  Diversity estimation is one of the FEW places rarefying is")
print("  defensible - richness is intrinsically depth-dependent. For")
print("  DIFFERENTIAL ABUNDANCE, rarefying throws away data and power; use an")
print("  offset or a bias-corrected model instead.")

print(f"\n  Group comparison after rarefying:")
for c in ["observed", "shannon"]:
    a = div_r.loc[(meta['group'] == 'ctrl').values, c]
    b = div_r.loc[(meta['group'] == 'case').values, c]
    print(f"    {c:<10} ctrl {a.mean():8.3f}  case {b.mean():8.3f}  "
          f"p = {st.ttest_ind(a, b, equal_var=False).pvalue:.4f}")

# %% [markdown]
# ## 6. Beta diversity: PERMANOVA and the dispersion trap, eq. (26.6)

# %%
header("6. PERMANOVA must be paired with a DISPERSION test (26.6)")

def aitchison_distance(counts, pseudocount=0.5):
    """stats.md eq. (18.5): Euclidean distance between clr vectors."""
    Z = clr(counts, pseudocount).values
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    return d


def bray_curtis_matrix(counts):
    """stats.md eq. (18.4)."""
    R = counts.div(counts.sum(axis=1), axis=0).values
    n = R.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        num = np.abs(R[i] - R).sum(axis=1)
        den = (R[i] + R).sum(axis=1)
        d[i] = num / np.maximum(den, 1e-12)
    return d


def permanova(D, group, n_perm=999, seed=0):
    """stats.md eq. (26.6). Pseudo-F from the distance matrix, permuted labels."""
    rng = np.random.default_rng(seed)
    g = np.asarray(group)
    n = len(g)
    D2 = D ** 2

    def pseudo_f(lab):
        total = D2.sum() / (2 * n)
        within = 0.0
        for lv in np.unique(lab):
            m = lab == lv
            nk = m.sum()
            within += D2[np.ix_(m, m)].sum() / (2 * nk)
        a = len(np.unique(lab))
        return ((total - within) / (a - 1)) / (within / (n - a))

    f_obs = pseudo_f(g)
    null = np.array([pseudo_f(rng.permutation(g)) for _ in range(n_perm)])
    return f_obs, (1 + np.sum(null >= f_obs)) / (n_perm + 1)


def permdisp(D, group, n_perm=999, seed=0):
    """PERMDISP / betadisper: are the groups equally SPREAD about their centroid?"""
    rng = np.random.default_rng(seed)
    g = np.asarray(group)
    # Distance to the group medoid as a simple dispersion measure.
    disp = np.empty(len(g))
    for lv in np.unique(g):
        m = np.where(g == lv)[0]
        sub = D[np.ix_(m, m)]
        medoid = m[np.argmin(sub.sum(axis=1))]
        disp[m] = D[m, medoid]
    obs = abs(np.mean(disp[g == np.unique(g)[0]])
              - np.mean(disp[g == np.unique(g)[1]]))
    null = []
    for _ in range(n_perm):
        pg = rng.permutation(g)
        null.append(abs(np.mean(disp[pg == np.unique(g)[0]])
                        - np.mean(disp[pg == np.unique(g)[1]])))
    return obs, (1 + np.sum(np.array(null) >= obs)) / (n_perm + 1)


grp = meta["group"].values
for name, Dm in [("Aitchison (18.5)", aitchison_distance(counts)),
                 ("Bray-Curtis (18.4)", bray_curtis_matrix(counts))]:
    f, p = permanova(Dm, grp, n_perm=299, seed=1)
    do, dp = permdisp(Dm, grp, n_perm=299, seed=1)
    print(f"  {name:<22} PERMANOVA F = {f:6.3f}, p = {p:.4f}   |   "
          f"dispersion difference = {do:.4f}, p = {dp:.4f}")

print("\n  A significant PERMANOVA with a significant DISPERSION test is")
print("  ambiguous: the groups may differ in CENTROID (location), in SPREAD")
print("  (dispersion), or both. Report both tests, always.")

# Demonstrate the trap directly, and find the condition under which it bites.
print("\n  TRAP DEMO - groups with IDENTICAL centroids but different spread.")
print("  How often does PERMANOVA wrongly declare a location difference?\n")
print(f"  {'design':<34}{'sd ratio':>10}{'PERMANOVA rejection rate':>27}")
for (na, nb, lab) in [(60, 60, "balanced (60 vs 60)"),
                      (100, 20, "unbalanced, SMALL grp dispersed"),
                      (20, 100, "unbalanced, LARGE grp dispersed")]:
    for sd_ratio in [2.0, 3.0]:
        rate = 0
        n_rep = 25
        for rep in range(n_rep):
            r = np.random.default_rng(4000 + rep)
            Zt = np.vstack([r.normal(0, 1.0, (na, 20)),
                            r.normal(0, sd_ratio, (nb, 20))])
            Dt = np.sqrt(((Zt[:, None, :] - Zt[None, :, :]) ** 2).sum(axis=2))
            gt = np.array(["a"] * na + ["b"] * nb)
            rate += permanova(Dt, gt, n_perm=149, seed=rep)[1] < 0.05
        print(f"  {lab:<34}{sd_ratio:>10.1f}{rate/n_rep:>26.0%}")

print("\n  The result is sharper than the usual warning. PERMANOVA is fairly")
print("  ROBUST to dispersion differences when the design is BALANCED. It")
print("  breaks down badly when the groups are unbalanced AND the SMALLER")
print("  group is the more dispersed one - then it rejects almost always,")
print("  with identical centroids. (Anderson & Walsh 2013 document exactly")
print("  this asymmetry.)")
print("\n  Two practical consequences:")
print("   * Balance your group sizes - yet another payoff from design.")
print("   * Always report PERMDISP alongside PERMANOVA, and say which of")
print("     'different centroids' and 'different spread' your data support.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
mean_abs_ctrl = absolute.loc[(meta["group"] == "ctrl").values].mean()
mean_abs_case = absolute.loc[(meta["group"] == "case").values].mean()
mean_rel_ctrl = rel.loc[(meta["group"] == "ctrl").values].mean()
mean_rel_case = rel.loc[(meta["group"] == "case").values].mean()
ok = (mean_abs_ctrl > 0) & (mean_rel_ctrl > 0)
ax.scatter(np.log2(mean_abs_case[ok] / mean_abs_ctrl[ok]),
           np.log2(mean_rel_case[ok] / mean_rel_ctrl[ok]),
           s=10, alpha=0.5,
           c=["C3" if t else "grey" for t in truth["changed"][ok]])
ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
ax.plot([-1, 4], [-1, 4], "k--", lw=1)
ax.set_xlabel("TRUE log2 fold change (absolute)")
ax.set_ylabel("observed log2 FC (relative)")
ax.set_title("Eq. (2.7): closure shifts everything down", fontsize=9)

ax = axes[0, 1]
ax.scatter(div["depth"], div["observed"], s=14, alpha=0.6, label="raw")
ax.scatter(div["depth"], div_r["observed"], s=14, alpha=0.6, label="rarefied")
ax.set_xlabel("sequencing depth"); ax.set_ylabel("observed taxa")
ax.set_title("Eq. (26.5): richness tracks depth", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
Dm = aitchison_distance(counts)
Dc = Dm ** 2
J = np.eye(len(Dc)) - np.ones_like(Dc) / len(Dc)
B = -0.5 * J @ Dc @ J
w, V = np.linalg.eigh(B)
coords = V[:, ::-1][:, :2] * np.sqrt(np.maximum(w[::-1][:2], 0))
for lv, c in [("ctrl", "C0"), ("case", "C1")]:
    m = grp == lv
    ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.7, color=c, label=lv)
ax.set_xlabel("PCoA 1"); ax.set_ylabel("PCoA 2")
ax.set_title("Aitchison PCoA (eq. 18.5)", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
_r = np.random.default_rng(6)
Zc = np.vstack([_r.normal(0, 1.0, (100, 20)), _r.normal(0, 3.0, (20, 20))])
gtrap = np.array(["large n, tight"] * 100 + ["small n, dispersed"] * 20)
for lv, c in [("large n, tight", "C0"), ("small n, dispersed", "C1")]:
    m = gtrap == lv
    ax.scatter(Zc[m, 0], Zc[m, 1], s=14, alpha=0.6, color=c, label=lv)
ax.set_title("Trap: same centroid, unbalanced dispersion", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "microbiome.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/microbiome.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Spurious correlation from closure
#
# Compute the correlation matrix of the relative abundances and of the clr
# values, for taxa whose ABSOLUTE abundances are independent by construction.
# Which one is honest?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# sub = [c for c in counts.columns[:40]]
# Cabs = np.corrcoef(absolute[sub].values, rowvar=False)
# Crel = np.corrcoef(rel[sub].values, rowvar=False)
# Cclr = np.corrcoef(clr(counts)[sub].values, rowvar=False)
# off = ~np.eye(len(sub), dtype=bool)
# for lab, C in [("ABSOLUTE (the truth)", Cabs), ("relative (closed)", Crel),
#                ("clr", Cclr)]:
#     print(f"  {lab:<24} mean off-diag r = {C[off].mean():+.4f}, "
#           f"max |r| = {np.abs(C[off]).max():.3f}")
#
# # Closure forces the average correlation NEGATIVE (eq. 2.7). The clr partially
# # removes it but is not a complete fix: clr rows sum to zero, so clr values
# # also carry a residual negative constraint. For network inference use
# # proportionality (phi, rho_p) or SPIEC-EASI, which estimate a sparse
# # PRECISION matrix (eq. 10.6) on log-ratios rather than raw correlations.

# %% [markdown]
# ### Problem 2: Absolute quantification rescues the analysis
#
# Suppose you also measured total bacterial load by qPCR. Multiply the relative
# abundances by the (known) total and redo the differential-abundance test.
# How much does it improve?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# total_load = absolute.sum(axis=1)                    # "qPCR" measurement
# abs_est = rel.mul(total_load, axis=0)
# logA = np.log(abs_est + 1)
# g = (meta["group"] == "case").astype(float).values
# Xd = np.column_stack([np.ones(len(g)), g])
# beta, *_ = np.linalg.lstsq(Xd, logA.values, rcond=None)
# resid = logA.values - Xd @ beta
# s2 = (resid**2).sum(axis=0) / (len(g)-2)
# se = np.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
# p_abs = pd.Series(2*st.t.sf(np.abs(beta[1]/np.maximum(se,1e-12)), len(g)-2),
#                   index=counts.columns)
# print(f"  {'method':<40}{'rej':>7}{'TP':>6}{'FP':>6}{'sens':>9}{'FDP':>8}")
# score(da_ancombc_like(counts, meta), truth, "bias-corrected relative (26.4)")
# score(p_abs, truth, "ABSOLUTE abundance (rel x qPCR total)")
#
# # With an absolute anchor the question becomes answerable directly and both
# # sensitivity and FDP improve. This is why spike-ins and qPCR total-load
# # measurements are worth the effort: they convert an unidentifiable question
# # into an identifiable one. Without them, state that your claim is about
# # RELATIVE abundance.

# %% [markdown]
# ### Problem 3: Prevalence filtering
#
# Filter taxa present in at least 10%, 25% and 50% of samples, and record how
# sensitivity and FDP change. Why must the filter be outcome-independent?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'prevalence filter':>19}{'taxa kept':>12}{'rej':>7}{'TP':>6}"
#       f"{'FP':>6}{'sens':>9}{'FDP':>8}")
# for thr in [0.0, 0.10, 0.25, 0.50]:
#     keep = (counts > 0).mean(axis=0) >= thr
#     sub = counts.loc[:, keep]
#     p = da_ancombc_like(sub, meta)
#     q = st.false_discovery_control(p.values)
#     rej = pd.Series(q < 0.05, index=p.index)
#     t = truth.loc[p.index]
#     tp = int((rej & t["changed"]).sum()); fp = int((rej & ~t["changed"]).sum())
#     print(f"  {thr:>19.0%}{int(keep.sum()):>12}{rej.sum():>7}{tp:>6}{fp:>6}"
#           f"{tp/max(truth['changed'].sum(),1):>9.2f}"
#           f"{fp/max(rej.sum(),1):>8.3f}")
#
# # Prevalence is computed from the counts alone, with no reference to the group
# # labels, so it is OUTCOME-INDEPENDENT and cannot break FDR control (Module 08,
# # section 6). Filtering on "taxa that differ between groups" would. Note the
# # trade-off: aggressive filtering removes rare taxa that may be exactly the
# # biologically interesting ones. Pre-specify the rule.

# %% [markdown]
# ## What to take away
#
# 1. State whether your claim is about **relative** or **absolute** abundance.
#    Only spike-ins or qPCR support absolute claims.
# 2. Transform with log-ratios before distance, ordination, correlation or PCA.
# 3. Handle zeros explicitly; report the pseudocount and a sensitivity analysis.
# 4. Use bias-corrected differential abundance (26.4).
# 5. **Pair every PERMANOVA with a dispersion test.**
# 6. Rarefy only for diversity, never for differential abundance.
#
# **Next:** `37_spatial_omics.py`
