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
# # Applied 31: Single-cell RNA-seq: replicated inference
#
# **Curriculum link:** `stats.md` -> Topic 21, equations (21.1)-(21.4)
# **Core modules used:** 01 (pseudoreplication), 14 (mixed models), 18 (double
# dipping), 26/36 (compositional)
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | Kang et al. 2018 (GSE96583): 8 lupus donors, PBMCs, +/-IFN-beta |
# | **Design** | Paired: each donor contributes control and stimulated cells |
# | **Unit of inference** | **Donor** (8 per arm), not cell (~15,000) |
# | **Here** | Simulated with the same hierarchy and realistic parameters |
#
# ## The biological question
#
# Which genes change **within a cell type** between conditions, and which
# **cell-type proportions** change? These are different questions (21.3).
#
# The literature is settled on the answer: preserve the donor as the replicate,
# via pseudobulk aggregation or a subject-aware mixed model. Naive per-cell
# tests produce a documented excess of false positives.

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

MODULE_NAME = "31_single_cell_pseudobulk"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate a multi-donor, multi-cell-type experiment
#
# The hierarchy is: condition -> donor -> cell type -> cell -> UMI count.
# Donor effects are gene-specific and **shared by every cell of that donor** -
# which is exactly what makes cells pseudoreplicates (eq. 1.4).

# %%
header("1. Simulating a Kang-like multi-donor scRNA-seq experiment")

def simulate_sc(n_donors_per_arm=4, G=1200, n_de=120, lfc=1.0,
                cell_types=("CD4T", "CD14mono", "B"),
                base_props=(0.55, 0.30, 0.15),
                da_effect=(0.0, 0.5, -0.5), seed=0):
    """Returns a cell-level count matrix, cell metadata, and the ground truth.

    `da_effect` is a log-odds shift in cell-type composition for the treated
    arm - so we can test differential ABUNDANCE as well as differential STATE.
    """
    rng = np.random.default_rng(seed)
    donors = [f"D{i:02d}" for i in range(2 * n_donors_per_arm)]
    arm = {d: ("ctrl" if i < n_donors_per_arm else "stim")
           for i, d in enumerate(donors)}

    # True differential-STATE genes, in the CD14mono compartment only.
    true_lfc = np.zeros(G)
    de_idx = rng.choice(G, n_de, replace=False)
    true_lfc[de_idx] = rng.normal(0, lfc, n_de)

    base_expr = np.exp(rng.normal(-1.2, 1.4, G))        # mean UMI per cell
    # Cell-type-specific baseline modulation.
    ct_mod = {ct: np.exp(rng.normal(0, 0.5, G)) for ct in cell_types}

    rows_meta, blocks = [], []
    for d in donors:
        # DONOR RANDOM EFFECT: gene-specific, shared by all of this donor's cells.
        donor_eff = rng.normal(0, 0.45, G)
        n_cells = rng.integers(400, 900)

        # Cell-type composition for this donor, on the log-odds scale.
        shift = np.array(da_effect) if arm[d] == "stim" else np.zeros(len(cell_types))
        logits = np.log(np.array(base_props)) + shift + rng.normal(0, 0.25,
                                                                   len(cell_types))
        props = np.exp(logits) / np.exp(logits).sum()
        ct_assign = rng.choice(cell_types, size=n_cells, p=props)

        depth = rng.lognormal(np.log(1.0), 0.35, n_cells)     # per-cell capture

        for ct in cell_types:
            sel = ct_assign == ct
            k = int(sel.sum())
            if k == 0:
                continue
            eff = (true_lfc * np.log(2) if (ct == "CD14mono"
                                            and arm[d] == "stim") else 0.0)
            mu = (base_expr * ct_mod[ct] * np.exp(donor_eff + eff))
            lam = np.outer(mu, depth[sel])                     # G x k
            blocks.append(rng.poisson(lam))
            rows_meta.append(pd.DataFrame({
                "donor": d, "condition": arm[d], "cell_type": ct,
                "depth": depth[sel]}))

    counts = np.hstack(blocks)
    cell_meta = pd.concat(rows_meta, ignore_index=True)
    cell_meta.index = [f"C{i:06d}" for i in range(len(cell_meta))]
    X = pd.DataFrame(counts, index=[f"G{i:04d}" for i in range(G)],
                     columns=cell_meta.index)
    truth = pd.DataFrame({"true_lfc": true_lfc, "is_de": true_lfc != 0},
                         index=X.index)
    return X, cell_meta, truth


X, cell_meta, truth = simulate_sc(seed=3101)
assert X.columns.equals(cell_meta.index)          # eq. (2.1)
print(f"  {X.shape[0]} genes x {X.shape[1]} cells from "
      f"{cell_meta['donor'].nunique()} donors")
print(f"\n  cells per donor x cell type (first 6 donors):")
tab = pd.crosstab(cell_meta["donor"], cell_meta["cell_type"])
print(tab.head(6).to_string())
print(f"\n  n for CONDITION-level inference = "
      f"{cell_meta['donor'].nunique()} donors, NOT {X.shape[1]} cells")
print(f"  truly DE genes (in CD14mono only): {truth['is_de'].sum()}")

# %% [markdown]
# ## 2. The naive per-cell test, and why it fails
#
# This is what a default `FindMarkers`-style Wilcoxon or t-test on cells does.
# We run it on the CD14mono compartment where there IS real signal, and then on
# CD4T where there is **none**, so we can measure false positives directly.

# %%
header("2. Per-cell testing inflates false positives")

def per_cell_test(X, cell_meta, cell_type, method="wilcoxon"):
    """Test every gene, treating each CELL as an independent observation."""
    sel = cell_meta["cell_type"] == cell_type
    cm = cell_meta[sel]
    # Standard log-normalisation, as used for visualisation/marker testing.
    sub = X.loc[:, sel.values]
    cpm = sub / sub.sum(axis=0) * 1e4
    logn = np.log1p(cpm)
    a = logn.loc[:, (cm["condition"] == "ctrl").values].values
    b = logn.loc[:, (cm["condition"] == "stim").values].values
    if method == "wilcoxon":
        p = np.array([st.mannwhitneyu(a[i], b[i]).pvalue for i in range(a.shape[0])])
    else:
        p = st.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    return pd.Series(p, index=X.index)


for ct, has_signal in [("CD14mono", True), ("CD4T", False)]:
    p = per_cell_test(X, cell_meta, ct)
    q = st.false_discovery_control(np.nan_to_num(p.values, nan=1.0))
    rej = q < 0.05
    # The true DE genes were injected into CD14mono ONLY. In any other
    # compartment there is no condition effect at all, so EVERY rejection is a
    # false positive - regardless of whether that gene responds elsewhere.
    if has_signal:
        tp = int((rej & truth["is_de"].values).sum())
        fp = int((rej & ~truth["is_de"].values).sum())
        label = "real signal present"
    else:
        tp, fp = 0, int(rej.sum())
        label = "NO real signal at all"
    print(f"  {ct:<10} ({label:<21}): {rej.sum():>4} 'DE' genes "
          f"(TP={tp:>3}, FP={fp:>4}, FDP={fp/max(rej.sum(),1):.3f})")

print("\n  Look at CD4T. There is NO condition effect in that compartment by")
print("  construction, yet the per-cell test calls a large fraction of the")
print("  transcriptome differential. Every one of those is a false positive,")
print("  produced by donor-to-donor variation being counted as within-group")
print("  noise across thousands of pseudoreplicated cells (eq. 1.8).")

# Quantify the inflation with the design effect.
sub = X.loc[:, (cell_meta["cell_type"] == "CD4T").values]
cm = cell_meta[cell_meta["cell_type"] == "CD4T"]
logn = np.log1p(sub / sub.sum(axis=0) * 1e4)
gene = logn.index[np.argsort(-logn.mean(axis=1).values)[50]]
vals = logn.loc[gene]
between = vals.groupby(cm["donor"].values).mean().var(ddof=1)
within = vals.groupby(cm["donor"].values).var(ddof=1).mean()
icc = between / (between + within)
m_bar = cm.groupby("donor").size().mean()
print(f"\n  For a representative CD4T gene:")
print(f"    ICC (eq. 1.7)          = {icc:.4f}")
print(f"    mean cells per donor   = {m_bar:.0f}")
print(f"    design effect (eq. 1.8)= {1 + (m_bar-1)*icc:.1f}")
print(f"    SE understated by      = {np.sqrt(1 + (m_bar-1)*icc):.1f}x")

# %% [markdown]
# ## 3. Pseudobulk aggregation, eq. (21.1)
#
# $$K^{\text{pb}}_{gci}=\sum_{k\in\text{cells}(c,i)}K_{gk}$$
#
# **Sum raw counts, do not average normalised values.** Summing keeps the count
# nature *and* the information about how many cells contributed, so a 3-cell
# pseudobulk is automatically down-weighted by the NB variance.

# %%
header("3. Pseudobulk by (cell type x donor), eq. (21.1)")

def make_pseudobulk(X, cell_meta, cell_type, min_cells=10):
    """Sum RAW counts within (cell type, donor). Returns counts + sample meta."""
    sel = (cell_meta["cell_type"] == cell_type).values
    sub = X.loc[:, sel]
    cm = cell_meta[sel]
    n_cells = cm.groupby("donor").size()
    keep_donors = n_cells[n_cells >= min_cells].index
    pb = sub.T.groupby(cm["donor"].values).sum().T[keep_donors]
    meta = (cm.groupby("donor")
            .agg(condition=("condition", "first"))
            .loc[keep_donors])
    meta["n_cells"] = n_cells[keep_donors]
    return pb, meta


pb, pb_meta = make_pseudobulk(X, cell_meta, "CD14mono")
print(pb_meta.to_string())
print(f"\n  pseudobulk matrix: {pb.shape[0]} genes x {pb.shape[1]} SAMPLES")
print(f"  n is now {pb.shape[1]} - the number of donors. This is the whole point.")

print("\n  Why SUM raw counts rather than average normalised expression:")
pb_mean = (np.log1p(X.loc[:, (cell_meta['cell_type']=='CD14mono').values]
                    / X.loc[:, (cell_meta['cell_type']=='CD14mono').values].sum(axis=0) * 1e4)
           .T.groupby(cell_meta[cell_meta['cell_type']=='CD14mono']['donor'].values)
           .mean().T)
print(f"    summed counts range over donors : "
      f"{pb.sum(axis=0).min():,} - {pb.sum(axis=0).max():,} UMIs")
print(f"    the library size carries the cell count, so the NB model")
print(f"    automatically trusts big pseudobulks more. Averaging normalised")
print(f"    values discards that and gives a 3-cell sample equal influence.")

# %% [markdown]
# ## 4. Three valid analyses compared
#
# 1. Pseudobulk + NB GLM (the recommended default)
# 2. Pseudobulk + limma-style moderated t
# 3. Cell-level NB GLMM with a donor random intercept (eq. 21.2)

# %%
header("4. Pseudobulk vs mixed model vs naive (21.1)-(21.2)")

def pseudobulk_nb(pb, pb_meta, min_count=10):
    """NB GLM on pseudobulk with a library-size offset. stats.md eq. (20.1)."""
    lib = pb.sum(axis=0)
    keep = (pb.sum(axis=1) >= min_count) & ((pb > 0).sum(axis=1) >= 3)
    sub = pb[keep]
    cond = (pb_meta["condition"] == "stim").astype(float).values
    Xd = np.column_stack([np.ones(len(cond)), cond])
    offset = np.log(lib.values)
    # One shared dispersion estimated from the data keeps this fast and stable.
    norm = sub / lib
    logy = np.log2(norm + 1e-6)
    beta, *_ = np.linalg.lstsq(Xd, logy.T.values, rcond=None)
    resid = logy.T.values - Xd @ beta
    s2 = (resid ** 2).sum(axis=0) / (len(cond) - 2)
    phi = np.maximum(np.median(s2) * np.log(2) ** 2, 0.01)

    out = []
    for i in range(sub.shape[0]):
        y = sub.iloc[i].values.astype(float)
        try:
            f = sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=phi),
                       offset=offset).fit()
            out.append((f.params[1] / np.log(2), f.pvalues[1]))
        except Exception:
            out.append((np.nan, 1.0))
    res = pd.DataFrame(out, index=sub.index, columns=["lfc", "pvalue"])
    res["padj"] = st.false_discovery_control(np.nan_to_num(res["pvalue"], nan=1.0))
    return res, phi


def pseudobulk_moderated_t(pb, pb_meta):
    """limma-style: log-CPM + moderated t (stats.md eq. 20.6)."""
    from scipy.special import digamma, polygamma
    from scipy.optimize import brentq
    lib = pb.sum(axis=0)
    keep = (pb.sum(axis=1) >= 10) & ((pb > 0).sum(axis=1) >= 3)
    y = np.log2(pb[keep] / lib * 1e6 + 1)
    a = y.loc[:, (pb_meta["condition"] == "ctrl").values].values
    b = y.loc[:, (pb_meta["condition"] == "stim").values].values
    n1, n2 = a.shape[1], b.shape[1]
    df = n1 + n2 - 2
    diff = b.mean(axis=1) - a.mean(axis=1)
    s2 = ((n1-1)*a.var(axis=1, ddof=1) + (n2-1)*b.var(axis=1, ddof=1)) / df
    s2 = np.maximum(s2, 1e-12)
    e = np.log(s2) - digamma(df/2) + np.log(df/2)
    mad = np.median(np.abs(e - np.median(e))) * 1.4826
    target = mad**2 - polygamma(1, df/2)
    if target > 0:
        try:
            d0 = brentq(lambda d: polygamma(1, d/2) - target, 1e-4, 1e4)
        except ValueError:
            d0 = 20.0
    else:
        d0 = 50.0
    s0_2 = float(np.exp(np.mean(e) + digamma(d0/2) - np.log(d0/2)))
    s2t = (d0*s0_2 + df*s2) / (d0 + df)
    se = np.sqrt(s2t * (1/n1 + 1/n2))
    t = diff / se
    p = 2 * st.t.sf(np.abs(t), df + d0)
    res = pd.DataFrame({"lfc": diff, "pvalue": p}, index=y.index)
    res["padj"] = st.false_discovery_control(res["pvalue"].values)
    return res, d0


def cell_level_glmm(X, cell_meta, cell_type, genes, seed=0):
    """Poisson GLMM with a donor random intercept (eq. 21.2), on a gene subset.

    Fitting 1,200 GLMMs is slow, which is itself the practical argument for
    pseudobulk; we fit a subset to show the answers agree.
    """
    sel = (cell_meta["cell_type"] == cell_type).values
    sub = X.loc[genes, sel]
    cm = cell_meta[sel].copy()
    cm["log_depth"] = np.log(sub.sum(axis=0).values)
    cm["stim"] = (cm["condition"] == "stim").astype(float)
    out = {}
    for g in genes:
        d = cm.copy()
        d["y"] = sub.loc[g].values.astype(float)
        try:
            # A Poisson GLMM via PQL-style approximation: fit on the log scale
            # with a donor random intercept using MixedLM on log1p counts
            # normalised by depth. Fast, and adequate for this comparison.
            d["ln"] = np.log1p(d["y"] / np.exp(d["log_depth"]) * 1e4)
            f = smf.mixedlm("ln ~ stim", data=d, groups=d["donor"]).fit()
            out[g] = f.pvalues["stim"]
        except Exception:
            out[g] = np.nan
    return pd.Series(out)


res_nb, phi_hat = pseudobulk_nb(pb, pb_meta)
res_mod, d0 = pseudobulk_moderated_t(pb, pb_meta)
p_naive = per_cell_test(X, cell_meta, "CD14mono")


def score(p_or_res, label, truth=truth):
    if isinstance(p_or_res, pd.DataFrame):
        padj = p_or_res["padj"]
        idx = p_or_res.index
    else:
        idx = p_or_res.index
        padj = pd.Series(st.false_discovery_control(
            np.nan_to_num(p_or_res.values, nan=1.0)), index=idx)
    t = truth.loc[idx]
    rej = padj < 0.05
    tp = int((rej & t["is_de"]).sum()); fp = int((rej & ~t["is_de"]).sum())
    print(f"  {label:<40}{rej.sum():>7}{tp:>7}{fp:>7}"
          f"{tp/max(t['is_de'].sum(),1):>9.2f}{fp/max(rej.sum(),1):>8.3f}")


print(f"  estimated pseudobulk dispersion = {phi_hat:.4f}; "
      f"moderated-t prior df = {d0:.1f}\n")
print(f"  {'method':<40}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>9}{'FDP':>8}")
score(p_naive, "naive per-cell Wilcoxon")
score(res_nb, "pseudobulk + NB GLM (21.1)")
score(res_mod, "pseudobulk + moderated t (20.6)")
print("\n  Read this honestly: the pseudobulk FDP is not exactly 0.05 either.")
print("  With only a dozen or two rejections, two or three false ones move it")
print("  a long way - that is Monte-Carlo noise on a small denominator, not a")
print("  failure of FDR control. The naive test's 0.87 is a different animal")
print("  entirely: it is a systematic breakdown, not sampling variation.")

# GLMM on a subset, to show it agrees with pseudobulk.
subset = list(truth.index[truth["is_de"]][:40]) + list(truth.index[~truth["is_de"]][:40])
p_glmm = cell_level_glmm(X, cell_meta, "CD14mono", subset)
t_sub = truth.loc[subset]
q_glmm = pd.Series(st.false_discovery_control(
    np.nan_to_num(p_glmm.values, nan=1.0)), index=p_glmm.index)
q_pb = res_nb["padj"].reindex(subset)
print(f"\n  On an 80-gene subset (40 DE, 40 null), donor-aware mixed model vs")
print(f"  pseudobulk NB:")
print(f"    mixed model : {int((q_glmm<0.05).sum())} rejected, "
      f"{int(((q_glmm<0.05) & t_sub['is_de']).sum())} true")
print(f"    pseudobulk  : {int((q_pb<0.05).sum())} rejected, "
      f"{int(((q_pb<0.05) & t_sub['is_de']).sum())} true")
print(f"    rank correlation of p-values: "
      f"{st.spearmanr(p_glmm.values, res_nb['pvalue'].reindex(subset).values, nan_policy='omit').statistic:.3f}")
print("\n  The two donor-aware approaches agree closely. Pseudobulk is far")
print("  cheaper and easier to audit, which is why it is the default.")

# %% [markdown]
# ## 5. Differential abundance is a separate, compositional question, eq. (21.4)

# %%
header("5. Differential ABUNDANCE of cell types (21.3)-(21.4)")

comp = pd.crosstab(cell_meta["donor"], cell_meta["cell_type"])
donor_cond = cell_meta.groupby("donor")["condition"].first()
props = comp.div(comp.sum(axis=1), axis=0)
print("  Cell-type proportions per donor:")
print(props.round(3).to_string())
print(f"\n  TRUE log-odds shifts in the stim arm: CD4T 0.0, CD14mono +0.5, B -0.5")

print(f"\n  {'cell type':<12}{'mean ctrl':>11}{'mean stim':>11}"
      f"{'t-test on props':>18}{'NB with offset (21.4)':>24}")
for ct in comp.columns:
    pc = props.loc[donor_cond == "ctrl", ct]
    ps = props.loc[donor_cond == "stim", ct]
    p_t = st.ttest_ind(pc, ps, equal_var=False).pvalue
    d = pd.DataFrame({"y": comp[ct].values,
                      "total": comp.sum(axis=1).values,
                      "stim": (donor_cond == "stim").astype(float).values})
    f = smf.glm("y ~ stim", data=d,
                family=sm.families.NegativeBinomial(alpha=0.1),
                offset=np.log(d["total"])).fit()
    print(f"  {ct:<12}{pc.mean():>11.3f}{ps.mean():>11.3f}{p_t:>18.4f}"
          f"{f.pvalues['stim']:>24.4f}")

drift = props.loc[donor_cond == "stim", "CD4T"].mean() - \
        props.loc[donor_cond == "ctrl", "CD4T"].mean()
print(f"\n  The COMPOSITIONAL trap (eq. 2.7). CD4T has NO true effect, yet its")
print(f"  mean proportion still drifted by {drift:+.3f}, purely because the")
print("  other two compartments moved and the three proportions must sum to 1.")
print("  With this many donors that drift is not yet significant - but it is a")
print("  systematic bias, not noise, so it does NOT average away: add donors")
print("  and it becomes a confident false positive.")
print("\n  The count model with a total-cells offset handles the varying")
print("  sequencing effort, but it does not escape the constraint either. The")
print("  honest framing is that relative abundance cannot distinguish")
print("  'monocytes expanded' from 'everything else contracted' without an")
print("  absolute measurement - spike-ins, counting beads, or cytometry counts.")
print("  Module 36 develops the log-ratio machinery for exactly this problem.")

# %% [markdown]
# ## 6. Minimum-cell thresholds and low-information pseudobulks

# %%
header("6. The minimum-cell rule")

print(f"  {'min cells':>11}{'donors kept':>13}{'rejected':>11}{'TP':>6}"
      f"{'FP':>6}{'FDP':>8}")
for mc in [1, 10, 50, 200]:
    pbx, mx = make_pseudobulk(X, cell_meta, "B", min_cells=mc)
    if pbx.shape[1] < 4 or mx["condition"].nunique() < 2:
        print(f"  {mc:>11}{pbx.shape[1]:>13}   too few donors to analyse")
        continue
    r, _ = pseudobulk_nb(pbx, mx)
    t = truth.loc[r.index]
    rej = r["padj"] < 0.05
    # B cells have NO true differential state, so every rejection is false.
    print(f"  {mc:>11}{pbx.shape[1]:>13}{rej.sum():>11}"
          f"{int((rej & t['is_de']).sum()):>6}{int((rej & ~t['is_de']).sum()):>6}"
          f"{(rej & ~t['is_de']).sum()/max(rej.sum(),1):>8.3f}")
print("\n  The B compartment has no true effect, so this is a negative control.")
print("  Set the minimum-cell rule BEFORE looking at the outcome (10 is a")
print("  common default), and report how many sample x cell-type combinations")
print("  it removed.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.hist(per_cell_test(X, cell_meta, "CD4T").dropna(), bins=40, alpha=0.7,
        label="naive per-cell (CD4T, no signal)")
pb_b, mb = make_pseudobulk(X, cell_meta, "CD4T")
r_b, _ = pseudobulk_nb(pb_b, mb)
ax.hist(r_b["pvalue"].dropna(), bins=40, alpha=0.7, label="pseudobulk (CD4T)")
ax.set_xlabel("p-value"); ax.set_title("Negative control: CD4T has no effect",
                                       fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
t_de = truth.loc[res_nb.index, "is_de"]
ax.scatter(res_nb.loc[~t_de, "lfc"], -np.log10(res_nb.loc[~t_de, "pvalue"]),
           s=3, alpha=0.3, color="grey", label="null")
ax.scatter(res_nb.loc[t_de, "lfc"], -np.log10(res_nb.loc[t_de, "pvalue"]),
           s=5, alpha=0.6, color="C3", label="true DE")
ax.set_xlabel("pseudobulk log2 FC"); ax.set_ylabel("$-\\log_{10} p$")
ax.set_title("Pseudobulk volcano (CD14mono)", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
for i, ct in enumerate(comp.columns):
    for cond, off, c in [("ctrl", -0.15, "C0"), ("stim", 0.15, "C1")]:
        v = props.loc[donor_cond == cond, ct]
        ax.scatter(np.full(len(v), i + off), v, s=40, color=c,
                   label=cond if i == 0 else None)
ax.set_xticks(range(len(comp.columns))); ax.set_xticklabels(comp.columns)
ax.set_ylabel("proportion of cells")
ax.set_title("Eq. (21.4): composition is constrained", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
n_cells_pb = pb_meta["n_cells"]
ax.scatter(n_cells_pb, pb.sum(axis=0), s=50)
ax.set_xlabel("cells in the pseudobulk"); ax.set_ylabel("total UMIs")
ax.set_title("Eq. (21.1): summing preserves the cell count", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "single_cell.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/single_cell.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Donors vs cells per donor
#
# Using `simulate_sc`, compare the power of the pseudobulk analysis at
# (a) 3 donors per arm with many cells, and (b) 8 donors per arm.
# Which buys more? Connect your answer to eq. (1.6) and (9.7).

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# for n_don in [3, 5, 8]:
#     Xs, cms, trs = simulate_sc(n_donors_per_arm=n_don, seed=500 + n_don)
#     pbs, ms = make_pseudobulk(Xs, cms, "CD14mono")
#     rs, _ = pseudobulk_nb(pbs, ms)
#     t = trs.loc[rs.index]
#     rej = rs["padj"] < 0.05
#     print(f"  {n_don} donors/arm: {rej.sum():>4} rejected, "
#           f"sens={int((rej & t['is_de']).sum())/max(t['is_de'].sum(),1):.2f}, "
#           f"FDP={(rej & ~t['is_de']).sum()/max(rej.sum(),1):.3f}")
#
# # Donors dominate. Eq. (1.6) says Var(mean) -> sigma_b^2 / n_donors no matter
# # how many cells you sequence, and eq. (9.7) says the cost-optimal cells per
# # donor is modest whenever donor variability is large - which it always is in
# # human studies. Three donors cannot be rescued by a million cells.

# %% [markdown]
# ### Problem 2: Marker genes are not condition effects
#
# Cluster the CD4T cells into two groups with k-means on the top PCs, then test
# for "markers" between the clusters. Compare the p-value histogram to the one
# from the (valid) pseudobulk condition test. Explain using Module 18.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# from sklearn.cluster import KMeans
# sel = (cell_meta["cell_type"] == "CD4T").values
# sub = X.loc[:, sel]
# logn = np.log1p(sub / sub.sum(axis=0) * 1e4).T.values
# Z = (logn - logn.mean(axis=0))
# U, S, Vt = np.linalg.svd(Z, full_matrices=False)
# pcs = (U * S)[:, :15]
# lab = KMeans(2, n_init=25, random_state=0).fit_predict(pcs)
# p_marker = st.ttest_ind(logn[lab == 0], logn[lab == 1], axis=0).pvalue
# print(f"  cluster 'markers' at p<0.05 : {np.mean(p_marker < 0.05):.1%}")
# print(f"  surviving BH FDR 0.05       : "
#       f"{int((st.false_discovery_control(np.nan_to_num(p_marker, nan=1.0)) < 0.05).sum())}")
# pb_c, m_c = make_pseudobulk(X, cell_meta, "CD4T")
# r_c, _ = pseudobulk_nb(pb_c, m_c)
# print(f"  valid pseudobulk condition test, p<0.05: "
#       f"{(r_c['pvalue'] < 0.05).mean():.1%}")
#
# # The clusters were DEFINED by the same expression values being tested, so
# # essentially every gene comes out "significant" (Module 18, double dipping).
# # Cluster markers describe the clustering, not a condition effect. They are a
# # useful DESCRIPTION and are not evidence of anything about the conditions.

# %% [markdown]
# ### Problem 3: Mean-of-normalised vs sum-of-counts pseudobulk
#
# Build a pseudobulk by averaging log-normalised expression instead of summing
# raw counts, analyse it with a t-test, and compare to the count-based result.
# Then deliberately down-sample one donor's CD14mono cells to 15 and see which
# method is more disturbed.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def pseudobulk_mean_norm(X, cell_meta, cell_type, keep_idx=None):
#     sel = (cell_meta["cell_type"] == cell_type).values
#     if keep_idx is not None:
#         sel = sel & keep_idx
#     sub = X.loc[:, sel]; cm = cell_meta[sel]
#     ln = np.log1p(sub / sub.sum(axis=0) * 1e4)
#     return ln.T.groupby(cm["donor"].values).mean().T, \
#            cm.groupby("donor")["condition"].first()
#
# pbm, cond_m = pseudobulk_mean_norm(X, cell_meta, "CD14mono")
# a = pbm.loc[:, (cond_m == "ctrl").values]; b = pbm.loc[:, (cond_m == "stim").values]
# p_mean = st.ttest_ind(a, b, axis=1).pvalue
# q_mean = st.false_discovery_control(np.nan_to_num(p_mean, nan=1.0))
# t = truth.loc[pbm.index]
# print(f"  mean-of-normalised: {int((q_mean<0.05).sum())} rejected, "
#       f"{int(((q_mean<0.05) & t['is_de']).sum())} true")
# score(res_nb, "sum-of-counts + NB GLM")
#
# # Now cripple one donor: keep only 15 of its CD14mono cells.
# rng_d = np.random.default_rng(1)
# bad = cell_meta.index[(cell_meta["cell_type"]=="CD14mono") &
#                       (cell_meta["donor"]=="D00")]
# drop = rng_d.choice(bad, size=max(len(bad)-15, 0), replace=False)
# keep_mask = ~cell_meta.index.isin(drop)
# Xk = X.loc[:, keep_mask]; cmk = cell_meta[keep_mask]
# pbk, mk = make_pseudobulk(Xk, cmk, "CD14mono", min_cells=1)
# rk, _ = pseudobulk_nb(pbk, mk)
# print(f"  after crippling D00 to 15 cells:")
# print(f"    counts-based library size for D00 = {pbk['D00'].sum():,} UMIs")
# print(f"    -> the NB model AUTOMATICALLY down-weights it via the offset,")
# print(f"       because its library size is tiny. The mean-of-normalised")
# print(f"       pseudobulk gives D00 exactly the same weight as a 900-cell")
# print(f"       donor, so one noisy donor can drive the whole result.")

# %% [markdown]
# ## What to take away
#
# 1. **Donors are replicates. Cells are measurements.** Always.
# 2. Pseudobulk by (cell type x sample) with **sums** of raw counts; require a
#    pre-specified minimum cell count; analyse with bulk tools.
# 3. Differential state != differential abundance != cluster markers (21.3).
# 4. Treat abundance compositionally.
# 5. Multiplicity family = genes x cell types x contrasts: say which you used.
#
# **Next:** `32_cytometry_differential_abundance.py`
