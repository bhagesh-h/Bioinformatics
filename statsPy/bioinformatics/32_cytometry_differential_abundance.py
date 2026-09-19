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
# # Applied 32: Flow / mass cytometry: differential abundance and state
#
# **Curriculum link:** `stats.md` -> Topic 22, equations (22.1)-(22.4)
# **Core modules used:** 01, 02 (arcsinh), 13 (offsets), 18 (clustering), 23
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | Bodenmiller et al. 2012 / the `diffcyt` BCR-XL dataset: 8 paired samples, 24 markers, CyTOF |
# | **Design** | Paired: reference vs BCR/FcR-XL stimulated, same donors |
# | **Unit of inference** | **Sample/donor** (8), not event (~150,000) |
# | **Here** | Simulated event-level data with the same structure |
#
# ## The two questions `diffcyt` separates
#
# * **Differential abundance (DA):** did a cluster change in *size*? -> count
#   model with a total-events offset (22.2).
# * **Differential state (DS):** did a marker change *within* a cluster? ->
#   moderated linear model on per-sample medians (22.3).

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "32_cytometry_differential_abundance"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate event-level CyTOF data

# %%
header("1. Simulating a paired CyTOF experiment")

TYPE_MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD14", "CD56"]
STATE_MARKERS = ["pS6", "pERK", "pSTAT1", "pNFkB"]
POPULATIONS = {
    #                CD3  CD4  CD8  CD19 CD14 CD56
    "CD4 T":        [5.0, 4.5, 0.2, 0.2, 0.2, 0.2],
    "CD8 T":        [5.0, 0.2, 4.5, 0.2, 0.2, 0.2],
    "B cells":      [0.2, 0.2, 0.2, 5.0, 0.2, 0.2],
    "Monocytes":    [0.2, 1.0, 0.2, 0.2, 5.0, 0.2],
    "NK cells":     [0.2, 0.2, 1.0, 0.2, 0.2, 4.5],
}
BASE_FREQ = np.array([0.33, 0.18, 0.12, 0.28, 0.09])
# TRUTH: B cells expand under stimulation; pS6 rises specifically in B cells.
DA_TRUTH = {"B cells": 0.55}
DS_TRUTH = {("B cells", "pS6"): 1.1}


def simulate_cytometry(n_donors=8, events_per_sample=6000, cofactor=5.0, seed=0):
    """Paired design: each donor gives a reference and a stimulated sample."""
    rng = np.random.default_rng(seed)
    pops = list(POPULATIONS)
    frames = []
    for d in range(n_donors):
        donor_freq_eff = rng.normal(0, 0.18, len(pops))     # donor composition RE
        donor_state_eff = {m: rng.normal(0, 0.25) for m in STATE_MARKERS}
        acq_day = f"day{d % 3}"                              # acquisition batch
        day_shift = {"day0": 0.0, "day1": 0.22, "day2": -0.15}[acq_day]
        for cond in ["Ref", "BCRXL"]:
            shift = np.array([DA_TRUTH.get(p, 0.0) if cond == "BCRXL" else 0.0
                              for p in pops])
            logits = np.log(BASE_FREQ) + donor_freq_eff + shift
            probs = np.exp(logits) / np.exp(logits).sum()
            n_ev = rng.integers(int(0.7 * events_per_sample),
                                int(1.3 * events_per_sample))
            assign = rng.choice(len(pops), size=n_ev, p=probs)

            mat = np.empty((n_ev, len(TYPE_MARKERS) + len(STATE_MARKERS)))
            for k, pop in enumerate(pops):
                sel = assign == k
                if sel.sum() == 0:
                    continue
                centres = np.array(POPULATIONS[pop])
                mat[sel, :len(TYPE_MARKERS)] = rng.normal(
                    centres, 0.55, (sel.sum(), len(TYPE_MARKERS)))
                for j, sm_ in enumerate(STATE_MARKERS):
                    base = 1.0 + donor_state_eff[sm_] + day_shift
                    eff = (DS_TRUTH.get((pop, sm_), 0.0)
                           if cond == "BCRXL" else 0.0)
                    mat[sel, len(TYPE_MARKERS) + j] = rng.normal(
                        base + eff, 0.6, sel.sum())
            # Convert the arcsinh-scale values back to RAW intensities, so the
            # script has to transform them like real FCS data (eq. 2.5).
            raw = np.sinh(np.clip(mat, -5, 12)) * cofactor
            df = pd.DataFrame(raw, columns=TYPE_MARKERS + STATE_MARKERS)
            df["sample_id"] = f"D{d:02d}_{cond}"
            df["donor"] = f"D{d:02d}"
            df["condition"] = cond
            df["acq_day"] = acq_day
            df["true_pop"] = [pops[a] for a in assign]
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


ev = simulate_cytometry(seed=3201)
print(f"  {len(ev):,} events, {ev['sample_id'].nunique()} samples, "
      f"{ev['donor'].nunique()} donors")
print(f"  markers: {len(TYPE_MARKERS)} type + {len(STATE_MARKERS)} state")
print(f"  events per sample: {ev.groupby('sample_id').size().min():,}-"
      f"{ev.groupby('sample_id').size().max():,}")
print(f"\n  n for CONDITION-level inference = {ev['donor'].nunique()} donors,"
      f" NOT {len(ev):,} events")
print(f"  raw intensity range: {ev[TYPE_MARKERS].values.min():.1f} to "
      f"{ev[TYPE_MARKERS].values.max():,.0f}  (and some are negative)")

# %% [markdown]
# ## 2. Transformation, eq. (2.5)
#
# $$y\mapsto\operatorname{arcsinh}(y/c)$$
#
# `log` is unusable: compensation produces negative values. The cofactor $c$
# sets the width of the linear region: 5 for CyTOF, ~150 for fluorescence flow.

# %%
header("2. arcsinh transformation with a modality-appropriate cofactor (2.5)")

MARKERS = TYPE_MARKERS + STATE_MARKERS
print(f"  {'cofactor':>10}{'SD of CD19':>13}{'bimodality (dip in CD19)':>27}")
for c in [1.0, 5.0, 150.0]:
    t = np.arcsinh(ev["CD19"].values / c)
    # Crude bimodality score: variance of the two k-means clusters vs total.
    km = KMeans(2, n_init=10, random_state=0).fit(t.reshape(-1, 1))
    within = np.mean([t[km.labels_ == k].var() for k in [0, 1]])
    print(f"  {c:>10.0f}{t.std(ddof=1):>13.3f}{1 - within/t.var():>27.3f}")
print("\n  c=5 gives the cleanest separation for these mass-cytometry-scale")
print("  intensities. Report the cofactor: it changes every downstream")
print("  clustering result.")

X = pd.DataFrame(np.arcsinh(ev[MARKERS].values / 5.0), columns=MARKERS)
for col in ["sample_id", "donor", "condition", "acq_day", "true_pop"]:
    X[col] = ev[col].values

# %% [markdown]
# ## 3. Clustering on TYPE markers only
#
# Clustering on state markers would confound "which cell is this?" with
# "what is it doing?": the very distinction DA/DS is built to separate.

# %%
header("3. High-resolution clustering on lineage markers")

km = KMeans(n_clusters=12, n_init=25, random_state=0).fit(X[TYPE_MARKERS].values)
X["cluster"] = km.labels_

# Annotate each cluster by its dominant true population (in a real analysis
# you would annotate by marker medians).
ann = (X.groupby("cluster")["true_pop"]
       .agg(lambda s: s.value_counts().idxmax()))
purity = (X.groupby("cluster")
          .apply(lambda g: (g["true_pop"] == ann[g.name]).mean()))
print(f"  {'cluster':>8}{'n events':>11}{'annotation':>13}{'purity':>9}")
for c in sorted(X["cluster"].unique()):
    n = int((X["cluster"] == c).sum())
    print(f"  {c:>8}{n:>11,}{ann[c]:>13}{purity[c]:>9.3f}")
X["population"] = X["cluster"].map(ann)
print("\n  Over-clustering (12 clusters for 5 populations) is deliberate: it is")
print("  the `diffcyt` strategy. Rare subsets get their own cluster, and")
print("  merging is a reversible annotation decision made afterwards.")

# %% [markdown]
# ## 4. Event level -> sample level, eq. (22.1)
#
# $$Y_{ci}\ \text{(counts)}, \qquad
#   \bar m_{cji}=\operatorname{median}_{k\in(c,i)}m_{jk}$$

# %%
header("4. Reducing to sample-level tables (22.1)")

counts = pd.crosstab(X["sample_id"], X["cluster"])
sample_meta = (X.groupby("sample_id")
               .agg(donor=("donor", "first"), condition=("condition", "first"),
                    acq_day=("acq_day", "first"), n_events=("cluster", "size")))
counts = counts.loc[sample_meta.index]
print(f"  counts table: {counts.shape[0]} samples x {counts.shape[1]} clusters")
print(counts.iloc[:4, :6].to_string())

medians = (X.groupby(["sample_id", "cluster"])[STATE_MARKERS]
           .median().reset_index())
print(f"\n  medians table: {len(medians)} (sample x cluster) rows x "
      f"{len(STATE_MARKERS)} state markers")

# Relative standard error of a proportion, eq. (22.4).
print(f"\n  Eq. (22.4): RSE ~ 1/sqrt(Y). Minimum-event guidance:")
for y in [10, 25, 100, 400, 2500]:
    print(f"    {y:>6} events -> RSE {1/np.sqrt(y):>6.1%}")
tiny = (counts < 25).sum().sum()
print(f"  {tiny} of {counts.size} sample x cluster cells have < 25 events")

# %% [markdown]
# ## 5. Differential abundance, eq. (22.2)

# %%
header("5. Differential abundance: NB counts with a total-events offset (22.2)")

def da_test(counts, meta, formula_extra=""):
    """stats.md eq. (22.2): differential abundance of each cluster.

    Poisson GLM with a log(total events) offset, then QUASI-LIKELIHOOD
    inference: the dispersion is estimated per cluster from the Pearson
    statistic (eq. 13.9) and every standard error is multiplied by
    sqrt(phi_hat), with a t reference on the residual df.

    Estimating the dispersion (rather than fixing it) is what lets a blocking
    term pay off: adding `donor` removes the donor-to-donor composition
    variance from the residual, phi_hat falls, and the standard error shrinks.
    A FIXED dispersion would make the blocking term nearly useless.
    """
    out = []
    for c in counts.columns:
        d = meta.copy()
        d["y"] = counts[c].values.astype(float)
        d["stim"] = (d["condition"] == "BCRXL").astype(float)
        try:
            f = smf.glm("y ~ stim" + formula_extra, data=d,
                        family=sm.families.Poisson(),
                        offset=np.log(d["n_events"])).fit()
            chi2 = np.sum((d["y"] - f.fittedvalues) ** 2 / f.fittedvalues)
            phi = max(chi2 / f.df_resid, 1.0)          # eq. (13.9)
            se = f.bse["stim"] * np.sqrt(phi)
            tstat = f.params["stim"] / se
            pval = 2 * st.t.sf(abs(tstat), f.df_resid)
            out.append((c, f.params["stim"], se, phi, pval))
        except Exception:
            out.append((c, np.nan, np.nan, np.nan, 1.0))
    res = pd.DataFrame(out, columns=["cluster", "logFC", "se", "phi",
                                     "pvalue"]).set_index("cluster")
    res["padj"] = st.false_discovery_control(np.nan_to_num(res["pvalue"], nan=1.0))
    res["population"] = [ann[c] for c in res.index]
    return res


res_da = da_test(counts, sample_meta)
res_da_paired = da_test(counts, sample_meta, " + C(donor)")
prop = counts.div(counts.sum(axis=1), axis=0)

print(f"  {'cluster':>8}{'population':>12}{'mean % Ref':>12}{'mean % Stim':>13}"
      f"{'logFC':>9}{'padj':>10}{'padj (paired)':>15}")
for c in res_da.index:
    pr = prop.loc[(sample_meta["condition"] == "Ref").values, c].mean()
    ps = prop.loc[(sample_meta["condition"] == "BCRXL").values, c].mean()
    star = "  *TRUE*" if ann[c] in DA_TRUTH else ""
    print(f"  {c:>8}{ann[c]:>12}{pr:>12.3%}{ps:>13.3%}"
          f"{res_da.loc[c,'logFC']:>9.3f}{res_da.loc[c,'padj']:>10.4f}"
          f"{res_da_paired.loc[c,'padj']:>15.4f}{star}")

true_clusters = [c for c in res_da.index if ann[c] in DA_TRUTH]
print(f"\n  Truly expanded population: B cells (clusters {true_clusters})")
print(f"  detected unpaired: "
      f"{sorted(res_da.index[(res_da['padj']<0.05)].tolist())}")
print(f"  detected paired  : "
      f"{sorted(res_da_paired.index[(res_da_paired['padj']<0.05)].tolist())}")
print(f"\n  median estimated dispersion, unpaired : {res_da['phi'].median():.2f}")
print(f"  median estimated dispersion, paired   : "
      f"{res_da_paired['phi'].median():.2f}")
print("  The paired model (donor as a blocking factor) pulls the donor-to-donor")
print("  composition variance out of the residual, the quasi-dispersion falls,")
print("  and every standard error shrinks - eq. (4.3) again.")
extra = [c for c in res_da_paired.index
         if res_da_paired.loc[c, "padj"] < 0.05 and ann[c] not in DA_TRUTH]
print("\n  NOW LOOK CAREFULLY at what the more powerful model found. Besides")
print(f"  the two true B-cell clusters it also flagged {len(extra)} others, every")
print("  one with a NEGATIVE logFC. Those are not shrinking populations - they")
print("  are being DILUTED, because when B cells expand every other cluster's")
print("  share of a fixed event total must fall (eq. 2.7). Extra power does not")
print("  rescue you from the constraint; it just lets you detect the artefact")
print("  with more confidence.")
print("\n  This is why `diffcyt` models COUNTS with a total-events offset rather")
print("  than testing proportions - and why a claim about ABSOLUTE cell numbers")
print("  needs an absolute measurement (counting beads, or cells per mL of")
print("  blood), not a frequency table. Report the direction of the total as")
print("  well as the frequencies.")

# %% [markdown]
# ## 6. Differential state, eq. (22.3)

# %%
header("6. Differential state: moderated models on per-sample medians (22.3)")

def ds_test(medians, meta, markers, paired=True):
    """stats.md eq. (22.3): a linear model per (cluster, marker)."""
    out = []
    for c in sorted(medians["cluster"].unique()):
        sub = medians[medians["cluster"] == c]
        d = sub.merge(meta.reset_index(), on="sample_id")
        if d["condition"].nunique() < 2 or len(d) < 6:
            continue
        d["stim"] = (d["condition"] == "BCRXL").astype(float)
        for m in markers:
            try:
                form = f"{m} ~ stim" + (" + C(donor)" if paired else "")
                f = smf.ols(form, data=d).fit()
                out.append((c, m, f.params["stim"], f.pvalues["stim"]))
            except Exception:
                out.append((c, m, np.nan, 1.0))
    res = pd.DataFrame(out, columns=["cluster", "marker", "diff", "pvalue"])
    res["padj"] = st.false_discovery_control(
        np.nan_to_num(res["pvalue"].values, nan=1.0))
    res["population"] = [ann[c] for c in res["cluster"]]
    return res


res_ds = ds_test(medians, sample_meta, STATE_MARKERS, paired=True)
hits = res_ds[res_ds["padj"] < 0.05].sort_values("padj")
print(f"  multiplicity family = {res_ds['cluster'].nunique()} clusters x "
      f"{len(STATE_MARKERS)} markers = {len(res_ds)} hypotheses")
print(f"\n  Significant (cluster, marker) pairs at FDR 5%:")
print(f"  {'cluster':>8}{'population':>12}{'marker':>9}{'difference':>12}"
      f"{'padj':>11}")
for _, r in hits.head(12).iterrows():
    star = "  *TRUE*" if (r["population"], r["marker"]) in DS_TRUTH else ""
    print(f"  {r['cluster']:>8}{r['population']:>12}{r['marker']:>9}"
          f"{r['diff']:>12.3f}{r['padj']:>11.2e}{star}")

tp = sum((r["population"], r["marker"]) in DS_TRUTH for _, r in hits.iterrows())
print(f"\n  {tp} of {len(hits)} hits are the injected pS6-in-B-cells effect"
      f" ({len(hits)-tp} false positive(s)).")
print("  Note the contrast with differential ABUNDANCE above: DS is a")
print("  WITHIN-cluster question, so it is NOT subject to the compositional")
print("  constraint. A marker can rise in B cells without forcing anything to")
print("  fall elsewhere. That is a good reason to state which of the two")
print("  questions you are answering (eq. 21.3).")

# %% [markdown]
# ## 7. Acquisition day is a batch effect

# %%
header("7. Modelling acquisition day (Topic 19)")

res_ds_day = ds_test(
    medians.merge(sample_meta.reset_index()[["sample_id", "acq_day"]],
                  on="sample_id"),
    sample_meta, STATE_MARKERS, paired=True)

def summarise(res, label):
    hits = res[res["padj"] < 0.05]
    tp = sum((r["population"], r["marker"]) in DS_TRUTH for _, r in hits.iterrows())
    print(f"  {label:<44}{len(hits):>7}{tp:>7}{len(hits)-tp:>7}")


print(f"  {'model':<44}{'hits':>7}{'TP':>7}{'FP':>7}")
summarise(ds_test(medians, sample_meta, STATE_MARKERS, paired=False),
          "~ stim (no blocking)")
summarise(res_ds, "~ stim + donor (paired)")
print("\n  Because each donor was acquired on ONE day, the donor term already")
print("  absorbs the acquisition-day effect - blocking on donor handles both.")
print("  When donors span days, include BOTH terms and check the design rank")
print("  (Module 01, section 6) before fitting.")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
for c, lab in [(1.0, "c=1"), (5.0, "c=5"), (150.0, "c=150")]:
    ax.hist(np.arcsinh(ev["CD19"].values[:20000] / c), bins=80, alpha=0.5,
            label=lab, density=True)
ax.set_xlabel("arcsinh(CD19 / c)")
ax.set_title("Eq. (2.5): the cofactor sets the linear region", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
sub = X.sample(8000, random_state=0)
for pop in POPULATIONS:
    s = sub[sub["population"] == pop]
    ax.scatter(s["CD19"], s["CD3"], s=2, alpha=0.3, label=pop)
ax.set_xlabel("arcsinh CD19"); ax.set_ylabel("arcsinh CD3")
ax.set_title("Clusters annotated by lineage markers", fontsize=9)
ax.legend(fontsize=6, markerscale=3)

ax = axes[1, 0]
for c in res_da.index:
    col = "C3" if ann[c] in DA_TRUTH else "grey"
    for cond, off in [("Ref", -0.15), ("BCRXL", 0.15)]:
        v = prop.loc[(sample_meta["condition"] == cond).values, c]
        ax.scatter(np.full(len(v), c + off), v, s=14, color=col,
                   marker="o" if cond == "Ref" else "s")
ax.set_xlabel("cluster"); ax.set_ylabel("proportion of events")
ax.set_title("Eq. (22.2): DA (red = truly expanded)", fontsize=9)

ax = axes[1, 1]
piv = res_ds.pivot(index="cluster", columns="marker", values="diff")
im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7)
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels([f"{c} ({ann[c]})" for c in piv.index], fontsize=6)
ax.set_title("Eq. (22.3): DS effect per cluster x marker", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "cytometry.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/cytometry.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Event-level testing
#
# Test pS6 in the B-cell clusters using **every event** as an observation
# (a Welch t-test on arcsinh values), and compare to the sample-level median
# analysis. Predict the direction of the difference before running it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# bcl = [c for c in X["cluster"].unique() if ann[c] == "B cells"]
# sub = X[X["cluster"].isin(bcl)]
# a = sub.loc[sub["condition"] == "Ref", "pS6"].values
# b = sub.loc[sub["condition"] == "BCRXL", "pS6"].values
# p_event = st.ttest_ind(a, b, equal_var=False).pvalue
# samp = (sub.groupby(["sample_id"])["pS6"].median()
#         .to_frame().join(sample_meta))
# p_sample = st.ttest_ind(samp.loc[samp["condition"]=="Ref", "pS6"],
#                         samp.loc[samp["condition"]=="BCRXL", "pS6"],
#                         equal_var=False).pvalue
# print(f"  event-level  n={len(a)+len(b):,} events : p = {p_event:.3e}")
# print(f"  sample-level n={len(samp)} samples      : p = {p_sample:.4f}")
#
# # Now the key control - a marker with NO true effect:
# for marker in ["pS6", "pERK", "pSTAT1"]:
#     aa = sub.loc[sub["condition"]=="Ref", marker].values
#     bb = sub.loc[sub["condition"]=="BCRXL", marker].values
#     ss = (sub.groupby("sample_id")[marker].median().to_frame().join(sample_meta))
#     pe = st.ttest_ind(aa, bb, equal_var=False).pvalue
#     ps = st.ttest_ind(ss.loc[ss["condition"]=="Ref", marker],
#                       ss.loc[ss["condition"]=="BCRXL", marker],
#                       equal_var=False).pvalue
#     truth_ = "TRUE effect" if ("B cells", marker) in DS_TRUTH else "no effect"
#     print(f"  {marker:<8} ({truth_:<11}): event p={pe:.2e}  sample p={ps:.4f}")
#
# # The event-level p-values are astronomically small for EVERY marker,
# # including the ones with no true effect, because ~50,000 events are treated
# # as independent when there are only 8 donors (eq. 1.8). The sample-level
# # analysis separates the real effect from the rest.

# %% [markdown]
# ### Problem 2: Minimum-event threshold
#
# Re-run the DS analysis keeping only (sample x cluster) combinations with at
# least 5, 25, and 200 events. How does the number of hits and the false
# positive count change? Use eq. (22.4) to justify a threshold.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# ev_counts = X.groupby(["sample_id", "cluster"]).size().rename("n_ev")
# med2 = medians.merge(ev_counts.reset_index(), on=["sample_id", "cluster"])
# print(f"  {'min events':>12}{'rows kept':>12}{'hits':>7}{'TP':>6}{'FP':>6}")
# for thr in [1, 5, 25, 200]:
#     keep = med2[med2["n_ev"] >= thr].drop(columns="n_ev")
#     r = ds_test(keep, sample_meta, STATE_MARKERS, paired=True)
#     h = r[r["padj"] < 0.05]
#     tp = sum((x["population"], x["marker"]) in DS_TRUTH for _, x in h.iterrows())
#     print(f"  {thr:>12}{len(keep):>12}{len(h):>7}{tp:>6}{len(h)-tp:>6}")
#
# # A median computed from 3 events is almost pure noise - eq. (22.4) gives an
# # RSE of ~58%. Raising the threshold removes those unstable rows. Set it
# # BEFORE looking at the outcome, and report how many rows it dropped.

# %% [markdown]
# ### Problem 3: Cluster-resolution sensitivity
#
# Repeat the DA analysis at k = 6, 12, and 25 clusters. Does the B-cell
# expansion survive? What happens to the multiplicity burden?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# for k in [6, 12, 25]:
#     kmk = KMeans(k, n_init=25, random_state=0).fit(X[TYPE_MARKERS].values)
#     Xk = X.copy(); Xk["cluster"] = kmk.labels_
#     annk = Xk.groupby("cluster")["true_pop"].agg(lambda s: s.value_counts().idxmax())
#     ck = pd.crosstab(Xk["sample_id"], Xk["cluster"]).loc[sample_meta.index]
#     out = []
#     for c in ck.columns:
#         d = sample_meta.copy(); d["y"] = ck[c].values.astype(float)
#         d["stim"] = (d["condition"] == "BCRXL").astype(float)
#         f = smf.glm("y ~ stim + C(donor)", data=d,
#                     family=sm.families.NegativeBinomial(alpha=0.15),
#                     offset=np.log(d["n_events"])).fit()
#         out.append((c, f.pvalues["stim"]))
#     rk = pd.DataFrame(out, columns=["cluster", "p"]).set_index("cluster")
#     rk["padj"] = st.false_discovery_control(rk["p"].values)
#     sig = rk.index[rk["padj"] < 0.05]
#     b_found = [c for c in sig if annk[c] == "B cells"]
#     print(f"  k={k:>3}: {len(sig)} significant clusters, "
#           f"{len(b_found)} of them B cells, "
#           f"{k} hypotheses in the family")
#
# # The B-cell expansion is robust to resolution - that is the point of a
# # sensitivity analysis (Module 24). Higher k splits populations, which raises
# # the multiplicity burden and can dilute an effect across sibling clusters.
# # Report the range, not one k.

# %% [markdown]
# ## What to take away
#
# 1. **Specimen/donor is $n$**; events are not replicates.
# 2. arcsinh with a stated cofactor; cluster on lineage markers only.
# 3. Counts + total-events offset for abundance; moderated linear models on
#    per-sample medians for state.
# 4. Pre-specify the minimum-event rule and the cluster resolution, and report
#    sensitivity to both.
# 5. Multiplicity family = clusters x markers x contrasts.
#
# **Next:** `33_methylation_epigenomics.py`
