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
# # Applied 39: Functional enrichment and pathway statistics
#
# **Curriculum link:** `stats.md` -> Topic 29, equations (29.1)-(29.5)
# **Core modules used:** 01 (design effect!), 08 (multiplicity), 30
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | MSigDB Hallmark / GO-BP sets applied to an RNA-seq result |
# | **Input** | The full ranked statistic vector, plus gene->set membership |
# | **Key threats** | Wrong universe, threshold loss, **inter-gene correlation**, detectability bias |
# | **Here** | Simulated with correlated gene modules and a known enriched set |
#
# ## The single most important fact
#
# Genes in a pathway are **co-regulated**, so they are not independent draws.
# The variance of a set-level statistic is inflated by the *same* design effect
# that clustered cells produce (eq. 1.8 = eq. 29.5). Ignoring it is the main
# source of false pathway discoveries.

# %%
import math
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "39_gene_set_enrichment"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate expression with correlated modules and gene sets

# %%
header("1. A differential-expression result with CORRELATED gene sets")

def simulate_for_enrichment(G=2000, n_per=8, n_sets=60, set_size=40,
                            n_true_sets=2, effect=1.7, rho=0.45, seed=0):
    """Genes live in correlated modules; gene sets are drawn over those modules.

    `rho` is the within-module correlation - the parameter that drives the
    variance inflation of eq. (29.5).
    """
    rng = np.random.default_rng(seed)
    genes = [f"G{i:04d}" for i in range(G)]

    # Modules of co-regulated genes.
    module_size = set_size
    n_modules = G // module_size
    module_of = np.repeat(np.arange(n_modules), module_size)[:G]

    # Gene sets: the first `n_true_sets` align exactly with modules 0,1.
    sets = {}
    for s in range(n_true_sets):
        sets[f"SET_TRUE_{s}"] = [genes[i] for i in np.where(module_of == s)[0]]
    for s in range(n_sets - n_true_sets):
        if s < (n_sets - n_true_sets) // 2:
            # HALF the null sets are also COHERENT (drawn from one module),
            # which is what real pathways look like.
            m = rng.integers(n_true_sets, n_modules)
            members = np.where(module_of == m)[0]
        else:
            # The other half are random gene collections (incoherent).
            members = rng.choice(G, module_size, replace=False)
        sets[f"SET_null_{s}"] = [genes[i] for i in members]

    # Expression: shared module factor induces within-module correlation rho.
    n = 2 * n_per
    group = np.r_[np.zeros(n_per), np.ones(n_per)]
    X = np.empty((G, n))
    for m in range(n_modules):
        idx = np.where(module_of == m)[0]
        f = rng.normal(0, 1, n)                        # module factor
        X[idx] = (np.sqrt(rho) * f[None, :]
                  + np.sqrt(1 - rho) * rng.normal(0, 1, (len(idx), n)))

    # TRUE effect: only the genes of modules 0 and 1 respond.
    true_de = np.zeros(G, bool)
    for s in range(n_true_sets):
        idx = np.where(module_of == s)[0]
        true_de[idx] = True
        X[idx] += effect * group[None, :]

    # Detectability: expression level predicts power (long/high genes win).
    expr_level = np.exp(rng.normal(0, 1.0, G))
    noise_scale = 1.0 / np.sqrt(1 + expr_level / 2)
    X = X * noise_scale[:, None]

    res = st.ttest_ind(X[:, group == 0], X[:, group == 1], axis=1)
    stats = pd.DataFrame({
        "t": -res.statistic, "p": res.pvalue,
        "padj": st.false_discovery_control(res.pvalue),
        "expr_level": expr_level, "true_de": true_de,
        "module": module_of}, index=genes)
    true_sets = [f"SET_TRUE_{s}" for s in range(n_true_sets)]
    return stats, sets, true_sets, X, group


stats, gene_sets, TRUE_SETS, X, group = simulate_for_enrichment(seed=3901)
print(f"  {len(stats)} genes, {len(gene_sets)} gene sets of "
      f"{len(next(iter(gene_sets.values())))} genes each")
print(f"  truly enriched sets: {TRUE_SETS}")
print(f"  genes with a true effect: {int(stats['true_de'].sum())}")
print(f"  genes significant at FDR 5%: {int((stats['padj'] < 0.05).sum())}")

# Verify the induced inter-gene correlation.
for label, sname in [("TRUE set", TRUE_SETS[0]),
                     ("coherent null set", "SET_null_0"),
                     ("random null set", f"SET_null_{len(gene_sets)-3}")]:
    idx = [g for g in gene_sets[sname] if g in stats.index]
    sub = X[[stats.index.get_loc(g) for g in idx]]
    C = np.corrcoef(sub)
    off = ~np.eye(len(C), dtype=bool)
    print(f"  mean inter-gene correlation, {label:<20}: {C[off].mean():+.3f}")

# %% [markdown]
# ## 2. Over-representation analysis, eq. (29.1)
#
# $$\Pr(X\ge k)=\sum_{x\ge k}\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}$$

# %%
header("2. ORA and the three ways it goes wrong (29.1)")

def ora(sig_genes, gene_sets, universe):
    """stats.md eq. (29.1): hypergeometric over-representation."""
    universe = set(universe)
    sig = set(sig_genes) & universe
    N, n = len(universe), len(sig)
    rows = []
    for name, members in gene_sets.items():
        S = set(members) & universe
        K = len(S)
        k = len(S & sig)
        if K == 0:
            continue
        p = st.hypergeom.sf(k - 1, N, K, n)
        rows.append({"set": name, "k": k, "K": K, "n": n, "N": N, "p": p})
    r = pd.DataFrame(rows).set_index("set")
    r["padj"] = st.false_discovery_control(r["p"].values)
    return r


sig = stats.index[stats["padj"] < 0.05]
tested_universe = stats.index                      # genes actually TESTED
genome_universe = list(stats.index) + [f"UNTESTED{i}" for i in range(6000)]

r_correct = ora(sig, gene_sets, tested_universe)
r_wrong_universe = ora(sig, gene_sets, genome_universe)

def summarise(r, label):
    rej = r["padj"] < 0.05
    tp = int(rej.reindex(TRUE_SETS).fillna(False).sum())
    print(f"  {label:<44}{rej.sum():>8}{tp:>6}{rej.sum()-tp:>6}")


print(f"  {'analysis':<44}{'sig sets':>8}{'TP':>6}{'FP':>6}")
summarise(r_correct, "universe = genes actually TESTED")
summarise(r_wrong_universe, "universe = whole 'genome' (WRONG)")
print("\n  Inflating the universe with genes that were never testable makes")
print("  every overlap look surprising. The universe must be the set of genes")
print("  that COULD have been detected and tested.")

# Threshold sensitivity - ORA throws away the ranking.
print(f"\n  ORA depends entirely on an arbitrary threshold:")
print(f"  {'FDR cutoff':>12}{'genes in list':>15}{'sig sets':>11}{'TRUE found':>13}")
for cut in [0.001, 0.01, 0.05, 0.10, 0.25]:
    s2 = stats.index[stats["padj"] < cut]
    r2 = ora(s2, gene_sets, tested_universe)
    rej = r2["padj"] < 0.05
    print(f"  {cut:>12.3f}{len(s2):>15}{rej.sum():>11}"
          f"{int(rej.reindex(TRUE_SETS).fillna(False).sum()):>13}")
print("  A gene at q = 0.051 contributes nothing; one at q = 0.049 contributes")
print("  fully. That discontinuity is why RANKED methods are preferred.")

# %% [markdown]
# ## 3. Competitive vs self-contained nulls, eq. (29.2)-(29.3)

# %%
header("3. Two different null hypotheses (29.2)-(29.3)")

print("""  SELF-CONTAINED (29.2): 'no gene in this set is associated'
      -> permute SAMPLE labels. Answers: is anything happening here at all?
  COMPETITIVE   (29.3): 'genes in this set are no more associated than
                         genes outside it'
      -> permute GENE labels. Answers: is this set special relative to the
         rest of the transcriptome?

  They can disagree. If a treatment shifts the whole transcriptome, a
  self-contained test is significant for nearly every set while a competitive
  test is significant for almost none. State which you mean.""")

def self_contained_test(stats_t, members, X, group, n_perm=199, seed=0):
    """Permute SAMPLE labels; statistic = mean |t| in the set."""
    rng = np.random.default_rng(seed)
    idx = [stats_t.index.get_loc(g) for g in members if g in stats_t.index]
    obs = np.mean(np.abs(stats_t["t"].values[idx]))
    null = []
    for _ in range(n_perm):
        gp = rng.permutation(group)
        r = st.ttest_ind(X[idx][:, gp == 0], X[idx][:, gp == 1], axis=1)
        null.append(np.mean(np.abs(np.nan_to_num(r.statistic))))
    return (1 + np.sum(np.array(null) >= obs)) / (n_perm + 1)


def competitive_test_naive(stats_t, members):
    """Permute GENE labels implicitly: a t-test of in-set vs out-of-set stats.

    This is the NAIVE competitive test - it assumes genes are independent.
    """
    inset = stats_t.loc[[g for g in members if g in stats_t.index], "t"].values
    outset = stats_t.loc[~stats_t.index.isin(members), "t"].values
    return st.ttest_ind(inset, outset, equal_var=False).pvalue


rows = []
for name, members in list(gene_sets.items())[:14]:
    rows.append({
        "set": name,
        "self_contained": self_contained_test(stats, members, X, group,
                                              n_perm=99, seed=1),
        "competitive_naive": competitive_test_naive(stats, members),
        "is_true": name in TRUE_SETS})
cmp_df = pd.DataFrame(rows).set_index("set")
print(f"\n  {'set':<16}{'self-contained p':>18}{'competitive p':>16}{'truth':>8}")
for name, r in cmp_df.iterrows():
    print(f"  {name:<16}{r['self_contained']:>18.4f}"
          f"{r['competitive_naive']:>16.2e}{'TRUE' if r['is_true'] else '-':>8}")

# %% [markdown]
# ## 4. Inter-gene correlation, eq. (29.5): THE key correction
#
# $$\mathrm{VIF}=1+(m-1)\bar\rho$$
#
# **This is the same formula as eq. (1.8).** Correlated units: cells in a
# donor, or genes in a pathway: inflate variance identically.

# %%
header("4. CAMERA: dividing out the inter-gene correlation (29.5)")

def mean_inter_gene_correlation(X, idx):
    """Estimate rho-bar from the residual expression of the set's genes."""
    sub = X[idx]
    if len(sub) < 2:
        return 0.0
    C = np.corrcoef(sub)
    off = ~np.eye(len(C), dtype=bool)
    return float(np.nanmean(C[off]))


def camera_test(stats_t, members, X):
    """stats.md eq. (29.5): a competitive test with the VIF divided out.

    This is the essential idea of limma::camera - estimate the mean pairwise
    correlation among the set's genes and inflate the variance of the set-level
    statistic by 1 + (m-1)*rho_bar before computing the p-value.
    """
    names = [g for g in members if g in stats_t.index]
    idx = [stats_t.index.get_loc(g) for g in names]
    m = len(idx)
    inset = stats_t["t"].values[idx]
    mask = np.ones(len(stats_t), bool); mask[idx] = False
    outset = stats_t["t"].values[mask]

    rho_bar = mean_inter_gene_correlation(X, idx)
    vif = max(1.0 + (m - 1) * rho_bar, 1e-6)           # eq. (29.5)

    n1, n2 = m, len(outset)
    s2 = ((n1 - 1) * inset.var(ddof=1) + (n2 - 1) * outset.var(ddof=1)) / (n1 + n2 - 2)
    # The in-set mean's variance is inflated by the VIF.
    se = np.sqrt(s2 * (vif / n1 + 1.0 / n2))
    tstat = (inset.mean() - outset.mean()) / se
    p = 2 * st.t.sf(abs(tstat), n1 + n2 - 2)
    return p, rho_bar, vif


rows = []
for name, members in gene_sets.items():
    p_naive = competitive_test_naive(stats, members)
    p_cam, rho_bar, vif = camera_test(stats, members, X)
    rows.append({"set": name, "p_naive": p_naive, "p_camera": p_cam,
                 "rho_bar": rho_bar, "vif": vif, "is_true": name in TRUE_SETS})
cam = pd.DataFrame(rows).set_index("set")
cam["padj_naive"] = st.false_discovery_control(cam["p_naive"].values)
cam["padj_camera"] = st.false_discovery_control(cam["p_camera"].values)

print(f"  {'method':<40}{'sig sets':>10}{'TP':>6}{'FP':>6}{'FDP':>8}")
for col, lab in [("padj_naive", "naive competitive (independence)"),
                 ("padj_camera", "CAMERA-style, VIF divided out (29.5)")]:
    rej = cam[col] < 0.05
    tp = int((rej & cam["is_true"]).sum()); fp = int((rej & ~cam["is_true"]).sum())
    print(f"  {lab:<40}{rej.sum():>10}{tp:>6}{fp:>6}"
          f"{fp/max(rej.sum(),1):>8.3f}")

print(f"\n  {'set type':<26}{'mean rho-bar':>15}{'mean VIF':>11}"
      f"{'sqrt(VIF)':>12}")
coherent = cam.index.str.contains("TRUE") | \
           cam.index.isin([f"SET_null_{s}" for s in range((len(gene_sets)-2)//2)])
for lab, m in [("coherent (one module)", coherent), ("random gene collections", ~coherent)]:
    print(f"  {lab:<26}{cam.loc[m,'rho_bar'].mean():>15.3f}"
          f"{cam.loc[m,'vif'].mean():>11.2f}"
          f"{np.sqrt(cam.loc[m,'vif'].mean()):>12.2f}")

print("\n  If CAMERA finds fewer sets than the naive test, that is the point,")
print("  not a defect - it is removing sets whose apparent significance came")
print("  from within-set correlation rather than from signal. It IS more")
print("  conservative, and with a strong VIF it can lose genuinely enriched")
print("  sets too; report both the correlation estimate and the corrected p.")
print("\n  Coherent sets - the ones that look like real pathways - carry a large")
print("  VIF. The naive test treats their 40 genes as 40 independent pieces of")
print("  evidence when they are closer to one. Dividing the VIF out is why")
print("  CAMERA controls the error rate where the naive test does not.")
print("\n  Note this is EXACTLY eq. (1.8) from Module 01. Correlated units")
print("  inflate variance the same way whether they are cells in a donor or")
print("  genes in a pathway.")

# %% [markdown]
# ## 5. A ranked enrichment score, eq. (29.4)

# %%
header("5. GSEA-style running enrichment score (29.4)")

def enrichment_score(ranked_genes, members, weights=None, w=1.0):
    """stats.md eq. (29.4). Returns the running sum and its extreme value."""
    members = set(members)
    N = len(ranked_genes)
    in_set = np.array([g in members for g in ranked_genes])
    N_S = in_set.sum()
    if N_S == 0 or N_S == N:
        return np.zeros(N), 0.0
    r = np.abs(weights) ** w if weights is not None else np.ones(N)
    N_R = r[in_set].sum()
    step = np.where(in_set, r / N_R, -1.0 / (N - N_S))
    running = np.cumsum(step)
    es = running[np.argmax(np.abs(running))]
    return running, float(es)


def gsea_test(stats_t, members, n_perm=499, seed=0, w=1.0):
    """Normalised ES with a GENE-label permutation null (competitive)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-stats_t["t"].values)
    ranked = stats_t.index.values[order]
    wts = stats_t["t"].values[order]
    _, es = enrichment_score(ranked, members, wts, w)
    m = len([g for g in members if g in stats_t.index])
    null = []
    for _ in range(n_perm):
        fake = rng.choice(ranked, m, replace=False)
        _, e = enrichment_score(ranked, fake, wts, w)
        null.append(e)
    null = np.array(null)
    side = null[null >= 0] if es >= 0 else null[null < 0]
    p = (1 + np.sum(np.abs(side) >= abs(es))) / (len(side) + 1) if len(side) else 1.0
    nes = es / np.mean(np.abs(side)) if len(side) else np.nan
    return es, nes, p


print(f"  {'set':<16}{'ES':>9}{'NES':>9}{'p (gene perm)':>16}{'truth':>8}")
gsea_rows = []
for name in list(gene_sets)[:10]:
    es, nes, p = gsea_test(stats, gene_sets[name], n_perm=299, seed=2)
    gsea_rows.append({"set": name, "ES": es, "NES": nes, "p": p,
                      "is_true": name in TRUE_SETS})
    print(f"  {name:<16}{es:>9.3f}{nes:>9.3f}{p:>16.4f}"
          f"{'TRUE' if name in TRUE_SETS else '-':>8}")

print("\n  The ES walks the ranked list, so no threshold is needed and the")
print("  magnitude of each statistic contributes (w=1). But note: this")
print("  GENE-label permutation is a COMPETITIVE null that still assumes genes")
print("  are exchangeable - i.e. it does NOT fix the correlation problem of")
print("  section 4. GSEA's sample-permutation variant does, at the cost of")
print("  needing enough samples to permute (with n=3 per group there are only")
print(f"  {math.comb(6, 3)} distinct splits, so the minimum p-value is "
      f"{1/math.comb(6, 3):.3f}).")

# %% [markdown]
# ## 6. Detectability bias

# %%
header("6. Detectability bias: highly expressed genes win by default")

print(f"  Correlation between expression level and |t|: "
      f"{st.spearmanr(stats['expr_level'], stats['t'].abs()).statistic:+.3f}")
print(f"\n  {'expression quintile':<24}{'% significant':>16}")
qs = pd.qcut(stats["expr_level"], 5, labels=[f"Q{i+1}" for i in range(5)])
for lvl in qs.cat.categories:
    m = qs == lvl
    print(f"  {str(lvl):<24}{(stats.loc[m,'padj'] < 0.05).mean():>16.1%}")

# A set enriched for highly expressed genes, with NO true effect.
top_expr = stats.nlargest(40, "expr_level").index.tolist()
p_biased = competitive_test_naive(stats, top_expr)
p_biased_cam, _, _ = camera_test(stats, top_expr, X)
print(f"\n  A synthetic set of the 40 MOST EXPRESSED genes (no true effect):")
print(f"    naive competitive p = {p_biased:.4f}")
print(f"    CAMERA-style p      = {p_biased_cam:.4f}")
print("\n  Pathways enriched for long, highly expressed genes (neuronal, ECM)")
print("  come out significant in almost any RNA-seq screen. Correct for the")
print("  detectability covariate, as `goseq` does for gene length and")
print("  `missMethyl::gometh` does for probe number (Module 33).")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
order = np.argsort(-stats["t"].values)
ranked = stats.index.values[order]
wts = stats["t"].values[order]
for name, c in [(TRUE_SETS[0], "C3"), ("SET_null_0", "C0")]:
    run, es = enrichment_score(ranked, gene_sets[name], wts)
    ax.plot(run, color=c, label=f"{name} (ES={es:.2f})")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("rank in the ordered list"); ax.set_ylabel("running ES")
ax.set_title("Eq. (29.4): GSEA running score", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
ax.scatter(cam["rho_bar"], cam["vif"], s=25,
           c=["C3" if t else "grey" for t in cam["is_true"]])
mm = 40
rr = np.linspace(0, cam["rho_bar"].max(), 50)
ax.plot(rr, 1 + (mm - 1) * rr, "k--", lw=1, label="$1+(m-1)\\bar\\rho$")
ax.set_xlabel("mean inter-gene correlation $\\bar\\rho$")
ax.set_ylabel("VIF")
ax.set_title("Eq. (29.5) = eq. (1.8)", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.scatter(-np.log10(cam["p_naive"]), -np.log10(cam["p_camera"]), s=25,
           c=["C3" if t else "grey" for t in cam["is_true"]])
lim = [0, max(-np.log10(cam["p_naive"]).max(), 1)]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("naive competitive $-\\log_{10}p$")
ax.set_ylabel("CAMERA-style $-\\log_{10}p$")
ax.set_title("Correlation correction moves null sets down", fontsize=9)

ax = axes[1, 1]
ax.scatter(np.log(stats["expr_level"]), stats["t"].abs(), s=3, alpha=0.2)
ax.set_xlabel("log expression level"); ax.set_ylabel("|t statistic|")
ax.set_title("Detectability bias", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "enrichment.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/enrichment.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: How bad is the naive competitive test under a pure null?
#
# Simulate data with **no** differential expression at all, but with correlated
# modules, and measure the false-positive rate of the naive competitive test
# versus the CAMERA-style test.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# fp_naive, fp_cam, n_tests = 0, 0, 0
# for rep in range(12):
#     s0, sets0, _, X0, g0 = simulate_for_enrichment(
#         n_true_sets=0, effect=0.0, seed=7000 + rep)
#     for name, members in sets0.items():
#         fp_naive += competitive_test_naive(s0, members) < 0.05
#         fp_cam += camera_test(s0, members, X0)[0] < 0.05
#         n_tests += 1
# print(f"  {n_tests} set tests under a COMPLETE null (no DE at all):")
# print(f"    naive competitive false-positive rate : {fp_naive/n_tests:.1%}")
# print(f"    CAMERA-style false-positive rate      : {fp_cam/n_tests:.1%}")
# print(f"    (nominal 5%)")
#
# # Under a complete null the naive test is anticonservative for COHERENT sets,
# # because their genes' statistics move together and the set mean is far more
# # variable than independence predicts. Dividing by the VIF restores
# # calibration. Try raising `rho` to 0.7 and watch the gap widen.

# %% [markdown]
# ### Problem 2: The universe, quantified
#
# Vary the number of untested genes added to the universe from 0 to 20,000 and
# plot how many sets become "significant". Explain using eq. (29.1).

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'untested genes added':>22}{'universe N':>13}{'sig sets':>11}")
# for extra in [0, 2000, 6000, 20000]:
#     uni = list(stats.index) + [f"X{i}" for i in range(extra)]
#     r = ora(sig, gene_sets, uni)
#     print(f"  {extra:>22,}{len(uni):>13,}{int((r['padj'] < 0.05).sum()):>11}")
#
# # In eq. (29.1) the universe size N enters the denominator binomial C(N, n).
# # Adding genes that were never tested makes any observed overlap look less
# # probable under the null, so p-values fall and "significance" is manufactured.
# # The universe must be the DETECTABLE, TESTED gene set - typically the rows
# # surviving your expression filter (Module 30, section 2).

# %% [markdown]
# ### Problem 3: Redundant, overlapping sets
#
# Build several heavily overlapping sets around one true module and show that
# they all come out significant. How should a results table present them?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# base = gene_sets[TRUE_SETS[0]]
# rng_r = np.random.default_rng(17)
# nested = {}
# for frac in [1.0, 0.8, 0.6, 0.4]:
#     k = int(len(base) * frac)
#     pad = list(rng_r.choice(stats.index, len(base) - k, replace=False))
#     nested[f"overlap_{int(frac*100)}pct"] = list(base[:k]) + pad
# print(f"  {'set':<20}{'overlap with TRUE':>20}{'CAMERA p':>12}")
# for name, members in nested.items():
#     ov = len(set(members) & set(base)) / len(base)
#     p, _, _ = camera_test(stats, members, X)
#     print(f"  {name:<20}{ov:>19.0%}{p:>12.2e}")
#
# # All of them are significant, because they are mostly the SAME GENES. A
# # results table listing them as separate discoveries triple-counts one finding.
# # Report: (a) the leading-edge genes driving each set, (b) a redundancy measure
# # (Jaccard overlap between significant sets), and (c) a collapsed
# # representative per cluster of similar sets. Pathway names are not
# # independent discoveries.

# %% [markdown]
# ## What to take away
#
# 1. Prefer ranked/competitive tests over threshold-based ORA.
# 2. Define the universe as the **tested, detectable** genes.
# 3. **Account for inter-gene correlation (29.5)**: it is the same design
#    effect as eq. (1.8).
# 4. Correct for detectability bias (length, expression, probe number).
# 5. Report leading-edge genes and set overlap; record annotation versions.
#
# **Next:** `40_networks_and_multiomics.py`
