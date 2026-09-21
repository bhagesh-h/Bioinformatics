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
# # Applied 24: Genotypes, GWAS, and statistical genetics
#
# **Curriculum link:** `stats.md` -> Topic 24, equations (24.1)-(24.9)
# **Core modules used:** 08 (multiplicity), 13 (logistic), 17 (PCs), 31, 32
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | A two-ancestry case/control GWAS (1000 Genomes-style LD structure) |
# | **Assay** | SNP dosages $g\in\{0,1,2\}$ |
# | **Key threats** | Population structure, relatedness, LD, extreme multiplicity |
# | **Here** | Simulated with ancestry structure, LD blocks and a known causal variant |
#
# ## The central problem
#
# Ancestry creates allele-frequency differences that correlate with phenotype.
# Uncorrected, this produces genome-wide false positives that look exactly like
# real associations.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "24_gwas_association"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulate a structured cohort with LD blocks

# %%
header("1. Genotypes with ancestry structure and LD")

def simulate_gwas(n=2000, n_snps=4000, block_size=25, n_causal=3,
                  beta_causal=0.45, ancestry_effect=0.9, seed=0):
    """Two ancestral populations, LD blocks, and a phenotype that depends on
    ancestry as well as on a few causal variants."""
    rng = np.random.default_rng(seed)
    pop = rng.binomial(1, 0.45, n)                       # ancestry label

    # Allele frequencies differ between populations (F_ST-like divergence).
    anc_freq = rng.uniform(0.1, 0.9, n_snps)
    fst = 0.12
    f0 = np.clip(rng.beta(anc_freq * (1 - fst) / fst,
                          (1 - anc_freq) * (1 - fst) / fst), 0.02, 0.98)
    f1 = np.clip(rng.beta(anc_freq * (1 - fst) / fst,
                          (1 - anc_freq) * (1 - fst) / fst), 0.02, 0.98)

    G = np.empty((n, n_snps), dtype=np.int8)
    n_blocks = n_snps // block_size
    for b in range(n_blocks):
        cols = slice(b * block_size, (b + 1) * block_size)
        # One latent haplotype per block per individual creates LD.
        for a in range(2):                              # two haplotypes
            latent = rng.normal(0, 1, n)
            for k, j in enumerate(range(cols.start, cols.stop)):
                f = np.where(pop == 1, f1[j], f0[j])
                # Correlate with the block latent, decaying with distance.
                rho = 0.85 ** k
                u = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
                thr = st.norm.ppf(1 - f)
                hap = (u > thr).astype(np.int8)
                if a == 0:
                    G[:, j] = hap
                else:
                    G[:, j] += hap

    causal = rng.choice(n_snps, n_causal, replace=False)
    # Phenotype: ancestry has a DIRECT effect (the confounder), plus true SNPs.
    lin = (ancestry_effect * pop
           + G[:, causal].astype(float) @ np.full(n_causal, beta_causal)
           - 1.0)
    y = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    return G, y, pop, causal, np.arange(n_snps)


G, y, pop, causal, snp_pos = simulate_gwas(seed=3401)
n, n_snps = G.shape
print(f"  {n} individuals x {n_snps} SNPs")
print(f"  case prevalence: overall {y.mean():.3f}; "
      f"pop0 {y[pop==0].mean():.3f}, pop1 {y[pop==1].mean():.3f}")
print(f"  causal SNPs: {sorted(causal)}")
print("\n  Ancestry raises BOTH allele frequencies and disease risk. That is a")
print("  textbook confounder (Module 32, fork structure).")

# %% [markdown]
# ## 2. Quality control, eq. (24.2)

# %%
header("2. QC: MAF, call rate, Hardy-Weinberg (24.2)")

def hwe_pvalue(g):
    """stats.md eq. (24.2). Chi-square goodness of fit for HWE."""
    n_aa = np.sum(g == 0); n_ab = np.sum(g == 1); n_bb = np.sum(g == 2)
    N = n_aa + n_ab + n_bb
    if N == 0:
        return 1.0
    p = (2 * n_bb + n_ab) / (2 * N)
    exp = np.array([N * (1 - p) ** 2, 2 * N * p * (1 - p), N * p ** 2])
    obs = np.array([n_aa, n_ab, n_bb])
    keep = exp > 0
    if keep.sum() < 2:
        return 1.0
    return st.chisquare(obs[keep], exp[keep], ddof=1).pvalue


maf = np.minimum(G.mean(axis=0) / 2, 1 - G.mean(axis=0) / 2)
# CRITICAL: HWE is tested in CONTROLS only. A real disease association can
# cause HWE deviation in cases, so testing everyone throws away true signals.
hwe_ctrl = np.array([hwe_pvalue(G[y == 0, j]) for j in range(n_snps)])
hwe_all = np.array([hwe_pvalue(G[:, j]) for j in range(n_snps)])

keep = (maf > 0.01) & (hwe_ctrl > 1e-6)
print(f"  MAF > 0.01              : {int((maf > 0.01).sum())} SNPs pass")
print(f"  HWE in CONTROLS > 1e-6  : {int((hwe_ctrl > 1e-6).sum())} pass")
print(f"  HWE in EVERYONE > 1e-6  : {int((hwe_all > 1e-6).sum())} pass")
print(f"  combined                : {int(keep.sum())} of {n_snps} SNPs retained")
print(f"\n  causal SNPs surviving QC: {[int(c) for c in causal if keep[c]]}")
print("  Note the HWE columns differ. Apply HWE filtering to CONTROLS only -")
print("  a genuine association can push CASES out of equilibrium, and filtering")
print("  on everyone would discard exactly the variants you are looking for.")

# %% [markdown]
# ## 3. The naive scan, and genomic inflation, eq. (24.1), (24.4)
#
# $$\lambda_{\mathrm{GC}}=\frac{\operatorname{median}(\chi^2_{\text{obs}})}{0.4549}$$

# %%
header("3. Uncorrected scan -> genome-wide false positives (24.1), (24.4)")

def scan(G, y, covars=None, keep=None):
    """Per-SNP additive logistic regression, eq. (24.1)."""
    idx = np.where(keep)[0] if keep is not None else np.arange(G.shape[1])
    base = [np.ones(len(y))]
    if covars is not None:
        base += [covars[:, k] for k in range(covars.shape[1])]
    B = np.column_stack(base)
    out = np.full(G.shape[1], np.nan)
    betas = np.full(G.shape[1], np.nan)
    for j in idx:
        X = np.column_stack([B, G[:, j].astype(float)])
        try:
            f = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            out[j] = f.pvalues[-1]; betas[j] = f.params[-1]
        except Exception:
            pass
    return out, betas


def lambda_gc(p):
    """stats.md eq. (24.4). median of a chi2_1 is 0.4549."""
    p = p[np.isfinite(p) & (p > 0)]
    chi2 = st.chi2.isf(p, 1)
    return np.median(chi2) / 0.4549


ALPHA_GW = 5e-8
p_naive, b_naive = scan(G, y, keep=keep)
print(f"  lambda_GC (uncorrected) = {lambda_gc(p_naive):.3f}   "
      f"(1.00 = calibrated)")
print(f"  SNPs below 5e-8         : {int(np.nansum(p_naive < ALPHA_GW))}")
print(f"  of which are causal     : "
      f"{int(np.nansum((p_naive < ALPHA_GW) & np.isin(np.arange(n_snps), causal)))}")
print(f"\n  Bonferroni threshold for {int(keep.sum())} tested SNPs: "
      f"{0.05/keep.sum():.2e}")
print(f"  The conventional 5e-8 (eq. 24.5) is 0.05 / 10^6, a correction for the")
print("  ~1 million effectively independent common variants in European LD.")

# %% [markdown]
# ## 4. Correcting for structure: ancestry PCs, eq. (24.3)

# %%
header("4. Principal components of the genotype matrix (24.3)")

def genotype_pcs(G, keep, k=10, ld_thin=5):
    """PCs of the standardised genotype matrix. Thinning reduces LD's influence
    on the leading components - a standard step (as is MHC removal in practice).
    """
    idx = np.where(keep)[0][::ld_thin]
    Z = G[:, idx].astype(float)
    Z = (Z - Z.mean(axis=0)) / np.maximum(Z.std(axis=0), 1e-8)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return (U * S)[:, :k], S**2 / np.sum(S**2)


pcs, pve = genotype_pcs(G, keep)
print(f"  {'PC':>4}{'PVE':>9}{'|corr with ancestry|':>24}")
for j in range(5):
    print(f"  {j+1:>4}{pve[j]:>9.2%}{abs(np.corrcoef(pcs[:,j], pop)[0,1]):>24.3f}")
print("\n  PC1 captures ancestry almost perfectly. In a real study you would")
print("  also plot PCs against a reference panel (1000 Genomes) to LABEL the")
print("  ancestry groups rather than just adjust for them.")

for n_pc in [0, 1, 2, 5, 10]:
    cov = pcs[:, :n_pc] if n_pc else None
    p_adj, b_adj = scan(G, y, covars=cov, keep=keep)
    n_sig = int(np.nansum(p_adj < ALPHA_GW))
    n_true = int(np.nansum((p_adj < ALPHA_GW) & np.isin(np.arange(n_snps), causal)))
    print(f"  {n_pc:>2} PCs: lambda_GC = {lambda_gc(p_adj):.3f}, "
          f"{n_sig:>3} genome-wide hits, {n_true} of them causal")
    if n_pc == 10:
        p_final, b_final = p_adj, b_adj

print("\n  Adjusting for ancestry PCs restores calibration (lambda -> 1) and")
print("  leaves the true signals. Note lambda_GC > 1 is NOT automatically")
print("  confounding: a highly polygenic trait genuinely inflates it at large")
print("  n. LD score regression separates polygenicity (slope) from")
print("  confounding (intercept); the QQ plot is the visual version.")

# %% [markdown]
# ## 5. LD and clumping, eq. (24.6)
#
# $$r^2=\frac{D^2}{p_A(1-p_A)p_B(1-p_B)}$$

# %%
header("5. LD means hits come in blocks (24.6)")

sig_idx = np.where(np.isfinite(p_final) & (p_final < ALPHA_GW))[0]
print(f"  {len(sig_idx)} SNPs pass 5e-8 after PC adjustment")

def clump(p, G, sig_idx, r2_thresh=0.1, window=60):
    """Greedy clumping: take the best SNP, drop its LD partners, repeat."""
    order = sig_idx[np.argsort(p[sig_idx])]
    lead, taken = [], set()
    for j in order:
        if j in taken:
            continue
        lead.append(j)
        lo, hi = max(0, j - window), min(G.shape[1], j + window)
        for k in range(lo, hi):
            if k in sig_idx and k not in taken:
                gj, gk = G[:, j].astype(float), G[:, k].astype(float)
                if gj.std() > 0 and gk.std() > 0:
                    if np.corrcoef(gj, gk)[0, 1] ** 2 > r2_thresh:
                        taken.add(k)
        taken.add(j)
    return lead


lead_snps = clump(p_final, G, sig_idx)
print(f"  after clumping (r^2 < 0.1): {len(lead_snps)} independent loci")
print(f"  lead SNPs: {sorted(lead_snps)}")
print(f"  true causal SNPs: {sorted(causal)}")
print(f"  lead SNPs that ARE causal: "
      f"{sorted(set(lead_snps) & set(causal.tolist()))}")

for L in lead_snps[:3]:
    near = [c for c in causal if abs(c - L) < 30]
    if near:
        c0 = near[0]
        r2 = np.corrcoef(G[:, L].astype(float), G[:, c0].astype(float))[0, 1] ** 2
        print(f"    lead SNP {L}: causal variant {c0} is {abs(L-c0)} SNPs away, "
              f"r^2 = {r2:.3f}, p_lead = {p_final[L]:.2e}, "
              f"p_causal = {p_final[c0]:.2e}")
print("\n  THE LEAD SNP IS OFTEN NOT THE CAUSAL VARIANT. It is the one that")
print("  happened to be typed and happened to tag the causal haplotype best in")
print("  this sample. Fine-mapping computes posterior inclusion probabilities")
print("  and a 95% credible set rather than naming the top SNP.")

# %% [markdown]
# ## 6. Polygenic scores and leakage, eq. (24.8)

# %%
header("6. Polygenic scores: the leakage trap (24.8)")

from sklearn.model_selection import train_test_split

def build_pgs(G_train, y_train, G_test, keep, covars_train=None,
              covars_test=None, p_thresh=1e-3, use_all_for_weights=None):
    """Clumping + thresholding PGS, eq. (24.8)."""
    if use_all_for_weights is not None:
        Gw, yw, cw = use_all_for_weights
    else:
        Gw, yw, cw = G_train, y_train, covars_train
    p_w, b_w = scan(Gw, yw, covars=cw, keep=keep)
    sel = np.where(np.isfinite(p_w) & (p_w < p_thresh))[0]
    if len(sel) == 0:
        return np.zeros(len(G_test)), sel
    return G_test[:, sel].astype(float) @ b_w[sel], sel


idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.35,
                                  random_state=0, stratify=y)
from sklearn.metrics import roc_auc_score

# (a) HONEST: weights from the training split only.
s_test, sel_h = build_pgs(G[idx_tr], y[idx_tr], G[idx_te], keep,
                          covars_train=pcs[idx_tr], covars_test=pcs[idx_te])
auc_honest = roc_auc_score(y[idx_te], s_test)

# (b) LEAKY: weights estimated on ALL samples, then "validated" on the test set.
s_leak, sel_l = build_pgs(G[idx_tr], y[idx_tr], G[idx_te], keep,
                          use_all_for_weights=(G, y, pcs))
auc_leak = roc_auc_score(y[idx_te], s_leak)

print(f"  weights from TRAINING only : {len(sel_h):>4} SNPs, test AUC = "
      f"{auc_honest:.3f}")
print(f"  weights from ALL samples   : {len(sel_l):>4} SNPs, test AUC = "
      f"{auc_leak:.3f}   <- inflated")
print(f"  optimism from the leak     : {auc_leak - auc_honest:+.3f}")

# Portability: does a score trained in one ancestry work in the other?
tr_pop0 = np.where(pop == 0)[0]
te_pop0 = tr_pop0[:len(tr_pop0)//3]; tr_pop0 = tr_pop0[len(tr_pop0)//3:]
te_pop1 = np.where(pop == 1)[0]
s0, _ = build_pgs(G[tr_pop0], y[tr_pop0], G[te_pop0], keep)
s1, _ = build_pgs(G[tr_pop0], y[tr_pop0], G[te_pop1], keep)
auc_same = roc_auc_score(y[te_pop0], s0)
auc_other = roc_auc_score(y[te_pop1], s1)
print(f"\n  Portability (trained entirely in population 0):")
print(f"    tested in population 0 : AUC = {auc_same:.3f}")
print(f"    tested in population 1 : AUC = {auc_other:.3f}")
print(f"    relative loss          : {1 - (auc_other-0.5)/max(auc_same-0.5,1e-9):.0%}")
print("  Real PGS lose 50-80% of their predictive R^2 across ancestries,")
print("  because LD patterns and allele frequencies differ. ALWAYS report the")
print("  ancestry of both the training and the test data.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
for label, pv, c in [("uncorrected", p_naive, "C3"),
                     ("10 ancestry PCs", p_final, "C0")]:
    q = pv[np.isfinite(pv) & (pv > 0)]
    obs = -np.log10(np.sort(q))
    exp = -np.log10(np.arange(1, len(obs)+1) / (len(obs)+1))
    ax.plot(exp, obs, ".", ms=2, color=c,
            label=f"{label} ($\\lambda$={lambda_gc(pv):.2f})")
ax.plot([0, exp.max()], [0, exp.max()], "k--", lw=1)
ax.set_xlabel("expected $-\\log_{10}p$"); ax.set_ylabel("observed")
ax.set_title("Eq. (24.4): QQ plot", fontsize=9)
ax.legend(fontsize=7)

ax = axes[0, 1]
ok = np.isfinite(p_final)
ax.scatter(np.arange(n_snps)[ok], -np.log10(p_final[ok]), s=2, alpha=0.4,
           c=["C3" if j in set(causal.tolist()) else "grey"
              for j in np.arange(n_snps)[ok]])
ax.axhline(-np.log10(ALPHA_GW), color="red", ls="--", label="5e-8")
ax.set_xlabel("SNP index"); ax.set_ylabel("$-\\log_{10}p$")
ax.set_title("Manhattan plot (PC-adjusted)", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 0]
for pcode, c in [(0, "C0"), (1, "C1")]:
    ax.scatter(pcs[pop == pcode, 0], pcs[pop == pcode, 1], s=5, alpha=0.5,
               color=c, label=f"population {pcode}")
ax.set_xlabel(f"PC1 ({pve[0]:.1%})"); ax.set_ylabel(f"PC2 ({pve[1]:.1%})")
ax.set_title("Eq. (24.3): ancestry PCs", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
if len(lead_snps):
    L = lead_snps[0]
    lo, hi = max(0, L-60), min(n_snps, L+60)
    r2s = [np.corrcoef(G[:, L].astype(float), G[:, k].astype(float))[0, 1]**2
           if G[:, k].std() > 0 else 0 for k in range(lo, hi)]
    ax.scatter(range(lo, hi), -np.log10(np.clip(p_final[lo:hi], 1e-300, 1)),
               c=r2s, cmap="viridis", s=18)
    ax.axvline(L, color="red", ls="--", label="lead SNP")
    for c0 in causal:
        if lo <= c0 < hi:
            ax.axvline(c0, color="k", ls=":", label="causal")
    ax.set_xlabel("SNP index"); ax.set_ylabel("$-\\log_{10}p$")
    ax.set_title("Eq. (24.6): LD around the lead SNP", fontsize=9)
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "gwas.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/gwas.png")

# %% [markdown]
# # PROBLEMS

# %% [markdown]
# ### Problem 1: Why 5e-8?
#
# Estimate the number of *effectively independent* tests in this genotype
# matrix (e.g. by counting the eigenvalues of the SNP correlation matrix needed
# to explain 99.5% of the variance) and derive the appropriate Bonferroni
# threshold. Compare it to `0.05 / n_snps`.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# idx = np.where(keep)[0][:1500]
# Z = G[:, idx].astype(float)
# Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
# ev = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
# ev = np.clip(ev, 0, None)
# m_eff = int(np.searchsorted(np.cumsum(ev)/ev.sum(), 0.995) + 1)
# print(f"  SNPs tested                  : {len(idx)}")
# print(f"  effectively independent tests: {m_eff}")
# print(f"  naive Bonferroni 0.05/M      : {0.05/len(idx):.2e}")
# print(f"  LD-aware  0.05/M_eff         : {0.05/m_eff:.2e}")
# print(f"  ratio                        : {len(idx)/m_eff:.2f}x too strict")
#
# # LD makes neighbouring tests redundant, so a naive Bonferroni over every SNP
# # is CONSERVATIVE. The genome-wide 5e-8 is calibrated to ~1e6 effectively
# # independent common variants, NOT to the number of SNPs on the array - which
# # is why denser arrays do not require a stricter threshold, while whole-genome
# # sequencing (which adds genuinely independent rare variants) does.

# %% [markdown]
# ### Problem 2: Case/control ancestry imbalance
#
# Re-simulate with `ancestry_effect=0` (ancestry no longer affects disease) but
# sample cases preferentially from population 1. Show that $\lambda_{GC}$ still
# inflates, and that PCs still fix it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# G2, y2, pop2, causal2, _ = simulate_gwas(ancestry_effect=0.0, seed=777)
# rng2 = np.random.default_rng(5)
# # Deliberately over-sample population 1 among cases (a recruitment artefact).
# want = np.where((y2 == 1) & (pop2 == 1))[0]
# drop = np.where((y2 == 1) & (pop2 == 0))[0]
# drop = rng2.choice(drop, size=int(0.7*len(drop)), replace=False)
# sel = np.setdiff1d(np.arange(len(y2)), drop)
# G2s, y2s, pop2s = G2[sel], y2[sel], pop2[sel]
# maf2 = np.minimum(G2s.mean(0)/2, 1 - G2s.mean(0)/2)
# keep2 = maf2 > 0.01
# p2, _ = scan(G2s, y2s, keep=keep2)
# pcs2, _ = genotype_pcs(G2s, keep2)
# p2adj, _ = scan(G2s, y2s, covars=pcs2[:, :10], keep=keep2)
# print(f"  case fraction from pop1: {pop2s[y2s==1].mean():.2f} vs "
#       f"control {pop2s[y2s==0].mean():.2f}")
# print(f"  lambda_GC uncorrected : {lambda_gc(p2):.3f}")
# print(f"  lambda_GC with 10 PCs : {lambda_gc(p2adj):.3f}")
# print(f"  genome-wide hits: {int(np.nansum(p2 < 5e-8))} -> "
#       f"{int(np.nansum(p2adj < 5e-8))}")
#
# # Ancestry does not have to CAUSE the disease. It only has to be associated
# # with case/control STATUS - here through recruitment - to confound every SNP
# # whose frequency differs between populations. This is selection, i.e. the
# # collider/selection structure of Module 32.

# %% [markdown]
# ### Problem 3: Mendelian randomisation and the F-statistic
#
# Using the causal SNPs as instruments for a simulated exposure, compute the
# IVW estimate (eq. 24.9). Then add a *weak* instrument (a SNP with no real
# effect) and observe what happens to the estimate and the F-statistic.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# rng3 = np.random.default_rng(31)
# TRUE_CAUSAL = 0.6
# Xexp = G[:, causal].astype(float) @ np.full(len(causal), 0.5) \
#        + rng3.normal(0, 1, n)
# U = rng3.normal(0, 1, n)                                # confounder
# Yout = TRUE_CAUSAL * Xexp + 1.2 * U + rng3.normal(0, 1, n)
# Xexp = Xexp + 1.2 * U                                   # confound the exposure too
#
# def ivw(instruments):
#     bx, se_x, by, se_y = [], [], [], []
#     for j in instruments:
#         g = G[:, j].astype(float)
#         fx = sm.OLS(Xexp, sm.add_constant(g)).fit()
#         fy = sm.OLS(Yout, sm.add_constant(g)).fit()
#         bx.append(fx.params[1]); se_x.append(fx.bse[1])
#         by.append(fy.params[1]); se_y.append(fy.bse[1])
#     bx, se_x, by, se_y = map(np.array, (bx, se_x, by, se_y))
#     est = np.sum(bx*by/se_y**2) / np.sum(bx**2/se_y**2)   # eq. (24.9)
#     F = np.mean((bx/se_x)**2)                             # instrument strength
#     return est, F
#
# est_strong, F_strong = ivw(list(causal))
# weak = [j for j in range(n_snps) if j not in set(causal.tolist())][:8]
# est_weak, F_weak = ivw(list(causal) + weak)
# ols_conf = sm.OLS(Yout, sm.add_constant(Xexp)).fit().params[1]
# print(f"  TRUE causal effect            = {TRUE_CAUSAL:.3f}")
# print(f"  naive OLS (confounded)        = {ols_conf:.3f}")
# print(f"  IVW, strong instruments only  = {est_strong:.3f}  (mean F = {F_strong:.1f})")
# print(f"  IVW, plus 8 weak instruments  = {est_weak:.3f}  (mean F = {F_weak:.1f})")
#
# # The relevance assumption is checkable: mean F > 10 is the usual rule. Weak
# # instruments bias the IVW estimate TOWARD the confounded OLS value. The other
# # two MR assumptions - independence and exclusion restriction (no pathway from
# # the SNP to the outcome except through the exposure) - are NOT testable.
# # MR-Egger and weighted-median estimators are sensitivity analyses for
# # horizontal pleiotropy, not proofs that it is absent.

# %% [markdown]
# ## What to take away
#
# 1. Do QC first, and test HWE in **controls only**.
# 2. Control structure with PCs or an LMM; verify with $\lambda_{GC}$ and a QQ
#    plot: but remember polygenicity also inflates $\lambda$.
# 3. Clump before counting "independent loci"; fine-map before naming a gene.
# 4. Never evaluate a PGS in samples that contributed to its weights, and
#    report ancestry transferability.
#
# **Next:** `25_proteomics_missing_values.py`
