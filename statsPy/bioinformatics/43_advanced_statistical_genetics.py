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
# # Applied 43: Fine-mapping, colocalisation, and heritability
#
# **Curriculum link:** `stats.md` -> Topic 43, equations (43.1)-(43.8)
# **Core modules used:** 08, 32, 33, 24
#
# ## Dataset card
#
# | | |
# |---|---|
# | **Real analogue** | A GWAS locus taken forward: fine-mapping, eQTL colocalisation, LDSC, MR |
# | **Key threats** | LD, the single-causal assumption, confounding vs polygenicity, invalid instruments, winner's curse |
# | **Here** | Simulated genotypes with realistic LD and a known causal variant |
#
# ## What module 34 left unfinished
#
# Module 24 found associated loci and stopped. This module does the four things
# that come next, and each one rests on an assumption worth stating out loud.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "43_advanced_statistical_genetics"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(43)

# %% [markdown]
# ## 1. A locus with LD and one causal variant

# %%
header("1. Simulating a locus with realistic linkage disequilibrium")


def simulate_locus(n=6000, n_snp=60, causal=25, beta=0.16, rho=0.99, seed=0):
    """An AR(1) LD structure: correlation decays with distance along the locus."""
    r = np.random.default_rng(seed)
    idx = np.arange(n_snp)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n_snp))
    liab = r.normal(size=(n, n_snp)) @ L.T
    maf = r.uniform(0.15, 0.45, n_snp)
    thr = st.norm.ppf(1 - maf)
    G = (liab > thr).astype(float) + (liab > st.norm.ppf(1 - maf / 3)).astype(float)
    y = beta * G[:, causal] + r.normal(0, 1, n)
    return G, y, causal, Sigma


G, y, CAUSAL, Sigma = simulate_locus(seed=1)
n, n_snp = G.shape


def scan(G, y):
    b, s, p = [], [], []
    for j in range(G.shape[1]):
        f = sm.OLS(y, sm.add_constant(G[:, j])).fit()
        b.append(f.params[1]); s.append(f.bse[1]); p.append(f.pvalues[1])
    return np.array(b), np.array(s), np.array(p)


beta_hat, se_hat, p_hat = scan(G, y)
lead = int(np.argmin(p_hat))
print(f"  {n} individuals, {n_snp} SNPs, causal variant at index {CAUSAL}")
print(f"  lead SNP (smallest p) is index {lead}, p = {p_hat[lead]:.2e}")
print(f"  correlation between lead and causal: r = {Sigma[lead, CAUSAL]:.3f}")
print(f"  SNPs with p < 5e-8: {np.sum(p_hat < 5e-8)}")
print("""
  The whole LD block clears genome-wide significance, so the locus is
  unambiguous but the VARIANT is not. Whether the lead SNP happens to be the
  causal one is partly luck, which a single dataset cannot show you. Section 2
  repeats the experiment to measure it.""")

# %% [markdown]
# ## 2. Fine-mapping, eq. (43.1)-(43.3)

# %%
header("2. Approximate Bayes factors and credible sets (43.1)-(43.3)")


def abf(beta, se, W=0.04):
    """Wakefield's ABF, eq. (43.1). W is the prior variance of the effect."""
    z2 = (beta / se) ** 2
    r_ = W / (se ** 2 + W)
    return np.sqrt(1 - r_) * np.exp(z2 * r_ / 2)


def credible_set(pip, level=0.95):
    """Eq. (43.3): smallest set whose PIPs sum to `level`."""
    o = np.argsort(-pip)
    c = np.cumsum(pip[o])
    k = int(np.searchsorted(c, level) + 1)
    return o[:k]


bf = abf(beta_hat, se_hat)
pip = bf / bf.sum()                                   # eq. (43.2)
cs = credible_set(pip)
print(f"  {'SNP':<7}{'p':>11}{'PIP (43.2)':>13}{'in 95% set':>13}")
for j in np.argsort(-pip)[:6]:
    print(f"  {j:<7}{p_hat[j]:>11.2e}{pip[j]:>13.3f}"
          f"{('yes' if j in cs else 'no'):>13}")
print(f"\n  95% credible set: {len(cs)} variants, "
      f"{'contains' if CAUSAL in cs else 'MISSES'} the causal variant")
print(f"  PIP of the true causal variant: {pip[CAUSAL]:.3f}")
print(f"  PIP of the lead SNP: {pip[lead]:.3f}")

print("""
  Now repeat the whole experiment 20 times, which is the only way to see how
  often the lead SNP is the right one:
""")
sizes, pips, hits, leadok = [], [], 0, 0
for i in range(20):
    Gi, yi, ci, _ = simulate_locus(seed=400 + i)
    b_, s_, p_ = scan(Gi, yi)
    pp = abf(b_, s_); pp = pp / pp.sum()
    cs_ = credible_set(pp)
    sizes.append(len(cs_)); pips.append(pp[ci])
    hits += ci in cs_
    leadok += int(np.argmin(p_)) == ci
print(f"  {'mean 95% credible set size':<40}{np.mean(sizes):>12.1f}")
print(f"  {'set contains the causal variant':<40}{hits/20:>12.0%}")
print(f"  {'LEAD SNP is the causal variant':<40}{leadok/20:>12.0%}")
print(f"  {'mean PIP of the causal variant':<40}{np.mean(pips):>12.3f}")
print("""
  The set nearly always contains the causal variant, and the lead SNP is it
  only some of the time. Reporting a short set is therefore honest in a way
  that naming the top hit is not.""")

# %% [markdown]
# ### 2b. The harder, and more common, case: the causal variant is untyped
#
# Arrays genotype a subset of variants, so the true causal one is frequently
# absent and represented only by proxies in LD with it.

# %%
header("2b. When the causal variant is not on the array")

sizes2 = []
for i in range(20):
    Gi, yi, ci, _ = simulate_locus(seed=400 + i)
    keep = np.setdiff1d(np.arange(Gi.shape[1]), [ci])    # drop the causal SNP
    b_, s_, p_ = scan(Gi[:, keep], yi)
    pp = abf(b_, s_); pp = pp / pp.sum()
    sizes2.append(len(credible_set(pp)))
print(f"  {'mean set size, causal variant TYPED':<40}{np.mean(sizes):>12.1f}")
print(f"  {'mean set size, causal variant UNTYPED':<40}{np.mean(sizes2):>12.1f}")
print("""
  With the causal variant absent, the credible set roughly doubles and its
  posterior probability is spread over proxies, none of which is the answer.
  The method cannot tell you this has happened: it reports a set with the
  usual 95% label, and every variant in it is innocent.

  That is the honest reading of a credible set. It is a statement about which
  TYPED variants are compatible with the data, not a guarantee that the causal
  variant is among them.

  Note the assumption behind all of it. Eqs. (43.1)-(43.3) assume EXACTLY ONE
  causal variant in the locus. If two variants act independently, the
  single-causal PIPs are wrong, and methods such as SuSiE that allow several
  are needed. Problem 2 breaks exactly this assumption.""")

# %% [markdown]
# ## 3. Colocalisation, eq. (43.4)

# %%
header("3. Do two traits share a causal variant? (43.4)")


def coloc(b1, s1, b2, s2, p1=1e-4, p2=1e-4, p12=1e-5):
    """Posterior over the five hypotheses, eq. (43.4)."""
    a1, a2 = abf(b1, s1), abf(b2, s2)
    H1 = p1 * np.sum(a1)
    H2 = p2 * np.sum(a2)
    H3 = p1 * p2 * (np.sum(a1) * np.sum(a2) - np.sum(a1 * a2))
    H4 = p12 * np.sum(a1 * a2)
    H0 = 1.0
    tot = H0 + H1 + H2 + H3 + H4
    return np.array([H0, H1, H2, H3, H4]) / tot


# Trait 2 shares the causal variant.
y2_shared = 0.14 * G[:, CAUSAL] + rng.normal(0, 1, n)
# Trait 3 has a DIFFERENT causal variant in the same LD block.
other = CAUSAL + 12
y3_distinct = 0.14 * G[:, other] + rng.normal(0, 1, n)

for lab, y_ in (("shares the causal variant", y2_shared),
                ("different variant, same block", y3_distinct)):
    b2, s2, _ = scan(G, y_)
    post = coloc(beta_hat, se_hat, b2, s2)
    print(f"  {lab}")
    print(f"    P(H0 none)={post[0]:.3f}  P(H1 trait1 only)={post[1]:.3f}  "
          f"P(H2 trait2 only)={post[2]:.3f}")
    print(f"    P(H3 distinct variants)={post[3]:.3f}  "
          f"P(H4 SHARED variant)={post[4]:.3f}\n")

print("""  H4 is high when the traits genuinely share a variant and H3 takes over
  when they do not, which is the intended behaviour.

  The practical hazard is that strong LD makes H3 and H4 hard to separate, and
  a modest posterior for H3 is frequently reported as if it supported sharing.
  Read all five numbers, and treat colocalisation as evidence about a LOCUS,
  not proof of a mechanism.""")

# %% [markdown]
# ## 4. Polygenicity or confounding? LDSC, eq. (43.5)-(43.6)

# %%
header("4. LD score regression separates the two causes of inflation (43.5)")


# M is the number of causal variants GENOME-WIDE, not the number we simulate.
# Using the simulated count here is a common slip and inflates chi-squares
# by orders of magnitude.
M_GENOME = 1_000_000


def ldsc_sim(n=200_000, n_snp=4000, h2=0.35, confound=0.0, seed=0):
    """Chi-squares under polygenicity plus optional stratification."""
    r = np.random.default_rng(seed)
    # LD scores: how many variants each SNP tags.
    ld = r.gamma(3.0, 1.5, n_snp) + 1.0
    # Eq. (43.5): E[chi2] = N h2 / M * ld + 1 + N a
    expected = n * h2 / M_GENOME * ld + 1.0 + confound
    chi2 = expected * r.chisquare(1, n_snp)
    return chi2, ld


print("  averaged over 30 replicate GWAS, N = 200,000\n")
print(f"  {'scenario':<34}{'lambda_GC':>11}{'LDSC slope':>12}"
      f"{'h2 (43.6)':>11}{'intercept':>11}")
N = 200_000
for lab, h2, cf in (("no signal, no confounding", 0.0, 0.0),
                    ("polygenic, no confounding", 0.35, 0.0),
                    ("no signal, stratification", 0.0, 0.35),
                    ("polygenic AND stratified", 0.35, 0.35)):
    acc = []
    for i in range(30):
        chi2, ld = ldsc_sim(h2=h2, confound=cf, seed=3000 + i)
        lam = np.median(chi2) / st.chi2.ppf(0.5, 1)
        f = sm.OLS(chi2, sm.add_constant(ld)).fit()
        acc.append([lam, f.params[1], f.params[1] * M_GENOME / N, f.params[0]])
    m = np.mean(acc, 0)                                       # eq. (43.6)
    print(f"  {lab:<34}{m[0]:>11.3f}{m[1]:>12.5f}{m[2]:>11.3f}{m[3]:>11.3f}")

print("""
  true h2 is 0.35 in the polygenic rows; the true intercept is 1.00 without
  stratification and 1.35 with it.""")
print("""
  Read the lambda column first: it is elevated in BOTH the polygenic row and
  the stratified row, and cannot tell them apart. That is the limitation
  Module 24 flagged and could not resolve.

  The LDSC columns resolve it. The SLOPE responds to heritability, because
  real polygenic signal accumulates in proportion to how many variants a SNP
  tags. The INTERCEPT responds to confounding, which inflates every SNP
  equally regardless of its LD score.

  So an intercept near 1 with a large slope is a genuinely polygenic trait; an
  intercept well above 1 is a warning about population structure or cryptic
  relatedness, whatever lambda says.""")

# %% [markdown]
# ## 5. Mendelian randomisation with invalid instruments, eq. (43.7)

# %%
header("5. IVW, Egger, weighted median and mode (43.7)")


def mr_estimators(bx, sx, by, sy):
    w = 1.0 / sy ** 2
    ivw = np.sum(w * bx * by) / np.sum(w * bx ** 2)                # eq. (24.9)
    # MR-Egger: weighted regression of by on bx WITH an intercept.
    X = sm.add_constant(bx)
    eg = sm.WLS(by, X, weights=w).fit()
    # Weighted median of the per-instrument ratio estimates.
    ratio = by / bx
    rw = (bx ** 2) / sy ** 2
    o = np.argsort(ratio)
    cw = np.cumsum(rw[o]) / np.sum(rw)
    med = ratio[o][np.searchsorted(cw, 0.5)]
    # Mode-based: the densest cluster of ratio estimates.
    grid = np.linspace(ratio.min(), ratio.max(), 512)
    dens = np.sum(np.exp(-0.5 * ((grid[:, None] - ratio[None, :]) / 0.08) ** 2), axis=1)
    mode = grid[int(np.argmax(dens))]
    return ivw, eg.params[1], eg.params[0], med, mode


TRUE_CAUSAL = 0.30
print(f"  true causal effect = {TRUE_CAUSAL}\n")
print(f"  {'instruments':<32}{'IVW':>9}{'Egger':>9}{'Egger int':>11}"
      f"{'W median':>10}{'mode':>9}")
for lab, n_bad, pleio_dir in (("all valid", 0, 0.0),
                              ("30% invalid, balanced", 9, 0.0),
                              ("30% invalid, directional", 9, 0.18)):
    r = np.random.default_rng(64)
    K = 30
    bx = r.uniform(0.10, 0.28, K)                 # instrument-exposure effects
    sx = np.full(K, 0.012)
    pleio = np.zeros(K)
    if n_bad:
        who = r.choice(K, n_bad, replace=False)
        pleio[who] = (r.normal(pleio_dir, 0.12, n_bad) if pleio_dir
                      else r.normal(0, 0.12, n_bad))
    by = TRUE_CAUSAL * bx + pleio + r.normal(0, 0.012, K)
    sy = np.full(K, 0.012)
    ivw, eg_s, eg_i, med, mode = mr_estimators(bx, sx, by, sy)
    print(f"  {lab:<32}{ivw:>9.3f}{eg_s:>9.3f}{eg_i:>11.3f}{med:>10.3f}{mode:>9.3f}")

print("""
  With valid instruments everything agrees, which is the case where the choice
  does not matter.

  With BALANCED pleiotropy, IVW stays roughly unbiased because the invalid
  effects cancel. With DIRECTIONAL pleiotropy it is badly biased, and the
  Egger intercept moves away from zero, which is precisely what it is for.

  The estimators are ordered by the strength of what they assume: IVW needs
  every instrument valid, the weighted median needs over half the weight
  valid, and the mode needs only that the valid ones form the largest cluster.
  Weaker assumptions cost precision, so report all of them. Agreement is the
  evidence; a single estimator is a choice you have not justified.""")

# %% [markdown]
# ## 6. Winner's curse, eq. (43.8)

# %%
header("6. Effects estimated in the data that selected them (43.8)")

r = np.random.default_rng(71)
M_SNP, N_DISC = 20000, 8000
true_b = np.zeros(M_SNP)
causal_idx = r.choice(M_SNP, 400, replace=False)
true_b[causal_idx] = r.normal(0, 0.035, 400)
se_d = 1 / np.sqrt(N_DISC)
b_disc = true_b + r.normal(0, se_d, M_SNP)
b_rep = true_b + r.normal(0, se_d, M_SNP)          # independent replication

for thresh_z in (4.0, 5.0, 5.45):
    sel = np.abs(b_disc / se_d) > thresh_z
    if sel.sum() < 5:
        continue
    print(f"  |z| > {thresh_z}: {sel.sum():>4} selected   "
          f"discovery |b| = {np.abs(b_disc[sel]).mean():.4f}   "
          f"replication |b| = {np.abs(b_rep[sel]).mean():.4f}   "
          f"true |b| = {np.abs(true_b[sel]).mean():.4f}")

print("""
  The replication estimate is consistently smaller than the discovery
  estimate, and the truth is smaller still. Nothing failed to replicate: the
  discovery estimate was inflated by the act of selection (43.8).

  Two consequences worth carrying. Replication studies powered on the
  discovery effect size are systematically underpowered. And polygenic scores
  built with discovery weights over-fit, which is why PRS weights must be
  shrunk or re-estimated in independent data.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].scatter(range(n_snp), -np.log10(p_hat), s=18, color="grey")
ax[0].scatter(cs, -np.log10(p_hat[cs]), s=30, color="steelblue", label="95% set")
ax[0].scatter([CAUSAL], [-np.log10(p_hat[CAUSAL])], s=60, color="firebrick",
              marker="*", label="causal")
ax[0].set_xlabel("SNP index"); ax[0].set_ylabel("-log10 p")
ax[0].set_title("Fine-mapping (43.2)-(43.3)"); ax[0].legend(fontsize=8)

chi2, ld = ldsc_sim(h2=0.35, confound=0.35, seed=9)
ax[1].scatter(ld, chi2, s=5, alpha=0.2, color="grey")
bins = np.quantile(ld, np.linspace(0, 1, 15))
mids = [(bins[i] + bins[i+1]) / 2 for i in range(14)]
means = [chi2[(ld >= bins[i]) & (ld < bins[i+1])].mean() for i in range(14)]
ax[1].plot(mids, means, "o-", color="firebrick")
ax[1].set_xlabel("LD score"); ax[1].set_ylabel(r"$\chi^2$")
ax[1].set_title("LDSC (43.5): slope = h2, intercept = confounding")

r = np.random.default_rng(64); K = 30
bx = r.uniform(0.10, 0.28, K); pleio = np.zeros(K)
who = r.choice(K, 9, replace=False); pleio[who] = r.normal(0.18, 0.12, 9)
by = TRUE_CAUSAL * bx + pleio + r.normal(0, 0.012, K)
ax[2].scatter(bx, by, s=25, color="steelblue")
ax[2].scatter(bx[who], by[who], s=25, color="firebrick", label="invalid")
xs = np.linspace(0, bx.max(), 10)
ax[2].plot(xs, TRUE_CAUSAL * xs, "k--", lw=1.5, label="truth")
ax[2].set_xlabel("SNP-exposure effect"); ax[2].set_ylabel("SNP-outcome effect")
ax[2].set_title("MR with directional pleiotropy"); ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "advanced_genetics.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'advanced_genetics.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: How big is a credible set, and when does it help?
#
# Vary the strength of LD and the sample size, and record the size of the 95%
# credible set and whether it contains the causal variant.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'LD rho':<9}{'n':>8}{'set size':>11}{'contains causal':>18}"
#       f"{'PIP of causal':>16}{'lead is causal':>17}")
# for rho in (0.70, 0.92, 0.98):
#     for n_ in (3000, 12000):
#         sizes, hits, pips, leadok = [], 0, [], 0
#         for i in range(12):
#             Gi, yi, ci, _ = simulate_locus(n=n_, rho=rho, seed=300 + i)
#             b_, s_, p_ = scan(Gi, yi)
#             pp = abf(b_, s_); pp = pp / pp.sum()
#             cs_ = credible_set(pp)
#             sizes.append(len(cs_)); hits += ci in cs_
#             pips.append(pp[ci]); leadok += int(np.argmin(p_)) == ci
#         print(f"  {rho:<9.2f}{n_:>8}{np.mean(sizes):>11.1f}{hits/12:>18.0%}"
#               f"{np.mean(pips):>16.3f}{leadok/12:>17.0%}")
#
# # Two levers, and they do different things. More SAMPLES sharpen the PIPs and
# # shrink the set, because the association signal is estimated more precisely.
# # Stronger LD widens the set, because more variants are statistically
# # indistinguishable from the causal one no matter how much data you have.
# #
# # The last column is the point of the exercise: the lead SNP is frequently
# # not the causal variant, and its being lead is partly luck. A credible set
# # reports that honestly. This is why fine-mapping results are given as sets
# # and why functional follow-up should target the set, not the top hit.

# %% [markdown]
# ### Problem 2: Break the single-causal-variant assumption
#
# Put TWO independent causal variants in the locus and see what the
# single-causal fine-mapping does.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# r2 = np.random.default_rng(88)
# idx = np.arange(60); Sg = 0.92 ** np.abs(idx[:, None] - idx[None, :])
# L2 = np.linalg.cholesky(Sg + 1e-8 * np.eye(60))
# for sep, lab in ((30, "two causals, far apart (low LD)"),
#                  (6, "two causals, close (high LD)")):
#     c1, c2 = 15, 15 + sep
#     hits1 = hits2 = 0; sizes = []
#     for i in range(12):
#         liab = r2.normal(size=(6000, 60)) @ L2.T
#         Gx = (liab > st.norm.ppf(0.7)).astype(float)
#         yx = 0.16 * Gx[:, c1] + 0.16 * Gx[:, c2] + r2.normal(0, 1, 6000)
#         b_, s_, p_ = scan(Gx, yx)
#         pp = abf(b_, s_); pp = pp / pp.sum()
#         cs_ = credible_set(pp)
#         sizes.append(len(cs_)); hits1 += c1 in cs_; hits2 += c2 in cs_
#     print(f"  {lab}")
#     print(f"    mean set size {np.mean(sizes):.1f}, contains causal 1 "
#           f"{hits1/12:.0%}, contains causal 2 {hits2/12:.0%}")
#
# # When the two causal variants are far apart in LD, the single-causal model
# # concentrates on one of them and the credible set frequently MISSES the
# # other entirely. The 95% claim is about a model with one causal variant, and
# # that model is false here, so the coverage claim does not apply.
# #
# # When they are close, the set tends to cover the region containing both, so
# # the failure is less visible but the PIPs are still not interpretable as
# # per-variant probabilities.
# #
# # This is why SuSiE and FINEMAP allow multiple credible sets. If you use a
# # single-causal method, say so, and check whether conditioning on the lead
# # SNP leaves a second independent signal.

# %% [markdown]
# ### Problem 3: Winner's curse and replication power
#
# Design a replication study powered on the discovery effect size, then
# measure how often it actually replicates.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# sel = np.abs(b_disc / se_d) > 5.0
# b_sel = np.abs(b_disc[sel]); b_true_sel = np.abs(true_b[sel])
# print(f"  {sel.sum()} variants selected at |z| > 5\n")
# print(f"  {'powered on':<26}{'assumed |b|':>13}{'n needed':>11}"
#       f"{'actual power':>15}")
# for lab, assumed in (("discovery estimate", b_sel.mean()),
#                      ("truth (unknowable)", b_true_sel.mean())):
#     # n for 80% power at alpha = 0.05, two-sided
#     n_need = int(((1.96 + 0.84) / assumed) ** 2)
#     se_r = 1 / np.sqrt(n_need)
#     # actual power against the TRUE effects of the selected variants
#     pw = np.mean(np.abs(b_true_sel) / se_r > 1.96)
#     print(f"  {lab:<26}{assumed:>13.4f}{n_need:>11,}{pw:>15.1%}")
#
# # Powering the replication on the discovery estimate produces a study that is
# # far short of 80% power, because the effect it was designed to detect is
# # larger than the effect that exists. Roughly half the "failures to
# # replicate" in such a design are failures of the design.
# #
# # Two practical responses. Shrink the discovery estimate before powering,
# # using a winner's-curse correction or simply the lower bound of its
# # confidence interval. And when a replication fails, compare the two
# # CONFIDENCE INTERVALS rather than the two p-values: overlapping intervals
# # with a smaller replication point estimate is the signature of selection,
# # not of a false original finding.

# %% [markdown]
# ## What to take away
#
# 1. The lead SNP is often not the causal one, and if the causal variant is
#    untyped it is not in the set at all. Report a **credible set** (43.3),
#    and state the single-causal assumption it rests on.
# 2. **Colocalisation** (43.4) has five hypotheses. Read all of them; LD makes
#    H3 and H4 hard to separate.
# 3. $\lambda_{GC}$ cannot distinguish polygenicity from confounding. The LDSC
#    **slope** gives heritability, the **intercept** gives confounding (43.5).
# 4. In MR, report IVW, Egger, weighted median and mode together. They assume
#    progressively less and cost progressively more precision (43.7).
# 5. **Winner's curse** (43.8) inflates discovery effects, underpowers
#    replications, and over-fits polygenic scores.
#
# **Next:** `44_power_and_design_for_omics.py`
