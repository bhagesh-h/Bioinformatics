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
# # Module 38: Meta-analysis and evidence synthesis
#
# **Curriculum link:** `stats.md` -> Topic 38, equations (38.1)-(38.7)
# **Assumes:** Modules 05, 14.
#
# ## What you will learn
#
# 1. Inverse-variance pooling (38.1), and when a single pooled number is the
#    wrong summary.
# 2. **Heterogeneity**: $Q$ (38.2), $I^2$ (38.3), $\tau^2$ (38.4), and why
#    $I^2$ is a proportion rather than an amount.
# 3. Random effects (38.5) change the **estimand**, they do not fix anything.
# 4. The **prediction interval** (38.6) is the honest summary under
#    heterogeneity.
# 5. Publication bias, and why random effects make it worse.
# 6. Why combining p-values (38.7) is not meta-analysis.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "38_meta_analysis"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(27)


def fixed_effect(est, se):
    """Eq. (38.1)."""
    w = 1.0 / se ** 2
    theta = np.sum(w * est) / np.sum(w)
    return theta, np.sqrt(1.0 / np.sum(w))


def heterogeneity(est, se):
    """Q (38.2), I^2 (38.3), tau^2 by DerSimonian-Laird (38.4)."""
    w = 1.0 / se ** 2
    theta_fe = np.sum(w * est) / np.sum(w)
    Q = np.sum(w * (est - theta_fe) ** 2)
    k = len(est)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / denom) if denom > 0 else 0.0
    p_q = st.chi2.sf(Q, k - 1)
    return Q, p_q, I2, tau2


def random_effects(est, se):
    """Eq. (38.5), plus the prediction interval (38.6)."""
    _, _, _, tau2 = heterogeneity(est, se)
    w = 1.0 / (se ** 2 + tau2)
    theta = np.sum(w * est) / np.sum(w)
    se_theta = np.sqrt(1.0 / np.sum(w))
    k = len(est)
    half = st.t.ppf(0.975, max(k - 2, 1)) * np.sqrt(tau2 + se_theta ** 2)
    return theta, se_theta, tau2, (theta - half, theta + half)


# %% [markdown]
# ## 1. Five studies, one question

# %%
header("1. Inverse-variance pooling (38.1)")

# Five studies of the same true effect, differing only in size.
TRUE = 0.35
n_i = np.array([40, 80, 150, 60, 900])
se = 2.0 / np.sqrt(n_i)
est = rng.normal(TRUE, se)

print(f"  true effect = {TRUE}\n")
print(f"  {'study':<9}{'n':>7}{'estimate':>11}{'SE':>8}{'weight %':>11}")
w = 1 / se ** 2
for i in range(len(est)):
    print(f"  {'S' + str(i+1):<9}{n_i[i]:>7}{est[i]:>11.3f}{se[i]:>8.3f}"
          f"{100*w[i]/w.sum():>11.1f}")

fe, fe_se = fixed_effect(est, se)
print(f"\n  fixed-effect pooled = {fe:.3f} (95% CI {fe-1.96*fe_se:.3f} to "
      f"{fe+1.96*fe_se:.3f})")
print(f"  simple unweighted mean = {est.mean():.3f}")
print("""
  Inverse-variance weighting is not a convention, it is the minimum-variance
  combination. Note the weight column: the largest study carries most of the
  answer, which is correct when every study estimates the same thing.

  That last clause is the entire question of this module.""")

# %% [markdown]
# ## 2. When the studies disagree, eq. (38.2)-(38.4)

# %%
header("2. Detecting heterogeneity (38.2)-(38.4)")


def make_studies(k=8, tau=0.0, base_n=120, seed=0):
    r = np.random.default_rng(seed)
    n = r.integers(base_n // 3, base_n * 3, k)
    s = 2.0 / np.sqrt(n)
    theta_i = r.normal(TRUE, tau, k)            # study-specific true effects
    return r.normal(theta_i, s), s


print("  averaged over 400 replicate meta-analyses of k = 10 studies each\n")
print(f"  {'true tau':<10}{'mean Q':>9}{'Q p<0.05':>10}{'mean I2':>10}"
      f"{'mean tau2-hat':>15}{'SD of tau2-hat':>16}{'true tau2':>11}")
for tau in (0.0, 0.10, 0.25, 0.50):
    acc = np.array([heterogeneity(*make_studies(k=10, tau=tau, seed=3300 + i))
                    for i in range(400)])
    print(f"  {tau:<10.2f}{acc[:, 0].mean():>9.2f}{(acc[:, 1] < 0.05).mean():>10.2f}"
          f"{acc[:, 2].mean():>10.2f}{acc[:, 3].mean():>15.4f}"
          f"{acc[:, 3].std():>16.4f}{tau ** 2:>11.4f}")

print("""
  On average tau2-hat lands close to the truth, so DerSimonian-Laird is
  roughly unbiased here. Now read the SD column: at k = 10 the spread of
  tau2-hat is about as large as tau2 itself. A SINGLE meta-analysis can easily
  report tau2-hat = 0 when tau2 is genuinely 0.01, or double the true value.

  So treat a reported tau2 as a rough indication, not a measurement, and never
  conclude "no heterogeneity" from one small tau2-hat.

  Two further cautions. Q has low power with few studies, so p > 0.05 for Q is not
  evidence of homogeneity when k is small. And I2 is a PROPORTION of total
  variability, not an amount: large studies have small within-study variance,
  so the same absolute heterogeneity produces a much larger I2. An I2 of 90%
  among huge studies can be practically irrelevant, and an I2 of 20% among
  tiny ones can hide a lot.""")

# %% [markdown]
# ## 3. Random effects change the question

# %%
header("3. Fixed and random effects answer different questions")

e_, s_ = make_studies(k=10, tau=0.30, seed=5)
fe, fe_se = fixed_effect(e_, s_)
re_, re_se, tau2, pred = random_effects(e_, s_)
_, _, I2, _ = heterogeneity(e_, s_)

print(f"  10 studies, real between-study SD = 0.30, I2 = {I2:.2f}\n")
print(f"  {'model':<26}{'estimate':>10}{'95% CI':>22}{'width':>9}")
print(f"  {'fixed effect (38.1)':<26}{fe:>10.3f}"
      f"{f'({fe-1.96*fe_se:.2f}, {fe+1.96*fe_se:.2f})':>22}{3.92*fe_se:>9.3f}")
print(f"  {'random effects (38.5)':<26}{re_:>10.3f}"
      f"{f'({re_-1.96*re_se:.2f}, {re_+1.96*re_se:.2f})':>22}{3.92*re_se:>9.3f}")
print(f"  {'prediction interval (38.6)':<26}{'':>10}"
      f"{f'({pred[0]:.2f}, {pred[1]:.2f})':>22}{pred[1]-pred[0]:>9.3f}")

print("""
  The fixed-effect interval is the narrowest and the most misleading: it
  answers "where is THE effect" in a world where there is no single effect.

  The random-effects interval covers the MEAN of the distribution of effects.
  It is wider, but it is still an interval for an average.

  The prediction interval covers where the NEXT study would land, and it is
  much wider than both. If someone asks "what should I expect if I run this
  experiment", that is the interval that answers them. Reporting only the
  random-effects mean under heterogeneity hides most of the uncertainty that
  matters for a decision.""")

# %% [markdown]
# ## 4. Random effects up-weight small studies

# %%
header("4. Why random effects are dangerous under publication bias")

w_fe = 1 / s_ ** 2
_, _, _, t2 = heterogeneity(e_, s_)
w_re = 1 / (s_ ** 2 + t2)
order = np.argsort(-s_)                   # smallest studies first
print(f"  {'study SE':<12}{'fixed-effect weight %':>24}{'random-effects weight %':>26}")
for i in order[:5]:
    print(f"  {s_[i]:<12.3f}{100*w_fe[i]/w_fe.sum():>24.1f}"
          f"{100*w_re[i]/w_re.sum():>26.1f}")

print("""
  Adding tau2 to every denominator compresses the weights toward equality, so
  small studies gain influence. That is correct if small studies are simply
  noisier draws from the same distribution.

  It is harmful if small studies are systematically different, which is
  exactly what publication bias produces: a small study is publishable only if
  it found a large effect. Random effects then give extra weight to precisely
  the studies most likely to be biased.""")

# Demonstrate it.
header("4b. Publication bias, measured")

def biased_meta(k=20, tau=0.2, publish_bias=True, seed=0):
    r = np.random.default_rng(seed)
    e_all, s_all = [], []
    tries = 0
    while len(e_all) < k and tries < 5000:
        tries += 1
        n = r.integers(25, 400)
        s = 2.0 / np.sqrt(n)
        v = r.normal(r.normal(TRUE, tau), s)
        if publish_bias:
            # Small studies only get published if they reach significance.
            if n < 300 and abs(v / s) < 1.96:
                continue
        e_all.append(v); s_all.append(s)
    return np.array(e_all), np.array(s_all)

print(f"  true effect = {TRUE}\n")
print(f"  {'literature':<34}{'fixed':>9}{'random':>9}{'Egger p':>10}")
for lab, bias in (("complete (no bias)", False), ("only significant small studies", True)):
    fes, res, eggers = [], [], []
    for i in range(150):
        e2, s2 = biased_meta(publish_bias=bias, seed=800 + i)
        fes.append(fixed_effect(e2, s2)[0])
        res.append(random_effects(e2, s2)[0])
        # Egger's test: regress the standard normal deviate on precision and
        # test the INTERCEPT. The slope estimates the effect; the intercept is
        # what measures funnel asymmetry. Testing the slope instead would just
        # detect a non-zero effect, which is not what we are asking.
        y_ = e2 / s2; x_ = 1.0 / s2
        X_ = np.column_stack([np.ones_like(x_), x_])
        coef, *_ = np.linalg.lstsq(X_, y_, rcond=None)
        resid = y_ - X_ @ coef
        dof = len(y_) - 2
        s2e = resid @ resid / dof
        se_int = np.sqrt(s2e * np.linalg.inv(X_.T @ X_)[0, 0])
        eggers.append(2 * st.t.sf(abs(coef[0] / se_int), dof))
    print(f"  {lab:<34}{np.mean(fes):>9.3f}{np.mean(res):>9.3f}"
          f"{np.mean(np.array(eggers) < 0.05):>10.1%}")

print("""
  Under publication bias both estimators are inflated, and the random-effects
  estimate is inflated MORE, because it leans on the small studies that were
  filtered. The Egger column reports how often the asymmetry test catches it.

  Always look at a funnel plot and test asymmetry before quoting a pooled
  effect, and be aware the test has poor power with few studies.""")

# %% [markdown]
# ## 5. Combining p-values is not meta-analysis, eq. (38.7)

# %%
header("5. Fisher's method answers a different question (38.7)")


def fisher(p):
    return st.chi2.sf(-2 * np.sum(np.log(np.maximum(p, 1e-300))), 2 * len(p))


scenarios = {
    "5 studies, all weakly positive": np.array([0.04, 0.08, 0.06, 0.09, 0.05]),
    "4 null studies, 1 very strong": np.array([0.8, 0.6, 0.45, 0.9, 1e-6]),
    "3 positive, 2 equally negative": np.array([0.01, 0.02, 0.03, 0.01, 0.02]),
}
print(f"  {'scenario':<34}{'Fisher p (38.7)':>18}")
for lab, pv in scenarios.items():
    print(f"  {lab:<34}{fisher(pv):>18.2e}")

print("""
  Every scenario is 'significant', including the one where a single study
  carries everything and the one where the studies could be pointing in
  opposite directions. Fisher's method tests the GLOBAL NULL that no study has
  any effect. Rejecting it tells you something happened somewhere.

  It gives no effect size, no direction and no interval, so it cannot answer
  "how big is it" or "should I expect this to work". Use it for screening
  (does this gene do anything in any of these datasets), never as a summary of
  evidence for an effect.""")

# %% [markdown]
# ## 6. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

e_f, s_f = make_studies(k=9, tau=0.25, seed=11)
fe_f, fese_f = fixed_effect(e_f, s_f)
re_f, rese_f, _, pred_f = random_effects(e_f, s_f)
ypos = np.arange(len(e_f))
ax[0].errorbar(e_f, ypos, xerr=1.96 * s_f, fmt="o", color="steelblue", ms=4)
ax[0].axvline(re_f, color="firebrick", lw=2)
ax[0].axvspan(pred_f[0], pred_f[1], color="firebrick", alpha=0.10)
ax[0].axvline(0, color="black", lw=1, ls=":")
ax[0].set_yticks(ypos); ax[0].set_yticklabels([f"S{i+1}" for i in ypos], fontsize=8)
ax[0].set_xlabel("effect"); ax[0].set_title("Forest plot, band = prediction interval")

e_b, s_b = biased_meta(k=30, publish_bias=True, seed=3)
ax[1].scatter(e_b, 1 / s_b, s=18, color="firebrick", label="biased literature")
e_c, s_c = biased_meta(k=30, publish_bias=False, seed=3)
ax[1].scatter(e_c, 1 / s_c, s=18, color="steelblue", alpha=0.6, label="complete")
ax[1].axvline(TRUE, color="black", lw=1)
ax[1].set_xlabel("effect"); ax[1].set_ylabel("precision (1/SE)")
ax[1].set_title("Funnel plot"); ax[1].legend(fontsize=8)

taus = np.linspace(0, 0.6, 20)
i2s = []
for t_ in taus:
    vals = [heterogeneity(*make_studies(k=12, tau=t_, seed=int(t_*1000)+j))[2]
            for j in range(20)]
    i2s.append(np.mean(vals))
ax[2].plot(taus, i2s, "o-", color="steelblue")
ax[2].set_xlabel(r"true between-study SD $\tau$"); ax[2].set_ylabel(r"$I^2$")
ax[2].set_title("Heterogeneity detected (38.3)")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "meta_analysis.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'meta_analysis.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Does the pooled interval cover the truth?
#
# Measure the coverage of the fixed-effect interval, the random-effects
# interval and the prediction interval as heterogeneity grows.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  coverage of the MEAN effect, and of a FUTURE study\n")
# print(f"  {'tau':<8}{'fixed CI':>11}{'random CI':>12}{'prediction':>13}"
#       f"{'PI covers next study':>23}")
# for tau in (0.0, 0.15, 0.30, 0.50):
#     cf = cr = cp = cn = 0; R = 1500
#     for i in range(R):
#         e2, s2 = make_studies(k=10, tau=tau, seed=4000 + i)
#         f_, fs_ = fixed_effect(e2, s2)
#         r_, rs_, _, pi_ = random_effects(e2, s2)
#         cf += abs(f_ - TRUE) < 1.96 * fs_
#         cr += abs(r_ - TRUE) < 1.96 * rs_
#         cp += pi_[0] <= TRUE <= pi_[1]
#         # a genuinely new study drawn from the same distribution
#         nxt = np.random.default_rng(9000 + i).normal(TRUE, tau) if tau > 0 else TRUE
#         cn += pi_[0] <= nxt <= pi_[1]
#     print(f"  {tau:<8.2f}{cf/R:>11.1%}{cr/R:>12.1%}{cp/R:>13.1%}{cn/R:>23.1%}")
#
# # At tau = 0 everything works. As heterogeneity grows the fixed-effect
# # interval collapses toward zero coverage of the mean, because it treats
# # between-study variation as if it did not exist. The random-effects interval
# # holds up much better.
# #
# # The last column is the one that matters for a reader. The prediction
# # interval keeps covering where a NEW study lands, which is the question
# # anyone planning an experiment is actually asking. A random-effects CI is
# # not designed to do that and should not be read as if it were.

# %% [markdown]
# ### Problem 2: How many studies before heterogeneity is detectable?
#
# $Q$ is a hypothesis test and shares every weakness of one. Find the power of
# the heterogeneity test as a function of the number of studies.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  power of the Q test to detect real heterogeneity\n")
# print(f"  {'k studies':<12}" + "".join(f"{f'tau={t}':>12}" for t in (0.0, 0.2, 0.4)))
# for k in (3, 5, 10, 20, 50):
#     row = ""
#     for tau in (0.0, 0.2, 0.4):
#         hits = np.mean([heterogeneity(*make_studies(k=k, tau=tau,
#                                                     seed=6000 + i))[1] < 0.05
#                         for i in range(600)])
#         row += f"{hits:>12.2f}"
#     print(f"  {k:<12}{row}")
#
# # The tau = 0 column is the false positive rate and sits near 0.05 throughout,
# # so the test is calibrated.
# #
# # The other columns show the problem: with 3 to 5 studies, which is the size
# # of most meta-analyses in biology, the test misses real heterogeneity most
# # of the time. "Q was not significant so we used fixed effects" is therefore
# # a weak argument, and with few studies it is close to no argument at all.
# #
# # The practical response is to decide between fixed and random effects on
# # grounds of what the studies ARE, different populations, protocols and
# # platforms, rather than on a test that cannot see the answer.

# %% [markdown]
# ### Problem 3: Meta-analysis across omics datasets
#
# Pool a gene's effect across five RNA-seq studies with different platforms
# and depths. Compare pooling effect sizes against pooling p-values.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# r = np.random.default_rng(77)
# G = 3000
# true_lfc = np.zeros(G); true_lfc[:200] = r.normal(0, 0.8, 200)
# study_se = [0.25, 0.30, 0.55, 0.22, 0.60]          # differing quality
# batch_shift = [0.0, 0.0, 0.35, 0.0, -0.30]         # two studies are biased
# E, S, P = [], [], []
# for se_s, sh in zip(study_se, batch_shift):
#     e = r.normal(true_lfc + sh, se_s, G)
#     E.append(e); S.append(np.full(G, se_s))
#     P.append(2 * st.norm.sf(np.abs(e / se_s)))
# E = np.array(E); S = np.array(S); P = np.array(P)
# is_de = true_lfc != 0
#
# def bh(p):
#     m = len(p); o = np.argsort(p); q = np.empty(m)
#     q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
#     return np.clip(q, 0, 1)
#
# rows = []
# fe_p = np.empty(G); re_p = np.empty(G); fi_p = np.empty(G)
# for g in range(G):
#     f_, fs_ = fixed_effect(E[:, g], S[:, g])
#     r_, rs_, _, _ = random_effects(E[:, g], S[:, g])
#     fe_p[g] = 2 * st.norm.sf(abs(f_ / fs_))
#     re_p[g] = 2 * st.norm.sf(abs(r_ / rs_))
#     fi_p[g] = fisher(P[:, g])
# print(f"  {'method':<32}{'discoveries':>13}{'FDP':>8}{'power':>8}")
# for lab, pv in (("fixed effect on effect sizes", fe_p),
#                 ("random effects on effect sizes", re_p),
#                 ("Fisher on p-values (38.7)", fi_p)):
#     rej = bh(pv) < 0.05
#     print(f"  {lab:<32}{rej.sum():>13}{(rej & ~is_de).sum()/max(rej.sum(),1):>8.3f}"
#           f"{(rej & is_de).sum()/is_de.sum():>8.3f}")
#
# # Two of the five studies carry a systematic shift, the kind an uncorrected
# # batch or platform difference produces. Fixed effects ignore that and report
# # a high false discovery proportion. Random effects absorb the shift into
# # tau2 and are far better calibrated, at some cost in power.
# #
# # Fisher's method is the worst of the three here, and for a reason worth
# # understanding: it combines EVIDENCE AGAINST THE NULL regardless of
# # DIRECTION, so a gene pushed up in two biased studies and down in the others
# # still accumulates a small combined p-value. Direction is information, and
# # p-value pooling throws it away.
# #
# # For omics meta-analysis, pool effect sizes with random effects, and check
# # for study-level shifts before pooling at all.

# %% [markdown]
# ## What to take away
#
# 1. Inverse-variance weighting (38.1) is optimal **if** every study estimates
#    the same thing. That is the assumption, not a detail.
# 2. Report $\tau^2$ and $I^2$, and remember $I^2$ is a proportion. $Q$ has
#    little power with few studies, so a non-significant $Q$ proves nothing.
# 3. Random effects change the estimand to the **mean of a distribution** and
#    up-weight small studies, which publication bias exploits.
# 4. Under heterogeneity the **prediction interval** (38.6) is the summary a
#    reader planning a study actually needs.
# 5. Combining p-values (38.7) tests the global null and discards direction
#    and magnitude.
#
# **Next:** `39_conformal_prediction.py`
