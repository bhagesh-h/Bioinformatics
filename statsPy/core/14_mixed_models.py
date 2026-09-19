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
# # Module 14: Mixed, multilevel, and repeated-measures models
#
# **Curriculum link:** `stats.md` -> Topic 14, equations (14.1)-(14.8)
#
# This module is the formal machinery that fixes Module 01's pseudoreplication.
#
# ## What you will learn
#
# 1. The marginal covariance $\mathbf V=\mathbf{ZGZ}^\top+\mathbf R$ (14.2): why
#    a mixed model produces correct standard errors automatically.
# 2. ICC (14.3) recovered as a variance-component ratio.
# 3. **Partial pooling / BLUPs (14.5)-(14.6)**: shrinkage is the same formula as
#    empirical Bayes (Module 23).
# 4. Why the boundary LRT for $\sigma_b^2=0$ is a $\chi^2$ **mixture**, not
#    $\chi^2_1$.
# 5. Mixed model vs pseudobulk aggregation: when they agree.
# 6. GEE (14.8) and when the sandwich estimator is trustworthy.

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

warnings.filterwarnings("ignore")   # mixed-model convergence chatter

MODULE_NAME = "14_mixed_models"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. A clustered dataset, and three ways to analyse it

# %%
header("1. Wrong, crude, and correct analyses of clustered data")

def simulate_clustered(n_donors=12, m_per=40, effect=0.6,
                       sigma_b=1.0, sigma_e=1.0, seed=0):
    """Two-arm study with donors nested in arms (stats.md eq. 1.4)."""
    rng = np.random.default_rng(seed)
    donor_id = np.repeat(np.arange(n_donors), m_per)
    arm = np.repeat(rng.permutation([0, 1] * (n_donors // 2)), m_per)
    b = np.repeat(rng.normal(0, sigma_b, n_donors), m_per)
    y = 5.0 + effect * arm + b + rng.normal(0, sigma_e, n_donors * m_per)
    return pd.DataFrame({"y": y, "arm": arm, "donor": donor_id})


d = simulate_clustered(seed=1401)
print(f"  {d['donor'].nunique()} donors x "
      f"{len(d)//d['donor'].nunique()} cells = {len(d)} observations")

# (a) WRONG: ordinary regression on every cell.
m_naive = smf.ols("y ~ arm", data=d).fit()
# (b) CRUDE but valid: aggregate to donor means (pseudobulk), then regress.
agg = d.groupby(["donor", "arm"], as_index=False)["y"].mean()
m_agg = smf.ols("y ~ arm", data=agg).fit()
# (c) CORRECT and efficient: mixed model with a donor random intercept.
m_lmm = smf.mixedlm("y ~ arm", data=d, groups=d["donor"]).fit(reml=True)

print(f"\n  {'analysis':<34}{'beta_arm':>10}{'SE':>10}{'p':>11}")
print(f"  {'OLS on all cells (WRONG)':<34}{m_naive.params['arm']:>10.4f}"
      f"{m_naive.bse['arm']:>10.4f}{m_naive.pvalues['arm']:>11.2e}")
print(f"  {'OLS on donor means (pseudobulk)':<34}{m_agg.params['arm']:>10.4f}"
      f"{m_agg.bse['arm']:>10.4f}{m_agg.pvalues['arm']:>11.4f}")
print(f"  {'mixed model (random intercept)':<34}{m_lmm.params['arm']:>10.4f}"
      f"{m_lmm.bse['arm']:>10.4f}{m_lmm.pvalues['arm']:>11.4f}")
print(f"\n  The naive SE is {m_agg.bse['arm']/m_naive.bse['arm']:.1f}x too small.")
print("  Pseudobulk and the mixed model agree closely - for a BALANCED design")
print("  they are near-equivalent, which is why pseudobulk is the recommended")
print("  default for single-cell differential state (stats.md Topic 21).")

# %% [markdown]
# ## 2. Variance components and ICC, eq. (14.2)-(14.3)

# %%
header("2. Variance components and the ICC (14.2)-(14.3)")

sigma_b2 = float(m_lmm.cov_re.iloc[0, 0])
sigma_e2 = float(m_lmm.scale)
icc = sigma_b2 / (sigma_b2 + sigma_e2)
print(f"  estimated sigma_b^2 (between donors) = {sigma_b2:.4f}  (truth 1.00)")
print(f"  estimated sigma_e^2 (within donor)   = {sigma_e2:.4f}  (truth 1.00)")
print(f"  ICC = sigma_b^2/(sigma_b^2+sigma_e^2) = {icc:.4f}  (eq. 14.3)")
m_per = len(d) // d["donor"].nunique()
print(f"\n  design effect 1+(m-1)*ICC with m={m_per}: "
      f"{1 + (m_per-1)*icc:.2f}  (eq. 1.8)")
print(f"  predicted SE inflation of the naive analysis: "
      f"{np.sqrt(1 + (m_per-1)*icc):.2f}x")
print(f"  observed                                    : "
      f"{m_agg.bse['arm']/m_naive.bse['arm']:.2f}x")
print("  The mixed model has RECOVERED the design effect from the data.")

# %% [markdown]
# ## 3. Partial pooling and BLUPs, eq. (14.5)-(14.6)
#
# $$\hat b_j=\underbrace{\frac{\sigma_b^2}{\sigma_b^2+\sigma_e^2/n_j}}_{\lambda_j}
#   \big(\bar y_j-\mathbf x_j^\top\hat{\boldsymbol\beta}\big)$$
#
# Small clusters are shrunk hard toward the population mean; large clusters are
# trusted. Compare this with eq. (33.3): it is the same formula.

# %%
header("3. Shrinkage: small clusters are trusted less (14.6)")

rng = np.random.default_rng(1402)
# Deliberately UNBALANCED: cluster sizes from 2 to 200.
sizes = [2, 3, 5, 10, 20, 50, 100, 200]
rows = []
true_b = rng.normal(0, 1.0, len(sizes))
for j, (nj, bj) in enumerate(zip(sizes, true_b)):
    for v in rng.normal(4.0 + bj, 1.0, nj):
        rows.append({"y": v, "donor": f"D{j}", "nj": nj, "true_b": bj})
du = pd.DataFrame(rows)

mu = smf.mixedlm("y ~ 1", data=du, groups=du["donor"]).fit(reml=True)
s_b2, s_e2 = float(mu.cov_re.iloc[0, 0]), float(mu.scale)
blups = mu.random_effects
grand = float(mu.params["Intercept"])

print(f"  sigma_b^2 = {s_b2:.4f},  sigma_e^2 = {s_e2:.4f}")
print(f"  {'donor':<8}{'n_j':>5}{'raw mean dev':>14}{'lambda_j (14.6)':>17}"
      f"{'BLUP':>9}{'true b_j':>10}")
for j, nj in enumerate(sizes):
    key = f"D{j}"
    raw = du.loc[du["donor"] == key, "y"].mean() - grand
    lam = s_b2 / (s_b2 + s_e2 / nj)                 # eq. (14.6)
    print(f"  {key:<8}{nj:>5}{raw:>14.4f}{lam:>17.4f}"
          f"{float(blups[key].iloc[0]):>9.4f}{true_b[j]:>10.4f}")

print("\n  For n_j=2 the shrinkage factor is small, so the BLUP is pulled far")
print("  toward 0 - the data barely constrain that donor. For n_j=200 lambda")
print("  is nearly 1 and the BLUP tracks the raw mean. This is exactly the")
print("  empirical-Bayes borrowing that limma applies across genes (Module 23).")

# %% [markdown]
# ## 4. The boundary problem: testing $\sigma_b^2=0$
#
# $H_0:\sigma_b^2=0$ puts the null on the **boundary** of the parameter space,
# so the LRT statistic is a 50:50 mixture $\tfrac12\chi^2_0+\tfrac12\chi^2_1$,
# not $\chi^2_1$. Using $\chi^2_1$ doubles the p-value (conservative).

# %%
header("4. LRT on the boundary is a chi2 mixture, not chi2_1")

rng = np.random.default_rng(1403)
stats_null = []
for rep in range(300):
    dn = simulate_clustered(n_donors=10, m_per=8, effect=0.0,
                            sigma_b=0.0, sigma_e=1.0, seed=5000 + rep)
    try:
        f_mm = smf.mixedlm("y ~ arm", data=dn, groups=dn["donor"]).fit(reml=False)
        f_ols = smf.ols("y ~ arm", data=dn).fit()
        stats_null.append(max(0.0, 2 * (f_mm.llf - f_ols.llf)))
    except Exception:
        pass
stats_null = np.array(stats_null)
print(f"  {len(stats_null)} simulations with sigma_b^2 TRULY zero")
print(f"  proportion of LRT statistics exactly ~0 : "
      f"{np.mean(stats_null < 1e-6):.3f}   (theory: 0.5)")
crit_wrong = st.chi2.ppf(0.95, 1)
crit_mix = st.chi2.ppf(0.90, 1)          # 95th pct of 0.5*chi2_0 + 0.5*chi2_1
print(f"\n  rejection rate using chi2_1 crit ({crit_wrong:.3f}) : "
      f"{np.mean(stats_null > crit_wrong):.3f}   <- conservative")
print(f"  rejection rate using mixture crit ({crit_mix:.3f}) : "
      f"{np.mean(stats_null > crit_mix):.3f}   <- calibrated at 0.05")

# %% [markdown]
# ## 5. Degrees of freedom matter when clusters are few
#
# Without a Satterthwaite/Kenward-Roger correction, a $z$-based GLMM p-value is
# **anticonservative with few clusters**. `statsmodels` `MixedLM` reports
# $z$-based p-values, so we check the damage.

# %%
header("5. z-based mixed-model p-values with few clusters")

def fpr_by_clusters(n_donors, m_per=20, n_sim=400, seed=0):
    """False-positive rate under a TRUE null, mixed model vs pseudobulk t-test."""
    mm_hits = agg_hits = ok = 0
    for rep in range(n_sim):
        dd = simulate_clustered(n_donors=n_donors, m_per=m_per, effect=0.0,
                                sigma_b=1.0, sigma_e=1.0, seed=seed + rep)
        try:
            f = smf.mixedlm("y ~ arm", data=dd, groups=dd["donor"]).fit(reml=True)
            mm_hits += f.pvalues["arm"] < 0.05
        except Exception:
            continue
        a = dd.groupby(["donor", "arm"], as_index=False)["y"].mean()
        p = st.ttest_ind(a.loc[a["arm"] == 0, "y"], a.loc[a["arm"] == 1, "y"],
                         equal_var=False).pvalue
        agg_hits += p < 0.05
        ok += 1
    return mm_hits / ok, agg_hits / ok, ok


print(f"  {'donors':>8}{'LMM (z-based) FPR':>20}{'pseudobulk t FPR':>19}")
for nd in [4, 6, 10, 20, 40]:
    a_, b_, ok = fpr_by_clusters(nd, seed=nd * 777)
    print(f"  {nd:>8}{a_:>19.1%}{b_:>19.1%}")

print("\n  With 4-6 donors the z-based LMM p-value is anticonservative, while the")
print("  pseudobulk t-test (which uses the right small-sample df) holds 5%.")
print("  In R, use lmerTest's Satterthwaite or Kenward-Roger df. In Python,")
print("  with few clusters, prefer the aggregate analysis - it is transparent,")
print("  auditable, and correctly calibrated.")

# %% [markdown]
# ## 6. Random slopes and repeated measures, eq. (28.1)-(28.2)

# %%
header("6. Random intercepts vs random slopes")

rng = np.random.default_rng(1404)
n_subj, n_visit = 24, 5
rows = []
for s in range(n_subj):
    arm = s % 2
    b0 = rng.normal(0, 1.2)                     # subject intercept
    b1 = rng.normal(0, 0.35)                    # subject SLOPE (heterogeneity)
    for v in range(n_visit):
        t = v
        y = (10 + b0) + (0.20 + 0.30 * arm + b1) * t + rng.normal(0, 0.5)
        rows.append({"y": y, "t": t, "arm": arm, "subject": f"S{s:02d}"})
dl = pd.DataFrame(rows)

m_ri = smf.mixedlm("y ~ t * arm", data=dl, groups=dl["subject"]).fit(reml=True)
m_rs = smf.mixedlm("y ~ t * arm", data=dl, groups=dl["subject"],
                   re_formula="~t").fit(reml=True)

print(f"  TRUE treatment x time interaction = 0.300")
print(f"  {'model':<28}{'beta(t:arm)':>13}{'SE':>9}{'p':>11}")
print(f"  {'random intercept only':<28}{m_ri.params['t:arm']:>13.4f}"
      f"{m_ri.bse['t:arm']:>9.4f}{m_ri.pvalues['t:arm']:>11.4f}")
print(f"  {'random intercept + slope':<28}{m_rs.params['t:arm']:>13.4f}"
      f"{m_rs.bse['t:arm']:>9.4f}{m_rs.pvalues['t:arm']:>11.4f}")
print(f"\n  estimated random-slope SD = "
      f"{np.sqrt(float(m_rs.cov_re.iloc[1,1])):.4f}  (truth 0.35)")
print("  Ignoring genuine slope heterogeneity understates the SE of the")
print("  treatment x time interaction - the estimand of a longitudinal trial.")

# %% [markdown]
# ## 7. GEE and the sandwich estimator, eq. (14.8)
#
# GEE estimates the **marginal** (population-average) effect and gets valid SEs
# from the sandwich even if the working correlation is wrong: **provided there
# are many clusters**.

# %%
header("7. GEE: robust, but only with many clusters (14.8)")

d_big = simulate_clustered(n_donors=60, m_per=15, effect=0.5, seed=1405)
gee_ind = sm.GEE.from_formula("y ~ arm", groups="donor", data=d_big,
                              cov_struct=sm.cov_struct.Independence()).fit()
gee_exch = sm.GEE.from_formula("y ~ arm", groups="donor", data=d_big,
                               cov_struct=sm.cov_struct.Exchangeable()).fit()
lmm_big = smf.mixedlm("y ~ arm", data=d_big, groups=d_big["donor"]).fit()

print(f"  {'method':<38}{'beta_arm':>10}{'SE':>10}")
print(f"  {'GEE, working correlation = independence':<38}"
      f"{gee_ind.params['arm']:>10.4f}{gee_ind.bse['arm']:>10.4f}")
print(f"  {'GEE, working correlation = exchangeable':<38}"
      f"{gee_exch.params['arm']:>10.4f}{gee_exch.bse['arm']:>10.4f}")
print(f"  {'mixed model':<38}{lmm_big.params['arm']:>10.4f}"
      f"{lmm_big.bse['arm']:>10.4f}")
print("\n  Both GEE SEs are similar despite different working correlations -")
print("  that is the sandwich robustness. But with 60 clusters. Standard")
print("  guidance is J >= 40; with 6 donors the sandwich is badly biased and")
print("  a mixed model (or pseudobulk) is safer.")

print("\n  Subject-specific vs population-average (eq. 14.7): for the IDENTITY")
print("  link they coincide, which is why this trap is invisible in linear")
print("  models. In a logistic GLMM, beta_marginal ~ beta_conditional /")
print("  sqrt(1 + 0.346 sigma_b^2):")
for s2 in [0.5, 1.0, 2.0, 4.0]:
    print(f"    sigma_b^2 = {s2:.1f} -> attenuation factor "
          f"{1/np.sqrt(1 + 0.346*s2):.3f}")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for don, grp in d.groupby("donor"):
    axes[0].scatter(grp["arm"] + rng.normal(0, 0.04, len(grp)), grp["y"],
                    s=4, alpha=0.4)
    axes[0].scatter([grp["arm"].iloc[0]], [grp["y"].mean()], s=70,
                    marker="_", color="k")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["control", "treated"])
axes[0].set_ylabel("y")
axes[0].set_title("Cells cluster within donors", fontsize=9)

lam = [s_b2 / (s_b2 + s_e2 / nj) for nj in sizes]
raws = [du.loc[du["donor"] == f"D{j}", "y"].mean() - grand
        for j in range(len(sizes))]
bl = [float(blups[f"D{j}"].iloc[0]) for j in range(len(sizes))]
axes[1].plot(sizes, raws, "o-", label="raw deviation")
axes[1].plot(sizes, bl, "s-", label="BLUP (shrunk)")
axes[1].plot(sizes, true_b, "^--", label="true $b_j$")
axes[1].set_xscale("log"); axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("cluster size $n_j$")
axes[1].set_title("Eq. (14.6): shrinkage depends on $n_j$", fontsize=9)
axes[1].legend(fontsize=7)

for s, grp in dl.groupby("subject"):
    axes[2].plot(grp["t"], grp["y"], "-", lw=0.8,
                 color="C0" if grp["arm"].iloc[0] == 0 else "C1", alpha=0.6)
axes[2].set_xlabel("visit"); axes[2].set_ylabel("y")
axes[2].set_title("Random slopes: subjects diverge", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "mixed_models.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/mixed_models.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 14)
#
# 1. Enumerate every level of clustering before modelling. Random effects come
#    from the **design**, not from model fit.
# 2. Fixed blocking effects when clusters are few and you only want to adjust;
#    random effects when clusters are many or you want to predict cluster values.
# 3. Use Satterthwaite/Kenward-Roger df; never trust $z$-based GLMM p-values
#    with few clusters (section 5).
# 4. When in doubt with a simple balanced design, **aggregate to the cluster
#    level**: transparent and nearly always valid.
#
# **Next:** `15_missing_data_and_censoring.py`
