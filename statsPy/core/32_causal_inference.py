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
# # Module 32: Causal inference for observational bioinformatics
#
# **Curriculum link:** `stats.md` -> Topic 32, equations (32.1)-(32.8)
#
# Adjusting for more covariates feels safer. **It is not.** Which covariates to
# adjust for is determined by the assumed causal structure, and some
# adjustments *create* bias.
#
# ## What you will learn
#
# 1. Confounder vs mediator vs collider: each simulated with known truth.
# 2. **Collider bias**, the counterintuitive one, and how sample selection
#    creates it.
# 3. The g-formula (32.2), propensity scores and IPTW (32.3)-(32.4).
# 4. Overlap and balance diagnostics (not p-values).
# 5. Doubly robust AIPW (32.5).
# 6. Mediation (32.6)-(32.7) and the E-value (32.8).

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

MODULE_NAME = "32_causal_inference"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. The three elementary structures
#
# | Structure | Path | Adjusting for M... |
# |---|---|---|
# | Chain (mediator) | $A\to M\to Y$ | **blocks** the indirect effect |
# | Fork (confounder) | $A\leftarrow C\to Y$ | **removes** confounding |
# | Collider | $A\to K\leftarrow Y$ | **creates** spurious association |

# %%
header("1. Confounder, mediator, collider: adjust or not?")

N = 40000

# --- FORK: C confounds A -> Y. TRUE direct effect of A on Y is 0.8 ----------
rng = np.random.default_rng(2201)
C = rng.normal(0, 1, N)
A_f = rng.binomial(1, 1 / (1 + np.exp(-(0.9 * C))))
Y_f = 0.8 * A_f + 1.2 * C + rng.normal(0, 1, N)
d_f = pd.DataFrame({"A": A_f, "Y": Y_f, "C": C})

# --- CHAIN: A -> M -> Y. TOTAL effect 0.5*1.0 = 0.5, DIRECT effect 0.3 ------
A_c = rng.binomial(1, 0.5, N)
M = 1.0 * A_c + rng.normal(0, 1, N)
Y_c = 0.3 * A_c + 0.5 * M + rng.normal(0, 1, N)
d_c = pd.DataFrame({"A": A_c, "Y": Y_c, "M": M})

# --- COLLIDER: A -> K <- Y. TRUE effect of A on Y is EXACTLY 0 -------------
A_k = rng.binomial(1, 0.5, N)
Y_k = rng.normal(0, 1, N)                     # independent of A by construction
K = 1.2 * A_k + 1.2 * Y_k + rng.normal(0, 1, N)
d_k = pd.DataFrame({"A": A_k, "Y": Y_k, "K": K})

print(f"  {'structure':<26}{'truth':>8}{'crude':>10}{'adjusted':>11}  verdict")
b_crude = smf.ols("Y ~ A", data=d_f).fit().params["A"]
b_adj = smf.ols("Y ~ A + C", data=d_f).fit().params["A"]
print(f"  {'FORK (C = confounder)':<26}{0.80:>8.2f}{b_crude:>10.3f}"
      f"{b_adj:>11.3f}  ADJUST")

b_crude = smf.ols("Y ~ A", data=d_c).fit().params["A"]
b_adj = smf.ols("Y ~ A + M", data=d_c).fit().params["A"]
print(f"  {'CHAIN (M = mediator)':<26}{0.80:>8.2f}{b_crude:>10.3f}"
      f"{b_adj:>11.3f}  DO NOT adjust for a total effect")
print(f"  {'   (total = 0.3+0.5*1.0 = 0.8; adjusting returns the DIRECT 0.3)':<70}")

b_crude = smf.ols("Y ~ A", data=d_k).fit().params["A"]
b_adj = smf.ols("Y ~ A + K", data=d_k).fit().params["A"]
print(f"  {'COLLIDER (K)':<26}{0.00:>8.2f}{b_crude:>10.3f}"
      f"{b_adj:>11.3f}  NEVER adjust")
print("\n  Adjusting for the collider MANUFACTURES a strong association where")
print("  the truth is exactly zero. 'Adjust for everything you measured' is")
print("  not caution - it is a way to generate false findings.")

# %% [markdown]
# ## 2. Collider bias via sample selection
#
# The bioinformatics version: filtering samples on a variable that both the
# exposure and the outcome influence: cell-quality filters, "responders only"
# cohorts, hospital-based sampling.

# %%
header("2. Selection on a collider (the bioinformatics version)")

rng = np.random.default_rng(2202)
N = 30000
# Gene expression and a phenotype, INDEPENDENT of each other.
expr = rng.normal(0, 1, N)
pheno = rng.normal(0, 1, N)
# Sample quality depends on BOTH (e.g. both raise RNA yield, so both raise the
# chance the sample passes QC).
quality = 0.9 * expr + 0.9 * pheno + rng.normal(0, 1, N)
passed_qc = quality > np.quantile(quality, 0.6)      # keep the top 40%

print(f"  TRUE correlation(expression, phenotype) = "
      f"{np.corrcoef(expr, pheno)[0,1]:+.4f}  (they are independent)")
print(f"  correlation AMONG QC-PASSING SAMPLES    = "
      f"{np.corrcoef(expr[passed_qc], pheno[passed_qc])[0,1]:+.4f}"
      f"   <- manufactured")
print(f"\n  {'QC stringency (kept)':>22}{'observed correlation':>24}")
for keep in [1.0, 0.8, 0.6, 0.4, 0.2]:
    sel = quality > np.quantile(quality, 1 - keep)
    print(f"  {keep:>21.0%}{np.corrcoef(expr[sel], pheno[sel])[0,1]:>24.4f}")
print("\n  The stricter the filter, the stronger the fake association. Draw your")
print("  QC and selection steps INTO the causal graph - they are causal")
print("  structures, not neutral preprocessing.")

# %% [markdown]
# ## 3. The g-formula, propensity scores and IPTW, eq. (32.2)-(32.4)

# %%
header("3. Standardisation, propensity scores, IPTW (32.2)-(32.4)")

def make_observational(n=6000, true_ate=1.5, seed=0, overlap=1.0):
    """Observational cohort with two measured confounders.

    `overlap` scales the strength of confounding: larger = worse overlap.
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.normal(0, 1, n)
    Z2 = rng.binomial(1, 0.4, n)
    ps = 1 / (1 + np.exp(-(overlap * (1.3 * Z1 + 1.1 * Z2 - 0.6))))
    A = rng.binomial(1, ps)
    Y = 2.0 + true_ate * A + 1.8 * Z1 + 1.4 * Z2 + rng.normal(0, 1, n)
    return pd.DataFrame({"Y": Y, "A": A, "Z1": Z1, "Z2": Z2, "ps_true": ps})


TRUE_ATE = 1.5
d = make_observational(true_ate=TRUE_ATE, seed=2203)

# (a) crude
crude = d.loc[d.A == 1, "Y"].mean() - d.loc[d.A == 0, "Y"].mean()

# (b) regression adjustment (coefficient)
reg = smf.ols("Y ~ A + Z1 + Z2", data=d).fit().params["A"]

# (c) g-formula / standardisation, eq. (32.2): average the contrast over the
#     covariate distribution of the WHOLE sample.
m_out = smf.ols("Y ~ A * (Z1 + Z2)", data=d).fit()
d1 = d.copy(); d1["A"] = 1
d0 = d.copy(); d0["A"] = 0
gform = (m_out.predict(d1) - m_out.predict(d0)).mean()

# (d) IPTW, eq. (32.3)-(32.4)
ps_model = smf.logit("A ~ Z1 + Z2", data=d).fit(disp=0)
e = ps_model.predict(d)
w = d["A"] / e + (1 - d["A"]) / (1 - e)
iptw = (np.sum(w * d["A"] * d["Y"]) / np.sum(w * d["A"])
        - np.sum(w * (1 - d["A"]) * d["Y"]) / np.sum(w * (1 - d["A"])))

# (e) AIPW / doubly robust, eq. (32.5)
mu1 = m_out.predict(d1); mu0 = m_out.predict(d0)
aipw = np.mean(mu1 - mu0
               + d["A"] * (d["Y"] - mu1) / e
               - (1 - d["A"]) * (d["Y"] - mu0) / (1 - e))

print(f"  TRUE ATE = {TRUE_ATE}")
print(f"  {'estimator':<40}{'estimate':>10}{'bias':>9}")
for lab, v in [("crude difference of means", crude),
               ("regression coefficient", reg),
               ("g-formula / standardisation (32.2)", gform),
               ("IPTW (32.4)", iptw),
               ("AIPW, doubly robust (32.5)", aipw)]:
    print(f"  {lab:<40}{v:>10.4f}{v-TRUE_ATE:>9.4f}")

# Doubly robust means: correct if EITHER model is right.
print("\n  'Doubly robust' demonstrated - deliberately MISSPECIFY one model:")
m_bad = smf.ols("Y ~ A", data=d).fit()            # outcome model omits Z1, Z2
mu1b = m_bad.predict(d1); mu0b = m_bad.predict(d0)
aipw_bad_out = np.mean(mu1b - mu0b + d["A"] * (d["Y"] - mu1b) / e
                       - (1 - d["A"]) * (d["Y"] - mu0b) / (1 - e))
ps_bad = smf.logit("A ~ 1", data=d).fit(disp=0).predict(d)   # PS omits Z1, Z2
aipw_bad_ps = np.mean(mu1 - mu0 + d["A"] * (d["Y"] - mu1) / ps_bad
                      - (1 - d["A"]) * (d["Y"] - mu0) / (1 - ps_bad))
print(f"    AIPW with WRONG outcome model, right PS : {aipw_bad_out:.4f}")
print(f"    AIPW with right outcome model, WRONG PS : {aipw_bad_ps:.4f}")
print(f"    (both close to {TRUE_ATE}; you need only ONE of the two correct)")

# %% [markdown]
# ## 4. Diagnose overlap and balance: not with p-values

# %%
header("4. Positivity (overlap) and covariate balance")

def smd(x, a, weights=None):
    """Standardised mean difference: the right balance diagnostic."""
    if weights is None:
        weights = np.ones(len(x))
    m1 = np.average(x[a == 1], weights=weights[a == 1])
    m0 = np.average(x[a == 0], weights=weights[a == 0])
    v1 = np.average((x[a == 1] - m1) ** 2, weights=weights[a == 1])
    v0 = np.average((x[a == 0] - m0) ** 2, weights=weights[a == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)


print(f"  {'covariate':<12}{'SMD before':>13}{'SMD after IPTW':>17}  target |SMD| < 0.1")
for col in ["Z1", "Z2"]:
    print(f"  {col:<12}{smd(d[col].values, d['A'].values):>13.3f}"
          f"{smd(d[col].values, d['A'].values, w.values):>17.3f}")

print(f"\n  Overlap (positivity). Propensity range by arm:")
for a in [0, 1]:
    pe = e[d["A"] == a]
    print(f"    arm {a}: min {pe.min():.4f}, 1st pct {np.percentile(pe,1):.4f}, "
          f"99th pct {np.percentile(pe,99):.4f}, max {pe.max():.4f}")
print(f"    largest IPTW weight = {w.max():.1f}  (weights > ~10 are a warning)")

print("\n  Now the SAME analysis with POOR overlap (strong confounding):")
for ov in [1.0, 2.5, 4.0]:
    dd = make_observational(seed=2204, overlap=ov)
    ee = smf.logit("A ~ Z1 + Z2", data=dd).fit(disp=0).predict(dd)
    ww = dd["A"] / ee + (1 - dd["A"]) / (1 - ee)
    est = (np.sum(ww * dd["A"] * dd["Y"]) / np.sum(ww * dd["A"])
           - np.sum(ww * (1-dd["A"]) * dd["Y"]) / np.sum(ww * (1-dd["A"])))
    print(f"    overlap scale {ov:.1f}: min(e)={ee.min():.5f}, "
          f"max weight={ww.max():8.1f}, IPTW ATE={est:.3f} "
          f"(bias {est-TRUE_ATE:+.3f})")
print("  Propensities near 0 or 1 produce enormous weights and unstable")
print("  estimates. That is a POSITIVITY VIOLATION - trimming or restricting to")
print("  the region of common support is more honest than extrapolating.")

print("\n  NOTE: balance is judged by SMD, NOT by p-values. A p-value for")
print("  balance just measures your sample size, not the size of the imbalance.")

# %% [markdown]
# ## 5. Mediation, eq. (32.6)-(32.7)

# %%
header("5. Decomposing a total effect (32.6)-(32.7)")

rng = np.random.default_rng(2205)
n = 20000
A = rng.binomial(1, 0.5, n)
# Cell composition mediates part of the exposure's effect on methylation.
Mcell = 0.9 * A + rng.normal(0, 1, n)
Ym = 0.4 * A + 0.7 * Mcell + rng.normal(0, 1, n)
dm = pd.DataFrame({"A": A, "M": Mcell, "Y": Ym})

TOTAL = 0.4 + 0.7 * 0.9
m_tot = smf.ols("Y ~ A", data=dm).fit()
m_med = smf.ols("M ~ A", data=dm).fit()
m_out2 = smf.ols("Y ~ A + M", data=dm).fit()
nde = m_out2.params["A"]
nie = m_med.params["A"] * m_out2.params["M"]      # product of coefficients

print(f"  TRUE total effect  = {TOTAL:.3f}")
print(f"  estimated total    = {m_tot.params['A']:.4f}")
print(f"  direct effect NDE  = {nde:.4f}   (truth 0.400)")
print(f"  indirect NIE       = {nie:.4f}   (truth {0.7*0.9:.3f})")
print(f"  NDE + NIE          = {nde + nie:.4f}  <- eq. (32.6)")
print(f"  proportion mediated= {nie/(nde+nie):.1%}")
print("\n  This is the correct framing for 'is the methylation effect mediated")
print("  by cell composition?' (Topic 23). But the product-of-coefficients")
print("  approach assumes NO exposure-mediator interaction AND no unmeasured")
print("  mediator-outcome confounding - which randomising A does NOT deliver.")

# %% [markdown]
# ## 6. Sensitivity: the E-value, eq. (32.8)

# %%
header("6. How strong would an unmeasured confounder have to be? (32.8)")

def e_value(rr):
    """stats.md eq. (32.8). For RR < 1, apply to 1/RR."""
    rr = max(rr, 1 / rr)
    return rr + np.sqrt(rr * (rr - 1))


print(f"  {'observed RR':>13}{'E-value':>10}  interpretation")
for rr in [1.1, 1.25, 1.5, 2.0, 3.0, 5.0]:
    ev = e_value(rr)
    note = ("trivially explained away" if ev < 1.7 else
            "easily explained away" if ev < 2.2 else
            "would need a strong unmeasured confounder" if ev < 4 else
            "robust to plausible confounding")
    print(f"  {rr:>13.2f}{ev:>10.2f}  {note}")
print("\n  The E-value is the minimum association (on the RR scale) an")
print("  unmeasured confounder would need with BOTH exposure and outcome to")
print("  fully explain the observed effect. Reporting it converts 'residual")
print("  confounding is possible' into a quantitative claim readers can judge.")
print("  Compare it to the strength of your MEASURED confounders for context.")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sel = quality > np.quantile(quality, 0.6)
axes[0].scatter(expr[~sel][:3000], pheno[~sel][:3000], s=3, alpha=0.2,
                label="failed QC")
axes[0].scatter(expr[sel][:3000], pheno[sel][:3000], s=3, alpha=0.3,
                color="C3", label="passed QC")
axes[0].set_xlabel("expression"); axes[0].set_ylabel("phenotype")
axes[0].set_title("Selection on a collider creates correlation", fontsize=9)
axes[0].legend(fontsize=7)

axes[1].hist(e[d.A == 0], bins=40, alpha=0.6, density=True, label="untreated")
axes[1].hist(e[d.A == 1], bins=40, alpha=0.6, density=True, label="treated")
axes[1].set_xlabel("propensity score $e(Z)$")
axes[1].set_title("Eq. (32.3): overlap check", fontsize=9)
axes[1].legend(fontsize=7)

rrs = np.linspace(1.01, 6, 200)
axes[2].plot(rrs, [e_value(r) for r in rrs], lw=2)
axes[2].plot(rrs, rrs, "k--", lw=1, label="y = RR")
axes[2].set_xlabel("observed risk ratio"); axes[2].set_ylabel("E-value")
axes[2].set_title("Eq. (32.8): sensitivity to unmeasured confounding",
                  fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "causal.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/causal.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 32)
#
# 1. **Draw the DAG first.** Adjustment sets come from the graph, not from
#    "everything available" or stepwise selection.
# 2. Never adjust for a mediator when estimating a total effect; never adjust
#    for or select on a collider.
# 3. Check positivity and covariate balance (SMD), not model fit.
# 4. State the untestable assumptions explicitly and quantify sensitivity.
# 5. Batch, selection and technical filtering are causal structures too -
#    draw them in the DAG.
#
# **Next:** `33_bayesian_and_shrinkage.py`
