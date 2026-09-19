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
# # Module 13: Generalised linear models
#
# **Curriculum link:** `stats.md` -> Topic 13, equations (13.1)-(13.12)
#
# ## What you will learn
#
# 1. The exponential family (13.1)-(13.2): mean and variance from one function.
# 2. **IRLS (13.3)-(13.4) implemented from scratch**: a GLM is a weighted
#    linear model on a locally linearised response.
# 3. Logistic regression (13.6): odds ratios, non-collapsibility, separation
#    and Firth's fix (13.7).
# 4. **Offsets (13.8)**: counts vs rates. The single most common GLM error.
# 5. Overdispersion (13.9): quasi-Poisson vs negative binomial.
# 6. Wald vs LRT vs score (13.11), and why the LRT is safer.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "13_generalized_linear_models"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. The three components, and the family table

# %%
header("1. Family, link, variance function (13.1)-(13.2)")

fam = pd.DataFrame([
    ("Gaussian",      "R",          "1",              "identity", "mean difference"),
    ("Binomial",      "{0,1}",      "mu(1-mu)",       "logit",    "log odds ratio"),
    ("Poisson",       "{0,1,2,..}", "mu",             "log",      "log rate ratio"),
    ("Neg. binomial", "{0,1,2,..}", "mu + phi*mu^2",  "log",      "log fold change"),
    ("Gamma",         "(0, inf)",   "mu^2",           "log",      "log ratio of means"),
], columns=["family", "support", "V(mu)", "canonical link", "exp(beta) means"])
print(fam.to_string(index=False))

# %% [markdown]
# ## 2. IRLS from scratch, eq. (13.3)-(13.5)
#
# $$z_i=\eta_i+(y_i-\mu_i)g'(\mu_i), \qquad w_i=\frac{1}{g'(\mu_i)^2V(\mu_i)},
#   \qquad \boldsymbol\beta^{(t+1)}=(\mathbf X^\top\mathbf W\mathbf X)^{-1}\mathbf X^\top\mathbf W\mathbf z$$

# %%
header("2. Implementing Fisher scoring (13.3)-(13.5)")

def irls(X, y, family="binomial", offset=None, max_iter=50, tol=1e-10):
    """stats.md eq. (13.3)-(13.5). Supports logit-binomial and log-Poisson."""
    n, p = X.shape
    offset = np.zeros(n) if offset is None else offset
    beta = np.zeros(p)
    for it in range(max_iter):
        eta = X @ beta + offset
        if family == "binomial":
            mu = 1.0 / (1.0 + np.exp(-eta))
            V = mu * (1 - mu)
            gprime = 1.0 / np.clip(V, 1e-12, None)     # d eta / d mu for logit
        elif family == "poisson":
            mu = np.exp(eta)
            V = mu
            gprime = 1.0 / np.clip(mu, 1e-12, None)    # d eta / d mu for log
        else:
            raise ValueError(family)

        z = (eta - offset) + (y - mu) * gprime          # working response (13.3)
        w = 1.0 / (gprime**2 * np.clip(V, 1e-12, None)) # working weights  (13.3)

        # Weighted least squares step (13.4), solved stably.
        sw = np.sqrt(w)
        beta_new = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Variance from eq. (13.5) with dispersion a(phi) = 1
    XtWX = (X * w[:, None]).T @ X
    cov = np.linalg.inv(XtWX)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "iterations": it + 1,
            "weights": w, "mu": mu}


rng = np.random.default_rng(1301)
n = 400
x1 = rng.normal(0, 1, n)
x2 = rng.binomial(1, 0.5, n)
eta_true = -0.5 + 1.1 * x1 + 0.8 * x2
yb = rng.binomial(1, 1 / (1 + np.exp(-eta_true)))
Xd = np.column_stack([np.ones(n), x1, x2])

mine = irls(Xd, yb, "binomial")
theirs = sm.GLM(yb, Xd, family=sm.families.Binomial()).fit()
print(f"  logistic - converged in {mine['iterations']} IRLS iterations")
print(f"  {'term':<12}{'mine beta':>12}{'mine SE':>11}"
      f"{'statsmodels beta':>19}{'statsmodels SE':>17}")
for i, nm in enumerate(["intercept", "x1", "x2"]):
    print(f"  {nm:<12}{mine['beta'][i]:>12.6f}{mine['se'][i]:>11.6f}"
          f"{theirs.params[i]:>19.6f}{theirs.bse[i]:>17.6f}")
print(f"  max |difference| = {np.max(np.abs(mine['beta'] - theirs.params)):.2e}")

# Poisson with an offset, same routine.
expo = rng.uniform(0.5, 2.0, n)
yp = rng.poisson(expo * np.exp(0.3 + 0.7 * x1))
mp = irls(Xd, yp, "poisson", offset=np.log(expo))
tp = sm.GLM(yp, Xd, family=sm.families.Poisson(), offset=np.log(expo)).fit()
print(f"\n  Poisson+offset - max |mine - statsmodels| = "
      f"{np.max(np.abs(mp['beta'] - tp.params)):.2e}")

# %% [markdown]
# ## 3. Logistic regression: odds ratios and their traps, eq. (13.6)

# %%
header("3. Odds ratios, risk ratios, non-collapsibility (13.6)")

rng = np.random.default_rng(1302)
N = 30000
z = rng.normal(0, 1, N)                       # a PROGNOSTIC factor, not a confounder
a = rng.binomial(1, 0.5, N)                   # randomised exposure
pr = 1 / (1 + np.exp(-(-0.5 + 0.9 * a + 1.5 * z)))
yy = rng.binomial(1, pr)
d = pd.DataFrame({"y": yy, "a": a, "z": z})

m_crude = smf.logit("y ~ a", data=d).fit(disp=0)
m_adj = smf.logit("y ~ a + z", data=d).fit(disp=0)
print(f"  TRUE conditional log-OR for a (by construction) = 0.900")
print(f"  crude    log-OR = {m_crude.params['a']:.4f}  (OR = "
      f"{np.exp(m_crude.params['a']):.3f})")
print(f"  adjusted log-OR = {m_adj.params['a']:.4f}  (OR = "
      f"{np.exp(m_adj.params['a']):.3f})")
print("\n  The exposure was RANDOMISED, so z cannot be a confounder - yet the")
print("  estimate changed. This is NON-COLLAPSIBILITY (stats.md Topic 13):")
print("  logistic coefficients shift when covariates are added even without")
print("  confounding. Do NOT read a shift in the OR as evidence of confounding.")

# OR vs RR at different baseline risks.
print("\n  OR is not RR unless the outcome is rare:")
print(f"  {'baseline risk':>15}{'OR':>7}{'implied RR':>13}")
for p0 in [0.01, 0.05, 0.20, 0.40, 0.60]:
    OR = 2.0
    RR = OR / (1 - p0 + p0 * OR)
    print(f"  {p0:>15.2f}{OR:>7.1f}{RR:>13.3f}")

# %% [markdown]
# ### Separation and Firth's penalised likelihood, eq. (13.7)

# %%
print("\n--- Complete separation: the MLE diverges ---")
xs = np.array([-3., -2., -1.5, -1., 1., 1.5, 2., 3.])
ys = np.array([0, 0, 0, 0, 1, 1, 1, 1])          # x perfectly predicts y
Xs = np.column_stack([np.ones(8), xs])
try:
    f_sep = sm.GLM(ys, Xs, family=sm.families.Binomial()).fit(maxiter=200)
    print(f"  unpenalised MLE slope = {f_sep.params[1]:10.3f}  "
          f"SE = {f_sep.bse[1]:10.3f}   <- both exploding")
except Exception as exc:
    print(f"  unpenalised fit failed: {exc}")


def firth_logit(X, y, max_iter=200, tol=1e-8):
    """stats.md eq. (13.7): Firth's penalised likelihood (Jeffreys prior).

    The modified score is U*(beta)_j = U(beta)_j + sum_i h_ii (0.5 - mu_i) x_ij,
    which always yields finite estimates under separation.
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        XtWX = (X * W[:, None]).T @ X
        inv = np.linalg.pinv(XtWX)
        H = (X * W[:, None]) @ inv @ X.T
        h = np.diag(H)
        U = X.T @ (y - mu + h * (0.5 - mu))          # penalised score
        step = inv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    se = np.sqrt(np.diag(np.linalg.pinv((X * (mu*(1-mu))[:, None]).T @ X)))
    return beta, se


bf, sef = firth_logit(Xs, ys)
print(f"  Firth penalised slope = {bf[1]:10.3f}  SE = {sef[1]:10.3f}   <- finite")
print("  Separation is common with small n and rare outcomes. Firth is the fix.")

# %% [markdown]
# ## 4. Offsets: counts vs rates, eq. (13.8)
#
# $$\log\mathbb E[Y_i]=\log t_i+\mathbf x_i^\top\boldsymbol\beta$$
#
# The offset has a **fixed coefficient of 1**. Putting $\log t_i$ in as a free
# covariate answers a different question; omitting it answers a wrong one.

# %%
header("4. The offset: the most common GLM error (13.8)")

rng = np.random.default_rng(1303)
n = 200
group = rng.binomial(1, 0.5, n)
# Library size is UNBALANCED between groups - a very common situation.
libsize = np.exp(rng.normal(np.where(group == 1, 11.5, 10.8), 0.25))
TRUE_LOG_RR = 0.0                            # NO real rate difference
counts = rng.poisson(libsize * np.exp(-6.0 + TRUE_LOG_RR * group))
dd = pd.DataFrame({"y": counts, "g": group, "lib": libsize})

m_no_offset = smf.glm("y ~ g", data=dd, family=sm.families.Poisson()).fit()
m_offset = smf.glm("y ~ g", data=dd, family=sm.families.Poisson(),
                   offset=np.log(dd["lib"])).fit()
m_covar = smf.glm("y ~ g + np.log(lib)", data=dd,
                  family=sm.families.Poisson()).fit()

print(f"  TRUE log rate ratio = {TRUE_LOG_RR:.3f}")
print(f"  {'model':<34}{'log RR':>10}{'p':>11}")
print(f"  {'y ~ g  (NO offset)':<34}{m_no_offset.params['g']:>10.4f}"
      f"{m_no_offset.pvalues['g']:>11.3e}   <- spurious!")
print(f"  {'y ~ g + offset(log lib)':<34}{m_offset.params['g']:>10.4f}"
      f"{m_offset.pvalues['g']:>11.4f}   <- correct")
print(f"  {'y ~ g + log(lib) as covariate':<34}{m_covar.params['g']:>10.4f}"
      f"{m_covar.pvalues['g']:>11.4f}")
print(f"       (its log(lib) coefficient = {m_covar.params['np.log(lib)']:.4f}, "
      f"freely estimated rather than fixed at 1)")
print("\n  Without the offset the model detects a difference in DEPTH and reports")
print("  it as a difference in BIOLOGY. This is why RNA-seq size factors enter")
print("  as offsets (stats.md eq. 20.1).")

# %% [markdown]
# ## 5. Overdispersion, eq. (13.9): quasi-Poisson vs negative binomial

# %%
header("5. Detecting and handling overdispersion (13.9)")

rng = np.random.default_rng(1304)
n = 150
xg = rng.binomial(1, 0.5, n)
PHI = 0.4
# NOTE the parameterisation: numpy's negative_binomial takes (n_successes, p).
# With dispersion phi the standard reparameterisation is r = 1/phi and
# p = r/(r+mu), which gives mean mu and variance mu + phi*mu^2 (eq. 4.9).
nb_r = 1 / PHI
mu_true = np.exp(2.5 + 0.35 * xg)
y_od = rng.negative_binomial(nb_r, nb_r / (nb_r + mu_true), size=n)
do = pd.DataFrame({"y": y_od, "g": xg})

m_pois = smf.glm("y ~ g", data=do, family=sm.families.Poisson()).fit()
pearson_chi2 = np.sum((do["y"] - m_pois.fittedvalues) ** 2 / m_pois.fittedvalues)
phi_hat = pearson_chi2 / m_pois.df_resid                     # eq. (13.9)
print(f"  Pearson dispersion estimate phi_hat = {phi_hat:.3f}  (1.0 = no overdispersion)")
print(f"  -> Poisson SEs are too small by a factor sqrt(phi) = {np.sqrt(phi_hat):.2f}")

def quasi_poisson(fit, term):
    """Quasi-Poisson inference, done by hand from a Poisson fit.

    Quasi-likelihood keeps the SAME point estimates and multiplies every
    standard error by sqrt(phi_hat), where phi_hat is the Pearson dispersion
    of stats.md eq. (13.9). Inference then uses t_{n-p} rather than z.
    """
    chi2 = np.sum((fit.model.endog - fit.fittedvalues) ** 2 / fit.fittedvalues)
    phi = chi2 / fit.df_resid
    se = fit.bse[term] * np.sqrt(phi)
    tstat = fit.params[term] / se
    return se, 2 * st.t.sf(abs(tstat), fit.df_resid), phi


qp_se, qp_p, _ = quasi_poisson(m_pois, "g")
m_nb = smf.glm("y ~ g", data=do,
               family=sm.families.NegativeBinomial(alpha=PHI)).fit()

print(f"\n  {'model':<22}{'beta_g':>10}{'SE':>10}{'p':>11}  variance model")
print(f"  {'Poisson':<22}{m_pois.params['g']:>10.4f}{m_pois.bse['g']:>10.4f}"
      f"{m_pois.pvalues['g']:>11.4f}  Var = mu")
print(f"  {'quasi-Poisson':<22}{m_pois.params['g']:>10.4f}{qp_se:>10.4f}"
      f"{qp_p:>11.4f}  Var = phi*mu   (linear)")
print(f"  {'negative binomial':<22}{m_nb.params['g']:>10.4f}{m_nb.bse['g']:>10.4f}"
      f"{m_nb.pvalues['g']:>11.4f}  Var = mu+phi*mu^2 (quadratic)")
print("\n  All three give the same POINT estimate; they differ entirely in the")
print("  standard error. Calibration check under a TRUE null:")

hits = {"Poisson": 0, "quasi-Poisson": 0, "NB": 0}
N_SIM = 800
for _ in range(N_SIM):
    xg2 = rng.binomial(1, 0.5, n)
    # size=n is essential: without it numpy returns a SCALAR and every row of
    # the response would be identical. A good reminder to check shapes.
    y2 = rng.negative_binomial(nb_r, nb_r / (nb_r + np.exp(2.5)), size=n)
    d2 = pd.DataFrame({"y": y2, "g": xg2})
    f1 = smf.glm("y ~ g", data=d2, family=sm.families.Poisson()).fit()
    hits["Poisson"] += f1.pvalues["g"] < 0.05
    hits["quasi-Poisson"] += quasi_poisson(f1, "g")[1] < 0.05
    hits["NB"] += smf.glm("y ~ g", data=d2,
                          family=sm.families.NegativeBinomial(alpha=PHI)
                          ).fit().pvalues["g"] < 0.05
for k, v in hits.items():
    print(f"    {k:<16} false-positive rate = {v/N_SIM:.1%}")
print("  Poisson on overdispersed counts is badly anticonservative.")

# %% [markdown]
# ## 6. Wald vs LRT vs score, eq. (13.11)

# %%
header("6. Three tests, one hypothesis (13.11)")

def lrt(model_full, model_reduced):
    """stats.md eq. (13.11): 2*(l_1 - l_0) ~ chi2_df."""
    stat = 2 * (model_full.llf - model_reduced.llf)
    ddf = int(model_full.df_model - model_reduced.df_model)
    return stat, ddf, st.chi2.sf(stat, ddf)


full = smf.logit("y ~ a + z", data=d).fit(disp=0)
red = smf.logit("y ~ z", data=d).fit(disp=0)
wald = (full.params["a"] / full.bse["a"]) ** 2
lr_stat, lr_df, lr_p = lrt(full, red)
print(f"  Wald chi2 = {wald:.4f}, p = {st.chi2.sf(wald,1):.3e}")
print(f"  LRT  chi2 = {lr_stat:.4f}, p = {lr_p:.3e}")
print("  They agree asymptotically. Now near separation, where they do not:")

xs2 = np.array([-3., -2., -1.5, -1., -0.2, 1., 1.5, 2., 3., 0.1])
ys2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
d3 = pd.DataFrame({"y": ys2, "x": xs2})
f_full = sm.GLM(ys2, np.column_stack([np.ones(10), xs2]),
                family=sm.families.Binomial()).fit(maxiter=100)
f_red = sm.GLM(ys2, np.ones((10, 1)), family=sm.families.Binomial()).fit()
w2 = (f_full.params[1] / f_full.bse[1]) ** 2
l2 = 2 * (f_full.llf - f_red.llf)
print(f"    Wald chi2 = {w2:.4f}, p = {st.chi2.sf(w2,1):.4f}   <- no power")
print(f"    LRT  chi2 = {l2:.4f}, p = {st.chi2.sf(l2,1):.4f}   <- correct")
print("  The Wald statistic COLLAPSES as |beta| grows because its SE grows")
print("  faster (the Hauck-Donner effect). PREFER THE LRT.")

# %% [markdown]
# ## 7. Deviance residuals, eq. (13.12)

# %%
header("7. Use deviance residuals, not raw ones (13.12)")

m = smf.glm("y ~ g", data=do, family=sm.families.NegativeBinomial(alpha=PHI)).fit()
print(f"  deviance = {m.deviance:.4f}, df = {int(m.df_resid)}, "
      f"deviance/df = {m.deviance/m.df_resid:.3f}")
print(f"  raw residual SD varies with the mean: "
      f"{np.std(do['y'] - m.fittedvalues, ddof=1):.3f} overall, but")
lowm = m.fittedvalues < np.median(m.fittedvalues)
print(f"    low-mean half : {np.std((do['y']-m.fittedvalues)[lowm], ddof=1):.3f}")
print(f"    high-mean half: {np.std((do['y']-m.fittedvalues)[~lowm], ddof=1):.3f}")
dev = m.resid_deviance
print(f"  deviance residuals are homoscedastic by construction: "
      f"{np.std(dev[lowm], ddof=1):.3f} vs {np.std(dev[~lowm], ddof=1):.3f}")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

xg_ = np.linspace(-3.5, 3.5, 200)
axes[0].plot(xg_, 1/(1+np.exp(-(-0.5 + 1.1*xg_))), lw=2)
# len(yb), not n: `n` was rebound to 150 in section 5. Shadowing bugs like
# this are exactly why later modules keep simulation blocks self-contained.
axes[0].scatter(x1, yb + rng.normal(0, 0.02, len(yb)), s=5, alpha=0.25)
axes[0].set_xlabel("x1"); axes[0].set_ylabel("P(y=1)")
axes[0].set_title("Eq. (13.6): logistic mean function", fontsize=9)

axes[1].scatter(dd["lib"], dd["y"], c=dd["g"], cmap="coolwarm", s=12)
axes[1].set_xlabel("library size (exposure)"); axes[1].set_ylabel("count")
axes[1].set_title("Eq. (13.8): counts scale with exposure", fontsize=9)

axes[2].scatter(m.fittedvalues, m.resid_deviance, s=14, label="deviance")
axes[2].scatter(m.fittedvalues, (do["y"] - m.fittedvalues) /
                np.std(do["y"] - m.fittedvalues, ddof=1), s=14, alpha=0.5,
                label="raw (scaled)")
axes[2].axhline(0, color="k", lw=1)
axes[2].set_xlabel("fitted mean"); axes[2].set_ylabel("residual")
axes[2].set_title("Eq. (13.12): deviance residuals", fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "glm.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/glm.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 13)
#
# 1. Choose the family from the outcome's support and mean-variance
#    relationship; choose the link from the estimand you want to report.
# 2. Counts with varying exposure require an offset (13.8), always.
# 3. Check the dispersion (13.9). Overdispersed counts modelled as Poisson give
#    dramatically anticonservative p-values.
# 4. Report both scales: the odds/rate ratio **and** a predicted probability or
#    rate at meaningful covariate values.
#
# **Next:** `14_mixed_models.py`
