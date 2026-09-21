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
# # Module 15: Missing data, censoring, and measurement limits
#
# **Curriculum link:** `stats.md` -> Topic 15, equations (15.1)-(15.9)
#
# ## What you will learn
#
# 1. MCAR / MAR / MNAR (15.1)-(15.3) simulated side by side, so you can *see*
#    which methods survive which mechanism.
# 2. Multiple imputation and **Rubin's rules (15.4)-(15.7)** implemented from
#    scratch: and why single imputation undercovers.
# 3. Left-censoring (limit of detection) as a **Tobit likelihood (15.8)**, not
#    as missingness.
# 4. The one diagnostic plot every proteomics report needs.
# 5. Why imputation must live inside the cross-validation fold.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "15_missing_data_and_censoring"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. The three mechanisms, eq. (15.1)-(15.3)
#
# * **MCAR**: missingness independent of everything.
# * **MAR**: depends only on *observed* data: recoverable by conditioning.
# * **MNAR**: depends on the *unobserved* value itself: not recoverable
#   without an untestable assumption.
#
# Critically, **MAR and MNAR cannot be distinguished from the observed data.**

# %%
header("1. Simulating MCAR, MAR and MNAR (15.1)-(15.3)")

def make_data(n=600, seed=0):
    """Complete data: y depends on x (the target estimand is beta_x = 1.5)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y})


def impose_missing(df, mechanism, rate=0.4, seed=0):
    """Blank out some y values under the requested mechanism."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if mechanism == "MCAR":                       # eq. (15.1)
        p = np.full(n, rate)
    elif mechanism == "MAR":                      # eq. (15.2): depends on x
        lin = 2.2 * df["x"].values
        p = 1 / (1 + np.exp(-(lin - np.log((1 - rate) / rate))))
    elif mechanism == "MNAR":                     # eq. (15.3): depends on y
        lin = 2.2 * (df["y"].values - df["y"].mean()) / df["y"].std()
        p = 1 / (1 + np.exp(-(lin - np.log((1 - rate) / rate))))
    else:
        raise ValueError(mechanism)
    out = df.copy()
    out["missing"] = rng.random(n) < p
    out.loc[out["missing"], "y"] = np.nan
    return out


TRUE_BETA = 1.5
print(f"  target estimand: beta_x = {TRUE_BETA}\n")
print(f"  {'mechanism':<12}{'% missing':>11}{'complete-case beta':>21}"
      f"{'bias':>9}")
for mech in ["MCAR", "MAR", "MNAR"]:
    biases = []
    for rep in range(200):
        full = make_data(seed=1500 + rep)
        obs = impose_missing(full, mech, seed=2500 + rep)
        cc = obs.dropna(subset=["y"])
        biases.append(smf.ols("y ~ x", data=cc).fit().params["x"] - TRUE_BETA)
    obs1 = impose_missing(make_data(seed=1500), mech, seed=2500)
    print(f"  {mech:<12}{obs1['missing'].mean():>10.1%}"
          f"{TRUE_BETA + np.mean(biases):>21.4f}{np.mean(biases):>9.4f}")

print("\n  MCAR: complete-case is unbiased (just wasteful).")
print("  MAR : complete-case is UNBIASED HERE because missingness depends only")
print("        on x, which is in the model. That is not general - if it")
print("        depended on another covariate you omitted, it would be biased.")
print("  MNAR: complete-case is biased, and no amount of imputation fixes it")
print("        without an assumption you cannot test.")

# %% [markdown]
# ### The MAR case where complete-case really does fail
#
# When missingness in a *covariate* depends on the outcome, complete-case
# analysis is biased even under MAR.

# %%
print("\n--- MAR in a covariate, depending on the outcome ---")
biases_cc, biases_mi = [], []
for rep in range(150):
    rng = np.random.default_rng(3000 + rep)
    n = 500
    x = rng.normal(0, 1, n)
    z = 0.8 * x + rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x + 0.7 * z + rng.normal(0, 1, n)
    d = pd.DataFrame({"x": x, "z": z, "y": y})
    # x is missing with a probability driven by the OBSERVED y: this is MAR.
    pmiss = 1 / (1 + np.exp(-(1.5 * (y - y.mean()) / y.std() - 0.4)))
    d.loc[rng.random(n) < pmiss, "x"] = np.nan
    cc = d.dropna()
    if len(cc) > 30:
        biases_cc.append(smf.ols("y ~ x + z", data=cc).fit().params["x"] - 1.5)
print(f"  complete-case bias in beta_x: {np.mean(biases_cc):+.4f}")
print("  Under MAR, the fix is a method that CONDITIONS ON THE OBSERVED DATA")
print("  driving missingness - multiple imputation including y, or a")
print("  likelihood-based analysis.")

# %% [markdown]
# ## 2. Multiple imputation and Rubin's rules, eq. (15.4)-(15.7)
#
# $$\bar Q=\frac1M\sum_m\hat Q_m, \qquad T=\bar U+\Big(1+\frac1M\Big)B$$
#
# The $(1+1/M)B$ term is exactly what single imputation omits: which is why
# single imputation gives intervals that are too narrow.

# %%
header("2. Rubin's rules from scratch (15.4)-(15.7)")

def impute_once(df, target, predictors, rng):
    """Stochastic regression imputation (the core step of MICE).

    Fit target ~ predictors on the complete cases, then draw imputed values
    from the PREDICTIVE distribution - adding residual noise is what makes it
    'proper'; predicting the conditional mean alone shrinks variability.
    """
    cc = df.dropna(subset=[target] + predictors)
    X = sm.add_constant(cc[predictors].values)
    fit = sm.OLS(cc[target].values, X).fit()
    miss = df[target].isna()
    if miss.sum() == 0:
        return df[target].values
    Xm = sm.add_constant(df.loc[miss, predictors].values, has_constant="add")
    # Draw beta from its sampling distribution, then add residual noise.
    beta_draw = rng.multivariate_normal(fit.params, fit.cov_params())
    sigma = np.sqrt(fit.mse_resid)
    vals = Xm @ beta_draw + rng.normal(0, sigma, miss.sum())
    out = df[target].values.copy()
    out[miss.values] = vals
    return out


def rubin_pool(estimates, variances, M):
    """stats.md eq. (15.4)-(15.7)."""
    Q_bar = np.mean(estimates)                       # (15.4)
    U_bar = np.mean(variances)                       # (15.5) within
    B = np.var(estimates, ddof=1) if M > 1 else 0.0  # (15.5) between
    T = U_bar + (1 + 1 / M) * B                      # (15.6)
    gamma = (1 + 1 / M) * B / T if T > 0 else 0.0    # fraction missing info
    nu = (M - 1) * (1 + U_bar / ((1 + 1 / M) * B)) ** 2 if B > 0 else np.inf
    return {"estimate": Q_bar, "se": np.sqrt(T), "df": nu, "fmi": gamma}


def run_mi(df, M, seed):
    """Impute M times, analyse each, pool with Rubin's rules."""
    rng = np.random.default_rng(seed)
    ests, vars_ = [], []
    for _ in range(M):
        d = df.copy()
        d["y"] = impute_once(d, "y", ["x"], rng)
        f = smf.ols("y ~ x", data=d).fit()
        ests.append(f.params["x"])
        vars_.append(f.bse["x"] ** 2)
    return rubin_pool(ests, vars_, M)


# Coverage study: does each strategy's 95% CI actually cover beta_x = 1.5?
print("  Coverage of a nominal 95% CI for beta_x, MCAR with 40% of y missing:")
cover = {"complete case": 0, "single imputation": 0, "MI (M=20)": 0}
width = {k: [] for k in cover}
N_REP = 250
for rep in range(N_REP):
    full = make_data(n=300, seed=4000 + rep)
    obs = impose_missing(full, "MCAR", rate=0.4, seed=5000 + rep)

    f_cc = smf.ols("y ~ x", data=obs.dropna(subset=["y"])).fit()
    lo, hi = f_cc.conf_int().loc["x"]
    cover["complete case"] += lo <= TRUE_BETA <= hi
    width["complete case"].append(hi - lo)

    r1 = run_mi(obs, M=1, seed=6000 + rep)
    lo = r1["estimate"] - 1.96 * r1["se"]; hi = r1["estimate"] + 1.96 * r1["se"]
    cover["single imputation"] += lo <= TRUE_BETA <= hi
    width["single imputation"].append(hi - lo)

    r20 = run_mi(obs, M=20, seed=7000 + rep)
    tcrit = st.t.ppf(0.975, min(r20["df"], 1e6))
    lo = r20["estimate"] - tcrit * r20["se"]; hi = r20["estimate"] + tcrit * r20["se"]
    cover["MI (M=20)"] += lo <= TRUE_BETA <= hi
    width["MI (M=20)"].append(hi - lo)

print(f"  {'method':<22}{'coverage':>11}{'mean CI width':>16}")
for k in cover:
    print(f"  {k:<22}{cover[k]/N_REP:>11.1%}{np.mean(width[k]):>16.4f}")
print("\n  Single imputation treats imputed values as if observed, so its CI is")
print("  too narrow and UNDERCOVERS. Multiple imputation adds the between-")
print("  imputation term (1+1/M)B and recovers nominal coverage.")

r = run_mi(impose_missing(make_data(n=300, seed=99), "MCAR", 0.4, seed=99),
           M=20, seed=99)
print(f"\n  Example pooled result: beta = {r['estimate']:.4f}, "
      f"SE = {r['se']:.4f}, df = {r['df']:.1f}")
print(f"  fraction of missing information = {r['fmi']:.3f}   (eq. 15.7)")
print("  Use M >= 20 when the FMI is substantial; the old 'M=5' advice is")
print("  outdated for efficiency.")

# %% [markdown]
# ### The imputation model must be at least as rich as the analysis model
#
# Omitting the outcome from the imputation model biases associations **toward
# zero**: the commonest multiple-imputation error.

# %%
print("\n--- Omitting the outcome from the imputation model ---")
b_with, b_without = [], []
for rep in range(150):
    rng = np.random.default_rng(8000 + rep)
    n = 400
    x = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 1, n)
    d = pd.DataFrame({"x": x, "y": y})
    d.loc[rng.random(n) < 0.4, "x"] = np.nan        # MCAR in x

    # (a) impute x USING y (correct)
    d1 = d.copy()
    d1["x"] = impute_once(d1, "x", ["y"], rng)
    b_with.append(smf.ols("y ~ x", data=d1).fit().params["x"])

    # (b) impute x from its own marginal mean only (outcome omitted)
    d2 = d.copy()
    d2["x"] = d2["x"].fillna(d2["x"].mean())
    b_without.append(smf.ols("y ~ x", data=d2).fit().params["x"])

print(f"  true beta_x                              = {TRUE_BETA:.4f}")
print(f"  imputing x from y (outcome included)     = {np.mean(b_with):.4f}")
print(f"  mean-imputing x (outcome omitted)        = {np.mean(b_without):.4f}"
      f"   <- attenuated")

# %% [markdown]
# ## 3. Left-censoring is NOT missingness, eq. (15.8)
#
# A proteomics value below the limit of detection $L$ is not unknown: it is
# **known to be below $L$**. The correct likelihood is censored (Tobit):
#
# $$L(\mu,\sigma)=\prod_{y_i>L}\frac1\sigma\phi\Big(\frac{y_i-\mu}{\sigma}\Big)
#   \prod_{y_i\le L}\Phi\Big(\frac{L-\mu}{\sigma}\Big)$$

# %%
header("3. Limit of detection: censored likelihood vs substitution (15.8)")

def tobit_mle(y_obs, censored, L):
    """stats.md eq. (15.8): MLE of (mu, sigma) under left-censoring at L."""
    n_cen = int(censored.sum())

    def nll(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        # Uncensored part: the density at each observed value.
        ll_obs = np.sum(st.norm.logpdf(y_obs[~censored], mu, sigma))
        # Censored part: EACH censored observation contributes log Phi((L-mu)/sigma).
        # Multiplying by n_cen is essential - the term is identical for every
        # censored point, so summing a scalar once would drop n_cen - 1 of them.
        ll_cen = n_cen * st.norm.logcdf((L - mu) / sigma)
        return -(ll_obs + ll_cen)

    start = [float(np.mean(y_obs[~censored])) if (~censored).any() else L,
             float(np.log(max(np.std(y_obs[~censored], ddof=1), 1e-3)))]
    res = minimize(nll, x0=start, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return res.x[0], np.exp(res.x[1])


rng = np.random.default_rng(1501)
TRUE_MU, TRUE_SD, L = 20.0, 3.0, 18.0
print(f"  truth: mu = {TRUE_MU}, sigma = {TRUE_SD}, LOD = {L}")

rows = []
for rep in range(400):
    z = rng.normal(TRUE_MU, TRUE_SD, 120)
    cen = z < L
    y = np.where(cen, np.nan, z)

    mu_drop = np.nanmean(y)                        # (a) drop censored values
    sd_drop = np.nanstd(y, ddof=1)
    y_half = np.where(cen, L / 2, z)               # (b) substitute L/2
    y_sq2 = np.where(cen, L / np.sqrt(2), z)       # (c) substitute L/sqrt(2)
    y_lod = np.where(cen, L, z)                    # (d) substitute L
    mu_t, sd_t = tobit_mle(np.where(cen, L, z), cen, L)   # (e) censored MLE

    rows.append({"drop_mu": mu_drop, "drop_sd": sd_drop,
                 "half_mu": y_half.mean(), "half_sd": y_half.std(ddof=1),
                 "sq2_mu": y_sq2.mean(), "sq2_sd": y_sq2.std(ddof=1),
                 "lod_mu": y_lod.mean(), "lod_sd": y_lod.std(ddof=1),
                 "tobit_mu": mu_t, "tobit_sd": sd_t,
                 "pct_cen": cen.mean()})
res = pd.DataFrame(rows)
print(f"  average censored fraction: {res['pct_cen'].mean():.1%}\n")
print(f"  {'strategy':<28}{'mean mu_hat':>13}{'bias':>9}"
      f"{'mean sigma_hat':>16}{'bias':>9}")
for key, lab in [("drop", "drop censored values"), ("half", "substitute L/2"),
                 ("sq2", "substitute L/sqrt(2)"), ("lod", "substitute L"),
                 ("tobit", "censored MLE (15.8)")]:
    mu_m, sd_m = res[f"{key}_mu"].mean(), res[f"{key}_sd"].mean()
    print(f"  {lab:<28}{mu_m:>13.4f}{mu_m-TRUE_MU:>9.4f}"
          f"{sd_m:>16.4f}{sd_m-TRUE_SD:>9.4f}")

print("\n  Every substitution rule is biased in BOTH the mean and the SD.")
print("  Dropping censored values biases the mean UP. The censored likelihood")
print("  recovers both parameters essentially without bias.")

# %% [markdown]
# ## 4. The diagnostic every proteomics report needs
#
# Plot per-feature missingness rate against mean observed abundance. A strong
# negative slope is the signature of abundance-dependent (MNAR) missingness.

# %%
header("4. Diagnosing the mechanism from missingness vs abundance")

rng = np.random.default_rng(1502)
G, n = 1500, 20
true_abundance = rng.normal(22, 2.5, G)
Y = rng.normal(true_abundance[:, None], 0.8, (G, n))

# Mixture: 70% of features MNAR (low abundance -> more missing), 30% MAR.
is_mnar = rng.random(G) < 0.7
p_missing = np.where(
    is_mnar,
    1 / (1 + np.exp((true_abundance - 20.5) / 0.6)),   # abundance-dependent
    0.15)                                              # flat
M = rng.random((G, n)) < p_missing[:, None]
Yobs = np.where(M, np.nan, Y)

miss_rate = np.isnan(Yobs).mean(axis=1)
mean_obs = np.nanmean(Yobs, axis=1)
ok = np.isfinite(mean_obs)
r = st.spearmanr(mean_obs[ok], miss_rate[ok])
print(f"  Spearman(mean observed abundance, missingness rate) = "
      f"{r.statistic:+.3f}  (p = {r.pvalue:.2e})")
print("  A strong NEGATIVE correlation => abundance-dependent missingness.")
print(f"\n  {'mean-abundance quintile':<26}{'mean missingness':>18}")
qs = pd.qcut(mean_obs[ok], 5, labels=[f"Q{i+1}" for i in range(5)])
for q, grp in pd.DataFrame({"q": qs, "m": miss_rate[ok]}).groupby("q",
                                                                 observed=True):
    print(f"  {str(q):<26}{grp['m'].mean():>18.3f}")
print("\n  Treat the two classes differently: MNAR-style (down-shifted or")
print("  censored) handling for features missing systematically in one")
print("  condition, MAR-appropriate imputation for sporadic gaps. Applying ONE")
print("  imputer to both is the standard mistake (stats.md Topic 25).")

# %% [markdown]
# ## 5. Imputation is a learned transformation: it belongs inside the fold

# %%
header("5. Imputing before cross-validation leaks information")

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

rng = np.random.default_rng(1503)
n, p = 120, 40
X = rng.normal(0, 1, (n, p))
y = X[:, 0] * 0.0 + rng.normal(0, 1, n)          # NO real signal at all
Xm = X.copy()
Xm[rng.random((n, p)) < 0.3] = np.nan

from sklearn.impute import KNNImputer


def make_imputer(kind):
    """Mean imputation uses only COLUMN statistics; kNN borrows across ROWS."""
    return SimpleImputer() if kind == "mean" else KNNImputer(n_neighbors=5)


def cv_r2(X, y, impute_inside, kind="mean", seed=0):
    kf = KFold(5, shuffle=True, random_state=seed)
    scores = []
    if not impute_inside:
        # LEAKY: impute using statistics computed from ALL rows, then CV.
        X = make_imputer(kind).fit_transform(X)
    for tr, te in kf.split(X):
        if impute_inside:
            model = Pipeline([("imp", make_imputer(kind)),
                              ("lm", LinearRegression())])
        else:
            model = LinearRegression()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        scores.append(1 - np.sum((y[te]-pred)**2) / np.sum((y[te]-y[tr].mean())**2))
    return np.mean(scores)


print(f"  Data contain NO signal; honest R^2 should be <= 0.")
print(f"  Averaged over 30 datasets, so the comparison is not one draw:\n")
print(f"  {'imputer':<34}{'before CV (leaky)':>19}{'inside the fold':>18}"
      f"{'optimism':>11}")
for kind, lab in (("mean", "column mean (no row borrowing)"),
                  ("knn", "kNN, k=5 (borrows across ROWS)")):
    lk, hn = [], []
    for rep in range(30):
        r2 = np.random.default_rng(1600 + rep)
        Xr = r2.normal(0, 1, (n, p))
        yr = r2.normal(0, 1, n)                     # NO real signal at all
        Xmr = Xr.copy()
        Xmr[r2.random((n, p)) < 0.3] = np.nan
        lk.append(cv_r2(Xmr, yr, False, kind, seed=rep))
        hn.append(cv_r2(Xmr, yr, True, kind, seed=rep))
    print(f"  {lab:<34}{np.mean(lk):>19.4f}{np.mean(hn):>18.4f}"
          f"{np.mean(lk) - np.mean(hn):>+11.4f}")

print("""
  Both rows should be at or below zero, because there is nothing to predict.

  Mean imputation leaks little: it only ever uses COLUMN statistics, so a
  held-out row contributes almost nothing to how it is filled.

  kNN imputation leaks much more, and the reason is structural: it fills a
  cell by copying from the most similar ROWS, and when it is fitted on all the
  data those neighbours include the test rows themselves. The test row is
  partly reconstructed from its own values before the model ever sees it.

  This is the rule behind the classification in Module 31: what matters is not
  whether a step is 'preprocessing' but whether it BORROWS ACROSS ROWS. Wrap
  every learned step in a Pipeline so the fold boundary is enforced
  structurally rather than by discipline.""")

# %% [markdown]
# ## 6. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

full = make_data(n=800, seed=77)
for i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
    obs = impose_missing(full, mech, seed=77)
    axes[0].scatter(full["x"][obs["missing"]], full["y"][obs["missing"]],
                    s=6, alpha=0.5, label=f"{mech} missing")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y (true value)")
axes[0].set_title("Which points go missing under each mechanism", fontsize=9)
axes[0].legend(fontsize=7)

z = rng.normal(TRUE_MU, TRUE_SD, 3000)
axes[1].hist(z, bins=60, alpha=0.5, label="true distribution")
axes[1].hist(z[z >= L], bins=60, alpha=0.7, label="observed (> LOD)")
axes[1].axvline(L, color="red", lw=2, label="LOD")
axes[1].set_title("Eq. (15.8): left-censoring", fontsize=9)
axes[1].legend(fontsize=7)

axes[2].scatter(mean_obs[ok], miss_rate[ok], s=4, alpha=0.3)
axes[2].set_xlabel("mean observed abundance")
axes[2].set_ylabel("missingness rate")
axes[2].set_title(f"MNAR signature (Spearman {r.statistic:+.2f})", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "missing_data.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/missing_data.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 15)
#
# 1. Name the mechanism for each variable and justify it from how the assay
#    works: you cannot test MAR vs MNAR.
# 2. Never impute the outcome to create observations; model it instead.
# 3. Use $M\ge20$ imputations when the fraction of missing information is
#    substantial, and include the outcome in the imputation model.
# 4. Model detection limits as **censoring** (15.8), not as missingness.
# 5. Report a sensitivity analysis over plausible MNAR departures: not optional
#    for proteomics.
#
# **Next:** `16_matrix_algebra.py`
