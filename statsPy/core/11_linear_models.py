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
# # Module 11: Linear regression as the core framework
#
# **Curriculum link:** `stats.md` -> Topic 11, equations (11.1)-(11.13)
#
# Almost every classical method in this course is
# $\mathbf y=\mathbf X\boldsymbol\beta+\boldsymbol\varepsilon$ with a different
# $\mathbf X$. Learn the algebra once.
#
# ## What you will learn
#
# 1. OLS from the normal equations (11.2), the hat matrix (11.3), and why you
#    should never invert $\mathbf X^\top\mathbf X$ numerically.
# 2. Coefficient standard errors and t-tests (11.4)-(11.5).
# 3. Nested-model F tests (11.6) and the uselessness of $R^2$ for selection.
# 4. **Confidence vs prediction intervals** (11.8)-(11.9).
# 5. Leverage, residuals, Cook's distance (11.10)-(11.11), VIF (11.12).
# 6. Splines (11.13): linear models are linear in the *parameters*.
# 7. The assumption hierarchy: which violations actually matter.

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

MODULE_NAME = "11_linear_models"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. OLS from scratch, eq. (11.2)-(11.5)

# %%
header("1. Normal equations, hat matrix, standard errors (11.2)-(11.5)")

rng = np.random.default_rng(1101)
n = 60
age = rng.uniform(25, 75, n)
treated = rng.binomial(1, 0.5, n)
batch = rng.integers(0, 3, n)
y = (2.0 + 0.05 * age + 1.4 * treated + 0.6 * (batch == 1)
     - 0.4 * (batch == 2) + rng.normal(0, 1.0, n))

df = pd.DataFrame({"y": y, "age": age, "treated": treated,
                   "batch": pd.Categorical(batch)})


def ols_from_scratch(X, y):
    """stats.md eq. (11.2)-(11.5), via QR (never form X'X explicitly)."""
    n, p = X.shape
    Q, R = np.linalg.qr(X)                      # numerically stable
    beta = np.linalg.solve(R, Q.T @ y)          # eq. (11.2)
    fitted = X @ beta
    resid = y - fitted
    rss = resid @ resid
    sigma2 = rss / (n - p)                      # eq. (11.4)
    XtX_inv = np.linalg.inv(R) @ np.linalg.inv(R).T
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    tvals = beta / se                           # eq. (11.5)
    H = Q @ Q.T                                 # eq. (11.3), the hat matrix
    return {"beta": beta, "se": se, "t": tvals,
            "p": 2 * st.t.sf(np.abs(tvals), n - p),
            "sigma2": sigma2, "rss": rss, "H": H,
            "leverage": np.diag(H), "resid": resid, "fitted": fitted,
            "XtX_inv": XtX_inv}


X = np.column_stack([np.ones(n), age, treated,
                     (batch == 1).astype(float), (batch == 2).astype(float)])
names = ["intercept", "age", "treated", "batch1", "batch2"]
mine = ols_from_scratch(X, y)

fit = smf.ols("y ~ age + treated + C(batch)", data=df).fit()
print(f"  {'term':<12}{'beta (mine)':>13}{'SE (mine)':>11}{'t':>9}"
      f"{'statsmodels beta':>19}")
for i, nm in enumerate(names):
    print(f"  {nm:<12}{mine['beta'][i]:>13.5f}{mine['se'][i]:>11.5f}"
          f"{mine['t'][i]:>9.3f}{fit.params.iloc[i]:>19.5f}")
print(f"\n  max |mine - statsmodels| = "
      f"{np.max(np.abs(mine['beta'] - fit.params.values)):.2e}")

# Hat-matrix properties.
H = mine["H"]
print(f"\n  Hat matrix (11.3): symmetric? {np.allclose(H, H.T)}   "
      f"idempotent? {np.allclose(H @ H, H)}   trace = {np.trace(H):.6f} "
      f"(= p = {X.shape[1]})")

# %% [markdown]
# ### Why QR, not the normal equations
#
# $\kappa(\mathbf X^\top\mathbf X)=\kappa(\mathbf X)^2$: forming the normal
# equations squares the condition number and loses half your digits.

# %%
print("\n--- Numerical stability on a nearly-collinear design ---")
t = np.linspace(0, 1, 40)
Xbad = np.column_stack([np.ones(40), t, t**2, t**3, t**4, t**5])
ybad = 1 + 2 * t + rng.normal(0, 0.01, 40)
print(f"  condition number of X      : {np.linalg.cond(Xbad):.3e}")
print(f"  condition number of X'X    : {np.linalg.cond(Xbad.T @ Xbad):.3e}")
b_normal = np.linalg.solve(Xbad.T @ Xbad, Xbad.T @ ybad)
b_qr = np.linalg.lstsq(Xbad, ybad, rcond=None)[0]
print(f"  max |normal-equations - QR|: {np.max(np.abs(b_normal - b_qr)):.3e}")
print("  statsmodels and R's lm() both use QR/SVD for exactly this reason.")

# %% [markdown]
# ## 2. Nested-model F test and $R^2$, eq. (11.6)-(11.7)

# %%
header("2. Partial F test (11.6); why R^2 cannot select models (11.7)")

fit_full = smf.ols("y ~ age + treated + C(batch)", data=df).fit()
fit_red = smf.ols("y ~ age + treated", data=df).fit()

p1, p0 = len(fit_full.params), len(fit_red.params)
F = ((fit_red.ssr - fit_full.ssr) / (p1 - p0)) / (fit_full.ssr / (n - p1))
p_val = st.f.sf(F, p1 - p0, n - p1)
print(f"  manual F (11.6) = {F:.4f}, df=({p1-p0}, {n-p1}), p = {p_val:.4f}")
sm_anova = sm.stats.anova_lm(fit_red, fit_full)
print(f"  statsmodels F   = {sm_anova['F'].iloc[1]:.4f}, "
      f"p = {sm_anova['Pr(>F)'].iloc[1]:.4f}")

print("\n  R^2 always increases when you add ANY predictor - even pure noise:")
d2 = df.copy()
prev_r2 = smf.ols("y ~ age", data=d2).fit().rsquared
print(f"    {'model':<34}{'R^2':>9}{'adj R^2':>10}")
print(f"    {'y ~ age':<34}{prev_r2:>9.4f}"
      f"{smf.ols('y ~ age', data=d2).fit().rsquared_adj:>10.4f}")
for k in range(1, 6):
    d2[f"noise{k}"] = rng.normal(0, 1, n)
    form = "y ~ age + " + " + ".join(f"noise{j}" for j in range(1, k + 1))
    f_ = smf.ols(form, data=d2).fit()
    print(f"    {'+ ' + str(k) + ' pure-noise predictors':<34}"
          f"{f_.rsquared:>9.4f}{f_.rsquared_adj:>10.4f}")
print("  Adjusted R^2 penalises p, but neither is a validation metric (Topic 31).")

# %% [markdown]
# ## 3. Confidence vs prediction intervals, eq. (11.8)-(11.9)
#
# The extra `1` in (11.9) is the irreducible noise of one new observation. The
# prediction interval does **not** shrink to zero as $n\to\infty$.

# %%
header("3. CI for the mean vs PI for an observation (11.8)-(11.9)")

fit_simple = smf.ols("y ~ age", data=df).fit()
x0 = pd.DataFrame({"age": [50.0]})
pred = fit_simple.get_prediction(x0)
ci = pred.conf_int(alpha=0.05)[0]
pi = pred.summary_frame(alpha=0.05)
print(f"  at age = 50: fitted = {pred.predicted_mean[0]:.4f}")
print(f"    95% CI for the MEAN  [{ci[0]:.4f}, {ci[1]:.4f}]  width "
      f"{ci[1]-ci[0]:.4f}")
print(f"    95% PI for an OBS    [{pi['obs_ci_lower'].iloc[0]:.4f}, "
      f"{pi['obs_ci_upper'].iloc[0]:.4f}]  width "
      f"{pi['obs_ci_upper'].iloc[0]-pi['obs_ci_lower'].iloc[0]:.4f}")

print("\n  As n grows, the CI shrinks to zero but the PI converges to ~ +/-1.96 sigma:")
for nn in [30, 300, 3000, 30000]:
    a = rng.uniform(25, 75, nn)
    yy = 2 + 0.05 * a + rng.normal(0, 1.0, nn)
    f_ = smf.ols("y ~ age", data=pd.DataFrame({"y": yy, "age": a})).fit()
    sfr = f_.get_prediction(x0).summary_frame(alpha=0.05)
    print(f"    n={nn:>6}: CI width {sfr['mean_ci_upper'].iloc[0]-sfr['mean_ci_lower'].iloc[0]:.4f}"
          f"   PI width {sfr['obs_ci_upper'].iloc[0]-sfr['obs_ci_lower'].iloc[0]:.4f}")

# %% [markdown]
# ## 4. Leverage, influence, collinearity, eq. (11.10)-(11.12)

# %%
header("4. Diagnostics: leverage, Cook's D, VIF (11.10)-(11.12)")

infl = fit_full.get_influence()
lev = infl.hat_matrix_diag
cooks = infl.cooks_distance[0]
p_ = len(fit_full.params)
print(f"  leverage rule of thumb 2p/n = {2*p_/n:.4f}: "
      f"{np.sum(lev > 2*p_/n)} points flagged")
print(f"  Cook's D rule of thumb 4/n  = {4/n:.4f}: "
      f"{np.sum(cooks > 4/n)} points flagged")
print(f"  max leverage = {lev.max():.4f}, max Cook's D = {cooks.max():.4f}")
print(f"  from-scratch leverage matches: "
      f"{np.allclose(np.sort(lev), np.sort(mine['leverage']))}")

print("\n  High leverage != influential. Both are needed:")
print(f"    {'point':<8}{'leverage':>10}{'|std resid|':>13}{'Cook D':>10}")
sr = infl.resid_studentized_internal
for i in np.argsort(-cooks)[:4]:
    print(f"    {i:<8}{lev[i]:>10.4f}{abs(sr[i]):>13.4f}{cooks[i]:>10.4f}")

# VIF
def vif(X, j):
    """stats.md eq. (11.12): 1/(1-R_j^2) from regressing column j on the rest."""
    others = np.delete(X, j, axis=1)
    beta = np.linalg.lstsq(others, X[:, j], rcond=None)[0]
    resid = X[:, j] - others @ beta
    r2 = 1 - resid.var() / X[:, j].var()
    return 1 / (1 - r2)


print("\n  VIF (11.12) for a design with an induced collinearity:")
z1 = rng.normal(0, 1, n)
z2 = z1 * 0.97 + rng.normal(0, 0.24, n)         # strongly collinear with z1
Xv = np.column_stack([np.ones(n), age, z1, z2])
for j, nm in zip([1, 2, 3], ["age", "z1", "z2"]):
    v = vif(Xv, j)
    print(f"    {nm:<6} VIF = {v:7.2f}  -> SE inflated {np.sqrt(v):.2f}x"
          f"{'   <-- serious' if v > 10 else ''}")

# %% [markdown]
# ## 5. Nonlinearity via splines, eq. (11.13)
#
# "Linear model" means linear in the **parameters**. A spline basis buys smooth
# nonlinearity while keeping every tool above.

# %%
header("5. Splines: linear in the parameters (11.13)")

rng = np.random.default_rng(1102)
t = np.sort(rng.uniform(0, 24, 120))                 # hours in a time course
true_curve = 3 + 2 * np.sin(t / 24 * 2 * np.pi) + 0.06 * t
expr = true_curve + rng.normal(0, 0.35, 120)
grp = np.tile([0, 1], 60)
expr = expr + 0.8 * grp * np.sin(t / 24 * 2 * np.pi)   # treatment x time effect
dt = pd.DataFrame({"expr": expr, "t": t, "grp": grp})

m_lin = smf.ols("expr ~ t", data=dt).fit()
m_spl = smf.ols("expr ~ bs(t, df=5)", data=dt).fit()
m_int = smf.ols("expr ~ bs(t, df=5) * C(grp)", data=dt).fit()

print(f"  {'model':<34}{'df':>5}{'RSS':>10}{'adj R^2':>10}")
for nm, m in [("expr ~ t (linear)", m_lin),
              ("expr ~ bs(t, df=5)", m_spl),
              ("expr ~ bs(t, df=5) * group", m_int)]:
    print(f"  {nm:<34}{int(m.df_model):>5}{m.ssr:>10.3f}{m.rsquared_adj:>10.4f}")

av = sm.stats.anova_lm(m_spl, m_int)
print(f"\n  Treatment x time interaction test (11.6): F = {av['F'].iloc[1]:.3f}, "
      f"p = {av['Pr(>F)'].iloc[1]:.3e}")
print("  This is the correct way to ask 'do the two groups follow different")
print("  smooth trajectories?' (stats.md Topic 28).")

# %% [markdown]
# ## 6. Which assumptions actually matter
#
# In order of severity: independence > correct mean structure >
# homoscedasticity > normality of residuals.

# %%
header("6. Ranking the assumption violations by damage done")

def fpr_under(violation, n_sim=2500, n=30, seed=0):
    """False-positive rate for a TRUE null slope under each violation."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        x = rng.normal(0, 1, n)
        if violation == "none":
            yy = rng.normal(0, 1, n)
        elif violation == "non-normal residuals":
            yy = rng.standard_t(3, n) * 0.7          # heavy tails, mean model OK
        elif violation == "heteroscedasticity":
            yy = rng.normal(0, 0.2 + 1.5 * np.abs(x), n)
        elif violation == "dependence (clusters)":
            # 6 clusters of 5; the cluster effect is correlated with x
            cl = np.repeat(rng.normal(0, 1.2, 6), 5)
            x = np.repeat(rng.normal(0, 1, 6), 5) + rng.normal(0, 0.3, n)
            yy = cl + rng.normal(0, 0.5, n)
        f = smf.ols("y ~ x", data=pd.DataFrame({"y": yy, "x": x})).fit()
        hits += f.pvalues["x"] < 0.05
    return hits / n_sim


print(f"  {'violation':<28}{'false-positive rate':>21}  (nominal 5%)")
for v in ["none", "non-normal residuals", "heteroscedasticity",
          "dependence (clusters)"]:
    print(f"  {v:<28}{fpr_under(v, seed=abs(hash(v)) % 997):>20.1%}")

print("\n  Non-normal residuals: barely matters at n=30 (the CLT, eq. 4.5).")
print("  Heteroscedasticity : distorts SEs - fix with HC3 (Module 05) or weights.")
print("  DEPENDENCE         : catastrophic, and INVISIBLE in residual plots.")
print("  Fix dependence by design or with a mixed model (Module 14).")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.scatter(fit_full.fittedvalues, fit_full.resid, s=18)
ax.axhline(0, color="k", lw=1)
lo = np.polyfit(fit_full.fittedvalues, fit_full.resid, 1)
ax.set_xlabel("fitted"); ax.set_ylabel("residual")
ax.set_title("Residuals vs fitted (linearity)", fontsize=9)

ax = axes[0, 1]
st.probplot(fit_full.resid / np.sqrt(fit_full.mse_resid), dist="norm", plot=ax)
ax.set_title("Normal Q-Q (least important assumption)", fontsize=9)

ax = axes[1, 0]
ax.scatter(lev, sr, s=18)
ax.axvline(2 * p_ / n, color="red", ls="--", label="$2p/n$")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("leverage $h_{ii}$"); ax.set_ylabel("studentised residual")
ax.set_title("Eq. (11.10)-(11.11): leverage vs residual", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
# Keep the grid strictly inside the observed range: a B-spline basis is
# undefined beyond its outermost knots, which is exactly the boundary
# behaviour that makes NATURAL splines preferable for extrapolation.
grid = pd.DataFrame({"t": np.linspace(dt["t"].min(), dt["t"].max(), 200)})
for g, c in [(0, "C0"), (1, "C1")]:
    sel = dt["grp"] == g
    ax.scatter(dt.loc[sel, "t"], dt.loc[sel, "expr"], s=12, color=c, alpha=0.6)
    gg = grid.copy(); gg["grp"] = g
    ax.plot(gg["t"], m_int.predict(gg), color=c, lw=2, label=f"group {g}")
ax.set_xlabel("time (h)"); ax.set_ylabel("expression")
ax.set_title("Eq. (11.13): spline x group interaction", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "linear_models.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/linear_models.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 11)
#
# 1. Write the design matrix and check its rank first.
# 2. Choose covariates from a causal diagram (Module 22), **not** from stepwise
#    selection, which invalidates all subsequent p-values.
# 3. Centre continuous predictors involved in interactions: $\beta_1$ is the
#    effect "when $x_2=0$", which may be outside the data.
# 4. Prefer robust SEs over dropping data when variance is non-constant.
#
# **Next:** `12_anova_and_contrasts.py`
