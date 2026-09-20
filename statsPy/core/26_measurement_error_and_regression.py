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
# # Module 26: Measurement error, attenuation, and regression to the mean
#
# **Curriculum link:** `stats.md` -> Topic 37, equations (37.1)-(37.6)
# **Assumes:** Modules 11, 21.
#
# ## What you will learn
#
# 1. Noise in a **predictor** biases its coefficient toward zero (37.2)-(37.3).
#    Noise in the **outcome** does not. The asymmetry decides where to spend
#    assay budget.
# 2. How to correct for it when reliability is known (37.4), and why the
#    correction inflates the standard error too.
# 3. That with several predictors, error in one biases the coefficients of the
#    others, in either direction.
# 4. **Regression to the mean** (37.5)-(37.6): selecting on an extreme value
#    guarantees an apparent effect with no treatment and no biology.

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

MODULE_NAME = "26_measurement_error_and_regression"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(26)


def slope(x, y):
    return sm.OLS(y, sm.add_constant(x)).fit().params[1]


# %% [markdown]
# ## 1. Noise in the predictor attenuates, eq. (37.1)-(37.3)

# %%
header("1. Error in X shrinks the slope; error in Y does not")

N, BETA, SD_X, SD_E = 4000, 2.0, 1.0, 1.0
print(f"  true slope = {BETA}\n")
print(f"  {'noise SD':<12}{'lambda (37.2)':>15}{'noise in X':>13}{'noise in Y':>13}"
      f"{'predicted':>12}")
for sd_u in (0.0, 0.5, 1.0, 2.0):
    bx, by = [], []
    for _ in range(300):
        x = rng.normal(0, SD_X, N)
        y = BETA * x + rng.normal(0, SD_E, N)
        bx.append(slope(x + rng.normal(0, sd_u, N), y))      # error in X
        by.append(slope(x, y + rng.normal(0, sd_u, N)))      # error in Y
    lam = SD_X ** 2 / (SD_X ** 2 + sd_u ** 2)                # eq. (37.2)
    print(f"  {sd_u:<12.1f}{lam:>15.3f}{np.mean(bx):>13.3f}{np.mean(by):>13.3f}"
          f"{lam * BETA:>12.3f}")

print("""
  The 'noise in X' column matches the prediction column exactly: the observed
  slope is lambda times the true slope, eq. (37.3). The 'noise in Y' column
  does not move at all.

  That asymmetry is worth internalising. If your predictor is an assay readout
  and your outcome is a clean clinical endpoint, your effect size is
  systematically too small and no amount of data fixes it. If it is the other
  way round, you have lost precision but not accuracy.""")

# %% [markdown]
# ## 2. What attenuation does to inference

# %%
header("2. Attenuation does not just shrink the estimate")

sd_u = 1.0
lam = SD_X ** 2 / (SD_X ** 2 + sd_u ** 2)
res = {"clean": [], "noisy": [], "power_clean": 0, "power_noisy": 0}
SMALL = 0.25                                   # a small but real effect
R = 600
for _ in range(R):
    x = rng.normal(0, SD_X, 200)
    y = SMALL * x + rng.normal(0, SD_E, 200)
    w = x + rng.normal(0, sd_u, 200)
    fc = sm.OLS(y, sm.add_constant(x)).fit()
    fn = sm.OLS(y, sm.add_constant(w)).fit()
    res["clean"].append(fc.params[1]); res["noisy"].append(fn.params[1])
    res["power_clean"] += fc.pvalues[1] < 0.05
    res["power_noisy"] += fn.pvalues[1] < 0.05

print(f"  true effect {SMALL}, reliability lambda = {lam:.2f}, n = 200\n")
print(f"  {'predictor':<22}{'mean estimate':>15}{'power':>9}")
print(f"  {'measured without error':<22}{np.mean(res['clean']):>15.3f}"
      f"{res['power_clean']/R:>9.2f}")
print(f"  {'measured with error':<22}{np.mean(res['noisy']):>15.3f}"
      f"{res['power_noisy']/R:>9.2f}")

print(f"""
  Attenuation costs you twice. The estimate shrinks toward zero, AND the power
  to detect it falls, because the signal has been diluted while the outcome
  noise stayed put.

  A study that reports "no significant association" with an assay-based
  predictor may have had a real association it could not see. Report the
  assay's reliability so a reader can judge.""")

# %% [markdown]
# ## 3. Correcting with a known reliability, eq. (37.4)
#
# Replicate measurements give you $\lambda$, and $\lambda$ gives you the
# correction. The corrected estimate is unbiased; it is also less precise, and
# the interval must say so.

# %%
header("3. Correcting for attenuation (37.4)")

sd_u = 1.0
lam_true = SD_X ** 2 / (SD_X ** 2 + sd_u ** 2)
ests, cors, cover_naive, cover_corr = [], [], 0, 0
for _ in range(1000):
    x = rng.normal(0, SD_X, 400)
    y = BETA * x + rng.normal(0, SD_E, 400)
    # Two replicate measurements of the same true x.
    w1 = x + rng.normal(0, sd_u, 400)
    w2 = x + rng.normal(0, sd_u, 400)
    # Reliability estimated from the replicates, not assumed.
    lam_hat = np.corrcoef(w1, w2)[0, 1]
    f = sm.OLS(y, sm.add_constant(w1)).fit()
    b, se = f.params[1], f.bse[1]
    ests.append(b)
    cors.append(b / lam_hat)
    cover_naive += abs(b - BETA) < 1.96 * se
    cover_corr += abs(b / lam_hat - BETA) < 1.96 * se / lam_hat   # eq. (37.4)

print(f"  true slope {BETA}, true lambda {lam_true:.3f}\n")
print(f"  {'estimator':<34}{'mean':>9}{'bias':>9}{'95% CI coverage':>18}")
print(f"  {'naive (ignore the error)':<34}{np.mean(ests):>9.3f}"
      f"{np.mean(ests)-BETA:>9.3f}{cover_naive/1000:>18.1%}")
print(f"  {'corrected by lambda-hat (37.4)':<34}{np.mean(cors):>9.3f}"
      f"{np.mean(cors)-BETA:>9.3f}{cover_corr/1000:>18.1%}")

print("""
  The naive interval has essentially zero coverage: it is a tight interval in
  the wrong place, which is the worst combination. Dividing by lambda recovers
  the truth, and dividing the standard error by lambda as well brings coverage
  most of the way back.

  It does not reach 95%, and the shortfall is informative. The corrected
  interval treats lambda-hat as though it were the known lambda, so it ignores
  the uncertainty in the reliability estimate itself. Problem 3 shows what
  happens when that uncertainty is large.

  Note what supplied lambda: a second measurement of the same samples. Without
  replicates you cannot estimate reliability, and without reliability you
  cannot correct. Build replicates into the design.""")

# %% [markdown]
# ## 4. With several predictors, it is not just shrinkage

# %%
header("4. Error in one predictor biases the others")

print("  Two correlated predictors. Only X1 is measured with error.\n")
print(f"  {'corr(X1,X2)':<14}{'beta1 true':>11}{'beta1 est':>11}"
      f"{'beta2 true':>11}{'beta2 est':>11}")
for rho in (0.0, 0.5, 0.8):
    b1s, b2s = [], []
    for _ in range(400):
        z = rng.normal(0, 1, (1500, 2))
        x1 = z[:, 0]
        x2 = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
        y = 1.0 * x1 + 0.0 * x2 + rng.normal(0, 1, 1500)
        w1 = x1 + rng.normal(0, 1.0, 1500)          # only X1 is noisy
        f = sm.OLS(y, sm.add_constant(np.column_stack([w1, x2]))).fit()
        b1s.append(f.params[1]); b2s.append(f.params[2])
    print(f"  {rho:<14.1f}{1.0:>11.2f}{np.mean(b1s):>11.3f}{0.0:>11.2f}"
          f"{np.mean(b2s):>11.3f}")

print("""
  X2 is measured perfectly and has NO true effect, yet its coefficient grows
  as the correlation rises. The model, unable to see the true X1, uses X2 as a
  partial proxy for it.

  So "measurement error only attenuates" is true for simple regression and
  false the moment you adjust for anything. A clean covariate can absorb the
  effect of a noisy one and appear important. This is a real mechanism behind
  spurious 'independent predictors' in clinical models.""")

# %% [markdown]
# ## 5. Regression to the mean, eq. (37.5)-(37.6)
#
# No treatment, no biology, no effect. Select on an extreme baseline and the
# follow-up moves toward the mean by an amount you can predict exactly.

# %%
header("5. Regression to the mean (37.5)-(37.6)")

MU, SD = 100.0, 15.0
n = 20000
print(f"  A biomarker with mean {MU} and SD {SD}, measured twice.")
print(f"  NOTHING happens between the measurements.\n")
print(f"  {'reliability rho':<18}{'selected top 10%':>19}{'observed change':>18}"
      f"{'predicted (37.6)':>19}")
for rho in (0.9, 0.7, 0.5, 0.3):
    true = rng.normal(MU, SD * np.sqrt(rho), n)
    y1 = true + rng.normal(0, SD * np.sqrt(1 - rho), n)
    y2 = true + rng.normal(0, SD * np.sqrt(1 - rho), n)
    sel = y1 >= np.quantile(y1, 0.90)
    obs = (y2[sel] - y1[sel]).mean()
    pred = (rho - 1) * (y1[sel].mean() - MU)                  # eq. (37.6)
    print(f"  {rho:<18.1f}{y1[sel].mean():>19.1f}{obs:>18.2f}{pred:>19.2f}")

print("""
  The observed change matches eq. (37.6) at every reliability. A study that
  enrolled the top decile and reported an average drop would be reporting
  arithmetic, not pharmacology.

  Note the direction of the dependence: the effect is LARGEST when rho is
  SMALL, that is, when the assay is noisiest. Poor measurement does not merely
  add noise here, it creates a systematic apparent effect.

  The fix is a control group selected the same way. Both arms regress by the
  same amount and the difference between them is clean. A single-arm
  before-and-after study in a selected group cannot be rescued by analysis.""")

# %% [markdown]
# ## 6. The same trap inside a model: adjusting for baseline

# %%
header("6. Change scores versus adjusting for baseline")

R, n = 500, 300
res = {"change": [], "ancova": []}
for _ in range(R):
    true = rng.normal(MU, SD * np.sqrt(0.6), n)
    y1 = true + rng.normal(0, SD * np.sqrt(0.4), n)
    arm = rng.integers(0, 2, n)
    # A real treatment effect of -3, plus regression to the mean for everyone.
    y2 = true + rng.normal(0, SD * np.sqrt(0.4), n) - 3.0 * arm
    d = y2 - y1
    res["change"].append(sm.OLS(d, sm.add_constant(arm.astype(float))).fit().params[1])
    X = sm.add_constant(np.column_stack([arm.astype(float), y1]))
    res["ancova"].append(sm.OLS(y2, X).fit().params[1])

print(f"  true treatment effect = -3.00, randomised arms\n")
print(f"  {'analysis':<40}{'estimate':>11}{'SD':>9}")
print(f"  {'change score (y2 - y1)':<40}{np.mean(res['change']):>11.3f}"
      f"{np.std(res['change']):>9.3f}")
print(f"  {'ANCOVA: y2 ~ arm + y1':<40}{np.mean(res['ancova']):>11.3f}"
      f"{np.std(res['ancova']):>9.3f}")

print("""
  Both are unbiased here, because the arms were RANDOMISED, so baseline
  imbalance is only chance. ANCOVA is more precise, and that is the standard
  reason to prefer it.

  The picture changes entirely if allocation depended on the baseline. Then
  the change score is biased by regression to the mean and ANCOVA is not, and
  the gap can be large. Randomise, and adjust for baseline in the model rather
  than selecting on it.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

sds = np.linspace(0, 2.5, 30)
lams = SD_X ** 2 / (SD_X ** 2 + sds ** 2)
obs = []
for s_ in sds:
    x = rng.normal(0, SD_X, 4000)
    y = BETA * x + rng.normal(0, SD_E, 4000)
    obs.append(slope(x + rng.normal(0, s_, 4000), y))
ax[0].plot(sds, obs, "o", ms=4, color="steelblue", label="simulated")
ax[0].plot(sds, BETA * lams, color="firebrick", lw=2, label=r"$\lambda\beta$ (37.3)")
ax[0].set_xlabel("SD of measurement error in X"); ax[0].set_ylabel("observed slope")
ax[0].set_title("Attenuation"); ax[0].legend(fontsize=8)

true = rng.normal(MU, SD * np.sqrt(0.5), 4000)
y1 = true + rng.normal(0, SD * np.sqrt(0.5), 4000)
y2 = true + rng.normal(0, SD * np.sqrt(0.5), 4000)
sel = y1 >= np.quantile(y1, 0.90)
ax[1].scatter(y1[~sel], y2[~sel], s=3, alpha=0.15, color="grey")
ax[1].scatter(y1[sel], y2[sel], s=5, alpha=0.5, color="firebrick")
lims = [y1.min(), y1.max()]
ax[1].plot(lims, lims, "k--", lw=1)
ax[1].axhline(MU, color="steelblue", lw=1)
ax[1].set_xlabel("baseline"); ax[1].set_ylabel("follow-up")
ax[1].set_title("Selected on baseline (red), no treatment")

rhos = np.linspace(0.1, 0.95, 25)
shift = []
for r_ in rhos:
    t = rng.normal(MU, SD * np.sqrt(r_), 8000)
    a = t + rng.normal(0, SD * np.sqrt(1 - r_), 8000)
    b = t + rng.normal(0, SD * np.sqrt(1 - r_), 8000)
    s_ = a >= np.quantile(a, 0.90)
    shift.append((b[s_] - a[s_]).mean())
ax[2].plot(rhos, shift, "o-", color="firebrick")
ax[2].axhline(0, color="black", lw=1)
ax[2].set_xlabel(r"reliability $\rho$"); ax[2].set_ylabel("apparent change")
ax[2].set_title("Worse assays, bigger fake effect")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "measurement_error.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'measurement_error.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Where should the assay budget go?
#
# You can afford to halve the measurement error on either the predictor or the
# outcome, but not both. Which buys more, and does the answer depend on what
# you are trying to do?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def study(sd_ux, sd_uy, beta=0.4, n=250, R=800):
#     est, hit = [], 0
#     for _ in range(R):
#         x = rng.normal(0, 1, n)
#         y = beta * x + rng.normal(0, 1, n)
#         f = sm.OLS(y + rng.normal(0, sd_uy, n),
#                    sm.add_constant(x + rng.normal(0, sd_ux, n))).fit()
#         est.append(f.params[1]); hit += f.pvalues[1] < 0.05
#     return np.mean(est), hit / R
#
# print(f"  true beta = 0.40\n")
# print(f"  {'scenario':<34}{'estimate':>11}{'bias':>9}{'power':>9}")
# for lab, sx, sy in (("both assays noisy", 1.0, 1.0),
#                     ("halve error in the PREDICTOR", 0.5, 1.0),
#                     ("halve error in the OUTCOME", 1.0, 0.5)):
#     e, p_ = study(sx, sy)
#     print(f"  {lab:<34}{e:>11.3f}{e-0.4:>9.3f}{p_:>9.2f}")
#
# # Improving the PREDICTOR reduces bias and raises power. Improving the
# # OUTCOME leaves the estimate where it was and raises power too, sometimes by
# # more, because outcome noise sits directly in the residual variance.
# #
# # So the answer depends on the goal, which is the real lesson:
# #   if you want an unbiased EFFECT SIZE, spend on the predictor;
# #   if you only want to DETECT the effect, spend wherever the variance is
# #   larger, which is often the outcome.
# #
# # Most papers want the effect size and most budgets are spent on the outcome.

# %% [markdown]
# ### Problem 2: Manufacture a treatment effect from nothing
#
# Design a single-arm study that reports a large, highly significant
# improvement for a treatment that does nothing at all. Then show which design
# change removes it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# n, rho = 400, 0.5
# t = rng.normal(MU, SD * np.sqrt(rho), n)
# base = t + rng.normal(0, SD * np.sqrt(1 - rho), n)
# after = t + rng.normal(0, SD * np.sqrt(1 - rho), n)      # NO treatment effect
# elig = base >= np.quantile(base, 0.80)                   # "enrol the worst 20%"
# d = after[elig] - base[elig]
# tt = st.ttest_1samp(d, 0)
# print(f"  Single-arm trial, enrolled the worst 20% at baseline, n = {elig.sum()}")
# print(f"    mean change    = {d.mean():+.2f}")
# print(f"    paired t-test  p = {tt.pvalue:.2e}")
# print(f"    Cohen's d      = {d.mean()/d.std(ddof=1):.2f}")
# print(f"    TRUE effect    = 0.00")
#
# # Now add a control group selected by exactly the same rule.
# arm = rng.integers(0, 2, n)
# d_all = after - base
# ctl = d_all[elig & (arm == 0)]; trt = d_all[elig & (arm == 1)]
# tt2 = st.ttest_ind(trt, ctl)
# print(f"\n  Same data, randomised control selected the same way:")
# print(f"    treated change = {trt.mean():+.2f}, control change = {ctl.mean():+.2f}")
# print(f"    difference     = {trt.mean()-ctl.mean():+.2f}, p = {tt2.pvalue:.3f}")
#
# # The single-arm study reports a large, overwhelmingly significant
# # improvement for a treatment with no effect. Every number in it is correct;
# # the design is what is wrong.
# #
# # The control arm regresses by the same amount, so the difference is clean.
# # This is why single-arm before-and-after studies in selected patients are
# # weak evidence however large the p-value, and it is the mechanism behind a
# # great many "promising pilot study" results that later fail.

# %% [markdown]
# ### Problem 3: Does correcting for attenuation ever hurt?
#
# The correction in (37.4) divides by an estimated $\lambda$. Investigate what
# happens when $\hat\lambda$ is itself noisy, as it is with few replicates.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  true slope {BETA}, true lambda {lam_true:.3f}\n")
# print(f"  {'n used to estimate lambda':<30}{'mean corrected':>16}"
#       f"{'SD':>9}{'worst run':>12}")
# N_STUDY = 2000                 # the main study, held fixed across rows
# for n_rel in (20, 50, 200, 2000):
#     cor = []
#     for _ in range(600):
#         x = rng.normal(0, SD_X, N_STUDY)
#         y = BETA * x + rng.normal(0, SD_E, N_STUDY)
#         w1 = x + rng.normal(0, 1.0, N_STUDY)
#         # lambda estimated from a replicate subset of only n_rel samples
#         idx = rng.choice(N_STUDY, n_rel, replace=False)
#         w2 = x[idx] + rng.normal(0, 1.0, n_rel)
#         lam_hat = np.clip(np.corrcoef(w1[idx], w2)[0, 1], 0.05, 1.0)
#         cor.append(sm.OLS(y, sm.add_constant(w1)).fit().params[1] / lam_hat)
#     cor = np.array(cor)
#     print(f"  {n_rel:<30}{cor.mean():>16.3f}{cor.std():>9.3f}{cor.max():>12.2f}")
#
# # With plenty of replicate pairs the correction is well behaved. With few, it
# # becomes unstable and skewed upward, because lambda-hat appears in a
# # DENOMINATOR: a lambda that happens to come out small produces an enormous
# # corrected slope. The mean is dragged by those runs and the worst run can be
# # far from the truth.
# #
# # This is a general hazard of ratio estimators and the reason to prefer
# # methods that model the error directly, such as SIMEX, structural equation
# # models, or a Bayesian measurement-error model, over dividing by a noisy
# # constant. It is also an argument for estimating reliability from a
# # dedicated, adequately sized reproducibility experiment rather than from a
# # handful of incidental duplicates.

# %% [markdown]
# ## What to take away
#
# 1. Noise in a **predictor** biases toward zero by $\lambda$ (37.2)-(37.3).
#    Noise in the **outcome** costs precision, not accuracy.
# 2. Attenuation costs estimate and power together, so "not significant" with
#    a noisy predictor is weak evidence of no effect.
# 3. Correct with (37.4) only if reliability came from real replicates, and
#    inflate the standard error as well.
# 4. With multiple predictors, error in one distorts the others. A clean
#    covariate can steal the credit from a noisy one.
# 5. **Selecting on an extreme baseline guarantees an apparent effect**, and
#    the worse the assay the bigger it is. Use a control group selected the
#    same way.
#
# **Next:** `27_meta_analysis.py`
