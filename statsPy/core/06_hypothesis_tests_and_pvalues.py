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
# # Module 06: Hypothesis tests and p-values
#
# **Curriculum link:** `stats.md` -> Topic 6, equations (6.1)-(6.7)
#
# ## What you will learn
#
# 1. What probability a p-value reports (6.1): and the three things it does not.
# 2. Uniformity under the null (6.2) and **the p-value histogram**, the single
#    most useful diagnostic in high-throughput biology.
# 3. False-positive risk (6.4): why "significant" often means "probably wrong".
# 4. Type S and Type M errors (6.5): the winner's curse.
# 5. Combining p-values (6.6)-(6.7).

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "06_hypothesis_tests_and_pvalues"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Uniformity under the null, eq. (6.2)
#
# $$p \mid H_0 \sim \mathrm{Uniform}(0,1)$$
#
# Everything in Topic 8 (multiple testing) rests on this. Let us confirm it, and
# then see what it looks like when assumptions are violated.

# %%
header("1. p-values are uniform under the null (6.2)")

rng = np.random.default_rng(601)

def pvalues_under_null(n_tests=20000, n=8, seed=0, kind="clean"):
    """Generate p-values from a true null under several scenarios."""
    rng = np.random.default_rng(seed)
    if kind == "clean":
        a = rng.normal(0, 1, (n_tests, n))
        b = rng.normal(0, 1, (n_tests, n))
        return st.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    if kind == "unmodelled_batch":
        # A batch effect shared by the first half of each group, ignored by the
        # test. The null is still "no condition effect" but the model is wrong.
        batch = rng.normal(0, 1.2, (n_tests, 1))
        a = rng.normal(0, 1, (n_tests, n)) + batch
        b = rng.normal(0, 1, (n_tests, n)) - batch
        return st.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    if kind == "discrete":
        # Small-count Fisher exact tests: p-values are discrete and conservative.
        out = np.empty(n_tests)
        for i in range(n_tests):
            tbl = rng.binomial(10, 0.3, size=(2, 2)) + 1
            out[i] = st.fisher_exact(tbl).pvalue
        return out
    raise ValueError(kind)


for kind in ["clean", "unmodelled_batch", "discrete"]:
    p = pvalues_under_null(5000, seed=hash(kind) % 999, kind=kind)
    ks = st.kstest(p, "uniform")
    print(f"  {kind:<18} P(p<0.05) = {np.mean(p < 0.05):.4f}   "
          f"KS vs Uniform p = {ks.pvalue:.2e}")

print("\n  clean            -> ~0.05, uniform. The test is calibrated.")
print("  unmodelled_batch -> inflated. A small p indicts the WHOLE model")
print("                      (eq. 6.1 conditions on every assumption), not")
print("                      just the null of no condition effect.")
print("  discrete         -> conservative: P(p<0.05) < 0.05.")

# %% [markdown]
# ### How to read a p-value histogram
#
# Do this before adjusting anything (Topic 8). The shape tells you what is wrong.

# %%
header("Diagnosing four p-value histograms")

rng = np.random.default_rng(602)
G = 8000

scenarios = {}
# (a) Healthy: mostly null, some real signal.
null_p = np.random.default_rng(1).uniform(0, 1, int(0.85 * G))
alt_z = np.random.default_rng(2).normal(3.5, 1, G - len(null_p))
scenarios["healthy (flat + spike at 0)"] = np.concatenate(
    [null_p, 2 * st.norm.sf(np.abs(alt_z))])
# (b) Anticonservative: unmodelled dependence.
scenarios["anticonservative (all small)"] = st.beta.rvs(
    0.3, 1, size=G, random_state=3)
# (c) Conservative: discrete/over-corrected.
scenarios["conservative (hump near 1)"] = st.beta.rvs(
    1, 0.4, size=G, random_state=4)
# (d) Hump in the middle: wrong null or correlated features.
scenarios["mid-hump (wrong null)"] = np.clip(
    np.random.default_rng(5).normal(0.5, 0.13, G), 1e-6, 1 - 1e-6)

print(f"{'histogram shape':<32}{'P(p<0.05)':>11}{'P(p>0.9)':>10}  interpretation")
interp = {
    "healthy (flat + spike at 0)": "proceed to FDR",
    "anticonservative (all small)": "fix the model first",
    "conservative (hump near 1)": "discreteness/over-correction",
    "mid-hump (wrong null)": "wrong null or strong dependence",
}
for name, p in scenarios.items():
    print(f"{name:<32}{np.mean(p < 0.05):>11.3f}{np.mean(p > 0.9):>10.3f}  "
          f"{interp[name]}")

print("\n  RULE: adjusting a pathological histogram produces confident nonsense.")

# %% [markdown]
# ## 2. False-positive risk, eq. (6.4)
#
# $$\mathrm{PPV}=\frac{(1-\beta)\pi}{(1-\beta)\pi+\alpha(1-\pi)}$$
#
# The probability that a "significant" finding is real depends on the **prior**
# probability that the hypothesis was true: which p-values say nothing about.

# %%
header("2. What fraction of 'significant' findings are real? (6.4)")

def ppv(power, prior, alpha=0.05):
    """stats.md eq. (6.4)."""
    return power * prior / (power * prior + alpha * (1 - prior))


print(f"{'prior P(H1)':>12}" + "".join(f"{f'power={p}':>13}"
                                       for p in [0.2, 0.5, 0.8, 0.95]))
for prior in [0.01, 0.05, 0.10, 0.30, 0.50]:
    row = "".join(f"{ppv(p, prior):>12.1%} " for p in [0.2, 0.5, 0.8, 0.95])
    print(f"{prior:>12.2f}{row}")

print("\n  Worked example from stats.md: prior = 0.10, power = 0.5, alpha = 0.05")
print(f"    PPV = {ppv(0.5, 0.10):.1%}  -> nearly half of 'significant' results")
print("    are false, with PERFECT statistical practice.")
print("\n  Lowering alpha helps, but raising POWER helps more and also fixes")
print("  the Type M problem below.")

# Verify by direct simulation of a screen.
rng = np.random.default_rng(603)
G, PRIOR, EFFECT, n = 20000, 0.10, 1.2, 10
is_alt = rng.random(G) < PRIOR
delta = np.where(is_alt, EFFECT, 0.0)
a = rng.normal(delta[:, None], 1.0, (G, n))
b = rng.normal(0.0, 1.0, (G, n))
p = st.ttest_ind(a, b, axis=1, equal_var=False).pvalue
sig = p < 0.05
emp_power = np.mean(sig[is_alt])
print(f"\n  Simulated screen: G={G}, prior={PRIOR}, per-test power="
      f"{emp_power:.2f}")
print(f"    observed PPV among significant = {np.mean(is_alt[sig]):.1%}")
print(f"    predicted by eq. (6.4)         = {ppv(emp_power, PRIOR):.1%}")

# %% [markdown]
# ## 3. Type S and Type M errors, eq. (6.5): the winner's curse

# %%
header("3. Conditioning on significance exaggerates effects (6.5)")

rng = np.random.default_rng(604)
SIGMA = 1.0
print(f"{'true effect':>12}{'power':>9}{'Type S':>9}{'Type M (exaggeration)':>24}")
for true_effect in [0.1, 0.2, 0.5, 1.0, 2.0]:
    n = 10
    est = rng.normal(true_effect, SIGMA * np.sqrt(2 / n), 60000)
    se = SIGMA * np.sqrt(2 / n)
    sig = np.abs(est / se) > 1.96
    power = sig.mean()
    if sig.sum() > 0:
        type_s = np.mean(np.sign(est[sig]) != np.sign(true_effect))
        type_m = np.mean(np.abs(est[sig])) / abs(true_effect)
    else:
        type_s = type_m = np.nan
    print(f"{true_effect:>12.2f}{power:>9.1%}{type_s:>9.1%}{type_m:>23.2f}x")

print("\n  At low power the published effect is 2-5x too large, and can even have")
print("  the WRONG SIGN. This is why underpowered findings 'fail to replicate'")
print("  even when the effect is genuinely non-zero.")

# %% [markdown]
# ## 4. One-sided vs two-sided

# %%
header("4. Sidedness is a design decision, not a data decision")

rng = np.random.default_rng(605)
n, TRUE = 12, 0.6
p_two, p_right, p_wrong_side = [], [], []
for _ in range(5000):
    x = rng.normal(TRUE, 1, n)
    p_two.append(st.ttest_1samp(x, 0).pvalue)
    p_right.append(st.ttest_1samp(x, 0, alternative="greater").pvalue)
    p_wrong_side.append(st.ttest_1samp(x, 0, alternative="less").pvalue)
print(f"  true effect = +{TRUE}, n = {n}")
print(f"    two-sided power                  : {np.mean(np.array(p_two)<0.05):.1%}")
print(f"    one-sided, correct direction     : {np.mean(np.array(p_right)<0.05):.1%}")
print(f"    one-sided, wrong direction       : {np.mean(np.array(p_wrong_side)<0.05):.1%}")
print("\n  A one-sided test gains power in the predicted direction and has ZERO")
print("  power against the opposite one. Choose the side before seeing data.")

# %% [markdown]
# ## 5. Combining p-values, eq. (6.6)-(6.7)

# %%
header("5. Fisher and Stouffer combination (6.6)-(6.7)")

def fisher_combine(pvals):
    """stats.md eq. (6.6): X2 = -2 sum log p ~ chi2_2K."""
    stat = -2.0 * np.sum(np.log(pvals))
    return stat, st.chi2.sf(stat, 2 * len(pvals))


def stouffer_combine(pvals, weights=None, signs=None):
    """stats.md eq. (6.7): weighted z, preserving direction."""
    z = st.norm.isf(np.asarray(pvals))
    if signs is not None:
        z = z * np.asarray(signs)
    w = np.ones_like(z) if weights is None else np.asarray(weights, float)
    stat = np.sum(w * z) / np.sqrt(np.sum(w ** 2))
    return stat, st.norm.sf(stat)


studies = pd.DataFrame({
    "study": ["A", "B", "C", "D"],
    "n": [20, 150, 40, 300],
    "p_one_sided": [0.12, 0.04, 0.20, 0.03],
    "direction": [+1, +1, -1, +1],       # study C points the OTHER way
})
print(studies.to_string(index=False))

f_stat, f_p = fisher_combine(studies["p_one_sided"].values)
s_stat, s_p = stouffer_combine(studies["p_one_sided"].values,
                               weights=np.sqrt(studies["n"].values))
s_dir_stat, s_dir_p = stouffer_combine(
    studies["p_one_sided"].values, weights=np.sqrt(studies["n"].values),
    signs=studies["direction"].values)

print(f"\n  Fisher (6.6)                     : X2={f_stat:.3f}, p={f_p:.5f}")
print(f"  Stouffer, weights sqrt(n)        : z ={s_stat:.3f}, p={s_p:.5f}")
print(f"  Stouffer, direction-aware        : z ={s_dir_stat:.3f}, p={s_dir_p:.5f}")
print("\n  Fisher ignores direction entirely, so four studies pointing in four")
print("  different directions can still look 'significant'. For meta-analysis")
print("  across studies, use the direction-aware version - or better, combine")
print("  EFFECT SIZES rather than p-values (Topic 5).")

# %% [markdown]
# ## 6. Figure: the four p-value histograms

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, (name, p) in zip(axes.ravel()[:4], scenarios.items()):
    ax.hist(p, bins=40, range=(0, 1), color="steelblue", edgecolor="white")
    ax.axhline(len(p) / 40, color="red", ls="--", lw=1, label="uniform")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("p-value")
    ax.legend(fontsize=7)

# PPV curve
priors = np.linspace(0.005, 0.5, 200)
for pw in [0.2, 0.5, 0.8]:
    axes[1, 1].plot(priors, ppv(pw, priors), label=f"power={pw}")
axes[1, 1].axhline(0.5, color="k", ls=":")
axes[1, 1].set_xlabel("prior P(H1 true)")
axes[1, 1].set_ylabel("PPV")
axes[1, 1].set_title("Eq. (6.4): false-positive risk", fontsize=9)
axes[1, 1].legend(fontsize=7)

# Type M
rng = np.random.default_rng(606)
effs = np.linspace(0.05, 2.0, 40)
tm = []
for te in effs:
    est = rng.normal(te, np.sqrt(2 / 10), 30000)
    sig = np.abs(est / np.sqrt(2 / 10)) > 1.96
    tm.append(np.mean(np.abs(est[sig])) / te if sig.sum() else np.nan)
axes[1, 2].plot(effs, tm, lw=2)
axes[1, 2].axhline(1, color="k", ls="--")
axes[1, 2].set_xlabel("true effect size")
axes[1, 2].set_ylabel("exaggeration ratio")
axes[1, 2].set_title("Eq. (6.5): the winner's curse", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "pvalues.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/pvalues.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 6)
#
# 1. Report the exact p-value alongside the estimate and interval. Never stars,
#    never "n.s.".
# 2. "$p>0.05$" means the data are compatible with $H_0$: and with many
#    non-zero effects. Use TOST (Module 05) to claim equivalence.
# 3. Decide sidedness, stopping rule, covariates, filtering and the multiplicity
#    family **before** the analysis.
# 4. Check the p-value histogram before adjusting anything.
#
# ## Self-check
#
# * A p-value of 0.001: what is it the probability of? *(Of a statistic at
#   least this extreme, IF the null and every model assumption hold.)*
# * Prior 5%, power 40%, alpha = 0.05. What fraction of your hits are real?
#   *(`ppv(0.4, 0.05)` ~= 29.6%.)*
#
# **Next:** `07_ttests_ranks_permutation.py`
