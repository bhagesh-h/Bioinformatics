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
# # Module 12: ANOVA, factorial designs, and contrasts
#
# **Curriculum link:** `stats.md` -> Topic 12, equations (12.1)-(12.7)
#
# ## What you will learn
#
# 1. The sum-of-squares decomposition (12.1) is Pythagoras, and $F=t^2$ for two
#    groups (12.2).
# 2. **Contrasts (12.3) are where the science lives**: how to build and test them.
# 3. Coding schemes change parameter *meaning*, not fit.
# 4. Interaction as a difference of differences (12.4)-(12.5), and why main
#    effects become ambiguous when it is present.
# 5. Type I/II/III sums of squares, and why the argument is usually a
#    distraction.
# 6. Effect sizes (12.6) and post-hoc multiplicity (12.7).

# %%
import os
import itertools

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "12_anova_and_contrasts"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. The decomposition and the omnibus F, eq. (12.1)-(12.2)

# %%
header("1. SS decomposition and F (12.1)-(12.2)")

rng = np.random.default_rng(1201)
# A dose-response experiment: 4 ordered doses, 8 replicates each.
doses = np.array([0, 1, 2, 3])
true_means = np.array([10.0, 10.6, 11.6, 13.0])      # roughly linear in dose
n_per = 8
y = np.concatenate([rng.normal(m, 1.2, n_per) for m in true_means])
g = np.repeat([f"D{d}" for d in doses], n_per)
df = pd.DataFrame({"y": y, "dose": g,
                   "dose_num": np.repeat(doses, n_per)})

grand = y.mean()
group_means = df.groupby("dose")["y"].mean()
counts = df.groupby("dose")["y"].count()

ss_total = np.sum((y - grand) ** 2)
ss_between = np.sum(counts.values * (group_means.values - grand) ** 2)
ss_within = sum(((df.loc[df["dose"] == lvl, "y"] - group_means[lvl]) ** 2).sum()
                for lvl in group_means.index)

k, N = len(group_means), len(y)
F = (ss_between / (k - 1)) / (ss_within / (N - k))
p = st.f.sf(F, k - 1, N - k)

print(f"  SS_total   = {ss_total:9.4f}")
print(f"  SS_between = {ss_between:9.4f}")
print(f"  SS_within  = {ss_within:9.4f}")
print(f"  sum        = {ss_between + ss_within:9.4f}   <- eq. (12.1) holds exactly")
print(f"\n  F({k-1}, {N-k}) = {F:.4f}, p = {p:.3e}")
print(f"  scipy f_oneway: F = "
      f"{st.f_oneway(*[df.loc[df['dose']==l,'y'] for l in group_means.index]).statistic:.4f}")

# For k=2, F = t^2 exactly.
two = df[df["dose"].isin(["D0", "D3"])]
a = two.loc[two["dose"] == "D0", "y"].values
b = two.loc[two["dose"] == "D3", "y"].values
print(f"\n  k=2 check: t^2 = {st.ttest_ind(a,b,equal_var=True).statistic**2:.6f}"
      f"   F = {st.f_oneway(a,b).statistic:.6f}   <- eq. (B.6)")

# %% [markdown]
# ## 2. Contrasts: where the science lives, eq. (12.3)
#
# $$L=\sum_j c_j\mu_j,\ \sum_j c_j=0; \qquad
#   \widehat{\operatorname{Var}}(\hat L)=\hat\sigma^2\sum_j\frac{c_j^2}{n_j}$$

# %%
header("2. Building and testing contrasts (12.3)")

def test_contrast(df, group_col, value_col, weights, label=""):
    """stats.md eq. (12.3): estimate, SE, t and p for a contrast of group means."""
    levels = sorted(df[group_col].unique())
    means = df.groupby(group_col)[value_col].mean()[levels].values
    ns = df.groupby(group_col)[value_col].count()[levels].values
    k = len(levels)
    N = ns.sum()
    # Pooled within-group variance = MS_within
    ss_w = sum(((df.loc[df[group_col] == l, value_col]
                 - means[i]) ** 2).sum() for i, l in enumerate(levels))
    ms_w = ss_w / (N - k)
    c = np.asarray(weights, float)
    assert abs(c.sum()) < 1e-10, "contrast weights must sum to zero"
    est = float(c @ means)
    se = float(np.sqrt(ms_w * np.sum(c**2 / ns)))
    t = est / se
    dfree = N - k
    return {"label": label, "estimate": est, "se": se, "t": t,
            "df": dfree, "p": 2 * st.t.sf(abs(t), dfree),
            "ci": (est - st.t.ppf(0.975, dfree) * se,
                   est + st.t.ppf(0.975, dfree) * se)}


contrasts = {
    "any treated vs control":  [-1, 1/3, 1/3, 1/3],
    "highest vs lowest":       [-1, 0, 0, 1],
    "linear trend":            [-3, -1, 1, 3],
    "quadratic trend":         [1, -1, -1, 1],
    "cubic trend":             [-1, 3, -3, 1],
}
print(f"  {'contrast':<26}{'estimate':>10}{'SE':>8}{'t':>8}{'p':>11}"
      f"{'95% CI':>22}")
results = []
for lab, w in contrasts.items():
    r = test_contrast(df, "dose", "y", w, lab)
    results.append(r)
    print(f"  {lab:<26}{r['estimate']:>10.4f}{r['se']:>8.4f}{r['t']:>8.3f}"
          f"{r['p']:>11.3e}   [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")

print("\n  The LINEAR TREND contrast is by far the most significant - it is the")
print("  question the design was built to answer. An omnibus F followed by")
print("  all-pairs tests would be much less powerful.")

# Orthogonality (balanced design: sum c_j d_j = 0).
print("\n  Orthogonality of the trend contrasts (balanced design):")
trend = {k_: np.array(v, float) for k_, v in contrasts.items()
         if "trend" in k_}
for (n1, c1), (n2, c2) in itertools.combinations(trend.items(), 2):
    print(f"    <{n1:<16}, {n2:<16}> = {c1 @ c2:+.1f}")
print("  All zero -> they partition SS_between into independent pieces:")
ms_w = ss_within / (N - k)
ss_parts = {kk: (test_contrast(df, 'dose', 'y', vv)['estimate'] ** 2
                 / np.sum(np.asarray(vv, float) ** 2 / counts.values))
            for kk, vv in trend.items()}
print(f"    sum of the 3 trend SS = {sum(ss_parts.values()):.4f}   "
      f"SS_between = {ss_between:.4f}")

# %% [markdown]
# ## 3. Coding schemes change meaning, not fit

# %%
header("3. Treatment vs sum-to-zero coding")

fit_treat = smf.ols("y ~ C(dose, Treatment)", data=df).fit()
fit_sum = smf.ols("y ~ C(dose, Sum)", data=df).fit()

print(f"  Treatment coding: intercept = {fit_treat.params.iloc[0]:.4f}  "
      f"(= mean of reference D0 = {group_means['D0']:.4f})")
print(f"  Sum-to-zero     : intercept = {fit_sum.params.iloc[0]:.4f}  "
      f"(= unweighted grand mean = {group_means.mean():.4f})")
print(f"\n  RSS identical?  {np.isclose(fit_treat.ssr, fit_sum.ssr)}  "
      f"({fit_treat.ssr:.6f} vs {fit_sum.ssr:.6f})")
print(f"  Fitted values identical? "
      f"{np.allclose(fit_treat.fittedvalues, fit_sum.fittedvalues)}")
print("\n  Same model, different parameterisation. Reporting 'the effect of")
print("  treatment' without stating the coding is ambiguous.")

# %% [markdown]
# ## 4. Factorial designs and interaction, eq. (12.4)-(12.5)

# %%
header("4. Interaction is a difference of differences (12.4)-(12.5)")

rng = np.random.default_rng(1202)
n_cell = 10
cells = {("WT", "veh"): 10.0, ("WT", "drug"): 12.0,
         ("KO", "veh"): 10.5, ("KO", "drug"): 10.7}   # drug works only in WT
rows = []
for (geno, trt), mu in cells.items():
    for v in rng.normal(mu, 1.0, n_cell):
        rows.append({"y": v, "genotype": geno, "treatment": trt})
fac = pd.DataFrame(rows)

m_add = smf.ols("y ~ C(genotype) + C(treatment)", data=fac).fit()
m_int = smf.ols("y ~ C(genotype) * C(treatment)", data=fac).fit()

cm = fac.groupby(["genotype", "treatment"])["y"].mean()
simple_wt = cm[("WT", "drug")] - cm[("WT", "veh")]
simple_ko = cm[("KO", "drug")] - cm[("KO", "veh")]
print("  Cell means:")
print(cm.round(3).to_string())
print(f"\n  simple effect of drug in WT = {simple_wt:+.4f}")
print(f"  simple effect of drug in KO = {simple_ko:+.4f}")
print(f"  interaction (difference of differences) = {simple_wt - simple_ko:+.4f}")

av = sm.stats.anova_lm(m_add, m_int)
print(f"\n  interaction test: F = {av['F'].iloc[1]:.3f}, "
      f"p = {av['Pr(>F)'].iloc[1]:.4f}")

main_drug = (cm[("WT","drug")] + cm[("KO","drug")])/2 - \
            (cm[("WT","veh")] + cm[("KO","veh")])/2
print(f"\n  'main effect' of drug (averaged over genotype) = {main_drug:+.4f}")
print("  But the drug does nothing in KO. The main effect describes a")
print("  hypothetical average genotype that nobody studied. When an")
print("  interaction is present, REPORT SIMPLE EFFECTS (eq. 12.5).")

# The efficiency argument for factorial designs.
print("\n  Factorial efficiency: a 2x2 with N units estimates BOTH main effects")
print("  as precisely as two separate N/2 one-factor experiments - and gets")
print("  the interaction for free.")

# %% [markdown]
# ## 5. Type I / II / III sums of squares
#
# They differ only for **unbalanced** designs. The honest framing: stop arguing
# about labels and write down the two models you want to compare.

# %%
header("5. Type I/II/III SS: only matters when unbalanced")

# Balanced version:
print("  BALANCED design - Type I SS with both term orders:")
for form in ["y ~ C(genotype) + C(treatment)", "y ~ C(treatment) + C(genotype)"]:
    t1 = sm.stats.anova_lm(smf.ols(form, data=fac).fit(), typ=1)
    print(f"    {form:<38} " +
          "  ".join(f"{i}={v:.3f}" for i, v in t1["sum_sq"].items()
                    if i != "Residual"))

# Unbalanced version: drop some rows.
rng = np.random.default_rng(1203)
unbal = fac.drop(fac[(fac["genotype"] == "KO") &
                     (fac["treatment"] == "drug")].sample(6, random_state=3).index)
print(f"\n  UNBALANCED design (cell sizes "
      f"{unbal.groupby(['genotype','treatment']).size().values}):")
for form in ["y ~ C(genotype) + C(treatment)", "y ~ C(treatment) + C(genotype)"]:
    t1 = sm.stats.anova_lm(smf.ols(form, data=unbal).fit(), typ=1)
    print(f"    Type I, {form:<38} " +
          "  ".join(f"{i}={v:.3f}" for i, v in t1["sum_sq"].items()
                    if i != "Residual"))
t2 = sm.stats.anova_lm(smf.ols("y ~ C(genotype) + C(treatment)",
                               data=unbal).fit(), typ=2)
print(f"    Type II (order-independent)            " +
      "  ".join(f"{i}={v:.3f}" for i, v in t2["sum_sq"].items()
                if i != "Residual"))
print("\n  Type I depends on the order of terms; Type II does not. Type III")
print("  requires SUM-TO-ZERO contrasts to be interpretable - with treatment")
print("  coding it tests hypotheses nobody wants.")

# %% [markdown]
# ## 6. Effect sizes and post-hoc multiplicity, eq. (12.6)-(12.7)

# %%
header("6. eta^2 vs omega^2 (12.6); Tukey HSD (12.7)")

ms_within = ss_within / (N - k)
eta2 = ss_between / ss_total
omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
print(f"  eta^2   = {eta2:.4f}   (biased upward)")
print(f"  omega^2 = {omega2:.4f}   (eq. 12.6, less biased)")

print("\n  All-pairs comparisons - raw vs Tukey-adjusted (12.7):")
levels = sorted(df["dose"].unique())
means = df.groupby("dose")["y"].mean()[levels]
ns = df.groupby("dose")["y"].count()[levels]
print(f"  {'pair':<12}{'difference':>12}{'raw p':>11}{'Tukey p':>11}"
      f"{'Tukey 95% CI':>22}")
for i, j in itertools.combinations(range(k), 2):
    diff = means.iloc[j] - means.iloc[i]
    se = np.sqrt(ms_within * (1/ns.iloc[i] + 1/ns.iloc[j]))
    t = diff / se
    raw_p = 2 * st.t.sf(abs(t), N - k)
    # Studentised range: q = |t| * sqrt(2)
    q = abs(t) * np.sqrt(2)
    tukey_p = st.studentized_range.sf(q, k, N - k)
    qcrit = st.studentized_range.ppf(0.95, k, N - k)
    half = qcrit / np.sqrt(2) * se
    print(f"  {levels[i]}-{levels[j]:<10}{diff:>12.4f}{raw_p:>11.4f}"
          f"{tukey_p:>11.4f}   [{diff-half:+.3f}, {diff+half:+.3f}]")

print("\n  Tukey controls FWER across all 6 pairs. But note: the single")
print("  pre-specified LINEAR TREND contrast (section 2) had p = "
      f"{results[2]['p']:.2e},")
print("  far stronger than any adjusted pairwise test. Pre-specify!")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, lvl in enumerate(levels):
    v = df.loc[df["dose"] == lvl, "y"]
    axes[0].scatter(np.full(len(v), i) + rng.normal(0, 0.05, len(v)), v, s=18,
                    alpha=0.7)
    axes[0].hlines(v.mean(), i - 0.25, i + 0.25, color="k", lw=2)
axes[0].axhline(grand, color="red", ls="--", label="grand mean")
axes[0].set_xticks(range(k)); axes[0].set_xticklabels(levels)
axes[0].set_xlabel("dose"); axes[0].set_ylabel("response")
axes[0].set_title("Eq. (12.1): between vs within variation", fontsize=9)
axes[0].legend(fontsize=7)

est = [r["estimate"] for r in results]
lab = [r["label"] for r in results]
ci_lo = [r["ci"][0] for r in results]
ci_hi = [r["ci"][1] for r in results]
ypos = np.arange(len(est))
axes[1].errorbar(est, ypos, xerr=[np.array(est)-np.array(ci_lo),
                                  np.array(ci_hi)-np.array(est)],
                 fmt="o", capsize=4)
axes[1].axvline(0, color="k", ls="--")
axes[1].set_yticks(ypos); axes[1].set_yticklabels(lab, fontsize=7)
axes[1].set_xlabel("contrast estimate")
axes[1].set_title("Eq. (12.3): contrasts with 95% CIs", fontsize=9)

for geno, c in [("WT", "C0"), ("KO", "C1")]:
    vals = [cm[(geno, "veh")], cm[(geno, "drug")]]
    axes[2].plot([0, 1], vals, "o-", color=c, lw=2, label=geno)
axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["vehicle", "drug"])
axes[2].set_ylabel("response")
axes[2].set_title("Eq. (12.5): non-parallel lines = interaction", fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "anova.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/anova.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 12)
#
# 1. Pre-specify contrasts. Omnibus F then all-pairs is rarely the best use of
#    the data.
# 2. Check the interaction before interpreting main effects.
# 3. For ordered doses, a trend contrast has far more power than all-pairs.
# 4. Balance the design when you can: it makes contrasts orthogonal and makes
#    the SS-type question disappear.
#
# **Next:** `13_generalized_linear_models.py`
