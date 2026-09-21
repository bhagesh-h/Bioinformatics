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
# # Module 40: Designing a simulation study
#
# **Curriculum link:** `stats.md` -> Topic 40, equations (40.1)-(40.4)
# **Assumes:** Modules 05, 09, 34.
#
# ## What you will learn
#
# Every module in this course evaluates a method by simulation. This one makes
# that method explicit, because a simulation is an experiment and deserves the
# same design discipline you would demand of a wet-lab one.
#
# 1. **ADEMP**: aims, data-generating mechanism, estimand, methods,
#    performance measures. State them before writing code.
# 2. Performance measures (40.1)-(40.2), and the identity linking them.
# 3. **Monte Carlo standard error** (40.3)-(40.4). A simulation result without
#    an MCSE is a number without an error bar.
# 4. **Common random numbers**: compare methods on shared datasets.
# 5. Why a benchmark every method passes discriminates nothing.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "40_simulation_and_benchmarking"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(29)

# %% [markdown]
# ## 1. ADEMP, stated before any code is written
#
# The example throughout: estimating the centre of a skewed distribution.

# %%
header("1. ADEMP for the study in this module")

print("""  AIMS        Which estimator of central tendency should be used for a
              right-skewed biomarker, and does the answer depend on n?

  DATA        Y ~ LogNormal(mu = 0, sigma = 0.9), n in {10, 30, 100}.
              Chosen because biomarkers are right-skewed and this has
              closed-form truth.

  ESTIMAND    The POPULATION MEAN, exp(mu + sigma^2 / 2).
              Stated explicitly, because the median estimates something
              else and comparing them without saying so is meaningless.

  METHODS     sample mean; sample median; 20% trimmed mean;
              exp(mean of logs), the naive back-transform.

  PERFORMANCE Bias, empirical SE, RMSE, and CI coverage. Each with a
              Monte Carlo standard error.""")

SIGMA = 0.9
TRUTH = np.exp(0 + SIGMA ** 2 / 2)
print(f"\n  estimand (population mean) = {TRUTH:.4f}")
print("""
  Write this down first. A simulation with no stated estimand cannot be
  wrong, which is another way of saying it cannot be informative. The most
  common failure in published simulation studies is comparing methods that
  target different quantities.""")

# %% [markdown]
# ## 2. Performance measures and their Monte Carlo error, eq. (40.1)-(40.3)

# %%
header("2. Every performance number needs an MCSE (40.1)-(40.3)")


def estimators(y):
    return {
        "mean": y.mean(),
        "median": np.median(y),
        "trimmed 20%": st.trim_mean(y, 0.2),
        "exp(mean log)": np.exp(np.log(y).mean()),
    }


def run_study(n, n_sim, seed=0):
    r = np.random.default_rng(seed)
    out = {k: [] for k in estimators(np.array([1.0, 2.0]))}
    for _ in range(n_sim):
        y = r.lognormal(0, SIGMA, n)
        for k, v in estimators(y).items():
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


N_SIM = 2000
res = run_study(n=30, n_sim=N_SIM, seed=1)
print(f"  n = 30, n_sim = {N_SIM}, estimand = {TRUTH:.4f}\n")
print(f"  {'method':<16}{'bias':>9}{'MCSE':>8}{'EmpSE':>9}{'RMSE':>9}"
      f"{'% bias':>9}")
for k, v in res.items():
    bias = v.mean() - TRUTH                                   # eq. (40.1)
    emp = v.std(ddof=1)
    rmse = np.sqrt(np.mean((v - TRUTH) ** 2))                 # eq. (40.2)
    mcse = emp / np.sqrt(N_SIM)                               # MCSE of bias
    print(f"  {k:<16}{bias:>9.4f}{mcse:>8.4f}{emp:>9.4f}{rmse:>9.4f}"
          f"{100*bias/TRUTH:>9.1f}")

print(f"""
  The MCSE column tells you which differences are real. The sample mean's bias
  is SMALLER than its own MCSE, so it is indistinguishable from zero;
  reporting it as "slightly biased" would be reporting noise. The other three
  biases are a hundred times their MCSE and are unambiguously real.

  Check the identity in eq. (40.2), MSE = bias^2 + EmpSE^2:""")
for k, v in list(res.items())[:2]:
    lhs = np.mean((v - TRUTH) ** 2)
    rhs = (v.mean() - TRUTH) ** 2 + (N_SIM - 1) / N_SIM * v.std(ddof=1) ** 2
    print(f"    {k:<16} MSE = {lhs:.6f},  bias^2 + EmpSE^2 = {rhs:.6f}")

print("""
  The median and the log-based estimator are heavily biased FOR THIS
  ESTIMAND, and that is not a criticism of them. They estimate the median,
  which for this distribution is 1.000 rather than 1.499. Naming the estimand
  is what turns an unfair comparison into an informative one.""")

# %% [markdown]
# ## 3. How many repetitions? eq. (40.4)

# %%
header("3. Choosing n_sim from the precision you need (40.4)")

print("  For a coverage or rejection RATE near p, eq. (40.3) gives")
print("  MCSE = sqrt(p(1-p)/n_sim), so eq. (40.4) inverts it:\n")
print(f"  {'target MCSE':<16}{'p = 0.05':>12}{'p = 0.50':>12}{'p = 0.80':>12}")
for target in (0.01, 0.005, 0.002, 0.001):
    row = "".join(f"{int(np.ceil(p*(1-p)/target**2)):>12,}" for p in (0.05, 0.5, 0.8))
    print(f"  {target:<16.3f}{row}")

print("""
  To claim a test holds its 5% level to within half a percentage point you
  need about 1,900 repetitions. To resolve it to a tenth of a point you need
  about 47,500.

  This is why "the false positive rate was 4% in 100 runs" is not evidence of
  anything: the MCSE there is 2%, so 4% and 8% are not distinguishable.""")

# Demonstrate the instability directly.
print(f"\n  The same quantity estimated with different n_sim:")
print(f"  {'n_sim':<10}{'5 independent estimates of the type I error rate':>52}")
for n_sim in (100, 1000, 10000):
    vals = []
    for rep in range(5):
        r = np.random.default_rng(3000 + rep * 97 + n_sim)
        hits = sum(st.ttest_ind(r.normal(0, 1, 15), r.normal(0, 1, 15)).pvalue < 0.05
                   for _ in range(n_sim))
        vals.append(hits / n_sim)
    print(f"  {n_sim:<10}" + "".join(f"{v:>10.3f}" for v in vals))

# %% [markdown]
# ## 4. Common random numbers
#
# Comparing methods on the *same* simulated datasets removes between-dataset
# variation from the comparison, exactly as pairing does in an experiment.

# %%
header("4. Compare methods paired, not independently")

n_sim = 1000
# Paired: both estimators see identical data.
r = np.random.default_rng(11)
paired_a, paired_b = [], []
for _ in range(n_sim):
    y = r.lognormal(0, SIGMA, 30)
    paired_a.append(y.mean()); paired_b.append(st.trim_mean(y, 0.2))
paired_a, paired_b = np.array(paired_a), np.array(paired_b)

# Unpaired: each estimator gets its own datasets.
r1 = np.random.default_rng(12); r2 = np.random.default_rng(13)
unp_a = np.array([r1.lognormal(0, SIGMA, 30).mean() for _ in range(n_sim)])
unp_b = np.array([st.trim_mean(r2.lognormal(0, SIGMA, 30), 0.2) for _ in range(n_sim)])

d_p = paired_a - paired_b
d_u = unp_a - unp_b
print(f"  difference in RMSE between two estimators, n_sim = {n_sim}\n")
print(f"  {'design':<26}{'mean difference':>18}{'MCSE of difference':>21}")
print(f"  {'shared datasets':<26}{d_p.mean():>18.4f}{d_p.std(ddof=1)/np.sqrt(n_sim):>21.5f}")
print(f"  {'independent datasets':<26}{d_u.mean():>18.4f}{d_u.std(ddof=1)/np.sqrt(n_sim):>21.5f}")
print(f"\n  correlation between the two estimators on shared data: "
      f"{np.corrcoef(paired_a, paired_b)[0,1]:.3f}")
print(f"  precision gained by pairing: "
      f"{(d_u.std(ddof=1)/d_p.std(ddof=1))**2:.1f}x fewer repetitions for the same MCSE")

print("""
  The two estimators are strongly correlated when computed on the same data,
  so most of the variation cancels in the difference. Pairing therefore buys a
  large reduction in the repetitions needed, for free.

  This is the simulation equivalent of a paired design (Topic 12), and the
  same reasoning applies: remove the variation you do not care about.""")

# %% [markdown]
# ## 5. A benchmark that everything passes is not a benchmark

# %%
header("5. Include a case where the method should fail")

def benchmark(scenario, n_sim=1200, seed=0):
    r = np.random.default_rng(seed)
    out = {"t pooled": 0, "t Welch": 0, "Wilcoxon": 0, "permutation": 0}
    for _ in range(n_sim):
        if scenario == "normal, null":
            a, b = r.normal(0, 1, 25), r.normal(0, 1, 25)
        elif scenario == "lognormal, null":
            a, b = r.lognormal(0, 1.2, 25), r.lognormal(0, 1.2, 25)
        elif scenario == "unequal var, EQUAL n":
            a, b = r.normal(0, 1, 25), r.normal(0, 4, 25)
        elif scenario == "unequal var, UNEQUAL n":
            a, b = r.normal(0, 4, 10), r.normal(0, 1, 40)
        elif scenario == "normal, real effect":
            a, b = r.normal(0, 1, 25), r.normal(0.8, 1, 25)
        out["t pooled"] += st.ttest_ind(a, b).pvalue < 0.05
        out["t Welch"] += st.ttest_ind(a, b, equal_var=False).pvalue < 0.05
        out["Wilcoxon"] += st.mannwhitneyu(a, b).pvalue < 0.05
        pool = np.concatenate([a, b]); na = len(a)
        obs = abs(b.mean() - a.mean())
        nd = np.array([abs(p_[na:].mean() - p_[:na].mean())
                       for p_ in (r.permutation(pool) for _ in range(99))])
        out["permutation"] += (1 + np.sum(nd >= obs)) / 100 < 0.05
    return {k: v / n_sim for k, v in out.items()}


print(f"  {'scenario':<26}{'t pooled':>10}{'t Welch':>10}{'Wilcoxon':>11}"
      f"{'permutation':>13}{'target':>9}")
for sc in ("normal, null", "lognormal, null", "unequal var, EQUAL n",
           "unequal var, UNEQUAL n", "normal, real effect"):
    r_ = benchmark(sc, seed=len(sc) * 37)
    tgt = "high" if "effect" in sc else "0.05"
    print(f"  {sc:<26}{r_['t pooled']:>10.3f}{r_['t Welch']:>10.3f}"
          f"{r_['Wilcoxon']:>11.3f}{r_['permutation']:>13.3f}{tgt:>9}")

print("""
  The first row separates nothing: every method is correct on clean normal
  data, which is why a benchmark consisting only of that row would be
  uninformative.

  The rows that discriminate are the awkward ones, and they do not all break
  the same method.

  With unequal variance but EQUAL group sizes, the pooled t-test is fine: the
  variance misspecification cancels. Wilcoxon is the one that inflates,
  because it tests stochastic equality rather than equality of means, and two
  distributions with the same mean and different spreads are not
  stochastically equal.

  Make the group sizes UNEQUAL and the pooled t-test fails badly. It pools the
  two variances by sample size, so the larger group's variance dominates the
  estimate; when the SMALL group is the variable one the test is far too
  liberal. Welch holds throughout, which is why it is the sensible default.
  The permutation test does not rescue this either: permuting assumes the two
  groups are exchangeable under the null, and different variances mean they
  are not.

  A benchmark with only the first two rows would have reported all four
  methods as equivalent.

  Design benchmarks around the cases where methods SHOULD differ, always
  include a null, and always include a case the method under test is expected
  to fail. A comparison in which your preferred method wins every scenario is
  usually a sign that the scenarios were chosen after the fact.""")

# %% [markdown]
# ## 6. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

for k, colour in zip(res, ("steelblue", "firebrick", "seagreen", "darkorange")):
    ax[0].hist(res[k], bins=45, alpha=0.5, color=colour, label=k, density=True)
ax[0].axvline(TRUTH, color="black", lw=2)
ax[0].set_xlabel("estimate"); ax[0].set_title("Four estimators, one estimand")
ax[0].legend(fontsize=7)

ns = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
mcse = np.sqrt(0.05 * 0.95 / ns)
ax[1].plot(ns, mcse, "o-", color="steelblue")
ax[1].axhline(0.005, color="firebrick", ls="--", label="0.5 point precision")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("n_sim"); ax[1].set_ylabel("MCSE at p = 0.05")
ax[1].set_title("Eq. (40.3)"); ax[1].legend(fontsize=8)

ax[2].hist(d_u, bins=40, alpha=0.6, color="firebrick", label="independent")
ax[2].hist(d_p, bins=40, alpha=0.6, color="steelblue", label="shared data")
ax[2].set_xlabel("difference between estimators")
ax[2].set_title("Common random numbers"); ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "simulation_design.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'simulation_design.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: How many repetitions did this course need?
#
# Several modules report a false positive rate from a few hundred runs. Work
# out the MCSE of those claims and decide which are safe.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  {'claim':<42}{'n_sim':>8}{'rate':>8}{'MCSE':>8}{'95% CI':>18}")
# claims = [("module 08: BH holds its FDR", 500, 0.048),
#           ("E1: confounded design false positives", 500, 0.238),
#           ("F5: p-hacking inflates alpha", 5000, 0.098),
#           ("module 25: e-BH under dependence", 200, 0.031)]
# for lab, n_s, rate in claims:
#     mcse = np.sqrt(rate * (1 - rate) / n_s)
#     print(f"  {lab:<42}{n_s:>8}{rate:>8.3f}{mcse:>8.4f}"
#           f"{f'({rate-1.96*mcse:.3f}, {rate+1.96*mcse:.3f})':>18}")
#
# # The claims differ in how much they can bear.
# #
# # "The confounded design gives 24% false positives" is safe: the interval is
# # nowhere near 5%, so the qualitative conclusion is unambiguous.
# #
# # "BH holds its FDR at 4.8%" from 500 runs has an interval of roughly 3% to
# # 7%. That supports "BH is approximately calibrated" and does NOT support
# # "BH is slightly conservative here", which is the kind of over-reading an
# # MCSE prevents.
# #
# # The general rule: use few repetitions for claims about large qualitative
# # differences, and many for claims about whether a rate equals its nominal
# # value. Report n_sim so a reader can do this arithmetic themselves.

# %% [markdown]
# ### Problem 2: Build a simulation study end to end
#
# Compare three ways of handling a skewed outcome: t-test on the raw values,
# t-test on logs, and a Wilcoxon test. Write the ADEMP first, then run it.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# # AIMS: which test for a right-skewed outcome with a multiplicative effect?
# # DATA: y = lognormal, treated group multiplied by a fold change.
# # ESTIMAND: the null is "no difference in distribution"; under the
# #           alternative the effect is MULTIPLICATIVE.
# # METHODS: t on raw, t on log, Wilcoxon.
# # PERFORMANCE: type I error and power, each with an MCSE.
# def one(n, fold, seed):
#     r = np.random.default_rng(seed)
#     a = r.lognormal(0, 1.0, n)
#     b = r.lognormal(np.log(fold), 1.0, n)
#     return (st.ttest_ind(a, b).pvalue,
#             st.ttest_ind(np.log(a), np.log(b)).pvalue,
#             st.mannwhitneyu(a, b).pvalue)
#
# NS = 3000
# print(f"  {'fold change':<14}{'t raw':>18}{'t on log':>18}{'Wilcoxon':>18}")
# for fold in (1.0, 1.3, 1.8):
#     hits = np.zeros(3)
#     for i in range(NS):
#         hits += np.array(one(30, fold, 200000 + i)) < 0.05
#     rates = hits / NS
#     cells = "".join(f"{r_:>11.3f} +/-{np.sqrt(r_*(1-r_)/NS):>5.3f}" for r_ in rates)
#     print(f"  {fold:<14.1f}{cells}")
#
# # At fold = 1.0 all three hold the nominal level, so all three are VALID.
# # Validity was never the question.
# #
# # At fold > 1 the log t-test and Wilcoxon are clearly more powerful than the
# # t-test on raw values, because the effect is multiplicative and the raw
# # scale spreads it across a long tail. The MCSEs are small enough that the
# # ordering is real rather than noise, which is exactly what reporting them
# # lets you assert.
# #
# # Notice the estimand did the work again. On the log scale the t-test targets
# # a ratio of geometric means, which is the quantity the data generating
# # mechanism actually contains.

# %% [markdown]
# ### Problem 3: A benchmark designed to be won
#
# Show how a scenario set can be chosen so that a preferred method wins, then
# show what an honest scenario set reveals.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def power_of(method, n, shift, tail, seed, NS=1200):
#     r = np.random.default_rng(seed); hits = 0
#     for _ in range(NS):
#         if tail == "normal":
#             a, b = r.normal(0, 1, n), r.normal(shift, 1, n)
#         else:
#             a = r.standard_t(2, n); b = r.standard_t(2, n) + shift
#         p_ = (st.ttest_ind(a, b).pvalue if method == "t"
#               else st.mannwhitneyu(a, b).pvalue)
#         hits += p_ < 0.05
#     return hits / NS
#
# print("  A 'benchmark' using only normal data:\n")
# print(f"  {'scenario':<26}{'t-test':>10}{'Wilcoxon':>11}{'winner':>10}")
# for sh in (0.4, 0.6, 0.8):
#     t_ = power_of("t", 30, sh, "normal", 1); w_ = power_of("w", 30, sh, "normal", 1)
#     print(f"  {f'normal, shift {sh}':<26}{t_:>10.3f}{w_:>11.3f}"
#           f"{('t-test' if t_ > w_ else 'Wilcoxon'):>10}")
# print("\n  The same comparison with a heavy-tailed scenario added:\n")
# print(f"  {'scenario':<26}{'t-test':>10}{'Wilcoxon':>11}{'winner':>10}")
# for sh in (0.4, 0.8):
#     t_ = power_of("t", 30, sh, "t2", 2); w_ = power_of("w", 30, sh, "t2", 2)
#     print(f"  {f'heavy-tailed, shift {sh}':<26}{t_:>10.3f}{w_:>11.3f}"
#           f"{('t-test' if t_ > w_ else 'Wilcoxon'):>10}")
#
# # Restricted to normal data, the t-test wins every row, and a paper could
# # report that table truthfully and conclude the t-test is superior.
# #
# # Add one heavy-tailed scenario and the ordering reverses. Nothing in the
# # first table was false; it was incomplete, and incompleteness is how
# # benchmarks mislead without anyone lying.
# #
# # Two defences when reading a benchmark: ask which scenarios are ABSENT, and
# # ask whether the authors' own method was the one the scenarios were designed
# # around. Two defences when writing one: pre-specify the scenarios, and
# # include the case where your method should lose.

# %% [markdown]
# ## What to take away
#
# 1. Write **ADEMP** before code. Naming the **estimand** is what makes a
#    comparison meaningful.
# 2. Report **bias, EmpSE, RMSE and coverage**, and check
#    MSE = bias$^2$ + EmpSE$^2$ as an arithmetic sanity test.
# 3. Every simulated number needs an **MCSE** (40.3). Choose $n_{\text{sim}}$
#    from the precision your claim requires (40.4).
# 4. Use **common random numbers**: compare methods on shared datasets.
# 5. A benchmark without a null, and without a case your method should fail,
#    discriminates nothing.
#
# **You have finished the core track.** Continue with the applied
# modules, starting with
# **Next:** `../bioinformatics/20_bulk_rnaseq_differential_expression.py`
