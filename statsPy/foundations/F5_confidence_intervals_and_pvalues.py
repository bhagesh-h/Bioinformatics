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
# # Foundations 5: Confidence intervals and p-values
#
# **Curriculum link:** `stats.md` -> Part 0, §0.7-§0.8, equations (0.13)-(0.15)
# **Assumes:** `F1`-`F4`.
#
# ## Why this module matters most
#
# Confidence intervals and p-values are the two things every paper reports and
# the two things most often misread. Both are statements about a **procedure**
# repeated many times, not about your particular result: and almost every
# misinterpretation comes from forgetting that.
#
# This module builds both from scratch and, importantly, **checks them by
# simulation**, so you see the procedure succeed at exactly the advertised
# rate rather than taking anyone's word for it.

# %%
import os

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "F5_confidence_intervals_and_pvalues"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(4)
TRUE_MU, TRUE_SIGMA = 50.0, 10.0

# %% [markdown]
# ## 1. Building a confidence interval, eq. (0.13)-(0.14)
#
# `estimate +/- critical value x standard error`. That is the whole recipe.

# %%
header("1. One interval, built by hand")

n = 16
s_ = rng.normal(TRUE_MU, TRUE_SIGMA, n)
xbar, s = s_.mean(), s_.std(ddof=1)
se = s / np.sqrt(n)
# With sigma unknown we use the t distribution's critical value, which is a
# little larger than 1.96 and so makes the interval appropriately wider.
crit = st.t.ppf(0.975, df=n - 1)
lo, hi = xbar - crit * se, xbar + crit * se

print(f"  sample mean  x-bar = {xbar:.3f}")
print(f"  sample SD    s     = {s:.3f}")
print(f"  standard error     = s/sqrt(n) = {se:.3f}")
print(f"  critical value     = {crit:.3f}   (t with {n-1} df; normal would use 1.960)")
print(f"\n  95% CI = {xbar:.3f} +/- {crit:.3f} x {se:.3f} = ({lo:.3f}, {hi:.3f})")
print(f"  true value = {TRUE_MU}  ->  {'INSIDE' if lo <= TRUE_MU <= hi else 'OUTSIDE'}")

# %% [markdown]
# ## 2. What "95% confident" actually means
#
# The natural reading is wrong, and simulation shows exactly what the right
# reading is. We build 10,000 intervals and count how many catch the truth.

# %%
header("2. The coverage check: does the procedure really work 95% of the time?")

def ci(sample, level=0.95):
    k = len(sample)
    m, se_ = sample.mean(), sample.std(ddof=1) / np.sqrt(k)
    c = st.t.ppf(0.5 + level / 2, df=k - 1)
    return m - c * se_, m + c * se_

print(f"  {'level':<10}{'n':<8}{'intervals built':>18}{'contained the truth':>22}")
for level in (0.80, 0.90, 0.95, 0.99):
    hits = sum(lo_ <= TRUE_MU <= hi_
               for lo_, hi_ in (ci(rng.normal(TRUE_MU, TRUE_SIGMA, 16), level)
                                for _ in range(10_000)))
    print(f"  {level:<10.0%}{16:<8}{10_000:>18,}{hits/10_000:>22.1%}")

print("""
  The procedure delivers what it promises. That is the ONLY guarantee a
  confidence interval carries, and it is a guarantee about the METHOD.

  CORRECT   : "if I repeated this experiment many times, 95% of the intervals
               I built would contain the true value."
  INCORRECT : "there is a 95% probability the true value is in MY interval."

  Why is the second wrong? Your interval is now a fixed pair of numbers and
  the truth is a fixed number. Either it is in there or it is not - there is
  no randomness left to have a probability about. The randomness was in the
  sampling, which has already happened.

  (If you genuinely want "95% probability the truth is in here", that is a
  CREDIBLE interval and requires Bayesian machinery and a prior - Topic 33.
  In practice the two often nearly coincide, which is why the sloppy reading
  usually survives contact with reality.)""")

# %% [markdown]
# ## 3. Read the width, not whether it excludes zero
#
# The most common misuse of an interval is to check one thing about it and
# throw the rest away.

# %%
header("3. Two 'non-significant' results that mean opposite things")

studies = [
    ("large, precise study",  0.1, 0.15),
    ("small, noisy study",    2.5, 1.55),
]
print(f"  {'study':<26}{'estimate':>10}{'95% CI':>22}{'p':>9}  verdict")
for name, est, se_ in studies:
    lo_, hi_ = est - 1.96 * se_, est + 1.96 * se_
    p = 2 * st.norm.sf(abs(est / se_))
    print(f"  {name:<26}{est:>10.2f}{f'({lo_:+.2f}, {hi_:+.2f})':>22}{p:>9.3f}"
          f"  {'n.s.' if p > 0.05 else 'sig'}")

print("""
  Both are "not significant". They say completely different things.

    The first RULES OUT anything larger than about 0.4. That is a real,
    informative, publishable finding: the effect, if any, is small.

    The second rules out nothing. Effects of 0, of 2, of 5 are all consistent
    with it. It is not evidence of absence - it is absence of evidence.

  Reporting both as "no significant difference" throws away the only
  information that distinguishes them. THIS is the strongest practical
  argument for reporting intervals, and the reason "no significant
  difference" must never be written as "no difference".""")

# %% [markdown]
# ## 4. What a p-value is, eq. (0.15)
#
# Assume nothing is going on. Ask how surprising your data would be. That
# surprise, expressed as a probability, is the p-value.

# %%
header("4. Building a p-value from scratch, by simulation")

# One experiment: 12 vs 12, with a real difference of 0.8 units.
group_a = rng.normal(50.0, 2.0, 12)
group_b = rng.normal(50.8, 2.0, 12)
observed_diff = group_b.mean() - group_a.mean()

# The null hypothesis says the labels are meaningless. So SHUFFLE them, and
# see how big a difference appears by chance alone. This is a permutation
# test: no formula, no distributional assumption, just the definition.
pooled = np.concatenate([group_a, group_b])
null_diffs = np.empty(20_000)
for i in range(20_000):
    perm = rng.permutation(pooled)
    null_diffs[i] = perm[12:].mean() - perm[:12].mean()

p_perm = np.mean(np.abs(null_diffs) >= abs(observed_diff))

print(f"  observed difference between the groups : {observed_diff:+.4f}")
print(f"  if the labels were meaningless, differences this big or bigger")
print(f"  happened in {np.sum(np.abs(null_diffs) >= abs(observed_diff)):,} of 20,000 shuffles")
print(f"\n  p (permutation) = {p_perm:.4f}")
print(f"  p (t-test)      = {st.ttest_ind(group_b, group_a).pvalue:.4f}   <- agrees")

print("""
  Read the definition off the simulation:

      p = P( data at least this extreme | nothing is going on )

  Note what is on each side of the bar. The p-value is the probability of THE
  DATA given the HYPOTHESIS. It is not the probability of the hypothesis given
  the data - that is the reversal from F2, section 4, and it is the single
  most common error in interpreting statistics.

  A p-value of 0.03 does NOT mean "a 3% chance there is no real effect".""")

# %% [markdown]
# ## 5. The kinds of p-value
#
# "The p-value" is not one thing. Knowing which kind you have is the
# difference between a valid analysis and an invalid one.

# %%
header("5. Five ways to get a p-value for the same comparison")

a = rng.normal(0.0, 1.0, 14)
b = rng.normal(0.9, 1.0, 14)

# (1) Parametric: assume normality, use a formula.
p_t = st.ttest_ind(a, b).pvalue
# (2) Welch: same, but not assuming equal variances (the safer default).
p_welch = st.ttest_ind(a, b, equal_var=False).pvalue
# (3) Permutation: shuffle labels; assumes only exchangeability.
pool = np.concatenate([a, b]); obs = b.mean() - a.mean()
nd = np.array([(lambda q: q[14:].mean() - q[:14].mean())(rng.permutation(pool))
               for _ in range(20_000)])
p_permut = np.mean(np.abs(nd) >= abs(obs))
# (4) Rank-based: throw away the values, keep the order.
p_rank = st.mannwhitneyu(a, b, alternative="two-sided").pvalue
# (5) Bootstrap: resample each group, see how often the difference crosses 0.
boot = np.array([rng.choice(b, 14, True).mean() - rng.choice(a, 14, True).mean()
                 for _ in range(20_000)])
p_boot = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))

print(f"  {'kind':<16}{'method':<26}{'assumes':<30}{'p':>9}")
rows = [
    ("parametric", "Student t-test", "normality, equal variance", p_t),
    ("parametric", "Welch t-test", "normality only", p_welch),
    ("permutation", "shuffle the labels", "exchangeability", p_permut),
    ("rank-based", "Mann-Whitney U", "nothing about shape", p_rank),
    ("bootstrap", "resample each group", "sample is representative", p_boot),
]
for kind, meth, ass, p in rows:
    print(f"  {kind:<16}{meth:<26}{ass:<30}{p:>9.4f}")

print("""
  On clean, well-behaved data they agree closely - as they should. They
  diverge when their assumptions diverge: on skewed data with outliers the
  t-test and the rank test can disagree sharply, and then you must decide
  WHICH QUESTION you are asking (a difference in means, or a tendency for one
  group to exceed the other - they are not the same question).

  The two that need extra care:

    PERMUTATION assumes only that the labels are exchangeable under the null.
    That makes it robust - but you must shuffle WITHIN the structure of your
    design. With batches or paired samples, free shuffling builds a null that
    is simply wrong (exercise E2 measures exactly this).

    ONE-SIDED vs TWO-SIDED is a separate axis entirely, and the next section
    is about it.""")

# %% [markdown]
# ## 6. One-sided, two-sided, and p-hacking

# %%
header("6. One-sided tests, and why they are abused")

p_two = st.ttest_ind(a, b).pvalue
p_one = st.ttest_ind(a, b, alternative="less").pvalue
print(f"  two-sided p = {p_two:.4f}")
print(f"  one-sided p = {p_one:.4f}   <- exactly half, when the effect is in")
print(f"                              the predicted direction")

print("""
  A one-sided test is legitimate only when a difference in the OTHER direction
  would be acted on identically to no difference at all, and only when you
  declared it before seeing the data. That is rare. Switching to one-sided
  because two-sided gave 0.06 is cheating, and it has a name.""")

# Demonstrate p-hacking: try several analyses, report the best.
header("6b. What 'trying a few things' does to your error rate")

def one_experiment(rng_):
    """Two groups with NO real difference, measured 4 ways."""
    x = rng_.normal(0, 1, 15); y = rng_.normal(0, 1, 15)
    return [
        st.ttest_ind(x, y).pvalue,                              # the plain test
        st.ttest_ind(x, y, alternative="less").pvalue,          # one-sided
        st.ttest_ind(x, y, alternative="greater").pvalue,       # other side
        st.mannwhitneyu(x, y, alternative="two-sided").pvalue,  # switch test
    ]

R = 5000
honest = hacked = 0
for _ in range(R):
    ps = one_experiment(rng)
    honest += ps[0] < 0.05          # decide the test in advance
    hacked += min(ps) < 0.05        # try all four, report the smallest
print(f"  {R:,} experiments in which NOTHING is going on:\n")
print(f"  {'strategy':<46}{'false positive rate':>20}")
print(f"  {'pre-specified two-sided t-test':<46}{honest/R:>20.1%}")
print(f"  {'try 4 analyses, report the smallest p':<46}{hacked/R:>20.1%}")

print("""
  Four analyses is a modest amount of flexibility - far less than a real
  analyst has. It roughly doubles the false positive rate - and a real
  analysis has far more than four forks in it.

  The reported p-value is then meaningless, because a p-value describes the
  PROCEDURE that produced it, and the procedure was "try things until one
  works". That procedure does not have a 5% error rate.

  The same mechanism drives:
    p-hacking        - trying analyses until one is significant
    HARKing          - inventing the hypothesis after seeing the result
    optional stopping- collecting more data until p drops below 0.05
                       (this reaches significance eventually with certainty)
    subgroup fishing - 'it worked in the older female patients'

  The defence is always the same: DECIDE THE ANALYSIS BEFORE SEEING THE
  OUTCOME, and report everything you tried.""")

# %% [markdown]
# ## 7. Where does 0.05 come from, and what are $\alpha$ and $\beta$?

# %%
header("7. Two kinds of error, and the threshold that trades them off")

DIFF = 1.0
def error_rates(alpha, n=15, reps=4000):
    fp = np.mean([st.ttest_ind(rng.normal(0, 1, n), rng.normal(0, 1, n)).pvalue < alpha
                  for _ in range(reps)])
    tp = np.mean([st.ttest_ind(rng.normal(0, 1, n), rng.normal(DIFF, 1, n)).pvalue < alpha
                  for _ in range(reps)])
    return fp, tp

print(f"  {'alpha':<12}{'Type I (false pos)':>22}{'power (1 - Type II)':>22}")
for alpha in (0.20, 0.10, 0.05, 0.01, 0.001):
    fp, tp = error_rates(alpha)
    print(f"  {alpha:<12}{fp:>22.3f}{tp:>22.3f}")

print("""
  TYPE I error  - you claim an effect that is not there. Its rate IS alpha:
                  whatever threshold you choose, you get that false positive
                  rate. The first column simply reproduces the first column.

  TYPE II error - you miss an effect that is there. Its rate is beta, and
                  POWER = 1 - beta is the second column.

  Lowering alpha buys fewer false positives and pays in missed real effects.
  There is no setting that avoids both; the threshold is a statement about
  which error is more costly IN YOUR SITUATION.

  0.05 is not a law of nature. Ronald Fisher suggested one-in-twenty in the
  1920s as a rough working line, explicitly as a rule of thumb. It survived
  because a shared default stops people choosing a threshold after seeing
  the answer - which is a real benefit, but not a mathematical one.

  Sensible thresholds vary enormously with the consequences:
    screening assay, cheap follow-up      alpha = 0.10
    ordinary single hypothesis            alpha = 0.05
    genome-wide association study         alpha = 5e-8   (Topic 24)
    particle physics discovery            alpha = 3e-7

  The modern position - the American Statistical Association's 2016 and 2019
  statements - is that 'statistically significant' should not be treated as a
  conclusion at all. Report the estimate, the interval, and the p-value; let
  the reader see the evidence rather than a verdict.""")

# %% [markdown]
# ## 8. Many tests at once: FWER and FDR

# %%
header("8. What happens when you test 20,000 things")

G = 20_000
null_p = rng.uniform(0, 1, G)              # 20,000 genes, none of them real

print(f"  {G:,} tests, NOT ONE of which has a real effect:\n")
print(f"  {'rule':<42}{'discoveries':>14}{'all false?':>12}")
print(f"  {'raw p < 0.05':<42}{np.sum(null_p < 0.05):>14,}{'yes':>12}")

from statsmodels.stats.multitest import multipletests
for meth, label in (("bonferroni", "Bonferroni (controls FWER)"),
                    ("holm", "Holm (controls FWER, more powerful)"),
                    ("fdr_bh", "Benjamini-Hochberg (controls FDR)")):
    rej = multipletests(null_p, alpha=0.05, method=meth)[0]
    print(f"  {label:<42}{rej.sum():>14,}{'yes':>12}")

# Now with some real signal mixed in.
mixed = np.concatenate([rng.uniform(0, 1, 19_000),      # 19,000 nulls
                        rng.beta(0.2, 12, 1_000)])      # 1,000 real effects
truth = np.concatenate([np.zeros(19_000, bool), np.ones(1_000, bool)])
print(f"\n  Now 20,000 tests of which 1,000 ARE real:\n")
print(f"  {'rule':<42}{'discoveries':>13}{'false':>8}{'FDP':>8}{'power':>8}")
for meth, label in (("bonferroni", "Bonferroni (FWER)"),
                    ("holm", "Holm (FWER)"),
                    ("fdr_bh", "Benjamini-Hochberg (FDR)")):
    rej = multipletests(mixed, alpha=0.05, method=meth)[0]
    fdp = np.sum(rej & ~truth) / max(rej.sum(), 1)
    print(f"  {label:<42}{rej.sum():>13,}{np.sum(rej & ~truth):>8}"
          f"{fdp:>8.3f}{np.mean(rej[truth]):>8.3f}")

print("""
  Two different promises, and you must know which one you want:

    FWER (Bonferroni, Holm) - controls the probability of making EVEN ONE
      false claim. Appropriate when a single false positive is expensive: a
      confirmatory clinical endpoint, a claim you will build a programme on.
      Very conservative when there are many tests, so power collapses.

    FDR (Benjamini-Hochberg) - controls the EXPECTED PROPORTION of your
      discoveries that are false. You accept that ~5% of your 200-gene list is
      wrong in exchange for getting a list at all. This is the right default
      in genomics, and its adjusted values are called q-values.

  Look at the power column. That difference is why genomics runs on FDR.

  Topic 8 covers this properly, including what happens when the tests are
  correlated - which, for genes in a pathway, they always are.""")

# %% [markdown]
# ## 9. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

for i in range(40):
    smp = rng.normal(TRUE_MU, TRUE_SIGMA, 16)
    lo_, hi_ = ci(smp)
    covers = lo_ <= TRUE_MU <= hi_
    ax[0].plot([lo_, hi_], [i, i], color="steelblue" if covers else "firebrick", lw=1.6)
    ax[0].plot(smp.mean(), i, "o", ms=2.5,
               color="steelblue" if covers else "firebrick")
ax[0].axvline(TRUE_MU, color="black", lw=1.5)
ax[0].set_yticks([]); ax[0].set_xlabel("value")
ax[0].set_title("40 intervals; red ones miss")

ax[1].hist(null_diffs, bins=70, color="grey", density=True)
ax[1].axvline(observed_diff, color="firebrick", lw=2,
              label=f"observed ({observed_diff:+.2f})")
ax[1].axvline(-observed_diff, color="firebrick", lw=2, ls=":")
ax[1].set_title(f"Permutation null (p = {p_perm:.3f})")
ax[1].set_xlabel("difference under shuffled labels"); ax[1].legend(fontsize=8)

alphas = np.array([0.5, 0.2, 0.1, 0.05, 0.01, 0.001])
fps, tps = zip(*[error_rates(al, reps=1500) for al in alphas])
ax[2].plot(alphas, fps, "o-", color="firebrick", label="Type I rate")
ax[2].plot(alphas, tps, "o-", color="steelblue", label="power")
ax[2].set_xscale("log"); ax[2].set_xlabel(r"threshold $\alpha$")
ax[2].set_title("The trade-off you are choosing"); ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "intervals_and_pvalues.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'intervals_and_pvalues.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Does the interval still work when the data is ugly?
#
# The coverage check in section 2 used normal data. Try it on strongly skewed
# data at several sample sizes.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# true_mean_ln = np.exp(0 + 1.4**2 / 2)          # mean of lognormal(0, 1.4)
# print(f"  A 95% interval SHOULD contain the truth 95% of the time.\n")
# print(f"  {'n':<8}{'normal data':>16}{'skewed data':>16}")
# for n_ in (5, 10, 30, 100, 500):
#     cov_n = np.mean([ci(rng.normal(0, 1, n_))[0] <= 0 <= ci(rng.normal(0, 1, n_))[1]
#                      for _ in range(4000)])
#     cov_s = np.mean([(lambda l, h: l <= true_mean_ln <= h)(
#                         *ci(rng.lognormal(0, 1.4, n_))) for _ in range(4000)])
#     print(f"  {n_:<8}{cov_n:>16.1%}{cov_s:>16.1%}")
#
# # The normal column sits at 95% at every n - as it must, since the method was
# # derived for exactly that case.
# #
# # The skewed column is too LOW at small n: the interval misses more often than
# # advertised, so you are over-confident. It improves as n grows, because the
# # central limit theorem (F4) gradually makes the sample mean normal.
# #
# # This is what an assumption violation actually looks like. Not a crash, not
# # an error message - just a procedure quietly delivering 88% coverage while
# # claiming 95%. The fixes are a bootstrap interval, a transformation (log),
# # or more data. Topic 5 covers all three.

# %% [markdown]
# ### Problem 2: The p-value is not the probability you are wrong
#
# Among all the times you see $p < 0.05$, how often is there really no effect?
# Show that the answer depends on something the p-value does not know.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def run(n_tests, frac_real, effect=0.8, n=20):
#     real = rng.random(n_tests) < frac_real
#     ps = np.array([st.ttest_ind(rng.normal(0, 1, n),
#                                 rng.normal(effect if r else 0.0, 1, n)).pvalue
#                    for r in real])
#     sig = ps < 0.05
#     return sig.sum(), np.mean(~real[sig]) if sig.sum() else np.nan
#
# print(f"  {'% of hypotheses that are REAL':<34}{'significant':>13}"
#       f"{'of those, % FALSE':>20}")
# for frac in (0.90, 0.50, 0.10, 0.01):
#     nsig, fdr = run(4000, frac)
#     print(f"  {frac:<34.0%}{nsig:>13,}{fdr:>20.1%}")
#
# # The threshold is 0.05 in every row. The chance that a significant result is
# # wrong ranges from a few percent to most of them.
# #
# # The p-value cannot know this, because it is computed ASSUMING the null is
# # true - it never considers how plausible the null was to begin with. So
# # "p < 0.05" carries completely different weight for a pre-registered
# # hypothesis with good prior support than for the 4,000th exploratory test.
# #
# # This is the same base-rate effect as the screening test in F2, and it is
# # the honest answer to "why do so many published findings fail to replicate?"
# # It is also why FDR methods (section 8) are framed around the PROPORTION of
# # discoveries that are false, which is exactly the quantity in the last
# # column.

# %% [markdown]
# ### Problem 3: Optional stopping
#
# Collect data one observation at a time, testing after each, and stop as soon
# as $p < 0.05$. There is no real effect. How often do you "succeed"?

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# def peek_until_significant(max_n, alpha=0.05):
#     """Add one observation per group at a time; stop at the first p < alpha."""
#     x, y = list(rng.normal(0, 1, 5)), list(rng.normal(0, 1, 5))
#     for _ in range(max_n - 5):
#         x.append(rng.normal()); y.append(rng.normal())
#         if st.ttest_ind(x, y).pvalue < alpha:
#             return True, len(x)
#     return False, len(x)
#
# print(f"  No real effect. Nominal false positive rate: 5%.\n")
# print(f"  {'max n allowed':<18}{'reached p < 0.05':>20}")
# for max_n in (5, 10, 20, 50, 100, 200):
#     hits = [peek_until_significant(max_n)[0] for _ in range(1500)]
#     print(f"  {max_n:<18}{np.mean(hits):>20.1%}")
#
# # Testing once at n = 5 gives the advertised 5%. Allowing yourself to keep
# # looking pushes it far higher, and it keeps climbing with the budget - with
# # unlimited data it reaches 100%. The p-value WILL dip below 0.05 eventually
# # just by wandering, and if you stop the moment it does, you always "win".
# #
# # This is why "we added a few more samples because the trend looked
# # promising" invalidates a p-value, and why clinical trials must pre-specify
# # their sample size or use a formal sequential design with corrected
# # boundaries (O'Brien-Fleming, alpha-spending).
# #
# # Note the pattern across all three problems and section 6b: the p-value is a
# # property of the PROCEDURE. Change the procedure - by trying variants, by
# # peeking, by choosing which hypotheses to test after looking - and the
# # number on the screen no longer means what its definition says.

# %% [markdown]
# ## What to take away
#
# 1. A confidence interval is a guarantee about the **procedure**: 95% of such
#    intervals contain the truth. Not "95% probability mine does".
# 2. **Read the width.** Two non-significant results can mean opposite things.
# 3. A p-value is $P(\text{data} \mid \text{no effect})$: never
#    $P(\text{no effect} \mid \text{data})$.
# 4. There are many kinds of p-value (parametric, permutation, rank,
#    bootstrap, exact) and one-vs-two-sided is a separate axis. Know which you
#    have.
# 5. $\alpha$ **is** your false positive rate; power is $1-\beta$. 0.05 is a
#    convention, not a law, and should follow the cost of each error.
# 6. With many tests, choose **FWER** (no false claims) or **FDR** (a tolerable
#    proportion of false claims) deliberately.
# 7. Every p-value describes a **procedure**. Trying variants, peeking, or
#    choosing hypotheses afterwards breaks it.
#
# **Next:** `F6_connecting_variables.py`
