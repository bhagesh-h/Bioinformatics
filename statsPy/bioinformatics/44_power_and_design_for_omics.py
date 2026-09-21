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
# # Applied 44: Power and design for omics experiments
#
# **Curriculum link:** `stats.md` -> Topic 44, equations (44.1)-(44.3)
# **Core modules used:** 01, 09, 40, 21
#
# ## The question
#
# Fixed budget. How many donors, how many cells per donor, how deep?
#
# ## A refinement to Module 01
#
# Module 01 showed that sub-samples hit a floor at $\sigma_d^2/n$ and concluded
# that replicates buy power and sub-samples do not. That is correct. This
# module locates the floor, and the location changes the practical advice: in
# scRNA-seq the per-cell noise is so large that cells DO buy power, steeply,
# until they suddenly stop.

# %%
import os
import warnings

import numpy as np
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODULE_NAME = "44_power_and_design_for_omics"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


rng = np.random.default_rng(44)

# %% [markdown]
# ## 1. Where the floor actually is, eq. (44.1)

# %%
header("1. Var(pseudobulk mean) = sigma_d^2/n + sigma_c^2/(nm)  (44.1)")

SD_DONOR = 0.30          # biological variation between donors, log scale
print(f"  donor SD = {SD_DONOR}. The per-cell SD depends on the gene's counts.\n")
print(f"  {'cells/donor':<14}" + "".join(f"{f'sigma_c={s}':>14}" for s in (0.5, 1.5, 4.0)))
for m in (1, 5, 20, 100, 500, 2000):
    row = ""
    for sc in (0.5, 1.5, 4.0):
        v = SD_DONOR ** 2 / 8 + sc ** 2 / (8 * m)
        row += f"{np.sqrt(v):>14.4f}"
    print(f"  {m:<14}{row}")
print(f"\n  floor (m -> infinity), n = 8 donors: {SD_DONOR/np.sqrt(8):.4f}")

print("""
  Read across a row and then down a column. With a well-measured gene
  (sigma_c = 0.5) the floor is reached by about 20 cells and extra cells are
  wasted, which is the Module 01 conclusion.

  With a lowly expressed gene (sigma_c = 4.0) you are still far above the
  floor at 500 cells per donor. For that gene, cells are the binding
  constraint, not donors.

  Both statements are eq. (44.1). Which one applies depends on the gene, and a
  real experiment contains both kinds at once.""")

# %% [markdown]
# ## 2. Simulated power across the three-way design

# %%
header("2. Power over donors, cells and depth")


def power_pseudobulk(n_donors, n_cells, depth, effect=0.5, n_sim=400, seed=0):
    """Simulate a two-group pseudobulk comparison.

    Per-cell counts are Poisson with mean proportional to depth, so the
    per-cell measurement noise falls as depth rises. Donor effects are the
    irreducible biological variation."""
    r = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        vals = []
        for grp in (0, 1):
            donor_mean = r.normal(grp * effect, SD_DONOR, n_donors)
            lam = np.exp(donor_mean)[:, None] * depth
            counts = r.poisson(lam, size=(n_donors, n_cells))
            # Pseudobulk: sum counts per donor, then log.
            vals.append(np.log(counts.sum(1) + 1) - np.log(n_cells * depth))
        hits += st.ttest_ind(vals[1], vals[0]).pvalue < 0.05
    return hits / n_sim


print(f"  effect = 0.5 on the log scale, depth = 1.0 (a moderately expressed gene)\n")
print(f"  {'donors':<9}" + "".join(f"{f'{m} cells':>12}" for m in (5, 20, 100, 500)))
for nd in (3, 5, 8, 15):
    row = "".join(f"{power_pseudobulk(nd, m, 1.0, seed=nd*10+m):>12.2f}"
                  for m in (5, 20, 100, 500))
    print(f"  {nd:<9}{row}")

print("""
  Read ACROSS: cells help, and then they stop helping. Read DOWN: donors keep
  helping all the way.

  The plateau is the floor from eq. (44.1). Past it, sequencing more cells
  from the same people adds cost and no information about the between-group
  difference. Before it, cells are genuinely buying power.

  The practical question is therefore not 'cells or donors' but 'where is the
  plateau for the genes I care about', and that depends on their expression.""")

# %% [markdown]
# ## 3. The same table for a lowly expressed gene

# %%
header("3. Depth moves the plateau")

print(f"  {'depth':<9}" + "".join(f"{f'{m} cells':>12}" for m in (5, 20, 100, 500)))
for dp in (0.1, 0.5, 2.0):
    row = "".join(f"{power_pseudobulk(6, m, dp, seed=int(dp*100)+m):>12.2f}"
                  for m in (5, 20, 100, 500))
    print(f"  {dp:<9.1f}{row}")

print("""
  At low depth the gene is barely detected per cell, so the per-cell noise is
  enormous and more cells keep helping well past 100. At high depth the
  plateau arrives early.

  This is the trade behind the standard advice that SHALLOW sequencing of MANY
  cells usually beats deep sequencing of few: for a fixed number of reads,
  spreading them over more cells reduces the sampling term in (44.1) faster
  than it raises per-cell precision. The advice is empirical, and it follows
  from this table rather than from a principle.""")

# %% [markdown]
# ## 4. Cost, eq. (44.2)

# %%
header("4. Optimising under a budget (44.2)")

C_DONOR, C_CELL = 1200.0, 0.30             # currency per donor, per cell
BUDGET = 20000.0
print(f"  budget {BUDGET:,.0f}; donor costs {C_DONOR:.0f}, cell costs {C_CELL}")
print("  Every row spends the whole budget. Two genes, same design.\n")
print(f"  {'donors':<9}{'cells each':>12}{'total cost':>13}"
      f"{'power, well-measured':>22}{'power, lowly expr.':>22}")
grid = [3, 5, 7, 9, 11, 13, 15, 16]
pw_hi, pw_lo = {}, {}
for nd in grid:
    m = int((BUDGET / nd - C_DONOR) / C_CELL)
    if m < 5:
        continue
    pw_hi[nd] = power_pseudobulk(nd, min(m, 2000), 1.0, n_sim=300, seed=nd)
    pw_lo[nd] = power_pseudobulk(nd, min(m, 20000), 0.02, n_sim=300,
                                 seed=nd + 500)
    cost = nd * (C_DONOR + m * C_CELL)
    print(f"  {nd:<9}{m:>12,}{cost:>13,.0f}{pw_hi[nd]:>22.2f}{pw_lo[nd]:>22.2f}")
print(f"\n  best design, well-measured gene : {max(pw_hi, key=pw_hi.get)} donors")
print(f"  best design, lowly expressed gene: {max(pw_lo, key=pw_lo.get)} donors")

print("""
  The two genes want different designs out of the same money.

  For the WELL-MEASURED gene the plateau from eq. (44.1) arrives after a few
  dozen cells, so every design on this table already clears it. Cells beyond
  that buy nothing, donors buy everything, and the best designs sit at the far
  edge of the table, where cells are an afterthought. This is Module 01's
  advice, and here it is correct.

  For the LOWLY EXPRESSED gene the plateau is thousands of cells away. Power
  rises with donors at first, peaks, and then FALLS: past the peak, each extra
  donor takes so much money from the cell budget that every donor is measured
  too badly to be worth having. That is a genuinely interior optimum, and no
  rule of thumb would have found it.

  Note this optimum is for ONE estimand, differential expression between
  groups. The next section shows a different estimand with a different
  optimum, using exactly the same money.""")

# %% [markdown]
# ## 5. A different question, a different design, eq. (44.3)

# %%
header("5. Detecting a rare cell type (44.3)")

print(f"  P(at least one cell of a type at frequency q) = 1 - (1-q)^m\n")
print(f"  {'cells/donor':<14}" + "".join(f"{f'q={q}':>12}" for q in (0.05, 0.01, 0.002)))
for m in (20, 100, 500, 2000, 5000):
    row = "".join(f"{1 - (1-q)**m:>12.3f}" for q in (0.05, 0.01, 0.002))
    print(f"  {m:<14}{row}")

print("""
  For a type at 0.2% frequency you need thousands of cells per donor just to
  SEE it, and seeing one cell is not the same as quantifying it: to estimate
  its abundance you need enough cells for the count to be stable.

  Compare with section 2, where 100 cells per donor was already past the
  plateau. Same money, opposite advice, because the estimand changed.

  This is the practical reason design questions cannot be answered generically.
  'How many cells should I sequence' has no answer until someone says what the
  experiment is for.""")

# %% [markdown]
# ## 6. Simulate the actual pipeline, not a formula

# %%
header("6. Closed-form power versus the pipeline you will run")

# Closed-form: two-sample t-test on donor means with known variance.
def closed_form_power(nd, m, effect=0.5, sc=1.2):
    var = SD_DONOR ** 2 + sc ** 2 / m
    se = np.sqrt(2 * var / nd)
    ncp = effect / se
    crit = st.t.ppf(0.975, 2 * nd - 2)
    return 1 - st.nct.cdf(crit, 2 * nd - 2, ncp) + st.nct.cdf(-crit, 2 * nd - 2, ncp)


print(f"  {'donors':<9}{'cells':<9}{'closed form':>14}{'simulated pipeline':>21}")
for nd, m in ((4, 50), (6, 50), (6, 200), (10, 200)):
    print(f"  {nd:<9}{m:<9}{closed_form_power(nd, m):>14.2f}"
          f"{power_pseudobulk(nd, m, 1.0, n_sim=400, seed=nd*m):>21.2f}")

print("""
  The closed form is in the right region but not reliable at small n, because
  it assumes a variance structure the pipeline does not exactly have: the
  pseudobulk log transform, the Poisson sampling and the estimated variance
  all matter.

  For a design that will be analysed with DESeq2, a mixed model or a
  pseudobulk pipeline, simulate THAT pipeline. A power calculation for a test
  you are not going to run is a number, not a plan. This is the ADEMP
  discipline of Module 40 applied to design.""")

# %% [markdown]
# ## 7. Figure

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ms = np.array([1, 2, 5, 10, 20, 50, 100, 300, 1000, 3000])
for sc, colour in ((0.5, "steelblue"), (1.5, "darkorange"), (4.0, "firebrick")):
    ax[0].plot(ms, np.sqrt(SD_DONOR**2 / 8 + sc**2 / (8 * ms)), "o-", color=colour,
               label=f"$\\sigma_c$ = {sc}")
ax[0].axhline(SD_DONOR / np.sqrt(8), color="black", ls="--", lw=1, label="floor")
ax[0].set_xscale("log"); ax[0].set_xlabel("cells per donor")
ax[0].set_ylabel("SE of the group mean")
ax[0].set_title("Eq. (44.1): where the floor is"); ax[0].legend(fontsize=7)

for nd, colour in ((3, "firebrick"), (6, "darkorange"), (12, "steelblue")):
    pw = [power_pseudobulk(nd, m, 1.0, n_sim=200, seed=nd + m) for m in (5, 20, 100, 500)]
    ax[1].plot((5, 20, 100, 500), pw, "o-", color=colour, label=f"{nd} donors")
ax[1].set_xscale("log"); ax[1].set_xlabel("cells per donor"); ax[1].set_ylabel("power")
ax[1].set_title("Cells plateau, donors do not"); ax[1].legend(fontsize=8)

mm = np.logspace(1, 4, 40)
for q, colour in ((0.05, "steelblue"), (0.01, "darkorange"), (0.002, "firebrick")):
    ax[2].plot(mm, 1 - (1 - q) ** mm, color=colour, lw=2, label=f"q = {q}")
ax[2].set_xscale("log"); ax[2].set_xlabel("cells per donor")
ax[2].set_ylabel("P(cell type observed)")
ax[2].set_title("Eq. (44.3): a different estimand"); ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "omics_power.png"), dpi=110)
plt.close(fig)
print(f"\nFigure written to {os.path.join(OUT, 'omics_power.png')}")

# %% [markdown]
# # PROBLEMS
#
# ### Problem 1: Spend a fixed budget three ways
#
# With the same money, find the best design for (a) differential expression,
# (b) finding a 1% cell type, and (c) quantifying a lowly expressed gene.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# BUD = 20000.0
# print(f"  {'donors':<9}{'cells':>9}{'DE power':>11}{'P(see 1% type)':>17}"
#       f"{'low-gene power':>17}")
# for nd in (3, 6, 10, 20):
#     m = int((BUD / nd - C_DONOR) / C_CELL)
#     if m < 5: continue
#     m = min(m, 3000)
#     de = power_pseudobulk(nd, min(m, 600), 1.0, n_sim=250, seed=nd)
#     see = 1 - (1 - 0.01) ** m
#     low = power_pseudobulk(nd, min(m, 600), 0.15, n_sim=250, seed=nd + 7)
#     print(f"  {nd:<9}{m:>9,}{de:>11.2f}{see:>17.3f}{low:>17.2f}")
#
# # Three columns, three different optima from the same budget. Differential
# # expression wants donors. Seeing a rare cell type wants cells, and with few
# # donors you can afford a great many. The lowly expressed gene sits between
# # them, because it needs both depth per cell and donors to average over.
# #
# # There is no design that is best at all three, which is the point. Decide
# # the primary estimand before the budget meeting, and state explicitly what
# # the design is NOT powered for. A study powered for DE that then reports a
# # rare-population finding is reporting something its design cannot support.

# %% [markdown]
# ### Problem 2: How wrong is a power calculation with a guessed variance?
#
# Power calculations need $\sigma_d$, which you rarely know in advance.
# Quantify the consequences of guessing it wrong.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# TRUE_SD = 0.30
# print(f"  true donor SD = {TRUE_SD}. Design for 80% power at effect 0.5.\n")
# print(f"  {'assumed SD':<14}{'donors planned':>16}{'ACTUAL power':>15}")
# for guess in (0.15, 0.22, 0.30, 0.45, 0.60):
#     nd = 3
#     while nd < 200:
#         var = guess ** 2 + 1.2 ** 2 / 100
#         se = np.sqrt(2 * var / nd)
#         if 1 - st.nct.cdf(st.t.ppf(0.975, 2*nd-2), 2*nd-2, 0.5/se) > 0.80:
#             break
#         nd += 1
#     actual = power_pseudobulk(nd, 100, 1.0, n_sim=500, seed=int(guess*100))
#     print(f"  {guess:<14.2f}{nd:>16}{actual:>15.2f}")
#
# # Underestimating the donor SD by a factor of two roughly halves the achieved
# # power, and the study is planned, funded and run before anyone finds out.
# # Overestimating wastes money but is at least safe.
# #
# # Two responses. Estimate sigma_d from PILOT or PUBLIC data on the same
# # tissue and platform rather than guessing, which is what tools like scPower
# # do. And report power across a RANGE of plausible variances rather than a
# # single number, so the reader can see how fragile the plan is.

# %% [markdown]
# ### Problem 3: Does adding donors always beat adding cells?
#
# Find the regime where the answer is no.

# %%
# ---- YOUR CODE HERE ----------------------------------------------------


# ---- SOLUTION (uncomment to check) -------------------------------------
# print(f"  Starting from 6 donors x 30 cells. Which upgrade helps more?\n")
# print(f"  {'gene depth':<14}{'baseline':>11}{'+2 donors':>12}{'x4 cells':>11}"
#       f"{'better buy':>13}")
# for dp in (0.05, 0.2, 1.0, 4.0):
#     base = power_pseudobulk(6, 30, dp, n_sim=500, seed=int(dp*100))
#     more_d = power_pseudobulk(8, 30, dp, n_sim=500, seed=int(dp*100)+1)
#     more_c = power_pseudobulk(6, 120, dp, n_sim=500, seed=int(dp*100)+2)
#     better = "donors" if more_d > more_c else "cells"
#     print(f"  {dp:<14.2f}{base:>11.2f}{more_d:>12.2f}{more_c:>11.2f}{better:>13}")
#
# # At high depth the gene is measured well in every cell, the per-cell term in
# # (44.1) is already small, and donors win, exactly as Module 01 says.
# #
# # At low depth the per-cell term dominates and quadrupling the cells wins,
# # sometimes clearly. Module 01's conclusion is not wrong; it describes the
# # regime past the plateau, and lowly expressed genes are not in it.
# #
# # The honest summary is that 'donors, not cells' is a good DEFAULT and a bad
# # RULE. Check where your genes of interest sit relative to the plateau before
# # committing a budget, because for the lowly expressed genes that motivate
# # many single-cell studies the default advice is backwards.

# %% [markdown]
# ## What to take away
#
# 1. Eq. (44.1) is eq. (1.5). What changed is knowing **where the floor is**,
#    and for lowly expressed genes it is far out.
# 2. Cells buy power steeply and then stop. Donors keep buying it.
# 3. Depth moves the plateau, which is why shallow-and-many usually beats
#    deep-and-few for a fixed read budget.
# 4. **The estimand decides the design.** DE wants donors, rare populations
#    want cells, and the same budget gives different optima.
# 5. Simulate the pipeline you will actually run, not a t-test you will not.
#
# **Next:** `45_longitudinal_causal.py`
