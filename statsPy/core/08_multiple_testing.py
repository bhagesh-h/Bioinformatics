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
# # Module 08: Multiple testing and selective inference
#
# **Curriculum link:** `stats.md` -> Topic 8, equations (8.1)-(8.11)
#
# ## What you will learn
#
# 1. FWER (8.1) vs FDR (8.2): different guarantees, different use cases.
# 2. Bonferroni (8.3), Holm (8.4), BH (8.5)-(8.7), BY (8.8) implemented from
#    their definitions and checked against `statsmodels`.
# 3. That **Holm dominates Bonferroni**: there is never a reason to use
#    Bonferroni for FWER.
# 4. Storey's $\hat\pi_0$ and q-values (8.9)-(8.10): recovering the power that
#    BH leaves on the table.
# 5. Local FDR (8.11): why a gene *at* the threshold is much less certain than
#    the nominal 5%.
# 6. Independent filtering: and the outcome-dependent filtering that breaks it.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "08_multiple_testing"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. A realistic screen
#
# We simulate $G$ features of which a known subset is truly non-null, so we can
# *measure* the realised FWER and FDR rather than trusting the label.

# %%
header("1. Simulating a differential-expression screen with known truth")

def simulate_screen(G=10000, pi0=0.85, effect=1.1, n=6, seed=0):
    """Return p-values, effect estimates, mean expression, and truth labels."""
    rng = np.random.default_rng(seed)
    is_alt = rng.random(G) >= pi0
    # Mean expression governs POWER but is independent of the p-value under the
    # null - the requirement for valid independent filtering (section 6).
    base_mean = np.exp(rng.normal(2.0, 1.6, G))
    noise_sd = 1.0 / np.sqrt(1 + base_mean / 10)      # more expression, less noise
    delta = np.where(is_alt, effect * rng.choice([-1, 1], G), 0.0)
    a = rng.normal(delta[:, None], noise_sd[:, None], (G, n))
    b = rng.normal(0.0, noise_sd[:, None], (G, n))
    res = st.ttest_ind(a, b, axis=1, equal_var=False)
    return pd.DataFrame({
        "p": res.pvalue,
        "estimate": a.mean(axis=1) - b.mean(axis=1),
        "base_mean": base_mean,
        "is_alt": is_alt,
    })


screen = simulate_screen(seed=801)
G = len(screen)
print(f"  G = {G} features, true pi0 = {1 - screen['is_alt'].mean():.3f}")
print(f"  unadjusted p < 0.05: {(screen['p'] < 0.05).sum()} features")
print(f"    of which truly null: "
      f"{((screen['p'] < 0.05) & ~screen['is_alt']).sum()}")
print("  Testing 10,000 nulls at alpha=0.05 gives ~500 false positives by design.")

# %% [markdown]
# ## 2. The procedures, implemented from their definitions

# %%
header("2. Bonferroni (8.3), Holm (8.4), BH (8.5)-(8.7), BY (8.8)")

def bonferroni_adjust(p):
    """stats.md eq. (8.3)."""
    return np.minimum(1.0, len(p) * np.asarray(p))


def holm_adjust(p):
    """stats.md eq. (8.4), expressed as adjusted p-values with monotonicity."""
    p = np.asarray(p, float)
    G = len(p)
    order = np.argsort(p)
    adj = np.empty(G)
    running = 0.0
    for rank, idx in enumerate(order):          # rank = j-1, 0-based
        running = max(running, (G - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bh_adjust(p):
    """stats.md eq. (8.7): BH adjusted p-values (step-up, enforced monotone)."""
    p = np.asarray(p, float)
    G = len(p)
    order = np.argsort(p)
    ps = p[order]
    raw = ps * G / np.arange(1, G + 1)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]   # min over m >= j
    adj = np.empty(G)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def by_adjust(p):
    """stats.md eq. (8.8): BH with the harmonic-number penalty."""
    G = len(p)
    H_G = np.sum(1.0 / np.arange(1, G + 1))
    return np.minimum(1.0, bh_adjust(p) * H_G)


p = screen["p"].values
mine = {"bonferroni": bonferroni_adjust(p), "holm": holm_adjust(p),
        "BH": bh_adjust(p), "BY": by_adjust(p)}
theirs = {k: multipletests(p, method=m)[1] for k, m in
          [("bonferroni", "bonferroni"), ("holm", "holm"),
           ("BH", "fdr_bh"), ("BY", "fdr_by")]}
print(f"  {'method':<12}{'max |mine - statsmodels|':>26}")
for k in mine:
    print(f"  {k:<12}{np.max(np.abs(mine[k] - theirs[k])):>26.2e}")
print("  (all ~1e-16: the from-scratch implementations are exact)")

H_G = np.sum(1.0 / np.arange(1, G + 1))
print(f"\n  Harmonic number H_G for G={G}: {H_G:.3f}  (~ ln G + 0.5772 = "
      f"{np.log(G) + 0.5772:.3f})")
print(f"  BY therefore pays a ~{H_G:.1f}-fold penalty over BH.")

# %% [markdown]
# ## 3. Measuring the realised error rates
#
# Because we know the truth, we can check that each procedure delivers what it
# promises: and see what it costs in discoveries.

# %%
header("3. Realised FWER and FDR (8.1)-(8.2), and the power cost")

def evaluate(reject, is_alt):
    V = np.sum(reject & ~is_alt)            # false positives
    S = np.sum(reject & is_alt)             # true positives
    R = V + S
    return {"rejected": R, "true pos": S, "false pos": V,
            "FDP": V / max(R, 1), "sensitivity": S / max(is_alt.sum(), 1)}


rows = []
for name, adj in [("none", p)] + list(mine.items()):
    r = evaluate(adj < 0.05, screen["is_alt"].values)
    rows.append({"method": name, **r})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n  'FDP' is the realised false discovery PROPORTION in this one dataset;")
print("  FDR (8.2) is its EXPECTATION over repeated experiments - shown next.")

# %%
print("\n--- Averaging over 200 repeated screens ---")
res = {k: {"fwer": [], "fdp": [], "sens": []}
       for k in ["bonferroni", "holm", "BH", "BY"]}
for rep in range(200):
    s = simulate_screen(G=3000, seed=9000 + rep)
    truth = s["is_alt"].values
    pv = s["p"].values
    for name, fn in [("bonferroni", bonferroni_adjust), ("holm", holm_adjust),
                     ("BH", bh_adjust), ("BY", by_adjust)]:
        rej = fn(pv) < 0.05
        V = np.sum(rej & ~truth)
        res[name]["fwer"].append(V >= 1)
        res[name]["fdp"].append(V / max(rej.sum(), 1))
        res[name]["sens"].append(np.sum(rej & truth) / truth.sum())

print(f"  {'method':<12}{'FWER':>9}{'FDR':>9}{'sensitivity':>13}  guarantee")
guarantee = {"bonferroni": "FWER <= 0.05", "holm": "FWER <= 0.05",
             "BH": "FDR  <= 0.05", "BY": "FDR  <= 0.05 (any dependence)"}
for k in res:
    print(f"  {k:<12}{np.mean(res[k]['fwer']):>9.3f}"
          f"{np.mean(res[k]['fdp']):>9.3f}{np.mean(res[k]['sens']):>13.3f}"
          f"  {guarantee[k]}")

n_extra = sum(np.sum((holm_adjust(simulate_screen(G=3000, seed=9000+r)["p"].values) < 0.05))
              - np.sum((bonferroni_adjust(simulate_screen(G=3000, seed=9000+r)["p"].values) < 0.05))
              for r in range(40))
print("\n  Bonferroni and Holm both keep FWER at/below 0.05. Holm rejects")
print("  EVERYTHING Bonferroni rejects and sometimes more, so it weakly")
print(f"  dominates: across 40 screens it made {n_extra} extra rejection(s).")
print("  The two nearly coincide here because so few tests clear either")
print("  threshold - but Holm is never worse, so there is no reason to")
print("  prefer Bonferroni. BH trades controlled FDR for far more sensitivity.")
print("  Note BH's realised FDR sits BELOW 0.05: it is conservative by the")
print("  factor pi0 (eq. 8.6), which motivates Storey's correction.")

# %% [markdown]
# ## 4. Storey's $\hat\pi_0$ and q-values, eq. (8.9)-(8.10)

# %%
header("4. Recovering the power BH leaves on the table (8.9)-(8.10)")

def storey_pi0(p, lam=0.5):
    """stats.md eq. (8.9): estimate the null proportion from the flat tail."""
    p = np.asarray(p)
    return min(1.0, np.mean(p > lam) / (1.0 - lam))


def qvalues(p, lam=0.5):
    """stats.md eq. (8.10): Storey q-values."""
    p = np.asarray(p, float)
    G = len(p)
    pi0 = storey_pi0(p, lam)
    order = np.argsort(p)
    ps = p[order]
    raw = pi0 * G * ps / np.arange(1, G + 1)
    q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    q = np.empty(G)
    q[order] = np.minimum(q_sorted, 1.0)
    return q, pi0


q, pi0_hat = qvalues(p)
true_pi0 = 1 - screen["is_alt"].mean()
print(f"  true pi0        = {true_pi0:.4f}")
print(f"  Storey pi0_hat  = {pi0_hat:.4f}   (lambda = 0.5)")
print(f"\n  {'lambda':>8}{'pi0_hat':>10}")
for lam in [0.2, 0.4, 0.5, 0.6, 0.8]:
    print(f"  {lam:>8.1f}{storey_pi0(p, lam):>10.4f}")
print("  (pi0_hat is sensitive to lambda; the spline-smoothed version of")
print("   Storey's method averages over a range of lambda.)")

bh = bh_adjust(p)
truth = screen["is_alt"].values
print(f"\n  BH at 0.05      : {np.sum(bh < 0.05):>5} discoveries, "
      f"FDP = {np.sum((bh < 0.05) & ~truth)/max(np.sum(bh<0.05),1):.4f}")
print(f"  q-value at 0.05 : {np.sum(q < 0.05):>5} discoveries, "
      f"FDP = {np.sum((q < 0.05) & ~truth)/max(np.sum(q<0.05),1):.4f}")
print("  More discoveries, FDR still controlled.")

# %% [markdown]
# ## 5. Local FDR, eq. (8.11): the gene *at* the threshold
#
# $$\mathrm{fdr}(z)=\frac{\pi_0f_0(z)}{f(z)}$$
#
# FDR is the **average** over the whole rejection region. A feature that just
# scraped past the 5% threshold has a much higher probability of being null.

# %%
header("5. FDR is an average; local fdr is per-feature (8.11)")

sel = bh < 0.05
if sel.sum() > 30:
    ranked = np.argsort(p)
    ranked = ranked[sel[ranked]]                  # rejected, in p order
    k = len(ranked)
    edges = [(0, k // 5), (2 * k // 5, 3 * k // 5), (4 * k // 5, k)]
    labels = ["strongest 20%", "middle 20%", "weakest 20% (at threshold)"]
    print(f"  {'bin of the rejected list':<32}{'realised FDP':>14}")
    for (lo, hi), lab in zip(edges, labels):
        idx = ranked[lo:hi]
        fdp = np.mean(~truth[idx])
        print(f"  {lab:<32}{fdp:>14.3f}")
    print(f"\n  Overall FDP across all {k} rejections: "
          f"{np.mean(~truth[ranked]):.3f}")
    print("  The features near the cut-off are FAR more likely to be null than")
    print("  the nominal 5% suggests. Rank by effect size, and treat the tail")
    print("  of a significant list with appropriate scepticism.")

# %% [markdown]
# ## 6. Independent filtering: and how to break it
#
# Filtering on a statistic that is independent of the p-value **under the null**
# but predicts power (mean expression) raises discoveries while preserving FDR
# control. Filtering on anything **outcome-dependent** invalidates everything.

# %%
header("6. Independent filtering vs outcome-dependent filtering")

s = simulate_screen(G=10000, seed=802)
truth = s["is_alt"].values

# (a) No filter
rej_all = bh_adjust(s["p"].values) < 0.05

# (b) VALID: filter on mean expression, which never saw the group labels.
keep = s["base_mean"] > np.quantile(s["base_mean"], 0.40)
rej_filt = np.zeros(len(s), bool)
rej_filt[keep.values] = bh_adjust(s.loc[keep, "p"].values) < 0.05

# (c) INVALID: filter on the observed effect size, which is outcome-derived.
keep_bad = np.abs(s["estimate"]) > np.quantile(np.abs(s["estimate"]), 0.40)
rej_bad = np.zeros(len(s), bool)
rej_bad[keep_bad.values] = bh_adjust(s.loc[keep_bad, "p"].values) < 0.05

for name, rej in [("no filter", rej_all),
                  ("filter on mean expression (VALID)", rej_filt),
                  ("filter on |effect size| (INVALID)", rej_bad)]:
    V = np.sum(rej & ~truth)
    print(f"  {name:<36} rejected={rej.sum():>5}  "
          f"true pos={np.sum(rej & truth):>4}  FDP={V/max(rej.sum(),1):.4f}")

# Show the invalidity properly: run it under a COMPLETE null.
print("\n  Same three strategies under a COMPLETE null (no real effects at all):")
fdr_bad, fdr_good = [], []
for rep in range(120):
    sn = simulate_screen(G=2000, pi0=1.0, seed=7000 + rep)
    kg = sn["base_mean"] > np.quantile(sn["base_mean"], 0.40)
    kb = np.abs(sn["estimate"]) > np.quantile(np.abs(sn["estimate"]), 0.40)
    fdr_good.append(np.sum(bh_adjust(sn.loc[kg, "p"].values) < 0.05))
    fdr_bad.append(np.sum(bh_adjust(sn.loc[kb, "p"].values) < 0.05))
print(f"    mean false discoveries, expression filter : {np.mean(fdr_good):.2f}")
print(f"    mean false discoveries, effect-size filter: {np.mean(fdr_bad):.2f}")
print("  With NO real signal the valid filter finds ~0; the outcome-dependent")
print("  filter manufactures discoveries out of noise.")

# %% [markdown]
# ## 7. Defining the family
#
# This is a scientific decision, not a software default.

# %%
header("7. What is the family? (genes x cell types x contrasts)")

n_genes, n_celltypes, n_contrasts = 20000, 8, 3
print(f"  {n_genes} genes x {n_celltypes} cell types x {n_contrasts} contrasts "
      f"= {n_genes*n_celltypes*n_contrasts:,} hypotheses")
print(f"\n  {'adjustment family':<42}{'Bonferroni threshold':>22}")
for label, m in [("within one gene list only", n_genes),
                 ("genes x cell types", n_genes * n_celltypes),
                 ("everything", n_genes * n_celltypes * n_contrasts)]:
    print(f"  {label:<42}{0.05/m:>22.3e}")
print("\n  Adjusting within each cell type answers: 'among the genes I called in")
print("  THIS cell type and contrast, what fraction are wrong?' - a legitimate")
print("  but DIFFERENT claim from a global one. State which you mean.")

# %% [markdown]
# ## 8. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(screen.loc[~screen["is_alt"], "p"], bins=40, range=(0, 1),
             alpha=0.7, label="true null")
axes[0].hist(screen.loc[screen["is_alt"], "p"], bins=40, range=(0, 1),
             alpha=0.7, label="true alternative")
axes[0].axhline(pi0_hat * G / 40, color="red", ls="--",
                label=f"$\\hat\\pi_0$ level = {pi0_hat:.2f}")
axes[0].set_xlabel("p-value"); axes[0].set_title("Eq. (8.9): estimating $\\pi_0$",
                                                 fontsize=9)
axes[0].legend(fontsize=7)

ps = np.sort(p)
j = np.arange(1, G + 1)
axes[1].loglog(j, ps, ".", ms=1, label="sorted p")
axes[1].loglog(j, 0.05 * j / G, "r-", label="BH line $j\\alpha/G$")
axes[1].axhline(0.05 / G, color="g", ls="--", label="Bonferroni $\\alpha/G$")
axes[1].set_xlabel("rank $j$"); axes[1].set_ylabel("p-value")
axes[1].set_title("Eq. (8.5): the BH step-up rule", fontsize=9)
axes[1].legend(fontsize=7)

for k in res:
    axes[2].scatter(np.mean(res[k]["fdp"]), np.mean(res[k]["sens"]), s=90)
    axes[2].annotate(k, (np.mean(res[k]["fdp"]), np.mean(res[k]["sens"])),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
axes[2].axvline(0.05, color="red", ls="--", label="target FDR")
axes[2].set_xlabel("realised FDR"); axes[2].set_ylabel("sensitivity")
axes[2].set_title("The power/strictness trade-off", fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "multiple_testing.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/multiple_testing.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 8)
#
# 1. FDR (BH) for discovery screens; **Holm**: never plain Bonferroni - for a
#    handful of confirmatory claims.
# 2. Look at the p-value histogram (Module 06) first.
# 3. Filter only on outcome-independent statistics, and pre-specify the rule.
# 4. Rank the surviving list by effect size and its interval. A gene with
#    $q=0.001$ and $\mathrm{LFC}=0.05$ is precise and uninteresting.
#
# **Next:** `09_power_and_sample_size.py`
