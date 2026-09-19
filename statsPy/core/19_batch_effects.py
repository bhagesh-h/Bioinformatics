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
# # Module 19: Batch effects and unwanted variation
#
# **Curriculum link:** `stats.md` -> Topic 19, equations (19.1)-(19.6)
#
# ## What you will learn
#
# 1. The location-scale model (19.1) and a from-scratch ComBat-style correction
#    (19.2): including **why the biological design must be supplied**.
# 2. Identifiability (19.3): complete confounding is unfixable, and partial
#    confounding is quantifiable.
# 3. **Model-based adjustment (19.4) vs correcting the matrix**: and why the
#    former is right for inference.
# 4. Surrogate variables (19.5) when batch is unrecorded.
# 5. The two-sided diagnostic: batch removal **and** biology preservation.

# %%
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "19_batch_effects"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Simulating the location-scale model, eq. (19.1)
#
# $$Y_{ijg}=\alpha_g+\mathbf x_j^\top\boldsymbol\beta_g+\gamma_{ig}+\delta_{ig}\varepsilon_{ijg}$$

# %%
header("1. A multi-batch experiment with known truth (19.1)")

def simulate_batches(n_per_group=12, G=2000, n_de=200, effect=1.8,
                     batch_shift=1.5, batch_scale=1.6, confounding=0.0, seed=0):
    """Two conditions across two batches.

    `confounding` in [0,1] controls how strongly condition and batch are
    associated: 0 = perfectly balanced, 1 = completely confounded.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    condition = np.array(["ctrl"] * n_per_group + ["trt"] * n_per_group)
    # Assign batch: with probability `confounding` follow the condition.
    follow = rng.random(n) < confounding
    batch = np.where(follow,
                     np.where(condition == "ctrl", "b1", "b2"),
                     rng.choice(["b1", "b2"], n))
    alpha = rng.normal(8.0, 1.5, G)                    # baseline expression
    beta = np.zeros(G)
    beta[:n_de] = effect * rng.choice([-1, 1], n_de)   # TRUE differential genes
    gamma = rng.normal(0, batch_shift, G)              # batch mean shift
    delta = np.exp(rng.normal(0, np.log(batch_scale), G))  # batch scale factor

    Y = np.empty((G, n))
    for j in range(n):
        shift = gamma if batch[j] == "b2" else 0.0
        scale = delta if batch[j] == "b2" else 1.0
        eff = beta * (condition[j] == "trt")
        Y[:, j] = alpha + eff + shift + scale * rng.normal(0, 1.0, G)

    meta = pd.DataFrame({"condition": condition, "batch": batch},
                        index=[f"S{i:02d}" for i in range(n)])
    expr = pd.DataFrame(Y, index=[f"G{i:04d}" for i in range(G)],
                        columns=meta.index)
    truth = np.zeros(G, bool); truth[:n_de] = True
    return expr, meta, truth


expr, meta, truth = simulate_batches(seed=1901)
print(pd.crosstab(meta["condition"], meta["batch"]))
print(f"\n  {truth.sum()} of {len(truth)} genes are truly differential")

# %% [markdown]
# ## 2. Diagnose before correcting: PC x metadata, eq. (17.6)

# %%
header("2. Which factor dominates the variance?")

def pc_metadata_table(expr, meta, n_pc=4):
    X = (expr.T - expr.T.mean(axis=0)).values
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores, pve = U * S, S**2 / np.sum(S**2)
    rows = []
    for j in range(n_pc):
        z = scores[:, j]
        row = {"PC": j + 1, "PVE": pve[j]}
        for col in meta.columns:
            D = pd.get_dummies(meta[col], drop_first=True).astype(float).values
            Xd = np.column_stack([np.ones(len(z)), D])
            resid = z - Xd @ np.linalg.lstsq(Xd, z, rcond=None)[0]
            row[col] = 1 - resid.var() / z.var()
        rows.append(row)
    return pd.DataFrame(rows).set_index("PC"), scores, pve


tab, scores_raw, pve_raw = pc_metadata_table(expr, meta)
print(tab.round(3).to_string())
print("\n  PC1 is dominated by BATCH. Build this table before interpreting")
print("  anything (stats.md eq. 17.6).")

# %% [markdown]
# ## 3. Identifiability, eq. (19.3)
#
# Complete confounding is not a hard problem: it is an **unanswerable** one.

# %%
header("3. Complete vs partial confounding (19.3)")

def design_rank(meta):
    A = pd.get_dummies(meta["condition"], drop_first=True).astype(float).values
    B = pd.get_dummies(meta["batch"], drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(meta)), A, B])
    return np.linalg.matrix_rank(X), X.shape[1]


print(f"  {'confounding':<14}{'cond x batch table':<26}{'rank/ncol':>12}"
      f"{'R^2(cond~batch)':>18}{'VIF':>8}")
for conf in [0.0, 0.5, 0.8, 1.0]:
    e, m, _ = simulate_batches(confounding=conf, seed=1902 + int(conf * 10))
    ct = pd.crosstab(m["condition"], m["batch"]).values.ravel()
    r, nc = design_rank(m)
    cvec = (m["condition"] == "trt").astype(float).values
    bvec = (m["batch"] == "b2").astype(float).values
    r2 = np.corrcoef(cvec, bvec)[0, 1] ** 2
    vif = 1 / (1 - r2) if r2 < 1 else np.inf
    print(f"  {conf:<14.1f}{str(ct):<26}{f'{r}/{nc}':>12}{r2:>18.3f}{vif:>8.2f}")

print("\n  At confounding = 1.0 the design matrix is RANK DEFICIENT: the")
print("  condition effect is not estimable by ANY method. At 0.8 it is")
print("  estimable but the SE is inflated by sqrt(VIF) - recoverable but")
print("  expensive. The correct response to the first case is to report it.")

# %% [markdown]
# ## 4. ComBat-style correction from scratch, eq. (19.2)

# %%
header("4. Location-scale correction, with and without the design (19.2)")

def combat_like(Y, batch, mod=None):
    """A from-scratch location-scale batch correction (stats.md eq. 19.2).

    This is the non-empirical-Bayes core of ComBat. `mod` is the biological
    design matrix; supplying it is what PROTECTS the biology from being
    removed along with the batch effect.
    """
    Y = np.asarray(Y, float)
    G, n = Y.shape
    batch = np.asarray(batch)
    levels = np.unique(batch)

    # Design: intercept + biology (if given) + batch indicators.
    B = pd.get_dummies(pd.Series(batch)).astype(float).values   # full dummy set
    X = B if mod is None else np.column_stack([B, mod])
    # Estimate the grand mean and biological coefficients with batch present.
    coef, *_ = np.linalg.lstsq(X, Y.T, rcond=None)              # (p x G)
    n_b = B.shape[1]
    batch_means = coef[:n_b]                                     # gamma per batch
    # Grand mean weighted by batch size (the standardisation reference).
    w = np.array([(batch == l).sum() for l in levels], float) / n
    grand = w @ batch_means
    bio = np.zeros((1, G)) if mod is None else mod @ coef[n_b:]

    # Standardised residuals, then per-batch scale factors (delta).
    fitted = X @ coef
    resid = Y.T - fitted
    pooled_var = (resid ** 2).sum(axis=0) / (n - X.shape[1])
    Ystar = np.empty_like(Y)
    for i, l in enumerate(levels):
        sel = batch == l
        delta2 = (resid[sel] ** 2).mean(axis=0) / np.maximum(pooled_var, 1e-12)
        delta = np.sqrt(np.maximum(delta2, 1e-12))
        # eq. (19.2): remove batch mean and scale, then add back grand mean+bio
        centred = (Y[:, sel].T - fitted[sel]) / delta
        Ystar[:, sel] = (centred + grand + (bio[sel] if mod is not None else 0)).T
    return Ystar


def de_test(Ymat, meta):
    a = Ymat[:, (meta["condition"] == "ctrl").values]
    b = Ymat[:, (meta["condition"] == "trt").values]
    return st.ttest_ind(a, b, axis=1).pvalue


def de_adjusted(Ymat, meta):
    """Model-based adjustment, stats.md eq. (19.4): batch as a covariate."""
    cond = (meta["condition"] == "trt").astype(float).values
    bat = (meta["batch"] == "b2").astype(float).values
    Xd = np.column_stack([np.ones(len(cond)), cond, bat])
    coef, *_ = np.linalg.lstsq(Xd, Ymat.T, rcond=None)
    resid = Ymat.T - Xd @ coef
    dof = len(cond) - Xd.shape[1]
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
    return 2 * st.t.sf(np.abs(coef[1] / se), dof)


def evaluate(pv, truth, label):
    q = st.false_discovery_control(pv)
    rej = q < 0.05
    tp, fp = np.sum(rej & truth), np.sum(rej & ~truth)
    print(f"  {label:<40}{rej.sum():>7}{tp:>7}{fp:>7}"
          f"{tp/max(truth.sum(),1):>10.2f}{fp/max(rej.sum(),1):>8.3f}")


def compare_all(conf, seed):
    e, m, tr = simulate_batches(confounding=conf, seed=seed)
    Ym = e.values
    bio = pd.get_dummies(m["condition"], drop_first=True).astype(float).values
    print(f"\n  --- confounding = {conf:.1f};  condition x batch table = "
          f"{pd.crosstab(m['condition'], m['batch']).values.ravel()} ---")
    print(f"  {'analysis':<40}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>10}{'FDP':>8}")
    evaluate(de_test(Ym, m), tr, "no correction, no adjustment")
    evaluate(de_test(combat_like(Ym, m["batch"].values, None), m), tr,
             "matrix corrected WITHOUT the design")
    evaluate(de_test(combat_like(Ym, m["batch"].values, bio), m), tr,
             "matrix corrected WITH the design")
    evaluate(de_adjusted(Ym, m), tr, "batch IN THE MODEL (19.4)")
    return e, m, tr


# Balanced design: correction without covariates is HARMLESS, because
# condition is orthogonal to batch. This is the case for good study design.
compare_all(0.0, 1901)

# Partially confounded design: now the design matters enormously.
expr_c, meta_c, truth_c = compare_all(0.6, 1911)

print("\n  Read the two blocks together:")
print("   * BALANCED: every strategy works. Removing batch means cannot touch")
print("     the condition effect because the two are orthogonal. This is the")
print("     strongest argument for balanced allocation at the design stage.")
print("   * CONFOUNDED: correcting WITHOUT the design strips out real biology")
print("     along with the batch, because part of the condition effect lives")
print("     in the batch means it is removing. Supplying the design, or")
print("     adjusting in the model, protects it.")
print("\n  Look at the FDP column in the confounded block. Model-based")
print("  adjustment (19.4) holds FDP near the nominal 0.05, while the")
print("  design-aware CORRECTED MATRIX recovers sensitivity but badly")
print("  overshoots FDP. That gap is the cost of treating Y* as observed:")
print("  the correction was ESTIMATED, and a corrected matrix discards that")
print("  uncertainty, whereas putting batch in the design propagates it into")
print("  every standard error. This is why the rule is: correct the matrix for")
print("  VISUALISATION, adjust in the model for INFERENCE.")

# Keep the balanced objects around for the diagnostics section.
Y = expr.values
bio_mod = pd.get_dummies(meta["condition"], drop_first=True).astype(float).values
Y_without = combat_like(Y, meta["batch"].values, None)
Y_with = combat_like(Y, meta["batch"].values, bio_mod)

# %% [markdown]
# ## 5. Surrogate variables when batch is unrecorded, eq. (19.5)

# %%
header("5. Estimating unrecorded batch as surrogate variables (19.5)")

expr2, meta2, truth2 = simulate_batches(seed=1903)
Y2 = expr2.values
bio2 = pd.get_dummies(meta2["condition"], drop_first=True).astype(float).values
Xbio = np.column_stack([np.ones(len(meta2)), bio2])

# eq. (19.5): remove the biological model, decompose the residuals.
coef_bio, *_ = np.linalg.lstsq(Xbio, Y2.T, rcond=None)
R = Y2.T - Xbio @ coef_bio
U, S, Vt = np.linalg.svd(R, full_matrices=False)
n_sv = 1
SV = U[:, :n_sv]

hidden = (meta2["batch"] == "b2").astype(float).values
print(f"  |correlation(SV1, true hidden batch)| = "
      f"{abs(np.corrcoef(SV[:, 0], hidden)[0, 1]):.4f}")
print(f"  residual variance explained by SV1    = {S[0]**2/np.sum(S**2):.1%}")

def de_with_sv(Ymat, bio, svs):
    Xd = np.column_stack([np.ones(len(bio)), bio, svs])
    coef, *_ = np.linalg.lstsq(Xd, Ymat.T, rcond=None)
    resid = Ymat.T - Xd @ coef
    dof = Ymat.shape[1] - Xd.shape[1]
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
    return 2 * st.t.sf(np.abs(coef[1] / se), dof)


print(f"\n  {'analysis':<44}{'rej':>7}{'TP':>7}{'FP':>7}{'sens':>10}{'FDP':>8}")
evaluate(de_test(Y2, meta2), truth2, "ignore the hidden batch")
evaluate(de_with_sv(Y2, bio2.ravel(), SV), truth2, "adjust for 1 surrogate variable")
evaluate(de_adjusted(Y2, meta2), truth2, "adjust for the TRUE batch (oracle)")
print("\n  The surrogate variable recovers most of what knowing the true batch")
print("  would have given. RUV is the alternative, using negative-control")
print("  features (eq. 19.6) - valid only if those controls really are")
print("  unaffected by the biology, which is an assumption to defend.")

# %% [markdown]
# ## 6. The two-sided diagnostic
#
# Report a **mixing** metric and a **biology-preservation** metric. Reporting
# only mixing rewards overcorrection: the dominant failure mode.

# %%
header("6. Batch removal AND biology preservation")

def batch_r2_pc1(Ymat, meta):
    X = (Ymat.T - Ymat.T.mean(axis=0))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    z = (U * S)[:, 0]
    out = {}
    for col in ["batch", "condition"]:
        D = pd.get_dummies(meta[col], drop_first=True).astype(float).values
        Xd = np.column_stack([np.ones(len(z)), D])
        resid = z - Xd @ np.linalg.lstsq(Xd, z, rcond=None)[0]
        out[col] = 1 - resid.var() / z.var()
    return out


def knn_batch_mixing(Ymat, meta, k=5):
    """Fraction of each sample's k nearest neighbours from the OTHER batch.

    A simple kBET-style mixing score. Perfect mixing gives ~0.5 for two
    equally sized batches; 0 means batches are completely separate.
    """
    X = (Ymat.T - Ymat.T.mean(axis=0))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    b = meta["batch"].values
    frac = []
    for i in range(len(b)):
        nn = np.argsort(D[i])[:k]
        frac.append(np.mean(b[nn] != b[i]))
    return float(np.mean(frac))


Yc = expr_c.values
bio_c = pd.get_dummies(meta_c["condition"], drop_first=True).astype(float).values
print("  (using the PARTIALLY CONFOUNDED dataset, where the choice matters)")
print(f"  {'matrix':<40}{'PC1 R^2 batch':>15}{'PC1 R^2 cond':>14}"
      f"{'kNN mixing':>12}{'true-DE sens':>14}")
for label, M in [("uncorrected", Yc),
                 ("corrected WITHOUT design", combat_like(Yc, meta_c["batch"].values, None)),
                 ("corrected WITH design", combat_like(Yc, meta_c["batch"].values, bio_c))]:
    r2 = batch_r2_pc1(M, meta_c)
    q = st.false_discovery_control(de_test(M, meta_c))
    sens = np.sum((q < 0.05) & truth_c) / truth_c.sum()
    print(f"  {label:<40}{r2['batch']:>15.3f}{r2['condition']:>14.3f}"
          f"{knn_batch_mixing(M, meta_c):>12.3f}{sens:>14.2f}")

print("\n  Both corrections mix the batches equally well (the mixing column).")
print("  Only the design-aware one PRESERVES the biology (the sensitivity")
print("  column). If you report ONLY a mixing metric you will score the")
print("  overcorrected matrix as the best - the exact failure mode that recent")
print("  batch-effect reviews warn about (stats.md Topic 19).")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (label, M) in zip(axes, [
        ("uncorrected (confounded)", Yc),
        ("corrected WITHOUT design", combat_like(Yc, meta_c["batch"].values, None)),
        ("corrected WITH design", combat_like(Yc, meta_c["batch"].values, bio_c))]):
    X = (M.T - M.T.mean(axis=0))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    sc, pv = U * S, S**2 / np.sum(S**2)
    for b, mk in zip(["b1", "b2"], ["o", "s"]):
        for c, col in zip(["ctrl", "trt"], ["C0", "C3"]):
            sel = ((meta_c["batch"] == b) & (meta_c["condition"] == c)).values
            ax.scatter(sc[sel, 0], sc[sel, 1], marker=mk, color=col, s=55,
                       label=f"{b}/{c}")
    ax.set_xlabel(f"PC1 ({pv[0]:.0%})"); ax.set_ylabel(f"PC2 ({pv[1]:.0%})")
    ax.set_title(label, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
axes[0].legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "batch_effects.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/batch_effects.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 19)
#
# 1. **Prevent**: randomise samples across batches, balance conditions within
#    batch, include bridging/reference samples, record every processing variable.
# 2. **Diagnose before correcting**: cross-tabulate condition x batch, check the
#    design rank, and build the PC x metadata table.
# 3. If completely confounded, **stop and say so**.
# 4. Adjust in the model (19.4) for inference; correct the matrix (19.2) only
#    for visualisation/clustering/prediction: and keep the correction inside
#    cross-validation folds (Module 21).
# 5. Always evaluate **both** batch removal and biology preservation.
#
# **Next:** `20_survival_and_longitudinal.py`
