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
# # Module 16: Matrix algebra for high-dimensional biology
#
# **Curriculum link:** `stats.md` -> Topic 16, equations (16.1)-(16.11)
#
# ## What you will learn
#
# 1. Projection (16.2): and that **simple regression, correlation and
#    projection are one idea**.
# 2. The centering matrix (16.3) and the covariance/Gram duality (16.4): when
#    $p\gg n$, decompose the small matrix.
# 3. Rank and identifiability (16.5): the algebra of a confounded design.
# 4. SVD (16.6)-(16.7) and the Eckart-Young theorem (16.8).
# 5. The pseudoinverse (16.9) and the condition number (16.10).
# 6. Quadratic forms (16.11): why ANOVA sums of squares are $\chi^2$.

# %%
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_NAME = "16_matrix_algebra"
OUT = os.path.join(os.environ.get("STATS_OUT", "results"), MODULE_NAME)
os.makedirs(OUT, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def header(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# %% [markdown]
# ## 1. Projection = regression = correlation, eq. (16.1)-(16.2)

# %%
header("1. Three names for one operation (16.1)-(16.2)")

rng = np.random.default_rng(1601)
n = 50
x = rng.normal(0, 1, n)
y = 1.7 * x + rng.normal(0, 1, n)

xc, yc = x - x.mean(), y - y.mean()

proj_coef = (xc @ yc) / (xc @ xc)                  # eq. (16.2)
ols_slope = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y,
                            rcond=None)[0][1]
corr = (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))

print(f"  projection coefficient  <x,y>/<x,x> = {proj_coef:.8f}")
print(f"  OLS slope                           = {ols_slope:.8f}")
print(f"  Pearson r                           = {corr:.8f}")
print(f"  r * ||y||/||x||                     = "
      f"{corr * np.linalg.norm(yc)/np.linalg.norm(xc):.8f}   <- equals the slope")
print(f"\n  cos(angle between centred vectors) = "
      f"{xc @ yc / (np.linalg.norm(xc)*np.linalg.norm(yc)):.6f}")
print("  Correlation IS the cosine of the angle between the centred vectors.")

# %% [markdown]
# ## 2. Centering matrix and the covariance/Gram duality, eq. (16.3)-(16.4)
#
# $\mathbf S$ ($p\times p$) and $\mathbf K$ ($n\times n$) share the same
# non-zero eigenvalues. **When $p\gg n$, decompose the small one.**

# %%
header("2. S and K share eigenvalues - use the smaller one (16.3)-(16.4)")

n_samp, p_feat = 12, 4000
X = rng.normal(0, 1, (n_samp, p_feat))
C = np.eye(n_samp) - np.ones((n_samp, n_samp)) / n_samp     # eq. (16.3)
Xc = C @ X

print(f"  C symmetric? {np.allclose(C, C.T)}   idempotent? "
      f"{np.allclose(C @ C, C)}   rank(C) = {np.linalg.matrix_rank(C)} "
      f"(= n-1 = {n_samp-1})")
print(f"  column means after centering: max |mean| = "
      f"{np.abs(Xc.mean(axis=0)).max():.2e}")

t0 = time.time()
S = Xc.T @ Xc / (n_samp - 1)                        # p x p  (4000 x 4000)
ev_S = np.linalg.eigvalsh(S)[::-1][:n_samp]
t_S = time.time() - t0

t0 = time.time()
K = Xc @ Xc.T                                       # n x n  (12 x 12)
ev_K = np.linalg.eigvalsh(K)[::-1] / (n_samp - 1)
t_K = time.time() - t0

print(f"\n  {n_samp} samples x {p_feat} features")
print(f"  top 5 eigenvalues from the {p_feat}x{p_feat} covariance : {ev_S[:5]}")
print(f"  top 5 eigenvalues from the {n_samp}x{n_samp} Gram matrix : {ev_K[:5]}")
print(f"  max |difference| = {np.max(np.abs(ev_S[:n_samp-1] - ev_K[:n_samp-1])):.2e}")
print(f"\n  timing: covariance route {t_S*1000:8.2f} ms")
print(f"          Gram route       {t_K*1000:8.2f} ms   "
      f"({t_S/max(t_K,1e-9):.0f}x faster)")
print(f"  Note rank: at most min(n-1, p) = {min(n_samp-1, p_feat)} non-zero")
print("  eigenvalues exist, no matter how many features you measured.")

# %% [markdown]
# ## 3. Rank and identifiability, eq. (16.5)
#
# This is the algebra behind "your design cannot answer the question".

# %%
header("3. Rank deficiency = an unanswerable design (16.5)")

def design_report(condition, batch, label):
    cond = pd.get_dummies(pd.Series(condition), drop_first=True).astype(float)
    bat = pd.get_dummies(pd.Series(batch), drop_first=True).astype(float)
    Xd = np.column_stack([np.ones(len(condition)), cond.values, bat.values])
    r = np.linalg.matrix_rank(Xd)
    sv = np.linalg.svd(Xd, compute_uv=False)
    print(f"  {label}")
    print(f"    shape {Xd.shape}, rank {r}, singular values "
          f"{np.round(sv, 4)}")
    if r < Xd.shape[1]:
        print(f"    -> RANK DEFICIENT: {Xd.shape[1]-r} direction(s) unidentifiable")
        # Find the null-space vector: the exact linear dependency.
        _, _, Vt = np.linalg.svd(Xd)
        print(f"    null-space vector (the aliasing): {np.round(Vt[-1], 4)}")
    else:
        print(f"    -> full rank: every effect is estimable")
    return r == Xd.shape[1]


design_report(["ctrl", "trt"] * 6, np.repeat(["b1", "b2", "b3"], 4),
              "Balanced design:")
print()
design_report(["ctrl"] * 6 + ["trt"] * 6, ["b1"] * 6 + ["b2"] * 6,
              "Confounded design:")
print("\n  The null-space vector states the exact linear combination that the")
print("  data cannot distinguish. No software resolves this - only a new design.")

# %% [markdown]
# ## 4. SVD and Eckart-Young, eq. (16.6)-(16.8)

# %%
header("4. SVD properties and optimal low-rank approximation (16.6)-(16.8)")

rng = np.random.default_rng(1602)
# A matrix with genuine rank-3 structure plus noise.
n_s, p_f, true_rank = 40, 200, 3
Ztrue = rng.normal(0, 1, (n_s, true_rank))
Wtrue = rng.normal(0, 1, (true_rank, p_f))
A = Ztrue @ Wtrue + rng.normal(0, 0.5, (n_s, p_f))

U, D, Vt = np.linalg.svd(A, full_matrices=False)     # eq. (16.6)
print(f"  A is {A.shape}; SVD gives U{U.shape}, D{D.shape}, Vt{Vt.shape}")
print(f"  U orthonormal? {np.allclose(U.T @ U, np.eye(U.shape[1]))}")
print(f"  V orthonormal? {np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))}")
print(f"  reconstruction error ||A - U D Vt|| = "
      f"{np.linalg.norm(A - U @ np.diag(D) @ Vt):.2e}")
print(f"\n  first 8 singular values: {np.round(D[:8], 3)}")
print("  The gap after the 3rd value reveals the true rank.")

# eq. (16.7): SVD gives both eigendecompositions at once.
ev_ata = np.linalg.eigvalsh(A.T @ A)[::-1][:5]
print(f"\n  eigenvalues of A'A : {np.round(ev_ata, 3)}")
print(f"  D^2                : {np.round(D[:5]**2, 3)}   <- eq. (16.7)")

# eq. (16.8): Eckart-Young.
print(f"\n  Eckart-Young check - ||A - A_k||_F^2 should equal sum_{{j>k}} d_j^2:")
print(f"  {'k':>4}{'actual':>14}{'sum d_j^2 (j>k)':>18}{'random rank-k':>16}")
for k in [1, 2, 3, 5, 10]:
    Ak = U[:, :k] @ np.diag(D[:k]) @ Vt[:k]
    actual = np.linalg.norm(A - Ak) ** 2
    theory = np.sum(D[k:] ** 2)
    # A random rank-k approximation, for comparison
    Rq = rng.normal(0, 1, (p_f, k))
    Q, _ = np.linalg.qr(Rq)
    Arand = A @ Q @ Q.T
    print(f"  {k:>4}{actual:>14.3f}{theory:>18.3f}"
          f"{np.linalg.norm(A - Arand)**2:>16.3f}")
print("  The truncated SVD is provably OPTIMAL; a random subspace is far worse.")

# %% [markdown]
# ## 5. Pseudoinverse and condition number, eq. (16.9)-(16.10)

# %%
header("5. Pseudoinverse and numerical conditioning (16.9)-(16.10)")

# A rank-deficient design: infinitely many solutions exist.
Xrd = np.array([[1., 1., 0.],
                [1., 1., 0.],
                [1., 0., 1.],
                [1., 0., 1.]])
yrd = np.array([2.0, 2.2, 3.1, 2.9])
print(f"  design shape {Xrd.shape}, rank {np.linalg.matrix_rank(Xrd)} "
      f"-> NOT full rank")
beta_pinv = np.linalg.pinv(Xrd) @ yrd               # eq. (16.9)
print(f"  minimum-norm solution: {np.round(beta_pinv, 4)}, "
      f"norm = {np.linalg.norm(beta_pinv):.4f}")
# Another exact solution, found by adding a null-space vector.
_, _, Vt_rd = np.linalg.svd(Xrd)
null_vec = Vt_rd[-1]
alt = beta_pinv + 3.0 * null_vec
print(f"  another exact solution: {np.round(alt, 4)}, "
      f"norm = {np.linalg.norm(alt):.4f}")
print(f"  both fit equally well: RSS {np.sum((yrd - Xrd@beta_pinv)**2):.6f} "
      f"vs {np.sum((yrd - Xrd@alt)**2):.6f}")
print("  The pinv picks the minimum-norm one. Useful - but if you needed it,")
print("  your design was rank-deficient and you should know why.")

print(f"\n  Condition number (16.10) and the cost of the normal equations:")
print("  {:<26}{:>13}{:>14}{:>13}".format("design", "kappa(X)",
                                              "kappa(X'X)", "digits lost"))
for label, M in [("well-conditioned", rng.normal(0, 1, (60, 5))),
                 ("mild collinearity", None), ("severe collinearity", None)]:
    if label == "mild collinearity":
        b = rng.normal(0, 1, 60)
        M = np.column_stack([np.ones(60), b, b + rng.normal(0, 0.3, 60)])
    elif label == "severe collinearity":
        b = rng.normal(0, 1, 60)
        M = np.column_stack([np.ones(60), b, b + rng.normal(0, 0.002, 60)])
    k = np.linalg.cond(M)
    print(f"  {label:<26}{k:>13.3e}{np.linalg.cond(M.T@M):>14.3e}"
          f"{np.log10(k**2) - np.log10(k):>13.1f}")
print("  kappa(X'X) = kappa(X)^2, so forming the normal equations DOUBLES the")
print("  exponent - you lose half your significant digits. Use QR or SVD.")

# %% [markdown]
# ## 6. Quadratic forms: why ANOVA sums of squares are $\chi^2$, eq. (16.11)

# %%
header("6. Quadratic forms in projection matrices (16.11)")

rng = np.random.default_rng(1603)
n = 30
Xd = np.column_stack([np.ones(n), np.repeat([0, 1], n // 2)])
H = Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T            # hat matrix
Mres = np.eye(n) - H                                # residual projector

print(f"  rank(H)  = {np.linalg.matrix_rank(H)} = p = {Xd.shape[1]}")
print(f"  rank(I-H)= {np.linalg.matrix_rank(Mres)} = n-p = {n - Xd.shape[1]}")
print(f"  H idempotent? {np.allclose(H @ H, H)}   "
      f"(I-H) idempotent? {np.allclose(Mres @ Mres, Mres)}")
print(f"  H(I-H) = 0 (orthogonal)? {np.allclose(H @ Mres, 0)}")

# Under the null (no group effect), y'(I-H)y / sigma^2 ~ chi2_{n-p}
SIGMA = 2.0
q_stats = []
for _ in range(20000):
    yv = rng.normal(5.0, SIGMA, n)                  # no group effect
    q_stats.append(yv @ Mres @ yv / SIGMA**2)
q_stats = np.array(q_stats)
dfree = n - Xd.shape[1]
print(f"\n  simulated mean of y'(I-H)y/sigma^2 = {q_stats.mean():.4f}   "
      f"(chi2_{dfree} mean = {dfree})")
print(f"  simulated variance                 = {q_stats.var(ddof=1):.4f}   "
      f"(chi2 variance = {2*dfree})")
print(f"  KS test vs chi2_{dfree}: p = "
      f"{st.kstest(q_stats, 'chi2', args=(dfree,)).pvalue:.3f}")
print("\n  This is Cochran's theorem in action: SS terms are quadratic forms in")
print("  orthogonal projection matrices, their df are the ranks, and their")
print("  independence is what makes the ANOVA F-ratio an F (Module 12).")

# %% [markdown]
# ## 7. Figure

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(np.arange(1, 21), D[:20], "o-")
axes[0].axvline(true_rank + 0.5, color="red", ls="--",
                label=f"true rank = {true_rank}")
axes[0].set_xlabel("component"); axes[0].set_ylabel("singular value")
axes[0].set_title("Eq. (16.6): the spectrum reveals rank", fontsize=9)
axes[0].legend(fontsize=7)

ks = np.arange(1, 21)
err = [np.linalg.norm(A - U[:, :k] @ np.diag(D[:k]) @ Vt[:k])**2 for k in ks]
axes[1].semilogy(ks, err, "o-", label="truncated SVD")
axes[1].semilogy(ks, [np.sum(D[k:]**2) for k in ks], "x--",
                 label="$\\sum_{j>k} d_j^2$")
axes[1].set_xlabel("rank $k$"); axes[1].set_ylabel("$\\|A-A_k\\|_F^2$")
axes[1].set_title("Eq. (16.8): Eckart-Young", fontsize=9)
axes[1].legend(fontsize=7)

axes[2].hist(q_stats, bins=70, density=True, alpha=0.7)
gx = np.linspace(0, q_stats.max(), 300)
axes[2].plot(gx, st.chi2.pdf(gx, dfree), "r-", lw=2, label=f"$\\chi^2_{{{dfree}}}$")
axes[2].set_title("Eq. (16.11): quadratic form is $\\chi^2$", fontsize=9)
axes[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "matrix_algebra.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure written to {OUT}/matrix_algebra.png")

# %% [markdown]
# ## Decision rules (from `stats.md` Topic 16)
#
# 1. Decide which dimension you are decomposing: samples or features - before
#    interpreting any output.
# 2. Use SVD/QR, never explicit inverses.
# 3. Check $\operatorname{rank}(\mathbf X)$ and $\kappa(\mathbf X)$ before
#    fitting $G$ feature-wise models with the same design.
# 4. Exploit the $\mathbf S\leftrightarrow\mathbf K$ duality when $p\gg n$.
#
# **Next:** `17_pca.py`
