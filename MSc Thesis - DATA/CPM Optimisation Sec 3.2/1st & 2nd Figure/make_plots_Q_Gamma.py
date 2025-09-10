#!/usr/bin/env python3
# make_plots_Q_Gamma.py –  ΣCPM heat-maps, γ* stats, ηCIM curves
# ------------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
sns.set(style="whitegrid", context="talk")

DATA_DIR = Path("DATA")          # B*_full.csv files
PLOT_DIR = Path("ASSESSOR_PLOTS"); PLOT_DIR.mkdir(exist_ok=True)

bud_order = ["B1", "B2", "B3"]
bud_color = dict(B1="steelblue", B2="mediumvioletred", B3="red")

# ────────── load & merge ------------------------------------------------
frames = []
for b in bud_order:
    f = DATA_DIR / f"{b}_full.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    d = pd.read_csv(f); d["budget"] = b; frames.append(d)
df = pd.concat(frames, ignore_index=True)

# corrected ΣCPM  --------------------------------------------------------
df["Sigma_CPM"] = (df["min_W"] - df["best_len"]).clip(lower=0.0)
df["global_best_norm"] = np.minimum(df["best_len"], df["min_W"])

def save(fig, name):
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig); print(" •", name)

def med_iqr(s):
    return pd.Series(dict(
        med=s.median(),
        q1 =s.quantile(0.25),
        q3 =s.quantile(0.75)
    ))

# ========================================================================
# 1.  ΣCPM heat-maps  (one per budget)
# ========================================================================
for bud in bud_order:
    mat = (df[df.budget == bud]
             .groupby(["Q", "gamma"])["Sigma_CPM"]
             .mean().unstack("gamma").sort_index())
    plt.figure(figsize=(10, 3))
    sns.heatmap(mat, cmap="viridis",
                cbar_kws=dict(label=r"$\Sigma_{CPM}$"))
    plt.xlabel(r"$\gamma$"); plt.ylabel("Q")
    plt.title(rf"$\Sigma_{{CPM}}$ heat-map – {bud}")
    save(plt.gcf(), f"heat_Sigma_{bud}.png")

# ========================================================================
# 2 & 3.  γ* mean heat-map  and  γ* IQR-width heat-map
# ========================================================================
def gamma_star(slc):
    return slc.loc[slc["eta_CIM"].idxmax(), "gamma"]

gs = (df.groupby(["slice_id", "budget", "Q"])
        .apply(gamma_star).reset_index(name="gamma_star"))

gs_mean = (gs.groupby(["budget", "Q"])["gamma_star"]
             .mean().unstack("budget"))
plt.figure(figsize=(6, 3))
sns.heatmap(gs_mean, annot=True, fmt=".3f",
            cmap="magma_r", cbar_kws=dict(label=r"$\gamma^*$"))
plt.title(r"Mean $\gamma^*$  (max $\eta_{CIM}$)")
plt.ylabel("Q"); plt.xlabel("budget")
save(plt.gcf(), "heat_gamma_star_mean.png")

gs_iqr = (gs.groupby(["budget", "Q"])["gamma_star"]
            .agg(lambda x: x.quantile(0.75) - x.quantile(0.25))
            .unstack("budget"))
plt.figure(figsize=(6, 3))
sns.heatmap(gs_iqr, annot=True, fmt=".3f",
            cmap="coolwarm_r", cbar_kws=dict(label=r"IQR$(\gamma^*)$"))
plt.title(r"IQR width of $\gamma^*$ across slices")
plt.ylabel("Q"); plt.xlabel("budget")
save(plt.gcf(), "heat_gamma_star_IQR.png")

# ========================================================================
# 4.  ηCIM vs γ  (median curve + IQR band)  –  one figure per Q
# ========================================================================
for Q in sorted(df.Q.unique()):
    sub = df[df.Q == Q]
    agg = (sub.groupby(["budget", "gamma"])["eta_CIM"]
             .agg(med="median",
                  q1 =lambda x: x.quantile(0.25),
                  q3 =lambda x: x.quantile(0.75))
             .reset_index())
    plt.figure(figsize=(7, 4))
    for bud in bud_order:
        a = agg[agg.budget == bud]
        plt.plot(a.gamma, a.med, color=bud_color[bud],
                 label=bud, linewidth=1.5)
        plt.fill_between(a.gamma, a.q1, a.q3,
                         color=bud_color[bud], alpha=0.25)
    plt.xlabel(r"$\gamma$"); plt.ylabel(r"$\eta_{CIM}$")
    plt.title(f"$\\eta_{{CIM}}$ vs $\\gamma$  (Q={Q})")
    plt.legend()
    save(plt.gcf(), f"Q{Q}_etaCIM_linear.png")

print("\nFinished – figures saved in", PLOT_DIR.resolve())
