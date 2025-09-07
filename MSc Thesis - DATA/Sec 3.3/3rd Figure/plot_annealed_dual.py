#!/usr/bin/env python3
# plot_annealed_aggregate.py  – final aesthetics
#
# • reads *_results_annealed_Q_gamma.csv from TO_PLOT/
# • builds running-min curves, averages → CSV
# • plots mean curves (log-y), x-ticks every 200 iterations
#   red = perfect CIM, blue = CPM annealing

from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----- style (moderate large fonts) ---------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except (OSError, ValueError):
    warnings.warn("seaborn-v0_8 style not found; using seaborn default")
    plt.style.use("seaborn")

plt.rcParams.update({
    "font.size":       18,
    "axes.titlesize":  22,
    "axes.labelsize":  20,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 2.5,
})

PLOT_KW  = dict(dpi=200, bbox_inches="tight")
FOLDER   = Path("TO_PLOT")
CSV_MASK = "*_results_annealed_Q_gamma.csv"
OUT_CSV  = FOLDER / "aggregate_global.csv"
OUT_PNG  = FOLDER / "aggregate_global.png"

csv_files = list(FOLDER.glob(CSV_MASK))
if not csv_files:
    raise FileNotFoundError(f"No files matching {CSV_MASK} in {FOLDER}")

required = {"outer_iter", "best_norm", "efficiency"}
rows = []

for csv in csv_files:
    df = pd.read_csv(csv).sort_values("outer_iter")
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{csv.name} missing: {', '.join(missing)}")

    best = df["best_norm"].cummin()
    opt  = (df["best_norm"] / df["efficiency"]).cummin()
    rows.append(pd.DataFrame({"outer_iter": df["outer_iter"],
                              "best": best, "opt": opt}))

all_runs = pd.concat(rows, ignore_index=True)

agg = (all_runs.groupby("outer_iter")
       .agg(best_avg=("best", "mean"),
            opt_avg =("opt",  "mean"))
       .reset_index())

agg.to_csv(OUT_CSV, index=False)
print("✓ wrote", OUT_CSV.name)

# ---------------- plot ----------------------------------------------
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(agg["outer_iter"], agg["best_avg"],
        color="tab:blue", label="CPM annealing (average)")
ax.plot(agg["outer_iter"], agg["opt_avg"],
        color="tab:red",  label="Perfect CIM (average)")

ax.set_yscale("log")
ax.set_xlabel("Iteration Number")
ax.set_ylabel("Shortest Norm found")
ax.set_title("Shortest norm found across iterations,\n"
             "for a perfect CIM vs. CPM with imperfect annealing procedure",
             pad=30)                     # extra spacing
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.legend()

fig.savefig(OUT_PNG, **PLOT_KW)
plt.close(fig)
print("✓ wrote", OUT_PNG.name)
print("Finished – plots are in", FOLDER)
