#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

RESULTS_FOLDER = "CIM_performance_results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def generate_random_candidates(X, n, low=1, high=10, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=(X, n))

def run_cim_empirical(search_basis, n_runs=10**6, seed=None):
    # Each run: random ±1 combination of search_basis
    np.random.seed(seed)
    K, n = search_basis.shape
    norm_counts = {}  # norm (rounded) -> count
    norm_raw = []
    for i in range(n_runs):
        spins = np.random.choice([-1,1], size=K)
        vec = spins @ search_basis
        # Exclude zero vector results
        if np.all(vec == 0):
            continue
        norm = np.linalg.norm(vec)
        norm_rounded = np.round(norm, 4)  # To avoid binning issues
        norm_raw.append(norm)
        if norm_rounded not in norm_counts:
            norm_counts[norm_rounded] = 0
        norm_counts[norm_rounded] += 1
        if (i+1) % (n_runs // 10) == 0:
            print(f"Progress: {i+1}/{n_runs} CIM runs")
    return norm_counts, np.array(norm_raw)

def plot_histogram(norm_counts, norm_raw, n, X, n_runs, results_dir):
    plt.figure(figsize=(8,5))
    # Use raw norms for a smooth histogram
    plt.hist(norm_raw, bins=100, alpha=0.75, color='royalblue')
    plt.xlabel("Norm of CIM output vector")
    plt.ylabel("Number of occurrences")
    plt.title(f"CIM Output Histogram (n={n}, X={X}, runs={n_runs})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"histogram_n{n}_X{X}_runs{n_runs}.pdf"))
    plt.close()

    # Also save the raw histogram as .txt for reproducibility
    hist_txt = os.path.join(results_dir, f"histogram_n{n}_X{X}_runs{n_runs}.txt")
    with open(hist_txt, "w") as f:
        f.write("Norm (rounded),Count\n")
        for k in sorted(norm_counts):
            f.write(f"{k},{norm_counts[k]}\n")
    print(f"Histogram and data saved to {results_dir}")

def main():
    # Parameters to vary
    n_list = [5, 10, 20]
    X_list = [5, 10, 20]
    n_runs = 10**6
    seed = 42

    now = datetime.datetime.now()
    results_dir = os.path.join(
        RESULTS_FOLDER,
        now.strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(results_dir, exist_ok=False)

    for n in n_list:
        for X in X_list:
            print(f"\n--- Running: n={n}, X={X} ---")
            search_basis = generate_random_candidates(X, n, low=1, high=10, seed=seed+n*100+X)
            norm_counts, norm_raw = run_cim_empirical(search_basis, n_runs=n_runs, seed=seed+n*100+X)
            plot_histogram(norm_counts, norm_raw, n, X, n_runs, results_dir)

if __name__ == "__main__":
    main()
