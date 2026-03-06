import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time
import matplotlib.pyplot as plt
import platform
import multiprocessing
from scipy.optimize import curve_fit
from scipy.stats import linregress
from orbitalspectrum import PolarDecomposition

# ============================================================
# Utility Functions
# ============================================================

def gaussian_density(X, Y, sigma=1.0):
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def generate_grid(Nx, Ny):
    x = np.linspace(-10, 10, Nx)
    y = np.linspace(-10, 10, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, gaussian_density(X, Y)


# ============================================================
# Timing Infrastructure
# ============================================================

def time_single_run(Nx, Ny, Nr, Ntheta):
    x, y, f_xy = generate_grid(Nx, Ny)

    t0 = time.perf_counter()
    PolarDecomposition(
        f_xy,
        x,
        y,
        Nr=Nr,
        Ntheta=Ntheta,
        recon_err_tol=1.0,
        recon_power_tol=1e-12,
        m_abs_max=None,
    )
    t1 = time.perf_counter()

    return t1 - t0


def averaged_runtime(Nx, Ny, Nr, Ntheta, repeats=5, warmups=2):
    # Warm-up runs (not timed)
    for _ in range(warmups):
        time_single_run(Nx, Ny, Nr, Ntheta)

    times = []
    for _ in range(repeats):
        t = time_single_run(Nx, Ny, Nr, Ntheta)
        times.append(t)

    return np.mean(times), np.std(times)


# ============================================================
# True Complexity Models
# ============================================================

def model_linear(N, a, b):
    return a + b * N


def model_nlogn(N, a, b):
    return a + b * N * np.log2(N)


def model_power(N, a, alpha):
    return a * N**alpha


# ============================================================
# Scaling Study
# ============================================================

def scaling_study(param_name, param_values, fixed_params, repeats=5):
    means = []
    stds = []

    for val in param_values:
        kwargs = fixed_params.copy()
        kwargs[param_name] = val

        mean, std = averaged_runtime(**kwargs, repeats=repeats)
        means.append(mean)
        stds.append(std)

        print(f"{param_name}={val:6d}  mean={mean:.5f}s  std={std:.5f}s")

    return np.array(param_values), np.array(means), np.array(stds)


# ============================================================
# Regression and Model Comparison
# ============================================================

def fit_and_compare(x, y, label):

    print(f"\n--- Regression analysis for {label} ---")

    # Linear model
    popt_lin, _ = curve_fit(model_linear, x, y)
    y_lin = model_linear(x, *popt_lin)

    # N log N model
    popt_nlogn, _ = curve_fit(model_nlogn, x, y)
    y_nlogn = model_nlogn(x, *popt_nlogn)

    # Power-law model
    slope, intercept, r_value, _, _ = linregress(np.log10(x), np.log10(y))
    y_power = 10**intercept * x**slope

    # Compute residual norms
    err_lin = np.linalg.norm(y - y_lin)
    err_nlogn = np.linalg.norm(y - y_nlogn)
    err_power = np.linalg.norm(y - y_power)

    print(f"Linear model residual: {err_lin:.4e}")
    print(f"N log N model residual: {err_nlogn:.4e}")
    print(f"Power-law exponent: {slope:.4f}")
    print(f"Power-law residual: {err_power:.4e}")

    return y_lin, y_nlogn, y_power, slope


# ============================================================
# Plotting
# ============================================================

def plot_scaling(x, y, std, fits, xlabel):

    y_lin, y_nlogn, y_power, slope = fits

    plt.figure(figsize=(7, 6))
    plt.errorbar(x, y, yerr=std, fmt='o', label="Measured")
    plt.loglog(x, y_lin, '--', label="Linear fit")
    plt.loglog(x, y_nlogn, '--', label="N log N fit")
    plt.loglog(x, y_power, '--', label=f"Power fit (α={slope:.2f})")

    plt.xlabel(xlabel)
    plt.ylabel("Runtime (s)")
    plt.title(f"Scaling vs {xlabel}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main CPC-Level Benchmark
# ============================================================

if __name__ == "__main__":

    print("===================================================")
    print("CPC-Level PolarDecomposition Benchmark")
    print("===================================================")
    print(f"Platform: {platform.platform()}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print("Threads forced to 1 for reproducibility.")
    print("===================================================")

    Nx = Ny = 256

    # ---------------------------
    # Ntheta Scaling
    # ---------------------------

    print("\nScaling vs Ntheta")
    Ntheta_vals = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])

    x_theta, y_theta, std_theta = scaling_study(
        param_name="Ntheta",
        param_values=Ntheta_vals,
        fixed_params=dict(Nx=Nx, Ny=Ny, Nr=512, Ntheta=512),
        repeats=7,
    )

    fits_theta = fit_and_compare(x_theta, y_theta, "Ntheta")
    plot_scaling(x_theta, y_theta, std_theta, fits_theta, "Ntheta")

    # ---------------------------
    # Nr Scaling
    # ---------------------------

    print("\nScaling vs Nr")
    Nr_vals = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])

    x_r, y_r, std_r = scaling_study(
        param_name="Nr",
        param_values=Nr_vals,
        fixed_params=dict(Nx=Nx, Ny=Ny, Nr=512, Ntheta=1024),
        repeats=7,
    )

    fits_r = fit_and_compare(x_r, y_r, "Nr")
    plot_scaling(x_r, y_r, std_r, fits_r, "Nr")

    print("\nBenchmark complete.")