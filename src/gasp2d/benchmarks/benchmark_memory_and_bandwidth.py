import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time
import psutil
import threading
import multiprocessing as mp
import matplotlib.pyplot as plt
import platform
from scipy.optimize import curve_fit
from orbitalspectrum import PolarDecomposition


# ============================================================
# Memory Sampling Thread
# ============================================================

def monitor_memory(pid, stop_event, peak_container, interval=0.01):
    proc = psutil.Process(pid)
    peak = proc.memory_info().rss
    while not stop_event.is_set():
        rss = proc.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(interval)
    peak_container.append(peak)


# ============================================================
# Worker (isolated process)
# ============================================================

def worker(Nr, Ntheta, return_dict):

    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f_xy = np.exp(-(X**2 + Y**2))

    stop_event = threading.Event()
    peak_container = []

    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(os.getpid(), stop_event, peak_container)
    )
    monitor_thread.start()

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

    stop_event.set()
    monitor_thread.join()

    return_dict["peak_rss"] = peak_container[0]
    return_dict["runtime"] = t1 - t0


# ============================================================
# Run Isolated
# ============================================================

def run_isolated(Nr, Ntheta):

    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(target=worker, args=(Nr, Ntheta, return_dict))
    p.start()
    p.join()

    return return_dict["runtime"], return_dict["peak_rss"]


# ============================================================
# Models
# ============================================================

def model_linear(N, A, B):
    return A + B*N

def model_power(N, C, alpha):
    return C*N**alpha


# ============================================================
# Scaling Study
# ============================================================

def memory_scaling_study(Nr_vals, Ntheta):

    N_list = []
    runtimes = []
    peaks = []

    for Nr in Nr_vals:

        runtime, peak = run_isolated(Nr, Ntheta)

        N = Nr * Ntheta
        N_list.append(N)
        runtimes.append(runtime)
        peaks.append(peak)

        print(f"Nr={Nr:6d}  "
              f"N={N:10d}  "
              f"PeakRSS={peak/1e6:.2f} MB  "
              f"time={runtime:.4f}s")

    return np.array(N_list), np.array(runtimes), np.array(peaks)


# ============================================================
# Model Fitting
# ============================================================

def fit_models(N, M):

    popt_lin, _ = curve_fit(model_linear, N, M)
    popt_pow, _ = curve_fit(model_power, N, M, maxfev=20000)

    M_lin = model_linear(N, *popt_lin)
    M_pow = model_power(N, *popt_pow)

    err_lin = np.linalg.norm(M - M_lin)
    err_pow = np.linalg.norm(M - M_pow)

    print("\n--- Memory Model Comparison ---")
    print(f"Linear residual: {err_lin:.4e}")
    print(f"Power residual:  {err_pow:.4e}")

    A, B = popt_lin
    _, alpha = popt_pow

    print("\n--- Linear Fit Parameters ---")
    print(f"Fixed overhead A = {A/1e6:.3f} MB")
    print(f"Asymptotic slope B = {B:.3f} bytes per grid point")

    print("\n--- Power Fit Parameter ---")
    print(f"Exponent α = {alpha:.4f}")

    return M_lin, M_pow, B


# ============================================================
# Bandwidth Table
# ============================================================

def print_bandwidth_table(N, runtimes, peaks):

    bandwidth_MB = peaks / runtimes / 1e6
    bandwidth_GB = peaks / runtimes / 1e9

    print("\n===================================================")
    print("Bandwidth Table")
    print("===================================================")
    print(f"{'N':>12} {'Peak MB':>12} {'Time (s)':>10} "
          f"{'MB/s':>12} {'GB/s':>10}")
    print("---------------------------------------------------")

    for n, m, t, bMB, bGB in zip(N, peaks, runtimes,
                                  bandwidth_MB, bandwidth_GB):

        print(f"{n:12d} "
              f"{m/1e6:12.2f} "
              f"{t:10.4f} "
              f"{bMB:12.2f} "
              f"{bGB:10.3f}")

    print("===================================================")

    return bandwidth_MB, bandwidth_GB


# ============================================================
# Plotting
# ============================================================

def plot_memory_models(N, M, M_lin, M_pow):

    plt.figure(figsize=(7,6))
    plt.loglog(N, M/1e6, 'o', label="Measured Peak RSS")
    plt.loglog(N, M_lin/1e6, '--', label="Linear fit")
    plt.loglog(N, M_pow/1e6, '--', label="Power fit")
    plt.xlabel("Total grid points N")
    plt.ylabel("Peak RSS (MB)")
    plt.title("Memory Scaling")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bandwidth(N, peaks, runtimes):

    bandwidth = peaks / runtimes

    plt.figure(figsize=(7,6))
    plt.loglog(N, bandwidth/1e9, 'o-')
    plt.xlabel("Total grid points N")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title("Empirical Memory Bandwidth")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("===================================================")
    print("CPC-Level Isolated Memory & Bandwidth Benchmark")
    print("===================================================")
    print(f"Platform: {platform.platform()}")
    print("Each size executed in fresh process.")
    print("===================================================")

    Ntheta_fixed = 1024
    Nr_vals = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    N, runtimes, peaks = memory_scaling_study(Nr_vals, Ntheta_fixed)

    M_lin, M_pow, slope = fit_models(N, peaks)

    bandwidth_MB, bandwidth_GB = print_bandwidth_table(N, runtimes, peaks)

    plot_memory_models(N, peaks, M_lin, M_pow)
    plot_bandwidth(N, peaks, runtimes)

    print("\nBenchmark complete.")