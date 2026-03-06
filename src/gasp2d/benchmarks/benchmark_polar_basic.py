import numpy as np
import time
import matplotlib.pyplot as plt
import platform
import multiprocessing

from orbitalspectrum import PolarDecomposition


# ============================================================
#   Test Function: 2D Gaussian (QHO ground state density)
# ============================================================

def gaussian_density(X, Y, sigma=1.0):
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


# ============================================================
#   Benchmark Function
# ============================================================

def run_single_benchmark(Nx, Ny, Nr, Ntheta):
    x = np.linspace(-10, 10, Nx)
    y = np.linspace(-10, 10, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    f_xy = gaussian_density(X, Y)

    t0 = time.perf_counter()
    pd = PolarDecomposition(
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


# ============================================================
#   Scaling Study
# ============================================================

def scaling_vs_Ntheta():
    Nx = Ny = 256
    Nr = 256

    Ntheta_values = [128, 256, 512, 1024, 2048]
    times = []

    for Ntheta in Ntheta_values:
        t = run_single_benchmark(Nx, Ny, Nr, Ntheta)
        times.append(t)
        print(f"Ntheta={Ntheta:4d}  Time={t:.4f} s")

    return Ntheta_values, times


def scaling_vs_Nr():
    Nx = Ny = 256
    Ntheta = 512

    Nr_values = [64, 128, 256, 512, 768]
    times = []

    for Nr in Nr_values:
        t = run_single_benchmark(Nx, Ny, Nr, Ntheta)
        times.append(t)
        print(f"Nr={Nr:4d}  Time={t:.4f} s")

    return Nr_values, times


# ============================================================
#   Main Execution
# ============================================================

if __name__ == "__main__":

    print("===================================================")
    print("PolarDecomposition Benchmark")
    print("===================================================")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print("===================================================")

    # --------------------------------------------------------
    # Scaling vs Ntheta
    # --------------------------------------------------------

    print("\nScaling vs Ntheta")
    Ntheta_vals, times_theta = scaling_vs_Ntheta()

    # --------------------------------------------------------
    # Scaling vs Nr
    # --------------------------------------------------------

    print("\nScaling vs Nr")
    Nr_vals, times_r = scaling_vs_Nr()

    # --------------------------------------------------------
    # Plot Results
    # --------------------------------------------------------

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    # Scaling vs Ntheta
    axs[0].plot(Ntheta_vals, times_theta, marker='o')
    axs[0].set_xlabel("Ntheta")
    axs[0].set_ylabel("Runtime (s)")
    axs[0].set_title("Scaling vs Ntheta")
    axs[0].grid(True)

    # Scaling vs Nr
    axs[1].plot(Nr_vals, times_r, marker='o')
    axs[1].set_xlabel("Nr")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].set_title("Scaling vs Nr")
    axs[1].grid(True)

    plt.show()
