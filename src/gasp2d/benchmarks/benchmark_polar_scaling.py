import numpy as np
import time
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import platform
import multiprocessing
from scipy.stats import linregress

from orbitalspectrum import PolarDecomposition


def gaussian_density(X, Y, sigma=1.0):
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def profile_run(Nx, Ny, Nr, Ntheta, print_stats=True):

    x = np.linspace(-10, 10, Nx)
    y = np.linspace(-10, 10, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f_xy = gaussian_density(X, Y)

    profiler = cProfile.Profile()
    profiler.enable()

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

    profiler.disable()

    total_time = t1 - t0

    if print_stats:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(15)
        print("\nTop cumulative time functions:")
        print(s.getvalue())

    return total_time


def scaling_study(param_name, param_values, fixed_params):

    times = []

    for val in param_values:
        kwargs = fixed_params.copy()
        kwargs[param_name] = val

        t = profile_run(**kwargs, print_stats=False)
        times.append(t)
        print(f"{param_name}={val:5d}   time={t:.4f} s")

    return np.array(param_values), np.array(times)


def fit_power_law(x, y):

    logx = np.log10(x)
    logy = np.log10(y)

    slope, intercept, r_value, _, _ = linregress(logx, logy)

    return slope, intercept, r_value


def plot_scaling(x, y, xlabel):

    slope, intercept, r_value = fit_power_law(x, y)

    fit_line = 10**intercept * x**slope

    plt.figure(figsize=(6, 5))
    plt.loglog(x, y, 'o', label="Measured")
    plt.loglog(x, fit_line, '--', label=f"Fit: slope={slope:.2f}")

    plt.xlabel(xlabel)
    plt.ylabel("Runtime (s)")
    plt.title(f"Scaling vs {xlabel}")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    print(f"\nScaling exponent for {xlabel}: {slope:.3f}")
    print(f"R^2 (log-log fit): {r_value**2:.4f}")

    plt.show()

if __name__ == "__main__":

    print("===================================================")
    print("Advanced PolarDecomposition Benchmark")
    print("===================================================")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print("===================================================")

    Nx = Ny = 256

    # Ntheta Scaling 

    print("\nRunning scaling vs Ntheta...")
    Ntheta_vals = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    x_theta, y_theta = scaling_study(
        param_name="Ntheta",
        param_values=Ntheta_vals,
        fixed_params=dict(Nx=Nx, Ny=Ny, Nr=256, Ntheta=512),
    )

    plot_scaling(x_theta, y_theta, "Ntheta")

    # Nr Scaling


    print("\nRunning scaling vs Nr...")
    Nr_vals = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    x_r, y_r = scaling_study(
        param_name="Nr",
        param_values=Nr_vals,
        fixed_params=dict(Nx=Nx, Ny=Ny, Nr=256, Ntheta=512),
    )

    plot_scaling(x_r, y_r, "Nr")

    # Single full profiling run

    print("\nRunning full cProfile analysis at large resolution...")
    profile_run(
        Nx=Nx,
        Ny=Ny,
        Nr=512,
        Ntheta=1024,
        print_stats=True,
    )
