import numpy as np
from time import time
from matplotlib import pyplot as plt

import orbitalspectrum as orb

analyticexample = True

def psi_nx_ny(nx, ny, ax=1.0, ay=1.0):
    from scipy.special import hermite, factorial
    """
    Returns a function ψ(x,y) for given quantum numbers (nx, ny)
    and oscillator lengths (ax, ay).
    """
    Hx = hermite(nx)
    Hy = hermite(ny)
    norm = 1.0 / np.sqrt(np.pi * ax * ay * (2**(nx+ny) * factorial(nx) * factorial(ny)))
    def psi(x, y):
        return norm * Hx(x/ax) * Hy(y/ay) * np.exp(-0.5*((x/ax)**2 + (y/ay)**2))
    return psi

#---------------------------------------------------------------------------------------
#          Example with Analytic Solution: Interaction Energy between Gaussians
#---------------------------------------------------------------------------------------

Nx = Ny = 101

x = np.linspace(-10, 10, Nx) 
y = np.linspace(-10, 10, Ny)

X, Y = np.meshgrid(x, y, indexing = 'ij')

QHO_Ground_State = psi_nx_ny(0, 0, ax = 1.0, ay = 1.0)

example = {'name':"2D Gaussian (Isotropic QHO Ground State)",
           'func': np.abs(QHO_Ground_State(X, Y))**2}

ti_cdh = time()
f = orb.PolarDecomposition(example['func'], x, y, Nr = 256, Ntheta = 512, recon_err_tol = 1.0, recon_power_tol = 1e-15, m_abs_max = None,)
tf_cdh = time()

print(f'Decomposition time: {tf_cdh - ti_cdh}')
f.plot_harmonics_with_hist(title = f"{example['name']}")
f.plot_original_vs_reconstructions(title = f"{example['name']}")
plt.show()


