import numpy as np
from numpy.fft import fft
from scipy.ndimage import map_coordinates
from scipy.integrate import cumulative_trapezoid
from matplotlib import pyplot as plt
from collections.abc import Mapping

from enum import Enum
class _functype(Enum):
    _callable = 0
    _discrete = 1

class PolarDecomposition(Mapping):

    """
    Polar harmonic decomposition of a 2D scalar field f(x, y).

    This class converts a function defined on a Cartesian grid into polar
    coordinates and performs an angular Fourier decomposition

        f(r, θ) = ∑_m ρ_m(r) e^{i m θ}

    where ρ_m(r) are radial harmonic coefficients.

    The decomposition is computed numerically by:
        1. Interpolating the Cartesian field onto a polar grid.
        2. Performing a discrete Fourier transform along the angular direction.
        3. Optionally selecting the minimal subset of harmonics needed to
           satisfy a reconstruction error tolerance.

    Parameters
    ----------
    f_xy : callable or ndarray of shape (Nx, Ny)
        Either:
        - A callable function f(x, y), or
        - A 2D array sampled on the Cartesian grid defined by (x, y).

    x : ndarray of shape (Nx,)
        Cartesian x-coordinates.

    y : ndarray of shape (Ny,)
        Cartesian y-coordinates.

    Nr : int or None
        Number of radial grid points. If None, defaults to min(Nx, Ny)//2.

    Ntheta : int, default=512
        Number of angular sampling points in [0, 2π).

    rmax : float or None
        Maximum radial extent. If None, chosen from grid extent.

    m_abs_max : int or None
        Maximum absolute angular momentum |m| retained in the decomposition.
        If None, all Fourier modes are kept.

    recon_err_tol : float, default=1.0
        Target RMS reconstruction error (%) for automatic harmonic selection.

    recon_power_tol : float, default=0.01
        Radial power tolerance used when computing cutoff radii.

    center : bool, default=True
        If True, shift coordinates so that the centroid of |f|² is at the origin.

    normalize : bool, default=False
        If True, normalize such that ∫ |f(x,y)|² dx dy = 1.

    interp_method : str, default='cubic'
        Interpolation method passed to scipy RegularGridInterpolator.

    Attributes
    ----------
    r : ndarray of shape (Nr,)
        Radial grid.

    theta : ndarray of shape (Ntheta,)
        Angular grid.

    rho : dict[int, ndarray]
        Dictionary mapping harmonic index m to radial coefficient ρ_m(r).

    m_sorted : list[int]
        Harmonics retained after automatic selection.

    powers : ndarray
        Integrated radial power of each retained harmonic.

    power_fracs : dict[int, float]
        Fractional power contribution of each retained harmonic.

    r_cutoff : dict[int, float]
        Radial cutoff enclosing specified fraction of harmonic power.

    f_polar : ndarray of shape (Nr, Ntheta)
        Field interpolated onto polar grid.

    f_recon : ndarray
        Reconstructed field from selected harmonics.

    recon_error : float
        Final RMS reconstruction error (%).
    """

    def __init__(self, 
                 f_xy, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 Nr: int | None,
                 Ntheta: int | None = 512,
                 rmax: float | None = None,
                 m_abs_max: int | None = None,
                 recon_err_tol: float = 1.0,
                 recon_power_tol: float = 0.01,
                 center: bool = True,
                 normalize: bool = False,
                 interp_method: str = 'cubic'):
        
        self._classification = self._classify_function_type(f_xy)
        self._f_xy = f_xy
        self.x, self._Nx = x, x.shape[0]
        self.y, self._Ny = y, y.shape[0]
        self._Nr = Nr
        self._Ntheta = Ntheta
        self._rmax = rmax
        self._m_abs_max = m_abs_max
        self._recon_err_tol = recon_err_tol
        self._recon_power_tol = recon_power_tol
        self._center = center
        self._normalize = normalize
        self._interp_method = interp_method

        self._prepfunctions()
        self._decompose()

    def __getitem__(self, key):
        """
        Access radial harmonic coefficients.

        Supports:
            p[m]            → ρ_m(r)
            p[m, i]         → ρ_m(r)[i]
            p[m, i:j]       → sliced radial values

        Parameters
        ----------
        key : int or tuple
            Harmonic index m, optionally followed by radial indexing.

        Returns
        -------
        ndarray or scalar
            Radial harmonic coefficient(s).

        Raises
        ------
        KeyError
            If harmonic m is not present.
        """

        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Empty index.")
            m = int(key[0])
            radial_index = key[1:]
        else:
            m = int(key)
            radial_index = ()

        if m not in self.rho:
            raise KeyError(f"Harmonic m={m} is not contained in kept m's: {list(self.rho.keys())}")

        rho_m = self.rho[m]

        return rho_m[radial_index] if radial_index else rho_m
    
    def __iter__(self):
        """
        Iterate over available harmonic indices m.
        """
        return iter(self.rho)


    def __len__(self):
        """
        Return number of retained harmonics.
        """
        return len(self.rho)


    def __contains__(self, m):
        """
        Return True if harmonic m is available.
        """
        return int(m) in self.rho
    
    def keys(self):
        """
        Return harmonic indices.
        """
        return self.rho.keys()


    def values(self):
        """
        Return radial harmonic coefficient arrays.
        """
        return self.rho.values()


    def items(self):
        """
        Iterate over (m, ρ_m(r)) pairs.
        """
        return self.rho.items()
        

    def _prepfunctions(self):
        """
        Prepare Cartesian and polar grids and interpolate field.

        This method:
            1. Builds Cartesian meshgrid.
            2. Evaluates or assigns f(x, y).
            3. Optionally normalizes ∫ |f|² dx dy.
            4. Computes centroid of |f|².
            5. Defines polar grid (r, θ).
            6. Interpolates field onto polar coordinates.

        Notes
        -----
        Interpolation is performed using scipy RegularGridInterpolator.
        Points outside the Cartesian grid are filled with zero.

        After execution, the following attributes are defined:
            - self.r
            - self.theta
            - self.f_polar
        """

        self._X, self._Y = np.meshgrid(self.x, self.y, indexing='ij')         # x and y meshgrids; shape = (len(x), len(y))
        
        self._f = self._f_xy(self._X, self._Y) if self._classification is _functype._callable else self._f_xy

        # normalization of f_xy
        if self._normalize:
            norm = np.sqrt(np.trapezoid(np.trapezoid(np.abs(self._f)**2, self.y), self.x))
            if norm > 0: 
                self._f = self._f / norm
            else:
                raise ValueError('f_xy is zero function.')
            
        self._f_sqr = np.abs(self._f)**2 # modulus squared of f_xy; density function if f_xy is a wave function

        # center the cartesian coordinates around the centroid of f_xy
        self._x_centroid = np.sum(self._X * self._f_sqr)/np.sum(self._f_sqr)
        self._y_centroid = np.sum(self._Y * self._f_sqr)/np.sum(self._f_sqr)

        if self._center:
            self.x_c = self.x - self._x_centroid
            self.y_c = self.y - self._y_centroid
        else:
            self.x_c = self.x.copy()
            self.y_c = self.y.copy()

        # define the polar grid coordinates
        if self._Nr is None: 
            self._Nr = min(self._Nx, self._Ny)//2
        if self._rmax is None: 
            self._rmax = np.sqrt(self.x_c.max()**2 + self.y_c.max()**2)

        self.r      = np.linspace(0, self._rmax, self._Nr)                    # radial distance space
        self.theta  = np.linspace(0, 2*np.pi, self._Ntheta, endpoint = False) # angular space
        self.R, self.TH  = np.meshgrid(self.r, self.theta, indexing='ij')     # radial and angular grid space
        self.Xp, self.Yp = self.R*np.cos(self.TH), self.R*np.sin(self.TH)     # cartesian grids of sampled polar coordinates

        # --------------------------------------------------------
        # FAST interpolation using map_coordinates
        # --------------------------------------------------------

        dx = self.x_c[1] - self.x_c[0]
        dy = self.y_c[1] - self.y_c[0]

        xmin = self.x_c[0]
        ymin = self.y_c[0]

        # Convert physical polar coordinates → fractional array indices
        ix = (self.Xp - xmin) / dx
        iy = (self.Yp - ymin) / dy

        coords = np.vstack([ix.ravel(), iy.ravel()])

        # order=1 → linear
        # order=3 → cubic (much faster than RGI cubic)
        interp_order = 1 if self._interp_method == "linear" else 3

        self.f_polar = map_coordinates(
            self._f,
            coords,
            order=interp_order,
            mode="constant",
            cval=0.0
        ).reshape(self._Nr, self._Ntheta)
            
        # if self._classification is _functype._discrete:
        #    
        #    # check if dimensions of x and y position arrays match dimensions of f_xy
        #    if self._Nx != self._func_Nx:
        #        raise IndexError(f'The number of x points ({self._Nx}) != from the dim 0 of f_xy')
        #    if self._Ny != self._func_Ny:
        #        raise IndexError(f'The number of x points ({self._Ny}) != from the dim 1 of f_xy')        
        #    
        #    self.f_polar = self._f_interp(np.stack((self.Xp.ravel(), self.Yp.ravel()), -1)).reshape(self._Nr, self._Ntheta)

    def _classify_function_type(self, input_func) -> str:
        """
        Classify input field as callable or discrete array.

        Parameters
        ----------
        input_func : callable or array-like

        Returns
        -------
        _functype
            Enum indicating whether input is:
            - _callable : function f(x, y)
            - _discrete : 2D array sampled on Cartesian grid

        Raises
        ------
        TypeError
            If input is empty, not 2D, or invalid.

        Notes
        -----
        Discrete input must have shape (Nx, Ny) consistent with
        coordinate arrays x and y.
        """
        
        if callable(input_func):
            self._func_Nx, self._func_Ny = None, None
            return _functype._callable

        try:
            arr = np.array(input_func)

            if arr.size == 0:
                raise TypeError('Input function is empty.')

            ndim = arr.ndim
            
            if ndim != 2:
                raise TypeError(f'Discrete input function is {ndim}D, must be 2D')

            self._func_Nx, self._func_Ny = arr.shape
            
            return _functype._discrete

        except Exception as e:
            raise TypeError(f"Error during classification: {e}")
        
    
    def _reconstruct(self, theta, m_vals, rho_m_r, keep_m = None):
        """
        Reconstruct field from selected harmonics.

        Computes

            f(r, θ) = ∑_m ρ_m(r) e^{i m θ}

        Parameters
        ----------
        theta : ndarray of shape (Ntheta,)
            Angular grid.

        m_vals : ndarray of shape (Nm,)
            Available harmonic indices.

        rho_m_r : ndarray of shape (Nr, Nm)
            Radial harmonic coefficients.

        keep_m : sequence of int or None
            Subset of harmonics to include. If None, all are used.

        Returns
        -------
        ndarray of shape (Nr, Ntheta)
            Reconstructed field in polar coordinates.
        """

        if keep_m is None:
            keep_m = m_vals

        m_mask = np.isin(m_vals, keep_m)
        m_sel = m_vals[m_mask]
        rho_m_r = rho_m_r[:, m_mask]
        f_recon = rho_m_r @ np.exp(1j * np.outer(m_sel, theta)) # shape = (Nr, Nm) @ (Nm, Nθ) = (Nr, Nθ)
        return f_recon
    
    def _relative_error_percent(self, f_orig, f_recon):
        """
        Compute RMS relative reconstruction error.

        Parameters
        ----------
        f_orig : ndarray
            Original field.

        f_recon : ndarray
            Reconstructed field.

        Returns
        -------
        float
            Relative RMS error in percent:

                100 × ||f_recon − f_orig||₂ / ||f_orig||₂

            Returns NaN if denominator is zero.
        """

        num   = np.linalg.norm(f_recon - f_orig)
        denom = np.linalg.norm(f_orig)
        
        if denom == 0:
            return np.nan
        
        return num/denom*100.0
        
    def _decompose(self):
        """
        Perform angular Fourier decomposition and harmonic selection.
        Optimized version: vectorized radial cutoff computation.
        """

        # -------------- Angular FFT --------------
        m_fft = np.arange(self._Ntheta)
        m_fft = np.where(m_fft <= self._Ntheta // 2,
                         m_fft,
                         m_fft - self._Ntheta)

        if self._m_abs_max is not None:
            m_mask = np.abs(m_fft) <= int(self._m_abs_max)
        else:
            m_mask = np.ones_like(m_fft, bool)

        self._Nm_mmax = m_mask.sum()
        self.m_vals = m_fft[m_mask]

        self._rho_m_r = (
            fft(self.f_polar, axis=1) / self._Ntheta
        )[:, m_mask]  # shape (Nr, Nm)

        #  Radial densities and total powers
        radial_density = np.abs(self._rho_m_r)**2 * self.r[:, None]

        # Total powers (one trapezoid along axis=0)
        self.powers = np.trapezoid(radial_density, self.r, axis=0)

        total_power_sum = self.powers.sum()
        if total_power_sum > 0:
            self._power_fracs = self.powers / total_power_sum
        else:
            self._power_fracs = np.zeros_like(self.powers)

        # ------------ Calculating radial cutoffs --------------
        
        dr = self.r[1] - self.r[0]  # uniform grid assumed

        cdf = np.zeros_like(radial_density)
        cdf[1:] = dr * np.cumsum(0.5 * (radial_density[:-1] + radial_density[1:]), axis=0)

        totals = cdf[-1]  # total power per harmonic
        targets = (1 - self._recon_power_tol) * totals  
        
        Nm = self._Nm_mmax
        r_cutoff = np.full(Nm, np.nan)

        for j in range(Nm):

            total = totals[j]
            if total <= 0:
                continue

            target = targets[j]
            idx = np.searchsorted(cdf[:, j], target)

            if idx <= 0:
                r_cutoff[j] = self.r[0]
                continue
            if idx >= len(self.r):
                r_cutoff[j] = self.r[-1]
                continue

            r0 = self.r[idx - 1]
            r1 = self.r[idx]
            c0 = cdf[idx - 1, j]
            c1 = cdf[idx, j]

            if c1 == c0:
                r_cutoff[j] = r0
            else:
                t = (target - c0) / (c1 - c0)
                r_cutoff[j] = r0 + t * (r1 - r0)

        self._r_cutoff = r_cutoff

        # -------------- ranking harmonics based on fractional power--------------
        order = np.argsort(-self._power_fracs)
        m_sorted = [self.m_vals[order[0]]]

        iteration = 1
        while iteration <= self._Ntheta:

            self.f_recon = self._reconstruct(
                self.theta,
                self.m_vals,
                self._rho_m_r,
                keep_m=m_sorted
            )

            self.recon_error = self._relative_error_percent(
                self.f_polar,
                self.f_recon
            )

            if (self.recon_error <= self._recon_err_tol
                    or len(m_sorted) >= len(self.m_vals)):
                break

            m_sorted.append(self.m_vals[order[len(m_sorted)]])
            iteration += 1

        # Outputs
        
        order_ = order[:len(m_sorted)]

        self.powers = self.powers[order_]
        self._rho_m_r = self._rho_m_r[:, order_]
        self._power_fracs = self._power_fracs[order_]
        self._r_cutoff = self._r_cutoff[order_]

        self.m_sorted = m_sorted
        self.Nm = len(m_sorted)

        self.rho = {int(m): self._rho_m_r[:, i] for i, m in enumerate(m_sorted)}

        self.power_fracs = {int(m): self._power_fracs[i] for i, m in enumerate(m_sorted)}

        self.r_cutoff = {int(m): self._r_cutoff[i] for i, m in enumerate(m_sorted)}


    def plot_harmonics_with_hist(self, title = ''):
        """
        Plot radial harmonic amplitudes and power histogram.

        Produces a two-panel figure:
            Left: |ρ_m(r)| vs r for retained harmonics.
            Right: Bar chart of fractional power contributions.

        Parameters
        ----------
        title : str, optional
            Figure title.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axs : ndarray of matplotlib.axes.Axes

        Notes
        -----
        Vertical dashed lines indicate radial cutoff for each harmonic.
        """

        maxabs = max(np.abs(self.m_sorted))
        if maxabs == 0:
            maxabs = 3

        fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout = 'constrained')
        ax1, ax2 = axs

        for i in range(len(self.m_sorted)):
            m = int(self.m_sorted[i])
            abs_rho = np.abs(self._rho_m_r[:, i])
            ax1.plot(self.r, abs_rho, label = f"m={m}, {self._power_fracs[i]*100:.4f}%")
            ax1.axvline(self._r_cutoff[i], color = ax1.lines[-1].get_color(), ls = '--')
        ax1.grid(ls = "--", alpha = 0.4)
        ax1.set_xlabel(r"$r ~[\mathrm{\AA}]$")
        ax1.set_ylabel(r"$|\rho_m(r)|$")
        ax1.legend(fontsize="small")
        ax1.set_title(fr"Radial components")

        ax2.bar(self.m_sorted, self._power_fracs)
        ax2.set_xlabel(r"$m$")
        ax2.set_ylabel("Fractional power")
        ax2.set_ylim(0, 1.01)
        ax2.set_xlim(-maxabs, maxabs)
        ax2.set_xticks(np.arange(-maxabs, maxabs+1))
        ax2.grid(ls = "--", alpha = 0.4)
        fig.suptitle(title)
        return fig, axs

    def plot_original_vs_reconstructions(self, title="Reconstruction comparison"):
        """
        Compare original and reconstructed polar fields.

        Produces side-by-side pcolormesh plots of:
            - |f(r, θ)| (original)
            - |f_recon(r, θ)| (reconstructed)

        Parameters
        ----------
        title : str
            Figure title.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axs : ndarray of matplotlib.axes.Axes

        Notes
        -----
        Visualization is shown in Cartesian coordinates
        (x = r cos θ, y = r sin θ).
        """

        labels = ["Original", f"Reconstructed\nError = {self.recon_error:.2f}%, |m_max|={np.abs(max(self.m_sorted))}"]
        fields = [self.f_polar, self.f_recon]

        r = self.r
        theta = self.theta

        r_edges = np.concatenate(([r[0] - (r[1]-r[0])/2], 
                                  (r[:-1] + r[1:]) / 2, 
                                  [r[-1] + (r[-1]-r[-2])/2]))
        theta_edges = np.concatenate(([theta[0] - (theta[1]-theta[0])/2],
                                      (theta[:-1] + theta[1:]) / 2,
                                      [theta[-1] + (theta[-1]-theta[-2])/2]))

        R_edges, TH_edges = np.meshgrid(r_edges, theta_edges, indexing="ij")
        X_edges, Y_edges = R_edges*np.cos(TH_edges), R_edges*np.sin(TH_edges)

        fig, axs = plt.subplots(1, 2, figsize = (10, 4.5), layout = 'constrained')
        for ax, lab, F in zip(axs, labels, fields):
            pcm = ax.pcolormesh(X_edges, Y_edges, np.abs(F), cmap="inferno", shading="auto")
            ax.set_aspect("equal")
            ax.set_title(lab)
            ax.set_xlabel(r'$x-x_{\mathrm{centroid}}$')
            ax.set_ylabel(r'$y-y_{\mathrm{centroid}}$')
            plt.colorbar(pcm, ax = ax, fraction = 0.046, pad = 0.04)
        fig.suptitle(title)
        return fig, ax
