import jax
import jax.numpy as jnp
from jax.scipy.special import factorial

def Wave(
    amplitudes: list[float],
    frequency: float,
):
    """Function for computing time-harmonic electric fields.

    Args:
        amplitudes: electric field amplitudes in xyz-components
        frequency: angular frequency

    Returns:
       Function that computes the electric field as a functon of time
    """
    def _field(t, real = True):
        val = (jnp.exp(1j * frequency * t) * static_part)
        return val.real if real else val
    static_part = jnp.array(amplitudes)
    return _field

def Ramp(
    amplitudes: list[float],
    frequency: float,
    ramp_duration: float,
    time_ramp: float,
):
    """Function for computing ramping up time-harmonic electric fields.

    Args:
        amplitudes: electric field amplitudes in xyz-components
        frequency: angular frequency
        ramp_duration: specifies how long does the electric field ramps up
        time_ramp: specifies time at which the field starts to ramp up

    Returns:
       Function that computes the electric field as a functon of time
    """
    def _field(t, real = True):        
        val =  (
            static_part
            * jnp.exp(1j * frequency * t)
            / (1 + 1.0 * jnp.exp(-ramp_constant * (t - time_ramp)))
        )
        return val.real if real else val
    static_part = jnp.array(amplitudes)
    p = 0.99
    ramp_constant = 2 * jnp.log(p / (1 - p)) / ramp_duration
    return _field


def Pulse(
    amplitudes: list[float],
    frequency: float,
    peak: float,
    fwhm: float,
):
    """Function for computing temporally located time-harmonics electric fields. The pulse is implemented as a temporal Gaussian.

    Args:
        amplitudes: electric field amplitudes in xyz-components
        frequency: angular frequency of the electric field
        peak: time where the pulse reaches its peak
        fwhm: full width at half maximum

    Returns:
       Function that computes the electric field
    """
    def _field(t, real = True):
        val = (
            static_part
            * jnp.exp(-1j * jnp.pi / 2 + 1j * frequency * (t - peak))
            * jnp.exp(-((t - peak) ** 2) / sigma**2)
        )
        return val.real if real else val
    static_part = jnp.array(amplitudes)
    sigma = fwhm / (2.0 * jnp.sqrt(jnp.log(2)))
    return _field

def _binom(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

def _laguerre(k, n, x):
    """Generalized (associated) Laguerre polynomial :math:`L_n^k(x)`.

    Args:
        k: Associated index (integer, typically k >= 0).
        n: Polynomial order (integer, n >= 0).
        x: Argument (scalar or array-like; may be JAX array).

    Returns:
        Value of :math:`L_n^k(x)` evaluated at ``x`` (same shape as ``x``).

    Notes:
        This implements the finite series definition. It is convenient and explicit,
        but not the most numerically stable for large ``n`` or large ``x``.
    """
    prefac = 1/factorial(n)
    res = 0
    for i in range(n+1):
        res += factorial(n) / factorial(i) * _binom(k+n, n - i) * (-x)**i        
    return prefac * res

def laguerre_gaussian(x, y, z, p, l, w0, wavelength):
    """Paraxial Laguerre–Gaussian mode (scalar complex field envelope).

    Implements a standard paraxial LG mode with radial index ``p`` and azimuthal
    index ``l``. The field is returned as a complex scalar envelope evaluated at
    coordinates (x, y, z).

    Args:
        x: x-coordinate (scalar or array-like; may be JAX array).
        y: y-coordinate (scalar or array-like; may be JAX array).
        z: z-coordinate (scalar or array-like; may be JAX array).
        p: Radial index (p >= 0).
        l: Azimuthal index / OAM charge (integer; may be negative).
        w0: Beam waist at focus (z=0).
        wavelength: Wavelength (same units as x, y, z, w0).

    Returns:
        Complex scalar LG field value(s) at (x, y, z).

    Notes:
        - This is a scalar paraxial mode (transverse field envelope).
        - The normalization/prefactor here follows the common convention
          :math:`\\sqrt{2 p!/(\\pi (p+|l|)!)}\\, 1/w(z)` up to any global phase.
        - The phase term used here follows your original implementation.
          If you need the standard curvature term ``exp(-i k r^2 / (2 R(z)))``,
          implement it explicitly.
    """
    # normalization factor
    z_r = w0**2 * jnp.pi / wavelength
    w_z = w0 * jnp.sqrt(1 + (z/z_r)**2)
    nom, denom = 2 * factorial(p), jnp.pi * factorial(p + jnp.abs(l))
    prefac = jnp.sqrt(nom/denom) / w_z

    # cylindrical radius
    rho = jnp.sqrt(x**2 + y**2)

    # split expression into 4 factors
    term1 = jnp.power(jnp.sqrt(2) * rho / w_z, jnp.abs(l))

    term2 = jnp.exp(-rho**2/w_z**2) * jnp.exp(1j * l * jnp.arctan2(y, x))

    term3 = _laguerre(jnp.abs(l), p, 2 * rho**2 / w_z**2)

    z_tilde = z + rho**2 * z / (2*(z**2 + z_r**2) )
    gouy_phase = jnp.arctan2(z, z_r)
    k = 2*jnp.pi/wavelength
    term4 = jnp.exp(1j *k*z_tilde - 1j*(2*p + jnp.abs(l) + 1)*gouy_phase)

    return term1 * term2 * term3 * term4

def Skyrmion(a_1, a_2, l1, l2, p1, p2, w0, wavelength, e_1 = None, e_2 = None, flake = None):
    """Construct a (vector) optical skyrmion field from two LG modes.

    Builds a paraxial complex electric field as a superposition of two LG modes
    (with indices ``(p1,l1)`` and ``(p2,l2)``) multiplied by two orthogonal
    polarization vectors ``e_1`` and ``e_2``.

    By default, ``e_1`` and ``e_2`` are right/left circular polarization vectors
    in the transverse xy-plane.

    Args:
        a_1: Complex (or real) amplitude for the first mode/polarization component.
        a_2: Complex (or real) amplitude for the second mode/polarization component.
        l1: Azimuthal index of the first LG mode.
        l2: Azimuthal index of the second LG mode.
        p1: Radial index of the first LG mode (p1 >= 0).
        p2: Radial index of the second LG mode (p2 >= 0).
        w0: Beam waist at focus (z=0).
        wavelength: Wavelength (same units as spatial coordinates).
        e_1: Polarization vector for component 1 (shape (3,)). Defaults to RCP.
        e_2: Polarization vector for component 2 (shape (3,)). Defaults to LCP.
        flake: Optional object with attribute ``positions`` of shape (N, 3).
            If provided, the field is evaluated on these positions and a
            time-harmonic real-valued field function ``E(t)`` is returned for
            time-domain simulations.

    Returns:
        If ``flake is None``:
            A function ``E(x_grid, y_grid, z)`` that returns an array of shape
            ``(Nx, Ny, 3)`` with complex vector field values.
        If ``flake is not None``:
            A function ``E(t)`` returning a real-valued array of shape ``(N, 3)``
            giving the time-harmonic electric field on ``flake.positions``.

    Notes:
        - The returned vector field is paraxial/transverse (Ez component is zero
          for the default polarization vectors).
        - For polarization-skyrmion calculations, build Stokes parameters from the
          transverse components (or circular basis) and normalize.
    """

    if e_1 is None:
        e_1 = 1/jnp.sqrt(2) * jnp.array([1, 1j, 0])
    if e_2 is None:
        e_2 = 1/jnp.sqrt(2) * jnp.array([1, -1j, 0])

    lg1 = lambda x, y, z : laguerre_gaussian(x, y, z, p1, l1, w0, wavelength)
    lg2 = lambda x, y, z : laguerre_gaussian(x, y, z, p2, l2, w0, wavelength)
        
    def _field(x, y, z):
        return a_1 * lg1(x, y, z) * e_1 + a_2 * lg2(x, y, z) * e_2

    if flake is None:        
        return jax.vmap(
            jax.vmap(
                _field, in_axes = (None, 0, None)
            ),
            in_axes = (0, None, None)
        )

    # flake given => want to use for time domain sims
    
    def _field_r(r):
        return _field(r[0], r[1], r[2])
    
    static_part = jax.vmap(_field_r)(flake.positions)

    return lambda t : (static_part * jnp.exp(1j * 2 * jnp.pi / wavelength * t)).real


def get_skyrmion_number(sv, eps = 1e-24):
    """Compute the skyrmion number from a (discretized) Stokes field via Berg's method.

    This implements the solid-angle / triangulation formula commonly attributed to
    Berg & Lüscher: the skyrmion number is the sum of oriented spherical triangle
    areas (solid angles) over a 2D grid, divided by ``4π``.

    Args:
        sv: Stokes field array of shape ``(N, M, 4)`` with components
            ``[S0, S1, S2, S3]`` at each grid point.
        eps: Relative intensity threshold for masking. Pixels with
            ``S0 <= eps * max(S0)`` are excluded to avoid numerical noise from
            division by very small intensities.

    Returns:
        Scalar skyrmion number (topological charge) as a JAX scalar.

    Notes:
        - The method assumes the normalized Stokes vector
          ``s = (S1,S2,S3)/S0`` maps the plane to (approximately) a compact domain,
          i.e. the polarization becomes (nearly) constant at the boundary of the
          numerical window. In practice, choose a sufficiently large spatial window.
        - This implementation relies on JAX's out-of-bounds indexing behavior
          (it clamps indices rather than raising), so the boundary contributes
          negligibly if the field is constant there. See the JAX note:
          https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
        - If you prefer strict array bounds, change the index ranges to
          ``jnp.arange(N-1)``, ``jnp.arange(M-1)`` and avoid out-of-bounds access.
    """    
    def angle(a, b, c):
            return 2.0 * jnp.arctan2(a @ jnp.cross(b, c), 1 + a @ b + a @ c + b @ c)

    def triangle(i, j):
        a, b, c, d = arr[i, j], arr[i + 1, j], arr[i, j + 1], arr[i + 1, j + 1]
        return angle(a,c,b) + angle(a,c,d)

    # intensities
    S0 = sv[:, :, 0]
    
    # "forbidden" intensity values
    mask = S0 > eps * jnp.max(S0)

    # normalized value
    arr  = sv[:, :, 1:] / S0[:, :, None]                 

    # NOTE: the field should be essentially constant on the domain boundary, so it is okay to rely on JAX's  out-of-bounds behavior: # https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
    # TL;DR: make the integration domain sufficiently large
    n, m, _ = arr.shape    
    triangles = jax.vmap(jax.vmap(triangle, in_axes = (None, 0) ), in_axes = (0, None))(jnp.arange(n), jnp.arange(m)) * mask
    
    return triangles.sum() / (4 * jnp.pi)

def stokes_vector(f):
    """Build a function that computes Stokes parameters from a complex vector E-field.

    The returned function evaluates the field ``E = f(x,y,z)`` and projects it onto
    right/left circular polarization basis vectors (in the transverse xy-plane).
    It then returns the Stokes parameters ``[S0, S1, S2, S3]`` in the circular basis.

    Args:
        f: Callable electric field function ``f(x, y, z) -> (..., 3)`` returning a
            complex-valued vector field (Ex, Ey, Ez). Typically paraxial with Ez=0.

    Returns:
        Callable ``sv(x, y, z)`` that returns an array of shape ``(..., 4)``
        containing ``[S0, S1, S2, S3]``.

    Notes:
        - This uses circular basis vectors
          ``e_r = (x + i y)/sqrt(2)``, ``e_l = (x - i y)/sqrt(2)``.
        - The sign convention for ``S2`` depends on the chosen Stokes definition.
          Keep it consistent with your skyrmion-number code / expected sign.
    """
    e_r = 1/jnp.sqrt(2) * jnp.array([1,  1j, 0])
    e_l = 1/jnp.sqrt(2) * jnp.array([1, -1j, 0])

    def _inner(x, y, z):
        E = f(x, y, z)                # (...,3) complex
        Er = E @ jnp.conj(e_r)        # (...)
        El = E @ jnp.conj(e_l)        # (...)

        S0 = jnp.abs(Er)**2 + jnp.abs(El)**2
        S1 = 2 * jnp.real(Er * jnp.conj(El))
        S2 = 2 * jnp.imag(Er * jnp.conj(El))
        S3 = jnp.abs(Er)**2 - jnp.abs(El)**2

        sv = jnp.stack([S0, S1, S2, S3], axis=-1)
        
        return sv

    return _inner
