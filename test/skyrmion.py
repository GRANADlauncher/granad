import jax
from jax.scipy.special import factorial
import jax.numpy as jnp
import matplotlib.pyplot as plt

# for testing
# from scipy.special import assoc_laguerre

# # x, n, k = 1001, 2, 0
# # theirs = assoc_laguerre(x, n, k)
# # ours = laguerre(k, n, x)
# # print(theirs, ours)

e_r = 1/jnp.sqrt(2) * jnp.array([1, 1j, 0])
e_l = 1/jnp.sqrt(2) * jnp.array([1, -1j, 0])

def binom(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

def laguerre(k, n, x):
    prefac = 1/factorial(n)
    res = 0
    for i in range(n+1):
        res += factorial(n) / factorial(i) * binom(k+n, n - i) * (-x)**i        
    return prefac * res


def psi(tau, x, l, p, gamma, w0):
    """
    equation 2 in https://doi.org/10.1038/s41377-025-02028-0
    """
    gts = gamma**2 * tau**2 # gamma tau squared
    fac1 = ( jnp.sqrt(gts + x**2) / w0)**jnp.abs(l)
    fac2 = jnp.exp(-(gts + x**2) / w0**2)
    fac3 = laguerre(jnp.abs(l), p, 2 * (gts + x**2) / w0**2)
    fac4 = jnp.exp(1j * l * jnp.arctan(x / (gamma * tau) ) )   
    return fac1 * fac2 * fac3 * fac4

def e_field(tau, x, y, y0, c0, gamma, w0, phi_gamma):
    prefac = jnp.exp(-y**2/y0**2)
    right = jnp.cos(c0) * psi(tau, x, 0, 0, gamma, w0) * e_r
    left = jnp.sin(c0) * jnp.exp(1j * phi_gamma) * psi(tau, x, 1, 0, gamma, w0) * e_l

    return prefac * (left + right)

def intensity(E):
    """
    E: (..., 3) complex vector field
    returns scalar intensity
    """
    return jnp.sum(jnp.abs(E)**2, axis=-1)

def evaluate_field_on_grid(field_fn, taus, xs, ys):
    """
    field_fn: function(tau, x, y) -> (3,) complex
    taus, xs, ys: 1D arrays

    returns: intensity grid of shape (Nt, Nx, Ny)
    """

    # vectorize field evaluation
    E_vec = jax.vmap(
        jax.vmap(
            jax.vmap(field_fn, in_axes=(0, None, None)),
            in_axes=(None, 0, None)
        ),
        in_axes=(None, None, 0)
    )    
    vals = E_vec(taus, xs, ys)
    
    return vals

def stokes_parameters(E):
    Er = E @ e_r
    El = E @ e_l

    S0 = jnp.abs(Er)**2 + jnp.abs(Er)**2
    S1 = 2 * jnp.real(e_r * jnp.conj(e_l))
    S2 = 2 * jnp.imag(e_r * jnp.conj(e_l))
    S3 = jnp.abs(Er)**2 - jnp.abs(Er)**2

    return S1 / S0, S2 / S0, S3 / S0

def plot_intensity():
    # parameters, best guess essentially
    y0 = 1.5
    w0 = 1
    gamma = 1
    phi_gamma = jnp.pi/4

    # field functions
    mode_00 = lambda tau, x, y : e_field(tau, x, y, y0, jnp.pi, gamma, w0, phi_gamma)
    mode_01 = lambda tau, x, y : e_field(tau, x, y, y0, jnp.pi/2, gamma, w0, phi_gamma)

    # grids
    taus = jnp.linspace(-1, 1, 30)
    xs   = jnp.linspace(-1, 1, 20)
    ys   = jnp.linspace(-2, 2, 10)

    E00 = evaluate_field_on_grid(mode_00, taus, xs, ys) # Intensity[y, x, tau]
    E01 = evaluate_field_on_grid(mode_01, taus, xs, ys)

    # compute intensity
    I00 = intensity(E00)
    I01 = intensity(E01)

    for i in range(10):
        fig, axs = plt.subplots(1, 2)
        im0 = axs[0].imshow(I00[0, :, :], aspect = "auto")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(I01[0, :, :], aspect = "auto")
        fig.colorbar(im1, ax=axs[1])

        plt.savefig(f"result_{i}.pdf")

# mixed mode stokes plot
y0 = 1.5
w0 = 1
c0 = jnp.pi/4 # equal mixture
gamma = 1
phi_gamma = jnp.pi/4

mode_mixed = lambda tau, x, y : e_field(tau, x, y, y0, c0, gamma, w0, phi_gamma)

# grids
taus = jnp.linspace(-1, 1, 30)
xs   = jnp.linspace(-1, 1, 20)
ys   = jnp.linspace(-2, 2, 10)
