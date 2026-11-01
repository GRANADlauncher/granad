import jax
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad import MaterialCatalog, Hexagon, Skyrmion, laguerre_gaussian
from matplotlib.ticker import MaxNLocator

# ============================================================
# Helpers
# ============================================================

E_CHARGE = 1.0
HBAR = 1.0


def bond_currents_from_rhos(
    flake,
    rhos,
    include_charge=True,
    hbar=HBAR,
    e_charge=E_CHARGE,
    mask_hoppings=True,
    hopping_tol=1e-14,
):
    """
    Compute bond currents J_ij(t) from a stack of density matrices.

    Convention:
        J[i, j, t] = current flowing from j -> i

    For a single-particle tight-binding Hamiltonian H and density matrix rho,
        J_ij(t) = -(2 e / hbar) Im[ H_ij * rho_ji(t) ]

    Parameters
    ----------
    flake : object
        Must provide flake.hamiltonian and flake.positions.
    rhos : array
        Shape (N, N, T) or (T, N, N) or (N, N).
    """
    H = jnp.asarray(flake.hamiltonian)
    rhos = jnp.asarray(rhos)

    N = H.shape[0]

    prefactor = (-2.0 / hbar) * e_charge
    J = prefactor * jnp.imag(H[None, :, :] * rhos)

    if mask_hoppings:
        hopping_mask = jnp.abs(H) > hopping_tol
        hopping_mask = hopping_mask & (~jnp.eye(N, dtype=bool))
        J = jnp.where(hopping_mask[None, :, :], J, 0.0)
    else:
        J = J.at[jnp.diag_indices(N)].set(0.0)

    return J


def angular_fourier_bond_currents(
    flake,
    J,
    ell_vals,
    center=None,
    mask=None,
    tangential=True,
    normalize=False,
    eps=1e-30,
):
    """
    Discrete angular Fourier transform of bond currents.

    Parameters
    ----------
    flake : object
        Must provide flake.positions with shape (N, 2).
    J : array
        Shape (N, N, T) or (N, N).
        Convention assumed:
            J[i, j, t] = current flowing from j -> i
        so J[j, i, t] is current along directed bond i -> j.
    ell_vals : array-like
        Angular modes, e.g. [-2, -1, 0, 1, 2].
    center : array-like or None
        Origin for angular decomposition. Defaults to mean position.
    mask : array or None
        Optional bond mask of shape (N, N).
    tangential : bool
        If True, project onto local tangential direction first.
    normalize : bool
        If True, divide by sum(abs(weights)).
    """
    R = jnp.asarray(flake.positions[:, :2])
    N = R.shape[0]
    ell_vals = jnp.asarray(ell_vals)

    if center is None:
        center = jnp.mean(R, axis=0)
    center = jnp.asarray(center)

    Ri = R[:, None, :]
    Rj = R[None, :, :]

    mids = 0.5 * (Ri + Rj) - center
    phi_mid = jnp.arctan2(mids[..., 1], mids[..., 0])

    bond_vec = Rj - Ri  # directed bond i -> j
    bond_len = jnp.linalg.norm(bond_vec, axis=-1, keepdims=True)
    bond_hat = bond_vec / jnp.maximum(bond_len, eps)

    if tangential:
        rnorm = jnp.linalg.norm(mids, axis=-1, keepdims=True)
        phi_hat = jnp.stack([-mids[..., 1], mids[..., 0]], axis=-1)
        phi_hat = phi_hat / jnp.maximum(rnorm, eps)
        geom_weight = jnp.sum(bond_hat * phi_hat, axis=-1)
    else:
        geom_weight = jnp.ones((N, N), dtype=R.dtype)

    geom_weight = geom_weight * (1.0 - jnp.eye(N, dtype=R.dtype))

    if mask is not None:
        mask = jnp.asarray(mask)
        if mask.shape != (N, N):
            raise ValueError(f"mask must have shape ({N}, {N})")
        geom_weight = geom_weight * mask

    # directed current along i -> j, TODO: not even sure whether we need this
    I_dir = jnp.transpose(J, (0, 2, 1))
    Iw = geom_weight[None] * I_dir

    phase = jnp.exp(-1j * ell_vals[:, None, None] * phi_mid[None, :, :])
    F = jnp.einsum("lij,tij->lt", phase, Iw)

    if normalize:
        denom = jnp.sum(jnp.abs(geom_weight))
        F = F / jnp.maximum(denom, eps)

    return F, phi_mid, geom_weight


if __name__ == "__main__":
    print("Building flake...")
    flake = MaterialCatalog.get("graphene").cut_flake(Hexagon(10))
    flake.shift_by_vector(-flake.positions[flake.center_index])
    
    print("Setting up field...")
    a_1, a_2 = 1.0, 1.0
    l1, l2 = 2, 0
    p1, p2 = 1, 1
    w0 = 5.0
    wavelength = 1.0
    
    lg1 = laguerre_gaussian(p1, l1, w0, wavelength, 1e-4 * jnp.array([1, 0, 0]), flake=flake)
    lg2 = laguerre_gaussian(p2, l2, w0, wavelength, 1e-4 * jnp.array([1, 0, 0]), flake=flake)
    skyrmion_x = lambda t : lg1(t) +  lg2(t)
    
    lg3 = laguerre_gaussian(p1, l1, w0, wavelength, 1e-4 * jnp.array([0, 1, 0]), flake=flake)
    lg4 = laguerre_gaussian(p2, l2, w0, wavelength, 1e-4 * jnp.array([0, -1, 0]), flake=flake)
    skyrmion_y = lambda t : lg3(t) +  lg4(t)
        
    # is_skyrmion = "laguerre_gaussian_only"
    name = "skyrmion_0_2"
    # name = "skyrmion_0_x_only"

    js = []
    for illumination in [skyrmion_x, skyrmion_y]:
        print("Running time domain simulation...")
        result = flake.master_equation(
            relaxation_rate=1 / 10,
            illumination=illumination,
            expectation_values=[flake.velocity_operator],
            density_matrix=["full"],
            end_time=40,
        )

        velocity_output = result.output[0]
        rhos = (result.output[1] - flake.initial_density_matrix[None])

        print("Extracting bond currents...")
        J = bond_currents_from_rhos(flake, rhos)

        js.append(J)

    J = js[0] - 1j * js[1]
    # J = js[0]

    print("Performing angular fourier decomposition...")    
    ell_vals = jnp.arange(-6, 7)
    F_ell_t, phi_mid, weights = angular_fourier_bond_currents(
        flake,
        J,
        ell_vals=ell_vals,
        tangential=True,
        normalize=True,
    )

    
    fig, axs = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True)    

    # selected angular modes in time
    ts = result.time_axis
    for ell in range(-6, 6):
        idx = int(jnp.where(ell_vals == ell)[0][0])
        axs[0].plot(ts, jnp.real(F_ell_t[idx]), label=f"Re F_{{{ell}}}")
    axs[0].set_title("Real part of angular Fourier modes of bond currents")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel(r"$Re F_{\ell}$")
    axs[0].legend(ncol=3)

    # mode magnitude heatmap
    im = axs[1].imshow(
        jnp.abs(F_ell_t),
        aspect="auto",
        origin="lower",
        extent=[float(ts[0]), float(ts[-1]), float(ell_vals[0]), float(ell_vals[-1])],
    )
    axs[1].set_title(r"$|F_{\ell}(t)|$ for angular Fourier modes of bond currents")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel(r"$\ell$")
    plt.colorbar(im, ax=axs[1], label=r"$|F_{\ell}|$")

    # plt.show()
    plt.savefig(f"{name}.pdf")

    flake.show_2d()

    # late = slice(max(0, len(ts) // 2), len(ts))
    # late_strength = jnp.mean(jnp.abs(F_ell_t[:, late]), axis=1)

    # print(r"Late-time average |F_ell|:")
    # for ell, val in zip(ell_vals.tolist(), late_strength.tolist()):
    #     print(f"  ell={ell:>2d} : {val:.6e}")
