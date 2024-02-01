import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def evolution(
    stack: granad.Stack,
    time: jax.Array,
    field,
    dissipation = None,
    coulomb_strength: float = 1.0,
    postprocess = None,
) -> tuple[granad.Stack, jax.Array]:
    """Propagate a stack forward in time.

    :param stack: stack object
    :param time: time axis
    :param field: electric field function
    :param dissipation: dissipation function
    :param coulomb_strength: scaling factor applied to coulomb interaction strength
    :param postprocess: a function applied to the density matrix after each time step
    :returns: (stack with rho_0 set to the current state, array containing all results from postprocess at all timesteps)
    """

    def integrate(rho, time):
        e_field, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ext = jnp.sum(stack.positions * e_field.real.T, axis=1)
        p_ind = coulomb @ charge
        # print(p_ind.shape)
        # import pdb; pdb.set_trace()
        h_total = stack.hamiltonian + jnp.diag(p_ext) - jnp.diag(p_ind)
        if dissipation:
            return rho - 1j * dt * (h_total @ rho - rho @ h_total) + dt * dissipation( delta_rho ), postprocess(rho) if postprocess else rho
        else:
            return rho - 1j * dt * (h_total @ rho - rho @ h_total), postprocess(rho) if postprocess else rho

    dt = time[1] - time[0]
    coulomb = stack.coulomb * coulomb_strength

    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T
    rho, rhos = jax.lax.scan(
        jax.jit(integrate), stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T, time
    )

    return (
        stack.replace(rho_0=stack.eigenvectors.conj().T @ rho @ stack.eigenvectors),
        rhos,
    )


## custom function for performing the fourier transform
def get_fourier_transform(t_linspace, function_of_time):
    function_of_omega = np.fft.fft(function_of_time) / len(t_linspace)
    omega_axis = (
        2
        * np.pi
        * len(t_linspace)
        / np.max(t_linspace)
        * np.fft.fftfreq(function_of_omega.shape[-1])
    )
    return omega_axis, function_of_omega


# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(20),
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66]
)
sb.set_hopping(hopping_graphene)


coulomb_graphene = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb_graphene)

# create the stack object
stack = sb.get_stack()

amplitudes = [1, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

# propagate in time
time_axis = jnp.linspace(0, 10, 10**3)

import time
t = time.time()
new_stack, occupations = evolution(stack, time_axis, field_func, None, postprocess = jnp.diag)
print(t-time.time())

# calculate dipole moment
dipole_moment = granad.induced_dipole_moment(stack, occupations)

#  perform fourier transform
omega_axis, dipole_omega = get_fourier_transform(time_axis, dipole_moment[:, 0])

# we also need the x-component of the electric field as a single function
electric_field = granad.electric_field(
    amplitudes, frequency, stack.positions[0, :]
)(time_axis)
_, field_omega = get_fourier_transform(time_axis, electric_field[0])

# plot result
omega_max = 100
component = 0
polarizability = dipole_omega / field_omega
spectrum = -omega_axis[:omega_max] * np.imag(polarizability[:omega_max])
plt.plot(omega_axis[:omega_max], np.abs(spectrum) ** (1 / 2))
plt.xlabel(r"$\hbar\omega$", fontsize=20)
plt.ylabel(r"$\sigma(\omega)$", fontsize=25)
plt.legend()
plt.show()
