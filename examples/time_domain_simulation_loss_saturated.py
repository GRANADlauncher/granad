import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Chain(6),
    lattice_type=granad.LatticeType.CHAIN,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1,
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
stack = sb.get_stack( from_state = 0, to_state = 2)

amplitudes = [0.0, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

# propagate in time
gamma = 10
time_axis = jnp.linspace(0, 1/gamma, 200000)

# allow transfer from higher energies to lower energies only if the
# two energy levels are not degenerate
diff = stack.energies[:,None] - stack.energies
gamma_matrix = gamma * jnp.logical_and( diff < 0, jnp.abs(diff) > stack.eps, )

relaxation_function = granad.lindblad( stack, gamma_matrix.T )

new_stack, occupations = granad.evolution(stack, time_axis, field_func, relaxation_function, postprocess = lambda x : jnp.diag( stack.eigenvectors.conj().T @ x @ stack.eigenvectors) )
occupations = np.array( occupations )

# plot occupations as function of time
skip = 100
for i, occ in enumerate(occupations[::skip].T):
    plt.plot( time_axis[::skip] * gamma, stack.electrons * occ.real, label = round(stack.energies[i],2) )
plt.legend()
plt.show()
