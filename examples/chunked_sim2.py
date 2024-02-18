import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

cc_distance=1.42028166
# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(5),
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=cc_distance*jnp.sqrt(3),
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
#plot initial energy state occupations
plt.scatter([0]*len(stack.energies),stack.electrons*jnp.diag(stack.rho_0).real)
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
time_axis = jnp.linspace(0, 1/gamma, 100001)

# allow transfer from higher energies to lower energies only if the
# two energy levels are not degenerate
diff = stack.energies[:,None] - stack.energies
gamma_matrix = gamma * jnp.logical_and( diff < 0, jnp.abs(diff) > stack.eps, )

relaxation_function = granad.lindblad( stack, gamma_matrix.T )

# max amount of memory (in bytes) to be allocated to the resulting array
max_memory = 10**8
print(f"Max memory= {max_memory/1e9} GB")
# the index after which to keep the results (Result==> diagonal elements of rho)
keep_after_index = time_axis.size - int(max_memory / stack.rho_0.diagonal().nbytes)

# split time axis into two arrays: discard/keep results for the first/second array
#if memory requried is less than max memory available: keep the whole time range 
if keep_after_index<0:
    split_axis=[time_axis]
else:
    split_axis = jnp.split(time_axis, [keep_after_index])

for i, t in enumerate( split_axis ):
    if i == 0:
        if keep_after_index<0:
            #first array => keep all the elements as it is less than max available memory
            postprocess=lambda rho: jnp.diag(granad.to_energy_basis(stack,rho)) #In this implementation evolution (integration) is done on rho in site basis
        else:
            # first array => let JAX set the return value to None ()
            postprocess = lambda rho : None
    else:
        # second array => usual postprocessing
        postprocess = lambda rho: jnp.diag(granad.to_energy_basis(stack,rho))

    stack, energystate_occupations = granad.evolution(stack,
                                          t,
                                          field_func,
                                          relaxation_function,
                                          postprocess = postprocess)

# plot occupations as function of time
plt.plot( split_axis[-1], stack.electrons * energystate_occupations.real, label = [round(float(energy),2) for energy in stack.energies] )
plt.xlabel("Time")
plt.ylabel("Energy state occupation (# of electrons)")
plt.title("Time evolution of Energy state occupation level")
plt.legend()
plt.show()

