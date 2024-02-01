# import numpy as np
# get = lambda n, x : np.where( np.abs(np.abs(stack.coulomb[n]) - x) < 1e-10)


# lattice_coupling = coulomb_graphene
# lattice_basis = lattice_coupling.lattice.lattice_basis
# orbital_positions = lattice_coupling.lattice.orbital_positions

# couplings = jnp.array([x + 0j for x in lattice_coupling.couplings])
# distances = jnp.sort(
#     jnp.unique(
#         jnp.round(
#             jnp.array(
#                 [
#                     jnp.linalg.norm(i * basis_vec + orbital_vec)
#                     for orbital_vec in orbital_positions
#                     for basis_vec in lattice_basis
#                     for i in range(len(couplings))
#                 ]
#             ),
#             8,
#         )
#     )
# )[: len(couplings)]
