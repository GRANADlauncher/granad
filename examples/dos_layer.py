import jax
import jax.numpy as jnp

import granad

import matplotlib.pyplot as plt


# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(7.4),
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_type=granad.LatticeType.HONEYCOMB,
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

omegas = jnp.linspace( min(stack.energies) - 0.1, max(stack.energies)  + 0.1, 100 )
dos = []
ldos = []
for o in omegas:
    dos.append( granad.dos( stack, o, broadening = 10 ) )
    ldos.append( granad.ldos( stack, o, site_index= 0, broadening = 10 ) )

plt.plot(omegas, dos)
plt.xlabel(r'$\omega$ (eV)')
plt.ylabel('DOS')
plt.show()

plt.plot(omegas, ldos)
plt.xlabel(r'$\omega$ (eV)')
plt.ylabel('LDOS')
plt.show()
