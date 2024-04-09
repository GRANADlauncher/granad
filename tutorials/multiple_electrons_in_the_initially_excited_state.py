# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# ## Multiple electrons in the initially excited state
#
# This example demonstrates how to initially excite more than one electron. You can do this by passing lists to the arguements of get_stack.

# +
# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
shape=granad.Triangle(4.1),
lattice_edge=granad.LatticeEdge.ARMCHAIR,
lattice_type=granad.LatticeType.HONEYCOMB,
lattice_constant=2.46,
)
sb.add("pz", graphene)

u = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66]
)

sb.set_hopping(u)

c = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(c)

# create the stack object
stack = sb.get_stack()
print(stack.rho_0.diagonal() * stack.electrons, stack.homo)

# create the stack object with a single excitation involving two electrons
stack = sb.get_stack( from_state = [1], to_state = [2], excited_electrons = [2], doping = 0 )
print(stack.rho_0.diagonal() * stack.electrons, stack.homo)

# create the stack object with two excitations
stack = sb.get_stack( from_state = [1,0], to_state = [2,4], excited_electrons = [2,1], doping = 0 )
print(stack.rho_0.diagonal() * stack.electrons, stack.homo)
# -
