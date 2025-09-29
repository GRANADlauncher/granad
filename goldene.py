from granad import *

sk_file = "Au-Au.skf"

# gold_fcc = (Material("gold_fcc")
#             .lattice_constant(4.078)  # experimental fcc lattice constant for Au (Å)
#             .lattice_basis([
#                 [0, 0.5, 0.5],
#                 [0.5, 0, 0.5],
#                 [0.5, 0.5, 0]
#             ])
#             .add_atom(atom="Au", position=(0, 0, 0), orbitals=["s"])  # fcc primitive cell, one atom basis
#             .add_slater_koster_interaction("Au", "Au", sk_file, num_neighbors = 20)
#             )

# flake = gold_fcc.cut_flake( [(0,5), (0,5), (0,3)] )
# flake.show_3d()
# flake.show_energies()


# https://www.nature.com/articles/s44160-024-00518-4
goldene = (Material("goldene")
           .lattice_constant(2.62)  # use 2.62 Å (exp) or 2.735 Å (DFT relaxed)
           .lattice_basis([
               [1, 0, 0],
               [-0.5, jnp.sqrt(3)/2, 0]
           ])
           .add_atom(atom = "Au", position=(0, 0), orbitals = ["s"])  # one Au atom per primitive cell (P6/mmm)
           .add_slater_koster_interaction("Au", "Au", sk_file, num_neighbors = 20)
           )

flake = goldene.cut_flake(Rectangle(40, 40))
flake.show_2d()
flake.show_energies()
