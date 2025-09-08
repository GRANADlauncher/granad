from granad import *

flake = get_graphene().cut_flake(Triangle(15))
for orb in flake:
    orb.kind = "z"

# just for pz
params = {"ppp" : -2.7, "pps" : 0.48}
cutoff = lambda vec : jnp.exp(1 -jnp.linalg.norm(vec) / 1.41)

flake.set_hamiltonian_slater_koster(flake[0], flake[0], params, cutoff)
flake.show_energies()

dist = flake.positions[0] - flake.positions[1]
dist /= jnp.linalg.norm(dist)
l, m, n = dist

assert flake.hamiltonian[0, 1].round(1) == -2.7, "SK param wrong"
