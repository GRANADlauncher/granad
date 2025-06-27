import jax.numpy as jnp
from granad import *
from granad._plotting import *

def show_energies(orbs, B, display = None, label = None, e_max = None, e_min = None):
    """
    Depicts the energy and occupation landscape of a stack, with energies plotted on the y-axis and eigenstates ordered by size on the x-axis.

    Parameters:
        `orbs`: An object containing the orbital data, including energies, electron counts, and initial density matrix.
        `display` (jnp.Array, optional): Array to annotate the energy states.
            - If `None`, electronic occupation is used.
        `label` (jnp.Array, optional): Label for the colorbar.
            - If `None`, "initial state occupation" is used.
        `e_max` (float, optional): The upper limit of the energy range to display on the y-axis. 
            - If `None`, the maximum energy is used by default.
            - This parameter allows you to zoom into a specific range of energies for a more focused view.
        `e_min` (float, optional): The lower limit of the energy range to display on the y-axis.
            - If `None`, the minimum energy is used by default.
            - This parameter allows you to filter out higher-energy states and focus on the lower-energy range.

    Notes:
        The scatter plot displays the eigenstate number on the x-axis and the corresponding energy (in eV) on the y-axis.
        The color of each point represents the initial state occupation, calculated as the product of the electron count and the initial density matrix diagonal element for each state.
        A color bar is added to indicate the magnitude of the initial state occupation for each eigenstate.
    """
    from matplotlib.ticker import MaxNLocator
    e_max = (e_max or orbs.energies.max()) 
    e_min = (e_min or orbs.energies.min())
    widening = (e_max - e_min) * 0.01 # 1% larger in each direction
    e_max += widening
    e_min -= widening
    energies_filtered_idxs = jnp.argwhere( jnp.logical_and(orbs.energies <= e_max, orbs.energies >= e_min))
    state_numbers = energies_filtered_idxs[:, 0]
    energies_filtered = orbs.energies[energies_filtered_idxs]

    if display is None:
        display = jnp.diag(orbs.electrons * orbs.initial_density_matrix_e)
    label = label or "initial state occupation"
    
    colors =  display[energies_filtered_idxs]

    fig, ax = plt.subplots(1, 1)
    plt.colorbar(
        ax.scatter(
            state_numbers,
            energies_filtered,
            c=colors,
        ),
        label=label,
    )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(e_min, e_max)    


def peierls_coupling(B, t, r1, r2):
    plus = r1 + r2
    minus = r1 - r2
    return complex(t * jnp.exp(1j * B * minus[0] * plus[1]))

def get_graphene_b_field(B, t, shape):
    """obtain tb model for graphene in magnetic field from peierls substitution

    Args:
        B : B-field strength
        t : nn hopping
    
    """
    graphene_peierls = (
        Material("graphene_peierls")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz", atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz")
        .add_interaction(
            "hamiltonian",
            participants=("pz", "pz"),
            parameters = [0],
        )
        .add_interaction(
            "coulomb",
            participants=("pz", "pz"),
            expression = ohno_potential(1.42)
        )
    )
    
    flake = graphene_peierls.cut_flake(shape)
    distances = jnp.round(jnp.linalg.norm(flake.positions - flake.positions[:, None], axis = -1), 4)
    nn = 1.5
    for i, orb1 in enumerate(flake):
        for j in range(i+1):
            if 0 < distances[i, j] <= nn:
                orb2 = flake[j]
                flake.set_hamiltonian_element(orb1, orb2, peierls_coupling(B, t, orb1.position, orb2.position))
    return flake

def make_plots():
    shape = Hexagon(30)
    flake = get_graphene().cut_flake(shape)
    flake.show_energies()
    flake.show_2d()

    # show quantization
    flake = get_graphene_b_field(1, -2.7, shape)
    flake.show_2d()
    flake.show_energies()
    omegas = jnp.linspace(0, 10)

    dos = jax.vmap(lambda w : flake.get_dos(w, broadening = 0.05))(omegas)
    plt.plot(omegas, dos)
    plt.show()
