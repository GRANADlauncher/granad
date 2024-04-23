from collections import defaultdict
from functools import wraps

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def _plot_wrapper(plot_func):
    @wraps(plot_func)
    def wrapper(*args, **kwargs):
        plot_func(*args, **{key: val for key, val in kwargs.items() if key != "name"})
        try:
            plt.savefig(kwargs["name"])
            plt.close()
        except KeyError:
            plt.show()

    return wrapper

@_plot_wrapper
def show_2d(orbs, show_tags=None, show_index=False, display = None):
    """
    Generates a 2D scatter plot representing the positions of orbitals in the xy-plane,
    filtered by specified tags. Optionally colors points by eigenvector amplitudes.

    Parameters:
    - orbs (list): List of orbital objects, each with attributes 'tag' and 'position'.
    - show_tags (list of str, optional): Tags used to filter orbitals for display.
    - show_index (bool): If True, indexes of the orbitals will be shown on the plot.
    - display: N-element array to display

    Returns:
    - None: A 2D scatter plot is displayed.
    """
    
    # Determine which tags to display
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    # Prepare data structures for plotting
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for orb in orbs:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(orbs.index(orb))

    # Create plot
    fig, ax = plt.subplots()
    
    if display is not None:
        cmap = plt.cm.viridis
        colors = jnp.abs(display) / jnp.abs(display).max()
        scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], c=colors, edgecolor='black', cmap=cmap)
        cbar = fig.colorbar(scatter, ax=ax)
    else:
        # Color by tags if no show_state is given
        unique_tags = list(set(orb.tag for orb in orbs))
        color_map = {tag: plt.cm.get_cmap('tab10')(i / len(unique_tags)) for i, tag in enumerate(unique_tags)}
        for tag, positions in tags_to_pos.items():
            positions = jnp.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], label=tag, color=color_map[tag], edgecolor='white', alpha=0.7)
        plt.legend(title='Orbital Tags')

    # Optionally annotate points with their indexes
    if show_index:
        for orb in orbs:
            pos = orb.position
            idx = orbs.index(orb)
            ax.annotate(str(idx), (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Finalize plot settings
    plt.title('Orbital Positions in the xy-plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.grid(True)
    ax.axis('equal')

@_plot_wrapper
def show_3d(orbs, show_tags=None, show_index=False, display = None):
    """
    Generates a 3D scatter plot representing the positions of orbitals in 3D space,
    filtered by specified tags. Optionally colors points by eigenvector amplitudes.

    Parameters:
    - orbs (list): List of orbital objects, each with attributes 'tag', 'position', and 'eigenvectors'.
    - show_tags (list of str, optional): Tags used to filter orbitals for display.
    - show_index (bool): If True, indexes of the orbitals will be shown on the plot.
    - display: N-element array to display

    Returns:
    - None: A 3D scatter plot is displayed.
    """
    
    # Determine which tags to display
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    # Prepare data structures for plotting
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for orb in orbs:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(orbs.index(orb))

    # Prepare 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if display is not None:
        cmap = plt.cm.viridis
        colors = jnp.abs(display) / jnp.abs(display).max()
        scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], [orb.position[2] for orb in orbs], c=colors, edgecolor='black', cmap=cmap, depthshade=True)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Eigenvector Magnitude')
    else:
        # Color by tags if no show_state is given
        unique_tags = list(set(orb.tag for orb in orbs))
        color_map = {tag: plt.cm.get_cmap('tab10')(i / len(unique_tags)) for i, tag in enumerate(unique_tags)}
        for tag, positions in tags_to_pos.items():
            positions = jnp.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], label=tag, color=color_map[tag], edgecolor='white', alpha=0.7)
        plt.legend(title='Orbital Tags')

    # Optionally annotate points with their indexes
    if show_index:
        for orb in orbs:
            pos = orb.position
            idx = orbs.index(orb)
            ax.text(pos[0], pos[1], pos[2], str(idx), color='black', size=10)

    # Finalize plot settings
    ax.set_title('Orbital Positions in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)

@_plot_wrapper
def show_energies(orbs):
    """Depicts the energy and occupation landscape of a stack (energies are plotted on the y-axis ordered by size)

    - `stack`: stack object
    """
    fig, ax = plt.subplots(1, 1)
    plt.colorbar(
        ax.scatter(
            jnp.arange(orbs.energies.size),
            orbs.energies,
            c=jnp.diag(orbs.electrons * orbs.initial_density_matrix_e),
        ),
        label="ground state occupation",
    )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")


@_plot_wrapper
def show_expectation_value_time(
    orbs,
    expectation_value,
    time: jax.Array = None,
    indicate_eigenstate = True,
    ylabel = None,
    thresh: float = 1e-2,
):
    """Depicts an expectation value as a function of time.

    - `expectation_value`: TxN array, where T => time
    - `time`: time axis
    - `indicate_eigenstate`: whether to associate the i-th energy eigenstate to the i-th column of expectation_value
    - `thresh`: plotting threshold.  o_t is plotted if max(o_t) - min(o_t) > thresh
    """
    time = time if time is not None else jnp.arange(expectation_values.shape[0])
    fig, ax = plt.subplots(1, 1)
    for idx in jnp.nonzero(
        jnp.abs(jnp.amax(expectation_value, axis=0) - jnp.amin(expectation_value, axis=0)) > thresh
    )[0]:
        ax.plot(time, expectation_value[:, idx], label=f"{float(orbs.energies[idx]):2.2f} eV")

    time_label = "time steps" if time.dtype == int else r"time [$\hbar$/eV]"
    ax.set_xlabel(time_label)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if indicate_eigenstate == True:
        plt.legend()


@_plot_wrapper
def show_induced_field(orbs, x=None, y=None, z=None, component = 0, density_matrix=None):
    """Displays the normalized logarithm of the absolute value of the induced field in 2D

    - `x`: 
    - `y`: 
    - `z`: 
    - `component`: field component to display
    - `density_matrix` : if not given, initial density_matrix 
    """

    density_matrix = (
        density_matrix if density_matrix is not None else orbs.initial_density_matrix
    )
    charge = density_matrix.diag().real

    x = jnp.linspace(-5, 5, 100)
    y = jnp.linspace(-5, 5, 100)
    z = jnp.linspace(-1, 1, 10)
    X, Y, Z = jnp.meshgrid(x, y, z)
    positions = jnp.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    induced_field = get_induced_electric_field(
        get_coulomb_field_to_from(orbs.positions, positions), charge
    )
    E_induced_abs_rescaled = jnp.log(jnp.abs(14.39 * induced_field))

    fig, ax = plt.subplots(1, 1)
    fig.colorbar(
        ax.contourf(
            first, second, E_induced_abs_rescaled[:, component].reshape(first.shape)
        ),
        label=r"$\log(|E|/|E_0|)$",
    )

    ax.scatter(*zip(*orbs.positions[:, plane_indices[plane][:2]]), s=16)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])

def _display_lattice_cut(positions, selected_positions, polygon = None):
    fig, ax = plt.subplots(1, 1)
    if polygon is not None:
        patch = plt.Polygon(
            polygon[:-1], edgecolor="orange", facecolor="none", linewidth=2
        )
        ax.add_patch(patch)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="datalim")
    plt.grid(True)
    plt.scatter(x=positions[:, 0], y=positions[:, 1])
    plt.scatter(x=selected_positions[:, 0], y=selected_positions[:, 1])
    plt.axis("equal")
    plt.show()
