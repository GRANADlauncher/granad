from collections import defaultdict
from functools import wraps

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad._numerics import get_coulomb_field_to_from

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
def show_2d(orbs, show_tags=None, show_index=False, display = None, scale = False, cmap = None, circle_scale : float = 1e3, title = None):
    """
    Generates a 2D scatter plot representing the positions of orbitals in the xy-plane,
    filtered by specified tags. Optionally colors and sizes points by, e.g., eigenvector
    amplitudes.

    Parameters:
    - orbs (list): List of orbital objects, each with attributes 'tag' and 'position'.
    - show_tags (list of str, optional): Tags used to filter orbitals for display.
    - show_index (bool): If True, indexes of the orbitals will be shown on the plot.
    - display: N-element array to display
    - circle_size (float): larger values mean larger circles 
    - title: title of the plot

    Returns:
    - None: A 2D scatter plot is displayed.
    """

    # decider whether to take abs val and normalize 
    def scale_vals( vals ):
        return jnp.abs(vals) / jnp.abs(vals).max() if scale else vals
    
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
        cmap = plt.cm.bwr if cmap is None else cmap
        colors = scale_vals(display)
        scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], c=colors, edgecolor='black', cmap=cmap, s = circle_scale*jnp.abs(display) )
        scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], color='black', s=10, marker='o')
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
        for orb in [orb for orb in orbs if orb.tag in show_tags]:
            pos = orb.position
            idx = orbs.index(orb)
            ax.annotate(str(idx), (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Finalize plot settings
    plt.title('Orbital positions in the xy-plane' if title is None else title)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.grid(True)
    ax.axis('equal')

@_plot_wrapper
def show_3d(orbs, show_tags=None, show_index=False, display = None, scale = False, cmap = None, circle_scale : float = 1e3, title = None):
    """
    Generates a 3D scatter plot representing the positions of orbitals in 3D space,
    filtered by specified tags. Optionally colors and sizes points by, e.g., eigenvector
    amplitudes.

    Parameters:
    - orbs (list): List of orbital objects, each with attributes 'tag', 'position', and 'eigenvectors'.
    - show_tags (list of str, optional): Tags used to filter orbitals for display.
    - show_index (bool): If True, indexes of the orbitals will be shown on the plot.
    - display: N-element array to display
    - circle_scale: larger values means larger circles
    - title: title of the plot

    Returns:
    - None: A 3D scatter plot is displayed.
    """
    # decider whether to take abs val and normalize 
    def scale_vals( vals ):
        return jnp.abs(vals) / jnp.abs(vals).max() if scale else vals

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
        cmap = plt.cm.bwr if cmap is None else cmap
        colors = scale_vals( display )
        scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], [orb.position[2] for orb in orbs], c=colors, edgecolor='black', cmap=cmap, depthshade=True, s = circle_scale*jnp.abs(display))
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
        for orb in [orb for orb in orbs if orb.tag in show_tags]:
            pos = orb.position
            idx = orbs.index(orb)
            ax.text(pos[0], pos[1], pos[2], str(idx), color='black', size=10)

    # Finalize plot settings
    ax.set_title('Orbital positions in 3D' if title is not None else title)
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
        label="initial state occupation",
    )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")


@_plot_wrapper
def show_res(
    orbs,
    res,
    plot_only : jax.Array = None,
    plot_labels : list[str] = None,
    show_illumination = False,
    omega_max = None,
    omega_min = None,
):
    """Depicts an expectation value as a function of time.


    - `res`: result object
    - `plot_only`: only these indices will be plotted
    - `plot_legend`: names associated to the indexed quantities
    """
    def _show( obs, name ):
        ax.plot(x_axis, obs, label = name)
    
    fig, ax = plt.subplots(1, 1)    
    ax.set_xlabel(r"time [$\hbar$/eV]")
    plot_obs = res.output
    illu = res.td_illumination
    x_axis = res.time_axis
    cart_list = ["x", "y", "z"]
    
    
    if omega_max is not None and omega_min is not None:
        plot_obs = res.ft_output( omega_max, omega_min )
        x_axis, illu = res.ft_illumination( omega_max, omega_min )
        ax.set_xlabel(r"$\omega$ [$\hbar$ eV]")
    
    for obs in plot_obs:
        obs = obs if plot_only is None else obs[:, plot_only]
        for i, obs_flat in enumerate(obs.T):
            label = '' if plot_labels is None else plot_labels[i]
            _show( obs_flat, label )
        if show_illumination == True:
            for component, illu_flat in enumerate(illu.T):            
                _show(illu_flat, f'illumination_{cart_list[component]}')
            
    plt.legend()


@_plot_wrapper
def show_induced_field(orbs, x, y, z, component = 0, density_matrix=None):
    """Displays the normalized logarithm of the absolute value of the induced field in 2D

    - `x`: 
    - `y`: 
    - `component`: field component to display
    - `density_matrix` : if not given, initial density_matrix 
    """

    density_matrix = (
        density_matrix if density_matrix is not None else orbs.initial_density_matrix
    )
    charge = density_matrix.diagonal().real

    X, Y, Z = jnp.meshgrid(x, y, z)
    positions = jnp.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T    

    induced_field = jnp.einsum('jir,i->jr', get_coulomb_field_to_from(orbs.positions, positions, jnp.arange(len(positions))), charge)
    
    E_induced_abs_rescaled = jnp.log(jnp.abs(induced_field) / jnp.abs(induced_field).max() )

    fig, ax = plt.subplots(1, 1)
    fig.colorbar(
        ax.contourf(
            X[:, :, 0], Y[:, :, 0], E_induced_abs_rescaled[:, component].reshape(X[:, :, 0].shape),
            cmap = plt.cm.bwr
        ),
        label=r"$\log(|E|/|E_0|)$",
    )

    ax.scatter(*zip(*orbs.positions[:, :2]), color = 'black', s=8)
    ax.axis('equal')


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
