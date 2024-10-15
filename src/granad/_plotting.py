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
    Generates a 2D scatter plot representing the positions of orbitals in the xy-plane, with optional filtering, coloring, and sizing.

    Parameters:
        `orbs` (list): List of orbital objects, each containing attributes such as 'tag' (for labeling) and 'position' (xy-coordinates).
        `show_tags` (list of str, optional): Filters the orbitals to display based on their tags. Only orbitals with matching tags will be shown. If `None`, all orbitals are displayed.
        `show_index` (bool, optional): If `True`, displays the index of each orbital next to its corresponding point on the plot.
        `display` (array-like, optional): Data used to color and scale the points (e.g., eigenvector amplitudes). Each value corresponds to an orbital.
        `scale` (bool, optional): If `True`, the values in `display` are normalized and their absolute values are used.
        `cmap` (optional): Colormap used for the scatter plot when `display` is provided. If `None`, a default colormap (`bwr`) is used.
        `circle_scale` (float, optional): A scaling factor for the size of the scatter plot points. Larger values result in larger circles. Default is 1000.
        `title` (str, optional): Custom title for the plot. If `None`, the default title "Orbital positions in the xy-plane" is used.

    Notes:
        If `display` is provided, the points are colored and sized according to the values in the `display` array, and a color bar is added to the plot.
        If `show_index` is `True`, the indices of the orbitals are annotated next to their corresponding points.
        The plot is automatically adjusted to ensure equal scaling of the axes, and grid lines are displayed.
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
        ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], color='black', s=10, marker='o')
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
    Generates a 3D scatter plot representing the positions of orbitals in 3D space, with optional filtering, coloring, and sizing.

    Parameters:
        `orbs` (list): List of orbital objects, each containing attributes such as 'tag' (for labeling) and 'position' (3D coordinates).
        `show_tags` (list of str, optional): Filters the orbitals to display based on their tags. Only orbitals with matching tags will be shown. If `None`, all orbitals are displayed.
        `show_index` (bool, optional): If `True`, displays the index of each orbital next to its corresponding point on the plot.
        `display` (array-like, optional): Data used to color and scale the points (e.g., eigenvector amplitudes). Each value corresponds to an orbital.
        `scale` (bool, optional): If `True`, the values in `display` are normalized and their absolute values are used.
        `cmap` (optional): Colormap used for the scatter plot when `display` is provided. If `None`, a default colormap (`bwr`) is used.
        `circle_scale` (float, optional): A scaling factor for the size of the scatter plot points. Larger values result in larger circles. Default is 1000.
        `title` (str, optional): Custom title for the plot. If `None`, the default title "Orbital positions in 3D" is used.

    Notes:
        If `display` is provided, the points are colored and sized according to the values in the `display` array, and a color bar is added to the plot.
        If `show_index` is `True`, the indices of the orbitals are annotated next to their corresponding points.
        The plot is automatically adjusted to display grid lines and 3D axes labels for X, Y, and Z.
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
    """
    Depicts the energy and occupation landscape of a stack, with energies plotted on the y-axis and eigenstates ordered by size on the x-axis.

    Parameters:
        `orbs`: An object containing the orbital data, including energies, electron counts, and initial density matrix.

    Notes:
        The scatter plot displays the eigenstate number on the x-axis and the corresponding energy (in eV) on the y-axis.
        The color of each point represents the initial state occupation, calculated as the product of the electron count and the initial density matrix diagonal element for each state.
        A color bar is added to indicate the magnitude of the initial state occupation for each eigenstate.
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
    """
    Visualizes the evolution of an expectation value over time or frequency, based on the given simulation results.

    Parameters:
        `orbs`: Not typically required in most use cases, as this function is generally attached to a 'flake' object (e.g., `flake.show_res`).
        `res`: A result object containing the simulation data, including the output values and corresponding time or frequency axis.
        `plot_only` (jax.Array, optional): Indices of specific components to be plotted. If not provided, all components will be plotted.
        `plot_labels` (list[str], optional): Labels for each plotted quantity. If not provided, no labels will be added.
        `show_illumination` (bool, optional): Whether to include illumination data in the plot. If `True`, illumination components are displayed.
        `omega_max` (optional): Upper bound for the frequency range, used when plotting in the frequency domain.
        `omega_min` (optional): Lower bound for the frequency range, used when plotting in the frequency domain.

    Notes:
        The function adapts automatically to display either time-dependent or frequency-dependent results based on the presence of `omega_max` and `omega_min`.
        If `show_illumination` is enabled, the function plots the illumination components (`x`, `y`, `z`) as additional curves.
        The x-axis label changes to represent time or frequency, depending on the mode of operation.
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
    """
    Displays a 2D plot of the normalized logarithm of the absolute value of the induced field, for a given field component.

    Parameters:
        `orbs`: An object containing the orbital data and field information.
        `x` (array-like): x-coordinates for the 2D grid on which the field is evaluated.
        `y` (array-like): y-coordinates for the 2D grid on which the field is evaluated.
        `z` (float): z-coordinate slice at which the field is evaluated in the xy-plane.
        `component` (int, optional): The field component to display (default is 0). Represents the direction (e.g., x, y, or z) of the field.
        `density_matrix` (optional): The density matrix used to calculate the induced field. If not provided, the initial density matrix will be used.

    Note:
        The plot visualizes the induced field's magnitude using a logarithmic scale for better representation of variations in field strength.
        The field is normalized before applying the logarithm, ensuring that relative differences in field strength are emphasized.
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
