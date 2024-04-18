import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import wraps

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
def show_2d(
        orbs,
        show_selected_tags = None,
        show_hilbert_space_index = False
):
    """Shows a 2D scatter plot in the xy-plane of selected orbitals. Selections are made by a list 
    of tags.

    - `show_selected_tags` : a list of strings 
    """
    show_selected_tags = set((x.tag for x in orbs)) if show_selected_tags is None else set(show_selected_tags)
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for i, orb in enumerate(orbs):
        if orb.tag in show_selected_tags:
            tags_to_pos[ orb.tag ].append( orb.position )
            tags_to_idxs[ orb.tag ].append( i )
        
    fig, ax = plt.subplots(1, 1)
    for tag, positions in tags_to_pos.items():
        positions = jnp.array(positions)
        plt.scatter( x = positions[:,0], y = positions[:,1], label = tag )
        if not show_hilbert_space_index:
            continue
        for i, idx in enumerate(tags_to_idxs[tag]):
            ax.annotate( str(idx), ( positions[i, 0], positions[i, 1], ),)            
    plt.legend()
    ax.axis("equal")
    
@_plot_wrapper
def show_charge_distribution_3d(orbs, density_matrix = None):
    """Displays the ground state charge distribution of the stack in 3D

    - `stack`: stack object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    charge = orbs.get_charge( density_matrix )
    sp = ax.scatter(*zip(*orbs.positions[:, :2]), zs=orbs.positions[:, 2], c=charge)
    plt.colorbar(sp)


@_plot_wrapper
def show_charge_distribution_2d(orbs, density_matrix = None, plane: str = "xy"):
    """Displays the ground state charge distribution of the stack in 2D

    - `stack`: object representing system state
    - `plane`: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'
    """
    indices = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
    fig, ax = plt.subplots(1, 1)
    charge = orbs.get_charge( density_matrix )
    sp = ax.scatter(*zip(*orbs.positions[:, indices[plane]]), c=charge)
    ax.axis("equal")
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    plt.colorbar(sp)


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
            c=jnp.diag(orbs.electrons * orbs.initial_density_matrix),
        ),
        label="ground state occupation",
    )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")


@_plot_wrapper
def show_energy_occupations(
    orbs,
    time: jax.Array,
    density_matrices_or_solution,
    thresh: float = 1e-2,
):
    """Depicts energy occupations as a function of time.

    - `stack`: a stack object
    - `occupations`: list of energy occupations (complex arrays). The occupation at timestep n is given by `occupations[n]`.
    - `time`: time axis
    - `thresh`: plotting threshold. an occupation time series o_t is selected for plotting if it outgrows/outshrinks this bound. More exactly: o_t is plotted if max(o_t) - min(o_t) > thresh
    """
    if not isinstance( density_matrices_or_solution, jax.Array ):
        density_matrices_or_solution = density_matrices_or_solution.ys
    occupations = jnp.diagonal( density_matrices_or_solution, axis1 = -1, axis2 = -2).real
    fig, ax = plt.subplots(1, 1)
    for idx in jnp.nonzero(
        jnp.abs(jnp.amax(occupations, axis=0) - jnp.amin(occupations, axis=0)) > thresh
    )[0]:
        ax.plot(time, occupations[:, idx], label=f"{float(orbs.energies[idx]):2.2f} eV")
    ax.set_xlabel(r"time [$\hbar$/eV]")
    ax.set_ylabel("occupation of eigenstate")
    plt.legend()



# FIXME: also broken, is this actually needed?
@_plot_wrapper
def show_electric_field_space(
    first: jax.Array,
    second: jax.Array,
    plane: str,
    time: jax.Array,
    field_func,
    args: dict,
    component: int = 0,
    flag: int = 0,
):
    """Shows the external electric field on a spatial grid at a fixed point in time.

    - `first`: grid coordinates. get passed directly as meshgrid(frist, second).
    - `second`: grid coordinates. get passed directly as meshgrid(frist, second).
    - `plane`: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'. E.g. 'xy' means: make a plot in xy-plane and use "first"-parameter as x-axis, "second"-parameter as y-axis
    - `time`: time to evalute the field at
    - `field_func`: a function taking in parameters as given by args and an additional argument "positions" that produces a closure that gives the electric field as function of time
    - `args`: arguments to field_func as a dictionary, The "positions"-argument must be dropped.
    - `component`: 0 => plot x, 1 => plot y, 2 => plot z
    - `flag`: 0 => plot real, 1 => plot imag, 2 => plot abs
    """
    plane_indices = {
        "xy": jnp.array([0, 1, 2]),
        "xz": jnp.array([0, 2, 1]),
        "yz": jnp.array([2, 0, 1]),
    }
    funcs = [
        lambda field, t: field(t).real,
        lambda field, t: field(t).imag,
        lambda field, t: jnp.abs(field(t)),
    ]

    labels = ["Re(E)", "Im(E)", "|E|"]
    first, second = jnp.meshgrid(first, second)
    dim = first.size
    pos = jnp.concatenate(
        (
            jnp.stack((first, second), axis=2).reshape(dim, 2),
            jnp.expand_dims(jnp.zeros(dim), 1),
        ),
        axis=1,
    )[:, plane_indices[plane]]
    fig, ax = plt.subplots(1, 1)
    fig.colorbar(
        ax.contourf(
            first,
            second,
            funcs[flag](field_func(**args, positions=pos), time)[component].reshape(
                first.shape
            ),
        ),
        label=labels[flag],
    )
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])

# FIXME: also broken, is this actually needed?
@_plot_wrapper
def show_electric_field_time(time: jax.Array, field: jax.Array, flag: int = 0):
    """Shows the external electric field with its (x,y,z)-components as a function of time at a fixed spatial point.

    - `time`: array of points in time for field evaluation
    - `field`: output of an electric field function
    - `flag`: 0 => plot real, 1 => plot imag, 2 => plot abs
    """
    fig, ax = plt.subplots(1, 1)
    funcs = [
        lambda x: x.real,
        lambda x: x.imag,
        lambda x: jnp.abs(x),
    ]
    labels = ["Re(E)", "Im(E)", "|E|"]
    ax.plot(time, funcs[flag](jnp.array([jnp.squeeze(field(t)) for t in time])))
    ax.set_xlabel(r"time [$\hbar$/eV]")
    ax.set_ylabel(labels[flag])

@_plot_wrapper
def show_induced_field_at(
        orbs,
        positions,
        x = None,
        y = None,
        z = None,
        density_matrix = None
):
    """Displays the normalized logarithm of the absolute value of the induced field in 2D

    - `rho`: density matrix
    - `electrons`: number of electrons
    - `eigenvectors`: eigenvectors of the corresponding stack (as stored in a stack object)
    - `positions`: positions of the orbitals in the stack
    - `first`: grid coordinates. get passed directly as meshgrid(frist, second).
    - `second`: grid coordinates. get passed directly as meshgrid(frist, second).
    - `plane`: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'. E.g. 'xy' means: make a plot in xy-plane and use "first"-parameter as x-axis, "second"-parameter as y-axis
    - `component`: 0 => plot x, 1 => plot y, 2 => plot z
    - `norm` : constant to normalize the field
    - `plot_stack`: if True, add a scatter plot indicating the positions of the orbitals in the stack
    """
    
    density_matrix = density_matrix if density_matrix is not None else orbs.density_matrix
    charge = density_matrix.diag().real

    x = jnp.linspace(-5, 5, 100)
    y = jnp.linspace(-5, 5, 100)
    z = jnp.linspace(-1, 1, 10)
    X, Y, Z = jnp.meshgrid(x, y, z)
    positions = jnp.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    
    induced_field = get_induced_electric_field( get_coulomb_field_to_from(orbs.positions, positions), charge)    
    E_induced_abs_rescaled = jnp.log( jnp.abs(14.39 * induced_field) )
    
    fig, ax = plt.subplots(1, 1)
    fig.colorbar(
        ax.contourf(first, second, E_induced_abs_rescaled[:, component].reshape(first.shape)),
        label=r"$\log(|E|/|E_0|)$",
    )
    
    ax.scatter(*zip(*orbs.positions[:, plane_indices[plane][:2]]), s=16)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])

# TODO: this is also not nice
def _display_lattice_cut( polygon_vertices, positions, selected_positions ):
    fig, ax = plt.subplots()
    patch = plt.Polygon(polygon_vertices[:-1], edgecolor='orange', facecolor='none', linewidth=2)    
    ax.add_patch(patch)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='datalim')
    plt.grid(True)
    plt.scatter( x = positions[:,0], y = positions[:,1] )
    plt.scatter( x = selected_positions[:,0], y = selected_positions[:,1])    
    plt.axis('equal')
    plt.show()
