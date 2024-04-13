    # I REALLY HATE THE PLOTTING, this should be moved to a different file or decided via getattr 
    @_plot_wrapper
    @recomputes
    def show_eigenstate3D(
        orbs,
        show_state: int = 0,
        show_orbitals: list[str] = None,
        indicate_size: bool = True,
        color_orbitals: bool = True,
        annotate_hilbert: bool = True,
    ):
        """Shows a 3D scatter plot of how selected orbitals in a stack contribute to an eigenstate.
        In the plot, orbitals are annotated with a color. The color corresponds either to the contribution to the selected eigenstate or to the type of the orbital.
        Optionally, orbitals can be annotated with a number corresponding to the hilbert space index.

        - `stack`: object representing system state
        - `show_state`: eigenstate index to show. (0 => eigenstate with lowest energy)
        - `show_orbitals`: orbitals to show. if None, all orbitals are shown.
        - `indicate_size`: if True, bigger points are orbitals contributing more strongly to the selected eigenstate.
        - `color_orbitals`: if True, identical orbitals are colored identically and a legend is displayed listing all the different orbital types. if False, the color corresponds to the contribution to the sublattice.
        - `annotate_hilbert`: if True, the orbitals are annotated with a number corresponding to the hilbert space index.
        """
        stack = orb.stack
        show_orbitals = stack.unique_ids if show_orbitals is None else show_orbitals
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for orb in show_orbitals:
            idxs = jnp.nonzero(stack.ids == stack.unique_ids.index(orb))[0]
            ax.scatter(
                *zip(*stack.positions[idxs, :2]),
                zs=stack.positions[idxs, 2],
                s=(
                    6000 * jnp.abs(stack.eigenvectors[idxs, show_state])
                    if indicate_size
                    else 40
                ),
                c=stack.sublattice_ids[idxs] if not color_orbitals else None,
                label=orb,
            )
            if annotate_hilbert:
                for idx in idxs:
                    ax.text(*stack.positions[idx, :], str(idx), "x")
        if color_orbitals:
            plt.legend()

    @_plot_wrapper
    @recomputes
    def show_eigenstate2D(
        self,
        plane: str = "xy",
        show_state: int = 0,
        show_orbitals: bool = None,
        indicate_size: bool = True,
        color_orbitals: bool = True,
        annotate_hilbert: bool = True,
    ):
        """Shows a 2D scatter plot of how selected orbitals in a stack contribute to an eigenstate.
        In the plot, orbitals are annotated with a color. The color corresponds either to the contribution to the selected eigenstate or to the type of the orbital.
        Optionally, orbitals can be annotated with a number corresponding to the hilbert space index.

        - `stack`: object representing system state
        - `plane`: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'.
        - `show_state`: eigenstate index to show. (0 => eigenstate with lowest energy)
        - `show_orbitals`: list of strings. orbitals to show. if None, all orbitals are shown.
        - `indicate_size`: if True, bigger points are orbitals contributing more strongly to the selected eigenstate.
        - `color_orbitals`: if True, identical orbitals are colored identically and a legend is displayed listing all the different orbital types. if False, the color corresponds to the sublattice.
        - `annotate_hilbert`: if True, the orbitals are annotated with a number corresponding to the hilbert space index.
        """
        stack = orbs.stack
        indices = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
        show_orbitals = stack.unique_ids if show_orbitals is None else show_orbitals
        fig, ax = plt.subplots(1, 1)
        for orb in show_orbitals:
            idxs = jnp.nonzero(stack.ids == stack.unique_ids.index(orb))[0]
            ax.scatter(
                *zip(*stack.positions[idxs, :][:, indices[plane]]),
                s=(
                    6000 * jnp.abs(stack.eigenvectors[idxs, show_state])
                    if indicate_size
                    else 40
                ),
                c=stack.sublattice_ids[idxs] if not color_orbitals else None,
                label=orb,
            )
            if annotate_hilbert:
                for idx in idxs:
                    ax.annotate(
                        str(idx),
                        (
                            stack.positions[idx, indices[plane][0]],
                            stack.positions[idx, indices[plane][1]],
                        ),
                    )
        if color_orbitals:
            plt.legend()
        ax.axis("equal")

    @_plot_wrapper
    @recomputes
    def show_charge_distribution3D(orbs):
        """Displays the ground state charge distribution of the stack in 3D

        - `stack`: stack object
        """
        stack = orbs.stack
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        charge = stack.electrons * jnp.diag(
            stack.eigenvectors @ stack.rho_0.real @ stack.eigenvectors.conj().T
        )
        sp = ax.scatter(*zip(*stack.positions[:, :2]), zs=stack.positions[:, 2], c=charge)
        plt.colorbar(sp)


    @_plot_wrapper
    @recomputes
    def show_charge_distribution2D(orbs, plane: str = "xy"):
        """Displays the ground state charge distribution of the stack in 2D

        - `stack`: object representing system state
        - `plane`: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'
        """
        stack = orbs.stack
        indices = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
        fig, ax = plt.subplots(1, 1)
        charge = stack.electrons * jnp.diag(
            stack.eigenvectors @ stack.rho_0.real @ stack.eigenvectors.conj().T
        )
        sp = ax.scatter(*zip(*stack.positions[:, indices[plane]]), c=charge)
        ax.axis("equal")
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        plt.colorbar(sp)


    @_plot_wrapper
    @recomputes
    def show_energies(orbs: OrbitalList):
        """Depicts the energy and occupation landscape of a stack (energies are plotted on the y-axis ordered by size)

        - `stack`: stack object
        """
        stack = orbs.stack
        fig, ax = plt.subplots(1, 1)
        plt.colorbar(
            ax.scatter(
                jnp.arange(stack.energies.size),
                stack.energies,
                c=stack.electrons * jnp.diag(stack.rho_0.real),
            ),
            label="ground state occupation",
        )
        ax.set_xlabel("eigenstate number")
        ax.set_ylabel("energy (eV)")


    @_plot_wrapper
    @recomputes
    def show_energy_occupations(
        orbs,
        occupations: list[jax.Array],
        time: jax.Array,
        thresh: float = 1e-2,
    ):
        """Depicts energy occupations as a function of time.

        - `stack`: a stack object
        - `occupations`: list of energy occupations (complex arrays). The occupation at timestep n is given by `occupations[n]`.
        - `time`: time axis
        - `thresh`: plotting threshold. an occupation time series o_t is selected for plotting if it outgrows/outshrinks this bound. More exactly: o_t is plotted if max(o_t) - min(o_t) > thresh
        """
        stack = orbs.stack
        fig, ax = plt.subplots(1, 1)
        occ = jnp.array([stack.electrons * r.real for r in occupations])
        for idx in jnp.nonzero(
            jnp.abs(jnp.amax(occ, axis=0) - jnp.amin(occ, axis=0)) > thresh
        )[0]:
            ax.plot(time, occ[:, idx], label=f"{float(stack.energies[idx]):2.2f} eV")
        ax.set_xlabel(r"time [$\hbar$/eV]")
        ax.set_ylabel("occupation of eigenstate")
        plt.legend()

        
## plotting
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
def show_electric_field_space(
    first: jax.Array,
    second: jax.Array,
    plane: str,
    time: jax.Array,
    field_func: Callable[[float], jax.Array],
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
def show_induced_field(
    rho: jax.Array,
    electrons: int,
    eigenvectors: jax.Array,
    positions: jax.Array,
    first: jax.Array,
    second: jax.Array,
    plane: str = "xy",
    component: int = 0,
    norm: int = 1,
    plot_stack: bool = True,
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
    plane_indices = {
        "xy": jnp.array([0, 1, 2]),
        "xz": jnp.array([0, 2, 1]),
        "yz": jnp.array([2, 0, 1]),
    }
    first, second = jnp.meshgrid(first, second)
    dim = first.size
    vec_r = jnp.ones((dim, 3, positions.shape[0])) * jnp.swapaxes(
        positions, 0, 1
    ) - jnp.expand_dims(
        jnp.concatenate(
            (
                jnp.stack((first, second), axis=2).reshape(dim, 2),
                jnp.expand_dims(jnp.zeros(dim), 1),
            ),
            axis=1,
        )[:, plane_indices[plane]],
        2,
    )
    r_point_charge = jnp.nan_to_num(
        vec_r / jnp.expand_dims(jnp.linalg.norm(vec_r, axis=1) ** 3, 1),
        posinf=0.0,
        neginf=0.0,
    )
    charge = electrons * jnp.diag(eigenvectors @ rho.real @ eigenvectors.conj().T)
    E_induced = jnp.log(
        jnp.abs(14.39 * jnp.sum(r_point_charge * charge.real, axis=2)) / norm
    )
    fig, ax = plt.subplots(1, 1)
    fig.colorbar(
        ax.contourf(first, second, E_induced[:, component].reshape(first.shape)),
        label=r"$\log(|E|/|E_0|)$",
    )
    if plot_stack:
        ax.scatter(*zip(*positions[:, plane_indices[plane][:2]]), s=16)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])

    
