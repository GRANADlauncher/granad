"""GRANAD: GRAphene Nanoantennas with ADatoms 
"""

## imports for main computations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from functools import reduce
from typing import Callable, Union
import jax
from flax import struct
import jax as jax
from jax import Array, lax
import jax.numpy as jnp
import diffrax

## complex precision
from jax import config

config.update("jax_enable_x64", True)

## typing
from typing import NewType
from collections.abc import Iterable
from enum import Enum, auto
from functools import wraps

## TYPE ALIASES
FieldFunc = Callable[[float], Array]
DissipationFunc = Callable[[Array], Array]
InputDict = NewType("InputDict", dict[str, Callable[[float], complex]])
TransitionsDict = NewType(
    "TransitionsDict", Union[None, dict[tuple[str, str], list[float]]]
)


## CLASSES
@struct.dataclass
class Orbital:
    """An orbital.
    It is only used briefly during stack construction and immediately discarded after.

    :param orbital_id: Orbital ID, e.g. "pz_graphene".
    :param position: Orbital position like :math:`(x,y,z)`, in Ångström.
    :param occupation: number of electrons in the non-interacting orbital (:code:`1` or :code:`0`)
    """

    orbital_id: str
    position: jax.Array
    occupation: int = 1
    sublattice_id: int = 0


@struct.dataclass
class Stack:
    """A stack of orbitals.

    :param hamiltonian: TB-Hamiltonian of the system.
    :param coulomb: Coulomb matrix of the system.
    :param rho_0: Density matrix corresponding to the initial state. If a simulation is interrupted at time t, then it can be resumed using the density matrix at time t as rho_0.
    :param rho_stat: Density matrix corresponding to the final state.
    :param energies: Independent particle energies as given by the eigenvalues of the TB-hamiltonian.
    :param eigenvectors: Eigenvectors of the TB-Hamiltonian. These are used to transform between *site* and *energy* basis.
    :param positions: Positions of the orbitals, :math:`N \\times 3`-dimensional.
    :param unique_ids: Collection of unique strings. Each string identifies an orbital type occuring in the stack. Typically something like "pz_graphene".
    :param ids: The integer at the n-th position corresponds to the type of the n-th orbital via its index in unique_ids. For example, for unique_ids = ["A", "B"], one has unique_ids[0] = "A". Additionally, if ids[10] = 0, then the orbital at 10 is of type "A". This is useful if you want to identify quantities corresponding to orbitals. So, say unique_ids = ["adatom_level1", "adatom_level2", "carbon"] and you want to know the position of the adatom. Then you can just perform the ugly lookup stack.positions[ jnp.argwhere(ids == stack.unique_ids.index("adatom_level1")), :]. This is what the function convenience-position function does.
    :param eps: numerical precision threshold for identifiying degeneracies in the eigenenergies.
    :param homo: index of the homo, such that stack.energies[homo] is the energy of the ground state homo.
    :param electrons: total number of electrons in the system.
    :param from_state: the state from which electrons have been taken for the initial excited transition, the state translates to HOMO - from_state
    :param to_state: the state into which electrons haven been put for the initial excited transition, the state translates to HOMO + from_state
    :param beta: corresponds to :math:`\\frac{1}{k_B T}`
    """

    hamiltonian: Array
    coulomb: Array
    rho_0: Array
    rho_stat: Array
    energies: Array
    eigenvectors: Array
    positions: Array
    unique_ids: list[str]
    ids: Array
    sublattice_ids: Array
    eps: float
    homo: int
    electrons: int
    from_state: int
    to_state: int
    excited_electrons: int
    beta: float
    spin_degeneracy: int
    transitions: TransitionsDict


class LatticeType(Enum):
    CHAIN = auto()
    SSH = auto()
    HONEYCOMB = auto()


class LatticeEdge(Enum):
    NONE = auto()
    ARMCHAIR = auto()
    ZIGZAG = auto()


## GEOMETRY
# TODO: this is somewhat broken
@struct.dataclass
class Chain:
    """A chain.

    :param length: length of the chain
    """

    length: float

    def _layout(self, lattice_constant: float) -> Array:
        n = int(self.length / lattice_constant)
        return jnp.expand_dims(jnp.arange(2 * n), 1) % 2


@struct.dataclass
class Triangle:
    """An equilateral triangle.

    :param base: Base length of the triangle
    """

    base: float

    def _layout(self, lattice_constant: float) -> Array:
        n = int(self.base / lattice_constant)

        size = 2 * n + 1
        shape = np.zeros((size, n), dtype=bool)
        for i in range(n):
            shape[n + i, i::2] = True
            shape[n - i, i::2] = True
        return jnp.array(shape)


@struct.dataclass
class Hexagon:
    """A hexagon.

    :param side_length: Side length of the hexagon
    """

    side_length: float

    def _layout(self, lattice_constant: float) -> Array:
        n = int(self.side_length / lattice_constant)

        width = 2 * n + 1
        height = 4 * n - 2
        shape = np.zeros((height, width))

        for i in range(n):
            shape[i : (height - i) : 2, n + i] = True
            shape[i : (height - i) : 2, n - i] = True

        return jnp.array(shape)


@struct.dataclass
class Rhomboid:
    """A rhomboid.

    :param x: side length in x-diretion
    :param y: side length in y-diretion
    """

    x: float
    y: float

    def _layout(self, lattice_constant: float) -> Array:
        x, y = int(self.x / lattice_constant), int(self.y / lattice_constant)
        shape = np.zeros((2 * (x + y), y), dtype=bool)
        for i in range(y):
            upper = 2 * x + i
            shape[i:upper:2, i] = True
        return jnp.array(shape)


@struct.dataclass
class Rectangle:
    """A rectangle.

    :param x: side length in x-diretion
    :param y: side length in y-diretion
    """

    x: float
    y: float

    def _layout(self, lattice_constant: float) -> Array:
        x, y = int(self.x / lattice_constant), int(self.y / lattice_constant)

        height = y * 2
        width = 2 * x - 1
        shape = np.zeros((height, width), dtype=bool)
        for i in range(height):
            shape[i, (i % 2) : width : 2] = True
        return jnp.array(shape)


class Lattice:
    """Monoatomic lattice cut.

    A triangular graphene lattice cut would be described as follows:

    .. code-block:: python

        graphene = granad.Lattice(
                   shape=triangle,
                   lattice_edge=granad.LatticeEdge.ARMCHAIR,
                   lattice_type=granad.LatticeType.HONEYCOMB,
                   lattice_constant=2.46,
                  )

    :param shape: a boolean array indicating the shape of the structure
    :param lattice_type: lattice type
    :param lattice_edge: edge type
    :param lattice_constant: lattice constant of the infinite lattice
    :param shift: spacial shift to apply to all coordinates
    """

    ## entries in this dict follow the scheme:
    ## (lattice_type, edge_type) : (confguration, scale, lattice basis, unit cell)
    ## the first entry is a redundant unit cell, which is copied to produce the correct edge type
    ## the the second entry refers to the scaling applied to a layout array
    ## the last two entries in the tuple refer to the infinite lattice

    _structures = {
        (LatticeType.HONEYCOMB, LatticeEdge.ARMCHAIR): (
            jnp.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [3 / 2, -jnp.sqrt(3) / 2, 0],
                    [1, -2 * jnp.sqrt(3) / 2, 0],
                    [0, -2 * jnp.sqrt(3) / 2, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                ]
            ),
            jnp.array([3 / 2, 3 / 2 * jnp.sqrt(3)]),
            jnp.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                ]
            ),
            jnp.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [3 / 2, -jnp.sqrt(3) / 2, 0],
                    [1, -2 * jnp.sqrt(3) / 2, 0],
                    [0, -2 * jnp.sqrt(3) / 2, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                ]
            ),
        ),
        (LatticeType.HONEYCOMB, LatticeEdge.ZIGZAG): (
            jnp.array(
                [
                    [-jnp.sqrt(3) / 2, 1 / 2, 0],
                    [0, 1, 0],
                    [jnp.sqrt(3) / 2, 1 / 2, 0],
                    [-jnp.sqrt(3) / 2, -1 / 2, 0],
                    [0, -1, 0],
                    [jnp.sqrt(3) / 2, -1 / 2, 0],
                ]
            ),
            jnp.array([1 / 2 * jnp.sqrt(3), 3 / 2]),
            jnp.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                ]
            ),
            jnp.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [3 / 2, -jnp.sqrt(3) / 2, 0],
                    [1, -2 * jnp.sqrt(3) / 2, 0],
                    [0, -2 * jnp.sqrt(3) / 2, 0],
                    [-1 / 2, -jnp.sqrt(3) / 2, 0],
                ]
            ),
        ),
        (LatticeType.CHAIN, LatticeEdge.NONE): (
            jnp.array([[0.0, 0, 0]]),
            jnp.array([0.5, 1.0]),
            jnp.array(
                [
                    [1.0, 0.0, 0.0],
                ]
            ),
            jnp.array(
                [
                    [0, 0, 0],
                ]
            ),
        ),
        (LatticeType.SSH, LatticeEdge.NONE): (
            jnp.array([[0.0, 0, 0], [0.3, 0, 0]]),
            jnp.array([0.5, 1.0]),
            jnp.array(
                [
                    [1.0, 0.0, 0.0],
                ]
            ),
            jnp.array([[0, 0, 0], [0.3, 0.0, 0.0]]),
        ),
    }

    def __init__(
        self,
        shape: Union[Chain, Triangle, Hexagon, Rhomboid, Rectangle],
        lattice_type: LatticeType,
        lattice_edge: LatticeEdge,
        lattice_constant: float,
        shift=jnp.zeros(3),
    ):
        """Initialization"""

        try:
            site_distance = self._site_distance(lattice_constant, lattice_type)
            self.configuration, self.scale, self.lattice_basis, self.unit_cell = [
                x * site_distance
                for x in self._structures[(lattice_type, lattice_edge)]
            ]
        except KeyError:
            raise Exception(
                f"Lattice {lattice_type} with edge {lattice_edge} is not supported!"
            )

        self.shift = shift
        self.shape = shape._layout(lattice_constant)

        if not (lattice_type, lattice_edge) in self._structures.keys():
            raise Exception(f"Lattice {lattice_type} not implemented!")

    @staticmethod
    def _site_distance(lattice_constant: float, lattice_type: LatticeType) -> float:
        if lattice_type == LatticeType.HONEYCOMB:
            return lattice_constant / jnp.sqrt(3)
        else:
            return lattice_constant

    def _orbitals(self, orbital_id: str, occupation: int = 1) -> list[Orbital]:
        """Returns a list of orbitals in a geometric arrangement."""

        shape, shift, configuration, scale = (
            self.shape,
            self.shift,
            self.configuration,
            self.scale,
        )

        # TODO: fix sublattice_id
        def add_orbitals(position: Array):
            return [
                Orbital(
                    position=jnp.array(
                        list(scale * position + displacement[:2] + shift[:2])
                        + [shift[2] + displacement[2]]
                    ),
                    orbital_id=orbital_id,
                    occupation=occupation,
                    sublattice_id=i % 2,
                )
                for i, displacement in enumerate(configuration)
            ]

        orbitals = reduce(
            lambda x, y: x + y, map(lambda pos: add_orbitals(pos), jnp.argwhere(shape))
        )
        return [
            orbitals[i]
            for i in jnp.unique(
                jnp.round(jnp.array([orb.position for orb in orbitals]), 8),
                axis=0,
                return_index=True,
            )[1]
        ]


@struct.dataclass
class Spot:
    """Single point in 3D space. Can represent isolated adatoms.

    :param position: point, e.g. :code:`[0.0, 1.0, 2.0]`
    """

    position: list[float]


@struct.dataclass
class DistanceCoupling:
    """Distance-dependent coupling.

    For example, a model direct channel repulsion of :math:`\\frac{1}{r}` between
    two orbitals identified by the strings :code:`"X"` and :code:`"Y"` can be modelled
    as follows:

    .. code-block:: python

       coulomb_repulsion = SpotCoupling(
       orbital_id1 = "X", # this id must match the id given to stack.add
       orbital_id2 = "Y", # this id must match the id given to stack.add
       coupling = lambda r : 1/r + 0j
       )


    :param orbital_id1: orbital id
    :param orbital_id2: orbital id
    :param coupling_function: coupling function, must return a **complex number**
    """

    orbital_id1: str
    orbital_id2: str
    coupling_function: Callable[[float], complex]


@struct.dataclass
class SpotCoupling:
    """Onsite coupling. Can hold the coupling between two orbitals at an adatom.

    For example, the onsite energy of a single adatom orbital identified by the string :code:`"A"` can be modelled
    as follows:

    .. code-block:: python

       onsite_adatom = SpotCoupling(
       orbital_id1 = "A", # this id must match the id given to stack.add
       orbital_id2 = "A",
       coupling = 0.0
       )

    :param orbital_id1: orbital id
    :param orbital_id2: orbital id
    :param coupling: coupling strength
    """

    orbital_id1: str
    orbital_id2: str
    coupling: float


@struct.dataclass
class LatticeCoupling:
    """Coupling in a lattice material.

    Constructor accepts a lattice object and a list of couplings.
    The i-th entry in the couplings-list determines the coupling to the i-th neighbor,
    i.e. :code:`couplings[0]` is the onsite coupling, :code:`couplings[2]` is the NNN-coupling.

    For example, the Coulomb coupling in spin-polarized graphene with an onsite strength
    of :code:`16.522`, nearest-neighbor interaction of :code:`8.64`, next-to-nearest neighbor
    interaction of :code:`5.333` and a classical coupling for orbitals further away can be modelled
    as follows:

    .. code-block:: python

       coulomb_graphene = LatticeCoupling(
       orbital_id1 = "pz", # this id must match the id given to stack.add
       orbital_id2 = "pz",
       lattice = graphene,
       couplings = [16.522, 8.64, 5.333],
       coupling_function = lambda d: 14.399 / d + 0j,
       )

    :param orbital_id1: orbital id
    :param orbital_id2: orbital id
    :param lattice: lattice material
    :param couplings: couplings
    :param coupling_function: function to apply to all neighbours not covered by couplings
    """

    orbital_id1: str
    orbital_id2: str
    lattice: Lattice
    couplings: list[float]
    coupling_function: Callable[[float], complex] = lambda x: 0j


@struct.dataclass
class LatticeSpotCoupling:
    """Determines coupling between a lattice material and an isolated spot.

    Constructor accepts a list of couplings.
    The i-th entry in the couplings-list determines the coupling between the isolated spot
    and its i-th neighbor in the lattice cut,
    i.e. :code:`couplings[0]` is the onsite coupling, :code:`couplings[2]` is the NNN-coupling.

    For example, the Coulomb coupling of a single adatom orbital by the string :code:`"A"` to
    a flake orbital identified by the string :code:`"pz"` with an onsite strength
    of :code:`16.522`, nearest-neighbor interaction of :code:`8.64`, next-to-nearest neighbor
    interaction of :code:`5.333` and a classical coupling for orbitals further away can be modelled
    as follows:

    .. code-block:: python

       coulomb_graphene = LatticeSpotCoupling(
       lattice_id = "pz", # this id must match the id given to stack.add
       spot_id = "A", # this id must match the id given to stack.add
       couplings = [16.522, 8.64, 5.333],
       coupling_function = lambda d: 14.399 / d + 0j,
       )

    :param lattice_id: bulk orbital id
    :param spot_id: isolated orbital id
    :param couplings: couplings list
    :param coupling_function: function to apply to all neighbours not covered by couplings
    """

    lattice_id: str
    spot_id: str
    couplings: list[float]
    coupling_function: Callable[[float], complex] = lambda x: 0j


class StackBuilder:
    """Incremental specification of a composite nanomaterial.

    Orbitals are added one by one and their TB-hopping and coulomb interactions specified.
    Once the specification is finished,
    the method :py:meth:`granad.numerics.StackBuilder.build_stack` is to be called to produce a
    :py:class:`granad.numerics.StackBuilder.Stack` object for further use in numerical simulations.
    """

    def __init__(self):
        """Initialization."""
        self.orbitals: list[Orbital] = []
        self.hopping: dict[tuple[str, str], Callable[[float], complex]] = {}
        self.coulomb: dict[tuple[str, str], Callable[[float], complex]] = {}

        self.spots: int = 0
        self.lattices: int = 0

    @property
    def orbital_ids(self):
        return list(set([orb.orbital_id for orb in self.orbitals]))

    def add(self, orbital_id: str, material: Union[Spot, Lattice], occupation: int = 1):
        """Adds new orbitals to the stack.

        :param orbital_id: new orbital to be added
        :param material: material, an isolated system or a lattice cut
        :param occupation: number of electrons carried by the orbital
        """

        if isinstance(material, Spot):
            # for a single site, add the orbital at that position
            self.orbitals.append(Orbital(orbital_id, material.position, occupation))

        elif isinstance(material, Lattice):
            # for a lattice, append a list of orbitals in the specified configuartion
            self.orbitals += material._orbitals(orbital_id, occupation)
        else:
            raise TypeError(f"Incorrect material type {material}!")

    def _set_coupling(
        self,
        coupling_dict: dict,
        coupling: Union[
            DistanceCoupling, SpotCoupling, LatticeCoupling, LatticeSpotCoupling
        ],
    ):
        if isinstance(coupling, DistanceCoupling):
            coupling_dict[(coupling.orbital_id1, coupling.orbital_id2)] = jax.jit(
                coupling.coupling_function
            )
        elif isinstance(coupling, SpotCoupling):
            coupling_dict[(coupling.orbital_id1, coupling.orbital_id2)] = jax.jit(
                lambda d: coupling.coupling + 0j
            )
        elif isinstance(coupling, LatticeCoupling):
            coupling_dict[(coupling.orbital_id1, coupling.orbital_id2)] = (
                self._lattice_coupling(coupling)
            )
        elif isinstance(coupling, LatticeSpotCoupling):
            lattice_pos = self.get_positions(coupling.lattice_id)
            spot_pos = self.get_positions(coupling.spot_id)
            coupling_dict[(coupling.lattice_id, coupling.spot_id)] = (
                self._lattice_spot_coupling(
                    coupling, lattice_pos=lattice_pos, spot_pos=spot_pos
                )
            )
        else:
            raise TypeError(f"Incorrect coupling type {coupling}!")

    def set_hopping(
        self,
        hopping: Union[
            DistanceCoupling, SpotCoupling, LatticeCoupling, LatticeSpotCoupling
        ],
    ):
        """Specifies the TB-hopping between two types of orbitals.

        :param hopping: hopping strength
        """

        self._set_coupling(self.hopping, hopping)

    def set_coulomb(
        self,
        coulomb: Union[
            DistanceCoupling, SpotCoupling, LatticeCoupling, LatticeSpotCoupling
        ],
    ):
        """Specifies the Coulomb interaction between two types of orbitals.

        :param coulomb: coulomb coupling
        """

        self._set_coupling(self.coulomb, coulomb)

    def get_positions(self, orbital_id: str = None) -> Array:
        """Gets the positions in the stack.

        :param orbital_id: If supplied, only get the positions of this orbital
        :returns: A :math:`N \\times 3`-array, where :math:`N` is the number of orbitals
        """

        if orbital_id is None:
            return jnp.array([x.position for x in self.orbitals])
        else:
            return jnp.array(
                [x.position for x in self.orbitals if x.orbital_id == orbital_id]
            )

    def show2D(self):
        """Plots all orbitals in 2D"""

        norm = Normalize(0, len(self.orbital_ids))

        fig, ax = plt.subplots(1, 1)
        for orb in self.orbital_ids:
            positions = self.get_positions(orb)
            color = norm(self.orbital_ids.index(orb)) * jnp.ones_like(positions[:, 0])
            ax.scatter(
                *zip(*positions[:, :2]),
                s=40,
                c=color,
                label=orb,
            )

        pos = self.get_positions()
        for i in range(len(self.orbitals)):
            ax.annotate(
                str(i),
                (
                    pos[i, 0],
                    pos[i, 1],
                ),
            )
        ax.axis("equal")
        plt.legend()
        plt.show()

    def show3D(self):
        """Plots all orbitals in 3D"""

        norm = Normalize(0, len(self.orbital_ids))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for orb in self.orbital_ids:
            positions = self.get_positions(orb)
            color = norm(self.orbital_ids.index(orb)) * jnp.ones_like(positions[:, 0])
            ax.scatter(
                *zip(*positions[:, :2]),
                zs=positions[:, 2],
                s=40,
                c=color,
                label=orb,
            )

        pos = self.get_positions()
        for i in range(len(self.orbitals)):
            ax.text(*pos[i, :], str(i), "x")

        plt.legend()
        plt.show()

    def _ensure_combinations(self):
        for name, coupling_dict in [
            ("hopping", self.hopping),
            ("coulomb", self.coulomb),
        ]:
            for id1 in self.orbital_ids:
                for id2 in self.orbital_ids:
                    if not (
                        (id1, id2) in coupling_dict.keys()
                        or (id2, id1) in coupling_dict.keys()
                    ):
                        raise Exception(
                            f"Not all {name} combinations specified, missing {(id1,id2)}!"
                        )

    @property
    def sublattice_ids(self):
        return [orb.sublattice_id for orb in self.orbitals]

    def get_stack(
        self,
        field_func: Callable[[float], Array] = lambda x: 0j,
        from_state: int = 0,
        to_state: int = 0,
        excited_electrons: int = 1,
        doping: int = 0,
        beta: float = jnp.inf,
        eps: float = 1e-5,
        spin_degenerate: bool = True,
        transitions: TransitionsDict = None,
        pingback: bool = True,
    ) -> Stack:
        """Get stack object for numerical simulations.

        :param field_func: position-dependent additive term for hamiltonian (e.g. electric field)
        :param from_state: optional, the state from which electrons to take for the initial excited transition, the state translates to HOMO - from_state
        :param to_state: optional, the state into which electrons to put for the initial excited transition, the state translates to HOMO + from_state
        :param doping: optional, how many doping electrons to throw into the stack
        :param beta: thermodynamic beta :math:`\\frac{1}{k_B T}`
        :param eps: optional, default 1e-5: numerical threshold. if two energies E1,E2 fullfill abs(E1-E2) < eps, they are considered degenerate
        :returns: stack object
        """

        self._ensure_combinations()

        stack = _stack(
            self.orbitals,
            self.hopping,
            self.coulomb,
            field_func,
            from_state,
            to_state,
            excited_electrons,
            doping,
            beta,
            jnp.array(self.sublattice_ids),
            eps,
            2.0 if spin_degenerate else 1.0,
            transitions,
        )
        if pingback:
            print(f"Built stack with {stack.electrons} electrons")

        return stack

    @staticmethod
    def _lattice_spot_coupling(
        lattice_spot_coupling: LatticeSpotCoupling,
        *,
        lattice_pos: Array,
        spot_pos: Array,
        tolerance: float = 1e-5,
    ):
        couplings = jnp.array([x + 0j for x in lattice_spot_coupling.couplings])

        distances = jnp.sort(
            jnp.unique(jnp.round(jnp.linalg.norm(lattice_pos - spot_pos, axis=1), 8))
        )[: len(couplings)]

        def inner(d):
            return lax.cond(
                jnp.min(jnp.abs(d - distances)) < tolerance,
                lambda x: couplings[jnp.argmin(jnp.abs(x - distances))],
                lattice_spot_coupling.coupling_function,
                d,
            )

        return jax.jit(inner)

    @staticmethod
    def _lattice_coupling(
        lattice_coupling: LatticeCoupling,
        tolerance: float = 1e-5,
    ):
        """Takes in some orbitals and geometry. Returns a function that is suitable as a value in an InputDict. The returned function takes a distance and returns the appropriate nearest neighbour coupling.

        :param lattice_coupling:
        :param tolerance:
        :returns: A function that is suitable as a value in an InputDict
        """

        lattice_basis = lattice_coupling.lattice.lattice_basis
        unit_cell = lattice_coupling.lattice.unit_cell
        couplings = jnp.array([x + 0j for x in lattice_coupling.couplings])

        positions = []
        for i in range(len(couplings)):
            for pos in unit_cell:
                for v in lattice_basis:
                    positions.append(pos + i * v)
        positions = jnp.array(positions)
        distances = jnp.unique(
            jnp.round(jnp.linalg.norm(positions - positions[:, None, :], axis=2), 8)
        )[: len(couplings)]

        def inner(d):
            return lax.cond(
                jnp.min(jnp.abs(d - distances)) < tolerance,
                lambda x: couplings[jnp.argmin(jnp.abs(x - distances))],
                lattice_coupling.coupling_function,
                d,
            )

        return jax.jit(inner)


## CONVERTING "Orbital" LISTS TO A STACK (HAMILTONIAN, COULOMB, EIGENVECTORS, ENERGIES, POSITIONS)
@jax.jit
def _encode(i, j):
    m = jnp.maximum(i, j) - 1
    return i + j + (m * (m + 1) * 0.5).astype(int)


def _make_fun(hf, cf):
    def inner(index, arg):
        return lax.switch(index, hf, arg), lax.switch(index, cf, arg)

    return jax.jit(inner)


def _unpack(orbs, hopping, coulomb):
    position = jnp.array([orb.position for orb in orbs])
    unique_ids = list(set(reduce(lambda x, y: x + y, hopping.keys())))
    h_dict = {
        tuple(sorted((unique_ids.index(x[0]), unique_ids.index(x[1])))): y
        for x, y in hopping.items()
    }
    c_dict = {
        tuple(sorted((unique_ids.index(x[0]), unique_ids.index(x[1])))): y
        for x, y in coulomb.items()
    }
    h_list, c_list = [
        h_dict[(j, i)] for i in range(len(unique_ids)) for j in range(i + 1)
    ], [c_dict[(j, i)] for i in range(len(unique_ids)) for j in range(i + 1)]
    ids = jnp.array([unique_ids.index(orb.orbital_id) for orb in orbs])
    return position, ids, _make_fun(h_list, c_list), unique_ids


def _hamiltonian_coulomb(pos, ids, func):
    def inner(i, j):
        return lax.cond(
            i >= j,
            func,
            lambda x, y: tuple(map(lambda x: x.conj(), func(x, y))),
            _encode(ids[i], ids[j]),
            jnp.linalg.norm(pos[i] - pos[j]),
        )

    return jax.vmap(jax.vmap(inner, (0, None), 0), (None, 0), 0)


def fermi(e, beta, mu):
    return 1 / (jnp.exp(beta * (e - mu)) + 1)


def _density_matrix(
    energies,
    electrons,
    spin_degeneracy,
    eps,
    from_state,
    to_state,
    excited_electrons,
    beta,
) -> tuple[Array, int]:
    """Calculates the normalized spin-traced 1RDM. For zero temperature, accordning to the Aufbau principle.
    At finite temperature, the chemical potential is determined for the thermal density matrix.

    :param energies: IP energies of the nanoflake
    """

    if beta == jnp.inf:
        return _density_aufbau(
            energies,
            electrons,
            spin_degeneracy,
            eps,
            from_state,
            to_state,
            excited_electrons,
        )
    return _density_thermo(energies, electrons, spin_degeneracy, eps, beta)


# TODO: notify if gradient descent does not converge
def _density_thermo(
    energies, electrons, spin_degeneracy, eps, beta, learning_rate=0.1, max_iter=10
):

    def loss(mu):
        return (spin_degeneracy * fermi(energies, beta, mu).sum() - electrons) ** 2

    # Compute the gradient of the loss function
    grad_loss = jax.grad(loss)

    # Define the gradient descent update step
    def update_step(mu):
        gradient = grad_loss(mu)
        new_mu = mu - learning_rate * gradient
        return new_mu

    # Define the condition for the while loop to continue
    # Here we simply run for a fixed number of iterations
    def cond_fun(val):
        mu, i = val
        return jnp.logical_or(loss(mu) > 1e-6, i < max_iter)

    # Define the body of the while loop
    def body_fun(val):
        mu, i = val
        new_mu = update_step(mu)
        return new_mu, i + 1

    # Initialize mu and iteration counter
    mu_init = 0.0  # initial value
    i_init = 0  # initial iteration

    # Run the gradient descent
    final_mu, final_iter = jax.lax.while_loop(cond_fun, body_fun, (mu_init, i_init))

    return (
        jnp.diag(spin_degeneracy * fermi(energies, beta, final_mu)) / electrons,
        jnp.nan,
    )


def _density_aufbau(
    energies, electrons, spin_degeneracy, eps, from_state, to_state, excited_electrons
):

    def _occupation(flags, spin_degeneracy, fraction):
        return jax.vmap(
            lambda flag: lax.switch(
                flag,
                [lambda x: spin_degeneracy, lambda x: fraction, lambda x: 0.0],
                flag,
            )
        )(flags)

    def _flags(index):
        """
        returns a tuple
        first element : array, labels all energies relative to a level given by energies[index] with
        0 => is_below_level, 1 => is_level, 2 => is_above_level
        second element : degeneracy of level, identified by the number of energies e fullfilling |e - level| < eps
        """
        is_level = jnp.abs(energies[index] - energies) < eps
        is_larger = energies - energies[index] > eps
        degeneracy = is_level.sum()
        return is_level + 2 * is_larger, degeneracy

    def _excitation(take_from, put_to, number):
        flags_from, deg_from = _flags(take_from)
        flags_to, deg_to = _flags(put_to)

        # transform to flag array, where 0 => else, 1 => from, 2 => to
        flags = (flags_from == 1) + (flags_to == 1) * 2
        return jax.vmap(
            lambda flag: lax.switch(
                flag,
                [
                    lambda x: 0.0,
                    lambda x: -number / deg_from,
                    lambda x: number / deg_to,
                ],
                flag,
            )
        )(flags)

    def _body_fun(index, occupation):
        return lax.cond(
            -from_state[index] == to_state[index],
            lambda x: x,
            lambda x: x
            + _excitation(
                homo - from_state[index],
                homo + to_state[index],
                excited_electrons[index],
            ),
            occupation,
        )

    homo = jnp.array(jnp.ceil(electrons / spin_degeneracy), int) - 1
    # determine where and how often this energy occurs
    flags, degeneracy = _flags(homo)
    # compute electrons at the fermi level
    below = (energies < energies[homo] - eps).sum()
    homo = jnp.nonzero(flags, size=1)[0]
    remaining_electrons = electrons - spin_degeneracy * below
    # compute ground state occupations by distributing the remaining electrons across all degenerate levels
    occupation = _occupation(flags, spin_degeneracy, remaining_electrons / degeneracy)
    # excited states
    occupation = lax.fori_loop(0, to_state.size, _body_fun, occupation)

    return jnp.diag(occupation) / electrons, homo


def _stack(
    orbs: list[Orbital],
    hopping: InputDict,
    coulomb: InputDict,
    field_func: Callable[[float], Array],
    from_state: int,
    to_state: int,
    excited_electrons: int,
    doping: int,
    beta: float,
    sublattice_ids: Array,
    eps: float,
    spin_degeneracy: int,
    transitions: TransitionsDict,
) -> Stack:
    """Takes a list of orbitals and two dictionaries specifying the hopping rates and coulomb interactions and produces a stack that holds the state of the system before it is propagated in time (e.g. the eigenenergies and density matrix).

    :param orbs: list of orbitals
    :param hopping: input dictionary for the hopping rates
    :param coulomb: input dictionary for the coulomb couplings
    :param field_func: position-dependent additive term for hamiltonian (e.g. electric field)
    :param from_state: optional, the state from which electrons to take for the initial excited transition, the state translates to HOMO - from_state
    :param to_state: optional, the state into which electrons to put for the initial excited transition, the state translates to HOMO + from_state
    :param doping: optional, how many doping electrons to throw into the stack
    :param eps: optional, default 1e-5: numerical threshold. if two energies E1,E2 fullfill abs(E1-E2) < eps, they are considered degenerate
    :returns: stack object
    """

    pos, ids, func, unique_ids = _unpack(orbs, hopping, coulomb)
    idxs = jnp.arange(pos.shape[0])

    hamiltonian, coulomb = _hamiltonian_coulomb(pos, ids, func)(idxs, idxs)

    electrons = sum(x.occupation for x in orbs) + doping

    eigenvectors, energies = lax.linalg.eigh(hamiltonian + field_func(pos))

    # TODO check same length
    from_state = jnp.array([from_state] if isinstance(from_state, int) else from_state)
    to_state = jnp.array([to_state] if isinstance(to_state, int) else to_state)
    excited_electrons = jnp.array(
        [excited_electrons] if isinstance(excited_electrons, int) else excited_electrons
    )

    eq_arr = jnp.array([0])
    rho_0, homo = _density_matrix(
        energies,
        electrons,
        spin_degeneracy,
        eps,
        from_state,
        to_state,
        excited_electrons,
        beta,
    )
    rho_stat, _ = _density_matrix(
        energies, electrons, spin_degeneracy, eps, eq_arr, eq_arr, eq_arr, beta
    )

    return Stack(
        hamiltonian + field_func(pos),
        coulomb,
        rho_0,
        rho_stat,
        energies,
        eigenvectors,
        pos,
        unique_ids,
        ids,
        sublattice_ids,
        eps,
        homo,
        electrons,
        from_state,
        to_state,
        excited_electrons,
        beta,
        spin_degeneracy,
        transitions,
    )


## DIPOLE TRANSITIONS
def dipole_transitions(
    stack: Stack,
    add_induced: bool = False,
) -> Callable:
    """Takes into account dipole transitions.

    :param stack:
    :param add_induced: add induced field to the field acting on the dipole
    :returns: transition function as additional input for evolution function
    """

    def inner(charge, H, E):
        def element(i, j):
            return lax.cond(
                # check if position index combination corresponds to an adatom orbital
                jnp.any(jnp.all(indices == jnp.array([i, j]), axis=2)),
                lambda x: lax.switch(
                    2 * (i == j) + (i < j),
                    [
                        lambda x: x
                        + 0.5 * moments[ind(i), :] @ (E[:, i] + induced_E[ind(i), :]),
                        lambda x: x
                        + (
                            0.5 * moments[ind(i), :] @ (E[:, i] + induced_E[ind(i), :])
                        ).conj(),
                        lambda x: x + stack.positions[i, :] @ induced_E[ind(i), :].real,
                    ],
                    x,
                ),
                lambda x: x,
                H[i, j],
            )

        # array of shape orbitals x 3, such that induced_E[i, :] corresponds to the induced field at the position of the adatom associated with the i-th transition
        induced_E = (
            14.39
            * jnp.tensordot(
                r_point_charge, charge.real, axes=[0, 0]
            )  # computes \sum_i r_i/|r_i|^3 * Q_i
            * add_induced
        )
        return jax.vmap(jax.vmap(jax.jit(element), (0, None), 0), (None, 0), 0)(
            idxs, idxs
        )

    if stack.transitions is None:
        return jax.jit(lambda c, h, e: h)

    # map position index to transition index
    ind = jax.jit(lambda i: jnp.argmin(jnp.abs(i - indices[:, 0, :])))
    idxs = jnp.arange(stack.positions.shape[0])

    indices, moments = [], []
    for (orb1, orb2), moment in stack.transitions.items():
        i1, i2 = (
            jnp.where(stack.ids == stack.unique_ids.index(orb1))[0],
            jnp.where(stack.ids == stack.unique_ids.index(orb2))[0],
        )
        assert (
            i1.size == i2.size == 1
        ), "Dipole transitions are allowed only between orbitals with unique names in the entire stack (name each dipole orbital differently, e.g. d1,d2 for 1 dipole or d11, d12 and d21, d22 for two dipoles)"
        i1, i2 = i1[0], i2[0]
        assert jnp.allclose(
            stack.positions[i1], stack.positions[i2]
        ), "Dipole transitions must happen between orbitals at the same location"
        indices.append([[i1, i2], [i2, i1], [i1, i1], [i2, i2]])
        moments += [moment, moment]

    indices, moments = jnp.array(indices), jnp.array(moments)

    # array of shape positions x orbitals x 3, with entries vec_r[i, o, :] = r_i - r_o
    vec_r = (
        jnp.repeat(
            stack.positions[:, jnp.newaxis, :], 2 * len(stack.transitions), axis=1
        )
        - stack.positions[indices[:, 0, :].flatten(), :]
    )

    # array of shape positions x orbitals x 3, with entries r_point_charge[i, o, :] = (r_o - r_i)/|r_o - r_i|^3
    r_point_charge = jnp.nan_to_num(
        vec_r / jnp.expand_dims(jnp.linalg.norm(vec_r, axis=2) ** 3, 2),
        posinf=0.0,
        neginf=0.0,
    )

    return jax.jit(inner)


## DISSIPATION
def relaxation(tau: float) -> Callable:
    """Function for modelling dissipation according to the relaxation approximation.

    :param tau: relaxation time
    :returns: JIT-compiled closure that is needed for computing the dissipative part of the lindblad equation
    """
    return jax.jit(lambda r, rs: -(r - rs) / (2 * tau))


def lindblad(stack, gamma, saturation):
    """Function for modelling dissipation according to the saturated lindblad equation. TODO: ref paper

    :param stack: object representing the state of the system
    :param gamma: symmetric (or lower triangular) NxN matrix. The element gamma[i,j] corresponds to the transition rate from state i to state j
    :param saturation: a saturation functional to apply, defaults to a sharp turn-off
    :returns: JIT-compiled closure that is needed for computing the dissipative part of the lindblad equation
    """

    commutator_diag = jnp.diag(gamma)
    gamma_matrix = gamma.astype(complex)
    saturation_vmapped = jax.vmap(saturation, 0, 0)

    def inner(r, rs):
        # convert rho to energy basis
        r = stack.eigenvectors.conj().T @ r @ stack.eigenvectors

        # extract occupations
        diag = jnp.diag(r) * stack.electrons

        # apply the saturation functional to turn off elements in the gamma matrix
        gamma = gamma_matrix * saturation_vmapped(diag)[None, :]

        a = jnp.diag(gamma.T @ jnp.diag(r))
        mat = jnp.diag(jnp.sum(gamma, axis=1))
        b = -1 / 2 * (mat @ r + r @ mat)
        val = a + b

        return stack.eigenvectors @ val @ stack.eigenvectors.conj().T

    return inner


## self-consistency
def get_self_consistent(
    stack: Stack, iterations: int = 50, mix: float = 0.05, accuracy: float = 1e-6
) -> Stack:
    """Get a stack with a self-consistent IP Hamiltonian.

    :param stack: a stack object
    :param iterations:
    :param mix:
    :param accuracy:
    :returns: a stack object
    """

    def _to_site_basis(ev, mat):
        return ev @ mat @ ev.conj().T

    def _phi(rho):
        return stack.coulomb @ jnp.diag(rho - rho_uniform)

    def _stop(args):
        return jnp.logical_and(
            jnp.linalg.norm(_phi(args[0]) - _phi(args[1])) > accuracy,
            args[2] < iterations,
        )

    def _loop(args):
        rho, rho_old, idx = args
        ham_new = stack.hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)

        # diagonalize
        energies, eigenvectors = jnp.linalg.eigh(ham_new)

        # new density matrix
        rho_energy, _ = _density_matrix(
            energies,
            stack.electrons,
            stack.spin_degeneracy,
            stack.eps,
            eq_arr,
            eq_arr,
            eq_arr,
            stack.beta,
        )

        return _to_site_basis(eigenvectors, rho_energy), rho, idx + 1

    # construct a uniform density matrix in real space
    _, arr, counts = jnp.unique(
        jnp.round(stack.positions, 8), return_inverse=True, return_counts=True, axis=0
    )
    normalization = 2.0 if int(stack.spin_degeneracy) == 1 else 1.0
    rho_uniform = jnp.diag((1 / counts)[arr[:, 0]]) / (
        stack.energies.size * normalization
    )
    eq_arr = jnp.array([0])

    # first induced potential
    rho_old = jnp.zeros_like(stack.hamiltonian)
    rho = _to_site_basis(stack.eigenvectors, stack.rho_stat)

    # sc loop
    rho, rho_old, idx = lax.while_loop(_stop, _loop, (rho, rho_old, 0))
    if idx == iterations - 1:
        raise Exception("Self-consistent procedure did not converge!!")

    # new hamiltonian and initial state
    ham_new = stack.hamiltonian + _phi(rho) * mix + _phi(rho_old) * (1 - mix)
    energies, eigenvectors = jnp.linalg.eigh(ham_new)
    rho_0, homo = _density_matrix(
        energies,
        stack.electrons,
        stack.spin_degeneracy,
        stack.eps,
        stack.from_state,
        stack.to_state,
        stack.excited_electrons,
        stack.beta,
    )

    return stack.replace(
        hamiltonian=ham_new,
        rho_0=rho_0,
        rho_stat=eigenvectors.conj().T @ rho @ eigenvectors,
        energies=energies,
        eigenvectors=eigenvectors,
        homo=homo,
    )


## ELECTRIC FIELDS
def electric_field(
    amplitudes: list[float],
    frequency: float,
    positions: Array,
    k_vector: list[float] = [0.0, 0.0, 1.0],
):
    """Function for computing time-harmonic electric fields.

    :param amplitudes: electric field amplitudes in xyz-components
    :param frequency: frequency
    :param positions: position for evaluation
    :returns: JIT-compiled closure that computes the electric field as a functon of time
    """
    static_part = jnp.expand_dims(jnp.array(amplitudes), 1) * jnp.exp(
        -1j * jnp.pi / 2 + 1j * positions @ jnp.array(k_vector)
    )
    return jax.jit(lambda t: jnp.exp(1j * frequency * t) * static_part)


def electric_field_with_ramp_up(
    amplitudes: list[float],
    frequency: float,
    positions: Array,
    ramp_duration: float,
    time_ramp: float,
    k_vector: list[float] = [0.0, 0.0, 1.0],
):
    """Function for computing ramping up time-harmonic electric fields.

    :param amplitudes: electric field amplitudes in xyz-components
    :param frequency: frequency
    :param positions: positions for evaluation
    :param ramp_duration: specifies how long does the electric field ramps up
    :param time_ramp:
    :returns: JIT-compiled closure that computes the electric field as a functon of time
    """
    static_part = jnp.expand_dims(jnp.array(amplitudes), 1) * jnp.exp(
        -1j * positions @ jnp.array(k_vector)
    )
    p = 0.99
    ramp_constant = 2 * jnp.log(p / (1 - p)) / ramp_duration
    return jax.jit(
        lambda t: static_part
        * jnp.exp(1j * frequency * t)
        / (1 + 1.0 * jnp.exp(-ramp_constant * (t - time_ramp)))
    )


def electric_field_pulse(
    amplitudes: list[float],
    frequency: float,
    positions: Array,
    peak: float,
    fwhm: float,
    k_vector: list[float] = [0.0, 0.0, 1.0],
):
    """Function for computing temporally located time-harmonics electric fields. The pulse is implemented as a temporal Gaussian.

    :param amplitudes: electric field amplitudes in xyz-components
    :param frequency: frequency of the electric field
    :param positions: positions where the electric field is evaluated
    :param peak: time where the pulse reaches its peak
    :param fwhm: full width at half maximum
    :returns: JIT-compiled closure that computes the electric field as a functon of time
    """

    static_part = jnp.expand_dims(jnp.array(amplitudes), 1) * jnp.exp(
        -1j * positions @ jnp.array(k_vector)
    )
    sigma = fwhm / (2.0 * jnp.sqrt(jnp.log(2)))
    return jax.jit(
        lambda t: static_part
        * jnp.exp(-1j * jnp.pi / 2 + 1j * frequency * (t - peak))
        * jnp.exp(-((t - peak) ** 2) / sigma**2)
    )


def dos(stack: Stack, omega: float, broadening: float = 0.1) -> Array:
    """IP-DOS of a nanomaterial stack.

    :param stack: a stack object
    :param omega: frequency
    :param broadening: numerical brodening parameter to replace Dirac Deltas with
    """

    broadening = 1 / broadening
    prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
    gaussians = jnp.exp(-((stack.energies - omega) ** 2) / 2 * broadening**2)
    return prefactor * jnp.sum(gaussians)


def ldos(stack: Stack, omega: float, site_index: int, broadening: float = 0.1) -> Array:
    """IP-LDOS of a nanomaterial stack.

    :param stack: a stack object
    :param omega: frequency
    :param site_index: the site index to evaluate the LDOS at
    :param broadening: numerical brodening parameter to replace Dirac Deltas with
    """

    broadening = 1 / broadening
    weight = jnp.abs(stack.eigenvectors[site_index, :]) ** 2
    prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
    gaussians = jnp.exp(-((stack.energies - omega) ** 2) / 2 * broadening**2)
    return prefactor * jnp.sum(weight * gaussians)


## TIME PROPAGATION
def electric_multipole_operator(stack, n: int = 1):
    """Returns n-th electric multipole operator starting with n = 1 => dipole."""

    def compute_expression(i, carry):
        if i <= 0:
            return carry
        return compute_expression(i - 1, jnp.einsum(exp[i - 1], dip, carry))

    def build_expression(i, carry):
        if i <= 0:
            return carry
        sequence = "".join(alphabet[: (i + 1)])
        sequence_in = sequence[:1] + "1" + sequence[1:]
        sequence_out = sequence[:1] + "23" + sequence[1:]
        return build_expression(i - 1, carry + [f"123,{sequence_in}->{sequence_out}"])

    alphabet = [chr(i) for i in range(ord("a"), ord("z") + 1)]
    exp = build_expression(n - 1, [])
    dip = granad.position_operator(stack)
    return compute_expression(n - 1, dip)


def position_operator(stack):
    N = stack.positions.shape[0]
    pos = jnp.zeros((N, N, 3))
    for i in range(3):
        pos = pos.at[:, :, i].set(jnp.diag(stack.positions[:, i] / 2))
    if not stack.transitions is None:
        for key, value in stack.transitions.items():
            value = jnp.array(value)
            i, j = indices(stack, key[0]), indices(stack, key[1])
            k = value.nonzero()[0]
            pos = pos.at[i, j, k].set(value[k])
    return pos + jnp.transpose(pos, (1, 0, 2))


def velocity_operator(stack):
    if stack.transitions is None:
        x_times_h = jnp.einsum("ij,ir->ijr", stack.hamiltonian, stack.positions)
        h_times_x = jnp.einsum("ij,jr->ijr", stack.hamiltonian, stack.positions)
    else:
        positions = position_operator(stack)
        x_times_h = jnp.einsum("kj,ikr->ijr", stack.hamiltonian, positions)
        h_times_x = jnp.einsum("ik,kjr->ijr", stack.hamiltonian, positions)
    return -1j * (x_times_h - h_times_x)


# TODO: units, reliably identify steady-states
def evolution(
    stack: Stack,
    time: Array,
    field: FieldFunc,
    dissipation: DissipationFunc = lambda x, y: 0.0,
    coulomb_strength: float = 1.0,
    saveat=None,
    solver=diffrax.Dopri5(),
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    spatial=False,
    add_induced=False,
):

    def rhs_uniform(time, rho, args):
        e_field, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ext = jnp.sum(stack.positions * e_field.real.T, axis=1)
        p_ind = coulomb @ charge
        h_total = transition(charge, stack.hamiltonian, e_field) + jnp.diag(
            p_ext - p_ind
        )
        return -1j * (h_total @ rho - rho @ h_total) + dissipation(rho, rho_stat)

    def rhs_spatial(time, rho, args):
        vector_potential, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ind = coulomb @ charge

        # we need to go from velocity to momentum under minimal coupling, i.e. p -> p - qA; H -> H - q A v + q^2/(2m) A^2
        h_total = (
            stack.hamiltonian
            - q * jnp.einsum("ijr, ir -> ij", v, vector_potential)
            + jnp.diag(q**2 / m * 0.5 * jnp.sum(vector_potential**2, axis=1) - p_ind)
        )
        return -1j * (h_total @ rho - rho @ h_total) + dissipation(rho, rho_stat)

    coulomb = stack.coulomb * coulomb_strength
    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T

    if spatial:
        q = 1
        m = 1
        v = velocity_operator(stack)
        rhs = rhs_spatial
    else:
        rhs = rhs_uniform
        transition = dipole_transitions(stack, add_induced)

    term = diffrax.ODETerm(rhs)
    rho_init = stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T
    saveat = diffrax.SaveAt(ts=time if saveat is None else saveat)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=time[0],
        t1=time[-1],
        dt0=time[1] - time[0],
        y0=rho_init,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return (
        stack.replace(
            rho_0=stack.eigenvectors.conj().T @ sol.ys[-1] @ stack.eigenvectors
        ),
        sol,
    )


def evolution_old(
    stack: Stack,
    time: Array,
    field: FieldFunc,
    dissipation: DissipationFunc = None,
    coulomb_strength: float = 1.0,
    postprocess: Callable[[Array], Array] = None,
    add_induced=False,
) -> tuple[Stack, Array]:
    """Propagate a stack forward in time.

    :param stack: stack object
    :param time: time axis
    :param field: electric field function
    :param dissipation: dissipation function
    :param coulomb_strength: scaling factor applied to coulomb interaction strength
    :param dipole_transition: a function describing dipolar transition radiation in the stack
    :param postprocess: a function applied to the density matrix after each time step
    :returns: (stack with rho_0 set to the current state, array containing all results from postprocess at all timesteps)
    """

    def integrate(rho, time):
        e_field, delta_rho = field(time), rho - rho_stat
        charge = -jnp.diag(delta_rho) * stack.electrons
        p_ext = jnp.sum(stack.positions * e_field.real.T, axis=1)
        p_ind = coulomb @ charge
        h_total = transition(charge, stack.hamiltonian, e_field) + jnp.diag(
            p_ext - p_ind
        )
        if dissipation:
            return (
                rho
                - 1j * dt * (h_total @ rho - rho @ h_total)
                + dt * dissipation(rho, rho_stat),
                postprocess(rho) if postprocess else rho,
            )
        else:
            return (
                rho - 1j * dt * (h_total @ rho - rho @ h_total),
                postprocess(rho) if postprocess else rho,
            )

    dt = time[1] - time[0]
    coulomb = stack.coulomb * coulomb_strength
    rho_stat = stack.eigenvectors @ stack.rho_stat @ stack.eigenvectors.conj().T
    transition = dipole_transitions(stack, add_induced=add_induced)
    rho, rhos = jax.lax.scan(
        integrate, stack.eigenvectors @ stack.rho_0 @ stack.eigenvectors.conj().T, time
    )

    return (
        stack.replace(rho_0=stack.eigenvectors.conj().T @ rho @ stack.eigenvectors),
        rhos,
    )


## INDEPENDENT-PARTICLE CALCULATIONS
def transition_energies(stack: Stack) -> Array:
    """Computes independent-particle transition energies associated with the TB-Hamiltonian of a stack.

    :param stack:
    :returns: square array, the element :code:`arr[i,j]` contains the transition energy from :code:`i` to :code:`j`
    """
    return jnp.abs(jnp.expand_dims(stack.energies, 1) - stack.energies)


def wigner_weisskopf(stack: Stack, component: int = 0) -> Array:
    """Calculcates Wigner-Weisskopf transiton rates.

    :param stack:
    :param component: component of the dipolar transition to take :math:`(0,1,2) \\rightarrow (x,y,z)`

    :returns: square array, the element :code:`arr[i,j]` contains the transition rate from :code:`i` to :code:`j`
    """
    charge = 1.602e-19
    eps_0 = 8.85 * 1e-12
    hbar = 1.0545718 * 1e-34
    c = 3e8  # 137 (a.u.)
    factor = 1.6e-29 * charge / (3 * jnp.pi * eps_0 * hbar**2 * c**3)
    te = transition_energies(stack)
    return (
        (te * (te > stack.eps)) ** 3
        * jnp.squeeze(transition_dipole_moments(stack)[:, :, component] ** 2)
        * factor
    )


# TODO: this should be changed in accordance with the newly defined position operator
def transition_dipole_moments(stack: Stack) -> Array:
    r"""Compute transition dipole moments for all states :math:`i,j` as :math:`\braket{i | \hat{r} | j}`.

    :param stack: stack object with N orbitals
    :returns: Transition dipole moments as a complex :math:`N \times N \times 3` - matrix, where the last component is the direction of the dipole moment.
    """
    return jnp.einsum(
        "li,lj,lk", stack.eigenvectors.conj(), stack.eigenvectors, stack.positions
    ) * jnp.expand_dims(
        jnp.ones_like(stack.eigenvectors) - jnp.eye(stack.eigenvectors.shape[0]), 2
    )


## INTERACTION
def epi(stack: Stack, rho: Array, omega: float, epsilon: float = None) -> float:
    r"""Calculates the EPI (Energy-based plasmonicity index) of a mode at :math:`\hbar\omega` in the absorption spectrum of a structure.

    :param stack: stack object
    :param rho: density matrix
    :param omega: energy at which the system has been CW-illuminated (:math:`\hbar\omega` in eV)
    :param epsilon: small broadening parameter to ensure numerical stability, if :code:`None`, stack.eps is chosen
    :returns: EPI, a number between :code:`0` (single-particle-like) and :code:`1` (plasmonic).
    """
    epsilon = stack.eps if epsilon is None else epsilon
    rho_without_diagonal = jnp.abs(rho - jnp.diag(jnp.diag(rho)))
    rho_normalized = rho_without_diagonal / jnp.linalg.norm(rho_without_diagonal)
    te = transition_energies(stack)
    excitonic_transitions = (
        rho_normalized / (te * (te > stack.eps) - omega + 1j * epsilon) ** 2
    )
    return 1 - jnp.sum(jnp.abs(excitonic_transitions * rho_normalized)) / (
        jnp.linalg.norm(rho_normalized) * jnp.linalg.norm(excitonic_transitions)
    )


# TODO: get t-matrix in some way
def rpa_polarizability_function(
        stack, tau, polarization, coulomb_strength, phi_ext = None, hungry=True
):
    def _polarizability(omega):
        x = sus(omega)
        ro = x @ jnp.linalg.inv(one - c @ x) @ phi_ext
        return -pos @ ro

    one = jnp.identity(stack.electrons)
    pos = stack.positions[:, polarization]
    phi_ext = pos if phi_ext is None else phi_ext
    c = stack.coulomb * coulomb_strength
    sus = bare_susceptibility_function(stack, tau, hungry)
    return _polarizability


def bare_susceptibility_function(stack, tau, hungry=True):

    def _sum_subarrays(arr):
        """Sums subarrays in 1-dim array arr. Subarrays are defined by n x 2 array indices as [ [start1, end1], [start2, end2], ... ]"""
        arr = jnp.r_[0, arr.cumsum()][indices]
        return arr[:, 1] - arr[:, 0]

    def _susceptibility(omega):

        def susceptibility_element(site_1, site_2):
            # calculate per-energy contributions
            prefac = eigenvectors[site_1, :] * eigenvectors[site_2, :] / delta_omega
            f_lower = prefac * (energies - omega_low) * occupation
            g_lower = prefac * (energies - omega_low) * (1 - occupation)
            f_upper = prefac * (-energies + omega_up) * occupation
            g_upper = prefac * (-energies + omega_up) * (1 - occupation)
            f = (
                jnp.r_[0, _sum_subarrays(f_lower)][mask]
                + jnp.r_[0, _sum_subarrays(f_upper)][mask2]
            )
            g = (
                jnp.r_[0, _sum_subarrays(g_lower)][mask]
                + jnp.r_[0, _sum_subarrays(g_upper)][mask2]
            )
            b = jnp.fft.ihfft(f, n=2 * f.size, norm="ortho") * jnp.fft.ihfft(
                g[::-1], n=2 * f.size, norm="ortho"
            )
            Sf1 = jnp.fft.hfft(b)[:-1]
            Sf = -Sf1[::-1] + Sf1
            eq = 2.0 * Sf / (omega - omega_grid_extended + 1j / (2.0 * tau))
            return -jnp.sum(eq)

        if hungry:
            return jax.vmap(
                jax.vmap(susceptibility_element, (0, None), 0), (None, 0), 0
            )(sites, sites)
        return lax.map(
            lambda i: lax.map(lambda j: susceptibility_element(i, j), sites), sites
        )

    # unpacking
    energies = stack.energies.real
    eigenvectors = stack.eigenvectors.real
    occupation = jnp.diag(stack.rho_0).real * stack.electrons / stack.spin_degeneracy
    sites = jnp.arange(energies.size)
    freq_number = 2**12
    omega_max = jnp.real(max(stack.energies[-1], -stack.energies[0])) + 0.1
    omega_grid = jnp.linspace(-omega_max, omega_max, freq_number)

    # build two arrays sandwiching the energy values: omega_low contains all frequencies bounding energies below, omega_up bounds above
    upper_indices = jnp.argmax(omega_grid > energies[:, None], axis=1)
    omega_low = omega_grid[upper_indices - 1]
    omega_up = omega_grid[upper_indices]
    delta_omega = omega_up[0] - omega_low[0]

    omega_dummy = jnp.linspace(-2 * omega_grid[-1], 2 * omega_grid[-1], 2 * freq_number)
    omega_3 = omega_dummy[1:-1]
    omega_grid_extended = jnp.insert(omega_3, int(len(omega_dummy) / 2 - 1), 0)

    # indices for grouping energies into contributions to frequency points.
    # e.g. energies like [1,1,2,3] on a frequency grid [0.5, 1.5, 2.5, 3.5]
    # the contribution array will look like [ f(1, eigenvector), f(1, eigenvector'), f(2, eigenvector_2), f(3, eigenvector_3) ]
    # we will have to sum the first two elements
    # we do this by building an array "indices" of the form: [ [0, 2], [2,3], [3, 4] ]
    omega_low_unique, indices = jnp.unique(omega_low, return_index=True)
    indices = jnp.r_[jnp.repeat(indices, 2)[1:], indices[-1] + 1].reshape(
        omega_low_unique.size, 2
    )

    # mask for inflating the contribution array to the full size given by omega_grid.
    comparison_matrix = omega_grid[:, None] == omega_low_unique[None, :]
    mask = (jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1)
    comparison_matrix = omega_grid[:, None] == jnp.unique(omega_up)[None, :]
    mask2 = (jnp.argmax(comparison_matrix, axis=1) + 1) * comparison_matrix.any(axis=1)
    return _susceptibility


## POSTPROCESSING
def indices(stack: Stack, orbital_id: str) -> Array:
    """Gets indices of a specific orbital.

    Can be used to calculate, e.g. positions and energies corresponding to that orbital in the stack.

    :param stack: stack object
    :param orbital_id: orbital identifier as contained in :class:`granad.numerics.Stack.unique_ids`
    :returns: integer array corresponding to the indices of the orbitals, such that e.g. :code:`stack.energies[indices]` gives the energies associated with the orbitals.
    """
    return jnp.nonzero(stack.ids == stack.unique_ids.index(orbital_id))[0]


# TODO: this should be changed in accordance with the newly defined position operator
def induced_dipole_moment(stack: Stack, rhos_diag: Array) -> Array:
    """
    Calculates the induced dipole moment for a collection of density matrices.

    :param stack: stack object
    :param rhos_diag: :math:`N \times m` time-dependent site occupation matrix, indexed by rhos_diag[timestep,site_number]
    :returns: dipole_moments, :math:`N \times 3` matrix containing the induced dipole moment :math:`(p_x, p_y, p_z)` at :math:`N` times
    """
    return (
        (jnp.diag(to_site_basis(stack, stack.rho_stat)) - rhos_diag).real
        @ stack.positions
        * stack.electrons
    )


def induced_field(
    stack: Stack, positions: Array, density_matrix: Union[Array, None] = None
) -> Array:
    """Classical approximation to the induced (local) field in a stack.

    :param stack: a stack object
    :param positions: positions to evaluate the field on, must be of shape N x 3
    :density_matrix: if given, compute the field corresponding to this density matrix. otherwise, use :code:`stack.rho_0`.
    """

    # determine whether to use argument or the state of the stack
    density_matrix = density_matrix if density_matrix is not None else stack.rho_0

    # distance vector array from field sources to positions to evaluate field on
    vec_r = stack.positions[:, None] - positions

    # scalar distances
    denominator = jnp.linalg.norm(vec_r, axis=2) ** 3

    # normalize distance vector array
    point_charge = jnp.nan_to_num(
        vec_r / denominator[:, :, None], posinf=0.0, neginf=0.0
    )

    # compute charge via occupations in site basis
    charge = stack.electrons * jnp.diag(to_site_basis(stack, density_matrix)).real

    # induced field is a sum of point charges, i.e. \vec{r} / r^3
    e_field = 14.39 * jnp.sum(point_charge * charge[:, None, None], axis=0)
    return e_field


def to_site_basis(stack: Stack, matrix: Array) -> Array:
    """Transforms an arbitrary matrix from energy to site basis.

    :param stack: stack object
    :param matrix: square array in energy basis
    :returns: square array in site basis
    """
    return stack.eigenvectors @ matrix @ stack.eigenvectors.conj().T


def to_energy_basis(stack: Stack, matrix: Array) -> Array:
    """Transforms an arbitrary matrix from site to energy basis.

    :param stack: stack object
    :param matrix: square array in energy basis
    :returns: square array in energy basis
    """
    return stack.eigenvectors.conj().T @ matrix @ stack.eigenvectors


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
def show_energies(stack: Stack):
    """Depicts the energy and occupation landscape of a stack (energies are plotted on the y-axis ordered by size)

    :param stack: stack object
    """
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
def show_energy_occupations(
    stack: Stack,
    occupations: list[Array],
    time: Array,
    thresh: float = 1e-2,
):
    """Depicts energy occupations as a function of time.

    :param stack: a stack object
    :param occupations: list of energy occupations (complex arrays). The occupation at timestep n is given by :code:`occupations[n]`.
    :param time: time axis
    :param thresh: plotting threshold. an occupation time series o_t is selected for plotting if it outgrows/outshrinks this bound. More exactly: o_t is plotted if max(o_t) - min(o_t) > thresh
    """
    fig, ax = plt.subplots(1, 1)
    occ = jnp.array([stack.electrons * r.real for r in occupations])
    for idx in jnp.nonzero(
        jnp.abs(jnp.amax(occ, axis=0) - jnp.amin(occ, axis=0)) > thresh
    )[0]:
        ax.plot(time, occ[:, idx], label=f"{float(stack.energies[idx]):2.2f} eV")
    ax.set_xlabel(r"time [$\hbar$/eV]")
    ax.set_ylabel("occupation of eigenstate")
    plt.legend()


@_plot_wrapper
def show_electric_field_space(
    first: Array,
    second: Array,
    plane: str,
    time: Array,
    field_func: Callable[[float], Array],
    args: dict,
    component: int = 0,
    flag: int = 0,
):
    """Shows the external electric field on a spatial grid at a fixed point in time.

    :param first: grid coordinates. get passed directly as meshgrid(frist, second).
    :param second: grid coordinates. get passed directly as meshgrid(frist, second).
    :param plane: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'. E.g. 'xy' means: make a plot in xy-plane and use "first"-parameter as x-axis, "second"-parameter as y-axis
    :param time: time to evalute the field at
    :param field_func: a function taking in parameters as given by args and an additional argument "positions" that produces a closure that gives the electric field as function of time
    :param args: arguments to field_func as a dictionary, The "positions"-argument must be dropped.
    :param component: 0 => plot x, 1 => plot y, 2 => plot z
    :param flag: 0 => plot real, 1 => plot imag, 2 => plot abs
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
def show_electric_field_time(time: Array, field: Array, flag: int = 0):
    """Shows the external electric field with its (x,y,z)-components as a function of time at a fixed spatial point.

    :param time: array of points in time for field evaluation
    :param field: output of an electric field function
    :param flag: 0 => plot real, 1 => plot imag, 2 => plot abs
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
def show_eigenstate3D(
    stack: Stack,
    show_state: int = 0,
    show_orbitals: list[str] = None,
    indicate_size: bool = True,
    color_orbitals: bool = True,
    annotate_hilbert: bool = True,
):
    """Shows a 3D scatter plot of how selected orbitals in a stack contribute to an eigenstate.
    In the plot, orbitals are annotated with a color. The color corresponds either to the contribution to the selected eigenstate or to the type of the orbital.
    Optionally, orbitals can be annotated with a number corresponding to the hilbert space index.

    :param stack: object representing system state
    :param show_state: eigenstate index to show. (0 => eigenstate with lowest energy)
    :param show_orbitals: orbitals to show. if None, all orbitals are shown.
    :param indicate_size: if True, bigger points are orbitals contributing more strongly to the selected eigenstate.
    :param color_orbitals: if True, identical orbitals are colored identically and a legend is displayed listing all the different orbital types. if False, the color corresponds to the contribution to the sublattice.
    :param annotate_hilbert: if True, the orbitals are annotated with a number corresponding to the hilbert space index.
    """
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
def show_eigenstate2D(
    stack: Stack,
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

    :param stack: object representing system state
    :param plane: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'.
    :param show_state: eigenstate index to show. (0 => eigenstate with lowest energy)
    :param show_orbitals: list of strings. orbitals to show. if None, all orbitals are shown.
    :param indicate_size: if True, bigger points are orbitals contributing more strongly to the selected eigenstate.
    :param color_orbitals: if True, identical orbitals are colored identically and a legend is displayed listing all the different orbital types. if False, the color corresponds to the sublattice.
    :param annotate_hilbert: if True, the orbitals are annotated with a number corresponding to the hilbert space index.
    """
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
def show_charge_distribution3D(stack: Stack):
    """Displays the ground state charge distribution of the stack in 3D

    :param stack: stack object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    charge = stack.electrons * jnp.diag(
        stack.eigenvectors @ stack.rho_0.real @ stack.eigenvectors.conj().T
    )
    sp = ax.scatter(*zip(*stack.positions[:, :2]), zs=stack.positions[:, 2], c=charge)
    plt.colorbar(sp)


@_plot_wrapper
def show_charge_distribution2D(stack: Stack, plane: str = "xy"):
    """Displays the ground state charge distribution of the stack in 2D

    :param stack: object representing system state
    :param plane: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'
    """
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
def show_induced_field(
    rho: Array,
    electrons: int,
    eigenvectors: Array,
    positions: Array,
    first: Array,
    second: Array,
    plane: str = "xy",
    component: int = 0,
    norm: int = 1,
    plot_stack: bool = True,
):
    """Displays the normalized logarithm of the absolute value of the induced field in 2D

    :param rho: density matrix
    :param electrons: number of electrons
    :param eigenvectors: eigenvectors of the corresponding stack (as stored in a stack object)
    :param positions: positions of the orbitals in the stack
    :param first: grid coordinates. get passed directly as meshgrid(frist, second).
    :param second: grid coordinates. get passed directly as meshgrid(frist, second).
    :param plane: which plane to use for field evaluation. one of 'xy', 'xz', 'yz'. E.g. 'xy' means: make a plot in xy-plane and use "first"-parameter as x-axis, "second"-parameter as y-axis
    :param component: 0 => plot x, 1 => plot y, 2 => plot z
    :param norm : constant to normalize the field
    :param plot_stack: if True, add a scatter plot indicating the positions of the orbitals in the stack
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
