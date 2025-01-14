import pprint
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from functools import wraps

import jax
import jax.numpy as jnp
from matplotlib.path import Path

from granad import _watchdog
from granad.orbitals import Orbital, OrbitalList
from granad._plotting import _display_lattice_cut

def zero_coupling(d):
    """
    Returns a zero coupling constant as a complex number.

    Args:
        d (float): A parameter (typically representing distance or some other factor) that is ignored by the function, as the output is always zero.

    Returns:
        complex: Returns 0.0 as a complex number (0.0j).
    """
    return 0.0j

def ohno_potential( offset = 0, start = 14.399 ):
    """
    Generates a callable that represents a regularized Coulomb-like potential.

    The potential function is parameterized to provide flexibility in adjusting the starting value and an offset,
    which can be used to avoid singularities at zero distance.

    Args:
        offset (float): The offset added to the distance to prevent division by zero and to regularize the potential at short distances. Defaults to 0.
        start (float): The initial strength or scaling factor of the potential. Defaults to 14.399.

    Returns:
        Callable[[float], complex]: A function that takes a distance 'd' and returns the computed Coulomb-like potential as a complex number.

    Note:
        ```python
        potential = ohno_potential()
        print(potential(1))  # Output: (14.399 + 0j) if default parameters used
        ```
    """
    def inner(d):
        """Coupling with a (regularized) Coulomb-like potential"""
        return start / (jnp.linalg.norm(d) + offset) + 0j
    return inner


def cut_flake_1d( material, unit_cells, plot=False):
    """
    Cuts a one-dimensional flake from the material based on the specified number of unit cells
    and optionally plots the lattice and orbital positions.

    Parameters:
        material (Material): The material instance from which to cut the flake.
        unit_cells (int): The number of unit cells to include in the flake.
        plot (bool, optional): If True, displays a plot of the orbital positions within the lattice.
                               Default is False.

    Returns:
        list: A list of orbitals positioned within the specified range of the material's lattice.

    Note:
        The function utilizes internal methods of the `Material` class to compute positions and
        retrieve orbital data, ensuring that the positions are unique and correctly mapped to the
        material's grid.
    """
    
    orbital_positions_uc =  material._get_positions_in_uc()
    grid = material._get_grid( [(0, unit_cells)] )
    orbital_positions = material._get_positions_in_lattice( orbital_positions_uc, grid )
    if plot:
        _display_lattice_cut( orbital_positions, orbital_positions )

    orbital_positions = jnp.unique( orbital_positions, axis = 0)        
    return material._get_orbital_list( orbital_positions, grid )

def cut_flake_2d( material, polygon, plot=False, minimum_neighbor_number: int = 2):
    """
    Cuts a two-dimensional flake from the material defined within the bounds of a specified polygon.
    It further prunes the positions to ensure that each atom has at least the specified minimum number of neighbors.
    Optionally, the function can plot the initial and final positions of the atoms within the polygon.

    Parameters:
        material (Material): The material instance from which to cut the flake.
        polygon (array-like): An array of coordinates defining the vertices of the polygon within which to cut the flake.
        plot (bool, optional): If True, plots the lattice and the positions of atoms before and after pruning.
                               Default is False.
        minimum_neighbor_number (int, optional): The minimum number of neighbors each atom must have to remain in the final positions.
                                                 Default is 2.

    Returns:
        list: A list of orbitals positioned within the specified polygon and satisfying the neighbor condition.

    Note:
        The function assumes the underlying lattice to be in the xy-plane.
    """
    def _prune_neighbors(
            positions, minimum_neighbor_number, remaining_old=jnp.inf
    ):
        """
        Recursively prunes positions to ensure each position has a sufficient number of neighboring positions
        based on a minimum distance calculated from the unique set of distances between positions.

        Parameters:
            positions (array-like): Array of positions to prune.
            minimum_neighbor_number (int): Minimum required number of neighbors for a position to be retained.
            remaining_old (int): The count of positions remaining from the previous iteration; used to detect convergence.

        Returns:
            array-like: Array of positions that meet the neighbor count criterion.
        """
        if minimum_neighbor_number <= 0:
            return positions
        distances = jnp.round(
            jnp.linalg.norm(positions[:, material.periodic] - positions[:, None, material.periodic], axis=-1), 4
        )
        minimum = jnp.unique(distances)[1]
        mask = (distances <= minimum).sum(axis=0) > minimum_neighbor_number
        remaining = mask.sum()
        if remaining_old == remaining:
            return positions[mask]
        else:
            return _prune_neighbors(
                positions[mask], minimum_neighbor_number, remaining
            )

    # to cover the plane, we solve the linear equation P = L C, where P are the polygon vertices, L is the lattice basis and C are the coefficients
    L = material._lattice_basis[material.periodic,:2] * material.lattice_constant
    coeffs = jnp.linalg.inv(L.T) @ polygon.T * 1.1

    # we just take the largest extent of the shape
    u1, u2 = jnp.ceil( coeffs ).max( axis = 1)
    l1, l2 = jnp.floor( coeffs ).min( axis = 1)
    grid = material._get_grid( [ (int(l1), int(u1)), (int(l2), int(u2)) ]  )

    # get atom positions in the unit cell in fractional coordinates
    orbital_positions =  material._get_positions_in_uc()
    unit_cell_fractional_atom_positions = jnp.unique(
        jnp.round(orbital_positions, 6), axis=0
            )

    initial_atom_positions = material._get_positions_in_lattice(
        unit_cell_fractional_atom_positions, grid
    ) 

    polygon_path = Path(polygon)
    flags = polygon_path.contains_points(initial_atom_positions[:, :2])
    
    # get atom positions where every atom has at least minimum_neighbor_number neighbors
    final_atom_positions = _prune_neighbors(
        initial_atom_positions[flags], minimum_neighbor_number
    )

    if plot == True:
        _display_lattice_cut(
            initial_atom_positions, final_atom_positions, polygon
        )
    return material._get_orbital_list(final_atom_positions, grid)

def cut_flake_generic( material, grid_range ):
    """
    Cuts a flake from the material using a specified grid range. This method is generic and can be applied
    to materials of any dimensionality.

    The function calculates the positions of orbitals within the unit cell, projects these onto the full
    lattice based on the provided grid range, and ensures that each position is unique. The result is a list
    of orbitals that are correctly positioned within the defined grid.

    Parameters:
        material (Material): The material instance from which to cut the flake.
        grid_range (list of tuples): Each tuple in the list specifies the range for the grid in that dimension.
                                     For example, [(0, 10), (0, 5)] defines a grid that extends from 0 to 10
                                     in the first dimension and from 0 to 5 in the second dimension.

    Returns:
        list: A list of orbitals within the specified grid range, uniquely positioned.

    Note:
        The grid_range parameter should be aligned with the material's dimensions and lattice structure,
        as mismatches can lead to incorrect or inefficient slicing of the material.
    """
    orbital_positions_uc =  material._get_positions_in_uc()
    grid = material._get_grid( grid_range)
    orbital_positions = material._get_positions_in_lattice( orbital_positions_uc, grid )
    orbital_positions = jnp.unique( orbital_positions, axis = 0)        
    return material._get_orbital_list( orbital_positions, grid )

def _finalize(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        for species in self.species.keys():
            self._species_to_groups[species] = _watchdog._Watchdog.next_value()
        if "grid_range" in kwargs:
            return cut_flake_generic(self, *args, **kwargs)
        if self.dim == 1:
            return cut_flake_1d(self, *args, **kwargs)        
        elif self.dim == 2:
            return cut_flake_2d(self, *args, **kwargs)
        else:
            return cut_flake_generic(self, *args, **kwargs)
    return wrapper

class Material:
    """
    Represents a material in a simulation, encapsulating its physical properties and interactions.

    Attributes:
        name (str): The name of the material.
        species (dict): Dictionary mapping species names to their quantum numbers and associated atoms.
                        Each species is defined with properties like principal quantum number (n),
                        angular momentum quantum number (l), magnetic quantum number (m), spin quantum number (s),
                        and the atom type.
        orbitals (defaultdict[list]): A mapping from species to lists of orbitals. Each orbital is represented
                                      as a dictionary containing the orbital's position and an optional tag
                                      for further identification.
        interactions (defaultdict[dict]): Describes the interactions between orbitals within the material.
                                         Each interaction is categorized by type (e.g., 'hamiltonian', 'Coulomb'),
                                         and includes the participants, parameters like 
                                         [onsite, offsite_nearest_neighbor, offsite_next_to_nearest_neighbor, ...], and                                         
                                         an optional mathematical expression defining the interaction for the coupling beyound 
                                         the len(parameters) - th nearest neighbor.

    Note:
        The `Material` class is used to define a material's structure and properties step-by-step.
        An example is constructing the material graphene, with specific lattice properties,
        orbitals corresponding to carbon's p_z orbitals, and defining hamiltonian and Coulomb interactions
        among these orbitals. 

        ```python
        graphene = (
            Material("graphene")
            .lattice_constant(2.46)
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0]
            ])
            .add_orbital_species("pz", l=1, atom='C')
            .add_orbital(position=(0, 0), tag="sublattice_1", species="pz")
            .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=[0.0, -2.66],
            )
            .add_interaction(
                "coulomb",
                participants=("pz", "pz"),
                parameters=[16.522, 8.64, 5.333],
                expression=lambda r : 1/r + 0j
            )
        )
        ```
    """
    def __init__(self, name):
        self.name = name
        self.species = {}
        self.orbitals = defaultdict(list)
        self.interactions = defaultdict(dict)
        self._species_to_groups=  {}
        self.dim = None
        
    def __str__(self):
        description = f"Material: {self.name}\n"
        if self.lattice_constant:
            description += f"  Lattice Constant: {self.lattice_constant} Å\n"
        if self.lattice_basis:
            description += f"  Lattice Basis: \n{self._lattice_basis}\n"
        
        if self.species:
            description += "  Orbital Species:\n"
            for species_name, attributes in self.species.items():
                description += f"    {species_name} characterized by (n,l,m,s, atom name) = {attributes}\n"
        
        if self.orbitals:
            description += "  Orbitals:\n"
            for spec, orbs in self.orbitals.items():
                for orb in orbs:
                    description += f"    Position: {orb['position']}, Tag: {orb['tag']}, Species: {spec}\n"
        
        if self.interactions:
            description += "  Interactions:\n"
            for type_, interaction in self.interactions.items():
                for participants, coupling in interaction.items():
                    description += f"""   Type: {type_}, Participants: {participants}:
                    NN Couplings: {', '.join(map(str, coupling[0]))}
                    """
                    # Check if there's a docstring on the function
                    if coupling[1].__doc__ is not None:
                        function_description = coupling[1].__doc__
                    else:
                        function_description = "No description available for this function."
                        
                    description += f"Other neighbors: {function_description}\n"
        
        return description
    
    def lattice_constant(self, value):
        """
        Sets the lattice constant for the material.

        Parameters:
            value (float): The lattice constant value.

        Returns:
            Material: Returns self to enable method chaining.
        """
        self.lattice_constant = value
        return self

    def lattice_basis(self, values, periodic = None):
        """
        Defines the lattice basis vectors and specifies which dimensions are periodic.

        Parameters:
            values (list of list of float): A list of vectors representing the lattice basis.
            periodic (list of int, optional): Indices of the basis vectors that are periodic. Defaults to all vectors being periodic.

        Returns:
            Material: Returns self to enable method chaining.
        """
        self._lattice_basis = jnp.array(values)
        total = set(range(len(self._lattice_basis)))        
        periodic = set(periodic) if periodic is not None else total
        self.periodic = list(periodic)
        self.finite = list(total - periodic)
        self.dim = len(self.periodic)                                              
        return self
    
    @_finalize
    def cut_flake( self ):
        """
        Finalizes the material construction by defining a method to cut a flake of the material,
        according to the material's dimensions like this

        1D material : materials.cut_flake_1d
        2D material : materials.cut_flake_2d
        3D material and higher : materials.cut_flake_generic

        This method is intended to be called after all material properties (like lattice constants, 
        basis, orbitals, and interactions) have been fully defined.

        Note:
        This method does not take any parameters and does not return any value. Its effect is
        internal to the state of the Material object and is meant to prepare the material for
        simulation by implementing necessary final structural adjustments.
        """
        pass

    def add_orbital(self, position, species, tag = ''):
        """
        Sets the lattice constant for the material.

        Parameters:
            value (float): The lattice constant value.

        Returns:
            Material: Returns self to enable method chaining.
        """
        self.orbitals[species].append({'position': position, 'tag': tag})
        return self

    def add_orbital_species( self, name, n = 0, l = 0, m = 0, s = 0, atom  = ''):
        """
        Adds a species definition for orbitals in the material.

        Parameters:
            name (str): The name of the orbital species.
            n (int): Principal quantum number.
            l (int): Orbital angular momentum quantum number.
            m (int): Magnetic quantum number.
            s (int): Spin quantum number.
            atom (str, optional): Name of the atom the orbital belongs to.

        Returns:
            Material: Returns self to enable method chaining.
        """
        self.species[name] = (n,l,m,s,atom)
        return self

    def add_interaction(self, interaction_type, participants, parameters = None, expression = zero_coupling):
        """
        Adds an interaction between orbitals specified by an interaction type and participants.

        Parameters:
            interaction_type (str): The type of interaction (e.g., 'hamiltonian', 'Coulomb').
            participants (tuple): A tuple identifying the participants in the interaction.
            parameters (dict): Parameters relevant to the interaction.
            expression (function): A function defining the mathematical form of the interaction.

        Returns:
            Material: Returns self to enable method chaining.
        """            
        self.interactions[interaction_type][participants] =  (parameters if parameters is not None else [], expression)
        return self

    def _get_positions_in_uc( self, species = None ):
        if species is None:
            return jnp.array( [x["position"] for orb in list(self.orbitals.values()) for x in orb] )
        return jnp.array( [orb_group['position'] for s in species for orb_group in self.orbitals[s] ] )

    def _get_positions_in_lattice(self, uc_positions, grid):
        shift = jnp.array(uc_positions) @ self._lattice_basis
        return self.lattice_constant * (
            grid @ self._lattice_basis + shift[:, None, :]
        ).reshape(shift.shape[0] * grid.shape[0], 3)

    def _get_grid(self, ns ):
        grid = [(1,) for i in range( len(self.finite) + len(self.periodic)) ]
        for i, p in enumerate(self.periodic):
            grid[p] = range(*ns[i])
        return jnp.array( list( product( *(x for x in grid) ) ) )
    
    def _keep_matching_positions(self, positions, candidates):
        idxs = (
            jnp.round(jnp.linalg.norm(positions - candidates[:, None], axis=-1), 4) == 0
        ).nonzero()[0]
        return candidates[idxs]

    def _couplings_to_function(
        self, couplings, outside_fun, species
    ):

        # no couplings
        if len(couplings) == 0:
            return outside_fun

        # vector couplings
        if all(isinstance(i, list) for i in couplings):
            return self._vector_couplings_to_function(couplings, outside_fun, species)

        # distance couplings
        return self._distance_couplings_to_function(couplings, outside_fun, species)
        
    
    def _vector_couplings_to_function(self, couplings, outside_fun, species):

        vecs, couplings_vals = jnp.array(couplings).astype(float)[:, :3], jnp.array(couplings).astype(complex)[:, 3]
        distances = jnp.linalg.norm(vecs, axis=1)
        
        def inner(d):
            return jax.lax.cond(
                jnp.min(jnp.abs(jnp.linalg.norm(d) - distances)) < 1e-5,
                lambda x: couplings_vals[jnp.argmin(jnp.linalg.norm(d - vecs, axis=1))],
                outside_fun,
                d,
            )
        return inner

    def _distance_couplings_to_function(self, couplings, outside_fun, species):
        
        couplings = jnp.array(couplings).astype(complex)
        grid = self._get_grid( [ (0, len(couplings)) for i in range(self.dim) ] )
        pos_uc_1 = self._get_positions_in_uc( (species[0],) )
        pos_uc_2 = self._get_positions_in_uc( (species[1],) )
        positions_1 = self._get_positions_in_lattice(pos_uc_1, grid )
        positions_2 = self._get_positions_in_lattice(pos_uc_2, grid )
        
        distances = jnp.unique(
            jnp.round(jnp.linalg.norm(positions_1 - positions_2[:, None, :], axis=2), 5)
        )[: len(couplings)]

        def inner(d):
            d = jnp.linalg.norm(d)
            return jax.lax.cond(
                jnp.min(jnp.abs(d - distances)) < 1e-5,
                lambda x: couplings[jnp.argmin(jnp.abs(x - distances))],
                outside_fun,
                d,
            )
        
        return inner

    def _set_couplings(self, setter_func, interaction_type):
        interaction_dict = self.interactions[interaction_type]
        for (species_1, species_2), couplings in interaction_dict.items():
            distance_func = self._couplings_to_function(
                *couplings, (species_1, species_2)
            )
            setter_func(self._species_to_groups[species_1], self._species_to_groups[species_2], distance_func)


    def _get_orbital_list(self, allowed_positions, grid):
                
        raw_list, layer_index = [], 0
        for species, orb_group in self.orbitals.items():
            
            for orb_uc in orb_group:
            
                uc_positions = jnp.array( [orb_uc['position']] )

                rs_positions = self._get_positions_in_lattice( uc_positions, grid )

                final_positions = self._keep_matching_positions( allowed_positions, rs_positions )

                for position in final_positions:
                    orb = Orbital(
                        position = position,
                        layer_index = layer_index,
                        tag=orb_uc['tag'],
                        group_id = self._species_to_groups[species],                        
                        energy_level = self.species[species][0],
                        angular_momentum = self.species[species][1],
                        angular_momentum_z= self.species[species][2],
                        spin=self.species[species][3],
                        atom_name=self.species[species][4]
                    )
                    layer_index += 1
                    raw_list.append(orb)

        orbital_list = OrbitalList(raw_list)
        self._set_couplings(orbital_list.set_hamiltonian_groups, "hamiltonian")
        self._set_couplings(orbital_list.set_coulomb_groups, "coulomb")
        return orbital_list
    
def get_mos2():
    return (Material("MoS2")
            .lattice_constant(3.16)  # Approximate lattice constant of monolayer MoS2
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0],  # Hexagonal lattice
                [0, 0, 1],
            ], periodic = [0,1])
            .add_orbital_species("d_molybdenum", l=2, atom='Mo')
            .add_orbital_species("p_sulfur", l=1, atom='S')
            .add_orbital(position=(0, 0, 0), tag="sublattice_molybdenum", species="d_molybdenum")
            .add_orbital(position=(1/3, 2/3, -1.5), tag="sublattice_sulfur_top", species="p_sulfur")
            .add_orbital(position=(1/3, 2/3, 1.5), tag="sublattice_sulfur_bottom", species="p_sulfur")
            .add_interaction(
                "hamiltonian",
                participants=("d_molybdenum", "p_sulfur"),
                parameters=[-3.352]  
            )
            .add_interaction(
                "hamiltonian",
                participants=("p_sulfur", "p_sulfur"),
                parameters=[-2., 0.1, 0.1]  
            )
            .add_interaction(
                "hamiltonian",
                participants=("d_molybdenum", "d_molybdenum"),
                parameters=[0.201, 0.2175]  
            )
            .add_interaction(
                "coulomb",
                participants=("d_molybdenum", "p_sulfur"),
                expression = ohno_potential(1)
            )
            .add_interaction(
                "coulomb",
                participants=("p_sulfur", "p_sulfur"),
                expression = ohno_potential(1)
            )
            .add_interaction(
                "coulomb",
                participants=("d_molybdenum", "d_molybdenum"),
                expression = ohno_potential(1)
            )
            )

    
def get_hbn():
    return (Material("hBN")
            .lattice_constant(2.50)  # Approximate lattice constant of hBN
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0],  # Hexagonal lattice
            ])
            .add_orbital_species("pz_boron", l=1, atom='B')
            .add_orbital_species("pz_nitrogen", l=1, atom='N')
            .add_orbital(position=(0, 0), tag="B", species="pz_boron")
            .add_orbital(position=(-1/3, -2/3), tag="N", species="pz_nitrogen")
            .add_interaction(
                "hamiltonian",
                participants=("pz_boron", "pz_boron"),
                parameters=[2.46, -0.04],  
            )
            .add_interaction(
                "hamiltonian",
                participants=("pz_nitrogen", "pz_nitrogen"),
                parameters=[-2.55, -0.04],  
            )
            .add_interaction(
                "hamiltonian",
                participants=("pz_boron", "pz_nitrogen"),
                parameters=[-2.16],  
            )
            .add_interaction(
                "coulomb",
                participants=("pz_boron", "pz_boron"),
                expression = ohno_potential(1)
            )
            .add_interaction(
                "coulomb",
                participants=("pz_nitrogen", "pz_nitrogen"),
                expression = ohno_potential(1)
            )
            .add_interaction(
                "coulomb",
                participants=("pz_boron", "pz_nitrogen"),
                expression = ohno_potential(1)
            )
            )

    
def get_graphene():
    return (Material("graphene")
            .lattice_constant(2.46)
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0]
            ])
            .add_orbital_species("pz", l=1, atom='C')
            .add_orbital(position=(0, 0), tag="sublattice_1", species="pz")
            .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=[0.0, -2.66],
            )
            .add_interaction(
                "coulomb",
                participants=("pz", "pz"),
                parameters=[16.522, 8.64, 5.333],
                expression=ohno_potential(0)
            )
            )

def get_ssh(delta = 0.2, displacement = 0.8):
    """Returns an ssh model with hoppings (eV)
    
    [onsite, nn, nnn] = [0.0, 1 + delta, 1 - delta]

    and the 2-atom unit cell (Angström)

    pos = [0, 0.8]    
    """
    return (Material("ssh")
            .lattice_constant(2.46)
            .lattice_basis([
                [1, 0, 0],
            ])
            .add_orbital_species("pz", l=1, atom='C')
            .add_orbital(position=(0,), tag="sublattice_1", species="pz")
            .add_orbital(position=(displacement,), tag="sublattice_2", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=[0.0, 1 + delta, 1 - delta],
            )
            .add_interaction(
                "coulomb",
                participants=("pz", "pz"),
                parameters=[16.522, 8.64, 5.333],
                expression=ohno_potential(0)
            )
            )

def get_metal_1d():
    return (Material("metal_1d")
            .lattice_constant(2.46)
            .lattice_basis([
                [1, 0, 0],
            ])
            .add_orbital_species("pz", l=1, atom='C')
            .add_orbital(position=(0,), tag="", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=[0.0, 2.66],
            )
            .add_interaction(
                "coulomb",
                participants=("pz", "pz"),
                parameters=[16.522, 8.64, 5.333],
                expression=ohno_potential(0)
            )
            )

class MaterialCatalog:
    """
    A class to manage and access built-in material properties within a simulation or modeling framework.
    
    This class provides a central repository for predefined materials, allowing for easy retrieval
    and description of their properties.

    Attributes:
        _materials (dict): A private dictionary that maps material names to their respective data objects.
                           This dictionary is pre-populated with several example materials such as graphene and MoS2.

    Methods:
        get(material): Retrieves the data object associated with the given material name.
        describe(material): Prints a description or the data object of the specified material.
        available(): Prints a list of all available materials stored in the catalog.
    """
    _materials = {"graphene" : get_graphene, "ssh" : get_ssh, "metal_1d" : get_metal_1d, "MoS2" : get_mos2, "hBN" : get_hbn }

    @staticmethod
    def get(material : str):
        """
        Retrieves the material data object for the specified material.

        Args:
            material (str): The name of the material to retrieve.

        Returns:
            The data object associated with the specified material.

        Example:
            ```python
            graphene_data = MaterialCatalog.get('graphene')
            ```
        """
        return MaterialCatalog._materials[material]()
    
    @staticmethod
    def describe(material : str):
        """
        Prints a description or the raw data of the specified material from the catalog.

        Args:
            material (str): The name of the material to describe.

        Example:
            ```python
            MaterialCatalog.describe('graphene')
            ```
        """
        print(MaterialCatalog._materials[material]())
    
    @staticmethod
    def available():
        """
        Prints a list of all materials available in the catalog.

        Example:
            ```python
            MaterialCatalog.available()
            ```
        """
        available_materials = "\n".join(MaterialCatalog._materials.keys())
        print(f"Available materials:\n{available_materials}")
