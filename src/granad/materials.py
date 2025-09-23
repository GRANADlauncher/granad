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
from granad._graphene_special import _cut_flake_graphene

from granad.parsing import parse_sk_file

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
        polygon (Polygon): A polygon objects with a vertices property holding an array of coordinates defining the vertices of the polygon within which to cut the flake.
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

    if material.name in ['graphene', 'graphene_spinful_hubbard'] and polygon.polygon_id in ["hexagon", "triangle"]:
        n, m, vertices, final_atom_positions, initial_atom_positions, sublattice = _cut_flake_graphene(polygon.polygon_id, polygon.edge_type, polygon.side_length, material.lattice_constant)
                
        raw_list, layer_index = [], -1
        for i, position in enumerate(final_atom_positions):
            layer_index += 1
            if material.name == "graphene":
                orb = Orbital(
                position = position,
                layer_index = layer_index,
                tag="sublattice_1" if sublattice[i] == "A" else "sublattice_2",
                group_id = material._species_to_groups["pz"],                        
                spin=material.species["pz"][0],
                atom_name=material.species["pz"][1]
                    )
            
                raw_list.append(orb)
            else:
                orb = Orbital(
                position = position,
                layer_index = layer_index,
                tag="sublattice_1" if sublattice[i] == "A" else "sublattice_2",
                group_id = material._species_to_groups["pz+"],                        
                spin=material.species["pz+"][0],
                atom_name=material.species["pz+"][1]
                )
                layer_index += 1
                raw_list.append(orb)
                
                orb = Orbital(
                position = position,
                layer_index = layer_index,
                tag="sublattice_1" if sublattice[i] == "A" else "sublattice_2",
                group_id = material._species_to_groups["pz-"],                        
                spin=material.species["pz-"][0],
                atom_name=material.species["pz-"][1]
                )
                layer_index += 1
                raw_list.append(orb)

        orbital_list = OrbitalList(raw_list)
        material._set_couplings(orbital_list.set_hamiltonian_groups, "hamiltonian")
        material._set_couplings(orbital_list.set_coulomb_groups, "coulomb")
        orb_list = orbital_list

    else:
        # to cover the plane, we solve the linear equation P = L C, where P are the polygon vertices, L is the lattice basis and C are the coefficients
        vertices = polygon.vertices
        L = material._lattice_basis[material.periodic,:2] * material.lattice_constant
        coeffs = jnp.linalg.inv(L.T) @ vertices.T * 1.1

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

        polygon_path = Path(vertices)
        flags = polygon_path.contains_points(initial_atom_positions[:, :2])        
        pruned_atom_positions = initial_atom_positions[flags]

        # get atom positions where every atom has at least minimum_neighbor_number neighbors
        final_atom_positions = _prune_neighbors(
            pruned_atom_positions, minimum_neighbor_number
        )
        orb_list = material._get_orbital_list(final_atom_positions, grid)

    if plot == True:
        _display_lattice_cut(
            initial_atom_positions, final_atom_positions, vertices
        )
    return orb_list

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
                        Each species is defined with properties like spin quantum number (s), and the atom type.
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
            .add_orbital_species("pz", atom='C')
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
        self.supported_orbitals =  {"s", "px", "py", "pz"}        
        
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
    
    # TODO: big uff
    # TODO: check if parity harmful in sk equations, include d, f orbs, spin
    def add_slater_koster_interaction(self, atom1, atom2, sk_file, num_neighbors = 1):
        """couples orbitals on atom1, atom2 according to slater koster parameters given in dftb file sk_file up to num_neighbors
        """
        
        # slater koster matrix element functions, sorted by combinations, as in https://link.aps.org/doi/10.1103/PhysRev.94.1498 table I
        matrix_element_map = {
            ("s", "s") : lambda l, m, n, params: params["sss"],
            ("px", "s") : lambda l, m, n, params: l * params["sps"],
            ("py", "s") : lambda l, m, n, params : m * params["sps"],
            ("pz", "s") : lambda l, m, n, params : n * params["sps"],
            ("px", "px") : lambda l, m, n, params : l**2 * params["pps"] + (1 - l**2) * params["ppp"],
            ("py", "py") : lambda l, m, n, params : m**2 * params["pps"] + (1 - m**2) * params["ppp"],
            ("pz", "pz") : lambda l, m, n, params : n**2 * params["pps"] + (1 - n**2) * params["ppp"],
            ("px", "py") : lambda l, m, n, params : l*m * params["pps"] + (1 - l*m) * params["ppp"],
            ("px", "pz") : lambda l, m, n, params : l*n * params["pps"] + (1 - l*n) * params["ppp"],
            ("py", "pz") : lambda l, m, n, params : m*n * params["pps"] + (1 - m*n) * params["ppp"]
        }

        def unpack(params, idx):
            return {k : vals[idx] for k, vals in params.items() }

        # computes either ham or overlap
        def coupling_entry(params, comb, vec):
            length = jnp.linalg.norm(vec)

            # index in params is lowered
            idx = jnp.argwhere(jnp.abs(length - distances) < 1e-5)[0][0] - 1

            # direction cosines
            l, m, n = vec / length

            # return everything as a list
            return vec.tolist() + [matrix_element_map[comb](l, m, n, unpack(params, idx))]
            
        # distance range
        cutoff = num_neighbors + 1
        
        # orbitals hosted on each atom
        atom1_species = [orb for orb in self.supported_orbitals if f"{orb}_{atom1}" in self.species]
        atom2_species = [orb for orb in self.supported_orbitals if f"{orb}_{atom2}" in self.species]

        # scalar distances and vectors pointing from atom1 to atom2 in a lattice cut covering at least num_neighbors (we just pick two random representatives)
        distances, distance_vecs = self._get_distances_and_vecs(f"{atom1_species[0]}_{atom1}",
                                                                f"{atom2_species[0]}_{atom2}",
                                                                cutoff)

        # unique offsite distance vecs (we don't need onsite, because we handle it ungracefully)
        idxs = jnp.logical_and(jnp.linalg.norm(distance_vecs, axis = 2) < distances.max() + 1e-5, jnp.linalg.norm(distance_vecs, axis = 2) - 1e-5 > 0)
        offsite_vecs = jnp.unique(distance_vecs[idxs], axis = 0)

        # dftb files use atomic units
        sk_params = parse_sk_file(sk_file, distances * 0.5)

        # unique combinations of orbitals
        combinations = set(tuple(sorted((i, j))) for i in atom1_species for j in atom2_species)

        # for each orbital combination: loop over distances => at each distance, store couplings in the form [vec, coupling]
        for comb in combinations:
            hamiltonian = [coupling_entry(sk_params["integrals"], comb, vec) for vec in offsite_vecs]
            overlap = [coupling_entry(sk_params["overlap"], comb, vec) for vec in offsite_vecs]
            # dummy
            coulomb = []
            
            # onsite is special case idrc
            if atom1 == atom2 and comb[0] == comb[1]:
                hamiltonian += [[0, 0, 0., sk_params["onsite"][comb[0][:1]][0] ]]
                overlap     += [[0, 0, 0, 1.]]
                coulomb     += [[0, 0, 0., sk_params["hubbard"][comb[0][:1]][0]]]

            # add interactions as regular vector couplings
            full_name = (f"{comb[0]}_{atom1}", f"{comb[1]}_{atom2}")
            self.add_interaction("hamiltonian", full_name, hamiltonian)
            self.add_interaction("overlap", full_name, overlap)
            self.add_interaction("coulomb", full_name, coulomb)

        return self

    def add_atom(self, atom, position, orbitals = None, spinful = False):
        """adds atom and all indicated orbitals
        """
        orbitals = orbitals or self.supported_orbitals        
        if len(self.supported_orbitals - set(orbitals)) < 0:
            raise AttributeError(f"Unsupported orbitals in {orbitals}. Currently, only {self.supported_orbitals} are valid.")
        
        for orb in orbitals:
            species = f"{orb}_{atom}"
            if spinful == True:
                raise NotImplementedError("No spinful orbs in atom")
                # self.add_orbital(position, f"{species}_up", atom = atom, kind = orb, s = 1)
                # self.add_orbital(position, f"{species}_down", atom = atom, kind = orb, s = -1)
            else:
                self.add_orbital(position, species, atom = atom, kind = orb)
                
        return self

    def add_orbital(self, position, species, tag = '', atom = '', s = 0, kind = None):
        """adds orbital
        """
        if species not in self.species:
            self.add_orbital_species(species, s, atom)
        self.orbitals[species].append({'position': position, 'tag': tag, 'kind' : kind})
        return self

    # we only need this function to assign an id to a "batch" of orbitals (eg all pz orbs in graphene)
    def add_orbital_species( self, name, s = 0, atom  = ''):
        """
        Adds a species definition for orbitals in the material.

        Parameters:
            name (str): The name of the orbital species.
            s (int): Spin quantum number.
            atom (str, optional): Name of the atom the orbital belongs to.

        Returns:
            Material: Returns self to enable method chaining.
        """
        self.species[name] = (s,atom)
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

    # produces distances and distance vectors between sites of orbitals s1, s2 on a lattice cut of radius given by cutoff
    def _get_distances_and_vecs(self, s1, s2, cutoff):
        grid = self._get_grid( [ (0, cutoff) for i in range(self.dim) ] )
        pos_uc_1 = self._get_positions_in_uc( (s1,) )
        pos_uc_2 = self._get_positions_in_uc( (s2,) )
        positions_1 = self._get_positions_in_lattice(pos_uc_1, grid )
        positions_2 = self._get_positions_in_lattice(pos_uc_2, grid )

        distance_vecs = positions_1 - positions_2[:, None, :]
        distances = jnp.round(jnp.linalg.norm(distance_vecs, axis=2), 5)
        dist_unique = jnp.unique(distances)[:cutoff]

        return dist_unique, distance_vecs
        
    def _distance_couplings_to_function(self, couplings, outside_fun, species):
        
        couplings = jnp.array(couplings).astype(complex)
        distances, _ = self._get_distances_and_vecs(species[0], species[1], len(couplings))
        
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
                        kind=orb_uc['kind'],
                        group_id = self._species_to_groups[species],                        
                        spin=self.species[species][0],
                        atom_name=self.species[species][1]
                    )
                    layer_index += 1
                    raw_list.append(orb)

        orbital_list = OrbitalList(raw_list)
        self._set_couplings(orbital_list.set_hamiltonian_groups, "hamiltonian")
        self._set_couplings(orbital_list.set_coulomb_groups, "coulomb")
        self._set_couplings(orbital_list.set_overlap_groups, "overlap")
        return orbital_list

def get_hbn(lattice_constant = 2.50, bb_hoppings = None, nn_hoppings = None, bn_hoppings = None):
    """
    Get a material representation for hexagonal boron nitride (hBN).

    Parameters:
    - lattice_constant (float): The lattice constant for hBN. Default is 2.50.
    - bb_hoppings (list or None): Hopping parameters for B-B interactions. 
                                  Default is [2.46, -0.04].
    - nn_hoppings (list or None): Hopping parameters for nearest-neighbor interactions. 
                                  Default is [-2.55, -0.04].
    - bn_hoppings (list or None): Hopping parameters for B-N interactions. 
                                  Default is [-2.16].

    Default values are derived from the study of the electronic structure of hexagonal boron nitride (hBN).
    See [Giraud et al.](https://www.semanticscholar.org/paper/Study-of-the-Electronic-Structure-of-hexagonal-on-Unibertsitatea-Thesis/ff1e000bbad5d8e2df5f85cb724b1a9e42a8b0f0) for more details.

    Returns:
    - A tuple containing the lattice constant and hopping parameters.
    """
    bb_hoppings = [2.46, -0.04] if bb_hoppings is None else bb_hoppings
    bn_hoppings = [-2.16]  if bn_hoppings is None else bn_hoppings
    nn_hoppings = [-2.55, -0.04]  if nn_hoppings is None else nn_hoppings
    
    return (Material("hBN")
            .lattice_constant(lattice_constant)  # Approximate lattice constant of hBN
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0],  # Hexagonal lattice
            ])
            .add_orbital_species("pz_boron", atom='B')
            .add_orbital_species("pz_nitrogen", atom='N')
            .add_orbital(position=(0, 0), tag="B", species="pz_boron")
            .add_orbital(position=(-1/3, -2/3), tag="N", species="pz_nitrogen")
            .add_interaction(
                "hamiltonian",
                participants=("pz_boron", "pz_boron"),
                parameters=bb_hoppings,  
            )
            .add_interaction(
                "hamiltonian",
                participants=("pz_nitrogen", "pz_nitrogen"),
                parameters=nn_hoppings,  
            )
            .add_interaction(
                "hamiltonian",
                participants=("pz_boron", "pz_nitrogen"),
                parameters=bn_hoppings,
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

def get_mos2():
    """
    Generates a MoS2 model based on parameters from [Bert Jorissen, Lucian Covaci, and Bart Partoens, SciPost Phys. Core 7, 004 (2024)](https://scipost.org/SciPostPhysCore.7.1.004), taking into account even-parity eigenstates.

    Returns:
        Material: A `Material` object representing the MoS2 model.

    Example:
        >>> mos2 = get_mos2()
        >>> print(mos2)
    """
    reference_vector = jnp.array([1,1,0])
    ref = reference_vector[:2] / jnp.linalg.norm(reference_vector[:2])            

    # Onsite energies
    epsilon_M_e_0 = -6.475
    epsilon_M_e_1 = -4.891
    epsilon_X_e_0 = -7.907
    epsilon_X_e_1 = -9.470

    # First nearest neighbor hopping parameters (u1)
    u1_e_0 = 0.999
    u1_e_1 = -1.289
    u1_e_2 = 0.795
    u1_e_3 = -0.688
    u1_e_4 = -0.795

    # Second nearest neighbor hopping parameters for metal (u2,Me)
    u2_Me_0 = -0.048
    u2_Me_1 = 0.580
    u2_Me_2 = -0.074
    u2_Me_3 = -0.414
    u2_Me_4 = -0.299
    u2_Me_5 = 0.045

    # Second nearest neighbor hopping parameters for chalcogen (u2,Xe)
    u2_Xe_0 = 0.795
    u2_Xe_1 = -0.248
    u2_Xe_2 = 0.164
    u2_Xe_3 = -0.002
    u2_Xe_4 = -0.283
    u2_Xe_5 = -0.174

    onsite_x = jnp.array([epsilon_X_e_0, epsilon_X_e_0, epsilon_X_e_1])
    
    # poor man's back transformation since odd orbitals are discarded
    onsite_x /= 2
    
    onsite_m = jnp.array([epsilon_M_e_0, epsilon_M_e_1, epsilon_M_e_1])

    nn = jnp.array([
        [0, 0, u1_e_0],
        [u1_e_1, u1_e_2, 0],
        [u1_e_3, u1_e_4, 0]
    ])

    # poor man's back transformation since odd orbitals are discarded
    nn /= jnp.sqrt(2)

    nnn_M = jnp.array([
        [u2_Me_0,         u2_Me_1,         u2_Me_2],
        [u2_Me_1,         u2_Me_3,         u2_Me_4],
        [-u2_Me_2,       -u2_Me_4,         u2_Me_5]
    ])

    nnn_X = jnp.array([
        [u2_Xe_0,         u2_Xe_1,         u2_Xe_2],
        [-u2_Xe_1,        u2_Xe_3,         u2_Xe_4],
        [-u2_Xe_2,        u2_Xe_4,         u2_Xe_5] ]
    )
    # poor man's back transformation since odd orbitals are discarded
    nnn_X /= 2

    gamma = 2 * jnp.pi / 3  # 120 degrees in radians

    R_X_e = jnp.array([
        [jnp.cos(gamma), -jnp.sin(gamma), 0],
        [jnp.sin(gamma),  jnp.cos(gamma), 0],
        [0,              0,             1]
    ])

    theta = 2 * gamma      # 240 degrees for d_x2-y2 and d_xy rotation

    R_M_e = jnp.array([
        [1.0,           0.0,           0.0],
        [0.0,  jnp.cos(theta), -jnp.sin(theta)],
        [0.0,  jnp.sin(theta),  jnp.cos(theta)]
    ])

    nn_list = jnp.stack([nn, R_X_e @ nn @ R_M_e.T, R_X_e.T @ nn @ R_M_e])
    nnn_M_list = jnp.stack([nnn_M, R_M_e @ nnn_M @ R_M_e.T, R_M_e.T @ nnn_M @ R_M_e])
    nnn_X_list = jnp.stack([nnn_X, R_X_e @ nnn_X @ R_X_e.T, R_X_e.T @ nnn_X @ R_X_e])
    
    d_orbs =  ["dz2", "dx2-y2", "dxy"]
    p_orbs = ["px", "py", "pz"]
    orbs = d_orbs + p_orbs
    
    def generate_coupling(orb1, orb2):
        orb1_idx = orbs.index(orb1) % 3
        orb2_idx = orbs.index(orb2) % 3

        # Select the correct matrix stack
        arr = nn_list
        onsite = onsite_m
        if orb1[0] == "p" and orb2[0] == "p":
            arr = nnn_X_list
            onsite = onsite_x
        elif orb1[0] == "d" and orb2[0] == "d":
            arr = nnn_M_list

        def nn_coupling(vec):
            vec /= jnp.linalg.norm(vec)
            
            # Compute angle between ref and vec
            angle = jnp.arctan2(vec[1], vec[0]) - jnp.arctan2(ref[1], ref[0])
            angle = jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi  # Map to [-π, π]
            branch = 0 * jnp.logical_and(angle >= -jnp.pi / 3, angle <= jnp.pi / 3) + 1 * (angle < -jnp.pi / 3) + 2 * (angle >= jnp.pi/3)
            idx = jax.lax.switch(
                branch,
                [lambda : 0, lambda : 1, lambda : 2],
            )
            
            return arr[idx][orb1_idx, orb2_idx]

        def coupling(vec):
            length = jnp.linalg.norm(vec[:2])
            thresh = 3.4
            branch = 0 * (length == 0) + 1 * jnp.logical_and(0 < length, length < thresh) + 2 * (length >= thresh)
            
            return jax.lax.switch(branch,
                                  [lambda x : onsite[orb1_idx],
                                   lambda x : nn_coupling(x),
                                   lambda x : 0. 
                                   ],
                                  vec[:2]
                                  )

        return coupling

    mat = (Material("MoS2")
    .lattice_constant(3.16)  # Approximate lattice constant of monolayer MoS2
    .lattice_basis([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0],  # Hexagonal lattice
        [0, 0, 1],
    ], periodic = [0,1])
    .add_orbital_species("dz2", atom='Mo')
    .add_orbital_species("dx2-y2", atom='Mo')
    .add_orbital_species("dxy", atom='Mo')
    .add_orbital_species("px", atom='S')
    .add_orbital_species("py", atom='S')
    .add_orbital_species("pz", atom='S')
    .add_orbital(position=(0, 0, 0), tag="dz2", species="dz2")
    .add_orbital(position=(0, 0, 0), tag="dx2-y2", species="dx2-y2")
    .add_orbital(position=(0, 0, 0), tag="dxy", species="dxy")
    .add_orbital(position=(1/3, 2/3, 1.5), tag="px_top", species="px")
    .add_orbital(position=(1/3, 2/3, 1.5), tag="py_top", species="py")
    .add_orbital(position=(1/3, 2/3, 1.5), tag="pz_top", species="pz")
    .add_orbital(position=(1/3, 2/3, -1.5), tag="px_bottom", species="px")
    .add_orbital(position=(1/3, 2/3, -1.5), tag="py_bottom", species="py")
    .add_orbital(position=(1/3, 2/3, -1.5), tag="pz_bottom", species="pz")
           )

    for orb1 in orbs:
        for orb2 in orbs:
            mat = (mat.add_interaction("hamiltonian",
                                       participants = (orb1, orb2),
                                       expression = generate_coupling(orb1, orb2)
                                       )
                   .add_interaction("coulomb",
                                    participants = (orb1, orb2),
                                    expression = ohno_potential(1)
                                    )
                   )            
    return mat

    
def get_graphene(hoppings = None):
    """
    Generates a graphene model based on parameters from 
    [David Tománek and Steven G. Louie, Phys. Rev. B 37, 8327 (1988)](https://doi.org/10.1103/PhysRevB.37.8327).

    Args:
        hoppings (list, optional): Hopping parameters for pz-pz interactions.  Default is [onsite, nn] = [0, -2.66], as specified in the reference.

    Returns:
        Material: A `Material` object representing the graphene model, which includes:
            - **Lattice Structure**:
                - Lattice constant: 2.46 Å.
                - Hexagonal lattice basis vectors: [1, 0, 0] and [-0.5, sqrt(3)/2, 0].
            - **Orbitals**:
                - Two sublattices, each with a single "pz" orbital, positioned at (0, 0) and (-1/3, -2/3).
            - **Hamiltonian Interaction**:
                - Nearest-neighbor hopping: [0.0 (onsite energy), hopping (default -2.66 eV)].
            - **Coulomb Interaction**:
                - Parameterized by the Ohno potential with parameters [16.522, 8.64, 5.333].

    Example:
        >>> graphene_model = get_graphene(hopping=-2.7)
        >>> print(graphene_model)
    """
    hoppings = hoppings or [0, -2.66]
    return (Material("graphene")
            .lattice_constant(2.46)
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0]
            ])
            .add_orbital_species("pz",  atom='C')
            .add_orbital(position=(0, 0), tag="sublattice_1", species="pz")
            .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=hoppings,
            )
            .add_interaction(
                "coulomb",
                participants=("pz", "pz"),
                parameters=[16.522, 8.64, 5.333],
                expression=ohno_potential(0)
            )
            )

def get_ssh(delta = 0.2, displacement = 0.4, base_hopping = -2.66, lattice_const = 2.84):
    """
    Generates an SSH (Su-Schrieffer-Heeger) model with specified hopping parameters and a 2-atom unit cell.

    Args:
        delta (float, optional): A parameter controlling the alternating hopping amplitudes in the model. 
            - The nearest-neighbor hopping amplitudes are defined as [1 + delta, 1 - delta]. Default is 0.2.
        displacement (float, optional): The displacement of the second atom in the unit cell along the x-axis (in Ångström). 
            - Determines the position of the second atom relative to the first. Default is 0.4. Takes values between 0 and 1.
        base_hopping (float, optional): base hopping value on which symmetrically intra and inter unit-cell hopping rates are applied, defaults to -2.66 eV.
        lattice constant (float, optional):  distance between two unict cells, defaults to 2*1.42 = 2.84 Ångström (since each unit cell contains two sites).

    Returns:
        Material: An SSH model represented as a `Material` object, including:
            - Lattice structure with a lattice constant of 2.46 Å.
            - Two pz orbitals (one per sublattice) placed at [0] and [displacement].
            - Nearest-neighbor (NN) hopping amplitudes: [base_hopping*(1 + delta), base_hopping*(1 - delta)].
            - Coulomb interactions parameterized by Ohno potential.
    """
    return (Material("ssh")
            .lattice_constant(lattice_const) #Changed to  2*a_cc
            .lattice_basis([
                [1, 0, 0],
            ])
            .add_orbital_species("pz", atom='C')
            .add_orbital(position=(0,), tag="sublattice_1", species="pz")
            .add_orbital(position=(displacement,), tag="sublattice_2", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=[0.0, -2.66 + delta*(-2.66), -2.66 - delta*(-2.66)],
            )
            .add_interaction(
                "coulomb",
                participants=("pz", "pz"),
                parameters=[16.522, 8.64, 5.333],
                expression=ohno_potential(0)
            )
            )

def get_chain(hopping = -2.66, lattice_const = 1.42):
    """
    Generates a 1D metallic chain model with specified hopping and Coulomb interaction parameters.

    Args:
        hopping (float, optional): nn hopping, defaults to -2.66 eV.
        lattice constant (float, optional): nn distance, defaults to 1.42 Ångström

    Returns:
        Material: A `Material` object representing the 1D metallic chain, which includes:
            - **Lattice Structure**: 
                - Lattice constant: 2.46 Å.
                - Lattice basis: [1, 0, 0] (1D chain along the x-axis).
            - **Orbital**:
                - Single orbital species: "pz" (associated with Carbon atoms).
                - One orbital per unit cell, positioned at [0].

    Example:
        >>> metal_chain = get_chain()
        >>> print(metal_chain)
    """
    return (Material("chain")
            .lattice_constant(lattice_const)
            .lattice_basis([
                [1, 0, 0],
            ])
            .add_orbital_species("pz", atom='C')
            .add_orbital(position=(0,), tag="", species="pz")
            .add_interaction(
                "hamiltonian",
                participants=("pz", "pz"),
                parameters=[0.0, hopping],
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
    _materials = {"graphene" : get_graphene, "ssh" : get_ssh, "chain" : get_chain, "hBN" : get_hbn }

    @staticmethod
    def get(material : str, **kwargs):
        """
        Retrieves the material data object for the specified material. Additional keyword arguments are given to the corresponding material function.

        Args:
            material (str): The name of the material to retrieve.

        Returns:
            The data object associated with the specified material.

        Example:
            ```python
            graphene_data = MaterialCatalog.get('graphene')
            ```
        """
        return MaterialCatalog._materials[material](**kwargs)
    
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
