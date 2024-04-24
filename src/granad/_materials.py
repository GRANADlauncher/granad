import pprint
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product

import jax
import jax.numpy as jnp
from matplotlib.path import Path

from granad import _watchdog
from granad.orbitals import Orbital, OrbitalList
from granad._plotting import _display_lattice_cut


def _cut_flake_1d( material, unit_cells, plot=False):
    orbital_positions_uc =  material._get_positions_in_uc()
    grid = material._get_grid( [(0, unit_cells)] )
    orbital_positions = material._get_positions_in_lattice( orbital_positions_uc, grid )
    if plot:
        _display_lattice_cut( orbital_positions, orbital_positions )

    orbital_positions = jnp.unique( orbital_positions, axis = 0)        
    return material._get_orbital_list( orbital_positions, grid )

def _cut_flake_2d( material, polygon, plot=False, minimum_neighbor_number: int = 2):
    def _prune_neighbors(
            positions, minimum_neighbor_number, remaining_old=jnp.inf
    ):
        if minimum_neighbor_number <= 0:
            return positions
        distances = jnp.round(
            jnp.linalg.norm(positions - positions[:, None], axis=-1), 4
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

    # shift the polygon into the positive xy plane
    min_values = jnp.min(polygon, axis=0)
    translation = jnp.where(min_values < 0, -min_values, 0)
    polygon += translation 

    # we compute the maximum extension of the polyon
    max_values = jnp.max(polygon, axis=0)
    max_dim = max_values.argmax()
    max_vec = jnp.abs(material._lattice_basis)[material.periodic, max_dim].argmax()
    n = max_values[max_dim] / material._lattice_basis[material.periodic,:][max_vec, max_dim].item() / 2
    n_rounded = jnp.ceil(jnp.abs(n)) + 1
    n = int(jnp.sign(n) * n_rounded)
    grid = material._get_grid( [ (0,n), (0,n) ] )
    
    # get atom positions in the unit cell in fractional coordinates
    orbital_positions =  material._get_positions_in_uc()
    unit_cell_fractional_atom_positions = jnp.unique(
        jnp.round(orbital_positions, 6), axis=0
            )

    # get all atom positions in a plane completely covering the polygon
    initial_atom_positions = material._get_positions_in_lattice(
        unit_cell_fractional_atom_positions, grid
    )
    
    # get atom positions within the polygon
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

def _cut_flake_generic( material, grid_range ):
    orbital_positions_uc =  material._get_positions_in_uc()
    grid = material._get_grid( grid_range)
    orbital_positions = material._get_positions_in_lattice( orbital_positions_uc, grid )
    orbital_positions = jnp.unique( orbital_positions, axis = 0)        
    return material._get_orbital_list( orbital_positions, grid )
    
def _finalize(method):
    def wrapper(self, *args, **kwargs):
        for species in self.species.keys():
            self._species_to_groups[species] = _watchdog._Watchdog.next_value()
        if self.dim == 1:
            return _cut_flake_1d(self, *args, **kwargs)        
        elif self.dim == 2:
            return _cut_flake_2d(self, *args, **kwargs)
        else:
            return _cut_flake_generic(self, *args, **kwargs)
    return wrapper

class Material:
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
            description += f"  Lattice Basis: {self._lattice_basis}\n"
        
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
                    description += f"    Type: {type_}, Participants: {participants}, Couplings (neighbor, function): {coupling}\n"
        
        return description
    
    def lattice_constant(self, value):
        self.lattice_constant = value
        return self

    def lattice_basis(self, values, periodic = None):
        self._lattice_basis = jnp.array(values)
        total = set(range(len(self._lattice_basis)))        
        periodic = set(periodic) if periodic is not None else total
        self.periodic = list(periodic)
        self.finite = list(total - periodic)
        self.dim = len(self.periodic)                                              
        return self
    
    @_finalize
    def cut_flake( self ):
        pass

    def add_orbital(self, position, species, tag = ''):
        self.orbitals[species].append({'position': position, 'tag': tag})
        return self

    def add_orbital_species( self, name, n = 0, l = 0, m = 0, s = 0, atom  = ''):
        self.species[name] = (n,l,m,s,atom)
        return self

    def add_interaction(self, interaction_type, participants, parameters, expression = lambda x : 0j):
        self.interactions[interaction_type][participants] =  (parameters, lambda x : expression(x) + 0j)
        return self

    def _get_positions_in_uc( self, species = None ):
        if species is None:
            return jnp.array( [x["position"] for orb in list(self.orbitals.values()) for x in orb] )
        else:
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
        couplings = jnp.array(couplings) + 0.0j        
        grid = self._get_grid( [ (0, len(couplings)) for i in range(self.dim) ] )
        fractional_positions = self._get_positions_in_uc( species )
        positions = self._get_positions_in_lattice(fractional_positions, grid )
        
        distances = jnp.unique(
            jnp.round(jnp.linalg.norm(positions - positions[:, None, :], axis=2), 8)
        )[: len(couplings)]

        def inner(d):
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
        self._set_couplings(orbital_list.set_groups_hopping, "hopping")
        self._set_couplings(orbital_list.set_groups_coulomb, "coulomb")
        return orbital_list        
    
_graphene = (
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
        "hopping",
        participants=("pz", "pz"),
        parameters=[0.0, 2.66],
    )
    .add_interaction(
        "coulomb",
        participants=("pz", "pz"),
        parameters=[16.522, 8.64, 5.333],
        expression=lambda d: 14.399 / d
    )
)

_ssh = (
    Material("ssh")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
    ])
    .add_orbital_species("pz", l=1, atom='C')
    .add_orbital(position=(0,), tag="sublattice_1", species="pz")
    .add_orbital(position=(0.8,), tag="sublattice_2", species="pz")
    .add_interaction(
        "hopping",
        participants=("pz", "pz"),
        parameters=[0.0, 1 + 0.2, 1 - 0.2],
    )
    .add_interaction(
        "coulomb",
        participants=("pz", "pz"),
        parameters=[16.522, 8.64, 5.333],
        expression=lambda d: 14.399 / d
    )
)

_metal_1d = (
    Material("metal_1d")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
    ])
    .add_orbital_species("pz", l=1, atom='C')
    .add_orbital(position=(0,), tag="", species="pz")
    .add_interaction(
        "hopping",
        participants=("pz", "pz"),
        parameters=[0.0, 2.66],
    )
    .add_interaction(
        "coulomb",
        participants=("pz", "pz"),
        parameters=[16.522, 8.64, 5.333],
        expression=lambda d: 14.399 / d
    )
)

class MaterialCatalog:
    _materials = {"graphene" : _graphene, "ssh" : _ssh, "metal_1d" : _metal_1d }

    @staticmethod
    def get(material):
        return MaterialCatalog._materials[material]
    
    @staticmethod
    def describe(material):
        print(MaterialCatalog._materials[material])
    
    @staticmethod
    def available():
        available_materials = "\n".join(MaterialCatalog._materials.keys())
        print(f"Available materials:\n{available_materials}")
