# TODO: 3D extension?
import pprint
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product

import jax
import jax.numpy as jnp
from matplotlib.path import Path

from . import _watchdog
from .orbitals import Orbital, OrbitalList

# TODO: hmmm
from ._plotting import _display_lattice_cut

# from granad._plotting import _display_lattice_cut
# from granad._orbitals import OrbitalList, Orbital
# from granad._shapes import *
# from granad import _watchdog
# import matplotlib.pyplot as plt


# TODO: the entire class is a big uff
@dataclass(frozen=True)
class Material2D:
    """positions must be fractional coordinates.
    lattice constant must be angström.
    """

    orbitals: dict
    hopping: dict
    coulomb: dict
    lattice_basis: list
    lattice_constant: float

    _materials = {
        "graphene": {
            "orbitals": [
                {
                    "position": (0, 0),
                    "shift": (0, 0, 0),
                    "attributes": (0, 1, 0, 0, "C"),
                    "tag": "sublattice_1",
                },
                {
                    "position": (-1 / 3, -2/3),
                    "shift": (0, 0, 0),
                    "attributes": (0, 1, 0, 0, "C"),
                    "tag": "sublattice_2",
                },
            ],
            "lattice_basis": [
                (1, 0, 0),  # First lattice vector
                (-1/2, jnp.sqrt(3)/2, 0),  # Second lattice vector
            ],
            # couplings are defined by combinations of orbital quantum numbers
            "hopping": {
                ((0, 1, 0, 0, "C"), (0, 1, 0, 0, "C")): ([0.0, 2.66], lambda d: 0j)
            },
            "coulomb": {
                ((0, 1, 0, 0, "C"), (0, 1, 0, 0, "C")): (
                    [16.522, 8.64, 5.333],
                    lambda d: 14.399 / d + 0j,
                )
            },
            "lattice_constant": 2.46,
        },
    }

    @classmethod
    def get(cls, material):
        return cls(**cls._materials[material])

    @staticmethod
    def available():
        available_materials = "\n".join(Material2D._materials.keys())
        print(f"Available materials:\n{available_materials}")

    def __post_init__(self):
        lattice_basis = jnp.array(self.lattice_basis)
        if len(self.lattice_basis) == 1:
            lattice_basis = jnp.array(self.lattice_basis + [(0, 0, 0)])
        object.__setattr__(self, "lattice_basis", lattice_basis)

    def _prune_neighbors(
        self, positions, minimum_neighbor_number, remaining_old=jnp.inf
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
            return self._prune_neighbors(
                positions[mask], minimum_neighbor_number, remaining
            )

    def _get_positions_in_lattice(self, positions_in_unit_cell, coefficients):
        shift = jnp.array(positions_in_unit_cell) @ self.lattice_basis
        return self.lattice_constant * (
            coefficients @ self.lattice_basis + shift[:, None, :]
        ).reshape(shift.shape[0] * coefficients.shape[0], 3)

    def _keep_matching_positions(self, positions, candidates):
        idxs = (
            jnp.round(jnp.linalg.norm(positions - candidates[:, None], axis=-1), 4) == 0
        ).nonzero()[0]
        return candidates[idxs]

    # very nasty, TODO: find better solution than rounding stuff, hardcoding 1e-5
    def _neighbor_couplings_to_function(
        self, couplings, outside_fun, attribute_combination
    ):
        matching_orbs = filter(
            lambda orb_spec: orb_spec["attributes"] in attribute_combination,
            self.orbitals,
        )
        fractional_positions = jnp.array([x["position"] for x in matching_orbs])
        n = len(couplings)
        repetitions = product(range(0, n), range(0, n))
        coefficients = jnp.array(list(repetitions))
        positions = self._get_positions_in_lattice(fractional_positions, coefficients)
        couplings = jnp.array(couplings) + 0.0j
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

    def _set_couplings(self, setter_func, cdict, group_id_to_attributes):
        for group_id_1, attributes_1 in group_id_to_attributes.items():
            for group_id_2, attributes_2 in group_id_to_attributes.items():
                try:
                    neighbor_couplings = cdict[(attributes_1, attributes_2)]
                except KeyError:
                    continue
                distance_func = self._neighbor_couplings_to_function(
                    *neighbor_couplings, (attributes_1, attributes_2)
                )
                setter_func(group_id_1, group_id_2, distance_func)

    # TODO: simplify vector determination
    def cut_orbitals(
            self, polygon_vertices, plot=False, minimum_neighbor_number: int = 2
    ):

        # shift the polygon into the positive xy plane
        min_values = jnp.min(polygon_vertices, axis=0)
        translation = jnp.where(min_values < 0, -min_values, 0)
        polygon_vertices += translation

        # we compute the maximum extension of the polyon
        max_values = jnp.max(polygon_vertices, axis=0)
        max_dim = max_values.argmax()
        max_vec = jnp.abs(self.lattice_basis)[:, max_dim].argmax()
        n = max_values[max_dim] / self.lattice_basis[max_vec, max_dim].item() / 2
        n_rounded = jnp.ceil(jnp.abs(n)) + 1
        n = int(jnp.sign(n) * n_rounded)
        repetitions = product(range(0, n), range(0, n))
        coefficients = jnp.array(list(repetitions))

        # get atom positions in the unit cell in fractional coordinates
        orbital_positions = jnp.array(
            [orbital["position"] for orbital in self.orbitals]
        )
        unit_cell_fractional_atom_positions = jnp.unique(
            jnp.round(orbital_positions, 6), axis=0
        )

        # get all atom positions in a plane completely covering the polygon
        initial_atom_positions = self._get_positions_in_lattice(
            unit_cell_fractional_atom_positions, coefficients
        )

        # get atom positions within the polygon
        polygon = Path(polygon_vertices)
        flags = polygon.contains_points(initial_atom_positions[:, :2])

        # get atom positions where every atom has at least minimum_neighbor_number neighbors
        final_atom_positions = self._prune_neighbors(
            initial_atom_positions[flags], minimum_neighbor_number
        )

        if plot == True:
            _display_lattice_cut(
                polygon_vertices, initial_atom_positions, final_atom_positions
            )
        return self._get_orbital_list(final_atom_positions, coefficients)

    def _get_orbital_list(self, final_atom_positions=None, coefficients=None):

        # groups together orbitals with identical quantum numbers
        orbitals_by_quantum_numbers = defaultdict(list)

        for orbital_spec in self.orbitals:
            # Use the attributes tuple as the key
            attr_key = tuple(orbital_spec["attributes"])
            # Append the dictionary to the list corresponding to the attributes key
            orbitals_by_quantum_numbers[attr_key].append(orbital_spec)

        # TODO: uff
        raw_list = []
        group_id_to_attributes = {}
        for attributes, orbital_properties_list in orbitals_by_quantum_numbers.items():

            # orbitals with identical quantum numbers get assigned the same group_id
            group_id = _watchdog._Watchdog.next_value()
            energy_level, angular_momentum, angular_momentum_z, spin, atom_name = (
                attributes
            )
            group_id_to_attributes[group_id] = attributes

            # a custom attribute, e.g. sublattice id may distinguish them
            layer_index = 0
            for orbital_property in orbital_properties_list:
                plane_orbital_positions = self._get_positions_in_lattice(
                    jnp.array([orbital_property["position"]]), coefficients
                )
                if final_atom_positions is not None:
                    plane_orbital_positions = self._keep_matching_positions(
                        final_atom_positions, plane_orbital_positions
                    )

                tag = orbital_property["tag"]
                for position in plane_orbital_positions:
                    orb = Orbital(
                        position=tuple(float(x) for x in position),
                        layer_index=layer_index,
                        tag=tag,
                        energy_level=energy_level,
                        angular_momentum=angular_momentum,
                        angular_momentum_z=angular_momentum_z,
                        spin=spin,
                        atom_name=atom_name,
                        group_id=group_id,
                    )
                    layer_index += 1
                    raw_list.append(orb)

        orbital_list = OrbitalList(raw_list)
        self._set_couplings(
            orbital_list.set_groups_hopping, self.hopping, group_id_to_attributes
        )
        self._set_couplings(
            orbital_list.set_groups_coulomb, self.coulomb, group_id_to_attributes
        )

        return orbital_list


# # TODO: the entire class is a big uff
# @dataclass( frozen  = True )
# class Material1D:
#     """positions must be fractional coordinates.
#     lattice constant must be angström.
#     """
#     orbitals : dict
#     hopping : dict
#     coulomb : dict
#     lattice_basis : list
#     lattice_constant : float

#     _materials = {
#         "ssh" : {
#             "orbitals": [
#                 { "position" : (0,), "shift" : (0,1,0), "attributes" : (0,1,0,0,None), "tag" : "sublattice_1" },
#                 { "position" : (0.8,), "shift" : (0,1,0),  "attributes" : (0,1,0,0,None), "tag" : "sublattice_2" }
#             ],
#             "lattice_basis": [
#                 (1.0, 0.0, 0.0)
#             ],
#             # couplings are defined by combinations of orbital quantum numbers
#             "hopping": { ((0,1,0,0,None), (0,1,0,0,None)) : ([0.0, 1., 0.5],lambda d: 0j)  },
#             "coulomb": {  ((0,1,0,0,None), (0,1,0,0,None)) : ([16.522, 8.64, 5.333],lambda d: 1/d+0j)  },
#             "lattice_constant" : 2.46},
#         "metallic_chain" : {
#             "orbitals": [
#                 { "position" : (0,), "shift" : (0,1,0), "attributes" : (0,1,0,0,None), "tag" : "" }
#             ],
#             "lattice_basis": [
#                 (1.0, 0.0, 0.0)
#             ],
#             # couplings are defined by combinations of orbital quantum numbers
#             "hopping": { ((0,1,0,0,None), (0,1,0,0,None)) : ([0.0, -2.66],lambda d: 0j)  },
#             "coulomb": {  ((0,1,0,0,None), (0,1,0,0,None)) : ([16.522, 8.64, 5.333],lambda d: 1/d+0j)  },
#             "lattice_constant" : 2.46},
#         }

#     @classmethod
#     def get(cls, material):
#         return cls( **cls._materials[material] )

#     @staticmethod
#     def available():
#         available_materials = '\n'.join( Material._materials.keys() )
#         print( f"Available materials:\n{available_materials}" )

#     def __post_init__(self):
#         lattice_basis = jnp.array( self.lattice_basis )
#         if len(self.lattice_basis) == 1:
#             lattice_basis = jnp.array( self.lattice_basis + [(0,0,0)] )
#         object.__setattr__(self, 'lattice_basis', lattice_basis)
