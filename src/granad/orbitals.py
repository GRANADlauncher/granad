import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import wraps
from pprint import pformat
from typing import Callable, Optional, Union

import diffrax
import jax
import jax.numpy as jnp

from granad import _numerics, _plotting, _watchdog


@dataclass
class Orbital:
    """
    Represents the quantum state of an electron in an atom with specific properties.

    Attributes:
        position (jax.Array): The position of the orbital in space, initialized by default to a zero position.
                              This field is not used in hashing or comparison of instances.
        layer_index (Optional[int]): An optional index representing the layer of the orbital within its atom,
                                     may be None if not specified.
        tag (Optional[str]): An optional tag for additional identification or categorization of the orbital,
                             defaults to None.
        energy_level (Optional[int]): The principal quantum number indicating the energy level of the orbital,
                                      can be None.
        angular_momentum (Optional[int]): The quantum number representing the angular momentum of the orbital,
                                          optional and can be None.
        angular_momentum_z (Optional[int]): The magnetic quantum number related to the z-component of the orbital's
                                            angular momentum, optional.
        spin (Optional[int]): The spin quantum number of the orbital, indicating its intrinsic angular momentum,
                              optional and may be None.
        atom_name (Optional[str]): The name of the atom this orbital belongs to, can be None if not applicable.
        group_id (int): A group identifier for the orbital, automatically assigned by a Watchdog class
                        default factory method. For example, all pz orbitals in a single graphene flake get the same 
                        group_id.

    Key Functionality:
        The most important attributes of an orbtial are
         
        group_id (automatically generated, not recommended to be set it by the user)
        tag (user-defined or predefined for existing materials)
    """
    position: jax.Array = field(default_factory=lambda : jnp.array([0, 0, 0]), hash=False, compare=False)
    layer_index: Optional[int] = None
    tag: Optional[str] = None
    energy_level: Optional[int] = None
    angular_momentum: Optional[int] = None
    angular_momentum_z: Optional[int] = None
    spin: Optional[int] = None
    atom_name: Optional[str] = None
    group_id: _watchdog.GroupId = field(default_factory=_watchdog._Watchdog.next_value)

    def __post_init__(self):
        object.__setattr__(self, "position", jnp.array(self.position).astype(float))

    def __hash__(self):
        # Include only immutable fields in hash calculation
        return hash(
            (
                self.layer_index,
                self.tag,
                self.energy_level,
                self.angular_momentum,
                self.angular_momentum_z,
                self.angular_momentum,
                self.spin,
                self.atom_name,
                self.group_id.id,
            )
        )

    def __str__(self):
        return pformat(vars(self), sort_dicts=False)

    def __eq__(self, other):
        if not isinstance(other, Orbital):
            return NotImplemented
        return self.group_id == other.group_id and self.layer_index == other.layer_index

    def __lt__(self, other):
        if not isinstance(other, Orbital):
            return NotImplemented
        return self.group_id < self.group_id

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __ne__(self, other):
        return not self == other


class _SortedTupleDict(dict):

    def __getitem__(self, key):
        sorted_key = tuple(sorted(key))
        return super().__getitem__(sorted_key)

    def __setitem__(self, key, value):
        sorted_key = tuple(sorted(key))
        super().__setitem__(sorted_key, value)

    def __contains__(self, key):
        sorted_key = tuple(sorted(key))
        return super().__contains__(sorted_key)

    def group_id_items(self):
        """Yields items where all elements of the key tuple are group ids."""
        for key, value in self.items():
            if all(isinstance(k, _watchdog.GroupId) for k in key):
                yield (key, value)

    def orbital_items(self):
        """Yields items where all elements of the key tuple are orbitals."""
        for key, value in self.items():
            if all(isinstance(k, Orbital) for k in key):
                yield (key, value)
                
@dataclass
class Params:
    """
    A data class for storing parameters necessary for running a simulation involving electronic states and transitions.

    Attributes:
        excitation (jax.Array): from state, to state, excited electrons 
        eps (float): Numerical precision used for identifying degenerate eigenstates. Defaults to 1e-5.
        beta (float): Inverse temperature parameter (1/kT) used in thermodynamic calculations. Set to
                      `jax.numpy.inf` by default, implying zero temperature.
        self_consistency_params (dict): A dictionary to hold additional parameters required for self-consistency
                                        calculations within the simulation. Defaults to an empty dictionary.
        spin_degeneracy (float): Factor to account for the degeneracy of spin states in the simulation. Typically
                               set to 2, considering spin up and spin down. 
        electrons (Optional[int]): The total number of electrons in the structure. If not provided, it is assumed
                                   that the system's electron number needs to be calculated or is managed elsewhere.

    Note:
        This object should not be created directly, but is rather used to encapsulate (ephemeral) internal state
        of OrbitalList.
    """
    electrons : int
    excitation : list[jax.Array] = field(default_factory=lambda : [jnp.array([0]), jnp.array([0]), jnp.array([0])])
    eps : float = 1e-5
    beta : float = jnp.inf
    self_consistency_params : dict =  field(default_factory=dict)
    spin_degeneracy : float = 2.0

    def __add__( self, other ):
        if isinstance(other, Params):
            return Params( electrons = self.electrons + other.electrons )        
        raise ValueError
    
# rule: new couplings are registered here
@dataclass
class Couplings:
    hamiltonian : _SortedTupleDict = field(default_factory=_SortedTupleDict)
    coulomb : _SortedTupleDict = field(default_factory=_SortedTupleDict)
    dipole_transitions : _SortedTupleDict = field(default_factory=_SortedTupleDict)

    def __add__( self, other ):
        if isinstance(other, Couplings):
            return Couplings()
        raise ValueError        
                
@dataclass
class TDResult:
    illumination : Callable
    time_axis : jax.Array
    final_density_matrix : jax.Array
    output : list[jax.Array]

    def ft_output( self, omega_max, omega_min ):
        ft = lambda o : _numerics.get_fourier_transform(self.time_axis, o, omega_max, omega_min, False)
        return [ft(o) for o in self.output]

    def ft_illumination( self, omega_max, omega_min, return_omega_axis = True ):
        return _numerics.get_fourier_transform(self.time_axis, self.td_illumination, omega_max, omega_min, return_omega_axis)

    @property
    def td_illumination( self ):
        return jax.vmap(self.illumination)(self.time_axis)

    
def mutates(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._recompute = True
        return func(self, *args, **kwargs)    

    return wrapper


def recomputes(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._recompute:
            self._build()
            self._recompute = False
        return func(self, *args, **kwargs)

    return wrapper

def plotting_methods(cls):
    # attach all plotting methods
    for name in dir(_plotting):
        method = getattr(_plotting, name)
        if callable(method) and name.startswith("show"):
            setattr(cls, name, method)            
    return cls
    
@plotting_methods
class OrbitalList:
    """
    A class that encapsulates a list of orbitals, providing an interface similar to a standard Python list,
    while also maintaining additional functionalities for coupling orbitals and managing their relationships.

    The class stores orbitals in a wrapped Python list and handles the coupling of orbitals using dictionaries,
    where the keys are tuples of orbital identifiers (orb_id), and the values are the couplings (either a float
    or a function representing the coupling strength or mechanism between the orbitals).

    The class also stores simulation parameters like the number of electrons and temperature in a dataclass.
    
    The class computes physical observables (energies etc) lazily on the fly, when they are needed. If there is 
    a basis (either site or energy) to reasonably associate with a quantity, the class exposes quantity_x as an attribute
    for the site basis and quantity_e as an attribute for the energy basis. By default, all quantities are in site basis, so
    quantity_x == quantity.

    The class exposes simulation methods.
     
    Attributes:
        orbitals (list): The underlying list that stores the orbitals.
        couplings (dict): A dictionary where keys are tuples of orbital identifiers and values are the couplings
                          (either float values or functions).

    Key Functionalities:
        - **Orbital Identification**: Orbitals can be identified either by their group_id, a direct
          reference to the orbital object itself, or via a user-defined tag.
        - **Index Access**: Orbitals can be accessed and managed by their index in the list, allowing for
          list-like manipulation (addition, removal, access).
        - **Coupling Definition**: Allows for the definition and adjustment of couplings between pairs of orbitals,
          identified by a tuple of their respective identifiers. These couplings can dynamically represent the
          interaction strength or be a computational function that defines the interaction.

    Note:
        The coupling values can be dynamically modified. When two orbital lists are added, their couplings are merged, 
        and their simulation parameters are wiped.
    """
    def __init__( self, orbs = None, couplings = None, params = None, recompute = True):
        self._list = orbs if orbs is not None else []
        self.couplings = couplings if couplings is not None else Couplings()
        self.params = params if params is not None else Params( len(orbs) )
        self._recompute = recompute    
        
    def __getattr__(self, property_name):
        if property_name.endswith("_x"):
            return getattr(self, property_name[:-2])
        elif property_name.endswith("_e"):
            return self.transform_to_energy_basis( getattr(self, property_name[:-2]) )

    def __len__(self):
        return len(self._list)

    # can't mutate, because orbitals are immutable
    def __getitem__(self, position):
        return self._list[position]

    def __repr__(self):
        return repr(self._list)

    def __str__(self):
        info = f"List with {len(self)} orbitals, {self.electrons} electrons."
        groups = "\n".join(
            [
                f"group id {key} : {val} orbitals"
                for key, val in Counter(self.get_group_ids()).items()
            ]
        )
        return "\n".join((info, groups)) + "\n" + str(self._list)

    def __iter__(self):
        return iter(self._list)

    def __add__(self, other):
        if not self._are_orbs(other):
            raise TypeError

        if any(orb in other for orb in self._list):
            raise ValueError

        if isinstance(other, OrbitalList):
            new_list = (self._list + list(other)).copy()
            new_couplings = self.couplings + other.couplings 
            new_params = self.params + other.params 
            
        return OrbitalList( new_list, new_couplings, new_params )

    @mutates
    def __setitem__(self, position, value):
        if isinstance(value, Orbital):
            self._list[position] = value
        raise TypeError

    def _delete_coupling(self, orb, coupling):
        keys_to_remove = [key for key in coupling if orb in key]
        for key in keys_to_remove:
            del coupling[key]

    @mutates
    def __delitem__(self, position):
        orb = self._list[position]
        self._delete_coupling(orb, self.coupling.hamiltonian)
        self._delete_coupling(orb, self.couplings.coulomb)
        del self._list[position]

    @staticmethod
    def _are_orbs(candidate):
        return all(isinstance(orb, Orbital) for orb in candidate)
    
    @mutates
    def _set_coupling(self, orb1, orb2, val_or_func, coupling):
        for o1 in orb1:
            for o2 in orb2:
                coupling[(o1, o2)] = val_or_func

    def _hamiltonian_coulomb(self):

        def fill_matrix(matrix, coupling_dict):

            dummy = jnp.arange(len(self))
            triangle_mask = dummy[:, None] >= dummy

            # TODO: in principle we can build a big tensor NxNxgroups, vmap over the last axis and sum the groups
            # first, we loop over all group_id couplings => interactions between groups
            for key, function in coupling_dict.group_id_items():
                # TODO:  big uff:  we rely on the correct ordering of the group_ids for cols and rows, first key is always smaller than last keys => we get upper triangular valid indices
                # if it were the other way around, these would be zeroed by the triangle mask
                cols = group_ids == key[0].id
                rows = (group_ids == key[1].id)[:, None]
                combination_indices = jnp.logical_and(rows, cols)
                valid_indices = jnp.logical_and(triangle_mask, combination_indices)
                function = jax.vmap(function)
                matrix = matrix.at[valid_indices].set(
                    function(distances[valid_indices])
                )

            # we now set single elements
            rows, cols, vals = [], [], []
            for key, val in coupling_dict.orbital_items():
                rows.append(self._list.index(key[0]))
                cols.append(self._list.index(key[1]))
                vals.append(val)

            matrix = matrix.at[rows, cols].set(vals)

            return matrix + matrix.conj().T - jnp.diag(jnp.diag(matrix))

        # TODO: rounding
        positions = self.positions
        distances = jnp.round(
            jnp.linalg.norm(positions - positions[:, None], axis=-1), 6
        )
        group_ids = jnp.array( [orb.group_id.id for orb in self._list] )

        hamiltonian = fill_matrix(
            jnp.zeros((len(self), len(self))).astype(complex), self.couplings.hamiltonian
        )
        coulomb = fill_matrix(
            jnp.zeros((len(self), len(self))).astype(complex), self.couplings.coulomb
        )
        return hamiltonian, coulomb

    # TODO: abstract this boilerplate away
    @mutates
    def set_dipole_element(self, orb1, orb2, arr):
        """
        Sets a dipole transition for specified orbital or index pairs.

        Parameters:
            orb_or_index1 (int or Orbital): Identifier or orbital for the first part of the transition.
            orb_or_index2 (int or Orbital): Identifier or orbital for the second part of the transition.
            arr (jax.Array): The 3-element array containing dipole transition elements.
        """
        self._set_coupling(self.filter_orbs(orb1, int), self.filter_orbs(orb2, int), arr, self.couplings.dipole_transitions)
        
    def set_hamiltonian_groups(self, orb1, orb2, func):
        """
        Sets the hamiltonian coupling between two groups of orbitals.

        Parameters:
            orb_or_group_id1 (int or Orbital): Identifier or orbital for the first group.
            orb_or_group_id2 (int or Orbital): Identifier or orbital for the second group.
            func (callable): Function that defines the hamiltonian interaction.

        Notes:
            The function `func` should be complex-valued.
        """
        self._set_coupling(
            self.filter_orbs(orb1, _watchdog.GroupId), self.filter_orbs(orb2, _watchdog.GroupId), self._ensure_complex(func), self.couplings.hamiltonian
        )

    def set_coulomb_groups(self, orb1, orb2, func):
        """
        Sets the Coulomb coupling between two groups of orbitals.

        Parameters:
            orb_or_group_id1 (int or Orbital): Identifier or orbital for the first group.
            orb_or_group_id2 (int or Orbital): Identifier or orbital for the second group.
            func (callable): Function that defines the Coulomb interaction.

        Notes:
            The function `func` should be complex-valued.
        """
        self._set_coupling(
            self.filter_orbs(orb1, _watchdog.GroupId), self.filter_orbs(orb2, _watchdog.GroupId), self._ensure_complex(func), self.couplings.coulomb
        )

    def set_hamiltonian_element(self, orb1, orb2, val):
        """
        Sets an element of the Hamiltonian matrix between two orbitals or indices.

        Parameters:
            orb_or_index1 (int or Orbital): Identifier or orbital for the first element.
            orb_or_index2 (int or Orbital): Identifier or orbital for the second element.
            val (complex): The complex value to set for the Hamiltonian element.
        """
        self._set_coupling(self.filter_orbs(orb1, int), self.filter_orbs(orb2, int), self._ensure_complex(val), self.couplings.hamiltonian)

    def set_coulomb_element(self, orb1, orb2, val):
        """
        Sets a Coulomb interaction element between two orbitals or indices.

        Parameters:
            orb_or_index1 (int or Orbital): Identifier or orbital for the first element.
            orb_or_index2 (int or Orbital): Identifier or orbital for the second element.
            val (complex): The complex value to set for the Coulomb interaction element.
        """
        self._set_coupling(self.filter_orbs(orb1, int), self.filter_orbs(orb2, int), self._ensure_complex(val), self.couplings.coulomb)

    def _ensure_complex(self, func_or_val):
        if callable(func_or_val):
            return lambda x: func_or_val(x) + 0.0j
        if isinstance(func_or_val, (int, float, complex)):
            return func_or_val + 0.0j
        raise TypeError

    def _build(self):

        assert len(self) > 0

        self._hamiltonian, self._coulomb = self._hamiltonian_coulomb()

        # TODO: there is something weird happening here!        
        self._eigenvectors, self._energies = jax.lax.linalg.eigh(self._hamiltonian)

        self._initial_density_matrix = _numerics._density_matrix(
            self._energies,
            self.params.electrons,
            self.params.spin_degeneracy,
            self.params.eps,
            self.params.excitation,
            self.params.beta,
        )
        self._stationary_density_matrix = _numerics._density_matrix(
            self._energies,
            self.params.electrons,
            self.params.spin_degeneracy,
            self.params.eps,
            Params(0).excitation,
            self.params.beta,
        )

        if self.params.self_consistency_params:
            (
                self._hamiltonian,
                self._initial_density_matrix,
                self._stationary_density_matrix,
                self._energies,
                self._eigenvectors,
            ) = _get_self_consistent(
                self._hamiltonian,
                self._coulomb,
                self.positions,
                self.params.spin_degeneracy,
                self.params.electrons,
                self.params.eps,
                self._eigenvectors,
                self._static_density_matrix,
                **self.params.self_consistent_params,
            )

        self._initial_density_matrix = self.transform_to_site_basis( self._initial_density_matrix )

        self._stationary_density_matrix = self.transform_to_site_basis( self._stationary_density_matrix )
            
    def set_open_shell( self ):
        if any( orb.spin is None for orb in self._list ):
            raise ValueError
        self.simulation_params.spin_degeneracy = 1.0

    def set_closed_shell( self ):
        self.simulation_params.spin_degeneracy = 2.0
        
    def index(self, orb):
        return self._list.index(orb)
        
    @mutates
    def append(self, other):
        """
        Appends an orbital to the list, ensuring it is not already present.

        Parameters:
            other (Orbital): The orbital to append.

        Raises:
            TypeError: If `other` is not an instance of Orbital.
            ValueError: If `other` is already in the list.
        """
        if not isinstance(other, Orbital):
            raise TypeError
        if other in self:
            raise ValueError
        self._list.append(other)

    # TODO: replace if else with action map
    def filter_orbs( self, orb_id, t ):
        if type(orb_id) == t:
            return [orb_id]
        
        # index to group, orb, tag => group_id / orb / tag at index,
        if isinstance(orb_id, int) and isinstance(t, _watchdog.GroupId):
            return [self._list[orb_id].group_id]
        if isinstance(orb_id, int) and isinstance(t, Orbital):
            return [self._list[orb_id]]
        if isinstance(orb_id, int) and isinstance(t, str):
            return [self._list[orb_id].tag]

        # group to index, orb, tag => group_id / orb / tag at index,
        if isinstance(orb_id, _watchdog.GroupId) and isinstance(t, str):
            return [ orb.tag for orb in self if orb.group_id == orb_id ]
        if isinstance(orb_id, _watchdog.GroupId) and isinstance(t, group_id):
            return [ orb.group_id for orb in self if orb.group_id == orb_id ]
        if isinstance(orb_id, _watchdog.GroupId) and isinstance(t, int):
            return [ self._list.index(orb) for orb in self if orb.group_id == orb_id ]

        # tag to group, orb, index => group_id / orb / tag at index,
        if isinstance(orb_id, str) and isinstance(t, _watchdog.GroupId):
            return [orb.group_id for orb in self if orb.tag == orb_id]
        if isinstance(orb_id, str) and isinstance(t, int):
            return [self._list.index(orb) for orb in self if orb.tag == orb_id]
        if isinstance(orb_id, str) and isinstance(t, Orbital):
            return [orb for orb in self if orb.tag == orb_id]

        # orb to index, group, tag
        if isinstance(orb_id, Orbital) and isinstance(t, _watchdog.GroupId):
            return [orb_id.group_id]
        if isinstance(orb_id, Orbital) and isinstance(t, int):
            return [self._list.index(orb_id)]
        if isinstance(orb_id, Orbital) and isinstance(t, str):
            return [orb_id.tag]

        
    @mutates
    def shift_by_vector(self, orb_id, translation_vector):
        """
        Shifts all orbitals with a specific tag by a given vector.

        Parameters:
            tag_or_group_id (str or int or list[int]): The tag, group_id to match orbitals.
            translation_vector (jax.Array): The vector by which to translate the orbital positions.

        Notes:
            This operation mutates the positions of the matched orbitals.
        """
        for orb in self.get( orb_id, Orbital ):
            orb.position += jnp.array(translation_vector)

    @mutates
    def set_position(self, orb_id, position):
        """
        Sets the position of all orbitals with a specific tag.

        Parameters:
            tag (str): The tag to match orbitals.
            position (jax.Array): The vector at which to move the orbitals

        Notes:
            This operation mutates the positions of the matched orbitals.
        """
        for orb in self.filter_orbs( orb_id, Orbital ):
            orb.position = position

            
    @mutates
    def set_self_consistent(self, sc_params):
        """
        Configures the list for self-consistent field calculations.

        Parameters:
            sc_params (dict): Parameters for self-consistency.
        """
        self.self_consistency_params = sc_params

    @mutates
    def set_excitation(self, from_state, to_state, excited_electrons):
        """
        Sets up an excitation process from one state to another with specified electrons.

        Parameters:
            from_state (int, list, or jax.Array): The initial state index or indices.
            to_state (int, list, or jax.Array): The final state index or indices.
            excited_electrons (int, list, or jax.Array): The indices of electrons to be excited.

        Notes:
            The states and electron indices may be specified as scalars, lists, or arrays.
        """
        def maybe_int_to_arr(maybe_int):
            if isinstance(maybe_int, int):
                return jnp.array([maybe_int])
            if isinstance(maybe_int, list):
                maybe_int = jnp.array(maybe_int)
            if isinstance(maybe_int, jax.Array):
                return (
                    jnp.array(maybe_int)
                    if maybe_int.ndim > 1
                    else jnp.array([maybe_int])
                )
            raise TypeError

        self.params.excitation = [maybe_int_to_arr(from_state), maybe_int_to_arr(to_state), maybe_int_to_arr(excited_electrons)]
        
    @property
    def positions(self):
        return jnp.array([orb.position for orb in self._list])

    # TODO: too verbose
    @property
    def electrons( self ):
        return self.params.electrons

    @mutates
    def set_electrons( self, val ):
        self.params.electrons = val

    @property
    def spin_degeneracy( self ):
        return self.params.spin_degeneracy
        
    @property
    @recomputes
    def homo(self):
        return (self.electrons * self.stationary_density_matrix_e).real.diagonal().round(2).nonzero()[0][-1].item()
    
    @property
    @recomputes
    def eigenvectors(self):
        return self._eigenvectors

    @property
    @recomputes
    def energies(self):
        return self._energies

    @property
    @recomputes
    def hamiltonian(self):
        return self._hamiltonian

    @property
    @recomputes
    def coulomb(self):
        return self._coulomb
    
    @property
    @recomputes
    def initial_density_matrix(self):
        return self._initial_density_matrix
    
    @property
    @recomputes
    def stationary_density_matrix(self):
        return self._stationary_density_matrix
    
    @property
    @recomputes
    def quadrupole_operator(self):
        """
        Calculates the quadrupole operator based on the dipole operator terms. It combines products of the dipole terms and their differences from the identity matrix scaled by the diagonal components.

        Returns:
           jax.Array: A tensor representing the quadrupole operator.
        """

        dip = self.dipole_operator
        term = jnp.einsum("ijk,jlm->ilkm", dip, dip)
        diag = jnp.einsum("ijk,jlk->il", dip, dip)
        diag = jnp.einsum("ij,kl->ijkl", diag, jnp.eye(term.shape[-1]))
        return 3 * term - diag
    
    @property
    @recomputes
    def dipole_operator(self):
        """
        Computes the dipole operator using positions and transition values. The diagonal is set by position components, and the off-diagonal elements are set by transition matrix values.

        Returns:
           jax.Array: A 3D tensor representing the dipole operator, symmetrized and complex conjugated.
        """

        N = self.positions.shape[0]
        dipole_operator = jnp.zeros((3, N, N)).astype(complex)
        for i in range(3):
            dipole_operator = dipole_operator.at[i, :, :].set(
                jnp.diag(self.positions[:, i] / 2)
            )
        for orbital_combination, value in self.couplings.dipole_transitions.items():
            i, j = self._list.index(orbital_combination[0]), self._list.index(
                orbital_combination[1]
            )
            k = value.nonzero()[0]
            dipole_operator = dipole_operator.at[k, i, j].set(value[k])
        return dipole_operator + jnp.transpose(dipole_operator, (0, 2, 1)).conj()
    
    @property
    @recomputes
    def velocity_operator(self):
        """
        Calculates the velocity operator as the commutator of position with the Hamiltonian using matrix multiplications.
        
        Returns:
           jax.Array: A tensor representing the velocity operator, computed as a differential of position and Hamiltonian.
        """

        if self.couplings.dipole_transitions is None:
            x_times_h = jnp.einsum("ij,iL->ijL", self._hamiltonian, self.positions)
            h_times = jnp.einsum("ij,jL->ijL", self._hamiltonian, self.positions)
        else:
            positions = self.dipole_operator
            x_times_h = jnp.einsum("kj,Lik->Lij", self._hamiltonian, positions)
            h_times = jnp.einsum("ik,Lkj->Lij", self._hamiltonian, positions)
        return -1j * (x_times_h - h_times)

    @property
    @recomputes
    def transition_energies(self):
        """
        Computes independent-particle transition energies associated with the TB-Hamiltonian of a stack.

        Returns:
           jax.Array: The element `arr[i,j]` contains the transition energy from `i` to `j`.
        """
        return self._energies[:, None] - self._energies

    @property
    @recomputes
    def wigner_weisskopf_transition_rates(self):
        """
        Calculates Wigner-Weisskopf transition rates based on transition energies and dipole moments transformed to the energy basis.
        
        Returns:
           jax.Array: The element `arr[i,j]` contains the transition rate from `i` to `j`.
        """
        charge = 1.602e-19
        eps_0 = 8.85 * 1e-12
        hbar = 1.0545718 * 1e-34
        c = 3e8  # 137 (a.u.)
        factor = 1.6e-29 * charge / (3 * jnp.pi * eps_0 * hbar**2 * c**3)
        te = self.transition_energies
        transition_dipole_moments = self.dipole_operator_e
        return (
            (te * (te > self.eps)) ** 3
            * jnp.squeeze(transition_dipole_moments**2)
            * factor
        )

    @staticmethod
    def _transform_basis(observable, vectors):
        dims_einsum_strings = {2: "ij,jk,lk->il", 3: "ij,mjk,lk->mil"}
        einsum_string = dims_einsum_strings[(observable.ndim)]
        return jnp.einsum(einsum_string, vectors, observable, vectors.conj())

    def transform_to_site_basis(self, observable):
        """
        Transforms an observable to the site basis using eigenvectors of the system.

        Parameters:
           observable (jax.Array): The observable to transform.

        Returns:
           jax.Array: The transformed observable in the site basis.
        """
        return self._transform_basis(observable, self._eigenvectors)

    def transform_to_energy_basis(self, observable):
        """
        Transforms an observable to the energy basis using the conjugate transpose of the system's eigenvectors.

        Parameters:
           observable (jax.Array): The observable to transform.

        Returns:
           jax.Array: The transformed observable in the energy basis.
        """

        return self._transform_basis(observable, self._eigenvectors.conj().T)

    @recomputes
    def get_charge(density_matrix = None):
        """
        Calculates the charge distribution from a given density matrix or from the initial density matrix if not specified.

        Parameters:
           density_matrix (jax.Array, optional): The density matrix to use for calculating charge.

        Returns:
           jax.Array: A diagonal array representing charges at each site.
        """
        if density_matrix is None:
            return jnp.diag(
                self.initial_density_matrix
                * self.electrons
            )
        else:
            return jnp.diag(density_matrix * self.electrons)

    @recomputes
    def get_dos(self, omega: float, broadening: float = 0.1):
        """
        Calculates the density of states (DOS) of a nanomaterial stack at a given frequency with broadening.

        Parameters:
           omega (float): The frequency at which to evaluate the DOS.
           broadening (float, optional): The numerical broadening parameter to replace Dirac Deltas.

        Returns:
           float: The integrated density of states at the specified frequency.
        """

        broadening = 1 / broadening
        prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
        gaussians = jnp.exp(-((self._energies - omega) ** 2) / 2 * broadening**2)
        return prefactor * jnp.sum(gaussians)

    @recomputes
    def get_ldos(self, omega: float, site_index: int, broadening: float = 0.1):
        """
        Calculates the local density of states (LDOS) at a specific site and frequency within a nanomaterial stack.

        Parameters:
           omega (float): The frequency at which to evaluate the LDOS.
           site_index (int): The site index to evaluate the LDOS at.
           broadening (float, optional): The numerical broadening parameter to replace Dirac Deltas.

        Returns:
           float: The local density of states at the specified site and frequency.
        """

        broadening = 1 / broadening
        weight = jnp.abs(self._eigenvectors[site_index, :]) ** 2
        prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
        gaussians = jnp.exp(-((self._energies - omega) ** 2) / 2 * broadening**2)
        return prefactor * jnp.sum(weight * gaussians)

    @recomputes
    def get_epi(self, density_matrix_stat: jax.Array, omega: float, epsilon: float = None) -> float:
        """
        Calculates the energy-based plasmonicity index (EPI) for a given density matrix and frequency.

        Parameters:
           density_matrix_stat (jax.Array): The density matrix to consider for EPI calculation.
           omega (float): The frequency to evaluate the EPI at.
           epsilon (float, optional): The small imaginary part to stabilize the calculation, defaults to internal epsilon if not provided.
        
        Returns:
           float: The EPI.
        """

        epsilon = epsilon if epsilon is not None else self.eps
        density_matrix_stat_without_diagonal = jnp.abs(density_matrix_stat - jnp.diag(jnp.diag(density_matrix_stat)))
        density_matrix_stat_normalized = density_matrix_stat_without_diagonal / jnp.linalg.norm(density_matrix_stat_without_diagonal)
        te = self.transition_energies
        excitonic_transitions = (
            density_matrix_stat_normalized / (te * (te > self.eps) - omega + 1j * epsilon) ** 2
        )
        return 1 - jnp.sum(jnp.abs(excitonic_transitions * density_matrix_stat_normalized)) / (
            jnp.linalg.norm(density_matrix_stat_normalized) * jnp.linalg.norm(excitonic_transitions)
        )

    @recomputes
    def get_induced_field(self, positions: jax.Array, density_matrix):
        """
        Calculates the induced electric field at specified positions based on a given density matrix.

        Parameters:
           positions (jax.Array): The positions at which to evaluate the induced field.
           density_matrix (jax.Array): The density matrix used to calculate the induced field.

        Returns:
           jax.Array: The resulting electric field vector at each position.
        """


        # distance vector array from field sources to positions to evaluate field on
        vec_r = self.positions[:, None] - positions

        # scalar distances
        denominator = jnp.linalg.norm(vec_r, axis=2) ** 3

        # normalize distance vector array
        point_charge = jnp.nan_to_num(
            vec_r / denominator[:, :, None], posinf=0.0, neginf=0.0
        )

        # compute charge via occupations in site basis
        charge = self.electrons * density_matrix.real

        # induced field is a sum of point charges, i.e. \vec{r} / r^3
        e_field = 14.39 * jnp.sum(point_charge * charge[:, None, None], axis=0)
        return e_field        
    
    def get_expectation_value(self, *, operator, density_matrix, induced = True):
        """
        Calculates the expectation value of an operator with respect to a given density matrix using tensor contractions specified for different dimensionalities of the input arrays.

        Parameters:
           operator (jax.Array): The operator for which the expectation value is calculated.
           density_matrix (jax.Array): The density matrix representing the state of the system.

        Returns:
           jax.Array: The calculated expectation value(s) depending on the dimensions of the operator and the density matrix.
        """

        dims_einsum_strings = {
            (3, 2): "ijk,kj->i",
            (3, 3): "ijk,lkj->li",
            (2, 3): "ij,kji->k",
            (2, 2): "ij,ji->",
        }
        correction = self.stationary_density_matrix_x if induced == True else 0
        return self.electrons * jnp.einsum(
            dims_einsum_strings[(operator.ndim, density_matrix.ndim)],
            operator,
            correction - density_matrix,
        )


    # TODO: rewrite _td functions
    def _td_postprocessing_func_list(self, expectation_values, density_matrix, custom_computation ):
        """Builds the list of funtions to be applied to the density matrix in postprocessing.
        """
        
        computation = []
        if expectation_values is not None:
            if isinstance(expectation_values, jax.Array):
                expectation_values = [expectation_values]
            ops = jnp.concatenate( expectation_values)
            computation.append( lambda rho : self.get_expectation_value(operator = ops, density_matrix = rho) )
        if density_matrix is not None:            
            if isinstance(density_matrix, str):
                density_matrix = [density_matrix]
            for option in density_matrix:
                if option == "occ_x":
                    computation.append( lambda rho : jnp.diagonal(rho, axis1=-1, axis2=-2) )
                elif option == "occ_e":
                    computation.append( lambda rho : jnp.diagonal(rho, axis1=-1, axis2=-2) )
                elif option == "full":
                    computation.append( lambda rho : rho )
        if custom_computation is not None:
            if callable(custom_computation):
                custom_computation = [custom_computation]
            computation.extend( custom_computation )
        assert len(computation) > 0, "Specify what to compute!"
        return [jax.jit(f) for f in computation]

    def _td_time_axis( self, grid, start_time, end_time, dt, max_mem_gb):
        # if grid is an array, sample these times, else subsample time axis
        time_axis = grid
        if not isinstance(grid, jax.Array):
            steps_time = jnp.ceil( (end_time - start_time) / dt ).astype(int)
            time_axis = jnp.linspace(start_time, end_time, steps_time)[::grid]
        # number of rhos in a single RAM batch 
        size = (self.initial_density_matrix.size * self.initial_density_matrix.itemsize) / 1e9
        matrices_per_batch = jnp.floor( max_mem_gb / size  ).astype(int).item()
        assert matrices_per_batch > 0, "Density matrix exceeds allowed max memory."
        # batch time axis accordingly
        splits = jnp.ceil(time_axis.size /  matrices_per_batch ).astype(int).item()
        return jnp.array_split(time_axis, [matrices_per_batch * i for i in range(1, splits)] )

    def _td_dynamic_functions( self, relaxation_rate, illumination, saturation ):
        relaxation_rate = relaxation_rate if relaxation_rate is not None else 0.0                        
        if callable(relaxation_rate):
            relaxation_function = relaxation_rate
        elif isinstance(relaxation_rate, jax.Array):
            # TODO: check leak
            relaxation_function = _numerics.saturation_lindblad( self.eigenvectors, relaxation_rate, saturation, self.electrons )
        else:
            relaxation_function = _numerics.decoherence_time( relaxation_rate )

        if illumination is None:
            illumination = lambda t : jnp.array([0.,0.,0.])            
        if not callable(illumination):
            raise TypeError("Provide a function for e-field")

        return illumination, relaxation_function
    
    @recomputes
    def td_run(            
            self,
            *,
            end_time : float,
            start_time : float = 0.0,
            dt : float = 1e-4,
            grid : int = 100,
            max_mem_gb : float = 0.5,

            initial_density_matrix : Optional[jax.Array] = None,

            coulomb_strength : float = 1.0,
            
            illumination : Callable = None,
            
            saturation : Callable = None,
            relaxation_rate : Union[float, jax.Array, Callable] = None,

            compute_at : Optional[jax.Array] = None,

            expectation_values : Optional[list[jax.Array]] = None,
            density_matrix : Optional[list[str]] = None,
            computation : Optional[Callable] = None,

            use_rwa : bool = False,

            solver = diffrax.Dopri5(),
            stepsize_controller = diffrax.PIDController(rtol=1e-10,atol=1e-10),

            dry_run : bool = False,
    ):

        """
        Simulates the time evolution of the density matrix for a given system under specified conditions and external fields.

        Parameters:

        Returns:
            ResultTD
        """

        # external functions
        illumination, relaxation = self._td_dynamic_functions(relaxation_rate, illumination, saturation)

        # batched time axis 
        time_axis = self._td_time_axis(grid, start_time, end_time, dt, max_mem_gb)
        print(f"Batched time axis {time_axis.shape}")

        # applied to density matrix batch
        pp_fun_list = self._td_postprocessing_func_list(expectation_values, density_matrix, computation)
        
        # coulomb field propagator, if any
        coulomb_field_to_from = _numerics.get_coulomb_field_to_from(
            self.positions, self.positions, compute_at
        )

        # we start with the flake dm if nothing else is supplied
        initial_density_matrix = self.initial_density_matrix if initial_density_matrix is None else initial_density_matrix        

        ## integrate
        final_density_matrix, output = _numerics.integrate_master_equation(
            self.hamiltonian,
            self.coulomb * coulomb_strength,
            self.dipole_operator,
            self.electrons,
            self.velocity_operator,
            self.initial_density_matrix,
            self.stationary_density_matrix,
            time_axis,
            illumination,
            relaxation,
            coulomb_field_to_from,
            use_rwa,
            solver,
            stepsize_controller,
            dt,    
            pp_fun_list,
        )
        
        return TDResult(
            time_axis = jnp.concatenate(time_axis),
            illumination = illumination,
            final_density_matrix = final_density_matrix,
            output = output,
        )

    # TODO: decouple rpa numerics from orbital datataype
    def get_polarizability_rpa(
        self,
        omegas,            
        relaxation_rate,
        polarization,
        coulomb_strength=1.0,
        hungry=0,
        phi_ext=None,
    ):
        """
        Calculates the random phase approximation (RPA) polarizability of the system at given frequencies under specified conditions.

        Parameters:
           omegas (jax.Array): Frequencies at which to calculate polarizability. If given as an nxm array, this function will be applied vectorized to the batches given by the last axis in omegas.
           relaxation_rate (float): The relaxation time parameter.
           polarization (jax.Array): Polarization directions or modes.
           coulomb_strength (float): The strength of Coulomb interaction in the calculations.
           hungry (int): speed up the simulation up, higher numbers (max 2) increase RAM usage.
           phi_ext (Optional[jax.Array]): External potential influences, if any.

        Returns:
           jax.Array: The calculated polarizabilities at the specified frequencies.
        """

        alpha = _numerics.rpa_polarizability_function(
            self, relaxation_rate, polarization, coulomb_strength, phi_ext, hungry
        )
        if omegas.ndim == 1:        
            return jax.lax.map(alpha, omegas)
        else:
            return jnp.concatenate( [ jax.vmap(alpha)(omega) for omega in omegas ] )

    def get_susceptibility_rpa(
            self, omegas, relaxation_rate, coulomb_strength=1.0, hungry=0
    ):
        """
        Computes the random phase approximation (RPA) susceptibility of the system over a range of frequencies.

        Parameters:
           omegas (jax.Array): The frequencies at which to compute susceptibility.
           relaxation_rate (float): The relaxation time affecting susceptibility calculations.
           coulomb_strength (float): The strength of Coulomb interactions considered in the calculations.
           hungry (int): speed up the simulation up, higher numbers (max 2) increase RAM usage.

        Returns:
           jax.Array: The susceptibility values at the given frequencies.
        """

        sus = _numerics.rpa_polarizability_function(
            self, relaxation_rate, coulomb_strength, hungry
        )
        return jax.lax.map(sus, omegas)    

    def get_mean_field_hamiltonian( self, overlap = None ):
        """convert an orbital list to a set of parameters usable for the rhf procedure. 
        currently, only an empirical direct channel interaction specified specified in a 
        the list's coulomb dict is taken into account.
        """
        # Since we consider <1i|U|i1> => U_{11ii}
        eri = self.coulomb[None, None]
        overlap = overlap if overlap is not None else jnp.eye(self.hamiltonian.shape[0])
        return _rhf( self.hamiltonian, eri, overlap, self.electrons )
    
    @property
    def atoms( self ):
        atoms_pos = defaultdict(list)
        for orb in self._list:
            atoms_pos[orb.atom_name] += [[str(x) for x in orb.position]]
        return atoms_pos
    
    def to_xyz( self, name : str = None ):
        atoms = self.atoms
        number_of_atoms = sum( [len(x) for x in atoms.values()] )
        str_rep = str(number_of_atoms) + "\n\n"
        
        for atom, positions in atoms.items():
            for pos in positions:
                str_rep += f'{atom} {" ".join(pos)}\n'

        if name is None:
            return str_rep

        with open( name, "w" ) as f:
            f.writelines(str_rep)

    # TODO: this far too simplistic
    @classmethod
    def from_xyz( cls, name : str ):
        orbs, group_id = [], _watchdog._Watchdog.next_value()        
        with open(name, 'r') as f:
            for line in f:
                processed = line.strip().split()
                if len(processed) <= 1:
                    continue                
                atom_name, x, y, z = processed
                
                orbs.append( Orbital( group_id = group_id,
                                      atom_name = atom_name,
                                      position = [float(x), float(y), float(z)] )  )
        return cls( orbs )
