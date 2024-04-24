from collections import Counter
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
    group_id: int = field(default_factory=_watchdog._Watchdog.next_value)

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
                self.group_id,
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
            if all(isinstance(k, int) for k in key):
                yield (key, value)

    def orbital_items(self):
        """Yields items where all elements of the key tuple are orbitals."""
        for key, value in self.items():
            if all(isinstance(k, Orbital) for k in key):
                yield (key, value)


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

@dataclass
class SimulationParams:
    """
    A data class for storing parameters necessary for running a simulation involving electronic states and transitions.

    Attributes:
        from_state (jax.Array): An array where each element is the index of an electronic state from which
                                electrons are excited. Defaults to an array containing a single zero.
        to_state (jax.Array): An array where each element is the index of an electronic state to which
                              electrons are excited. Defaults to an array containing a single zero.
        excited_electrons (jax.Array): An array where each element indicates the number of electrons excited
                                       between the corresponding states in `from_state` and `to_state`.
                                       Defaults to an array containing a single zero.
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
    from_state : jax.Array = field(default_factory=lambda : jnp.array([0]))
    to_state : jax.Array = field(default_factory=lambda : jnp.array([0]))
    excited_electrons : jax.Array = field(default_factory=lambda : jnp.array([0]))    
    eps : float = 1e-5
    beta : float = jnp.inf
    self_consistency_params : dict =  field(default_factory=dict)
    spin_degeneracy : float = 2.0
    electrons : Optional[int] = None
    

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

    def __init__(self, orbs, _hopping_dict=None, _coulomb_dict=None, _transitions_dict=None):
        # couplings are dicts mapping orbital pairs to couplings
        self._hopping_dict = (
            _hopping_dict if _hopping_dict is not None else _SortedTupleDict()
        )
        self._coulomb_dict = (
            _coulomb_dict if _coulomb_dict is not None else _SortedTupleDict()
        )
        self._transitions_dict =_transitions_dict if _transitions_dict is not None else _SortedTupleDict()


        # contains all high-level simulation information
        self._list = list(orbs) if orbs is not None else []

        # flag for recomputing state
        self._recompute = True

        self.simulation_params = SimulationParams()
        
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
        excited = f"{self.excited_electrons} electrons excited from {self.from_state} to {self.to_state}."
        groups = "\n".join(
            [
                f"group id {key} : {val} orbitals"
                for key, val in Counter(self.get_group_ids()).items()
            ]
        )
        return "\n".join((info, excited, groups))

    def __iter__(self):
        return iter(self._list)

    def __add__(self, other):
        if not self._are_orbs(other):
            raise TypeError

        if any(orb in other for orb in self._list):
            raise ValueError

        if isinstance(other, OrbitalList):
            new_hopping_dict = self._hopping_dict.copy()
            new_hopping_dict.update(other._hopping_dict)
            new_coulomb_dict = self._coulomb_dict.copy()
            new_coulomb_dict.update(other._coulomb_dict)
            new_transitions_dict = self._transitions_dict.copy()
            new_transitions_dict.update(other._transitions_dict)
            

        return OrbitalList(
            (self._list + list(other)).copy(),
            _SortedTupleDict(new_hopping_dict),
            _SortedTupleDict(new_coulomb_dict),
            _SortedTupleDict(new_transitions_dict),
        )

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
        self._delete_coupling(orb, self._hopping_dict)
        self._delete_coupling(orb, self._coulomb_dict)
        del self._list[position]

    @staticmethod
    def _are_orbs(candidate):
        return all(isinstance(orb, Orbital) for orb in candidate)

    @mutates
    def _set_coupling(self, orb_or_group_id1, orb_or_group_id2, val_or_func, coupling):
        coupling[(orb_or_group_id1, orb_or_group_id2)] = val_or_func

    def _hamiltonian_coulomb(self):

        def fill_matrix(matrix, coupling_dict):

            dummy = jnp.arange(len(self))
            triangle_mask = dummy[:, None] >= dummy

            # TODO: in principle we can build a big tensor NxNxgroups, vmap over the last axis and sum the groups
            # first, we loop over all group_id couplings => interactions between groups
            for key, function in coupling_dict.group_id_items():
                # TODO:  big uff:  we rely on the correct ordering of the group_ids for cols and rows, first key is always smaller than last keys => we get upper triangular valid indices
                # if it were the other way around, these would be zeroed by the triangle mask
                cols = group_ids == key[0]
                rows = (group_ids == key[1])[:, None]
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
        positions = self._get_positions()
        distances = jnp.round(
            jnp.linalg.norm(positions - positions[:, None], axis=-1), 6
        )
        group_ids = jnp.array(self.get_group_ids())

        hamiltonian = fill_matrix(
            jnp.zeros((len(self), len(self))).astype(complex), self._hopping_dict
        )
        coulomb = fill_matrix(
            jnp.zeros((len(self), len(self))).astype(complex), self._coulomb_dict
        )

        return hamiltonian, coulomb

    def _get_positions(self):
        return jnp.array([orb.position for orb in self._list])

    def _ensure_complex(self, func_or_val):
        if callable(func_or_val):
            return lambda x: func_or_val(x) + 0.0j
        if isinstance(func_or_val, (int, float, complex)):
            return func_or_val + 0.0j
        raise TypeError

    def _maybe_orbs_to_group_ids(self, maybe_orbs):
        def convert(maybe_orb):
            # TODO: check if this is really a group_id
            if isinstance(maybe_orb, int):
                return maybe_orb
            if isinstance(maybe_orb, Orbital):
                return maybe_orb.group_id
            return "You have passed something that is neither an orbital nor a group_id"
        return [convert(x) for x in maybe_orbs]


    def _maybe_indices_to_orbs(self, maybe_indices):
        def convert(maybe_index):
            if isinstance(maybe_index, int):
                return self._list[maybe_index]
            if isinstance(maybe_index, Orbital):
                return maybe_index
            return "You have passed something that is neither an orbital nor an index"
        return [convert(x) for x in maybe_indices]
        
    def _build(self):

        assert len(self) > 0
        self._positions = self._get_positions()

        self._hamiltonian, self._coulomb = self._hamiltonian_coulomb()

        self._eigenvectors, self._energies = jax.lax.linalg.eigh(self._hamiltonian)
        
        self._initial_density_matrix = _numerics._density_matrix(
            self._energies,
            self.electrons,
            self.spin_degeneracy,
            self.eps,
            self.from_state,
            self.to_state,
            self.excited_electrons,
            self.beta,
        )
        self._stationary_density_matrix = _numerics._density_matrix(
            self._energies,
            self.electrons,
            self.spin_degeneracy,
            self.eps,
            jnp.array([0]),
            jnp.array([0]),
            jnp.array([0]),
            self.beta,
        )

        if self.self_consistency_params:
            (
                self._hamiltonian,
                self._initial_density_matrix,
                self._stationary_density_matrix,
                self._energies,
                self._eigenvectors,
            ) = _get_self_consistent(
                self._hamiltonian,
                self._coulomb,
                self._positions,
                self.spin_degeneracy,
                self.electrons,
                self.eps,
                self._eigenvectors,
                self._static_density_matrix,
                **self.self_consistent_params,
            )

        self._initial_density_matrix = self.transform_to_site_basis( self._initial_density_matrix )
        self._stationary_density_matrix = self.transform_to_site_basis( self._stationary_density_matrix )
            
    def get_group_ids(self):
        """
        Retrieves a list of group IDs for all orbitals managed by this object.

        Returns:
            List[int]: A list of group IDs for each orbital.
        """
        return [orb.group_id for orb in self._list]

    def get_unique_group_ids(self):
        """
        Retrieves a unique set of group IDs from all orbitals.

        Returns:
            List[int]: A list of unique group IDs.
        """
        return list(set(self.get_group_ids()))

    def set_groups_hopping(self, orb_or_group_id1, orb_or_group_id2, func):
        """
        Sets the hopping coupling between two groups of orbitals.

        Parameters:
            orb_or_group_id1 (int or Orbital): Identifier or orbital for the first group.
            orb_or_group_id2 (int or Orbital): Identifier or orbital for the second group.
            func (callable): Function that defines the hopping interaction.

        Notes:
            The function `func` should be complex-valued.
        """
        group_id1, group_id2 = self._maybe_orbs_to_group_ids(
            (orb_or_group_id1, orb_or_group_id2)
        )
        self._set_coupling(
            group_id1, group_id2, self._ensure_complex(func), self._hopping_dict
        )

    def set_groups_coulomb(self, orb_or_group_id1, orb_or_group_id2, func):
        """
        Sets the Coulomb coupling between two groups of orbitals.

        Parameters:
            orb_or_group_id1 (int or Orbital): Identifier or orbital for the first group.
            orb_or_group_id2 (int or Orbital): Identifier or orbital for the second group.
            func (callable): Function that defines the Coulomb interaction.

        Notes:
            The function `func` should be complex-valued.
        """
        group_id1, group_id2 = self._maybe_orbs_to_group_ids(
            (orb_or_group_id1, orb_or_group_id2)
        )
        self._set_coupling(
            group_id1, group_id2, self._ensure_complex(func), self._coulomb_dict
        )

    def set_hamiltonian_element(self, orb_or_index1, orb_or_index2, val):
        """
        Sets an element of the Hamiltonian matrix between two orbitals or indices.

        Parameters:
            orb_or_index1 (int or Orbital): Identifier or orbital for the first element.
            orb_or_index2 (int or Orbital): Identifier or orbital for the second element.
            val (complex): The complex value to set for the Hamiltonian element.
        """
        orb1, orb2 = self._maybe_indices_to_orbs((orb_or_index1, orb_or_index2))
        self._set_coupling(orb1, orb2, self._ensure_complex(val), self._hopping_dict)

    def set_coulomb_element(self, orb_or_index1, orb_or_index2, val):
        """
        Sets a Coulomb interaction element between two orbitals or indices.

        Parameters:
            orb_or_index1 (int or Orbital): Identifier or orbital for the first element.
            orb_or_index2 (int or Orbital): Identifier or orbital for the second element.
            val (complex): The complex value to set for the Coulomb interaction element.
        """
        orb1, orb2 = self._maybe_indices_to_orbs((orb_or_index1, orb_or_index2))
        self._set_coupling(orb1, orb2, self._ensure_complex(val), self._coulomb_dict)

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

    @mutates
    def shift_by_vector(self, tag_or_group_id, translation_vector):
        """
        Shifts all orbitals with a specific tag by a given vector.

        Parameters:
            tag_or_group_id (str or int or list[int]): The tag, group_id to match orbitals.
            translation_vector (jax.Array): The vector by which to translate the orbital positions.

        Notes:
            This operation mutates the positions of the matched orbitals.
        """
        if isinstance(tag_or_group_id, str):            
            orbs = [orb for orb in self._list if orb.tag == tag_or_group_id]
        elif isinstance(tag_or_group_id, int):
            orbs = [orb for orb in self._list if orb.group_id == tag_or_group_id]
        else:
            orbs = [orb for orb in self._list if orb.group_id in tag_or_group_id]
            
        for orb in orbs:
            orb.position += jnp.array(translation_vector)

    @mutates
    def set_position(self, tag, position):
        """
        Sets the position of all orbitals with a specific tag.

        Parameters:
            tag (str): The tag to match orbitals.
            position (jax.Array): The vector at which to move the orbitals

        Notes:
            This operation mutates the positions of the matched orbitals.
        """
        orbs = [orb for orb in self._list if orb.tag == tag]
        for orb in orbs:
            orb.position = position

            
    @mutates
    def make_self_consistent(self, sc_params):
        """
        Configures the list for self-consistent field calculations.

        Parameters:
            sc_params (dict): Parameters for self-consistency.
        """
        self.self_consistency_params = sc_params

    @mutates
    def set_electrons( self, number ):
        self.simulation_params.electrons = number 

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

        self.simulation_params.from_state = maybe_int_to_arr(from_state)
        self.simulation_params.to_state = maybe_int_to_arr(to_state)
        self.simulation_params.excited_electrons = maybe_int_to_arr(excited_electrons)

    @mutates
    def set_dipole_transition(self, orb_or_index1, orb_or_index2, arr):
        """
        Sets a dipole transition for specified orbital or index pairs.

        Parameters:
            orb_or_index1 (int or Orbital): Identifier or orbital for the first part of the transition.
            orb_or_index2 (int or Orbital): Identifier or orbital for the second part of the transition.
            arr (jax.Array): The 3-element array containing dipole transition elements.
        """
        orb1, orb2 = self._maybe_indices_to_orbs((orb_or_index1, orb_or_index2))
        self._transitions_dict[(orb_or_index1, orb_or_index2)] = jnp.array(arr).astype(
            complex
        )

    @property
    @recomputes
    def homo(self):
        return (self.electrons * self.stationary_density_matrix_e).real.diagonal().round(2).nonzero()[0][-1].item()

    @property
    def electrons(self):
        if self.simulation_params.electrons is None:
            return len(self._list)
        return self.simulation_params.electrons

    @property
    def spin_degeneracy(self):
        return self.simulation_params.spin_degeneracy

    @property
    def from_state(self):
        return self.simulation_params.from_state
    
    @property
    def to_state(self):
        return self.simulation_params.to_state
    
    @property
    def excited_electrons(self):
        return self.simulation_params.excited_electrons

    @property
    def beta(self):
        return self.simulation_params.beta

    @property
    def eps(self):
        return self.simulation_params.eps
    
    @property
    @recomputes
    def positions(self):
        return self._positions

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
                jnp.diag(self._positions[:, i] / 2)
            )
        for orbital_combination, value in self._transitions_dict.items():
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

        if self._transitions_dict is None:
            x_times_h = jnp.einsum("ij,iL->ijL", self._hamiltonian, self._positions)
            h_times = jnp.einsum("ij,jL->ijL", self._hamiltonian, self._positions)
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

    def transform_to_energy_basis(self, observable):#
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
        vec_r = self._positions[:, None] - positions

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

    def get_expectation_value_time_domain(self, *args, **kwargs):
        """
        Calculates the time-domain expectation value of an operator, corrected for induced effects based on the stationary density matrix.

        Parameters:
        The same as for get_density_matrix_time_domain, except operator

        Returns:
           Tuple[jax.Array, jax.Array]: A tuple containing the time axis and the calculated expectation values over time.
        """

        operator = kwargs.pop("operator", None)
        return_density = kwargs.pop("return_density", False)
        time_axis, density_matrices = self.get_density_matrix_time_domain(
            *args, **kwargs
        )
        expectation_value = self.get_expectation_value(
            density_matrix = density_matrices, operator = operator
        )
        if return_density == True:
            return time_axis, expecation_value, density_matrices
        return time_axis, expectation_value        

    def get_expectation_value_frequency_domain(self, *args, **kwargs):
        """
        Computes the frequency-domain expectation values by transforming time-domain data obtained from expectation values calculations.

        Parameters:
        The same as for get_density_matrix_time_domain, except omega_min, omega_max and the operator.

        Returns:
           Tuple[jax.Array, jax.Array, jax.Array]: Frequencies and corresponding expectation values, and optionally transformed electric field data.
        """

        density_matrices = kwargs.pop("density_matrices", None)
        time_axis = kwargs.pop("time", None)
        omega_min = kwargs.pop("omega_min", 0)
        omega_max = kwargs.pop("omega_max", 100)

        if density_matrices is None:
            time_axis, exp_val_td = self.get_expectation_value_time_domain(*args, **kwargs)
        else:
            operator = kwargs.pop("operator", None)            
            exp_val_td = self.get_expectation_value(density_matrix = density_matrices, operator = operator)
                        
        omega, exp_val_omega = _numerics.get_fourier_transform(time_axis, exp_val_td)
        mask = (omega >= omega_min) & (omega <= omega_max)
        try:
            electric_field = jax.vmap(kwargs["illumination"])(time_axis)
            field_omega = _numerics.get_fourier_transform(
                time_axis, electric_field, return_omega_axis=False
            )
            return omega[mask], exp_val_omega[mask], field_omega[mask]
        except KeyError:
            return omega[mask], exp_val_omega[mask]

    @recomputes
    def get_density_matrix_time_domain(
        self,
        end_time: float,
        illumination: Callable[[float], jax.Array],
        start_time: Optional[float] = None,
        steps_time: Optional[int] = None,
        skip: Optional[int] = None,
        relaxation_rate: Union[float, jax.Array] = None,
        saturation_functional: Callable[[float], float] = lambda x: 1
        / (1 + jnp.exp(-1e6 * (2.0 - x))),
        use_old_method: bool = False,
        include_induced_contribution: bool = False,
        use_rwa=False,
        compute_only_at=None,
        coulomb_strength=1.0,
        solver=diffrax.Dopri5(),
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
        initial_density_matrix : Optional[jax.Array] = None,
    ):
        """
        Simulates the time evolution of the density matrix for a given system under specified conditions and external fields.
        
        Parameters:
           end_time (float): The end time for the simulation.
           illumination (Callable[[float], jax.Array]): A function that returns the electric field at a given time.
           start_time (Optional[float]): The start time for the simulation, defaults to zero.
           steps_time (Optional[int]): The number of time steps to simulate, defaults to int(end_time * 1000)
           skip (Optional[int]): The interval at which to record results, defaults to 1, i.e. record every density matrix.
           relaxation_rate (Union[float, jax.Array]): The relaxation rates to be applied: if constant, the phenomenological term is applied, if an NxN array, the saturated lindblad model is applied.
           saturation_functional (Callable[[float], float]): A function defining the saturation behavior, defaults to smoothed-out step function.
           use_old_method (bool): Flag to use the old RK method.
           include_induced_contribution (bool): Whether to include induced contributions in the simulation.
           use_rwa (bool): Whether to apply the rotating wave approximation.
           compute_only_at (Optional[any]): Specific orbital indices at which the induced field computation is performed.
           coulomb_strength (float): Strength of Coulomb interactions.
           solver (diffrax.Solver): The differential equation solver to use.
           stepsize_controller (diffrax.StepSizeController): The controller for the solver's step size.
           initial_density_matrix (Union[jax.Array,None]): if given, used as initial density matrix instead

        Returns:
           Tuple[jax.Array, jax.Array]: The time axis and the simulated density matrices at specified time intervals.
        """

        # Time axis creation
        start_time = float(start_time) if start_time is not None else 0.0
        steps_time = int(steps_time) if steps_time is not None else int(end_time * 1000)
        time_axis = jnp.linspace(start_time, end_time, steps_time)
        skip = skip if skip is not None else 1

        # Determine relaxation function based on the input type
        if relaxation_rate is None:
            relaxation_function = lambda r: 0.0
        elif isinstance(relaxation_rate, jax.Array):
            relaxation_function = _numerics.lindblad_saturation_functional(
                self._eigenvectors,
                relaxation_rate,
                saturation_functional,
                self.electrons,
                self._stationary_density_matrix,
            )
        else:
            relaxation_function = _numerics.relaxation_time_approximation(
                relaxation_rate,
                self.stationary_density_matrix,
            )

        # Verify that illumination is a callable
        if not callable(illumination):
            raise TypeError("Provide a function for e-field")

        # Initialize common variables
        coulomb_field_to_from = _numerics.get_coulomb_field_to_from(
            self.positions, self.positions, compute_only_at
        )
        initial_density_matrix = self.initial_density_matrix if initial_density_matrix is None else initial_density_matrix

        # TODO: not very elegant: we just dump every argument in there by default
        return time_axis[::skip], _numerics.integrate_master_equation(
            self._hamiltonian,
            coulomb_strength * self._coulomb,
            self.dipole_operator,
            self.electrons,
            self.velocity_operator,
            initial_density_matrix,
            self.stationary_density_matrix,
            time_axis,
            illumination,
            relaxation_function,
            coulomb_field_to_from,
            include_induced_contribution,
            use_rwa,
            solver,
            stepsize_controller,
            use_old_method,
            skip,
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
