import os
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace, asdict
from functools import wraps
from pprint import pformat
from typing import Callable, NamedTuple, Optional, Union, Dict, Any

import diffrax
import jax
import jax.numpy as jnp

from granad import _numerics, _plotting, _watchdog, potentials, dissipators


@dataclass
class Orbital:
    """
    Attributes:
        position (jax.Array): The position of the orbital in space, initialized by default to a zero position.
                              This field is not used in hashing or comparison of instances.
        layer_index (Optional[int]): An optional index representing the layer of the orbital within its atom,
                                     may be None if not specified.
        tag (Optional[str]): An optional tag for additional identification or categorization of the orbital,
                             defaults to None.
        spin (Optional[int]): The spin quantum number of the orbital, indicating its intrinsic angular momentum,
                              optional and may be None. *Note* This is experimental.
        atom_name (Optional[str]): The name of the atom this orbital belongs to, can be None if not applicable.
        group_id (int): A group identifier for the orbital, automatically assigned by a Watchdog class
                        default factory method. For example, all pz orbitals in a single graphene flake get the same 
                        group_id.
    """
    position: jax.Array = field(default_factory=lambda : jnp.array([0, 0, 0]), hash=False, compare=False)
    layer_index: Optional[int] = None
    tag: Optional[str] = None
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
        return self.group_id < other.group_id

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
    Stores parameters characterizing a given structure.

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
    mean_field_params : dict =  field(default_factory=dict)
    spin_degeneracy : float = 2.0

    def __add__( self, other ):
        if isinstance(other, Params):
            return Params(self.electrons + other.electrons)        
        raise ValueError
    
# rule: new couplings are registered here
@dataclass
class Couplings:
    """
    A data class for representing orbital couplings.

    Attributes:
        hamiltonian (_SortedTupleDict): A dictionary-like container holding Hamiltonian terms.
        coulomb (_SortedTupleDict): A dictionary-like container for Coulomb interaction terms.
        dipole_transitions (_SortedTupleDict): A dictionary-like container for storing dipole transition elements.
    """

    hamiltonian : _SortedTupleDict = field(default_factory=_SortedTupleDict)
    coulomb : _SortedTupleDict = field(default_factory=_SortedTupleDict)
    dipole_transitions : _SortedTupleDict = field(default_factory=_SortedTupleDict)

    def __str__( self ):
        return 

    def __add__( self, other ):
        if isinstance(other, Couplings):            
            return Couplings(
                _SortedTupleDict(self.hamiltonian | other.hamiltonian),
                _SortedTupleDict(self.coulomb | other.coulomb),
                _SortedTupleDict(self.dipole_transitions | other.dipole_transitions)
            )
        raise ValueError        

    
class TDArgs(NamedTuple):
    hamiltonian : jax.Array
    energies : jax.Array
    coulomb_scaled : jax.Array
    initial_density_matrix : jax.Array
    stationary_density_matrix : jax.Array
    eigenvectors : jax.Array
    dipole_operator : jax.Array
    electrons : jax.Array
    relaxation_rate : jax.Array
    propagator : jax.Array
    spin_degeneracy : jax.Array
    positions : jax.Array

@dataclass
class TDResult:
    """
    A data class for storing the results of time-dependent simulations.

    Attributes:
        td_illumination (jax.Array): An array containing the time-dependent illumination function applied to the system,
                                     typically representing an external electromagnetic field.
        time_axis (jax.Array): An array representing the time points at which the simulation was evaluated.
        final_density_matrix (jax.Array): The resulting density matrix at the end of the simulation, representing the
                                          state of the system.
        output (list[jax.Array]): A list of arrays containing various output data from the simulation, such as observables
                                  over time.
        extra_attributes (Dict[str, Any]): A dictionary saving any other quantity of interest (e.g. absorption spectra), by default empty.

    """

    td_illumination : jax.Array = field(default_factory=lambda: jnp.array([]))
    time_axis : jax.Array = field(default_factory=lambda: jnp.array([]))
    final_density_matrix : jax.Array = field(default_factory=lambda: jnp.array([[]]))
    output : list[jax.Array] = field(default_factory=list)
    extra_attributes: Dict[str, Any] = field(default_factory=dict)  # Stores dynamic attributes
    
    def ft_output( self, omega_max, omega_min ):
        """
        Computes the Fourier transform of each element in the output data across a specified frequency range.

        Args:
            omega_max (float): The maximum frequency bound for the Fourier transform.
            omega_min (float): The minimum frequency bound for the Fourier transform.

        Returns:
            list[jax.Array]: A list of Fourier transformed arrays corresponding to each element in the `output` attribute,
                              evaluated over the specified frequency range.

        Note:
            This method applies a Fourier transform to each array in the `output` list to analyze the frequency components
            between `omega_min` and `omega_max`.
        """
        ft = lambda o : _numerics.get_fourier_transform(self.time_axis, o, omega_max, omega_min, False)
        return [ft(o) for o in self.output]

    def ft_illumination( self, omega_max, omega_min, return_omega_axis = True ):
        """
        Calculates the Fourier transform of the time-dependent illumination function over a specified frequency range,
        with an option to return the frequency axis.

        Args:
            omega_max (float): The maximum frequency limit for the Fourier transform.
            omega_min (float): The minimum frequency limit for the Fourier transform.
            return_omega_axis (bool): If True, the function also returns the frequency axis along with the Fourier
                                      transformed illumination function. Defaults to True.

        Returns:
            jax.Array, optional[jax.Array]: The Fourier transformed illumination function. If `return_omega_axis` is True,
                                            a tuple containing the Fourier transformed data and the corresponding frequency
                                            axis is returned. Otherwise, only the Fourier transformed data is returned.

        """
        return _numerics.get_fourier_transform(self.time_axis, self.td_illumination, omega_max, omega_min, return_omega_axis)

    def add_extra_attribute(self,name: str,value: Any):
        """
        Dynamically adds an attribute to the 'extra_attributes' field.

        Args:
            name (str): Name of the new attribute to be added.
            value (Any): Value of the attribute.
        """
        self.extra_attributes[name]=value
        print(f"Extra attribute '{name}' is added.")

    def remove_extra_attribute(self,name: str):
        """
        Dynamically deletes an attribute from 'extra_attributes'.

        Args:
            name (str): Name of the attribute to be removed.
        """
        if name not in self.extra_attributes:
            raise KeyError(f"The attribute '{name}' does not exist in 'extra_attributes'. ")
        else: 
            del self.extra_attributes[name]
            print(f"Extra attribute '{name}' is removed.")

    def show_extra_attribute_list(self):
        """
        Displays all available extra attributes. 
        """
        print(list(self.extra_attributes.keys()))

    def get_attribute(self, name: str):
        """
        Returns the value of any specified attribute, no matter the original class attributes or the extra ones.

        Args:
            name (str): Name of the attribute.
            
        Return:
            Value of the attribute.
        """
        if name in self.__dict__.keys():
            return self.__dict__[name]
            
        elif name in self.extra_attributes.keys():
            return self.extra_attributes[name]
            
        else:
            raise KeyError(f"The attribute '{name}' does not exist ")
    
    def save(self, name, save_only=None):
        """
        Saves the TDResult into a .npz file

        Args:
            name (str): The filename prefix for saving.
            save_only (list, optional): List of attribute names to save selectively.
        """
        data = asdict(self) 
        
        if save_only:
            data.update(self.extra_attributes) # flatted dict with key from both orginal data and extra_attributes dictionary
            data={k:v for k,v in data.items() if k in save_only } # filtered data
            
        jnp.savez(f"{name}.npz", **data)

    @classmethod
    def load( cls, name ):
        """
        Constructs a TDResult object from saved data.
    
        Args:
            name (str): The filename (without extension) from which to load the data.
    
        Returns:
            TDResult: A TDResult object constructed from the saved data.
    
        Note:
            If the 'save_only' option was used earlier, the TDResult object will be created 
            with only the available data, and missing fields will be filled with empty values 
            of their corresponding types.
        """
        with jnp.load(f'{name}.npz',allow_pickle=True) as data:
            data=dict(**data)
            primary_attribute_list=['td_illumination','time_axis','final_density_matrix','output','extra_attributes']
            dynamic_attributes={k:v for k,v in data.items() if k not in primary_attribute_list}
            
            return cls(
                
                td_illumination = jnp.asarray(data.get('td_illumination',[])),
                
                time_axis = jnp.asarray(data.get('time_axis',[])),
                
                final_density_matrix = jnp.asarray(data.get('final_density_matrix',[[]])),
                
                output=[jnp.asarray(arr) for arr in data.get('output', [])],
                
                extra_attributes=data.get('extra_attributes',dynamic_attributes).item()
                  
            )

    
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
        _list (list) : the underlying list that contains the orbitals
        params (Params): Simulation parameters like electron count and temperature.
        couplings (_SortedTupleDict): A (customized) dictionary where keys are tuples of orbital identifiers and values are the couplings
                          (either float values or functions).

    Note:
        - **Orbital Identification**: Orbitals can be identified either by their group_id, a direct
          reference to the orbital object itself, or via a user-defined tag.
        - **Index Access**: Orbitals can be accessed and managed by their index in the list, allowing for
          list-like manipulation (addition, removal, access).
        - **Coupling Definition**: Allows for the definition and adjustment of couplings between pairs of orbitals,
          identified by a tuple of their respective identifiers. These couplings can dynamically represent the
          interaction strength or be a computational function that defines the interaction.
    """
    def __init__( self, orbs = None, couplings = None, params = None, recompute = True):
        self._list = orbs if orbs is not None else []
        self.couplings = couplings if couplings is not None else Couplings( )
        self.params = params if params is not None else Params( len(orbs) )
        self._recompute = recompute    
        
    def __getattr__(self, property_name):
        if property_name.endswith("_x"):
            original_name = property_name[:-2]
            try:
                return getattr(self, original_name)
            except AttributeError:
                pass 
        elif property_name.endswith("_e"):
            original_name = property_name[:-2]
            try:
                return self.transform_to_energy_basis(getattr(self, original_name))
            except AttributeError:
                pass            
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {property_name!r}")

    def __len__(self):
        return len(self._list)

    # can't mutate, because orbitals are immutable
    def __getitem__(self, position):
        return self._list[position]

    def __repr__(self):
        info = f"List with {len(self)} orbitals, {self.electrons} electrons."
        exc = self.params.excitation
        info += f"\nExcitation: {exc[2]} electrons excited from energy levels {exc[0]} to {exc[1]}."
        info += f"\nIncluded tags with number of orbitals: {dict(Counter(o.tag for o in self))}"
        return info 

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
        self._delete_coupling(orb, self.couplings.hamiltonian)
        self._delete_coupling(orb, self.couplings.coulomb)
        self._delete_coupling(orb, self.couplings.dipole_transitions)
        self.params.electrons -= 1
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

            # matrix is NxN and hermitian
            # we fill the upper triangle with a mask and make the matrix hermitian by adding adding its conjugate transpose
            dummy = jnp.arange(len(self))
            triangle_mask = dummy[:, None] >= dummy
            
            # first, we loop over all group_id couplings => interactions between groups
            for key, function in coupling_dict.group_id_items():
                # if it were the other way around, these would be zeroed by the triangle mask
                cols = group_ids == key[0].id
                rows = (group_ids == key[1].id)[:, None] 
                combination_indices = jnp.logical_and(rows, cols)                
                valid_indices = jnp.logical_and(triangle_mask, combination_indices)
                
                # hotfix
                if valid_indices.sum() == 0:
                    valid_indices = jnp.logical_and(triangle_mask.T, combination_indices)
                    
                function = jax.vmap(function)
                matrix = matrix.at[valid_indices].set(
                    function(distances[valid_indices])
                )

            matrix += matrix.conj().T - jnp.diag(jnp.diag(matrix))

            # we now set single elements
            rows, cols, vals = [], [], []
            for key, val in coupling_dict.orbital_items():
                rows.append(self._list.index(key[0]))
                cols.append(self._list.index(key[1]))
                vals.append(val)

            vals = jnp.array(vals)
            matrix = matrix.at[rows, cols].set(vals)
            matrix = matrix.at[cols, rows].set(vals.conj())

            return matrix

        positions = self.positions
        distances = jnp.round(positions - positions[:, None], 6)
        group_ids = jnp.array( [orb.group_id.id for orb in self._list] )

        hamiltonian = fill_matrix(
            jnp.zeros((len(self), len(self))).astype(complex), self.couplings.hamiltonian
        )
        coulomb = fill_matrix(
            jnp.zeros((len(self), len(self))).astype(complex), self.couplings.coulomb
        )
        return hamiltonian, coulomb

    @mutates
    def set_dipole_element(self, orb1, orb2, arr):
        """
        Sets a dipole transition for specified orbital or index pairs.

        Parameters:
            orb1: Identifier for orbital(s) for the first part of the transition.
            orb2: Identifier for orbital(s) for the second part of the transition.
            arr (jax.Array): The 3-element array containing dipole transition elements.
        """
        self._set_coupling(self.filter_orbs(orb1, Orbital), self.filter_orbs(orb2, Orbital), jnp.array(arr).astype(complex), self.couplings.dipole_transitions)
        
    def set_hamiltonian_groups(self, orb1, orb2, func):
        """
        Sets the hamiltonian coupling between two groups of orbitals.

        Parameters:
            orb1: Identifier for orbital(s) for the first group.
            orb2: Identifier for orbital(s) for the second group.
            func (callable): Function that defines the hamiltonian interaction.

        Note:
            The function `func` should be complex-valued.
        """
        self._set_coupling(
            self.filter_orbs(orb1, _watchdog.GroupId), self.filter_orbs(orb2, _watchdog.GroupId), self._ensure_complex(func), self.couplings.hamiltonian
        )

    def set_coulomb_groups(self, orb1, orb2, func):
        """
        Sets the Coulomb coupling between two groups of orbitals.

        Parameters:
            orb1: Identifier for orbital(s) for the first group.
            orb2: Identifier for orbital(s) for the second group.
            func (callable): Function that defines the Coulomb interaction.

        Note:
            The function `func` should be complex-valued.
        """
        self._set_coupling(
            self.filter_orbs(orb1, _watchdog.GroupId), self.filter_orbs(orb2, _watchdog.GroupId), self._ensure_complex(func), self.couplings.coulomb
        )

    def set_onsite_hopping(self, orb, val):
        """
        Sets onsite hopping element of the Hamiltonian matrix.

        Parameters:
            orb: Identifier for orbital(s).
            val (real): The value to set for the onsite hopping.
        """
        self.set_hamiltonian_element(orb, orb, val)        

    def set_hamiltonian_element(self, orb1, orb2, val):
        """
        Sets an element of the Hamiltonian matrix between two orbitals or indices.

        Parameters:
            orb1: Identifier for orbital(s) for the first element.
            orb2: Identifier for orbital(s) for the second element.
            val (complex): The complex value to set for the Hamiltonian element.
        """
        self._set_coupling(self.filter_orbs(orb1, Orbital), self.filter_orbs(orb2, Orbital), self._ensure_complex(val), self.couplings.hamiltonian)

    def set_coulomb_element(self, orb1, orb2, val):
        """
        Sets a Coulomb interaction element between two orbitals or indices.

        Parameters:
            orb1: Identifier for orbital(s) for the first element.
            orb2: Identifier for orbital(s) for the second element.
            val (complex): The complex value to set for the Coulomb interaction element.
        """
        self._set_coupling(self.filter_orbs(orb1, Orbital), self.filter_orbs(orb2, Orbital), self._ensure_complex(val), self.couplings.coulomb)

    @property
    def center_index(self):
        """index of approximate center orbital of the structure"""
        distances = jnp.round(jnp.linalg.norm(self.positions - self.positions[:, None], axis = -1), 4)
        return jnp.argmin(distances.sum(axis=0))
    
    def localization(self, neighbor_number : int = 6):
        """Compute edge localization of eigenstates according to
        
        $$
        \\frac{\sum_{j \, edge} |\phi_{j}|^2}{\sum_i |\phi_i|^2 }
        $$

        Edges are identified based on the number of next-to-next-to nearest neighbors (nnn).
        
        Args:
            neighbor_number (int): nnn used to identify edges. Depends on lattice and orbital number.
            Defaults to nnn = 6 for the case of a hexagonal lattice with a single orbital per site.
            For more orbitals use nnn * num_orbitals.

        Returns:
            jax.Array: localization, where i-th entry corresponds to i-th energy eigenstate
        """
    
        # edges => neighboring unit cells are incomplete => all points that are not inside a "big hexagon" made up of nearest neighbors
        positions, states, energies = self.positions, self.eigenvectors, self.energies 

        distances = jnp.round(jnp.linalg.norm(positions - positions[:, None], axis = -1), 4)
        nnn = jnp.unique(distances)[2]
        mask = (distances == nnn).sum(axis=0) < neighbor_number

        # localization => how much eingenstate 
        l = (jnp.abs(states[mask, :])**2).sum(axis = 0) # vectors are normed

        return l

    def _ensure_complex(self, func_or_val):
        if callable(func_or_val):
            return lambda x: func_or_val(x) + 0.0j
        if isinstance(func_or_val, (int, float, complex)):
            return func_or_val + 0.0j
        raise TypeError

    def _build(self):

        assert len(self) > 0

        self._hamiltonian, self._coulomb = self._hamiltonian_coulomb()

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

        if len(self.params.self_consistency_params) != 0:
            (
                self._hamiltonian,
                self._initial_density_matrix,
                self._stationary_density_matrix,
                self._energies,
                self._eigenvectors,
            ) = _numerics._get_self_consistent(
                self._hamiltonian,
                self._coulomb,
                self.positions,
                self.params.excitation,
                self.params.spin_degeneracy,
                self.params.electrons,
                self.params.eps,
                self._eigenvectors,
                self._stationary_density_matrix,
                **self.params.self_consistency_params,
            )

        if len(self.params.mean_field_params) != 0:
            (
                self._hamiltonian,
                self._initial_density_matrix,
                self._stationary_density_matrix,
                self._energies,
                self._eigenvectors,
            ) = _numerics._mf_loop(
                self._hamiltonian,
                self._coulomb,
                self.params.excitation,
                self.params.spin_degeneracy,
                self.params.electrons,
                self.params.eps,
                **self.params.mean_field_params,
            )

        eps = 1e-1
        lower = -eps
        upper = 2 + eps
        if jnp.any(self._initial_density_matrix.diagonal() < lower) or jnp.any(self._initial_density_matrix.diagonal() > upper):
            raise Exception("Occupation numbers in initial density matrix are invalid.")

        if jnp.any(self._stationary_density_matrix.diagonal() < lower) or jnp.any(self._stationary_density_matrix.diagonal() > upper ) :
            raise Exception("Occupation numbers in stationary density matrix are invalid.")
            
        self._initial_density_matrix = self.transform_to_site_basis( self._initial_density_matrix )

        self._stationary_density_matrix = self.transform_to_site_basis( self._stationary_density_matrix )
            
    def set_open_shell( self ):
        if any( orb.spin is None for orb in self._list ):
            raise ValueError
        self.params.spin_degeneracy = 1.0

    def set_closed_shell( self ):
        self.params.spin_degeneracy = 2.0
        
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
        self.params.electrons += 1

    def filter_orbs( self, orb_id, t ):
        """maps a given orb_id (such as an index or tag) to a list of the required type t"""
        def filter_single_orb(orb_id, t):
            if type(orb_id) == t:
                return [orb_id]

            # index to group, orb, tag => group_id / orb / tag at index,
            if isinstance(orb_id, int) and t == _watchdog.GroupId:
                return [self._list[orb_id].group_id]
            if isinstance(orb_id, int) and t == Orbital:
                return [self._list[orb_id]]
            if isinstance(orb_id, int) and t == str:
                return [self._list[orb_id].tag]

            # group to index, orb, tag => group_id / orb / tag at index,
            if isinstance(orb_id, _watchdog.GroupId) and t == str:
                return [ orb.tag for orb in self if orb.group_id == orb_id ]
            if isinstance(orb_id, _watchdog.GroupId) and t == Orbital:
                return [ orb for orb in self if orb.group_id == orb_id ]
            if isinstance(orb_id, _watchdog.GroupId) and t == int:
                return [ i for i, orb in enumerate(self) if orb.group_id == orb_id ]

            # tag to group, orb, index => group_id / orb / tag at index,
            if isinstance(orb_id, str) and t == _watchdog.GroupId:
                return [orb.group_id for orb in self if orb.tag == orb_id]
            if isinstance(orb_id, str) and t == int:
                return [i for i, orb in enumerate(self) if orb.tag == orb_id]
            if isinstance(orb_id, str) and t == Orbital:
                return [orb for orb in self if orb.tag == orb_id]

            # orb to index, group, tag
            if isinstance(orb_id, Orbital) and t == _watchdog.GroupId:
                return [orb_id.group_id]
            if isinstance(orb_id, Orbital) and t == int:
                return [self._list.index(orb_id)]
            if isinstance(orb_id, Orbital) and t == str:
                return [orb_id.tag]

        if not isinstance(orb_id, OrbitalList):
            orb_id = [orb_id]

        return [ x for orb in orb_id for x in filter_single_orb(orb, t) ]
            

        
    @mutates
    def shift_by_vector(self, translation_vector, orb_id = None):
        """
        Shifts all orbitals with a specific tag by a given vector.

        Parameters:
            translation_vector (list or jax.Array): The vector by which to translate the orbital positions.
            orb_id: Identifier for the orbital(s) to shift.

        Note:
            This operation mutates the positions of the matched orbitals.
        """
        filtered_orbs = self.filter_orbs( orb_id, Orbital ) if orb_id is not None else self
        for orb in filtered_orbs:
            orb.position += jnp.array(translation_vector)

    @mutates
    def set_position(self, position, orb_id = None):
        """
        Sets the position of all orbitals with a specific tag.

        Parameters:
            position (list or jax.Array): The vector at which to move the orbitals
            orb_id: Identifier for the orbital(s) to shift.

        Note:
            This operation mutates the positions of the matched orbitals.
        """
        filtered_orbs = self.filter_orbs( orb_id, Orbital ) if orb_id is not None else self
        for orb in filtered_orbs:
            orb.position = position

    @mutates
    def rotate(self, x, phi, axis = 'z'):
        """rotates all orbitals an angle phi around a point p around axis.    
    
        Args:
        x : jnp.ndarray
            A 3D point around which to rotate.
        phi : float
            Angle by which to rotate.
        axis : str
            Axis to rotate around ('x', 'y', or 'z'). Default is 'z'.
        """
        
        # Define the rotation matrix based on the specified axis
        if axis == 'x':
            rotation_matrix = jnp.array([
                [1, 0, 0],
                [0, jnp.cos(phi), -jnp.sin(phi)],
                [0, jnp.sin(phi), jnp.cos(phi)]
            ])
        elif axis == 'y':
            rotation_matrix = jnp.array([
                [jnp.cos(phi), 0, jnp.sin(phi)],
                [0, 1, 0],
                [-jnp.sin(phi), 0, jnp.cos(phi)]
            ])
        elif axis == 'z':
            rotation_matrix = jnp.array([
                [jnp.cos(phi), -jnp.sin(phi), 0],
                [jnp.sin(phi), jnp.cos(phi), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        for orb in self._list:
            # Perform the rotation (translate along x, rotate, translate back)
            self.set_position(rotation_matrix @ (orb.position - x) + x, orb)
             
    @mutates
    def set_self_consistent(self, **kwargs):
        """
        Configures the parameters for self-consistent field (SCF) calculations.

        This function sets up the self-consistency parameters used in iterative calculations 
        to update the system's density matrix until convergence is achieved.

        Args:
            **kwargs: Keyword arguments to override the default self-consistency parameters. 
                The available parameters are:

                - `accuracy` (float, optional): The convergence criterion for self-consistency. 
                  Specifies the maximum allowed difference between successive density matrices.
                  Default is 1e-6.

                - `mix` (float, optional): The mixing factor for the density matrix during updates.
                  This controls the contribution of the new density matrix to the updated one.
                  Values closer to 1 favor the new density matrix, while smaller values favor 
                  smoother convergence. Default is 0.3.

                - `iterations` (int, optional): The maximum number of iterations allowed in the 
                  self-consistency cycle. Default is 500.

                - `coulomb_strength` (float, optional): A scaling factor for the Coulomb matrix.
                  This allows tuning of the strength of Coulomb interactions in the system. 
                  Default is 1.0.

        Example:
            >>> model.set_self_consistent(accuracy=1e-7, mix=0.5, iterations=1000)
            >>> print(model.params.self_consistency_params)
            {'accuracy': 1e-7, 'mix': 0.5, 'iterations': 1000, 'coulomb_strength': 1.0}
        """
        default = {"accuracy" : 1e-6, "mix" : 0.3, "iterations" : 500, "coulomb_strength" : 1.0}
        self.params.self_consistency_params = default | kwargs

    @mutates
    def set_mean_field(self, **kwargs):
        """
        Configures the parameters for mean field calculations.
        If no other parameters are passed, a standard direct channel Hartree-Fock calculation is performed.
        Note that this procedure differs slightly from the self-consistent field procedure.

        This function sets up the mean field parameters used in iterative calculations 
        to update the system's density matrix until convergence is achieved.

        Args:
            **kwargs: Keyword arguments to override the default self-consistency parameters. 
                The available parameters are:

                - `accuracy` (float, optional): The convergence criterion for self-consistency. 
                  Specifies the maximum allowed difference between successive density matrices.
                  Default is 1e-6.

                - `mix` (float, optional): The mixing factor for the density matrix during updates.
                  This controls the contribution of the new density matrix to the updated one.
                  Values closer to 1 favor the new density matrix, while smaller values favor 
                  smoother convergence. Default is 0.3.

                - `iterations` (int, optional): The maximum number of iterations allowed in the 
                  self-consistency cycle. Default is 500.

                - `coulomb_strength` (float, optional): A scaling factor for the Coulomb matrix.
                  This allows tuning of the strength of Coulomb interactions in the system. 
                  Default is 1.0.
        
                - `f_mean_field` (Callable, optional): A function for computing the mean field term.
                  First argument is density matrix, second argument is single particle hamiltonian.
                  Can be used, e.g., for full HF by passing a closure containing ERIs.       
                  Default is None.
        
                - `f_build` (Callable, optional): Construction of the density matrix from energies and eigenvectors. If None, single-particle energy levels are filled according to number of electrons.
                  Default is None.
        
                - `rho_0` (jax.Array, optional): Initial guess for the density matrix. If None, zeros are used.
                   Default is None.

        Example:
            >>> model.set_mean_field(accuracy=1e-7, mix=0.5, iterations=1000)
            >>> print(model.params.mean_field_params)
            {'accuracy': 1e-7, 'mix': 0.5, 'iterations': 1000, 'coulomb_strength': 1.0, 'f_mean_field': None}
        """
        default = {"accuracy" : 1e-6, "mix" : 0.3, "iterations" : 500, "coulomb_strength" : 1.0, "f_mean_field" : None, "f_build" : None, "rho_0" : None}
        self.params.mean_field_params = default | kwargs


    @mutates
    def set_excitation(self, from_state, to_state, excited_electrons):
        """
        Sets up an excitation process from one state to another with specified electrons.

        Parameters:
            from_state (int, list, or jax.Array): The initial state index or indices.
            to_state (int, list, or jax.Array): The final state index or indices.
            excited_electrons (int, list, or jax.Array): The indices of electrons to be excited.

        Note:
            The states and electron indices may be specified as scalars, lists, or arrays.
        """
        def maybe_int_to_arr(maybe_int):
            if isinstance(maybe_int, int):
                return jnp.array([maybe_int])
            if isinstance(maybe_int, list):
                return jnp.array(maybe_int)
            raise TypeError

        self.params.excitation = [maybe_int_to_arr(from_state), maybe_int_to_arr(to_state), maybe_int_to_arr(excited_electrons)]
        
    @property
    def positions(self):
        return jnp.array([orb.position for orb in self._list])

    @property
    def electrons( self ):
        return self.params.electrons

    @mutates
    def set_electrons( self, val ):
        assert val <= self.params.spin_degeneracy * len(self), "Max electrons exceeded"
        self.params.electrons = val

    @property
    def eps( self ):
        return self.params.eps

    @mutates
    def set_eps( self, val ):
        self.params.eps = val

    @property
    def spin_degeneracy( self ):
        return self.params.spin_degeneracy
        
    @property
    @recomputes
    def homo(self):
        return (self.electrons * self.stationary_density_matrix_e).real.diagonal().round(2).nonzero()[0][-1].item()
    
    @property
    @recomputes
    def lumo(self):
        return (self.electrons * self.stationary_density_matrix_e).real.diagonal().round(2).nonzero()[0][-1].item() + 1
    
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
                jnp.diag(self.positions[:, i] / 2) #-jnp.average(self.positions[:,i]))
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
    def oam_operator(self):
        """
        Calculates the orbital angular momentum operator from the dipole $P$ and velocity operator $J$ as $L_{k} = \epsilon_{ijk} P_j J_k$.
        
        Returns:
           jax.Array: A 3 x N x N tensor representing the orbital angular momentum operator
        """
        epsilon = jnp.array([[[ 0,  0,  0],
                             [ 0,  0,  1],
                             [ 0, -1,  0]],
                            [[ 0,  0, -1],
                             [ 0,  0,  0],
                             [ 1,  0,  0]],
                            [[ 0,  1,  0],
                             [-1,  0,  0],
                             [ 0,  0,  0]]])
        
        return jnp.einsum('ijk,jlm,kmn->iln', epsilon, self.dipole_operator, self.velocity_operator)

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
        charge = 1.602e-19   # C
        eps_0 = 8.85 * 1e-12 # F/m
        hbar = 1.0545718 * 1e-34 # Js
        c = 3e8  # 137 (a.u.) # m/s
        angstroem = 1e-10 # m
        factor = (charge/hbar)**3 * (charge*angstroem)**2  / (3 * jnp.pi * eps_0 * hbar * c**3)
        te = self.transition_energies
        transition_dipole_moments_squared = jnp.sum(self.dipole_operator_e**2, axis = 0)
        factor2 = hbar/charge # transfer Gamma back to code units
        return (
            (te * (te > self.eps)) ** 3
            * transition_dipole_moments_squared
            * factor * factor2
        ).real

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
    def get_charge(self, density_matrix = None):
        """
        Calculates the charge distribution from a given density matrix or from the initial density matrix if not specified.

        Parameters:
           density_matrix (jax.Array, optional): The density matrix to use for calculating charge. 
                                                 If omitted, the initial density matrix is used.

        Returns:
           jax.Array: A diagonal array representing charges at each site.
        """
        density_matrix = self.initial_density_matrix if density_matrix is None else density_matrix
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

    def get_args( self, relaxation_rate = 0.0, coulomb_strength = 1.0, propagator = None):
        return TDArgs(
            self.hamiltonian,
            self.energies,
            self.coulomb * coulomb_strength,
            self.initial_density_matrix,
            self.stationary_density_matrix,
            self.eigenvectors,
            self.dipole_operator,
            self.electrons,
            relaxation_rate,
            propagator,
            self.spin_degeneracy,
            self.positions
            )

    @staticmethod
    def get_hamiltonian(illumination = None, use_rwa = False, add_induced = False):
        """Dict holding terms of the default hamiltonian: bare + coulomb + dipole gauge coupling to external field  + (optional) induced field (optionally in RWA)"""
        contents = {}
        contents["bare_hamiltonian"] = potentials.BareHamiltonian()
        contents["coulomb"] = potentials.Coulomb()
        if illumination is not None:
            contents["potential"] = potentials.DipoleGauge(illumination, use_rwa)
        if add_induced == True:
            contents["induced"] = potentials.Induced( )
        return contents

    @staticmethod
    def get_dissipator(relaxation_rate = None, saturation = None):
        """Dict holding the term of the default dissipator: either decoherence time from relaxation_rate as float and ignored saturation or lindblad from relaxation_rate as array and saturation function"""
        if relaxation_rate is None and saturation is None:
            return {"no_dissipation" : lambda t, r, args : 0.0}
        if isinstance(relaxation_rate, float):
            return { "decoherence_time" : dissipators.DecoherenceTime() }
        func  = (lambda x: 1 / (1 + jnp.exp(-1e6 * (2.0 - x)))) if saturation is None else saturation
        return {"lindblad" : dissipators.SaturationLindblad(func) }        

    def get_postprocesses( self, expectation_values, density_matrix ):
        postprocesses = {}
        if isinstance(expectation_values, jax.Array):
            expectation_values = [expectation_values]
        if expectation_values is not None:
            ops = jnp.concatenate( expectation_values)
            postprocesses["expectation_values"] = lambda rho, args: self.get_expectation_value(operator=ops,density_matrix=rho)

        if density_matrix is None:
            return postprocesses

        if isinstance(density_matrix, str):
            density_matrix = [density_matrix]
        for option in density_matrix:
            if option == "occ_x":
                postprocesses[option] = lambda rho, args: args.electrons * jnp.diagonal(rho, axis1=-1, axis2=-2) 
            elif option == "occ_e":
                postprocesses[option] = lambda rho, args: args.electrons * jnp.diagonal( args.eigenvectors.conj().T @ rho @ args.eigenvectors, axis1=-1, axis2=-2) 
            elif option == "full":
                postprocesses[option] = lambda rho, args: rho
            elif option == "diag_x":
                postprocesses[option] = lambda rho, args: jnp.diagonal(rho, axis1=-1, axis2=-2)
            elif option == "diag_e":
                postprocesses[option] = lambda rho, args: jnp.diagonal( args.eigenvectors.conj().T @ rho @ args.eigenvectors, axis1=-1, axis2=-2) 
            

        return postprocesses


    @recomputes
    def master_equation(            
            self,
            *,
            end_time : float,
            start_time : float = 0.0,
            dt : float = 1e-4,
            grid : Union[int, jax.Array] = 100,
            max_mem_gb : float = 0.5,

            initial_density_matrix : Optional[jax.Array] = None,

            coulomb_strength : float = 1.0,
            
            illumination : Callable = None,
            
            relaxation_rate : Optional[Union[float, jax.Array]] = None,

            compute_at : Optional[jax.Array] = None,

            expectation_values : Optional[list[jax.Array]] = None,
            density_matrix : Optional[list[str]] = None,

            use_rwa : bool = False,

            solver = diffrax.Dopri5(),
            stepsize_controller = diffrax.PIDController(rtol=1e-10,atol=1e-10),

            hamiltonian : dict = None,
            dissipator : dict = None,
            postprocesses : dict = None,
            rhs_args = None,

    ):
        """
        Simulates the time evolution of the density matrix, computing observables, density matrices or extracting custom information.

        Args:
            end_time (float): The final time for the simulation.
            start_time (float): The starting time for the simulation. Defaults to 0.0.
            dt (float): The time step size for the simulation. Defaults to 1e-4.
            grid (Union[int, jax.Array]): Determines the output times for the simulation results. If an integer, results
                                          are saved every 'grid'-th time step. If an array, results are saved at the
                                          specified times.
            max_mem_gb (float): Maximum memory in gigabytes allowed for each batch of intermediate density matrices.
            initial_density_matrix (Optional[jax.Array]): The initial state of the density matrix. If not provided,
                                                          `self.initial_density_matrix` is used.
            coulomb_strength (float): Scaling factor for the Coulomb interaction matrix.
            illumination (Callable): Function describing the time-dependent external illumination applied to the system.
            relaxation_rate (Union[float, jax.Array, Callable]): Specifies the relaxation dynamics. A float indicates a
                                                                 uniform decoherence time, an array provides state-specific
                                                                 rates.
            compute_at (Optional[jax.Array]): The orbitals indexed by this array will experience induced fields.
            expectation_values (Optional[list[jax.Array]]): Expectation values to compute during the simulation.
            density_matrix (Optional[list[str]]): Tags for additional density matrix computations. "full", "occ_x", "occ_e", "diag_x", "diag_e". May be deprecated.
            computation (Optional[Callable]): Additional computation to be performed at each step.
            use_rwa (bool): Whether to use the rotating wave approximation. Defaults to False.
            solver: The numerical solver instance to use for integrating the differential equations.
            stepsize_controller: Controller for adjusting the solver's step size based on error tolerance.
            hamiltonian: dict of functions representing terms in the hamiltonian. functions must have signature `t, r, args->jax.Array`. keys don't matter.
            dissipator:: dict of functions representing terms in the dissipator. functions must have signature `t, r, args->jax.Array`. keys don't matter.
            postprocesses: (bool): dict of functions representing information to extract from the simulation. functions must have signature `r, args->jax.Array`. keys don't matter.
            rhs_args: arguments passed to hamiltonian, dissipator, postprocesses during the simulation. namedtuple.

        Returns:
            ResultTD
        """


        # arguments to evolution function
        if rhs_args is None:
            rhs_args = self.get_args( relaxation_rate,
                                      coulomb_strength,
                                      _numerics.get_coulomb_field_to_from(self.positions, self.positions, compute_at) )

        if illumination is None:
            illumination = lambda t : jnp.array( [0j, 0j, 0j] )

        # each of these functions is applied to a density matrix batch
        postprocesses = self.get_postprocesses( expectation_values, density_matrix ) if postprocesses is None else postprocesses

        # hermitian rhs
        hamiltonian = self.get_hamiltonian(illumination, use_rwa, compute_at is not None) if hamiltonian is None else hamiltonian

        # non hermitian rhs
        dissipator = self.get_dissipator(relaxation_rate, None) if dissipator is None else dissipator

        # set reasonable default 
        initial_density_matrix = initial_density_matrix if initial_density_matrix is not None else rhs_args.initial_density_matrix
        
        try:        
            return self._integrate_master_equation( list(hamiltonian.values()), list(dissipator.values()), list(postprocesses.values()), rhs_args, illumination, solver, stepsize_controller, initial_density_matrix, start_time, end_time, grid, max_mem_gb, dt )
        except Exception as e:
            print(f"Simulation crashed with exception {e}. Try increasing the time mesh and make your sure your illumination is differentiable. The full diffrax traceback follows below.")
            traceback.print_stack()

    @staticmethod
    def _integrate_master_equation( hamiltonian, dissipator, postprocesses, rhs_args, illumination, solver, stepsize_controller, initial_density_matrix, start_time, end_time, grid, max_mem_gb, dt ):
        
        # batched time axis to save memory 
        mat_size = initial_density_matrix.size * initial_density_matrix.itemsize / 1e9
        time_axis = _numerics.get_time_axis( mat_size = mat_size, grid = grid, start_time = start_time, end_time = end_time, max_mem_gb = max_mem_gb, dt = dt )

        ## integrate
        final, output = _numerics.td_run(
            initial_density_matrix,
            _numerics.get_integrator(hamiltonian, dissipator, postprocesses, solver, stepsize_controller, dt),
            time_axis,
            rhs_args)
        
        return TDResult(
            td_illumination = jax.vmap(illumination)(jnp.concatenate(time_axis)) ,
            output = output,
            final_density_matrix = final,
            time_axis = jnp.concatenate( time_axis )
        )

    
    def get_ip_green_function(self, A, B, omegas, occupations = None, energies = None, mask = None, relaxation_rate = 1e-1):
        """independent-particle greens function at the specified frequency according to 

        $$
        G_{AB}(\omega) = \sum_{nm} \\frac{P_m - P_n}{\omega + E_m - E_n + i e} A_{nm} B_{mn}
        $$

        Parameters: 
          A, B : operators *in energy basis*, square jax.Array
          omegas (jax.Array) : frequency grid
          rho_e (jax.Array) : energy occupations, if omitted, current density matrix diagonal is used
          energies (jax.Array) : energies, if omitted, current energies are used
          mask (jax.Array): boolean mask excluding energy states from the summation
          relaxation_rate (float): broadening parameter
        
        Returns:
          jax.Array: Values of the Green's function
        """

        def inner(omega):
            return jnp.trace( (delta_occ / (omega + delta_e + 1j*relaxation_rate)) @ operator_product)

        print("Computing Greens function. Remember we default to site basis")
        
        operator_product =  A.T * B
        occupations = self.initial_density_matrix_e.diagonal() * self.electrons if occupations is None else occupations
        energies = self.energies if energies is None else energies        
        delta_occ = (occupations[:, None] - occupations)
        if mask is not None:        
            delta_occ = delta_occ.at[mask].set(0) 
        delta_e = energies[:, None] - energies
        
        return jax.lax.map(jax.jit(inner), omegas)

    def get_polarizability_rpa(
        self,
        omegas,            
        polarization,
        coulomb_strength=1.0,
        relaxation_rate=1/10,
        hungry=0,
        phi_ext=None,
        args = None,
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
           args (Optional): numeric representation of an orbital list, as obtained by `get_args`

        Returns:
           jax.Array: The calculated polarizabilities at the specified frequencies.
        """

        if args is None:
            args = self.get_args(relaxation_rate = relaxation_rate, coulomb_strength = coulomb_strength, propagator = None)
        alpha = _numerics.rpa_polarizability_function(args, polarization, hungry, phi_ext)
        if omegas.ndim == 1:        
            return jax.lax.map(alpha, omegas)
        else:
            return jnp.concatenate( [ jax.vmap(alpha)(omega) for omega in omegas ] )

    def get_susceptibility_rpa(
            self, omegas, relaxation_rate=1/10, coulomb_strength=1.0, hungry=0, args = None,
    ):
        """
        Computes the random phase approximation (RPA) susceptibility of the system over a range of frequencies.

        Parameters:
           omegas (jax.Array): The frequencies at which to compute susceptibility.
           relaxation_rate (float): The relaxation time affecting susceptibility calculations.
           coulomb_strength (float): The strength of Coulomb interactions considered in the calculations.
           hungry (int): speed up the simulation up, higher numbers (max 2) increase RAM usage.
           args (Optional): numeric representation of an orbital list, as obtained by `get_args`

        Returns:
           jax.Array: The susceptibility values at the given frequencies.
        """
        if args is None:
            args = self.get_args(relaxation_rate = relaxation_rate, coulomb_strength = coulomb_strength, propagator = None)
        sus = _numerics.rpa_polarizability_function( args, hungry )
        return jax.lax.map(sus, omegas)    
    
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
