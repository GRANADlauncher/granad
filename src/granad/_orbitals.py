from dataclasses import dataclass, fields, field
from collections import Counter
from pprint import pformat
from typing import Optional, Callable, Union
from functools import wraps
import jax
import jax.numpy as jnp
import diffrax

# TODO: hmmm
from . import _plotting, _watchdog, _numerics

# TODO: clean this up, repair doc strings, some naming conventions are weird, e.g. position_operator should be get_position_operator, make einsum magic more versatile, rethink the "use private members in recompute" idea
# TODO: HF for parameter estimation

@dataclass(frozen=True)
class Orbital:
    position: tuple[float, float, float]
    tag: Optional[str] = None
    energy_level: Optional[int] = None
    angular_momentum: Optional[int] = None
    angular_momentum_z: Optional[int] = None
    spin: Optional[int] = None
    atom_name: Optional[str] = None
    group_id: int = field(default_factory=_watchdog._Watchdog.next_value)


    def __post_init__(self):
        object.__setattr__(self, 'position', tuple(map(float, self.position)))

    def __str__( self ):
        return pformat( vars(self), sort_dicts = False )

    # TODO: bla bla bla ... this should be shorter but im too tired
    def __eq__(self, other):
        if not isinstance(other, Orbital):
            return NotImplemented
        return self.group_id == other.group_id and self.position == other.position and self.name == other.name and self.info == other.info
    
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
    
# TODO: inheritance :///, typecheck keys
class SortedTupleDict(dict):  
  
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
    for name in dir(_plotting):
        method = getattr(_plotting, name)
        if callable(method) and name.startswith('show'):
            setattr(cls, name, method)
    return cls

@plotting_methods
class OrbitalList:
    """A list of orbitals.
    """
        
    def __init__(self, orbs, _hopping_dict = None, _coulomb_dict = None ):
        # couplings are dicts mapping orbital pairs to couplings
        self._hopping_dict = _hopping_dict if _hopping_dict is not None else SortedTupleDict() 
        self._coulomb_dict =  _coulomb_dict if _coulomb_dict is not None else SortedTupleDict() 
        self._transitions = SortedTupleDict()
        
        # contains all high-level simulation information
        self._list = list(orbs) if orbs is not None else []
                
        # flag for recomputing state
        self._recompute = True

        # TODO: this is a bunch of stuff and must be better doable
        self.from_state = jnp.array([0])    
        self.to_state = jnp.array([0])    
        self.excited_electrons =jnp.array([0])    
        self.eps  = 1e-5
        self.beta = jnp.inf        
        self.self_consistency_params = {}
        self.spin_degeneracy  = 2.0
        self.electrons = len( self._list ) 

    def __len__(self):
        return len(self._list)
    
    # can't mutate, because orbitals are immutable
    def __getitem__(self, position):
        return self._list[position]

    def __repr__( self ):
        return repr(self._list)

    # TODO: hmmm
    def __str__(self):
        info = f"List with {len(self)} orbitals, {self.electrons} electrons."
        excited = f"{self.excited_electrons} electrons excited from {self.from_state} to {self.to_state}."
        groups = '\n'.join( [f'group id {key} : {val} orbitals' for key, val in Counter( self.get_group_ids()).items() ])
        return '\n'.join( (info, excited, groups ) )

    def __iter__(self):
        return iter(self._list)

    # TODO: uff, addition, or, in general, mutation should wipe all attributes except for coupling
    def __add__(self, other):        
        if not self._are_orbs( other ):
            raise TypeError

        if any(orb in other for orb in self._list):
            raise ValueError

        if isinstance( other, OrbitalList ):
            self._hopping_dict.update( other._hopping_dict )
            self._coulomb_dict.update( other._coulomb_dict )
            
        return OrbitalList(self._list + list(other), self._hopping_dict, self._coulomb_dict)                    
    
    @mutates
    def __setitem__(self, position, value):
        if isinstance(value, Orbital):
            self._list[position] = value
        raise TypeError

    def _delete_coupling( self, orb, coupling ):                
        keys_to_remove = [key for key in coupling if orb in key]        
        for key in keys_to_remove:
            del coupling[key]
            
    @mutates
    def __delitem__(self, position):        
        orb = self._list[position]
        self._delete_coupling( orb, self._hopping_dict )
        self._delete_coupling( orb, self._coulomb_dict )
        del self._list[position]
            
    @staticmethod
    def _are_orbs(candidate):
        return all(isinstance(orb, Orbital) for orb in candidate)

    @mutates
    def _set_coupling( self, orb_or_group_id1, orb_or_group_id2, val_or_func, coupling):
        coupling[ (orb_or_group_id1, orb_or_group_id2) ] = val_or_func

    def get_group_ids( self ):
        return [orb.group_id for orb in self._list]
    
    def get_unique_group_ids( self ):
        return list( set( self.get_group_ids() ) )

    # TODO: we may want to differentiate through this, also this is private so better not wrap return val into container
    def _hamiltonian_coulomb( self ):        

        def fill_matrix( matrix, coupling_dict ):

            # TODO: there should be an internal
            dummy = jnp.arange( len(self) )
            triangle_mask = dummy[:,None] >= dummy

            # TODO: in principle we can build a big tensor NxNxgroups, vmap over the last axis and sum the groups
            # first, we loop over all group_id couplings => interactions between groups
            for key, function in coupling_dict.group_id_items():                
                # TODO:  big uff:  we rely on the correct ordering of the group_ids for cols and rows, first key is always smaller than last keys => we get upper triangular valid indices
                # if it were the other way around, these would be zeroed by the triangle mask
                cols = group_ids == key[0]
                rows = (group_ids == key[1])[:, None]
                combination_indices = jnp.logical_and( rows, cols )                
                valid_indices = jnp.logical_and( triangle_mask, combination_indices )
                function = jax.vmap( function )
                matrix = matrix.at[valid_indices].set( function( distances[valid_indices] ) )
                
            # we now set single elements
            rows, cols, vals = [], [], []
            for key, val in coupling_dict.orbital_items():
                rows.append( self._list.index( key[0] ) )
                cols.append( self._list.index( key[1] ) )
                vals.append( val )
                
            matrix = matrix.at[rows, cols].set(vals)

            return matrix + matrix.conj().T - jnp.diag(jnp.diag(matrix))
        
        # TODO: oh noes rounding again, but don't know what to do else
        positions = self._get_positions()
        distances = jnp.round( jnp.linalg.norm( positions - positions[:, None], axis = -1 ), 6 )
        group_ids = jnp.array( self.get_group_ids() )
        
        hamiltonian = fill_matrix( jnp.zeros( (len(self), len(self)) ).astype(complex), self._hopping_dict )
        coulomb = fill_matrix( jnp.zeros( (len(self), len(self)) ).astype(complex), self._coulomb_dict )

        return hamiltonian, coulomb

    def _get_positions( self ):
        return jnp.array( [ o.position for o in self._list ] )
    
    def _ensure_complex( self, func_or_val ):
        if callable(func_or_val):
            return lambda x: func_or_val(x) + 0.0j
        if isinstance(func_or_val, (int, float, complex)):            
            return func_or_val + 0.0j
        raise TypeError

    # TODO: bla bla bla ... incredibly verbose, but couldn't think of anything better yet
    def _maybe_orbs_to_group_ids( self, maybe_orbs ):
        def convert( maybe_orb ):
            # TODO: check if this is really a group_id
            if isinstance(maybe_orb, int):                
                return maybe_orb
            if isinstance(maybe_orb, Orbital):
                return maybe_orb.group_id
            return "You have passed something that is neither an orbital nor a group_id"
        return [convert(x) for x in maybe_orbs]

    def set_groups_hopping( self, orb_or_group_id1, orb_or_group_id2, func ):
        group_id1, group_id2 = self._maybe_orbs_to_group_ids( (orb_or_group_id1, orb_or_group_id2) )
        self._set_coupling( group_id1, group_id2, self._ensure_complex(func), self._hopping_dict )

    def set_groups_coulomb( self, orb_or_group_id1, orb_or_group_id2, func ):
        group_id1, group_id2 = self._maybe_orbs_to_group_ids( (orb_or_group_id1, orb_or_group_id2) )          
        self._set_coupling( group_id1, group_id2, self._ensure_complex(func), self._coulomb_dict )

    def _maybe_indices_to_orbs( self, maybe_indices ):
        def convert( maybe_index ):
            if isinstance(maybe_index, int):
                return self._list[maybe_index]
            if isinstance(maybe_index, Orbital):
                return maybe_index
            return "You have passed something that is neither an orbital nor an index"
        return [convert(x) for x in maybe_indices]
        
    def set_hamiltonian_element( self, orb_or_index1, orb_or_index2, val ):
        orb1, orb2 = self._maybe_indices_to_orbs( (orb_or_index1, orb_or_index2) )              
        self._set_coupling( orb1, orb2, self._ensure_complex(val), self._hopping_dict )

    def set_coulomb_element( self, orb_or_index1, orb_or_index2, val ):
        orb1, orb2 = self._maybe_indices_to_orbs( (orb_or_index1, orb_or_index2) )
        self._set_coupling( orb1, orb2, self._ensure_complex(val), self._coulomb_dict )

    @mutates
    def append(self, other):
        if not isinstance(other, Orbital):
            raise TypeError
        if other in self:
            raise ValueError            
        self._list.append( other )

    def _build( self ):
        
        # TODO: uff
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
        self._stationary_density_matrix  = _numerics._density_matrix(
            self._energies,
            self.electrons,
            self.spin_degeneracy,
            self.eps,
            jnp.array([0]),
            jnp.array([0]),
            jnp.array([0]),
            self.beta
        )
        
        # TODO: uff
        if self.self_consistency_params:
            self._hamiltonian, self._initial_density_matrix, self._stationary_density_matrix, self._energies, self._eigenvectors =  _get_self_consistent(
                self._hamiltonian, self._coulomb, self._positions, self.spin_degeneracy, self.electrons, self.eps, self._eigenvectors, self._static_density_matrix, **self.self_consistent_params )

    # TODO: validate
    @mutates
    def make_self_consistent( self, sc_params ):
        self.params.self_consistent_params = sc_params

    # TODO: uff
    @mutates
    def set_excitation( self, from_state, to_state, excited_electrons ):
        def maybe_int_to_arr( maybe_int ):            
            if isinstance(maybe_int, int):
                return jnp.array([maybe_int])
            if isinstance( maybe_int, list ):            
                maybe_int = jnp.array(maybe_int)
            if isinstance( maybe_int, jax.Array ):            
                return jnp.array(maybe_int) if maybe_int.ndim > 1 else jnp.array([maybe_int])
            raise TypeError
            
        self.from_state = maybe_int_to_arr(from_state)
        self.to_state = maybe_int_to_arr(to_state)
        self.excited_electrons = maybe_int_to_arr(excited_electrons)
        
    @mutates
    def set_dipole_transition( self, orb_or_index1, orb_or_index2, arr ):        
        orb1, orb2 = self._maybe_indices_to_orbs( (orb_or_index1, orb_or_index2) )              
        self._transitions[ (orb_or_index1, orb_or_index2) ] = jnp.array(arr).astype(complex)
        
    # TODO: bla bla bla
    @property
    @recomputes
    def homo(self):
        # TODO: hmmm
        return jnp.diag(self._stationary_density_matrix).nonzero()[0][-1]

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

    # TODO: uff decorator inception, also should return copies to avoid weirdness
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
        dip = self.dipole_operator
        term = jnp.einsum("ijk,jlm->ilkm", dip, dip)
        diag = jnp.einsum("ijk,jlk->il", dip, dip)
        diag = jnp.einsum("ij,kl->ijkl", diag, jnp.eye(term.shape[-1]))
        return 3 * term - diag

    @property
    @recomputes
    def dipole_operator(self):
        N = self.positions.shape[0]
        dipole_operator = jnp.zeros((3, N, N)).astype(complex)
        for i in range(3):
            dipole_operator = dipole_operator.at[i, :, :].set(jnp.diag(self._positions[:, i] / 2))
        for orbital_combination, value in self._transitions.items():
            i, j = self._list.index(orbital_combination[0]), self._list.index(orbital_combination[1])
            k = value.nonzero()[0]
            dipole_operator = dipole_operator.at[k, i, j].set(value[k])
        return dipole_operator + jnp.transpose(dipole_operator, (0, 2, 1)).conj()

    @property
    @recomputes
    def velocity_operator(self):
        if self._transitions is None:
            x_times_h = jnp.einsum("ij,iL->ijL", self._hamiltonian, self._positions)
            h_times_x = jnp.einsum("ij,jL->ijL", self._hamiltonian, self._positions)
        else:
            positions = self.dipole_operator
            x_times_h = jnp.einsum("kj,Lik->Lij", self._hamiltonian, positions)
            h_times_x = jnp.einsum("ik,Lkj->Lij", self._hamiltonian, positions)
        return -1j * (x_times_h - h_times_x)


    @property
    @recomputes
    def transition_energies(self) :
        """Computes independent-particle transition energies associated with the TB-Hamiltonian of a stack.

        - `stack`:

        **Returns:**

        array, the element `arr[i,j]` contains the transition energy from `i` to `j`
        """
        return self._energies[:,None] - self._energies
    
    @property    
    @recomputes
    def wigner_weisskopf_transition_rates(self) :
        """Calculcates Wigner-Weisskopf transiton rates.

            - `stack`:
            - `component`: component of the dipolar transition to take $(0,1,2) \rightarrow (x,y,z)$`


        **Returns:**

         array, the element `arr[i,j]` contains the transition rate from `i` to `j`
        """
        charge = 1.602e-19
        eps_0 = 8.85 * 1e-12
        hbar = 1.0545718 * 1e-34
        c = 3e8  # 137 (a.u.)
        factor = 1.6e-29 * charge / (3 * jnp.pi * eps_0 * hbar**2 * c**3)
        te = self.transition_energies
        transition_dipole_moments = self.transform_to_energy_basis(self.dipole_operator)
        return (
            (te * (te > self.eps)) ** 3
            * jnp.squeeze( transition_dipole_moments ** 2)
            * factor
        )

    @staticmethod
    def _transform_basis( observable, vectors ):
        dims_einsum_strings = {  2 : 'ij,jk,lk->il' , 3 : 'ij,mjk,lk->mil' }
        einsum_string = dims_einsum_strings[(observable.ndim)]
        return jnp.einsum(einsum_string, vectors, observable, vectors.conj())
        
    
    def transform_to_site_basis(self, observable):
        return self._transform_basis( observable, self._eigenvectors)

    def transform_to_energy_basis(self, observable):
        return self._transform_basis( observable, self._eigenvectors.conj().T)

    @recomputes
    def get_charge( density_matrix : None ):
        if density_matrix is None:
            return jnp.diag(self.transform_to_site_basis( self.initial_density_matrix ) * self.electrons)
        else:
            return jnp.diag(density_matrix * self.electrons)
    
    @recomputes
    def get_dos(self, omega: float, broadening: float = 0.1) :
        """IP-DOS of a nanomaterial stack.

        - `stack`: a stack object
        - `omega`: frequency
        - `broadening`: numerical brodening parameter to replace Dirac Deltas with
        """

        broadening = 1 / broadening
        prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
        gaussians = jnp.exp(-((self._energies - omega) ** 2) / 2 * broadening**2)
        return prefactor * jnp.sum(gaussians)

    # TODO: make compatbile with orbital
    @recomputes
    def get_ldos(
        self, omega: float, site_index: int, broadening: float = 0.1
    ) :
        """IP-LDOS of a nanomaterial stack.

        - `stack`: a stack object
        - `omega`: frequency
        - `site_index`: the site index to evaluate the LDOS at
        - `broadening`: numerical brodening parameter to replace Dirac Deltas with
        """

        broadening = 1 / broadening
        weight = jnp.abs(self._eigenvectors[site_index, :]) ** 2
        prefactor = 1 / (jnp.sqrt(2 * jnp.pi) * broadening)
        gaussians = jnp.exp(-((self._energies - omega) ** 2) / 2 * broadening**2)
        return prefactor * jnp.sum(weight * gaussians)

    @recomputes
    def get_epi(self, rho: jax.Array, omega: float, epsilon: float = None) -> float:
        epsilon = self.params.eps if epsilon is None else epsilon
        rho_without_diagonal = jnp.abs(rho - jnp.diag(jnp.diag(rho)))
        rho_normalized = rho_without_diagonal / jnp.linalg.norm(rho_without_diagonal)
        te = self.transition_energies
        excitonic_transitions = (
            rho_normalized / (te * (te > self.eps) - omega + 1j * epsilon) ** 2
        )
        return 1 - jnp.sum(jnp.abs(excitonic_transitions * rho_normalized)) / (
            jnp.linalg.norm(rho_normalized) * jnp.linalg.norm(excitonic_transitions)
        )
    
    @recomputes
    def get_induced_field(
        self, positions: jax.Array, density_matrix
    ) :

        # distance vector array from field sources to positions to evaluate field on
        vec_r = self._positions[:, None] - positions

        # scalar distances
        denominator = jnp.linalg.norm(vec_r, axis=2) ** 3

        # normalize distance vector array
        point_charge = jnp.nan_to_num(
            vec_r / denominator[:, :, None], posinf=0.0, neginf=0.0
        )

        # compute charge via occupations in site basis
        charge = self.electrons * self.transform_to_site_basis(density_matrix).real

        # induced field is a sum of point charges, i.e. \vec{r} / r^3
        e_field = 14.39 * jnp.sum(point_charge * charge[:, None, None], axis=0)
        return e_field

    @staticmethod
    def get_expectation_value( operator, density_matrix ):
        dims_einsum_strings = { (3,2): 'ijk,kj->i', (3,3): 'ijk,lkj->il', (2,3): 'ij,kji->k', (2,2): 'ij,ji->'}
        return jnp.einsum( dims_einsum_strings[(operator.ndim, density_matrix.ndim)], operator, density_matrix )
        
    # TODO: uff, all of the methods below should be rewritten
    def get_expectation_value_time_domain( self, *args, **kwargs ):
        operator = kwargs.pop('operator', None)
        induced = kwargs.pop('induced', True)
        correction = self.transform_to_site_basis(self.stationary_density_matrix) if induced else 0        
        time_axis, density_matrices = self.get_density_matrix_time_domain( *args, **kwargs )
        try:
            return time_axis, self.electrons* self.get_expectation_value( correction - density_matrices.ys, operator )
        except AttributeError:
            return time_axis, self.electrons* self.get_expectation_value( correction - density_matrices, operator )

    def get_expectation_value_frequency_domain(self, *args, **kwargs):
        omega_min = kwargs.pop('omega_min', 0)
        omega_max = kwargs.pop('omega_max', 100)
        time_axis, exp_val_td = self.get_expectation_value_time_domain( *args, **kwargs )        
        omega, exp_val_omega = _numerics.get_fourier_transform( time_axis, exp_val_td )        
        mask = (omega >= omega_min) & (omega <= omega_max)
        try:
            electric_field = jax.vmap(kwargs['illumination'])(time_axis)
            field_omega = _numerics.get_fourier_transform(time_axis, electric_field, return_omega_axis = False)
            return omega[mask], exp_val_omega[mask], field_omega[mask]
        except KeyError:            
            return omega[mask], exp_val_omega[mask]

    # TODO: solve the Lindblad equation analytically
    @recomputes
    def get_density_matrix_time_domain_analytical(
        self,
        end_time: float, 
        illumination: Callable[[float], jax.Array], 
        relaxation_rate: Union[float, jax.Array] = None, 
        steps_time: Optional[int] = None, 
        saturation_functional: Callable[[float], float] = lambda x: 1 / (1 + jnp.exp(-1e6 * (2.0 - x))),
        use_old_method: bool = False,
        include_induced_contribution : bool = False,
        compute_only_at = None,
        coulomb_strength = 1.0):
        return NotImplemented

    # TODO: das hier ist absolut überkonfigurierbar und kotzt mich richtig an
    @recomputes
    def get_density_matrix_time_domain(
        self,
        end_time : float,
        illumination: Callable[[float], jax.Array], 
        start_time: Optional[float] = None,
        steps_time : Optional[int] = None,
        skip : Optional[int] = None,
        relaxation_rate: Union[float, jax.Array] = None, 
        saturation_functional: Callable[[float], float] = lambda x: 1 / (1 + jnp.exp(-1e6 * (2.0 - x))),
        use_old_method: bool = False,
        include_induced_contribution : bool = False,
        use_rwa = False,
        compute_only_at = None,
        coulomb_strength = 1.0,
        solver = diffrax.Dopri5(),
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    ):
        # Time axis creation
        start_time = float(start_time) if start_time is not None else 0.0
        steps_time = int(steps_time) if steps_time is not None else int(end_time * 1000)
        time_axis = jnp.linspace(start_time, end_time, steps_time)
        skip = skip if skip is not None else 1            

        # Determine relaxation function based on the input type
        if relaxation_rate is None:
            relaxation_function = lambda r : 0.0
        elif isinstance(relaxation_rate, jax.Array):
            relaxation_function = _numerics.lindblad_saturation_functional(
                self._eigenvectors, relaxation_rate, saturation_functional, self.electrons, self._stationary_density_matrix)
        else:
            relaxation_function = _numerics.relaxation_time_approximation(
                relaxation_rate, self.transform_to_site_basis(self._stationary_density_matrix) )            

        # Verify that illumination is a callable
        if not callable(illumination):
            raise TypeError("Provide a function for e-field")
        
        # Initialize common variables
        initial_density_matrix = self.transform_to_site_basis(self._initial_density_matrix)
        stationary_density_matrix = self.transform_to_site_basis(self._stationary_density_matrix)
        coulomb_field_to_from = _numerics.get_coulomb_field_to_from( self.positions, self.positions, compute_only_at )        

        # TODO: not very elegant: we just dump every argument in there by default
        return time_axis[::skip], _numerics.integrate_master_equation(
            self._hamiltonian, coulomb_strength * self._coulomb,
            self.dipole_operator, self.electrons, self.velocity_operator,
            initial_density_matrix, stationary_density_matrix,
            time_axis, illumination, relaxation_function,
            coulomb_field_to_from, include_induced_contribution, use_rwa,
            solver, stepsize_controller, use_old_method, skip )
    
    # TODO: uff, again verbose
    def get_polarizability_rpa( omegas, relaxation_time, polarization, coulomb_strength = 1.0, hungry = False, phi_ext = None ):
        alpha = _numerics.rpa_polarizability_function( self, relaxation_time, polarization, coulomb_strength, phi_ext, hungry  )
        return jax.lax.map(alpha, omegas)

    
    def get_susceptibility_rpa( omegas, relaxation_time, coulomb_strength = 1.0, hungry = False ):
        sus = _numerics.rpa_polarizability_function( self, relaxation_time, coulomb_strength, hungry  )
        return jax.lax.map(sus, omegas)
