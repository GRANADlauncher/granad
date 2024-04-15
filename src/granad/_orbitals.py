from dataclasses import dataclass, fields, field
import jax.numpy as jnp
from functools import wraps
import jax
import warnings
import uuid

# TODO: hmmm
from ._core import *
from . import _observables
from . import _watchdog

# TODO: some naming conventions are weird, e.g. position_operator should be get_position_operator etc., I use ints and uuid interchangeably, which is also not great, maybe command pattern to guard against illegal dict inserts, perhaps do HF for parameter estimation?
    
## CLASSES
@dataclass(frozen  = True)
class Orbital:
    """An orbital.
    
     - `orbital_id`: Orbital ID, e.g. "pz_graphene".
     - `position`: Orbital position like $(x,y,z)$, in Ångström.
     - `occupation`: number of electrons in the non-interacting orbital (`1` or `0`)
    """

    orbital_name : str
    position : tuple[float, float, float]
    occupation : int = 1
    uuid : int = field(default_factory=_watchdog._Watchdog.next_value)
    quantum_numbers : tuple[int, int, int, int]  = None

    # TODO: bla bla bla ... this should be shorter but im too tired
    def __eq__(self, other):
        if not isinstance(other, Orbital):
            return NotImplemented
        return self.uuid == other.uuid and self.position == other.position and self.orbital_name == other.orbital_name and self.atom == other.atom

    def __lt__(self, other):
        if not isinstance(other, Orbital):
            return NotImplemented
        return self.uuid < self.uuid

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __ne__(self, other):
        return not self == other
    
def _default_jax_array():
    return jnp.array([0, 0, 0])

@dataclass
class Params:
    """simulation parameters. specify when you have finished setting up your orbital list. wiped whenever orbital lists are concatenated or new elements are added.
    """
    
    # TODO: uff... quick and dirty
    from_state : jax.Array = field(default_factory=_default_jax_array)
    to_state : jax.Array = field(default_factory=_default_jax_array)
    excited_electrons : jax.Array = field(default_factory=_default_jax_array)
    doping : int = 0
    eps : float = 1e-5
    beta : jax.Array= jnp.inf        
    sc_params : dict = field(default_factory=dict)
    transitions : dict = field(default_factory=dict)
    spin_degeneracy = 2.0
    
    # self-consistency
    sc = False

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

    def uuid_items(self):
        """Yields items where all elements of the key tuple are strings."""
        for key, value in self.items():
            if all(isinstance(k, int) for k in key):
                yield (key, value)
                
    def orbital_items(self):
        """Yields items where all elements of the key tuple are strings."""
        for key, value in self.items():
            if all(isinstance(k, Orbital) for k in key):
                yield (key, value)


# TODO: uff
def mutates(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args[0]._recompute_stack = True
        return func(*args, **kwargs)
    return wrapper

def add_observables_methods(cls):
    for name in dir(_observables):
        method = getattr(_observables, name)
        if callable(method) and not name.startswith("__"):
            setattr(cls, name, method)
    return cls

class OrbitalList:
    """A list of orbitals.
    """
    
    def __init__(self, orbs, hopping = None, coulomb = None ):
        # couplings are dicts mapping orbital pairs to couplings
        self._hopping = hopping if hopping is not None else SortedTupleDict()
        self._coulomb = coulomb if coulomb is not None else SortedTupleDict()
        
        # contains all high-level simulation information
        self._list = list(orbs) if orbs is not None else []
        self.params = Params()
                
        # contains all numerical simulation information
        self._stack = None
        self._recompute_stack = True

        # list of fields that are directly looked up from the stack
        self._stack_fields = [ n.name for n in fields(Stack)]
        
    def __len__(self):
        return len(self._list)
    
    # can't mutate, because orbitals are immutable
    def __getitem__(self, position):
        return self._list[position]    
        
    def __str__(self):
        return str(self._list)

    def __repr__(self):
        def concatenate_couplings( cd  ):
            res = ''
            for key, val in cd.items():
                res += f'{key}, {val}'
            return res
                
        info = f"Orbital list with {len(self)} orbitals.\n"
        hop = concatenate_couplings( self._hopping )
        coul = concatenate_couplings( self._coulomb )
        params = str( self.params )
        res = info + hop + coul + params        
        return res 

    def __iter__(self):
        return iter(self._list)

    # TODO: uff, addition, or, in general, mutation should wipe all attributes except for coupling
    def __add__(self, other):        
        if not self._are_orbs( other ):
            return other + self

        if isinstance( other, OrbitalList ):
            self._hopping.update( other._hopping )
            self._coulomb.update( other._coulomb )
            
        return OrbitalList(self._list + list(other), self._hopping, self._coulomb)                    
    
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
        self._delete_coupling( orb, self._hopping )
        self._delete_coupling( orb, self._coulomb )
        del self._list[position]

    def __getattr__(self, item):
        if not item in self._stack_fields:        
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
        
        if self._recompute_stack:
            self.build()
            self._recompute_stack = False
            
        return getattr( self.stack, item )

    
    @staticmethod
    def _are_orbs(candidate):
        return all(isinstance(orb, Orbital) for orb in candidate)

    @mutates
    def _set_coupling( self, orb_or_uuid1, orb_or_uuid2, val_or_func, coupling):
        coupling[ (orb_or_uuid1, orb_or_uuid2) ] = val_or_func

    def get_layers( self ):
        return [orb.uuid for orb in self._list]
    
    def get_unique_layers( self ):
        return list( set( self.get_layers() ) )

    def _hamiltonian_coulomb( self ):        

        def fill_matrix( matrix, coupling_dict ):

            # TODO: there should be an internal
            dummy = jnp.arange( len(self._list) )
            triangle_mask = dummy[:,None] >= dummy

            # TODO: in principle we can build a big tensor NxNxlayers, vmap over the last axis and sum the layers
            # first, we loop over all uuid couplings => interactions between layers
            for key, function in coupling_dict.uuid_items():
                # select ids only in upper triangle
                rows = uuid_ints == key[0]
                cols = uuid_ints == key[1]
                valid_indices = jnp.logical_and( triangle_mask, jnp.logical_and( rows, cols ) )
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
        positions = self.get_positions()
        distances = jnp.round( jnp.linalg.norm( positions - positions[:, None], axis = -1 ), 6 )
        uuid_ints = jnp.array( self.get_layers() )
        
        # TODO: mention that uuid == layer uuid
        hamiltonian = fill_matrix( jnp.zeros( (len(self), len(self)) ).astype(complex), self._hopping )
        coulomb = fill_matrix( jnp.zeros( (len(self), len(self)) ).astype(complex), self._coulomb )

        return hamiltonian, coulomb

    
    def _ensure_complex( self, val ):
        return val + 0.0j

    # TODO: bla bla bla ... incredibly verbose, but couldn't think of anything better yet
    def set_layers_hopping( self, uuid1, uuid2, func ):
        self._set_coupling( uuid1, uuid2, func, self._hopping )

    def set_layers_coulomb( self, uuid1, uuid2, func ):
        self._set_coupling( uuid1, uuid2, func, self._coulomb )

    def _maybe_ints_to_orbs( self, maybe_ints ):
        def convert( orb ):
            if isinstance(orb, int):
                return self._list[orb]
            if isinstance(orb, Orbital):
                return orb
            return "You have passed something that is neither an orbital nor an int"
        return [convert(x) for x in maybe_ints]
        
    def set_hamiltonian_element( self, orb_or_int1, orb_or_int2, val ):
        orb1, orb2 = self._maybe_ints_to_orbs( (orb_or_int1, orb_or_int2) )              
        self._set_coupling( orb1, orb2, self._ensure_complex(val), self._hopping )

    def set_coulomb_element( self, orb_or_int1, orb_or_int2, val ):
        orb1, orb2 = self._maybe_ints_to_orbs( (orb_or_int1, orb_or_int2) )
        self._set_coupling( orb1, orb2, self._ensure_complex(val), self._coulomb )

    def append(self, other):
        if not isinstance(other, Orbital):
            raise TypeError
        if other in self:
            raise ValueError            
        self._list.append( other )

    def build( self ):
        
        hamiltonian, coulomb = self._hamiltonian_coulomb()
                
        eigenvectors, energies = jax.lax.linalg.eigh(hamiltonian)
        
        # TODO: this is too verbose, but to my credit, the numerics is even worse ... eh wait...
        rho_0, homo = density_matrix(
            energies,
            self.get_electrons(),
            self.params.spin_degeneracy,
            self.params.eps,
            self.params.from_state,
            self.params.to_state,
            self.params.excited_electrons,
            self.params.beta,
        )
        rho_stat, _ = density_matrix(
            energies, self.get_electrons(), self.params.spin_degeneracy, self.params.eps, jnp.array([0]), jnp.array([0]), jnp.array([0]), self.params.beta
        )

        # TODO: fix
        unique_ids = 0
        ids = 0
        stack =  Stack(hamiltonian, coulomb, rho_0, rho_stat, energies, eigenvectors, self.get_positions(), unique_ids, ids, self.params.eps, homo, self.get_electrons(), self.params.from_state, self.params.to_state, self.params.excited_electrons, self.params.beta, self.params.spin_degeneracy, self.params.transitions )
        
        if self.params.sc_params:
            stack = _get_self_consistent( stack, **self.sc_params )

        self.stack = stack

    # TODO: should be private?
    def get_electrons( self ):
        return sum( o.occupation for o in self._list ) + self.params.doping
    
    def get_positions ( self ):
        return jnp.array( [ o.position for o in self._list ] )

    @mutates
    def make_self_consistent( self, sc_params ):
        self.params.sc = sc_params

    # TODO: uff
    @mutates
    def excite( self, from_state, to_state, excited_electrons ):
        from_state = jnp.array([from_state] if isinstance(from_state, int) else from_state)
        to_state = jnp.array([to_state] if isinstance(to_state, int) else to_state)
        excited_electrons = jnp.array(
            [excited_electrons] if isinstance(excited_electrons, int) else excited_electrons
        )
        self.params.from_state = from_state
        self.params.to_state = to_state
        self.params.excited_electrons = excited_electrons
