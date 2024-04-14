from flax import struct
from dataclasses import dataclass, fields, field
import numpy as np
import jax.numpy as jnp
from matplotlib.path import Path
from functools import wraps
import jax
import itertools
import warnings
from granad import *
import json
from _shapes import shapes
from pkg_resources import resource_filename, resource_stream
import uuid

# TODO JAXIFX

@struct.dataclass
class Stack:
    """A stack of orbitals.

    - `hamiltonian`: TB-Hamiltonian of the system.
    - `coulomb`: Coulomb matrix of the system.
    - `rho_0`: Density matrix corresponding to the initial state. If a simulation is interrupted at time t, then it can be resumed using the density matrix at time t as rho_0.
    - `rho_stat`: Density matrix corresponding to the final state.
    - `energies`: Independent particle energies as given by the eigenvalues of the TB-hamiltonian.
    - `eigenvectors`: Eigenvectors of the TB-Hamiltonian. These are used to transform between *site* and *energy* basis.
    - `positions`: Positions of the orbitals, $N \\times 3$ - dimensional.
    - `unique_ids`: Collection of unique strings. Each string identifies an orbital type occuring in the stack. Typically something like `"pz_graphene"`.
    - `ids`: The integer at the n-th position corresponds to the type of the n-th orbital via its index in `unique_ids`.
    - `eps`: numerical precision threshold for identifiying degeneracies in the eigenenergies.
    - `homo`: index of the homo, such that stack.energies[homo] is the energy of the ground state homo.
    - `electrons`: total number of electrons in the system.
    - `from_state`: the state from which electrons have been taken for the initial excited transition, the state translates to HOMO - from_state
    - `to_state`: the state into which electrons haven been put for the initial excited transition, the state translates to HOMO + from_state
    - `beta`: corresponds to $(k_B T)^{-1}$
    """

    hamiltonian: jax.Array
    coulomb: jax.Array
    rho_0: jax.Array
    rho_stat: jax.Array
    energies: jax.Array
    eigenvectors: jax.Array
    positions: jax.Array
    unique_ids: list[str] # really needed? only once for dipoles
    ids: jax.Array  # really needed? only once for dipoles
    eps: float
    homo: int
    electrons: int
    from_state: int
    to_state: int
    excited_electrons: int
    beta: float
    spin_degeneracy: float
    transitions: dict

# TODO: this is not thread-safe
class Counter:
    _counter = 0

    @classmethod
    def next_value(cls):
        cls._counter += 1
        return cls._counter

## CLASSES
@dataclass(frozen  = True)
class Orbital:
    """An orbital.
    It is only used briefly during stack construction and immediately discarded after.

     - `orbital_id`: Orbital ID, e.g. "pz_graphene".
     - `position`: Orbital position like $(x,y,z)$, in Ångström.
     - `occupation`: number of electrons in the non-interacting orbital (`1` or `0`)
    """

    orbital_name : str
    position : tuple[float, float, float]
    occupation : int = 1
    atom : str = ''
    material = None
    uuid : int = field(default_factory=Counter.next_value)

    # TODO: bla bla bla ... this should be shorter but im too tired
    def __eq__(self, other):
        if not isinstance(other, Orbital):
            return NotImplemented
        return self.uuid == other.uuid

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

# TODO: cleanup this horrible code, this was just quick-and-dirty
# TODO: 3D extension?
@dataclass(frozen = True)
class Material:
    # dictionary mapping orbital names to atomic positions in the bulk unit cell like "orbital_id" : [x,y,z] 
    orbitals : dict    
    # bulk lattice basis: only used if you want to cut finite pieces from an infinite bulk
    lattice_basis : list = None
    lattice_constant : float = None    
    # coupling information
    hopping : dict = None
    coulomb : dict = None

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.loads(json.load(f))
        return cls(**data)

    @property
    def _orbs( self ):
        return [ o for o in self.orbitals.keys() ]

    def get_translations( self, repetitions ):
        return list(itertools.product( *(range(-repetitions, repetitions+1) for x in range(self.basis_size) ) ))
        
    def cover_plane( self, translation_units, orbs = None ):
        if orbs is None:
            orbs = self._orbs

        positions = []
        for orb in orbs:            
            orbital_positions = self.orbitals[orb] 
            orbital_shifts = np.array(orbital_positions) @ self.lattice_basis        
            translations =  np.array(translation_units) @ self.lattice_basis        
            pos = jnp.array( [shift + translations for shift in orbital_shifts] )
            pos = self.lattice_constant * pos.reshape( pos.shape[0] * pos.shape[1], 3 )
            positions.append( pos )
        return positions

    @property
    def basis_size(self):
        size = np.array(self.lattice_basis).shape[0]
        if size > 2:
            raise NotImplementedError
        return size

    @staticmethod
    def _prune( points, remove_neighbors, remaining_old = jnp.inf ):
        if remove_neighbors <= 0:
            return points
        distances = jnp.round( jnp.linalg.norm( points - points[:, None], axis = -1), 6 )
        minimum = jnp.unique(distances)[1]
        mask = (distances <= minimum).sum(axis = 1)  > remove_neighbors
        # remove every point with less neighbors
        remaining = mask.sum()
        if remaining_old == remaining:            
            return points[mask, :]
        else:
            return Material._prune( points[mask, :], remove_neighbors, remaining )

    # very nasty, TODO: find better solution than rounding stuff, hardcoding 1e-5
    def neighbor_couplings_to_function(self, couplings, orb_1, orb_2 ):
        translations = self.get_translations( len(couplings) )
        positions = jnp.concatenate( self.cover_plane( translations, [orb_1, orb_2] ) )
        couplings = jnp.array( couplings ) + 0.0j
        
        distances = jnp.unique(
            jnp.round(jnp.linalg.norm(positions - positions[:, None, :], axis=2), 8)
        )[: len(couplings)]

        def inner(d):
            return jax.lax.cond(
                jnp.min(jnp.abs(d - distances)) < 1e-5,
                lambda x: couplings[jnp.argmin(jnp.abs(x - distances))],
                lambda x: 0.0j,
                d,
            )

        return inner

    @staticmethod
    def join_keys( keys ):
        return '-'.join( keys )
    
    def _set_couplings( self, func, cdict, uuid ):
        for orb1 in self._orbs:
            for orb2 in self._orbs:

                try:
                    distance_func = self.neighbor_couplings_to_function( cdict[self.join_keys( (orb1, orb2) )], orb1, orb2 )
                    func( uuid, uuid, distance_func )                    
                except KeyError:
                    pass


    # TODO: planes too big in some cases
    def cut( self, polygon_vertices, preview = False, remove_neighbors : int = 2 ):        
        # Unzip into separate lists of x and y coordinates
        x_coords, y_coords = zip(*polygon_vertices)
        
        # Find the maximum x and y values
        max_x = np.abs(np.array(x_coords)).max()
        max_y = np.abs(np.array(y_coords)).max()
        lattice_max_x = self.lattice_constant * np.max( self.lattice_basis, axis = 0)[0]        
        lattice_max_y = self.lattice_constant * np.max( self.lattice_basis, axis = 0)[1]

        # translate by lattice vectors
        repetitions = int( np.nan_to_num( np.ceil(max( max_x / lattice_max_x, max_y / lattice_max_y )), posinf = 0) )
        translation_units = self.get_translations( repetitions )

        # get all positions in a plane
        all_positions = self.cover_plane( translation_units )        
        polygon = Path( polygon_vertices )
        flags = [ polygon.contains_points( p[:,:2] ) for p in all_positions ]

        # TODO: uff again
        positions, orbs = [], []        
        for i,p in enumerate(flags):
            positions.append( self._prune(all_positions[i][flags[i]], remove_neighbors) )
            orbs += [ self._orbs[i] for x in range(positions[-1].shape[0]) ]
        positions = jnp.concatenate(  positions )

        # TODO: big uff, all_positions changes type depending on bool
        if preview:
            all_positions = jnp.concatenate( all_positions )
            fig, ax = plt.subplots()
            patch = plt.Polygon(polygon_vertices[:-1], edgecolor='orange', facecolor='none', linewidth=2)
            ax.add_patch(patch)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='datalim')
            plt.grid(True)
            plt.scatter( x = all_positions[:,0], y = all_positions[:,1])
            plt.scatter( x = positions[:,0], y = positions[:,1])
            plt.axis('equal')
            plt.show()

        uuid = Counter.next_value()
        orbital_list = OrbitalList( [ Orbital( position = tuple(p), orbital_name = orbs[i], uuid = uuid  ) for i,p in enumerate(positions) ] )

        self._set_couplings( orbital_list.set_layers_hopping, self.hopping, uuid )
        self._set_couplings( orbital_list.set_layers_coulomb, self.coulomb, uuid )

        return orbital_list

    
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
    
    # non-empirical TB
    slater_koster = False

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


# TODO: state pattern for emprical, SK ?
class OrbitalList:

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
        return f"{self.__class__.__name__}({self._list})"

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

    # TODO: this is incredibly dangerous if there are couplings pointing to this orbital, would need to check similar things with slicing
    @mutates
    def __delitem__(self, position):
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

    # any modification of couplings SHOULD ONLY HAPPEN HERE!!!, TODO: type check
    @mutates
    def _set_coupling( self, orb_or_string1, orb_or_string2, val_or_func, coupling):
        coupling[ (orb_or_string1, orb_or_string2) ] = val_or_func

    def get_layers( self ):
        return list( map( lambda orb : orb.uuid, self._list) )
    
    def get_unique_layers( self ):
        return list( set( self.get_layers() ) )

    def _hamiltonian_coulomb_empirical( self ):        

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

            return matrix + matrix.conj().T
        
        # TODO: oh noes rounding again, but don't know what to do else
        positions = self.get_positions()
        distances = jnp.round( jnp.linalg.norm( positions - positions[:, None], axis = -1), 6 )
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
        
    def set_hamiltonian_element( self, orb1, orb2, val ):
        if isinstance(orb1, int):
            orb1 = self._list[orb1]
        if isinstance(orb2, int):
            orb2 = self._list[orb2]
        self._set_coupling( orb1, orb2, self._ensure_complex(val), self._hopping )

    def set_coulomb_element( self, orb1, orb2, val ):
        if isinstance(orb1, int):
            orb1 = self._list[orb1]
        if isinstance(orb2, int):
            orb2 = self._list[orb2]
        self._set_coupling( orb1, orb2, self._ensure_complex(val), self._coulomb )
        
    # TODO: implement
    def append(self, other):
        return NotImplemented

    def build( self ):
        
        hamiltonian, coulomb = self._hamiltonian_coulomb_empirical( )

        if self.params.slater_koster:
            # in Slater-Koster, we compute the Hamiltonian from the Orbital shape
            hamiltonian_sk, coulomb_sk = self._hamiltonian_coulomb_slater_koster()
            hamiltonian, coulomb = combine(hamiltonian, hamiltonian_sk), combine(coulomb, coulomb_sk)
                
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

    def relax( self ):
        return NotImplemented    

    def _hamiltonian_coulomb_slater_koster():
        return NotImplemented

def test_recompute():
    pos = [0.0, 0.0, 0.0]
    orbs =  OrbitalList( [Orbital("A", pos)] )
    assert orbs.recompute_stack == True
    orbs.energies
    assert orbs.recompute_stack == False

def test_set_elements():
    pos = [0.0, 0.0, 0.0]
    orbs1 =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
    orbs1.set_hamiltonian_element( 0, 1, 0.3j )
    assert orbs1.hamiltonian[0, 1] == 0.3j
    assert orbs1.hamiltonian[1, 0] == -0.3j

def test_combine_lists():
    pos = [0.0, 0.0, 0.0]
    orbs1 =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
    orbs1.set_hamiltonian_element( 0, 1, 0.3j )
    assert orbs1.hamiltonian[0, 1] == 0.3j
    assert orbs1.hamiltonian[1, 0] == -0.3j

    orbs1.set_hamiltonian_element( 0, 0, 0.3 )
    assert orbs1.hamiltonian[0, 0] == 0.3

    orbs1.set_hamiltonian_element( 1, 1, 0.4 )
    assert orbs1.hamiltonian[1, 1] == 0.4

    orbs2 =  OrbitalList( [ Orbital("C", pos), Orbital("D", pos) ] )
    orbs2.set_hamiltonian_element( 0, 0, 0.1 )
    orbs2.set_hamiltonian_element( 1, 1, 0.2 )
    orbs2.set_hamiltonian_element( 0, 1, 0.4j )

    orbs = orbs1 + orbs2
    orbs.set_hamiltonian_element( 0, 2, 0.7 )
    orbs.set_hamiltonian_element( 1, 3, 0.8 )
    assert orbs.hamiltonian[0, 2] == 0.7
    assert orbs.hamiltonian[1, 3] == 0.8

    
# def test_material_loading():
# graphene cutting 
graphene = {
    "orbitals": {
        "p_z": [
            (0, 0),  # Position of the first carbon atom in fractional coordinates
            (-1/3, -2/3)   # Position of the second carbon atom
        ]
    },
    "lattice_basis": [
        (1.0, 0.0, 0.0),  # First lattice vector
        (-0.5, 0.86602540378, 0.0)  # Second lattice vector (approx for sqrt(3)/2)
    ],
    "hopping": { "p_z-p_z" : [0.0, 2.66]  },
    "coulomb": { "p_z-p_z" : [0.0, 2.66]  },
    "lattice_constant" : 2.46,
}

graphene = Material(**graphene)
orbs = graphene.cut( 20*shapes.triangle, preview = True )

pos = (0.0, 0.0, 0.0)
orbs1 =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
orbs1.set_hamiltonian_element( 0, 1, 0.3j )
# assert orbs1.hamiltonian[0, 1] == 0.3j
# assert orbs1.hamiltonian[1, 0] == -0.3j

orbs += orbs1

print( orbs.hamiltonian )
