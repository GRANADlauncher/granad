# TODO: 3D extension?
import jax
import jax.numpy as jnp
from itertools import combinations, product
from matplotlib.path import Path
import pprint

# TODO: hmmm
from ._plotting import _display_lattice_cut
from ._orbitals import OrbitalList, Orbital
from . import _watchdog

class Material:
    """positions must be fractional coordinates. 
    lattice constant must be angström.    
    """
    _materials = _materials
    
    @classmethod
    def get(cls, material):
        return cls( **cls._materials[material] )

    @staticmethod
    def materials():
        available_materials = '\n'.join( Material._materials.keys() )
        print f"Available materials:\n{available_materials}"
    
    def _prune_neighbors( self, positions, minimum_neighbor_number, remaining_old = jnp.inf ):
        if minimum_neighbor_number <= 0:
            return positions
        distances = jnp.round( jnp.linalg.norm( positions - positions[:, None], axis = -1), 4 )
        minimum = jnp.unique(distances)[1]
        mask = (distances <= minimum).sum(axis = 0)  > minimum_neighbor_number
        remaining = mask.sum()        
        if remaining_old == remaining:
            return positions[mask, :]        
        else:
            return self._prune_neighbors( positions[mask, :], minimum_neighbor_number, remaining )

    def _get_orbs_positions_dict( self, m, n, orbs = None ):
        if orbs is None:
            orbs = self.orbitals.keys()

        orbs_positions_dict = {}
        n = (-n-1,n+1) if n > 0 else (n-1,-n+1)
        m = (-m-1,m+1) if m > 0 else (m-1,-m+1)
        coefficients = jnp.array( list(product(range(*n), range(*m))) )
        lattice_basis = jnp.array(self.lattice_basis)
        for orb in orbs:
            orbital_shift = jnp.array(self.orbitals[orb]["position"]) @ lattice_basis
            pos = self.lattice_constant * (coefficients @ lattice_basis + orbital_shift)
            orbs_positions_dict[orb] = pos
        return orbs_positions_dict
    
    # very nasty, TODO: find better solution than rounding stuff, hardcoding 1e-5
    def _neighbor_couplings_to_function(self, couplings, qm_comb ):
        orbs = filter(lambda orb : self.orbitals[orb]["quantum_numbers"] in qm_comb, self.orbitals.keys() )
        orbs_positions_dict = self._get_orbs_positions_dict( len(couplings), len(couplings), orbs )
        positions = jnp.concatenate( list(orbs_positions_dict.values()) )
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
    
    def _set_couplings( self, setter_func, cdict, uuid ):
        quantum_numbers = (self.orbitals[x]["quantum_numbers"] for x in  self.orbitals.keys())
        for comb in combinations( quantum_numbers, 2 ):
            try:
                neighbor_couplings = cdict[comb]
            except KeyError:
                pass            
            distance_func = self._neighbor_couplings_to_function(neighbor_couplings, comb)
            setter_func( uuid, uuid, distance_func )                    

    
    def _create_orbital(self, orb, pos, uuid):
        return Orbital(name = orb, position = tuple(float(x) for x in pos), uuid = uuid, quantum_numbers = self.orbitals[orb]["quantum_numbers"] ) 
    
    # TODO: planes too big in some cases
    def cut_orbitals( self, polygon_vertices, plot = False, minimum_neighbor_number : int = 2 ):        
        # Unzip into separate lists of x and y coordinates
        x_coords, y_coords = zip(*polygon_vertices)
        max_x =jnp.abs(jnp.array(x_coords)).max()
        max_y =jnp.abs(jnp.array(y_coords)).max()
        basis = jnp.array(self.lattice_basis) * self.lattice_constant
        frac_x =  jnp.nan_to_num( max_x / basis[:,0], posinf = 1, neginf = 1)
        frac_y = jnp.nan_to_num( max_y / basis[:,1], posinf = 1, neginf = 1)
        m_max = frac_x[ jnp.abs(frac_x).argmax() ]
        m_max = jnp.floor(m_max)  if m_max < 0 else jnp.ceil(m_max) 
        n_max = frac_y[ jnp.abs(frac_y).argmax() ]
        n_max = jnp.floor(n_max) if n_max < 0 else jnp.ceil(n_max) 
        orbs_positions_dict = self._get_orbs_positions_dict( int(m_max), int(n_max) )        


        # get all positions within the polygon and remove isolated spots
        polygon = Path( polygon_vertices )
        all_positions = jnp.concatenate( list( orbs_positions_dict.values() ) )
        flags = polygon.contains_points( all_positions[:,:2] )
        selected_positions  = self._prune_neighbors( all_positions[flags], minimum_neighbor_number )
        
        # associate positions to orbitals
        orbs_selected_positions_dict = {}
        for orb, orb_positions in orbs_positions_dict.items():
            idxs = (jnp.round( jnp.linalg.norm( orb_positions - selected_positions[:, None], axis = -1), 4 ) == 0).nonzero()[0]
            orbs_selected_positions_dict[orb] = selected_positions[ idxs, : ]
            
        if plot:
            _display_lattice_cut( polygon_vertices, all_positions, selected_positions )

        # prepare the orbital list, making sure that all orbitals share the same uuid TODO: rename, uuid is something else tbh
        uuid = _watchdog._Watchdog.next_value()
        raw_list = [ self._create_orbital(orb, pos, uuid) for orb, positions in orbs_selected_positions_dict.items() for pos in positions ]        
        orbital_list = OrbitalList( raw_list )        
        self._set_couplings( orbital_list.set_layers_hopping, self.hopping, uuid )
        self._set_couplings( orbital_list.set_layers_coulomb, self.coulomb, uuid )
        
        return orbital_list


_materials =
"graphene" : {
    "orbitals": {
    (0,1,0,0,"C","sublattice_1") : (0,0), 
    (0,1,0,0,"C","sublattice_2") : (-1/3,-2/3) }
    "lattice_basis": [
        (1.0, 0.0, 0.0),  # First lattice vector
        (-0.5, 0.86602540378, 0.0)  # Second lattice vector (approx for sqrt(3)/2)
    ],        
    # couplings are defined by combinations of orbital quantum numbers
    "hopping": { ((0,1,0,0,"C"), (0,1,0,0,"C")) : ([0.0, 2.66],)  },
    "coulomb": {  ((0,1,0,0,"C"), (0,1,0,0,"C")) : ([16.522, 8.64, 5.333],lambda d: 1/d)  },
    "lattice_constant" : 2.46,
},
"ssh" : {},
"metallic_chain" : {},
}


