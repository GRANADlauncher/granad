# TODO: 3D extension?

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from itertools import combinations, product
from matplotlib.path import Path
import json

# TODO: hmmm
from ._plotting import _show_lattice_cut
from ._orbitals import OrbitalList, Orbital
from . import _watchdog

@dataclass(frozen = True)
class Material:
    orbitals : dict    
    lattice_basis : list = None
    lattice_constant : float = None    
    hopping : dict = None
    coulomb : dict = None    

    def _prune_neighbors( self, positions, all_positions, minimum_neighbor_number, remaining_old = jnp.inf ):
        if minimum_neighbor_number <= 0:
            return positions
        distances = jnp.round( jnp.linalg.norm( positions - all_positions[:, None], axis = -1), 6 )
        minimum = jnp.unique(distances)[1]
        mask = (distances <= minimum).sum(axis = 0)  > minimum_neighbor_number
        remaining = mask.sum()
        if remaining_old == remaining:
            return positions[mask, :]        
        else:
            return self._prune_neighbors( positions[mask, :], all_positions, minimum_neighbor_number, remaining )

    def _get_translations( self, repetitions ):
        basis_size = jnp.array(self.lattice_basis).shape[0]
        return list(product( *(range(-repetitions, repetitions+1) for x in range(basis_size) ) ))

    def _get_orbs_positions_dict( self, coefficients, orbs = None ):
        if orbs is None:
            orbs = self.orbitals.keys()

        orbs_positions_dict = {}
        coefficients = jnp.array( coefficients )
        lattice_basis = jnp.array(self.lattice_basis)
        for orb in orbs:
            orbital_shift = jnp.array(self.orbitals[orb]["position"]) @ lattice_basis
            pos = self.lattice_constant * (coefficients @ lattice_basis + orbital_shift)
            orbs_positions_dict[orb] = pos
        return orbs_positions_dict
    
    # very nasty, TODO: find better solution than rounding stuff, hardcoding 1e-5
    def _neighbor_couplings_to_function(self, couplings, qm_comb ):
        orbs = filter(lambda orb : self.orbitals[orb]["quantum_numbers"] in qm_comb, self.orbitals.keys() )
        orbs_positions_dict = self._get_orbs_positions_dict( self._get_translations( len(couplings) ), orbs )
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
        return Orbital(orbital_name = orb, position = tuple(float(x) for x in pos), uuid = uuid, quantum_numbers = self.orbitals[orb]["quantum_numbers"] ) 
    
    # TODO: planes too big in some cases
    def cut( self, polygon_vertices, plot = False, minimum_neighbor_number : int = 2 ):        
        # Unzip into separate lists of x and y coordinates
        x_coords, y_coords = zip(*polygon_vertices)
        
        # get all positions in a plane covering the selection
        max_x =jnp.abs(jnp.array(x_coords)).max()
        max_y =jnp.abs(jnp.array(y_coords)).max()
        lattice_max_x = self.lattice_constant *jnp.array( self.lattice_basis).max( axis = 0)[0]        
        lattice_max_y = self.lattice_constant *jnp.array( self.lattice_basis ).max( axis = 0)[1]
        repetitions = int(jnp.nan_to_num(jnp.ceil(max( max_x / lattice_max_x, max_y / lattice_max_y )), posinf = 0) )
        orbs_positions_dict = self._get_orbs_positions_dict( self._get_translations( repetitions ) )        

        # get all positions within the polygon
        polygon = Path( polygon_vertices )

        # TODO: uff, i have clearly not thought of the data well enough
        all_positions = jnp.concatenate( list( orbs_positions_dict.values() ) )
        flags = polygon.contains_points( all_positions[:,:2] )            
        all_positions_in_polygon = all_positions[flags]

        # get the positions within the polygon 
        orbs_selected_positions_dict = {}
        for orb, orb_positions in orbs_positions_dict.items():
            flags = polygon.contains_points( orb_positions[:,:2] )            
            positions_in_polygon = orb_positions[flags]
            orbs_selected_positions_dict[orb] = self._prune_neighbors( positions_in_polygon, all_positions_in_polygon, minimum_neighbor_number )
            
        # do some optional plotting
        selected_positions = jnp.concatenate( list( orbs_selected_positions_dict.values() ) )
        if plot:
            _show_lattice_cut( polygon_vertices, all_positions, selected_positions )

        # prepare the orbital list, making sure that all orbitals share the same uuid TODO: rename, uuid is something else tbh
        uuid = _watchdog._Watchdog.next_value()
        raw_list = [ self._create_orbital(orb, pos, uuid) for orb, positions in orbs_selected_positions_dict.items() for pos in positions ]        
        orbital_list = OrbitalList( raw_list )        
        self._set_couplings( orbital_list.set_layers_hopping, self.hopping, uuid )
        self._set_couplings( orbital_list.set_layers_coulomb, self.coulomb, uuid )
        
        return orbital_list


# TODO: make this get all attributes from a collection of JSON files
class MaterialDatabase():
    """positions must be fractional coordinates. lattice constant must be anström.    
    """
    graphene = {
    "orbitals": {
        "pz0" : { "position" : (0,0), "quantum_numbers"  : (0,1,0,0) },  
        "pz1" : { "position" : (-1/3,-2/3), "quantum_numbers" : (0,1,0,0) }, 
        },
    "lattice_basis": [
        (1.0, 0.0, 0.0),  # First lattice vector
        (-0.5, 0.86602540378, 0.0)  # Second lattice vector (approx for sqrt(3)/2)
    ],        
    # couplings are defined by combinations of orbital quantum numbers
    "hopping": { ((0,1,0,0), (0,1,0,0)) : [0.0, 2.66]  },
    "coulomb": {  ((0,1,0,0), (0,1,0,0)) : [16.522, 8.64, 5.333]  },
    "lattice_constant" : 2.46,
}
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls(**data)
