from granad import *

def test_create():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [Orbital(pos)] )
    assert orbs._recompute == True

def test_update_on_access():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [Orbital( pos)] )
    assert orbs._recompute == True
    orbs.energies
    assert orbs._recompute == False

def test_set_elements():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [ Orbital( pos), Orbital( pos) ] )
    orbs.set_hamiltonian_element( 0, 1, 0.3j )
    assert orbs.hamiltonian[0, 1] == 0.3j
    assert orbs.hamiltonian[1, 0] == -0.3j
    orbs.set_coulomb_element( 0, 1, 0.3j )
    assert orbs.coulomb[0, 1] == 0.3j
    assert orbs.coulomb[1, 0] == -0.3j

def test_add():
    pos = (0.0, 0.0, 0.0)
    orbs_ab =  OrbitalList( [ Orbital( pos), Orbital( pos) ] )
    orbs_cd =  OrbitalList( [ Orbital(pos), Orbital(pos) ] )
    orbs = orbs_ab + orbs_cd
    assert len( orbs ) == 4

def test_append():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [ Orbital( pos), Orbital( pos) ] )
    single_orb =  Orbital(pos)
    orbs.append( single_orb )
    assert len( orbs ) == 3

def test_append_no_duplicate():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [ Orbital( pos), Orbital( pos) ] )
    try:
        orbs.append( orbs[0] )
    except:
        return
    raise ValueError

def test_delete():
    pos = (0.0, 0.0, 0.0)
    a, b = Orbital( pos), Orbital( pos)
    orbs =  OrbitalList( [ a, b ] )
    orbs.set_hamiltonian_element(0, 1, 1)    
    del orbs[ 0 ]
       
    assert (a, b) not in orbs._hopping_dict
    assert (a, b) not in orbs._coulomb_dict    
    assert orbs.hamiltonian.shape == (1,1)
    assert orbs.coulomb.shape == (1,1)
    
def test_material_coupling():
    def get_unique_nonzero( arr ):
        unique = jnp.unique( jnp.array(arr) )
        return unique[unique.nonzero()]
    
    graphene = Materials.get("graphene")
    orbs = graphene.cut_flake(Triangle(20, armchair = True), plot = False)    
    assert jnp.allclose( get_unique_nonzero( orbs.hamiltonian ), get_unique_nonzero( list(graphene.hopping.values())[0][0] ).flatten() )
    
def test_layer_orbital_coupling():    
    graphene = Material.get("graphene")
    orbs = graphene.cut_orbitals(9*Shapes.triangle_rotated, plot = False)
    orbs.append( Orbital( (0.0, 0.0, 1.1)) )
    orbs.append( Orbital( (0.0, 0.0, 1.1)) )    
    orbs.set_hamiltonian_element( -1, 0, 0.3j )
    orbs.set_coulomb_element( -1, 0, 0.3j )    
    func = lambda d : d
    orbs.set_groups_hopping( orbs[-1].group_id, orbs[0].group_id, func )
    orbs.set_groups_coulomb( orbs[-1].group_id, orbs[0].group_id, func )
    assert orbs.hamiltonian[-1, 0] == 0.3j
    assert orbs.coulomb[-1, 0] == 0.3j
    assert orbs.hamiltonian[-1,:].nonzero()[0].size == len(orbs) - 2
    assert orbs.coulomb[-1,:].nonzero()[0].size == len(orbs) - 2
    assert jnp.all(orbs.coulomb[-1,:] == orbs.hamiltonian[-1,:])
    assert jnp.count_nonzero(orbs.hamiltonian[-2,:]) == 0

    orbs = OrbitalList( [Orbital( (0.0, 0.0, 1.1)), Orbital( (0.0, 0.0, 1.1)) ] )
    graphene = Material.get("graphene")
    orbs += graphene.cut_orbitals(9*Shapes.triangle_rotated, plot = False)
    orbs.set_hamiltonian_element( -1, 0, 0.3j )
    orbs.set_coulomb_element( -1, 0, 0.3j )
    orbs.set_groups_hopping( orbs[-1].group_id, orbs[0].group_id, func )
    orbs.set_groups_coulomb( orbs[-1].group_id, orbs[0].group_id, func )
    assert orbs.hamiltonian[-1, 0] == 0.3j
    assert orbs.coulomb[-1, 0] == 0.3j
    assert orbs.hamiltonian[0,:].nonzero()[0].size == len(orbs) - 2
    assert orbs.coulomb[0,:].nonzero()[0].size == len(orbs) - 2
    assert jnp.all(orbs.coulomb[0,:] == orbs.hamiltonian[0,:])
    assert jnp.count_nonzero(orbs.hamiltonian[1,:]) == 0
