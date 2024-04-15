from granad import *

def test_create():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [Orbital("A", pos)] )
    assert orbs._recompute_stack == True

def test_update_on_access():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [Orbital("A", pos)] )
    assert orbs._recompute_stack == True
    orbs.energies
    assert orbs._recompute_stack == False

def test_set_elements():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
    orbs.set_hamiltonian_element( 0, 1, 0.3j )
    assert orbs.hamiltonian[0, 1] == 0.3j
    assert orbs.hamiltonian[1, 0] == -0.3j
    orbs.set_coulomb_element( 0, 1, 0.3j )
    assert orbs.coulomb[0, 1] == 0.3j
    assert orbs.coulomb[1, 0] == -0.3j

def test_add():
    pos = (0.0, 0.0, 0.0)
    orbs_ab =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
    orbs_cd =  OrbitalList( [ Orbital("C", pos), Orbital("D", pos) ] )
    orbs = orbs_ab + orbs_cd
    assert len( orbs ) == 4

def test_append():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
    single_orb =  Orbital("C", pos)
    orbs.append( single_orb )
    assert len( orbs ) == 3

def test_append_no_duplicate():
    pos = (0.0, 0.0, 0.0)
    orbs =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
    try:
        orbs.append( orbs[0] )
    except:
        return
    raise ValueError

def test_delete():
    pos = (0.0, 0.0, 0.0)
    a, b = Orbital("A", pos), Orbital("B", pos)
    orbs =  OrbitalList( [ a, b ] )
    orbs.set_hamiltonian_element(0, 1, 1)    
    del orbs[ 0 ]
       
    assert (a, b) not in orbs._hopping
    assert (a, b) not in orbs._coulomb    
    assert orbs.hamiltonian.shape == (1,1)
    assert orbs.coulomb.shape == (1,1)
    
def test_material_coupling():
    def get_unique_nonzero( arr ):
        unique = jnp.unique( jnp.array(arr) )
        return unique[unique.nonzero()]
    
    graphene = Material(**MaterialDatabase.graphene)
    orbs = graphene.cut(10*Shapes.triangle, plot = False)
    assert jnp.allclose( get_unique_nonzero( orbs.hamiltonian ), get_unique_nonzero( list(graphene.hopping.values()) ).flatten() )
    assert jnp.allclose( get_unique_nonzero( orbs.coulomb ), get_unique_nonzero( list(graphene.coulomb.values()) ).flatten()  )
    
