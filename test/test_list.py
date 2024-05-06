from granad import *

def test_create():
    orbs =  OrbitalList( [Orbital()] )

def test_update_on_access():
    orbs =  OrbitalList( [Orbital()] )
    assert orbs._recompute == True
    orbs.energies
    assert orbs._recompute == False

def test_set_elements():
    orbs =  OrbitalList( [ Orbital(), Orbital() ] )
    orbs.set_hamiltonian_element( 0, 1, 0.3j )
    assert orbs.hamiltonian[0, 1] == 0.3j
    assert orbs.hamiltonian[1, 0] == -0.3j
    orbs.set_coulomb_element( 0, 1, 0.3j )
    assert orbs.coulomb[0, 1] == 0.3j
    assert orbs.coulomb[1, 0] == -0.3j
    orbs.set_hamiltonian_element( 0, 0, -0.1 )
    orbs.set_hamiltonian_element( 1, 1, 0.1 )
    assert orbs.hamiltonian[0, 0] == -0.1
    assert orbs.hamiltonian[1, 1] == 0.1

def test_add():
    orbs_ab =  OrbitalList( [ Orbital(), Orbital() ] )
    orbs_cd =  OrbitalList( [ Orbital(), Orbital() ] )
    orbs = orbs_ab + orbs_cd
    assert len( orbs ) == 4

def test_append():
    orbs =  OrbitalList( [ Orbital(), Orbital() ] )
    single_orb =  Orbital()
    orbs.append( single_orb )
    assert len( orbs ) == 3

def test_append_no_duplicate():
    orbs =  OrbitalList( [ Orbital(), Orbital() ] )
    try:
        orbs.append( orbs[0] )
    except:
        return
    raise ValueError

def test_delete():
    a, b = Orbital(), Orbital()
    orbs =  OrbitalList( [ a, b ] )
    orbs.set_hamiltonian_element(0, 1, 1)    
    del orbs[ 0 ]
       
    assert (a, b) not in orbs.couplings.hamiltonian
    assert (a, b) not in orbs.couplings.coulomb
    assert orbs.hamiltonian.shape == (1,1)
    assert orbs.coulomb.shape == (1,1)
    
def test_material_coupling():
    def get_unique_nonzero( arr ):
        unique = jnp.unique( jnp.array(arr) )
        return unique[unique.nonzero()]
    
    graphene = MaterialCatalog.get("graphene")
    hopping = [ v for vals in graphene.interactions['hopping'].values() for v in vals[0] ]
    orbs = graphene.cut_flake(Triangle(20, armchair = True), plot = False)    
    assert jnp.allclose(
        get_unique_nonzero( orbs.hamiltonian ),
        get_unique_nonzero( hopping ) )
    
def test_layer_orbital_coupling():    
    graphene = MaterialCatalog.get("graphene")
    orbs = graphene.cut_flake(Triangle(20, armchair = True), plot = False)

    # we append two orbs
    orbs.append( Orbital( (0.0, 0.0, 1.1)) )
    orbs.append( Orbital( (0.0, 0.0, 1.1)) )

    # we set the coupling between the last orbital and the first pz orbital
    orbs.set_hamiltonian_element( -1, 0, 0.3j )
    orbs.set_coulomb_element( -1, 0, 0.3j )

    # for all other pz orbitals, we apply a function
    func = lambda d : d
    orbs.set_hamiltonian_groups( orbs[-1].group_id, orbs[0].group_id, func )
    orbs.set_coulomb_groups( orbs[-1].group_id, orbs[0].group_id, func )

    # check whether elements > groups
    assert jnp.abs(orbs.hamiltonian[-1, 0].imag) == 0.3
    assert jnp.abs(orbs.coulomb[-1, 0].imag) == 0.3

    # check whether coupling functon has been applied to the pz orbitals
    assert orbs.hamiltonian[-1,:].nonzero()[0].size == len(orbs) - 2
    assert orbs.coulomb[-1,:].nonzero()[0].size == len(orbs) - 2
    assert jnp.all(orbs.coulomb[-1,:] == orbs.hamiltonian[-1,:])
    assert jnp.count_nonzero(orbs.hamiltonian[-2,:]) == 0

    # same thing for two lists
    orbs = OrbitalList( [Orbital( (0.0, 0.0, 1.1)), Orbital( (0.0, 0.0, 1.1)) ] )
    graphene = MaterialCatalog.get("graphene")
    orbs += graphene.cut_flake(Triangle(20, armchair = True), plot = False)
    orbs.set_hamiltonian_element( -1, 0, 0.3j )
    orbs.set_coulomb_element( -1, 0, 0.3j )
    orbs.set_hamiltonian_groups( orbs[-1].group_id, orbs[0].group_id, func )
    orbs.set_coulomb_groups( orbs[-1].group_id, orbs[0].group_id, func )
    assert jnp.abs(orbs.hamiltonian[-1, 0].imag) == 0.3
    assert jnp.abs(orbs.coulomb[-1, 0].imag) == 0.3
    assert orbs.hamiltonian[0,:].nonzero()[0].size == len(orbs) - 2
    assert orbs.coulomb[0,:].nonzero()[0].size == len(orbs) - 2
    assert jnp.all(orbs.coulomb[0,:] == orbs.hamiltonian[0,:])
    assert jnp.count_nonzero(orbs.hamiltonian[1,:]) == 0
