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
    
    try:
        orbs._hopping[ (a, b) ]
        raise ValueError
    except KeyError:
        pass
    
    assert orbs.hamiltonian.shape == (1,1)
    assert orbs.coulomb.shape == (1,1)

    
def test_material_list():
    
    graphene = Material(**graphene)
    orbs = graphene.cut( 10*Shapes.triangle )    
    assert orbs.energies.size > 0
    
# def test_material_loading():
#     # graphene cutting 
#     graphene = {
#         "orbitals": {
#             "p_z": [
#                 (0, 0),  # Position of the first carbon atom in fractional coordinates
#                 (-1/3, -2/3)   # Position of the second carbon atom
#             ]
#         },
#         "lattice_basis": [
#             (1.0, 0.0, 0.0),  # First lattice vector
#             (-0.5, 0.86602540378, 0.0)  # Second lattice vector (approx for sqrt(3)/2)
#         ],
#         "hopping": { "p_z-p_z" : [0.0, 2.66]  },
#         "coulomb": { "p_z-p_z" : [0.0, 2.66]  },
#         "lattice_constant" : 2.46,
#     }

#     graphene = Material(**graphene)
#     orbs = graphene.cut( 10*Shapes.triangle, preview = False )

#     pos = (0.0, 0.0, 0.0)
#     orbs1 =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
#     orbs1.set_hamiltonian_element( 0, 1, 0.3j )
#     # assert orbs1.hamiltonian[0, 1] == 0.3j
#     # assert orbs1.hamiltonian[1, 0] == -0.3j

#     orbs += orbs1
#     orbs.set_hamiltonian_element( 0, 1, 0.3j )
#     print( orbs.hamiltonian )

# # graphene cutting 
# graphene = {
#     "orbitals": {
#         "p_z": [
#             (0, 0),  # Position of the first carbon atom in fractional coordinates
#             (-1/3, -2/3)   # Position of the second carbon atom
#         ]
#     },
#     "lattice_basis": [
#         (1.0, 0.0, 0.0),  # First lattice vector
#         (-0.5, 0.86602540378, 0.0)  # Second lattice vector (approx for sqrt(3)/2)
#     ],
#     "hopping": { "p_z-p_z" : [0.0, 2.66]  },
#     "coulomb": { "p_z-p_z" : [0.0, 2.66]  },
#     "lattice_constant" : 2.46,
# }

# graphene = Material(**graphene)
# orbs = graphene.cut( 10*Shapes.triangle, preview = False )

# pos = (0.0, 0.0, 0.0)
# orbs1 =  OrbitalList( [ Orbital("A", pos), Orbital("B", pos) ] )
# orbs1.set_hamiltonian_element( 0, 1, 0.3j )
# # assert orbs1.hamiltonian[0, 1] == 0.3j
# # assert orbs1.hamiltonian[1, 0] == -0.3j

# orbs += orbs1
# orbs.set_hamiltonian_element( 0, 1, 0.3j )
# print( orbs.hamiltonian )

# orbs.append(  Orbital("B", pos) )
# orbs.append(  orbs[0] )
