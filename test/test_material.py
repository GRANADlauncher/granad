from granad import *

def test_from_dict():
    # graphene cutting 
    graphene = {
        "orbitals": {
            "p_z": [
                {"pos" : (0, 0), "sublattice" : 0} ,  # Position of the first carbon atom in fractional coordinates
                 {"pos" : (-1/3, -2/3), "sublattice" : 1}   # Position of the second carbon atom
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

def test_from_json():
    import json
    # graphene cutting 
    graphene = {
        "orbitals": {
            "p_z": [
                {"pos" : (0, 0), "sublattice" : 0} ,  # Position of the first carbon atom in fractional coordinates
                 {"pos" : (-1/3, -2/3), "sublattice" : 1}   # Position of the second carbon atom
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
    with open('graphene.json', 'w') as f:
        json.dump(graphene, f, indent=4)  
    graphene = Material.from_json('graphene.json')    
test_from_json()
