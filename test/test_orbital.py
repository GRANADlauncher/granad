from granad import *

def test_uuid():
    pos = (0.0, 0.0, 0.0)
    orb_a =  Orbital("A", pos)
    orb_b =  Orbital("B", pos)
    assert orb_a.uuid != orb_b.uuid    
    graphene = Material(**MaterialDatabase.graphene)
    orbs = graphene.cut(10*Shapes.triangle, plot = False)
    assert orbs[0].uuid != orb_a.uuid != orb_b.uuid


