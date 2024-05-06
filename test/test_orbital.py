from granad import *

def test_group_id():
    orb_a =  Orbital()
    orb_b =  Orbital()
    assert orb_a.group_id != orb_b.group_id    
    graphene = MaterialCatalog.get("graphene")
    orbs = graphene.cut_flake(Triangle(10), plot = False)
    assert orbs[0].group_id != orb_a.group_id != orb_b.group_id


