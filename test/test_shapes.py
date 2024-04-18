from granad import *

def test_shapes():
    graphene = Material.get("graphene")
    for shape in dir(Shapes):
        if shape[:2] == '__':
            continue
        orbs = graphene.cut_orbitals(10*getattr(Shapes,shape), plot = True)
        # not the best test ever
        assert len(orbs) > 0
