from granad import *

def test_shapes():
    graphene = MaterialCatalog.get("graphene")
    for shape in [Triangle(10), Hexagon(10), Rhomboid(10,10), Rectangle(10,10)]:
        if shape[:2] == '__':
            continue
        orbs = graphene.cut_flake(shape, plot = False)
        # not the best test ever
        assert len(orbs) > 0
