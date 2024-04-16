from granad import *

def test_shapes():
    graphene = Material(**MaterialDatabase.graphene)
    for shape in dir(Shapes):
        if shape[:2] == '__':
            continue
        orbs = graphene.cut(10*getattr(Shapes,shape))
        # not the best test ever
        assert len(orbs) > 0
