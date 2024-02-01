import granad

sb = granad.StackBuilder()

spot = granad.Spot(position=[0.0, 0.0, 0.0])
sb.add("A", spot)
sb.add("B", spot, occupation=0)

# onsite hopping
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=0))
sb.set_hopping(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_hopping(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=2))

# onsite coulomb
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="A", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="A", orbital_id2="B", coupling=1))
sb.set_coulomb(granad.SpotCoupling(orbital_id1="B", orbital_id2="B", coupling=1))

stack = sb.get_stack()
granad.show_energies(stack)
