# Imports
from granad import MaterialCatalog,Triangle
import jax.numpy as jnp
from granad import Wave
import diffrax
from granad import Orbital, OrbitalList

#Flake
Flake=MaterialCatalog.get("graphene").cut_flake(Triangle(10,armchair=True),plot=True,minimum_neighbor_number = 1)

#Adatom
lower_5d=Orbital()
upper_6s=Orbital()
Au=OrbitalList([lower_5d,upper_6s])
Au.set_hamiltonian_element(lower_5d,lower_5d,-1.32)
Au.set_hamiltonian_element(upper_6s,upper_6s,-.29)
Au.set_dipole_element(lower_5d,upper_6s,[0,0,0])
Au.set_electrons(3)

#Flake+Adatom
Au_Flake=Flake+Au
coupling_pz=Au_Flake[1]
top_position=coupling_pz.position+jnp.array([0,0,2.56])
Au_Flake.set_position(top_position,Au)
# Au-Flake Couplings
Au_Flake.set_hamiltonian_element(coupling_pz, Au, -2.66*.2)
Au_Flake.show_3d()

# Time Domain Simulations
wave=Wave(amplitudes=[.17,0,0],frequency=.969)#.969)
result=Au_Flake.master_equation(
    dt=1e-5,
    end_time=430,
    grid=100,
    illumination=wave,
    use_rwa=False,
    coulomb_strength=1,
    density_matrix=['occ_e'],
    relaxation_rate=None,
    max_mem_gb=15,
    solver = diffrax.Dopri8()
   

)
states_to_plot=range(Au_Flake.homo-3,1+Au_Flake.homo+3)
labels=[str(i) for i in states_to_plot]
Au_Flake.show_res(result,plot_only=[i for i in states_to_plot],plot_labels=labels)