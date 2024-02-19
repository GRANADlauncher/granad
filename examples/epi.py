#===========First run same simulation as 'time_domain_simulation_loss.py' to produce the absorption spectra=============#
import granad
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

## custom function for performing the fourier transform
def get_fourier_transform(t_linspace, function_of_time):
    function_of_omega = jnp.fft.fft(function_of_time) / len(t_linspace)
    omega_axis = (
        2
        * jnp.pi
        * len(t_linspace)
        / jnp.max(t_linspace)
        * jnp.fft.fftfreq(function_of_omega.shape[-1])
    )
    return omega_axis, function_of_omega


# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Triangle(7.4),
    lattice_type=granad.LatticeType.HONEYCOMB,
    lattice_edge=granad.LatticeEdge.ARMCHAIR,
    lattice_constant=2.46,
)
sb.add("pz", graphene)

hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66]
)
sb.set_hopping(hopping_graphene)


coulomb_graphene = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb_graphene)

# create the stack object
#sb.show2D()
stack = sb.get_stack()
orginal_stack=stack #keep the orginal stack (rho_0) for future CW illumination
amplitudes = [1e-5, 0, 0]
frequency = 4
peak = 4
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm
)

# propagate in time
T=200
steps=2000001
time_axis = jnp.linspace(0,T,steps)

new_stack, occupations = granad.evolution(
    stack,
    time_axis,
    field_func,
    granad.relaxation(10),
    postprocess = jnp.diag
)

# calculate dipole moment
dipole_moment = granad.induced_dipole_moment(stack, occupations)

#  perform fourier transform
component = 0
omega_axis, dipole_omega = get_fourier_transform(time_axis, dipole_moment[:, component])

# we also need the x-component of the electric field as a single function
electric_field = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[component,:],peak, fwhm
)(time_axis)
_, field_omega = get_fourier_transform(time_axis, electric_field[component,:])

# plot result
omega_max = jnp.int32(8*T//(2*jnp.pi)) #we restrict upto 8 eV
omega_axis,dipole_omega,field_omega=omega_axis[:omega_max],dipole_omega[:omega_max],field_omega[:omega_max]
polarizability = dipole_omega / field_omega
spectrum = -omega_axis * jnp.imag(polarizability)

#======NOW get the omegas corresponding to absorption peaks===================#
def get_omegas_for_peaks(omegas,sigmas,limit=None,height=0,prominence=1):
    from scipy.signal import find_peaks
    
    if limit!=None:
        lower_limit,upper_limit=limit
        lower_idx=jnp.argmin(jnp.abs(o-lower_limit))
        upper_idx=jnp.argmin(jnp.abs(o-upper_limit))
        omegas_cut= omegas[lower_idx:upper_idx]
        sigmas_cut=sigmas[lower_idx:upper_idx]
    else:
        omegas_cut,sigmas_cut=omegas,sigmas
    peaks_idx,_= find_peaks(sigmas_cut,height=height,prominence=prominence)
    return omegas_cut[peaks_idx],peaks_idx,omegas_cut,sigmas_cut
omegas_for_peaks,*_=get_omegas_for_peaks(omega_axis,spectrum)

#==========Calculate EPI of the first peak in the spextrum ============================#
omega_for_epi=omegas_for_peaks[0]

#evolute upto 50 cycles of illumination or max time T=200 wich ever is maximum
ncycles=50
t_period=2.0*jnp.pi/omega_for_epi
T=max(ncycles*t_period,200)
ncycles=T/t_period
step=int(jnp.ceil(T*2000)) #devide unit time interval into 2000 intervals
time_axis=jnp.linspace(0,T,step)

#Now we make a CW illumination at the frequency omega_for_epi and keep the density matrix at the last time step of the evolution
CW_field_func = granad.electric_field(amplitudes = [1e-5, 0, 0],frequency=omega_for_epi, positions=stack.positions[0, :])

# Chunked Time Evolution. For details see the example 'chunked_sim1.py'
stack=orginal_stack
max_memory = 10**9
print(f"Max memory= {max_memory/1e9} GB")
keep_after_index = time_axis.size - int(max_memory / stack.rho_0.nbytes)
if keep_after_index<0:split_axis=[time_axis]
else:split_axis = jnp.split(time_axis, [keep_after_index])
for i, t in enumerate( split_axis ):
    if i == 0:
        if keep_after_index<0:postprocess=lambda x: x
        else:postprocess = lambda x : None
    else:postprocess = lambda x: x
    stack, some_last_rhos_SiteBasis = granad.evolution(stack,
                                          t,
                                          CW_field_func,
                                          granad.relaxation(10),
                                          postprocess = postprocess)

last_rho_EnergyBasis=granad.to_energy_basis(stack,some_last_rhos_SiteBasis[-1,:,:])
some_last_rhos_SiteBasis=None
EPI=float(granad.epi(stack, last_rho_EnergyBasis, omega_for_epi, epsilon=.05))
print(f'EPI corresponding to the peak Energy={round(omega_for_epi,2)} is {EPI}$')
