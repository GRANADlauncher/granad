from granad import *
from granad._numerics import *
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def test_rabi():
    occupations_old = jnp.array( [[0j, (1+0j)], [(0.0024638913840934317+0j), (0.9975361086159059+0j)], [(0.009439610097185275+0j), (0.9905603899028148+0j)], [(0.019866046496283395+0j), (0.9801339535037163+0j)], [(0.032555648338359174+0j), (0.9674443516616407+0j)], [(0.046826257653933605+0j), (0.953173742346067+0j)], [(0.06290673104581448+0j), (0.9370932689541853+0j)], [(0.08186489990478929-1.3877787807814457e-17j), (0.9181351000952105-1.3877787807814457e-17j)], [(0.1050663021590199-1.3877787807814457e-17j), (0.8949336978409798-1.3877787807814457e-17j)], [(0.13344369283596066+6.938893903907228e-18j), (0.866556307164036+6.938893903907228e-18j)], [(0.16695964633465715+8.673617379884035e-19j), (0.8330403536653427+8.673617379884035e-19j)], [(0.2045169577670455+0j), (0.7954830422329542+0j)], [(0.24431622391076907+0j), (0.7556837760892303+0j)], [(0.28444882873036503-1.3877787807814457e-17j), (0.7155511712696347-1.3877787807814457e-17j)], [(0.3234562787791079+2.7755575615628914e-17j), (0.676543721220892+2.7755575615628914e-17j)], [(0.36067088983629325+2.7755575615628914e-17j), (0.6393291101637064+2.7755575615628914e-17j)], [(0.3962864697375787+0j), (0.6037135302624219+0j)], [(0.4312040968760631+0j), (0.5687959031239369+0j)], [(0.4667277953918714+0j), (0.5332722046081283+0j)], [(0.5041728983836291+0j), (0.49582710161637134+0j)], [(0.544441461271429+0j), (0.4555585387285712+0j)], [(0.5876420500381162-1.3877787807814457e-17j), (0.4123579499618834-1.3877787807814457e-17j)], [(0.6328743285618935+0j), (0.36712567143810576+0j)], [(0.6783152768512879+0j), (0.3216847231487112+0j)], [(0.7216771602068128+0j), (0.27832283979318756+0j)], [(0.7609446339351671+0j), (0.23905536606483224+0j)], [(0.7951079416482287+0j), (0.20489205835177043+0j)], [(0.8245247140835956+2.7755575615628914e-17j), (0.1754752859164067+2.7755575615628914e-17j)], [(0.8506715818927751+0j), (0.14932841810722433+0j)], [(0.8753601649915238+1.3877787807814457e-17j), (0.12463983500847373+1.3877787807814457e-17j)], [(0.8998062809427204+0j), (0.10019371905727867+0j)], [(0.9240379075331356+0j), (0.0759620924668635+0j)], [(0.9469287737968378+0j), (0.05307122620316155+0j)], [(0.9667846956381883+0j), (0.03321530436181097+0j)], [(0.9821275819263993+3.469446951953614e-18j), (0.017872418073599943+3.469446951953614e-18j)], [(0.9922770214853304+0j), (0.007722978514668959+0j)], [(0.9975005274610449+3.469446951953614e-18j), (0.0024994725389542605+3.469446951953614e-18j)], [(0.9987453496278487+0j), (0.0012546503721505495+0j)], [(0.9971345868521586+0j), (0.002865413147841625+0j)], [(0.9934557801396431+0j), (0.006544219860355756+0j)], [(0.9878235867858742+0j), (0.01217641321412467+0j)], [(0.9796179829328355+6.938893903907228e-18j), (0.020382017067160307+6.938893903907228e-18j)], [(0.967719930638776+0j), (0.03228006936122142+0j)], [(0.9509868401582654+0j), (0.04901315984173415+0j)], [(0.9288223551621037-3.469446951953614e-18j), (0.07117764483789465-3.469446951953614e-18j)], [(0.9016154926321921+0j), (0.09838450736780796+0j)], [(0.8708089278408382+0j), (0.12919107215916084+0j)], [(0.8384665955873825+0j), (0.16153340441261677+0j)], [(0.8064481275322855-1.3877787807814457e-17j), (0.1935518724677099-1.3877787807814457e-17j)], [(0.7755554358361351+0j), (0.22444456416386382+0j)], [(0.7451162986277254+0j), (0.25488370137227334+0j)], [(0.7132910087234716+0j), (0.2867089912765277+0j)], [(0.677995805558201+0j), (0.3220041944417975+0j)], [(0.6379740658460638+1.3877787807814457e-17j), (0.3620259341539343+1.3877787807814457e-17j)], [(0.5934613516137499-3.469446951953614e-18j), (0.40653864838624876-3.469446951953614e-18j)], [(0.5461384044321757+0j), (0.4538615955678233+0j)], [(0.49846675435873355+0j), (0.5015332456412653+0j)], [(0.4527911960655344+0j), (0.5472088039344645+0j)], [(0.4106247772156902+0j), (0.5893752227843084+0j)], [(0.3723518071933689+2.7755575615628914e-17j), (0.6276481928066299+2.7755575615628914e-17j)], [(0.33735054287881217+0j), (0.6626494571211867+0j)], [(0.3043854791050724+0j), (0.6956145208949257+0j)], [(0.2720882865203828+0j), (0.7279117134796155+0j)], [(0.23939347834074623+0j), (0.7606065216592524+0j)], [(0.20585428470549888+0j), (0.7941457152945005+0j)], [(0.1717964437666442+1.3877787807814457e-17j), (0.8282035562333536+1.3877787807814457e-17j)], [(0.13827594083116018-6.938893903907228e-18j), (0.8617240591688387-6.938893903907228e-18j)], [(0.1068225187259226+0j), (0.8931774812740759+0j)], [(0.07900440952547913+1.734723475976807e-18j), (0.9209955904745193+1.734723475976807e-18j)], [(0.05593796381040583+0j), (0.9440620361895926+0j)], [(0.037938729988095665+0j), (0.9620612700119028+0j)], [(0.024496417138187346+0j), (0.9755035828618106+0j)], [(0.014621553309201866+0j), (0.9853784466907971+0j)], [(0.007410796171401575-3.469446951953614e-18j), (0.9925892038285973-3.469446951953614e-18j)], [(0.0025375582748977505+3.469446951953614e-18j), (0.9974624417250997+3.469446951953614e-18j)], [(0.00040330853094411485+0j), (0.9995966914690544+0j)], [(0.0018781475863282292+0j), (0.9981218524136706+0j)], [(0.007791924602207685+0j), (0.9922080753977907+0j)], [(0.018459050238517405+3.469446951953614e-18j), (0.9815409497614811+3.469446951953614e-18j)], [(0.03346973826123054+0j), (0.9665302617387681+0j)], [(0.051817047600691335+6.938893903907228e-18j), (0.9481829523993073+6.938893903907228e-18j)], [(0.07226806912677017+0j), (0.9277319308732248+0j)], [(0.09380932887586835+1.3877787807814457e-17j), (0.9061906711241299+1.3877787807814457e-17j)], [(0.11600680511216627+0j), (0.8839931948878323+0j)], [(0.13917592346027824-1.3877787807814457e-17j), (0.8608240765397193-1.3877787807814457e-17j)], [(0.1643129198055882+0j), (0.8356870801944096+0j)], [(0.19278465587565305+0j), (0.8072153441243454+0j)], [(0.22582546553619465-1.3877787807814457e-17j), (0.7741745344638039-1.3877787807814457e-17j)], [(0.2639645710282149+1.3877787807814457e-17j), (0.7360354289717836+1.3877787807814457e-17j)], [(0.3065901486007151-6.938893903907228e-18j), (0.6934098513992836-6.938893903907228e-18j)], [(0.3518883985805217+0j), (0.6481116014194772+0j)], [(0.39730590199717597+0j), (0.6026940980028226+0j)], [(0.4404519140688915+0j), (0.5595480859311077+0j)], [(0.48007768361742004+0j), (0.5199223163825788+0j)], [(0.5166276606313535+2.7755575615628914e-17j), (0.4833723393686453+2.7755575615628914e-17j)], [(0.5520058758658833+0j), (0.4479941241341154+0j)], [(0.5886094951283771+0j), (0.4113905048716229+0j)], [(0.6281088589895645+1.3877787807814457e-17j), (0.3718911410104343+1.3877787807814457e-17j)], [(0.67060727049109-6.938893903907228e-18j), (0.3293927295089084-6.938893903907228e-18j)], [(0.7145835448650841+0j), (0.2854164551349142+0j)]] )

    lower_level = Orbital( (0,0,0) )
    upper_level = Orbital( (0,0,0) )
    adatom = OrbitalList( [lower_level, upper_level] )
    adatom.set_hamiltonian_element( lower_level, lower_level, 0 )
    adatom.set_hamiltonian_element( upper_level, upper_level, 2 )
    adatom.set_hamiltonian_element( upper_level, lower_level, 1 )
    adatom.set_coulomb_element( lower_level, lower_level, 1 )
    adatom.set_coulomb_element( upper_level, upper_level, 1 )
    adatom.set_coulomb_element( upper_level, lower_level, 1 )
    adatom.electrons = 1
    adatom.set_dipole_transition( upper_level, lower_level, [1/2,0,0] )
    adatom.set_excitation( adatom.homo, adatom.homo+1, 1)

    # this and the other code are phase shifted, this is why the amplitude is complex and likely also the reason for the second decimal deviations 
    wave = Wave(amplitudes = [1j, 0, 0], frequency = max(adatom.energies) - min(adatom.energies) )

    # compare to old version
    # diffrax
    time_axis, density_matrices = adatom.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 1e4, illumination = wave, use_rwa = True, use_old_method = False )
    occupations = jnp.diagonal( adatom.transform_to_energy_basis(density_matrices.ys), axis1=-1, axis2=-2)

    plt.plot(time_axis[::100], occupations_old)
    plt.plot(time_axis[::100], occupations[::100], '--')
    plt.show()    

    
    np.testing.assert_allclose( occupations[::100], occupations_old, atol = 1e-1 )

    # hand made RK
    time_axis, density_matrices = adatom.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 1e4, illumination = wave, use_rwa = True, use_old_method = True )
    occupations = jnp.diagonal( adatom.transform_to_energy_basis(density_matrices), axis1=-1, axis2=-2)
    np.testing.assert_allclose( occupations[::100], occupations_old, atol = 1e-1 )
    
    # non-RWA should agree for fine enough grid
    # diffrax
    time_axis, density_matrices = adatom.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 1e4, illumination = wave, use_rwa = False, use_old_method = False )
    occupations = jnp.diagonal( adatom.transform_to_energy_basis(density_matrices.ys), axis1=-1, axis2=-2)

    # hand made RK
    time_axis_rk, density_matrices = adatom.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 1e6, illumination = wave, use_rwa = False, use_old_method = True )
    occupations_rk = jnp.diagonal( adatom.transform_to_energy_basis(density_matrices), axis1=-1, axis2=-2)
    plt.plot(time_axis_rk, occupations_rk)
    plt.plot(time_axis, occupations, '--')
    plt.show()    
    np.testing.assert_allclose( occupations, occupations_rk[::100], atol = 1e-2 )

# TODO: doesnt work
def test_vector_potential():
    lower_level = Orbital( (0,0,0) )
    upper_level = Orbital( (0,0,0) )
    adatom = OrbitalList( [lower_level, upper_level] )
    adatom.set_hamiltonian_element( lower_level, lower_level, 0 )
    adatom.set_hamiltonian_element( upper_level, upper_level, 2 )
    adatom.set_hamiltonian_element( upper_level, lower_level, 1 )
    adatom.set_coulomb_element( lower_level, lower_level, 1 )
    adatom.set_coulomb_element( upper_level, upper_level, 1 )
    adatom.set_coulomb_element( upper_level, lower_level, 1 )
    adatom.electrons = 1
    adatom.set_dipole_transition( upper_level, lower_level, [1/2,0,0] )
    adatom.set_excitation( adatom.homo, adatom.homo+1, 1)

    omega = max(adatom.energies) - min(adatom.energies)
    omega = 1
    electric_field = lambda t : jnp.array( [ -jnp.sin( omega * t ), 0, 0 ] )
    vector_potential = lambda t : jnp.array( [ [ jnp.cos( omega * t ), 0, 0 ], [ jnp.cos( omega * t ), 0, 0 ] ]  )

    # check difference between vector potential and electric field approach
    # electric field
    time_axis, density_matrices = adatom.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 1e4, illumination = electric_field, use_rwa = False, use_old_method = False )
    occupations_efield = jnp.diagonal( adatom.transform_to_energy_basis(density_matrices.ys), axis1=-1, axis2=-2)

    # full vector potential term
    time_axis, density_matrices = adatom.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 1e4, illumination = vector_potential, use_rwa = False, use_old_method = False )
    occupations_vp = jnp.diagonal( adatom.transform_to_energy_basis(density_matrices.ys), axis1=-1, axis2=-2)

    plt.plot( time_axis, occupations_efield )
    plt.plot( time_axis, occupations_vp, '--' )
    plt.show()

# TODO: test with all possible arguments
def test_rhs():
    lower_level = Orbital( (0,0,0) )
    upper_level = Orbital( (0,0,0) )
    adatom = OrbitalList( [lower_level, upper_level] )
    adatom.set_hamiltonian_element( lower_level, lower_level, 0 )
    adatom.set_hamiltonian_element( upper_level, upper_level, 2 )
    adatom.set_hamiltonian_element( upper_level, lower_level, 1 )
    adatom.set_coulomb_element( lower_level, lower_level, 1 )
    adatom.set_coulomb_element( upper_level, upper_level, 1 )
    adatom.set_coulomb_element( upper_level, lower_level, 1 )
    adatom.electrons = 1
    adatom.set_dipole_transition( upper_level, lower_level, [1,0,0] )
    adatom.set_excitation( adatom.homo, adatom.homo+1, 1)
    illumination = Wave(amplitudes = [-1j, 0, 0], frequency = max(adatom.energies) - min(adatom.energies) )

    time_axis = jnp.linspace(0,10, int(1e4))

    include_induced_contribution = False
    relaxation_function = lambda r : 0.0
    coulomb_field_to_from = get_coulomb_field_to_from( adatom.positions, adatom.positions, None )        

    initial_density_matrix = adatom.transform_to_site_basis(adatom._initial_density_matrix)
    stationary_density_matrix = adatom.transform_to_site_basis(adatom._stationary_density_matrix)


    rhs = get_rhs_master_equation( adatom.hamiltonian, adatom.coulomb,
                adatom.dipole_operator, adatom.electrons, adatom.velocity_operator,
                stationary_density_matrix,
                time_axis, illumination, relaxation_function,
                coulomb_field_to_from, include_induced_contribution)

    rhs(0, initial_density_matrix, 0) 

def test_dissipation():    
    return

def test_dissipation_saturated():
    return

graphene = Material2D.get("graphene" )
flake = graphene.cut_orbitals( Triangle(15, armchair = True) + jnp.array([10, 10]), plot = False )
# flake.show_2d()
pulse = Pulse( amplitudes = [1e-5j, 0, 0], frequency = 2.3, peak = 5, fwhm = 2 )

time, dipole_moments = flake.get_expectation_value_time_domain(
    operator = flake.dipole_operator,
    relaxation_rate = 1/10,
    end_time = 40,
    steps_time = 1e5,
    illumination = pulse,
    skip = 100,
    use_old_method = False,
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10)
)
plt.plot(time, dipole_moments)
plt.savefig('res.pdf')
plt.close()

omegas, pol_omega, pulse_omega = flake.get_expectation_value_frequency_domain(
    operator = flake.dipole_operator,
    end_time = 40,
    steps_time = 1e5,
    illumination = pulse,
    skip = 100,
    relaxation_rate = 1/10,
    omega_min = 0,
    omega_max = 16,
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
)
# def get_fourier_transform(t_linspace, function_of_time):
#     function_of_omega = np.fft.fft(function_of_time) / len(t_linspace)
#     omega_axis = (
#         2
#         * np.pi
#         * len(t_linspace)
#         / np.max(t_linspace)
#         * np.fft.fftfreq(function_of_omega.shape[-1])
#     )
#     return omega_axis, function_of_omega
# omega_axis_new, dipole_omega_new = get_fourier_transform(
#     time, dipole_moments[:, 0] )
# electric_field = jax.vmap(pulse)(time)
# _, field_omega_new = get_fourier_transform(time, electric_field[:,0])
# omega_max = 100
# component = 0
# # alpha = p / E
# polarizability = dipole_omega_new / field_omega_new
# spectrum = -omega_axis_new[:omega_max] * np.imag(polarizability[:omega_max])
# plt.plot(omega_axis_new[:omega_max], np.abs(spectrum) ** (1 / 2), "--", label="new")
# plt.legend()
# plt.xlabel(r"$\hbar\omega$", fontsize=20)
# plt.ylabel(r"$\sigma(\omega)$", fontsize=25)
# plt.savefig('abs_new.pdf')




spectrum = jnp.abs(electric_field); plt.plot(time_axis, jnp.abs(spectrum).T ); plt.savefig('pulse.pdf')

# _, field_omega_old = get_fourier_transform(time_axis, electric_field[0])
spectrum = jnp.abs(jax.vmap(pulse)(time))
plt.plot(time, jnp.abs(spectrum) )
plt.savefig('pulse.pdf')
plt.close()


spectrum = -(pol_omega / pulse_omega).imag * omegas[:,None]
plt.plot(omegas, jnp.abs(spectrum)**0.5 )
plt.savefig('abs.pdf')
plt.close()

spectrum = jnp.abs(pol_omega)
plt.plot(omegas, jnp.abs(spectrum) )
plt.savefig('dip.pdf')
plt.close()

spectrum = jnp.abs(pulse_omega)
plt.plot(omegas, jnp.abs(spectrum) )
plt.savefig('pulse.pdf')
# omegas, pol = flake.get_polarizability_time_domain( end_time = 10, steps_time = 1e5, relaxation_rate = 1/10, illumination = pulse )

# propagate in time
# gamma = 1
# time_axis = jnp.linspace(0, 1 / gamma, int(1e5))

# allow transfer from higher energies to lower energies only if the
# two energy levels are not degenerate
# diff = orbs.energies[:, None] - orbs.energies
# gamma_matrix = gamma * jnp.logical_and(
#     diff < 0,
#     jnp.abs(diff) > orbs.eps,
# )

# wave = Wave(amplitudes = [0, 0, 0], frequency = 2  )
# time_axis, density_matrices = orbs.get_density_matrix_time_domain( start_time = 0, end_time = 1/gamma, steps_time = 1e5, illumination = wave, use_old_method = True )
# occ = jnp.diagonal( orbs.transform_to_energy_basis( density_matrices ), axis1 = -1, axis2=-2)
# # jnp.diagonal( adatom.transform_to_energy_basis(density_matrices.ys), axis1=-1, axis2=-2)

# plt.plot( time_axis, occ*orbs.electrons )
# plt.show()
    

def test_rpa():
    return

def test_operators():
    return
