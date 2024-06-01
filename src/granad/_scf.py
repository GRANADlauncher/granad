import time
import jax
import jax.numpy as jnp

# reference
from pyqint import PyQInt, cgf
import numpy as np
from copy import deepcopy

from granad import *

# TODO: mean-field channels as flake couplings: coulomb, cooper, exchange
# TODO: high level scf into flake that sets params, positions from scf
# TODO: doc strings
# TODO: remove hard-coded ranges (determine max range at start?)
# TODO: are type casts ... le bad?
# TODO: exploit symmetry as follows:
#   1. introduce array `dont_compute_these_index_combinations` with index combinations not to compute.
#   2. orbitals = jnp.concatenate(orbitals, jnp.arange(orbitals.size))
#   3. in integration, check ( orbital1[-1] , orbital2[-1] , orbital3[-1] , orbital4[-1] ) in index array. yes => return 0, no => compute
#   4. result = vmap ... => build full matrix with function fill_missing_values(result)
# TODO: performance: normalization, factorial, binom, double_factorial are slow
# TODO: put norms into orbital array?
# TODO: DRY norm unpacking same in all mat elem procedures

def get_atomic_charges():
    return {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
        'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
    }

# TODO: tail-call for n > len(vals)
def double_factorial(n):
    vals = jnp.array([1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025])
    n_max = vals.size
    rng = jnp.arange(n_max)
    return (n > 0) * jnp.where(rng == n, vals[rng], 0).sum() + (n < 0)

def factorial(n):
    vals = jnp.array([1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,6227020800,87178291200,1307674368000])
    rng = jnp.arange(vals.size)
    return jnp.where(rng == n, vals[rng], 0).sum()

def binom(n, m):
    return (n > 0) * (m > 0) * (n > m) * (factorial(n) / (factorial(n - m) * factorial(m)) - 1) + 1

def gaussian_1d(o1, o2, x1, x2, gamma):
    """computes 1D gaussian integral"""
    
    def prefactor(i):
        s = 2*i
        cond = jnp.logical_and( rng < s + 1, jnp.logical_and(s-o1 <= rng, rng <= o2) )
        arr = jnp.where(cond, rng, -1)
        return jax.lax.map(
            lambda t : jax.lax.cond(
                t >= 0,
                lambda t : binom(o1, s-t)*binom(o2, t)*jnp.power(x1,(o1-s+t))*jnp.power(x2,(o2-t)),
                lambda t : 0.0,
                t),
            arr).sum()
    
    def term(i):
        return prefactor(i)*double_factorial(2*i-1)/jnp.power(2*gamma,i)

    rng = jnp.arange(20)
    arr = jnp.where(rng < 1 + jnp.floor(0.5*(o1+o2)), rng, -1)

    return jax.lax.map(
        lambda i : jax.lax.cond(
            i >= 0,
            lambda i : term(i),
            lambda  i : 0.0,
            i),
        arr).sum()

def gaussian_3d(lmn1, lmn2, a1, a2, r1, r2, r):
    """computes 3D gaussian integral"""
    
    gamma = a1+a2
    center = (a1*r1+a2*r2)/gamma
    pre = jnp.power((jnp.pi/gamma),1.5) * jnp.exp(-a1*a2*r/gamma)
    
    # gaussian integral as function of cartesian index
    f = lambda i : gaussian_1d(lmn1[i], lmn2[i],  center[i] - r1[i],  center[i] - r2[i], gamma)

    return pre * jax.vmap(f)(jnp.arange(3)).prod()

def gaussian_kinetic_term(lmn1, lmn2, a1, a2, r1, r2, r):
    """computes gaussian kinetic term for two GTOs"""

    term0  = a2 * (2.0*lmn2.sum()+3.0) * gaussian_3d(lmn1, lmn2, a1, a2, r1, r2, r)

    term1 = -2.0 * jnp.pow(a2, 2.0) * (gaussian_3d(lmn1, lmn2.at[0].set(lmn2[0]+2), a1, a2, r1, r2, r)
                                       + gaussian_3d(lmn1, lmn2.at[1].set(lmn2[1]+2), a1, a2, r1, r2, r)
                                       + gaussian_3d(lmn1, lmn2.at[2].set(lmn2[2]+2), a1, a2, r1, r2, r)
                                       )

    prefac = lmn2*(lmn2 - 1)
    term2 = -0.5 * (prefac[0]*gaussian_3d(lmn1, lmn2.at[0].set(lmn2[0]-2), a1, a2, r1, r2, r)
                    + prefac[1]*gaussian_3d(lmn1, lmn2.at[1].set(lmn2[1]-2), a1, a2, r1, r2, r)
                    + prefac[2]*gaussian_3d(lmn1, lmn2.at[2].set(lmn2[2]-2), a1, a2, r1, r2, r)
                    )
    
    return term0 + term1 + term2

def gaussian_norm(lmn, alphas):
    nom = jax.vmap(lambda a : jnp.pow(2.0, 2.0*(lmn.sum()) + 1.5) * jnp.pow(a, lmn.sum() + 1.5))(alphas)
    denom = (jax.vmap(lambda i : double_factorial(2*i-1))(lmn)).prod() * jnp.pow(jnp.pi, 1.5)
    return jnp.sqrt(nom / denom)    

def two_body_gaussian(n, orbital_i, orbital_j, orbital_k, orbital_l):
    r"""2 body matrix element of 1/|x-x'| in gaussian basis
    $U_{ijkl} = \int dx \int dx' \overline{\phi}_i(x) \overline{\phi}_j(x') 1/|x-x'| \phi_k(x') \phi_l(x)$"""
    print("Compiling gaussian two body")
    return

def nuclear_gaussian(n, orbitals, nuclei):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} 1/|x_n - x| \phi_j(x)$"""
    print("Compiling gaussian nuclear")
    return

def kinetic_gaussian(n, orbital_i, orbital_j):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} - \nabla \phi_j(x)$"""
    print("Compiling gaussian kinetic")

    # unpack array
    coefficients_i, alphas_i, lmn_i, pos_i = orbital_i[:n], orbital_i[n:2*n], orbital_i[-6:-3], orbital_i[-3:]
    coefficients_j, alphas_j, lmn_j, pos_j = orbital_j[:n], orbital_j[n:2*n], orbital_j[-6:-3], orbital_j[-3:]
    
    # normalization 
    norms_i = gaussian_norm(lmn_i, alphas_i)
    norms_j = gaussian_norm(lmn_j, alphas_j)
    
    # scalar distance is loop invariant
    r = jnp.linalg.norm(pos_i - pos_j)**2

    # vectorized integration as a matrix
    integral = jax.vmap(jax.vmap(lambda a, b : gaussian_kinetic_term(lmn_i, lmn_j, a, b, pos_i, pos_j, r), (0, None), 0), (None, 0), 0)

    # corresponding prefactor matrix
    prefac = norms_i[:, None] * norms_j * coefficients_i[:, None] * coefficients_j
    
    return (prefac * integral(alphas_i, alphas_j) ).sum()

def overlap_gaussian(n, orbital_i, orbital_j):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} \phi_j(x)$

    Args:
    n : slicing parameter for orbital array unpacking
    orbitals : orbital array, represents two orbitals

    Returns:
    float, overlap element
    """    
    print("Compiling gaussian overlap")

    # unpack array
    coefficients_i, alphas_i, lmn_i, pos_i = orbital_i[:n], orbital_i[n:2*n], orbital_i[-6:-3], orbital_i[-3:]
    coefficients_j, alphas_j, lmn_j, pos_j = orbital_j[:n], orbital_j[n:2*n], orbital_j[-6:-3], orbital_j[-3:]
    
    # normalization
    norms_i = gaussian_norm(lmn_i, alphas_i)
    norms_j = gaussian_norm(lmn_j, alphas_j)

    # scalar distance is loop invariant
    r = jnp.linalg.norm(pos_i - pos_j)**2

    # vectorized integration as a matrix
    integral = jax.vmap(jax.vmap(lambda a, b : gaussian_3d(lmn_i, lmn_j, a, b, pos_i, pos_j, r), (0, None), 0), (None, 0), 0)

    # corresponding prefactor matrix
    prefac = norms_i[:, None] * norms_j * coefficients_i[:, None] * coefficients_j
    
    return (prefac * integral(alphas_i, alphas_j) ).sum()

def expand_gaussian(orb_list, expansion):
    """converts OrbitalList into a representation compatible with gaussian integration.
    
    Args:
    orb_list: OrbtalList 
    expansion: dictionary mapping group ids to arrays
    
    Returns:
    """
    
    # translate orbital list to array representation
    orbitals = jnp.array( [jnp.concatenate([jnp.concatenate(expansion[o.group_id]), o.position]) for o in orb_list] )

    # array representation of nuclei, index map to keep track of orbitals => nuclei
    nuclei_positions, idxs, orbitals_to_nuclei = jnp.unique(orbitals[:,:3], axis = 0, return_index = True, return_inverse = True)
    atomic_charges = get_atomic_charges()
    nuclei_charges = jnp.array([float(atomic_charges[o.atom_name]) for o in orb_list])[idxs]
    nuclei = jnp.hstack( [nuclei_positions, nuclei_charges[:,None]] )

    size = expansion[orb_list[0].group_id][0].size

    # TODO: naja...
    overlap = jax.jit(lambda orb1, orb2: overlap_gaussian(size, orb1, orb2))
    overlap(orbitals[0], orbitals[0])
    
    kinetic = jax.jit(lambda orb1, orb2: kinetic_gaussian(size, orb1, orb2))
    kinetic(orbitals[0], orbitals[0])

    # kinetic = overlap = None
    
    return overlap, kinetic, orbitals, nuclei, orbitals_to_nuclei

def update_positions(orbitals, nuclei, orbitals_to_nuclei):
    """Ensures that orbital positions match their nuclei."""    
    return orbitals.at[:, :3].set(nuclei[orbitals_to_nuclei, :3])

def merge(h_e, c_e, idxs, oem, tem):
    """Combines empirical and ab-initio operator representations.

    Args:
    h_e : empirical hamiltonian
    c_e : empirical scf channels
    idxs : indices that should be overwritten
    oem : ab-initio one electron matrix elements
    tem : ab-initio two electron matrix elements
    """
    return NotImplemented

def scf(hamiltonian, cooper, coulomb, exchange):
    """low-level scf computation for a potential hybrid model of empirical and ab-initio parameters.
    
    Args:
    hamiltonian: NxN array containing empirical TB parameters

    cooper: SC coupling parameters, if None, don't consider this channel

    coulomb: direct channel coupling parameters, if None, don't consider this channel

    exchange: exchange coupling parameters, if None, don't consider this channel

    Returns:
    SCF Hamiltonian, Order Parameter
    """
    # cooper channel => build PH symmetric ham
    # else, start loop
    return NotImplemented

# convention used for built-in gaussian basis: tuples have the form (coefficients, alphas, lmn), where lmn is exponent of cartesian coordinates x,y,z
sto_3g = {
    "pz" : (jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
            jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
            jnp.array( [ 0,0,1 ]) )
    }


def get_reference(flake, expansion):
    """constructs PyQInt reference list of CGFs"""
    ang2bohr = 1 #1.8897259886
    cgfs = []
    for orb in flake:
        cs, alphas = expansion[orb.group_id][0], expansion[orb.group_id][1]
        cgf1 = cgf( (orb.position*ang2bohr).tolist()  )    
        cgf1.add_gto(cs[0], alphas[0], 0, 0, 1)
        cgf1.add_gto(cs[1], alphas[1], 0, 0, 1)
        cgf1.add_gto(cs[2], alphas[2], 0, 0, 1)    
        cgfs.append( cgf1 )
    return cgfs

def test_elements():
    # set up flake    
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(10))

    # modelling only pz orbs
    expansion = { flake[0].group_id : sto_3g["pz"] }

    # prepare data and functions for matrix element evaluation    
    overlap, kinetic, orbitals, nuclei, orbs_to_nuclei = expand_gaussian(flake, expansion)
    
    integrator = PyQInt()    

    i, j = 0, 3  
    cgfs = get_reference(flake, expansion)    
    cfg1, cfg2 = cgfs[i], cgfs[j]

    jax_s = overlap(orbitals[i], orbitals[j])
    pyqint_s = integrator.overlap(cfg1, cfg2)

    print("###OVERLAP###")
    print( "JAX", jax_s)
    print( "PyQint",  pyqint_s)    
    print( "delta", jnp.abs(jax_s - pyqint_s) )    

    jax_s = kinetic(orbitals[i], orbitals[j])
    pyqint_s = integrator.kinetic(cfg1, cfg2)
    
    print("###KINETIC###")    
    print( "JAX", jax_s)
    print( "PyQint",  pyqint_s)    
    print( "delta", jnp.abs(jax_s - pyqint_s) )    

def test_matrix():
    # set up flake    
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(10))

    # modelling only pz orbs
    expansion = { flake[0].group_id : sto_3g["pz"] }

    # prepare data and functions for matrix element evaluation    
    overlap, kinetic, orbitals, nuclei, orbs_to_nuclei = expand_gaussian(flake, expansion)
    
    cgfs = get_reference(flake, expansion)
    integrator = PyQInt()    

    get_s = jax.jit(jax.vmap(jax.vmap(lambda o1, o2 : overlap(o1, o2), (0, None), 0), (None, 0), 0))
    start = time.time()
    jax_s = get_s(orbitals, orbitals)
    print(time.time() - start)
    
    pyqint_s = np.zeros_like(jax_s)
    start = time.time()
    for i in range(len(flake)):
        for j in range(len(flake)):
            pyqint_s[i,j] = integrator.overlap(cgfs[i], cgfs[j])
    print(time.time()-start)

    print("###OVERLAP###")
    print( "delta", jnp.linalg.norm(jax_s - pyqint_s) )    


def test_cgfs():
    # set up flake    
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(10))

    # modelling only pz orbs
    expansion = { flake[0].group_id : sto_3g["pz"] }

    # prepare data and functions for matrix element evaluation    
    overlap, kinetic, orbitals, nuclei, orbs_to_nuclei = expand_gaussian(flake, expansion)
    
    integrator = PyQInt()    

    cgfs = get_reference(flake, expansion)
    
    i, j = 0, 1
    gto1 = cgfs[i].gtos[0]
    gto2 = cgfs[j].gtos[0]
    print(gto1, gto2)
    
    n = 3
    orbital_1, orbital_2 = orbitals[i], orbitals[j]    
    coefficients_1, alphas_1, lmn_1, r_1 = orbital_1[:n], orbital_1[n:2*n], orbital_1[-6:-3], orbital_1[-3:]
    coefficients_2, alphas_2, lmn_2, r_2 = orbital_2[:n], orbital_2[n:2*n], orbital_2[-6:-3], orbital_2[-3:]
    a_1, a_2 = alphas_1[i], alphas_2[i]    
    r = jnp.linalg.norm(r_1 - r_2)**2    

    vals1 = jnp.array([2,1,2])
    vals2 = jnp.array([1,1,2])
    
    gto1.l = vals1[0]
    gto1.m = vals1[1]
    gto1.n = vals1[2]
    gto1.p = r_1.tolist()

    gto2.l = vals2[0]
    gto2.m = vals2[1]
    gto2.n = vals2[2]
    gto2.p = r_2.tolist()

    print(gaussian_3d(lmn_1.at[:3].set(vals1), lmn_2.at[:3].set(vals2), a_1, a_2, r_1, r_2, r))
    print(integrator.overlap_gto(gto1, gto2))
    print(gaussian_kinetic_term(lmn_1.at[:3].set(vals1), lmn_2.at[:3].set(vals2), a_1, a_2, r_1, r_2, r))
    print(integrator.kinetic_gto(gto1, gto2))    
        
if __name__ == '__main__':
    # test_cgfs()
    # test_elements()
    test_matrix()
