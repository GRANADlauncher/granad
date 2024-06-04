import time
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma, gammainc

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

def get_atomic_charge(atom):
    charges = {
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
    return charges[atom]

# TODO: I suspect all these functions suck in terms of performance
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

def boys(n, x):
    r"""Computes $F_n(T) = \int_0^1 dx x^{2m} e^{-Tx^2}$"""
    return gamma(0.5 + n) * gammainc(0.5 + n, x) / (2*x**(0.5 + n))

def gaussian_norm(lmn, alphas):
    nom = jax.vmap(lambda a : jnp.pow(2.0, 2.0*(lmn.sum()) + 1.5) * jnp.pow(a, lmn.sum() + 1.5))(alphas)
    denom = (jax.vmap(lambda i : double_factorial(2*i-1))(lmn)).prod() * jnp.pow(jnp.pi, 1.5)
    return jnp.sqrt(nom / denom)    

def two_body_gaussian(n, orbital_i, orbital_j, orbital_k, orbital_l):
    r"""2 body matrix element of 1/|x-x'| in gaussian basis
    $U_{ijkl} = \int dx \int dx' \overline{\phi}_i(x) \overline{\phi}_j(x') 1/|x-x'| \phi_k(x') \phi_l(x)$"""
    print("Compiling gaussian two body")
    return

def _unpack_loop():
    return

def nuclear_gaussian(n, orbital_i, orbital_j, nucleus):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} 1/|x_n - x| \phi_j(x)$"""
    print("Compiling gaussian nuclear")
    return

def kinetic_gaussian(n, orbital_i, orbital_j):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} - \nabla \phi_j(x)$"""
    print("Compiling gaussian kinetic")
    return

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
    return

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

    nuclear = jax.jit(lambda orb1, orb2, nuc: nuclear_gaussian(size, orb1, orb2, nuc))
    nuclear(orbitals[0], orbitals[0], nuclei[0])
    
    # kinetic = overlap = nuclear = None
    
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

# TODO: prbly smarter to bound individual nestings
def transform_nested_loop(f, bound, *arrs):
    """transforms a dynamic bounds nested loop computation with base case function f
    for a in arrs[0]:
      for b in arrs[1]:
        ...
          val = f(val, a, b, ..., **params)
    into a JAX-JIT compatible chain of `scan` calls.

    where the dynamic bounds are decided by the bound function before calling into f.

    This function is partially based on mattj's ingenious solution https://github.com/google/jax/discussions/10401#discussioncomment-2610609
    """
    
    def bound_eval(loop, bound):
        """bypasses loop computation if bound evals to True"""

        return lambda val, *xs, **params : jax.lax.cond( bound(*xs, **params), lambda : loop(val, *xs, **params), lambda : val)
    
    # wrap loop into dynamic bound evaluation function before capturing in closure 
    loop = bound_eval(lambda val, *xs, **params : f(val, *xs, **params), bound)
    
    def add(loop, arr):
        """adds another nesting level to `loop` over the values contained in `arr`"""
        
        def wrapper(val, *xs, **params):
            """wraps the bounded loop computation on `arr` into a scan, discarding the array value accumulator"""
            
            print("compiled nesting") # poor mans JIT debug            
            val, _ = jax.lax.scan(lambda val, x : (loop(val, *xs, x, **params), None), val, arr)
            return val

        # start from base case by adding induction cases
        return wrapper

    for arr in reversed(arrs):
        loop = add(loop, arr)
    
    return loop

# convention used for built-in gaussian basis: tuples have the form (coefficients, alphas, lmn), where lmn is exponent of cartesian coordinates x,y,z
sto_3g = {
    "pz" : (jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
            jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
            jnp.array( [ 0,0,1 ]) )
    }



if __name__ == '__main__':
    1
    # test_cgfs()
    # test_elements()
    # test_matrix()

    # boys_g = lambda n, t : 0.5*jnp.pow(t,-n-0.5) * jax.lax.igamma(n+0.5,t)
    # boys_g =  lambda n, x : 

