import time
import jax
import jax.numpy as jnp

# reference
from pyqint import PyQInt, cgf
import numpy as np
from copy import deepcopy

from granad import *

# TODO: relax : jax.jit(jax.grad(scf(nuc, unpack(nuc, orb_spec), ...), arg_names = ['nuc']))), where unpack keeps the orbitals at their places
# TODO: mean-field channels as flake couplings: coulomb, cooper, exchange
# TODO: high level scf into flake that sets params, positions from scf
# TODO: doc strings
# TODO: remove hard-coded ranges
# TODO: are type casts ... le bad?
# TODO: for fixed basis, all involved arrays need to be same shape to avoid recompilation!

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
    
def double_factorial(n):
    rng = jnp.arange(1,10)
    arr = jnp.where(rng < n - 1, rng, 1)
    return arr[::2].prod()

def factorial(n):
    rng = jnp.arange(1,10)
    arr = jnp.where(rng < n - 1, rng, 1)
    return arr.prod()

def binom(n, m):
    return factorial(n) / (factorial(n - m) * factorial(m))

def gaussian_1d(o1, o2, x1, x2, gamma):
    """computes 1D gaussian integral"""
    
    def prefactor(i):
        s = 2*i
        arr = jnp.where( jnp.logical_and(s-o1 <= rng, rng <= o2), rng, -1)
        return jax.lax.map(
            lambda t : jax.lax.cond(
                t > 0,
                lambda t : binom(o1, s-t)*binom(o2, t)*jnp.power(x1,(o1-s+t))*jnp.power(x2,(o2-t)),
                lambda t : 0.0,
                t),
            arr).sum()
    
    def term(i):
        return prefactor(i)*double_factorial(2*i-1)/jnp.power(2*gamma,i)
    
    rng = jnp.arange(10)
    arr = jnp.where(rng < 1 + jnp.floor(0.5*(o1+o2)), rng, -1)

    return jax.lax.map(
        lambda i : jax.lax.cond(
            i > 0,
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
    f = lambda i : gaussian_1d(lmn1[i], lmn2[i], r1[i], r2[i], gamma)

    return pre * jax.vmap(f)(jnp.arange(3)).sum()

def gaussian_norm(lmn, alphas):
    nom = jax.vmap(lambda a : jnp.pow(2.0, 2.0*(lmn.sum()) + 1.5) * jnp.pow(a, lmn.sum() + 1.5))(alphas)
    denom = (jax.vmap(lambda i : double_factorial(2*i-1))(lmn)).prod() * jnp.pow(jnp.pi, 1.5)
    return jnp.sqrt(nom / denom)

def eri_gaussian(n, orbitals):
    r"""2 body matrix element of 1/|x-x'| in gaussian basis
    $U_{ijkl} = \int dx \int dx' \overline{\phi}_i(x) \overline{\phi}_j(x') 1/|x-x'| \phi_k(x') \phi_l(x)$"""
    return

def nuclear_gaussian(n, orbitals, nuclei):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} 1/|x_n - x| \phi_j(x)$"""
    return

def kinetic_gaussian(n, orbitals, nuclei):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} - \nabla \phi_j(x)$"""
    return

def overlap_gaussian(n, orbitals, nuclei):
    r"""1 body matrix element in gaussian basis
    $U_{ij} = \int dx \overline{\phi_i(x)} \sum_{x_n} \phi_j(x)$

    Args:
    n : slicing parameter for orbital array unpacking
    orbitals : orbital array, represents two orbitals
    nuclei : nuclei array, ignored

    Returns:
    float, overlap element
    """

    # unpack array
    pos_i, alphas_i, coefficients_i = orbitals[:3]
    pos_j, alphas_j, coefficients_j = orbitals[:3]    
    
    # normalization (could also be stored in arr tuple)
    norms_i = gaussian_norm(lmns_i, alphas_i)
    norms_j = gaussian_norm(lmns_j, alphas_j)

    # scalar distance is loop invariant
    r = jnp.linalg.norm(pos_i - pos_j)**2

    # vectorized integration as a matrix
    integral = jax.vmap(jax.vmap(lambda a, b : gaussian_3d(lmn_i, lmn_j, a, b, pos_i, pos_j, r), (0, None), 0), (None, 0), 0)

    # corresponding prefactor matrix
    prefac = norms_i[:, None] * norms_j * coefficients_i[:, None] * coefficients_j
    
    return (prefac * integral(a1, a2) ).sum()

def expand_gaussian(orb_list, expansion):
    """converts OrbitalList into a representation compatible with gaussian integration.
    
    Args:
    orb_list: OrbtalList 
    expansion: dictionary mapping group ids to arrays
    
    Returns:
    integrators, orbitals, nuclei, orbs_to_nuc
    """
    
    # orbital positions
    orbital_positions = jnp.array(
        [jnp.concatenate([o.position, jnp.concatenate()]) for o in orb_list]
    )

    # nuclei stuff
    nuclei_positions, orbitals_to_nuclei = jnp.unique(orbital_positions, axis = 0, return_index = True)
    atomic_charges = get_atomic_charges()
    nuclei_charges = jnp.array([float(atomic_charges[o.atom_name]) for o in orb_list])[orbitals_to_nuclei]    

    # TODO: uff
    orbital_specs = []
    for i, o in enumerate(orbs):
        orbital_specs.append(tuple(specs))

    return orbitals, nuclei, orbs_to_nuc

def get_masks(orbitals):
    """Returns two mask tuples `pack_mask`, `unpack_mask`. 

    The first (second) mask in each tuple reshapes during one (two) electron matrix computation.
       
    The masks in `pack_mask` reshape an orbital array. 
    In the resulting array, each row is a combination of two (four) orbitals.

    The masks `unpack_mask` reshape a 1D array of one (two) electron matrix elements into a 2 (4) - dim tensor.

    The following symmetries are taken into account:

    OEI: symmetric matrix
    TEI: symmetry U_{abcd} = U_{dcba}
    """    
    return NotImplemented

def get_matrix_elements(funcs):
    """Returns a closure that computes the list of functions.
    """
    def compute(args):        
        return [f(args) for f in funcs]    
    return compute

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

# TODO: this is only useful in structure relaxation and enforcing this is not very elegant, but linear so idc
def update_positions(orbitals, nuclei, idxs):
    """Ensures that orbital positions match their nuclei."""    
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


if __name__ == '__main__':
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(10))

    # modelling only pz orbs
    expansion = { flake[0].group_id : sto_3g["pz"] }

    # integrator params
    nuclei_positions, nuclei_charges, spec = expand(flake, expansion)
    orbitals = unpack(nuclei_positions, spec)

        
    # ### PYQINT FOR REFERENCE ###
    # ang2bohr = 1 #1.8897259886
    # cgfs = []
    # for orb in flake:
    #     cs, alphas = expansion_dict[orb.group_id]["coefficients"], expansion_dict[orb.group_id]["alphas"]
    #     cgf1 = cgf( (orb.position*ang2bohr).tolist()  )    
    #     cgf1.add_gto(cs[0], alphas[0], 0, 0, 1)
    #     cgf1.add_gto(cs[1], alphas[1], 0, 0, 1)
    #     cgf1.add_gto(cs[2], alphas[2], 0, 0, 1)    
    #     cgfs.append( cgf1 )
    # integrator = PyQInt()

    # ### OVERLAP COMPARISON ###
    # for i in range(len(flake)):
    #     for j in range(i+1):
    #         res = overlap_gaussian(i, j, expansion)
    #         print("JAX:", i, j, res)
    #         res = integrator.overlap(cgfs[i], cgfs[j])
    #         print("Pyqint:", i, j, res)
    #         break
