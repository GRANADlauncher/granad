import time
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

from granad import *

from pyqint import PyQInt, cgf, gto
import numpy as np

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
# TODO: performance: normalization, factorial, binomial, double_factorial are slow
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

### ELEMENTARY FUNCTIONS ###
# TODO: I suspect all these functions suck in terms of performance, also: loop for n > len(vals)
def gamma_fun(s, x):
    return gammainc(s+0.5, x) * gamma(s+0.5) * 0.5 * jnp.pow(x,-s-0.5) 
    
def get_binomial_prefactor(l_max_range):    
    def body(val, t, s, ia, ib, xpa, xpb):
        return jax.lax.cond(jnp.logical_and(jnp.logical_and(t <= s, s-ia<=t), t <= ib),                            
                            lambda : val + binomial(ia, s-t) *
                            binomial(ib, t) *
                            xpa ** (ia-s+t)  *
                            xpb ** (ib-t),
                            lambda : val)

    def wrapper(*params):
        return jax.lax.scan(lambda val, t : (body(val, t, *params), None), 0.0, l_max_range)[0]

    return wrapper

def double_factorial(n):
    vals = jnp.array([1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025])
    n_max = vals.size
    rng = jnp.arange(n_max)
    return (n > 0) * jnp.where(rng == n, vals[rng], 0).sum() + (n < 0)

# def factorial(n):
#     vals = jnp.array([1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600,6227020800,87178291200,1307674368000])
#     return jax.lax.cond(n < vals.size,
#                         lambda : vals[n],
#                         lambda : 1 )

def binomial(n, m):
    return (n > 0) * (m > 0) * (n > m) * (factorial(n) / (factorial(n - m) * factorial(m)) - 1) + 1

def gaussian_norm(lmn, alphas):
    nom = jax.vmap(lambda a : jnp.pow(2.0, 2.0*(lmn.sum()) + 1.5) * jnp.pow(a, lmn.sum() + 1.5))(alphas)
    denom = (jax.vmap(lambda i : double_factorial(2*i-1))(lmn)).prod() * jnp.pow(jnp.pi, 1.5)
    return jnp.sqrt(nom / denom)


### OVERLAP ###
def get_overlap_1d(l_max_range):
    def body(val, i, l1, l2, x1, x2, gamma):
        return jax.lax.cond(i <  1 + jnp.floor(0.5*(l1+l2)),                            
                            lambda : val + binomial_prefactor(2*i, l1, l2, x1, x2) * double_factorial(2*i-1) / jnp.pow(2*gamma, i),
                            lambda : val)

    def wrapper(*params):
        return jax.lax.scan(lambda val, i : (body(val, i, *params), None), 0.0, l_max_range)[0]

    binomial_prefactor = get_binomial_prefactor(l_max_range)
    
    return wrapper

def get_overlap(l_max):
    def overlap(alpha1, lmn1, pos1, alpha2, lmn2, pos2):
        rab2 = jnp.linalg.norm(pos1-pos2)**2
        gamma = alpha1 + alpha2
        p = (alpha1*pos1 + alpha2*pos2) / gamma
        pre = jnp.pow(jnp.pi/gamma, 1.5) * jnp.exp(-alpha1*alpha2*rab2/gamma)

        vpa =  p - pos1
        vpb =  p - pos2

        wx = overlap_1d(lmn1[0], lmn2[0], vpa[0], vpb[0], gamma)
        wy = overlap_1d(lmn1[1], lmn2[1], vpa[1], vpb[1], gamma)
        wz = overlap_1d(lmn1[2], lmn2[2], vpa[2], vpb[2], gamma)
        
        return pre*wx*wy*wz

    l_max_range = jnp.arange(l_max)
    overlap_1d  = get_overlap_1d(l_max_range)
    
    return overlap


### KINETIC ###
def get_kinetic(l_max):
    
    def kinetic(alpha1, lmn1, pos1, alpha2, lmn2, pos2):
        term = alpha2 * (2.0 * lmn2.sum() + 3.0) * overlap(alpha1, lmn1, pos1, alpha2, lmn2, pos2)

        lmn2_inc = lmn2 + 2
        term += -2.0 * jnp.pow(alpha2, 2) * (overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[0].set(lmn2_inc[0]), pos2) +
                                               overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[1].set(lmn2_inc[1]), pos2) +
                                               overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[2].set(lmn2_inc[2]), pos2) )

        lmn2_dec = lmn2 - 2
        term += -0.5 * ( (lmn2[0] * (lmn2[0] - 1)) * overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[0].set(lmn2_dec[0]), pos2) +
                 lmn2[1]*(lmn2[1]-1) * overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[1].set(lmn2_dec[1]), pos2) +
                 lmn2[2]*(lmn2[2]-1) * overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[2].set(lmn2_dec[2]), pos2) )
        
        return term

    overlap = get_overlap(l_max)
    
    return kinetic


### NUCLEAR ###
def get_a_array(l_max):

    def a_term(i, r, u, l1, l2, pa, pb, cp, g):
        return ( jnp.pow(-1,i) * binomial_prefactor(i,l1,l2,pa,pb)*
                 jnp.pow(-1,u) * factorial(i)*jnp.pow(cp,i-2*r-2*u)*
                 jnp.pow(0.25/g,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u) )
    
    def a_legal(iI, i, r, u, l1, l2):
        return jnp.logical_and( jnp.logical_and( jnp.logical_and(i < l1 + l2 + 1, r <= jnp.ceil(i/2)),  u <= jnp.ceil((i-2*r)/2)),  iI == i - 2 * r - u )

    def a_wrapped(iI, i, r, u, l1, l2, pa, pb, cp, g):
        return jax.lax.cond(a_legal(iI, i, r, u, l1, l2), lambda: a_term(i, r, u, l1, l2, pa, pb, cp, g), lambda : 0.0) 

    def a_loop(iI, l1, l2, pa, pb, cp, g):
        return jax.lax.fori_loop(0, imax,
                lambda i, vx: vx + jax.lax.fori_loop(0, rmax,
                    lambda r, vy: vy + jax.lax.fori_loop(0, umax,
                        lambda u, vz: vz + a_wrapped(iI, i, r, u, l1, l2, pa, pb, cp, g),
                            0),
                            0),
                            0)
    
    # TODO: this sucks, bc it introduces an additional loop
    def a_array(l1, l2, pa, pb, cp, g):
        return jax.vmap(lambda iI : a_loop(iI, l1, l2, pa, pb, cp, g))(i_max_range)    

    imax = 2*l_max + 1
    rmax = jnp.floor(imax/2).astype(int)  + 1
    umax = rmax

    i_max_range = jnp.arange(imax)
    binomial_prefactor = get_binomial_prefactor(i_max_range)
    return a_array

def get_nuclear(l_max):

    def loop_body(i, j, k, lmn, rg):
        return jax.lax.cond( jnp.logical_and(jnp.logical_and(i <= lmn[0], j <= lmn[1]), k <= lmn[2]), lambda: gamma_fun(i+j+k , rg), lambda: 0.0)
        

    def loop(ax, ay, az, lmn, rg):
        return jax.lax.fori_loop(0, lim,
                    lambda i, vx: vx + jax.lax.fori_loop(0, lim,
                        lambda j, vy: vy + jax.lax.fori_loop(0, lim,
                            lambda k, vz: vz + ax[i] * ay[j] * az[k] * loop_body(i, j, k, lmn, rg),
                                0),
                                0),
                                0)
        
        
    def nuclear(alpha1, lmn1, pos1, alpha2, lmn2, pos2, nuc):
        gamma = alpha1 + alpha2
        p = (alpha1*pos1 + alpha2*pos2) / gamma
        rab2 = jnp.linalg.norm(pos1-pos2)**2
        rcp2 = jnp.linalg.norm(nuc-p)**2
        
        # TODO: this looks vectorizable
        vpa = p - pos1
        vpb = p - pos2
        vpn = p - nuc
        ax = a_array(lmn1[0], lmn2[0], vpa[0], vpb[0], vpn[0], gamma)
        ay = a_array(lmn1[1], lmn2[1], vpa[1], vpb[1], vpn[1], gamma)
        az = a_array(lmn1[2], lmn2[2], vpa[2], vpb[2], vpn[2], gamma)
        res = loop(ax, ay, az, lmn1+lmn2, rcp2*gamma)
        return -2.0 * jnp.pi / gamma * jnp.exp(-alpha1*alpha2*rab2/gamma) * res

    lim = 2*l_max+1
    a_array = get_a_array(l_max)
    
    return nuclear


### REPULSION ###
# TODO: i almost vomit every time i have to read this
def get_b_array(l_max):

    def bb0(i, r, g):
        return factorial(i) / factorial(r) / factorial(i - 2*r) * jnp.pow(4*g,r-i)

    def fb(i, l1, l2, p-a, p-b):
        return binomial_prefactor(i, l1, l2, p-a, p-b) * bb0(i, r, g)
    
    def b_term(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta):
        a, b = i1+i2-2*(r1+r2),u
        return (fB(i1,l1,l2,px,ax,bx,r1,g1)*
                jnp.pow(-1,i2) * fB(i2,l3,l4,qx,cx,dx,r2,g2)*
                jnp.pow(-1,u)* factorial(a) / factorial(b) / factorial(a - 2*b)*
                jnp.pow(qx-px,i1+i2-2*(r1+r2)-2*u)/
                jnp.pow(delta,i1+i2-2*(r1+r2)-u))
    
    def b_legal(i, i1, i2, r1, r2, u, l1, l2, l3, l4):
        return jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_and(i1 < l1 + l2 + 1, i2 < l3 + l4 + 1),
                        r1 < jnp.ceil(i1/2)),
                    r2 < jnp.ceil(i2/2)),
                u<jnp.floor((i1+i2)/2)-r1-r2+1),
            i == i1+i2-2*(r1+r2)-u)

    def b_wrapped(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta):
        return jax.lax.cond(b_legal(i, i1, i2, r1, r2, u, l1, l2, l3, l4), lambda: b_term(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta), lambda : 0.0) 

    def b_loop(i, l1, l2, pa, pb, cp, g):
            return jax.lax.fori_loop(0, inner_max,
                    lambda i1, acc_i1: acc_i1 + jax.lax.fori_loop(0, inner_max,
                        lambda i2, acc_i2: acc_i2 + jax.lax.fori_loop(0, inner_max,
                            lambda r1, acc_r1: acc_r1 + jax.lax.fori_loop(0, inner_max,
                                lambda r2, acc_r2: acc_r2 + jax.lax.fori_loop(0, inner_max,
                                    lambda u, acc_u: acc_u + b_wrapped(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta),
                                        0),
                                        0),
                                        0),
                                        0),
                                     0)
    
    # TODO: this sucks, bc it introduces an additional loop
    def b_array(l1, l2, pa, pb, cp, g):
        return jax.vmap(lambda iI : b_loop(iI, l1, l2, pa, pb, cp, g))(i_max_range)    

    imax = 4*l_max + 1
    inner_max = 2*l_max + 1

    i_max_range = jnp.arange(imax)
    binomial_prefactor = get_binomial_prefactor(i_max_range)
    return b_array

def get_repulsion(l_max):

    def loop_body(i, j, k, lmn, rg):
        return jax.lax.cond( jnp.logical_and(jnp.logical_and(i <= lmn[0], j <= lmn[1]), k <= lmn[2]), lambda: gamma_fun(i+j+k , rg), lambda: 0.0)
        

    def loop(bx, by, bz, lmn, rg):
        return jax.lax.fori_loop(0, lim,
                lambda i, vx: vx + jax.lax.fori_loop(0, lim,
                    lambda j, vy: vy + jax.lax.fori_loop(0, lim,
                        lambda k, vz: vz + bx[i] * by[j] * bz[k] * loop_body(i, j, k, lmn, rg),
                        0),
                        0),
                        0)

    def repulsion(alpha1, lmn1, pos1, alpha2, lmn2, pos2, alpha3, lmn3, pos3, alpha4, lmn4, pos4):
        
        rab2 = jnp.linalg.norm(pos1-pos2)**2
        rcd2 = jnp.linalg.norm(pos3-pos4)**2
        
        gamma12 = alpha1 + alpha2
        p = (alpha1*pos1 + alpha2*pos2) / gamma12
        
        gamma34 = alpha3 + alpha4        
        q = (alpha3*pos3 + alpha4*pos4) / gamma34

        rpq2 = jnp.linalg.norm(p-q)**2
        
        delta = 0.25 * (1.0 / gamma12 + 1.0 / gamma34)

        bx=b_array(lmn1[0], lmn2[0], lmn3[0], lmn4[0], p[0], pos1[0], pos2[0], q[0], pos3[0], pos4[0], gamma12, gamma34, delta)
        by=b_array(lmn1[1], lmn2[1], lmn3[1], lmn4[1], p[1], pos1[1], pos2[1], q[1], pos3[1], pos4[1], gamma12, gamma34, delta)
        bz=b_array(lmn1[2], lmn2[2], lmn3[2], lmn4[2], p[2], pos1[2], pos2[2], q[2], pos3[2], pos4[2], gamma12, gamma34, delta)

        res = loop(bx, by, bz, lmn1+lmn2+lmn3+lmn4, 0.25*rpq2/delta)
        return 2.0 * jnp.pow(jnp.pi, 2.5) / (gamma12*gamma34*jnp.sqrt(gamma12+gamma34)) * jnp.exp(-alpha1*alpha2*rab2/gamma12) * jnp.exp(-alpha3*alpha4*rcd2/gamma34) * ret
        

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

### UNWRAP ###
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

    # max l for static loops compatible with JIT+reverse AD
    l_max = orbitals[3:6].max()
    
    return *get_gaussian_functions(l_max), orbitals, nuclei, orbitals_to_nuclei

# TODO: decocrate gto functions with norm factor loop and compile directive
def one_body_wrapper():
    return NotImplemented

def two_body_wrapper():
    return NotImplemented

def get_gaussian_functions(l_max):
    overlap = get_overlap(l_max)
    kinetic = get_kinetic(l_max)
    nuclear = get_nuclear(l_max)
    repulsion = get_repulsion(l_max)

    return overlap, kinetic, nuclear, repulsion

### PYQINT REFERENCE NAMESPACE: USE EXPOSED FUNCS FROM LIB OR PURE PYTHON IMPLS OF PYQINT C++ FUNCS ###
class Reference:

    @staticmethod 
    def get_reference(flake, expansion):
        def get_cgf( orb ):
            ret = cgf(orb.position.tolist())
            exp = expansion[orb.group_id]

            for i in range(len(exp[0])):
                ret.add_gto( exp[0][i], exp[1][i], *(exp[2].tolist()) )

            return ret

        return list(map(get_cgf, flake))

    @staticmethod
    def binomial_prefactor(s, ia, ib, xpa, xpb):
        from scipy.special import binom
        sum = 0.0
        for t in range(s + 1):
            if (s - ia <= t) and (t <= ib):
                sum += binomial(ia, s - t) * binomial(ib, t) * (xpa ** (ia - s + t)) * (xpb ** (ib - t))
        return sum

    @staticmethod
    def nuclear(a, l1, m1, n1, alpha1, b, l2, m2, n2, alpha2, c):
        import scipy.special
        
        Fgamma = lambda s, x : scipy.special.gammainc(s+0.5, x) * 0.5 * jnp.pow(x,-s-0.5) * scipy.special.gamma(s+0.5)

        gamma = alpha1 + alpha2

        p = (alpha1 * a + alpha2 * b) / gamma
        rab2 = jnp.linalg.norm(a - b)**2
        rcp2 = jnp.linalg.norm(c - p)**2

        ax = Reference.a_array(l1, l2, p[0] - a[0], p[0] - b[0], p[0] - c[0], gamma)
        ay = Reference.a_array(m1, m2, p[1] - a[1], p[1] - b[1], p[1] - c[1], gamma)
        az = Reference.a_array(n1, n2, p[2] - a[2], p[2] - b[2], p[2] - c[2], gamma)

        sum = 0.0

        for i in range(l1 + l2 + 1):
            for j in range(m1 + m2 + 1):
                for k in range(n1 + n2 + 1):
                    sum += ax[i] * ay[j] * az[k] * Fgamma(i + j + k, rcp2 * gamma)

        return -2.0 * jnp.pi / gamma * jnp.exp(-alpha1 * alpha2 * rab2 / gamma) * sum

    @staticmethod
    def a_array(l1, l2, pa, pb, cp, g):
        imax = l1 + l2 + 1
        arrA = [0] * imax

        for i in range(imax):
            for r in range(int(i/2)+1):
                for u in range( int((i-2*r)/2)+1):
                    iI = i - 2*r - u
                    arrA[iI] += Reference.A_term(i, r, u, l1, l2, pa, pb, cp, g) # some pure function call

        return arrA
    
    @staticmethod
    def A_term(i, r, u, l1, l2, pax, pbx, cpx, gamma):
        import math
        return (math.pow(-1, i) * Reference.binomial_prefactor(i, l1, l2, pax, pbx) *
                math.pow(-1, u) * factorial(i) * math.pow(cpx, i - 2 * r - 2 * u) *
                math.pow(0.25 / gamma, r + u) / factorial(r) / factorial(u) / factorial(i - 2 * r - 2 * u))


# TODO: test with minimum possible l_max
### QUICK TESTS ###
def test_binom():
    assert False
    
def test_gaussian_norm():
    assert False

def test_double_factorial():
    assert False

def test_factorial():
    # The first 20 factorial values
    expected_values = [
        1,               # 0!
        1,               # 1!
        2,               # 2!
        6,               # 3!
        24,              # 4!
        120,             # 5!
        720,             # 6!
        5040,            # 7!
        40320,           # 8!
        362880,          # 9!
        3628800,         # 10!
        39916800,        # 11!
        479001600,       # 12!
        6227020800,      # 13!
        87178291200,     # 14!
        1307674368000,   # 15!
        20922789888000,  # 16!
        355687428096000, # 17!
        6402373705728000,# 18!
        121645100408832000,# 19!
        2432902008176640000 # 20!
    ]
    
    for n in range(21):
        assert factorial(n) == expected_values[n], f"Test failed for n={n}: {factorial(n)} != {expected_values[n]}"

def test_gamma_fun():
    import scipy.special

    prec = 1e-6

    # Test case 1: gamma(1, 1)
    val = scipy.special.gammainc(1, 1) * scipy.special.gamma(1)
    assert abs(gamma_fun(1, 1.) - val) < prec

    # Test case 2: gamma(2, 2)
    val = scipy.special.gammainc(2, 2) * scipy.special.gamma(2)
    assert abs(gamma_fun(2, 2.) - val) < prec

    # Test case 3: gamma(3, 0.5)
    val = scipy.special.gammainc(3, 0.5) * scipy.special.gamma(3)
    assert abs(gamma_fun(3, 0.5) - val) < prec

    # Test case 4: gamma(0.5, 0.5)
    val = scipy.special.gammainc(0.5, 0.5) * scipy.special.gamma(0.5)
    assert abs(gamma_fun(0.5, 0.5) - val) < prec

    # Test case 5: gamma(5, 3)
    val = scipy.special.gammainc(5, 3) * scipy.special.gamma(5)
    assert abs(gamma_fun(5, 3.) - val) < prec

    # Test case 6: gamma(10, 10)
    val = scipy.special.gammainc(10, 10) * scipy.special.gamma(10)
    assert abs(gamma_fun(10, 10.) - val) < prec    


def test_binomial_prefactor():
    bf = get_binomial_prefactor(jnp.arange(10))
    s, ia, ib, xpa, xpb = 1, 1, 2, -0.1, 0.1
    bf = jax.jit(bf)
    bf(s, ia, ib, xpa, xpb)
    
    print(Reference.binomial_prefactor(s, ia, ib, xpa, xpb))
    print(bf(s, ia, ib, xpa, xpb))
    
    print(Reference.binomial_prefactor(s, ia, ib, xpa, xpb) - bf(s, ia, ib, xpa, xpb) )

    assert abs(Reference.binomial_prefactor(s, ia, ib, xpa, xpb) - bf(s, ia, ib, xpa, xpb)) < 1e-10

def test_gto_overlap():
    integrator = PyQInt()

    # parameters
    c_1, c_2 = 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2 = 0.3, 0.1
    lmn1, lmn2 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ])
    p_1, p_2 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))

    # overlap function
    overlap = get_gaussian_functions(jnp.concatenate([lmn1, lmn2]).max()+10)[0]
    overlap = jax.jit(overlap)

    print(integrator.overlap_gto(gto_1, gto_2))
    print(overlap(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2))

    assert abs(integrator.overlap_gto(gto_1, gto_2) - overlap(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2)) < 1e-10

def test_gto_kinetic():
    integrator = PyQInt()

    # parameters
    c_1, c_2 = 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2 = 0.3, 0.1
    lmn1, lmn2 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ])
    p_1, p_2 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))

    # overlap function
    kinetic = get_gaussian_functions(jnp.concatenate([lmn1, lmn2]).max()+10)[1]
    kinetic = jax.jit(kinetic)

    print(integrator.kinetic_gto(gto_1, gto_2))
    print(kinetic(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2))

    assert abs(integrator.kinetic_gto(gto_1, gto_2) - kinetic(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2)) < 1e-10

def test_a_array():
    l1, l2, pa, pb, cp, g = 2,3,0.1, 0.2, 0.3, 0.1

    a_array = get_a_array(max(l1,l2))
    # jax.make_jaxpr(a_array)(l1, l2, pa, pb, cp, g)
    a_array = jax.jit(a_array)

    print(a_array(l1, l2, pa, pb, cp, g))
    print(Reference.a_array(l1, l2, pa, pb, cp, g))

    assert jnp.allclose( a_array(l1, l2, pa, pb, cp, g)[:(l1+l2+1)], jnp.array(Reference.a_array(l1, l2, pa, pb, cp, g)) )

def test_a_array_grad():
    l1, l2, pa, pb, cp, g = 2,3,0.1, 0.2, 0.3, 0.1
    a_array = get_a_array(max(l1,l2))
    grad = jax.jit(jax.grad( lambda *xs : a_array(*xs).sum(), argnums = [2,3,4,5]))
    grad(l1,l2,pa,pb,cp,g)
    grad(l1,l2,pa,pb,cp,g)    
    
def test_gto_nuclear():
    integrator = PyQInt()

    # parameters
    c_1, c_2 = 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2 = 0.3, 0.1
    lmn1, lmn2 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ])
    p_1, p_2, nuc = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))

    # overlap function
    nuclear = get_gaussian_functions(jnp.concatenate([lmn1, lmn2]).max()+1)[2]
    nuclear = jax.jit(nuclear)
    print(nuclear(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, nuc))
    print(integrator.nuclear_gto(gto_1, gto_2, nuc.tolist()))

    assert abs(integrator.nuclear_gto(gto_1, gto_2, nuc.tolist()) - nuclear(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, nuc)) < 1e-10

def test_b_array():
    assert False

def test_gto_ee():
    assert False
    
def test_overlap():
    assert False

def test_kinetic():
    assert False

def test_nuclear():
    assert False

def test_ee():
    assert False

# TODO: what a shame. this function is so nice, but so useless :(
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
    # test_a_array()
    test_gto_nuclear()
