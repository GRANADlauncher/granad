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
# TODO: exploit symmetries via some sort of memoization
# TODO: optimize gaussian elementary functions (type casts)
# TODO: gaussian_norm : loop for n > len(vals)

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
def gamma_fun(s, x):
    return gammainc(s+0.5, x) * gamma(s+0.5) * 0.5 * jnp.pow(x,-s-0.5) 
    
def binomial(n, m):
    return (n > 0) * (m > 0) * (n > m) * (factorial(n) / (factorial(n - m) * factorial(m)) - 1) + 1

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

    l_max = l_max + 2
    overlap = get_overlap(l_max)
    
    return kinetic


### NUCLEAR ###
def get_a_array(l_max):

    def a_term(i, r, u, l1, l2, pa, pb, cp, g):
        return ( jnp.pow(-1,i) * binomial_prefactor(i,l1,l2,pa,pb)*
                 jnp.pow(-1,u) * factorial(i)*jnp.pow(cp,i-2*r-2*u)*
                 jnp.pow(0.25/g,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u) )
    
    def a_legal(iI, i, r, u, l1, l2):
        return jnp.logical_and( jnp.logical_and( jnp.logical_and(i < l1 + l2 + 1, r <= i//2),  u <= (i-2*r)//2),  iI == i - 2 * r - u )

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

    def fb(i, l1, l2, p, a, b, r, g):
        return binomial_prefactor(i, l1, l2, p-a, p-b) * bb0(i, r, g)
    
    def b_term(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta):
        a, b = i1+i2-2*(r1+r2),u
        return (fb(i1,l1,l2,px,ax,bx,r1,g1)*
                jnp.pow(-1,i2) * fb(i2,l3,l4,qx,cx,dx,r2,g2)*
                jnp.pow(-1,u)* factorial(a) / factorial(b) / factorial(a - 2*b)*
                jnp.pow(qx-px,i1+i2-2*(r1+r2)-2*u)/
                jnp.pow(delta,i1+i2-2*(r1+r2)-u))
    
    def b_legal(i, i1, i2, r1, r2, u, l1, l2, l3, l4):
        return jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_and(i1 < l1 + l2 + 1, i2 < l3 + l4 + 1),
                        r1 < i1//2 + 1),
                    r2 < i2//2 + 1),
                u<(i1+i2)//2-r1-r2+1),
            i == i1+i2-2*(r1+r2)-u)

    # def b_wrapped(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta):
    #     return jax.lax.cond(b_legal(i, i1, i2, r1, r2, u, l1, l2, l3, l4), lambda: b_term(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta), lambda : 0.0)
    
    def b_wrapped(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta):
        return b_term(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta)

    
    def b_loop(i, l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta):
        f = lambda i1, i2, r1, r2, u : b_wrapped(i, i1, i2, r1, r2, u, l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta)
        return jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(f,
                                                            in_axes=(0, None, None, None, None)), in_axes=(None, 0, None, None, None)), in_axes=(None, None, 0, None, None)), in_axes=(None, None, None, 0, None)), in_axes=(None, None, None, None, 0))(inner_i_range, inner_i_range, inner_i_range, inner_i_range, inner_i_range).sum()

    
    def b_array(l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta):
        return jax.vmap(lambda i : b_loop(i, l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta))(outer_i_range)    

    outer_i_range = jnp.arange(4*l_max + 1)
    binomial_prefactor = get_binomial_prefactor(outer_i_range)

    inner_i_range = jnp.arange(2*l_max + 1)
    
    return b_array

def get_repulsion(l_max):

    def loop_body(i, j, k, lmn, rg):
        return jax.lax.cond( jnp.logical_and(jnp.logical_and(i <= lmn[0], j <= lmn[1]), k <= lmn[2]), lambda: gamma_fun(i+j+k , rg), lambda: 0.0)
    
    def loop_body(i, j, k, lmn, rg):
        return gamma_fun(i+j+k , rg)

    
    def repulsion(alpha1, lmn1, pos1, alpha2, lmn2, pos2, alpha3, lmn3, pos3, alpha4, lmn4, pos4):

        print("I compiled")

        rab2 = jnp.linalg.norm(pos1-pos2)**2
        rcd2 = jnp.linalg.norm(pos3-pos4)**2
        
        gamma12 = alpha1 + alpha2
        p = (alpha1*pos1 + alpha2*pos2) / gamma12
        
        gamma34 = alpha3 + alpha4        
        q = (alpha3*pos3 + alpha4*pos4) / gamma34

        rpq2 = jnp.linalg.norm(p-q)**2
        
        delta = 0.25 * (1.0 / gamma12 + 1.0 / gamma34)

        # TODO: hmpf
        b = jax.vmap(lambda i : b_array(lmn1[i], lmn2[i], lmn3[i], lmn4[i], p[i], pos1[i], pos2[i], q[i], pos3[i], pos4[i], gamma12, gamma34, delta) )(jnp.arange(3))

        lmn= lmn1+lmn2+lmn3+lmn4
        l = jax.vmap(jax.vmap(jax.vmap(lambda i, j, k : loop_body(i, j, k, lmn, 0.25*rpq2/delta), in_axes=(0, None, None)), in_axes=(None, 0, None)), in_axes=(None, None, 0))(indices, indices, indices)

        res = b[2] @ (b[1] @ (b[0] @ l))
        return 2.0 * jnp.pow(jnp.pi, 2.5) / (gamma12*gamma34*jnp.sqrt(gamma12+gamma34)) * jnp.exp(-alpha1*alpha2*rab2/gamma12) * jnp.exp(-alpha3*alpha4*rcd2/gamma34) * res
    

    b_array = get_b_array(l_max)
    indices = jnp.arange(4*l_max+1)
    
    return repulsion

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
def one_body_wrapper(func, m):

    def inner(orb1, orb2, *args):
        cs1, alphas1, lmn1, ps1, cs2, alphas2, lmn2, ps2 = orb1[:m], orb1[m:2*m], orb1[-6:-3], orb1[-3:], orb2[:m], orb2[m:2*m], orb2[-6:-3], orb2[-3:]
        norms1 = gaussian_norm(lmn1, alphas1)
        norms2 = gaussian_norm(lmn2, alphas2)
        
        return jax.lax.fori_loop(0, imax, lambda i, acci:
                                 acci + jax.lax.fori_loop(0, imax, lambda j, accj :
                                                          accj + cs1[i] * cs2[j] * norms1[i] * norms2[j] * func(alphas1[i], lmn1, ps1, alphas2[j], lmn2, ps2, *args), 0), 0)

    # number of coefficients to be looped over
    imax = m 
    
    return inner

def two_body_wrapper(func, m):
    
    def inner(orb1, orb2, orb3, orb4):
        cs1, alphas1, lmn1, ps1, cs2, alphas2, lmn2, ps2, cs3, alphas3, lmn3, ps3, cs4, alphas4, lmn4, ps4 = orb1[:m], orb1[m:2*m], orb1[-6:-3], orb1[-3:], orb2[:m], orb2[m:2*m], orb2[-6:-3], orb2[-3:], orb3[:m], orb3[m:2*m], orb3[-6:-3], orb3[-3:], orb4[:m], orb4[m:2*m], orb4[-6:-3], orb4[-3:]

        norms1 = gaussian_norm(lmn1, alphas1)
        norms2 = gaussian_norm(lmn2, alphas2)
        norms3 = gaussian_norm(lmn3, alphas3)
        norms4 = gaussian_norm(lmn4, alphas4)

        # import pdb; pdb.set_trace()
        # print("le compile")
        
        return jax.lax.fori_loop(0, imax, lambda i, acci:
                                 acci + jax.lax.fori_loop(0, imax, lambda j, accj :
                                                          accj + jax.lax.fori_loop(0, imax, lambda k, acck :
                                                                                   acck + jax.lax.fori_loop(0, imax, lambda l, accl :
                                                                                                            accl + norms1[i] * norms2[j] * norms3[k] * norms4[l] * cs1[i] * cs2[j] * cs3[k] * cs4[l] * \
                                                                                                            func(alphas1[i], lmn1, ps1, alphas2[j], lmn2, ps2, alphas3[k], lmn3, ps3, alphas4[l], lmn4, ps4), 0), 0), 0), 0)
    
    # number of coefficients to be looped over
    imax = m
    
    return inner

def expand_gaussian(orb_list, expansion):
    """converts OrbitalList into a representation compatible with gaussian integration.
    
    Args:
    orb_list: OrbtalList 
    expansion: dictionary mapping group ids to arrays
    
    Returns:
    """
    
    # translate orbital list to array representation
    orbitals = jnp.array( [jnp.concatenate([jnp.concatenate(expansion[o.group_id]), o.position]) for o in orb_list] )

    # size of coefficients for unpacking array arguments
    coeff_size = expansion[orb_list[0].group_id][0].size

    # array representation of nuclei, index map to keep track of orbitals => nuclei
    nuclei_positions, idxs, orbitals_to_nuclei = jnp.unique(orbitals[:,:3], axis = 0, return_index = True, return_inverse = True)
    nuclei_charges = jnp.array([float(get_atomic_charge(o.atom_name)) for o in orb_list])[idxs]
    nuclei = jnp.hstack( [nuclei_positions, nuclei_charges[:,None]] )

    # max l for static loops compatible with JIT+reverse AD
    l_max = orbitals[3:6].max()

    # gto funcs 
    overlap, kinetic, nuclear, repulsion = get_gaussian_functions(l_max)

    # cgf expansion
    overlap = one_body_wrapper(get_overlap(l_max), coeff_size)
    kinetic = one_body_wrapper(get_kinetic(l_max), coeff_size)
    nuclear = one_body_wrapper(get_nuclear(l_max), coeff_size)
    repulsion = two_body_wrapper(get_repulsion(l_max), coeff_size)
    
    return overlap, kinetic, nuclear, repulsion, orbitals, nuclei, orbitals_to_nuclei

### PYQINT REFERENCE NAMESPACE: USE EXPOSED FUNCS FROM LIB OR PURE PYTHON IMPLS OF PYQINT C++ FUNCS ###
import math

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
        return (math.pow(-1, i) * Reference.binomial_prefactor(i, l1, l2, pax, pbx) *
                math.pow(-1, u) * factorial(i) * math.pow(cpx, i - 2 * r - 2 * u) *
                math.pow(0.25 / gamma, r + u) / factorial(r) / factorial(u) / factorial(i - 2 * r - 2 * u))    
    
### QUICK TESTS ###
def test_binom():
    assert False
    
def test_gaussian_norm():
    assert False

def test_double_factorial():
    assert False

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
    overlap = get_overlap(jnp.concatenate([lmn1, lmn2]).max()+10)
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
    kinetic = get_kinetic(jnp.concatenate([lmn1, lmn2]).max())
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
    arr = grad(l1,l2,pa,pb,cp,g)    

    assert not jnp.isnan(jnp.array(arr)).any()

def test_b_array_grad():
    l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta = 1, 2, 3, 4, 0.1, 0.2, 0.3, 0.1, 0.1, 0.2, 0.3, 0.1, 0.3

    b_array = get_b_array(max(l1,l2,l3,l4))
    grad = jax.jit(jax.grad( lambda *xs : b_array(*xs).sum(), argnums = [5,6,7,8,9,10,11,12]))
    grad(l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta)
    
    arr = grad(l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta)

    assert not jnp.isnan(jnp.array(arr)).any()
    
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
    nuclear = get_nuclear(jnp.concatenate([lmn1, lmn2]).max())
    nuclear = jax.jit(nuclear)
    print(nuclear(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, nuc))
    print(integrator.nuclear_gto(gto_1, gto_2, nuc.tolist()))

    assert abs(integrator.nuclear_gto(gto_1, gto_2, nuc.tolist()) - nuclear(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, nuc)) < 1e-10
        
def test_gto_repulsion():
    integrator = PyQInt()

    # parameters
    c_1, c_2, c_3, c_4 = 0.391957, 0.391957, 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2, alpha_3, alpha_4 = 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))
    gto_3 = gto(c_3, p_3.tolist(), alpha_3, *(lmn3.tolist()))
    gto_4 = gto(c_4, p_4.tolist(), alpha_4, *(lmn4.tolist()))

    # overlap function
    repulsion = get_repulsion(jnp.concatenate([lmn1, lmn2, lmn3, lmn4]).max())
    repulsion = jax.jit(repulsion)
    print(repulsion(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, alpha_3, lmn3, p_3, alpha_4, lmn4, p_4))
    print(integrator.repulsion_gto(gto_1, gto_2, gto_3, gto_4))

    import pdb; pdb.set_trace()

    assert abs(repulsion(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, alpha_3, lmn3, p_3, alpha_4, lmn4, p_4) - integrator.repulsion_gto(gto_1, gto_2, gto_3, gto_4)) < 1e-10
    
def test_overlap():
    
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])


    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )

    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    integrator = PyQInt()
    n1 = integrator.overlap(cgf1, cgf2)

    overlap = one_body_wrapper(get_overlap(jnp.concatenate([lmn1, lmn2]).max()), 2)
    n2 = overlap( orb1, orb2 )

    assert abs(n1-n2) < 1e-10
    
def test_kinetic():
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])

    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )
    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    integrator = PyQInt()
    n1 = integrator.kinetic(cgf1, cgf2)

    kinetic = one_body_wrapper(get_kinetic(jnp.concatenate([lmn1, lmn2]).max()), 2)    
    n2 = kinetic( orb1, orb2 )

    assert abs(n1-n2) < 1e-10

def test_nuclear():    
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])

    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )
    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    integrator = PyQInt()
    n1 = integrator.nuclear(cgf1, cgf2, p_3.tolist(), 1)

    nuclear = one_body_wrapper(get_nuclear(jnp.concatenate([lmn1, lmn2]).max()), 2)    
    n2 = nuclear( orb1, orb2, p_3 )

    assert abs(n1-n2) < 1e-10

def test_repulsion():
    
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])


    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 1, 1, 1, 1, 1, 1, 1, 1

    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )
    orb3 = jnp.concatenate( [jnp.array([c_31, c_32]), jnp.array([alpha_31, alpha_32]), lmn3, p_3] )
    orb4 = jnp.concatenate( [jnp.array([c_41, c_42]), jnp.array([alpha_41, alpha_42]), lmn4, p_4] )
    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    cgf3 = cgf(p_3.tolist())
    cgf3.add_gto( c_31, alpha_31, *lmn3.tolist() )
    cgf3.add_gto( c_32, alpha_32, *lmn3.tolist() )

    cgf4 = cgf(p_4.tolist())
    cgf4.add_gto( c_41, alpha_41, *lmn4.tolist() )
    cgf2.add_gto( c_42, alpha_42, *lmn4.tolist() )

    
    integrator = PyQInt()
    n1 = integrator.repulsion(cgf1, cgf2, cgf3, cgf4)

    repulsion = two_body_wrapper(get_repulsion(jnp.concatenate([lmn1, lmn2, lmn3, lmn4]).max()), 2)
    # repulsion  = jax.jit(repulsion)
    n2 = repulsion( orb1, orb2, orb3, orb4 )

    import pdb; pdb.set_trace()


    assert abs(n1-n2) < 1e-10

# convention used for built-in gaussian basis: tuples have the form (coefficients, alphas, lmn), where lmn is exponent of cartesian coordinates x,y,z
sto_3g = {
    "pz" : (jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
            jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
            jnp.array( [ 0,0,1 ]) )
    }



if __name__ == '__main__':
    # test_overlap()
    test_gto_repulsion()
