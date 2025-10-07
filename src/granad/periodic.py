import jax
import jax.numpy as jnp

from granad.orbitals import _fill_matrix
from granad import _numerics

# TODO: performance meh because nothing is cached and everything is recomputed
# TODO: attribute access
# TODO: makes material class kinda superfluous
# TODO: duplicated functionality!!! => orbitallist already implements response theory etc, better to factor out those methods to _numerics
class Periodic:
    """Turns an OrbitalList into a unit cell of a periodic structure"""
    
    def __init__(self, lst, lattice_vectors, fourier_cutoff):
        self.lst = lst
        self.lattice_vectors = lattice_vectors
        self.fourier_cutoff = fourier_cutoff

    @property
    def basis_size(self):
        return self.lattice_vectors.shape[0]    
        
    def get_intracell_phases(self, ks):
        """returns relative phases between orbitals in unit cell e^(ik(r-r')) as K x N x N array"""
        diff = self.lst.positions - self.lst.positions[:, None]
        return jnp.exp(-1j * diff @ ks).T

    def get_intercell_phases(self, ks):
        """returns relative phases between unit cells e^(ik(DR')) as K x N_f array"""        
        return jnp.exp(-1j * self.get_grid_vectors() @ ks).T

    def matrix_to_kspace(self, mat, ks):
        """turns matrix slice as N_f x N x N array to k-space as K x N x N array"""
        
        # phases between orbs in cell, K x N x N
        intracell_phases = jnp.ones_like(self.get_intracell_phases(ks))

        # phases between cells, K x Nf
        intercell_phases = self.get_intercell_phases(ks)

        # produces matrix-valued fourier-trafo, ie non-hermitian
        mat = jnp.einsum('kf, fij -> kij', intercell_phases, mat)
        make_hermitian = lambda m : jnp.triu(m) + jnp.triu(m, 1).conj().T
        mat = jax.vmap(make_hermitian)(mat)

        return intracell_phases * mat

    def matrix_to_kspace_derivative(self, mat, ks):
        """turns matrix slice as N_f x N x N array to k-space as 3 x K x N x N array representing k-derivative of matrix"""
        
        # phases between orbs in cell, K x N x N
        intracell_phases = self.get_intracell_phases(ks)

        # phases between cells, K x Nf
        intercell_phases = self.get_intercell_phases(ks)
        
        # d_k \sum_R e^{ik(R + r - r')} h_{rr'}(R) => i e^{ik(r-r')} \sum R e^{ikR} h_{rr'} + \sum_R e^{ikR} (r-r') h_{rr'}(R) = phase * \sum a(r) + b(r)

        # a-matrix: R * mat => 3 x N_f x N x N
        a = jnp.einsum('nc, nij -> cnij', self.get_grid_vectors(), mat)
        
        # b-matrix: dress matrix with distance vectors (r-r') => 3 x N_f x N x N
        diff = self.lst.positions - self.lst.positions[:, None]
        b = jnp.einsum('nij, ijc -> cnij', mat, diff)
        
        # make hermitian
        mat = -1j * jnp.einsum('kn, cnij -> ckij', intercell_phases, a+b)
        mat = jnp.triu(mat) + jnp.swapaxes(jnp.triu(mat, 1).conj(), -1, -2)

        # fourier trafo => 3 x K x N x N
        return -intracell_phases[None] * mat

    def get_grid_vectors(self):
        """returns grid vectors for fourier trafo as N_f x 3"""
        return self.get_grid(self.lattice_vectors, self.fourier_cutoff)

    def get_grid_reciprocal(self, n):
        return self.get_grid(self.reciprocal_basis.T, n).T
        
    def get_grid(self, vecs, n):
        from itertools import product
        grid_params = [(-n, n + 1) for i in range(self.basis_size)] 
        grid = jnp.array(list(product( *(range(*params) for params in grid_params))))
        return grid @ vecs

    def _matrix(self, coupling):
        grid_vectors = self.get_grid_vectors()
        group_ids = jnp.array( [orb.group_id.id for orb in self.lst._list] )            
        return jnp.array([_fill_matrix(self.lst.positions, self.lst.positions + vec, group_ids, coupling) for vec in grid_vectors])

    def get_hamiltonian(self, ks):        
        return self.matrix_to_kspace(self._matrix(self.lst.couplings.hamiltonian), ks)

    def get_overlap(self, ks):
        if self.lst.is_ortho:
            return jnp.repeat(jnp.eye(len(self.lst))[None], ks.shape[-1], axis = 0)
        return self.matrix_to_kspace(self._matrix(self.lst.couplings.overlap), ks)

    def get_hamiltonian_ortho(self, ks):
        h = self.get_hamiltonian(ks)
        
        s = self.get_overlap(ks)
        overlap_vals, overlap_vecs = jnp.linalg.eigh(s)  
        sqrt_overlap = overlap_vals**(-0.5)
        ortho_trafo = jnp.einsum('kij, kj, klj -> kil', overlap_vecs, sqrt_overlap, overlap_vecs)
        ortho_trafo_inv = jnp.linalg.inv(ortho_trafo)
        return jnp.einsum('kji, kjl, klm -> kim', ortho_trafo.conj(), h, ortho_trafo)

    @staticmethod
    def get_mu(energies, filling_fraction = 0.5, iters = 80):
        """determine chemical potential"""
        energies = energies.flatten().sort()
        lo, hi = energies.min(), energies.max()
        n = energies.shape[0]

        def dos_integral(mu):
            return (energies <= mu).sum() / n

        for _ in range(iters):
            mu = 0.5*(lo+hi)
            if dos_integral(mu) >= filling_fraction:
                hi = mu
            else:
                lo = mu

        return 0.5*(lo+hi)

    def get_density_matrix(self, ks, mu, gs = 2):
        vals, vecs = self.get_eigenbasis(ks)
        occupations = vals <= mu        
        return gs * jnp.einsum('kji, ki, kli -> kjl', vecs, occupations, vecs.conj())

    def get_eigenbasis(self, ks, ortho = False):
        if ortho:
            h = self.get_hamiltonian_ortho(ks)
        else:
            h = self.get_hamiltonian(ks)
        return jnp.linalg.eigh(h)
    
    def get_velocity_operator(self, ks, vecs = None):
        """returns velocity operator. If vecs is passed, transformed with vecs"""
        v = self.matrix_to_kspace_derivative(self._matrix(self.lst.couplings.hamiltonian), ks)
        if vecs is None:
            return v
        return jnp.einsum('kij, ckil, klm -> ckjm', vecs.conj(), v, vecs)
    
    @staticmethod
    def get_ip_conductivity_inter(v, energies, omegas, mu, beta, relaxation_rate = 0.1):
        """
        Compute the **interband (optical) contribution** to the longitudinal 
        conductivity tensor using the independent-particle Kubo–Greenwood formula.

        Parameters
        ----------
        v : array_like, shape (3, K, N, N)
            Velocity operator at each k-point in the **eigenbasis** of the Hamiltonian.
            The first axis corresponds to Cartesian components (x, y, z).
        energies : array_like, shape (K, N)
            Band energies εₙ(k) for each k-point.
        omegas : array_like, shape (W,)
            Frequency grid ω for which σ(ω) is evaluated.
        mu : float
            Chemical potential.
        beta : float
            Inverse temperature β = 1/(k_B T). Use jnp.inf for T → 0 limit.
        relaxation_rate : float, optional
            Phenomenological broadening Γ (energy units). Defaults to 0.1.

        Returns
        -------
        sigma_inter : array_like, shape (W, 3, 3)
            Complex interband optical conductivity tensor σᵢⱼ(ω).
            The imaginary part encodes dispersion (reactive) response, and
            the real part gives absorption.

        Notes
        -----
        Implements
            σᵢⱼ^(inter)(ω) ∝ ∑ₖ ∑_{n≠m}
            [(fₙ - fₘ)/(εₙ - εₘ)] ·
            [vᵢₙₘ(k) vⱼₘₙ(k)] /
            [ω + iΓ + εₙ - εₘ].

        The diagonal (n = m) terms are excluded; those are accounted for
        in the intraband (Drude) contribution.
        """

        def inner(idx):
            delta_e = energies[idx][:, None] - energies[idx]
            delta_occ = (occupations[idx][:, None] - occupations[idx])

            # same energy => 0 / 0 => nan, needs to be set to zero, considered in intraband
            mat = jnp.nan_to_num(delta_occ / delta_e)
            x = mat / (omegas[:, None, None] + 1j*relaxation_rate + delta_e)
            return jnp.nan_to_num(jnp.einsum('wnm, cmn, dnm -> wcd', x, v[:, idx], v[:, idx]))

        occupations = _numerics.fermi(energies, beta, mu)

        return jax.vmap(inner)(jnp.arange(energies.shape[0])).sum(axis = 0) / v.shape[-1]

    @staticmethod
    def get_ip_conductivity_intra(v, energies, omegas, mu, beta, relaxation_rate = 0.1):
        """
        Compute the **intraband (Drude)** contribution to the conductivity tensor
        within the independent-particle approximation.

        Parameters
        ----------
        v : array_like, shape (3, K, N, N)
            Velocity operator at each k-point in the **eigenbasis** of the Hamiltonian.
            The first axis corresponds to Cartesian components (x, y, z).
        energies : array_like, shape (K, N)
            Band energies εₙ(k) for each k-point.
        omegas : array_like, shape (W,)
            Frequency grid ω for which σ(ω) is evaluated.
        mu : float
            Chemical potential.
        beta : float
            Inverse temperature β = 1/(k_B T). Use jnp.inf for T → 0 limit.
        relaxation_rate : float, optional
            Phenomenological broadening Γ (energy units). Defaults to 0.1.

        Returns
        -------
        sigma_intra : array_like, shape (W, 3, 3)
            Complex intraband (Drude) conductivity tensor σᵢⱼ(ω).

        Notes
        -----
        Implements
            σᵢⱼ^(intra)(ω) ∝ i / (ω + iΓ) ·
            ∑ₖ ∑ₙ [ -∂f(εₙₖ)/∂εₙₖ ] vᵢₙₙ(k) vⱼₙₙ(k).

        The intraband term captures the free-carrier (Drude) response.
        It depends on the derivative of the Fermi function and hence
        only states within ~k_BT of the Fermi level contribute.
        """

        v_diag = jnp.real(jnp.diagonal(v, axis1=-2, axis2=-1))        
        prefac = 1j/(omegas + 1j*relaxation_rate)
        
        # way to large intermediate array
        deriv = jax.jacrev(_numerics.fermi)(energies, beta, mu)

        return prefac[:, None, None] * jnp.einsum('kiki, ckii, dkii -> cd', -deriv, v, v) / energies.shape[0]

    def show_2d(self, n): 
        grid = self.get_grid(self.lattice_vectors, n)
        pos = jnp.concatenate([self.lst.positions + vec for vec in grid])
        plt.scatter(pos[:, 0], pos[:, 1])
        plt.axis('equal')
        plt.show()
        
    def get_kgrid_monkhorst_pack(self, ns = None):
        """Returns uniform grid of k-points in the reciprocal primitive cell."""

        def prefac(n):
            i = jnp.arange(n)
            return (2*(i + 1) - n - 1) / (2*n)

        ns = ns or [40, 40, 40]
        n = self.basis_size

        if n == 1:
            ks = prefac(ns[0]) * self.reciprocal_basis[:, 0][:, None]
        elif n == 2:
            S1, S2 = jnp.meshgrid(prefac(ns[0]), prefac(ns[1]), indexing='ij')
            b1, b2 = self.reciprocal_basis.T
            ks = S1[..., None]*b1 + S2[..., None]*b2      
            ks = ks.reshape(-1, 3).T                        
        elif n == 3:
            b1, b2, b3 = self.reciprocal_basis.T
            S1, S2, S3 = jnp.meshgrid(prefac(ns[0]), prefac(ns[1]), prefac(ns[2]), indexing='ij')  
            ks = S1[..., None]*b1 + S2[..., None]*b2, S3[..., None]*b3
            ks = ks.reshape(-1, 3).T

        return ks

    @property
    def reciprocal_basis(self):
        """
        Return the reciprocal lattice basis as an n x 3 array B such that
        B @ self.lattice_vectors.T = 2π I_n. Works for n = 1, 2, or 3
        (lattice embedded in 3D). Raises a ValueError for degenerate lattices.
        """
        lv = self.lattice_vectors  # shape (n, 3)
        n = lv.shape[0]

        if n == 1:
            a1 = lv[0]
            denom = jnp.dot(a1, a1)
            if denom == 0:
                raise ValueError("Degenerate 1D lattice: |a1| = 0.")
            b1 = 2 * jnp.pi * a1 / denom
            return b1[:, None]

        elif n == 2:
            a1, a2 = lv[0], lv[1]
            nvec = jnp.cross(a1, a2)
            area_sq = jnp.dot(nvec, nvec)  # |a1 × a2|^2
            if area_sq == 0:
                raise ValueError("Degenerate 2D lattice: a1 and a2 are collinear.")
            b1 = 2 * jnp.pi * jnp.cross(a2, nvec) / area_sq
            b2 = 2 * jnp.pi * jnp.cross(nvec, a1) / area_sq
            return jnp.stack([b1, b2], axis=1)

        elif n == 3:
            a1, a2, a3 = lv[0], lv[1], lv[2]
            volume = jnp.dot(a1, jnp.cross(a2, a3))
            if volume == 0:
                raise ValueError("Degenerate 3D lattice: volume is zero.")
            b1 = 2 * jnp.pi * jnp.cross(a2, a3) / volume
            b2 = 2 * jnp.pi * jnp.cross(a3, a1) / volume
            b3 = 2 * jnp.pi * jnp.cross(a1, a2) / volume
            return jnp.stack([b1, b2, b3], axis=1)

        else:
            raise ValueError("lattice_vectors must have shape n x 3 with 1 <= n <= 3.")
