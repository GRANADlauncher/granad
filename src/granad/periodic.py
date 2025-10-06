import jax.numpy as jnp

from granad.orbitals import _fill_matrix

# TODO: attribute access
# TODO: makes material class kinda superfluous 
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
        intracell_phases = self.get_intracell_phases(ks)

        # phases between cells, K x Nf
        intercell_phases = self.get_intercell_phases(ks)
        
        return intracell_phases * jnp.einsum('kf, fij -> kij', intercell_phases, mat)

    def matrix_to_kspace_derivative(self, mat, ks):
        """turns matrix slice as N_f x N x N array to k-space as 3 x K x N x N array representing k-derivative of matrix"""
        
        # phases between orbs in cell, K x N x N
        intracell_phases = self.get_intracell_phases(ks)

        # phases between cells, K x Nf
        intercell_phases = self.get_intercell_phases(ks)
        
        # d_k \sum_R e^{ik(R + r - r')} h_{rr'}(R) => e^{ik(r - r')} \sum_R i(R + r - r') h_{rr'}(R) = e^{ik(r-r')} \sum iR \cdot (r-r')h_{rr'}(R) = e^{ik(r-r')} \sum_R iR t_{rr'}(R)

        # dress matrix with distance vectors (r-r')
        diff = self.lst.positions - self.lst.positions[:, None]
        mat = mat * diff

        # dress intercell phases by iR => 3 x K x Nf
        intercell_phases = jnp.einsum('ckn, kn -> ckn', self.get_grid_vectors(), intercell_phases)
        
        return intracell_phases[None] * jnp.einsum('ckf, cfij -> ckij', intercell_phases, mat)

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
        # return h
        
        if self.lst.is_ortho:
            return h
        
        s = self.get_overlap(ks)
        overlap_vals, overlap_vecs = jnp.linalg.eigh(s)  
        sqrt_overlap = overlap_vals**(-0.5)
        ortho_trafo = jnp.einsum('kij, kj, klj -> kil', overlap_vecs, sqrt_overlap, overlap_vecs)
        ortho_trafo_inv = jnp.linalg.inv(ortho_trafo)
        return jnp.einsum('kji, kjl, klm -> kim', ortho_trafo.conj(), h, ortho_trafo)

    @staticmethod
    def get_mu(energies, fraction = 0.5, gs = 1, iters = 80):
        # energies: (Nk, Nb), w: (Nk,) sum to 1
        lo, hi = energies.min(), energies.max()
        n = energies.shape[0]

        def dos_integral(mu):
            return gs * (energies <= mu).sum() / n

        for _ in range(iters):
            mu = 0.5*(lo+hi)
            if dos_integral(mu) >= fraction:
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

    def get_phases_derivative(self, ks):
        return
    
    def get_velocity_operator(self, ks):
        return

    def get_ip_green_function_inter(op1, op2):
        return
    
    def get_ip_green_function_intra(op1):
        return

    def show_2d(self, n): 
        grid = self.get_grid(self.lattice_vectors, n)
        pos = jnp.concatenate([self.lst.positions + vec for vec in grid])
        plt.scatter(pos[:, 0], pos[:, 1])
        plt.axis('equal')
        plt.show()

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
