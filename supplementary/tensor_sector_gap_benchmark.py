"""Numerical benchmark for the non-scalar covariance factorization.
The script checks the local Delta_2 roots, the TF Schur factor at z=0,
and the complete finite-N tensor block spectrum.
"""
import numpy as np


def local_roots(phi, psi, gamma):
    S = phi + psi
    # H=z^2+gamma*z+S, G=z^2+3 gamma*z+2 gamma^2+S
    H = np.poly1d([1.0, gamma, S])
    G = np.poly1d([1.0, 3.0 * gamma, 2.0 * gamma**2 + S])
    return np.roots(H * G - np.poly1d([4.0 * phi * psi]))


def tf_schur_zero(N, k, omega0, gamma):
    vals = []
    for ell in range(N):
        m = (k - ell) % N
        phi = omega0**2 + 4.0 * np.sin(np.pi * ell / N) ** 2
        psi = omega0**2 + 4.0 * np.sin(np.pi * m / N) ** 2
        S = phi + psi
        Delta0 = (phi - psi) ** 2 + 2.0 * gamma**2 * S
        vals.append(gamma * S / Delta0)
    return 1.0 + 2.0 * gamma * np.mean(vals)


def full_tensor_matrix(N, k, omega0, gamma):
    L = np.zeros((4 * N, 4 * N), dtype=float)
    for ell in range(N):
        m = (k - ell) % N
        phi = omega0**2 + 4.0 * np.sin(np.pi * ell / N) ** 2
        psi = omega0**2 + 4.0 * np.sin(np.pi * m / N) ** 2
        A, B, C, D = ell, N + ell, 2 * N + ell, 3 * N + ell
        L[A, B] = 1.0
        L[A, C] = 1.0
        L[B, A] = -psi
        L[B, B] = -gamma
        L[B, D] = 1.0
        L[C, A] = -phi
        L[C, C] = -gamma
        L[C, D] = 1.0
        L[D, B] = -phi
        L[D, C] = -psi
        L[D, D] = -2.0 * gamma
        # -2 gamma d, d=N^{-1} sum D_ell
        L[D, 3 * N : 4 * N] -= 2.0 * gamma / N
    return L


def benchmark(N_values=(8, 16, 32), k_values=(0, 1, 2), omega0=0.8, gamma=1.1):
    print(f"omega0={omega0}, gamma={gamma}")
    for N in N_values:
        local_gap = np.inf
        block_gap = np.inf
        schur_min = np.inf
        for k in k_values:
            schur_min = min(schur_min, tf_schur_zero(N, k, omega0, gamma))
            for ell in range(N):
                m = (k - ell) % N
                phi = omega0**2 + 4.0 * np.sin(np.pi * ell / N) ** 2
                psi = omega0**2 + 4.0 * np.sin(np.pi * m / N) ** 2
                local_gap = min(local_gap, -np.max(np.real(local_roots(phi, psi, gamma))))
            eig = np.linalg.eigvals(full_tensor_matrix(N, k, omega0, gamma))
            block_gap = min(block_gap, -np.max(np.real(eig)))
        print(f"N={N:3d}: local_gap={local_gap:.6e}, full_TF_gap={block_gap:.6e}, min_DTF(0)={schur_min:.6e}")


if __name__ == "__main__":
    benchmark()
