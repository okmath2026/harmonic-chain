"""Finite matrix checks of the spectrally prepared Gaussian construction.

The scalar secular equation computes the candidate slow root. Its covariance
reconstruction is then checked against the real-space planar Ito covariance
generator and a separately coded real-space local energy extractor. No
trajectories, stochastic sampling, or full covariance supermatrix is used.

Run from any directory with Python, NumPy, SciPy, and mpmath installed.
The JSON output is written beside this script. The tests cover the displayed
finite parameter choices; they do not replace the analytic uniform bounds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import mpmath as mp
import numpy as np
import scipy.linalg as la

from independent_scalar_matrix_audit import physical_matrices


def slow_root(n: int, k: int, omega: float, gamma: float):
    """High precision secular Newton/secant root, without target B4/C6."""
    with mp.workdps(70):
        w, g = mp.mpf(str(omega)), mp.mpf(str(gamma))
        t = [2 * mp.pi * j / n for j in range(n)]
        xi = 2 * mp.pi * k / n
        v = [w*w + 4*mp.sin(x/2)**2 for x in t]
        psi = [v[(k-j) % n] for j in range(n)]
        kappa = mp.fsum(mp.sin(t[j])**2/v[j] for j in range(n))/(g*n)

        def secular(z):
            total = mp.mpf("0")
            for vl, vm in zip(v, psi):
                h = z*(z+g)+vl+vm
                delta = ((z+2*g)*(z+g)+vl+vm)*h-4*vl*vm
                total += 2*g*(z+g)*h/delta
            return 1-total/n

        guess = -kappa*xi*xi
        root = mp.findroot(secular, (guess, guess*mp.mpf("0.95")),
                           tol=mp.mpf("1e-60"), maxsteps=100)
        residual = abs(secular(root))
        assert residual < mp.mpf("1e-55")
        return float(root), float(residual), float(kappa)


def candidate(n: int, k: int, omega: float, gamma: float, z: float):
    """Return unwhitened real-space complex covariance and S7 energy factor."""
    t = 2*np.pi*np.arange(n)/n
    xi = 2*np.pi*k/n
    v = omega*omega + 4*np.sin(t/2)**2
    paired = (k-np.arange(n)) % n
    vm = v[paired]
    h = z*(z+gamma)+v+vm
    delta = ((z+2*gamma)*(z+gamma)+v+vm)*h-4*v*vm
    aa = 4*gamma*(z+gamma)/delta
    bb = 2*gamma*(z*(z+gamma)+v-vm)/delta
    cc = 2*gamma*(z*(z+gamma)+vm-v)/delta
    dd = 2*gamma*(z+gamma)*h/delta

    # Planar dot amplitudes are divided by two to obtain each internal
    # component, and the resulting covariance is multiplied by N^{-1/2}.
    scalar_covariance = np.zeros((2*n, 2*n), dtype=complex)
    scalar_whitened = np.zeros_like(scalar_covariance)
    for ell, m in enumerate(paired):
        indices = np.ix_([ell, n+ell], [m, n+m])
        block = np.array([[aa[ell], bb[ell]], [cc[ell], dd[ell]]])
        block /= 2*np.sqrt(n)
        scalar_covariance[indices] = block
        scalar_whitened[indices] = np.diag([np.sqrt(v[ell]), 1]) @ block @ np.diag([np.sqrt(v[m]), 1])

    fourier = np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)
    transform = la.block_diag(fourier, fourier)
    # Bilinear covariance transforms use transpose, not adjoint. Thus the
    # inverse transform has F^* on the left and conjugate F on the right.
    covariance = np.kron(transform.conj().T @ scalar_covariance @ transform.conj(), np.eye(2))
    whitened_direct = np.kron(transform.conj().T @ scalar_whitened @ transform.conj(), np.eye(2))
    bond_factor = omega*omega + (1-np.exp(-1j*t))*(1-np.exp(-1j*(xi-t)))
    e_formula = np.mean(dd+bond_factor*aa)/2
    return covariance, whitened_direct, complex(e_formula), float(np.min(abs(delta)))


def local_mean_energies(covariance, n: int, omega: float):
    """Real-space local energy, including precisely the left bond."""
    energy = np.zeros(n, dtype=complex)
    for x in range(n):
        prev = (x-1) % n
        qx, qprev, px = slice(2*x, 2*x+2), slice(2*prev, 2*prev+2), slice(2*n+2*x, 2*n+2*x+2)
        energy[x] = (
            np.trace(covariance[px, px])/2
            + np.trace(covariance[qx, qx]+covariance[qprev, qprev]
                       -covariance[qx, qprev]-covariance[qprev, qx])/2
            + omega*omega*np.trace(covariance[qx, qx])/2
        )
    return energy


def energy_extractor(covariance, n: int, k: int, omega: float):
    phases = np.exp(-2j*np.pi*k*np.arange(n)/n)
    return np.dot(phases, local_mean_energies(covariance, n, omega))/np.sqrt(n)


def ito_covariance_action_sparse_noise(covariance, drift, n: int, gamma: float):
    """Physical Ito action, with only the nonzero 2-by-2 noise blocks."""
    derivative = drift @ covariance+covariance @ drift.T
    rotation = np.array([[0., 1.], [-1., 0.]])
    for x in range(n):
        p = slice(2*n+2*x, 2*n+2*x+2)
        derivative[p, p] += 2*gamma*(rotation @ covariance[p, p] @ rotation.T)
    return derivative


def complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def audit(n: int, k: int, omega: float, gamma: float, amplitude=.2, phase=.37):
    z, secular_residual, kappa = slow_root(n, k, omega, gamma)
    stiffness, drift, noises, _ = physical_matrices(n, omega, gamma)
    # Verify that the sparse on-site action uses the same physical rotation.
    rotation = np.array([[0., 1.], [-1., 0.]])
    for x, noise in enumerate(noises):
        p = slice(2*n+2*x, 2*n+2*x+2)
        assert np.array_equal(noise[p, p], rotation)
        assert np.count_nonzero(noise) == 2
    del noises
    values, vectors = la.eigh(stiffness)
    khalf = (vectors*np.sqrt(values)) @ vectors.T
    kinvhalf = (vectors/np.sqrt(values)) @ vectors.T
    ghalf = la.block_diag(np.kron(khalf, np.eye(2)), np.eye(2*n))
    ginvhalf = la.block_diag(np.kron(kinvhalf, np.eye(2)), np.eye(2*n))
    covariance, whitened_direct, e_formula, margin = candidate(n, k, omega, gamma, z)
    whitened = ghalf @ covariance @ ghalf
    e_physical = energy_extractor(covariance, n, k, omega)
    normalized = covariance/e_formula
    x = whitened/e_formula
    physical_action = ito_covariance_action_sparse_noise(normalized, drift, n, gamma)
    physical_residual = la.norm(physical_action-z*normalized, 'fro')/la.norm(normalized, 'fro')

    y = amplitude/2*(np.exp(1j*phase)*x+np.exp(-1j*phase)*x.conj())
    prepared_white = np.eye(4*n)/2+y
    prepared = ginvhalf @ prepared_white @ ginvhalf
    expected_profile = 1+amplitude/np.sqrt(n)*np.cos(2*np.pi*k*np.arange(n)/n+phase)
    measured_profile = local_mean_energies(prepared, n, omega)

    covariance0, z0_direct, e0_formula, _ = candidate(n, 0, omega, gamma, 0.)
    zero_white = ghalf @ covariance0 @ ghalf
    zero_identity = np.eye(4*n)/(2*np.sqrt(n))
    errors = {
        'bilinear_covariance_symmetry_fro': float(la.norm(normalized-normalized.T, 'fro')),
        'direct_whitened_formula_fro': float(la.norm(whitened-whitened_direct, 'fro')),
        'physical_ito_eigenmatrix_relative_fro': float(physical_residual),
        'left_bond_energy_formula_abs': float(abs(e_physical-e_formula)),
        'normalized_energy_amplitude_abs': float(abs(energy_extractor(normalized, n, k, omega)-1)),
        'prepared_covariance_imaginary_max': float(np.max(abs(prepared.imag))),
        'prepared_covariance_symmetry_fro': float(la.norm(prepared-prepared.T, 'fro')),
        'prepared_initial_energy_profile_max': float(np.max(abs(measured_profile-expected_profile))),
        'prepared_initial_complex_mode_abs': float(abs(energy_extractor(prepared, n, k, omega)-amplitude*np.exp(1j*phase)/2)),
        'zero_wave_identity_fro': float(la.norm(zero_white-zero_identity, 'fro')),
        'zero_wave_direct_whitening_fro': float(la.norm(z0_direct-zero_identity, 'fro')),
        'zero_wave_formula_energy_amplitude_abs': float(abs(e0_formula-1)),
        'zero_wave_physical_energy_amplitude_abs': float(abs(energy_extractor(covariance0, n, 0, omega)-1)),
    }
    tolerance = 1e-9
    assert max(errors.values()) < tolerance, errors
    min_physical = float(la.eigvalsh(prepared.real)[0])
    min_white = float(la.eigvalsh(prepared_white.real)[0])
    assert min_physical > 0 and min_white > 0
    assert abs(e_formula) > .5 and margin > 0
    return {
        'parameters': {'N': n, 'k': k, 'omega0': omega, 'gamma': gamma,
                       'beta': 2., 'amplitude': amplitude, 'phase': phase},
        'slow_root': z, 'secular_residual_70_digits': secular_residual,
        'initial_guess_diffusivity': kappa, 'local_denominator_margin': margin,
        'energy_amplitude_formula_real_imag': complex_pair(e_formula),
        'energy_amplitude_real_space_real_imag': complex_pair(e_physical),
        'normalized_whitened_frobenius_norm': float(la.norm(x, 'fro')),
        'normalized_whitened_operator_norm': float(la.svdvals(x)[0]),
        'sqrtN_times_normalized_operator_norm': float(np.sqrt(n)*la.svdvals(x)[0]),
        'prepared_min_eigenvalue_physical': min_physical,
        'prepared_min_eigenvalue_whitened': min_white,
        'maximum_asserted_error': max(errors.values()), 'absolute_error_tolerance': tolerance,
        'errors': errors, 'passed': True,
    }


def main():
    cases = [audit(n, 1, 1., gamma) for n in (32, 64) for gamma in (.7, 1.)]
    result = {
        'scope': 'Finite-dimensional arithmetic checks of S7; no trajectories or Monte Carlo.',
        'candidate': 'Secular slow root and explicit S7 Fourier reconstruction.',
        'independent_checks': 'Real-space planar Ito generator, left-bond local energy, and Gaussian positive definiteness.',
        'all_passed': all(row['passed'] for row in cases), 'cases': cases,
    }
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'all_passed': result['all_passed'], 'cases': len(cases),
                      'max_error': max(row['maximum_asserted_error'] for row in cases),
                      'minimum_covariance_eigenvalue': min(row['prepared_min_eigenvalue_physical'] for row in cases),
                      'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
