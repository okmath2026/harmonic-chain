"""Independent physical-covariance and matrix-Taylor audit.

No trajectories, Monte Carlo sampling, imported manuscript benchmark functions,
secular root fitting, or target B4/C6 coefficients enter the computed branch.

Route A builds the full planar Ito covariance generator in real space, projects
its symmetric O(2)-scalar covariance sector onto a translation character, and
only then checks the manuscript's Fourier equations and secular residual.

Route B builds a relative-coordinate Bloch matrix from left/right real-space
stiffness actions and the local momentum-diagonal Ito injection. Its eigenvalue
Taylor coefficients are obtained by a bordered linear-system recurrence.
The closed coefficients are evaluated only afterwards for comparison.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.linalg as la
import mpmath as mp


def physical_matrices(n: int, omega: float, gamma: float):
    shift = np.roll(np.eye(n), -1, axis=1)
    stiffness = (omega * omega + 2) * np.eye(n) - shift - shift.T
    k2 = np.kron(stiffness, np.eye(2))
    drift = np.block([
        [np.zeros((2*n, 2*n)), np.eye(2*n)],
        [-k2, -gamma*np.eye(2*n)]
    ])
    noises = []
    for x in range(n):
        b = np.zeros((4*n, 4*n))
        i = 2*n + 2*x
        b[i, i+1] = 1
        b[i+1, i] = -1
        noises.append(b)
    return stiffness, drift, noises, shift


def physical_covariance_action(c, drift, noises, gamma):
    """Literal planar Ito matrix equation, not scalar Fourier equations."""
    result = drift @ c + c @ drift.T
    for b in noises:
        result += 2*gamma * (b @ c @ b.T)
    return result


def symmetric_basis(d):
    answer = []
    for i in range(d):
        for j in range(i, d):
            c = np.zeros((d, d))
            if i == j:
                c[i, i] = 1
            else:
                c[i, j] = c[j, i] = 1 / math.sqrt(2)
            answer.append(c)
    return np.array(answer)


def evaluate_secular_only_for_comparison(z, n, k, omega, gamma):
    t = 2*np.pi*np.arange(n)/n
    xi = 2*np.pi*k/n
    phi = omega*omega + 2 - 2*np.cos(t)
    psi = omega*omega + 2 - 2*np.cos(xi-t)
    h = z*(z+gamma) + phi + psi
    g = (z+2*gamma)*(z+gamma) + phi + psi
    delta = g*h - 4*phi*psi
    return 1 - np.mean(2*gamma*(z+gamma)*h/delta), np.min(abs(delta))


def audit_physical_fourier(n=7, k=1, omega=1.0, gamma=0.7):
    stiffness, drift, noises, shift = physical_matrices(n, omega, gamma)
    basis = symmetric_basis(2*n)
    fourier = np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/math.sqrt(n)
    transform = la.block_diag(fourier, fourier)
    t = 2*np.pi*np.arange(n)/n
    frequencies = omega*omega + 2 - 2*np.cos(t)
    generator_columns = []
    translation_columns = []
    u = la.block_diag(shift, shift)
    max_fourier_error = 0.0
    max_scalar_leakage = 0.0
    for c in basis:
        full = np.kron(c, np.eye(2))
        full_derivative = physical_covariance_action(full, drift, noises, gamma)
        derivative = full_derivative[::2, ::2]
        max_scalar_leakage = max(max_scalar_leakage, float(np.max(
            abs(full_derivative - np.kron(derivative, np.eye(2))))))
        generator_columns.append(np.einsum("aij,ij->a", basis, derivative))
        translated = u @ c @ u.T
        translation_columns.append(np.einsum("aij,ij->a", basis, translated))
        f = transform @ c @ transform.T  # Bilinear, not a Hermitian covariance.
        observed = transform @ derivative @ transform.T
        predicted = np.zeros_like(observed)
        for ell in range(n):
            for m in range(n):
                a, b = f[ell, m], f[ell, n+m]
                cc, d = f[n+ell, m], f[n+ell, n+m]
                total = (ell+m) % n
                dbar = sum(f[n+j, n+(total-j) % n] for j in range(n))/n
                predicted[ell, m] = b + cc
                predicted[ell, n+m] = d - frequencies[m]*a - gamma*b
                predicted[n+ell, m] = d - frequencies[ell]*a - gamma*cc
                predicted[n+ell, n+m] = (
                    -frequencies[ell]*b - frequencies[m]*cc
                    -2*gamma*d + 2*gamma*dbar
                )
        max_fourier_error = max(max_fourier_error, float(np.max(abs(observed-predicted))))
    generator = np.column_stack(generator_columns)
    translation = np.column_stack(translation_columns)
    # Recover the entire physical translation block from the real-space matrix.
    powers = np.eye(len(basis), dtype=complex)
    projection = np.zeros_like(powers)
    for j in range(n):
        projection += np.exp(2j*np.pi*k*j/n)*powers/n
        powers = translation @ powers
    uu, ss, _ = la.svd(projection)
    rank = int(np.sum(ss > 0.5))
    block_basis = uu[:, :rank]
    block = block_basis.conj().T @ generator @ block_basis
    vals, vecs = la.eig(block)
    rows = []
    for i, z in enumerate(vals):
        c = np.einsum("a,aij->ij", block_basis @ vecs[:, i], basis)
        f = transform @ c @ transform.T
        dbar = sum(f[n+j, n+(k-j) % n] for j in range(n))/n
        residual, denominator_margin = evaluate_secular_only_for_comparison(
            z, n, k, omega, gamma)
        eligible = abs(dbar) > 1e-8 and denominator_margin > 1e-7
        rows.append({
            "eigenvalue": [float(z.real), float(z.imag)],
            "dbar_abs": float(abs(dbar)),
            "local_denominator_margin": float(denominator_margin),
            "secular_residual_abs": float(abs(residual)) if eligible else None,
            "eligible_for_secular_equation": bool(eligible),
        })
    rows.sort(key=lambda x: x["eigenvalue"][0], reverse=True)
    return {
        "N": n, "k": k, "omega0": omega, "gamma": gamma,
        "full_planar_covariance_dimension": 4*n,
        "symmetric_scalar_generator_dimension": len(basis),
        "physical_translation_block_dimension": rank,
        "physical_scalar_closure_error": max_scalar_leakage,
        "max_fourier_equation_error_all_symmetric_basis_matrices": max_fourier_error,
        "translation_commutator_norm": float(la.norm(generator@translation-translation@generator)),
        "projection_idempotence_norm": float(la.norm(projection@projection-projection)),
        "block_invariance_error": float(la.norm(generator@block_basis-block_basis@block)),
        "eigenvalues_and_posthoc_secular_residuals": rows,
    }


def relative_coordinate_matrix(n, omega, gamma):
    """Build from real-space stiffness on each index of a covariance.

    C_{x,y}=exp(i xi x)c_{y-x}. Left stiffness is
    c I-exp(i xi)shift_minus-exp(-i xi)shift_plus.
    Right stiffness is c I-shift_minus-shift_plus.
    The Ito injection is 2 gamma delta_{r,0} D_0.
    This is an analytic interpolation; a physical periodic block exists
    only when xi=2 pi k/N.
    """
    minus = np.roll(np.eye(n), 1, axis=1)
    plus = minus.T
    stiffness = (omega*omega+2)*np.eye(n)-minus-plus
    zero = np.zeros((n, n))
    ident = np.eye(n)
    injection = np.zeros((n, n))
    injection[0, 0] = 2*gamma
    matrix = np.block([
        [zero, ident, ident, zero],
        [-stiffness, -gamma*ident, zero, ident],
        [-stiffness, zero, -gamma*ident, ident],
        [zero, -stiffness, -stiffness, -2*gamma*ident+injection],
    ])
    delta = np.zeros(n)
    delta[0] = 1
    v0 = np.concatenate([la.solve(stiffness, delta), np.zeros(2*n), delta])
    left = np.concatenate([stiffness[0]/2, np.zeros(2*n), delta/2])
    assert abs(left@v0-1) < 1e-12
    assert la.norm(matrix@v0) < 1e-11
    assert la.norm(left@matrix) < 1e-11
    return matrix, v0, left, minus, plus


def independent_matrix_taylor(n, omega, gamma, order=6):
    matrix, v0, left, minus, plus = relative_coordinate_matrix(n, omega, gamma)
    d = 4*n
    bordered = np.zeros((d+1, d+1))
    bordered[:d, :d] = matrix
    bordered[:d, d] = -v0
    bordered[d, :d] = left
    factors = la.lu_factor(bordered)
    vectors = [v0]
    lambdas = [0.0]
    max_residual = 0.0
    # Expand in u=i xi, so all coefficient matrices and solves are real.
    def apply_coefficient(j, value):
        shift = (minus + (-1)**j*plus)/math.factorial(j)
        out = np.zeros(d)
        out[2*n:3*n] = shift @ value[:n]
        out[3*n:] = shift @ value[n:2*n]
        return out
    for degree in range(1, order+1):
        rhs = np.zeros(d)
        for j in range(1, degree+1):
            rhs -= apply_coefficient(j, vectors[degree-j])
        for j in range(1, degree):
            rhs += lambdas[j]*vectors[degree-j]
        target = np.r_[rhs, 0.0]
        solution = la.lu_solve(factors, target)
        max_residual = max(max_residual, float(la.norm(bordered@solution-target, np.inf)))
        vectors.append(solution[:d])
        lambdas.append(float(solution[d]))
    physical = [complex((1j)**j*lambdas[j]) for j in range(order+1)]
    return {
        "N": n, "omega0": omega, "gamma": gamma,
        "kappa": -physical[2].real,
        "B4": physical[4].real, "C6": physical[6].real,
        "max_odd_coefficient_abs": max(abs(physical[j]) for j in [1, 3, 5]),
        "max_bordered_recurrence_residual": max_residual,
    }


def high_precision_recurrence(n=16, digits=75):
    """Same bordered matrix recurrence in arbitrary precision, gamma=omega=1."""
    mp.mp.dps = digits
    nn, d = n, 4*n
    ident = mp.eye(n)
    minus = mp.zeros(n)
    for i in range(n):
        minus[i, (i-1) % n] = 1
    plus = minus.T
    stiffness = 3*ident-minus-plus
    matrix = mp.zeros(d)
    for i in range(n):
        for j in range(n):
            v = stiffness[i, j]
            matrix[n+i, j] = -v
            matrix[2*n+i, j] = -v
            matrix[3*n+i, n+j] = -v
            matrix[3*n+i, 2*n+j] = -v
        matrix[i, n+i] = matrix[i, 2*n+i] = 1
        matrix[n+i, n+i] = matrix[2*n+i, 2*n+i] = -1
        matrix[n+i, 3*n+i] = matrix[2*n+i, 3*n+i] = 1
        matrix[3*n+i, 3*n+i] = -2
    matrix[3*n, 3*n] += 2
    delta = mp.zeros(n, 1)
    delta[0] = 1
    a0 = mp.lu_solve(stiffness, delta)
    v0 = mp.zeros(d, 1)
    left = mp.zeros(1, d)
    for i in range(n):
        v0[i] = a0[i]
        left[0, i] = stiffness[0, i]/2
    v0[3*n] = 1
    left[0, 3*n] = mp.mpf("0.5")
    bordered = mp.zeros(d+1)
    for i in range(d):
        for j in range(d):
            bordered[i, j] = matrix[i, j]
        bordered[i, d] = -v0[i]
        bordered[d, i] = left[0, i]
    inverse = bordered**-1
    vectors, lambdas = [v0], [mp.mpf(0)]
    residuals = []
    for degree in range(1, 7):
        rhs = mp.zeros(d, 1)
        for j in range(1, degree+1):
            source = vectors[degree-j]
            for i in range(n):
                rhs[2*n+i] -= (source[(i-1) % n]+(-1)**j*source[(i+1) % n])/mp.factorial(j)
                rhs[3*n+i] -= (source[n+(i-1) % n]+(-1)**j*source[n+(i+1) % n])/mp.factorial(j)
        for j in range(1, degree):
            rhs += lambdas[j]*vectors[degree-j]
        target = mp.matrix([rhs[i] for i in range(d)]+[0])
        result = inverse*target
        vectors.append(mp.matrix([result[i] for i in range(d)]))
        lambdas.append(result[d])
        residuals.append(mp.norm(bordered*result-target, mp.inf))
    return {
        "N": n, "omega0": 1, "gamma": 1, "precision_decimal_digits": digits,
        "kappa": mp.nstr(lambdas[2], digits-8),
        "B4": mp.nstr(lambdas[4], digits-8),
        "C6": mp.nstr(-lambdas[6], digits-8),
        "max_recurrence_residual": mp.nstr(max(residuals), 8),
    }


def high_precision_physical_block(n=7, k=1, digits=75):
    """Physical Fourier block rebuilt at high precision, without root inversion.

    The one-component Ito injection below is the exact scalar-I_2 part
    of the literal planar noise action checked on every basis above.
    """
    mp.mp.dps = digits
    g = mp.mpf(7)/10
    stiffness = mp.zeros(n)
    for x in range(n):
        stiffness[x, x] += 3
        stiffness[x, (x-1) % n] -= 1
        stiffness[x, (x+1) % n] -= 1
    drift = mp.zeros(2*n)
    for x in range(n):
        drift[x, n+x] = 1
        drift[n+x, n+x] = -g
        for y in range(n):
            drift[n+x, y] = -stiffness[x, y]
    fourier = mp.zeros(2*n)
    inverse_fourier = mp.zeros(2*n)
    for channel in [0, 1]:
        for ell in range(n):
            for x in range(n):
                value = mp.exp(-2j*mp.pi*ell*x/n)/mp.sqrt(n)
                fourier[channel*n+ell, channel*n+x] = value
                inverse_fourier[channel*n+x, channel*n+ell] = mp.conj(value)
    pairs = [(a, b) for a in range(2*n) for b in range(a, 2*n)
             if (a % n+b % n) % n == k]
    block = mp.zeros(len(pairs))
    for column, (a, b) in enumerate(pairs):
        modal_covariance = mp.zeros(2*n)
        modal_covariance[a, b] = modal_covariance[b, a] = 1
        real_space_covariance = inverse_fourier*modal_covariance*inverse_fourier.T
        derivative = drift*real_space_covariance+real_space_covariance*drift.T
        for x in range(n):
            derivative[n+x, n+x] += 2*g*real_space_covariance[n+x, n+x]
        transformed = fourier*derivative*fourier.T
        for row, (aa, bb) in enumerate(pairs):
            block[row, column] = transformed[aa, bb]
    eigenvalues = mp.eig(block, left=False, right=False)
    eigenvalues = sorted(eigenvalues, key=lambda z: mp.re(z), reverse=True)
    residuals = []
    for z in eigenvalues:
        total = 0
        for ell in range(n):
            phi = 3-2*mp.cos(2*mp.pi*ell/n)
            psi = 3-2*mp.cos(2*mp.pi*(k-ell)/n)
            h = z*(z+g)+phi+psi
            gg = (z+2*g)*(z+g)+phi+psi
            total += 2*g*(z+g)*h/(gg*h-4*phi*psi)
        residuals.append(abs(1-total/n))
    return {
        "N": n, "k": k, "omega0": 1, "gamma": "0.7",
        "precision_decimal_digits": digits, "physical_block_dimension": len(pairs),
        "dominant_physical_eigenvalue_real": mp.nstr(mp.re(eigenvalues[0]), digits-8),
        "dominant_physical_eigenvalue_imaginary": mp.nstr(mp.im(eigenvalues[0]), 8),
        "dominant_physical_eigenvalue_secular_residual": mp.nstr(residuals[0], 8),
        "maximum_secular_residual_all_physical_block_eigenvalues": mp.nstr(max(residuals), 8),
    }


def claimed_closed_forms_for_posthoc_comparison(omega, gamma):
    """Never called by physical construction or Taylor recurrence."""
    c = omega*omega+2
    h = math.sqrt(c*c-4)
    r = 2/(c+h)
    kappa = r/(2*gamma)
    b = kappa/12+kappa*kappa*(gamma/(4*h)-1/(2*gamma))
    a4 = 19*r**4+22*r*r+4
    a2 = 60*r*(1-r*r)*(2*r*r+1)
    a0 = 180*r*r*(1-r*r)**2
    cc = -r*(a4*gamma**4-a2*gamma*gamma+a0)/(2880*gamma**5*(1-r*r)**2)
    return {"kappa": kappa, "B4": b, "C6": cc}


def main():
    output = Path(__file__).with_suffix(".json")
    physical = audit_physical_fourier()
    r = 2/(3+math.sqrt(5))
    gamma_b = math.sqrt(6*r*math.sqrt(5)/(2*math.sqrt(5)+3*r))
    parameters = [
        (1.0, 0.5), (1.0, 1.0), (1.0, 1.25), (1.0, 2.0),
        (1.0, gamma_b), (0.7, 0.8), (1.3, 0.7), (2.0, 1.0)
    ]
    rows = []
    for omega, gamma in parameters:
        for n in [16, 32, 64, 96, 128]:
            row = independent_matrix_taylor(n, omega, gamma)
            target = claimed_closed_forms_for_posthoc_comparison(omega, gamma)
            row["posthoc_claimed_coefficients"] = target
            row["absolute_posthoc_differences"] = {
                key: abs(row[key]-target[key]) for key in target
            }
            rows.append(row)
        print("matrix recurrence completed:", omega, gamma, flush=True)
    high = high_precision_recurrence()
    physical_high = high_precision_physical_block()
    double = independent_matrix_taylor(high["N"], 1, 1)
    high["double_vs_high_precision_abs_error"] = {
        key: mp.nstr(abs(mp.mpf(double[key])-mp.mpf(high[key])), 12)
        for key in ["kappa", "B4", "C6"]
    }
    result = {
        "method": __doc__,
        "inputs_to_independent_computation": [
            "physical Ito drift and site rotation matrices",
            "periodic nearest-neighbor stiffness",
            "uniform O(2)-scalar covariance embedding",
            "relative-coordinate Bloch interpolation",
            "bordered eigenvalue-Taylor recurrence"
        ],
        "not_inputs": [
            "the target quartic/sixth-order coefficients",
            "the manuscript secular equation for root computation",
            "fitted pole values or trajectory samples"
        ],
        "physical_generator_audit": physical,
        "matrix_taylor_coefficients_and_posthoc_comparison": rows,
        "arbitrary_precision_finite_volume_audit": high,
        "arbitrary_precision_physical_block_audit": physical_high,
        "limitations": [
            "The large-N comparison uses floating-point matrix arithmetic, not interval enclosures.",
            "The arbitrary-precision check validates finite-volume recurrence precision at N=16.",
            "Analytic interpolation between physical discrete wave numbers is specified explicitly.",
            "Uniform deterministic-to-stochastic transfer and deterministic poles are not tested."
        ]
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", output, flush=True)
    print("max physical Fourier equation error:", physical[
        "max_fourier_equation_error_all_symmetric_basis_matrices"], flush=True)
    eligible = [row["secular_residual_abs"] for row in physical[
        "eigenvalues_and_posthoc_secular_residuals"] if row["eligible_for_secular_equation"]]
    print("max eligible secular residual:", max(eligible), flush=True)
    for key in ["kappa", "B4", "C6"]:
        print("max N=128 posthoc", key, "difference:",
              max(row["absolute_posthoc_differences"][key] for row in rows if row["N"] == 128),
              flush=True)
    print("high precision:", high, flush=True)
    print("high precision physical block:", physical_high, flush=True)


if __name__ == "__main__":
    main()
