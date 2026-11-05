"""Exact symbolic normalization checks; no trajectories or sampling.

Run with Python and SymPy. Writes exact_normalization_check.json beside this file.
The dimension-independent bound is proved analytically in the manuscript;
this finite matrix calculation independently checks its transport conventions.
"""
from pathlib import Path
import json
import sympy as sp

N = 3
shift = sp.zeros(N)
for x in range(N):
    shift[x, (x + 1) % N] = 1
K = 3 * sp.eye(N) - shift - shift.T
C = (shift - shift.T) / 2
K2 = sp.kronecker_product(K, sp.eye(2))
C2 = sp.kronecker_product(C, sp.eye(2))
G = sp.diag(K2, sp.eye(2 * N))
cov = G.inv()  # beta=1
current = sp.BlockMatrix([
    [sp.zeros(2 * N), C2 / 2],
    [C2.T / 2, sp.zeros(2 * N)],
]).as_explicit()
var_current = sp.simplify(2 * sp.trace((current * cov) ** 2))
var_energy = sp.simplify(2 * sp.trace(((G / 2) * cov) ** 2))
Dlat = sp.simplify(sp.Rational(2, N) * sp.trace(C * K.inv() * C.T))
gamma, sigma, theta = sp.symbols('gamma sigma theta', real=True, nonzero=True)
kappa = sp.simplify(var_current / (gamma * var_energy))
assert var_current == sp.Rational(3, 4)
assert var_energy == 6
assert Dlat == sp.Rational(1, 4)
assert kappa == Dlat / (2 * gamma) == 1 / (8 * gamma)

c0 = sp.cos(sp.pi * theta)
c1 = sp.cos(sp.pi * theta + sp.pi)
Pc = sp.simplify((c0 + c1) / 2)
Pc2 = sp.trigsimp((c0**2 + c1**2) / 2)
assert Pc == 0
assert sp.trigsimp(Pc2 - (1 + sp.cos(2 * sp.pi * theta)) / 2) == 0
Gamma = sigma**2 / 2
matched_gamma = Gamma / 2
matched_kappa = sp.simplify(kappa.subs(gamma, matched_gamma))
assert matched_kappa == Dlat / Gamma

result = {
    'kind': 'exact symbolic identities; no stochastic simulation',
    'N': N, 'omega_0_squared': 1, 'beta': 1,
    'active_current_variance': str(var_current),
    'total_energy_variance': str(var_energy),
    'D_N_lat': str(Dlat),
    'reference_diffusivity': str(kappa),
    'P_cos': str(Pc), 'P_cos_squared': str(Pc2),
    'complete_fast_variance': str(Gamma),
    'matched_rotational_rate': str(matched_gamma),
    'matched_diffusivity': str(matched_kappa),
    'status': 'all exact assertions passed',
}
destination = Path(__file__).with_suffix('.json')
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
