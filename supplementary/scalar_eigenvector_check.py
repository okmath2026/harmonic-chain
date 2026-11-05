"""Exact local algebra behind the physical scalar eigenvector reconstruction."""
from pathlib import Path
import json
import sympy as sp

z, gamma, phi_l, phi_m = sp.symbols('z gamma phi_l phi_m')
h = z * (z + gamma) + phi_l + phi_m
delta = ((z + 2 * gamma) * (z + gamma) + phi_l + phi_m) * h - 4 * phi_l * phi_m
amplitude_a = 4 * gamma * (z + gamma) / delta
amplitude_b = 2 * gamma * (z * (z + gamma) + phi_l - phi_m) / delta
amplitude_c = 2 * gamma * (z * (z + gamma) + phi_m - phi_l) / delta
amplitude_d = 2 * gamma * (z + gamma) * h / delta
residuals = [
    z * amplitude_a - amplitude_b - amplitude_c,
    (z + gamma) * amplitude_b - amplitude_d + phi_m * amplitude_a,
    (z + gamma) * amplitude_c - amplitude_d + phi_l * amplitude_a,
    (z + 2 * gamma) * amplitude_d + phi_l * amplitude_b + phi_m * amplitude_c - 2 * gamma,
]
local = sp.Matrix([
    [z, -1, -1, 0],
    [phi_m, z + gamma, 0, -1],
    [phi_l, 0, z + gamma, -1],
    [0, phi_l, phi_m, z + 2 * gamma],
])
result = {
    'normalization': 'd_k=1; Delta!=0; secular equation supplies mean(D_l)=1',
    'four_equation_residuals': [str(sp.factor(value)) for value in residuals],
    'local_determinant_minus_Delta': str(sp.factor(local.det() - delta)),
}
assert result['four_equation_residuals'] == ['0'] * 4
assert result['local_determinant_minus_Delta'] == '0'
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
