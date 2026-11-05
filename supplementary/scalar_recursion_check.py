"""Check the local scalar secular recursion; no trajectories or sampling.

The three local coefficient identities are checked exactly in independent
symbols. A periodic quadrature then solves the triangular averaged recursion
for z2, z4, z6, without supplying the closed coefficients as inputs.
Numerical discrepancies are comparisons, not rigorous quadrature error bounds.
"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
from sixth_order_scalar_benchmark import coefficients

q = sp.symbols('q')
a, gamma = sp.symbols('a gamma', nonzero=True)
x, u, z2, z4, z6 = sp.symbols('x u z2 z4 z6')
z = z2*q + z4*q**2 + z6*q**3
rz = z*(z+gamma)
S = 2*a + x*q/2 - x*q**2/96
d2 = 4*u*q - u*q**2/3 + u*q**3/90
den = 2*gamma*(z+gamma)*(S+rz)
lam1 = z2/gamma + (x/2+gamma*z2)/(2*a)
lam2 = z4/gamma + (-x/96+gamma*z4+z2**2)/(2*a) + z2*(x/2+gamma*z2)/(2*gamma*a)
n1 = 4*u
n2 = -u/3 - gamma**2*z2**2
n3 = u/90 - 2*gamma**2*z2*z4 - 2*gamma*z2**3
den_poly = sp.Poly(sp.expand(den), q)
numerator_poly = sp.Poly(sp.expand(d2-rz**2), q)
local_coefficients = [
    z2/gamma+n1/(4*gamma**2*a),
    z4/gamma+(n2-n1*lam1)/(4*gamma**2*a),
    z6/gamma+(n3-n2*lam1+n1*(lam1**2-lam2))/(4*gamma**2*a),
]
# Check by multiplying the proposed local series by the exact denominator.
local_series = sum(coef*q**power for power, coef in enumerate(local_coefficients, 1))
identity = sp.Poly(sp.expand(den*(local_series-z/gamma)-(d2-rz**2)), q)
residuals = {
    'denominator_q1': sp.factor(den_poly.coeff_monomial(q)/(4*gamma**2*a)-lam1),
    'denominator_q2': sp.factor(den_poly.coeff_monomial(q**2)/(4*gamma**2*a)-lam2),
    'numerator_q1': sp.factor(numerator_poly.coeff_monomial(q)-n1),
    'numerator_q2': sp.factor(numerator_poly.coeff_monomial(q**2)-n2),
    'numerator_q3': sp.factor(numerator_poly.coeff_monomial(q**3)-n3),
}
for power in (1, 2, 3):
    residuals[f'local_E_q{power}'] = sp.factor(identity.coeff_monomial(q**power))
assert all(value == 0 for value in residuals.values()), residuals

def averaged_recursion(omega0, gamma, count=100_000):
    y = 2*np.pi*(np.arange(count)+0.5)/count
    x = np.cos(y)
    u = np.sin(y)**2
    a = omega0**2+2-2*x
    z2 = -np.mean(u/a)/gamma
    n1 = 4*u
    n2 = -u/3-gamma**2*z2**2
    lam1 = z2/gamma+(x/2+gamma*z2)/(2*a)
    E2 = z2/gamma+n1/(4*gamma**2*a)
    R4 = (n2-n1*lam1)/(4*gamma**2*a)
    z4 = gamma*np.mean(E2**2-R4)
    E4 = z4/gamma+R4
    lam2 = z4/gamma+(-x/96+gamma*z4+z2**2)/(2*a)+z2*(x/2+gamma*z2)/(2*gamma*a)
    n3 = u/90-2*gamma**2*z2*z4-2*gamma*z2**3
    R6 = (n3-n2*lam1+n1*(lam1**2-lam2))/(4*gamma**2*a)
    z6 = gamma*np.mean(2*E2*E4-E2**3-R6)
    return np.array([z2, z4, z6])

checks = []
for omega0 in (0.1, 1.0, 2.0, 3.0):
    for rate in (0.5, 1.0, 2.0):
        actual = averaged_recursion(omega0, rate)
        kappa, B4, C6 = coefficients(omega0, rate)
        expected = np.array([-kappa, B4, C6])
        errors = np.abs(actual-expected)
        assert np.max(errors) < 1e-11
        checks.append({'omega0': omega0, 'gamma': rate,
                       'recursion': actual.tolist(), 'closed': expected.tolist(),
                       'absolute_discrepancies': errors.tolist()})
report = {
    'scope': 'exact local series identities plus periodic-quadrature comparisons; no trajectory sampling',
    'symbolic_residuals': {name: str(value) for name, value in residuals.items()},
    'quadrature_points': 100_000,
    'cases': checks,
    'max_absolute_discrepancy': max(max(item['absolute_discrepancies']) for item in checks),
    'qualification': 'No rigorous quadrature error estimate is inferred from these discrepancies.',
}
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'exact_identities': len(residuals), 'all_zero': True, 'quadrature_cases': len(checks),
                  'max_absolute_discrepancy': report['max_absolute_discrepancy']}))
