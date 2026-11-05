"""Exact algebraic checks of the stochastic scalar dispersion identities.

This script uses only SymPy rational/algebraic arithmetic.  It checks
consequences of the manuscript's already derived fourth- and sixth-order
coefficients; it does not independently derive the secular equation, simulate
trajectories, sample data, or establish deterministic pole transfer.

Run from any directory with Python and SymPy installed.  The JSON output is
written beside this file.  The domain is 0 < r < 1 and gamma > 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    r, gamma = sp.symbols("r gamma", positive=True)
    beta = sp.symbols("beta", real=True)
    q = sp.symbols("q")
    x = r**2
    h = (1 - x) / r
    kappa = r / (2 * gamma)
    b4_raw = kappa / 12 + kappa**2 * (gamma / (4 * h) - 1 / (2 * gamma))
    a4 = 19 * r**4 + 22 * r**2 + 4
    a2 = 60 * r * (1 - r**2) * (2 * r**2 + 1)
    a0 = 180 * r**2 * (1 - r**2)**2
    c6_raw = -r * (a4 * gamma**4 - a2 * gamma**2 + a0) / (
        2880 * gamma**5 * (1 - r**2)**2
    )
    b4 = sp.factor(b4_raw / kappa)
    b6 = sp.factor(c6_raw / kappa)
    b_inf = (2 + x) / (24 * (1 - x))
    b_minus = (1 - 1 / sp.sqrt(5)) / 24
    b_plus = (1 + 1 / sp.sqrt(5)) / 24
    parabola = -sp.Rational(1, 360) + beta / 6 - 2 * beta**2

    gamma_b_sq = 6 * r * (1 - x) / (2 + x)
    gamma_m_sq = 2 * h
    gamma_minus_sq = 6 * r * (1 - x) * (
        5 * (2 * x + 1) - sp.sqrt(5) * (1 - x)
    ) / a4
    gamma_plus_sq = 6 * r * (1 - x) * (
        5 * (2 * x + 1) + sp.sqrt(5) * (1 - x)
    ) / a4
    gamma_b = sp.sqrt(gamma_b_sq)
    shift = gamma_b**3 / (180 * r)
    slope_at_b = r / (2 * gamma_b**3)

    def at_square(expression: sp.Expr, square: sp.Expr) -> sp.Expr:
        return sp.factor(expression.subs(gamma**2, square))

    residuals: dict[str, sp.Expr] = {}

    def check(name: str, expression: sp.Expr) -> None:
        residual = sp.factor(sp.simplify(expression))
        if residual != 0:
            raise AssertionError(f"{name}: nonzero exact residual {residual}")
        residuals[name] = residual

    check("universal_normalized_relation", b6 + sp.Rational(1, 360) - b4 / 6 + 2 * b4**2)
    check("normalized_quartic_monotonic_derivative", sp.diff(b4, gamma) - r / (2 * gamma**3))
    check("normalized_quartic_as_affine_inverse_rate", b4 - b_inf + r / (4 * gamma**2))
    check("quartic_at_gamma_B", at_square(b4, gamma_b_sq))
    check("sixth_at_gamma_B", at_square(b6, gamma_b_sq) + sp.Rational(1, 360))
    check("quartic_at_gamma_M", at_square(b4, gamma_m_sq) - sp.Rational(1, 12))
    check("sixth_at_gamma_M", at_square(b6, gamma_m_sq) + sp.Rational(1, 360))
    check("parabola_at_b_minus", parabola.subs(beta, b_minus))
    check("parabola_at_b_plus", parabola.subs(beta, b_plus))
    check("gamma_minus_corresponds_to_b_minus", at_square(b4, gamma_minus_sq) - b_minus)
    check("gamma_plus_corresponds_to_b_plus", at_square(b4, gamma_plus_sq) - b_plus)
    check("normalized_sixth_peak_identity", parabola - (sp.Rational(1, 1440) - 2 * (beta - sp.Rational(1, 24))**2))
    check("finite_wavelength_shift_times_slope", shift * slope_at_b - sp.Rational(1, 360))
    check("shift_equals_implicit_function_derivative", shift + at_square(b6, gamma_b_sq) / slope_at_b)

    # Formal inversion of -mu/(kappa*xi^2), with q=xi^2.
    lifetime_polynomial = 1 + b4 * q + (b4**2 + b6) * q**2
    inverse_series = sp.series(1 / (1 - b4 * q - b6 * q**2), q, 0, 3).removeO()
    for power in range(3):
        check(f"lifetime_inverse_series_q{power}", sp.expand(inverse_series - lifetime_polynomial).coeff(q, power))
    check("lifetime_second_relative_coefficient", b4**2 + b6 - (-sp.Rational(1, 360) + b4 / 6 - b4**2))
    check("lifetime_at_gamma_B_first_relative_coefficient", at_square(b4, gamma_b_sq))
    check("lifetime_at_gamma_B_second_relative_coefficient", at_square(b4**2 + b6, gamma_b_sq) + sp.Rational(1, 360))
    check("lifetime_at_gamma_M_second_relative_coefficient", at_square(b4**2 + b6, gamma_m_sq) - sp.Rational(1, 240))

    # Positive factor forms, used together with 0 < r < 1.
    s = r * (1 - x)
    lower_gap_factor = 6 * s * (1 - x) * (
        (9 - sp.sqrt(5)) * x + 6 - 2 * sp.sqrt(5)
    ) / (a4 * (2 + x))
    upper_gap_factor = 2 * s * (1 - x) * (
        (11 - 3 * sp.sqrt(5)) * x + 4
    ) / (x * a4)
    middle_gap_factor = 12 * sp.sqrt(5) * s * (1 - x) / a4
    check("gamma_minus_squared_minus_gamma_B_squared_factor", gamma_minus_sq - gamma_b_sq - lower_gap_factor)
    check("gamma_plus_squared_minus_gamma_minus_squared_factor", gamma_plus_sq - gamma_minus_sq - middle_gap_factor)
    check("gamma_M_squared_minus_gamma_plus_squared_factor", gamma_m_sq - gamma_plus_sq - upper_gap_factor)
    check("b_infinity_minus_one_twelfth_factor", b_inf - sp.Rational(1, 12) - x / (8 * (1 - x)))
    check("ordered_b_roots_gap", b_plus - b_minus - 1 / (12 * sp.sqrt(5)))
    check("upper_b_root_below_one_twelfth", sp.Rational(1, 12) - b_plus - b_minus)

    r_one = (3 - sp.sqrt(5)) / 2  # omega_0=1.
    curves = {
        "gamma_B": gamma_b_sq,
        "gamma_minus": gamma_minus_sq,
        "gamma_plus": gamma_plus_sq,
        "gamma_M": gamma_m_sq,
    }
    numerical = {
        "omega0": "1",
        "r": str(sp.N(r_one, 24)),
        **{name: str(sp.N(sp.sqrt(square.subs(r, r_one)), 24)) for name, square in curves.items()},
        "gamma_cont_xi_squared_shift_coefficient": str(sp.N(shift.subs(r, r_one), 24)),
    }
    result = {
        "method": "Exact SymPy algebra of the established stochastic scalar coefficient formulas; no simulation or sampling.",
        "domain": "0 < r < 1; gamma > 0; fixed positive pinning and a local positive-gamma neighborhood for the analytic shift.",
        "definitions": {
            "r": "D=(c-sqrt(c^2-4))/2, c=omega0^2+2",
            "h": str(h),
            "kappa": str(kappa),
            "B_sc": str(b4_raw),
            "C_sc": str(c6_raw),
            "b4": str(b4),
            "b6": str(b6),
            "b4_infinity": str(b_inf),
            "b4_minus": str(b_minus),
            "b4_plus": str(b_plus),
            "gamma_B_squared": str(gamma_b_sq),
            "gamma_minus_squared": str(gamma_minus_sq),
            "gamma_plus_squared": str(gamma_plus_sq),
            "gamma_M_squared": str(gamma_m_sq),
            "gamma_cont_xi_squared_shift_coefficient": str(shift),
            "lifetime_relative_series": "tau_sc/tau_heat = 1 + b4*xi^2 + (b4^2+b6)*xi^4 + O(xi^6)",
        },
        "exact_residuals": {name: str(value) for name, value in residuals.items()},
        "all_exact_residuals_zero": all(value == 0 for value in residuals.values()),
        "ordering_positive_factors": {
            "b4_minus": str(b_minus),
            "b4_plus_minus_b4_minus": str(1 / (12 * sp.sqrt(5))),
            "one_twelfth_minus_b4_plus": str(b_minus),
            "b4_infinity_minus_one_twelfth": str(x / (8 * (1 - x))),
            "gamma_minus_squared_minus_gamma_B_squared": str(lower_gap_factor),
            "gamma_plus_squared_minus_gamma_minus_squared": str(middle_gap_factor),
            "gamma_M_squared_minus_gamma_plus_squared": str(upper_gap_factor),
            "reason": "Each factor is strictly positive for 0<r<1; 6-2*sqrt(5), 9-sqrt(5), and 11-3*sqrt(5) are positive.",
            "conclusion": "gamma_B < gamma_minus < gamma_plus < gamma_M",
        },
        "omega0_one_values": numerical,
        "analytic_shift_scope": {
            "normalized_function": "H(q,gamma)=[mu_sc(xi,gamma)+kappa(gamma)*xi^2]/[kappa(gamma)*xi^4], q=xi^2; extend analytically at q=0.",
            "implicit_data": "H(0,gamma_B)=0; partial_gamma H=r/(2*gamma_B^3)>0; partial_q H=-1/360.",
            "conclusion": "There is a unique local analytic gamma_cont(q) near (q,gamma)=(0,gamma_B), with gamma_cont(q)=gamma_B+gamma_B^3*q/(180*r)+O(q^2).",
            "uniformity": "The established secular-function bounds can be chosen uniformly on a compact positive-gamma neighborhood of gamma_B at fixed r.",
            "faster_interval": "For sufficiently small real q>0 and gamma_B<gamma<gamma_cont(q), B_sc>0 but H(q,gamma)<0, so the scalar mode damps faster than continuum heat with the same kappa(gamma).",
            "not_asserted": "Global uniqueness, finite-N exact coefficient identities, higher-order deterministic pole transfer, or equality to discrete heat at all orders.",
        },
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: {len(residuals)} exact residuals are zero.")
    print("Ordering: gamma_B < gamma_minus < gamma_plus < gamma_M for every 0<r<1.")
    print("At gamma_B and gamma_M: C_sc/kappa = -1/360.")
    print("At gamma_M: the stochastic scalar pole matches discrete heat through xi^6.")
    print("omega0=1:", json.dumps(numerical))
    print("Saved:", output)


if __name__ == "__main__":
    main()
