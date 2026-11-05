"""Sixth-order scalar dispersion benchmark for the effective covariance sector.

The slow branch is mu(xi) = -kappa*xi**2 + B4*xi**4 + C6*xi**6 + O(xi**8).
All formulas use phi(t)=c-2*cos(t), c=omega0**2+2, gamma>0.
"""
from __future__ import annotations
import numpy as np


def coefficients(omega0: float, gamma: float) -> tuple[float, float, float]:
    """Return (kappa, B4, C6) from the closed formulas."""
    c = omega0 * omega0 + 2.0
    h = np.sqrt(c * c - 4.0)
    r = (c - h) / 2.0  # r equals the infinite-volume D, 0<r<1
    kappa = r / (2.0 * gamma)
    B4 = kappa / 12.0 + kappa * kappa * (gamma / (4.0 * h) - 1.0 / (2.0 * gamma))
    p4 = 19.0 * r**4 + 22.0 * r**2 + 4.0
    p2 = 60.0 * r * (1.0 - r**2) * (2.0 * r**2 + 1.0)
    p0 = 180.0 * r**2 * (1.0 - r**2) ** 2
    C6 = -r * (p4 * gamma**4 - p2 * gamma**2 + p0) / (
        2880.0 * gamma**5 * (1.0 - r**2) ** 2
    )
    return kappa, B4, C6


def c6_integral(omega0: float, gamma: float, n: int = 200_000) -> float:
    """Compute C6 from the sixth-order secular integral."""
    c = omega0 * omega0 + 2.0
    h = np.sqrt(c * c - 4.0)
    I = (c - h) / 4.0
    z2 = -I / gamma
    kappa = I / gamma
    z4 = kappa / 12.0 + kappa**2 * (gamma / (4.0 * h) - 1.0 / (2.0 * gamma))
    t = 2.0 * np.pi * (np.arange(n) + 0.5) / n
    x = np.cos(t)
    u = 1.0 - x * x
    a = c - 2.0 * x
    r2 = gamma * z2
    r4 = gamma * z4 + z2 * z2
    s2 = x / 2.0
    s4 = -x / 96.0
    e1 = (s2 + r2) / (2.0 * a)
    e2 = (s4 + r4) / (2.0 * a)
    w1 = -z2 / gamma - e1
    w2 = z2 * z2 / gamma**2 - z4 / gamma + e1 * e1 - e2 + (z2 / gamma) * e1
    d1, d2, d3 = 4.0 * u, -u / 3.0, u / 90.0
    t4 = (d2 + d1 * w1) / (4.0 * a * gamma**2)
    t6 = (d3 + d2 * w1 + d1 * w2) / (4.0 * a * gamma**2)
    v4 = r2 * r2 / (4.0 * a * gamma**2)
    v6 = (2.0 * r2 * r4 + r2 * r2 * w1) / (4.0 * a * gamma**2)
    e_2 = z2 / gamma + d1 / (4.0 * a * gamma**2)
    e_4 = z4 / gamma + t4 - v4
    residual6 = t6 - v6 - 2.0 * e_2 * e_4 + e_2**3
    return float(-gamma * residual6.mean())


def zero_curves(omega0: float) -> tuple[float, float]:
    """Return gamma_- and gamma_+ where C6=0."""
    c = omega0 * omega0 + 2.0
    h = np.sqrt(c * c - 4.0)
    r = (c - h) / 2.0
    den = 19.0 * r**4 + 22.0 * r**2 + 4.0
    base = 6.0 * r * (1.0 - r**2)
    gm2 = base * (5.0 * (2.0 * r**2 + 1.0) - np.sqrt(5.0) * (1.0 - r**2)) / den
    gp2 = base * (5.0 * (2.0 * r**2 + 1.0) + np.sqrt(5.0) * (1.0 - r**2)) / den
    return float(np.sqrt(gm2)), float(np.sqrt(gp2))


if __name__ == "__main__":
    for omega0 in (0.1, 1.0, 2.0, 3.0):
        for gamma in (0.5, 1.0, 2.0):
            kappa, b4, c6 = coefficients(omega0, gamma)
            check = c6_integral(omega0, gamma, n=100_000)
            print(f"omega0={omega0:g}, gamma={gamma:g}, "
                  f"kappa={kappa:.12g}, B4={b4:.12g}, "
                  f"C6={c6:.12g}, integral_error={check-c6:+.3e}")
        print("C6 zero curves gamma_- gamma_+ =", zero_curves(omega0))
