"""Normalized fourth/sixth scalar coefficients of the specified harmonic chain.

Both panels evaluate the closed analytic formulas directly. There are no fitted
coefficients, sampled trajectories, or simulation data. The right-hand relation
is independent of pinning within this nearest-neighbour chain.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = "fig_scalar_coefficients.pdf"
W0 = 1.0
c = W0**2 + 2.0
h = np.sqrt(c**2 - 4.0)
D = (c - h) / 2.0
A4 = 19.0 * D**4 + 22.0 * D**2 + 4.0
A2 = 60.0 * D * (1.0 - D**2) * (2.0 * D**2 + 1.0)
A0 = 180.0 * D**2 * (1.0 - D**2)**2


def kappa(gamma):
    return D / (2.0 * np.asarray(gamma))


def Bsc(gamma):
    gamma = np.asarray(gamma)
    kap = kappa(gamma)
    return kap / 12.0 + kap**2 * (gamma / (4.0 * h) - 1.0 / (2.0 * gamma))


def Csc(gamma):
    gamma = np.asarray(gamma)
    return -D * (A4 * gamma**4 - A2 * gamma**2 + A0) / (
        2880.0 * gamma**5 * (1.0 - D**2)**2
    )


def b4(gamma):
    return Bsc(gamma) / kappa(gamma)


def coefficient_relation(value):
    return -1.0 / 360.0 + value / 6.0 - 2.0 * value**2


b4_minus = (1.0 - 1.0 / np.sqrt(5.0)) / 24.0
b4_plus = (1.0 + 1.0 / np.sqrt(5.0)) / 24.0
gB = np.sqrt(6.0 * D * h / (2.0 * h + 3.0 * D))
gm = np.sqrt(6.0 * D * (1.0 - D**2) * (
    5.0 * (2.0 * D**2 + 1.0) - np.sqrt(5.0) * (1.0 - D**2)
) / A4)
gp = np.sqrt(6.0 * D * (1.0 - D**2) * (
    5.0 * (2.0 * D**2 + 1.0) + np.sqrt(5.0) * (1.0 - D**2)
) / A4)
gM = np.sqrt(2.0 * h)

gammas = np.linspace(0.6, 2.6, 3000)
normalized_fourth = b4(gammas)
normalized_sixth = Csc(gammas) / kappa(gammas)
identity_error = np.max(np.abs(
    normalized_sixth - coefficient_relation(normalized_fourth)
))
levels = np.array([0.0, b4_minus, b4_plus, 1.0 / 12.0])
critical_rates = np.array([gB, gm, gp, gM])
assert np.all(np.diff(normalized_fourth) > 0.0)
assert np.all(np.diff(critical_rates) > 0.0)
assert identity_error < 5e-15
assert np.max(np.abs(b4(critical_rates) - levels)) < 5e-15

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 10.0,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
negative_fill, positive_fill = "#f8e8e4", "#e6f2eb"
point_colors = ["#265d96", "#168268", "#c56b18", "#765190"]
markers = ["o", "^", "s", "D"]

# (a) Pinning-fixed monotone parameterization of the normalized quartic term.
ax = axes[0]
ax.axvspan(0.6, gm, color=negative_fill, zorder=0)
ax.axvspan(gm, gp, color=positive_fill, zorder=0)
ax.axvspan(gp, 2.6, color=negative_fill, zorder=0)
for y in levels:
    ax.axhline(y, color="#8c9299", lw=0.6, ls=(0, (3, 3)), zorder=1)
ax.plot(gammas, normalized_fourth, color="#253e59", lw=1.8, zorder=2)
for rate, level, color, marker in zip(critical_rates, levels, point_colors, markers):
    ax.plot(rate, level, marker=marker, ms=5.5, mfc="white", mec=color,
            mew=1.35, ls="none", zorder=4)

labels = [r"$\gamma_B$", r"$\gamma_-$", r"$\gamma_+$", r"$\gamma_M$"]
text_positions = [(0.79, -0.025), (1.00, 0.050), (1.57, 0.040), (2.22, 0.064)]
for rate, level, label, text_pos, color in zip(
        critical_rates, levels, labels, text_positions, point_colors):
    ax.annotate(label, xy=(rate, level), xytext=text_pos, color=color,
                fontsize=10.0, ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65})

for x, text, color in [
        (0.79, r"$b_6<0$", "#96564a"),
        ((gm + gp) / 2.0, r"$b_6>0$", "#276e50"),
        (2.08, r"$b_6<0$", "#96564a")]:
    ax.text(x, -0.145, text, ha="center", va="center", color=color, fontsize=9.0)

ax.set(xlim=(0.6, 2.6), ylim=(-0.17, 0.103), xlabel=r"Rotation rate $\gamma$",
       ylabel=r"$b_4=\mathcal{B}_{\mathrm{sc}}/\kappa_{\mathrm{eff}}$")
ax.set_yticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10])
ax.set_title(r"(a) Monotone quartic correction, $\omega_0=1$", pad=10)

# Right-edge labels identify the four horizontal reference levels.
for y, label in zip(levels, [r"$0$", r"$b_{4,-}$", r"$b_{4,+}$", r"$1/12$"]):
    ax.text(1.018, y, label, transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8.6, color="#4b5563")

# (b) The same four points lie on one pinning-independent coefficient curve.
ax = axes[1]
values = np.linspace(-0.02, 0.10, 2400)
ax.axvspan(-0.02, b4_minus, color=negative_fill, zorder=0)
ax.axvspan(b4_minus, b4_plus, color=positive_fill, zorder=0)
ax.axvspan(b4_plus, 0.10, color=negative_fill, zorder=0)
ax.axhline(0.0, color="#777e86", lw=0.8, zorder=1)
ax.axhline(-1.0 / 360.0, color="#8c9299", lw=0.6, ls=(0, (3, 3)), zorder=1)
ax.plot(values, coefficient_relation(values), color="#253e59", lw=1.8, zorder=2)
for level, color, marker in zip(levels, point_colors, markers):
    ax.plot(level, coefficient_relation(level), marker=marker, ms=5.5,
            mfc="white", mec=color, mew=1.35, ls="none", zorder=4)

point_text = [r"$b_4=0$", r"$b_{4,-}$", r"$b_{4,+}$", r"$b_4=1/12$"]
point_text_positions = [(-0.008, -0.00345), (0.015, 0.00065),
                        (0.067, 0.00065), (0.077, -0.00345)]
for level, text, position, color in zip(levels, point_text, point_text_positions, point_colors):
    ax.annotate(text, xy=(level, coefficient_relation(level)), xytext=position,
                color=color, fontsize=9.0, ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65})
ax.text(0.040, -0.00510, r"$b_6=-1/360+b_4/6-2b_4^2$",
        ha="center", va="center", fontsize=10.0)
ax.text(0.040, -0.00665, "Pinning independent within this chain",
        ha="center", va="center", fontsize=8.3, color="#4b5563")
ax.set(xlim=(-0.02, 0.10), ylim=(-0.0076, 0.00145),
       xlabel=r"$b_4=\mathcal{B}_{\mathrm{sc}}/\kappa_{\mathrm{eff}}$",
       ylabel=r"$b_6=\mathcal{C}_{\mathrm{sc}}/\kappa_{\mathrm{eff}}$")
ax.set_xticks([-0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax.set_title("(b) Normalized coefficient relation", pad=10)

for ax in axes:
    ax.tick_params(width=0.7, length=3.0)
    ax.spines[["top", "right"]].set_visible(False)

fig.subplots_adjust(left=0.080, right=0.985, bottom=0.170, top=0.865, wspace=0.38)
fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("wrote", OUT)
print(f"D={D:.12f}; gamma_B={gB:.12f}; gamma_-={gm:.12f}; "
      f"gamma_+={gp:.12f}; gamma_M={gM:.12f}")
print(f"b4_-={b4_minus:.12f}; b4_+={b4_plus:.12f}")
print("max normalized-coefficient identity discrepancy:", f"{identity_error:.3e}")
for name, rate, level in zip(["B", "-", "+", "M"], critical_rates, levels):
    print(f"point {name}: gamma={rate:.12f}; b4={level:.12f}; "
          f"b6={float(Csc(rate)/kappa(rate)):.12f}")
