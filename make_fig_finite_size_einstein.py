"""fig:finite-size-einstein -- D_N^lat vs N against the exact closed form.

Finite-volume arithmetic check of an identity proved in the text; not a
Monte Carlo or trajectory-sampling measurement.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpmath import mp

OUT = "fig_finite_size_einstein.pdf"
W0, GAMMA = 1.3, 0.7
c = W0**2 + 2
h = np.sqrt(c**2 - 4)
D = (c - h) / 2


def v_of_j(N):
    j = np.arange(N)
    return W0**2 + 4 * np.sin(np.pi * j / N) ** 2


def D_lattice(N):
    j = np.arange(N)
    return 2.0 * np.sum(np.sin(2 * np.pi * j / N) ** 2 / v_of_j(N)) / N


def D_closed(N):
    return D * (1 - D ** (N - 2)) / (1 - D**N)


def D_error(N):
    return -D * (1 - D**2) * D ** (N - 2) / (1 - D**N)


Ns = np.arange(4, 41)
lat = np.array([D_lattice(N) for N in Ns])
clo = np.array([D_closed(N) for N in Ns])
# Resolve the exponentially small difference from the lattice sum itself;
# ordinary double precision reaches its roundoff floor before N=40.
mp.dps = 50
c_mp = mp.mpf(str(W0)) ** 2 + 2
D_mp = (c_mp - mp.sqrt(c_mp ** 2 - 4)) / 2
rel = np.array([
    float(abs((mp.mpf(2) / int(N) * mp.fsum(
        mp.sin(2 * mp.pi * j / int(N)) ** 2 /
        (mp.mpf(str(W0)) ** 2 + 4 * mp.sin(mp.pi * j / int(N)) ** 2)
        for j in range(int(N)))) - D_mp) / D_mp)
    for N in Ns
])
pred = np.abs(D_error(Ns)) / D

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))

ax[0].axhline(D, color="0.55", lw=0.7, zorder=1)
ax[0].plot(Ns, clo, color="C1", lw=1.9, zorder=3,
           label=r"closed form $D(1-D^{N-2})/(1-D^N)$")
ax[0].plot(Ns, lat, "o", ms=3.6, mfc="white", mec="C0", mew=1.2, zorder=4,
           label=r"lattice sum $\frac{2}{N}\sum_j\frac{\sin^2(2\pi j/N)}{v_{N,j}}$")
ax[0].set_xlabel(r"$N$")
ax[0].set_ylabel(r"$D_N^{\mathrm{lat}}$")
ax[0].set_title(r"Einstein coefficient at $\omega_0=1.3$", fontsize=9.5)
ax[0].annotate(r"$D=0.294508$", xy=(24, D), xytext=(24.5, D - 0.004),
               color="0.35", fontsize=8.5)
lg = ax[0].legend(fontsize=7.4, loc="center right", framealpha=0.92)
lg.get_frame().set_linewidth(0.5)

ax[1].semilogy(Ns, pred, "-", color="C3", lw=1.5, zorder=3,
               label=r"$D^{N-2}$ prediction")
ax[1].semilogy(Ns, rel, "o", ms=3.8, mfc="white", mec="C3", mew=1.3, zorder=4,
               label=r"$|D_N^{\mathrm{lat}}-D|/D$")
ax[1].set_xlabel(r"$N$")
ax[1].set_ylabel(r"relative deviation")
ax[1].set_title(r"exponentially small, negative correction", fontsize=9.5)
lg = ax[1].legend(fontsize=7.4, loc="lower left", framealpha=0.92)
lg.get_frame().set_linewidth(0.5)

for a in ax:
    a.tick_params(labelsize=8.5)
    a.set_xlim(4, 40)

fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.155, wspace=0.30)
fig.savefig(OUT, format="pdf", bbox_inches="tight")
print("wrote", OUT)
print(f"D = {D:.12f}   kappa_eff = D/(2 gamma) = {D / (2 * GAMMA):.12f}")
print("max |lattice - closed| over N=4..40 :", np.max(np.abs(lat - clo)))
for N in (6, 10, 20, 40):
    print(f"  N={N:>2}: lattice={D_lattice(N):.12f} closed={D_closed(N):.12f} "
          f"err={D_error(N):+.3e} GK={D_closed(N) / (2 * GAMMA):.9f}")
