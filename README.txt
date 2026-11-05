SUPPLEMENTARY MATERIAL -- 21 October 2026

Long-wavelength energy relaxation and covariance transfer in pinned harmonic chains

Contents

The technical proofs are included in Appendices A-G of the main manuscript,
after its references. The main LaTeX source inputs appendices/technical_appendices.tex
and its four proof files. These belong to the manuscript source submission.
This separate Supplementary Material package contains the verification scripts,
recorded JSON outputs, figure-generation code and figures cited in the paper.
The former standalone Supplementary Information is retained only in the local
revision history and is not included in this submission archive.

1. Appendix numbering

The earlier supplement sections S1-S7 now correspond to Appendices A-G.
Older recorded computational outputs may still use S7 for Appendix G; their
values and the calculations have not changed. Infinite-window assumptions
are in Appendix D; the effective-chain fast-sector estimates are in Appendix B.

2. Run the calculations

Use Python 3.10 or newer. From the package root:
  python -m pip install -r requirements.txt
  python run_checks.py

The default command runs the normalization, eigenvector, scalar-recursion,
physical-dispersion, sixth-order integral, fast-sector matrix, and exact
telegraph-order and explicit comparison-constant checks.
To also run the larger prepared-covariance and independent matrix audits:
  python run_checks.py --full

The full audit can take appreciably longer. To also regenerate both figures:
  python run_checks.py --figures

The runner supplies each working directory and records command output and a
summary in verification_logs/. The successful six-check run used to prepare
the materials is preserved in recorded_checks/. Detailed outputs of the two
larger audits are the JSON files beside their scripts in supplementary/.
environment_versions.json records the package versions used for the supplied
outputs. The new telegraph_order_check.json records the exact eighth-order
counterexample and N=32,64 matrix comparisons; it is additional to that
archived six-check baseline. No network access is needed after dependencies are installed.

Each script can also be run separately. Run the two make_fig_*.py scripts from
the package root. Run the other scripts from supplementary/. The scripts that
write JSON place it beside their source files; the two remaining benchmarks
print comparisons, captured in recorded_checks/ for the supplied default run.

3. Scope

The calculations use exact symbolic identities, numerical quadrature and
finite-dimensional covariance matrices. No experimental data, trajectory
simulations or Monte Carlo samples were used. Floating-point discrepancies are
not rigorous quadrature error bounds or interval enclosures.

The effective-chain scalar formulas concern the specified pinned nearest-
neighbour chain. The deterministic comparisons average over an independent
stationary doubling-map environment, for the initial laws and scale conditions
stated in the manuscript. For Gaussian covariances constructed from the stochastic slow eigenmatrix,
finite-time logarithmic rates approximate that stochastic eigenvalue and its
fourth/sixth corrections; this does not identify deterministic spectral poles. Appendix D lists additional assumptions needed
for infinite-window response and pole transfer. The fast-sector gap estimates
refer to the effective chain.

4. AI-assisted tools

AI was used for language polishing and code checking. The author independently established and verified all results and conclusions and takes full responsibility for the accuracy and integrity of the work.

5. File integrity

supplementary_manifest.json gives relative filenames, sizes and SHA-256 digests.

Manuscript sources require the standard mathrsfs package in addition to the
previous mathematical packages. The main statements are Theorems 1-2. Corollary 1 is their application to
prepared finite-time logarithmic rates; Corollary 2 gives its error budget.

6. Proof and computation scope

The scripts audit the stochastic scalar and Einstein calculations and exact
dyadic transfer identities. They do not numerically test the deterministic
long-time comparison or prepared finite-time readout theorems.
The scalar covariance amplitudes are Q_l, R_l, S_l, P_l in the revised text
(qq, qp, pq, pp respectively). Internal script variables may retain A, B, C, D;
the linear systems and recorded values are unchanged.

7. Explicit comparison constants (23 October 2026)

From supplementary/, run:
  python comparison_constant_budget.py

The script uses exact SymPy arithmetic to evaluate the displayed local defect,
endpoint and accumulated-time constants at omega0=1, sigma=2. Its JSON also
records the N=32, epsilon=N^-3 momentum-law bound and the stochastic quartic
rate signal. These are formula evaluations, not sampled dynamical errors.
The crude upper bound is uninformative at that finite size; it does not show
that the true deterministic discrepancy is large. The eighth default check
is additional to the older six-check archived baseline.

8. Result hierarchy and limiting regime (23 October 2026)

The stochastic spectral relation and the deterministic mean comparison are the
two main theorems. The prepared logarithmic-rate result is Corollary 1, obtained
from the comparison, the constructed eigenmatrix and the stochastic Taylor
expansion. It fixes k as N grows, so the physical wave number tends to zero.
The finite-wavelength matching curve is a property of the limiting stochastic
branch, not a fixed-nonzero-wave-number deterministic spectral result.
All these statements hold at fixed positive pinning; no uniform unpinned limit
is claimed. Mathematical hypotheses and rates are unchanged by this relabeling.
