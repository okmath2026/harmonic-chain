# Harmonic-chain verification code

Supplementary code and recorded computational outputs for **Long-wavelength energy relaxation and covariance transfer in pinned harmonic chains**, by Qingyu Zhang.

This repository contains the computational supplement. The manuscript, proof appendices, submission correspondence and internal review records are not included.

## Requirements and use

Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python run_checks.py
```

The default runner executes eight checks. The two larger matrix audits are optional:

```bash
python run_checks.py --full
```

To regenerate both figures as well:

```bash
python run_checks.py --figures
```

The runner sets the working directories and writes its logs to `verification_logs/`. Individual scripts in `supplementary/` should be run from that directory; the two `make_fig_*.py` scripts should be run from the repository root.

## Contents

- `supplementary/`: exact symbolic calculations, deterministic quadrature, finite-dimensional covariance matrix checks, and their recorded JSON outputs.
- `recorded_checks/`: archived command outputs and the earlier six-check baseline summary. Later added checks have their own JSON outputs; this baseline is not a record of the expanded eight-check runner.
- `make_fig_*.py` and `fig_*.pdf`: figure-generation scripts and retained figures.
- `run_checks.py`: calculation runner.
- `requirements.txt` and `environment_versions.json`: dependencies and the package versions recorded with the supplied outputs.
- `README.txt`: detailed supplementary instructions and scope.
- `supplementary_manifest.json`: file sizes and SHA-256 hashes of the supplement inputs and recorded outputs.

## Scope of verification

These calculations check stochastic scalar dispersion and Einstein coefficients, finite-dimensional covariance identities and exact dyadic transfer identities. They are not trajectory simulations or numerical tests of deterministic long-time comparison theorems. Floating-point discrepancies do not replace rigorous quadrature bounds or analytic proofs. The fast-sector gap calculations concern the stochastic reference chain.

## Provenance

Prepared for public code sharing on 5 November 2026 from the supplementary materials assembled on 24 October 2026. Scientific scripts and numerical values are unchanged. One historical Windows output path in `recorded_checks/physical_dispersion_identities.txt` was replaced by its repository-relative path; the manifest was updated accordingly. The original submission files remain unchanged in the local submission workspace.

## License

No reuse license has been selected. Public availability does not by itself grant an open-source license.
