"""Run the archived calculations with explicit working directories."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='include the two larger matrix audits')
    parser.add_argument('--figures', action='store_true', help='also regenerate both figures')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    logs = root / 'verification_logs'
    logs.mkdir(exist_ok=True)
    jobs = [(root / 'supplementary', name) for name in (
        'exact_normalization_check.py', 'scalar_eigenvector_check.py',
        'scalar_recursion_check.py', 'physical_dispersion_identities.py',
        'sixth_order_scalar_benchmark.py', 'tensor_sector_gap_benchmark.py',
        'telegraph_order_check.py', 'comparison_constant_budget.py')]
    if args.full:
        jobs += [(root / 'supplementary', name) for name in (
            'prepared_mode_matrix_check.py', 'independent_scalar_matrix_audit.py')]
    if args.figures:
        jobs += [(root, name) for name in (
            'make_fig_scalar_coefficients.py', 'make_fig_finite_size_einstein.py')]
    records = []
    for folder, name in jobs:
        print('Running', name, flush=True)
        result = subprocess.run([sys.executable, name], cwd=folder,
                                capture_output=True, text=True, errors='replace')
        (logs / (Path(name).stem + '.txt')).write_text(
            result.stdout + '\nSTDERR:\n' + result.stderr, encoding='utf-8')
        records.append({'script': str((folder / name).relative_to(root)),
                        'exit_code': result.returncode})
        if result.returncode:
            print(result.stdout[-4000:] + result.stderr[-4000:], file=sys.stderr)
            break
    passed = len(records) == len(jobs) and all(r['exit_code'] == 0 for r in records)
    summary = {'python': sys.version, 'packages': {p: metadata.version(p) for p in
               ('numpy', 'scipy', 'sympy', 'mpmath', 'matplotlib')},
               'full_audit_requested': args.full, 'figures_requested': args.figures,
               'all_commands_completed_successfully': passed, 'commands': records,
               'scope': 'Process completion and recorded computational checks; not a replacement for the analytic proofs.'}
    (logs / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
