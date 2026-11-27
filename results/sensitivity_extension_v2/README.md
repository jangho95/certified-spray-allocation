# Sensitivity extension: verification version 2

This directory contains exact post-verification for the same 156 cases as
`../sensitivity_extension/`. The integer candidates and their objective values
are preserved. Continuous relaxations were solved again to improve feasible
upper bounds and supporting-plane lower bounds. The original results remain in
the version 1 directory.

## Main files

- `protocol.json`: rules, case list, environment, and source hashes fixed before the v2 runs.
- `rows.json`, `cases/`: complete results.
- `summary_cases.csv`: table of the main quantities for all 156 cases.
- `summary.json`, `distribution_summary.json`: aggregate results.
- `operator_*.npz`, `witness/`: operators, initial fields, lower-bound reference points, exact feasible continuous points, and integer candidates.
- `verification.json`: exact reconstruction of L, U, all integer candidate values, and feasible continuous upper bounds.
- `tests.json`: projection tests, known-optimum QP tests, and the box audit of earlier reference points.
- `independent_rational_checks.json`: two cases checked with a separate Fraction implementation.
- `mesh_comparison.json`: numerical analytic-integral comparisons. These values are distinct from the exact certificates for the stored discrete operators.
- `source_before/`, `source_after/`: snapshots of the changed scripts.

## Verify or replay

Set up the [Python environment](../../README.md) and run from the repository root:

```sh
python verification/verify_certificates.py --suite sensitivity --workers 4
```

The verifier reconstructs the certificates from saved operators and witnesses.
To recompute the cases in a separate working copy, run:

```sh
python tools/new_run.py ../spray-sensitivity-replay --build --stage sensitivity
```

The [sensitivity guide](../../docs/SENSITIVITY.md) describes the archived
versions, the replay output, and how to rebuild the tables.

## Guarantee and interpretation

All 156 certificates pass exact reconstruction. The largest continuous optimum
enclosure has absolute width 3.567e-12. Seventeen lower bounds improve, while all
integer candidates and U remain unchanged. The largest certificate-width
reduction is 0.0001727 percentage points.

The guarantee applies to the stored float64 operators and initial fields.
The cell-average Gaussian mesh family is a separate continuous extension of the
point-sampled benchmark. Its analytic-integral comparisons use floating-point
evaluation and do not certify physical model error. Thirty fields per distribution
provide descriptive statistics; no additional Wilks statement is claimed.
