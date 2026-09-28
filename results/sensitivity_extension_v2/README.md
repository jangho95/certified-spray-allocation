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

## Reproduce

From the experiments directory:

```sh
.venv/bin/python pipeline/sensitivity_extension_v2.py test
.venv/bin/python pipeline/sensitivity_extension_v2.py run --workers 4
.venv/bin/python pipeline/sensitivity_extension_v2.py verify --workers 4
```

`run` resumes existing v2 case files. Fresh calculations require a separate copy
with those v2 case files removed. The v1 results and inputs must be retained,
because v2 preserves the original allocations. `verify` reconstructs certificates
from saved operators and witnesses without optimizing allocations.

Version 1 scripts depend on the original shared routines. Their archived source
is in `source_before/`; v1 code hashes are historical records and should not be
rewritten to match the corrected routines.

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
