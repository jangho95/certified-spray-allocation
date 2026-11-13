# PSO follow-up on 59 independent random fields

This follow-up was specified after the existing field certificates were
available. It is not an externally preregistered experiment. All 59 stored
fields were used, with 25 distinct seeds per field and 5000 iterations.
The fields, seed schedule, settings and summaries were fixed before the
first PSO execution. The original execution protocol hash is preserved in
`public_copy_provenance.json` and the result records.

`protocol.json` is the publication copy, with repository-relative source
locations and its separate hash in `protocol.sha256`. Comments and
docstrings have been removed from the public Python sources. Original
and distributed hashes and executable-AST checks are recorded in
`provenance/pso_random_fields_sources.json` at the repository root.
The computational functions match the execution sources. The runner
locates the repository build output instead of a bundled executable.
The two reference-only scripts, `code/stage5_pso_same_budget.py` and
`code/stage5_long_report.py`, are the portable versions from `pipeline/`.

## Recorded results

`field_summaries.csv` and `summary.json` contain the planned summaries.
`fields/<field>/runs.csv` lists every run. Each attained field-budget pair
has exact supporting-plane and allocation witnesses under
`fields/<field>/budgets/`. `operator.json` and the copied field inputs
specify the stored binary64 model interpreted as exact rationals.

The per-field statistic is the sample median of 25 same-budget relative
excesses, enclosed by the medians of the verified endpoints. It is not a
population median over infinitely many seeds on a fixed field. The
population-percentile statement is restricted to independent Uniform[1,30]
fields on the benchmark grid, target and PSO setting.

The intermediate CSV line-ending correction is documented in
`deviations.json`. The first 88 postprocessing failures are retained in
`verification_attempts/attempt_1/`; all 88 pairs subsequently passed.
No PSO execution was repeated for this correction.

## Check without an optimizer

From the repository root:

    python verification/verify_pso_random_fields.py

The checker uses the standard library. It checks all supporting-plane
witnesses, candidate values, 1475 excess intervals, the planned aggregate
statistics, and iteration-4500/5000 counts from the compressed histories.
It does not rewrite the archived results. No history extraction is needed.

## Repeat an individual recorded PSO execution

Install the versions in the protocol and run from the repository root:

    python results/pso_random_fields/reproduce_one.py --field e4_01 --seed 100001

Output is placed in a new `replayed_runs/` folder and compared with the
recorded allocation. The historical results are preserved. Whole-sample
entry points are also retained in `run_experiment.py`; their completion
guard prevents silently replacing the finished sample.

The 1475 iteration histories are compressed in `histories.tar.gz`, retaining
their original relative names and bytes. `history_archive.json` records
their size and the archive hash. To inspect the CSV files directly, extract
the archive from this directory with `tar -xzf histories.tar.gz`.

## Build the C++ candidate solver

No compiled executable is distributed. The execution binary hash is retained
in the protocol as historical provenance. Its C++ source has the same
executable tokens as the execution source, with comments removed.
Before regenerating budget candidates, build from the repository root:

    python tools/build.py

The runner uses `build/pufoam_solver`. The solver-free checker and the
individual PSO replay above do not require that build.
