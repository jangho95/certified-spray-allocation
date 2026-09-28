# Sensitivity extension

## Data and scope

The extension contains 156 cases: six target thicknesses on four benchmark fields (24), 30 initial fields from each of four distributions (120), and three grid resolutions on four fields (12). The distribution sample is descriptive; it carries no Wilks percentile claim.

| Path | Contents |
| --- | --- |
| `inputs/sensitivity_extension/` | 120 canonical initial-field CSVs |
| `results/sensitivity_extension/` | First-pass protocol, results, operators and certificates |
| `results/sensitivity_extension_v2/` | Final results and certificates, refinement records and summaries |
| `work/sensitivity_extension/` | Per-run integer allocations and original solver outputs |
| `results/sensitivity_tables/` | Main Table 15, SI S13–S15 and their numerical checks |
| `pipeline/sensitivity_extension.py` | Model construction, candidate generation and numerical mesh comparisons |
| `pipeline/sensitivity_extension_v2.py` | Refined certificates with separate feasible continuous points |
| `pipeline/rigor_pilot.py` | Production exact arithmetic and feasible-point routines |
| `verification/verify_sensitivity.py` | Independent saved-data checks using `exact_oracle.py` |

Each `rows.json` record gives the input hash, budget, dimensions, exact rational strings `L_exact`, `U_exact`, `Vfeas_exact`, outward-rounded display values, and its witness path. Each witness contains the initial field, lower-bound reference, exactly feasible continuous point, selected integer allocation and all three candidates. Operators are stored in column-sparse arrays `col_ptr`, `cells`, `weights`, with `M` and `N`.

The final guarantees concern the stored float64 operator and initial field interpreted as exact rationals. The grid series uses a separate cell-average Gaussian model on `[0,50]^2`, with waypoints `(0.5+2i,0.5+2j)`, `i,j=0,...,24`. Analytic continuous evaluations and their discretization errors are numerical comparisons, not exact certificates of the continuous model. `delta_own` is `(V_c - V_h)/V_c`, displayed as a percentage.

## Verification without optimization

From the package root, with the documented Python environment:

```bash
python tools/verify_integrity.py
python verification/verify_certificates.py --suite sensitivity --workers 4
```

The verifier checks both versions: 156 v1 and 156 v2 cases, 936 candidate evaluations, input hashes, exact lower/upper values, exact continuous feasibility and directed rounding. It confirms that v2 preserves the integer allocations, does not lower any lower bound and does not increase any feasible continuous upper value. The final maximum continuous enclosure width is approximately `3.5669119147994212e-12`. Four recorded float dual exceedances are expected findings, not certificate failures.

The first pass has looser continuous upper values in some cases; those values can exceed the integer objective. This does not invalidate its lower bounds. The final v2 data are used in the manuscript.

## Recreate the tables

```bash
python pipeline/make_sensitivity_tables.py --output-dir /tmp/spray-sensitivity-tables
```

The generator reads saved exact rational values. Individual upper endpoints and certificate widths round up, lower endpoints round down. Distribution minima round down, maxima up, and medians/quartiles to nearest. It reproduces the distributed four TeX tables and `sensitivity_tables_check.json` without solving an optimization problem.

## Recompute the optimization in a new copy

```bash
python tools/new_run.py ../spray-sensitivity-replay --build --stage sensitivity
```

The build creates both `pufoam_solver` and `matrix_exchange`, using the pinned compiler and `-ffp-contract=off`. The replay uses all archived inputs and current public code, records a new protocol, and writes `results/sensitivity_extension_replay/`, `results/sensitivity_extension_v2_replay/` and `work/sensitivity_extension_replay/`. Recorded v1/v2 results are retained. New allocations can differ across platforms; comparisons of stored exact certificates do not depend on reproducing the optimizer's floating-point decisions.

For selected cases, first create a copy without `--stage`, then run inside that copy:

```bash
python pipeline/replay_sensitivity.py --workers 2 --case target_zero_T50 --case dist_correlated_uniform_03 --case mesh_center100_F50
```

Use a fresh copy for each replay. A subset run is marked as such and is not a reproduction of the full 156-case series. Historical drivers retain their original source-hash checks; use this replay entry point rather than changing archived protocols to match the distributed sources.

Public Python comments and docstrings have been removed, including the `source_before` and `source_after` snapshots. Historical source hashes in protocols identify the original experiment files; source-to-public mappings are recorded under `provenance/`. Raw inputs, result records and certificate arrays are unchanged by packaging.
