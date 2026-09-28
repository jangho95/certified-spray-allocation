# Reproducing and verifying results

## Environment

Use Linux/WSL and Python 3.12.14 with `requirements.lock.txt`. Dependencies are version-pinned; the file is not a cryptographic lock of downloaded wheels. The tested CPU architecture is x86-64. Native Windows execution is not supported by this package: process reporting uses `/proc`, and progress/registry locking uses `fcntl`.

From the package root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
```

The requirements include NumPy, SciPy, OSQP, Clarabel, PySCIPOpt, Matplotlib, CMake, Ninja, and Zig/Clang. No Gurobi license is required. Stage 6 uses SCIP for both integer formulations; it is not an independent-solver comparison.

## Verify the distributed files

```bash
python tools/verify_integrity.py
```

This checks SHA-256 hashes against `MANIFEST.json`. The manifest excludes itself. Extra runtime files are allowed; modified or missing listed files fail verification. Check the original distributed copy before executing scripts that replace outputs.

## Recompute exact certificates from stored witnesses

```bash
python verification/verify_certificates.py --workers 4
python verification/verify_certificates.py --suite pilot
python verification/verify_certificates.py --suite stage5
python verification/verify_certificates.py --suite stage9 --workers 4
python verification/verify_certificates.py --suite benchmark
python verification/verify_certificates.py --suite stage5_long25
python verification/verify_certificates.py --suite sensitivity --workers 4
```

The verifier uses separate integer/rational arithmetic in `verification/exact_oracle.py`. It checks the recorded supporting-plane lower bounds, feasible allocations, exact objective values, and directed rounding. It also checks the pilot/Stage 9 Gaussian correction. It reads the published witnesses and does not import the production certificate-generation arithmetic or call a solver. Keep assertions enabled: do not use `python -O`.

The expanded verification, including the long PSO comparison, took about 24 seconds with four workers during packaging. This is a verification measurement on the packaging machine, not a replacement for the recorded optimization or certificate-generation benchmarks.

## Build only

```bash
python tools/build.py
```

This produces `build/pufoam_solver` and `build/matrix_exchange` using pinned Zig/Clang through `tools/zigcxx`. It also creates package-internal relative symlinks to `inputs/` and `build/` for the original Stage 1 script layout. C++ flags include `-O3 -ffp-contract=off`; `-march=native` is not used. No distributed data file is rewritten by the build.

## Independent working copies

Recommended workflow:

```bash
python tools/new_run.py ../stop-spray-run --build
```

The destination must be a new directory outside this package. The tool copies the baseline code/data, omits build/environment/cache/report directories, records `RUN_COPY.json`, and optionally builds the solver. Activate the same environment before running commands inside this copy. Each copy needs roughly the package size plus the environment/build/new outputs.

Stages may overwrite baseline summaries or append registry records **in the working copy**. Do not mix newly generated values with the original recorded benchmark when quoting results. For a clean comparison, use one new copy per stage.

## Stage entry points

`tools/new_run.py DEST --build --stage K` creates a copy and runs the listed scripts for `K=1,...,9`. The `tables` option needs no C++ build.

| Stage | Entry points in `pipeline/` | Scope |
|---|---|---|
| 0 | `s0_consistency.py`, `s0_rng_check.py`, `s0_soc_check.py` | Model/RNG/SOC checks; invoke separately in a built working copy |
| 1 | `stage1_run.py` | Submitted tables, archive comparisons, PSO and solver runs |
| 2 | `stage2_e1.py` | Numerical lower-bound sensitivity and Clarabel comparison |
| 3 | `stage3_cmax.py`, `stage3_round_repair.py` | Upper-bound effects and integer candidate improvement |
| 4 | `stage4_run.py`, `stage4_reuse_integer_check.py` | Timing/memory and reuse-path integer candidates |
| 5 | `stage5_pso_same_budget.py`, `stage5_figure.py` | PSO reimplementation and same-budget comparison |
| 6 | `stage6_miqp.py`, `stage6_decompose.py` | SCIP pilot and gap decomposition |
| 7 | `stage7_range.py`, `stage7_refine.py`, `stage7_report.py` | Applicability sweep and bound refinement |
| 8 | `stage8_mass.py` | Nonuniform mass and bounded DP |
| 9 | `replay_stage9.py` | Retrospective rerun of the 59 archived fields |
| tables | `stage4_aggregate.py`, `stage5_figure.py`, `stage6_decompose.py`, `stage7_report.py`, `stage9_report.py` | Regenerate selected summaries/figures without optimization |
| figures | `figures_revised.py`, `initial_fields_3d.py`, `legacy_v5_figures.py`, `range_figures.py`, `stage5_long_report.py` | Recreate the nine figure files used in the revised paper and SI |
| pso-long | `replay_stage5_long25.py` | Repeat the 96 extension runs, retaining the four archived long runs; write a separate replay analysis |
| sensitivity | `replay_sensitivity.py` | Recompute 156 cases using archived inputs; write separate allocation and refined-certificate results |
| sensitivity-tables | `make_sensitivity_tables.py` | Regenerate main Table 15 and SI S13–S15 without optimization |

Stages 1, 5, and 6 can be substantially longer than a smoke test. Stage 6 has 45 jobs with 300-second SCIP limits. The recorded Stage 0 large continuous SOC failure is solver-specific; the package does not claim that packaging repairs that solver issue.

Example representative consistency check:

```bash
cd ../stop-spray-run
python pipeline/s0_consistency.py
```

Example table generation in a fresh copy:

```bash
python tools/new_run.py ../stop-spray-tables --stage tables
```

Run that second command from the original package root. Tables/figures are placed under the copy's `results/` and `reports/data/`. The report writer records progress in `reports/status.json`; no web server is required.

## Stage 9 and preregistration

The historical `stage9_e4.py register/run` guard checks original source/binary hashes. Public packaging changes the source hashes. Do not edit the historical preregistration to make those commands pass.

Use:

```bash
python tools/new_run.py ../stop-spray-stage9 --build --stage 9
```

The replay validates the archived input hashes, records current and historical code hashes, and writes to `results/stage9_replay/` and `work/stage9_replay/`. It preserves `results/stage9/`. This is a retrospective replication on existing inputs, not a new preregistered sample. Its report uses a separate replay name.

## Generate new exact certificates

The saved certificates normally suffice for review. To regenerate them, use a separate built working copy and run `pipeline/rigor_pilot.py`, `pipeline/stage5_rigor.py`, or `pipeline/stage9_rigor.py`, as appropriate. These generation scripts replace their respective certificate outputs/protocol files in that copy. `pipeline/stage9_rigor_post.py` regenerates postprocessing and the witness-file list. Do not describe a newly generated protocol as the historical preregistration.

## Numerical reproducibility

Canonical CSV inputs and stored integer allocations define the recorded experiment. Thread controls force one BLAS thread in the revised computational paths. Near-ties in rounding and floating-point decisions in repair can still depend on compiler, CPU, or library versions. Agreement of exact evaluation for a stored allocation is distinct from generating the identical allocation on another machine. Timing and peak memory are hardware-dependent.

## Revised benchmark sweep and figures

For the stored 84-instance sweep, the independent `benchmark` verification suite checks exact lower/upper values, budget/box/integer feasibility, all 252 candidate allocations, the selected incumbent, and the four summary maxima. It also independently checks the recorded numerical-bound violations. No solver runs are needed.

```bash
python tools/new_run.py ../stop-spray-figures --stage figures
```

This recreates all nine revised figure files in `results/revision_figures/`, including the front and solution maps, the initial-field surfaces, the kernel/count figures, the range study and the long PSO convergence plot. For a newly optimized sweep, use `python tools/new_run.py ../stop-spray-benchmark --build --stage benchmark`. That command is a new run and changes results in its separate copy; its timing or allocations need not match historical files bit for bit.

To regenerate the added column-mass diagnostics in a working copy, run `python pipeline/column_mass_audit.py`. It reconstructs the specified operators and checks their mass summaries against Stage 8 before reporting departures; it does not optimize allocations.

The target-thickness, distribution and grid-resolution extension has a separate [sensitivity guide](SENSITIVITY.md). Use its independent verification command for archived results; historical experiment drivers intentionally retain original source-hash guards.
