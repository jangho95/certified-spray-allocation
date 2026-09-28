# Fixed-kernel stop-and-spray allocation: code and data

Reproducibility package for **Separating operator, index, and certificate structure in fixed-kernel stop-and-spray allocation**, JangHo Seo and Joonwoo Lee.

This package contains the submitted-result archive, revision experiments (Stages 0–9 and the 156-case sensitivity extension), canonical inputs, integer allocations, PSO histories, and exact-arithmetic verification witnesses. The optimization model uses a fixed additive kernel and a fixed boundary rule.

[한국어 안내](README_ko.md) · [Execution guide](docs/REPRODUCING.md) · [Data guide](docs/DATA.md) · [Provenance and interpretation](docs/PROVENANCE.md)

## Quick verification

The tested platform is Linux/WSL with Python **3.12.14**. Run these commands from this directory, using a Python 3.12 installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
```

The certificate verifier uses NumPy and the Python standard library; it does not run an optimizer. Its six suites check the pilot's 6 cases, Stage 5's 40 field/budget pairs and 104 PSO allocations, Stage 9's 1,239 cases, the 84-case deterministic benchmark sweep, the expanded PSO comparison's 200 allocations (100 at each iteration limit), and the 156 sensitivity cases in both their original and final verification versions. These groups overlap. It also evaluates the two-candidate rule on the 59-field sample. Use ordinary Python, without `-O`, because verification uses assertions.

The expected Stage 9 maximum verified width is approximately **0.05144377935685342%**, rounded upward to **0.0515%**. Stage 5 has five recorded floating-point lower-bound violations; the verifier checks this finding rather than treating the old floating-point bounds as rigorous certificates.

## Build and rerun

Create a separate working copy to preserve the distributed results:

```bash
python tools/new_run.py ../stop-spray-stage3 --build --stage 3
```

This builds the C++ solver and reruns the low-count upper-bound comparison and rounding/repair experiments in the new directory. The build uses C++17, the pinned Zig/Clang compiler, and `-ffp-contract=off`, without `-march=native`.

To regenerate selected tables and figures from stored results:

```bash
python tools/new_run.py ../stop-spray-tables --stage tables
python tools/new_run.py ../stop-spray-figures --stage figures
```

Generated summaries are written to `reports/data/` and the relevant `results/stage*/` directories of that working copy. The private research dashboard is not required. See [the execution guide](docs/REPRODUCING.md) for other stages and the Stage 9 replay procedure.

## Contents

| Directory | Contents |
|---|---|
| `src/` | C++ simulator/repair solver and Python model, QP, PSO, and comparison code |
| `pipeline/` | Revision experiment drivers, report generation, and exact-bound generation |
| `inputs/` | Canonical initial fields, including the 59 Stage 9 fields, with manifests |
| `results/` | Recorded outputs, summaries, candidate registry, and exact certificates |
| `work/` | Per-run allocations, final fields, histories, and supporting Stage 1 modules |
| `reference/` | Original CSV archive and extracted submitted table text used in comparisons |
| `verification/` | Separate integer/rational arithmetic verifier for saved certificates |
| `tools/` | Portable build, working-copy creation, and integrity verification |
| `provenance/` | Source mapping and package validation results |

Keep `work/` and `reference/`: they are required for verification and reproduction. Historical copies named `pre_review*`, `*_v1*`, `*_before_*`, or `*backup*` are retained for audit and are not the final summaries.

## Validation of this package

The package was tested from a different directory using a newly installed environment:

- C++ compilation succeeded.
- Seven representative C++/Python model comparisons passed.
- All 3,184 stored-result comparisons were checked: 3,159 `ok`, 25 explained `warn`, zero `fail`.
- Ten rounding/repair cases were rerun; their reported widths agree with the recorded results.
- Five table/figure generation scripts completed.
- Saved exact-arithmetic certificates passed the separate verifier for the original three suites. The revision update also passes the new benchmark suite and regenerates the frozen figures; see `provenance/revision_validation.json`.

Details are in [provenance/validation.json](provenance/validation.json). This packaging check did not rerun every optimization experiment in Stages 0–9. Timing measurements depend on hardware and process configuration.

## Scientific and provenance notes

The PSO implementation (`paper_pso.py`) is a **same-simulator reimplementation from the prior publication's algorithm description**. The prior paper's original PSO source and allocations are not included. Comparisons against this implementation must be identified accordingly.

Historical inputs, results, and preregistration records are preserved. Public code has been adapted for relative paths, and all explanatory code comments and Python docstrings have been removed. Thus historical source hashes identify the earlier experiment code, not these public source files. `MANIFEST.json` identifies the distributed files; [provenance/source_mapping.json](provenance/source_mapping.json) records their source and changes.

Exact certification covers the stored float64 model and inputs for the pilot, Stage 5, Stage 9, the benchmark sweep, and the sensitivity extension. The pilot and Stage 9 also include a Gaussian-value correction for the same finite support and boundary folding. It is not a guarantee about physical model error. The sensitivity grid study certifies each stored discrete model; its analytic continuous-model comparisons remain numerical. Other stages retain their stated numerical guarantees.

See [NOTICE.md](NOTICE.md) for attribution and the current license status. Repository: [jangho95/certified-spray-allocation](https://github.com/jangho95/certified-spray-allocation). No release DOI has been assigned.

## Revision update

`results/bench_sweep/` contains the four benchmark fields at 21 budgets each, the stored operator and exact witnesses. The three candidate allocations per budget are in `work/stage9/bench/` (252 files). `pipeline/figures_revised.py` recreates the revised front and solution maps from these frozen witnesses, without optimization or a mutable registry lookup. Outputs are in `results/revision_figures/`.

```bash
python verification/verify_certificates.py --suite benchmark
python pipeline/figures_revised.py --output-dir /tmp/stop-spray-figures
```

The four maximum verified sweep widths are 0.10868866966488258%, 0.045443828341176475%, 0.10007512578728987%, and 0.051176733270915764% (zero, random s7, center-30, center-100). Ten zero-field budgets have floating-point dual values above independently computed feasible continuous objectives; the verified bounds are used for rigorous claims.

`results/stage9_rigor/two_candidate_check.json` records the post hoc verified widths for the submitted incumbent rule, with maximum 0.27791023119151836%. `pipeline/column_mass_audit.py` regenerates mean and maximum column-mass departures for both the reflective reference and each scenario's own mean. These additions do not modify the historical preregistration.

The latest update also contains `results/stage5_long25/`: all 100 long PSO runs, the corresponding 100 short runs, histories, allocations, exact witnesses and analysis. `python verification/verify_certificates.py --suite stage5_long25` independently checks all 200 allocations. The `figures` working-copy command regenerates all nine figure panels/files used in the revised main text and supplement. Their PNG outputs match the manuscript images byte for byte in the tested environment.

`results/revision_tables/` contains CSV snapshots of the displayed tables, with TeX notation preserved in cells. They retain display rounding; unrounded observations remain in the experiment result directories. See [the revision inventory](docs/REVISION_UPDATE.md) for data sources, figure scripts and long-run replay instructions.

## Target thickness, initial fields and grid resolution

The sensitivity extension adds 24 target-thickness cases, 120 fields from four distributions, and 12 grid-resolution cases. Final verified results are in `results/sensitivity_extension_v2/`; the first verification pass is preserved in `results/sensitivity_extension/`. Inputs, all three integer candidates, exact lower and upper values, and feasible continuous points are included.

```bash
python verification/verify_certificates.py --suite sensitivity --workers 4
python pipeline/make_sensitivity_tables.py --output-dir /tmp/spray-sensitivity-tables
```

The first command independently checks both versions, including all 936 candidate evaluations and the four recorded numerical-bound exceedances. The second regenerates main Table 15 and supplementary Tables S13–S15 from exact saved values, with the manuscript's rounding rules. See [the sensitivity guide](docs/SENSITIVITY.md) for data formats, build instructions and a separate-copy replay. Packaging validation is recorded in `provenance/sensitivity_20260929_validation.json`.
