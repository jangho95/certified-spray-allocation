# Revision data and figure inventory

The update includes the 84-case benchmark sweep, the expanded 25-run PSO comparison at 5,000 iterations for each of four fields, and the current figure-generation scripts. Historical protocols, raw allocations and inputs remain unchanged. Public Python scripts have relative paths and no explanatory comments or docstrings.

## Exact verification

Run `python verification/verify_certificates.py --workers 4`. Its six suites check the pilot, original PSO comparison, 59-field sample, benchmark sweep, expanded PSO comparison, and sensitivity extension. The expanded PSO suite independently evaluates 40 field/budget instances and all 200 PSO allocations. The sensitivity suite checks all 156 cases in both v1 and v2, including integer feasibility, exact objectives, supporting-plane bounds and feasible continuous upper values. No optimization is needed.

The packaging check is recorded in `provenance/revision_20260928_verification.json`. The optimization experiments were not rerun as part of packaging.

## Figure generation

Run `python tools/new_run.py ../revision-figures --stage figures` from the package root. The destination must not exist. Outputs are placed in that copy's `results/revision_figures/`.

| Figure file | Generator in `pipeline/` | Stored inputs |
|---|---|---|
| `problem_overview_rev` | `legacy_v5_figures.py` | `reference/legacy_figure_inputs/` |
| `initial_fields_3d` | `initial_fields_3d.py` | `reference/initial_fields_3d_inputs/` |
| `pareto_fronts_rev` | `figures_revised.py` | `results/bench_sweep/`, `inputs/` |
| `stage7_seriesA`, `stage7_seriesB` | `range_figures.py` | `results/stage7/range_rows.json` |
| `solution_maps_rev` | `figures_revised.py` | `results/bench_sweep/witness/`, stored operator and inputs |
| `gaussian_kernel_shapes_rev`, `cmax_count_ablation_rev` | `legacy_v5_figures.py` | `reference/legacy_figure_inputs/` |
| `pso_long25_convergence` | `stage5_long_report.py` | `results/stage5_long25/history/`, `analysis.json` |

All nine regenerated PNG images matched the current manuscript images byte for byte in the packaging environment. PDF metadata can differ between runs.

## Tables and underlying numerical records

`results/revision_tables/index.json` maps table numbers and manuscript labels to 28 CSV grids from main Tables 1–14 and supplementary Tables S1–S12. Tables S7 and S10 each have two grids. These are snapshots of the displayed cells, including TeX notation and display rounding; they are not replacements for unrounded numerical records.

Four further grids contain main Table 15 and SI S13–S15. The sensitivity generator reproduces their TeX tables from the final exact rows; [SENSITIVITY.md](SENSITIVITY.md) describes the inputs and commands. The SI S3 diagnostic inventory is updated to include these additions.

| Results | Numerical records |
|---|---|
| Candidate improvement and low counts | `results/stage3/`, `results/incumbents/` |
| Mass perturbations and DP | `results/stage8/` |
| Target and sweep verified bounds | `results/bench_sweep/`, `results/rigor_pilot/` |
| Timing, memory and reuse | `results/stage4/` |
| Integer optimization and gap decomposition | `results/stage6/` |
| Random-field summaries and verified widths | `results/stage9/`, `results/stage9_rigor/` |
| Spacing, kernel, residual and original sweep tables | `reference/`, `work/stage1/`, `results/stage1/` |
| Numerical tolerance study | `results/stage2/` |
| Mean-shot-count and kernel-width studies | `results/stage7/` |
| PSO comparisons and confidence summaries | `results/stage5/`, `results/stage5_rigor/`, `results/stage5_long25/` |

Existing stage-report scripts regenerate numerical summaries; `pipeline/stage5_long_report.py` additionally regenerates the Table S12 fragment in `results/revision_figures/tableS12.tex`. Table formatting and cross-references belong to the manuscript.

## Long PSO replay

The published results and independent verifier are ready to use. To repeat the extension in a fresh working copy:

```bash
python tools/new_run.py ../pso-long-replay --stage pso-long
```

The replay retains the four archived seed-1 long runs and recomputes the other 96 runs using the archived inputs and settings. It records a new protocol under `results/stage5_long25_replay/`, evaluates the resulting allocations and generates a report. Historical outputs under `results/stage5_long25/` remain intact. This replay is not a new statistical sample and was not run during packaging. Directly invoking the historical run script against the archived protocol would fail its historical code-hash check; use the replay entry point above.
