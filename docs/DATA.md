# Data guide

## Inputs and per-run data

`inputs/*.csv` stores canonical initial fields as headerless grids, with decimal precision sufficient to recover the stored float64 values. `inputs/manifest.json` records the input identities and hashes. `inputs/stage9/` contains the 59 statistical-study fields and its own manifest. Development fields with seeds 1–10 are separate from that sample.

`work/stage*/` stores the underlying run files, including allocations, final fields, PSO convergence histories, and intermediate candidates. The Stage 1 directory also contains modules used by later scripts; keep the whole directory. Typical formats:

| File | Format |
|---|---|
| `*_allocation.csv` | Header `waypoint,shot_count`; zero-based waypoint index and integer shot count |
| `*_history.csv` for Stage 5 PSO | `iter,gbest_variance,gbest_mean,gbest_budget,gbest_fit` |
| `*.npz` verification witness | NumPy archive of operator entries, initial field, reference point, budget, and/or allocations |
| `records.jsonl` | One run record per line, including settings, status, bounds, residuals, and provenance |
| `*.log` | Captured process/solver output; timing and paths refer to the historical run |

`reference/archive/` contains the original `JAMC/solver_cpp` CSV archive used for submitted-result comparisons. Some older values were stored with fewer digits. `reference/submitted_tables_*.tex` contains extracted table text needed by the comparison parser; these files are not standalone manuscripts.

## Result index

Paths below are relative to the package root.

| Study | Main result files | Underlying data |
|---|---|---|
| Stage 0: environment/model audit | `results/stage0/{inputs_check,rng_check,consistency,soc_check}.json` | `inputs/`, `work/stage0/`, solver logs |
| Stage 1: submitted results | `results/stage1/comparison_full.csv`, `comparison_summary.json`, `compare_*.json` | `work/stage1/`, `reference/archive/`, submitted table fragments |
| Stage 2: numerical sensitivity | `results/stage2/e1_runs.json`, `e1_clarabel.json`, `e1_summary.json` | Saved runs/settings and records in the same directory |
| Stage 3: C_max and candidates | `results/stage3/cmax_rows.json`, `table3_round_repair.json`, `target_round_repair.json` | `work/stage3/`, candidate registry |
| Stage 4: costs and reuse | `results/stage4/{single_summary,sweep_summary,pso_summary,reuse_integer_check}.json` | `work/stage4/`, `logs/`, per-run files in `results/stage4/` |
| Stage 5: PSO same-budget evaluation | `results/stage5/{pso_runs,pso_budgets,pso_summary}.json` | `work/stage5/`, candidate registry |
| Stage 6: integer solver pilot | `results/stage6/{scip_jobs,miqp_summary,formulation_decomposition}.json` | `work/stage6/`, solver logs/results |
| Stage 7: applicability | `results/stage7/range_rows.json`, `attainment.json` | `work/stage7/`; refinement data in `results/stage7/` |
| Stage 8: nonuniform mass | `results/stage8/{mass_structure,mass_dp,mass_dp_exact,mass_dp_meta}.json` | Parameters, counts, and timing metadata in these files |
| Stage 9: statistical study | `results/stage9/{preregistration,e4_runs,e4_fields,e4_summary,e4_diagnostics}.json` | `inputs/stage9/`, `work/stage9/`; `dev_*` files are separate development results |
| Exact pilot | `results/rigor_pilot/pilot_rows.json` | `results/rigor_pilot/witness/` |
| Exact Stage 5 | `results/stage5_rigor/{protocol,rows,summary}.json` | `results/stage5_rigor/witness/`, Stage 5 allocations, candidate registry, shared stored operator |
| Exact Stage 9 | `results/stage9_rigor/{protocol,rows,fields,summary,post}.json` | `operator.npz`, `witness/`, `VERIFY_FILES.json`, input CSVs |
| Sensitivity extension, final | `results/sensitivity_extension_v2/{rows,summary,mesh_comparison,distribution_summary}.json` | `inputs/sensitivity_extension/`, `operator_*.npz`, `witness/`, `work/sensitivity_extension/` |
| Sensitivity extension, first pass | `results/sensitivity_extension/{protocol,rows,verification}.json` | Original operator and witness files; integer candidates are preserved in v2 |
| Sensitivity tables | `results/sensitivity_tables/`, `results/revision_tables/` | Exact rows in the final sensitivity results; generator `pipeline/make_sensitivity_tables.py` |

## Revision additions

- `results/bench_sweep/{runs,rigor,summary}.json`, `operator.npz`, and `witness/`: 84 deterministic sweep cases and exact witnesses.
- `work/stage9/bench/`: three saved candidate allocations for every sweep case. Historical absolute paths in JSON are relocated by the package reader.
- `results/revision_figures/`: revised front and solution maps plus source-witness hashes.
- `results/stage9_rigor/two_candidate_check.json`: verified maxima for the submitted two-candidate procedure on the 59-field sample, as an additional analysis.
- `results/stage8/column_mass_audit.json`: mean and maximum absolute column-mass departures from the reflective reference and from each scenario's own mean.

## Candidate registry

`results/incumbents/registry.jsonl` records candidates with their run ID, method, objective, feasibility, and allocation hash. `results/incumbents/alloc/` contains the corresponding allocations. `best.json` selects the best registered feasible candidate for each instance.

This registry can accumulate later candidates. To reproduce a particular historical table, use that table's run records and allocation hash. The exact Stage 5 verifier resolves the reported Stage 5 allocation from its run records; it does not substitute a newer `best.json` candidate.

## Main symbols and units

| Name | Meaning |
|---|---|
| `F` | Field side length in grid cells |
| `M`, `N` | Number of field cells and allocation variables, respectively |
| `B` | Total shots, the exact sum of the integer allocation |
| `Cmax`, `C_max` | Per-variable upper limit; the usual lower limit is one shot |
| `s0`, `input` | Initial field and its canonical CSV |
| `L` | Lower bound, whose numerical or exact status depends on the result set |
| `U` | Objective of the selected feasible integer allocation |
| `P` | Objective of the PSO reimplementation's allocation |
| `V_QP`, `Q` | Computed continuous-relaxation objective; not automatically an exact optimum |
| `rel_gap` | Relative width, usually `(U-L)/U` as a fraction |
| `gap_*_pct`, `W_*_pct*`, Stage 9 `T1`/`T2` | Relative width expressed in percent; read each field name/schema |
| Stage 5 `rel_low`, `rel_high` | Fractions `P/U-1` and `P/L-1`, not percentages |
| `*_exact` | Rational number encoded as an integer or numerator/denominator string |
| `*_down`, `*_up` | Directed floating-point endpoint; the schema states whether it is a bound, width, or excess |

Stage 4 memory summaries report MiB. Some historical generic keys say `mb`; consult the stage-specific measurement definition. Stage 8 bitset storage and DP core time are distinct from peak process memory and full process time.

Some older JSON files use Python's nonstandard `Infinity`/`NaN` representation or string sentinels. Raw files are preserved. Public report output under `reports/` uses strict JSON with string representations for nonfinite values. Python's `json` reader handles the historical format used by these scripts.

## Current versus historical results

Use unversioned final summaries and the exact-verification directories for current conclusions. `pre_review_v1/`, `pilot_rows_v1.json`, `range_rows_before_refine.json`, `range_rows_refine_v1.json`, and files containing `backup` or `before_fix` record earlier analyses. They are supplied for provenance, not as additional independent observations.

In particular, Stage 5's numerical lower bounds have been superseded for rigorous claims by `stage5_rigor/`. Five zero-field budgets have tiny invalid numerical continuous lower bounds. Stage 9's preregistered numerical results and subsequent rigorous verification are separate analyses of the same 59 fields.
