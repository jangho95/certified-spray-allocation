# From the manuscript to the files

Table and section numbers follow the revised manuscript of November 2026.
They may change in the published version. The older guides in this folder
(`REVISION_UPDATE.md`, `SENSITIVITY.md`, `REPRODUCING.md`) and
`results/revision_tables/` use the numbering of the first revision; for
example, their Table 15 is now Table 9.

Paths are relative to the repository root. "Exact check" names the suite of
`verification/verify_certificates.py` that recomputes the result.

| Manuscript | Content | Files | Exact check |
|---|---|---|---|
| Table 5; Section 6.4.3; SI Table S18 | certificates at the four target budgets; the two cases with a waypoint at every cell | `results/rigor_pilot/` | `--suite pilot` |
| Table 6; SI Tables S7 and S12; SI Section 3 | PSO comparison at 500 and 5000 iterations | `results/stage5_long25/` | `--suite stage5_long25` |
| Table 7; SI Table S9; front and solution-map figures | budget sweep, 4 fields at 21 budgets | `results/bench_sweep/`, `work/stage9/bench/`, `results/revision_figures/` | `--suite benchmark` |
| Table 8; SI Tables S5 and S11; SI Section 2 | 59 random fields | `results/stage9/`, `results/stage9_rigor/`, `inputs/stage9/` | `--suite stage9` |
| Table 9; SI Tables S13–S15 | target thickness, field distributions and grid resolution | `results/sensitivity_extension_v2/` | `--suite sensitivity` |
| Table 10 | low shot counts and the cap comparison | `results/stage3/`, `work/stage3/` | none |
| Table 11; shot-count and kernel-width figure | series A and B | `results/stage7/`, `work/stage7/` | 11 cases, see the row for SI Table S22 |
| Tables 12 and 13; SI Section 8 | nonuniform column mass | `results/stage8/` | none |
| Table 14; SI Tables S16, S19 and S20 | integer solves on reduced instances | `results/stage6/`, `work/stage6/` | none |
| SI Table S10 | solver tolerance | `results/stage2/` | none |
| Section 6.6.3; SI Table S21 | computational cost | `results/stage4/`, `work/stage4/` | none |
| SI Table S22; SI Section 10 | exact checks of 11 kernel-width cases | `verification/kernel_width/` (same content as the supplementary archive of the manuscript) | `python verification/kernel_width/replay_kernel_witnesses.py` |
| counts and timings quoted in the text (Sections 6.1.1, 6.1.3, 6.2.2, 6.3.1, 6.4.2, 6.6.3; SI Sections 2 and 3) | PSO runs still improving at 5000 iterations, PSO wall times, clipped dual values, exchange-move counts, phase timings, fitness formula | computed from the files above | `python verification/second_revision_numbers.py` |

## Individual files named in the text

| Statement in the manuscript | File |
|---|---|
| reference simulator of the operator check (Section 6.1.2) | `src/pufoam_solver.cpp` |
| comparison matrix of the operator check | built by `src/scip_compare.py`; result in `results/stage0/consistency.json` |
| re-implemented PSO and its settings (SI Section 3, SI Table S17) | `src/paper_pso.py` |
| list of the 100 PSO runs with 5000 iterations | `results/stage5_long25/runs.json` |
| design record of those runs | `results/stage5_long25/protocol.json` |
| protocol record of the 59-field sample, written before the fields were evaluated | `results/stage9/preregistration.json` |
| the 10 development fields | `results/stage9/dev_fields.json`, `dev_runs.json`, `dev_summary.json` |
| rounded starting allocations at `B = 12500` (Section 6.4.2) | `work/stage7/A_zero_F50_k7_s1.75_B12500_round_idx.csv`, `work/stage3/t3_b12500_c50_round_idx_allocation.csv` |
| exchange-move counts at the target budgets (Section 6.1.1) | `work/sensitivity_extension/target_*_T200_round_exchange.log` |

The file name `preregistration.json` is historical. The record is internal
and was not registered with an external registry.
