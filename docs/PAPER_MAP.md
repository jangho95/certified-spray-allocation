# Manuscript results and verification files

This guide connects the paper’s tables and figures to their data files and verification commands.

Paths are relative to the repository root. The verification labels follow
Supplementary Table S3. Run `python verification/verify_certificates.py` with
the listed `--suite` option to check a group of results.

| Manuscript | Verification status | Files | Exact check |
|---|---|---|---|
| Table 5, Fig. S1 | Width (3) is verified in exact arithmetic. The displayed dual lower bound and Width (2) are numerical estimates (double precision). Fig. S1 shows the three-candidate allocations. | `results/rigor_pilot/`, `results/revision_figures/` | `--suite pilot` |
| Table 6, Tables S7 and S12, Fig. S4, SI Section 3 | Widths and optimization intervals use bounds evaluated in exact arithmetic. Table 6 and Table S12 also give sampling confidence intervals. The lower block of Table S7 and the trajectories in Fig. S4 give descriptive summaries. | `results/stage5_long25/` | `--suite stage5_long25` |
| Table 7, Fig. 3, Table S9 | Table 7 (3) and the black dashed curves are verified in exact arithmetic. Table 7 (2), the red and blue curves, and the widths in Table S9 are numerical estimates (double precision). | `results/bench_sweep/`, `work/stage9/bench/`, `results/revision_figures/` | `--suite benchmark` |
| Table 8, Tables S5 and S11, SI Section 2 | Widths in Tables 8 and S5 and the four width columns of S11 without a “verified” label are numerical estimates (double precision). The “verified” columns are post hoc checks in exact arithmetic. The main text gives confidence bounds based on verified widths. | `results/stage9/`, `results/stage9_rigor/`, `inputs/stage9/` | `--suite stage9` |
| Table 9, Section 6.3.1 | Optimization intervals use bounds evaluated in exact arithmetic. | `results/pso_random_fields/` | `python verification/verify_pso_random_fields.py` |
| Table 10, Tables S13–S15 | Certificate bounds and widths were verified in exact arithmetic. The distribution tables also give descriptive field statistics. The continuous variance and discretization comparisons are floating-point evaluations. | `results/sensitivity_extension_v2/` | `--suite sensitivity` |
| Table 11, Fig. S3, Table S6 | Displayed widths are numerical estimates (double precision). Table 5 reports the exact-arithmetic three-candidate width for the same zero-field target-budget allocation (`B=28348`). | `results/stage3/`, `work/stage3/` | `--suite pilot`, case `T-zero` only |
| Table 12, Fig. 4 | Displayed widths are numerical estimates (double precision). Table S22 gives the separate exact-arithmetic widths for 11 selected cases. | `results/stage7/`, `work/stage7/` | See the Table S22 row |
| Table S4, Fig. S2 | The widths in Table S4 are numerical estimates (double precision). Fig. S2 shows the input kernels. | `results/revision_tables/`, `results/revision_figures/` | — |
| Section 6.4.3, Table S18 | Displayed Table S18 widths are numerical estimates (double precision). Section 6.4.3 reports exact-arithmetic checks for the zero and center-30 cases at `s=1`. | `results/rigor_pilot/` | `--suite pilot`, cases `V-s1zero` and `V-s1c30` |
| Tables S6, S8 and S10 | Double-precision bound evaluations and solver diagnostics. | `results/stage2/`, `results/revision_tables/` | — |
| Tables 13 and 14, SI Section 8 | Floating-point diagnostics. The zero envelope widths under common mass follow analytically. | `results/stage8/` | — |
| Table 15, Tables S16, S19 and S20 | SCIP results at solver tolerances, with numerical relaxation bounds and derived diagnostics. | `results/stage6/`, `work/stage6/` | — |
| Section 6.6.3, Table S21 | Descriptive timings. | `results/stage4/`, `work/stage4/` | — |
| Table S22, SI Section 10 | Separate numerical estimates and widths verified in exact arithmetic for the same 11 selected allocations. | `verification/kernel_width/` | `python verification/kernel_width/replay_kernel_witnesses.py` |
| Counts and timings quoted in the text (Sections 6.1.1, 6.1.3, 6.2.2, 6.3.1, 6.4.2, 6.6.3, and SI Sections 2 and 3) | PSO runs still improving at 5000 iterations, PSO wall times, clipped dual values, exchange-move counts, phase timings, fitness formula | computed from the files above | `python verification/second_revision_numbers.py` |

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

The sampling protocol was recorded before the study runs and was not
registered externally.

## Random-field PSO follow-up

The Table 9 follow-up uses the same stored fields as Table 8. Its protocol
was set after the field certificates were available and before the follow-up
PSO runs. The [follow-up documentation](../results/pso_random_fields/README.md)
describes the protocol, archived histories, and verification data.
