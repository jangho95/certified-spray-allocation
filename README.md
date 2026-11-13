# Fixed-kernel stop-and-spray allocation: code and data

This repository holds the code, inputs and stored results for the manuscript
"Separating operator, index, and certificate structure in fixed-kernel
stop-and-spray allocation" by JangHo Seo and Joonwoo Lee.

The problem is to choose an integer number of spray shots at each waypoint so
that the final foam thickness is as even as possible for a given total number
of shots. The repository contains the simulator, the relaxation and rounding
code, the re-implemented PSO baseline, every allocation reported in the
manuscript, and the data needed to recheck the reported bounds in exact
arithmetic.

한국어 안내는 [README_ko.md](README_ko.md)에 있습니다.

## Check the stored results

The checks were run on Linux (WSL) with Python 3.12.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
```

The first script compares the files listed in `MANIFEST.json` with their
recorded hashes. The second recomputes the lower bounds and the variances of
the stored allocations in integer and rational arithmetic. It uses NumPy and
the Python standard library and runs no optimizer. Both scripts print a JSON
summary and stop with an error if a check fails. Do not run them with
`python -O`, because the checks are assertions.

`--suite NAME` runs one group of checks.

| Suite | What it rechecks |
|---|---|
| `pilot` | the four target-budget cases and the two cases with a waypoint at every cell |
| `stage5` | the first PSO comparison: 40 field–budget pairs and 104 allocations |
| `stage5_long25` | the PSO comparison in the manuscript: 200 allocations, 25 seeds per field at 500 and at 5000 iterations |
| `benchmark` | the budget sweep: 4 fields at 21 budgets each |
| `stage9` | 59 random fields at 21 budgets each (1,239 cases) |
| `sensitivity` | the 156 cases of the target-thickness, field-distribution and grid series |

For the 59 random fields, the largest verified width is 0.0515% after rounding up.

Two further checks were added with the second revision. Both stop with an
error if a value differs.

```bash
python verification/kernel_width/replay_kernel_witnesses.py
python verification/second_revision_numbers.py
```

The first rechecks the 11 kernel-width cases of Supplementary Table S22 and
needs NumPy. Its folder is the same as the supplementary archive of the
manuscript. The second recomputes, from the stored files, the counts and
timings quoted in the text of the revised manuscript. It uses the standard
library only.

## Rerun an experiment

Make a working copy first, so that the distributed results stay unchanged.

```bash
python tools/new_run.py ../stop-spray-stage3 --build --stage 3
```

This command builds the C++ solver and reruns the low-shot-count experiments
in the new directory. The build uses C++17 and the Zig/Clang compiler pinned
in the requirements, with `-ffp-contract=off` and without `-march=native`.

Tables and figures can be rebuilt from the stored results in the same way.

```bash
python tools/new_run.py ../stop-spray-tables --stage tables
python tools/new_run.py ../stop-spray-figures --stage figures
```

[docs/REPRODUCING.md](docs/REPRODUCING.md) describes the other stages. Run
times depend on the machine.

## Layout

| Directory | Contents |
|---|---|
| `src/` | C++ simulator and exchange search; Python model, QP, PSO and comparison code |
| `pipeline/` | scripts that ran the experiments and built the tables and figures |
| `inputs/` | initial thickness fields, including the 59 random fields |
| `results/` | stored outputs and the data used by the exact checks |
| `work/` | allocations, final fields and PSO histories of the individual runs |
| `reference/` | CSV files and table text of the first submission, used for comparison |
| `verification/` | the exact-arithmetic checker |
| `tools/` | build, working-copy and integrity scripts |
| `provenance/` | where each file came from and how the package was tested |
| `docs/` | guides to the data, the reruns and the manuscript tables |

The experiments were run in numbered stages, and the directory names keep
those numbers (`results/stage3`, `work/stage7`). [docs/PAPER_MAP.md](docs/PAPER_MAP.md)
lists the directory behind each table of the manuscript.
[docs/DATA.md](docs/DATA.md) explains the file formats.

Files and folders named `pre_review*`, `*_v1*`, `*_before_*` or `*backup*`
are earlier versions kept for the record. The checks need `work/` and
`reference/`, so keep both.

## Notes

- The PSO code (`src/paper_pso.py`) was re-implemented from the description
  in the authors' earlier article. It is not the code used in that article.
- The exact checks apply to the stored double-precision operators and
  initial fields. They do not address how well the model matches a physical
  coating process.
- The public source files carry no comments. The guides in `docs/` describe
  the entry points and the data formats.
- [NOTICE.md](NOTICE.md) gives the citation and the license status.

## PSO follow-up on independent fields

The comparison was extended to the existing 59 independent random fields with 25 new seeds per field and 5000 iterations (1475 runs). The follow-up protocol, allocations, exact witnesses, planned summaries and compressed histories are in [`results/pso_random_fields/`](results/pso_random_fields/README.md). Run `python verification/verify_pso_random_fields.py` for the solver-free check. The manuscript mapping is in [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md).
