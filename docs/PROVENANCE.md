# Where the files come from

## What was changed for the public copy

The inputs, allocations, verification data and numerical results are the
files produced by the experiments, byte for byte. In logs and result
records, the workspace prefix of the absolute paths was shortened to
`/home/user/experiments`, and nothing else in those files was changed.
`provenance/source_mapping.json` lists every file with its original and
distributed hash and marks each file whose bytes differ with
`bytes_identical: false`. In the protocol record of the 59-field study the
entropy tag is withheld. The recorded entropy value is unchanged.

The source code differs from the code that ran the experiments in six ways.

1. Comments and docstrings were removed.
2. Machine-specific paths were replaced with paths relative to the repository.
3. A loader maps the absolute paths stored in old JSON files to the
   repository when they are read. Apart from the shortened prefix, the old
   files are not rewritten.
4. Reports are written under `reports/` by a small local writer.
5. Entry points for building, rerunning, checking integrity and verifying
   certificates were added.
6. `pipeline/stage9_e4.py` uses the recorded entropy value of the 59-field
   study directly.

Manuscripts, correspondence with reviewers, drafts, virtual environments,
binaries and caches are not included. The table text in `reference/` was
taken from the first submission for numerical comparison. `work/stage1`
contains modules that later scripts import, so it must stay complete.

## Hashes

`MANIFEST.json` covers the files in this repository. Hashes of code and
binaries stored in older run records and in `original_sources.sha256` refer
to the code as it was when the experiments ran. Because comments were
removed, the public source files have different hashes even where the
computations are the same.

For the 59-field study, the order of work was a development pilot, the
generation of the 59 fields, a record of the input hashes, code hashes and
evaluation rules, and then the runs, each checked against the recorded
hashes. That record is local and was not registered externally. The exact
verification was added afterwards and has its own protocol.
`replay_stage9.py` reruns the study with the current code and marks the
result as a rerun.

## PSO

`src/paper_pso.py` and the copy used in stage 1 were written from the
algorithm description in the authors' earlier article. The original code and
allocations of that article are not in this repository. Stage 5 compares this
re-implementation, on the same simulator, with verified bounds at the budget
each run reached.

## What the exact checks cover

The checker treats the stored double-precision operator entries and initial
fields as exact rational numbers. It minimizes a supporting-plane lower bound
over the box and budget constraints in exact arithmetic, checks that each
allocation is integer and feasible, and evaluates its variance exactly.
Reported floating-point endpoints are rounded outward.

Exact checks exist for the six target-budget and dense-waypoint cases, the
PSO comparisons, the 59-field study, the budget sweep, the 156 sensitivity
cases and the 11 kernel-width cases. These sets overlap. The other results
keep numerical bounds, and the integer solves by SCIP are optimal within
SCIP's tolerances.

For the six cases and the 59-field study, the checker also bounds the
difference from ideal Gaussian kernel values on the same support and with the
same boundary rule. The choice of kernel model and the match to a physical
process are outside these checks.

In the 59-field study the sampling unit is the field, so the sample size is
59, not 1,239. The population is independent Uniform[1,30] cells with the
stated geometry, allocation procedure and 21-budget grid. The largest
verified width in the sample, rounded up, bounds the population 95th
percentile of the largest gap over that grid with confidence at least
1 − 0.95^59.

## Tests of this copy

`provenance/validation.json` records a fresh installation, a build in a
different directory, model consistency checks, comparisons with the submitted
tables, a rerun of ten stage 3 cases, the regeneration of tables and figures,
and the certificate checks. The long experiments were not all repeated.

`provenance/revision_update.json` lists the files added with the first
revision. The verification data and candidate allocations of the budget
sweep are unchanged copies, and its run record differs only in the shortened
path prefix. Figures are generated from the stored verification data. The
two-candidate check on the 59 fields and the column-mass audit were added
later and do not change the protocol record.
