# Provenance and interpretation

## Sources and package changes

This package was assembled from the revision experiment workspace. `provenance/source_mapping.json` records each copied file's relative source, original SHA-256, distributed SHA-256, and whether its bytes are unchanged. Its omission list identifies internal drafts/scripts and other excluded material.

Inputs, saved allocations, witnesses, and numerical results were copied without changing their bytes. In logs and result records, the workspace prefix of historical absolute paths was shortened to `/home/user/experiments` for the public release; no other bytes of those files changed. `source_mapping.json` keeps their original SHA-256 and marks them `bytes_identical: false`. The Stage 9 entropy tag is withheld in the preregistration record; the recorded entropy value is unchanged. The public source code differs through:

1. Removal of explanatory Python/C++/CMake comments and Python docstrings. Preprocessor directives and the shell interpreter directive remain executable syntax.
2. Replacement of machine-specific source/input/interpreter locations with package-relative paths and the active Python interpreter.
3. A path loader that maps historical absolute paths into this package at read time. Apart from the path-prefix change above, historical JSON/JSONL is not rewritten.
4. A local report writer replacing dependence on the private plan/dashboard, and report paths under `reports/`.
5. Build, independent-run, integrity, Stage 9 replay, and certificate-verification entry points added for this distribution.
6. `pipeline/stage9_e4.py` uses the recorded Stage 9 entropy value directly.

The table fragments in `reference/` were extracted from the submitted main/supplementary TeX sources for numerical comparison. Full manuscripts, reviewer correspondence, internal analysis drafts, virtual environments, binaries, and caches are outside this distribution.

There is no external file dependency on the original experiment workspace. Historical logs and metadata retain absolute paths (with the shortened prefix); the runtime loader relocates relevant references. `work/stage1` includes supporting modules intentionally, since later experiment drivers import them.

## Hashes and historical claims

`MANIFEST.json` covers the current distributed files. The earlier code/binary hashes in preregistration, run records, and `original_sources.sha256` refer to the code used in those recorded experiments. Removing comments changes hashes even when the numerical expressions are unchanged. A newly built binary also has its own hash.

Do not replace historical hashes with public-package hashes or present public source as byte-identical preregistered source. The saved numerical inputs/results remain verifiable through the current manifest and their individual input/allocation hashes. `replay_stage9.py` explicitly records a retrospective rerun with the current code.

The Stage 9 sequence was: development pilot; generation of the 59 inputs; recording of input/code hashes and evaluation rules; hash-checked execution. This is a local record, not independent proof of its timestamp or prior access. The subsequent exact-certificate analysis was post hoc and has its own protocol.

## PSO provenance

`src/paper_pso.py` and its Stage 1 counterpart implement a PSO-style procedure reconstructed from the prior paper's published algorithm description. They are not the prior paper's original source. The original source and allocations are unavailable in this package.

Stage 5 compares this reimplementation on the same simulator against certified bounds at each run's actual budget. It does not establish the exact optimality gap of the prior publication's unavailable allocations. Lower mean variance than a published table does not by itself prove that a same-budget optimality-gap comparison is conservative for those unavailable runs. The random initial field must also be matched before making an implementation-level numerical comparison.

## Numerical and exact guarantees

The exact verifier treats the saved float64 operator entries and initial fields as exact rational numbers. A supporting-plane lower bound is minimized over the continuous box/budget constraint set using exact arithmetic. Integer feasibility is checked before evaluating the upper bound. Reported floating-point endpoints are rounded outward.

The pilot and Stage 9 additionally bound the difference from ideal Gaussian values on the same finite support with the same folding rule. This correction does not cover kernel-model choice, discretization of a physical process, uncertain measurements, or nonlinear physics absent from the additive model.

Rigorous verification is supplied for the pilot's 6 cases, Stage 5's 40 field/budget pairs with 104 PSO allocations, Stage 9's 1,239 cases, and the 84-case benchmark sweep. These groups overlap and should not be summed as unique instances. This does not turn every numerical bound in Stages 1–8 into a rigorous one. The SCIP pilot's `optimal` status is subject to SCIP's stated numerical tolerances.

For Stage 9, the sample unit is one initial field, so `n=59`, not 1,239. The population is the specified independent Uniform[1,30] cell model, with the specified geometry, procedure, and 21-budget grid. If `G2` is the maximum actual relative optimality gap of the fixed allocation procedure over that grid, the verified sample widths bound the corresponding observed `G2` values. Their upward-rounded maximum can therefore bound the population 95th percentile of `G2` with confidence at least `1 - 0.95^59`. This is neither a worst-case guarantee nor a claim about the distribution of an adaptively selected post hoc certification procedure.

## Validation performed for packaging

`provenance/validation.json` records a new dependency installation, a build from a relocated directory, model consistency checks, submitted-table comparisons, a 10-case Stage 3 rerun, table/figure regeneration, and separate certificate verification. The tests used a disposable working copy so distributed historical data stayed unchanged.

These checks establish that the packaged entry points and saved evidence used in those checks work without the original directory tree. They are not a claim that all long-running stages were repeated or that arbitrary platforms produce identical heuristic trajectories.

## Revision update

The benchmark witnesses and 252 candidate CSVs are copied without changing their bytes; the benchmark run record differs only in the shortened path prefix. Figure generation reads the saved witness/operator rather than a changing best-candidate registry and records hashes. `provenance/revision_update.json` records added/changed sources and the package baseline. The new Stage 9 two-candidate check and column-mass audit are additional analyses, not changes to the historical preregistration. The independent verifier has been extended to check these benchmark certificates and the submitted-rule Stage 9 values.
