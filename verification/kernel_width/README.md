# Exact checks of eleven kernel-width cases

This archive rechecks the eleven Series B cases of the kernel-width study that
were selected by the study's numerical-accuracy rule. The integer allocations,
initial fields, kernels and budgets are the original ones.

The guarantee applies to the double-precision deposition operator and initial
field stored here. It does not cover the physical model or the other
kernel-width cases. Refining the reference point tightens the bound. All
reported bounds and variances are evaluated in exact rational arithmetic.

To replay the checks without any optimization solver, install NumPy and run:

```sh
python replay_kernel_witnesses.py
```

`protocol.json` names the inputs, allocations, selection rule, refinement
procedure and software versions. `cases/` and `summary.json` compare the
earlier numerical estimates with the verified values. `operators/` and
`witness/` hold the data the checker reads. `provenance/` keeps the original
field files, integer allocations and continuous reference points. The checker
verifies their hashes and compares them with the stored arrays. `SHA256SUMS`
lists the files of the archive.

The script that generated these data is not part of the archive. It needs the
experiment package and the optimization solvers. The archive alone is enough
to replay the checks.
