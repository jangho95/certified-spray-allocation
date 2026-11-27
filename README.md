# Fixed-kernel stop-and-spray allocation

Code and data for **Separating operator, index, and certificate structure in
fixed-kernel stop-and-spray allocation** by JangHo Seo and Joonwoo Lee.
[Paper citation and terms of use](NOTICE.md).

## Verify the results

From the repository root on Linux or WSL with Python 3.12, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python tools/verify_integrity.py
python verification/verify_certificates.py --workers 4
python verification/kernel_width/replay_kernel_witnesses.py
python verification/verify_pso_random_fields.py
```

The integrity check compares the distributed files against SHA-256 hashes.
The verification scripts recompute selected certificate bounds and allocation
objectives from their stored inputs in exact arithmetic without running an
optimizer. See the [verification and reproduction guide](docs/REPRODUCING.md)
for individual checks and instructions for rerunning experiments.

## Find the data and code

- [Paper tables and figures](docs/PAPER_MAP.md) links the reported results to
  their data, verification status, and checkers.
- [Data guide](docs/DATA.md) describes input fields, allocations, witnesses, and
  result file formats.
- [Inputs](inputs/), [saved results](results/), and [recorded runs](work/) hold
  the data used in the paper.
- [Simulation and optimization code](src/), [experiment scripts](pipeline/),
  and [verification code](verification/) contain the implementations.

[한국어 안내](README_ko.md)
