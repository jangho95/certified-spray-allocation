import determinism
import hashlib
import json
from pathlib import Path
import time

import stage9_e4 as s9

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / 'RUN_COPY.json').is_file():
        raise SystemExit('Run this through tools/new_run.py DEST --build --stage 9.')
    archived = json.loads((root / 'results/stage9/preregistration.json').read_text())
    manifest = json.loads((root / 'inputs/stage9/manifest.json').read_text())
    for name, item in manifest.items():
        if digest(root / 'inputs' / name) != item['sha256']:
            raise RuntimeError('Input changed: ' + name)
    s9.OUT = root / 'results/stage9_replay'
    s9.WORK = root / 'work/stage9_replay'
    if s9.OUT.exists():
        raise SystemExit('Replay results already exist.')
    s9.OUT.mkdir(parents=True)
    protocol = dict(kind='retrospective replication using the archived 59 inputs',
                    original_protocol_sha256=digest(root / 'results/stage9/preregistration.json'),
                    original_code_hashes=archived['code_sha256'], current_code_hashes=s9.code_hashes(),
                    original_outputs='results/stage9', new_outputs='results/stage9_replay',
                    claim='This is a rerun of existing inputs, not a new preregistered statistical sample.')
    (s9.OUT / 'replication_protocol.json').write_text(json.dumps(protocol, indent=2))
    jobs = [dict(set='replay', name=Path(name).stem, input=name) for name in sorted(manifest)]
    threads = determinism.assert_single_thread()
    started = time.perf_counter()
    results = s9.run_set(jobs, 'Archived-input replication')
    s9.finish(results, 'retrospective replication, 59 archived fields', 'e4_replay', 'replay', threads)
    print(json.dumps(dict(fields=len(results), elapsed_s=time.perf_counter()-started)))

if __name__ == '__main__':
    main()
