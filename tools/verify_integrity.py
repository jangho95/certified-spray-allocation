import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT / 'MANIFEST.json')
    args = p.parse_args()
    entries = json.loads(args.manifest.read_text())['files']
    errors = []
    for name, expected in entries.items():
        path = ROOT / name
        if not path.is_file():
            errors.append([name, 'missing'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            errors.append([name, 'sha256 mismatch'])
    print(json.dumps(dict(checked=len(entries), errors=errors, passed=not errors), indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
