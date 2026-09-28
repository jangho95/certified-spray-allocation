import json
from pathlib import Path
ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'requirements.lock.txt').is_file() and (p / 'pipeline').is_dir()))

def data_path(value):
    p = Path(value)
    text = str(p)
    if p.is_absolute():
        if '/experiments/' in text:
            return ROOT / text.split('/experiments/', 1)[1]
        if '/JAMC/solver_cpp/' in text:
            return ROOT / 'reference' / 'archive' / text.split('/JAMC/solver_cpp/', 1)[1]
        return p
    return ROOT / p

def relocate(value):
    if isinstance(value, str) and value.startswith('/') and ('/experiments/' in value or '/JAMC/solver_cpp/' in value):
        return str(data_path(value))
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    return value

def load_json(text, *args, **kwargs):
    return relocate(json.loads(text, *args, **kwargs))
