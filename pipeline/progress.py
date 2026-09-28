import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
DASH = EXP / 'reports'
DATA = DASH / 'data'
STATUS = DASH / 'status.json'
LOCK = DASH / '.status.lock'

def strict_json(value):
    if isinstance(value, float) and (not math.isfinite(value)):
        return '+∞' if value > 0 else '−∞' if value < 0 else 'NaN'
    if isinstance(value, dict):
        return {k: strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    return value

def _find(state, tid):
    stage = tid.split('.')[0]
    stages = state.setdefault('stages', [])
    st = next((s for s in stages if s['id'] == stage), None)
    if st is None:
        st = {'id': stage, 'tasks': []}
        stages.append(st)
    t = next((t for t in st['tasks'] if t['id'] == tid), None)
    if t is None:
        t = {'id': tid}
        st['tasks'].append(t)
    return t

def _update(fn):
    DASH.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        fn(state)
        state['updated'] = time.time()
        tmp = STATUS.with_suffix('.tmp')
        tmp.write_text(json.dumps(strict_json(state), ensure_ascii=False, indent=1, allow_nan=False))
        os.replace(tmp, STATUS)

def init(force=False):
    _update(lambda s: s.clear() if force else None)

def task(tid, status, detail=None, metrics=None):

    def update(s):
        t = _find(s, tid)
        t.update(status=status, detail=detail)
        if metrics:
            t.setdefault('metrics', {}).update(metrics)
    _update(update)

def detail(tid, text):
    _update(lambda s: _find(s, tid).update(detail=text))

def log(message):
    print(message, flush=True)

    def update(s):
        s.setdefault('log', []).append({'t': time.time(), 'msg': message})
        s['log'] = s['log'][-200:]
    _update(update)

def result(key, title, payload):
    DATA.mkdir(parents=True, exist_ok=True)
    temp = DATA / (key + '.tmp')
    temp.write_text(json.dumps(strict_json(payload), ensure_ascii=False, indent=1, allow_nan=False))
    os.replace(temp, DATA / (key + '.json'))
    _update(lambda s: s.setdefault('results', {}).update({key: {'title': title, 'updated': time.time()}}))
if __name__ == '__main__':
    init()
