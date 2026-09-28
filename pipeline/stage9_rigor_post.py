from package_paths import load_json
import determinism
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP / 'work' / 'stage1')]
import rigor_pilot as rp
import stage9_e4 as s9
import stage9_rigor as sr
from qp_audit import sha256_file

def main():
    OUT = sr.OUT
    rows = load_json((OUT / 'rows.json').read_text())
    out, meta = (rows['fields'], rows['meta'])
    fields = load_json((OUT / 'fields.json').read_text())
    summ = load_json((OUT / 'summary.json').read_text())['summary']
    for fr in out:
        p = OUT / 'witness' / f"{fr['name']}.npz"
        w = dict(np.load(p))
        s0 = np.asarray(s9.load_field(fr['input']), dtype=np.float64)
        if 's0' in w:
            assert np.array_equal(w['s0'], s0)
            continue
        np.savez(p, **w, s0=s0, Cmax=np.array(s9.CMAX), M=np.array(s9.M), input=np.array(fr['input']), input_sha256=np.array(sha256_file(s9.INPUTS / fr['input'])))
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        ver = dict(ex.map(sr.verify, out))
    verify_s = time.perf_counter() - t0
    assert all((all(v) for v in ver.values())), 're-verification failed'
    g = SimpleNamespace(field_size=50, waypoint_rows=25, waypoint_cols=25, spray_interval=2, kernel_size=7, sigma_x=1.75, sigma_y=1.75, boundary='reflect')
    pts = rp.make_waypoints(g)
    st = rp.build_stamps(g, pts)
    t0 = time.perf_counter()
    eps, _ = rp.ideal_eps(st, pts, 2, s9.CMAX)
    eps_s = time.perf_counter() - t0
    assert eps == meta['eps_ideal']
    files = sorted([OUT / 'operator.npz', OUT / 'rows.json', OUT / 'fields.json', OUT / 'summary.json', OUT / 'protocol.json', *(OUT / 'witness').glob('*.npz'), *s9.NEW.glob('e4_*.csv'), s9.NEW / 'manifest.json', EXP / 'pipeline' / 'rigor_pilot.py', EXP / 'pipeline' / 'stage9_rigor.py', EXP / 'pipeline' / 'stage9_rigor_post.py'])
    (OUT / 'VERIFY_FILES.json').write_text(json.dumps(dict(purpose='files needed to re-verify every Stage 9 rigorous bound without solvers: witness/<field>.npz carries s0, B, C_max, M, the bound points and the allocations; operator.npz carries A; the input CSVs are listed for traceability of s0; exact L/U are in rows.json', verify='pipeline/stage9_rigor.py: verify(field) for every field of rows.json (needs rigor_pilot.py for the exact arithmetic)', files={str(f.relative_to(EXP)): sha256_file(f) for f in files}), indent=1))
    post = dict(verify_wall_s=verify_s, workers=8, eps_s=eps_s, verified_fields=sum((all(v) for v in ver.values())), witness_self_contained=True)
    (OUT / 'post.json').write_text(json.dumps(post, indent=1))
    sr.publish(fields, summ, meta, out, post)
    return post
if __name__ == '__main__':
    print(main())
