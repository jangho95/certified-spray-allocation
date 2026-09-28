import os
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
ENV_BEFORE = {k: os.environ.get(k) for k in THREAD_VARS}
for _k in THREAD_VARS:
    os.environ[_k] = '1'

def thread_record() -> dict:
    rec = {'env_before': ENV_BEFORE, 'env_forced': {k: os.environ.get(k) for k in THREAD_VARS}}
    try:
        from threadpoolctl import threadpool_info
        rec['threadpools'] = [{k: p.get(k) for k in ('user_api', 'internal_api', 'num_threads', 'version')} for p in threadpool_info()]
    except ImportError:
        rec['threadpools'] = 'threadpoolctl unavailable'
    return rec

def assert_single_thread() -> dict:
    rec = thread_record()
    pools = rec['threadpools']
    if isinstance(pools, list):
        bad = [p for p in pools if p['num_threads'] != 1]
        if bad:
            raise RuntimeError(f'thread pools not single-threaded: {bad}')
    return rec
