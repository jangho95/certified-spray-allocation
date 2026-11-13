from common import *
import run_experiment as experiment
original = experiment.cpp_candidate

def compatible_candidate(input_path, B, prefix, init=None):
    if init:
        source = Path(init)
        normalized = source.with_name(source.stem + '_cpp_lf.csv')
        normalized.write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
        assert list(read_alloc(source)) == list(read_alloc(normalized))
        atomic(str(normalized) + '.format_check.json', dict(source=str(source), source_sha256=digest(source), cpp_input=str(normalized), cpp_input_sha256=digest(normalized), allocation_values_unchanged=True, change='CRLF to LF only'))
        init = str(normalized)
    return original(input_path, B, prefix, init)
if __name__ == '__main__':
    os.chdir(ROOT)
    check_protocol()
    experiment.cpp_candidate = compatible_candidate
    try:
        experiment.verify_all()
    except Exception as exc:
        atomic('verification_attempts/attempt_2_failure.json', dict(error=error_text(exc), time_utc=now()))
        print(error_text(exc), flush=True)
        raise SystemExit(1)
