import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    bin_dir = Path(sys.executable).parent
    cmake = bin_dir / 'cmake'
    ninja = bin_dir / 'ninja'
    if not cmake.exists() or not ninja.exists():
        raise SystemExit('Install requirements.lock.txt into the active Python environment first.')
    env = dict(os.environ, SPRAY_PYTHON=sys.executable)
    wrapper = ROOT / 'tools' / 'zigcxx'
    wrapper.chmod(0o755)
    subprocess.run([str(cmake), '-S', str(ROOT / 'src'), '-B', str(ROOT / 'build'), '-G', 'Ninja',
                    '-DCMAKE_MAKE_PROGRAM=' + str(ninja), '-DCMAKE_CXX_COMPILER=' + str(wrapper),
                    '-DCMAKE_BUILD_TYPE=Release'], check=True, env=env)
    subprocess.run([str(cmake), '--build', str(ROOT / 'build'), '--parallel', '2'], check=True, env=env)
    for parent in (ROOT / 'src', ROOT / 'work' / 'stage1'):
        for name in ('inputs', 'build'):
            target = parent / name
            relative = os.path.relpath(ROOT / name, parent)
            if target.is_symlink():
                target.unlink()
            if target.exists():
                raise RuntimeError(f'Refusing to replace a directory: {target}')
            target.symlink_to(relative, target_is_directory=True)
    print(ROOT / 'build' / 'pufoam_solver')

if __name__ == '__main__':
    main()
