"""Checks original toolkit admission without a GPU, downloads or compilation."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[4]
REVISION = '8c5b0fb00e6eca4c5bdc6e0ace7a0b8437d25b28'
source = subprocess.run(['git', 'show', f'{REVISION}:experiments/semantic-scale/tools/colab_bootstrap.py'],
                        cwd=REPO, check=True, capture_output=True, text=True).stdout
module = ast.parse(source)
function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == 'cuda_toolkit_usable')
namespace = {'Path': Path}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'original-toolkit-admission', 'exec'), namespace)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for name in ('bin/nvcc', 'include/cuda_runtime.h', 'include/cublas_v2.h'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    print(json.dumps({'revision': REVISION, 'admitted': namespace['cuda_toolkit_usable'](root),
                      'library_files': [str(path.relative_to(root)) for path in root.rglob('*.so*')]}))
