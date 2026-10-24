"""Recreate interrupted first checkout using only a temporary local Git origin."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[4]
notebook = json.loads((REPO / 'notebooks/PackTok_Semantic_Scale_Colab.ipynb').read_text())
functions = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        functions.extend(node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'checkout_source')
namespace = {'Path': Path, 're': re, 'subprocess': subprocess}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'checkout-source', 'exec'), namespace)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    origin, destination = root / 'origin', root / 'destination'
    def git(*args, cwd=None):
        return subprocess.run(['git', *map(str, args)], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    git('init', origin)
    (origin / 'example.txt').write_text('tracked content')
    git('add', 'example.txt', cwd=origin)
    git('-c', 'user.name=PackTok Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture', cwd=origin)
    commit = git('rev-parse', 'HEAD', cwd=origin)
    # This is the on-disk state after successful clone and failed first fetch.
    git('clone', '--no-checkout', origin, destination)
    print(json.dumps({'case': 'prior_clone_before_failed_first_fetch',
                      'worktree_entries': sorted(p.name for p in destination.iterdir()),
                      'index_exists': (destination / '.git/index').exists()}), flush=True)
    try:
        namespace['checkout_source'](destination, commit, str(origin))
    except subprocess.CalledProcessError as error:
        print(json.dumps({'recovery': 'failed', 'argv': error.cmd, 'returncode': error.returncode}), flush=True)
    else:
        print(json.dumps({'recovery': 'passed', 'tracked_file_exists': (destination / 'example.txt').exists()}), flush=True)
