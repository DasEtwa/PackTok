"""Two same-name local runtimes sharing one Drive run; no GPU or network."""
import json
from pathlib import Path
import sys
import tempfile
TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
import colab_storage as storage
from test_colab_storage import generation
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    a, b, drive = root / 'runtime-a', root / 'runtime-b', root / 'drive'
    image_a, pointer_a = generation(a / 'checkpoints', 2)
    generation(b / 'checkpoints', 1)
    mirror_a = storage.Mirror(a, drive, 'same-run')
    mirror_b = storage.Mirror(b, drive, 'same-run')
    mirror_a.sync(final=True)
    before = storage.checkpoint(drive / 'checkpoints', verify_payloads=True)[1]['step']
    caught = None
    try:
        mirror_b.sync(final=True)
    except BaseException as error:
        caught = type(error).__name__ + ': ' + str(error)
    after = storage.checkpoint(drive / 'checkpoints', verify_payloads=True)[1]['step']
    print(json.dumps({'drive_step_before': before, 'drive_step_after': after,
                      'newer_checkpoint_preserved': (drive / 'checkpoints' / pointer_a['directory']).exists(),
                      'stale_mirror_verified': mirror_b.verified, 'error': caught}), flush=True)
