"""Tiny independent review repros; no GPU/network or production-file edits."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
import colab_bootstrap as bootstrap
import colab_storage as storage
from test_colab_storage import generation


def restore_without_remote_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        local, drive = Path(tmp) / 'local', Path(tmp) / 'drive'
        local.mkdir()
        mirror = storage.Mirror(local, drive, 'run-a')
        (local / 'metrics.jsonl').write_text('{"stage":"train","step":1}\n')
        mirror.sync()
        generation(local / 'checkpoints', 2)
        (local / 'metrics.jsonl').write_text('{"stage":"train","step":1}\n{"stage":"train","step":2}\n')
        before = (local / 'metrics.jsonl').read_text()
        mirror.restore()
        after = (local / 'metrics.jsonl').read_text()
        print(json.dumps({'case': 'restore_without_remote_checkpoint',
                          'checkpoint_step': storage.checkpoint(local / 'checkpoints', verify_payloads=True)[1]['step'],
                          'before': before, 'after': after,
                          'lost_checkpoint_step_metric': '"step":2' not in after}), flush=True)


def descendant_watchdog():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        leader = root / 'leader.py'
        state = root / 'child.json'
        leader.write_text('''import json, os, signal, sys, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(sys.argv[1], "w") as f:
        json.dump({"group": os.getpgrp(), "child": os.getpid()}, f)
    print("descendant ready", flush=True)
    while True: time.sleep(1)
else:
    while True: time.sleep(1)
''')
        result = []
        def run():
            try:
                bootstrap.Commands(root / 'logs')([sys.executable, leader, state], timeout=1, label='descendant')
            except BaseException as error:
                result.append(type(error).__name__)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 3
        while not state.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert state.exists(), 'child did not initialize'
        process = json.loads(state.read_text())
        try:
            thread.join(timeout=18)
            print(json.dumps({'case': 'descendant_watchdog', 'timeout_seconds': 1,
                              'forced_kill_grace_seconds': 15, 'observed_after_seconds': 18,
                              'still_blocked': thread.is_alive(),
                              'child_alive': Path(f'/proc/{process["child"]}').exists(),
                              'command_result': result}), flush=True)
        finally:
            try:
                os.killpg(process['group'], signal.SIGKILL)
            except ProcessLookupError:
                pass
            thread.join(timeout=3)
            assert not thread.is_alive(), 'cleanup did not finish'

if __name__ == '__main__':
    restore_without_remote_checkpoint()
    descendant_watchdog()
