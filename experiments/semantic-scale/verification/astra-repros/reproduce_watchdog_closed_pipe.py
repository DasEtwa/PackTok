"""A descendant may keep running even if it does not keep stdout open."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
import colab_bootstrap as bootstrap

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    script = root / 'leader.py'
    state = root / 'state.json'
    script.write_text('''import json, os, signal, sys, time
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    with open(sys.argv[1], 'w') as f:
        json.dump({'group': os.getpgrp(), 'child': os.getpid()}, f)
    while True: time.sleep(1)
else:
    while True: time.sleep(1)
''')
    bootstrap._TERMINATION_GRACE_SECONDS = .2
    error = None
    try:
        bootstrap.Commands(root / 'logs')([sys.executable, script, state], timeout=.5)
    except BaseException as caught:
        error = type(caught).__name__
    process = json.loads(state.read_text())
    try:
        time.sleep(.5)
        proc = Path(f'/proc/{process["child"]}/stat')
        process_state = proc.read_text().split(') ', 1)[1].split()[0] if proc.exists() else None
        print(json.dumps({'case': 'descendant_closes_stdout', 'returned': error,
                          'child_state_after_timeout_and_grace': process_state,
                          'survived': process_state not in (None, 'Z')}), flush=True)
    finally:
        try: os.killpg(process['group'], signal.SIGKILL)
        except ProcessLookupError: pass
