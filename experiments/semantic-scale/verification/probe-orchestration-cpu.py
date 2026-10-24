"""Small operational reference with real Rust CPU training and local backup.

Installation/GPU discovery are mocked; GPU commands are explicitly routed through
`--cpu` with tiny FP32 settings. This is not a CUDA or actual Drive/FUSE test.
Run after cargo +1.90.0 build --release --manifest-path experiments/semantic-scale/Cargo.toml.
"""
from pathlib import Path
import json
import shutil
import sys
import tempfile
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/semantic-scale/tools'))
import colab_bootstrap as bootstrap
import colab_storage as storage

binary = REPO / 'experiments/semantic-scale/target/release/packtok-semantic-scale'
original_commands = bootstrap.Commands

class CpuCommands(original_commands):
    def __call__(self, arguments, **kwargs):
        args = [str(value) for value in arguments]
        if args and args[0] == str(binary) and len(args) > 1 and args[1] in ('preflight','train','evaluate'):
            args.append('--cpu')
        return super().__call__(args, **kwargs)

settings = bootstrap.Settings(context=8, accumulation=2, steps=4, train_minutes=1,
                              dtype='f32', run_name='cpu-orchestration-reference')
with tempfile.TemporaryDirectory(prefix='packtok-orchestration-cpu-') as temporary:
    root=Path(temporary); content=root/'content'; drive=root/'local-drive-fixture'
    with patch.object(bootstrap,'Commands',CpuCommands), \
         patch.object(bootstrap,'check_gpu',return_value={'name':'MOCKED_CPU_REFERENCE_ONLY'}), \
         patch.object(bootstrap,'ensure_system_packages',return_value=None), \
         patch.object(bootstrap,'ensure_cuda',return_value=None), \
         patch.object(bootstrap,'ensure_rust',return_value=Path('/unused/mock/cargo')), \
         patch.object(bootstrap,'build',return_value=binary):
        for attempt in range(2):
            preservation={}
            result=bootstrap.execute_pipeline(REPO,drive,settings,content=content,preservation=preservation)
            assert preservation['verified'] is True
            backup=drive/'runs'/settings.run_name
            pointer,index=storage.checkpoint(backup/'checkpoints',verify_payloads=True)
            assert index['step']==4
            metrics=(backup/'metrics.jsonl').read_text().splitlines()
            train_steps=[json.loads(line)['step'] for line in metrics if json.loads(line).get('stage')=='train']
            assert train_steps==[1,2,3,4],train_steps
            if attempt==0:
                # Simulate recycled local runtime; restore from verified backup.
                shutil.rmtree(content)
        print(json.dumps({'result':'PASS','attempts':2,'restored_optimizer_step':index['step'],
                          'canonical_steps':train_steps,'cuda_executed':False,'real_drive_tested':False,
                          'installation_mocked':True,'rust_training':'tiny FP32 CPU, explicit --cpu'},indent=2))
