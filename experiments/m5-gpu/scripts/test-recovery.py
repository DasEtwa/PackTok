"""CPU regression suite with isolated fake CLI/Drive; never allocates a GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import io
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


recovery = module('l4-recovery')
drive = module('drive-backup')
remote = module('remote-recovery')

FAKE_COLAB = '''#!/usr/bin/env python3
import os,json,sys,time,shutil,signal,subprocess
from pathlib import Path
r=Path(os.environ['MOCK_ROOT']); args=sys.argv[1:]; op=args[0]
with (r/'calls').open('a') as f: f.write(' '.join(args)+'\\n')
mode=os.environ.get('MOCK_MODE','pass')
with (r/'events.jsonl').open('a') as f: f.write(json.dumps(dict(op=op,pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),epoch=time.time()))+'\\n')
def signalled(signum,frame):
 with (r/'signals.jsonl').open('a') as f: f.write(json.dumps(dict(pid=os.getpid(),signal=signum,epoch=time.time()))+'\\n')
 sys.exit(128+signum)
signal.signal(signal.SIGTERM,signalled);signal.signal(signal.SIGINT,signalled)
if op=='version': print('Version: 0.7.4')
elif op=='usage': print('Current balance: 100.00 compute units\\nUsage rate: 1.54/hr\\nActive assignments: 1')
elif op=='sessions':
 print('[LIV] unrelated | Hardware: L4 | Variant: GPU')
 if (r/'swapped').exists(): print('[packtok-m5] alien | Hardware: L4 | Variant: GPU')
 if (r/'active').exists(): print('[?] fixture | Hardware: L4 | Variant: GPU' if mode=='leak' else '[packtok-m5] fixture | Hardware: L4 | Variant: GPU')
elif op=='new':
 assert args==['new','-s','packtok-m5','--gpu','L4']; (r/'active').touch()
 if mode=='unknown': sys.exit(7)
elif op=='status': print('[packtok-m5] fixture | Hardware: '+('T4' if mode=='hardware' else 'L4')+' | Variant: GPU')
elif op=='upload':
 if mode=='upload': sys.exit(8)
elif op=='exec':
 (r/'exec-pid.json').write_text(json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),epoch=time.time())))
 if mode=='orphan':
  child=subprocess.Popen(['/usr/bin/sleep','600']); (r/'descendant-pid.txt').write_text(str(child.pid))
 time.sleep(float(os.environ.get('MOCK_EXEC_SECONDS','0')))
 if mode=='timeout': time.sleep(10)
 if mode=='exec': sys.exit(7)
elif op=='download':
 if mode=='swapped': (r/'swapped').touch()
 if mode=='missing': sys.exit(0)
 if mode=='download': sys.exit(6)
 shutil.copyfile(r/('failure.tar.gz' if mode=='loader' else 'nopass.tar.gz' if mode=='nopass' else 'pass.tar.gz'),args[4])
elif op=='stop':
 assert args==['stop','-s','packtok-m5']
 if mode!='leak': (r/'active').unlink(missing_ok=True)
else: sys.exit(99)
'''


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'experiments/m5-gpu'
        (self.root/'provenance').mkdir(parents=True)
        (self.root/'scripts').mkdir()
        shutil.copyfile(HERE/'check-preflight.py', self.root/'scripts/check-preflight.py')
        (self.root/'scripts/remote-recovery.py').touch()
        (self.root/'provenance/l4-overfit-bundle-v5').mkdir()
        receipt=self.root/'provenance/recovery-20261008/drive-fixture.json'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text('{"status":"COMPLETE_VERIFIED","kind":"preflight-fixture"}')
        (receipt.parent/'completion-readback.json').write_bytes(receipt.read_bytes())
        bundle = self.root/'bundle-v5'
        bundle.mkdir()
        payload=bundle/'packtok-m5'; payload.mkdir()
        binary=payload/'packtok-m5'; binary.write_bytes(b'fixture binary')
        binary_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
        archive=bundle/'packtok-m5-bundle.tar.gz'; archive.write_bytes(b'fixture archive')
        archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        (bundle/'packtok-m5-expected-sha256.txt').write_text(archive_sha+'\n')
        frozen=dict(bundle_id='bundle-v5',source_commit='fixture-commit',binary_sha256=binary_sha,primary_seed=20261008,
                    mode='fresh-and-post18-overfit-500',overfit_updates=500,overfit_learning_rate=0.005,
                    overfit_weight_decay=0.0,overfit_input=[1,2,3,4,1,2,3,4],overfit_target=[2,3,4,1,2,3,4,1])
        (payload/'frozen-source.json').write_text(json.dumps(frozen))
        (bundle/'CPU_READY').write_text('source_commit=fixture-commit\n')
        rows=[]
        for i in range(14):
            name=f'packtok-m5-bundle.tar.gz.part{i:03d}'
            (bundle/name).write_bytes(b'fixture')
            rows.append(name)
        checksum_paths=[bundle/'CPU_READY',archive,*[bundle/name for name in rows]]
        (bundle/'bundle.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in checksum_paths))
        source=bundle/'source.sha256'
        source.write_text(f'{binary_sha}  {binary}\n')
        cpu=self.root/'provenance/l4-overfit-bundle-v5/CPU_CHECKS_PASS.json'
        cpu.write_text(json.dumps(dict(status='CPU_CHECKS_PASS',source_commit='fixture-commit',verified_files={})))
        readiness=self.root/'provenance/l4-overfit-bundle-v5/PREFLIGHT_READY.json'
        ready=dict(status='CPU_AND_DRIVE_READY',bundle_name='bundle-v5',source_commit='fixture-commit',
                   allocation_authorization_count=1,allocation_cap_seconds=1800,full_training_authorized=False,
                   bundle_sha256=archive_sha,binary_sha256=binary_sha,
                   drive_completion=str(receipt.relative_to(self.root)),
                   verified_files={str(cpu.relative_to(self.root)):hashlib.sha256(cpu.read_bytes()).hexdigest(),
                                   str(receipt.relative_to(self.root)):hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                   str((receipt.parent/'completion-readback.json').relative_to(self.root)):hashlib.sha256((receipt.parent/'completion-readback.json').read_bytes()).hexdigest()})
        readiness.write_text(json.dumps(ready))
        self.readiness=readiness
        (self.base/'bin').mkdir()
        fake=self.base/'bin/colab'
        fake.write_text(FAKE_COLAB)
        fake.chmod(0o755)
        for name,code,gate in [('pass',0,True),('failure',127,False),('nopass',0,False)]:
            content=self.base/name/'results'
            (content/'gate').mkdir(parents=True)
            (content/'exit-code.txt').write_text(str(code)+'\n')
            (content/'loader-resolution.txt').write_text('libcurand.so.10: cannot open shared object file\n')
            (content/'gate/preflight.jsonl').write_text('{"stage":"gate","status":"PASS"}\n' if gate else '')
            with tarfile.open(self.base/(name+'.tar.gz'),'w:gz') as a: a.add(content,arcname='results')
        self.env=patch.dict(os.environ, PATH=str(self.base/'bin')+':'+os.environ['PATH'], MOCK_ROOT=str(self.base), MOCK_MODE='pass')
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def invoke(self, mode):
        os.environ['MOCK_MODE']=mode
        p=subprocess.run(['python3',str(HERE/'l4-recovery.py'),'--preflight','--root',str(self.root),
                          '--bundle-name','bundle-v5','--readiness',str(self.readiness)],capture_output=True,text=True,timeout=15)
        calls=(self.base/'calls').read_text()
        return p,calls

    def test_success_and_unrelated_session(self):
        p,calls=self.invoke('pass')
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertNotIn('stop -s LIV',calls)
        self.assertFalse((self.base/'active').exists())
        self.assertEqual(calls.count('new -s'),1)
        p2,calls2=self.invoke('pass')
        self.assertNotEqual(p2.returncode,0)
        self.assertEqual(calls2.count('new -s'),1)

    def test_transport_success_without_archive_fails(self):
        p, calls = self.invoke('missing')
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse((self.base/'active').exists())
        self.assertTrue(next((self.root/'provenance').glob('*/result-missing.txt')).exists())

    def test_success_reaps_transport_descendants(self):
        p, calls = self.invoke('orphan')
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        pid = int((self.base/'descendant-pid.txt').read_text())
        status = Path(f'/proc/{pid}/stat')
        self.assertTrue(not status.exists() or status.read_text().split()[2] == 'Z')

    def test_wrong_hardware(self):
        p,calls=self.invoke('hardware')
        self.assertNotEqual(p.returncode,0)
        self.assertNotIn('exec -s',calls)
        self.assertIn('stop -s packtok-m5',calls)

    def test_reassigned_alias_is_not_stopped(self):
        p, calls = self.invoke('swapped')
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn('stop -s', calls)
        self.assertTrue((self.base/'active').exists())
        self.assertTrue((self.root/'bundle-v5/recovery-owned.json').exists())
        self.assertTrue(next((self.root/'provenance').glob('*/release-refused-0.txt')).exists())

    def test_unknown_allocation_endpoint_is_not_stopped(self):
        p, calls = self.invoke('unknown')
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn('stop -s', calls)
        self.assertTrue((self.root/'bundle-v5/recovery-owned.json').exists())

    def test_storage_readiness_missing_blocks_allocation(self):
        self.readiness.unlink()
        p,calls=self.invoke('pass')
        self.assertNotEqual(p.returncode,0)
        self.assertNotIn('new -s',calls)

    def test_loader_failure_hidden_by_transport(self):
        p,calls=self.invoke('loader')
        self.assertNotEqual(p.returncode,0)
        self.assertFalse((self.base/'active').exists())
        verification=next((self.root/'provenance').glob('*/result-verification.txt')).read_text()
        self.assertIn('remote exit 127',verification)

    def test_missing_gate_pass(self):
        p,calls=self.invoke('nopass')
        self.assertNotEqual(p.returncode,0)
        self.assertFalse((self.base/'active').exists())

    def test_upload_failure(self):
        p,calls=self.invoke('upload')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('stop -s packtok-m5',calls)

    def test_transfer_failure(self):
        p,calls=self.invoke('download')
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(calls.count('download -s'),2)
        self.assertFalse((self.base/'active').exists())

    def test_stale_owned_release_only(self):
        (self.base/'active').touch()
        (self.root/'bundle-v5/recovery-owned.json').write_text('{"endpoint":"fixture"}')
        p,calls=self.invoke('pass')
        self.assertNotEqual(p.returncode,0)
        self.assertNotIn('new -s',calls)
        self.assertFalse((self.base/'active').exists())

    def test_unowned_alias_untouched(self):
        (self.base/'active').touch()
        p,calls=self.invoke('pass')
        self.assertNotEqual(p.returncode,0)
        self.assertNotIn('stop -s',calls)
        self.assertNotIn('new -s',calls)

    def test_endpoint_survives(self):
        p,calls=self.invoke('leak')
        self.assertNotEqual(p.returncode,0)
        self.assertTrue((self.root/'bundle-v5/recovery-owned.json').exists())

    def test_watchdog_remote_hang(self):
        os.environ['MOCK_MODE']='timeout'
        s=recovery.Supervisor(self.root,work_seconds=2,bundle_name='bundle-v5',readiness=self.readiness)
        try:
            s.execute()
            with self.assertRaises((TimeoutError,subprocess.TimeoutExpired)):
                s.request()
        finally:
            s.release()
        self.assertTrue(s.released)

    def test_consumed_old_package_identity_is_rejected_before_inventory(self):
        p=subprocess.run(['python3',str(HERE/'l4-recovery.py'),'--preflight','--root',str(self.root),
                          '--bundle-name','bundle-v4','--readiness',str(self.readiness)],
                         capture_output=True,text=True,timeout=15)
        self.assertNotEqual(p.returncode,0)
        self.assertFalse((self.base/'calls').exists())
        self.assertFalse((self.base/'active').exists())


class Backup(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.artifact=self.root/'fixture.txt'
        self.artifact.write_bytes(b'PackTok harmless deterministic fixture\n')
        self.metadata=dict(source_commit='a'*40,variant='A',seed=20261008,configuration_sha256='b'*64,
                           dataset_sha256='c'*64,tokenizer_sha256='d'*64,kind='preflight-fixture')
        self.remote={}
        self.events=[]
        self.mode='pass'
        def transport(args):
            self.events.append(args)
            if args[0]=='mkdir': return ''
            self.assertEqual(args[0],'copyto')
            self.assertIn('--immutable',args)
            src,dst=args[1:3]
            if src.startswith('packtok-drive:'):
                value=self.remote[src]
                if self.mode=='mismatch' and not src.endswith('completion.json'): value=b'corrupted'
                Path(dst).write_bytes(value)
            else:
                if self.mode=='partial' and not dst.endswith('completion.json'):
                    self.remote[dst]=Path(src).read_bytes()[:3]
                    raise RuntimeError('partial upload')
                self.assertNotIn(dst,self.remote)
                self.remote[dst]=Path(src).read_bytes()
            return ''
        self.patch1=patch.object(drive,'run',transport)
        self.patch1.start()
        self.patch2=patch.object(drive.subprocess,'run',lambda *a,**k: subprocess.CompletedProcess(a,0,'packtok-drive: drive\n',''))
        self.patch2.start()

    def tearDown(self):
        self.patch2.stop(); self.patch1.stop(); self.temp.cleanup()

    def test_backup_recovery_success(self):
        record=drive.backup(self.artifact,self.metadata,self.root/'download')
        self.assertEqual(record['status'],'COMPLETE_VERIFIED')
        self.assertEqual(drive.sha(self.artifact),drive.sha(self.root/'download/fixture.txt'))
        manifest=[i for i,e in enumerate(self.events) if e[0]=='copyto' and e[2].endswith('completion.json')][0]
        download=[i for i,e in enumerate(self.events) if e[0]=='copyto' and e[1].startswith('packtok-drive:')][0]
        self.assertGreater(manifest,download)
        self.assertFalse(record['exact_training_resume'])

    def test_authentication_unavailable(self):
        with patch.object(drive.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')):
            with self.assertRaisesRegex(RuntimeError,'authentication unavailable'):
                drive.backup(self.artifact,self.metadata,self.root/'download')
        self.assertEqual(self.remote,{})

    def test_partial_checkpoint_upload(self):
        self.mode='partial'
        with self.assertRaisesRegex(RuntimeError,'partial upload'):
            drive.backup(self.artifact,self.metadata,self.root/'download')
        self.assertFalse(any(k.endswith('completion.json') for k in self.remote))

    def test_checkpoint_integrity_mismatch(self):
        self.mode='mismatch'
        with self.assertRaisesRegex(RuntimeError,'integrity mismatch'):
            drive.backup(self.artifact,self.metadata,self.root/'download')
        self.assertFalse(any(k.endswith('completion.json') for k in self.remote))

    def test_good_checkpoint_retention(self):
        self.remote['packtok-drive:PackTok/M5/preflight/old-good/model.safetensors']=b'known-good'
        self.mode='partial'
        with self.assertRaises(RuntimeError): drive.backup(self.artifact,self.metadata,self.root/'download')
        self.assertEqual(self.remote['packtok-drive:PackTok/M5/preflight/old-good/model.safetensors'],b'known-good')
        self.assertFalse(any(e[0] in ['sync','delete','purge'] for e in self.events))


class RemoteDiagnostics(unittest.TestCase):
    def test_early_loader_failure_archived_without_starting_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            root=base/'packtok-m5'
            (root/'runtime').mkdir(parents=True)
            (root/'packtok-m5').touch()
            (root/'frozen-source.json').write_text('{"bundle_id":"bundle-v5","source_commit":"frozen","mode":"fresh-and-post18-overfit-500"}')
            # The fixture unpack is simulated; the transport identity is real.
            archive_bytes=b'transport-fixture'
            for i in range(14):
                (base/f'packtok-m5-bundle.tar.gz.part{i:03d}').write_bytes(archive_bytes if i==0 else b'')
            (base/'packtok-m5-expected-sha256.txt').write_text(hashlib.sha256(archive_bytes).hexdigest()+'\n')
            stages=[]
            def command(name,args,env=None,timeout=30):
                stages.append(name)
                (base/'results'/f'{name}.txt').write_text('missing libcurand.so.10\n' if name.startswith('loader') else 'fixture\n')
                return 127 if name=='loader-resolution' else 0
            with patch.object(remote,'BASE',base), patch.object(remote,'RESULTS',base/'results'), \
                 patch.object(remote,'command',command), \
                 patch.object(remote.subprocess,'check_output',return_value='NVIDIA L4\n'):
                with self.assertRaisesRegex(RuntimeError,'remote failure 127'):
                    remote.main()
            self.assertNotIn('rust-console',stages)
            with tarfile.open(base/'packtok-m5-results.tar.gz') as archive:
                self.assertEqual(archive.extractfile('results/exit-code.txt').read(),b'127\n')
                self.assertIn(b'libcurand',archive.extractfile('results/loader-resolution.txt').read())

    def test_transport_bootstrap_failure_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            with patch.object(remote,'BASE',base), patch.object(remote,'RESULTS',base/'results'):
                with self.assertRaisesRegex(RuntimeError,'remote failure 1'):
                    remote.main()
            with tarfile.open(base/'packtok-m5-results.tar.gz') as archive:
                self.assertIn(b'bundle parts',archive.extractfile('results/failure.txt').read())


class ResultVerification(unittest.TestCase):
    def test_threshold_failure_is_preserved_without_becoming_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            gate=base/'results/gate'; gate.mkdir(parents=True)
            rows=[dict(stage='tiny-overfit-summary',state=state,passed_original_threshold=False)
                  for state in ['fresh','post-18']]
            rows.append(dict(stage='gate',status='FAIL'))
            (gate/'preflight.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (base/'results/exit-code.txt').write_text('1\n')
            archive_path=base/'results.tar.gz'
            with tarfile.open(archive_path,'w:gz') as archive:
                archive.add(base/'results',arcname='results')
            checked=subprocess.run(['python3',str(HERE/'check-preflight.py'),str(archive_path)],
                                   capture_output=True,text=True,timeout=10)
            self.assertNotEqual(checked.returncode,0)
            self.assertIn('Valid CUDA overfit failure preserved',checked.stderr)
            rows[-1]['status']='PASS'
            (gate/'preflight.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (base/'results/exit-code.txt').write_text('0\n')
            with tarfile.open(archive_path,'w:gz') as archive:
                archive.add(base/'results',arcname='results')
            passed=subprocess.run(['python3',str(HERE/'check-preflight.py'),str(archive_path)],
                                  capture_output=True,text=True,timeout=10)
            self.assertEqual(passed.returncode,0,passed.stdout+passed.stderr)


if __name__=='__main__':
    unittest.main(verbosity=2)
