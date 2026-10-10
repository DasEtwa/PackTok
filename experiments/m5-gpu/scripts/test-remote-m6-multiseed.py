#!/usr/bin/env python3
import importlib.util, os, shutil, tarfile, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; M5=ROOT/"experiments/m5-gpu"
BRIDGE=M5/"scripts/remote-bridge.py"; REMOTE=M5/"scripts/remote-m6-multiseed.sh"
def write(p,s,x=False):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s)
 if x:p.chmod(0o755)
def prep(root):
 pkg=root/"packtok-m5"; pkg.mkdir(parents=True); shutil.copy2(REMOTE,pkg/"remote.sh")
 write(pkg/"frozen-source.json",'{"test":"rehearsal"}\\n')
 for f in ["configs/m6-five-seed-2k-v1.json","configs/primary.json","provenance/corpus-v2-manifest.json",
 *[f"data/prepared-v2/{v}-{s}.seq" for v in "AC" for s in ("train","validation","test")],
 "artifacts/corpus-v2/A-0.packtok","artifacts/corpus-v2/C-0.packtok","artifacts/corpus-v2/A.mapping","artifacts/corpus-v2/C.mapping"]:write(pkg/f,"fixture\\n")
 write(pkg/"runtime/ld-linux-x86-64.so.2",'''#!/usr/bin/env bash
set -e
if [[ ${1:-} == --library-path ]]; then shift 2; fi
if [[ ${1:-} == --list ]]; then exit 0; fi
exec "$@"
''',True)
 write(pkg/"packtok-m5",'''#!/usr/bin/env python3
import json,os,pathlib,sys
a=sys.argv[1:]
if a[0]=="plan-extended":pathlib.Path(a[3]).write_text("{}");print("plan reached");raise SystemExit()
_,config,data,out,var,regime,seed,approval=a
assert os.environ.get("PACKTOK_M5_GPU_APPROVAL")==approval
pathlib.Path("results/invocations.txt").open("a").write(" ".join(a)+"\\n")
if os.environ.get("MOCK_FAIL")==f"{seed}:{var}":print("mock error",file=sys.stderr);raise SystemExit(9)
p=pathlib.Path(out);p.mkdir(parents=True,exist_ok=True)
rows=[{"stage":"validation","step":s} for s in (1,500,1000,1500,2000)]+[{"stage":"final","updates":2000,"targets":4096000}]
(p/"metrics.jsonl").write_text("".join(json.dumps(r,separators=(",",":"))+"\\n" for r in rows));(p/"latest.resume.safetensors").write_bytes(b"x")
''',True)
 bind=root/"bin";write(bind/"nvidia-smi",'''#!/usr/bin/env bash
if [[ " $* " == *' --loop-ms '* ]];then exec sleep 3600;fi
if [[ " $* " == *'--query-gpu=name --format=csv,noheader'* ]];then echo "NVIDIA L4";else echo "NVIDIA L4, mock, 24GiB";fi
''',True)
 tarpath=root/"packtok-m5-bundle.tar.gz"
 with tarfile.open(tarpath,"w:gz") as t:t.add(pkg,arcname="packtok-m5")
 (root/"packtok-m5-bundle.tar.gz.part000").write_bytes(tarpath.read_bytes())
 return bind
def execute(root,bind,approval,extra=None):
 spec=importlib.util.spec_from_file_location("bridge",BRIDGE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 env=os.environ.copy();env["PATH"]=str(bind)+os.pathsep+env["PATH"];env["PACKTOK_M5_RESULTS_ARCHIVE"]=str(root/"packtok-m5-results.tar.gz");env.pop("PACKTOK_M5_GPU_APPROVAL",None)
 if approval:env["PACKTOK_M5_GPU_APPROVAL"]=approval
 if extra:env.update(extra)
 return mod.run(root=root,env=env)
def files(root):
 with tarfile.open(root/"packtok-m5-results.tar.gz") as t:return {"/".join(Path(i.name).parts[1:]):t.extractfile(i).read() for i in t if i.isfile()}
class Rehearsal(unittest.TestCase):
 def test_success(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);b=prep(r);self.assertEqual(execute(r,b,"APPROVAL"),0);f=files(r)
   self.assertEqual(f["exit-code.txt"],b"0\n");self.assertIn(b"M6_FOUR_NEW_PAIRED_SEEDS_COMPLETE",f["pilot-status.txt"])
   inv=f["invocations.txt"].decode().splitlines();self.assertEqual(len(inv),8)
   self.assertIn("configs/m6-five-seed-2k-v1.json",inv[0]);self.assertIn("results/seed-20261009/A A T 20261009 APPROVAL",inv[0])
 def test_failure(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);b=prep(r);self.assertEqual(execute(r,b,"APPROVAL",{"MOCK_FAIL":"20261009:C"}),9);f=files(r)
   self.assertEqual(f["exit-code.txt"],b"9\n");self.assertIn(b"mock error",f["seed-20261009/C/console.txt"])
 def test_missing_approval(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);b=prep(r);self.assertEqual(execute(r,b,None),1);self.assertIn(b"required PACKTOK_M5_GPU_APPROVAL",files(r)["bootstrap-console.txt"])
if __name__=="__main__":unittest.main(verbosity=2)

