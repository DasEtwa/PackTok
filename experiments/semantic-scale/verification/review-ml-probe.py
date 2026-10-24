"""Independent tiny CPU integration probe; no GPU or large allocation."""
import pathlib,tempfile,subprocess,json,hashlib
base=pathlib.Path(tempfile.mkdtemp(prefix='packtok-scale-ml-review-'))
raw=base/'raw';raw.mkdir()
for split,text in [('train','idk this original train story about trees and houses ngl'),('validation','keine Ahnung das ist der separate Prüfsatz über Autos'),('test','to be honest the final text describes a distant ocean')]:
 (raw/f'{split}.txt').write_text(''.join(f'{text} {i:08x} item {i*i+1234567}.\n' for i in range(100)))
binary=pathlib.Path('/workspace/PackTok/experiments/semantic-scale/target/debug/packtok-semantic-scale')
logs=[]
def run(*args,check=True):
 result=subprocess.run([str(binary),*map(str,args)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 logs.append('$ '+repr(args)+'\nexit='+str(result.returncode)+'\n'+result.stdout)
 if check and result.returncode: raise RuntimeError(result.stdout)
 return result
try:
 run('prepare','--input',raw,'--output',base/'prepared')
 plan=base/'plan.json';output=base/'run'
 run('plan','--data',base/'prepared','--out',plan,'--preset','tiny','--context','8','--batch','1','--accum','1','--dtype','f32','--steps','4','--max-seconds','60','--checkpoint-every','2')
 run('preflight','--config',plan,'--output',output,'--cpu')
 run('train','--config',plan,'--output',output,'--cpu')
 checkpoints=output/'checkpoints'
 indexes=[p for p in checkpoints.glob('step-*/index.json') if json.loads(p.read_text())['step']==2]
 assert indexes
 index=indexes[0]
 latest={'format':1,'directory':index.parent.name,'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest()}
 (checkpoints/'latest.json').write_text(json.dumps(latest))
 run('train','--config',plan,'--output',output,'--cpu','--resume')
 metrics=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
 steps_logged=[x['step'] for x in metrics if x.get('stage')=='train']
 logs.append('observed train step history='+repr(steps_logged)+'\n')
 print('observed step history',steps_logged)
finally:
 out=pathlib.Path('/workspace/PackTok/experiments/semantic-scale/verification/review-ml-probes.txt')
 out.write_text('Temporary probe dir='+str(base)+'\n\n'+'\n'.join(logs))
 print('probe output preserved',out)
