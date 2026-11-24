"""Offline analysis of the small M6 fixture after release. Input is the supervisor result archive."""
import argparse
import hashlib
import json
import math
import re
import struct
import tarfile
from pathlib import Path


def compare(a,b):
    if len(a)!=len(b) or not a:
        raise ValueError("incompatible vector lengths")
    differences = [abs(x-y) for x,y in zip(a,b)]
    if not all(math.isfinite(x) for x in [*a,*b]):
        raise ValueError("nonfinite output")
    return dict(count=len(a),max_abs=max(differences),outside=sum(
        d > 1e-5 + 1e-3*abs(x) for d,x in zip(differences,a)))


def checkpoint(raw):
    if raw[:8]!=b"PKTRN002" or hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("invalid v2 image")
    pos=12+struct.unpack_from("<I",raw,8)[0]
    meta=json.loads(raw[12:pos])
    groups=[[],[],[]]
    for _,shape in meta["layout"]:
        size=math.prod(shape)
        for group in groups:
            group.extend(struct.unpack_from(f"<{size}f",raw,pos))
            pos+=4*size
    if pos!=len(raw)-32:
        raise ValueError("extra checkpoint data")
    return meta,groups


def analyze(get,backend):
    if get("exit-code.txt").strip()!=b"0":
        raise ValueError("remote fixture failed")
    logs={key:get(key+".log").decode() for key in ["control","interrupt","resume"]}
    def traces(log):
        return [row.split(" ",5) for row in re.findall(r"TRACE ([^\n]+)",log)]
    a=traces(logs["control"])
    b=traces(logs["interrupt"])+traces(logs["resume"])
    def exact(rows):
        return [[*r[:3],*r[4:]] for r in rows]
    def losses(rows):
        return [struct.unpack("<f",struct.pack("<I",int(r[3])))[0] for r in rows]
    processes=[json.loads(l) for l in get("processes.jsonl").splitlines()]
    if [p["exit"] for p in processes]!=[0,-9,0] or len({p["pid"] for p in processes})!=3:
        raise ValueError("unexpected process boundaries")
    step2="step-00000000000000000002.ptc"
    step4="step-00000000000000000004.ptc"
    raw=[get(d+"/"+step4) for d in ["full","split"]]
    (m1,s1),(m2,s2)=map(checkpoint,raw)
    metrics={"loss":compare(losses(a),losses(b))}
    for name,c,r in zip(["parameters","first_moments","second_moments"],s1,s2):
        metrics[name]=compare(c,r)
    checks=dict(restore_exact=get("audit/"+step2)==get("split/"+step2) and
                "RESTORE_BYTES_EXACT" in logs["resume"],
                trace_equal=exact(a)==exact(b) and [r[0] for r in a]==["1","2","3","4"],
                metadata_equal=m1==m2 and m1["position"]["step"]==4 and m1["spec"]["backend"]==backend,
                finite_within_tolerance=all(v["outside"]==0 for v in metrics.values()))
    sizes={k:[list(map(int,row)) for row in re.findall(r"PUBLISH_US (\d+) BYTES (\d+)",v)] for k,v in logs.items()}
    summary=dict(passed=all(checks.values()),checks=checks,checkpoint_bitwise_equal=raw[0]==raw[1],
                 loss_bits_equal=[r[3] for r in a]==[r[3] for r in b],metrics=metrics,
                 checkpoint_sha256=[hashlib.sha256(x).hexdigest() for x in raw],processes=processes,
                 publish_us_bytes=sizes,restore_us=int(re.search(r"RESTORE_US (\d+)",logs["resume"])[1]))
    if backend=="cuda":
        summary["hardware"]=get("hardware.txt").decode().strip()
        samples=[int(v) for v in re.findall(r",\s*(\d+)\s*$", get("vram.csv").decode(),re.M)]
        summary["vram_samples"]=len(samples)
        summary["peak_sampled_device_mib"]=max(samples,default=None)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    args=p.parse_args()
    with tarfile.open(args.archive) as tar:
        def read(name):
            item=tar.getmember("m6-result/"+name)
            if not item.isfile() or item.size>2**20:
                raise ValueError(name)
            return tar.extractfile(item).read()
        result=analyze(read,"cuda")
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 1)
