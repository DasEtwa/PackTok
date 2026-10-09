#!/usr/bin/env python3
"""Read-only analysis of frozen M5 PTM5SEQ training inputs."""
import argparse, array, bisect, collections, hashlib, json, math, re, struct, unicodedata
from pathlib import Path

SEQ_HEADER=bytes.fromhex("50544d3553455101")
MAP_HEADER=bytes.fromhex("50544d4150340001")
RECORD=struct.Struct("<II")
PACK={0:"FLAT",1:"TEXT",2:"NUMBER",3:"STRUCTURE",65535:"BYTE_FALLBACK"}
DOMAIN={"english-drama":"english-literature","english-prose":"english-literature",
        "german-prose":"german-prose","code":"rust-source",
        "synthetic-json":"synthetic-json","synthetic-unicode":"synthetic-unicode",
        "structured":"structured"}
WORD_DOMAINS={"english-literature","german-prose","synthetic-json","synthetic-unicode"}
IDENT=re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
NUMBER=re.compile(rb"[0-9]+")
JSON_KEY=re.compile(rb'"([^"]+)":')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def mapping(path):
    data=path.read_bytes()
    if len(data)<12 or data[:8]!=MAP_HEADER: raise ValueError("mapping header")
    n=struct.unpack_from("<I",data,8)[0]
    if n!=512 or len(data)!=12+6*n: raise ValueError("mapping size/vocabulary")
    rows=[struct.unpack_from("<HI",data,12+6*i) for i in range(n)]
    ids=collections.defaultdict(set)
    for pack,local in rows: ids[pack].add(local)
    if any(v!=set(range(len(v))) for v in ids.values()): raise ValueError("mapping local IDs")
    return rows

def sequence(path):
    data=path.read_bytes()
    if len(data)<16 or data[:8]!=SEQ_HEADER: raise ValueError("PTM5SEQ header")
    n=struct.unpack_from("<Q",data,8)[0]
    if len(data)!=16+n*RECORD.size: raise ValueError("PTM5SEQ size")
    return data,n

def spans_from_manifest(manifest,split,total):
    categories={s["name"]:DOMAIN.get(s["category"],s["category"]) for s in manifest["sources"]}
    spans=[]; offset=0
    for u in manifest["units"]:
        if u["proposed"]!=split or not u["kept"]: continue
        if u["source"] not in categories or u["end"]<=u["start"]: raise ValueError("bad manifest unit")
        d=categories[u["source"]]; end=offset+u["end"]-u["start"]
        if spans and spans[-1][2]==d: spans[-1]=(spans[-1][0],end,d)
        else: spans.append((offset,end,d))
        offset=end
    if offset!=total: raise ValueError(f"manifest covers {offset}, expected {total}")
    return spans

def locate(spans,start,end,cursor):
    while cursor<len(spans) and spans[cursor][1]<=start: cursor+=1
    if cursor==len(spans) or spans[cursor][0]>start: raise ValueError("span gap")
    d=spans[cursor][2]; covered=min(end,spans[cursor][1]); i=cursor
    while covered<end:
        i+=1
        if i==len(spans) or spans[i][0]!=covered: raise ValueError("token span gap")
        if spans[i][2]!=d: return "mixed-domain",cursor
        covered=min(end,spans[i][1])
    return d,cursor

def quantiles(hist):
    total=sum(hist.values())
    if not total:return {}
    def q(p):
        target=max(1,math.ceil(total*p)); seen=0
        for value,n in sorted(hist.items()):
            seen+=n
            if seen>=target:return value
    return {"min":min(hist),"p25":q(.25),"median":q(.5),"p75":q(.75),
            "p90":q(.9),"p95":q(.95),"p99":q(.99),"max":max(hist),
            "mean":sum(k*v for k,v in hist.items())/total}

def unicode_words(raw,spans):
    for start,end,domain in spans:
        if domain not in WORD_DOMAINS: continue
        text=raw[start:end].decode("utf-8"); pos=start; begin=None; chars=[]
        for ch in text:
            width=len(ch.encode("utf-8"))
            if unicodedata.category(ch)[0] in {"L","M"}:
                if begin is None: begin=pos
                chars.append(ch)
            elif begin is not None:
                yield begin,pos,domain,"".join(chars).casefold()
                begin=None; chars=[]
            pos+=width
        if begin is not None: yield begin,pos,domain,"".join(chars).casefold()

def lexemes(raw,spans,kind):
    for start,end,domain in spans:
        pattern=None
        if kind=="rust-identifiers" and domain=="rust-source": pattern=IDENT
        elif kind=="numbers": pattern=NUMBER
        elif kind=="json-keys" and domain=="synthetic-json":
            for m in JSON_KEY.finditer(raw,start,end):
                yield m.start(1),m.end(1),domain,m.group(1).decode("utf8")
        if pattern:
            for m in pattern.finditer(raw,start,end):
                yield m.start(),m.end(),domain,m.group().decode("ascii").casefold()

def fragmentation(raw,spans,ends,kind):
    matches=unicode_words(raw,spans) if kind=="unicode-words" else lexemes(raw,spans,kind)
    domains={}
    for a,b,d,s in matches:
        pieces=bisect.bisect_left(ends,b)+1-bisect.bisect_right(ends,a)
        if pieces<=0: raise ValueError("lexeme not covered by tokens")
        r=domains.setdefault(d,{"n":0,"multi":0,"pieces":collections.Counter()})
        r["n"]+=1; r["multi"]+=pieces>1; r["pieces"][pieces]+=1
    out={}
    for d,r in sorted(domains.items()):
        n=r["n"]
        out[d]={"lexemes":n,"multi_token_percent":100*r["multi"]/n,
                "mean_tokens_per_lexeme":sum(k*v for k,v in r["pieces"].items())/n,
                "piece_histogram":{str(k):v for k,v in sorted(r["pieces"].items())}}
    return out

def analyze_variant(root,variant,raw,spans):
    seq_path=root/"data/prepared-v2"/(variant+"-train.seq")
    map_path=root/"artifacts/corpus-v2"/(variant+".mapping")
    data,n=sequence(seq_path); ids=mapping(map_path)
    hist=collections.Counter(); lengths=collections.Counter(); packs=collections.Counter()
    pack_bytes=collections.Counter(); used=collections.defaultdict(set); per_domain={}
    ends=array.array("Q"); offset=0; cursor=0; mixed_n=mixed_bytes=0
    for i in range(n):
        token,size=RECORD.unpack_from(data,16+i*8)
        if token>=len(ids) or size==0: raise ValueError("invalid token record")
        end=offset+size; d,cursor=locate(spans,offset,end,cursor); pack,local=ids[token]
        ends.append(end); hist[token]+=1; lengths[size]+=1; packs[pack]+=1; pack_bytes[pack]+=size; used[pack].add(local)
        r=per_domain.setdefault(d,{"tokens":0,"bytes":0,"lengths":collections.Counter(),
                                   "packs":collections.Counter()})
        r["tokens"]+=1; r["bytes"]+=size; r["lengths"][size]+=1; r["packs"][pack]+=1
        if d=="mixed-domain": mixed_n+=1; mixed_bytes+=size
        offset=end
    if offset!=len(raw): raise ValueError(f"sequence represents {offset} not {len(raw)} raw bytes")
    allocation=collections.Counter(p for p,_ in ids); pack_summary={}
    for p,allocated in sorted(allocation.items()):
        pack_summary[PACK.get(p,f"PACK_{p}")]={"pack_id":p,"allocated_ids":allocated,
            "observed_ids":len(used[p]),"allocated_id_occupancy_percent":100*len(used[p])/allocated,
            "emitted_tokens":packs[p],"token_share_percent":100*packs[p]/n,
            "represented_bytes":pack_bytes[p],"bytes_per_emitted_token":pack_bytes[p]/packs[p]}
    domains={}
    for d,r in sorted(per_domain.items()):
        domains[d]={"tokens":r["tokens"],"represented_bytes":r["bytes"],
            "tokens_per_represented_byte":r["tokens"]/r["bytes"],
            "represented_bytes_per_token":r["bytes"]/r["tokens"],
            "token_length_bytes":quantiles(r["lengths"]),
            "pack_tokens":{PACK.get(p,f"PACK_{p}"):v for p,v in sorted(r["packs"].items())},
            "pack_token_shares_percent":{PACK.get(p,f"PACK_{p}"):100*v/r["tokens"] for p,v in sorted(r["packs"].items())}}
    entropy=-sum((v/n)*math.log2(v/n) for v in hist.values())
    top=[{"id":t,"count":c,"share_percent":100*c/n,"pack":PACK.get(ids[t][0],f"PACK_{ids[t][0]}"),
          "local_id":ids[t][1]} for t,c in hist.most_common(20)]
    fallback=sum(c for t,c in hist.items() if (ids[t][1]<256 if variant=="A" else ids[t][0]==65535))
    return {"input_sha256":{"sequence":sha(data),"mapping":file_sha(map_path),
             "tokenizer":file_sha(root/"artifacts/corpus-v2"/(variant+"-0.packtok"))},
        "total_tokens":n,"represented_raw_bytes":offset,"tokens_per_raw_byte":n/offset,
        "raw_bytes_per_token":offset/n,"token_length_bytes":quantiles(lengths),
        "single_byte_or_fallback_tokens":fallback,"single_byte_or_fallback_percent":100*fallback/n,
        "vocabulary":{"logical_ids":len(ids),"unique_ids_observed":len(hist),
            "entropy_bits_per_token":entropy,"effective_ids_2_pow_entropy":2**entropy,
            "allocated_ids_by_pack":{PACK.get(p,f"PACK_{p}"):v for p,v in sorted(allocation.items())},
            "pack_utilization":pack_summary},
        "global_token_frequency_top20":top,"domains":domains,
        "mixed_domain_tokens":{"tokens":mixed_n,"represented_bytes":mixed_bytes},
        "lexical_fragmentation":{k:fragmentation(raw,spans,ends,k) for k in
            ("unicode-words","numbers","rust-identifiers","json-keys")}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--m5-root",required=True,type=Path,
      help="Existing M5 data/artifact root; files are read only.")
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args(); root=args.m5_root.resolve()
    repo=Path(__file__).resolve().parents[3]
    manifest_path=root/"provenance/corpus-v2-manifest.json"
    manifest_raw=manifest_path.read_bytes(); manifest=json.loads(manifest_raw)
    raw=(root/"data/corpus-v2/train.txt").read_bytes()
    train=next(x for x in manifest["splits"] if x["split"]=="train")
    if len(raw)!=train["bytes"] or sha(raw)!=train["sha256"]: raise ValueError("train raw corpus hash mismatch")
    spans=spans_from_manifest(manifest,"train",len(raw))
    pilot_path=repo/"experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T154700Z/PILOT_RESULT.json"
    pilot=json.loads(pilot_path.read_text())
    pilot_config=repo/"experiments/m5-gpu/configs/pilot-2k-v1.json"
    if file_sha(pilot_config)!=pilot["config_sha256"]: raise ValueError("pilot config hash mismatch")
    if sha(manifest_raw)!=pilot["corpus_manifest_sha256"]: raise ValueError("manifest hash mismatch")
    input_manifest=root/"provenance/bundle-input-SHA256SUMS-v5.txt"
    input_hashes={line.split()[1]:line.split()[0] for line in input_manifest.read_text().splitlines() if line.strip()}
    variants={}
    for v in ("A","C"):
        sp=root/"data/prepared-v2"/(v+"-train.seq")
        art=root/"artifacts/corpus-v2"/(v+"-0.packtok")
        if file_sha(sp)!=input_hashes["./data/"+v+"-train.seq"]:
            raise ValueError(f"{v} sequence identity mismatch")
        if file_sha(art)!=input_hashes["./artifacts/"+v+"-0.packtok"]: raise ValueError(f"{v} tokenizer identity mismatch")
        mp=root/"artifacts/corpus-v2"/(v+".mapping")
        if file_sha(mp)!=input_hashes["./artifacts/"+v+".mapping"]: raise ValueError(f"{v} mapping identity mismatch")
        variants[v]=analyze_variant(root,v,raw,spans)
    result={"schema":"packtok-m6-m5-tokenizer-efficiency-v1",
      "method":"Read-only analysis of frozen M5 TRAIN bytes and exact PTM5SEQ outputs; no validation/test data or model training.",
      "m5":{"pilot_report_sha256":file_sha(pilot_path),"pilot_source_commit":pilot["source_commit"],
        "pilot_config_sha256":pilot["config_sha256"],"corpus_manifest_sha256":sha(manifest_raw),
        "train_raw_sha256":sha(raw),"train_raw_bytes":len(raw),
        "bundle_input_sha256_manifest":file_sha(input_manifest),
        "domain_raw_bytes":{d:sum(b-a for a,b,x in spans if x==d) for d in sorted({x for _,_,x in spans})},
        "pilot_initialization_sha256":pilot["initialization_sha256"],
        "target_positions_per_variant":pilot["target_positions_expected"],
        "sampled_training_raw_bytes":{v:pilot["variants"][v]["raw_training_bytes"] for v in ("A","C")}},
      "variants":variants,"limitations":[
        "Full-split tokenization metrics are not the unique-byte coverage of stochastic M5 training samples.",
        "Tokens crossing domain boundaries are classified as mixed-domain, matching the M5 scorer.",
        "Committed M5 records have aggregate sampled raw-byte exposure only, not domain-stratified sampled exposure.",
        "The script reads existing local ignored M5 data by path and never copies or modifies those inputs."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"train_bytes":len(raw),
      "variants":{v:{"tokens":variants[v]["total_tokens"],"bytes_per_token":variants[v]["raw_bytes_per_token"],
                      "fallback_pct":variants[v]["single_byte_or_fallback_percent"]} for v in ("A","C")},},sort_keys=True))
if __name__=="__main__": main()
