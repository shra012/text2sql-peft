import json, collections, sys
sys.path.insert(0,'wsp/notebook')
from analyze_nsc import classify, P
F4='ae81f3cb05285df3'
L=json.load(open('wsp/ledger.json')); recs={}
for p in L['passes']:
    if p['kind']=='gen' and p['edits_fingerprint']==F4:
        try:
            for l in open(p['out']+'/edited.jsonl'):
                r=json.loads(l)
                if not P[r['id']].get('probe_only'): recs[r['id']]=r
        except FileNotFoundError: pass
for long in (False, True):
    c=collections.Counter(); n=0
    for i,r in recs.items():
        if (P[i]['prompt_tokens']>2048)!=long: continue
        n+=1; k=classify(r)
        if k: c[k[0]]+=1
    print('long' if long else 'short', n, dict(c))
