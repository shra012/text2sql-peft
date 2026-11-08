import json, sys
P = {json.loads(l)['id']: json.loads(l) for l in open('wsp/practice.jsonl')}
F4='ae81f3cb05285df3'
L=json.load(open('wsp/ledger.json')); recs={}
for p in L['passes']:
    if p['kind']=='gen' and p['edits_fingerprint']==F4:
        try:
            for l in open(p['out']+'/edited.jsonl'):
                r=json.loads(l)
                if not P[r['id']].get('probe_only'): recs[r['id']]=r
        except FileNotFoundError: pass
pat=sys.argv[1]
for i,r in recs.items():
    if pat in r['error']:
        print(i, P[i]['db_id'], P[i]['prompt_tokens'], r['error'][:90]); print('   PRED', r['pred_sql'][:300].replace('\n',' ')); print('   GOLD', P[i]['gold_sql'][:300])
