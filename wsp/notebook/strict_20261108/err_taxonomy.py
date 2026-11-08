"""Free (no-pass) taxonomy of F4 failures on existing practice outputs, split by prompt length."""
import json, re, collections, sys
P = {json.loads(l)['id']: json.loads(l) for l in open('wsp/practice.jsonl')}
F4 = 'ae81f3cb05285df3'
L = json.load(open('wsp/ledger.json'))
recs = {}
for p in L['passes']:
    if p['kind'] != 'gen' or p['edits_fingerprint'] != F4: continue
    f = p['out'] + '/edited.jsonl'
    try:
        for l in open(f):
            r = json.loads(l)
            if r['id'] in P and not P[r['id']].get('probe_only'):
                recs[r['id']] = r
    except FileNotFoundError: pass
print('unique F4 outputs', len(recs))
def cls(r):
    e = r['error']
    if r['correct']: return 'correct'
    if not e.startswith('pred:'): return 'wrong_rows' if not e else 'other:' + e[:30]
    e = e[5:].strip()
    if 'no such column' in e: return 'no_such_column'
    if 'no such table' in e: return 'no_such_table'
    if 'syntax error' in e: return 'syntax:' + re.sub(r'near "(.*)".*', r'\1', e.split(':')[0] if False else e)[:40]
    if 'ambiguous' in e: return 'ambiguous'
    return 'pred_other:' + e[:50]
for name, sel in [('short<=2048', lambda i: P[i]['prompt_tokens'] <= 2048), ('long>2048', lambda i: P[i]['prompt_tokens'] > 2048)]:
    ids = [i for i in recs if sel(i)]
    c = collections.Counter(cls(recs[i]) for i in ids)
    n = len(ids)
    print(f'\n== {name}: n={n}')
    for k, v in c.most_common(25): print(f'  {k:60s} {v:4d} {v/n:.3f}')
    trunc = sum(len(recs[i]['raw']) > 1500 for i in ids)
    noblock = sum('```' not in recs[i]['raw'] for i in ids)
    multi = sum(recs[i]['raw'].count('```sql') > 1 for i in ids)
    print('  long raw >1500 chars', trunc, ' no ``` ', noblock, ' multiple sql blocks', multi)
