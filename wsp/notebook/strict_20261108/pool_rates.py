"""Pool-level prevalence of special identifiers (non-cohort practice ids only; cohort ids are never read)."""
import json, re, collections, sys
sys.path.insert(0,'wsp/notebook/strict_20261108')
from quote_stats import specials
held=set()
for n in 'AB':
    # only used to EXCLUDE ids, never to read their content
    held |= {l.strip() for l in open(f'wsp/protocol/cohort_{n}.txt')}
c=collections.Counter(); dbs=collections.Counter()
for l in open('wsp/practice.jsonl'):
    r=json.loads(l)
    if r['id'] in held or r.get('probe_only'): continue
    lg = r['prompt_tokens']>2048
    t,cl=specials(r['messages'][1]['content'])
    c['n',lg]+=1
    if any(x in r['gold_sql'] for x in t): c['gold_special_table',lg]+=1
    if any(x in r['gold_sql'] for x in t|cl): c['gold_special_any',lg]+=1; dbs[r['db_id'],lg]+=1
for k in sorted(c): print(k,c[k], round(c[k]/c['n',k[1]],3))
print(dbs.most_common(25))
