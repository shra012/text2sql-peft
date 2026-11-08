"""Build teacher-forced texts for pass 85 (non-cohort practice ids only)."""
import json, re, random, sys, collections
sys.path.insert(0,'wsp/notebook/strict_20261108')
from quote_stats import recs, P
from transformers import AutoTokenizer
from t2s.wsp.common import base_dir
tok = AutoTokenizer.from_pretrained(str(base_dir()))
held = set()
for n in 'AB': held |= {l.strip() for l in open(f'wsp/protocol/cohort_{n}.txt')}   # exclusion only
SPECIAL = {'regional_sales': ['Sales Orders','Sales Team','Store Locations'], 'airline': ['Air Carriers'],
           'disney': ['voice-actors'], 'legislator': ['current-terms','historical-terms','social-media'],
           'beer_factory': ['transaction']}
def quote(sql, tabs):
    for x in sorted(tabs, key=len, reverse=True):
        sql = re.sub(r'(?<![`"\[\w-])' + re.escape(x) + r'(?![\w`"\]-])', '`' + x + '`', sql)
    return sql
rng = random.Random(85)
pos_rows, neg_rows = [], []
for i, r in P.items():
    if i in held or r.get('probe_only'): continue
    if r['db_id'] in SPECIAL:
        tabs = SPECIAL[r['db_id']]
        if i in recs and any(t in recs[i]['raw'] for t in tabs):
            ans, src = quote(recs[i]['raw'], tabs), 'own'
        elif any(t in r['gold_sql'] for t in tabs):
            ans, src = '```sql\n' + quote(r['gold_sql'], tabs) + '\n```', 'gold'
        else: continue
        pos_rows.append((i, ans, src))
    elif i in recs:
        neg_rows.append((i, recs[i]['raw'], 'own_neg'))
rng.shuffle(pos_rows); rng.shuffle(neg_rows)
own = [x for x in pos_rows if x[2]=='own']; gold = [x for x in pos_rows if x[2]=='gold']
sel = own + gold[:774-len(own)] + neg_rows[:1024-len(own)-min(len(gold),774-len(own))]
print('own', len(own), 'gold avail', len(gold), 'neg', 1024-len(own)-len(gold), 'total', len(sel))
print(collections.Counter(P[i]['db_id'] for i,_,_ in sel).most_common(8))
with open('wsp/notebook/strict_20261108/texts_read85.jsonl','w') as f:
    for i, ans, src in sel:
        prompt = tok.apply_chat_template(P[i]['messages'], tokenize=False, add_generation_prompt=True)
        f.write(json.dumps({'id': f'{src}:{i}', 'prompt': prompt, 'answer': ans}) + '\n')
