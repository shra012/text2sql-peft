"""Paired comparison of a gen pass (snapshot reference vs edited), split by special-table DB vs other."""
import json, sys, re, collections
d = sys.argv[1]
P = {json.loads(l)['id']: json.loads(l) for l in open('wsp/practice.jsonl')}
ref = [json.loads(l) for l in open(f'{d}/snapshot_F4.jsonl')]; ed = {json.loads(l)['id']: json.loads(l) for l in open(f'{d}/edited.jsonl')}
SPECIAL = {'regional_sales': ['Sales Orders','Sales Team','Store Locations'], 'airline': ['Air Carriers'], 'disney': ['voice-actors'],
           'legislator': ['current-terms','historical-terms','social-media'], 'beer_factory': ['transaction']}
def unq(sql, tabs): return sum(len(re.findall(r'(?<![`"\[\w-])' + re.escape(t) + r'(?![\w`"\]-])', sql)) for t in tabs)
def syn(e): return 'syntax error' in e
for half in ('special', 'other'):
    c = collections.Counter()
    for r in ref:
        sp = P[r['id']]['db_id'] in SPECIAL
        if (half == 'special') != sp: continue
        e = ed[r['id']]; c['n'] += 1
        c['changed'] += r['raw'] != e['raw']
        c['fixed'] += e['correct'] and not r['correct']; c['broke'] += r['correct'] and not e['correct']
        c['F4 correct'] += r['correct']; c['Q correct'] += e['correct']
        c['F4 syntax'] += syn(r['error']); c['Q syntax'] += syn(e['error'])
        if sp:
            tabs = SPECIAL[P[r['id']]['db_id']]
            c['F4 unquoted special'] += unq(r['pred_sql'], tabs); c['Q unquoted special'] += unq(e['pred_sql'], tabs)
            c['F4 quoted special'] += sum(('`' + t + '`') in r['pred_sql'] for t in tabs); c['Q quoted special'] += sum(('`' + t + '`') in e['pred_sql'] for t in tabs)
        c['F4 backticked FROM'] += len(re.findall(r'(FROM|JOIN) `', r['pred_sql'])); c['Q backticked FROM'] += len(re.findall(r'(FROM|JOIN) `', e['pred_sql']))
    print(half, dict(c))
if len(sys.argv) > 2:
    for r in ref:
        e = ed[r['id']]
        if r['correct'] != e['correct']:
            print('FIXED' if e['correct'] else 'BROKE', r['id'], P[r['id']]['db_id']); print('  F4:', r['pred_sql'][:250].replace('\n', ' '), '|', r['error'][:60]); print('  Q :', e['pred_sql'][:250].replace('\n', ' '), '|', e['error'][:60])
