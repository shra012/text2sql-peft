"""Upper bound for a COLUMN-quoting fix: failed F4 answers with an unquoted special column; quote only columns."""
import json, re, sys, collections
sys.path.insert(0, 'wsp/notebook/strict_20261108')
from quote_stats import recs, P, specials, unquoted
from t2s.bird_eval import evaluate_batch_detailed
def main():
    items, meta = [], []
    allr = dict(recs)
    for f in ['wsp/passes/pass_86/edited.jsonl']:
        for l in open(f):
            r = json.loads(l); allr[r['id']] = r
    for i, r in allr.items():
        t, cl = specials(P[i]['messages'][1]['content'])
        uq = [x for x in sorted(cl, key=len, reverse=True) if unquoted(r['pred_sql'], x)]
        if not uq or r['correct']: continue
        sql = r['pred_sql']
        for x in uq: sql = re.sub(r'(?<![`"\[\w])' + re.escape(x) + r'(?![\w`"\]])', '`' + x + '`', sql)
        items.append({'db_path': P[i]['db_path'], 'pred_sql': sql, 'gold_sql': P[i]['gold_sql']}); meta.append((i, P[i]['db_id']))
    res = evaluate_batch_detailed(items, num_workers=8)
    print('n questions', len(allr), 'failed with unquoted special column', len(items), 'correct after quoting columns', sum(c for c, _ in res))
    print(collections.Counter(db for (i, db), (c, _) in zip(meta, res) if c))
if __name__ == '__main__': main()
