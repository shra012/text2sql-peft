"""Summarize recorded outcomes without model access or weight changes."""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ledger = json.loads((ROOT / 'wsp/ledger.json').read_text())
trials = [json.loads(line) for line in (HERE / 'trials.jsonl').read_text().splitlines()]
screens = []
confirmations = []
for trial in trials:
    number = trial['pass']
    report = json.loads((HERE / f'report_{number:02d}.json').read_text())
    row = {'pass': number, 'name': trial['trial']['name'],
           'family': trial['trial']['family'], 'fingerprint': trial['fingerprint'],
           'comparison': report.get('matched_control', report.get('all'))}
    if trial['trial'].get('cohort') == 'screen256.txt' and row['family'] != 'control':
        screens.append(row)
    if row['family'] == 'confirmation_candidate':
        row['parent_pass'] = trial['trial']['parent_pass']
        confirmations.append(row)
screens.sort(key=lambda r: (-r['comparison']['net'], r['pass']))
summary = {
    'passes_used': ledger['used'], 'budget': ledger['budget'],
    'extra_passes_since_user_requested_more_time': ledger['used'] - 25,
    'extra_passes_by_kind': dict(collections.Counter(p['kind'] for p in ledger['passes'] if p['n'] > 25)),
    'screened_weight_candidates': len(screens),
    'positive_screen_candidates': [r for r in screens if r['comparison']['net'] > 0],
    'confirmation_candidates': confirmations,
    'screen_by_family': {},
}
for family in sorted({r['family'] for r in screens}):
    rows = [r for r in screens if r['family'] == family]
    summary['screen_by_family'][family] = {'count': len(rows), 'best': rows[0]}
for name in ('validation_selection', 'validation_aggregate', 'decision'):
    path = HERE / f'{name}.json'
    if path.exists():
        summary[name] = json.loads(path.read_text())
(HERE / 'results_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'screen_by_family'}, indent=2))
