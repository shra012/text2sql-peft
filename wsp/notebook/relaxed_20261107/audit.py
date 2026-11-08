"""Check ledger, cohorts, and archived edit provenance without model inference."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ledger = json.loads((ROOT / 'wsp/ledger.json').read_text())
assert ledger['used'] <= ledger['budget'] == 100
assert [p['n'] for p in ledger['passes']] == list(range(1, ledger['used'] + 1))
rows = [json.loads(r) for r in (ROOT / 'wsp/practice.jsonl').read_text().splitlines()]
by_id = {r['id']: r for r in rows}
assert len(by_id) == len(rows)
normal = {i for i, r in by_id.items() if not r.get('probe_only')}
assert len(normal) == 8761
cohorts = {}
for name in ('screen256', 'confirm256', 'validate512'):
    ids = (HERE / f'{name}.txt').read_text().splitlines()
    assert len(ids) == len(set(ids)) and set(ids) <= normal
    cohorts[name] = set(ids)
assert len(cohorts['screen256']) == len(cohorts['confirm256']) == 256
assert len(cohorts['validate512']) == 512
assert not (cohorts['screen256'] & cohorts['confirm256'])
assert not (cohorts['validate512'] & (cohorts['screen256'] | cohorts['confirm256']))
assert cohorts['screen256'] | cohorts['confirm256'] == set((HERE / 'screen512.txt').read_text().splitlines())
passes = {p['n']: p for p in ledger['passes']}
trials = [json.loads(r) for r in (HERE / 'trials.jsonl').read_text().splitlines()]
archived_count = 0
recipe_only = []
for trial in trials:
    number = trial['pass']
    archive = HERE / 'candidates' / f'pass_{number:02d}'
    assert trial['fingerprint'] == passes[number]['edits_fingerprint']
    if not (archive / 'manifest.json').exists():
        assert number in (30, 31, 32, 33, 34)
        recipe_only.append(number)
        continue
    archived_count += 1
    manifest = json.loads((archive / 'manifest.json').read_text())
    assert manifest['fingerprint'] == trial['fingerprint'] == passes[number]['edits_fingerprint']
    assert manifest['trial'] == trial['trial']
    names = sorted(p.name[:-len('.safetensors')] for p in archive.glob('*.safetensors'))
    assert names == passes[number]['edits']
    assert all(not name.startswith(('lm_head.', 'model.embed_tokens.', 'model.norm.')) for name in names)
# Hash the accepted/final files and explicitly confirmed candidate archives.
checked = {}
for number in {37, 45} | {r['pass'] for r in trials if r['trial']['family'].startswith(('confirmation', 'validation'))}:
    archive = HERE / 'candidates' / f'pass_{number:02d}'
    if not archive.exists():
        continue
    digest = hashlib.sha256()
    for tensor in sorted(archive.glob('*.safetensors')):
        digest.update(tensor.name[:-len('.safetensors')].encode())
        digest.update(tensor.read_bytes())
    fingerprint = digest.hexdigest()[:16]
    assert fingerprint == passes[number]['edits_fingerprint']
    checked[str(number)] = fingerprint
result = {'status': 'passed', 'passes_used': ledger['used'], 'budget': ledger['budget'],
          'normal_practice_questions': len(normal), 'probe_questions': len(rows) - len(normal),
          'cohorts_disjoint': True, 'archived_trial_metadata_checked': archived_count,
          'early_rejected_trials_with_recipe_only': recipe_only,
          'archived_tensor_hashes_checked': checked}
(HERE / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
