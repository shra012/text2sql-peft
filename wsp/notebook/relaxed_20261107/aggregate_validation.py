"""Apply the recorded promotion rule to all four completed frozen-edit folds."""
import json
from pathlib import Path
from report import paired, records

HERE = Path(__file__).resolve().parent
trials = [json.loads(line) for line in (HERE / 'trials.jsonl').read_text().splitlines()]
controls = {r['trial']['fold']: r for r in trials if r['trial']['family'] == 'validation_control'}
candidates = {r['trial']['fold']: r for r in trials if r['trial']['family'] == 'validation_candidate'}
assert controls.keys() == candidates.keys() == {1, 2, 3, 4}
frozen = json.loads((HERE / 'frozen_candidate.json').read_text())
reference = {}
candidate = {}
folds = []
for fold in range(1, 5):
    control, edit = controls[fold], candidates[fold]
    assert edit['fingerprint'] == frozen['fingerprint']
    assert control['fingerprint'] == 'ae81f3cb05285df3'
    r, c = records(control['pass']), records(edit['pass'])
    assert r.keys() == c.keys() and len(r) == 512
    assert not (r.keys() & reference.keys())
    reference.update(r)
    candidate.update(c)
    folds.append({'fold': fold, 'control_pass': control['pass'], 'candidate_pass': edit['pass'], **paired(r, c)})
result = {'frozen_candidate': frozen, 'all': paired(reference, candidate), 'folds': folds}
for prefix, name in (('p', 'short'), ('x', 'long')):
    result[name] = paired(reference, {i: r for i, r in candidate.items() if i.startswith(prefix)})
rule = json.loads((HERE / 'validation_protocol.json').read_text())['promotion_gate']
positive_folds = sum(fold['net'] > 0 for fold in folds)
passed = (result['all']['net'] >= rule['minimum_pooled_net'] and
          positive_folds >= rule['minimum_positive_folds'] and
          result['all']['exact_p'] <= rule['maximum_two_sided_exact_mcnemar_p'])
result['gate'] = {'passed': passed, 'positive_folds': positive_folds, 'rules': rule}
assert result['all']['n'] == 2048
(HERE / 'validation_aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
