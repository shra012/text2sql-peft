"""Freeze one confirmed candidate before four disjoint validation cohorts."""
import json
from pathlib import Path
import random
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ledger=json.loads((ROOT/"wsp/ledger.json").read_text())
selection=json.loads((HERE/"confirmation_selection.json").read_text())
screens={r["pass"]:r for r in selection["selected"]}
eligible=[]
for trial in map(json.loads,(HERE/"trials.jsonl").read_text().splitlines()):
    if trial["trial"]["family"]!="confirmation_candidate":continue
    parent=trial["trial"]["parent_pass"]
    result=json.loads((HERE/f"report_{trial['pass']:02d}.json").read_text())["all"]
    if result["net"]>0:
        eligible.append({"parent_pass":parent,"confirmation_pass":trial["pass"],"fingerprint":trial["fingerprint"],
                         "screen":screens[parent]["screen"],"confirmation":result,
                         "pooled_net":screens[parent]["screen"]["net"]+result["net"]})
eligible.sort(key=lambda r:(-r["pooled_net"],-r["confirmation"]["net"],r["parent_pass"]))
if not eligible:
    (HERE/"validation_selection.json").write_text(json.dumps({"selected":None,"reason":"No positive screening candidate improved the separate confirmation cohort."},indent=2)+"\n")
    print("No candidate survived confirmation. Retain original F4.")
    raise SystemExit(0)
selected=eligible[0]
assert ledger["budget"]-ledger["used"]>=8
assert not (HERE/"frozen_candidate.json").exists()
(HERE/"frozen_candidate.json").write_text(json.dumps(selected,indent=2)+"\n")
shutil.copytree(HERE/"candidates"/f"pass_{selected['parent_pass']:02d}",HERE/"frozen_edit")
used=set()
for entry in ledger["passes"]:
    if entry["kind"]!="gen":continue
    for kind in ("original","edited"):
        path=Path(entry["out"])/f"{kind}.jsonl"
        if path.exists():used.update(r["id"] for r in map(json.loads,path.read_text().splitlines()) if r["id"].startswith(("p","x")))
first=(HERE/"validate512.txt").read_text().splitlines()
assert len(first)==512 and not (set(first)&used)
cohorts=[first];used.update(first)
normal=[r for r in map(json.loads,(ROOT/"wsp/practice.jsonl").read_text().splitlines()) if not r.get("probe_only") and r["id"] not in used]
rng=random.Random(3192607)
short=sorted(r["id"] for r in normal if r["id"].startswith("p"));rng.shuffle(short)
long=sorted(r["id"] for r in normal if r["id"].startswith("x"));rng.shuffle(long)
for fold in range(3):
    cohort=short[fold*256:(fold+1)*256]+long[fold*256:(fold+1)*256]
    assert len(cohort)==512 and not (set(cohort)&used)
    rng.shuffle(cohort);cohorts.append(cohort);used.update(cohort)
plan=[];start=ledger["used"]+1
for fold,cohort in enumerate(cohorts,1):
    filename=f"validation_fold{fold}.txt"
    (HERE/filename).write_text("\n".join(cohort)+"\n")
    control=start+(fold-1)*2
    plan.append({"name":f"validation{fold}_F4","script":"restore_f4.py","args":[],"family":"validation_control","fold":fold,"cohort":filename,"baseline":control})
    plan.append({"name":f"validation{fold}_candidate","script":"apply_archived.py","args":["--pass-number",str(selected["parent_pass"])],"family":"validation_candidate","parent_pass":selected["parent_pass"],"fold":fold,"cohort":filename,"baseline":control})
(HERE/"plan_validation.json").write_text(json.dumps(plan,indent=2)+"\n")
(HERE/"validation_selection.json").write_text(json.dumps({"selected":selected,"n":2048,"folds":4,"weights_frozen":True,"no_validation_driven_tuning":True},indent=2)+"\n")
print(json.dumps({"selected":selected,"n":2048,"folds":4},indent=2))
