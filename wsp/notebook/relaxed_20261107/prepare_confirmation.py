"""Select up to three positive screening candidates for a separate cohort."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ledger=json.loads((ROOT/"wsp/ledger.json").read_text())
assert ledger["used"]==80 and ledger["submitted"] is None
rows=[]
for row in map(json.loads,(HERE/"trials.jsonl").read_text().splitlines()):
    if row["trial"].get("cohort")!="screen256.txt" or row["trial"]["family"]=="control":continue
    report=json.loads((HERE/f"report_{row['pass']:02d}.json").read_text())
    result=report.get("matched_control",report["all"])
    if result["net"]>0 and row["fingerprint"]!="ae81f3cb05285df3":
        rows.append({"pass":row["pass"],"name":row["trial"]["name"],"fingerprint":row["fingerprint"],"screen":result})
rows.sort(key=lambda row:(-row["screen"]["net"],row["pass"]))
selected=[];seen=set()
for row in rows:
    if row["fingerprint"] in seen:continue
    selected.append(row);seen.add(row["fingerprint"])
    if len(selected)==3:break
control_pass=ledger["used"]+1
plan=[{"name":"F4_confirmation_control","script":"restore_f4.py","args":[],"family":"confirmation_control","cohort":"confirm256.txt","baseline":control_pass}]
for row in selected:
    plan.append({"name":"confirm_"+row["name"],"script":"apply_archived.py","args":["--pass-number",str(row["pass"])],
                 "family":"confirmation_candidate","parent_pass":row["pass"],"cohort":"confirm256.txt","baseline":control_pass})
(HERE/"confirmation_selection.json").write_text(json.dumps({"control_pass":control_pass,"selected":selected},indent=2)+"\n")
(HERE/"plan_confirmation.json").write_text(json.dumps(plan,indent=2)+"\n")
print(json.dumps({"control_pass":control_pass,"selected":selected},indent=2))
