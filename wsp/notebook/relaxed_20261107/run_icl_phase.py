"""Score correct few-shot probes, then read clean/corrupted context contrasts."""
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ledger=json.loads((ROOT/"wsp/ledger.json").read_text())
assert ledger["used"]==60 and ledger["submitted"] is None
subprocess.run([sys.executable,str(HERE/"restore_f4.py")],cwd=ROOT,check=True)
with (HERE/"pass61.log").open("w") as log:
    subprocess.run([sys.executable,"-m","t2s.wsp.tool","gen","--which","edited","--ids-file",str(HERE/"icl_probe512.txt")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(HERE/"report.py"),"61","--probe","--probe-metadata","icl_probe_metadata.json"],cwd=ROOT,check=True)
print("Few-shot diagnostic complete; recording clean/corrupted contrasts",flush=True)
with (HERE/"pass62.log").open("w") as log:
    subprocess.run([sys.executable,"-m","t2s.wsp.tool","read","--model","original","--capture","resid:11,resid:15,resid:19,resid:23","--positions","last","--texts-file",str(HERE/"icl_reads384.jsonl")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print("Few-shot phase complete at pass 62",flush=True)
