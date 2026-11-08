"""Run predeclared style diagnostics on restored F4, then activation reads."""
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ledger=json.loads((ROOT/"wsp/ledger.json").read_text())
assert ledger["used"]==45 and ledger["submitted"] is None
subprocess.run([sys.executable,str(HERE/"restore_f4.py")],cwd=ROOT,check=True)
with (HERE/"pass46.log").open("w") as log:
    subprocess.run([sys.executable,"-m","t2s.wsp.tool","gen","--which","edited","--ids-file",str(HERE/"style_probe512.txt")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(HERE/"report.py"),"46","--probe","--probe-metadata","style_probe_metadata.json"],cwd=ROOT,check=True)
print("Style probes complete; recording controlled states",flush=True)
with (HERE/"pass47.log").open("w") as log:
    subprocess.run([sys.executable,"-m","t2s.wsp.tool","read","--model","original","--capture","resid:11,resid:15,resid:19,resid:23","--positions","last","--texts-file",str(HERE/"style_reads448.jsonl")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print("Style phase complete at pass 47",flush=True)
