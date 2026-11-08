"""Sequentially execute the five remaining, predeclared practice passes.

Only t2s.wsp.tool runs the model. Failed commands halt this runner. The original
pass budget is enforced by the tool; this runner never modifies the ledger.
"""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = [
    (21, "edit_heads.py", ["--seed", "0"]),
    (22, "edit_binding.py", ["--seed", "2026"]),
    (23, "edit_heads.py", ["--seed", "2026"]),
    (24, "edit_binding.py", ["--ids-file", "wsp/notebook/ids_long512.txt"]),
    (25, "edit_heads.py", ["--ids-file", "wsp/notebook/ids_long512.txt"]),
]


def main():
    for number, script, cohort_args in PLAN:
        ledger = json.loads((ROOT / "wsp/ledger.json").read_text())
        assert ledger["used"] == number - 1 and ledger["submitted"] is None
        print(f"Starting pass {number}: {script}, {cohort_args}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, check=True)
        with (HERE / f"pass{number}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "t2s.wsp.tool", "gen", "--which", "both",
                            "--sample", "512", *cohort_args],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((ROOT / f"wsp/passes/pass_{number:02d}/summary.json").read_text())
        print(f"Completed pass {number}: {json.dumps(summary)}", flush=True)
        subprocess.run([sys.executable, str(HERE / "summarize.py")], cwd=ROOT,
                       stdout=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    main()
