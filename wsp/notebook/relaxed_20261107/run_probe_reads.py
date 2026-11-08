"""Run the fixed diagnostic/read sequence; all model execution uses the harness."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "wsp/notebook/relaxed_20261107/"
PLAN = [
    (27, ["gen", "--which", "edited", "--ids-file", PREFIX + "instruction_probe512.txt"]),
    (28, ["read", "--model", "original", "--capture",
          "resid:11,resid:15,resid:19,resid:23,resid:26,mlp_act:11,mlp_act:15,mlp_act:19,mlp_act:23",
          "--positions", "all", "--texts-file", PREFIX + "calibration16.jsonl"]),
    (29, ["read", "--model", "original", "--capture", "resid:11,resid:15,resid:19,resid:23,resid:26",
          "--positions", "last", "--texts-file", PREFIX + "contrast480.jsonl"]),
]


def main():
    for number, arguments in PLAN:
        ledger = json.loads((ROOT / "wsp/ledger.json").read_text())
        assert ledger["used"] == number - 1 and ledger["submitted"] is None
        print("Starting", number, arguments, flush=True)
        with (HERE / f"pass{number}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "t2s.wsp.tool", *arguments], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((ROOT / f"wsp/passes/pass_{number:02d}/summary.json").read_text())
        print("Completed", number, json.dumps(summary), flush=True)
        if number == 27:
            subprocess.run([sys.executable, str(HERE / "report.py"), "27", "--probe"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
