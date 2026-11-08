"""Sequential mechanism trials, specified explicitly in a JSON plan file."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for trial in plan:
        ledger = json.loads((ROOT / "wsp/ledger.json").read_text())
        number = ledger["used"] + 1
        assert not ledger["submitted"] and number <= ledger["budget"]
        print("Starting", number, trial, flush=True)
        subprocess.run([sys.executable, str(HERE / trial["script"]), *trial["args"]], cwd=ROOT, check=True)
        cohort = trial.get("cohort", "screen256.txt")
        with (HERE / f"pass{number}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "t2s.wsp.tool", "gen", "--which", "edited",
                            "--ids-file", str(HERE / cohort)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(HERE / "report.py"), str(number), "--baseline", str(trial.get("baseline", 26))],
                       cwd=ROOT, check=True)
        completed = json.loads((ROOT / "wsp/ledger.json").read_text())["passes"][-1]
        archive = HERE / "candidates" / f"pass_{number:02d}"
        archive.mkdir(parents=True, exist_ok=True)
        for tensor in (ROOT / "wsp/edits").glob("*.safetensors"):
            shutil.copy2(tensor, archive / tensor.name)
        (archive / "manifest.json").write_text(json.dumps({"trial": trial, "fingerprint": completed["edits_fingerprint"]}, indent=2) + "\n")
        with (HERE / "trials.jsonl").open("a") as f:
            f.write(json.dumps({"pass": number, "trial": trial, "fingerprint": completed["edits_fingerprint"],
                                "summary": completed["summary"]}) + "\n")
        print("Completed", number, json.dumps(completed["summary"]), flush=True)


if __name__ == "__main__":
    main()
