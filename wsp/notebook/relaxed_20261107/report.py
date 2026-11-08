"""Compare unchanged-prompt edit trials with their exact F4 cohort reference."""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASSES = HERE.parents[1] / "passes"


def records(number):
    return {r["id"]: r for r in map(json.loads, (PASSES / f"pass_{number:02d}/edited.jsonl").read_text().splitlines())}


def paired(reference, candidate):
    assert candidate.keys() <= reference.keys()
    ids = list(candidate)
    n = len(ids)
    fixed = sum(candidate[i]["correct"] and not reference[i]["correct"] for i in ids)
    broke = sum(reference[i]["correct"] and not candidate[i]["correct"] for i in ids)
    discordant = fixed + broke
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(fixed, broke) + 1)) / 2 ** discordant) if discordant else 1.
    return {"n": n, "reference_correct": sum(reference[i]["correct"] for i in ids),
            "candidate_correct": sum(candidate[i]["correct"] for i in ids),
            "reference_ex": round(100 * sum(reference[i]["correct"] for i in ids) / n, 2),
            "candidate_ex": round(100 * sum(candidate[i]["correct"] for i in ids) / n, 2),
            "fixed": fixed, "broke": broke, "net": fixed - broke, "exact_p": round(p, 6)}


def forms(rows):
    return {"with": sum(row["pred_sql"].lstrip().upper().startswith("WITH") for row in rows.values()),
            "comment": sum(row["pred_sql"].lstrip().startswith(("--", "/*")) for row in rows.values()),
            "sql_first": sum(row["sql_first"] for row in rows.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pass_number", type=int)
    parser.add_argument("--baseline", type=int, default=26)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-metadata", default="instruction_probe_metadata.json")
    args = parser.parse_args()
    reference, candidate = records(args.baseline), records(args.pass_number)
    if args.probe:
        metadata = json.loads((HERE / args.probe_metadata).read_text())
        by_variant = {}
        for row in metadata:
            by_variant.setdefault(row["variant"], {})[row["original_id"]] = candidate[row["id"]]
        result = {name: {**paired(reference, rows), "forms": forms(rows)} for name, rows in by_variant.items()}
    else:
        result = {"all": paired(reference, candidate)}
        result["forms"] = forms(candidate)
        for prefix, label in (("p", "short"), ("x", "long")):
            subset = {i: r for i, r in candidate.items() if i.startswith(prefix)}
            if subset:
                result[label] = paired(reference, subset)
        control_path = PASSES / "pass_34/edited.jsonl"
        if control_path.exists() and args.pass_number != 34:
            control = records(34)
            if candidate.keys() <= control.keys():
                result["matched_control"] = {"reference_pass": 34, **paired(control, candidate)}
    (HERE / f"report_{args.pass_number:02d}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
