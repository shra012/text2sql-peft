"""Read-only paired execution report for continuation practice passes."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASSES = HERE.parents[1] / "passes"
COHORTS = [
    ("binding_64", "seed0", 20, 14),
    ("heads_1.5", "seed0", 21, 14),
    ("qualifier", "seed0", 22, 14),
    ("FROM_gate", "seed0", 24, 14),
    ("F4_control", "seed0", 25, 14),
]


def records(number, kind="edited"):
    path = PASSES / f"pass_{number:02d}" / f"{kind}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def paired(reference, candidate):
    assert reference.keys() == candidate.keys()
    fixed = sum(candidate[i]["correct"] and not reference[i]["correct"] for i in reference)
    broke = sum(reference[i]["correct"] and not candidate[i]["correct"] for i in reference)
    n = fixed + broke
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(fixed, broke) + 1)) / 2 ** n) if n else 1.
    return {"reference_correct": sum(r["correct"] for r in reference.values()),
            "candidate_correct": sum(r["correct"] for r in candidate.values()),
            "fixed": fixed, "broke": broke, "net": fixed - broke,
            "mcnemar_exact_p": round(p, 6)}


def main():
    rows = []
    pooled = {}
    for name, cohort, number, f4_number in COHORTS:
        if not (PASSES / f"pass_{number:02d}" / "summary.json").exists():
            continue
        candidate = records(number)
        original = records(number, "original")
        f4 = records(f4_number)
        row = {"candidate": name, "cohort": cohort, "pass": number, "n": len(candidate),
               "ex": round(100 * sum(r["correct"] for r in candidate.values()) / len(candidate), 2),
               "vs_original": paired(original, candidate), "vs_f4": paired(f4, candidate),
               "no_such_column": sum("no such column" in r["error"] for r in candidate.values()),
               "sql_first": sum(r["sql_first"] for r in candidate.values())}
        if cohort == "seed0" and (PASSES / "pass_25/summary.json").exists():
            row["vs_f4_control"] = paired(records(25), candidate)
        rows.append(row)
        pool = pooled.setdefault(name, {"original": {}, "f4": {}, "candidate": {}})
        # The same question can occur in multiple cohorts. Count it once, taking
        # the first predeclared cohort's result, for the pooled paired test.
        for i in candidate:
            if i not in pool["candidate"]:
                for kind, source in (("original", original), ("f4", f4), ("candidate", candidate)):
                    pool[kind][i] = source[i]
    unique = {name: {"n": len(p["candidate"]),
                     "vs_original": paired(p["original"], p["candidate"]),
                     "vs_f4": paired(p["f4"], p["candidate"])} for name, p in pooled.items()}
    result = {"cohorts": rows, "unique_questions": unique,
              "caveat": "Practice data only; candidates selected using these data. Cached-baseline numerical variation and multiple comparisons limit p-value interpretation."}
    (HERE / "practice_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
