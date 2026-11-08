"""Counterfactual instruction probes over BIRD train questions only."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INSTRUCTIONS = {
    "neutral": "The provided text describes a SQLite database and a question about it. The task is to write a single SQLite query using that information. Return the query in a SQL code block.",
    "schema": "Use only tables and columns that appear in the schema. Locate the owning table of every selected column and include every required join. Keep table aliases consistent with the FROM clause. Return only SQL.",
    "plan": "Work out the required tables, joins, filters, and aggregation before choosing the SELECT expressions. Check the complete query against the question and the external knowledge. Return only SQL, without explaining your plan.",
    "expert": "Be a meticulous SQLite database engineer. Read the schema literally and use the supplied external knowledge to choose the exact conditions. Check that every identifier exists and every join connects the intended tables. Return only SQL.",
}


def main():
    practice_path = ROOT / "wsp/practice.jsonl"
    rows = {r["id"]: r for r in map(json.loads, practice_path.read_text().splitlines())}
    baseline_path = ROOT / "wsp/passes/pass_26/edited.jsonl"
    assert baseline_path.exists(), "Baseline must finish before selecting probe questions"
    screen = (HERE / "screen512.txt").read_text().splitlines()
    short = [i for i in screen if i.startswith("p")]
    long = [i for i in screen if i.startswith("x")]
    # Fixed ordering, not chosen using the questions' correctness.
    ids = short[:64] + long[:64]
    new, metadata = [], []
    for variant, instruction in INSTRUCTIONS.items():
        for original_id in ids:
            row = dict(rows[original_id])
            row["id"] = f"z{variant}_{original_id}"
            row["messages"] = [dict(message) for message in row["messages"]]
            row["messages"][0]["content"] += " " + instruction
            row["probe_only"] = True
            new.append(row)
            metadata.append({"id": row["id"], "original_id": original_id, "variant": variant})
    assert not any(row["id"] in rows for row in new)
    with practice_path.open("a") as f:
        for row in new:
            f.write(json.dumps(row) + "\n")
    (HERE / "instruction_probe512.txt").write_text("\n".join(r["id"] for r in new) + "\n")
    (HERE / "instruction_probe_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Smaller, balanced first-stage cohort for broad mechanism screens.
    (HERE / "screen256.txt").write_text("\n".join(short[:128] + long[:128]) + "\n")
    print("Prepared", len(new), "diagnostic prompts on", len(ids), "training questions.")


if __name__ == "__main__":
    main()
