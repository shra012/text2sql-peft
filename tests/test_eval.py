"""Gold-vs-gold sanity check for the BIRD EX evaluator (~1 min CPU), plus small unit tests.

.venv/bin/python tests/test_eval.py      (no pytest needed)
pytest tests/test_eval.py
"""

from __future__ import annotations

import sys
import time

from t2s.bird_eval import evaluate_batch_detailed, ex_match, execute
from t2s.data import load_raw
from t2s.prompt import extract_sql, format_completion


def test_extract_sql():
    assert extract_sql("blah\n```sql\nSELECT 1;\n```\nmore ```sql\nSELECT 2 ;  \n```") == "SELECT 2"
    assert extract_sql("```\nSELECT 3;\n```") == "SELECT 3"
    assert extract_sql("SELECT 4;\n") == "SELECT 4"
    assert extract_sql("```sql\nSELECT 5 FROM t") == "SELECT 5 FROM t"  # unterminated
    assert extract_sql(format_completion("SELECT 6")) == "SELECT 6"


def test_timeout_and_errors():
    rows = load_raw("dev")
    db = rows[0]["db_path"]
    t0 = time.monotonic()
    ok, res = execute(
        db,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c",
        timeout=2.0,
    )
    assert not ok and "timeout" in res and time.monotonic() - t0 < 10
    ok, res = execute(db, "SELEC nonsense")
    assert not ok
    ok, res = execute(db, "CREATE TABLE zz(a)")  # read-only connection
    assert not ok
    assert ex_match(db, "SELECT 1, 2", "SELECT 1, 2 UNION ALL SELECT 1, 2")  # set semantics
    assert not ex_match(db, "SELECT 2, 1", "SELECT 1, 2")  # column order matters


def gold_vs_gold(num_workers: int = 8) -> tuple[int, list[tuple[dict, str]]]:
    """Score every dev gold query against itself; returns (number of rows, failures)."""
    rows = load_raw("dev")
    items = [{"db_path": r["db_path"], "pred_sql": r["gold_sql"], "gold_sql": r["gold_sql"]} for r in rows]
    t0 = time.time()
    res = evaluate_batch_detailed(items, num_workers=num_workers)
    fails = [(r, err) for r, (ok, err) in zip(rows, res, strict=True) if not ok]
    n_ok = len(rows) - len(fails)
    print(
        f"gold-vs-gold: {n_ok}/{len(rows)} correct ({100 * n_ok / len(rows):.2f}%) in {time.time() - t0:.0f}s"
    )
    for r, err in fails:
        print(f"  FAIL qid={r['question_id']} db={r['db_id']}: {err}")
    return len(rows), fails


# These two gold queries take longer than BIRD's 30 s limit on their own, so 1532/1534 is the ceiling.
SLOW_GOLD = {518, 701}


def test_gold_vs_gold():
    n, fails = gold_vs_gold()
    assert n == 1534
    unexpected = [(r["question_id"], err) for r, err in fails if r["question_id"] not in SLOW_GOLD]
    assert not unexpected, f"gold queries failed: {unexpected}"


if __name__ == "__main__":
    test_extract_sql()
    test_timeout_and_errors()
    print("unit tests ok")
    n, fails = gold_vs_gold()
    sys.exit(0 if n == 1534 and {r["question_id"] for r, _ in fails} <= SLOW_GOLD else 1)
