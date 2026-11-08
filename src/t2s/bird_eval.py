"""BIRD execution accuracy (EX), mirroring the official evaluation logic.

Official BIRD: run predicted and gold SQL against the sqlite DB, fetchall(), and count
a hit iff set(pred_rows) == set(gold_rows); any error or timeout on the prediction
is a miss. Here the DB is opened read-only and the per-query timeout is enforced with
a sqlite progress handler (interrupts the VM in-process, so it cannot hang the way a
thread-based timeout can). A process pool parallelises a batch.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import sqlite3
import time
import urllib.parse

# Guard against runaway predictions (e.g. accidental cartesian products) exhausting RAM.
# Gold results on BIRD dev are far below this.
MAX_ROWS = 2_000_000
_FETCH_CHUNK = 10_000


# Model output is untrusted code. Besides the read-only open, an authorizer whitelists
# read-only actions, so ATTACH (which could open/create another file), PRAGMA, writes,
# DDL and transactions are refused before they run.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE
}


def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _connect_ro(db_path: str) -> sqlite3.Connection:
    uri = f"file:{urllib.parse.quote(str(db_path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.set_authorizer(_authorizer)
    # Some BIRD DBs contain non-UTF8 text; decode leniently instead of raising.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def execute(db_path: str, sql: str, timeout: float = 30.0) -> tuple[bool, list | str]:
    """Execute sql read-only. Returns (True, rows) or (False, error message)."""
    deadline = time.monotonic() + float(timeout)
    conn = None
    try:
        conn = _connect_ro(db_path)
        # Called every N VM instructions; a non-zero return aborts the statement.
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = conn.cursor()
        cur.execute(sql)
        rows: list = []
        while True:
            chunk = cur.fetchmany(_FETCH_CHUNK)
            if not chunk:
                break
            rows.extend(chunk)
            if len(rows) > MAX_ROWS:
                return False, f"too many rows (>{MAX_ROWS})"
        return True, rows
    except sqlite3.OperationalError as e:
        if time.monotonic() > deadline or "interrupted" in str(e):
            return False, f"timeout after {timeout}s"
        return False, f"OperationalError: {e}"
    except Exception as e:  # any failure is a miss
        return False, f"{type(e).__name__}: {e}"
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def _match_detail(db_path: str, pred_sql: str, gold_sql: str, timeout: float = 30.0) -> tuple[bool, str]:
    """(hit, error): error is "pred: ...", "gold: ..." or "compare: ...", empty on a clean comparison."""
    ok_p, res_p = execute(db_path, pred_sql, timeout)
    if not ok_p:
        return False, f"pred: {res_p}"
    ok_g, res_g = execute(db_path, gold_sql, timeout)
    if not ok_g:
        return False, f"gold: {res_g}"
    try:
        return set(res_p) == set(res_g), ""
    except TypeError as e:  # unhashable values; should not happen with sqlite
        return False, f"compare: {e}"


def ex_match(db_path: str, pred_sql: str, gold_sql: str, timeout: float = 30.0) -> bool:
    """True iff pred and gold both run and return the same set of rows."""
    return _match_detail(db_path, pred_sql, gold_sql, timeout)[0]


def _worker(task: tuple[str, str, str, float]) -> tuple[bool, str]:
    return _match_detail(*task)


def evaluate_batch_detailed(
    items: list[dict], num_workers: int = 8, timeout: float = 30.0
) -> list[tuple[bool, str]]:
    """Like evaluate_batch but returns (correct, error_message) per item."""
    tasks = [(it["db_path"], it["pred_sql"], it["gold_sql"], it.get("timeout", timeout)) for it in items]
    out: list[tuple[bool, str]] = [(False, "not run")] * len(tasks)
    if not tasks:
        return out
    # spawn: safe even when the parent has CUDA / vLLM threads alive.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=max(1, int(num_workers)))
    try:
        pending = [pool.apply_async(_worker, (t,)) for t in tasks]
        # Hard backstop in case a statement is not interruptible: 2 queries + slack.
        for idx, res in enumerate(pending):
            try:
                out[idx] = res.get(timeout=2 * tasks[idx][3] + 60)
            except mp.TimeoutError:
                out[idx] = (False, "hard timeout")
    finally:
        pool.terminate()
        pool.join()
    return out


def evaluate_batch(items: list[dict], num_workers: int = 8) -> list[bool]:
    """items: dicts with db_path, pred_sql, gold_sql. Returns per-item EX correctness."""
    return [ok for ok, _ in evaluate_batch_detailed(items, num_workers)]
