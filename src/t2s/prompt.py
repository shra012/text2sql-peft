"""Prompt construction for BIRD text-to-SQL.

Schema = CREATE TABLE DDL (from sqlite_master) + one comment line per column with
a few distinct example values. Prompts are fitted to a token budget by shrinking the
number of example values, then hard-truncating the schema text.
"""

from __future__ import annotations

import re
import sqlite3
import urllib.parse
from functools import cache

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema, external knowledge and a question, "
    "write one SQLite query that answers it. Return only the SQL in a ```sql code block."
)

MAX_VALUE_CHARS = 40
TRUNCATION_MARKER = "\n-- [schema truncated]"


def _connect_ro(db_path: str) -> sqlite3.Connection:
    # Trusted queries only, so no authorizer (unlike bird_eval): PRAGMA table_info must run.
    uri = f"file:{urllib.parse.quote(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_WS_OR_QUOTED = re.compile(r"(\"[^\"]*\"|`[^`]*`|'[^']*'|\[[^\]]*\])|[ \t]{2,}")


def _compact_ddl(ddl: str) -> str:
    """Collapse runs of spaces/tabs outside quoted identifiers/literals (saves tokens)."""
    lines = [
        _WS_OR_QUOTED.sub(lambda m: m.group(1) if m.group(1) is not None else " ", ln).rstrip()
        for ln in ddl.strip().splitlines()
    ]
    return "\n".join(ln for ln in lines if ln.strip())


def _fmt_value(v) -> str:
    if isinstance(v, bytes):
        return "<blob>"
    if isinstance(v, str):
        s = " ".join(v.split())  # collapse newlines/tabs
        if len(s) > MAX_VALUE_CHARS:
            s = s[:MAX_VALUE_CHARS] + "..."
        return "'" + s.replace("'", "''") + "'"
    return repr(v)


@cache
def _schema_parts(db_path: str, n_examples: int) -> tuple[str, ...]:
    """Return one text block per table (DDL + per-column example comments)."""
    conn = _connect_ro(db_path)
    try:
        cur = conn.cursor()
        tables = cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        parts = []
        for tname, ddl in tables:
            lines = [_compact_ddl(ddl).rstrip(";") + ";"]
            if n_examples > 0:
                cols = cur.execute(f"PRAGMA table_info({_qident(tname)})").fetchall()
                for col in cols:
                    cname = col[1]
                    try:
                        vals = cur.execute(
                            f"SELECT DISTINCT {_qident(cname)} FROM {_qident(tname)} "
                            f"WHERE {_qident(cname)} IS NOT NULL LIMIT {int(n_examples)}"
                        ).fetchall()
                    except sqlite3.Error:
                        vals = []
                    if vals:
                        shown = ", ".join(_fmt_value(v[0]) for v in vals)
                        lines.append(f"-- {tname}.`{cname}` examples: {shown}")
            parts.append("\n".join(lines))
        return tuple(parts)
    finally:
        conn.close()


def get_schema(db_path: str, n_examples: int = 3) -> str:
    """CREATE TABLE DDL for every table plus example-value comments per column (cached)."""
    return "\n\n".join(_schema_parts(str(db_path), int(n_examples)))


def build_messages(schema: str, question: str, evidence: str) -> list[dict]:
    user = f"{schema}\n\n-- External knowledge: {evidence}\n-- Question: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _n_tokens(tokenizer, messages: list[dict]) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(ids)


def fit_prompt_info(
    tokenizer, db_path: str, question: str, evidence: str, max_tokens: int = 3072
) -> tuple[list[dict], str, int]:
    """Like fit_prompt but returns (messages, fit_level, n_tokens).

    fit_level is "examples3", "examples1", "examples0" or "truncated".
    """
    for n in (3, 1, 0):
        msgs = build_messages(get_schema(db_path, n), question, evidence)
        n_tok = _n_tokens(tokenizer, msgs)
        if n_tok <= max_tokens:
            return msgs, f"examples{n}", n_tok

    # Hard-truncate the (example-free) schema text until it fits.
    schema = get_schema(db_path, 0)
    overflow = n_tok - max_tokens
    keep = len(schema)
    while True:
        # Shrink by the token overflow times a char/token ratio, with some slack.
        keep = max(0, keep - max(64, int(overflow * 4)))
        cut = schema[:keep]
        nl = cut.rfind("\n")
        if nl > 0:
            cut = cut[:nl]
        msgs = build_messages(cut + TRUNCATION_MARKER, question, evidence)
        n_tok = _n_tokens(tokenizer, msgs)
        if n_tok <= max_tokens or keep == 0:
            return msgs, "truncated", n_tok
        overflow = n_tok - max_tokens


def fit_prompt(tokenizer, db_path: str, question: str, evidence: str, max_tokens: int = 3072) -> list[dict]:
    """Build messages; shrink examples (3 -> 1 -> 0) then hard-truncate schema to fit."""
    return fit_prompt_info(tokenizer, db_path, question, evidence, max_tokens)[0]


_SQL_BLOCK = re.compile(r"```[ \t]*sql[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
_OPEN_SQL = re.compile(r"```[ \t]*sql[ \t]*\n?(.*)$", re.DOTALL | re.IGNORECASE)


def _clean(sql: str) -> str:
    return sql.strip().rstrip("; \t\r\n").strip()


def extract_sql(text: str) -> str:
    """Last ```sql block, else any ``` block, else raw text; strip trailing semicolons."""
    blocks = _SQL_BLOCK.findall(text)
    if blocks:
        return _clean(blocks[-1])
    blocks = _ANY_BLOCK.findall(text)
    if blocks:
        return _clean(blocks[-1])
    m = _OPEN_SQL.search(text)  # unterminated block (e.g. hit max_tokens)
    if m:
        return _clean(m.group(1))
    return _clean(text)


def format_completion(sql: str) -> str:
    return f"```sql\n{sql}\n```"
