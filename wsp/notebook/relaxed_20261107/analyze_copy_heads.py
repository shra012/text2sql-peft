"""Trace schema-token retrieval in cached attention inputs, CPU algebra only.

Scores are mechanistic diagnostics, not execution-accuracy evidence. Tokens from
gold SQL are never used. Only the model's own responses and database identifiers
define the query positions and schema-copy locations.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys

from t2s.wsp.common import base_dir
from t2s.wsp.kernel import load, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from attnlib import attn_heads

torch.set_num_threads(8)
LAYERS = (12, 16, 20, 24, 27)
KEYWORDS = set("SELECT FROM WHERE JOIN INNER LEFT RIGHT OUTER AS ON AND OR NOT NULL DISTINCT GROUP BY ORDER ASC DESC LIMIT HAVING COUNT SUM AVG MIN MAX CASE WHEN THEN ELSE END CAST INTEGER REAL TEXT LIKE IN BETWEEN UNION ALL WITH TRUE FALSE SQL".split())


def main():
    data = torch.load(HERE.parents[1] / "passes/pass_28/read.pt", weights_only=True)
    texts = {r["id"]: r for r in map(json.loads, (HERE / "calibration16.jsonl").read_text().splitlines())}
    practice = {r["id"]: r for r in map(json.loads, (HERE.parents[1] / "practice.jsonl").read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    embedding, norm = load("model.embed_tokens.weight"), load("model.norm.weight")
    aggregate, examples = defaultdict(list), defaultdict(list)
    for n, pid in enumerate(data["ids"]):
        text = texts[pid]["prompt"] + texts[pid]["answer"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = data["tokens"][n]
        assert encoded["input_ids"] == tokens.tolist()
        prompt_length = len(tokenizer(texts[pid]["prompt"], add_special_tokens=False)["input_ids"])
        user_start = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        user_end = text.index("<|im_end|>", user_start)
        kinds = []
        for a, b in encoded["offset_mapping"]:
            line_start = text.rfind("\n", 0, a) + 1
            line_end = text.find("\n", a)
            line = text[line_start:line_end if line_end >= 0 else len(text)].lstrip()
            kind = "question" if line.startswith(("-- Question", "-- External knowledge")) else "comment" if line.startswith("--") else "ddl"
            if a < user_start or a >= user_end:
                kind = "system"
            kinds.append(kind)
        connection = sqlite3.connect(f"file:{practice[pid]['db_path']}?mode=ro", uri=True)
        try:
            names = []
            for table, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                names.append(table)
                quoted = table.replace('"', '""')
                names.extend(row[1] for row in connection.execute(f'PRAGMA table_info("{quoted}")'))
        finally:
            connection.close()
        identifier_tokens = {i for name in names for i in tokenizer(" " + name, add_special_tokens=False)["input_ids"]
                             if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]+", tokenizer.decode([i]).strip())
                             and tokenizer.decode([i]).strip().upper() not in KEYWORDS}
        # A maximum of 20 positions per response keeps the attribution focused
        # on actual schema words rather than punctuation or alias digits.
        query_positions = [p for p in range(prompt_length - 1, len(tokens) - 1) if int(tokens[p + 1]) in identifier_tokens][:20]
        if not query_positions:
            continue
        query_tensor = torch.tensor(query_positions)
        target_tokens = tokens[query_tensor + 1].long()
        contrasts = norm * (embedding[target_tokens] - embedding[list(identifier_tokens)].mean(0))
        ddl = torch.tensor([p < prompt_length and kinds[p] == "ddl" for p in range(len(tokens))])
        comment = torch.tensor([p < prompt_length and kinds[p] == "comment" for p in range(len(tokens))])
        matching = tokens[None] == target_tokens[:, None]
        # Induction heads may read the preceding token rather than the copied
        # token itself; include either source location as a retrieval match.
        preceding = torch.zeros_like(matching)
        preceding[:, :-1] = matching[:, 1:]
        matching |= preceding
        for layer in LAYERS:
            residual = data[f"resid:{layer - 1}"][n].float()
            contribution, attention = attn_heads(layer, residual, query_positions)
            dla = torch.einsum("qhd,qd->qh", contribution, contrasts) / residual[query_tensor].square().mean(-1).sqrt()[:, None]
            prompt_mass = attention[:, :, :prompt_length].sum(-1)
            ddl_match = (attention * (matching & ddl[None])[:, None]).sum(-1)
            comment_match = (attention * (matching & comment[None])[:, None]).sum(-1)
            for head in range(12):
                key = f"L{layer}H{head}"
                for j, pos in enumerate(query_positions):
                    aggregate[key].append({"prompt_mass": float(prompt_mass[j, head]),
                                           "ddl_match": float(ddl_match[j, head]),
                                           "comment_match": float(comment_match[j, head]),
                                           "copy_dla": float(dla[j, head]), "long": pid.startswith("x")})
                if sum(e["id"].startswith(pid[0]) for e in examples[key]) < 2:
                    best = int(ddl_match[:, head].argmax())
                    weights = attention[best, head]
                    tops = weights.topk(4)
                    examples[key].append({"id": pid, "next_token": tokenizer.decode([int(target_tokens[best])]),
                                          "source": [{"text": tokenizer.decode(tokens[max(0, int(p) - 2):int(p) + 2]),
                                                      "mass": float(v), "region": "answer" if p >= prompt_length else kinds[int(p)]}
                                                     for v, p in zip(tops.values, tops.indices)]})
    result = []
    for key, rows in aggregate.items():
        summary = {name: sum(row[name] for row in rows) / len(rows)
                   for name in ("prompt_mass", "ddl_match", "comment_match", "copy_dla")}
        summary.update({"head": key, "n_positions": len(rows), "examples": examples[key]})
        for flag, label in ((False, "short"), (True, "long")):
            subset = [row for row in rows if row["long"] == flag]
            summary[label] = {name: sum(row[name] for row in subset) / len(subset)
                              for name in ("prompt_mass", "ddl_match", "comment_match", "copy_dla")}
        result.append(summary)
    result.sort(key=lambda row: row["ddl_match"], reverse=True)
    (HERE / "copy_heads.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result:
        print(json.dumps({k: v for k, v in row.items() if k != "examples"}), flush=True)


if __name__ == "__main__":
    main()
