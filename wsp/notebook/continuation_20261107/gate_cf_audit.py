"""Cached direct-path estimate for the proposed gate intervention, not SQL scoring."""
import json
from pathlib import Path
import re
import sys

from t2s.wsp.kernel import load, torch
from t2s.wsp.common import base_dir
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from attnlib import attn_heads, rms, W

torch.set_num_threads(8)
data = torch.load(HERE.parents[1] / "passes/pass_07/read.pt", weights_only=True)
design = torch.load(HERE / "gate_design.pt", weights_only=True)
texts = {r["id"]: r for r in map(json.loads, (HERE.parent / "texts_pass07.jsonl").read_text().splitlines())}
counterfactuals = {r["id"]: r for r in json.loads((HERE.parent / "alias_cf_rows.json").read_text())}
original = {r["id"]: r for r in map(json.loads, (HERE.parents[1] / "passes/pass_01/original.jsonl").read_text().splitlines())}
tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
embedding, norm = load("model.embed_tokens.weight"), load("model.norm.weight")
selected = design["neurons"]
gate = load("model.layers.27.mlp.gate_proj.weight")[selected]
up = load("model.layers.27.mlp.up_proj.weight")[selected]
down = load("model.layers.27.mlp.down_proj.weight")[:, selected]
rows = {}
for n, label in enumerate(data["ids"]):
    if label.startswith("S_"):
        continue
    tag, pid = label.split("_", 1)
    p = len(data["tokens"][n]) - len(tokenizer(texts[label]["answer"], add_special_tokens=False)["input_ids"]) - 1
    residual = data["resid:26"][n].float()
    heads, _ = attn_heads(27, residual, [p])
    x = rms(residual[p] + heads[0].sum(0), W(27, "post_attention_layernorm.weight"))
    before_gate = gate @ x
    after_gate = before_gate - design["gate_response"] * (x @ design["key"])
    delta = down @ ((torch.nn.functional.silu(after_gate) - torch.nn.functional.silu(before_gate)) * (up @ x))
    match = re.search(r"FROM\s+(\w+)\s+AS\s+T1\s+(?:INNER\s+)?JOIN\s+(\w+)\s+AS\s+T2\s", original[pid]["pred_sql"])
    a, b = [tokenizer(" " + table)["input_ids"][0] for table in match.groups()]
    c = counterfactuals[pid]
    owner, other = (a, b) if c["owner_is_t1"] else (b, a)
    shift = float(delta @ (norm * (embedding[owner] - embedding[other])) / 5.5)
    digit = c["digit_own"] if tag == "Aown" else 3 - c["digit_own"]
    before = c["own27" if tag == "Aown" else "swap27"]
    rows.setdefault(pid, {})[digit] = (before, before + shift)
result = {"mode": design["mode"], "n_pairs": len(rows)}
for digit in (1, 2):
    b = torch.tensor([r[digit][0] for r in rows.values()])
    a = torch.tensor([r[digit][1] for r in rows.values()])
    result[f"T{digit}"] = {"margin_before": float(b.mean()), "margin_after": float(a.mean()),
                           "consistent_before": int(((b > 0) if digit == 1 else (b < 0)).sum()),
                           "consistent_after": int(((a > 0) if digit == 1 else (a < 0)).sum())}
result["binding_effect_before"] = sum(r[1][0] - r[2][0] for r in rows.values()) / len(rows)
result["binding_effect_after"] = sum(r[1][1] - r[2][1] for r in rows.values()) / len(rows)
(HERE / "gate_cf_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
