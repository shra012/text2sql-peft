"""Remove a FROM-position component from 64 previously attributed MLP gates.

All computations are CPU algebra on captured layer-26 residuals. The existing
attnlib reconstructs only layer-27 attention at recorded positions, as in the
original editor's attribution scripts. No model is loaded or run end to end.
The new gate matrix suppresses selected gates at FROM with one rank-one update.
It does not fit SQL labels or optimize any accuracy/loss objective.
"""
import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
from t2s.wsp.common import base_dir
from t2s.wsp.kernel import list_edits, load, remove_edit, save_edit, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from attnlib import attn_heads, rms, W

torch.set_num_threads(8)
FORMAT = "model.layers.27.mlp.down_proj.weight"
GATE = "model.layers.27.mlp.gate_proj.weight"


def collect():
    cached = HERE / "gate_inputs.pt"
    if cached.exists():
        return torch.load(cached, weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    data = torch.load(HERE.parents[1] / "passes/pass_23/read.pt", weights_only=True)
    texts = {r["id"]: r for r in map(json.loads, (HERE / "gate_texts.jsonl").read_text().splitlines())}
    from_ids = {tokenizer.convert_tokens_to_ids("ĠFROM"), tokenizer.convert_tokens_to_ids("FROM")}
    groups = {"design_positive": [], "design_negative": [], "audit_positive": [], "audit_negative": []}
    for n, label in enumerate(data["ids"]):
        prompt_len = len(tokenizer(texts[label]["prompt"], add_special_tokens=False)["input_ids"])
        tokens = data["tokens"][n]
        positions = torch.arange(prompt_len - 1, len(tokens) - 1)
        residual = data["resid:26"][n].float()
        heads, _ = attn_heads(27, residual, positions)
        middle = residual[positions] + heads.sum(1)
        inputs = rms(middle, W(27, "post_attention_layernorm.weight"))
        positive = torch.tensor([int(tokens[p]) in from_ids for p in positions])
        cohort = "audit" if int(label[1:]) % 3 == 0 else "design"
        groups[f"{cohort}_positive"].append(inputs[positive])
        groups[f"{cohort}_negative"].append(inputs[~positive])
    groups = {name: torch.cat(rows) for name, rows in groups.items()}
    torch.save(groups, cached)
    return groups


def key(positive, negative):
    mean = positive.mean(0)
    covariance = negative.T @ negative / len(negative)
    k = torch.linalg.solve(covariance + .01 * covariance.diagonal().mean() * torch.eye(len(mean)), mean)
    return k / (k @ mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--mode", choices=("mean", "suppress"), default="suppress")
    args = parser.parse_args()
    groups = collect()
    k_audit = key(groups["design_positive"], groups["design_negative"])
    diagnostics = {}
    for name, rows in groups.items():
        score = rows @ k_audit
        diagnostics[name] = {"n": len(rows), "mean": float(score.mean()),
                             "q01_q50_q99": score.quantile(torch.tensor([.01, .5, .99])).tolist()}
    print("FROM gate-key audit:", json.dumps(diagnostics))
    positive = torch.cat([groups["design_positive"], groups["audit_positive"]])
    negative = torch.cat([groups["design_negative"], groups["audit_negative"]])
    k = key(positive, negative)
    binding = torch.load(HERE.parent / "l27_binding.pt", weights_only=True)
    f4 = torch.load(HERE.parent / "edit_f4_vectors.pt", weights_only=True)
    chosen = binding["eff"].argsort()
    chosen = chosen[f4["k"][chosen] == 0][:64]
    gate = load(GATE)
    # The mean-removal version did not improve the cached binding circuit.
    # A -8 gate shift removes 97% of these neurons' FROM activation RMS while
    # avoiding the broad perturbation of deleting the neurons everywhere.
    u = gate[chosen] @ positive.mean(0) if args.mode == "mean" else torch.full((len(chosen),), 8.)
    gate[chosen] -= torch.outer(u, k)
    # Diagnose how much of the original MLP output would change at these exact
    # cached inputs, compared with the broadly harmful column-zeroing candidate.
    up = load("model.layers.27.mlp.up_proj.weight")[chosen]
    old_gate = load(GATE)[chosen]
    down = load(FORMAT)[:, chosen]
    impact = {}
    for name, inputs in (("FROM", positive), ("other_answer", negative)):
        acts = inputs @ up.T
        before = torch.nn.functional.silu(inputs @ old_gate.T) * acts
        after = torch.nn.functional.silu(inputs @ gate[chosen].T) * acts
        residual_change = (after - before) @ down.T
        impact[name] = residual_change.square().mean(-1).sqrt().quantile(torch.tensor([.5, .99])).tolist()
    print("Residual-change RMS median/p99:", json.dumps(impact))
    torch.save({"key": k, "gate_response": u, "neurons": chosen,
                "diagnostics": diagnostics, "impact": impact, "mode": args.mode}, HERE / "gate_design.pt")
    if args.dry:
        return
    for name in list_edits():
        if name != FORMAT:
            remove_edit(name)
    print(save_edit(FORMAT, load_file(str(HERE / f"{FORMAT}.safetensors"))["tensor"]))
    print(save_edit(GATE, gate))


if __name__ == "__main__":
    main()
