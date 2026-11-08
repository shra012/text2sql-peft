"""Sparse, closed-form edit discouraging a T alias after SELECT/DISTINCT.

Hypothesis: the model can use column/table names it already knows instead of
opening a SELECT list with an alias whose table has not yet been specified.
This changes a naming habit, not SQL prompts or postprocessing. There is no
supervised target SQL, optimizer, gradient, or iterative fit.
"""
import argparse
from collections import Counter
from pathlib import Path

from safetensors.torch import load_file
from t2s.wsp.common import base_dir
from t2s.wsp.kernel import list_edits, load, remove_edit, save_edit, torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
NAME = "model.layers.27.mlp.down_proj.weight"
torch.set_num_threads(8)


def key(positive, negative, columns=128):
    # Select position-sensitive neurons, then make one ridge solve. All training
    # signals here describe a token position, never whether a SQL answer is right.
    score = (positive.mean(0) - negative.mean(0)).abs() / (
        positive.std(0) + negative.std(0) + .001)
    chosen = score.topk(columns).indices
    x = negative[:, chosen]
    covariance = x.T @ x / len(x)
    mean = positive[:, chosen].mean(0)
    small = torch.linalg.solve(covariance + .01 * covariance.diagonal().mean() * torch.eye(columns), mean)
    small /= small @ mean
    full = torch.zeros(positive.shape[1])
    full[chosen] = small
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=4.)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    data = torch.load(HERE.parents[1] / "passes/pass_03/read.pt", weights_only=True)
    select_ids = {tokenizer.convert_tokens_to_ids("SELECT"), tokenizer.convert_tokens_to_ids("ĠDISTINCT")}
    groups = {"design_positive": [], "design_negative": [], "audit_positive": [], "audit_negative": []}
    counts = Counter()
    for index, label in enumerate(data["ids"]):
        if not label.startswith("F_"):
            continue
        cohort = "audit" if int(label.split("_")[-1][1:]) % 3 == 0 else "design"
        tokens = data["tokens"][index]
        for j, position in enumerate(data["positions"][index].tolist()):
            group = "positive" if int(tokens[position]) in select_ids else "negative"
            groups[f"{cohort}_{group}"].append(data["mlp_act:27"][index][j].float())
        counts.update(tokens[data["positions"][index][0]:].tolist())
    groups = {name: torch.stack(rows) for name, rows in groups.items()}
    k_audit = key(groups["design_positive"], groups["design_negative"])
    diagnostics = {}
    for name, rows in groups.items():
        score = rows @ k_audit
        values = score.quantile(torch.tensor([.01, .5, .99]))
        diagnostics[name] = {"n": len(rows), "mean": float(score.mean()), "q01_q50_q99": values.tolist()}
    print("Position-key audit:", diagnostics)
    k = key(torch.cat([groups["design_positive"], groups["audit_positive"]]),
            torch.cat([groups["design_negative"], groups["audit_negative"]]))
    target = tokenizer.convert_tokens_to_ids("ĠT")
    embedding, norm = load("model.embed_tokens.weight"), load("model.norm.weight")
    keep = [token for token, _ in counts.most_common() if token != target][:400]
    constraints = norm * embedding[[target, *keep]]
    rhs = torch.zeros(len(constraints))
    rhs[0] = -args.delta * 5.5  # same pre-final-normalization RMS approximation as F4
    u = constraints.T @ torch.linalg.solve(constraints @ constraints.T + 1e-6 * torch.eye(len(rhs)), rhs)
    torch.save({"u": u, "k": k, "diagnostics": diagnostics, "delta": args.delta}, HERE / "qualifier_design.pt")
    if args.dry:
        return
    for name in list_edits():
        if name != NAME:
            remove_edit(name)
    weight = load_file(str(HERE / f"{NAME}.safetensors"))["tensor"].float()
    print(save_edit(NAME, weight + torch.outer(u, k)))


if __name__ == "__main__":
    main()
