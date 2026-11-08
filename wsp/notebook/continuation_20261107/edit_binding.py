"""Remove late MLP neurons that oppose the SELECT-alias -> FROM-table circuit.

Uses the existing own/swap counterfactual attribution, not a supervised SQL loss.
F4 is restored from its archived, exact bf16 tensor before applying each candidate.
No forward pass, gradients, optimizer, or iterative parameter fitting occurs here.
"""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
from t2s.wsp.kernel import list_edits, remove_edit, save_edit, torch

torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent
NAME = "model.layers.27.mlp.down_proj.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--allow-failed", action="store_true", help="Explicitly reproduce a rejected candidate")
    args = parser.parse_args()
    prior = HERE.parents[1] / "passes/pass_20/summary.json"
    if args.neurons == 64 and prior.exists() and not args.allow_failed:
        summary = json.loads(prior.read_text())
        if summary["edited"]["acc"] < summary["original"]["acc"] - .05:
            parser.exit(2, "Binding-64 was rejected after a >5-point practice regression; no further pass spent.\n")
    binding = torch.load(HERE.parent / "l27_binding.pt", weights_only=True)
    f4 = torch.load(HERE.parent / "edit_f4_vectors.pt", weights_only=True)
    order = binding["eff"].argsort()
    # Preserve every column used by the original format edit.
    order = order[f4["k"][order] == 0]
    chosen = order[:args.neurons]
    assert (binding["eff"][chosen] < 0).all()
    for name in list_edits():
        if name != NAME:
            remove_edit(name)
    weight = load_file(str(HERE / f"{NAME}.safetensors"))["tensor"].float()
    weight[:, chosen] = 0
    stats = save_edit(NAME, weight)
    label = f"binding_{args.neurons}"
    save_file({"tensor": weight.bfloat16().contiguous()}, str(HERE / f"{label}.safetensors"))
    torch.save({"neurons": chosen, "stats": stats}, HERE / f"{label}_design.pt")
    print(label, stats)
    print("Counterfactual anti-binding effect removed:", float(binding["eff"][chosen].sum()))


if __name__ == "__main__":
    main()
