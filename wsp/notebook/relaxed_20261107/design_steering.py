"""One-shot role keys and paired-instruction residual directions, CPU only.

Position labels and controlled prompts define the directions. There is no SQL
target loss, gradient, optimizer, or fitting of output weights to correct queries.
"""
import json
from pathlib import Path

from t2s.wsp.kernel import torch
from t2s.wsp.common import base_dir
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
PASSES = HERE.parents[1] / "passes"
torch.set_num_threads(8)
LAYERS = (11, 15, 19, 23)


def solve(positive, negative, columns=256):
    separation = (positive.mean(0) - negative.mean(0)).abs() / (positive.std(0) + negative.std(0) + .001)
    chosen = separation.topk(columns).indices
    x = negative[:, chosen]
    covariance = x.T @ x / len(x)
    mean = positive[:, chosen].mean(0)
    small = torch.linalg.solve(covariance + .01 * covariance.diagonal().mean() * torch.eye(columns), mean)
    small /= small @ mean
    k = torch.zeros(positive.shape[1])
    k[chosen] = small
    return k


def score_stats(x):
    return {"n": len(x), "mean": float(x.mean()), "q01_q50_q99": x.quantile(torch.tensor([.01, .5, .99])).tolist()}


def main():
    calibration = torch.load(PASSES / "pass_28/read.pt", weights_only=True)
    contrasts = torch.load(PASSES / "pass_29/read.pt", weights_only=True)
    metadata = json.loads((HERE / "contrast_metadata.json").read_text())
    texts = {r["id"]: r for r in map(json.loads, (HERE / "calibration16.jsonl").read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir()))
    prompt_lengths = {pid: len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]) for pid, row in texts.items()}
    lookup = {(row["original_id"], row["variant"], row["anchor"]): i for i, row in enumerate(metadata)}
    questions = sorted({row["original_id"] for row in metadata})
    assert [row["id"] for row in metadata] == contrasts["ids"]
    output, report = {}, {}
    for layer in LAYERS:
        groups = {"design_reply": [], "design_prompt": [], "audit_reply": [], "audit_prompt": []}
        for n, pid in enumerate(calibration["ids"]):
            cohort = "audit" if n % 4 == 0 else "design"
            acts = calibration[f"mlp_act:{layer}"][n].float()
            split = prompt_lengths[pid] - 1
            groups[f"{cohort}_prompt"].append(acts[:split])
            groups[f"{cohort}_reply"].append(acts[split:-1])
        groups = {name: torch.cat(rows) for name, rows in groups.items()}
        audit_key = solve(groups["design_reply"], groups["design_prompt"])
        diagnostics = {name: score_stats(rows @ audit_key) for name, rows in groups.items()}
        k = solve(torch.cat([groups["design_reply"], groups["audit_reply"]]),
                  torch.cat([groups["design_prompt"], groups["audit_prompt"]]))
        h = contrasts[f"resid:{layer}"]
        differences = {}
        for variant in ("neutral", "schema", "plan", "expert"):
            differences[variant] = torch.stack([
                h[lookup[(pid, variant, anchor)]][0].float() - h[lookup[(pid, "original", anchor)]][0].float()
                for pid in questions for anchor in range(3)])
        nuisance = differences["neutral"].mean(0)
        directions, details = {}, {}
        for variant in ("schema", "plan", "expert"):
            delta = differences[variant]
            mean = delta.mean(0)
            orthogonal = mean - nuisance * (mean @ nuisance) / nuisance.square().sum().clamp_min(1e-12)
            directions[variant] = {"raw": mean, "orthogonal": orthogonal}
            details[variant] = {"mean_norm": float(mean.norm()), "orthogonal_norm": float(orthogonal.norm()),
                                "mean_pair_norm": float(delta.norm(dim=1).mean()),
                                "coherence": float(torch.nn.functional.cosine_similarity(delta, mean[None]).mean())}
        output[layer] = {"key": k, "directions": directions, "diagnostics": diagnostics}
        report[layer] = {"key_audit": diagnostics, "directions": details, "nuisance_norm": float(nuisance.norm())}
        print("layer", layer, json.dumps(report[layer]), flush=True)
        del groups
    torch.save(output, HERE / "steering_design.pt")
    (HERE / "steering_design.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
