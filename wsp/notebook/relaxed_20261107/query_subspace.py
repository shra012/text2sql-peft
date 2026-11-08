"""Covariance-normalized, content-dependent alias-query contrasts.

Uses recorded T1/T2 counterfactual inputs, never a SQL loss. Every projection is
a closed-form factorization anchored to the original model, not an optimizer.
"""
import json
import argparse
from pathlib import Path
import sys

from t2s.wsp.kernel import load, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from attnlib import rope

torch.set_num_threads(4)


def design(geometry, layer, head, rank, gain, null_midpoints=False, split=False):
    difference = geometry["query_differences"]
    midpoint = geometry["query_midpoints"]
    if split:
        difference, midpoint = difference[:16], midpoint[:16]
    covariance = geometry["covariance"]
    chol = torch.linalg.cholesky(covariance + .01 * covariance.diagonal().mean() * torch.eye(1536))
    whitened = torch.linalg.solve_triangular(chol, difference.T, upper=False)
    if null_midpoints:
        centers = torch.linalg.solve_triangular(chol, (midpoint - geometry["mean"]).T, upper=False)
        q, _ = torch.linalg.qr(centers, mode="reduced")
        whitened -= q @ (q.T @ whitened)
    basis = torch.linalg.svd(whitened, full_matrices=False).U[:, :rank]
    values = chol @ basis
    keys = torch.linalg.solve_triangular(chol.T, basis, upper=True)
    weight = load(f"model.layers.{layer}.self_attn.q_proj.weight")[head*128:(head+1)*128]
    delta = (gain - 1) * (weight @ values) @ keys.T
    bias = -(delta @ geometry["mean"])
    background_rms = float(torch.trace(delta @ covariance @ delta.T).clamp_min(0).sqrt() / 128**.5)
    return delta, bias, {"background_query_delta_rms": background_rms,
                         "relative_weight_change": float(delta.norm() / weight.norm())}


def audit(geometries, rank, gain, null_midpoints, split=True):
    combined, stats = {}, {}
    for label, geometry in geometries.items():
        layer, head = map(int, label[1:].split("H"))
        delta, bias, stats[label] = design(geometry, layer, head, rank, gain, null_midpoints, split=split)
        training = {row["id"] for row in geometry["rows"][:16]} if split else set()
        for case in geometry["cases"]:
            if case["id"] in training:
                continue
            shift_query = delta @ case["query_input"] + bias
            rotated = rope(shift_query[None,None], torch.tensor([case["position"]]))[0,0]
            before = ((case["key"] @ case["query"]) / 128**.5).softmax(-1)
            after = ((case["key"] @ (case["query"] + rotated)) / 128**.5).softmax(-1)
            shift = float(((after - before) @ case["value"]) @ case["readout"])
            key = (case["id"], case["digit"])
            if key not in combined:
                combined[key] = [case["baseline_margin"], 0.]
            combined[key][1] += shift
    result = {"rank": rank, "gain": gain, "null_midpoints": null_midpoints, "projection": stats}
    for digit in (1, 2):
        values = [v for (pid, d), v in combined.items() if d == digit]
        before = torch.tensor([v[0] for v in values]); after = torch.tensor([sum(v) for v in values])
        result[f"T{digit}"] = {"n": len(values), "before_margin": float(before.mean()), "after_margin": float(after.mean()),
                               "consistent_before": int(((before > 0) if digit == 1 else (before < 0)).sum()),
                               "consistent_after": int(((after > 0) if digit == 1 else (after < 0)).sum())}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    geometries = torch.load(HERE / ("binding_wide_geometry.pt" if args.wide else "binding_qkv_geometry.pt"), weights_only=True)
    results = []
    for rank in ((4, 16, 32) if args.wide else (1, 4, 8)):
        for gain in (2., 4.):
            for null_midpoints in (False, True):
                result = audit(geometries, rank, gain, null_midpoints, split=not args.wide)
                results.append(result)
                print(json.dumps(result), flush=True)
    (HERE / ("query_wide_audit.json" if args.wide else "query_subspace_audit.json")).write_text(json.dumps(results, indent=2) + "\n")
