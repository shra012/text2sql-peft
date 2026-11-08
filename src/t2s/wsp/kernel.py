"""Weight-space programming: the editor's CPU-only view of the weights.

Mirrors the MIT CSAIL "Sparks of Weight-Space Programming" harness: the editor reads raw weights,
computes whatever it wants on CPU, and submits replacement tensors with save_edit(). No gradients,
no training, and the model is never run here (the forward-pass tool in t2s.wsp.tool does that,
under a fixed budget).

    from t2s.wsp.kernel import *
    names()                       # all parameter names
    W = load("model.layers.15.mlp.down_proj.weight")    # original weight, float32, CPU
    W2 = current(name)            # edited version if one was saved, else original
    save_edit(name, new_tensor)   # validated, stored as bf16 in wsp/edits/
"""

from __future__ import annotations

import os

# Must happen before torch is imported: the kernel never touches the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from t2s.wsp.common import BASE_FILE, EDITS, append_edit_log, base_dir, edit_path, list_edits

__all__ = ["names", "shape", "load", "current", "save_edit", "remove_edit", "list_edits", "config", "torch"]

torch.set_grad_enabled(False)

FORBIDDEN = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}

_handle = safe_open(str(BASE_FILE), "pt")


def config() -> dict:
    return json.loads((base_dir() / "config.json").read_text())


def names() -> list[str]:
    return sorted(_handle.keys())


def shape(name: str) -> tuple:
    return tuple(_handle.get_slice(name).get_shape())


def load(name: str) -> torch.Tensor:
    """Original (unedited) weight as float32 on CPU."""
    return _handle.get_tensor(name).float()


def current(name: str) -> torch.Tensor:
    """Edited weight if an edit is saved for `name`, else the original. float32, CPU."""
    path = edit_path(name)
    return load_file(str(path))["tensor"].float() if path.exists() else load(name)


def remove_edit(name: str) -> None:
    edit_path(name).unlink(missing_ok=True)
    append_edit_log({"action": "remove", "name": name})


def _validate(tensor: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
    """Return `tensor` as float32 CPU, or raise if it is not an acceptable replacement for `orig`."""
    t = torch.as_tensor(tensor).detach().float().cpu()
    if tuple(t.shape) != tuple(orig.shape):
        raise ValueError(f"shape {tuple(t.shape)} != {tuple(orig.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("tensor contains NaN or inf")
    # Least-squares scale c of t onto orig; reject t == c * orig for any c != 1.
    c = float((t * orig).sum() / (orig * orig).sum())
    if abs(c - 1) > 1e-6 and torch.allclose(t, c * orig, rtol=1e-4, atol=1e-7):
        raise ValueError(f"edits that only multiply the whole tensor by a constant ({c:.4f}) are not allowed")
    return t


def save_edit(name: str, tensor: torch.Tensor) -> dict:
    """Replace parameter `name` with `tensor`. Validated; stored as bf16 (what the model will use)."""
    if name not in _handle.keys():  # noqa: SIM118 - safe_open handles are not containers
        raise ValueError(f"unknown parameter {name}")
    if name in FORBIDDEN:
        raise ValueError(f"{name} may not be edited (embeddings / output layer / final norm are off limits)")
    orig = load(name)
    new_bf16 = _validate(tensor, orig).to(torch.bfloat16)
    changed = int((new_bf16 != orig.to(torch.bfloat16)).sum())
    if changed == 0:
        raise ValueError("edit does not change any weight after bf16 rounding")
    EDITS.mkdir(parents=True, exist_ok=True)
    save_file({"tensor": new_bf16.contiguous()}, str(edit_path(name)))
    delta = new_bf16.float() - orig
    stats = {
        "name": name,
        "changed": changed,
        "numel": orig.numel(),
        "delta_fro": float(delta.norm()),
        "rel_change": float(delta.norm() / orig.norm()),
    }
    append_edit_log({"action": "save", **stats})
    return stats
