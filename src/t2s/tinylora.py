"""TinyLoRA adapters (arXiv:2602.04118, "Learning to Reason in 13 Parameters")."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
STATE_ATTR = "tinylora_state"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class TinyLoRAState(nn.Module):
    """The trainable v vectors (one per tie key) plus the TinyLoRA config."""

    def __init__(self, config: dict, keys: list[str], u: int):
        super().__init__()
        self.config = dict(config)
        self.v = nn.ParameterDict({k: nn.Parameter(torch.zeros(u, dtype=torch.float32)) for k in keys})


class TinyLoRALinear(nn.Module):
    """Frozen nn.Linear plus the fp32 adapter path: y = base(x) + ((x @ V) @ R^T) @ US^T."""

    def __init__(
        self,
        base: nn.Linear,
        US: torch.Tensor,
        V: torch.Tensor,
        P: torch.Tensor,
        state: TinyLoRAState,
        v_key: str,
    ):
        super().__init__()
        self.base = base
        # Non-persistent so model.state_dict() keeps only the original weights.
        self.register_buffer("US", US.float().contiguous(), persistent=False)  # d_out x r
        self.register_buffer("V", V.float().contiguous(), persistent=False)  # d_in x r
        self.register_buffer("P", P.float().contiguous(), persistent=False)  # u x r x r
        self.v_key = v_key
        # A plain list keeps the shared state from being registered as a submodule of every
        # wrapper (which would duplicate its parameters and state_dict entries).
        self._state = [state]

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:  # some HF utilities peek at .weight
        return self.base.weight

    def v(self) -> torch.Tensor:
        return self._state[0].v[self.v_key]

    def R(self) -> torch.Tensor:
        return torch.einsum("u,uij->ij", self.v().float(), self.P)

    def delta_weight(self) -> torch.Tensor:
        """The fp32 (d_out x d_in) update US R V^T."""
        return self.US @ self.R() @ self.V.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            d = ((x.to(torch.float32) @ self.V) @ self.R().T) @ self.US.T
        return y + d.to(y.dtype)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.V.shape[1]}, "
            f"u={self.P.shape[0]}, v_key={self.v_key!r}"
        )


def _svd_bases(W: torch.Tensor, r: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-r truncated SVD of W in fp32 on `device`; returns US (d_out x r) and V (d_in x r)."""
    U, S, Vh = torch.linalg.svd(W.detach().to(device=device, dtype=torch.float32), full_matrices=False)
    U, S, V = U[:, :r], S[:r], Vh[:r].T.contiguous()
    # Singular pairs are only defined up to a joint sign flip, which changes US R V^T for a
    # non-diagonal R. Make the largest-|.| entry of each V column positive so the bases are
    # reproducible when recomputed on load.
    idx = V.abs().argmax(dim=0)
    sign = torch.sign(V[idx, torch.arange(r, device=device)])
    sign[sign == 0] = 1.0
    return U * sign * S, V * sign


def _v_key(name: str, tie: str) -> str:
    if tie == "all":
        return "all"
    if tie == "layer":
        m = _LAYER_RE.search(name)
        if m is None:
            raise ValueError(f"cannot find layer index in module name {name!r}")
        return f"layer_{int(m.group(1))}"
    if tie == "module":
        return name.replace(".", "__")
    raise ValueError(f"unknown tie mode {tie!r}")


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def get_state(model: nn.Module) -> TinyLoRAState:
    state = getattr(model, STATE_ATTR, None)
    if state is None:
        raise ValueError("model has no TinyLoRA adapters (call apply_tinylora first)")
    return state


def tinylora_modules(model: nn.Module) -> list[tuple[str, TinyLoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, TinyLoRALinear)]


def tinylora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(get_state(model).v.values())


def apply_tinylora(
    model: nn.Module,
    u: int = 13,
    r: int = 2,
    tie: str = "all",
    targets: Iterable[str] = DEFAULT_TARGETS,
    seed: int = 0,
    share_P: bool = False,
    svd_device: str | torch.device | None = None,
) -> nn.Module:
    """Freeze `model` and wrap every target nn.Linear with TinyLoRALinear, in place.

    Each wrapped weight W becomes W' = W + U S R V^T with R = sum_i v_i P_i (r x r), where U, S, V
    are the rank-r truncated SVD of W, P (u x r x r) is a fixed Gaussian scaled by 1/sqrt(r), and
    v (length u, fp32, zero-init so W' == W at step 0) is the only trainable tensor, tied as:
    tie="all" one v for every module (u params), "layer" one per decoder layer, "module" one per module.

    P is drawn per module, in module order, from a CPU generator seeded with `seed` (or once
    for all modules if share_P). SVD runs in fp32 on `svd_device` (default: CUDA if available).
    """
    if hasattr(model, STATE_ATTR):
        raise ValueError("TinyLoRA already applied to this model")
    targets = tuple(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    if svd_device is None:
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
    svd_device = torch.device(svd_device)

    linears = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets
    ]
    if not linears:
        raise ValueError(f"no nn.Linear modules matching {targets}")

    v_keys = [_v_key(name, tie) for name, _ in linears]
    config = dict(u=u, r=r, tie=tie, targets=list(targets), seed=seed, share_P=share_P)
    state = TinyLoRAState(config, list(dict.fromkeys(v_keys)), u)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / (r**0.5)
    shared_P = torch.randn(u, r, r, generator=gen) * scale if share_P else None
    for (name, lin), v_key in zip(linears, v_keys, strict=True):
        P = shared_P if share_P else torch.randn(u, r, r, generator=gen) * scale
        US, V = _svd_bases(lin.weight, r, svd_device)
        dev = lin.weight.device
        _set_submodule(model, name, TinyLoRALinear(lin, US.to(dev), V.to(dev), P.to(dev), state, v_key))

    first_dev = next(model.parameters()).device
    setattr(model, STATE_ATTR, state.to(first_dev))
    return model


@torch.no_grad()
def merged_weights(
    model: nn.Module, only_adapted: bool = True, dtype: torch.dtype | None = None
) -> dict[str, torch.Tensor]:
    """{original HF param name: tensor} with W' = W + US R V^T for adapted weights (fp32 math, cast
    to `dtype` or the base dtype). only_adapted=False adds every other parameter too, i.e. a full
    state for loading into e.g. vLLM."""
    out: dict[str, torch.Tensor] = {}
    adapted_prefixes = set()
    for name, mod in tinylora_modules(model):
        W, bias = mod.base.weight, mod.base.bias
        out[f"{name}.weight"] = (W.float() + mod.delta_weight()).to(dtype or W.dtype)
        adapted_prefixes.add(name + ".base.")
        if bias is not None and not only_adapted:
            out[f"{name}.bias"] = bias.to(dtype or bias.dtype)
    if only_adapted:
        return out
    for pname, p in model.named_parameters():
        if pname.startswith(STATE_ATTR + ".") or any(pname.startswith(pref) for pref in adapted_prefixes):
            continue
        out[pname] = p.detach().to(dtype or p.dtype)
    return out


@torch.no_grad()
def merge_and_unload(model: nn.Module) -> nn.Module:
    """Bake W' into the base Linears and remove the wrappers and the TinyLoRA state."""
    for name, mod in tinylora_modules(model):
        W = mod.base.weight
        W.copy_((W.float() + mod.delta_weight()).to(W.dtype))
        _set_submodule(model, name, mod.base)
    if hasattr(model, STATE_ATTR):
        delattr(model, STATE_ATTR)
    return model


def save_tinylora(model: nn.Module, out_dir: str | Path) -> None:
    """Save every v, the per-module P and the config; US/V are recomputed on load."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = get_state(model)
    modules = tinylora_modules(model)
    tensors = {f"v.{k}": p.detach().float().cpu().clone() for k, p in state.v.items()}
    for name, mod in modules:
        tensors[f"P.{name}"] = mod.P.detach().float().cpu().clone()  # clone: shared P must not share storage
    save_file(tensors, str(out_dir / "tinylora.safetensors"))

    cfg = dict(state.config)
    cfg["n_modules"] = len(modules)
    cfg["trainable_params"] = sum(p.numel() for p in state.v.values())
    base_name = getattr(getattr(model, "config", None), "_name_or_path", None)
    if base_name:
        cfg["base_model"] = base_name
    (out_dir / "tinylora_config.json").write_text(json.dumps(cfg, indent=2))


def load_tinylora(
    model: nn.Module, in_dir: str | Path, svd_device: str | torch.device | None = None
) -> nn.Module:
    """Apply TinyLoRA to a fresh base `model` from a saved config, then load v and P."""
    in_dir = Path(in_dir)
    cfg = json.loads((in_dir / "tinylora_config.json").read_text())
    apply_tinylora(
        model,
        u=cfg["u"],
        r=cfg["r"],
        tie=cfg["tie"],
        targets=tuple(cfg["targets"]),
        seed=cfg["seed"],
        share_P=cfg.get("share_P", False),
        svd_device=svd_device,
    )
    tensors = load_file(str(in_dir / "tinylora.safetensors"))
    with torch.no_grad():
        for k, p in get_state(model).v.items():
            p.copy_(tensors[f"v.{k}"].to(p.device))
        for name, mod in tinylora_modules(model):
            key = f"P.{name}"
            if key in tensors:  # otherwise keep the P regenerated from the seed
                mod.P.copy_(tensors[key].to(mod.P.device))
    return model
