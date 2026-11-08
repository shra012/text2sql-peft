"""TinyLoRA tests: CPU checks on a tiny random Qwen2, plus CUDA-only checks on the real 1.5B model.

Run with `pytest tests/test_tinylora.py` or `.venv/bin/python tests/test_tinylora.py`.
Override the real model with T2S_TEST_MODEL.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from t2s.tinylora import (
    apply_tinylora,
    load_tinylora,
    merge_and_unload,
    merged_weights,
    save_tinylora,
    tinylora_modules,
    tinylora_parameters,
)

MODEL = os.environ.get("T2S_TEST_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
N_TARGETS = 7  # q, k, v, o, gate, up, down per decoder layer
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (loads the real model)")


def _logits(model, inp: dict) -> torch.Tensor:
    with torch.no_grad():
        return model(**inp).logits.float()


def _n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_v(model, scale: float) -> None:
    torch.manual_seed(123)
    with torch.no_grad():
        tinylora_parameters(model)[0].copy_(torch.randn(13) * scale)


def _merge_and_reload(make_model: Callable, inp: dict, v_scale: float) -> dict[str, torch.Tensor]:
    """Apply TinyLoRA, set v, save, merge, reload onto a fresh base. Checks structure along the
    way and returns the base / adapter / merged / reloaded logits for tolerance checks."""
    model = make_model()
    n_layers = model.config.num_hidden_layers
    out = {"base": _logits(model, inp)}
    apply_tinylora(model, u=13, r=2, tie="all", seed=0)
    _set_v(model, v_scale)
    out["adapter"] = _logits(model, inp)

    mw = merged_weights(model)
    assert "model.layers.0.self_attn.q_proj.weight" in mw
    assert len(mw) == N_TARGETS * n_layers

    with tempfile.TemporaryDirectory() as d:
        save_tinylora(model, d)
        cfg = json.loads((Path(d) / "tinylora_config.json").read_text())
        assert cfg["trainable_params"] == 13 and cfg["n_modules"] == N_TARGETS * n_layers
        v_saved = tinylora_parameters(model)[0].detach().cpu().clone()
        P_saved = {n: m.P.detach().cpu().clone() for n, m in tinylora_modules(model)}

        merge_and_unload(model)
        assert not tinylora_modules(model)
        assert not hasattr(model, "tinylora_state")
        out["merged"] = _logits(model, inp)
        name = "model.layers.0.self_attn.q_proj.weight"
        assert torch.equal(model.state_dict()[name], mw[name])
        del model, mw

        reloaded = make_model()
        load_tinylora(reloaded, d)
        assert torch.equal(tinylora_parameters(reloaded)[0].cpu(), v_saved)
        for n, m in tinylora_modules(reloaded):
            assert torch.equal(m.P.cpu(), P_saved[n]), n
        out["reloaded"] = _logits(reloaded, inp)
    return out


def _check_grad_flows_to_v_only(model, inp: dict, autocast: bool) -> None:
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    apply_tinylora(model, u=13, r=2, tie="all")
    with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=autocast):
        out = model(**inp, labels=inp["input_ids"])
    out.loss.backward()
    v = tinylora_parameters(model)[0]
    assert v.grad is not None and v.grad.abs().sum() > 0 and v.grad.dtype == torch.float32
    others = [n for n, p in model.named_parameters() if p is not v and p.grad is not None]
    assert not others, others[:5]
    torch.optim.AdamW([v], lr=1e-2, weight_decay=0.0).step()
    assert v.abs().sum() > 0


# ----------------------------------------------------------------------------- CPU (tiny random Qwen2)


def _tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)  # same weights on every call, so a reload sees the same base
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg).float().cpu().eval()


def _tiny_inputs() -> dict:
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_tiny_param_counts():
    n_layers = 2
    for tie, expected in [("all", 13), ("layer", 13 * n_layers), ("module", 13 * N_TARGETS * n_layers)]:
        model = apply_tinylora(_tiny_model(), u=13, r=2, tie=tie, svd_device="cpu")
        assert _n_trainable(model) == expected, (tie, _n_trainable(model))
        assert sum(p.numel() for p in tinylora_parameters(model)) == expected
        assert len(tinylora_modules(model)) == N_TARGETS * n_layers


def test_tiny_zero_init_matches_base():
    model, inp = _tiny_model(), _tiny_inputs()
    ref = _logits(model, inp)
    apply_tinylora(model, u=13, r=2, tie="all", svd_device="cpu")
    assert torch.equal(_logits(model, inp), ref)


def test_tiny_merge_and_save_load():
    lg = _merge_and_reload(_tiny_model, _tiny_inputs(), v_scale=5.0)
    effect = (lg["adapter"] - lg["base"]).abs().max()
    assert effect > 1e-4, f"adapter had no effect ({effect})"
    torch.testing.assert_close(lg["merged"], lg["adapter"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lg["reloaded"], lg["adapter"], rtol=0, atol=0)


def test_tiny_full_merged_state_matches_plain_model():
    model = apply_tinylora(_tiny_model(), u=13, r=2, tie="all", svd_device="cpu")
    full = merged_weights(model, only_adapted=False)
    plain_names = {n for n, _ in _tiny_model().named_parameters()}
    assert set(full) == plain_names


def test_tiny_grad_flows_to_v_only():
    _check_grad_flows_to_v_only(_tiny_model(), _tiny_inputs(), autocast=False)


# ----------------------------------------------------------------------------- CUDA (real model)

_TOK: dict = {}


def _base() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()


def _inputs() -> dict:
    if "tok" not in _TOK:
        _TOK["tok"] = AutoTokenizer.from_pretrained(MODEL)
    text = "SELECT name FROM singer WHERE age > 30 ORDER BY age DESC; -- count the singers"
    return _TOK["tok"](text, return_tensors="pt").to("cuda")


@requires_cuda
def test_param_count_and_zero_init():
    model, inp = _base(), _inputs()
    ref = _logits(model, inp)
    apply_tinylora(model, u=13, r=2, tie="all")
    assert _n_trainable(model) == 13
    assert sum(p.numel() for p in tinylora_parameters(model)) == 13
    out = _logits(model, inp)
    assert torch.equal(out, ref), (out - ref).abs().max()
    del model
    torch.cuda.empty_cache()

    model = apply_tinylora(_base(), u=13, r=2, tie="layer")
    assert _n_trainable(model) == 13 * model.config.num_hidden_layers
    del model
    torch.cuda.empty_cache()


@requires_cuda
def test_merge_matches_adapter_and_save_load():
    lg = _merge_and_reload(_base, _inputs(), v_scale=0.05)
    torch.cuda.empty_cache()
    effect = (lg["adapter"] - lg["base"]).abs().mean()
    assert (lg["adapter"] - lg["base"]).abs().max() > 1e-3, "adapter had no effect"
    # Merged W' is rounded to bf16 while the adapter path is fp32, hence the loose tolerance.
    diff = (lg["merged"] - lg["adapter"]).abs()
    print(f"adapter vs merged: max|d|={diff.max():.4f} mean|d|={diff.mean():.5f} (effect {effect:.4f})")
    assert diff.mean() < 0.05 and diff.max() < 1.0
    assert diff.mean() < 0.25 * effect, (diff.mean(), effect)
    reload_diff = (lg["reloaded"] - lg["adapter"]).abs().max()
    print(f"reloaded vs adapter: max|d|={reload_diff:.5f}")
    assert reload_diff < 1e-2


@requires_cuda
def test_grad_flows_to_v_only():
    _check_grad_flows_to_v_only(_base(), _inputs(), autocast=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    import time

    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
        except unittest.SkipTest as e:
            print(f"SKIP {name}: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            continue
        print(f"PASS {name} ({time.time() - t:.1f}s)")
    sys.exit(1 if failed else 0)
