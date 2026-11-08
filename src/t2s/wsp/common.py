"""Shared paths and edit storage for weight-space programming (no CUDA side effects)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]
WSP = ROOT / "wsp"
EDITS = WSP / "edits"
EDIT_LOG = WSP / "edit_log.jsonl"
BASE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EDIT_SUFFIX = ".safetensors"


def base_dir() -> Path:
    """Local Hugging Face snapshot directory of the base model."""
    snaps = Path.home() / ".cache/huggingface/hub" / f"models--{BASE_ID.replace('/', '--')}" / "snapshots"
    return next(p for p in snaps.iterdir() if (p / "model.safetensors").exists())


BASE_FILE = base_dir() / "model.safetensors"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_edit_log(record: dict) -> None:
    """Append a timestamped record to wsp/edit_log.jsonl."""
    WSP.mkdir(parents=True, exist_ok=True)
    with open(EDIT_LOG, "a") as f:
        f.write(json.dumps({"t": timestamp(), **record}) + "\n")


def edit_path(name: str, edits_dir: Path = EDITS) -> Path:
    return edits_dir / f"{name}{EDIT_SUFFIX}"


def list_edits(edits_dir: Path = EDITS) -> list[str]:
    """Names of the parameters with a saved edit, sorted."""
    if not edits_dir.exists():
        return []
    return sorted(p.name[: -len(EDIT_SUFFIX)] for p in edits_dir.glob(f"*{EDIT_SUFFIX}"))


def edits_fingerprint(edits_dir: Path = EDITS) -> str:
    """Short hash of the edit names and file contents in edits_dir."""
    h = hashlib.sha256()
    for name in list_edits(edits_dir):
        h.update(name.encode())
        h.update(edit_path(name, edits_dir).read_bytes())
    return h.hexdigest()[:16]


def build_merged(out_dir: Path, edits_dir: Path = EDITS) -> Path:
    """Write a full HF checkpoint (base + edits in edits_dir); skipped if those edits are unchanged."""
    fingerprint = edits_fingerprint(edits_dir)
    marker = out_dir / "EDITS_FINGERPRINT"
    if marker.exists() and marker.read_text() == fingerprint:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_file(str(BASE_FILE))
    for name in list_edits(edits_dir):
        state[name] = load_file(str(edit_path(name, edits_dir)))["tensor"].to(state[name].dtype)
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in base_dir().iterdir():
        if f.name != "model.safetensors" and f.is_file():
            (out_dir / f.name).write_bytes(f.read_bytes())
    marker.write_text(fingerprint)
    return out_dir
