"""Animated results chart for the README: docs/results.gif.

    .venv/bin/python blog/tools/readme_gif.py [--out docs/results.gif]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CARD, INK, INK2, GRID, BASE_GREY = "#FCFCFB", "#0b0b0b", "#52514e", "#E2E8F0", "#94a3b8"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
ROWS = [  # run, label, what changes, colour
    ("qlora", "QLoRA", "74M numbers, 4-bit base", BLUE),
    ("lora", "LoRA", "74M numbers", BLUE),
    ("lora_fa", "LoRA-FA", "41M numbers", BLUE),
    ("full", "Full fine-tuning", "all 1.54B numbers", ORANGE),
    ("tinylora_grpo", "TinyLoRA + GRPO", "13 numbers", AQUA),
    ("wsp_f4", "LLM edits weights", "48K weights, no training", VIOLET),
]
GROW, HOLD, W, H, DPI = 28, 26, 9.0, 4.6, 100


def scores() -> tuple[float, list[float]]:
    ex = lambda run: json.loads((ROOT / f"results/{run}/metrics.json").read_text())["ex"]
    return ex("base"), [ex(run) for run, *_ in ROWS]


def ease(t: float) -> float:
    return 1 - (1 - t) ** 3


def frame(base: float, finals: list[float], t: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    lo, hi = 20, 49
    for i, ((_, label, what, color), final) in enumerate(zip(ROWS, finals, strict=True)):
        y = len(ROWS) - 1 - i
        value = base + (final - base) * ease(t)
        ax.barh(y, value - lo, left=lo, height=0.62, color=color, zorder=3)
        ax.text(lo - 0.4, y + 0.1, label, ha="right", va="center", fontsize=11.5, color=INK, weight="bold")
        ax.text(lo - 0.4, y - 0.24, what, ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(value + 0.35, y, f"{value:.1f}%", va="center", fontsize=11, color=INK, weight="bold")
        if t >= 1:
            ax.text(value + 3.1, y, f"+{final - base:.1f} pts", va="center", fontsize=9.5, color=INK2)
    ax.axvline(base, color=BASE_GREY, linestyle=(0, (4, 3)), linewidth=1.6, zorder=4)
    ax.text(base, len(ROWS) - 0.35, f"base model {base:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(ROWS) - 0.1)
    ax.set_yticks([])
    ax.set_xticks(range(lo, 46, 5), [f"{v}%" for v in range(lo, 46, 5)], fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.text(0.02, 0.94, "Six ways to change a 1.5B model, scored on BIRD text-to-SQL", fontsize=14,
             weight="bold", color=INK)
    subtitle = "Execution accuracy on 1,534 BIRD dev questions · Qwen2.5-Coder-1.5B · one 16 GB laptop GPU"
    fig.text(0.02, 0.885, subtitle, fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.83, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=CARD)
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/results.gif")
    out = ap.parse_args().out
    base, finals = scores()
    frames = [frame(base, finals, i / (GROW - 1)) for i in range(GROW)]
    frames += [frames[-1]] * HOLD
    palette = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[45] * GROW + [120] * HOLD,
                   loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
