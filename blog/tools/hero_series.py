"""Hero images (1600 x 900) for the 3-part blog series: dark background, purple glow, blue = trainable.

All important content stays inside the central 1600 x 700 band (y = 100..800 px), because blog cards crop
the image to cover.

  .venv/bin/python blog/tools/hero_series.py [--out DIR]   -> blog/series/hero_part{1,2,3}.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "blog/series"

BG = "#0b0a1f"
PANEL = "#13122b"
TEXT = "#f0efec"
MUTED = "#a9a7b8"
BLUE = "#3987e5"
GRAY = "#6f6e69"
ACCENT = "#915eff"
QUANT_GRAYS = ["#5b5a56", "#6f6e69", "#83827c", "#4f4e4a"]  # "rounded to 4-bit levels" look

W_PX, H_PX, DPI = 1600, 900, 160

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "text.color": TEXT})


def block(ax, x: float, y: float, w: float, h: float, color: str) -> None:
    """Filled rectangle with lower-left corner (x, y)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=2))


def canvas(glow_x: float = 115):
    """Figure in 160 x 90 data units (10 px each) with the soft purple glow behind the art."""
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.set_aspect("equal")
    ax.axis("off")
    yy, xx = np.mgrid[0:90:300j, 0:160:520j]
    glow = np.exp(-(((xx - glow_x) / 36) ** 2 + ((yy - 45) / 30) ** 2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", [BG, "#1d1450"])
    ax.imshow(glow, extent=(0, 160, 0, 90), origin="lower", cmap=cmap, zorder=0, aspect="auto")
    return fig, ax


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def title_block(ax, kicker: str, line1: str, line2: str, sub: str, y: float = 62) -> None:
    ax.text(14, y + 8.5, kicker, fontsize=11, fontweight="bold", va="center", color=ACCENT)
    ax.text(14, y, line1, fontsize=24, fontweight="bold", va="center", color=TEXT)
    ax.text(14, y - 8.5, line2, fontsize=24, fontweight="bold", va="center", color=ACCENT)
    ax.text(14, y - 19, sub, fontsize=10.5, va="center", color=MUTED, linespacing=1.5)


def legend(ax, items: list[tuple[float, str | list[str], str]], y: float = 30.5) -> None:
    """One row of (x, colour, label) swatches centred on height y; 4 colours make a 2 x 2 swatch."""
    for x, color, label in items:
        if isinstance(color, list):
            for q, shade in enumerate(color):
                block(ax, x + 1.1 * (q % 2), y - 1.1 + 1.1 * (q // 2), 1.1, 1.1, shade)
        else:
            block(ax, x, y - 1.1, 2.2, 2.2, color)
        ax.text(x + 3.4, y, label, fontsize=9.5, va="center")


def _matrix_cell_color(k: int, i: int, j: int, n: int, rng) -> str:
    """Cell colour of method k's 12 x 12 matrix (0 full FT, 1 LoRA, 2 LoRA-FA, 3 QLoRA, 4 TinyLoRA)."""
    if k == 0:
        return BLUE
    if k == 4:
        return GRAY
    band = 2
    in_b = i < band
    in_a = j >= n - band
    if in_b or (in_a and k != 2):
        return BLUE
    if k == 3:
        return QUANT_GRAYS[rng.integers(4)]
    return GRAY


def part1(out: Path) -> None:
    fig, ax = canvas()
    title_block(
        ax,
        "PART 1 OF 3  ·  FINE-TUNING, EXPLAINED",
        "5 ways to teach",
        "a language model.",
        "Full fine-tuning, LoRA, LoRA-FA, QLoRA and TinyLoRA,\n"
        "from one grid of numbers down to 13 trainable numbers",
    )
    legend(ax, [(14, BLUE, "trainable"), (32, GRAY, "frozen"), (47, QUANT_GRAYS, "frozen, stored in 4-bit")])
    rng = np.random.default_rng(7)
    n, c = 12, 1.2
    positions = [(93, 50), (110, 50), (127, 50), (101.5, 25), (118.5, 25)]
    labels = ["Full FT", "LoRA", "LoRA-FA", "QLoRA", "TinyLoRA"]
    for k, ((x, y), lab) in enumerate(zip(positions, labels, strict=True)):
        for i in range(n):
            for j in range(n):
                color = _matrix_cell_color(k, i, j, n, rng)
                ax.add_patch(
                    Rectangle(
                        (x + i * c, y + j * c),
                        c * 0.86,
                        c * 0.86,
                        facecolor=color,
                        edgecolor="none",
                        zorder=2,
                    )
                )
        if k == 4:  # TinyLoRA: 13 trainable cells scattered over the frozen matrix
            for cell in rng.choice(n * n, size=13, replace=False):
                i, j = divmod(int(cell), n)
                ax.add_patch(
                    Rectangle(
                        (x + i * c, y + j * c), c * 0.86, c * 0.86, facecolor=BLUE, edgecolor="none", zorder=3
                    )
                )
        ax.text(x + n * c / 2, y - 2.4, lab, ha="center", va="center", fontsize=8.5, color=MUTED)
    save(fig, out / "hero_part1.png")


def _code_window(ax, x0: float, y0: float, w: float, h: float) -> None:
    """Editor-style panel with traffic-light dots and a syntax-coloured SQL query."""
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=PANEL,
            edgecolor="#26244a",
            linewidth=1.2,
            zorder=2,
        )
    )
    for dx, col in [(3, "#ff5f57"), (5.6, "#febc2e"), (8.2, "#28c840")]:
        ax.add_patch(plt.Circle((x0 + dx, y0 + h - 2.6), 0.75, color=col, zorder=3))
    code = [
        [("SELECT", BLUE), (" T1.bond_id", TEXT)],
        [("FROM", BLUE), (" bond AS T1", TEXT)],
        [("JOIN", BLUE), (" molecule AS T2", TEXT)],
        [("  ON", BLUE), (" T1.molecule_id = T2.molecule_id", TEXT)],
        [("WHERE", BLUE), (" T2.label = '+'", TEXT)],
    ]
    char_w = 1.227  # monospace advance at fontsize 9.2, in data units
    for r, parts in enumerate(code):
        col_chars = 0
        for word, col in parts:
            ax.text(
                x0 + 3 + col_chars * char_w,
                y0 + h - 7 - r * 3.0,
                word,
                fontsize=9.2,
                color=col,
                family="DejaVu Sans Mono",
                va="center",
                zorder=3,
                fontweight="bold" if col == BLUE else "normal",
            )
            col_chars += len(word)


def part2(out: Path) -> None:
    fig, ax = canvas()
    title_block(
        ax,
        "PART 2 OF 3  ·  HOW I TESTED IT",
        "Text-to-SQL on BIRD,",
        "one laptop GPU.",
        "1,534 questions, 11 real databases, every answer\n"
        "graded by running the SQL. 16 GB of VRAM, 6 hours.",
    )
    _code_window(ax, 84, 52, 62, 22)
    rng = np.random.default_rng(3)
    nx, ny, c = 26, 6, 2.38
    gx, gy = 84, 22
    for i in range(nx):
        for j in range(ny):
            ok = rng.random() < 0.42
            ax.add_patch(
                Rectangle(
                    (gx + i * c, gy + j * c),
                    c * 0.82,
                    c * 0.82,
                    facecolor=BLUE if ok else GRAY,
                    edgecolor="none",
                    zorder=2,
                )
            )
    ax.text(
        gx,
        gy - 2.6,
        "each square: one question, run on the database and graded",
        fontsize=8.5,
        color=MUTED,
        va="center",
    )
    legend(ax, [(14, BLUE, "right rows"), (34, GRAY, "wrong or error")])
    save(fig, out / "hero_part2.png")


def part3(out: Path) -> None:
    fig, ax = canvas()
    title_block(
        ax,
        "PART 3 OF 3  ·  THE RESULTS",
        "13 parameters vs.",
        "74 million.",
        "Execution accuracy on BIRD dev for all five methods,\n"
        "what each run cost, and what the 13 numbers learned",
    )
    runs = [
        ("base", "Base"),
        ("full", "Full FT"),
        ("lora", "LoRA"),
        ("lora_fa", "LoRA-FA"),
        ("qlora", "QLoRA"),
        ("tinylora_sft", "Tiny SFT"),
        ("tinylora_grpo", "Tiny GRPO"),
    ]
    ex = {r: json.loads((ROOT / f"results/{r}/metrics.json").read_text())["ex"] for r, _ in runs}
    L, B, top = 86, 22, 70
    bw, gap = 6.2, 3.3
    vmax = 50.0
    for k, (r, lab) in enumerate(runs):
        x = L + k * (bw + gap)
        hgt = ex[r] / vmax * (top - B)
        col = GRAY if r == "base" else (ACCENT if r.startswith("tiny") else BLUE)
        ax.add_patch(
            FancyBboxPatch(
                (x, B),
                bw,
                hgt,
                boxstyle="round,pad=0,rounding_size=0.6",
                facecolor=col,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            x + bw / 2,
            B + hgt + 1.8,
            f"{ex[r]:.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT,
            zorder=3,
        )
        ax.text(x + bw / 2, B - 2.4, lab, ha="center", va="center", fontsize=7.4, color=MUTED, rotation=0)
    base_y = B + ex["base"] / vmax * (top - B)
    ax.plot(
        [L - 1.5, L + len(runs) * (bw + gap) - gap + 1.5],
        [base_y, base_y],
        color=MUTED,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        zorder=3,
    )
    ax.text(L, top + 4, "BIRD dev execution accuracy (%)", fontsize=9, color=MUTED, va="center")
    legend(ax, [(14, BLUE, "41M to 1.54B trained numbers"), (52, ACCENT, "13 trained numbers")])
    save(fig, out / "hero_part3.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the three blog hero images.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (part1, part2, part3):
        fn(args.out)
    print("wrote", ", ".join(str(args.out / f"hero_part{i}.png") for i in (1, 2, 3)))


if __name__ == "__main__":
    main()
