"""Part 3 chart p3_format_shift: how the answer *format* changed, from results/<run>/preds.jsonl.

Shows the share of answers that start straight with a ```sql block and the mean answer length.

  .venv/bin/python blog/tools/svg_series_part3.py [--out DIR]   -> blog/svg/p3_*.svg
"""

from __future__ import annotations

import json
from html import escape

from svg_charts import BLUE, FROZEN, INK, MUTED, ROOT, legend_item, svg, t, write_svgs

N_DEV = 1534
ROWS = [
    ("base", "Base model"),
    ("tinylora_sft", "TinyLoRA + SFT"),
    ("tinylora_grpo", "TinyLoRA + GRPO"),
    ("wsp_f4", "Weight-space programming"),
    ("lora", "LoRA"),
]


def format_stats(run: str) -> tuple[int, int, float]:
    """(answers starting with ```sql, number of answers, mean answer length in characters)."""
    with open(ROOT / f"results/{run}/preds.jsonl") as f:
        preds = [json.loads(line) for line in f]
    starts = sum(p["raw"].lstrip().startswith("```sql") for p in preds)
    mean_len = sum(len(p["raw"]) for p in preds) / len(preds)
    return starts, len(preds), mean_len


def chart_format_shift():
    W, H = 1000, 356
    stats = {r: format_stats(r) for r, _ in ROWS}
    b = ""
    panels = [
        ("Answers that start straight with ```sql", 0, 40),
        ("Average answer length (characters)", 1, 520),
    ]
    for title, which, x0 in panels:
        L, R, T = x0 + 150, x0 + 400, 128
        b += t(x0, 110, title, 14, INK, "start", "700")
        vmax = N_DEV if which == 0 else 700
        for i, (r, name) in enumerate(ROWS):
            starts, n, mean_len = stats[r]
            y = T + i * 44
            v = starts if which == 0 else mean_len
            w = v / vmax * (R - L)
            label = f"{starts:,} of {n:,}" if which == 0 else f"{mean_len:,.0f}"
            fill, stroke = (FROZEN, MUTED) if r == "base" else (BLUE, BLUE)
            if len(name) > 18 and " " in name:  # long names wrap onto two lines beside their bar
                first, second = name.rsplit(" ", 1)
                b += (
                    f'  <text x="{L - 10:.1f}" y="{y + 13:.1f}" font-size="13" fill="{INK}" '
                    f'text-anchor="end" font-weight="400"><tspan x="{L - 10:.1f}">{escape(first)}</tspan>'
                    f'<tspan x="{L - 10:.1f}" dy="14">{escape(second)}</tspan></text>\n'
                )
            else:
                b += t(L - 10, y + 19, name, 13, INK, "end")
            b += (
                f'  <rect x="{L}" y="{y + 4}" width="{max(w, 2):.1f}" height="22" rx="4" fill="{fill}" '
                f'stroke="{stroke}"><title>{escape(f"{name}: {label}")}</title></rect>\n'
            )
            b += t(L + w + 8, y + 20, label, 12.5, INK, "start", "600")
    b += legend_item(660, 52, FROZEN, "untrained", MUTED)
    b += legend_item(770, 52, BLUE, "changed")
    return svg(
        W,
        H,
        b,
        "What the tiny changes did: the shape of the answer",
        "All 1,534 BIRD dev answers. LoRA shown for comparison.",
    )


if __name__ == "__main__":
    write_svgs([("p3_format_shift", chart_format_shift)], "Part 3 answer-format chart (inline SVG).")
