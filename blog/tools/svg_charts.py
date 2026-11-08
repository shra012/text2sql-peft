"""Inline-SVG result charts for the blog, plus the shared house style used by svg_series_part*.py.

Style: light card #FCFCFB, #E2E8F0 border, bold title + grey subtitle, #2a78d6 blue. Palette slots 1-3
validated (light, all-pairs) with the dataviz validator; aqua's <3:1 contrast is relieved by the results
table shown right above the charts.

  .venv/bin/python blog/tools/svg_charts.py [--out DIR]   -> blog/svg/{methods,ex_*,training_cost,...}.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "blog/svg"

CARD, BORDER, INK, INK2, MUTED, GRID = "#FCFCFB", "#E2E8F0", "#0b0b0b", "#52514e", "#94a3b8", "#E8ECF2"
BLUE, ORANGE, AQUA, BLUE_SOFT = "#2a78d6", "#eb6834", "#1baf7a", "#EFF4FB"
FROZEN, WHITE = "#E5E7EB", "#ffffff"
FONT = "ui-sans-serif,system-ui,Helvetica,Arial,sans-serif"

RUNS = [
    ("base", "Base model"),
    ("full", "Full fine-tune"),
    ("lora", "LoRA"),
    ("lora_fa", "LoRA-FA"),
    ("qlora", "QLoRA"),
    ("tinylora_sft", "TinyLoRA + SFT"),
    ("tinylora_grpo", "TinyLoRA + GRPO"),
    ("wsp_f4", "Opus weight edit"),
]
WSP_CHANGED = 48_398  # weights changed by the LLM's direct edit (no training run, so no run_info.json)


def metrics(run: str) -> dict:
    return json.loads((ROOT / f"results/{run}/metrics.json").read_text())


def info(run: str) -> dict:
    if run == "wsp_f4":
        return {"trainable_params": WSP_CHANGED}
    p = ROOT / f"runs/{run}/run_info.json"
    return json.loads(p.read_text()) if p.exists() else {}


def svg(w: float, h: float, body: str, title: str, subtitle: str) -> str:
    """Wrap `body` in the house card: rounded border, bold title, grey subtitle."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'style="width:100%;min-width:660px;height:auto;display:block" font-family="{FONT}" role="img" '
        f'aria-label="{escape(title)}">\n'
        f'  <rect width="{w}" height="{h}" rx="14" fill="{CARD}" stroke="{BORDER}" stroke-width="2"/>\n'
        f'  <text x="28" y="40" font-size="19" font-weight="700" fill="{INK}">{escape(title)}</text>\n'
        f'  <text x="28" y="64" font-size="14" fill="{INK2}">{escape(subtitle)}</text>\n{body}</svg>'
    )


def t(
    x: float, y: float, s, size: float = 13, fill: str = INK2, anchor: str = "start", weight: str = "400"
) -> str:
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(str(s))}</text>\n'
    )


def legend_item(x: float, y: float, fill: str, label: str, stroke: str | None = None) -> str:
    """14 px colour swatch at (x, y) with its label to the right."""
    st = f' stroke="{stroke}"' if stroke else ""
    return f'  <rect x="{x}" y="{y}" width="14" height="14" rx="3" fill="{fill}"{st}/>\n' + t(
        x + 20, y + 12, label, 13, INK
    )


def y_grid(left: float, right: float, y_of: Callable[[float], float], values) -> str:
    """Horizontal grid lines with percent labels on the left."""
    b = ""
    for v in values:
        b += f'  <line x1="{left}" y1="{y_of(v):.1f}" x2="{right}" y2="{y_of(v):.1f}" stroke="{GRID}"/>\n'
        b += t(left - 10, y_of(v) + 4, f"{v}%", 12, INK2, "end")
    return b


def write_svgs(charts: list[tuple[str, Callable[[], str]]], description: str) -> None:
    """CLI shared by the generators: write each chart to <out>/<name>.svg."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default: blog/svg)")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in charts:
        (out / f"{name}.svg").write_text(fn())
        print("wrote", out / f"{name}.svg")


def chart_params() -> str:
    W, H = 1000, 470
    L, R, T, B = 80, 960, 100, 400
    xmin, xmax = 0.5, 10.0  # log10 params
    ymin, ymax = 26, 44
    X = lambda p: L + (math.log10(p) - xmin) / (xmax - xmin) * (R - L)
    Y = lambda v: B - (v - ymin) / (ymax - ymin) * (B - T)
    b = y_grid(L, R, Y, range(ymin, ymax + 1, 2))
    for e in range(1, 10):
        lab = str(10**e) if e <= 3 else f'10<tspan dy="-6" font-size="9">{e}</tspan>'
        b += (
            f'  <text x="{X(10**e):.1f}" y="{B + 22}" font-size="12" fill="{INK2}" '
            f'text-anchor="middle">{lab}</text>\n'
        )
    b += t(
        (L + R) / 2,
        B + 48,
        "Trainable numbers, or weights changed for the Opus edit (log scale)",
        13,
        INK2,
        "middle",
    )
    base = metrics("base")["ex"]
    b += (
        f'  <line x1="{L}" y1="{Y(base):.1f}" x2="{R}" y2="{Y(base):.1f}" stroke="{MUTED}" '
        f'stroke-width="2" stroke-dasharray="6 5"/>\n'
    )
    b += t(L + 8, Y(base) - 8, f"Base model, no training: {base:.2f}%", 13, INK2)
    pts = {r: (info(r)["trainable_params"], metrics(r)["ex"]) for r, _ in RUNS[1:]}
    label = {  # run: (name, dx, dy, anchor)
        "full": ("Full fine-tune", -12, -14, "end"),
        "lora": ("LoRA", 14, 18, "start"),
        "qlora": ("QLoRA", 14, -8, "start"),
        "lora_fa": ("LoRA-FA", -14, 5, "end"),
        "wsp_f4": ("Opus weight edit", 14, -12, "start"),
    }
    for r, (p, v) in pts.items():
        tip = f"{dict(RUNS)[r]}: {v:.2f}% with {p:,} trainable parameters"
        b += (
            f'  <circle cx="{X(p):.1f}" cy="{Y(v):.1f}" r="8" fill="{BLUE}" stroke="{CARD}" stroke-width="2">'
            f"<title>{escape(tip)}</title></circle>\n"
        )
        if r in label:
            name, dx, dy, anc = label[r]
            text = f"{name} {v:.2f}%" + (" (48,398 weights changed)" if r == "wsp_f4" else "")
            b += t(X(p) + dx, Y(v) + dy, text, 13, INK, anc, "600")
    p, v = pts["tinylora_sft"]  # both TinyLoRA runs sit on the same point: one shared label
    b += t(X(p) + 14, Y(v) - 10, f"TinyLoRA, SFT and GRPO: {v:.2f}% (both runs)", 13, INK, "start", "600")
    b += t(X(p) + 14, Y(v) + 8, "13 trainable numbers", 12, INK2)
    return svg(
        W,
        H,
        b,
        "Accuracy vs. how many numbers you train",
        "BIRD dev execution accuracy, 1,534 questions. Hover a point for exact values.",
    )


def _two_line_label(name: str) -> list[str]:
    if " + " in name:
        return name.split(" + ")
    return name.rsplit(" ", 1) if len(name) > 12 else [name]


def chart_difficulty() -> str:
    W, H = 1000, 470
    L, R, T, B = 70, 970, 110, 390
    ymax = 55
    Y = lambda v: B - v / ymax * (B - T)
    b = y_grid(L, R, Y, range(0, ymax + 1, 10))
    keys = [
        ("ex_simple", "Simple (925 Qs)", BLUE),
        ("ex_moderate", "Moderate (464)", ORANGE),
        ("ex_challenging", "Challenging (145)", AQUA),
    ]
    for i, (_, lab, col) in enumerate(keys):
        b += legend_item(28 + 170 * i, 84, col, lab)
    gw = (R - L) / len(RUNS)
    bw = 26
    for i, (r, name) in enumerate(RUNS):
        m = metrics(r)
        cx = L + gw * (i + 0.5)
        for j, (k, lab, col) in enumerate(keys):
            x = cx + (j - 1) * (bw + 2) - bw / 2
            v = m[k]
            tip = f"{name}, {lab.split(' (')[0].lower()}: {v:.2f}%"
            b += (  # bar with rounded top corners
                f'  <path d="M{x:.1f},{B} V{Y(v) + 4:.1f} Q{x:.1f},{Y(v):.1f} {x + 4:.1f},{Y(v):.1f} '
                f'H{x + bw - 4:.1f} Q{x + bw:.1f},{Y(v):.1f} {x + bw:.1f},{Y(v) + 4:.1f} V{B} Z" '
                f'fill="{col}">'
                f"<title>{escape(tip)}</title></path>\n"
            )
        parts = _two_line_label(name)
        b += t(cx, B + 22, parts[0], 12.5, INK, "middle", "600")
        if len(parts) > 1:
            b += t(cx, B + 38, ("+ " if " + " in name else "") + parts[1], 12.5, INK, "middle", "600")
    b += f'  <line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}"/>\n'
    return svg(
        W,
        H,
        b,
        "Accuracy by BIRD difficulty",
        "Only 145 challenging questions, so each one there is worth 0.7 points. Hover a bar for the value.",
    )


def _cost_note(run: str, key: str) -> str:
    if key != "peak_vram_gb":
        return ""
    if run == "full":
        return f" +{info(run)['optimizer_state_gb']:.1f} in RAM"
    if run == "tinylora_grpo":
        return " incl. vLLM"
    return ""


def chart_cost() -> str:
    W, H = 1000, 400
    rows = [(r, n) for r, n in RUNS[1:] if r != "wsp_f4"]  # training runs only
    panels = [  # (title, run_info key, scale, axis max, panel x)
        ("Peak GPU memory (GB)", "peak_vram_gb", 1.0, 16, 40),
        ("Training time (minutes)", "wall_seconds", 1 / 60, 140, 520),
    ]
    b = ""
    for title, key, scale, vmax, x0 in panels:
        L, R, T = x0 + 150, x0 + 440, 112
        b += t(x0, 98, title, 14, INK, "start", "700")
        for i, (r, name) in enumerate(rows):
            y = T + i * 44
            v = info(r)[key] * scale
            w = v / vmax * (R - L)
            b += t(L - 10, y + 19, name, 13, INK, "end")
            b += (
                f'  <rect x="{L}" y="{y + 4}" width="{max(w, 2):.1f}" height="22" rx="4" fill="{BLUE}">'
                f"<title>{escape(f'{name}: {v:.1f}')}</title></rect>\n"
            )
            b += t(L + w + 8, y + 20, f"{v:.1f}{_cost_note(r, key)}", 12.5, INK, "start", "600")
    return svg(
        W,
        H,
        b,
        "What each run cost on one 16 GB laptop GPU",
        "Peak GPU memory and wall-clock training time. GRPO also runs a second model copy to sample answers.",
    )


def chart_grpo() -> str:
    with open(ROOT / "runs/tinylora_grpo/train_log.csv") as f:
        rows = [r for r in csv.DictReader(f) if r.get("reward")]
    steps = [int(r["step"]) for r in rows]
    rew = [100 * float(r["reward"]) for r in rows]
    k = 10
    avg = [sum(rew[i - k + 1 : i + 1]) / k for i in range(k - 1, len(rew))]
    W, H = 1000, 440
    L, R, T, B = 70, 940, 112, 370
    ymin, ymax = 0, 50
    X = lambda s: L + (s - 1) / (steps[-1] - 1) * (R - L)
    Y = lambda v: B - (v - ymin) / (ymax - ymin) * (B - T)
    b = y_grid(L, R, Y, range(0, ymax + 1, 10))
    for s in (1, 15, 30, 45, 60, 75):
        b += t(X(s), B + 22, str(s), 12, INK2, "middle")
    b += t((L + R) / 2, B + 46, "GRPO step (each step: 8 questions x 8 sampled answers)", 13, INK2, "middle")
    raw = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, rew, strict=True))
    b += f'  <polyline points="{raw}" fill="none" stroke="{MUTED}" stroke-width="1.5"/>\n'
    sm = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps[k - 1 :], avg, strict=True))
    b += f'  <polyline points="{sm}" fill="none" stroke="{BLUE}" stroke-width="3" stroke-linejoin="round"/>\n'
    for s, v in zip(steps, rew, strict=True):  # invisible hover targets
        b += (
            f'  <circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="7" fill="transparent">'
            f"<title>Step {s}: {v:.1f}% of answers correct</title></circle>\n"
        )
    b += t(X(steps[k - 1]) - 6, Y(avg[0]) + 22, f"{avg[0]:.1f}%", 13, INK, "end", "600")
    b += t(R + 8, Y(avg[-1]) + 4, f"{avg[-1]:.1f}%", 13, INK, "start", "600")
    b += f'  <line x1="28" y1="90" x2="52" y2="90" stroke="{MUTED}" stroke-width="1.5"/>\n'
    b += t(58, 94, "each step", 13, INK)
    b += f'  <line x1="148" y1="90" x2="172" y2="90" stroke="{BLUE}" stroke-width="3"/>\n'
    b += t(178, 94, "average of the last 10 steps", 13, INK)
    return svg(
        W,
        H,
        b,
        "TinyLoRA + GRPO: share of sampled answers that were correct",
        "Reward is 1 when the SQL returns the right rows. "
        "Each step draws new questions, so single steps are noisy.",
    )


def _method_matrix(kind: str, wx: float, wy: float, ws: float) -> str:
    """W plus the method's trainable pieces (B/A for LoRA variants, v for TinyLoRA)."""
    trainable_w = kind == "full"
    b = (
        f'  <rect x="{wx}" y="{wy}" width="{ws}" height="{ws}" rx="6" '
        f'fill="{BLUE if trainable_w else FROZEN}" '
        f'stroke="{BLUE if trainable_w else MUTED}" stroke-width="1.5"/>\n'
    )
    if kind == "qlora":  # 4-bit: a coarse grid suggests rounding to few levels
        for g in range(1, 4):
            gx, gy = wx + g * ws / 4, wy + g * ws / 4
            grid_line = f'stroke="{MUTED}" stroke-width="1"/>\n'
            b += f'  <line x1="{gx:.1f}" y1="{wy}" x2="{gx:.1f}" y2="{wy + ws}" ' + grid_line
            b += f'  <line x1="{wx}" y1="{gy:.1f}" x2="{wx + ws}" y2="{gy:.1f}" ' + grid_line
    w_label = "W" + (" 4-bit" if kind == "qlora" else "")
    b += t(wx + ws / 2, wy + ws / 2 + 5, w_label, 14, WHITE if trainable_w else INK, "middle", "700")
    if kind in ("lora", "lorafa", "qlora"):
        bx = wx + ws + 14
        b += f'  <rect x="{bx}" y="{wy}" width="14" height="{ws}" rx="3" fill="{BLUE}"/>\n'
        b += t(bx + 7, wy - 6, "B", 12, INK, "middle", "700")
        a_frozen = kind == "lorafa"
        b += (
            f'  <rect x="{wx}" y="{wy + ws + 12}" width="{ws}" height="14" rx="3" '
            f'fill="{FROZEN if a_frozen else BLUE}" stroke="{MUTED if a_frozen else BLUE}"/>\n'
        )
        b += t(wx + ws + 8, wy + ws + 24, "A" + (" (frozen)" if a_frozen else ""), 12, INK, "start", "700")
    if kind == "tiny":
        vx = wx + ws + 14
        for j in range(4):
            b += f'  <rect x="{vx}" y="{wy + j * 13}" width="12" height="11" rx="2" fill="{BLUE}"/>\n'
        b += t(vx + 6, wy + 66, "v", 13, INK, "middle", "700")
        b += t(wx, wy + ws + 24, "U, Σ, V, P frozen", 12, INK2)
    return b


def chart_methods() -> str:
    W, H = 1000, 400
    cols = [  # (name, formula, trainable count, kind)
        ("Full fine-tune", "W' = W + ΔW", "1.54B trainable", "full"),
        ("LoRA", "W' = W + B·A", "73.9M trainable", "lora"),
        ("LoRA-FA", "W' = W + B·A, A frozen", "41.3M trainable", "lorafa"),
        ("QLoRA", "W 4-bit + B·A", "73.9M trainable", "qlora"),
        ("TinyLoRA", "W + UΣ(Σ vᵢPᵢ)Vᵀ", "13 trainable", "tiny"),
    ]
    cw = 190
    b = ""
    for i, (name, form, count, kind) in enumerate(cols):
        x0 = 20 + i * cw
        b += (
            f'  <rect x="{x0 + 6}" y="86" width="{cw - 12}" height="290" rx="12" fill="{WHITE}" '
            f'stroke="{BORDER}" stroke-width="1.5"/>\n'
        )
        b += t(x0 + cw / 2, 114, name, 15, INK, "middle", "700")
        b += _method_matrix(kind, x0 + 30, 140, 90)
        b += t(x0 + cw / 2, 316, form, 12.5, INK, "middle", "600")
        b += t(x0 + cw / 2, 346, count, 13.5, BLUE, "middle", "700")
    b += legend_item(620, 52, BLUE, "trained")
    b += legend_item(720, 52, FROZEN, "frozen", MUTED)
    return svg(W, H, b, "The five methods for one weight matrix W", "Counts are for Qwen2.5-Coder-1.5B.")


if __name__ == "__main__":
    write_svgs(
        [
            ("methods", chart_methods),
            ("ex_vs_params", chart_params),
            ("ex_by_difficulty", chart_difficulty),
            ("training_cost", chart_cost),
            ("tinylora_grpo_reward", chart_grpo),
        ],
        "Result charts for the blog (inline SVG).",
    )
