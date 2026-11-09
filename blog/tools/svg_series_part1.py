"""Explanatory inline-SVG pictures for part 1 of the blog series ("Fine-tuning, explained").

House style from svg_charts.py: blue = trained, light grey with slate stroke = frozen, orange / aqua as
2nd / 3rd colour.

  .venv/bin/python blog/tools/svg_series_part1.py [--out DIR]   -> blog/svg/p1_*.svg
"""

from __future__ import annotations

import math

from svg_charts import (
    AQUA,
    BLUE,
    BLUE_SOFT,
    BORDER,
    FROZEN,
    INK,
    INK2,
    MUTED,
    ORANGE,
    WHITE,
    svg,
    t,
    write_svgs,
)

AQUA_SOFT = "#E6F6EF"


def rect(x, y, w, h, fill, stroke="none", rx=4, sw=1.5, extra=""):
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{extra}/>\n'
    )


def line(x1, y1, x2, y2, stroke=MUTED, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{sw}"{d}/>\n'
    )


def arrow(x1, y1, x2, y2, color=MUTED, sw=2.0, head=8):
    """Straight line with a filled triangular head at (x2, y2)."""
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    px, py = -math.sin(ang) * head * 0.55, math.cos(ang) * head * 0.55
    return (
        line(x1, y1, bx, by, color, sw)
        + f'  <path d="M{x2:.1f},{y2:.1f} L{bx + px:.1f},{by + py:.1f} L{bx - px:.1f},{by - py:.1f} Z" '
        f'fill="{color}"/>\n'
    )


def fmt(v):
    """Number as shown in the pictures: shortest form, typographic minus."""
    if isinstance(v, str):
        return v
    return f"{v:g}".replace("-", "−")


def grid(x, y, vals, cell, fill, stroke, tfill=INK, size=14, weight="600", fills=None):
    """Matrix of numbers. vals: list of rows. fills: optional per-cell fill override."""
    b = ""
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            f = fills[i][j] if fills else fill
            b += rect(x + j * cell, y + i * cell, cell, cell, f, stroke, rx=0, sw=1.2)
            cx, cy = x + j * cell + cell / 2, y + i * cell + cell / 2 + size * 0.36  # 0.36: optical centring
            b += t(cx, cy, fmt(v), size, tfill, "middle", weight)
    b += rect(x, y, cell * len(vals[0]), cell * len(vals), "none", stroke, rx=2, sw=2)
    return b


def lines(x, y, rows, gap=22):
    """Left-aligned block of (text, size, fill, weight) rows."""
    b = ""
    for k, (s, size, fill, weight) in enumerate(rows):
        b += t(x, y + k * gap, s, size, fill, "start", weight)
    return b


def p1_matrix_times_vector():
    W, H = 1000, 330
    c = 66
    b = ""
    wx, wy = 60, 104
    fills = [[BLUE_SOFT, BLUE_SOFT], [WHITE, WHITE]]
    b += grid(wx, wy, [[2, 1], [0, 3]], c, WHITE, MUTED, fills=fills, size=20)
    b += t(wx + c, wy + 2 * c + 28, "W (the weights)", 14, INK, "middle", "700")
    b += t(wx + 2 * c + 32, wy + c + 8, "×", 26, INK2, "middle")
    xx = wx + 2 * c + 64
    b += grid(xx, wy, [[1], [2]], c, AQUA_SOFT, MUTED, size=20)
    b += t(xx + c / 2, wy + 2 * c + 28, "x (input)", 14, INK, "middle", "700")
    b += t(xx + c + 32, wy + c + 8, "=", 26, INK2, "middle")
    yx = xx + c + 64
    b += grid(yx, wy, [[4], [6]], c, WHITE, MUTED, fills=[[BLUE_SOFT], [WHITE]], size=20)
    b += t(yx + c / 2, wy + 2 * c + 28, "y (output)", 14, INK, "middle", "700")

    px = 560
    b += t(px, 118, "How each output number is made", 15, INK, "start", "700")
    b += t(px, 150, "Row 1 of W times x:  2×1 + 1×2 = 4", 15, INK, "start", "600")
    b += t(px, 178, "Row 2 of W times x:  0×1 + 3×2 = 6", 15, INK, "start", "400")
    b += t(px, 220, "Training only ever changes the numbers in W.", 14, INK2)
    b += t(px, 244, "One real layer of our model is a 1536 × 1536", 14, INK2)
    b += t(px, 266, "grid: 2,359,296 numbers. The model has 1.54 billion.", 14, INK2)
    return svg(
        W,
        H,
        b,
        "A layer is a grid of numbers times an input",
        "Toy 2×2 example. The highlighted row of W produces the highlighted output.",
    )


def p1_gradient_step():
    W, H = 1000, 420
    L, R, T, B = 80, 540, 100, 350
    wmin, wmax, lmax = 0.0, 1.6, 4.0
    X = lambda w: L + (w - wmin) / (wmax - wmin) * (R - L)
    Y = lambda v: B - v / lmax * (B - T)
    loss = lambda w: (2 * w - 2) ** 2
    b = ""
    for v in range(0, 5):
        b += line(L, Y(v), R, Y(v), "#E8ECF2", 1)
        b += t(L - 10, Y(v) + 4, str(v), 12, INK2, "end")
    for w in (0, 0.5, 1.0, 1.5):
        b += t(X(w), B + 20, f"{w:g}", 12, INK2, "middle")
    b += t((L + R) / 2, B + 44, "weight w", 13, INK2, "middle")
    b += t(L - 44, T - 14, "loss", 13, INK2, "start")
    pts = " ".join(f"{X(w):.1f},{Y(loss(w)):.1f}" for w in [i / 100 for i in range(0, 161)])
    b += f'  <polyline points="{pts}" fill="none" stroke="{MUTED}" stroke-width="2.5"/>\n'
    # tangent at w = 0.5: slope -4
    w0 = 0.5
    tl = lambda w: loss(w0) - 4 * (w - w0)
    b += line(X(0.25), Y(tl(0.25)), X(0.75), Y(tl(0.75)), ORANGE, 2.5)
    b += t(X(0.25) - 6, Y(tl(0.25)) - 6, "slope −4", 13, ORANGE, "end", "700")
    b += arrow(X(0.5) + 8, Y(loss(0.5)) + 6, X(0.9) - 8, Y(loss(0.9)) - 6, BLUE, 2)
    for w in (0.5, 0.9, 0.98):
        b += (
            f'  <circle cx="{X(w):.1f}" cy="{Y(loss(w)):.1f}" r="7" fill="{BLUE}" '
            f'stroke="{WHITE}" stroke-width="2"/>\n'
        )
    b += t(X(0.5) - 10, Y(loss(0.5)) + 22, "start", 13, INK, "end", "600")
    b += t(X(0.9) - 4, Y(loss(0.9)) - 14, "1", 13, INK, "middle", "700")
    b += t(X(0.98) + 6, Y(loss(0.98)) - 14, "2", 13, INK, "middle", "700")

    px = 590
    b += t(px, 112, "Model: prediction = w × x", 14, INK, "start", "700")
    b += t(px, 136, "x = 2, target = 2, learning rate = 0.1", 14, INK2)
    cols = [0, 60, 130, 230]
    table = [
        ("", "w", "prediction", "loss"),
        ("start", "0.50", "1.00", "1.00"),
        ("step 1", "0.90", "1.80", "0.04"),
        ("step 2", "0.98", "1.96", "0.0016"),
    ]
    for i, row in enumerate(table):
        for dx, cell in zip(cols, row, strict=True):
            b += t(
                px + dx, 178 + i * 24, cell, 14, INK2 if i == 0 else INK, "start", "400" if i == 0 else "600"
            )
    b += line(px, 184, px + 290, 184, BORDER, 1.5)
    rows = [
        ("The slope (the gradient) says which way", 14, INK2, "400"),
        ("is downhill. Each step moves w a little", 14, INK2, "400"),
        ("that way: w = w − 0.1 × slope.", 14, INK2, "400"),
    ]
    b += lines(px, 290, rows, 24)
    return svg(
        W,
        H,
        b,
        "Training is walking downhill on the loss",
        "Loss = (prediction − target)². One weight, two steps of gradient descent, all by hand.",
    )


def p1_memory():
    W, H = 1000, 360
    L, R = 250, 900
    gmax = 20.0
    X = lambda g: L + g / gmax * (R - L)
    b = ""
    leg = [("weights (bf16)", BLUE), ("gradients (bf16)", ORANGE), ("Adam's 2 running averages", AQUA)]
    lx = 28
    for lab, col in leg:
        b += rect(lx, 84, 14, 14, col, rx=3)
        b += t(lx + 20, 96, lab, 13, INK)
        lx += 40 + len(lab) * 7.2
    bars = [
        (
            "Plain Adam",
            "12 bytes per parameter",
            [(3.1, BLUE, "3.1"), (3.1, ORANGE, "3.1"), (12.3, AQUA, "12.3 (fp32)")],
            "18.5 GB",
        ),
        (
            "My full fine-tune",
            "8-bit Adam, paged",
            [(3.1, BLUE, "3.1"), (3.1, ORANGE, "3.1"), (3.1, AQUA, "3.1 (8-bit)")],
            "6.2 GB on GPU",
        ),
    ]
    for i, (name, sub, segs, total) in enumerate(bars):
        y = 140 + i * 82
        b += t(L - 14, y + 18, name, 14, INK, "end", "700")
        b += t(L - 14, y + 37, sub, 12.5, INK2, "end")
        x = L
        for k, (g, col, lab) in enumerate(segs):
            w = X(g) - L
            extra = ' fill-opacity="0.45"' if (i == 1 and k == 2) else ""
            b += rect(x, y, w - 2, 40, col, rx=4, extra=extra)
            b += t(x + w / 2 - 1, y + 25, lab, 13, WHITE if not extra else INK, "middle", "700")
            x += w
        b += t(x + 10, y + 25, total, 14, INK, "start", "700")
    b += line(X(16), 120, X(16), 312, INK, 2, "6 5")
    b += t(X(16) - 6, 334, "16 GB GPU", 13, INK, "end", "600")
    b += t(X(6.2) + 4, 290, "8-bit Adam state lives in CPU RAM (paged)", 12.5, INK2, "start")
    b += t(28, 334, "Counts for 1.54B parameters. Activations not shown.", 12.5, INK2)
    return svg(
        W,
        H,
        b,
        "Memory for full fine-tuning, before any data goes in",
        "Every trainable number needs its own gradient and optimizer state, not just itself.",
    )


def p1_lora_patch():
    W, H = 1000, 380
    c = 42
    y0 = 108
    Wv = [[0.2, -0.1, 0.4, 0.0], [0.3, 0.5, -0.2, 0.1], [-0.4, 0.2, 0.1, 0.3], [0.0, -0.3, 0.2, 0.6]]
    Bv = [[1], [2], [0], [-1]]
    Av = [[0.5, 0, 0, 0.5]]
    BA = [[b[0] * a for a in Av[0]] for b in Bv]
    b = ""
    x = 36
    b += grid(x, y0, Wv, c, FROZEN, MUTED, size=13)
    b += t(x + 2 * c, y0 + 4 * c + 26, "W, frozen", 14, INK, "middle", "700")
    b += t(x + 2 * c, y0 + 4 * c + 46, "16 numbers", 12.5, INK2, "middle")
    x += 4 * c + 22
    b += t(x, y0 + 2 * c + 9, "+", 26, INK2, "middle")
    x += 20
    b += grid(x, y0, Bv, c, BLUE, BLUE, WHITE, size=14)
    b += t(x + c / 2, y0 + 4 * c + 26, "B", 14, INK, "middle", "700")
    b += t(x + c / 2, y0 + 4 * c + 46, "4, trained", 12.5, INK2, "middle")
    x += c + 22
    b += t(x, y0 + 2 * c + 9, "×", 24, INK2, "middle")
    x += 20
    b += grid(x, y0, Av, c, BLUE, BLUE, WHITE, size=14)
    b += t(x + 2 * c, y0 + c + 24, "A", 14, INK, "middle", "700")
    b += t(x + 2 * c, y0 + c + 44, "4, trained", 12.5, INK2, "middle")
    x += 4 * c + 22
    b += t(x, y0 + 2 * c + 9, "=", 26, INK2, "middle")
    x += 20
    b += grid(x, y0, BA, c, BLUE_SOFT, BLUE, size=13)
    b += t(x + 2 * c, y0 + 4 * c + 26, "B·A, the patch", 14, INK, "middle", "700")
    b += t(x + 2 * c, y0 + 4 * c + 46, "16 changes, computed", 12.5, INK2, "middle")

    px = x + 4 * c + 34
    rows = [
        ("W' = W + B·A", 15, INK, "700"),
        ("", 13, INK, "400"),
        ("8 trained numbers move", 13.5, INK2, "400"),
        ("all 16 numbers of W.", 13.5, INK2, "400"),
        ("", 13, INK, "400"),
        ("Real layer, rank 64:", 13.5, INK, "700"),
        ("W has 2,359,296 numbers;", 13.5, INK2, "400"),
        ("B and A have 196,608.", 13.5, INK2, "400"),
    ]
    b += lines(px, 124, rows, 22)
    return svg(
        W,
        H,
        b,
        "LoRA: a thin patch added to a frozen matrix",
        "Rank 1 toy: a 4×1 column B times a 1×4 row A makes a full 4×4 grid.",
    )


NF4 = [
    -1.0,
    -0.6962,
    -0.5251,
    -0.3949,
    -0.2844,
    -0.1848,
    -0.0911,
    0.0,
    0.0796,
    0.1609,
    0.2461,
    0.3379,
    0.4407,
    0.5626,
    0.7230,
    1.0,
]


def p1_quant_levels():
    W, H = 1000, 360
    L, R = 80, 920
    X = lambda v: L + (v + 1) / 2 * (R - L)
    b = ""
    # row 1: even levels
    y1 = 190
    b += t(28, 106, "16 evenly spaced levels between −1 and 1", 14, INK, "start", "700")
    b += line(L, y1, R, y1, MUTED, 2)
    levels = [-1 + 2 * k / 15 for k in range(16)]
    for v in levels:
        b += (
            f'  <circle cx="{X(v):.1f}" cy="{y1}" r="5" fill="{WHITE}" stroke="{INK2}" stroke-width="1.5"/>\n'
        )
    for v, lab in [(-1, "−1"), (1, "1")]:
        b += t(X(v), y1 + 26, lab, 12.5, INK2, "middle")
    ex = [(0.137, 0.2), (-0.402, -1 + 2 * 4 / 15), (0.951, 1.0)]
    for orig, snap in ex:
        ox = X(orig)
        b += f'  <circle cx="{ox:.1f}" cy="{y1 - 52}" r="6" fill="{ORANGE}"/>\n'
        b += t(ox, y1 - 64, fmt(orig), 13, ORANGE, "middle", "700")
        b += arrow(ox, y1 - 45, X(snap), y1 - 9, INK2, 1.5, 7)
        b += f'  <circle cx="{X(snap):.1f}" cy="{y1}" r="6" fill="{BLUE}"/>\n'
        b += t(X(snap), y1 + 26, fmt(round(snap, 3)), 13, BLUE, "middle", "700")
    # row 2: NF4
    y2 = 300
    b += t(
        28,
        260,
        "NF4: the same 16 levels, packed closer together near 0 where most weights are",
        14,
        INK,
        "start",
        "700",
    )
    b += line(L, y2, R, y2, MUTED, 2)
    for v in NF4:
        b += f'  <circle cx="{X(v):.1f}" cy="{y2}" r="5" fill="{BLUE}" stroke="{WHITE}" stroke-width="1"/>\n'
    for v, lab in [(-1, "−1"), (0, "0"), (1, "1")]:
        b += t(X(v), y2 + 26, lab, 12.5, INK2, "middle")
    b += f'  <circle cx="{R - 250}" cy="102" r="6" fill="{ORANGE}"/>\n' + t(
        R - 240, 106, "original weight", 12.5, INK2
    )
    b += f'  <circle cx="{R - 120}" cy="102" r="6" fill="{BLUE}"/>\n' + t(
        R - 110, 106, "stored level", 12.5, INK2
    )
    return svg(
        W,
        H,
        b,
        "4-bit numbers: round every weight to one of 16 levels",
        "4 bits can hold 2⁴ = 16 different values. Each weight snaps to its nearest level.",
    )


def p1_tinylora():
    W, H = 1000, 420
    b = ""
    top = 112
    ws = 150
    b += rect(36, top, ws, ws, FROZEN, MUTED, rx=6)
    b += t(36 + ws / 2, top + ws / 2 + 6, "W", 18, INK, "middle", "700")
    b += t(36 + ws / 2, top + ws + 26, "W, frozen", 14, INK, "middle", "700")
    x = 36 + ws + 24
    b += t(x, top + ws / 2 + 9, "+", 26, INK2, "middle")
    # U Sigma: d x 2
    x += 22
    cw = 22
    b += rect(x, top, cw, ws, FROZEN, MUTED, rx=3)
    b += rect(x + cw, top, cw, ws, FROZEN, MUTED, rx=3)
    b += t(x + cw, top + ws + 26, "UΣ", 14, INK, "middle", "700")
    b += t(x + cw, top + ws + 44, "frozen", 12.5, INK2, "middle")
    # R 2x2
    rx_ = x + 2 * cw + 14
    for i in range(2):
        for j in range(2):
            b += rect(rx_ + j * cw, top + i * cw, cw, cw, BLUE_SOFT, BLUE, rx=2, sw=1.5)
    b += t(rx_ + cw, top - 10, "R", 14, INK, "middle", "700")
    # V^T: 2 x d
    vx = rx_ + 2 * cw + 14
    b += rect(vx, top, ws, cw, FROZEN, MUTED, rx=3)
    b += rect(vx, top + cw, ws, cw, FROZEN, MUTED, rx=3)
    b += t(vx + ws / 2, top + 2 * cw + 22, "Vᵀ, frozen", 14, INK, "middle", "700")
    b += t(vx + ws / 2, top - 10, "the 2 strongest directions of W", 12.5, INK2, "middle")
    # v vector, 13 cells, and arrow to R
    vy = 330
    n, cs, gap = 13, 22, 3
    vw = n * cs + (n - 1) * gap
    v0 = rx_ + cw - vw / 2 + 40
    for k in range(n):
        b += rect(v0 + k * (cs + gap), vy, cs, cs, BLUE, rx=3)
    b += t(
        v0 + vw / 2,
        vy + cs + 24,
        "v: 13 trained numbers, one copy shared by all 196 matrices",
        13.5,
        INK,
        "middle",
        "700",
    )
    b += arrow(v0 + vw / 2 - 40, vy - 6, rx_ + cw, top + 2 * cw + 8, BLUE, 2)
    b += t(rx_ + cw + 14, 266, "R = v₁P₁ + v₂P₂ + … + v₁₃P₁₃", 13.5, INK, "start", "600")
    b += t(rx_ + cw + 14, 286, "P₁ … P₁₃: fixed random 2×2 grids", 12.5, INK2, "start")

    px = vx + ws + 40
    rows = [
        ("W' = W + UΣ R Vᵀ", 15, INK, "700"),
        ("", 13, INK, "400"),
        ("SVD finds the strongest", 13.5, INK2, "400"),
        ("directions of W (frozen).", 13.5, INK2, "400"),
        ("R, a 2×2 mixer, re-weights", 13.5, INK2, "400"),
        ("how they combine.", 13.5, INK2, "400"),
        ("", 13, INK, "400"),
        ("v starts at 0, so R = 0", 13.5, INK2, "400"),
        ("and W' = W at step 0.", 13.5, INK2, "400"),
    ]
    b += lines(px, 124, rows, 22)
    return svg(
        W,
        H,
        b,
        "TinyLoRA: steer the strongest directions with 13 numbers",
        "Blue is trained or computed from trained numbers. Everything grey is frozen.",
    )


def p1_sft_mask():
    W, H = 1000, 280
    b = ""
    prompt = ["How", " many", " students", "?"]
    answer = ["SELECT", " COUNT", "(*)", " FROM", " students", "<end>"]
    x, y, h = 36, 118, 44
    spans = []
    for k, tok in enumerate(prompt + answer):
        is_ans = k >= len(prompt)
        w = 26 + 9.2 * len(tok.strip())
        b += rect(x, y, w, h, BLUE if is_ans else FROZEN, BLUE if is_ans else MUTED, rx=6)
        b += t(x + w / 2, y + 28, tok.strip(), 14, WHITE if is_ans else INK, "middle", "700")
        spans.append((x, x + w, is_ans))
        x += w + 8
    p0, p1 = spans[0][0], spans[len(prompt) - 1][1]
    a0, a1 = spans[len(prompt)][0], spans[-1][1]
    by = y + h + 18
    for s0, s1, col in [(p0, p1, MUTED), (a0, a1, BLUE)]:
        b += line(s0, by, s1, by, col, 2)
        b += line(s0, by - 6, s0, by, col, 2) + line(s1, by - 6, s1, by, col, 2)
    b += t((p0 + p1) / 2, by + 24, "prompt: label −100, ignored", 13.5, INK, "middle", "700")
    b += t((p0 + p1) / 2, by + 44, "the model reads it, no loss", 12.5, INK2, "middle")
    b += t((a0 + a1) / 2, by + 24, "answer: loss counted on every token", 13.5, INK, "middle", "700")
    b += t((a0 + a1) / 2, by + 44, '"after FROM, write students"', 12.5, INK2, "middle")
    return svg(
        W,
        H,
        b,
        "SFT: learn from the answer key, one token at a time",
        "Grey tokens are context. Blue tokens are what the model is graded on.",
    )


def p1_grpo_group():
    W, H = 1000, 430
    b = ""
    base, unit = 300, 120  # reward 1 = 120 px
    Y = lambda r: base - r * unit
    cols = [(120, 1, "correct"), (250, 0, "wrong rows"), (380, 0, "SQL error"), (510, 1, "correct")]
    mean = 0.5
    L, R = 60, 580
    b += line(L, base, R, base, MUTED, 1.5)
    b += line(L, Y(mean), R, Y(mean), INK, 2, "6 5")
    b += t(R + 8, Y(mean) + 4, "group mean 0.5", 13, INK, "start", "700")
    b += t(L - 8, Y(1) + 4, "1", 12, INK2, "end") + t(L - 8, base + 4, "0", 12, INK2, "end")
    b += line(L, Y(1), R, Y(1), "#E8ECF2", 1)
    for k, (cx, r, lab) in enumerate(cols):
        b += t(cx, 102, f"Answer {k + 1}", 14, INK, "middle", "700")
        b += t(cx, 122, lab, 12.5, AQUA if r else ORANGE, "middle", "700")
        bw = 44
        if r:
            b += rect(cx - bw / 2 - 16, Y(r), bw, r * unit, BLUE_SOFT, BLUE, rx=3)
        else:
            b += line(cx - bw / 2 - 16, base, cx + bw / 2 - 16, base, BLUE, 3)
        adv = (r - mean) / 0.5
        col = AQUA if adv > 0 else ORANGE
        ax = cx + 26
        b += arrow(ax, Y(mean), ax, Y(r) + (2 if adv > 0 else -2), col, 3.5, 12)
        b += t(cx, base + 24, f"reward {r}", 13, INK, "middle", "600")
        b += t(cx, base + 46, f"advantage {'+1' if adv > 0 else '−1'}", 13.5, col, "middle", "700")
        b += t(cx, base + 66, "push up" if adv > 0 else "push down", 12.5, INK2, "middle")
    px = 730
    rows = [
        ("mean = (1+0+0+1) / 4 = 0.5", 13.5, INK, "600"),
        ("spread (std) = 0.5", 13.5, INK, "600"),
        ("advantage = (reward − mean) / std", 13.5, INK, "600"),
        ("", 13, INK, "400"),
        ("If all 4 rewards are equal,", 13.5, INK2, "400"),
        ("every advantage is 0 and", 13.5, INK2, "400"),
        ("this question teaches nothing.", 13.5, INK2, "400"),
    ]
    b += lines(px, 150, rows, 24)
    return svg(
        W,
        H,
        b,
        "GRPO: grade a group of answers, then compare each to the group",
        "One question, 4 sampled answers. Arrows show the advantage: distance from the group mean.",
    )


if __name__ == "__main__":
    charts = [
        p1_matrix_times_vector,
        p1_gradient_step,
        p1_memory,
        p1_lora_patch,
        p1_quant_levels,
        p1_tinylora,
        p1_sft_mask,
        p1_grpo_group,
    ]
    write_svgs([(fn.__name__, fn) for fn in charts], "Part 1 explanatory pictures (inline SVG).")
