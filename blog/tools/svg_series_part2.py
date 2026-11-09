"""Inline-SVG pictures for part 2 of the blog series ("How I tested them"), house style of svg_charts.py.

Numbers come from runs/*/run_info.json and runs/_sched/{status.json,*.log} (job start = log mtime - duration).

  .venv/bin/python blog/tools/svg_series_part2.py [--out DIR]   -> blog/svg/p2_*.svg
"""

from __future__ import annotations

import json
import os
from html import escape

from svg_charts import (
    AQUA,
    BLUE,
    BLUE_SOFT,
    BORDER,
    FROZEN,
    GRID,
    INK,
    INK2,
    MUTED,
    ORANGE,
    ROOT,
    WHITE,
    info,
    legend_item,
    svg,
    t,
    write_svgs,
)


def arrow_defs():
    return (
        f'  <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>\n'
    )


def line(points, dashed=False, color=MUTED, width=2, head=True):
    """Polyline through `points`, by default ending in the #ah arrowhead."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    mk = ' marker-end="url(#ah)"' if head else ""
    return (
        f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{dash}'
        f' stroke-linejoin="round"{mk}/>\n'
    )


def box(x, y, w, h, head, lines, accent=BLUE, fill=WHITE, head_fill=INK, dashed=False, size=12.5):
    """Card with a bold heading, grey body lines and an optional coloured accent bar on the left."""
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    b = (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{BORDER}" '
        f'stroke-width="1.5"{dash}/>\n'
    )
    if accent:
        b += f'  <rect x="{x}" y="{y + 10}" width="5" height="{h - 20}" rx="2.5" fill="{accent}"/>\n'
    b += t(x + 16, y + 24, head, 14, head_fill, "start", "700")
    for i, s in enumerate(lines):
        b += t(x + 16, y + 44 + i * 18, s, size, INK2)
    return b


def chart_pipeline():
    W, H = 1000, 452
    b = arrow_defs()
    bw, bh, gap = 206, 92, 40
    xs = [28 + i * (bw + gap) for i in range(4)]
    y1 = 96
    row1 = [
        ("1. BIRD question", ["question + evidence hint", "+ the database file"], MUTED),
        ("2. Build the prompt", ["schema + example values,", "fitted to 3,072 tokens"], MUTED),
        ("3. Model writes SQL", ["Qwen2.5-Coder-1.5B replies", "with a sql code block"], BLUE),
        ("4. extract_sql()", ["pull the query out", "of the model's reply"], MUTED),
    ]
    for x, (h, ls, ac) in zip(xs, row1, strict=True):
        b += box(x, y1, bw, bh, h, ls, ac)
    for i in range(3):
        b += line([(xs[i] + bw + 4, y1 + bh / 2), (xs[i + 1] - 4, y1 + bh / 2)])
    # elbow from box 4 down and back to the start of row 2
    y2 = 240
    b += line(
        [
            (xs[3] + bw / 2, y1 + bh + 2),
            (xs[3] + bw / 2, y1 + bh + 22),
            (xs[0] + bw / 2, y1 + bh + 22),
            (xs[0] + bw / 2, y2 - 4),
        ]
    )
    b += box(
        xs[0],
        y2,
        bw,
        bh,
        "5. Sandbox runs it",
        ["read-only SQLite, allow-list,", "30 s limit, own process"],
        ORANGE,
    )
    y3 = y2 + bh + 20
    b += box(xs[0], y3, bw, 76, "Gold SQL from BIRD", ["run in the same sandbox"], FROZEN, fill="#F8FAFC")
    # compare box sits between the two sandbox runs
    cy = y2 + 40
    b += box(
        xs[1],
        cy,
        bw,
        bh,
        "6. Compare as sets",
        ["model rows vs gold rows,", "order and repeats ignored"],
        ORANGE,
    )
    b += line([(xs[0] + bw + 4, y2 + bh / 2), (xs[1] - 4, cy + 30)])
    b += line([(xs[0] + bw + 4, y3 + 38), (xs[1] - 4, cy + bh - 22)])
    b += box(xs[2], cy, bw, bh, "7. Score", ["1 = same set of rows", "0 = wrong rows, error, timeout"], AQUA)
    b += line([(xs[1] + bw + 4, cy + bh / 2), (xs[2] - 4, cy + bh / 2)])
    b += box(
        xs[3],
        cy,
        bw,
        bh,
        "Used twice",
        ["dev test: 1,534 questions", "GRPO reward: 4,800 answers"],
        None,
        fill="#F8FAFC",
        dashed=True,
    )
    return svg(
        W,
        H,
        b,
        "One question, start to finish",
        "Every score in this series comes out of this path. Only step 3 changes between methods.",
    )


def chart_grpo_loop():
    W, H = 1000, 474
    b = arrow_defs()
    lanes = [
        ("vLLM", "fast sampler", 132),
        ("CPU", "12 SQL workers", 242),
        ("Trainer", "PyTorch, 13 numbers", 352),
    ]
    lh = 96
    for name, sub, y in lanes:
        b += f'  <rect x="20" y="{y}" width="960" height="{lh}" rx="10" fill="#F8FAFC" stroke="{BORDER}"/>\n'
        b += t(36, y + 44, name, 15, INK, "start", "700")
        b += t(36, y + 64, sub, 12.5, INK2)
    bw, bh = 150, 80
    xs = [182, 340, 498, 656, 814]
    ly = {n: y + (lh - bh) / 2 for n, _, y in lanes}
    steps = [
        ("vLLM", "1. Sample", ["8 questions x 8", "= 64 SQL answers"], BLUE),
        ("CPU", "2. Grade", ["run each SQL,", "reward 1 or 0"], AQUA),
        ("Trainer", "3. Advantages", ["(reward - mean)", "/ spread, per 8"], ORANGE),
        ("Trainer", "4. Update v", ["fwd + bwd, 16 x 4", "answers, 1 step"], BLUE),
        ("vLLM", "5. Sync weights", ["push W' into", "vLLM's 196 layers"], BLUE),
    ]
    for x, (lane, h, ls, ac) in zip(xs, steps, strict=True):
        b += box(x, ly[lane], bw, bh, h, ls, ac, size=12)
    # arrows between steps: same lane -> straight; other lane -> leave from the bottom/top centre
    for i in range(4):
        x0, x1 = xs[i] + bw, xs[i + 1]
        y0, y1 = ly[steps[i][0]] + bh / 2, ly[steps[i + 1][0]] + bh / 2
        cx = xs[i] + bw / 2
        if abs(y0 - y1) < 1:
            b += line([(x0 + 3, y0), (x1 - 4, y1)])
        elif y1 > y0:
            b += line([(cx, ly[steps[i][0]] + bh + 2), (cx, y1), (x1 - 4, y1)])
        else:
            b += line([(cx, ly[steps[i][0]] - 2), (cx, y1), (x1 - 4, y1)])
    # vLLM asleep band during steps 2-4 (ends before the step 4 -> 5 arrow)
    sx0, sx1 = xs[1] - 6, xs[3] + bw / 2 - 14
    b += (
        f'  <rect x="{sx0}" y="{ly["vLLM"]}" width="{sx1 - sx0}" height="{bh}" rx="10" fill="{WHITE}" '
        f'stroke="{MUTED}" stroke-dasharray="6 5"/>\n'
    )
    b += t((sx0 + sx1) / 2, ly["vLLM"] + 34, "vLLM asleep: weights and KV cache", 13, INK, "middle", "600")
    b += t((sx0 + sx1) / 2, ly["vLLM"] + 54, "moved off the GPU (frees about 7 GiB)", 12.5, INK2, "middle")
    # loop back arrow along the top
    yt = 110
    b += line(
        [
            (xs[4] + bw / 2, ly["vLLM"] - 2),
            (xs[4] + bw / 2, yt),
            (xs[0] + bw / 2, yt),
            (xs[0] + bw / 2, ly["vLLM"] - 4),
        ],
        dashed=True,
    )
    mid = (xs[0] + xs[4] + bw) / 2
    b += f'  <rect x="{mid - 160}" y="{yt - 12}" width="320" height="24" fill="#FCFCFB"/>\n'
    b += t(mid, yt + 5, "next step: about 100 s per loop, 75 loops", 13, INK, "middle", "600")
    return svg(
        W,
        H,
        b,
        "One GRPO step",
        "vLLM and the trainer take turns on the GPU. The SQL grading runs on the CPU.",
    )


def chart_gpu_budget():
    W, H = 1000, 470
    L, R, T = 210, 900, 122
    vmax = 18.0
    X = lambda v: L + v / vmax * (R - L)
    rows = [
        ("full", "Full fine-tune"),
        ("lora", "LoRA"),
        ("lora_fa", "LoRA-FA"),
        ("qlora", "QLoRA"),
        ("tinylora_sft", "TinyLoRA + SFT"),
        ("tinylora_grpo", "TinyLoRA + GRPO"),
        ("eval", "Evaluation (vLLM)"),
    ]
    b = ""
    lx = 28
    for col, lab, stroke in [
        (BLUE, "PyTorch peak", None),
        (BLUE_SOFT, "rest of the GPU in use", BLUE),
        (ORANGE, "reserved by vLLM", None),
    ]:
        b += legend_item(lx, 84, col, lab, stroke)
        lx += 40 + len(lab) * 7.2  # ~7.2 px per character at 13 px
    for v in range(0, 19, 2):
        b += f'  <line x1="{X(v):.1f}" y1="{T - 6}" x2="{X(v):.1f}" y2="{T + 7 * 40}" stroke="{GRID}"/>\n'
        b += t(X(v), T + 7 * 40 + 20, f"{v}", 12, INK2, "middle")
    b += t((L + R) / 2, T + 7 * 40 + 42, "GPU memory in GB (billions of bytes)", 13, INK2, "middle")
    card = 16 * 1024**3 / 1e9
    for i, (r, name) in enumerate(rows):
        y = T + i * 40
        b += t(L - 12, y + 19, name, 13, INK, "end")
        if r == "eval":
            v = 0.5 * card
            b += (
                f'  <rect x="{L}" y="{y + 4}" width="{X(v) - L:.1f}" height="22" rx="4" fill="{ORANGE}">'
                f"<title>vLLM reserves half the card: {v:.1f} GB</title></rect>\n"
            )
            b += t(
                X(v) + 8, y + 20, f"{v:.1f}  (half the card, set by gpu-mem 0.5)", 12.5, INK, "start", "600"
            )
            continue
        ri = info(r)
        peak = ri["peak_vram_gb"]
        dev = ri.get("device_used_max_gb")
        if dev:
            b += (
                f'  <rect x="{L}" y="{y + 4}" width="{X(dev) - L:.1f}" height="22" rx="4" fill="{BLUE_SOFT}" '
                f'stroke="{BLUE}"><title>{escape(name)}: whole GPU in use {dev:.1f} GB</title></rect>\n'
            )
        b += (
            f'  <rect x="{L}" y="{y + 4}" width="{X(peak) - L:.1f}" height="22" rx="4" fill="{BLUE}">'
            f"<title>{escape(name)}: PyTorch peak {peak:.1f} GB</title></rect>\n"
        )
        end = X(dev or peak)
        note = f"{peak:.1f}" + (f" / {dev:.1f}" if dev else "")
        if r == "full":
            note += "  (+3.1 GB optimizer state in CPU RAM)"
        if r == "tinylora_grpo":
            note += "  (trainer + sleeping vLLM)"
        b += t(end + 8, y + 20, note, 12.5, INK, "start", "600")
    b += (
        f'  <line x1="{X(card):.1f}" y1="{T - 14}" x2="{X(card):.1f}" y2="{T + 7 * 40}" stroke="{INK}" '
        f'stroke-width="2" stroke-dasharray="6 5"/>\n'
    )
    b += t(X(card) - 6, T - 20, "card limit: 16 GiB = 17.2 GB", 12.5, INK, "end", "700")
    return svg(
        W,
        H,
        b,
        "How much GPU memory each job used",
        "Any single job fits easily. The trouble starts when two of them share the card.",
    )


def job_times():
    st = json.loads((ROOT / "runs/_sched/status.json").read_text())
    out = {}
    for n, v in st.items():
        log = ROOT / f"runs/_sched/{n}.log"
        if log.exists():
            end = os.path.getmtime(log)
            out[n] = (end - v["secs"], end)
    t0 = min(s for s, _ in out.values())
    return {n: ((s - t0) / 60, (e - t0) / 60) for n, (s, e) in out.items()}


def chart_schedule():
    J = job_times()
    W, H = 1000, 480
    L, R, T = 200, 950, 122
    tmax = 360
    X = lambda m: L + m / tmax * (R - L)
    rows = [
        ("TinyLoRA LR test", ["sweep_1e-3", "sweep_1e-2", "sweep_5e-2"], [], "3 x 40 steps"),
        ("TinyLoRA + GRPO", ["grpo_tinylora"], ["merge_tinylora_grpo", "eval_tinylora_grpo"], None),
        ("LoRA", ["sft_lora"], ["merge_lora", "eval_lora"], None),
        ("QLoRA", ["sft_qlora"], ["merge_qlora", "eval_qlora"], None),
        ("LoRA-FA", ["sft_lora_fa"], ["merge_lora_fa", "eval_lora_fa"], None),
        ("Full fine-tune", ["sft_full"], ["merge_full", "eval_full"], None),
        ("TinyLoRA + SFT", ["sft_tinylora"], ["merge_tinylora_sft", "eval_tinylora_sft"], None),
        ("QLoRA, 4-bit eval", [], ["eval_qlora_4bit"], None),
    ]
    rh = 36
    total = max(e for _, e in J.values())
    b = ""
    b += legend_item(28, 84, BLUE, "training")
    b += legend_item(148, 84, ORANGE, "merge + evaluate on 1,534 questions")
    bottom = T + len(rows) * rh
    for h in range(0, 7):
        hx = X(h * 60)
        b += f'  <line x1="{hx:.1f}" y1="{T - 6}" x2="{hx:.1f}" y2="{bottom}" stroke="{GRID}"/>\n'
        b += t(X(h * 60), bottom + 20, f"{h} h", 12, INK2, "middle")
    for i, (name, train, post, label) in enumerate(rows):
        y = T + i * rh
        b += t(L - 12, y + 21, name, 13, INK, "end")
        for j in train:
            s, e = J[j]
            b += (
                f'  <rect x="{X(s):.1f}" y="{y + 6}" width="{max(X(e) - X(s) - 1, 2):.1f}" '
                f'height="22" rx="4" '
                f'fill="{BLUE}"><title>{j}: {e - s:.1f} min</title></rect>\n'
            )
        if post:
            s, e = min(J[p][0] for p in post), max(J[p][1] for p in post)
            b += (
                f'  <rect x="{X(s):.1f}" y="{y + 6}" width="{max(X(e) - X(s), 3):.1f}" height="22" rx="3" '
                f'fill="{ORANGE}"><title>{", ".join(post)}: {e - s:.1f} min</title></rect>\n'
            )
        all_jobs = train + post
        end = max(J[j][1] for j in all_jobs)
        tr = sum(J[j][1] - J[j][0] for j in train)
        pt = sum(J[p][1] - J[p][0] for p in post)
        if label:
            txt = f"{label}, {tr:.0f} min"
        elif train:
            txt = f"{tr:.0f} min + {pt:.0f} min"
        else:
            txt = f"{pt:.0f} min"
        right_edge = X(end) + 8 + len(txt) * 7
        # put the label left of the bar if it would leave the card or run into the "all done" line
        if right_edge > W - 20 or (right_edge > X(total) - 4 and end < total - 1):
            start = min(J[j][0] for j in all_jobs)
            b += t(X(start) - 8, y + 22, txt, 12.5, INK, "end", "600")
        else:
            b += t(X(end) + 8, y + 22, txt, 12.5, INK, "start", "600")
    b += (
        f'  <line x1="{X(total):.1f}" y1="{T - 14}" x2="{X(total):.1f}" y2="{bottom}" stroke="{INK}" '
        f'stroke-width="2" stroke-dasharray="6 5"/>\n'
    )
    hh, mm = divmod(round(total), 60)
    b += t(X(total) - 6, T - 20, f"all done: {hh} h {mm} min", 12.5, INK, "end", "700")
    return svg(
        W,
        H,
        b,
        "The whole experiment, one job at a time",
        "Real start and end times from the scheduler, 01:21 to 07:11, nobody at the keyboard.",
    )


if __name__ == "__main__":
    write_svgs(
        [
            ("p2_pipeline", chart_pipeline),
            ("p2_grpo_loop", chart_grpo_loop),
            ("p2_gpu_budget", chart_gpu_budget),
            ("p2_schedule", chart_schedule),
        ],
        "Part 2 pictures (inline SVG).",
    )
