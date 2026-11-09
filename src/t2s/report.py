"""Collect results/<run>/metrics.json + runs/<run>/run_info.json into results/summary.{md,csv} and charts.

    python -m t2s.report [--results results] [--runs runs] [--out results/figures]

Charts are drawn for a dark page (the portfolio blog); palette validated with the dataviz validator
(slots 1-3 dark, all-pairs, on #0b0a1f).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = [  # (run name, display label)
    ("base", "Base (no tuning)"),
    ("full", "Full fine-tune"),
    ("lora", "LoRA"),
    ("lora_fa", "LoRA-FA"),
    ("qlora", "QLoRA"),
    ("qlora_4bit", "QLoRA (4-bit serve)"),
    ("tinylora_sft", "TinyLoRA + SFT"),
    ("tinylora_grpo", "TinyLoRA + GRPO"),
    ("wsp_f4", "Weight-space programming (Opus 5.5)"),
]
BASE_PARAMS = 1_543_714_304
WSP_CHANGED = 48_398  # weights changed by weight-space programming; it has no training run
SURFACE = "#0b0a1f"
TEXT = "#f0efec"
TEXT2 = "#c3c2b7"
MUTED = "#6f6e69"
GRID = "#2a2940"
S1, S2, S3 = "#3987e5", "#d95926", "#199e70"  # simple / moderate / challenging
SMOOTH_K = 10  # GRPO reward moving-average window


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": TEXT,
            "axes.labelcolor": TEXT2,
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )


def load(results: Path, runs: Path) -> list[dict]:
    """One row per run in RUNS that has a metrics.json."""
    rows = []
    for name, label in RUNS:
        m = results / name / "metrics.json"
        if not m.exists():
            continue
        met = json.loads(m.read_text())
        info_path = runs / name.replace("qlora_4bit", "qlora") / "run_info.json"  # same trained adapter
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        if name == "wsp_f4":
            info = {"trainable_params": WSP_CHANGED, "pct_of_base": 100 * WSP_CHANGED / BASE_PARAMS}
        rows.append(
            {
                "run": name,
                "label": label,
                "ex": met["ex"],  # metrics.json stores EX already in percent
                "ex_simple": met["ex_simple"],
                "ex_moderate": met["ex_moderate"],
                "ex_challenging": met["ex_challenging"],
                "trainable_params": info.get("trainable_params", 0),
                "pct": info.get("pct_of_base", info.get("pct", 0.0)),
                "peak_vram_gb": info.get("peak_vram_gb"),
                "train_hours": info["wall_seconds"] / 3600 if "wall_seconds" in info else None,
                "optimizer_state_gb": info.get("optimizer_state_gb"),
            }
        )
    return rows


def fmt_pct(pct: float) -> str:
    if pct >= 1:
        return f"{pct:.1f}%"
    if pct >= 0.001:
        return f"{pct:.3f}%"
    return "<0.001%" if pct > 0 else "0%"


def fmt_params(n: int) -> str:
    if n == 0:
        return "0"
    for unit, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= d:
            return f"{n / d:.2f}{unit}"
    return str(n)


def write_tables(rows: list[dict], out: Path) -> None:
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    base = next((r["ex"] for r in rows if r["run"] == "base"), None)
    lines = [
        "| Method | Trainable params | % of model | EX | Δ vs base | Simple | Moderate | Challenging "
        "| Peak VRAM | Train time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = f"{r['ex'] - base:+.1f}" if base is not None and r["run"] != "base" else "-"
        vram = f"{r['peak_vram_gb']:.1f} GB" if r["peak_vram_gb"] else "-"
        hours = f"{r['train_hours']:.1f} h" if r["train_hours"] else "-"
        params = fmt_params(r["trainable_params"])
        lines.append(
            f"| {r['label']} | {params} | {fmt_pct(r['pct'])} | {r['ex']:.1f}% | {delta} | "
            f"{r['ex_simple']:.1f} | {r['ex_moderate']:.1f} | {r['ex_challenging']:.1f} | {vram} | {hours} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def chart_params(rows: list[dict], figs: Path) -> None:
    base = next((r for r in rows if r["run"] == "base"), None)
    pts = [r for r in rows if r["run"] not in ("base", "qlora_4bit") and r["trainable_params"]]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.scatter(
        [r["trainable_params"] for r in pts],
        [r["ex"] for r in pts],
        s=90,
        color=S1,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    tiny = [r for r in pts if r["run"].startswith("tinylora")]
    labelled = [r for r in pts if not r["run"].startswith("tinylora")]
    if len({round(r["ex"], 2) for r in tiny}) == 1:  # both TinyLoRA runs on one point: one label
        labelled.append({**tiny[0], "label": "TinyLoRA (SFT and GRPO, 13 params)"})
    else:
        labelled += tiny
    offsets = {"lora": (8, -16), "qlora": (8, 4), "wsp_f4": (8, -18)}
    for r in labelled:
        ax.annotate(
            f"{r['label']}  {r['ex']:.1f}%",
            (r["trainable_params"], r["ex"]),
            xytext=offsets.get(r["run"], (8, 6)),
            textcoords="offset points",
            color=TEXT,
            fontsize=11,
        )
    if base:
        ax.axhline(base["ex"], color=MUTED, linestyle="--", linewidth=1.5)
        ax.annotate(
            f"Base model, no training: {base['ex']:.1f}%",
            (min(r["trainable_params"] for r in pts), base["ex"]),
            xytext=(0, 6),
            textcoords="offset points",
            color=TEXT2,
            fontsize=10,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("BIRD dev execution accuracy (%)")
    ax.set_title("Accuracy vs. how many weights you train")
    ax.set_xlim(3, 5e9)
    fig.tight_layout()
    fig.savefig(figs / "ex_vs_params.png", dpi=160)
    plt.close(fig)


def chart_difficulty(rows: list[dict], figs: Path) -> None:
    rs = [r for r in rows if r["run"] != "qlora_4bit"]
    x = np.arange(len(rs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = (
        ("ex_simple", "Simple", S1),
        ("ex_moderate", "Moderate", S2),
        ("ex_challenging", "Challenging", S3),
    )
    for i, (key, lab, col) in enumerate(bars):
        ax.bar(x + (i - 1) * w, [r[key] for r in rs], width=w - 0.02, color=col, label=lab, zorder=3)
    ax.set_xticks(x, [r["label"].replace(" + ", "\n+ ").replace(" (", "\n(") for r in rs], fontsize=10)
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_title("Accuracy by BIRD difficulty")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "ex_by_difficulty.png", dpi=160)
    plt.close(fig)


def _cost_note(key: str, r: dict) -> str:
    if key != "peak_vram_gb":
        return ""
    if r["run"] == "full" and r.get("optimizer_state_gb"):
        return f"  (+{r['optimizer_state_gb']:.1f} GB optimizer in CPU RAM)"
    if r["run"] == "tinylora_grpo":
        return "  (incl. vLLM sampler)"
    return ""


def chart_cost(rows: list[dict], figs: Path) -> None:
    rs = [r for r in rows if r["peak_vram_gb"] and r["run"] != "qlora_4bit"]
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    panels = (
        ("peak_vram_gb", "Peak training VRAM", "GB", 1.9),
        ("train_hours", "Training wall-clock", "h", 1.25),
    )
    for ax, (key, title, unit, headroom) in zip(axes, panels, strict=True):
        vals = [r[key] or 0 for r in rs]
        ax.barh([r["label"] for r in rs], vals, color=S1, height=0.6, zorder=3)
        for i, (v, r) in enumerate(zip(vals, rs, strict=True)):
            ax.annotate(
                f"{v:.1f} {unit}{_cost_note(key, r)}",
                (v, i),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=TEXT,
                fontsize=10,
            )
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * headroom)
    fig.tight_layout()
    fig.savefig(figs / "training_cost.png", dpi=160)
    plt.close(fig)


def chart_grpo(runs: Path, figs: Path) -> None:
    p = runs / "tinylora_grpo" / "train_log.csv"
    if not p.exists():
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))
    key = next((k for k in ("reward", "rewards/execution_reward/mean") if k in rows[0]), None)
    if not key:
        return
    steps, rew = zip(*[(int(r["step"]), float(r[key])) for r in rows if r.get(key)], strict=True)
    k = SMOOTH_K
    smooth = [sum(rew[i - k + 1 : i + 1]) / k for i in range(k - 1, len(rew))]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(steps, [100 * r for r in rew], color=MUTED, linewidth=1, label="each step")
    ax.plot(
        steps[k - 1 :], [100 * s for s in smooth], color=S1, linewidth=2.5, label=f"average of last {k} steps"
    )
    ax.annotate(
        f"{100 * smooth[0]:.1f}%",
        (steps[k - 1], 100 * smooth[0]),
        xytext=(-6, -16),
        textcoords="offset points",
        ha="right",
        color=TEXT,
    )
    ax.annotate(
        f"{100 * smooth[-1]:.1f}%",
        (steps[-1], 100 * smooth[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=TEXT,
    )
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel("GRPO step (8 questions x 8 sampled answers each)")
    ax.set_ylabel("Rollouts with correct result (%)")
    ax.set_title("TinyLoRA + GRPO: training reward (13 parameters)")
    fig.tight_layout()
    fig.savefig(figs / "tinylora_grpo_reward.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    results, runs, figs = Path(args.results), Path(args.runs), Path(args.out)
    figs.mkdir(parents=True, exist_ok=True)
    style()
    rows = load(results, runs)
    if not rows:
        raise SystemExit("no results yet")
    write_tables(rows, results)
    chart_params(rows, figs)
    chart_difficulty(rows, figs)
    chart_cost(rows, figs)
    chart_grpo(runs, figs)
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
