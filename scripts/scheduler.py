"""Memory-aware GPU job scheduler: runs the whole experiment DAG on one GPU.

A job starts when (a) all its deps succeeded and (b) its VRAM estimate fits in both the GPU's currently
free memory and the budget left after the estimates of running jobs.
Logs: runs/_sched/<job>.log, status: runs/_sched/status.json.

  .venv/bin/python scripts/scheduler.py [--only job1,job2] [--budget-gb 9.5] [--skip-done]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
SCHED = ROOT / "runs/_sched"
BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SWEEP_LRS = ["1e-3", "1e-2", "5e-2"]
SWEEP_STEPS = 40
POLL_SECONDS = 10
SETTLE_SECONDS = 75  # a just-launched job needs this long to allocate before nvidia-smi is trusted again
FREE_MARGIN_GB = 0.5


@dataclass
class Job:
    name: str
    cmd: list[str] | Callable[[], list[str]]  # callable: built at launch time (e.g. needs the sweep winner)
    vram_gb: float
    deps: list[str] = field(default_factory=list)
    priority: int = 0  # higher first
    state: str = "pending"  # pending|running|done|failed|skipped
    proc: subprocess.Popen | None = None
    t0: float = 0.0
    t1: float = 0.0


def sft(name: str, cfg: str, extra: list[str] | tuple = ()) -> list[str]:
    return [PY, "-m", "t2s.train_sft", "--config", f"configs/{cfg}.yaml", "--out", f"runs/{name}", *extra]


def merge(name: str) -> list[str]:
    return [PY, "-m", "t2s.merge", "--run", f"runs/{name}", "--out", f"checkpoints/{name}"]


def evaluate(name: str) -> list[str]:
    return [PY, "-m", "t2s.evaluate", "--model", f"checkpoints/{name}", "--name", name, "--gpu-mem", "0.5"]


def best_sweep_lr() -> str:
    """Lowest mean loss over the last 10 logged steps of each TinyLoRA-SFT sweep run."""
    best, best_loss = None, float("inf")
    for lr in SWEEP_LRS:
        p = ROOT / f"runs/tinylora_sweep_{lr}/train_log.csv"
        if not p.exists():
            continue
        with open(p) as f:
            losses = [float(r["loss"]) for r in csv.DictReader(f) if r.get("loss")]
        if not losses:
            continue
        tail = sum(losses[-10:]) / len(losses[-10:])
        print(f"[sched] sweep lr={lr}: tail loss {tail:.4f}", flush=True)
        if tail < best_loss:
            best, best_loss = lr, tail
    if best is None:
        raise RuntimeError("no sweep results")
    (SCHED / "best_tinylora_lr.txt").write_text(best)
    return best


def build_jobs() -> dict[str, Job]:
    jobs = [
        # biggest SFT jobs get priority so the solo full-FT job doesn't wait behind packed pairs
        Job("sft_full", sft("full", "full"), 8.5, priority=6),
        Job("sft_lora", sft("lora", "lora"), 6.0, priority=8),
        Job("sft_qlora", sft("qlora", "qlora"), 5.0, priority=7),
        Job("sft_lora_fa", sft("lora_fa", "lora_fa"), 5.5, priority=7),
    ]
    sweep_args = ["--max-steps", str(SWEEP_STEPS)]
    for lr in SWEEP_LRS:
        jobs.append(
            Job(
                f"sweep_{lr}",
                sft(f"tinylora_sweep_{lr}", "tinylora_sft", ["--lr", lr, *sweep_args]),
                5.0,
                priority=12,
            )
        )
    sweep_deps = [f"sweep_{lr}" for lr in SWEEP_LRS]
    jobs.append(
        Job(
            "sft_tinylora",
            lambda: sft("tinylora_sft", "tinylora_sft", ["--lr", best_sweep_lr()]),
            5.0,
            deps=sweep_deps,
            priority=5,
        )
    )
    # GRPO reuses the TinyLoRA-SFT LR (same step budget, same optimizer -> comparable |v|)
    grpo_cmd = [
        PY,
        "-m",
        "t2s.train_grpo",
        "--config",
        "configs/tinylora_grpo.yaml",
        "--out",
        "runs/tinylora_grpo",
    ]
    jobs.append(
        Job("grpo_tinylora", lambda: [*grpo_cmd, "--lr", best_sweep_lr()], 9.5, deps=sweep_deps, priority=11)
    )

    trained = [
        ("full", "sft_full"),
        ("lora", "sft_lora"),
        ("qlora", "sft_qlora"),
        ("lora_fa", "sft_lora_fa"),
        ("tinylora_sft", "sft_tinylora"),
        ("tinylora_grpo", "grpo_tinylora"),
    ]
    for name, train_job in trained:
        jobs.append(Job(f"merge_{name}", merge(name), 1.0, deps=[train_job], priority=20))
        jobs.append(Job(f"eval_{name}", evaluate(name), 9.0, deps=[f"merge_{name}"], priority=15))
    # QLoRA as actually deployed: 4-bit base + adapter
    qlora_4bit = [PY, "-m", "t2s.evaluate", "--model", BASE, "--name", "qlora_4bit"]
    qlora_4bit += ["--quantization", "bitsandbytes", "--lora", "runs/qlora/final", "--gpu-mem", "0.5"]
    jobs.append(Job("eval_qlora_4bit", qlora_4bit, 9.0, deps=["sft_qlora"], priority=4))
    evals = [j.name for j in jobs if j.name.startswith("eval_")]
    jobs.append(Job("report", [PY, "-m", "t2s.report"], 0.0, deps=evals, priority=0))
    return {j.name: j for j in jobs}


def gpu_free_gb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    return float(out.strip().splitlines()[0]) / 1024


def write_status(jobs: dict[str, Job]) -> None:
    st = {
        n: {"state": j.state, "secs": round((j.t1 or time.time()) - j.t0) if j.t0 else 0}
        for n, j in jobs.items()
    }
    (SCHED / "status.json").write_text(json.dumps(st, indent=1))


def mark_initial_done(jobs: dict[str, Job], only: str | None, skip_done: bool) -> None:
    if only:
        keep = set(only.split(","))
        for n, j in jobs.items():
            if n not in keep:
                j.state = "done"
    if skip_done:
        for n, j in jobs.items():
            if n.startswith("eval_") and (ROOT / f"results/{n[5:]}/metrics.json").exists():
                j.state = "done"
            if n.startswith("merge_") and (ROOT / f"checkpoints/{n[6:]}/config.json").exists():
                j.state = "done"


def reap_finished(jobs: dict[str, Job]) -> None:
    for n, j in jobs.items():
        if j.state == "running" and j.proc.poll() is not None:
            j.t1 = time.time()
            j.state = "done" if j.proc.returncode == 0 else "failed"
            print(
                f"[sched] {n} {j.state.upper()} rc={j.proc.returncode} in {(j.t1 - j.t0) / 60:.1f} min",
                flush=True,
            )


def skip_blocked(jobs: dict[str, Job]) -> None:
    for n, j in jobs.items():
        if j.state == "pending" and any(jobs[d].state in ("failed", "skipped") for d in j.deps):
            j.state = "skipped"
            print(f"[sched] {n} SKIPPED (dep failed)", flush=True)


def launch(j: Job, free: float) -> None:
    cmd = j.cmd() if callable(j.cmd) else j.cmd
    with open(SCHED / f"{j.name}.log", "w") as log:  # the child keeps its own copy of the fd
        j.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    j.state, j.t0 = "running", time.time()
    print(f"[sched] START {j.name} (est {j.vram_gb} GB, free {free:.1f} GB): {' '.join(cmd[2:])}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Serial by design: measured pairing cost ~35% throughput and WSL spills VRAM overflow to RAM.
    # Budget fits one training/eval job (>=5 GB) plus a small merge (1 GB), never two big jobs.
    ap.add_argument("--budget-gb", type=float, default=9.5)
    ap.add_argument("--only", default=None, help="comma list; others are treated as done")
    ap.add_argument("--skip-done", action="store_true", help="treat jobs whose output exists as done")
    args = ap.parse_args()
    SCHED.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    mark_initial_done(jobs, args.only, args.skip_done)
    last_launch = 0.0
    while True:
        reap_finished(jobs)
        skip_blocked(jobs)
        write_status(jobs)
        if all(j.state not in ("pending", "running") for j in jobs.values()):
            break
        ready = sorted(
            (
                j
                for j in jobs.values()
                if j.state == "pending" and all(jobs[d].state == "done" for d in j.deps)
            ),
            key=lambda j: -j.priority,
        )
        running_est = sum(j.vram_gb for j in jobs.values() if j.state == "running")
        settled = time.time() - last_launch > SETTLE_SECONDS
        free = gpu_free_gb()
        for j in ready:
            fits_budget = running_est + j.vram_gb <= args.budget_gb
            fits_free = j.vram_gb <= free - FREE_MARGIN_GB
            if not fits_budget or not (j.vram_gb == 0 or (fits_free and settled)):
                continue
            launch(j, free)
            if j.vram_gb:
                running_est += j.vram_gb
                last_launch = time.time()
                break  # one GPU launch per settle window
        time.sleep(POLL_SECONDS)
    failed = [n for n, j in jobs.items() if j.state in ("failed", "skipped")]
    print(f"[sched] ALL DONE. failed/skipped: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
