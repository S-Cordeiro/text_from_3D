"""Plot train vs validation curves of the two_cubes vision runs.

Usage: python scripts/plot_two_cubes.py [--out figures/two-cubes-train-val.png]
Reads runs/two_cubes_*/run_*/metrics.jsonl (local copies pulled from Bentley).
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    # (label, path, colour) — colours from the dataviz reference palette, slots 1–2.
    ("Constant LR 1e-4, batch 8", "runs/two_cubes_lr_1e-4/run_3", "#2a78d6"),
    ("Warmup + cosine, batch 16", "runs/two_cubes_cosine/run_1", "#eb6834"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    train, val = [], []
    for line in open(Path(path) / "metrics.jsonl"):
        row = json.loads(line)
        if row["split"] == "train":
            train.append(row)
        elif row["split"] == "val":
            val.append(row)
    steps_per_epoch = max(r["step"] for r in train) / max(r["epoch"] for r in train)
    return train, val, steps_per_epoch


def series(rows, key, steps_per_epoch, smooth=1):
    x = np.array([r["step"] / steps_per_epoch for r in rows])
    y = np.array([r[key] for r in rows], dtype=float)
    if key.endswith("_angle"):
        y = np.degrees(y)
    if smooth > 1:  # mean over one epoch: training batches are very noisy
        n = len(y) // smooth * smooth
        x = x[:n].reshape(-1, smooth).max(1)
        y = y[:n].reshape(-1, smooth).mean(1)
    return x, y


def ema_log(y, alpha=0.3):
    """Exponential moving average in log space (suits log-scale losses)."""
    out, acc = np.empty_like(y), math.log(y[0])
    for i, v in enumerate(y):
        acc = alpha * math.log(v) + (1 - alpha) * acc
        out[i] = math.exp(acc)
    return out


def plot_best(out, panels=(("total", "Total loss"), ("object_rotation", "Object rotation loss"))):
    """Train vs validation loss of the best run (constant LR), one panel per key."""
    _, path, _ = RUNS[0]
    train, val, spe = load(path)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels) + 1, 3.8),
                             dpi=150, squeeze=False)
    axes = axes[0]
    for ax, (key, title) in zip(axes, panels):
        tx, ty = series(train, key, spe, smooth=int(spe))
        vx, vy = series(val, key, spe)
        for x, y, colour in [(tx, ty, "#2a78d6"), (vx, vy, "#eb6834")]:
            ax.plot(x, y, color=colour, lw=1, alpha=0.25)  # raw epoch values
            ax.plot(x, ema_log(y), color=colour, lw=2)      # smoothed
        if key == "total":
            best = int(np.argmin(vy))
            ax.plot(vx[best], vy[best], "o", ms=7, color="#eb6834", mec="white", mew=1.5)
            ax.annotate(f"best checkpoint\nepoch {vx[best]:.0f}, loss {vy[best]:.3f}",
                        (vx[best], vy[best]), xytext=(-20, 75), textcoords="offset points",
                        ha="right", fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_yscale("log")
        ax.set_xlim(0, 80)
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} (log scale)", color=INK, loc="left", fontsize=11)
        ax.grid(True, color=GRID, lw=0.8)
    handles = [plt.Line2D([], [], color="#2a78d6", lw=2),
               plt.Line2D([], [], color="#eb6834", lw=2),
               plt.Line2D([], [], color=MUTED, lw=1, alpha=0.4)]
    labels = ["train (epoch mean)", "validation (500 scenes)", "unsmoothed"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("Two cubes, best run (constant LR 1e-4)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, facecolor="white")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/two-cubes-train-val.png")
    parser.add_argument("--best-out", default="figures/two-cubes-best-loss.png")
    parser.add_argument("--rotation-out", default="figures/two-cubes-rotation-loss.png")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    panels = [("total", "Total loss (log scale)", True),
              ("object_rotation_angle", "Object rotation error (°)", False)]

    for label, path, colour in RUNS:
        train, val, spe = load(path)
        for ax, (key, _, _) in zip(axes, panels):
            ax.plot(*series(train, key, spe, smooth=int(spe)), color=colour,
                    lw=1.5, alpha=0.45, ls="--")
            ax.plot(*series(val, key, spe), color=colour, lw=2, label=label)

    for ax, (key, title, log) in zip(axes, panels):
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_xlim(0, 80)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 40)

    # Legend: colour = run, line style = split.
    handles = [plt.Line2D([], [], color=c, lw=2) for _, _, c in RUNS]
    handles += [plt.Line2D([], [], color=MUTED, lw=2),
                plt.Line2D([], [], color=MUTED, lw=1.5, ls="--", alpha=0.6)]
    labels = [l for l, _, _ in RUNS] + ["validation (500 scenes)", "train (epoch mean)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out, facecolor="white")
    print(f"saved {args.out}")
    plot_best(args.best_out)
    plot_best(args.rotation_out, panels=(("object_rotation", "Object rotation loss"),))


if __name__ == "__main__":
    main()
