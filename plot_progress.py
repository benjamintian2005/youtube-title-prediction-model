"""Regenerates progress.png from champion.json's git history - one point per
promoted champion, in chronological order. Run any time after a new champion
is promoted to refresh the chart (mirrors karpathy/autoresearch's progress.png)."""
import json
import subprocess

import matplotlib.pyplot as plt

# Reference palette (see the dataviz skill's references/palette.md) - fixed
# categorical slot order, one hue per panel since each is its own single series.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e2e1dc"
SURFACE = "#fcfcfb"

OUT_PATH = "progress.png"


def _git(*args):
    # encoding/errors explicit: benchmark/dataset.csv contains titles with
    # emoji etc. that break the platform-default codec (cp1252 on Windows)
    return subprocess.run(
        ["git", *args], capture_output=True, check=True,
        encoding="utf-8", errors="replace",
    ).stdout


def load_champion_history():
    hashes = [
        line.split("|")[0]
        for line in _git("log", "--follow", "--format=%H|%ad", "--date=iso", "--", "champion.json").splitlines()
    ]
    points = []
    for h in reversed(hashes):  # oldest -> newest
        raw = _git("show", f"{h}:champion.json")
        data = json.loads(raw)
        n_rows = int(_git("show", f"{h}:benchmark/dataset.csv").count("\n")) - 1
        points.append({
            "log_r2": data["log_r2"],
            "mdape": data["mdape"],
            "n_rows": n_rows,
            "timestamp": data["timestamp"],
        })
    return points


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _line_panel(ax, x, y, color, title, y_fmt, higher_is_better=True):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", zorder=3)
    ax.scatter(x, y, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)
    best_i = (max if higher_is_better else min)(range(len(y)), key=lambda i: y[i])
    ax.annotate(
        y_fmt(y[-1]), (x[-1], y[-1]), textcoords="offset points", xytext=(8, 4),
        color=TEXT_PRIMARY, fontsize=10, fontweight="bold",
    )
    if best_i != len(y) - 1:
        ax.scatter([x[best_i]], [y[best_i]], s=64, facecolors="none",
                    edgecolors=color, linewidths=2, zorder=5)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    _style_axis(ax)


def main():
    points = load_champion_history()
    x = list(range(1, len(points) + 1))
    log_r2 = [p["log_r2"] for p in points]
    mdape = [p["mdape"] for p in points]
    n_rows = [p["n_rows"] for p in points]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    fig.suptitle(
        "youtube title predictor - autoresearch progress",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.text(
        0.01, 0.90,
        f"{len(points)} promoted champions so far - each point is a kept `program.md` experiment",
        color=TEXT_SECONDARY, fontsize=9.5,
    )

    _line_panel(axes[0], x, log_r2, BLUE, "val log_r2 (higher = better)", lambda v: f"{v:.2f}")
    _line_panel(axes[1], x, mdape, ORANGE, "val mdape % (lower = better)", lambda v: f"{v:.0f}%", higher_is_better=False)
    _line_panel(axes[2], x, n_rows, AQUA, "dataset size (rows)", lambda v: f"{v:,}")

    for ax in axes:
        ax.set_xlabel("champion generation", color=TEXT_SECONDARY, fontsize=9)
        ax.set_xticks(x)

    fig.text(
        0.01, 0.02,
        "open circle = best-so-far point if not the latest. raw log_mse isn't shown - it's not comparable "
        "across benchmark re-freezes; log_r2 is the cross-epoch-comparable metric used here.",
        color=TEXT_SECONDARY, fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"Wrote {OUT_PATH} ({len(points)} generations)")


if __name__ == "__main__":
    main()
