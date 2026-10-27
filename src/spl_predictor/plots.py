"""The figures used in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def model_comparison(backtest: pd.DataFrame, holdout: pd.DataFrame) -> plt.Figure:
    """Share of results picked correctly: backtest seasons and the 2025-26 season."""
    order = backtest.sort_values("accuracy").index
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hlines(y, backtest.loc[order, "accuracy"], holdout.loc[order, "accuracy"], color=GRID, lw=2, zorder=1)
    ax.scatter(backtest.loc[order, "accuracy"], y, s=70, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Backtest, 2016-17 to 2024-25")
    ax.scatter(holdout.loc[order, "accuracy"], y, s=70, color=ORANGE, marker="D", zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="2025-26 season")
    ax.set_yticks(list(y), order)
    ax.set_xlim(0.40, 0.62)
    ax.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Match results picked correctly (higher is better)")
    ax.set_title("Only logistic regression clearly beats always picking the home team")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2, handletextpad=0.3)
    return fig


def overfitting_gap(summary: pd.DataFrame) -> plt.Figure:
    """Log loss on the matches a model trained on against matches it had not seen."""
    order = summary.sort_values("gap").index
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hlines(y, summary.loc[order, "train_log_loss"], summary.loc[order, "log_loss"], color=GRID, lw=2, zorder=1)
    ax.scatter(summary.loc[order, "train_log_loss"], y, s=70, facecolor=SURFACE, edgecolor=BLUE,
               linewidth=2, zorder=3, label="Matches it trained on")
    ax.scatter(summary.loc[order, "log_loss"], y, s=70, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Unseen season")
    for row, name in zip(y, order):
        ax.annotate(f"{summary.loc[name, 'gap']:+.2f}",
                    (summary.loc[name, ["train_log_loss", "log_loss"]].max(), row),
                    xytext=(9, 0), textcoords="offset points", va="center", color=MUTED)
    ax.set_yticks(list(y), order)
    ax.set_xlim(0, 1.7)
    ax.set_xlabel("Log loss (lower is better)")
    ax.set_title("The tree models memorise the matches they trained on")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2, handletextpad=0.3)
    return fig


def season_table(table: pd.DataFrame, season: str) -> plt.Figure:
    """Pre-season forecast position for each club against where it finished."""
    table = table.sort_values("predicted_position")
    y = table["predicted_position"]
    n = len(table)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.hlines(y, table["predicted_position"], table["position"], color=GRID, lw=2, zorder=1)
    ax.scatter(table["predicted_position"], y, s=90, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(table["position"], y, s=34, color=ORANGE, marker="D", zorder=4, edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y, table["team"])
    ax.set_ylim(n + 0.7, 0.3)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("League position")
    ax.set_title(f"{season}: table from summed match predictions, made before the season")
    ax.legend(
        handles=[Line2D([], [], marker="o", color=BLUE, lw=0, markersize=9, label="Forecast"),
                 Line2D([], [], marker="D", color=ORANGE, lw=0, markersize=6, label="Actual finish")],
        loc="upper left", bbox_to_anchor=(0, -0.09), ncols=2, handletextpad=0.8, columnspacing=2.0,
    )
    return fig
