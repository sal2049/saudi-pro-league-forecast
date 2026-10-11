"""The three figures used in the README."""

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


def model_comparison(summary: pd.DataFrame, holdout_mae: pd.Series) -> plt.Figure:
    """Backtest error per model next to its error on the untouched 2025-26 season."""
    order = summary.sort_values("test_mae", ascending=False).index
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hlines(y, summary.loc[order, "test_mae"], holdout_mae[order], color=GRID, lw=2, zorder=1)
    ax.scatter(summary.loc[order, "test_mae"], y, s=70, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Backtest, 2021-22 to 2024-25")
    ax.scatter(holdout_mae[order], y, s=70, color=ORANGE, marker="D", zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Holdout, 2025-26")

    ax.set_yticks(list(y), order)
    ax.set_xlim(0, 4)
    ax.set_xlabel("Average position error (places, lower is better)")
    ax.set_title("Which model looks best depends on which season you test on")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=2, handletextpad=0.3)
    return fig


def overfitting_gap(summary: pd.DataFrame) -> plt.Figure:
    """Error on seasons a model trained on versus seasons it had never seen."""
    order = summary.sort_values("gap").index
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hlines(y, summary.loc[order, "train_mae"], summary.loc[order, "test_mae"],
              color=GRID, lw=2, zorder=1)
    ax.scatter(summary.loc[order, "train_mae"], y, s=70, facecolor=SURFACE, edgecolor=BLUE,
               linewidth=2, zorder=3, label="Seasons it trained on")
    ax.scatter(summary.loc[order, "test_mae"], y, s=70, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Unseen season")
    for row, name in zip(y, order):
        gap = summary.loc[name, "gap"]
        ax.annotate(f"{gap:+.1f}", (summary.loc[name, ["train_mae", "test_mae"]].max(), row),
                    xytext=(9, 0), textcoords="offset points", va="center", color=MUTED)

    ax.set_yticks(list(y), order)
    ax.set_xlim(0, 4)
    ax.set_xlabel("Average position error (places)")
    ax.set_title("The flexible models memorise the training seasons")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=2, handletextpad=0.3)
    return fig


def forecast_vs_actual(forecast: pd.DataFrame, final_table: pd.DataFrame) -> plt.Figure:
    """Forecast position and range for each club against where it finished."""
    actual = forecast["team"].map(final_table.set_index("team")["position"])
    y = forecast["pred_position"]
    n_teams = len(forecast)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.hlines(y, forecast["pos_low"], forecast["pos_high"], color=BLUE, alpha=0.18, lw=7,
              capstyle="round", zorder=1)
    ax.scatter(forecast["pred_position"], y, s=110, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(actual, y, s=34, color=ORANGE, marker="D", zorder=4,
               edgecolor=SURFACE, linewidth=1.2)

    labels = [f"{t} (promoted)" if p else t for t, p in zip(forecast["team"], forecast["promoted"])]
    ax.set_yticks(y, labels)
    ax.set_ylim(n_teams + 0.7, 0.3)
    ax.set_xlim(0.3, n_teams + 0.7)
    ax.set_xticks(range(1, n_teams + 1))
    ax.set_xlabel("League position")
    ax.set_title("2025-26: forecast made from data up to May 2025, and what happened")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, lw=0, markersize=9, label="Forecast"),
            Line2D([], [], color=BLUE, alpha=0.18, lw=7, label="80% range"),
            Line2D([], [], marker="D", color=ORANGE, lw=0, markersize=6, label="Actual finish"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.09), ncols=3, handletextpad=0.8,
        columnspacing=2.0,
    )
    return fig
