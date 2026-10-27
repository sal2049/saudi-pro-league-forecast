"""The figures used in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

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


def model_comparison(backtest: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> plt.Figure:
    """Share of results picked correctly, per model, in each evaluation set."""
    order = backtest.sort_values("accuracy").index
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    series = [(backtest, BLUE, "o", "Backtest, 2016-17 to 2024-25"),
              (validation, MUTED, "s", "2025-26 (validation)"),
              (test, ORANGE, "D", "2026-27 so far (test)")]
    for table, colour, marker, label in series:
        ax.scatter(table.loc[order, "accuracy"], y, s=60, color=colour, marker=marker, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2, label=label)
    ax.set_yticks(y, order)
    ax.set_xlim(0.35, 0.70)
    ax.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Match results picked correctly (higher is better)")
    ax.set_title("Logistic regression beats always picking the home team, every time")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncols=3, handletextpad=0.2, columnspacing=1.2)
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
    ax.set_xlim(0, max(1.2, summary["log_loss"].max() + 0.2))
    ax.set_xlabel("Log loss (lower is better)")
    ax.set_title("The tree models memorise the matches they trained on")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2, handletextpad=0.3)
    return fig


def probability_curve(model, train: pd.DataFrame) -> plt.Figure:
    """How the fitted model turns the gap into three probabilities."""
    fitted = clone(model).fit(train[["gap"]], train["result"])
    gaps = np.linspace(-2, 2, 81)
    probs = pd.DataFrame(fitted.predict_proba(pd.DataFrame({"gap": gaps})), columns=fitted.classes_, index=gaps)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.grid(axis="y")
    for result, colour, label in (("H", BLUE, "Home win"), ("D", MUTED, "Draw"), ("A", ORANGE, "Away win")):
        ax.plot(gaps, probs[result], color=colour, lw=2.2, label=label)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Gap: home team's goal difference per game over its last 50 matches, minus the away team's")
    ax.set_title("The draw is never the most likely result")
    ax.legend(loc="upper left", ncols=3, handletextpad=0.4)
    return fig
