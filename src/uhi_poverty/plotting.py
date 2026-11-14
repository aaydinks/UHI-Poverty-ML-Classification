"""Shared figure style: colour tokens, fonts, and the diverging SUHI scale.

Colours follow one fixed mapping per entity (a model keeps its colour in every
figure). Text and axes use ink tokens, never series colours. The first three
categorical slots pass the all-pairs colour-vision checks; nothing here uses more.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

MODEL_COLORS = {"lgbm": BLUE, "logistic": ORANGE, "majority": MUTED}
LABEL_COLORS = {0: BLUE, 1: ORANGE}

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281",
            "#0d366b"]
# Diverging blue (cooler than rural) <-> red (hotter than rural), grey midpoint at 0 °C.
DIVERGING = ["#104281", "#2a78d6", "#86b6ef", NEUTRAL, "#f2a3a2", "#e34948", "#8f1d1d"]

DPI = 600


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.dpi": 100,
    })


def suhi_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("suhi_diverging", DIVERGING, N=256)


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE, N=256)


def status_cmap(colors: list[str]) -> ListedColormap:
    return ListedColormap(colors)


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def note(fig, text: str, y: float = -0.02) -> None:
    """A small source/method note under the figure, in secondary ink."""
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7, color=INK_2, wrap=True)
