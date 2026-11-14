"""Number formatting, a consistency check, and small JSON and CSV writers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _missing(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(x))


def fmt_num(x, digits: int = 3) -> str:
    if _missing(x):
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):,.{digits}f}"


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.ndarray, pd.Series)):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean(o):
    """Replace float NaN with None so the JSON stays valid."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, default=_default) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def check(condition: bool, message: str) -> None:
    """Stop a script whose inputs fail a consistency check."""
    if not condition:
        raise AssertionError(f"check failed: {message}")


def write_table(df: pd.DataFrame, stem: Path) -> None:
    """Write `<stem>.csv`."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
