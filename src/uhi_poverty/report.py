"""Markdown tables, number formatting, and small JSON/text writers."""

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


def fmt_int(x) -> str:
    if _missing(x):
        return "–"
    return f"{int(round(float(x))):,}"


def fmt_pct(x, digits: int = 1) -> str:
    """A proportion (0–1) as a percentage."""
    if _missing(x):
        return "–"
    return f"{100 * float(x):.{digits}f} %"


def fmt_p(p) -> str:
    if _missing(p):
        return "–"
    p = float(p)
    if p < 1e-16:  # below double precision; scipy returns 0
        return "< 1e-16"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def fmt_ci(est, lo, hi, digits: int = 3) -> str:
    return f"{fmt_num(est, digits)} [{fmt_num(lo, digits)}, {fmt_num(hi, digits)}]"


def require(condition: bool, message: str) -> None:
    """Stop a README writer whose prose would no longer match its results."""
    if not condition:
        raise AssertionError(f"README text out of date: {message}")


def md_table(df: pd.DataFrame, digits: int | dict = 3, index: bool = False) -> str:
    """GitHub-flavoured Markdown table. Integers get thousands separators."""
    if index:
        df = df.reset_index()
    per_col = digits if isinstance(digits, dict) else {}
    default = 3 if isinstance(digits, dict) else digits

    def cell(v, col):
        if isinstance(v, str):
            return v.replace("|", "\\|")
        if _missing(v):
            return "–"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):,.{per_col.get(col, default)}f}"
        return str(v)

    def numeric(c):
        return pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    align = "|" + "|".join("---:" if numeric(c) else "---" for c in df.columns) + "|"
    body = [
        "| " + " | ".join(cell(v, c) for c, v in zip(df.columns, row, strict=True)) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, align, *body]) + "\n"


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


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_table(df: pd.DataFrame, stem: Path, digits: int | dict = 3, title: str | None = None,
                note: str | None = None) -> None:
    """Write `<stem>.csv` and `<stem>.md` side by side."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    parts = []
    if title:
        parts.append(f"# {title}\n")
    parts.append(md_table(df, digits))
    if note:
        parts.append(note)
    write_text("\n".join(parts), stem.with_suffix(".md"))
