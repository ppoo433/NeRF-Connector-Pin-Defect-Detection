from __future__ import annotations

"""
Split selection helpers.

We keep split files versioned under `splits/` (e.g. split_v1.csv, split_v2.csv).
Scripts should default to the latest split, but allow overriding for reproducibility.
"""

import os
from pathlib import Path

_SPLIT_VERSION_ENV = "DAIMA_SPLIT_VERSION"
_SPLIT_CSV_ENV = "DAIMA_SPLIT_CSV"


def resolve_split_csv(repo_root: Path, *, default_version: str = "split_v3") -> Path:
    """
    Resolve which split csv to use.

    Priority:
      1) DAIMA_SPLIT_CSV: explicit csv path (absolute or repo-root relative).
      2) DAIMA_SPLIT_VERSION: e.g. "split_v2" or "split_v2.csv".
      3) default_version.
    """
    p = os.environ.get(_SPLIT_CSV_ENV)
    if p is not None and str(p).strip() != "":
        pp = Path(str(p).strip())
        return pp if pp.is_absolute() else (repo_root / pp)

    v = os.environ.get(_SPLIT_VERSION_ENV, default_version)
    v = str(v).strip()
    if v.lower().endswith(".csv"):
        name = v
    else:
        name = f"{v}.csv"
    return repo_root / "splits" / name
