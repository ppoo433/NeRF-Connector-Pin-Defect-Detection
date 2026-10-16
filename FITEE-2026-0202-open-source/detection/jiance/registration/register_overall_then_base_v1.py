from __future__ import annotations

import sys
from pathlib import Path

_DAIMA_CTX = None


def _find_repo_root(from_path: Path) -> Path:
    p = from_path.resolve()
    start = p if p.is_dir() else p.parent
    for d in [start] + list(start.parents):
        if (d / "daima_runtime").is_dir():
            return d
    for d in [start] + list(start.parents):
        if (d / ".release-root").is_file():
            return d
    for d in [start] + list(start.parents):
        if (d / ".git").exists():
            return d
    return start


if __name__ == "__main__":
    try:
        sys.dont_write_bytecode = True
        _REPO_ROOT = _find_repo_root(Path(__file__))
        _JIANCE_DIR = Path(__file__).resolve().parents[1]
        for p in [_REPO_ROOT, _JIANCE_DIR]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        from daima_runtime.runlog import setup_run

        _DAIMA_CTX = setup_run(__file__)
    except Exception:
        _DAIMA_CTX = None

from reg_common import run_registration


def main() -> int:
    return int(run_registration("overall_then_base_v1", daima_ctx=_DAIMA_CTX))


if __name__ == "__main__":
    raise SystemExit(main())
