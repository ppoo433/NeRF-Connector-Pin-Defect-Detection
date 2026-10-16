"""Runtime helpers for the daima repo.

This package is intentionally stdlib-only so it can be imported from any of the
repo's virtual environments without extra dependencies.
"""

from __future__ import annotations

from .runlog import RunContext, setup_run  # noqa: F401

