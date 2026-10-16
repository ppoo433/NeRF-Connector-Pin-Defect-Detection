# -*- coding: utf-8 -*-
"""Shared sample-subset reporting helpers for formal eval entry points."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _natural_sample_key(text: str) -> tuple[int, str]:
    value = str(text or "").strip()
    return (len(value), value)


def sorted_sample_ids(sample_ids: Iterable[Any]) -> List[str]:
    normalized = {str(x or "").strip() for x in sample_ids if str(x or "").strip()}
    return sorted(normalized, key=_natural_sample_key)


def summarize_skip_reason_counts(rows: Sequence[Mapping[str, Any]], reason_field: str = "reason") -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in list(rows or []):
        reason = str(dict(row).get(reason_field, "") or "").strip() or "unknown"
        counter[str(reason)] += 1
    return {str(key): int(counter[key]) for key in sorted(counter.keys())}


def build_task_subset(
    *,
    sample_ids_requested: Iterable[Any],
    sample_ids_used: Iterable[Any],
    skipped: Sequence[Mapping[str, Any]] | None = None,
    not_applicable_sample_ids: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    requested = sorted_sample_ids(sample_ids_requested)
    used = sorted_sample_ids(sample_ids_used)
    not_applicable = sorted_sample_ids(not_applicable_sample_ids or [])
    skipped_rows = [dict(row) for row in list(skipped or [])]
    return {
        "sample_ids_requested": list(requested),
        "sample_count_requested": int(len(requested)),
        "sample_ids_used": list(used),
        "sample_count_used": int(len(used)),
        "skipped": skipped_rows,
        "skip_reason_counts": summarize_skip_reason_counts(skipped_rows),
        "not_applicable_sample_ids": list(not_applicable),
        "not_applicable_count": int(len(not_applicable)),
    }


def build_formal_sample_contract(
    *,
    policy: str,
    sample_ids_requested: Iterable[Any],
    sample_ids_used_global: Iterable[Any],
    global_skipped: Sequence[Mapping[str, Any]] | None,
    tasks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    requested = sorted_sample_ids(sample_ids_requested)
    used_global = sorted_sample_ids(sample_ids_used_global)
    skipped_rows = [dict(row) for row in list(global_skipped or [])]
    return {
        "schema_version": 1,
        "policy": str(policy),
        "sample_ids_requested": list(requested),
        "sample_count_requested": int(len(requested)),
        "sample_ids_used_global": list(used_global),
        "sample_count_used_global": int(len(used_global)),
        "global_skipped": skipped_rows,
        "global_skip_reason_counts": summarize_skip_reason_counts(skipped_rows),
        "tasks": {str(task_name): dict(task_payload) for task_name, task_payload in dict(tasks or {}).items()},
    }
