# -*- coding: utf-8 -*-
from __future__ import annotations

_DAIMA_CTX = None
if __name__ == "__main__":
    try:
        import sys
        from pathlib import Path

        def _find_repo_root(from_path: Path) -> Path:
            from_path = from_path.resolve()
            for d in [from_path] + list(from_path.parents):
                if (d / ".release-root").is_file():
                    return d
            for d in [from_path] + list(from_path.parents):
                if (d / ".git").exists():
                    return d
            return from_path.parent

        _REPO_ROOT = _find_repo_root(Path(__file__))
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        _JIANCE_DIR = (_REPO_ROOT / "jiance").resolve()
        if str(_JIANCE_DIR) not in sys.path:
            sys.path.insert(0, str(_JIANCE_DIR))

        from daima_runtime.runlog import setup_run

        _DAIMA_CTX = setup_run(__file__)
    except Exception:
        _DAIMA_CTX = None

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from .gt_native_loader import (
        GT_ROOT_DEFAULT,
        LOGIC_GT_SAMPLE_CACHE_ROOT,
        load_gt_authoritative_by_manifest_rows,
        validate_manifest_rows_against_authoritative_gt,
    )
except ImportError:
    from gt_native_loader import (
        GT_ROOT_DEFAULT,
        LOGIC_GT_SAMPLE_CACHE_ROOT,
        load_gt_authoritative_by_manifest_rows,
        validate_manifest_rows_against_authoritative_gt,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = REPO_ROOT / "jiance" / "paper_experiment" / "tolerance_band_v1_protocol"
DEFAULT_MANIFEST_ROLE = "mixed"


def _load_manifest_rows(manifest_csv: Path) -> List[Dict[str, str]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            rows.append(
                {
                    str(k or "").replace("\ufeff", "").strip(): str(v or "").strip()
                    for k, v in raw_row.items()
                }
            )
        return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): row.get(k, "") for k in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_path_from_role(role: str) -> Path:
    role_n = str(role or "").strip().lower()
    role_to_file = {
        "mixed": PROTOCOL_DIR / "mixed_manifest.csv",
        "dev": PROTOCOL_DIR / "dev_manifest.csv",
        "test": PROTOCOL_DIR / "test_manifest.csv",
        "adjudicated": PROTOCOL_DIR / "adjudicated_manifest.csv",
    }
    path = role_to_file.get(role_n)
    if path is None:
        raise FileNotFoundError(f"Unknown manifest role: {role!r}")
    return path.resolve()


def _safe_div(num: float, den: float, *, default: float = float("nan")) -> float:
    if float(den) == 0.0:
        return float(default)
    return float(num) / float(den)


def _summarize_logic_gt_cache_usage(sample_summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    cache_hit_count = 0
    cache_enabled_count = 0
    cache_path_count = 0
    for summary in sample_summaries.values():
        summary_dict = dict(summary or {})
        status = str(summary_dict.get("logic_gt_cache_status", "") or "").strip() or "unknown"
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        if str(summary_dict.get("logic_gt_cache_hit_yes_no", "") or "").strip().upper() == "Y":
            cache_hit_count += 1
        if status != "disabled":
            cache_enabled_count += 1
        if str(summary_dict.get("logic_gt_cache_path", "") or "").strip():
            cache_path_count += 1
    cache_miss_count = int(max(cache_enabled_count - cache_hit_count, 0))
    return {
        "sample_count": int(len(sample_summaries)),
        "cache_enabled_count": int(cache_enabled_count),
        "cache_hit_count": int(cache_hit_count),
        "cache_miss_count": int(cache_miss_count),
        "cache_path_count": int(cache_path_count),
        "cache_hit_rate": float(_safe_div(cache_hit_count, cache_enabled_count, default=0.0)),
        "cache_status_counts": dict(sorted(status_counts.items(), key=lambda item: item[0])),
    }


def _slot_rows_from_map(slot_map: Mapping[Any, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(row)
        for _key, row in sorted(
            ((key, dict(value)) for key, value in slot_map.items()),
            key=lambda item: (str(item[0][0]), int(item[0][1])),
        )
    ]


def _normalize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    return value


def _normalize_csv_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        out.append({str(k): _normalize_csv_value(v) for k, v in row_dict.items()})
    return out


def _fieldnames_from_rows(rows: Sequence[Mapping[str, Any]], fallback: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for row in rows:
        for key in row.keys():
            key_n = str(key)
            if key_n not in seen:
                seen[key_n] = None
    return list(seen.keys()) if seen else list(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute slot-level logic/geo GT artifacts for authoritative evaluation.")
    parser.add_argument("--manifest_csv", default="", help="Optional manifest CSV override")
    parser.add_argument(
        "--manifest_role",
        default=str(DEFAULT_MANIFEST_ROLE),
        choices=["mixed", "dev", "test", "adjudicated"],
        help="Protocol manifest role when --manifest_csv is not given",
    )
    parser.add_argument("--cache_root", default="", help="Optional prepared GT cache root override")
    parser.add_argument("--gt_root", default="", help="Optional authoritative GT root override")
    args = parser.parse_args()

    manifest_path = Path(str(args.manifest_csv).strip()).resolve() if str(args.manifest_csv).strip() else _manifest_path_from_role(str(args.manifest_role))
    gt_root = Path(str(args.gt_root).strip()).resolve() if str(args.gt_root).strip() else Path(GT_ROOT_DEFAULT).resolve()
    cache_root = Path(str(args.cache_root).strip()).resolve() if str(args.cache_root).strip() else Path(LOGIC_GT_SAMPLE_CACHE_ROOT).resolve()
    manifest_rows = validate_manifest_rows_against_authoritative_gt(_load_manifest_rows(manifest_path), gt_root)
    sample_ids = [
        str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
        for row in manifest_rows
        if str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
    ]

    t0 = time.perf_counter()
    gt_slot_map, gt_skipped, sample_summaries, _point_support_by_key = load_gt_authoritative_by_manifest_rows(
        gt_root,
        manifest_rows,
        include_point_support=False,
        use_sample_cache=True,
        cache_root=cache_root,
        require_sample_cache_hit=False,
        write_sample_cache_on_miss=True,
    )
    elapsed_sec = float(time.perf_counter() - t0)
    cache_stats = _summarize_logic_gt_cache_usage(sample_summaries)

    out_dir = (
        Path(_DAIMA_CTX.run_dir)
        if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None)
        else (REPO_ROOT / "debug_out" / "manual_prepare_logic_geo_gt")
    )
    per_slot_rows = _normalize_csv_rows(_slot_rows_from_map(gt_slot_map))
    sample_summary_rows = _normalize_csv_rows(
        [dict(value) for _key, value in sorted(sample_summaries.items(), key=lambda item: str(item[0]))]
    )
    skipped_rows = _normalize_csv_rows(
        [dict(row) for row in sorted(list(gt_skipped or []), key=lambda row: (str(row.get("sample_uid", "")), str(row.get("reason", ""))))]
    )

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "manifest_path": str(manifest_path),
                "gt_root": str(gt_root),
                "cache_root": str(cache_root),
            },
            outputs={
                "prepared_logic_geo_gt_per_slot_csv": str(out_dir / "prepared_logic_geo_gt_per_slot.csv"),
                "prepared_logic_geo_gt_sample_summary_csv": str(out_dir / "prepared_logic_geo_gt_sample_summary.csv"),
                "prepared_logic_geo_gt_skipped_csv": str(out_dir / "prepared_logic_geo_gt_skipped.csv"),
                "prepared_logic_geo_gt_report_json": str(out_dir / "prepared_logic_geo_gt_report.json"),
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "manifest_role": str(args.manifest_role),
                "manifest_csv": str(manifest_path),
                "gt_root": str(gt_root),
                "cache_root": str(cache_root),
            }
        )

    _write_csv(
        out_dir / "prepared_logic_geo_gt_per_slot.csv",
        per_slot_rows,
        _fieldnames_from_rows(per_slot_rows, ["sample_uid", "sample_id", "stable_id", "gt_missing", "gt_abnormal", "gt_geo_eligible"]),
    )
    _write_csv(
        out_dir / "prepared_logic_geo_gt_sample_summary.csv",
        sample_summary_rows,
        _fieldnames_from_rows(sample_summary_rows, ["sample_uid", "sample_id", "logic_gt_cache_hit_yes_no", "logic_gt_cache_path"]),
    )
    _write_csv(
        out_dir / "prepared_logic_geo_gt_skipped.csv",
        skipped_rows,
        _fieldnames_from_rows(skipped_rows, ["sample_uid", "sample_id", "reason", "error"]),
    )

    report = {
        "manifest_path": str(manifest_path),
        "gt_root": str(gt_root),
        "cache_root": str(cache_root),
        "sample_ids_requested": list(sample_ids),
        "sample_ids_prepared": sorted({str(key[0]) for key in gt_slot_map.keys()}),
        "gt_skipped": gt_skipped,
        "elapsed_sec": float(elapsed_sec),
        "cache_stats": dict(cache_stats),
    }
    _write_json(out_dir / "prepared_logic_geo_gt_report.json", report)

    print(f"[INFO] Prepared logic/geo GT under: {out_dir}")
    print(
        "[INFO] GT prepare summary: "
        f"elapsed={float(elapsed_sec):.2f}s "
        f"| sample_count={int(cache_stats.get('sample_count', 0))} "
        f"| cache_hit={int(cache_stats.get('cache_hit_count', 0))} "
        f"| cache_miss={int(cache_stats.get('cache_miss_count', 0))}"
    )


if __name__ == "__main__":
    main()
