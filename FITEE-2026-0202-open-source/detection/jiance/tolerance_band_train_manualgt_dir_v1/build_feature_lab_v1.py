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
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .build_geo_shadow_feature_bank_v3 import (
        BASE_NUMERIC_FEATURE_COLUMNS,
        DIRECTION_EVENT_FEATURE_COLUMNS,
        DIRECTION_SUMMARY_FEATURE_COLUMNS,
        ENHANCED_SHAPE_FEATURE_COLUMNS,
        EVENT_FEATURE_COLUMNS,
        REPO_ROOT,
        SHAPE_FEATURE_COLUMNS,
        _attach_neighbor_features,
        _attach_sample_common_mode_features,
        _attach_sample_group_features,
        _attach_sample_relative_features,
        _attach_slot_context_features,
        _latest_artifact,
        _load_detection_report_features,
        _read_csv,
        _resolve_existing_path,
        _safe_float,
        _safe_int,
        _direction_sequence_column_name,
        _sequence_column_name,
        _write_csv,
        _write_json,
    )
except ImportError:
    from tolerance_band_train_manualgt_dir_v1.build_geo_shadow_feature_bank_v3 import (
        BASE_NUMERIC_FEATURE_COLUMNS,
        DIRECTION_EVENT_FEATURE_COLUMNS,
        DIRECTION_SUMMARY_FEATURE_COLUMNS,
        ENHANCED_SHAPE_FEATURE_COLUMNS,
        EVENT_FEATURE_COLUMNS,
        REPO_ROOT,
        SHAPE_FEATURE_COLUMNS,
        _attach_neighbor_features,
        _attach_sample_common_mode_features,
        _attach_sample_group_features,
        _attach_sample_relative_features,
        _attach_slot_context_features,
        _latest_artifact,
        _load_detection_report_features,
        _read_csv,
        _resolve_existing_path,
        _safe_float,
        _safe_int,
        _direction_sequence_column_name,
        _sequence_column_name,
        _write_csv,
        _write_json,
    )


DEFAULT_TRAIN_FEATURE_BANK_RAW = ""
DEFAULT_TRAIN_SUMMARY_RAW = ""
DEFAULT_TEST_DETECTION_REPORT_RAW = ""
DEFAULT_TEST_DETECTION_PER_SLOT_RAW = ""
DEFAULT_MAX_TRAIN_SAMPLES = int(str(os.getenv("DAIMA_FEATURE_LAB_MAX_TRAIN_SAMPLES", "0") or "0"))
DEFAULT_MAX_TEST_SAMPLES = int(str(os.getenv("DAIMA_FEATURE_LAB_MAX_TEST_SAMPLES", "0") or "0"))

META_COLUMNS = [
    "split_role",
    "label_available",
    "sample_uid",
    "sample_id",
    "source_sample_id",
    "target_id",
    "gt_partition",
    "is_augmented",
    "augmentation_name",
    "augmentation_source_sample_id",
    "origin_sample_uid",
    "origin_source_sample_id",
    "cv_group_uid",
    "cv_group_reason",
    "stable_id",
    "gt_abnormal",
    "geo_included",
    "dataset_subset",
    "subset_is_new",
    "matched_detection",
    "matched_detection_report",
    "formal_decision_verdict",
    "formal_decision_reason",
    "baseline_verdict",
    "baseline_reason",
    "ab_shadow_verdict",
    "ab_shadow_reason",
    "tol_fail_subtype",
    "fit_status",
    "compare_alignment_mode",
    "ab_compare_mode",
    "ab_readout_mode",
    "is_main_chain_candidate",
    "is_current_fp",
    "is_current_tp",
    "is_current_fn",
    "is_current_tn",
]


def _default_output_dir() -> Path:
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None):
        return Path(_DAIMA_CTX.run_dir).resolve()
    return (REPO_ROOT / "debug_out" / "feature_lab_v1").resolve()


def default_train_feature_bank_csv() -> Path:
    path = Path(DEFAULT_TRAIN_FEATURE_BANK_RAW)
    if path.is_file():
        return path.resolve()
    return _latest_artifact("geo_shadow_feature_bank_v3.csv")


def default_train_summary_json() -> Path:
    path = Path(DEFAULT_TRAIN_SUMMARY_RAW)
    if path.is_file():
        return path.resolve()
    return _latest_artifact("geo_shadow_feature_bank_v3_summary.json")


def default_test_detection_report_json() -> Path:
    path = Path(DEFAULT_TEST_DETECTION_REPORT_RAW)
    if path.is_file():
        return path.resolve()
    return _latest_artifact("testset_detection_report.json")


def default_test_detection_per_slot_csv() -> Path:
    path = Path(DEFAULT_TEST_DETECTION_PER_SLOT_RAW)
    if path.is_file():
        return path.resolve()
    return _latest_artifact("testset_detection_per_slot.csv")


def _sample_uid(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_uid", "") or row.get("sample_id", "") or row.get("source_sample_id", "") or "").strip()


def _normalize_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return (_sample_uid(row), int(_safe_int(row.get("stable_id", -1), default=-1)))


def _limit_sample_rows(rows: Sequence[Mapping[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if max_samples <= 0:
        return [dict(row) for row in rows]
    seen: List[str] = []
    keep: set[str] = set()
    for row in rows:
        uid = _sample_uid(row)
        if not uid or uid in keep:
            continue
        seen.append(uid)
        keep.add(uid)
        if len(seen) >= max_samples:
            break
    return [dict(row) for row in rows if _sample_uid(row) in keep]


def _load_train_rows(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows = _limit_sample_rows(_read_csv(path), max_samples)
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["split_role"] = "train_supervised"
        row["label_available"] = 1
        row["gt_abnormal"] = int(_safe_int(row.get("gt_abnormal", 0), default=0))
        row["cv_group_uid"] = str(row.get("cv_group_uid", "") or row.get("sample_uid", "")).strip()
        out.append(row)
    return out


def _test_base_rows(
    *,
    detection_per_slot_csv: Path,
    detection_report_json: Path,
    max_samples: int,
) -> Tuple[List[Dict[str, Any]], int]:
    detection_rows = _limit_sample_rows(_read_csv(detection_per_slot_csv), max_samples)
    report_map = _load_detection_report_features(detection_report_json)
    rows: List[Dict[str, Any]] = []
    missing_report = 0
    max_seq_len = 0
    max_dir_seq_len = 0
    for det_row in detection_rows:
        sample_uid = _sample_uid(det_row)
        stable_id = int(_safe_int(det_row.get("stable_id", -1), default=-1))
        if not sample_uid or stable_id < 0:
            continue
        row: Dict[str, Any] = {
            "split_role": "test_blind",
            "label_available": 0,
            "sample_uid": sample_uid,
            "sample_id": str(det_row.get("sample_id", "")).strip(),
            "source_sample_id": str(det_row.get("source_sample_id", "")).strip(),
            "target_id": str(det_row.get("target_id", "")).strip(),
            "gt_partition": str(det_row.get("gt_partition", "")).strip(),
            "is_augmented": 0,
            "augmentation_name": "",
            "augmentation_source_sample_id": "",
            "origin_sample_uid": sample_uid,
            "origin_source_sample_id": str(det_row.get("source_sample_id", "")).strip(),
            "cv_group_uid": sample_uid,
            "cv_group_reason": "test_blind_sample_uid",
            "stable_id": stable_id,
            "gt_abnormal": "",
            "geo_included": int(_safe_int(det_row.get("pred_geo_eval_eligible", 1), default=1)),
            "dataset_subset": "test_blind",
            "subset_is_new": "",
            "matched_detection": 1,
            "formal_decision_verdict": str(det_row.get("formal_decision_verdict_pre_round2", det_row.get("formal_decision_verdict", ""))).strip(),
            "formal_decision_reason": str(det_row.get("formal_decision_reason_pre_round2", det_row.get("formal_decision_reason", ""))).strip(),
            "baseline_verdict": str(det_row.get("baseline_verdict", "")).strip(),
            "baseline_reason": str(det_row.get("baseline_reason", "")).strip(),
            "ab_shadow_verdict": str(det_row.get("ab_shadow_verdict", "")).strip(),
            "ab_shadow_reason": str(det_row.get("ab_shadow_reason", "")).strip(),
            "tol_fail_subtype": str(det_row.get("tol_fail_subtype", "")).strip(),
            "fit_status": str(det_row.get("fit_status", "")).strip(),
            "compare_alignment_mode": str(det_row.get("compare_alignment_mode", "")).strip(),
            "ab_compare_mode": str(det_row.get("ab_compare_mode", "")).strip(),
            "ab_readout_mode": str(det_row.get("ab_readout_mode", "")).strip(),
            "is_main_chain_candidate": int(
                str(det_row.get("tol_fail_subtype", "")).strip() == "lateral-only"
                and str(det_row.get("ab_shadow_reason", "")).strip() == "point_readout_positive"
            ),
            "is_current_fp": "",
            "is_current_tp": "",
            "is_current_fn": "",
            "is_current_tn": "",
        }
        for col in BASE_NUMERIC_FEATURE_COLUMNS:
            row[col] = _safe_float(det_row.get(col, float("nan")))

        report_features = report_map.get((sample_uid, stable_id))
        if report_features is None:
            missing_report += 1
            row["matched_detection_report"] = 0
            sequence_values: List[float] = []
            direction_sequence_values: List[float] = []
            for col in (
                EVENT_FEATURE_COLUMNS
                + SHAPE_FEATURE_COLUMNS
                + ENHANCED_SHAPE_FEATURE_COLUMNS
                + DIRECTION_SUMMARY_FEATURE_COLUMNS
                + DIRECTION_EVENT_FEATURE_COLUMNS
            ):
                row[col] = float("nan")
        else:
            row["matched_detection_report"] = 1
            sequence_values = list(report_features.get("sequence_values", []) or [])
            direction_sequence_values = list(report_features.get("direction_sequence_values", []) or [])
            for key, value in report_features.items():
                if key not in {"sequence_values", "direction_sequence_values"}:
                    row[key] = value
        max_seq_len = max(max_seq_len, len(sequence_values))
        max_dir_seq_len = max(max_dir_seq_len, len(direction_sequence_values))
        row["_sequence_values"] = sequence_values
        row["_direction_sequence_values"] = direction_sequence_values
        rows.append(row)

    _attach_sample_relative_features(rows)
    _attach_sample_group_features(rows)
    _attach_slot_context_features(rows)
    _attach_sample_common_mode_features(rows)
    _attach_neighbor_features(rows)

    for row in rows:
        sequence_values = list(row.pop("_sequence_values", []) or [])
        direction_sequence_values = list(row.pop("_direction_sequence_values", []) or [])
        for idx in range(max_seq_len):
            row[_sequence_column_name(idx)] = float(sequence_values[idx]) if idx < len(sequence_values) else float("nan")
        for idx in range(max_dir_seq_len):
            row[_direction_sequence_column_name(idx)] = (
                float(direction_sequence_values[idx]) if idx < len(direction_sequence_values) else float("nan")
            )
    return rows, int(missing_report)


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _feature_columns_from_summary(summary: Mapping[str, Any], test_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    hgb_pool = list(summary.get("hgb_feature_pool_v3_columns", []) or [])
    train_sequence = list(summary.get("sequence_feature_columns", []) or [])
    test_sequence = sorted(
        {
            str(key)
            for row in test_rows
            for key in row.keys()
            if str(key).startswith("seq_raw_")
        }
    )
    train_direction_sequence = list(summary.get("direction_sequence_feature_columns", []) or [])
    test_direction_sequence = sorted(
        {
            str(key)
            for row in test_rows
            for key in row.keys()
            if str(key).startswith("dir_seq_margin_")
        }
    )
    sequence = sorted(set(train_sequence).union(test_sequence))
    direction_sequence = sorted(set(train_direction_sequence).union(test_direction_sequence))
    hybrid = list(dict.fromkeys(hgb_pool + sequence + direction_sequence))
    return {
        "hgb_pool_v3": hgb_pool,
        "sequence": sequence,
        "direction_sequence": direction_sequence,
        "hybrid_v3": hybrid,
        "base_numeric": list(summary.get("base_numeric_feature_columns", []) or []),
        "event": list(summary.get("event_feature_columns", []) or []),
        "shape": list(summary.get("shape_feature_columns", []) or []),
        "enhanced_shape": list(summary.get("enhanced_shape_feature_columns", []) or []),
        "direction_summary": list(summary.get("direction_summary_feature_columns", []) or []),
        "direction_event": list(summary.get("direction_event_feature_columns", []) or []),
        "relative": list(summary.get("relative_feature_columns", []) or []),
        "group": list(summary.get("group_feature_columns", []) or []),
        "slot_context": list(summary.get("slot_context_feature_columns", []) or []),
        "common_mode": list(summary.get("common_mode_feature_columns", []) or []),
        "neighbor": list(summary.get("neighbor_feature_columns", []) or []),
    }


def _fieldnames(rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> List[str]:
    ordered = list(dict.fromkeys(META_COLUMNS + list(feature_columns)))
    extras: List[str] = []
    seen = set(ordered)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                extras.append(str(key))
                seen.add(str(key))
    return ordered + extras


def run_build(
    *,
    train_feature_bank_csv: Path,
    train_summary_json: Path,
    test_detection_per_slot_csv: Path,
    test_detection_report_json: Path,
    max_train_samples: int,
    max_test_samples: int,
) -> Dict[str, Path]:
    train_summary = _load_json(train_summary_json)
    train_rows = _load_train_rows(train_feature_bank_csv, max_train_samples)
    test_rows, missing_test_report = _test_base_rows(
        detection_per_slot_csv=test_detection_per_slot_csv,
        detection_report_json=test_detection_report_json,
        max_samples=max_test_samples,
    )
    feature_columns = _feature_columns_from_summary(train_summary, test_rows)
    all_rows = train_rows + test_rows
    fields = _fieldnames(all_rows, feature_columns["hybrid_v3"])

    out_dir = _default_output_dir()
    train_csv = (out_dir / "feature_lab_train.csv").resolve()
    test_csv = (out_dir / "feature_lab_test_blind.csv").resolve()
    combined_csv = (out_dir / "feature_lab_combined.csv").resolve()
    summary_json = (out_dir / "feature_lab_summary.json").resolve()
    summary = {
        "schema_version": 1,
        "mode": "feature_lab_train_supervised_test_blind",
        "test_gt_used_for_feature_selection": False,
        "selection_scope": "train_supervised_only",
        "train": {
            "feature_bank_csv": str(train_feature_bank_csv),
            "summary_json": str(train_summary_json),
            "row_count": int(len(train_rows)),
            "sample_count": int(len({str(row.get("sample_uid", "")) for row in train_rows if str(row.get("sample_uid", ""))})),
            "positive_count": int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in train_rows)),
            "negative_count": int(sum(1 for row in train_rows if int(_safe_int(row.get("gt_abnormal", 0), default=0)) == 0)),
        },
        "test_blind": {
            "detection_per_slot_csv": str(test_detection_per_slot_csv),
            "detection_report_json": str(test_detection_report_json),
            "row_count": int(len(test_rows)),
            "sample_count": int(len({str(row.get("sample_uid", "")) for row in test_rows if str(row.get("sample_uid", ""))})),
            "missing_detection_report_count": int(missing_test_report),
        },
        "feature_columns": feature_columns,
        "outputs": {
            "train_csv": str(train_csv),
            "test_blind_csv": str(test_csv),
            "combined_csv": str(combined_csv),
            "summary_json": str(summary_json),
        },
    }

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "train_feature_bank_csv": str(train_feature_bank_csv),
                "train_summary_json": str(train_summary_json),
                "test_detection_per_slot_csv": str(test_detection_per_slot_csv),
                "test_detection_report_json": str(test_detection_report_json),
            },
            outputs=summary["outputs"],
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "max_train_samples": int(max_train_samples),
                "max_test_samples": int(max_test_samples),
                "test_gt_used_for_feature_selection": False,
                "feature_count_hybrid_v3": int(len(feature_columns["hybrid_v3"])),
            }
        )

    _write_csv(train_csv, train_rows, fields)
    _write_csv(test_csv, test_rows, fields)
    _write_csv(combined_csv, all_rows, fields)
    _write_json(summary_json, summary)
    return {
        "train_csv": train_csv,
        "test_blind_csv": test_csv,
        "combined_csv": combined_csv,
        "summary_json": summary_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature-lab tables for supervised train discovery and blind test drift checks.")
    parser.add_argument("--train_feature_bank_csv", default=DEFAULT_TRAIN_FEATURE_BANK_RAW)
    parser.add_argument("--train_summary_json", default=DEFAULT_TRAIN_SUMMARY_RAW)
    parser.add_argument("--test_detection_per_slot_csv", default=DEFAULT_TEST_DETECTION_PER_SLOT_RAW)
    parser.add_argument("--test_detection_report_json", default=DEFAULT_TEST_DETECTION_REPORT_RAW)
    parser.add_argument("--max_train_samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--max_test_samples", type=int, default=DEFAULT_MAX_TEST_SAMPLES)
    args = parser.parse_args()

    outputs = run_build(
        train_feature_bank_csv=_resolve_existing_path(args.train_feature_bank_csv, default_train_feature_bank_csv()),
        train_summary_json=_resolve_existing_path(args.train_summary_json, default_train_summary_json()),
        test_detection_per_slot_csv=_resolve_existing_path(args.test_detection_per_slot_csv, default_test_detection_per_slot_csv()),
        test_detection_report_json=_resolve_existing_path(args.test_detection_report_json, default_test_detection_report_json()),
        max_train_samples=int(args.max_train_samples),
        max_test_samples=int(args.max_test_samples),
    )
    for key in ("train_csv", "test_blind_csv", "combined_csv", "summary_json"):
        print(str(outputs[key]))


if __name__ == "__main__":
    main()
