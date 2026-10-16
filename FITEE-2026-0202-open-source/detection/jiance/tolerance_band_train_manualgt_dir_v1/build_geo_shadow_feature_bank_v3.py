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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from baseline.common import compute_positive_event_features_v1

try:
    from .build_offline_feature_bank_v1 import (
        DEFAULT_AUTHORITATIVE_GEO_RAW,
        DEFAULT_DETECTION_PER_SLOT_RAW,
        REPO_ROOT,
        _resolve_existing_path,
        _safe_float,
        _safe_int,
        _write_csv,
        _write_json,
        build_feature_bank_rows,
        default_authoritative_geo_csv,
        default_detection_per_slot_csv,
    )
except ImportError:
    from build_offline_feature_bank_v1 import (
        DEFAULT_AUTHORITATIVE_GEO_RAW,
        DEFAULT_DETECTION_PER_SLOT_RAW,
        REPO_ROOT,
        _resolve_existing_path,
        _safe_float,
        _safe_int,
        _write_csv,
        _write_json,
        build_feature_bank_rows,
        default_authoritative_geo_csv,
        default_detection_per_slot_csv,
    )


DEFAULT_DETECTION_REPORT_RAW = ""
DEFAULT_AUTHORITATIVE_GEO_V2_RAW = ""
DEFAULT_DETECTION_PER_SLOT_V2_RAW = ""
DEFAULT_COMPARE_INVENTORY_RAW = ""

SEQUENCE_PREFIX = "seq_raw_"
DIR_SEQUENCE_PREFIX = "dir_seq_margin_"
SEQUENCE_INDEX_WIDTH = 3

BASE_NUMERIC_FEATURE_COLUMNS = [
    "needle_score_raw",
    "point_score_selected_raw_max",
    "pred_max_run_span_mm",
    "pred_dominant_run_frac",
    "pred_run_count",
    "pred_pos_bins",
    "pred_short_run_frac",
    "pred_short_run_bins",
    "pred_max_run_bins",
    "pair_dist_u",
    "ab_station_step_mm",
]

EVENT_FEATURE_COLUMNS = [
    "positive_event_count",
    "positive_total_bins",
    "positive_total_support_mm",
    "positive_total_area_mm",
    "dominant_event_rank",
    "dominant_event_start_idx",
    "dominant_event_end_idx",
    "dominant_event_bins",
    "dominant_event_span_mm",
    "dominant_event_support_mm",
    "dominant_event_area_mm",
    "dominant_event_peak_score",
    "dominant_event_mean_score",
    "dominant_event_start_score",
    "dominant_event_end_score",
    "dominant_event_monotone_forward_frac",
    "dominant_event_monotone_reverse_frac",
    "dominant_event_monotone_best_frac",
    "dominant_event_return_frac",
    "dominant_event_peak_at_edge",
]

SHAPE_FEATURE_COLUMNS = [
    "sequence_len",
    "sequence_finite_count",
    "sequence_finite_frac",
    "sequence_positive_frac",
    "sequence_mean",
    "sequence_std",
    "sequence_q50",
    "sequence_q75",
    "sequence_q90",
    "sequence_q95",
    "sequence_q99",
    "sequence_peak_idx",
    "sequence_peak_pos_frac",
    "sequence_peak_edge_distance_frac",
    "sequence_first_pos_idx",
    "sequence_last_pos_idx",
    "sequence_first_pos_frac",
    "sequence_last_pos_frac",
    "sequence_pre_peak_area_mm",
    "sequence_post_peak_area_mm",
    "sequence_post_peak_area_frac",
    "sequence_non_dominant_area_mm",
    "sequence_non_dominant_support_mm",
    "sequence_dominant_area_frac",
    "sequence_dominant_support_frac",
]

RELATIVE_SOURCE_COLUMNS = [
    "needle_score_raw",
    "point_score_selected_raw_max",
    "pred_max_run_span_mm",
    "dominant_event_support_mm",
    "dominant_event_area_mm",
    "dominant_event_peak_score",
    "positive_total_area_mm",
    "max_tangent_ratio",
    "tangent_decision_span_mm",
    "direction_dominant_peak_ratio",
    "direction_total_area_mm",
]

GENERALIZED_GROUP_SOURCE_COLUMNS = [
    "needle_score_raw",
    "positive_total_bins",
    "dominant_event_support_mm",
    "dominant_event_area_mm",
    "sequence_positive_frac",
    "max_tangent_ratio",
    "tangent_decision_points",
    "direction_event_count",
]

ENHANCED_SHAPE_FEATURE_COLUMNS = [
    "sequence_q99_minus_q90",
    "sequence_q95_minus_q50",
    "sequence_peak_minus_q90",
    "sequence_positive_center_of_mass_frac",
    "sequence_positive_spread_frac",
    "sequence_positive_area_first_third_mm",
    "sequence_positive_area_mid_third_mm",
    "sequence_positive_area_last_third_mm",
    "sequence_positive_area_first_third_frac",
    "sequence_positive_area_mid_third_frac",
    "sequence_positive_area_last_third_frac",
    "sequence_positive_edge10_frac",
    "sequence_positive_center50_frac",
    "sequence_abs_slope_mean",
    "sequence_abs_slope_q90",
    "sequence_abs_slope_max",
    "sequence_curvature_abs_mean",
    "sequence_curvature_abs_q90",
    "sequence_curvature_abs_max",
]

DIRECTION_SUMMARY_FEATURE_COLUMNS = [
    "direction_model_ok",
    "tangent_status_fail",
    "tangent_status_pass",
    "max_tangent_deg",
    "max_tangent_ratio",
    "tangent_margin_max",
    "tangent_margin_q95",
    "tangent_ratio_q95",
    "tangent_ratio_q99",
    "tangent_score_max",
    "tangent_positive_frac",
    "tangent_positive_area_mm",
    "tangent_over_cone_points",
    "tangent_over_cone_span_mm",
    "tangent_window_over_ratio_max",
    "tangent_window_over_span_mm_max",
    "tangent_fail_window_count",
    "tangent_fail_segment_count",
    "tangent_decision_points",
    "tangent_decision_frac",
    "tangent_decision_span_mm",
    "tangent_decision_area_mm",
    "tangent_decision_peak_ratio",
    "tangent_decision_mean_over_ratio",
]

DIRECTION_EVENT_FEATURE_COLUMNS = [
    "direction_event_count",
    "direction_segment_ge3_count",
    "direction_total_bins",
    "direction_total_support_mm",
    "direction_total_area_mm",
    "direction_dominant_start_idx",
    "direction_dominant_end_idx",
    "direction_dominant_bins",
    "direction_dominant_span_mm",
    "direction_dominant_peak_ratio",
    "direction_dominant_mean_over_ratio",
    "direction_dominant_window_over_ratio",
    "direction_dominant_window_over_span_mm",
    "direction_dominant_overlap_pos_bins",
    "direction_dominant_overlap_ratio",
    "direction_dominant_local_pos_peak_margin",
    "direction_dominant_local_pos_near_miss",
    "direction_dominant_is_isolated_spike",
]

SLOT_CONTEXT_FEATURE_COLUMNS = [
    "stable_id_numeric",
    "stable_id_norm",
    "stable_edge_distance",
    "stable_edge_distance_norm",
    "stable_left_half",
    "stable_right_half",
]

COMMON_MODE_SOURCE_COLUMNS = [
    "needle_score_raw",
    "positive_total_bins",
    "dominant_event_area_mm",
    "dominant_event_support_mm",
    "pred_max_run_span_mm",
    "sequence_positive_frac",
    "max_tangent_ratio",
    "tangent_decision_points",
    "direction_total_area_mm",
]

NEIGHBOR_SOURCE_COLUMNS = [
    "needle_score_raw",
    "positive_total_bins",
    "dominant_event_area_mm",
    "dominant_event_support_mm",
    "pred_max_run_span_mm",
    "sequence_positive_frac",
    "max_tangent_ratio",
    "tangent_decision_points",
    "direction_total_area_mm",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
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


def _latest_artifact(filename: str) -> Path:
    runs_root = REPO_ROOT / "debug_out" / "runs"
    cands = sorted(runs_root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No {filename} found under: {runs_root}")
    return cands[0].resolve()


def default_detection_report_json() -> Path:
    path = Path(DEFAULT_DETECTION_REPORT_RAW)
    if path.is_file():
        return path.resolve()
    try:
        return _latest_artifact("train_combined_detection_report.json")
    except FileNotFoundError:
        return _latest_artifact("testset_detection_report.json")


def default_authoritative_geo_csv_v2() -> Path:
    path = Path(DEFAULT_AUTHORITATIVE_GEO_V2_RAW)
    if path.is_file():
        return path.resolve()
    try:
        return _latest_artifact("manual_authoritative_geo_per_slot.csv")
    except FileNotFoundError:
        return _latest_artifact("authoritative_geo_per_slot.csv")


def default_detection_per_slot_csv_v2() -> Path:
    path = Path(DEFAULT_DETECTION_PER_SLOT_V2_RAW)
    if path.is_file():
        return path.resolve()
    try:
        return _latest_artifact("train_combined_detection_per_slot.csv")
    except FileNotFoundError:
        return _latest_artifact("testset_detection_per_slot.csv")


def default_compare_inventory_csv() -> Path:
    path = Path(DEFAULT_COMPARE_INVENTORY_RAW)
    if path.is_file():
        return path.resolve()
    try:
        return _latest_artifact("train_manual_compare_inventory.csv")
    except FileNotFoundError:
        return _latest_artifact("compare_inventory.csv")


def _default_output_dir() -> Path:
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None):
        return Path(_DAIMA_CTX.run_dir).resolve()
    return (REPO_ROOT / "debug_out" / "manual_geo_shadow_feature_bank_v3").resolve()


def _normalize_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return (
        str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip(),
        int(_safe_int(row.get("stable_id", -1), default=-1)),
    )


def _safe_sequence(arr: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for item in list(arr or []):
        try:
            out.append(float(item))
        except Exception:
            out.append(float("nan"))
    return out


def _sequence_column_name(idx: int) -> str:
    return f"{SEQUENCE_PREFIX}{idx:0{SEQUENCE_INDEX_WIDTH}d}"


def _direction_sequence_column_name(idx: int) -> str:
    return f"{DIR_SEQUENCE_PREFIX}{idx:0{SEQUENCE_INDEX_WIDTH}d}"


def _finite_max(arr: np.ndarray, default: float = float("nan")) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size > 0 else float(default)


def _finite_quantile_or_nan(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.quantile(finite, q)) if finite.size > 0 else float("nan")


def _bool_segments(mask_values: Sequence[Any]) -> List[Dict[str, int]]:
    mask = np.asarray(mask_values, dtype=bool).reshape(-1)
    segments: List[Dict[str, int]] = []
    start = -1
    for idx, flag in enumerate(mask):
        if bool(flag) and start < 0:
            start = idx
        elif not bool(flag) and start >= 0:
            segments.append({"start_idx": int(start), "end_idx": int(idx - 1)})
            start = -1
    if start >= 0:
        segments.append({"start_idx": int(start), "end_idx": int(len(mask) - 1)})
    return segments


def _compute_direction_features(
    *,
    tol_metrics: Mapping[str, Any],
    point_score_arr: Sequence[Any],
    station_step_mm: Any,
) -> Tuple[Dict[str, Any], List[float]]:
    step_mm = float(_safe_float(station_step_mm, default=float("nan")))
    if not np.isfinite(step_mm) or step_mm <= 0.0:
        step_mm = 0.2

    ratio_arr = np.asarray(_safe_sequence(list(tol_metrics.get("tangent_ratio_arr", []) or [])), dtype=np.float64)
    margin_values = _safe_sequence(list(tol_metrics.get("tangent_margin_arr", []) or []))
    if not margin_values and ratio_arr.size > 0:
        margin_values = [float(v - 1.0) if np.isfinite(v) else float("nan") for v in ratio_arr]
    margin_arr = np.asarray(margin_values, dtype=np.float64)
    score_arr = np.asarray(_safe_sequence(list(tol_metrics.get("tangent_score_arr", []) or [])), dtype=np.float64)
    window_ratio_arr = np.asarray(
        _safe_sequence(list(tol_metrics.get("tangent_window_over_ratio_arr", []) or [])),
        dtype=np.float64,
    )
    window_span_arr = np.asarray(
        _safe_sequence(list(tol_metrics.get("tangent_window_over_span_mm_arr", []) or [])),
        dtype=np.float64,
    )
    decision_mask = np.asarray(list(tol_metrics.get("tangent_decision_mask_arr", []) or []), dtype=bool).reshape(-1)
    point_arr = np.asarray(_safe_sequence(point_score_arr), dtype=np.float64)

    n = int(max(len(ratio_arr), len(margin_arr), len(score_arr), len(decision_mask), len(point_arr), 0))
    if n <= 0:
        base = {col: float("nan") for col in DIRECTION_SUMMARY_FEATURE_COLUMNS + DIRECTION_EVENT_FEATURE_COLUMNS}
        base.update(
            {
                "direction_model_ok": 0,
                "tangent_status_fail": 0,
                "tangent_status_pass": 0,
                "direction_event_count": 0,
                "direction_segment_ge3_count": 0,
                "direction_total_bins": 0,
                "direction_total_support_mm": 0.0,
                "direction_total_area_mm": 0.0,
            }
        )
        return base, []

    def _pad_float(arr: np.ndarray, fill: float = float("nan")) -> np.ndarray:
        out = np.full((n,), fill, dtype=np.float64)
        take = min(n, int(arr.size))
        if take > 0:
            out[:take] = arr[:take]
        return out

    def _pad_bool(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((n,), dtype=bool)
        take = min(n, int(arr.size))
        if take > 0:
            out[:take] = arr[:take]
        return out

    ratio_arr = _pad_float(ratio_arr)
    margin_arr = _pad_float(margin_arr)
    score_arr = _pad_float(score_arr, 0.0)
    window_ratio_arr = _pad_float(window_ratio_arr, 0.0)
    window_span_arr = _pad_float(window_span_arr, 0.0)
    decision_mask = _pad_bool(decision_mask)
    point_arr = _pad_float(point_arr)

    status = str(tol_metrics.get("direction_model_status", "") or "").strip().lower()
    tangent_status = str(tol_metrics.get("tangent_status", "") or "").strip().lower()
    positive_mask = np.isfinite(margin_arr) & (margin_arr > 0.0)
    decision_margin = np.where(decision_mask & np.isfinite(margin_arr), np.clip(margin_arr, 0.0, None), 0.0)
    decision_ratio = ratio_arr[decision_mask & np.isfinite(ratio_arr)]

    source_segments = [dict(seg) for seg in list(tol_metrics.get("tangent_fail_segments", []) or []) if isinstance(seg, Mapping)]
    if not source_segments:
        source_segments = _bool_segments(decision_mask)

    segments: List[Dict[str, Any]] = []
    for seg in source_segments:
        start_idx = int(_safe_int(seg.get("start_idx", seg.get("idx_start", -1)), default=-1))
        end_idx = int(_safe_int(seg.get("end_idx", seg.get("idx_end", -1)), default=-1))
        if start_idx < 0 or end_idx < start_idx or start_idx >= n:
            continue
        end_idx = min(end_idx, n - 1)
        seg_ratio = ratio_arr[start_idx : end_idx + 1]
        seg_margin = margin_arr[start_idx : end_idx + 1]
        seg_win_ratio = window_ratio_arr[start_idx : end_idx + 1]
        seg_win_span = window_span_arr[start_idx : end_idx + 1]
        seg_pos = point_arr[start_idx : end_idx + 1]
        pos_mask = np.isfinite(seg_pos) & (seg_pos > 0.0)
        bins = int(end_idx - start_idx + 1)
        span_mm = float(max(0, bins - 1) * step_mm)
        peak_ratio = _finite_max(seg_ratio)
        mean_over = float(np.nanmean(np.clip(seg_ratio - 1.0, 0.0, None))) if np.any(np.isfinite(seg_ratio)) else float("nan")
        local_peak = _finite_max(seg_pos)
        segments.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "bins": int(bins),
                "span_mm": float(span_mm),
                "peak_ratio": float(peak_ratio),
                "mean_over_ratio": float(mean_over),
                "area_mm": float(np.nansum(np.clip(seg_margin, 0.0, None)) * step_mm),
                "window_over_ratio": _finite_max(seg_win_ratio),
                "window_over_span_mm": _finite_max(seg_win_span),
                "overlap_pos_bins": int(np.count_nonzero(pos_mask)),
                "overlap_ratio": float(np.mean(pos_mask.astype(np.float64))) if pos_mask.size > 0 else 0.0,
                "local_pos_peak_margin": float(local_peak),
                "local_pos_near_miss": int(np.isfinite(local_peak) and local_peak >= -0.05),
                "is_isolated_spike": int(bins < 3),
            }
        )

    dominant = (
        sorted(
            segments,
            key=lambda x: (
                float(_safe_float(x.get("span_mm", float("nan")))),
                float(_safe_float(x.get("peak_ratio", float("nan")))),
                float(_safe_float(x.get("mean_over_ratio", float("nan")))),
                int(_safe_int(x.get("bins", 0), default=0)),
            ),
            reverse=True,
        )[0]
        if segments
        else {}
    )

    features: Dict[str, Any] = {
        "direction_model_ok": int(status == "ok"),
        "tangent_status_fail": int(tangent_status == "fail"),
        "tangent_status_pass": int(tangent_status == "pass"),
        "max_tangent_deg": _safe_float(tol_metrics.get("max_tangent_deg", float("nan"))),
        "max_tangent_ratio": _safe_float(tol_metrics.get("max_tangent_ratio", _finite_max(ratio_arr))),
        "tangent_margin_max": _finite_max(margin_arr),
        "tangent_margin_q95": _finite_quantile_or_nan(margin_arr, 0.95),
        "tangent_ratio_q95": _finite_quantile_or_nan(ratio_arr, 0.95),
        "tangent_ratio_q99": _finite_quantile_or_nan(ratio_arr, 0.99),
        "tangent_score_max": _finite_max(score_arr, default=0.0),
        "tangent_positive_frac": float(np.count_nonzero(positive_mask) / n) if n > 0 else 0.0,
        "tangent_positive_area_mm": float(np.nansum(np.clip(margin_arr, 0.0, None)) * step_mm),
        "tangent_over_cone_points": int(_safe_int(tol_metrics.get("tangent_over_cone_points", np.count_nonzero(positive_mask)), default=0)),
        "tangent_over_cone_span_mm": _safe_float(tol_metrics.get("tangent_over_cone_span_mm", float("nan"))),
        "tangent_window_over_ratio_max": _safe_float(tol_metrics.get("tangent_window_over_ratio_max", _finite_max(window_ratio_arr))),
        "tangent_window_over_span_mm_max": _safe_float(tol_metrics.get("tangent_window_over_span_mm_max", _finite_max(window_span_arr))),
        "tangent_fail_window_count": int(_safe_int(tol_metrics.get("tangent_fail_window_count", 0), default=0)),
        "tangent_fail_segment_count": int(_safe_int(tol_metrics.get("tangent_fail_segment_count", len(segments)), default=0)),
        "tangent_decision_points": int(_safe_int(tol_metrics.get("tangent_decision_points", np.count_nonzero(decision_mask)), default=0)),
        "tangent_decision_frac": float(np.count_nonzero(decision_mask) / n) if n > 0 else 0.0,
        "tangent_decision_span_mm": _safe_float(tol_metrics.get("tangent_decision_span_mm", float("nan"))),
        "tangent_decision_area_mm": float(np.sum(decision_margin) * step_mm),
        "tangent_decision_peak_ratio": float(np.max(decision_ratio)) if decision_ratio.size > 0 else float("nan"),
        "tangent_decision_mean_over_ratio": float(np.mean(np.clip(decision_ratio - 1.0, 0.0, None))) if decision_ratio.size > 0 else float("nan"),
        "direction_event_count": int(len(segments)),
        "direction_segment_ge3_count": int(sum(int(seg.get("bins", 0)) >= 3 for seg in segments)),
        "direction_total_bins": int(sum(int(seg.get("bins", 0)) for seg in segments)),
        "direction_total_support_mm": float(sum(float(_safe_float(seg.get("span_mm", 0.0), default=0.0)) for seg in segments)),
        "direction_total_area_mm": float(sum(float(_safe_float(seg.get("area_mm", 0.0), default=0.0)) for seg in segments)),
        "direction_dominant_start_idx": int(_safe_int(dominant.get("start_idx", -1), default=-1)) if dominant else -1,
        "direction_dominant_end_idx": int(_safe_int(dominant.get("end_idx", -1), default=-1)) if dominant else -1,
        "direction_dominant_bins": int(_safe_int(dominant.get("bins", 0), default=0)) if dominant else 0,
        "direction_dominant_span_mm": _safe_float(dominant.get("span_mm", float("nan"))) if dominant else float("nan"),
        "direction_dominant_peak_ratio": _safe_float(dominant.get("peak_ratio", float("nan"))) if dominant else float("nan"),
        "direction_dominant_mean_over_ratio": _safe_float(dominant.get("mean_over_ratio", float("nan"))) if dominant else float("nan"),
        "direction_dominant_window_over_ratio": _safe_float(dominant.get("window_over_ratio", float("nan"))) if dominant else float("nan"),
        "direction_dominant_window_over_span_mm": _safe_float(dominant.get("window_over_span_mm", float("nan"))) if dominant else float("nan"),
        "direction_dominant_overlap_pos_bins": int(_safe_int(dominant.get("overlap_pos_bins", 0), default=0)) if dominant else 0,
        "direction_dominant_overlap_ratio": _safe_float(dominant.get("overlap_ratio", float("nan"))) if dominant else float("nan"),
        "direction_dominant_local_pos_peak_margin": _safe_float(dominant.get("local_pos_peak_margin", float("nan"))) if dominant else float("nan"),
        "direction_dominant_local_pos_near_miss": int(_safe_int(dominant.get("local_pos_near_miss", 0), default=0)) if dominant else 0,
        "direction_dominant_is_isolated_spike": int(_safe_int(dominant.get("is_isolated_spike", 0), default=0)) if dominant else 0,
    }
    return features, [float(v) for v in margin_arr.tolist()]


def _sample_old36_ids(compare_inventory_csv: Path) -> set[str]:
    rows = _read_csv(compare_inventory_csv)
    return {
        str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
        for row in rows
        if str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
    }


def _compute_sequence_shape_features(
    arr_values: Sequence[Any],
    *,
    station_step_mm: Any,
    event_features: Mapping[str, Any],
) -> Dict[str, Any]:
    arr = np.asarray(list(arr_values or []), dtype=np.float64).reshape(-1)
    seq_len = int(arr.size)
    finite_mask = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite_mask))
    finite_vals = arr[finite_mask]
    step_mm_f = float(_safe_float(station_step_mm, default=float("nan")))
    if not np.isfinite(step_mm_f) or step_mm_f <= 0.0:
        step_mm_f = 0.2
    pos_mask = finite_mask & (arr > 0.0)
    pos_idx = np.flatnonzero(pos_mask)
    peak_idx = int(np.nanargmax(arr)) if finite_count > 0 else -1
    denom = float(max(seq_len - 1, 1))
    first_pos_idx = int(pos_idx[0]) if pos_idx.size > 0 else -1
    last_pos_idx = int(pos_idx[-1]) if pos_idx.size > 0 else -1
    peak_pos_frac = float(peak_idx / denom) if peak_idx >= 0 else float("nan")
    peak_edge_distance_frac = (
        float(min(peak_idx, max(seq_len - 1 - peak_idx, 0)) / denom)
        if peak_idx >= 0 and seq_len > 0
        else float("nan")
    )
    first_pos_frac = float(first_pos_idx / denom) if first_pos_idx >= 0 else float("nan")
    last_pos_frac = float(last_pos_idx / denom) if last_pos_idx >= 0 else float("nan")

    if peak_idx >= 0:
        pre_peak = arr[: peak_idx + 1]
        post_peak = arr[peak_idx:]
    else:
        pre_peak = np.asarray([], dtype=np.float64)
        post_peak = np.asarray([], dtype=np.float64)
    pre_peak_area_mm = float(np.nansum(np.clip(pre_peak, 0.0, None)) * step_mm_f) if pre_peak.size > 0 else 0.0
    post_peak_area_mm = float(np.nansum(np.clip(post_peak, 0.0, None)) * step_mm_f) if post_peak.size > 0 else 0.0
    total_area_mm = float(_safe_float(event_features.get("positive_total_area_mm", 0.0), default=0.0))
    dominant_area_mm = float(_safe_float(event_features.get("dominant_event_area_mm", 0.0), default=0.0))
    total_support_mm = float(_safe_float(event_features.get("positive_total_support_mm", 0.0), default=0.0))
    dominant_support_mm = float(_safe_float(event_features.get("dominant_event_support_mm", 0.0), default=0.0))

    def _quantile(q: float) -> float:
        if finite_vals.size <= 0:
            return float("nan")
        return float(np.quantile(finite_vals, q))

    return {
        "sequence_len": int(seq_len),
        "sequence_finite_count": int(finite_count),
        "sequence_finite_frac": float(finite_count / seq_len) if seq_len > 0 else 0.0,
        "sequence_positive_frac": float(pos_idx.size / seq_len) if seq_len > 0 else 0.0,
        "sequence_mean": float(np.mean(finite_vals)) if finite_vals.size > 0 else float("nan"),
        "sequence_std": float(np.std(finite_vals)) if finite_vals.size > 0 else float("nan"),
        "sequence_q50": _quantile(0.50),
        "sequence_q75": _quantile(0.75),
        "sequence_q90": _quantile(0.90),
        "sequence_q95": _quantile(0.95),
        "sequence_q99": _quantile(0.99),
        "sequence_peak_idx": int(peak_idx),
        "sequence_peak_pos_frac": float(peak_pos_frac),
        "sequence_peak_edge_distance_frac": float(peak_edge_distance_frac),
        "sequence_first_pos_idx": int(first_pos_idx),
        "sequence_last_pos_idx": int(last_pos_idx),
        "sequence_first_pos_frac": float(first_pos_frac),
        "sequence_last_pos_frac": float(last_pos_frac),
        "sequence_pre_peak_area_mm": float(pre_peak_area_mm),
        "sequence_post_peak_area_mm": float(post_peak_area_mm),
        "sequence_post_peak_area_frac": float(post_peak_area_mm / total_area_mm) if total_area_mm > 1e-12 else 0.0,
        "sequence_non_dominant_area_mm": float(max(total_area_mm - dominant_area_mm, 0.0)),
        "sequence_non_dominant_support_mm": float(max(total_support_mm - dominant_support_mm, 0.0)),
        "sequence_dominant_area_frac": float(dominant_area_mm / total_area_mm) if total_area_mm > 1e-12 else 0.0,
        "sequence_dominant_support_frac": float(dominant_support_mm / total_support_mm) if total_support_mm > 1e-12 else 0.0,
    }


def _finite_quantile(values: np.ndarray, q: float, default: float = float("nan")) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 0:
        return float(default)
    return float(np.quantile(finite, q))


def _compute_enhanced_sequence_shape_features(
    arr_values: Sequence[Any],
    *,
    station_step_mm: Any,
) -> Dict[str, Any]:
    arr = np.asarray(list(arr_values or []), dtype=np.float64).reshape(-1)
    seq_len = int(arr.size)
    step_mm_f = float(_safe_float(station_step_mm, default=float("nan")))
    if not np.isfinite(step_mm_f) or step_mm_f <= 0.0:
        step_mm_f = 0.2
    if seq_len <= 0:
        return {name: float("nan") for name in ENHANCED_SHAPE_FEATURE_COLUMNS}

    finite_mask = np.isfinite(arr)
    safe_arr = np.where(finite_mask, arr, 0.0)
    positive = np.clip(safe_arr, 0.0, None)
    positive_total_area = float(np.sum(positive) * step_mm_f)
    q50 = _finite_quantile(arr, 0.50)
    q90 = _finite_quantile(arr, 0.90)
    q95 = _finite_quantile(arr, 0.95)
    q99 = _finite_quantile(arr, 0.99)
    peak = float(np.nanmax(arr)) if np.any(finite_mask) else float("nan")

    idx = np.arange(seq_len, dtype=np.float64)
    denom = float(max(seq_len - 1, 1))
    weight_sum = float(np.sum(positive))
    if weight_sum > 1e-12:
        center = float(np.sum(idx * positive) / weight_sum / denom)
        spread = float(np.sqrt(np.sum(((idx / denom) - center) ** 2 * positive) / weight_sum))
    else:
        center = float("nan")
        spread = float("nan")

    thirds = np.array_split(np.arange(seq_len), 3)
    third_areas = [float(np.sum(positive[part]) * step_mm_f) if part.size > 0 else 0.0 for part in thirds]
    edge_n = max(int(round(seq_len * 0.10)), 1)
    center_lo = int(round(seq_len * 0.25))
    center_hi = int(round(seq_len * 0.75))
    positive_bins = int(np.count_nonzero(positive > 0.0))
    edge_bins = int(np.count_nonzero(positive[:edge_n] > 0.0) + np.count_nonzero(positive[max(seq_len - edge_n, 0) :] > 0.0))
    center_bins = int(np.count_nonzero(positive[center_lo:center_hi] > 0.0))

    first_diff = np.diff(safe_arr) / step_mm_f if seq_len >= 2 else np.asarray([], dtype=np.float64)
    second_diff = np.diff(safe_arr, n=2) / (step_mm_f * step_mm_f) if seq_len >= 3 else np.asarray([], dtype=np.float64)
    abs_slope = np.abs(first_diff)
    abs_curve = np.abs(second_diff)

    return {
        "sequence_q99_minus_q90": float(q99 - q90) if np.isfinite(q99) and np.isfinite(q90) else float("nan"),
        "sequence_q95_minus_q50": float(q95 - q50) if np.isfinite(q95) and np.isfinite(q50) else float("nan"),
        "sequence_peak_minus_q90": float(peak - q90) if np.isfinite(peak) and np.isfinite(q90) else float("nan"),
        "sequence_positive_center_of_mass_frac": float(center),
        "sequence_positive_spread_frac": float(spread),
        "sequence_positive_area_first_third_mm": float(third_areas[0]),
        "sequence_positive_area_mid_third_mm": float(third_areas[1]),
        "sequence_positive_area_last_third_mm": float(third_areas[2]),
        "sequence_positive_area_first_third_frac": float(third_areas[0] / positive_total_area) if positive_total_area > 1e-12 else 0.0,
        "sequence_positive_area_mid_third_frac": float(third_areas[1] / positive_total_area) if positive_total_area > 1e-12 else 0.0,
        "sequence_positive_area_last_third_frac": float(third_areas[2] / positive_total_area) if positive_total_area > 1e-12 else 0.0,
        "sequence_positive_edge10_frac": float(edge_bins / positive_bins) if positive_bins > 0 else 0.0,
        "sequence_positive_center50_frac": float(center_bins / positive_bins) if positive_bins > 0 else 0.0,
        "sequence_abs_slope_mean": float(np.mean(abs_slope)) if abs_slope.size > 0 else 0.0,
        "sequence_abs_slope_q90": _finite_quantile(abs_slope, 0.90, default=0.0),
        "sequence_abs_slope_max": float(np.max(abs_slope)) if abs_slope.size > 0 else 0.0,
        "sequence_curvature_abs_mean": float(np.mean(abs_curve)) if abs_curve.size > 0 else 0.0,
        "sequence_curvature_abs_q90": _finite_quantile(abs_curve, 0.90, default=0.0),
        "sequence_curvature_abs_max": float(np.max(abs_curve)) if abs_curve.size > 0 else 0.0,
    }


def _load_detection_report_features(report_json: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for sample in list(payload.get("samples", []) or []):
        sample_uid = str(sample.get("sample_uid", "") or sample.get("sample_id", "") or "").strip()
        for needle in list(sample.get("needles", []) or []):
            stable_id = int(_safe_int(needle.get("stable_id", -1), default=-1))
            if not sample_uid or stable_id < 0:
                continue
            tol_metrics = dict(needle.get("tol_metrics", {}) or {})
            point_score_arr = _safe_sequence(list(tol_metrics.get("point_score_selected_raw_arr", []) or []))
            step_mm = _safe_float(
                needle.get(
                    "ab_station_step_mm",
                    tol_metrics.get("pred_station_step_mm", tol_metrics.get("ab_station_step_mm", float("nan"))),
                ),
                default=float("nan"),
            )
            event_features = compute_positive_event_features_v1(
                point_score_selected_raw_arr=point_score_arr,
                station_step_mm=step_mm,
            )
            shape_features = _compute_sequence_shape_features(
                point_score_arr,
                station_step_mm=step_mm,
                event_features=event_features,
            )
            enhanced_shape_features = _compute_enhanced_sequence_shape_features(
                point_score_arr,
                station_step_mm=step_mm,
            )
            direction_features, direction_sequence_values = _compute_direction_features(
                tol_metrics=tol_metrics,
                point_score_arr=point_score_arr,
                station_step_mm=step_mm,
            )
            out[(sample_uid, stable_id)] = {
                "matched_detection_report": 1,
                "ab_station_step_mm": float(step_mm),
                "sequence_values": point_score_arr,
                "direction_sequence_values": direction_sequence_values,
                **event_features,
                **shape_features,
                **enhanced_shape_features,
                **direction_features,
            }
    return out


def _attach_sample_relative_features(rows: List[Dict[str, Any]]) -> List[str]:
    relative_columns: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_uid", "")), []).append(row)

    for source_col in RELATIVE_SOURCE_COLUMNS:
        stat_cols = [
            f"{source_col}_sample_median",
            f"{source_col}_sample_q90",
            f"{source_col}_sample_center_median",
            f"{source_col}_sample_zscore",
            f"{source_col}_sample_rank_frac",
        ]
        relative_columns.extend(stat_cols)
        for sample_rows in grouped.values():
            finite_pairs: List[Tuple[int, float]] = []
            for idx, row in enumerate(sample_rows):
                value = float(_safe_float(row.get(source_col, float("nan"))))
                if np.isfinite(value):
                    finite_pairs.append((idx, value))
            if not finite_pairs:
                for row in sample_rows:
                    for key in stat_cols:
                        row[key] = float("nan")
                continue
            values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
            median = float(np.median(values))
            q90 = float(np.quantile(values, 0.90))
            mean = float(np.mean(values))
            std = float(np.std(values))
            sorted_unique = sorted(values.tolist(), reverse=True)
            denom = float(max(len(sorted_unique) - 1, 1))
            for row_idx, value in finite_pairs:
                row = sample_rows[row_idx]
                try:
                    rank = sorted_unique.index(value)
                except ValueError:
                    rank = 0
                row[f"{source_col}_sample_median"] = float(median)
                row[f"{source_col}_sample_q90"] = float(q90)
                row[f"{source_col}_sample_center_median"] = float(value - median)
                row[f"{source_col}_sample_zscore"] = float((value - mean) / std) if std > 1e-9 else 0.0
                row[f"{source_col}_sample_rank_frac"] = float(1.0 - (rank / denom))
            missing_rows = {idx for idx in range(len(sample_rows))} - {idx for idx, _ in finite_pairs}
            for row_idx in missing_rows:
                row = sample_rows[row_idx]
                for key in stat_cols:
                    row[key] = float("nan")
    return relative_columns


def _attach_sample_group_features(rows: List[Dict[str, Any]]) -> List[str]:
    group_columns: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_uid", "")), []).append(row)

    for source_col in GENERALIZED_GROUP_SOURCE_COLUMNS:
        stat_cols = [
            f"{source_col}_sample_rank_ord",
            f"{source_col}_sample_top1_value",
            f"{source_col}_sample_top2_value",
            f"{source_col}_sample_top3_value",
            f"{source_col}_sample_gap_from_top1",
            f"{source_col}_sample_gap_from_top2",
            f"{source_col}_sample_ratio_to_top1",
            f"{source_col}_sample_ratio_to_top2",
            f"{source_col}_sample_value_share_total",
            f"{source_col}_sample_top1_share_total",
            f"{source_col}_sample_top2_share_total",
            f"{source_col}_sample_top3_share_total",
            f"{source_col}_sample_near_top75_count",
            f"{source_col}_sample_near_top90_count",
        ]
        group_columns.extend(stat_cols)
        for sample_rows in grouped.values():
            finite_pairs: List[Tuple[int, float]] = []
            for idx, row in enumerate(sample_rows):
                value = float(_safe_float(row.get(source_col, float("nan"))))
                if np.isfinite(value):
                    finite_pairs.append((idx, value))
            if not finite_pairs:
                for row in sample_rows:
                    for key in stat_cols:
                        row[key] = float("nan")
                continue

            values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
            order = np.argsort(values)[::-1]
            sorted_values = values[order]
            top1 = float(sorted_values[0]) if sorted_values.size >= 1 else float("nan")
            top2 = float(sorted_values[1]) if sorted_values.size >= 2 else float("nan")
            top3 = float(sorted_values[2]) if sorted_values.size >= 3 else float("nan")
            total = float(np.sum(values))
            near_top75_count = int(np.sum(values >= (0.75 * top1))) if np.isfinite(top1) and top1 > 0.0 else 0
            near_top90_count = int(np.sum(values >= (0.90 * top1))) if np.isfinite(top1) and top1 > 0.0 else 0
            top1_share = float(top1 / total) if total > 1e-12 and np.isfinite(top1) else float("nan")
            top2_share = float(np.sum(sorted_values[:2]) / total) if total > 1e-12 and sorted_values.size >= 2 else float("nan")
            top3_share = float(np.sum(sorted_values[:3]) / total) if total > 1e-12 and sorted_values.size >= 3 else float("nan")
            rank_lookup: Dict[int, int] = {int(orig_idx): int(rank + 1) for rank, orig_idx in enumerate(order.tolist())}
            for row_idx, value in finite_pairs:
                row = sample_rows[row_idx]
                rank_ord = int(rank_lookup.get(row_idx, len(values)))
                row[f"{source_col}_sample_rank_ord"] = int(rank_ord)
                row[f"{source_col}_sample_top1_value"] = float(top1)
                row[f"{source_col}_sample_top2_value"] = float(top2)
                row[f"{source_col}_sample_top3_value"] = float(top3)
                row[f"{source_col}_sample_gap_from_top1"] = float(top1 - value) if np.isfinite(top1) else float("nan")
                row[f"{source_col}_sample_gap_from_top2"] = float(top2 - value) if np.isfinite(top2) else float("nan")
                row[f"{source_col}_sample_ratio_to_top1"] = float(value / top1) if np.isfinite(top1) and abs(top1) > 1e-12 else float("nan")
                row[f"{source_col}_sample_ratio_to_top2"] = float(value / top2) if np.isfinite(top2) and abs(top2) > 1e-12 else float("nan")
                row[f"{source_col}_sample_value_share_total"] = float(value / total) if total > 1e-12 else float("nan")
                row[f"{source_col}_sample_top1_share_total"] = float(top1_share)
                row[f"{source_col}_sample_top2_share_total"] = float(top2_share)
                row[f"{source_col}_sample_top3_share_total"] = float(top3_share)
                row[f"{source_col}_sample_near_top75_count"] = int(near_top75_count)
                row[f"{source_col}_sample_near_top90_count"] = int(near_top90_count)
            missing_rows = {idx for idx in range(len(sample_rows))} - {idx for idx, _ in finite_pairs}
            for row_idx in missing_rows:
                row = sample_rows[row_idx]
                for key in stat_cols:
                    row[key] = float("nan")
    return group_columns


def _attach_slot_context_features(rows: List[Dict[str, Any]]) -> List[str]:
    for row in rows:
        sid = int(_safe_int(row.get("stable_id", -1), default=-1))
        edge_distance = min(max(sid, 0), max(19 - sid, 0)) if 0 <= sid <= 19 else -1
        row["stable_id_numeric"] = int(sid)
        row["stable_id_norm"] = float(sid / 19.0) if 0 <= sid <= 19 else float("nan")
        row["stable_edge_distance"] = int(edge_distance)
        row["stable_edge_distance_norm"] = float(edge_distance / 9.0) if edge_distance >= 0 else float("nan")
        row["stable_left_half"] = int(0 <= sid <= 9)
        row["stable_right_half"] = int(10 <= sid <= 19)
    return list(SLOT_CONTEXT_FEATURE_COLUMNS)


def _attach_sample_common_mode_features(rows: List[Dict[str, Any]]) -> List[str]:
    common_columns: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_uid", "")), []).append(row)

    for source_col in COMMON_MODE_SOURCE_COLUMNS:
        stat_cols = [
            f"{source_col}_sample_mean",
            f"{source_col}_sample_std",
            f"{source_col}_sample_q75",
            f"{source_col}_sample_q95",
            f"{source_col}_sample_center_mean",
            f"{source_col}_sample_above_median_count",
            f"{source_col}_sample_positive_count",
        ]
        common_columns.extend(stat_cols)
        for sample_rows in grouped.values():
            finite_pairs: List[Tuple[int, float]] = []
            for idx, row in enumerate(sample_rows):
                value = float(_safe_float(row.get(source_col, float("nan"))))
                if np.isfinite(value):
                    finite_pairs.append((idx, value))
            if not finite_pairs:
                for row in sample_rows:
                    for key in stat_cols:
                        row[key] = float("nan")
                continue
            values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
            mean = float(np.mean(values))
            std = float(np.std(values))
            median = float(np.median(values))
            q75 = float(np.quantile(values, 0.75))
            q95 = float(np.quantile(values, 0.95))
            above_median_count = int(np.count_nonzero(values > median))
            positive_count = int(np.count_nonzero(values > 0.0))
            for row_idx, value in finite_pairs:
                row = sample_rows[row_idx]
                row[f"{source_col}_sample_mean"] = float(mean)
                row[f"{source_col}_sample_std"] = float(std)
                row[f"{source_col}_sample_q75"] = float(q75)
                row[f"{source_col}_sample_q95"] = float(q95)
                row[f"{source_col}_sample_center_mean"] = float(value - mean)
                row[f"{source_col}_sample_above_median_count"] = int(above_median_count)
                row[f"{source_col}_sample_positive_count"] = int(positive_count)
            missing_rows = {idx for idx in range(len(sample_rows))} - {idx for idx, _ in finite_pairs}
            for row_idx in missing_rows:
                row = sample_rows[row_idx]
                for key in stat_cols:
                    row[key] = float("nan")
    return common_columns


def _attach_neighbor_features(rows: List[Dict[str, Any]]) -> List[str]:
    neighbor_columns: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_uid", "")), []).append(row)

    for source_col in NEIGHBOR_SOURCE_COLUMNS:
        stat_cols = [
            f"{source_col}_neighbor_prev_value",
            f"{source_col}_neighbor_next_value",
            f"{source_col}_neighbor_max_value",
            f"{source_col}_neighbor_mean_value",
            f"{source_col}_neighbor_delta_prev",
            f"{source_col}_neighbor_delta_next",
            f"{source_col}_neighbor_higher_count",
        ]
        neighbor_columns.extend(stat_cols)
        for sample_rows in grouped.values():
            by_sid = {int(_safe_int(row.get("stable_id", -1), default=-1)): row for row in sample_rows}
            for row in sample_rows:
                sid = int(_safe_int(row.get("stable_id", -1), default=-1))
                value = float(_safe_float(row.get(source_col, float("nan"))))
                prev_value = float(_safe_float(by_sid.get(sid - 1, {}).get(source_col, float("nan"))))
                next_value = float(_safe_float(by_sid.get(sid + 1, {}).get(source_col, float("nan"))))
                neighbor_values = np.asarray(
                    [item for item in (prev_value, next_value) if np.isfinite(item)],
                    dtype=np.float64,
                )
                row[f"{source_col}_neighbor_prev_value"] = float(prev_value)
                row[f"{source_col}_neighbor_next_value"] = float(next_value)
                row[f"{source_col}_neighbor_max_value"] = float(np.max(neighbor_values)) if neighbor_values.size > 0 else float("nan")
                row[f"{source_col}_neighbor_mean_value"] = float(np.mean(neighbor_values)) if neighbor_values.size > 0 else float("nan")
                row[f"{source_col}_neighbor_delta_prev"] = float(value - prev_value) if np.isfinite(value) and np.isfinite(prev_value) else float("nan")
                row[f"{source_col}_neighbor_delta_next"] = float(value - next_value) if np.isfinite(value) and np.isfinite(next_value) else float("nan")
                row[f"{source_col}_neighbor_higher_count"] = int(np.count_nonzero(neighbor_values > value)) if np.isfinite(value) else 0
    return neighbor_columns


def build_feature_bank_rows_v3(
    *,
    authoritative_geo_csv: Path,
    detection_per_slot_csv: Path,
    detection_report_json: Path,
    compare_inventory_csv: Path,
    sample_ids: Optional[Iterable[str]] = None,
    include_geo_excluded: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_rows = build_feature_bank_rows(
        authoritative_geo_csv=authoritative_geo_csv,
        detection_per_slot_csv=detection_per_slot_csv,
        sample_ids=sample_ids,
        include_geo_excluded=include_geo_excluded,
    )
    detection_report_map = _load_detection_report_features(detection_report_json)
    old36_ids = _sample_old36_ids(compare_inventory_csv)
    rows: List[Dict[str, Any]] = []
    max_seq_len = 0
    max_dir_seq_len = 0
    missing_detection_report = 0
    for base in base_rows:
        row = dict(base)
        key = _normalize_key(row)
        report_features = detection_report_map.get(key)
        if report_features is None:
            missing_detection_report += 1
            report_features = {
                "matched_detection_report": 0,
                "ab_station_step_mm": float("nan"),
                "sequence_values": [],
                "direction_sequence_values": [],
            }
            for col in (
                EVENT_FEATURE_COLUMNS
                + SHAPE_FEATURE_COLUMNS
                + ENHANCED_SHAPE_FEATURE_COLUMNS
                + DIRECTION_SUMMARY_FEATURE_COLUMNS
                + DIRECTION_EVENT_FEATURE_COLUMNS
            ):
                report_features[col] = float("nan")
        sequence_values = list(report_features.get("sequence_values", []) or [])
        direction_sequence_values = list(report_features.get("direction_sequence_values", []) or [])
        max_seq_len = max(max_seq_len, len(sequence_values))
        max_dir_seq_len = max(max_dir_seq_len, len(direction_sequence_values))
        row.update({k: v for k, v in report_features.items() if k not in {"sequence_values", "direction_sequence_values"}})
        row["dataset_subset"] = "old36" if str(row.get("sample_uid", "")) in old36_ids else "new11"
        row["subset_is_new"] = int(row["dataset_subset"] == "new11")
        row["_sequence_values"] = sequence_values
        row["_direction_sequence_values"] = direction_sequence_values
        rows.append(row)

    relative_columns = _attach_sample_relative_features(rows)
    group_columns = _attach_sample_group_features(rows)
    slot_context_columns = _attach_slot_context_features(rows)
    common_mode_columns = _attach_sample_common_mode_features(rows)
    neighbor_columns = _attach_neighbor_features(rows)
    sequence_columns = [_sequence_column_name(i) for i in range(max_seq_len)]
    direction_sequence_columns = [_direction_sequence_column_name(i) for i in range(max_dir_seq_len)]
    for row in rows:
        sequence_values = list(row.pop("_sequence_values", []) or [])
        direction_sequence_values = list(row.pop("_direction_sequence_values", []) or [])
        for idx, col in enumerate(sequence_columns):
            row[col] = float(sequence_values[idx]) if idx < len(sequence_values) else float("nan")
        for idx, col in enumerate(direction_sequence_columns):
            row[col] = float(direction_sequence_values[idx]) if idx < len(direction_sequence_values) else float("nan")

    core_feature_columns = (
        BASE_NUMERIC_FEATURE_COLUMNS
        + EVENT_FEATURE_COLUMNS
        + SHAPE_FEATURE_COLUMNS
        + ENHANCED_SHAPE_FEATURE_COLUMNS
        + DIRECTION_SUMMARY_FEATURE_COLUMNS
        + DIRECTION_EVENT_FEATURE_COLUMNS
    )
    context_feature_columns = core_feature_columns + list(relative_columns) + list(group_columns)
    enhanced_context_columns = (
        context_feature_columns
        + list(slot_context_columns)
        + list(common_mode_columns)
        + list(neighbor_columns)
    )
    feature_summary = {
        "schema_version": 3,
        "row_count": int(len(rows)),
        "sample_count": int(len({str(row.get("sample_uid", "")) for row in rows if str(row.get("sample_uid", ""))})),
        "cv_group_count": int(len({str(row.get("cv_group_uid", "") or row.get("sample_uid", "")) for row in rows if str(row.get("sample_uid", ""))})),
        "augmented_row_count": int(sum(int(_safe_int(row.get("is_augmented", 0), default=0)) for row in rows)),
        "positive_count": int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in rows)),
        "negative_count": int(sum(1 for row in rows if int(_safe_int(row.get("gt_abnormal", 0), default=0)) == 0)),
        "main_chain_count": int(sum(int(_safe_int(row.get("is_main_chain_candidate", 0), default=0)) for row in rows)),
        "old36_row_count": int(sum(1 for row in rows if str(row.get("dataset_subset", "")) == "old36")),
        "new11_row_count": int(sum(1 for row in rows if str(row.get("dataset_subset", "")) == "new11")),
        "missing_detection_report_count": int(missing_detection_report),
        "max_sequence_len": int(max_seq_len),
        "max_direction_sequence_len": int(max_dir_seq_len),
        "base_numeric_feature_columns": list(BASE_NUMERIC_FEATURE_COLUMNS),
        "event_feature_columns": list(EVENT_FEATURE_COLUMNS),
        "shape_feature_columns": list(SHAPE_FEATURE_COLUMNS),
        "enhanced_shape_feature_columns": list(ENHANCED_SHAPE_FEATURE_COLUMNS),
        "direction_summary_feature_columns": list(DIRECTION_SUMMARY_FEATURE_COLUMNS),
        "direction_event_feature_columns": list(DIRECTION_EVENT_FEATURE_COLUMNS),
        "relative_feature_columns": list(relative_columns),
        "group_feature_columns": list(group_columns),
        "slot_context_feature_columns": list(slot_context_columns),
        "common_mode_feature_columns": list(common_mode_columns),
        "neighbor_feature_columns": list(neighbor_columns),
        "sequence_feature_columns": list(sequence_columns),
        "direction_sequence_feature_columns": list(direction_sequence_columns),
        "handcrafted_core_columns": list(core_feature_columns),
        "handcrafted_context_columns": list(context_feature_columns),
        "handcrafted_context_v3_columns": list(enhanced_context_columns),
        "hgb_feature_pool_v3_columns": list(enhanced_context_columns),
        "hgb_feature_pool_v3_no_slot_columns": [
            col for col in enhanced_context_columns if col not in set(slot_context_columns)
        ],
        "hgb_feature_pool_v3_no_neighbor_columns": [
            col for col in enhanced_context_columns if col not in set(neighbor_columns)
        ],
        "hybrid_feature_columns": list(context_feature_columns) + list(sequence_columns) + list(direction_sequence_columns),
        "hybrid_v3_feature_columns": list(enhanced_context_columns) + list(sequence_columns) + list(direction_sequence_columns),
    }
    return rows, feature_summary


def run_build(
    *,
    authoritative_geo_csv: Path,
    detection_per_slot_csv: Path,
    detection_report_json: Path,
    compare_inventory_csv: Path,
    sample_ids: Optional[Iterable[str]] = None,
    include_geo_excluded: bool = False,
) -> Tuple[Path, Path]:
    rows, summary = build_feature_bank_rows_v3(
        authoritative_geo_csv=authoritative_geo_csv,
        detection_per_slot_csv=detection_per_slot_csv,
        detection_report_json=detection_report_json,
        compare_inventory_csv=compare_inventory_csv,
        sample_ids=sample_ids,
        include_geo_excluded=include_geo_excluded,
    )
    out_dir = _default_output_dir()
    csv_path = (out_dir / "geo_shadow_feature_bank_v3.csv").resolve()
    summary_path = (out_dir / "geo_shadow_feature_bank_v3_summary.json").resolve()
    meta_prefix = [
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
        "geo_case",
        "geo_pred",
        "accepted",
        "need_rescan",
        "geo_reject_reason",
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
    feature_cols = (
        summary["hgb_feature_pool_v3_columns"]
        + summary["sequence_feature_columns"]
        + summary["direction_sequence_feature_columns"]
    )
    fieldnames = meta_prefix + [col for col in feature_cols if col not in meta_prefix]
    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "authoritative_geo_csv": str(authoritative_geo_csv),
                "detection_per_slot_csv": str(detection_per_slot_csv),
                "detection_report_json": str(detection_report_json),
                "compare_inventory_csv": str(compare_inventory_csv),
            },
            outputs={
                "feature_bank_csv": str(csv_path),
                "summary_json": str(summary_path),
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "sample_ids": sorted(set(sample_ids or [])),
                "include_geo_excluded": bool(include_geo_excluded),
                "schema_version": 3,
                "feature_group_sizes": {
                    "base_numeric": int(len(summary["base_numeric_feature_columns"])),
                    "event": int(len(summary["event_feature_columns"])),
                    "shape": int(len(summary["shape_feature_columns"])),
                    "enhanced_shape": int(len(summary["enhanced_shape_feature_columns"])),
                    "direction_summary": int(len(summary["direction_summary_feature_columns"])),
                    "direction_event": int(len(summary["direction_event_feature_columns"])),
                    "relative": int(len(summary["relative_feature_columns"])),
                    "group": int(len(summary["group_feature_columns"])),
                    "slot_context": int(len(summary["slot_context_feature_columns"])),
                    "common_mode": int(len(summary["common_mode_feature_columns"])),
                    "neighbor": int(len(summary["neighbor_feature_columns"])),
                    "sequence": int(len(summary["sequence_feature_columns"])),
                    "direction_sequence": int(len(summary["direction_sequence_feature_columns"])),
                },
            }
        )
    _write_csv(csv_path, rows, fieldnames=fieldnames)
    _write_json(summary_path, summary)
    return csv_path, summary_path


def _parse_sample_ids(raw: str) -> Optional[set[str]]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return None
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build geo shadow feature bank v3 with enhanced HGB context features.")
    parser.add_argument("--authoritative_geo_csv", default=DEFAULT_AUTHORITATIVE_GEO_V2_RAW, help="Path to authoritative geo per-slot CSV.")
    parser.add_argument("--detection_per_slot_csv", default=DEFAULT_DETECTION_PER_SLOT_V2_RAW, help="Path to detection per-slot CSV.")
    parser.add_argument("--detection_report_json", default=DEFAULT_DETECTION_REPORT_RAW, help="Path to detection report JSON.")
    parser.add_argument("--compare_inventory_csv", default=DEFAULT_COMPARE_INVENTORY_RAW, help="Path to old compare inventory CSV for old36/new11 split.")
    parser.add_argument("--sample_ids", default="", help="Optional comma-separated sample_uid allowlist.")
    parser.add_argument("--include_geo_excluded", action="store_true", help="Include rows where geo_included != 1.")
    args = parser.parse_args()

    authoritative_geo_csv = _resolve_existing_path(args.authoritative_geo_csv, default_authoritative_geo_csv_v2())
    detection_per_slot_csv = _resolve_existing_path(args.detection_per_slot_csv, default_detection_per_slot_csv_v2())
    detection_report_json = _resolve_existing_path(args.detection_report_json, default_detection_report_json())
    compare_inventory_csv = _resolve_existing_path(args.compare_inventory_csv, default_compare_inventory_csv())
    sample_ids = _parse_sample_ids(args.sample_ids)
    csv_path, summary_path = run_build(
        authoritative_geo_csv=authoritative_geo_csv,
        detection_per_slot_csv=detection_per_slot_csv,
        detection_report_json=detection_report_json,
        compare_inventory_csv=compare_inventory_csv,
        sample_ids=sample_ids,
        include_geo_excluded=bool(args.include_geo_excluded),
    )
    print(str(csv_path))
    print(str(summary_path))


if __name__ == "__main__":
    main()
