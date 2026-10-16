from __future__ import annotations

import os

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

FROZEN_COMPARE_READOUT_PROTOCOL_V1: Dict[str, Any] = {
    "compare_mode": "bounded_local_slack_v1",
    "readout_mode": "topk_quantile_v1",
    "slack_radius_steps": 3,
    "topk_frac": 0.02,
    "topk_min_points": 3,
}

def get_frozen_compare_readout_protocol_v1() -> Dict[str, Any]:
    return dict(FROZEN_COMPARE_READOUT_PROTOCOL_V1)

def normalize_compare_readout_protocol_params(params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    defaults = get_frozen_compare_readout_protocol_v1()
    src = dict(params) if isinstance(params, Mapping) else {}

    compare_mode = str(src.get("compare_mode", defaults["compare_mode"]) or defaults["compare_mode"]).strip().lower()
    if compare_mode not in {"fixed_station", "bounded_local_slack_v1", "banded_monotone_slack_v1"}:
        compare_mode = str(defaults["compare_mode"])

    readout_mode = str(src.get("readout_mode", defaults["readout_mode"]) or defaults["readout_mode"]).strip().lower()
    if readout_mode not in {"raw_max", "topk_quantile_v1"}:
        readout_mode = str(defaults["readout_mode"])

    try:
        slack_radius_steps = int(float(str(src.get("slack_radius_steps", defaults["slack_radius_steps"]) or defaults["slack_radius_steps"])))
    except Exception:
        slack_radius_steps = int(defaults["slack_radius_steps"])
    slack_radius_steps = max(0, int(slack_radius_steps))

    try:
        topk_frac = float(str(src.get("topk_frac", defaults["topk_frac"]) or defaults["topk_frac"]))
    except Exception:
        topk_frac = float(defaults["topk_frac"])
    if not np.isfinite(topk_frac) or topk_frac <= 0.0 or topk_frac > 1.0:
        topk_frac = float(defaults["topk_frac"])

    try:
        topk_min_points = int(float(str(src.get("topk_min_points", defaults["topk_min_points"]) or defaults["topk_min_points"])))
    except Exception:
        topk_min_points = int(defaults["topk_min_points"])
    topk_min_points = max(1, int(topk_min_points))

    return {
        "compare_mode": str(compare_mode),
        "readout_mode": str(readout_mode),
        "slack_radius_steps": int(slack_radius_steps),
        "topk_frac": float(topk_frac),
        "topk_min_points": int(topk_min_points),
    }

def resolve_compare_readout_protocol_from_env() -> Dict[str, Any]:
    defaults = get_frozen_compare_readout_protocol_v1()
    return normalize_compare_readout_protocol_params(
        {
            "compare_mode": os.getenv("DAIMA_COMPARE_MODE", str(defaults["compare_mode"])),
            "readout_mode": os.getenv("DAIMA_READOUT_MODE", str(defaults["readout_mode"])),
            "slack_radius_steps": os.getenv("DAIMA_SLACK_RADIUS_STEPS", str(defaults["slack_radius_steps"])),
            "topk_frac": os.getenv("DAIMA_READOUT_TOPK_FRAC", str(defaults["topk_frac"])),
            "topk_min_points": os.getenv("DAIMA_READOUT_TOPK_MIN_POINTS", str(defaults["topk_min_points"])),
        }
    )

def compare_readout_protocol_matches_frozen(params: Optional[Mapping[str, Any]] = None) -> bool:
    active = normalize_compare_readout_protocol_params(params)
    frozen = normalize_compare_readout_protocol_params(get_frozen_compare_readout_protocol_v1())
    return (
        str(active["compare_mode"]) == str(frozen["compare_mode"])
        and str(active["readout_mode"]) == str(frozen["readout_mode"])
        and int(active["slack_radius_steps"]) == int(frozen["slack_radius_steps"])
        and abs(float(active["topk_frac"]) - float(frozen["topk_frac"])) <= 1e-12
        and int(active["topk_min_points"]) == int(frozen["topk_min_points"])
    )

def format_compare_readout_protocol(params: Optional[Mapping[str, Any]] = None) -> str:
    norm = normalize_compare_readout_protocol_params(params)
    return (
        f"{norm['compare_mode']} + {norm['readout_mode']}"
        f" (slack_radius_steps={norm['slack_radius_steps']}, "
        f"topk_frac={norm['topk_frac']:.4f}, "
        f"topk_min_points={norm['topk_min_points']})"
    )

def finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False

def cumulative_arclength_u(curve_u: Any) -> np.ndarray:
    pts = np.asarray(curve_u, dtype=np.float64).reshape(-1, 3)
    if pts.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return np.full((len(pts),), np.nan, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros((len(pts),), dtype=np.float64)
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    if not np.all(np.isfinite(d)):
        return np.full((len(pts),), np.nan, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(d))).astype(np.float64)

def resample_curve_by_count(curve_u: Any, n: int) -> np.ndarray:
    pts = np.asarray(curve_u, dtype=np.float64).reshape(-1, 3)
    if pts.size <= 0:
        return pts.reshape((0, 3))
    if len(pts) == 1:
        return np.repeat(pts, repeats=max(1, int(n)), axis=0)
    n = max(2, int(n))
    s = cumulative_arclength_u(pts)
    if len(s) != len(pts) or not np.all(np.isfinite(s)):
        return np.repeat(pts[:1], repeats=n, axis=0)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 1e-12:
        return np.repeat(pts[:1], repeats=n, axis=0)
    sample_s = np.linspace(0.0, total, n)
    idx = np.searchsorted(s, sample_s, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 2)
    seg = np.maximum(s[idx + 1] - s[idx], 1e-12)
    t = (sample_s - s[idx]) / seg
    out = pts[idx] + (pts[idx + 1] - pts[idx]) * t[:, None]
    return out.astype(np.float64)

def orient_curve_to_match_endpoints(defect_u: Any, nominal_u: Any) -> np.ndarray:
    defect = np.asarray(defect_u, dtype=np.float64).reshape(-1, 3)
    nominal = np.asarray(nominal_u, dtype=np.float64).reshape(-1, 3)
    if len(defect) < 2 or len(nominal) < 2:
        return defect
    cost_keep = float(np.linalg.norm(defect[0] - nominal[0]) + np.linalg.norm(defect[-1] - nominal[-1]))
    cost_flip = float(np.linalg.norm(defect[0] - nominal[-1]) + np.linalg.norm(defect[-1] - nominal[0]))
    return defect[::-1] if cost_flip < cost_keep else defect

def _normalize_rows(arr: Any) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.size <= 0:
        return x.reshape((0, 3))
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 0.0
    out = np.zeros_like(x, dtype=np.float64)
    valid = n.reshape(-1) > 1e-12
    if np.any(valid):
        out[valid] = x[valid] / n[valid]
    return out

def estimate_curve_tangents_windowed(curve_u: Any, unit_scale: float, *, window_mm: Any) -> np.ndarray:
    pts = np.asarray(curve_u, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return np.zeros((len(pts), 3), dtype=np.float64)

    window_mm_f = float(window_mm) if finite(window_mm) else 0.0
    if window_mm_f <= 1e-12:
        return _normalize_rows(np.gradient(pts, axis=0))

    unit_scale_f = float(unit_scale) if finite(unit_scale) and float(unit_scale) > 1e-12 else 1.0
    s_u = cumulative_arclength_u(pts)
    if len(s_u) != len(pts) or not np.all(np.isfinite(s_u)):
        return _normalize_rows(np.gradient(pts, axis=0))

    s_mm = s_u / (unit_scale_f + 1e-12)
    tangents = np.zeros((len(pts), 3), dtype=np.float64)
    last_idx = len(pts) - 1
    for i in range(len(pts)):
        s0 = float(s_mm[i])
        left = int(np.searchsorted(s_mm, s0 - window_mm_f, side="left"))
        right = int(np.searchsorted(s_mm, s0 + window_mm_f, side="right") - 1)
        left = min(max(left, 0), i)
        right = max(min(right, last_idx), i)
        if left == right:
            if i <= 0:
                right = min(1, last_idx)
            elif i >= last_idx:
                left = max(0, last_idx - 1)
            else:
                left = i - 1
                right = i + 1
        if right <= left:
            if i <= 0:
                left, right = 0, min(1, last_idx)
            elif i >= last_idx:
                left, right = max(0, last_idx - 1), last_idx
            else:
                left, right = i - 1, i + 1
        tangents[i] = pts[right] - pts[left]
    return _normalize_rows(tangents)

def _longest_true_span_mm(mask: Any, curve_u: Any, unit_scale: float) -> float:
    flag = np.asarray(mask, dtype=bool).reshape(-1)
    pts = np.asarray(curve_u, dtype=np.float64).reshape(-1, 3)
    n = min(len(flag), len(pts))
    if n < 2:
        return 0.0
    flag = flag[:n]
    s_mm = cumulative_arclength_u(pts[:n]) / (float(unit_scale) + 1e-12)
    if len(s_mm) != n or not np.all(np.isfinite(s_mm)):
        return 0.0
    best = 0.0
    start = None
    for idx, is_true in enumerate(flag):
        if is_true and start is None:
            start = idx
        elif (not is_true) and start is not None:
            best = max(best, float(s_mm[idx - 1] - s_mm[start]))
            start = None
    if start is not None:
        best = max(best, float(s_mm[n - 1] - s_mm[start]))
    return float(best)

def _true_segments_from_mask(mask: Any, s_mm: Any) -> List[Dict[str, Any]]:
    flag = np.asarray(mask, dtype=bool).reshape(-1)
    s_arr = np.asarray(s_mm, dtype=np.float64).reshape(-1)
    n = min(len(flag), len(s_arr))
    if n <= 0:
        return []
    flag = flag[:n]
    s_arr = s_arr[:n]
    if not np.all(np.isfinite(s_arr)):
        return []
    segments: List[Dict[str, Any]] = []
    start = None
    for idx, is_true in enumerate(flag):
        if is_true and start is None:
            start = idx
        elif (not is_true) and start is not None:
            end = idx - 1
            segments.append(
                {
                    "idx_start": int(start),
                    "idx_end": int(end),
                    "n_points": int(end - start + 1),
                    "s_start_mm": float(s_arr[start]),
                    "s_end_mm": float(s_arr[end]),
                    "span_mm": float(max(0.0, s_arr[end] - s_arr[start])),
                }
            )
            start = None
    if start is not None:
        end = n - 1
        segments.append(
            {
                "idx_start": int(start),
                "idx_end": int(end),
                "n_points": int(end - start + 1),
                "s_start_mm": float(s_arr[start]),
                "s_end_mm": float(s_arr[end]),
                "span_mm": float(max(0.0, s_arr[end] - s_arr[start])),
            }
        )
    return segments

def _windowed_true_mask_stats(mask: Any, curve_u: Any, unit_scale: float, window_mm: Any) -> Dict[str, Any]:
    flag = np.asarray(mask, dtype=bool).reshape(-1)
    pts = np.asarray(curve_u, dtype=np.float64).reshape(-1, 3)
    n = min(len(flag), len(pts))
    if n < 2:
        return {
            "window_mm": 0.0,
            "s_mm": np.zeros((0,), dtype=np.float64),
            "left_idx_arr": np.zeros((0,), dtype=np.int64),
            "right_idx_arr": np.zeros((0,), dtype=np.int64),
            "over_ratio_arr": np.zeros((0,), dtype=np.float64),
            "over_span_mm_arr": np.zeros((0,), dtype=np.float64),
            "raw_segments": [],
        }
    flag = flag[:n]
    pts = pts[:n]
    unit_scale_f = float(unit_scale) if finite(unit_scale) and float(unit_scale) > 1e-12 else 1.0
    s_mm = cumulative_arclength_u(pts) / (unit_scale_f + 1e-12)
    if len(s_mm) != n or not np.all(np.isfinite(s_mm)):
        return {
            "window_mm": 0.0,
            "s_mm": np.zeros((0,), dtype=np.float64),
            "left_idx_arr": np.zeros((0,), dtype=np.int64),
            "right_idx_arr": np.zeros((0,), dtype=np.int64),
            "over_ratio_arr": np.zeros((0,), dtype=np.float64),
            "over_span_mm_arr": np.zeros((0,), dtype=np.float64),
            "raw_segments": [],
        }

    step_mm = np.diff(s_mm)
    step_ref_mm = float(np.median(step_mm[np.isfinite(step_mm)])) if np.any(np.isfinite(step_mm)) else 0.0
    target_window_mm = float(window_mm) if finite(window_mm) else 0.0
    if target_window_mm <= 1e-9:
        target_window_mm = max(0.0, step_ref_mm * 2.0)
    half_window_mm = max(target_window_mm * 0.5, step_ref_mm, 1e-6)
    actual_window_mm = float(half_window_mm * 2.0)

    left_idx_arr = np.zeros((n,), dtype=np.int64)
    right_idx_arr = np.zeros((n,), dtype=np.int64)
    over_ratio_arr = np.zeros((n,), dtype=np.float64)
    over_span_mm_arr = np.zeros((n,), dtype=np.float64)
    for idx in range(n):
        s0 = float(s_mm[idx])
        left = int(np.searchsorted(s_mm, s0 - half_window_mm, side="left"))
        right = int(np.searchsorted(s_mm, s0 + half_window_mm, side="right") - 1)
        left = min(max(left, 0), idx)
        right = max(min(right, n - 1), idx)
        left_idx_arr[idx] = int(left)
        right_idx_arr[idx] = int(right)
        win_flag = flag[left : right + 1]
        if win_flag.size <= 0:
            continue
        over_ratio_arr[idx] = float(np.count_nonzero(win_flag)) / float(win_flag.size)
        if np.any(win_flag):
            over_span_mm_arr[idx] = float(_longest_true_span_mm(win_flag, pts[left : right + 1], unit_scale_f))

    return {
        "window_mm": float(actual_window_mm),
        "s_mm": s_mm.astype(np.float64),
        "left_idx_arr": left_idx_arr.astype(np.int64),
        "right_idx_arr": right_idx_arr.astype(np.int64),
        "over_ratio_arr": over_ratio_arr.astype(np.float64),
        "over_span_mm_arr": over_span_mm_arr.astype(np.float64),
        "raw_segments": _true_segments_from_mask(flag, s_mm),
    }

def _expand_window_centers_to_mask(
    center_mask: Any,
    left_idx_arr: Any,
    right_idx_arr: Any,
    n_points: int,
    *,
    limit_mask: Any = None,
) -> np.ndarray:
    center_flag = np.asarray(center_mask, dtype=bool).reshape(-1)
    left_arr = np.asarray(left_idx_arr, dtype=np.int64).reshape(-1)
    right_arr = np.asarray(right_idx_arr, dtype=np.int64).reshape(-1)
    n = min(int(n_points), len(center_flag), len(left_arr), len(right_arr))
    out = np.zeros((max(0, int(n_points)),), dtype=bool)
    for idx in range(n):
        if not center_flag[idx]:
            continue
        left = min(max(int(left_arr[idx]), 0), max(0, int(n_points) - 1))
        right = min(max(int(right_arr[idx]), left), max(0, int(n_points) - 1))
        out[left : right + 1] = True
    if limit_mask is not None:
        out &= np.asarray(limit_mask, dtype=bool).reshape(-1)[: len(out)]
    return out.astype(bool)

def _resolve_tangent_decision_params(tangent_window_mm: Any) -> Dict[str, float]:
    base_window_mm = float(tangent_window_mm) if finite(tangent_window_mm) and float(tangent_window_mm) > 0 else 1.0
    fail_window_mm = max(2.0, 2.0 * base_window_mm)
    fail_min_ratio = 0.80
    fail_min_span_mm = max(1.5, 0.75 * fail_window_mm)
    return {
        "fail_window_mm": float(fail_window_mm),
        "fail_min_ratio": float(fail_min_ratio),
        "fail_min_span_mm": float(fail_min_span_mm),
    }

def classify_tol_fail_subtype(
    *,
    max_lateral_ratio: Any,
    tip_short_ratio: Any,
    len_short_ratio: Any,
    eps: float = 1e-9,
) -> str:
    hits: List[str] = []
    if finite(max_lateral_ratio) and float(max_lateral_ratio) > 1.0 + float(eps):
        hits.append("lateral")
    if finite(tip_short_ratio) and float(tip_short_ratio) > 1.0 + float(eps):
        hits.append("tip")
    if finite(len_short_ratio) and float(len_short_ratio) > 1.0 + float(eps):
        hits.append("len")
    if not hits:
        return "none"
    if len(hits) == 1:
        return f"{hits[0]}-only"
    return "mixed"

def _resolve_compare_readout_ab_params() -> Dict[str, Any]:
    return resolve_compare_readout_protocol_from_env()

def _estimate_station_step_mm(nominal_n: np.ndarray, unit_scale_f: float) -> float:
    s_mm = cumulative_arclength_u(nominal_n) / (unit_scale_f + 1e-12)
    if len(s_mm) < 2:
        return float("nan")
    diff = np.diff(s_mm)
    if not np.any(np.isfinite(diff)):
        return float("nan")
    return float(np.median(diff))

def _pack_lateral_compare_metrics(
    *,
    ratio_arr: Any,
    shift_steps_arr: Any,
    station_step_mm: Any,
) -> Dict[str, Any]:
    ratio = np.asarray(ratio_arr, dtype=np.float64).reshape(-1)
    shifts = np.asarray(shift_steps_arr, dtype=np.int32).reshape(-1)
    if shifts.size != ratio.size:
        if shifts.size <= 0:
            shifts = np.zeros((ratio.size,), dtype=np.int32)
        elif shifts.size < ratio.size:
            shifts = np.pad(shifts, (0, int(ratio.size - shifts.size)), constant_values=0)
        else:
            shifts = shifts[: ratio.size]

    step_mm = float(station_step_mm) if finite(station_step_mm) and float(station_step_mm) > 0.0 else float("nan")
    shift_mm_arr = np.full((ratio.size,), np.nan, dtype=np.float64)
    if np.isfinite(step_mm):
        shift_mm_arr = shifts.astype(np.float64) * float(step_mm)

    abs_shift = np.abs(shifts.astype(np.float64))
    finite_shift_mm = shift_mm_arr[np.isfinite(shift_mm_arr)]
    return {
        "ratio_arr": ratio.astype(np.float64),
        "margin_arr": (ratio - 1.0).astype(np.float64),
        "score_arr": np.clip(ratio - 1.0, 0.0, 1.0).astype(np.float64),
        "shift_steps_arr": shifts.astype(np.int32),
        "shift_mm_arr": shift_mm_arr.astype(np.float64),
        "warp_nonzero_ratio": float(np.mean(abs_shift > 0.0)) if abs_shift.size else 0.0,
        "warp_p90_steps": float(np.quantile(abs_shift, 0.90)) if abs_shift.size else 0.0,
        "warp_max_steps": float(np.max(abs_shift)) if abs_shift.size else 0.0,
        "warp_p90_mm": float(np.quantile(np.abs(finite_shift_mm), 0.90)) if finite_shift_mm.size else 0.0,
        "warp_max_mm": float(np.max(np.abs(finite_shift_mm))) if finite_shift_mm.size else 0.0,
        "station_step_mm": float(step_mm),
    }

def _compute_bounded_local_slack_lateral_metrics(
    *,
    nominal_n: np.ndarray,
    defect_n: np.ndarray,
    tan: np.ndarray,
    tol_lat: np.ndarray,
    unit_scale_f: float,
    fixed_ratio_arr: np.ndarray,
    slack_radius_steps: int,
) -> Dict[str, Any]:
    n = int(len(nominal_n))
    fixed_ratio = np.asarray(fixed_ratio_arr, dtype=np.float64).reshape(-1)
    fixed_ratio = fixed_ratio[:n] if len(fixed_ratio) >= n else np.pad(fixed_ratio, (0, max(0, n - len(fixed_ratio))), constant_values=np.nan)
    step_mm = _estimate_station_step_mm(nominal_n, unit_scale_f)

    if n < 2 or int(slack_radius_steps) <= 0:
        return _pack_lateral_compare_metrics(
            ratio_arr=fixed_ratio,
            shift_steps_arr=np.zeros((n,), dtype=np.int32),
            station_step_mm=step_mm,
        )

    ratio_arr = np.full((n,), np.nan, dtype=np.float64)
    shift_steps_arr = np.zeros((n,), dtype=np.int32)
    for i in range(n):
        left = max(0, int(i) - int(slack_radius_steps))
        right = min(n - 1, int(i) + int(slack_radius_steps))
        best_ratio = float("inf")
        best_shift = 0
        for j in range(left, right + 1):
            delta_ij = defect_n[i] - nominal_n[j]
            proj_ij = float(np.dot(delta_ij, tan[j])) * tan[j]
            lateral_mm_ij = float(np.linalg.norm(delta_ij - proj_ij) / (unit_scale_f + 1e-12))
            tol_mm_ij = float(max(float(tol_lat[j]), 1e-12))
            ratio_ij = float(lateral_mm_ij / tol_mm_ij)
            shift_ij = int(j - i)
            if (ratio_ij + 1e-12) < best_ratio or (
                abs(ratio_ij - best_ratio) <= 1e-12 and abs(shift_ij) < abs(best_shift)
            ):
                best_ratio = float(ratio_ij)
                best_shift = int(shift_ij)
        ratio_arr[i] = float(best_ratio)
        shift_steps_arr[i] = int(best_shift)
    return _pack_lateral_compare_metrics(
        ratio_arr=ratio_arr,
        shift_steps_arr=shift_steps_arr,
        station_step_mm=step_mm,
    )

def _compute_banded_monotone_slack_lateral_metrics(
    *,
    nominal_n: np.ndarray,
    defect_n: np.ndarray,
    tan: np.ndarray,
    tol_lat: np.ndarray,
    unit_scale_f: float,
    fixed_ratio_arr: np.ndarray,
    slack_radius_steps: int,
    max_step_steps: int = 3,
    transition_penalty: float = 0.05,
    diag_penalty: float = 0.03,
    start_penalty: float = 0.03,
) -> Dict[str, Any]:
    n = int(len(nominal_n))
    fixed_ratio = np.asarray(fixed_ratio_arr, dtype=np.float64).reshape(-1)
    fixed_ratio = fixed_ratio[:n] if len(fixed_ratio) >= n else np.pad(fixed_ratio, (0, max(0, n - len(fixed_ratio))), constant_values=np.nan)
    step_mm = _estimate_station_step_mm(nominal_n, unit_scale_f)

    if n < 2 or int(slack_radius_steps) <= 0:
        return _pack_lateral_compare_metrics(
            ratio_arr=fixed_ratio,
            shift_steps_arr=np.zeros((n,), dtype=np.int32),
            station_step_mm=step_mm,
        )

    radius = int(max(0, slack_radius_steps))
    max_step = int(max(1, min(max_step_steps, max(1, radius))))
    band_norm = float(max(1, radius))
    ratio_mat = np.full((n, n), np.nan, dtype=np.float64)
    local_cost = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        left = max(0, i - radius)
        right = min(n - 1, i + radius)
        for j in range(left, right + 1):
            delta_ij = defect_n[i] - nominal_n[j]
            proj_ij = float(np.dot(delta_ij, tan[j])) * tan[j]
            lateral_mm_ij = float(np.linalg.norm(delta_ij - proj_ij) / (unit_scale_f + 1e-12))
            tol_mm_ij = float(max(float(tol_lat[j]), 1e-12))
            ratio_ij = float(lateral_mm_ij / tol_mm_ij)
            ratio_mat[i, j] = float(ratio_ij)
            local_cost[i, j] = float(ratio_ij + float(diag_penalty) * (abs(j - i) / band_norm))

    dp = np.full((n, n), np.inf, dtype=np.float64)
    prev = np.full((n, n), -1, dtype=np.int32)
    start_left = 0
    start_right = min(n - 1, radius)
    for j in range(start_left, start_right + 1):
        if not np.isfinite(local_cost[0, j]):
            continue
        dp[0, j] = float(local_cost[0, j] + float(start_penalty) * (abs(j) / band_norm))

    for i in range(1, n):
        left = max(0, i - radius)
        right = min(n - 1, i + radius)
        prev_left = max(0, i - 1 - radius)
        prev_right = min(n - 1, i - 1 + radius)
        for j in range(left, right + 1):
            if not np.isfinite(local_cost[i, j]):
                continue
            best_cost = float("inf")
            best_prev = -1
            cand_left = max(prev_left, j - max_step)
            cand_right = min(prev_right, j)
            for k in range(cand_left, cand_right + 1):
                prev_cost = float(dp[i - 1, k])
                if not np.isfinite(prev_cost):
                    continue
                delta_step = int(j - k)
                transition_cost = float(transition_penalty) * abs(float(delta_step) - 1.0)
                cand = float(prev_cost + local_cost[i, j] + transition_cost)
                if cand < best_cost:
                    best_cost = float(cand)
                    best_prev = int(k)
            if best_prev < 0:
                for k in range(prev_left, min(prev_right, j) + 1):
                    prev_cost = float(dp[i - 1, k])
                    if not np.isfinite(prev_cost):
                        continue
                    delta_step = int(j - k)
                    transition_cost = float(transition_penalty) * abs(float(delta_step) - 1.0)
                    cand = float(prev_cost + local_cost[i, j] + transition_cost)
                    if cand < best_cost:
                        best_cost = float(cand)
                        best_prev = int(k)
            if best_prev >= 0:
                dp[i, j] = float(best_cost)
                prev[i, j] = int(best_prev)

    end_left = max(0, n - 1 - radius)
    end_right = n - 1
    end_j = -1
    end_cost = float("inf")
    for j in range(end_left, end_right + 1):
        cur = float(dp[n - 1, j])
        if cur < end_cost:
            end_cost = float(cur)
            end_j = int(j)

    if end_j < 0 or not np.isfinite(end_cost):
        fallback = _compute_bounded_local_slack_lateral_metrics(
            nominal_n=nominal_n,
            defect_n=defect_n,
            tan=tan,
            tol_lat=tol_lat,
            unit_scale_f=unit_scale_f,
            fixed_ratio_arr=fixed_ratio,
            slack_radius_steps=radius,
        )
        fallback["path_cost_mean"] = float("nan")
        fallback["path_start_j"] = -1
        fallback["path_end_j"] = -1
        fallback["path_step_deviation_mean"] = float("nan")
        fallback["path_used_fallback"] = True
        return fallback

    j_by_i = np.zeros((n,), dtype=np.int32)
    cur_j = int(end_j)
    for i in range(n - 1, -1, -1):
        j_by_i[i] = int(cur_j)
        if i > 0:
            nxt = int(prev[i, cur_j])
            if nxt < 0:
                nxt = int(cur_j)
            cur_j = int(nxt)

    ratio_arr = np.asarray([ratio_mat[i, int(j_by_i[i])] for i in range(n)], dtype=np.float64)
    out = _pack_lateral_compare_metrics(
        ratio_arr=ratio_arr,
        shift_steps_arr=(j_by_i - np.arange(n, dtype=np.int32)),
        station_step_mm=step_mm,
    )
    if n >= 2:
        path_step_dev = np.abs(np.diff(j_by_i.astype(np.float64)) - 1.0)
        out["path_step_deviation_mean"] = float(np.mean(path_step_dev)) if path_step_dev.size else 0.0
    else:
        out["path_step_deviation_mean"] = float("nan")
    out["path_cost_mean"] = float(end_cost / max(1, n))
    out["path_start_j"] = int(j_by_i[0]) if j_by_i.size else -1
    out["path_end_j"] = int(j_by_i[-1]) if j_by_i.size else -1
    out["path_used_fallback"] = False
    return out

def _compute_needle_readout_metrics(
    point_score_raw_arr: Any,
    *,
    readout_mode: str,
    topk_frac: float,
    topk_min_points: int,
) -> Dict[str, Any]:
    arr = np.asarray(point_score_raw_arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "needle_score_raw": float("nan"),
            "needle_score_clipped": float("nan"),
            "needle_score_q95": float("nan"),
            "needle_score_topk_mean": float("nan"),
            "needle_score_topk_count": 0,
        }

    sorted_arr = np.sort(arr)
    topk_n = min(int(sorted_arr.size), max(int(topk_min_points), int(np.ceil(float(topk_frac) * float(sorted_arr.size)))))
    topk_vals = sorted_arr[-topk_n:] if topk_n > 0 else sorted_arr[-1:]
    q95 = float(np.quantile(sorted_arr, 0.95)) if sorted_arr.size else float("nan")
    topk_mean = float(np.mean(topk_vals)) if topk_vals.size else float("nan")

    mode_n = str(readout_mode).strip().lower()
    if mode_n == "topk_quantile_v1":
        score_raw = float(topk_mean)
    else:
        score_raw = float(np.max(sorted_arr))

    return {
        "needle_score_raw": float(score_raw),
        "needle_score_clipped": float(np.clip(score_raw, 0.0, 1.0)) if np.isfinite(score_raw) else float("nan"),
        "needle_score_q95": float(q95),
        "needle_score_topk_mean": float(topk_mean),
        "needle_score_topk_count": int(topk_n),
    }

def _compute_ab_shadow_verdict(
    *,
    needle_score_raw: Any,
    tip_short_ratio: Any,
    len_short_ratio: Any,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    if finite(tip_short_ratio) and float(tip_short_ratio) > 1.0 + float(eps):
        return {"verdict": "fail", "reason": "tip_short_fail"}
    if finite(len_short_ratio) and float(len_short_ratio) > 1.0 + float(eps):
        return {"verdict": "fail", "reason": "len_short_fail"}
    if finite(needle_score_raw) and float(needle_score_raw) > 0.0 + float(eps):
        return {"verdict": "fail", "reason": "point_readout_positive"}
    if finite(needle_score_raw):
        return {"verdict": "pass", "reason": "clear_pass"}
    return {"verdict": "unknown", "reason": "non_finite_needle_score"}

def evaluate_centerline_tolerance_with_direction(
    nominal_curve_u: Any,
    defect_curve_u: Any,
    *,
    tol_lateral_mm: Any,
    tol_tip_short_mm: Any,
    tol_len_short_mm: Any,
    unit_scale: Any,
    tol_tangent_axis_xyz: Any = None,
    tol_tangent_half_angle_deg: Any = None,
    tol_tangent_window_mm: Any = None,
    tol_tangent_model_status: Any = None,
    use_direction_score: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any], np.ndarray, np.ndarray]:
    nominal = np.asarray(nominal_curve_u, dtype=np.float64).reshape(-1, 3)
    defect = np.asarray(defect_curve_u, dtype=np.float64).reshape(-1, 3)
    tol_lat = np.asarray(tol_lateral_mm, dtype=np.float64).reshape(-1)
    if nominal.ndim != 2 or nominal.shape[1] != 3 or len(nominal) < 2:
        return None, {}, nominal, defect
    if defect.ndim != 2 or defect.shape[1] != 3 or len(defect) < 2:
        return None, {}, nominal, defect

    n = int(len(tol_lat)) if tol_lat.size else int(len(nominal))
    if n < 2:
        return None, {}, nominal, defect

    unit_scale_f = float(unit_scale) if finite(unit_scale) and float(unit_scale) > 1e-12 else 1.0
    nominal_n = resample_curve_by_count(nominal, n) if len(nominal) != n else nominal
    defect_o = orient_curve_to_match_endpoints(defect, nominal_n)
    defect_n = resample_curve_by_count(defect_o, n) if len(defect_o) != n else defect_o

    tan = _normalize_rows(np.gradient(nominal_n, axis=0))
    delta = defect_n - nominal_n
    proj = np.sum(delta * tan, axis=1, keepdims=True) * tan
    lateral_u = np.linalg.norm(delta - proj, axis=1)
    lateral_mm = lateral_u / (unit_scale_f + 1e-12)

    if tol_lat.size != n:
        tol_lat = np.interp(
            np.linspace(0.0, 1.0, n),
            np.linspace(0.0, 1.0, max(2, int(tol_lat.size))),
            tol_lat if tol_lat.size else np.zeros(n, dtype=np.float64),
        )
    denom = np.maximum(tol_lat, 1e-12)
    ratio = lateral_mm / denom
    lateral_margin = ratio - 1.0
    lateral_score = np.clip(ratio - 1.0, 0.0, 1.0)

    max_lateral_mm = float(np.max(lateral_mm)) if lateral_mm.size else float("nan")
    max_lateral_ratio = float(np.max(ratio)) if ratio.size else float("nan")

    axis = nominal_n[-1] - nominal_n[0]
    axis_n = float(np.linalg.norm(axis))
    axis = axis / (axis_n + 1e-12)
    tip_axial_mm = float(np.dot(defect_n[-1] - nominal_n[-1], axis) / (unit_scale_f + 1e-12))
    tip_short_mm = max(0.0, -tip_axial_mm)
    tol_tip = float(tol_tip_short_mm) if finite(tol_tip_short_mm) else float("nan")
    tip_short_ratio = (float(tip_short_mm) / tol_tip) if np.isfinite(tol_tip) and tol_tip > 1e-12 else float("nan")

    len_nom_mm = float(cumulative_arclength_u(nominal_n)[-1] / (unit_scale_f + 1e-12))
    len_def_mm = float(cumulative_arclength_u(defect_n)[-1] / (unit_scale_f + 1e-12))
    len_short_mm = max(0.0, float(len_nom_mm - len_def_mm))
    tol_len = float(tol_len_short_mm) if finite(tol_len_short_mm) else float("nan")
    len_short_ratio = (float(len_short_mm) / tol_len) if np.isfinite(tol_len) and tol_len > 1e-12 else float("nan")

    direction_model_status = str(tol_tangent_model_status).strip() if tol_tangent_model_status is not None else ""
    if not direction_model_status:
        direction_model_status = "ok"
    tangent_ratio = np.full((n,), np.nan, dtype=np.float64)
    tangent_deg = np.full((n,), np.nan, dtype=np.float64)
    tangent_half_angle_deg = np.full((n,), np.nan, dtype=np.float64)
    tangent_margin = np.full((n,), np.nan, dtype=np.float64)
    tangent_score = np.zeros((n,), dtype=np.float64)
    tangent_status = "model_insufficient"
    tangent_decision_mode = "window_ratio_span_v1"
    tangent_fail_window_mm = float("nan")
    tangent_fail_min_ratio = float("nan")
    tangent_fail_min_span_mm = float("nan")
    tangent_window_over_ratio_arr = np.zeros((n,), dtype=np.float64)
    tangent_window_over_span_mm_arr = np.zeros((n,), dtype=np.float64)
    tangent_window_fail_mask = np.zeros((n,), dtype=bool)
    tangent_decision_mask = np.zeros((n,), dtype=bool)
    tangent_fail_segments: List[Dict[str, Any]] = []
    if direction_model_status == "ok":
        axis_xyz = np.asarray(tol_tangent_axis_xyz, dtype=np.float64).reshape(-1, 3)
        half_angle = np.asarray(tol_tangent_half_angle_deg, dtype=np.float64).reshape(-1)
        if len(axis_xyz) >= 2 and len(half_angle) >= 2:
            if len(axis_xyz) != n:
                axis_xyz = resample_curve_by_count(axis_xyz, n)
                axis_xyz = _normalize_rows(axis_xyz)
            else:
                axis_xyz = _normalize_rows(axis_xyz)
            if len(half_angle) != n:
                half_angle = np.interp(
                    np.linspace(0.0, 1.0, n),
                    np.linspace(0.0, 1.0, max(2, int(len(half_angle)))),
                    half_angle,
                )
            tangent_half_angle_deg = np.asarray(half_angle, dtype=np.float64).reshape(-1)
            tangent_window = float(tol_tangent_window_mm) if finite(tol_tangent_window_mm) else 0.0
            defect_tan = estimate_curve_tangents_windowed(defect_n, unit_scale_f, window_mm=tangent_window)
            dots = np.einsum("ni,ni->n", defect_tan, axis_xyz)
            dots = np.clip(dots, -1.0, 1.0)
            tangent_deg = np.degrees(np.arccos(dots))
            tangent_ratio = tangent_deg / np.maximum(tangent_half_angle_deg, 1e-6)
            tangent_margin = tangent_ratio - 1.0
            tangent_score = np.clip(tangent_ratio - 1.0, 0.0, 1.0)
            tangent_over_mask = np.isfinite(tangent_ratio) & (tangent_ratio > 1.0 + 1e-9)
            decision_params = _resolve_tangent_decision_params(tangent_window)
            tangent_fail_window_mm = float(decision_params["fail_window_mm"])
            tangent_fail_min_ratio = float(decision_params["fail_min_ratio"])
            tangent_fail_min_span_mm = float(decision_params["fail_min_span_mm"])
            window_stats = _windowed_true_mask_stats(tangent_over_mask, nominal_n, unit_scale_f, tangent_fail_window_mm)
            tangent_window_over_ratio_arr = np.asarray(window_stats.get("over_ratio_arr", np.zeros((n,), dtype=np.float64)), dtype=np.float64).reshape(-1)
            tangent_window_over_span_mm_arr = np.asarray(window_stats.get("over_span_mm_arr", np.zeros((n,), dtype=np.float64)), dtype=np.float64).reshape(-1)
            if len(tangent_window_over_ratio_arr) != n:
                tangent_window_over_ratio_arr = np.interp(
                    np.linspace(0.0, 1.0, n),
                    np.linspace(0.0, 1.0, max(2, int(len(tangent_window_over_ratio_arr)))),
                    tangent_window_over_ratio_arr if len(tangent_window_over_ratio_arr) else np.zeros((2,), dtype=np.float64),
                )
            if len(tangent_window_over_span_mm_arr) != n:
                tangent_window_over_span_mm_arr = np.interp(
                    np.linspace(0.0, 1.0, n),
                    np.linspace(0.0, 1.0, max(2, int(len(tangent_window_over_span_mm_arr)))),
                    tangent_window_over_span_mm_arr if len(tangent_window_over_span_mm_arr) else np.zeros((2,), dtype=np.float64),
                )
            tangent_window_fail_mask = (
                np.isfinite(tangent_window_over_ratio_arr)
                & np.isfinite(tangent_window_over_span_mm_arr)
                & (tangent_window_over_ratio_arr >= tangent_fail_min_ratio - 1e-9)
                & (tangent_window_over_span_mm_arr >= tangent_fail_min_span_mm - 1e-9)
            )
            tangent_decision_mask = _expand_window_centers_to_mask(
                tangent_window_fail_mask,
                window_stats.get("left_idx_arr", np.zeros((0,), dtype=np.int64)),
                window_stats.get("right_idx_arr", np.zeros((0,), dtype=np.int64)),
                n,
                limit_mask=tangent_over_mask,
            )
            tangent_fail_segments = _true_segments_from_mask(tangent_decision_mask, window_stats.get("s_mm", np.zeros((0,), dtype=np.float64)))
            tangent_score = np.where(tangent_decision_mask, tangent_score, 0.0)
            tangent_status = "fail" if np.any(tangent_window_fail_mask) else "pass"
        else:
            direction_model_status = "model_insufficient"

    score_arr = np.maximum(lateral_score, tangent_score) if use_direction_score else lateral_score
    point_score_raw_arr = np.maximum(lateral_margin, tangent_margin) if use_direction_score else lateral_margin
    point_score_clipped_arr = np.asarray(score_arr, dtype=np.float64)

    ab_params = _resolve_compare_readout_ab_params()
    slack_metrics = _compute_bounded_local_slack_lateral_metrics(
        nominal_n=nominal_n,
        defect_n=defect_n,
        tan=tan,
        tol_lat=tol_lat,
        unit_scale_f=unit_scale_f,
        fixed_ratio_arr=ratio,
        slack_radius_steps=int(ab_params["slack_radius_steps"]),
    )
    point_score_raw_fixed_arr = np.asarray(point_score_raw_arr, dtype=np.float64)
    point_score_clipped_fixed_arr = np.asarray(point_score_clipped_arr, dtype=np.float64)
    point_score_raw_slack_arr = np.asarray(slack_metrics["margin_arr"], dtype=np.float64)
    point_score_clipped_slack_arr = np.asarray(slack_metrics["score_arr"], dtype=np.float64)

    compare_mode_ab = str(ab_params["compare_mode"])
    readout_mode_ab = str(ab_params["readout_mode"])
    monotone_metrics: Optional[Dict[str, Any]] = None
    if compare_mode_ab == "bounded_local_slack_v1":
        ab_point_score_raw_arr = point_score_raw_slack_arr
        ab_point_score_clipped_arr = point_score_clipped_slack_arr
        compare_metrics_shadow = slack_metrics
    elif compare_mode_ab == "banded_monotone_slack_v1":
        monotone_metrics = _compute_banded_monotone_slack_lateral_metrics(
            nominal_n=nominal_n,
            defect_n=defect_n,
            tan=tan,
            tol_lat=tol_lat,
            unit_scale_f=unit_scale_f,
            fixed_ratio_arr=ratio,
            slack_radius_steps=int(ab_params["slack_radius_steps"]),
        )
        ab_point_score_raw_arr = np.asarray(monotone_metrics["margin_arr"], dtype=np.float64)
        ab_point_score_clipped_arr = np.asarray(monotone_metrics["score_arr"], dtype=np.float64)
        compare_metrics_shadow = monotone_metrics
    else:
        ab_point_score_raw_arr = point_score_raw_fixed_arr
        ab_point_score_clipped_arr = point_score_clipped_fixed_arr
        compare_metrics_shadow = slack_metrics

    needle_readout = _compute_needle_readout_metrics(
        ab_point_score_raw_arr,
        readout_mode=readout_mode_ab,
        topk_frac=float(ab_params["topk_frac"]),
        topk_min_points=int(ab_params["topk_min_points"]),
    )
    ab_shadow_verdict = _compute_ab_shadow_verdict(
        needle_score_raw=needle_readout.get("needle_score_raw", float("nan")),
        tip_short_ratio=tip_short_ratio,
        len_short_ratio=len_short_ratio,
    )
    slack_gain = float(np.nanmax(point_score_raw_fixed_arr) - np.nanmax(ab_point_score_raw_arr)) if (
        np.any(np.isfinite(point_score_raw_fixed_arr)) and np.any(np.isfinite(ab_point_score_raw_arr))
    ) else float("nan")
    tangent_over_mask = np.isfinite(tangent_ratio) & (tangent_ratio > 1.0 + 1e-9)
    max_tangent_deg = float(np.nanmax(tangent_deg)) if np.any(np.isfinite(tangent_deg)) else float("nan")
    max_tangent_ratio = float(np.nanmax(tangent_ratio)) if np.any(np.isfinite(tangent_ratio)) else float("nan")
    tangent_window_over_ratio_max = (
        float(np.nanmax(tangent_window_over_ratio_arr)) if np.any(np.isfinite(tangent_window_over_ratio_arr)) else float("nan")
    )
    tangent_window_over_span_mm_max = (
        float(np.nanmax(tangent_window_over_span_mm_arr)) if np.any(np.isfinite(tangent_window_over_span_mm_arr)) else float("nan")
    )
    tangent_decision_span_mm = float(_longest_true_span_mm(tangent_decision_mask, nominal_n, unit_scale_f))

    metrics = {
        "unit_scale": float(unit_scale_f),
        "compare_alignment_mode": "fixed_station",
        "max_lateral_mm": float(max_lateral_mm),
        "max_lateral_ratio": float(max_lateral_ratio),
        "tip_short_mm": float(tip_short_mm),
        "tip_short_ratio": float(tip_short_ratio),
        "len_short_mm": float(len_short_mm),
        "len_short_ratio": float(len_short_ratio),
        "tol_fail_subtype": str(
            classify_tol_fail_subtype(
                max_lateral_ratio=max_lateral_ratio,
                tip_short_ratio=tip_short_ratio,
                len_short_ratio=len_short_ratio,
            )
        ),
        "nominal_length_mm": float(len_nom_mm),
        "defect_length_mm": float(len_def_mm),
        "direction_model_status": str(direction_model_status),
        "tangent_status": str(tangent_status),
        "tangent_decision_mode": str(tangent_decision_mode),
        "tangent_window_mm": float(tol_tangent_window_mm) if finite(tol_tangent_window_mm) else float("nan"),
        "tangent_fail_window_mm": float(tangent_fail_window_mm),
        "tangent_fail_min_ratio": float(tangent_fail_min_ratio),
        "tangent_fail_min_span_mm": float(tangent_fail_min_span_mm),
        "max_tangent_deg": float(max_tangent_deg),
        "max_tangent_ratio": float(max_tangent_ratio),
        "tangent_over_cone_points": int(np.count_nonzero(tangent_over_mask)),
        "tangent_over_cone_span_mm": float(_longest_true_span_mm(tangent_over_mask, nominal_n, unit_scale_f)),
        "tangent_window_over_ratio_max": float(tangent_window_over_ratio_max),
        "tangent_window_over_span_mm_max": float(tangent_window_over_span_mm_max),
        "tangent_fail_window_count": int(np.count_nonzero(tangent_window_fail_mask)),
        "tangent_fail_segment_count": int(len(tangent_fail_segments)),
        "tangent_decision_points": int(np.count_nonzero(tangent_decision_mask)),
        "tangent_decision_span_mm": float(tangent_decision_span_mm),
        "score_mode": "max_lateral_tangent" if use_direction_score and np.any(np.isfinite(tangent_ratio)) else "lateral_only",
        "lateral_mm_arr": lateral_mm.astype(float).tolist(),
        "tol_lateral_mm_arr": tol_lat.astype(float).tolist(),
        "lateral_ratio_arr": ratio.astype(float).tolist(),
        "lateral_margin_arr": lateral_margin.astype(float).tolist(),
        "lateral_score_arr": lateral_score.astype(float).tolist(),
        "point_score_raw_arr": point_score_raw_arr.astype(float).tolist(),
        "point_score_clipped_arr": point_score_clipped_arr.astype(float).tolist(),
        "point_score_raw_max": float(np.nanmax(point_score_raw_arr)) if np.any(np.isfinite(point_score_raw_arr)) else float("nan"),
        "point_score_clipped_max": float(np.nanmax(point_score_clipped_arr)) if np.any(np.isfinite(point_score_clipped_arr)) else float("nan"),
        "point_score_fixed_raw_arr": point_score_raw_fixed_arr.astype(float).tolist(),
        "point_score_fixed_clipped_arr": point_score_clipped_fixed_arr.astype(float).tolist(),
        "point_score_slack_raw_arr": point_score_raw_slack_arr.astype(float).tolist(),
        "point_score_slack_clipped_arr": point_score_clipped_slack_arr.astype(float).tolist(),
        "point_score_fixed_raw_max": float(np.nanmax(point_score_raw_fixed_arr)) if np.any(np.isfinite(point_score_raw_fixed_arr)) else float("nan"),
        "point_score_slack_raw_max": float(np.nanmax(point_score_raw_slack_arr)) if np.any(np.isfinite(point_score_raw_slack_arr)) else float("nan"),
        "point_score_monotone_raw_arr": np.asarray(monotone_metrics["margin_arr"], dtype=np.float64).astype(float).tolist() if monotone_metrics else [],
        "point_score_monotone_clipped_arr": np.asarray(monotone_metrics["score_arr"], dtype=np.float64).astype(float).tolist() if monotone_metrics else [],
        "point_score_monotone_raw_max": float(np.nanmax(np.asarray(monotone_metrics["margin_arr"], dtype=np.float64))) if monotone_metrics and np.any(np.isfinite(np.asarray(monotone_metrics["margin_arr"], dtype=np.float64))) else float("nan"),
        "point_score_selected_raw_arr": ab_point_score_raw_arr.astype(float).tolist(),
        "point_score_selected_clipped_arr": ab_point_score_clipped_arr.astype(float).tolist(),
        "point_score_selected_raw_max": float(np.nanmax(ab_point_score_raw_arr)) if np.any(np.isfinite(ab_point_score_raw_arr)) else float("nan"),
        "point_score_selected_clipped_max": float(np.nanmax(ab_point_score_clipped_arr)) if np.any(np.isfinite(ab_point_score_clipped_arr)) else float("nan"),
        "ab_compare_mode": str(compare_mode_ab),
        "ab_readout_mode": str(readout_mode_ab),
        "ab_slack_radius_steps": int(ab_params["slack_radius_steps"]),
        "ab_readout_topk_frac": float(ab_params["topk_frac"]),
        "ab_readout_topk_min_points": int(ab_params["topk_min_points"]),
        "ab_shift_steps_arr": np.asarray(compare_metrics_shadow["shift_steps_arr"], dtype=np.int32).astype(int).tolist(),
        "ab_shift_mm_arr": np.asarray(compare_metrics_shadow["shift_mm_arr"], dtype=np.float64).astype(float).tolist(),
        "ab_warp_nonzero_ratio": float(compare_metrics_shadow["warp_nonzero_ratio"]),
        "ab_warp_p90_steps": float(compare_metrics_shadow["warp_p90_steps"]),
        "ab_warp_max_steps": float(compare_metrics_shadow["warp_max_steps"]),
        "ab_warp_p90_mm": float(compare_metrics_shadow["warp_p90_mm"]),
        "ab_warp_max_mm": float(compare_metrics_shadow["warp_max_mm"]),
        "ab_station_step_mm": float(compare_metrics_shadow["station_step_mm"]),
        "ab_slack_gain": float(slack_gain),
        "ab_compare_gain": float(slack_gain),
        "ab_path_cost_mean": float(compare_metrics_shadow.get("path_cost_mean", float("nan")))
        if finite(compare_metrics_shadow.get("path_cost_mean", float("nan")))
        else float("nan"),
        "ab_path_step_deviation_mean": float(compare_metrics_shadow.get("path_step_deviation_mean", float("nan")))
        if finite(compare_metrics_shadow.get("path_step_deviation_mean", float("nan")))
        else float("nan"),
        "ab_path_start_j": float(compare_metrics_shadow.get("path_start_j", float("nan")))
        if finite(compare_metrics_shadow.get("path_start_j", float("nan")))
        else float("nan"),
        "ab_path_end_j": float(compare_metrics_shadow.get("path_end_j", float("nan")))
        if finite(compare_metrics_shadow.get("path_end_j", float("nan")))
        else float("nan"),
        "ab_path_used_fallback": bool(compare_metrics_shadow.get("path_used_fallback", False)),
        "needle_score_raw": float(needle_readout["needle_score_raw"]),
        "needle_score_clipped": float(needle_readout["needle_score_clipped"]),
        "needle_score_q95": float(needle_readout["needle_score_q95"]),
        "needle_score_topk_mean": float(needle_readout["needle_score_topk_mean"]),
        "needle_score_topk_count": int(needle_readout["needle_score_topk_count"]),
        "ab_shadow_verdict": str(ab_shadow_verdict["verdict"]),
        "ab_shadow_reason": str(ab_shadow_verdict["reason"]),
        "tangent_score_arr": tangent_score.astype(float).tolist(),
        "tangent_ratio_arr": tangent_ratio.astype(float).tolist(),
        "tangent_margin_arr": tangent_margin.astype(float).tolist(),
        "tangent_angle_deg_arr": tangent_deg.astype(float).tolist(),
        "tangent_half_angle_deg_arr": tangent_half_angle_deg.astype(float).tolist(),
        "tangent_window_over_ratio_arr": tangent_window_over_ratio_arr.astype(float).tolist(),
        "tangent_window_over_span_mm_arr": tangent_window_over_span_mm_arr.astype(float).tolist(),
        "tangent_window_fail_mask_arr": tangent_window_fail_mask.astype(bool).tolist(),
        "tangent_decision_mask_arr": tangent_decision_mask.astype(bool).tolist(),
        "tangent_fail_segments": tangent_fail_segments,
        "tangent_axis_xyz_arr": axis_xyz.astype(float).tolist() if 'axis_xyz' in locals() else [],
        "defect_tangent_xyz_arr": defect_tan.astype(float).tolist() if 'defect_tan' in locals() else [],
    }
    return np.asarray(score_arr, dtype=float), metrics, nominal_n, defect_n

def compute_baseline_verdict_v1(
    *,
    tol_status: Any,
    tangent_status: Any,
    fit_status: Any = "",
    metrics: Optional[Mapping[str, Any]] = None,
    reg_status: Any = "",
    paired: Any = None,
    icp_rmse_mm: Any = None,
    icp_fitness: Any = None,
    base_bias_final_norm_mm: Any = None,
    mode: str = "baseline_v1",
    pass_guard_ratio: float = 0.05,
    fail_guard_ratio: float = 0.10,
    hard_fail_margin_ratio: float = 0.20,
    min_paired: int = 18,
    min_icp_fitness: float = 0.0,
    max_icp_rmse_mm: float = 0.50,
    max_base_bias_mm: float = 0.05,
) -> Dict[str, Any]:
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    tol = str(tol_status).strip() or "unknown"
    tangent = str(tangent_status).strip() or "unknown"
    fit = str(fit_status).strip()
    reg = str(reg_status).strip()
    mode_n = str(mode).strip().lower() or "baseline_v1"
    compare_alignment_mode = str(metrics_map.get("compare_alignment_mode", "fixed_station") or "fixed_station")

    def _worst_ratio(keys: Sequence[str]) -> float:
        vals: List[float] = []
        for key in keys:
            v = metrics_map.get(key, float("nan"))
            if finite(v):
                vals.append(float(v))
        return float(max(vals)) if vals else float("nan")

    def _unique_flags(*groups: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for group in groups:
            for item in group:
                s = str(item).strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
        return out

    pos_ratio = _worst_ratio(("max_lateral_ratio", "tip_short_ratio", "len_short_ratio"))
    dir_ratio = _worst_ratio(("max_tangent_ratio",))
    pos_margin = float(pos_ratio - 1.0) if finite(pos_ratio) else float("nan")
    dir_margin = float(dir_ratio - 1.0) if finite(dir_ratio) else float("nan")

    if mode_n == "legacy":
        legacy = tol if tol else "unknown"
        confidence = "medium" if legacy in ("pass", "fail") else "low"
        return {
            "baseline_verdict": legacy,
            "baseline_reason": "legacy_tol_status",
            "uncertainty_flags": [],
            "position_margin_ratio": float(pos_margin),
            "direction_margin_ratio": float(dir_margin),
            "baseline_confidence_bucket": str(confidence),
            "compare_alignment_mode": str(compare_alignment_mode),
        }

    passthrough = {"missing", "model_insufficient", "reg_failed", "unsupported_method"}
    if tol in passthrough:
        return {
            "baseline_verdict": tol,
            "baseline_reason": tol,
            "uncertainty_flags": [],
            "position_margin_ratio": float(pos_margin),
            "direction_margin_ratio": float(dir_margin),
            "baseline_confidence_bucket": "low",
            "compare_alignment_mode": str(compare_alignment_mode),
        }

    hard_flags: List[str] = []
    soft_flags: List[str] = []
    if fit == "high_risk":
        hard_flags.append("fit_high_risk")
    if reg and reg.lower() not in ("ok", "success"):
        hard_flags.append(f"reg_{reg.lower()}")

    try:
        paired_i = int(paired) if paired is not None and str(paired).strip() != "" else None
    except Exception:
        paired_i = None
    if paired_i is not None and int(min_paired) > 0 and paired_i < int(min_paired):
        hard_flags.append("low_pairing")

    if finite(icp_rmse_mm) and float(max_icp_rmse_mm) > 0.0 and float(icp_rmse_mm) > float(max_icp_rmse_mm):
        soft_flags.append("reg_rmse_high")
    if (
        finite(icp_fitness)
        and float(min_icp_fitness) > 0.0
        and float(icp_fitness) < float(min_icp_fitness)
    ):
        soft_flags.append("reg_fitness_low")
    if (
        finite(base_bias_final_norm_mm)
        and float(max_base_bias_mm) > 0.0
        and float(base_bias_final_norm_mm) > float(max_base_bias_mm)
    ):
        soft_flags.append("base_bias_high")

    near_position = finite(pos_margin) and (-float(pass_guard_ratio) < float(pos_margin) < float(fail_guard_ratio))
    near_direction = (
        tangent in ("pass", "fail")
        and finite(dir_margin)
        and (-float(pass_guard_ratio) < float(dir_margin) < float(fail_guard_ratio))
    )
    disagree = tol in ("pass", "fail") and tangent in ("pass", "fail") and tol != tangent

    clear_pass = (
        tol == "pass"
        and tangent in ("pass", "model_insufficient")
        and finite(pos_margin)
        and float(pos_margin) <= -float(pass_guard_ratio)
        and (
            tangent == "model_insufficient"
            or not finite(dir_margin)
            or float(dir_margin) <= -float(pass_guard_ratio)
        )
    )
    clear_fail = (
        tol == "fail"
        and tangent == "fail"
        and finite(pos_margin)
        and float(pos_margin) >= float(fail_guard_ratio)
        and (not finite(dir_margin) or float(dir_margin) >= float(fail_guard_ratio))
    )
    hard_pos_fail = (
        tol == "fail"
        and tangent != "pass"
        and finite(pos_margin)
        and float(pos_margin) >= float(hard_fail_margin_ratio)
    )

    verdict = "need_rescan"
    reason = "insufficient_evidence"
    flags: List[str] = []
    if hard_flags:
        verdict = "need_rescan"
        reason = str(hard_flags[0])
        flags = _unique_flags(hard_flags, soft_flags)
    elif tol == "need_rescan" or tangent == "need_rescan":
        verdict = "need_rescan"
        reason = "upstream_need_rescan"
        flags = _unique_flags(soft_flags)
    elif hard_pos_fail:
        verdict = "fail"
        reason = "position_hard_fail"
        flags = _unique_flags(soft_flags)
    elif clear_fail and not soft_flags:
        verdict = "fail"
        reason = "position_direction_clear_fail"
    elif clear_pass and not soft_flags and not near_direction:
        verdict = "pass"
        reason = "clear_pass"
        if tangent == "model_insufficient":
            flags = ["tangent_model_insufficient"]
    elif disagree:
        verdict = "need_rescan"
        reason = "position_direction_disagree"
        flags = _unique_flags(["chain_disagree"], soft_flags)
    elif soft_flags:
        verdict = "need_rescan"
        reason = "soft_quality_gate"
        flags = _unique_flags(soft_flags)
    elif near_position or near_direction:
        verdict = "need_rescan"
        reason = "near_threshold"
        flags = _unique_flags(["near_threshold"])
    elif tol == "fail":
        verdict = "need_rescan"
        reason = "position_fail_not_clear"
        flags = _unique_flags([f"tangent_{tangent}" if tangent not in ("", "unknown") else ""], soft_flags)
    elif tol == "pass" and tangent in ("pass", "model_insufficient"):
        verdict = "need_rescan"
        reason = "pass_not_clear"
        flags = _unique_flags(["tangent_model_insufficient" if tangent == "model_insufficient" else ""], soft_flags)

    if verdict == "need_rescan" or verdict in passthrough:
        confidence = "low"
    elif verdict == "pass" and tangent == "model_insufficient":
        confidence = "medium"
    else:
        confidence = "high"
    return {
        "baseline_verdict": str(verdict),
        "baseline_reason": str(reason),
        "uncertainty_flags": list(flags),
        "position_margin_ratio": float(pos_margin),
        "direction_margin_ratio": float(dir_margin),
        "baseline_confidence_bucket": str(confidence),
        "compare_alignment_mode": str(compare_alignment_mode),
    }

FORMAL_DECISION_LOCAL_RULE_P4 = "p4_no_rescan_contiguous_prior_v1"

FORMAL_DECISION_LOCAL_PASS_MAX_SPAN_MM = 0.2

FORMAL_DECISION_LOCAL_PASS_MAX_RUNS = 1

FORMAL_DECISION_LOCAL_PASS_MAX_BINS = 2

FORMAL_DECISION_LOCAL_PASS_MAX_NEEDLE_SCORE = 0.1

FORMAL_DECISION_LOCAL_PASS_MAX_PEAK_SCORE = 0.6

FORMAL_DECISION_LOCAL_CONTIG_MAX_SPAN_MM = 1.2

FORMAL_DECISION_LOCAL_CONTIG_MAX_RUNS = 2

FORMAL_DECISION_LOCAL_CONTIG_MIN_DOM_FRAC = 0.8

FORMAL_DECISION_SEGMENT_STEP_MM_FALLBACK = 0.2

def _coerce_float_or_nan(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _coerce_int_or_zero(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0

def compute_selected_score_run_features_v1(
    *,
    point_score_selected_raw_arr: Any,
    station_step_mm: Any,
    positive_threshold: float = 0.0,
) -> Dict[str, Any]:
    arr = np.asarray(point_score_selected_raw_arr if point_score_selected_raw_arr is not None else [], dtype=np.float64).reshape(-1)
    step_mm_f = _coerce_float_or_nan(station_step_mm)
    if not np.isfinite(step_mm_f) or step_mm_f <= 0.0:
        step_mm_f = float(FORMAL_DECISION_SEGMENT_STEP_MM_FALLBACK)

    pos_idx = np.flatnonzero(np.isfinite(arr) & (arr > float(positive_threshold)))
    if pos_idx.size <= 0:
        return {
            "pred_pos_bins": 0,
            "pred_run_count": 0,
            "pred_max_run_bins": 0,
            "pred_max_run_span_mm": 0.0,
            "pred_dominant_run_frac": 0.0,
            "pred_singleton_run_count": 0,
            "pred_short_run_bins": 0,
            "pred_short_run_frac": 0.0,
            "pred_station_step_mm": float(step_mm_f),
        }

    gaps = np.where(np.diff(pos_idx) > 1)[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, pos_idx.size - 1]
    run_bins: List[int] = []
    singleton_run_count = 0
    short_run_bins = 0
    max_run_bins = 0
    max_run_span_mm = 0.0
    for start, end in zip(starts, ends):
        run = pos_idx[start : end + 1]
        bins = int(run.size)
        span_mm = float((run[-1] - run[0]) * step_mm_f)
        run_bins.append(bins)
        if bins == 1:
            singleton_run_count += 1
        if bins <= 3:
            short_run_bins += bins
        if bins > max_run_bins:
            max_run_bins = bins
            max_run_span_mm = span_mm
    pos_bins = int(pos_idx.size)
    return {
        "pred_pos_bins": int(pos_bins),
        "pred_run_count": int(len(run_bins)),
        "pred_max_run_bins": int(max_run_bins),
        "pred_max_run_span_mm": float(max_run_span_mm),
        "pred_dominant_run_frac": float(max_run_bins / pos_bins) if pos_bins > 0 else 0.0,
        "pred_singleton_run_count": int(singleton_run_count),
        "pred_short_run_bins": int(short_run_bins),
        "pred_short_run_frac": float(short_run_bins / pos_bins) if pos_bins > 0 else 0.0,
        "pred_station_step_mm": float(step_mm_f),
    }

def compute_positive_event_features_v1(
    *,
    point_score_selected_raw_arr: Any,
    station_step_mm: Any,
    positive_threshold: float = 0.0,
    monotone_eps: float = 1e-6,
) -> Dict[str, Any]:
    arr = np.asarray(point_score_selected_raw_arr if point_score_selected_raw_arr is not None else [], dtype=np.float64).reshape(-1)
    step_mm_f = _coerce_float_or_nan(station_step_mm)
    if not np.isfinite(step_mm_f) or step_mm_f <= 0.0:
        step_mm_f = float(FORMAL_DECISION_SEGMENT_STEP_MM_FALLBACK)

    finite_mask = np.isfinite(arr)
    pos_mask = finite_mask & (arr > float(positive_threshold))
    pos_idx = np.flatnonzero(pos_mask)
    pos_vals = np.maximum(arr[pos_mask] - float(positive_threshold), 0.0)
    total_support_mm = float(pos_idx.size * step_mm_f)
    total_area_mm = float(np.sum(pos_vals) * step_mm_f) if pos_vals.size > 0 else 0.0

    default_out = {
        "positive_event_count": 0,
        "positive_total_bins": int(pos_idx.size),
        "positive_total_support_mm": float(total_support_mm),
        "positive_total_area_mm": float(total_area_mm),
        "dominant_event_rank": -1,
        "dominant_event_start_idx": -1,
        "dominant_event_end_idx": -1,
        "dominant_event_bins": 0,
        "dominant_event_span_mm": 0.0,
        "dominant_event_support_mm": 0.0,
        "dominant_event_area_mm": 0.0,
        "dominant_event_peak_score": 0.0,
        "dominant_event_mean_score": 0.0,
        "dominant_event_start_score": 0.0,
        "dominant_event_end_score": 0.0,
        "dominant_event_monotone_forward_frac": 0.0,
        "dominant_event_monotone_reverse_frac": 0.0,
        "dominant_event_monotone_best_frac": 0.0,
        "dominant_event_return_frac": 0.0,
        "dominant_event_peak_at_edge": 0,
    }
    if pos_idx.size <= 0:
        return default_out

    gaps = np.where(np.diff(pos_idx) > 1)[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, pos_idx.size - 1]
    runs: List[Dict[str, Any]] = []
    eps = max(float(monotone_eps), 0.0)
    for rank, (start, end) in enumerate(zip(starts, ends)):
        run = pos_idx[start : end + 1]
        if run.size <= 0:
            continue
        vals = np.maximum(arr[run] - float(positive_threshold), 0.0)
        bins = int(run.size)
        span_mm = float((run[-1] - run[0]) * step_mm_f) if bins >= 2 else 0.0
        support_mm = float(bins * step_mm_f)
        area_mm = float(np.sum(vals) * step_mm_f)
        peak = float(np.max(vals)) if vals.size > 0 else 0.0
        mean = float(np.mean(vals)) if vals.size > 0 else 0.0
        start_score = float(vals[0]) if vals.size > 0 else 0.0
        end_score = float(vals[-1]) if vals.size > 0 else 0.0
        if vals.size >= 2:
            diffs = np.diff(vals)
            forward_frac = float(np.mean(diffs >= -eps))
            reverse_frac = float(np.mean(diffs <= eps))
            monotone_best = float(max(forward_frac, reverse_frac))
            abs_diff = float(np.sum(np.abs(diffs)))
            if abs_diff > 1e-12:
                return_frac = float(np.sum(np.maximum(-diffs, 0.0)) / abs_diff)
                return_frac = min(return_frac, float(np.sum(np.maximum(diffs, 0.0)) / abs_diff))
            else:
                return_frac = 0.0
        else:
            forward_frac = 1.0
            reverse_frac = 1.0
            monotone_best = 1.0
            return_frac = 0.0
        peak_at_edge = int(1 if peak <= 0.0 + 1e-12 or abs(start_score - peak) <= 1e-12 or abs(end_score - peak) <= 1e-12 else 0)
        runs.append(
            {
                "rank": int(rank),
                "start_idx": int(run[0]),
                "end_idx": int(run[-1]),
                "bins": int(bins),
                "span_mm": float(span_mm),
                "support_mm": float(support_mm),
                "area_mm": float(area_mm),
                "peak_score": float(peak),
                "mean_score": float(mean),
                "start_score": float(start_score),
                "end_score": float(end_score),
                "monotone_forward_frac": float(forward_frac),
                "monotone_reverse_frac": float(reverse_frac),
                "monotone_best_frac": float(monotone_best),
                "return_frac": float(return_frac),
                "peak_at_edge": int(peak_at_edge),
            }
        )

    if not runs:
        return default_out

    dominant = max(
        runs,
        key=lambda r: (
            float(r["area_mm"]),
            float(r["peak_score"]),
            float(r["support_mm"]),
            -int(r["rank"]),
        ),
    )
    out = dict(default_out)
    out.update(
        {
            "positive_event_count": int(len(runs)),
            "positive_total_bins": int(pos_idx.size),
            "positive_total_support_mm": float(total_support_mm),
            "positive_total_area_mm": float(total_area_mm),
            "dominant_event_rank": int(dominant["rank"]),
            "dominant_event_start_idx": int(dominant["start_idx"]),
            "dominant_event_end_idx": int(dominant["end_idx"]),
            "dominant_event_bins": int(dominant["bins"]),
            "dominant_event_span_mm": float(dominant["span_mm"]),
            "dominant_event_support_mm": float(dominant["support_mm"]),
            "dominant_event_area_mm": float(dominant["area_mm"]),
            "dominant_event_peak_score": float(dominant["peak_score"]),
            "dominant_event_mean_score": float(dominant["mean_score"]),
            "dominant_event_start_score": float(dominant["start_score"]),
            "dominant_event_end_score": float(dominant["end_score"]),
            "dominant_event_monotone_forward_frac": float(dominant["monotone_forward_frac"]),
            "dominant_event_monotone_reverse_frac": float(dominant["monotone_reverse_frac"]),
            "dominant_event_monotone_best_frac": float(dominant["monotone_best_frac"]),
            "dominant_event_return_frac": float(dominant["return_frac"]),
            "dominant_event_peak_at_edge": int(dominant["peak_at_edge"]),
        }
    )
    return out

def _evaluate_formal_local_override_v1(
    *,
    base_pred: str,
    local_override_rule: str,
    local_segment_fields: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str]:
    rule_n = str(local_override_rule or "").strip().lower()
    if not rule_n:
        return "clear", "clear"
    if rule_n != FORMAL_DECISION_LOCAL_RULE_P4:
        return "", "unsupported_local_rule"
    if str(base_pred).strip() != "fail":
        return "clear", "base_not_fail"

    fields = dict(local_segment_fields) if isinstance(local_segment_fields, Mapping) else {}
    tol_fail_subtype = str(fields.get("tol_fail_subtype", "")).strip()
    ab_shadow_reason = str(fields.get("ab_shadow_reason", "")).strip()
    if tol_fail_subtype != "lateral-only" or ab_shadow_reason != "point_readout_positive":
        return "clear", "not_main_chain_candidate"

    pred_pos_bins = _coerce_int_or_zero(fields.get("pred_pos_bins", 0))
    if pred_pos_bins <= 0:
        return "clear", "no_positive_support"

    pred_span_mm = _coerce_float_or_nan(fields.get("pred_max_run_span_mm", float("nan")))
    pred_run_count = _coerce_int_or_zero(fields.get("pred_run_count", 0))
    pred_dom_frac = _coerce_float_or_nan(fields.get("pred_dominant_run_frac", float("nan")))
    needle_score_raw = _coerce_float_or_nan(fields.get("needle_score_raw", float("nan")))
    peak_score_raw = _coerce_float_or_nan(fields.get("point_score_selected_raw_max", float("nan")))

    ultra_short = (
        np.isfinite(pred_span_mm)
        and pred_span_mm <= float(FORMAL_DECISION_LOCAL_PASS_MAX_SPAN_MM)
        and pred_run_count <= int(FORMAL_DECISION_LOCAL_PASS_MAX_RUNS)
        and pred_pos_bins <= int(FORMAL_DECISION_LOCAL_PASS_MAX_BINS)
        and np.isfinite(needle_score_raw)
        and 0.0 <= needle_score_raw <= float(FORMAL_DECISION_LOCAL_PASS_MAX_NEEDLE_SCORE)
        and np.isfinite(peak_score_raw)
        and peak_score_raw <= float(FORMAL_DECISION_LOCAL_PASS_MAX_PEAK_SCORE)
    )
    if ultra_short:
        return "pass", "ultra_short_low_score_pass"

    contiguous_short = (
        np.isfinite(pred_span_mm)
        and pred_span_mm <= float(FORMAL_DECISION_LOCAL_CONTIG_MAX_SPAN_MM)
        and pred_run_count <= int(FORMAL_DECISION_LOCAL_CONTIG_MAX_RUNS)
        and np.isfinite(pred_dom_frac)
        and pred_dom_frac >= float(FORMAL_DECISION_LOCAL_CONTIG_MIN_DOM_FRAC)
    )
    if contiguous_short:
        return "pass", "short_segment_contiguous_prior_pass"
    return "clear", "clear"

def compute_formal_decision_shadow_v1(
    *,
    base_pred: Any,
    gate_source_fields: Optional[Mapping[str, Any]] = None,
    gate_rule: str = "global_only_v1",
    min_paired: int = 18,
    local_override_rule: str = "",
    local_segment_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    fields = dict(gate_source_fields) if isinstance(gate_source_fields, Mapping) else {}
    base = str(base_pred).strip()
    gate_rule_n = str(gate_rule or "").strip().lower() or "global_only_v1"
    local_rule_n = str(local_override_rule or "").strip().lower()
    policy_name = (
        f"{gate_rule_n}__{local_rule_n}__ab_shadow_verdict"
        if local_rule_n
        else f"{gate_rule_n}__ab_shadow_verdict"
    )
    rule_version = "formal_decision_shadow_v2" if local_rule_n else "formal_decision_shadow_v1"

    trigger_hits: List[str] = []

    def _add_hit(text: str) -> None:
        s = str(text).strip()
        if s and s not in trigger_hits:
            trigger_hits.append(s)

    reg_status = str(fields.get("registration_status", "")).strip().lower()
    if reg_status and reg_status not in ("ok", "success"):
        _add_hit(f"registration:{reg_status}")

    try:
        paired_i = int(fields.get("paired_count", 0))
    except Exception:
        paired_i = 0
    if int(min_paired) > 0 and paired_i < int(min_paired):
        _add_hit("pairing:low_pairing")

    for side in ("start", "end"):
        key = f"triclass_{side}_risk_hit_min_n_slab"
        hit_val = fields.get(key, float("nan"))
        if finite(hit_val) and float(hit_val) > 0.5:
            _add_hit(f"{side}:min_n_slab<n_th")

    if gate_rule_n != "global_only_v1":
        return {
            "formal_decision_rule_version": str(rule_version),
            "formal_decision_policy_shadow": str(policy_name),
            "formal_decision_base_source_shadow": "ab_shadow_verdict",
            "formal_decision_base_pred_shadow": str(base),
            "formal_decision_gate_rule_shadow": str(gate_rule_n),
            "formal_decision_gate_status_shadow": "",
            "formal_decision_gate_reason_shadow": "unsupported_gate_rule",
            "formal_decision_gate_hit_list_shadow": [],
            "formal_decision_gate_hit_text_shadow": "",
            "formal_decision_local_rule_shadow": str(local_rule_n),
            "formal_decision_local_status_shadow": "",
            "formal_decision_local_reason_shadow": "",
            "formal_decision_verdict_shadow": "",
            "formal_decision_reason_shadow": "unsupported_gate_rule",
        }

    need_rescan = bool(trigger_hits)
    gate_status = "need_rescan" if need_rescan else "clear"
    gate_reason = "global_hard" if need_rescan else "clear"
    local_status, local_reason = _evaluate_formal_local_override_v1(
        base_pred=base,
        local_override_rule=local_rule_n,
        local_segment_fields=local_segment_fields,
    )

    if need_rescan:
        verdict = "need_rescan"
        reason = "quality_gate_global_hard"
    elif local_status == "pass":
        verdict = "pass"
        reason = str(local_reason)
    elif base in {"pass", "fail"}:
        verdict = str(base)
        reason = f"base_{base}"
    else:
        verdict = ""
        reason = "base_nonbinary"

    return {
        "formal_decision_rule_version": str(rule_version),
        "formal_decision_policy_shadow": str(policy_name),
        "formal_decision_base_source_shadow": "ab_shadow_verdict",
        "formal_decision_base_pred_shadow": str(base),
        "formal_decision_gate_rule_shadow": str(gate_rule_n),
        "formal_decision_gate_status_shadow": str(gate_status),
        "formal_decision_gate_reason_shadow": str(gate_reason),
        "formal_decision_gate_hit_list_shadow": list(trigger_hits),
        "formal_decision_gate_hit_text_shadow": ",".join(trigger_hits),
        "formal_decision_local_rule_shadow": str(local_rule_n),
        "formal_decision_local_status_shadow": str(local_status),
        "formal_decision_local_reason_shadow": str(local_reason),
        "formal_decision_verdict_shadow": str(verdict),
        "formal_decision_reason_shadow": str(reason),
    }
