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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from eval_sample_contract_v1 import build_formal_sample_contract, build_task_subset

try:
    from .detect_tolerance_interactive_1_11 import (
        TOLERANCE_JSON_PATH,
        _analyze_tolerance_pair,
        _centerline_extract_config,
        _safe_float,
        compute_board_transform,
        extract_centerlines_from_pcd,
        load_centerline_tolerance,
        match_needles_by_position,
        resolve_tolerance_reference_paths,
        transform_needle_infos,
    )
    from .eval_io import REPO_ROOT, _safe_int, _write_csv, _write_json
    from .gt_native_loader import (
        GT_ROOT_DEFAULT,
        LOGIC_GT_SAMPLE_CACHE_ROOT,
        _build_reference_infos,
        authoritative_gt_paths,
        load_gt_authoritative_by_manifest_rows,
        validate_manifest_rows_against_authoritative_gt,
    )
except ImportError:
    from detect_tolerance_interactive_1_11 import (
        TOLERANCE_JSON_PATH,
        _analyze_tolerance_pair,
        _centerline_extract_config,
        _safe_float,
        compute_board_transform,
        extract_centerlines_from_pcd,
        load_centerline_tolerance,
        match_needles_by_position,
        resolve_tolerance_reference_paths,
        transform_needle_infos,
    )
    from eval_io import REPO_ROOT, _safe_int, _write_csv, _write_json
    from gt_native_loader import (
        GT_ROOT_DEFAULT,
        LOGIC_GT_SAMPLE_CACHE_ROOT,
        _build_reference_infos,
        authoritative_gt_paths,
        load_gt_authoritative_by_manifest_rows,
        validate_manifest_rows_against_authoritative_gt,
    )


PROTOCOL_DIR = REPO_ROOT / "jiance" / "paper_experiment" / "tolerance_band_v1_protocol"
DEFAULT_MANIFEST_ROLE = "mixed"
DEFAULT_POINT_THRESHOLD = 0.0
DEFAULT_GEO_PRED_FIELD = "formal_decision_verdict"
DEFAULT_GT_SOURCE_MODE = "prepared_only"


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


def _read_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_match_path(path_like: Any) -> str:
    text = str(path_like or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(str(Path(text).resolve())))


def _latest_artifact(filename: str) -> Path:
    runs_root = REPO_ROOT / "debug_out" / "runs"
    candidates = sorted(runs_root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {filename} found under: {runs_root}")
    return candidates[0].resolve()


def _resolve_per_slot_csv_from_matching_detect_run(manifest_path: Path) -> Dict[str, Any]:
    runs_root = (REPO_ROOT / "debug_out" / "runs").resolve()
    detect_script_path = (REPO_ROOT / "jiance" / "tolerance_band_v1" / "detect_tolerance_on_testset.py").resolve()
    manifest_norm = _normalize_match_path(manifest_path)
    candidates: List[Dict[str, Any]] = []
    for run_dir in sorted(runs_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        paths_json = (run_dir / "paths.json").resolve()
        run_meta_json = (run_dir / "run_meta.json").resolve()
        if not paths_json.is_file() or not run_meta_json.is_file():
            continue
        try:
            paths_payload = _read_json_file(paths_json)
            run_meta = _read_json_file(run_meta_json)
        except Exception:
            continue
        script_path = _normalize_match_path(run_meta.get("script_path", ""))
        if script_path != _normalize_match_path(detect_script_path):
            continue
        inputs = dict(paths_payload.get("inputs", {}) or {})
        outputs = dict(paths_payload.get("outputs", {}) or {})
        candidate_manifest = _normalize_match_path(inputs.get("manifest_path", ""))
        candidate_per_slot = Path(str(outputs.get("testset_detection_per_slot_csv", "") or "")).resolve()
        if candidate_manifest != manifest_norm or not candidate_per_slot.is_file():
            continue
        candidates.append(
            {
                "run_id": str(run_meta.get("run_id", "") or run_dir.name),
                "run_dir": str(run_dir),
                "per_slot_csv": str(candidate_per_slot),
                "manifest_path": str(Path(str(inputs.get("manifest_path", "") or "")).resolve()),
            }
        )
    if not candidates:
        raise FileNotFoundError(
            "No compatible detect run found for manifest. "
            f"manifest_path={manifest_path} | "
            f"expected_script={detect_script_path} | "
            "Run detect_tolerance_on_testset.py with this manifest first, "
            "or pass --per_slot_csv explicitly."
        )
    best = candidates[0]
    return {
        "mode": "auto_match_manifest",
        "run_id": str(best.get("run_id", "")),
        "run_dir": str(best.get("run_dir", "")),
        "manifest_path": str(best.get("manifest_path", "")),
        "per_slot_csv": str(best.get("per_slot_csv", "")),
    }


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


def _safe_div(num: float, den: float, *, default: float = float("nan")) -> float:
    if float(den) == 0.0:
        return float(default)
    return float(num) / float(den)


def _resolve_gt_load_plan(*, gt_source_mode: str, emit_point_visual_sidecar: bool) -> Dict[str, Any]:
    mode = str(gt_source_mode or DEFAULT_GT_SOURCE_MODE).strip().lower() or DEFAULT_GT_SOURCE_MODE
    if mode not in {"auto", "live", "prepared_only"}:
        raise ValueError(f"Unsupported gt_source_mode: {gt_source_mode!r}")
    if bool(emit_point_visual_sidecar):
        return {
            "requested_mode": mode,
            "resolved_mode": "live_point_sidecar",
            "reason": (
                "point_sidecar_requires_live_gt_point_support"
                if mode != "prepared_only"
                else "point_sidecar_overrides_prepared_only_default"
            ),
            "include_point_support": True,
            "use_sample_cache": False,
            "require_sample_cache_hit": False,
            "write_sample_cache_on_miss": False,
        }
    if mode == "live":
        return {
            "requested_mode": mode,
            "resolved_mode": "live_logic_geo",
            "reason": "explicit_live",
            "include_point_support": False,
            "use_sample_cache": False,
            "require_sample_cache_hit": False,
            "write_sample_cache_on_miss": False,
        }
    if mode == "prepared_only":
        return {
            "requested_mode": mode,
            "resolved_mode": "prepared_only_logic_geo",
            "reason": "prepared_cache_required",
            "include_point_support": False,
            "use_sample_cache": True,
            "require_sample_cache_hit": True,
            "write_sample_cache_on_miss": False,
        }
    return {
        "requested_mode": mode,
        "resolved_mode": "auto_logic_geo",
        "reason": "auto_prefer_prepared_cache_then_live_fill",
        "include_point_support": False,
        "use_sample_cache": True,
        "require_sample_cache_hit": False,
        "write_sample_cache_on_miss": True,
    }


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


def _f1_from_pr(precision: float, recall: float) -> float:
    if precision != precision or recall != recall:
        return float("nan")
    if float(precision) + float(recall) <= 0.0:
        return float("nan")
    return float(2.0 * precision * recall / (precision + recall))


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, List[Dict[str, Any]]]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_score):
        n = min(len(y_true), len(y_score))
        y_true = y_true[:n]
        y_score = y_score[:n]
    positive_total = int(np.sum(y_true == 1))
    if positive_total <= 0 or len(y_true) <= 0:
        return float("nan"), []

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    tp = np.cumsum(y_true_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_true_sorted == 0).astype(np.float64)
    distinct_idxs = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_idxs, len(y_score_sorted) - 1]
    tp_k = tp[threshold_idxs]
    fp_k = fp[threshold_idxs]
    precision = tp_k / np.maximum(tp_k + fp_k, 1.0)
    recall = tp_k / float(positive_total)
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = float(np.sum((recall - recall_prev) * precision))
    curve_rows: List[Dict[str, Any]] = [{"threshold": "", "precision": 1.0, "recall": 0.0}]
    for idx, prec, rec in zip(threshold_idxs.tolist(), precision.tolist(), recall.tolist()):
        curve_rows.append(
            {
                "threshold": float(y_score_sorted[int(idx)]),
                "precision": float(prec),
                "recall": float(rec),
            }
        )
    return float(ap), curve_rows


def _logic_metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    per_slot_rows: List[Dict[str, Any]] = []
    for row in rows:
        gt_missing = int(row.get("gt_missing", 0))
        pred_missing = int(row.get("pred_missing", 0))
        if gt_missing == 1 and pred_missing == 1:
            tp += 1
            logic_case = "TP_logic"
        elif gt_missing == 0 and pred_missing == 1:
            fp += 1
            logic_case = "FP_logic"
        elif gt_missing == 1 and pred_missing == 0:
            fn += 1
            logic_case = "FN_logic"
        else:
            tn += 1
            logic_case = "TN_logic"
        per_slot_rows.append(
            {
                "sample_uid": str(row.get("sample_uid", row.get("sample_id", "")) or ""),
                "sample_id": str(row.get("sample_id", "")),
                "source_sample_id": str(row.get("source_sample_id", row.get("sample_id", "")) or ""),
                "gt_partition": str(row.get("gt_partition", "") or ""),
                "stable_id": int(row.get("stable_id", -1)),
                "gt_missing": int(gt_missing),
                "pred_missing": int(pred_missing),
                "logic_case": str(logic_case),
                "gt_logic_label": str(row.get("gt_logic_label", "")),
                "pred_pair_status": str(row.get("pred_pair_status", "")),
                "pred_missing_reason": str(row.get("pred_missing_reason", "")),
            }
        )

    precision_logic = _safe_div(tp, tp + fp)
    recall_logic = _safe_div(tp, tp + fn)
    f1_logic = _f1_from_pr(precision_logic, recall_logic)
    recall_present = _safe_div(tn, tn + fp)
    balanced_accuracy_logic = float(np.nanmean([recall_logic, recall_present]))
    precision_present = _safe_div(tn, tn + fn)
    f1_present = _f1_from_pr(precision_present, recall_present)
    macro_f1_logic = float(np.nanmean([f1_logic, f1_present]))
    return {
        "summary": {
            "n_logic": int(len(rows)),
            "tp_logic": int(tp),
            "fp_logic": int(fp),
            "fn_logic": int(fn),
            "tn_logic": int(tn),
            "precision_logic": float(precision_logic),
            "recall_logic": float(recall_logic),
            "f1_logic": float(f1_logic),
            "recall_present": float(recall_present),
            "balanced_accuracy_logic": float(balanced_accuracy_logic),
            "precision_present": float(precision_present),
            "f1_present": float(f1_present),
            "macro_f1_logic": float(macro_f1_logic),
        },
        "rows": per_slot_rows,
    }


def _geo_metrics_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    geo_pred_field: str,
) -> Dict[str, Any]:
    included_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    tp = fp = fn = tn = rej = 0
    abnormal_total = normal_total = abnormal_reject = 0

    for row in rows:
        gt_geo_eligible = int(row.get("gt_geo_eligible", 0))
        if gt_geo_eligible != 1:
            excluded_rows.append(
                {
                    "sample_uid": str(row.get("sample_uid", row.get("sample_id", "")) or ""),
                    "sample_id": str(row.get("sample_id", "")),
                    "source_sample_id": str(row.get("source_sample_id", row.get("sample_id", "")) or ""),
                    "gt_partition": str(row.get("gt_partition", "") or ""),
                    "stable_id": int(row.get("stable_id", -1)),
                    "geo_included": 0,
                    "geo_exclusion_reason": str(row.get("gt_exclusion_reason", "")) or "gt_not_geo_eligible",
                }
            )
            continue

        gt_abnormal = int(row.get("gt_abnormal", 0))
        abnormal_total += gt_abnormal
        normal_total += int(1 - gt_abnormal)
        verdict = str(row.get(geo_pred_field, "")).strip()
        geo_pred_source = str(geo_pred_field)
        geo_reject_reason = ""
        pred_geo_eval_eligible = int(row.get("pred_geo_eval_eligible", row.get("pred_present", 0)))
        if pred_geo_eval_eligible != 1:
            verdict = "need_rescan"
            geo_pred_source = "pred_geo_eval_eligible"
            geo_reject_reason = "pred_missing_no_geometry_support"
        elif verdict not in {"pass", "fail", "need_rescan"}:
            verdict = "need_rescan"
            geo_pred_source = str(geo_pred_field)
            geo_reject_reason = f"invalid_{geo_pred_field}"

        accepted = int(verdict in {"pass", "fail"})
        if verdict == "need_rescan":
            rej += 1
            abnormal_reject += gt_abnormal
            geo_case = "reject"
        elif verdict == "fail":
            if gt_abnormal == 1:
                tp += 1
                geo_case = "TP_geo"
            else:
                fp += 1
                geo_case = "FP_geo"
        else:
            if gt_abnormal == 1:
                fn += 1
                geo_case = "FN_geo"
            else:
                tn += 1
                geo_case = "TN_geo"

        included_rows.append(
            {
                "sample_uid": str(row.get("sample_uid", row.get("sample_id", "")) or ""),
                "sample_id": str(row.get("sample_id", "")),
                "source_sample_id": str(row.get("source_sample_id", row.get("sample_id", "")) or ""),
                "gt_partition": str(row.get("gt_partition", "") or ""),
                "stable_id": int(row.get("stable_id", -1)),
                "gt_abnormal": int(gt_abnormal),
                "gt_label": str(row.get("gt_label", "")),
                "geo_included": 1,
                "geo_case": str(geo_case),
                "geo_verdict_field": str(geo_pred_field),
                "geo_pred_source": str(geo_pred_source),
                "geo_pred": str(verdict),
                "accepted": int(accepted),
                "need_rescan": int(verdict == "need_rescan"),
                "geo_reject_reason": str(geo_reject_reason),
            }
        )

    n_geo = int(len(included_rows))
    n_cov = int(tp + fp + fn + tn)
    precision_abn = _safe_div(tp, tp + fp)
    recall_abn = _safe_div(tp, tp + fn)
    f1_abn = _f1_from_pr(precision_abn, recall_abn)
    precision_norm = _safe_div(tn, tn + fn)
    recall_norm = _safe_div(tn, tn + fp)
    f1_norm = _f1_from_pr(precision_norm, recall_norm)
    macro_f1_geo = float(np.nanmean([f1_abn, f1_norm]))
    balanced_accuracy_geo = float(np.nanmean([recall_abn, recall_norm]))
    coverage_geo = _safe_div(n_cov, n_geo)
    rescan_rate_geo = _safe_div(rej, n_geo)
    overall_abnormal_capture_geo = _safe_div(tp + abnormal_reject, abnormal_total)
    return {
        "summary": {
            "geo_pred_field": str(geo_pred_field),
            "n_geo": int(n_geo),
            "n_cov_geo": int(n_cov),
            "n_rej_geo": int(rej),
            "tp_geo": int(tp),
            "fp_geo": int(fp),
            "fn_geo": int(fn),
            "tn_geo": int(tn),
            "abnormal_total_geo": int(abnormal_total),
            "normal_total_geo": int(normal_total),
            "abnormal_reject_geo": int(abnormal_reject),
            "precision_abn_geo": float(precision_abn),
            "recall_abn_geo": float(recall_abn),
            "f1_abn_geo": float(f1_abn),
            "precision_norm_geo": float(precision_norm),
            "recall_norm_geo": float(recall_norm),
            "f1_norm_geo": float(f1_norm),
            "macro_f1_geo": float(macro_f1_geo),
            "balanced_accuracy_geo": float(balanced_accuracy_geo),
            "coverage_geo": float(coverage_geo),
            "rescan_rate_geo": float(rescan_rate_geo),
            "abnormal_recall_at_covered_geo": float(recall_abn),
            "overall_abnormal_capture_geo": float(overall_abnormal_capture_geo),
            "geo_excluded_count": int(len(excluded_rows)),
        },
        "rows": included_rows,
        "excluded_rows": excluded_rows,
    }


def _build_gt_point_predictions_by_manifest_rows(
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    gt_root: Path,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    point_predictions: Dict[Tuple[str, int], Dict[str, Any]] = {}
    ref_pack, tol_base_path, tol_pins_path, _tol_json = _build_reference_infos()
    infos_ref = list(ref_pack["infos_ref"])
    centerline_cfg = _centerline_extract_config()
    r_match_u = ref_pack["r_match_u"]
    for row in manifest_rows:
        sample_uid = str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
        sample_id = str(row.get("source_sample_id", "") or row.get("sample_id", "") or "").strip()
        if not sample_uid or not sample_id:
            continue
        pin_gt_path, base_gt_path = authoritative_gt_paths(
            gt_root,
            sample_id,
            gt_partition=str(row.get("gt_partition", "bad") or "bad"),
            gt_ply=str(row.get("authoritative_gt_ply", "") or row.get("gt_ply", "") or ""),
            gt_base_ply=str(row.get("authoritative_gt_base_ply", "") or row.get("gt_base_ply", "") or ""),
        )
        if not pin_gt_path.is_file() or not base_gt_path.is_file():
            continue
        T, reg_meta, _, _ = compute_board_transform(
            str(tol_base_path),
            str(base_gt_path),
            str(tol_pins_path),
            str(pin_gt_path),
            return_clean_clouds=True,
        )
        infos_def = extract_centerlines_from_pcd(str(pin_gt_path), config=centerline_cfg)
        infos_def_aligned = transform_needle_infos(infos_def, T)
        pairing = match_needles_by_position(infos_ref, infos_def_aligned, r_match=r_match_u)
        for pair in pairing.get("pairs", []):
            idx_ref = int(pair.get("normal_id", -1))
            idx_def = int(pair.get("defect_id", -1))
            if idx_ref < 0 or idx_ref >= len(infos_ref) or idx_def < 0 or idx_def >= len(infos_def_aligned):
                continue
            record, artifact = _analyze_tolerance_pair(
                infos_ref[idx_ref],
                infos_def_aligned[idx_def],
                pair,
                pairing,
                final_method="tolerance_band",
                reg_meta=reg_meta,
                step_mm=0.2,
            )
            if record is None or artifact is None:
                continue
            stable_id = int(record.get("stable_id", infos_ref[idx_ref].get("stable_id", idx_ref)))
            tol_metrics = dict(artifact.get("tol_metrics", {}))
            centerline_points = np.asarray(artifact.get("pts_centerline_def_final", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
            point_scores_raw = np.asarray(tol_metrics.get("point_score_selected_raw_arr", []), dtype=np.float64).reshape(-1)
            if len(centerline_points) <= 0 or len(point_scores_raw) <= 0:
                continue
            n = min(len(centerline_points), len(point_scores_raw))
            point_predictions[(str(sample_uid), int(stable_id))] = {
                "sample_uid": str(sample_uid),
                "sample_id": str(sample_id),
                "source_sample_id": str(sample_id),
                "stable_id": int(stable_id),
                "centerline_points": np.asarray(centerline_points[:n], dtype=np.float64),
                "point_scores_raw": np.asarray(point_scores_raw[:n], dtype=np.float64),
                "fit_status": str(record.get("fit_status", "")),
                "tol_status": str(record.get("tol_status", "")),
                "formal_decision_verdict": str(record.get("formal_decision_verdict", "")),
            }
    return point_predictions


def _point_metrics_for_scope(
    gt_slot_map: Mapping[Tuple[str, int], Mapping[str, Any]],
    point_support_by_key: Mapping[Tuple[str, int], Mapping[str, Any]],
    point_pred_by_key: Mapping[Tuple[str, int], Mapping[str, Any]],
    *,
    point_threshold: float,
    scope_name: str,
    scope_label: str,
    eligibility_field: str,
) -> Dict[str, Any]:
    eligible_items = [
        (key, dict(row))
        for key, row in gt_slot_map.items()
        if int(row.get(eligibility_field, 0)) == 1
    ]

    pooled_true: List[np.ndarray] = []
    pooled_score: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    per_needle_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []

    for key, slot_row in eligible_items:
        sample_uid, stable_id = str(key[0]), int(key[1])
        support = point_support_by_key.get((sample_uid, stable_id))
        pred_pack = point_pred_by_key.get((sample_uid, stable_id))
        source_sample_id = str(
            slot_row.get("source_sample_id", slot_row.get("sample_id", "")) or ""
        )
        if support is not None:
            source_sample_id = str(support.get("source_sample_id", support.get("sample_id", "")) or source_sample_id)
        elif pred_pack is not None:
            source_sample_id = str(pred_pack.get("source_sample_id", pred_pack.get("sample_id", "")) or source_sample_id)
        if support is None:
            excluded_rows.append(
                {
                    "point_scope": str(scope_name),
                    "sample_uid": sample_uid,
                    "sample_id": source_sample_id,
                    "stable_id": int(stable_id),
                    "point_exclusion_reason": "missing_point_support",
                }
            )
            continue
        if pred_pack is None:
            excluded_rows.append(
                {
                    "point_scope": str(scope_name),
                    "sample_uid": sample_uid,
                    "sample_id": source_sample_id,
                    "stable_id": int(stable_id),
                    "point_exclusion_reason": "missing_point_prediction",
                }
            )
            continue
        support_points = np.asarray(support.get("points_xyz_aligned", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
        gt_mask = np.asarray(support.get("gt_positive_mask", np.zeros(0, dtype=np.uint8)), dtype=np.uint8).reshape(-1)
        centerline_points = np.asarray(pred_pack.get("centerline_points", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
        point_scores = np.asarray(pred_pack.get("point_scores_raw", np.zeros(0, dtype=np.float64)), dtype=np.float64).reshape(-1)
        if len(support_points) <= 0 or len(gt_mask) <= 0 or len(centerline_points) <= 0 or len(point_scores) <= 0:
            excluded_rows.append(
                {
                    "point_scope": str(scope_name),
                    "sample_uid": sample_uid,
                    "sample_id": source_sample_id,
                    "stable_id": int(stable_id),
                    "point_exclusion_reason": "empty_point_arrays",
                }
            )
            continue
        tree = cKDTree(centerline_points)
        _, nearest_idx = tree.query(support_points, k=1, workers=-1)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
        mapped_scores = point_scores[np.clip(nearest_idx, 0, len(point_scores) - 1)]
        pred_mask = np.asarray(mapped_scores > float(point_threshold), dtype=np.uint8)

        tp = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
        fp = int(np.sum((pred_mask == 1) & (gt_mask == 0)))
        fn = int(np.sum((pred_mask == 0) & (gt_mask == 1)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1_from_pr(precision, recall)
        iou = _safe_div(tp, tp + fp + fn)
        ap, _curve_rows = _average_precision_binary(gt_mask, mapped_scores)

        pooled_true.append(np.asarray(gt_mask, dtype=np.uint8))
        pooled_score.append(np.asarray(mapped_scores, dtype=np.float64))
        pooled_pred.append(np.asarray(pred_mask, dtype=np.uint8))
        per_needle_rows.append(
            {
                "point_scope": str(scope_name),
                "point_scope_label": str(scope_label),
                "point_scope_eligibility_field": str(eligibility_field),
                "sample_uid": str(sample_uid),
                "sample_id": str(source_sample_id),
                "stable_id": int(stable_id),
                "point_count": int(len(gt_mask)),
                "gt_positive_count": int(np.sum(gt_mask == 1)),
                "pred_positive_count": int(np.sum(pred_mask == 1)),
                "point_precision_defined": int(tp + fp > 0),
                "point_recall_defined": int(tp + fn > 0),
                "point_f1_defined": int(bool(np.isfinite(precision) and np.isfinite(recall) and (float(precision) + float(recall) > 0.0))),
                "point_iou_defined": int(tp + fp + fn > 0),
                "point_ap_defined": int(np.sum(gt_mask == 1) > 0),
                "point_precision": float(precision),
                "point_recall": float(recall),
                "point_f1": float(f1),
                "point_iou": float(iou),
                "point_ap": float(ap),
                "point_threshold": float(point_threshold),
            }
        )

    exclusion_reason_counts: Dict[str, int] = {}
    for row in excluded_rows:
        reason = str(row.get("point_exclusion_reason", "") or "unknown")
        exclusion_reason_counts[reason] = int(exclusion_reason_counts.get(reason, 0) + 1)

    if pooled_true:
        y_true = np.concatenate(pooled_true, axis=0)
        y_score = np.concatenate(pooled_score, axis=0)
        y_pred = np.concatenate(pooled_pred, axis=0)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1_from_pr(precision, recall)
        iou = _safe_div(tp, tp + fp + fn)
        ap, pr_curve_rows = _average_precision_binary(y_true, y_score)
        summary = {
            "point_scope": str(scope_name),
            "point_scope_label": str(scope_label),
            "point_scope_eligibility_field": str(eligibility_field),
            "n_point_needles_requested": int(len(eligible_items)),
            "n_point_needles": int(len(per_needle_rows)),
            "n_point_needles_excluded": int(len(excluded_rows)),
            "point_exclusion_reason_counts_json": json.dumps(exclusion_reason_counts, ensure_ascii=False, sort_keys=True),
            "point_threshold": float(point_threshold),
            "point_count_total": int(len(y_true)),
            "point_positive_total": int(np.sum(y_true == 1)),
            "point_pred_positive_total": int(np.sum(y_pred == 1)),
            "point_precision_defined": int(tp + fp > 0),
            "point_recall_defined": int(tp + fn > 0),
            "point_f1_defined": int(bool(np.isfinite(precision) and np.isfinite(recall) and (float(precision) + float(recall) > 0.0))),
            "point_iou_defined": int(tp + fp + fn > 0),
            "point_ap_defined": int(np.sum(y_true == 1) > 0),
            "point_precision": float(precision),
            "point_recall": float(recall),
            "point_f1": float(f1),
            "point_iou": float(iou),
            "point_ap": float(ap),
        }
    else:
        summary = {
            "point_scope": str(scope_name),
            "point_scope_label": str(scope_label),
            "point_scope_eligibility_field": str(eligibility_field),
            "n_point_needles_requested": int(len(eligible_items)),
            "n_point_needles": 0,
            "n_point_needles_excluded": int(len(excluded_rows)),
            "point_exclusion_reason_counts_json": json.dumps(exclusion_reason_counts, ensure_ascii=False, sort_keys=True),
            "point_threshold": float(point_threshold),
            "point_count_total": 0,
            "point_positive_total": 0,
            "point_pred_positive_total": 0,
            "point_precision_defined": 0,
            "point_recall_defined": 0,
            "point_f1_defined": 0,
            "point_iou_defined": 0,
            "point_ap_defined": 0,
            "point_precision": float("nan"),
            "point_recall": float("nan"),
            "point_f1": float("nan"),
            "point_iou": float("nan"),
            "point_ap": float("nan"),
        }
        pr_curve_rows = []
    return {
        "summary": summary,
        "rows": per_needle_rows,
        "excluded_rows": excluded_rows,
        "pr_curve_rows": pr_curve_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Authoritative GT metrics for pin-level mainline evaluation (logic / geometry) with optional point visualization sidecar"
    )
    parser.add_argument("--per_slot_csv", default="", help="Optional per-slot CSV override")
    parser.add_argument("--manifest_csv", default="", help="Optional manifest CSV override")
    parser.add_argument(
        "--manifest_role",
        default=str(DEFAULT_MANIFEST_ROLE),
        choices=["mixed", "dev", "test", "adjudicated"],
        help="Protocol manifest role when --manifest_csv is not given",
    )
    parser.add_argument(
        "--geo_pred_field",
        default=str(DEFAULT_GEO_PRED_FIELD),
        help="Prediction field from per-slot CSV for geometry metrics",
    )
    parser.add_argument(
        "--gt_source_mode",
        default=str(DEFAULT_GT_SOURCE_MODE),
        choices=["auto", "live", "prepared_only"],
        help="How logic/geo GT is loaded: auto prefers prepared slot GT cache, live recomputes, prepared_only requires cache hit.",
    )
    parser.add_argument("--gt_cache_root", default="", help="Optional prepared logic/geo GT cache root override")
    parser.add_argument(
        "--point_threshold",
        default=float(DEFAULT_POINT_THRESHOLD),
        type=float,
        help="Binary threshold on point_score_selected_raw when deriving point masks",
    )
    parser.add_argument(
        "--emit_point_visual_sidecar",
        action="store_true",
        help="Emit point-level sidecar artifacts for visualization / localization only; not part of the formal pin-level mainline output.",
    )
    args = parser.parse_args()

    manifest_path = Path(str(args.manifest_csv).strip()).resolve() if str(args.manifest_csv).strip() else _manifest_path_from_role(str(args.manifest_role))
    if str(args.per_slot_csv).strip():
        per_slot_csv = Path(str(args.per_slot_csv).strip()).resolve()
        per_slot_resolution = {
            "mode": "explicit_arg",
            "run_id": "",
            "run_dir": str(per_slot_csv.parent),
            "manifest_path": str(manifest_path),
            "per_slot_csv": str(per_slot_csv),
        }
    else:
        per_slot_resolution = _resolve_per_slot_csv_from_matching_detect_run(manifest_path)
        per_slot_csv = Path(str(per_slot_resolution.get("per_slot_csv", "") or "")).resolve()
    manifest_rows = validate_manifest_rows_against_authoritative_gt(_load_manifest_rows(manifest_path), GT_ROOT_DEFAULT)
    sample_ids = [
        str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
        for row in manifest_rows
        if str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
    ]
    gt_cache_root = Path(str(args.gt_cache_root).strip()).resolve() if str(args.gt_cache_root).strip() else Path(LOGIC_GT_SAMPLE_CACHE_ROOT).resolve()
    gt_load_plan = _resolve_gt_load_plan(
        gt_source_mode=str(args.gt_source_mode),
        emit_point_visual_sidecar=bool(args.emit_point_visual_sidecar),
    )
    gt_load_t0 = time.perf_counter()
    gt_slot_map, gt_skipped, sample_summaries, point_support_by_key = load_gt_authoritative_by_manifest_rows(
        GT_ROOT_DEFAULT,
        manifest_rows,
        include_point_support=bool(gt_load_plan.get("include_point_support", True)),
        use_sample_cache=bool(gt_load_plan.get("use_sample_cache", False)),
        cache_root=gt_cache_root,
        require_sample_cache_hit=bool(gt_load_plan.get("require_sample_cache_hit", False)),
        write_sample_cache_on_miss=bool(gt_load_plan.get("write_sample_cache_on_miss", True)),
    )
    gt_load_elapsed_sec = float(time.perf_counter() - gt_load_t0)
    gt_cache_stats = _summarize_logic_gt_cache_usage(sample_summaries)
    per_slot_rows = _read_csv(per_slot_csv)
    pred_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in per_slot_rows:
        sample_id = str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
        stable_id = _safe_int(row.get("stable_id", -1), default=-1)
        if sample_id and stable_id >= 0:
            pred_by_key[(sample_id, int(stable_id))] = dict(row)

    merged_slot_rows: List[Dict[str, Any]] = []
    for key, gt_row in sorted(gt_slot_map.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))):
        pred_row = pred_by_key.get(key)
        if pred_row is None:
            raise RuntimeError(f"per-slot prediction missing for key={key!r} in {per_slot_csv}")
        merged_row = dict(gt_row)
        merged_row.update(
            {
                "pred_present": int(_safe_int(pred_row.get("pred_present", 0), default=0)),
                "pred_missing": int(_safe_int(pred_row.get("pred_missing", 0), default=0)),
                "pred_pair_status": str(pred_row.get("pred_pair_status", "")),
                "pred_missing_reason": str(pred_row.get("pred_missing_reason", "")),
                "pred_geo_eval_eligible": int(_safe_int(pred_row.get("pred_geo_eval_eligible", 0), default=0)),
                "fit_status": str(pred_row.get("fit_status", "")),
                "tol_status": str(pred_row.get("tol_status", "")),
                "tangent_status": str(pred_row.get("tangent_status", "")),
                "baseline_verdict": str(pred_row.get("baseline_verdict", "")),
                "baseline_reason": str(pred_row.get("baseline_reason", "")),
                "formal_decision_verdict": str(pred_row.get("formal_decision_verdict", "")),
                "formal_decision_reason": str(pred_row.get("formal_decision_reason", "")),
                "formal_decision_gate_status": str(pred_row.get("formal_decision_gate_status", "")),
                "formal_decision_gate_reason": str(pred_row.get("formal_decision_gate_reason", "")),
                "ab_shadow_verdict": str(pred_row.get("ab_shadow_verdict", "")),
                "ab_shadow_reason": str(pred_row.get("ab_shadow_reason", "")),
            }
        )
        merged_slot_rows.append(merged_row)

    logic_metrics = _logic_metrics_from_rows(merged_slot_rows)
    geo_metrics = _geo_metrics_from_rows(merged_slot_rows, geo_pred_field=str(args.geo_pred_field))
    out_dir = (
        Path(_DAIMA_CTX.run_dir)
        if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None)
        else (REPO_ROOT / "debug_out" / "manual_authoritative_metrics")
    )
    sample_ids_used_global = sorted({str(k[0]) for k in gt_slot_map.keys()})
    sample_subset_tasks: Dict[str, Any] = {
        "logic": build_task_subset(sample_ids_requested=sample_ids, sample_ids_used=sample_ids_used_global, skipped=gt_skipped),
        "geo": build_task_subset(sample_ids_requested=sample_ids, sample_ids_used=sample_ids_used_global, skipped=gt_skipped),
    }
    point_sidecar_outputs: Dict[str, str] = {}
    point_visual_sidecar: Dict[str, Any] = {
        "emitted": False,
        "usage": "visualization_only",
        "not_formal_metric": True,
        "description": "Point-level artifacts are sidecar outputs for localization / visualization only and are excluded from the formal pin-level mainline report.",
    }
    if bool(args.emit_point_visual_sidecar):
        point_eval_sample_ids = sorted(
            {
                str(key[0])
                for key, row in gt_slot_map.items()
                if int(row.get("gt_point_support_present", 0)) == 1
                or int(row.get("gt_point_global_eval_eligible", 0)) == 1
                or int(row.get("gt_point_eval_eligible", 0)) == 1
            }
        )
        manifest_rows_point = [
            dict(row)
            for row in list(manifest_rows or [])
            if str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip() in set(point_eval_sample_ids)
        ]
        point_pred_by_key = _build_gt_point_predictions_by_manifest_rows(
            manifest_rows_point,
            gt_root=Path(GT_ROOT_DEFAULT).resolve(),
        )
        point_metrics_global = _point_metrics_for_scope(
            gt_slot_map,
            point_support_by_key,
            point_pred_by_key,
            point_threshold=float(args.point_threshold),
            scope_name="global_all_geo",
            scope_label="all_geo_eligible_needles_with_point_support",
            eligibility_field="gt_point_global_eval_eligible",
        )
        point_metrics_positive_conditioned = _point_metrics_for_scope(
            gt_slot_map,
            point_support_by_key,
            point_pred_by_key,
            point_threshold=float(args.point_threshold),
            scope_name="positive_conditioned",
            scope_label="abnormal_needles_with_point_labels",
            eligibility_field="gt_point_eval_eligible",
        )
        point_sample_ids_used_global = sorted(
            {
                str(key[0])
                for key, row in gt_slot_map.items()
                if int(row.get("gt_point_global_eval_eligible", 0)) == 1
            }
        )
        point_not_applicable_global = sorted(set(sample_ids_used_global) - set(point_sample_ids_used_global))
        sample_subset_tasks["point_visual_sidecar"] = build_task_subset(
            sample_ids_requested=sample_ids,
            sample_ids_used=point_sample_ids_used_global,
            skipped=gt_skipped,
            not_applicable_sample_ids=point_not_applicable_global,
        )
        point_sidecar_outputs = {
            "authoritative_point_metrics_csv": str(out_dir / "authoritative_point_metrics.csv"),
            "authoritative_point_metrics_global_csv": str(out_dir / "authoritative_point_metrics_global.csv"),
            "authoritative_point_metrics_positive_conditioned_csv": str(out_dir / "authoritative_point_metrics_positive_conditioned.csv"),
            "authoritative_point_per_needle_csv": str(out_dir / "authoritative_point_per_needle.csv"),
            "authoritative_point_per_needle_global_csv": str(out_dir / "authoritative_point_per_needle_global.csv"),
            "authoritative_point_per_needle_positive_conditioned_csv": str(out_dir / "authoritative_point_per_needle_positive_conditioned.csv"),
            "authoritative_point_excluded_needles_csv": str(out_dir / "authoritative_point_excluded_needles.csv"),
            "authoritative_point_excluded_needles_global_csv": str(out_dir / "authoritative_point_excluded_needles_global.csv"),
            "authoritative_point_excluded_needles_positive_conditioned_csv": str(out_dir / "authoritative_point_excluded_needles_positive_conditioned.csv"),
            "authoritative_point_pr_curve_csv": str(out_dir / "authoritative_point_pr_curve.csv"),
            "authoritative_point_pr_curve_global_csv": str(out_dir / "authoritative_point_pr_curve_global.csv"),
            "authoritative_point_pr_curve_positive_conditioned_csv": str(out_dir / "authoritative_point_pr_curve_positive_conditioned.csv"),
        }
        point_visual_sidecar = {
            "emitted": True,
            "usage": "visualization_only",
            "not_formal_metric": True,
            "description": "Point-level artifacts were emitted for visualization / localization only and are excluded from the formal pin-level mainline report.",
            "paths": dict(point_sidecar_outputs),
        }
    sample_subset_contract = build_formal_sample_contract(
        policy="gt_native_pin_qc_gate_then_task_applicability",
        sample_ids_requested=sample_ids,
        sample_ids_used_global=sample_ids_used_global,
        global_skipped=gt_skipped,
        tasks=sample_subset_tasks,
    )
    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "manifest_path": str(manifest_path),
                "per_slot_csv": str(per_slot_csv),
                "gt_root": str(Path(GT_ROOT_DEFAULT).resolve()),
                "gt_cache_root": str(gt_cache_root),
            },
            outputs={
                "authoritative_logic_metrics_csv": str(out_dir / "authoritative_logic_metrics.csv"),
                "authoritative_geo_metrics_csv": str(out_dir / "authoritative_geo_metrics.csv"),
                "authoritative_logic_per_slot_csv": str(out_dir / "authoritative_logic_per_slot.csv"),
                "authoritative_geo_per_slot_csv": str(out_dir / "authoritative_geo_per_slot.csv"),
                "authoritative_geo_excluded_slots_csv": str(out_dir / "authoritative_geo_excluded_slots.csv"),
                "authoritative_metrics_report_json": str(out_dir / "authoritative_metrics_report.json"),
                **point_sidecar_outputs,
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "manifest_role": str(args.manifest_role),
                "geo_pred_field": str(args.geo_pred_field),
                "gt_source_mode_requested": str(gt_load_plan.get("requested_mode", args.gt_source_mode)),
                "gt_source_mode_resolved": str(gt_load_plan.get("resolved_mode", "")),
                "gt_source_reason": str(gt_load_plan.get("reason", "")),
                "gt_cache_root": str(gt_cache_root),
                "point_threshold": float(args.point_threshold),
                "emit_point_visual_sidecar": bool(args.emit_point_visual_sidecar),
            }
        )

    _write_csv(out_dir / "authoritative_logic_metrics.csv", [logic_metrics["summary"]], list(logic_metrics["summary"].keys()))
    _write_csv(out_dir / "authoritative_geo_metrics.csv", [geo_metrics["summary"]], list(geo_metrics["summary"].keys()))
    _write_csv(out_dir / "authoritative_logic_per_slot.csv", logic_metrics["rows"], list(logic_metrics["rows"][0].keys()) if logic_metrics["rows"] else ["sample_uid", "sample_id", "stable_id"])
    _write_csv(out_dir / "authoritative_geo_per_slot.csv", geo_metrics["rows"], list(geo_metrics["rows"][0].keys()) if geo_metrics["rows"] else ["sample_uid", "sample_id", "stable_id"])
    _write_csv(out_dir / "authoritative_geo_excluded_slots.csv", geo_metrics["excluded_rows"], list(geo_metrics["excluded_rows"][0].keys()) if geo_metrics["excluded_rows"] else ["sample_uid", "sample_id", "stable_id", "geo_exclusion_reason"])
    if bool(args.emit_point_visual_sidecar):
        _write_csv(out_dir / "authoritative_point_metrics.csv", [point_metrics_positive_conditioned["summary"]], list(point_metrics_positive_conditioned["summary"].keys()))
        _write_csv(out_dir / "authoritative_point_metrics_global.csv", [point_metrics_global["summary"]], list(point_metrics_global["summary"].keys()))
        _write_csv(out_dir / "authoritative_point_metrics_positive_conditioned.csv", [point_metrics_positive_conditioned["summary"]], list(point_metrics_positive_conditioned["summary"].keys()))
        _write_csv(out_dir / "authoritative_point_per_needle.csv", point_metrics_positive_conditioned["rows"], list(point_metrics_positive_conditioned["rows"][0].keys()) if point_metrics_positive_conditioned["rows"] else ["sample_uid", "sample_id", "stable_id"])
        _write_csv(out_dir / "authoritative_point_per_needle_global.csv", point_metrics_global["rows"], list(point_metrics_global["rows"][0].keys()) if point_metrics_global["rows"] else ["point_scope", "sample_uid", "sample_id", "stable_id"])
        _write_csv(out_dir / "authoritative_point_per_needle_positive_conditioned.csv", point_metrics_positive_conditioned["rows"], list(point_metrics_positive_conditioned["rows"][0].keys()) if point_metrics_positive_conditioned["rows"] else ["point_scope", "sample_uid", "sample_id", "stable_id"])
        _write_csv(out_dir / "authoritative_point_excluded_needles.csv", point_metrics_positive_conditioned["excluded_rows"], list(point_metrics_positive_conditioned["excluded_rows"][0].keys()) if point_metrics_positive_conditioned["excluded_rows"] else ["sample_uid", "sample_id", "stable_id", "point_exclusion_reason"])
        _write_csv(out_dir / "authoritative_point_excluded_needles_global.csv", point_metrics_global["excluded_rows"], list(point_metrics_global["excluded_rows"][0].keys()) if point_metrics_global["excluded_rows"] else ["point_scope", "sample_uid", "sample_id", "stable_id", "point_exclusion_reason"])
        _write_csv(out_dir / "authoritative_point_excluded_needles_positive_conditioned.csv", point_metrics_positive_conditioned["excluded_rows"], list(point_metrics_positive_conditioned["excluded_rows"][0].keys()) if point_metrics_positive_conditioned["excluded_rows"] else ["point_scope", "sample_uid", "sample_id", "stable_id", "point_exclusion_reason"])
        _write_csv(out_dir / "authoritative_point_pr_curve.csv", point_metrics_positive_conditioned["pr_curve_rows"], list(point_metrics_positive_conditioned["pr_curve_rows"][0].keys()) if point_metrics_positive_conditioned["pr_curve_rows"] else ["threshold", "precision", "recall"])
        _write_csv(out_dir / "authoritative_point_pr_curve_global.csv", point_metrics_global["pr_curve_rows"], list(point_metrics_global["pr_curve_rows"][0].keys()) if point_metrics_global["pr_curve_rows"] else ["threshold", "precision", "recall"])
        _write_csv(out_dir / "authoritative_point_pr_curve_positive_conditioned.csv", point_metrics_positive_conditioned["pr_curve_rows"], list(point_metrics_positive_conditioned["pr_curve_rows"][0].keys()) if point_metrics_positive_conditioned["pr_curve_rows"] else ["threshold", "precision", "recall"])

    report = {
        "manifest_path": str(manifest_path),
        "per_slot_csv": str(per_slot_csv),
        "per_slot_resolution": dict(per_slot_resolution),
        "sample_ids_requested": list(sample_ids),
        "sample_ids_used": list(sample_ids_used_global),
        "gt_skipped": gt_skipped,
        "gt_loading": {
            "requested_mode": str(gt_load_plan.get("requested_mode", args.gt_source_mode)),
            "resolved_mode": str(gt_load_plan.get("resolved_mode", "")),
            "reason": str(gt_load_plan.get("reason", "")),
            "cache_root": str(gt_cache_root),
            "include_point_support": bool(gt_load_plan.get("include_point_support", True)),
            "use_sample_cache": bool(gt_load_plan.get("use_sample_cache", False)),
            "require_sample_cache_hit": bool(gt_load_plan.get("require_sample_cache_hit", False)),
            "write_sample_cache_on_miss": bool(gt_load_plan.get("write_sample_cache_on_miss", True)),
            "gt_load_elapsed_sec": float(gt_load_elapsed_sec),
            "cache_stats": dict(gt_cache_stats),
        },
        "sample_subset_contract": sample_subset_contract,
        "gt_sample_summaries": sample_summaries,
        "formal_output_contract": {
            "primary_tasks": ["logic", "geo"],
            "point_role": "visualization_only_sidecar",
            "point_not_formal_metric": True,
        },
        "logic_metrics": logic_metrics["summary"],
        "geo_metrics": geo_metrics["summary"],
        "point_visual_sidecar": point_visual_sidecar,
    }
    _write_json(out_dir / "authoritative_metrics_report.json", report)
    print(f"[INFO] Wrote authoritative detection metrics under: {out_dir}")
    print(
        "[INFO] GT load: "
        f"requested={gt_load_plan.get('requested_mode', args.gt_source_mode)} "
        f"| resolved={gt_load_plan.get('resolved_mode', '')} "
        f"| elapsed={float(gt_load_elapsed_sec):.2f}s "
        f"| cache_hit={int(gt_cache_stats.get('cache_hit_count', 0))} "
        f"| cache_miss={int(gt_cache_stats.get('cache_miss_count', 0))}"
    )
    print(
        "[INFO] Logic mainline: "
        f"Macro-F1={float(logic_metrics['summary'].get('macro_f1_logic', float('nan'))):.4f} "
        f"| MissingRecall={float(logic_metrics['summary'].get('recall_logic', float('nan'))):.4f}"
    )
    print(
        "[INFO] Geo mainline: "
        f"Macro-F1={float(geo_metrics['summary'].get('macro_f1_geo', float('nan'))):.4f} "
        f"| Coverage={float(geo_metrics['summary'].get('coverage_geo', float('nan'))):.4f} "
        f"| RescanRate={float(geo_metrics['summary'].get('rescan_rate_geo', float('nan'))):.4f}"
    )
    if bool(args.emit_point_visual_sidecar):
        print("[INFO] Point visualization sidecar emitted (not part of formal pin-level metrics).")


if __name__ == "__main__":
    main()
