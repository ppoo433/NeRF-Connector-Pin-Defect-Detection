# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_DAIMA_CTX = None
if __name__ == "__main__":
    try:
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

import numpy as np

try:
    from .detect_tolerance_interactive_1_11 import (
        DETECTION_DISABLE_RESCAN,
        DETECTION_DISABLE_SHORT_TOLERANCE,
        FINAL_METHOD,
        PIN_UNIT_SCALE,
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
    
    from .gt_native_loader import GT_ROOT_DEFAULT, validate_manifest_rows_against_authoritative_gt
except ImportError:
    from detect_tolerance_interactive_1_11 import (
        DETECTION_DISABLE_RESCAN,
        DETECTION_DISABLE_SHORT_TOLERANCE,
        FINAL_METHOD,
        PIN_UNIT_SCALE,
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
    
    from gt_native_loader import GT_ROOT_DEFAULT, validate_manifest_rows_against_authoritative_gt

from baseline.common import (
    compare_readout_protocol_matches_frozen,
    format_compare_readout_protocol,
    get_frozen_compare_readout_protocol_v1,
    resolve_compare_readout_protocol_from_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = REPO_ROOT / "jiance" / "paper_experiment" / "tolerance_band_v1_protocol"
DEFAULT_SPLIT_ROLE = str(os.getenv("DAIMA_PROTOCOL_SPLIT", "mixed") or "mixed").strip().lower()
DEFAULT_MAX_SAMPLES = int(str(os.getenv("DAIMA_TEST_MAX_SAMPLES", "0") or "0"))


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_manifest_rows_from_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
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


def _load_manifest_rows(role: str) -> List[Dict[str, str]]:
    role_n = str(role).strip().lower()
    role_to_file = {
        "mixed": PROTOCOL_DIR / "mixed_manifest.csv",
        "dev": PROTOCOL_DIR / "dev_manifest.csv",
        "test": PROTOCOL_DIR / "test_manifest.csv",
        "adjudicated": PROTOCOL_DIR / "adjudicated_manifest.csv",
    }
    if role_n == "all":
        rows: List[Dict[str, str]] = []
        for key in ("dev", "test"):
            rows.extend(_load_manifest_rows(key))
        return rows
    path = role_to_file.get(role_n)
    if path is None:
        raise FileNotFoundError(f"Unknown or missing manifest for role={role!r}: {path}")
    return _load_manifest_rows_from_csv(path)


def _is_official_protocol_manifest(manifest_override: str) -> bool:
    manifest_text = str(manifest_override or "").strip()
    if not manifest_text:
        return True
    try:
        return Path(manifest_text).resolve().is_relative_to(PROTOCOL_DIR.resolve())
    except Exception:
        return False


def _guard_frozen_compare_readout_protocol(active_protocol: Dict[str, Any], manifest_override: str) -> None:
    active_text = format_compare_readout_protocol(active_protocol)
    frozen_protocol = get_frozen_compare_readout_protocol_v1()
    frozen_text = format_compare_readout_protocol(frozen_protocol)
    if compare_readout_protocol_matches_frozen(active_protocol):
        print(f"[INFO] compare/readout protocol: {active_text} [frozen]")
        return

    manifest_label = (
        str(Path(manifest_override).resolve())
        if str(manifest_override or "").strip()
        else f"protocol:{DEFAULT_SPLIT_ROLE}"
    )
    message = (
        "compare/readout protocol drift detected: "
        f"active={active_text} | frozen={frozen_text} | manifest={manifest_label}"
    )
    if _is_official_protocol_manifest(manifest_override):
        raise RuntimeError(message)
    print(f"[WARN] {message}")


def _normalize_match_path(path_like: Any) -> str:
    text = str(path_like or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(str(Path(text).resolve())))


def _build_reference_infos(tol_json: Dict[str, Any], tol_base_path: Path, tol_pins_path: Path) -> Dict[str, Any]:
    centerline_cfg = _centerline_extract_config()
    tol_unit_scale = _safe_float(tol_json.get("unit_scale", PIN_UNIT_SCALE), default=PIN_UNIT_SCALE)
    match_radius_mm = _safe_float(tol_json.get("match_radius_mm", 0.0), default=0.0)
    r_match_u = None
    if np.isfinite(match_radius_mm) and match_radius_mm > 0 and np.isfinite(tol_unit_scale) and tol_unit_scale > 1e-12:
        r_match_u = float(match_radius_mm) * float(tol_unit_scale)

    # Batch detect consumes the frozen tolerance model as the authoritative reference.
    infos_ref: List[Dict[str, Any]] = []
    for needle in (tol_json.get("needles", []) if isinstance(tol_json, dict) else []):
        try:
            sid = int(needle.get("stable_id", -1))
        except Exception:
            continue
        nominal = np.asarray(needle.get("nominal_curve_u", []), dtype=np.float64)
        if sid < 0 or nominal.ndim != 2 or nominal.shape[1] != 3 or len(nominal) < 2:
            continue
        info_ref = {
            "stable_id": int(sid),
            "curve_global_final": nominal,
            "center_point": np.mean(nominal, axis=0),
            "base_end_idx_local": 0,
            "tol_model_status": str(needle.get("status", "")),
            "tol_tangent_model_status": str(needle.get("tol_tangent_model_status", needle.get("status", ""))),
            "tol_unit_scale": float(tol_unit_scale),
            "tol_lateral_mm": needle.get("tol_lateral_mm", []),
            "tol_tip_short_mm": needle.get("tol_tip_short_mm", None),
            "tol_len_short_mm": needle.get("tol_len_short_mm", None),
            "tol_tangent_axis_xyz": needle.get("tol_tangent_axis_xyz", []),
            "tol_tangent_half_angle_deg": needle.get("tol_tangent_half_angle_deg", []),
            "tol_tangent_window_mm": needle.get("tol_tangent_window_mm", None),
            "tol_stats": needle.get("stats", {}),
        }
        infos_ref.append(info_ref)
    infos_ref.sort(key=lambda d: int(d.get("stable_id", 9999)))
    return {
        "infos_ref": infos_ref,
        "centerline_cfg": centerline_cfg,
        "tol_unit_scale": float(tol_unit_scale),
        "r_match_u": r_match_u,
        "match_radius_mm": float(match_radius_mm),
    }


def _build_case_report(
    manifest_row: Dict[str, str],
    *,
    infos_ref: List[Dict[str, Any]],
    centerline_cfg: Any,
    base_ref_path: Path,
    pins_ref_path: Path,
    r_match_u: Any,
    tol_json: Dict[str, Any],
) -> Dict[str, Any]:
    sample_uid = str(manifest_row.get("sample_uid", "") or "").strip()
    sample_id = str(manifest_row.get("source_sample_id", "") or manifest_row.get("sample_id", "") or "").strip()
    base_def_path = Path(str(manifest_row.get("base_ply", "")).strip())
    pins_def_path = Path(str(manifest_row.get("pin_ply", "")).strip())
    gt_ply = str(manifest_row.get("authoritative_gt_ply", "")).strip()
    gt_base_ply = str(manifest_row.get("authoritative_gt_base_ply", "")).strip()
    if not sample_id:
        sample_id = str(base_def_path.stem).split("_", 1)[0].strip() or str(pins_def_path.stem).split("_", 1)[0].strip()
    if not sample_uid:
        sample_uid = str(sample_id)

    T, reg_meta, _, _ = compute_board_transform(
        str(base_ref_path),
        str(base_def_path),
        str(pins_ref_path),
        str(pins_def_path),
        return_clean_clouds=True,
    )

    infos_def = extract_centerlines_from_pcd(str(pins_def_path), config=centerline_cfg)
    infos_def_aligned = transform_needle_infos(infos_def, T)
    pairing = match_needles_by_position(infos_ref, infos_def_aligned, r_match=r_match_u)

    analysis_report: Dict[str, Any] = {
        "sample_uid": str(sample_uid),
        "sample_id": str(sample_id),
        "source_sample_id": str(sample_id),
        "gt_partition": str(manifest_row.get("gt_partition", "") or ""),
        "final_method": "tolerance_band",
        "requested_final_method": str(FINAL_METHOD),
        "final_method_used": "tolerance_band",
        "pin_unit_scale": float(PIN_UNIT_SCALE),
        "protocol_split_role": str(manifest_row.get("split_role", "")),
        "gt_ply": str(gt_ply),
        "gt_base_ply": str(gt_base_ply),
        "missing_normal": list(pairing.get("missing_normal", [])) if isinstance(pairing, dict) else [],
        "extra_defect": list(pairing.get("extra_defect", [])) if isinstance(pairing, dict) else [],
        "needles": [],
        "paths": {
            "base_ref_path": str(base_ref_path),
            "pins_ref_path": str(pins_ref_path),
            "base_def_path": str(base_def_path),
            "pins_def_path": str(pins_def_path),
            "gt_ply": str(gt_ply),
            "gt_base_ply": str(gt_base_ply),
        },
        "registration": dict(reg_meta) if isinstance(reg_meta, dict) else {},
        "tolerance": {
            "tolerance_json_path": str(TOLERANCE_JSON_PATH),
            "reference_sample_id": str(tol_json.get("reference_sample_id", "")),
            "unit_scale": _safe_float(tol_json.get("unit_scale", float("nan"))),
            "match_radius_mm": _safe_float(tol_json.get("match_radius_mm", float("nan"))),
        },
    }

    for pair in pairing.get("pairs", []):
        idx_ref = int(pair.get("normal_id", -1))
        idx_def = int(pair.get("defect_id", -1))
        if idx_ref < 0 or idx_ref >= len(infos_ref) or idx_def < 0 or idx_def >= len(infos_def_aligned):
            continue
        record, _artifact = _analyze_tolerance_pair(
            infos_ref[idx_ref],
            infos_def_aligned[idx_def],
            pair,
            pairing,
            final_method="tolerance_band",
            reg_meta=reg_meta,
            step_mm=0.2,
        )
        if record is not None:
            analysis_report["needles"].append(record)
    analysis_report["formal_round2_runtime"] = {"enabled": False, "reason": "Final decisions are produced by TGAD LightGBM."}
    return analysis_report


def _flatten_per_needle_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_uid = str(report.get("sample_uid", ""))
    sample_id = str(report.get("sample_id", ""))
    gt_ply = str(report.get("gt_ply", ""))
    for needle in report.get("needles", []) if isinstance(report.get("needles", []), list) else []:
        rows.append(
            {
                "sample_uid": sample_uid,
                "sample_id": sample_id,
                "source_sample_id": sample_id,
                "gt_partition": str(report.get("gt_partition", "") or ""),
                "gt_ply": gt_ply,
                "stable_id": needle.get("stable_id", ""),
                "ref_index": needle.get("ref_index", ""),
                "def_index": needle.get("def_index", ""),
                "pair_dist_u": needle.get("pair_dist_u", ""),
                "fit_status": needle.get("fit_status", ""),
                "final_used_method": needle.get("final_used_method", ""),
                "tol_model_status": needle.get("tol_model_status", ""),
                "tol_status": needle.get("tol_status", ""),
                "tangent_status": needle.get("tangent_status", ""),
                "baseline_verdict": needle.get("baseline_verdict", ""),
                "baseline_reason": needle.get("baseline_reason", ""),
                "baseline_confidence_bucket": needle.get("baseline_confidence_bucket", ""),
                "uncertainty_flags": needle.get("uncertainty_flags", ""),
                "gate_status_shadow": needle.get("gate_status_shadow", ""),
                "gate_reason_shadow": needle.get("gate_reason_shadow", ""),
                "gate_hit_text": needle.get("gate_hit_text", ""),
                "gate_hard_hit_text": needle.get("gate_hard_hit_text", ""),
                "gate_soft_hit_text": needle.get("gate_soft_hit_text", ""),
                "gate_decision_rule_version": needle.get("gate_decision_rule_version", ""),
                "gate_hard_fail_count": needle.get("gate_hard_fail_count", ""),
                "formal_decision_rule_version": needle.get("formal_decision_rule_version", ""),
                "formal_decision_policy": needle.get("formal_decision_policy", ""),
                "formal_decision_base_source": needle.get("formal_decision_base_source", ""),
                "formal_decision_base_pred": needle.get("formal_decision_base_pred", ""),
                "formal_decision_gate_rule": needle.get("formal_decision_gate_rule", ""),
                "formal_decision_gate_status": needle.get("formal_decision_gate_status", ""),
                "formal_decision_gate_reason": needle.get("formal_decision_gate_reason", ""),
                "formal_decision_gate_hit_text": needle.get("formal_decision_gate_hit_text", ""),
                "formal_decision_local_rule": needle.get("formal_decision_local_rule", ""),
                "formal_decision_local_status": needle.get("formal_decision_local_status", ""),
                "formal_decision_local_reason": needle.get("formal_decision_local_reason", ""),
                "formal_decision_verdict": needle.get("formal_decision_verdict", ""),
                "formal_decision_reason": needle.get("formal_decision_reason", ""),
                "formal_decision_rule_version_pre_round2": needle.get("formal_decision_rule_version_pre_round2", ""),
                "formal_decision_policy_pre_round2": needle.get("formal_decision_policy_pre_round2", ""),
                "formal_decision_base_source_pre_round2": needle.get("formal_decision_base_source_pre_round2", ""),
                "formal_decision_base_pred_pre_round2": needle.get("formal_decision_base_pred_pre_round2", ""),
                "formal_decision_gate_rule_pre_round2": needle.get("formal_decision_gate_rule_pre_round2", ""),
                "formal_decision_gate_status_pre_round2": needle.get("formal_decision_gate_status_pre_round2", ""),
                "formal_decision_gate_reason_pre_round2": needle.get("formal_decision_gate_reason_pre_round2", ""),
                "formal_decision_gate_hit_text_pre_round2": needle.get("formal_decision_gate_hit_text_pre_round2", ""),
                "formal_decision_local_rule_pre_round2": needle.get("formal_decision_local_rule_pre_round2", ""),
                "formal_decision_local_status_pre_round2": needle.get("formal_decision_local_status_pre_round2", ""),
                "formal_decision_local_reason_pre_round2": needle.get("formal_decision_local_reason_pre_round2", ""),
                "formal_decision_verdict_pre_round2": needle.get("formal_decision_verdict_pre_round2", ""),
                "formal_decision_reason_pre_round2": needle.get("formal_decision_reason_pre_round2", ""),
                "formal_decision_round2_enabled": needle.get("formal_decision_round2_enabled", ""),
                "formal_decision_round2_policy": needle.get("formal_decision_round2_policy", ""),
                "formal_decision_round2_base_pred": needle.get("formal_decision_round2_base_pred", ""),
                "formal_decision_round2_reason": needle.get("formal_decision_round2_reason", ""),
                "formal_decision_round2_score": needle.get("formal_decision_round2_score", ""),
                "formal_decision_round2_threshold": needle.get("formal_decision_round2_threshold", ""),
                "formal_decision_round2_changed": needle.get("formal_decision_round2_changed", ""),
                "formal_decision_round2_score_source": needle.get("formal_decision_round2_score_source", ""),
                "formal_decision_round2_score_rank_ord": needle.get("formal_decision_round2_score_rank_ord", ""),
                "formal_decision_round2_evidence_rank_sum": needle.get("formal_decision_round2_evidence_rank_sum", ""),
                "formal_decision_round2_base_fail_count_in_sample": needle.get("formal_decision_round2_base_fail_count_in_sample", ""),
                "formal_decision_round2_g1_fail_count_in_sample": needle.get("formal_decision_round2_g1_fail_count_in_sample", ""),
                "formal_decision_round2_artifact_json": needle.get("formal_decision_round2_artifact_json", ""),
                "formal_decision_round2_candidate_name": needle.get("formal_decision_round2_candidate_name", ""),
                "formal_decision_round2_threshold_tag": needle.get("formal_decision_round2_threshold_tag", ""),
                "formal_decision_policy_shadow": needle.get("formal_decision_policy_shadow", ""),
                "formal_decision_base_source_shadow": needle.get("formal_decision_base_source_shadow", ""),
                "formal_decision_base_pred_shadow": needle.get("formal_decision_base_pred_shadow", ""),
                "formal_decision_gate_rule_shadow": needle.get("formal_decision_gate_rule_shadow", ""),
                "formal_decision_gate_status_shadow": needle.get("formal_decision_gate_status_shadow", ""),
                "formal_decision_gate_reason_shadow": needle.get("formal_decision_gate_reason_shadow", ""),
                "formal_decision_gate_hit_text_shadow": needle.get("formal_decision_gate_hit_text_shadow", ""),
                "formal_decision_local_rule_shadow": needle.get("formal_decision_local_rule_shadow", ""),
                "formal_decision_local_status_shadow": needle.get("formal_decision_local_status_shadow", ""),
                "formal_decision_local_reason_shadow": needle.get("formal_decision_local_reason_shadow", ""),
                "formal_decision_verdict_shadow": needle.get("formal_decision_verdict_shadow", ""),
                "formal_decision_reason_shadow": needle.get("formal_decision_reason_shadow", ""),
                "tol_fail_subtype": needle.get("tol_fail_subtype", ""),
                "point_score_raw_max": needle.get("point_score_raw_max", ""),
                "point_score_clipped_max": needle.get("point_score_clipped_max", ""),
                "point_score_fixed_raw_max": needle.get("point_score_fixed_raw_max", ""),
                "point_score_slack_raw_max": needle.get("point_score_slack_raw_max", ""),
                "point_score_selected_raw_max": needle.get("point_score_selected_raw_max", ""),
                "needle_score_raw": needle.get("needle_score_raw", ""),
                "needle_score_clipped": needle.get("needle_score_clipped", ""),
                "needle_score_q95": needle.get("needle_score_q95", ""),
                "needle_score_topk_mean": needle.get("needle_score_topk_mean", ""),
                "needle_score_topk_count": needle.get("needle_score_topk_count", ""),
                "ab_station_step_mm": needle.get("ab_station_step_mm", ""),
                "pred_pos_bins": needle.get("pred_pos_bins", ""),
                "pred_run_count": needle.get("pred_run_count", ""),
                "pred_max_run_bins": needle.get("pred_max_run_bins", ""),
                "pred_max_run_span_mm": needle.get("pred_max_run_span_mm", ""),
                "pred_dominant_run_frac": needle.get("pred_dominant_run_frac", ""),
                "pred_singleton_run_count": needle.get("pred_singleton_run_count", ""),
                "pred_short_run_bins": needle.get("pred_short_run_bins", ""),
                "pred_short_run_frac": needle.get("pred_short_run_frac", ""),
                "ab_compare_mode": needle.get("ab_compare_mode", ""),
                "ab_readout_mode": needle.get("ab_readout_mode", ""),
                "ab_slack_radius_steps": needle.get("ab_slack_radius_steps", ""),
                "ab_shadow_verdict": needle.get("ab_shadow_verdict", ""),
                "ab_shadow_reason": needle.get("ab_shadow_reason", ""),
                "ab_slack_gain": needle.get("ab_slack_gain", ""),
                "warp_severity": needle.get("warp_severity", ""),
                "ab_warp_p90_steps": needle.get("ab_warp_p90_steps", ""),
                "ab_warp_max_steps": needle.get("ab_warp_max_steps", ""),
                "ab_warp_p90_mm": needle.get("ab_warp_p90_mm", ""),
                "ab_warp_max_mm": needle.get("ab_warp_max_mm", ""),
                "ab_warp_nonzero_ratio": needle.get("ab_warp_nonzero_ratio", ""),
                "max_lateral_ratio": needle.get("max_lateral_ratio", ""),
                "tip_short_ratio": needle.get("tip_short_ratio", ""),
                "len_short_ratio": needle.get("len_short_ratio", ""),
                "max_tangent_ratio": needle.get("max_tangent_ratio", ""),
                "compare_alignment_mode": needle.get("compare_alignment_mode", ""),
                "gate_source_fields_text": needle.get("gate_source_fields_text", ""),
            }
        )
    return rows


def _flatten_per_slot_rows(report: Dict[str, Any], infos_ref: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_uid = str(report.get("sample_uid", ""))
    sample_id = str(report.get("sample_id", ""))
    gt_ply = str(report.get("gt_ply", ""))
    needles = list(report.get("needles", [])) if isinstance(report.get("needles", []), list) else []
    pair_by_ref_index: Dict[int, Dict[str, Any]] = {}
    for needle in needles:
        try:
            ref_index = int(needle.get("ref_index", -1))
        except Exception:
            continue
        if ref_index >= 0 and ref_index not in pair_by_ref_index:
            pair_by_ref_index[ref_index] = dict(needle)

    missing_ref_indices = set()
    for x in report.get("missing_normal", []) if isinstance(report.get("missing_normal", []), list) else []:
        try:
            missing_ref_indices.add(int(x))
        except Exception:
            continue

    slot_total_count = int(len(infos_ref))
    paired_count = int(len(needles))
    missing_count = int(len(missing_ref_indices))
    extra_defect_count = int(len(report.get("extra_defect", [])))

    for ref_index, info_ref in enumerate(infos_ref):
        stable_id = int(info_ref.get("stable_id", ref_index))
        paired_row = pair_by_ref_index.get(int(ref_index))
        pred_present = int(1 if paired_row is not None else 0)
        pred_missing = int(1 - pred_present)
        row = {
            "sample_uid": str(sample_uid),
            "sample_id": str(sample_id),
            "source_sample_id": str(sample_id),
            "gt_partition": str(report.get("gt_partition", "") or ""),
            "gt_ply": str(gt_ply),
            "stable_id": int(stable_id),
            "ref_index": int(ref_index),
            "slot_total_count": int(slot_total_count),
            "sample_paired_count": int(paired_count),
            "sample_missing_count": int(missing_count),
            "sample_extra_defect_count": int(extra_defect_count),
            "pred_present": int(pred_present),
            "pred_missing": int(pred_missing),
            "pred_pair_status": "present" if int(pred_present) == 1 else "missing",
            "pred_missing_reason": "missing_normal" if int(pred_missing) == 1 else "",
            "pred_geo_eval_eligible": int(pred_present),
            "def_index": "",
            "pair_dist_u": "",
            "fit_status": "",
            "tol_status": "",
            "tangent_status": "",
            "baseline_verdict": "",
            "baseline_reason": "",
            "formal_decision_verdict": "",
            "formal_decision_reason": "",
            "formal_decision_gate_status": "",
            "formal_decision_gate_reason": "",
            "formal_decision_local_rule": "",
            "formal_decision_local_status": "",
            "formal_decision_local_reason": "",
            "formal_decision_verdict_pre_round2": "",
            "formal_decision_reason_pre_round2": "",
            "formal_decision_round2_policy": "",
            "formal_decision_round2_reason": "",
            "formal_decision_round2_score": "",
            "formal_decision_round2_threshold": "",
            "formal_decision_round2_changed": "",
            "formal_decision_round2_score_source": "",
            "ab_shadow_verdict": "",
            "ab_shadow_reason": "",
            "tol_fail_subtype": "",
            "point_score_selected_raw_max": "",
            "needle_score_raw": "",
            "ab_station_step_mm": "",
            "pred_pos_bins": "",
            "pred_run_count": "",
            "pred_max_run_bins": "",
            "pred_max_run_span_mm": "",
            "pred_dominant_run_frac": "",
            "pred_singleton_run_count": "",
            "pred_short_run_bins": "",
            "pred_short_run_frac": "",
            "compare_alignment_mode": "",
            "ab_compare_mode": "",
            "ab_readout_mode": "",
        }
        if paired_row is not None:
            row.update(
                {
                    "def_index": paired_row.get("def_index", ""),
                    "pair_dist_u": paired_row.get("pair_dist_u", ""),
                    "fit_status": paired_row.get("fit_status", ""),
                    "tol_status": paired_row.get("tol_status", ""),
                    "tangent_status": paired_row.get("tangent_status", ""),
                    "baseline_verdict": paired_row.get("baseline_verdict", ""),
                    "baseline_reason": paired_row.get("baseline_reason", ""),
                    "formal_decision_verdict": paired_row.get("formal_decision_verdict", ""),
                    "formal_decision_reason": paired_row.get("formal_decision_reason", ""),
                    "formal_decision_gate_status": paired_row.get("formal_decision_gate_status", ""),
                    "formal_decision_gate_reason": paired_row.get("formal_decision_gate_reason", ""),
                    "formal_decision_local_rule": paired_row.get("formal_decision_local_rule", ""),
                    "formal_decision_local_status": paired_row.get("formal_decision_local_status", ""),
                    "formal_decision_local_reason": paired_row.get("formal_decision_local_reason", ""),
                    "formal_decision_verdict_pre_round2": paired_row.get("formal_decision_verdict_pre_round2", ""),
                    "formal_decision_reason_pre_round2": paired_row.get("formal_decision_reason_pre_round2", ""),
                    "formal_decision_round2_policy": paired_row.get("formal_decision_round2_policy", ""),
                    "formal_decision_round2_reason": paired_row.get("formal_decision_round2_reason", ""),
                    "formal_decision_round2_score": paired_row.get("formal_decision_round2_score", ""),
                    "formal_decision_round2_threshold": paired_row.get("formal_decision_round2_threshold", ""),
                    "formal_decision_round2_changed": paired_row.get("formal_decision_round2_changed", ""),
                    "formal_decision_round2_score_source": paired_row.get("formal_decision_round2_score_source", ""),
                    "ab_shadow_verdict": paired_row.get("ab_shadow_verdict", ""),
                    "ab_shadow_reason": paired_row.get("ab_shadow_reason", ""),
                    "tol_fail_subtype": paired_row.get("tol_fail_subtype", ""),
                    "point_score_selected_raw_max": paired_row.get("point_score_selected_raw_max", ""),
                    "needle_score_raw": paired_row.get("needle_score_raw", ""),
                    "ab_station_step_mm": paired_row.get("ab_station_step_mm", ""),
                    "pred_pos_bins": paired_row.get("pred_pos_bins", ""),
                    "pred_run_count": paired_row.get("pred_run_count", ""),
                    "pred_max_run_bins": paired_row.get("pred_max_run_bins", ""),
                    "pred_max_run_span_mm": paired_row.get("pred_max_run_span_mm", ""),
                    "pred_dominant_run_frac": paired_row.get("pred_dominant_run_frac", ""),
                    "pred_singleton_run_count": paired_row.get("pred_singleton_run_count", ""),
                    "pred_short_run_bins": paired_row.get("pred_short_run_bins", ""),
                    "pred_short_run_frac": paired_row.get("pred_short_run_frac", ""),
                    "compare_alignment_mode": paired_row.get("compare_alignment_mode", ""),
                    "ab_compare_mode": paired_row.get("ab_compare_mode", ""),
                    "ab_readout_mode": paired_row.get("ab_readout_mode", ""),
                }
            )
        rows.append(row)
    return rows


def _count_by_key(rows: List[Dict[str, Any]], key: str, value: str) -> int:
    return int(sum(1 for row in rows if str(row.get(key, "")).strip() == str(value)))


def _build_summary_row(report: Dict[str, Any]) -> Dict[str, Any]:
    needles = list(report.get("needles", [])) if isinstance(report.get("needles", []), list) else []
    round2_runtime = dict(report.get("formal_round2_runtime", {}) or {})
    return {
        "sample_uid": str(report.get("sample_uid", "")),
        "sample_id": str(report.get("sample_id", "")),
        "source_sample_id": str(report.get("sample_id", "")),
        "gt_partition": str(report.get("gt_partition", "") or ""),
        "gt_ply": str(report.get("gt_ply", "")),
        "ab_compare_mode": str(needles[0].get("ab_compare_mode", "")) if needles else "",
        "ab_readout_mode": str(needles[0].get("ab_readout_mode", "")) if needles else "",
        "formal_round2_runtime_applied": int(round2_runtime.get("applied", False)),
        "formal_round2_runtime_changed_count": int(_safe_float(round2_runtime.get("changed_count", 0.0), default=0.0)),
        "formal_round2_runtime_artifact_json": str(round2_runtime.get("artifact_json_path", "") or ""),
        "slot_total_count": int(len(needles) + len(report.get("missing_normal", []))),
        "paired_count": int(len(needles)),
        "missing_normal_count": int(len(report.get("missing_normal", []))),
        "extra_defect_count": int(len(report.get("extra_defect", []))),
        "tol_pass_count": _count_by_key(needles, "tol_status", "pass"),
        "tol_fail_count": _count_by_key(needles, "tol_status", "fail"),
        "tol_need_rescan_count": _count_by_key(needles, "tol_status", "need_rescan"),
        "baseline_pass_count": _count_by_key(needles, "baseline_verdict", "pass"),
        "baseline_fail_count": _count_by_key(needles, "baseline_verdict", "fail"),
        "baseline_need_rescan_count": _count_by_key(needles, "baseline_verdict", "need_rescan"),
        "ab_shadow_pass_count": _count_by_key(needles, "ab_shadow_verdict", "pass"),
        "ab_shadow_fail_count": _count_by_key(needles, "ab_shadow_verdict", "fail"),
        "ab_shadow_other_count": int(len(needles) - _count_by_key(needles, "ab_shadow_verdict", "pass") - _count_by_key(needles, "ab_shadow_verdict", "fail")),
        "formal_pass_count": _count_by_key(needles, "formal_decision_verdict", "pass"),
        "formal_fail_count": _count_by_key(needles, "formal_decision_verdict", "fail"),
        "formal_need_rescan_count": _count_by_key(needles, "formal_decision_verdict", "need_rescan"),
        "formal_shadow_pass_count": _count_by_key(needles, "formal_decision_verdict_shadow", "pass"),
        "formal_shadow_fail_count": _count_by_key(needles, "formal_decision_verdict_shadow", "fail"),
        "formal_shadow_need_rescan_count": _count_by_key(needles, "formal_decision_verdict_shadow", "need_rescan"),
        "formal_gate_need_rescan_count": _count_by_key(needles, "formal_decision_gate_status", "need_rescan"),
        "formal_gate_clear_count": _count_by_key(needles, "formal_decision_gate_status", "clear"),
        "shadow_gate_need_rescan_count": _count_by_key(needles, "gate_status_shadow", "need_rescan"),
        "shadow_gate_clear_count": _count_by_key(needles, "gate_status_shadow", "clear"),
        "formal_shadow_gate_need_rescan_count": _count_by_key(needles, "formal_decision_gate_status_shadow", "need_rescan"),
        "formal_shadow_gate_clear_count": _count_by_key(needles, "formal_decision_gate_status_shadow", "clear"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch tolerance-band detection on a manifest-defined sample set")
    parser.add_argument("--manifest_csv", default="", help="Optional custom manifest CSV override")
    parser.add_argument(
        "--skip_authoritative_gt_validation",
        action="store_true",
        help="Allow GT-free custom manifest inference without authoritative GT path validation.",
    )
    args = parser.parse_args()

    final_method = "tolerance_band"
    if final_method != "tolerance_band":
        raise RuntimeError("This batch entry is frozen to tolerance_band for round-1 protocol execution.")

    manifest_override = str(args.manifest_csv or "").strip()
    skip_authoritative_gt_validation = bool(args.skip_authoritative_gt_validation)
    active_ab_protocol = resolve_compare_readout_protocol_from_env()
    frozen_ab_protocol = get_frozen_compare_readout_protocol_v1()
    _guard_frozen_compare_readout_protocol(active_ab_protocol, manifest_override)

    tol_json = load_centerline_tolerance(TOLERANCE_JSON_PATH)
    tol_base_path, tol_pins_path = resolve_tolerance_reference_paths(tol_json or {})
    if tol_json is None or tol_base_path is None or tol_pins_path is None:
        raise RuntimeError(f"tolerance model / reference files not available: {TOLERANCE_JSON_PATH}")

    ref_pack = _build_reference_infos(tol_json, Path(tol_base_path), Path(tol_pins_path))
    infos_ref = ref_pack["infos_ref"]
    centerline_cfg = ref_pack["centerline_cfg"]
    r_match_u = ref_pack["r_match_u"]
    manifest_rows = (
        _load_manifest_rows_from_csv(Path(manifest_override).resolve())
        if manifest_override
        else _load_manifest_rows(DEFAULT_SPLIT_ROLE)
    )
    if not skip_authoritative_gt_validation:
        manifest_rows = validate_manifest_rows_against_authoritative_gt(manifest_rows, GT_ROOT_DEFAULT)
        sample_to_gt: Dict[str, str] = {}
        for row in manifest_rows:
            sample_key = str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
            gt_ply = str(row.get("authoritative_gt_ply", "") or row.get("gt_ply", "") or "").strip()
            prev = sample_to_gt.get(sample_key, "")
            if sample_key and prev and _normalize_match_path(prev) != _normalize_match_path(gt_ply):
                raise RuntimeError(
                    "manifest mixes multiple GT partitions/files for the same sample_uid within one detect run: "
                    f"sample_uid={sample_key} prev_gt_ply={prev} new_gt_ply={gt_ply}"
                )
            if sample_key:
                sample_to_gt[sample_key] = gt_ply
    if DEFAULT_MAX_SAMPLES > 0:
        manifest_rows = manifest_rows[:DEFAULT_MAX_SAMPLES]

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "protocol_dir": str(PROTOCOL_DIR),
                "manifest_role": str(DEFAULT_SPLIT_ROLE),
                "manifest_path": str(Path(manifest_override).resolve()) if manifest_override else (str(PROTOCOL_DIR / f"{DEFAULT_SPLIT_ROLE}_manifest.csv") if DEFAULT_SPLIT_ROLE != "all" else str(PROTOCOL_DIR)),
                "tolerance_json_path": str(TOLERANCE_JSON_PATH),
                "gt_dir": str(Path(GT_ROOT_DEFAULT).resolve()),
                "ref_base_ply": str(tol_base_path),
                "ref_pins_ply": str(tol_pins_path),
            },
            outputs={
                "testset_detection_report_json": str(Path(_DAIMA_CTX.run_dir) / "testset_detection_report.json") if getattr(_DAIMA_CTX, "run_dir", None) else "",
                "testset_detection_summary_csv": str(Path(_DAIMA_CTX.run_dir) / "testset_detection_summary.csv") if getattr(_DAIMA_CTX, "run_dir", None) else "",
                "testset_detection_per_needle_csv": str(Path(_DAIMA_CTX.run_dir) / "testset_detection_per_needle.csv") if getattr(_DAIMA_CTX, "run_dir", None) else "",
                "testset_detection_per_slot_csv": str(Path(_DAIMA_CTX.run_dir) / "testset_detection_per_slot.csv") if getattr(_DAIMA_CTX, "run_dir", None) else "",
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "protocol_split_role": str(DEFAULT_SPLIT_ROLE),
                "manifest_override_csv": str(Path(manifest_override).resolve()) if manifest_override else "",
                "max_samples": int(DEFAULT_MAX_SAMPLES),
                "final_method": str(final_method),
                "tolerance_json_path": str(TOLERANCE_JSON_PATH),
                "baseline_min_paired": 18,
                "baseline_max_icp_rmse_mm": 0.50,
                "DETECTION_DISABLE_SHORT_TOLERANCE": bool(DETECTION_DISABLE_SHORT_TOLERANCE),
                "DETECTION_DISABLE_RESCAN": bool(DETECTION_DISABLE_RESCAN),
                "FORMAL_ROUND2_RUNTIME_ENABLE": bool(False),
                "FORMAL_ROUND2_RUNTIME_ARTIFACT_JSON": str(""),
                "skip_authoritative_gt_validation": bool(skip_authoritative_gt_validation),
                "DAIMA_COMPARE_MODE": str(active_ab_protocol["compare_mode"]),
                "DAIMA_READOUT_MODE": str(active_ab_protocol["readout_mode"]),
                "DAIMA_SLACK_RADIUS_STEPS": str(active_ab_protocol["slack_radius_steps"]),
                "DAIMA_READOUT_TOPK_FRAC": str(active_ab_protocol["topk_frac"]),
                "DAIMA_READOUT_TOPK_MIN_POINTS": str(active_ab_protocol["topk_min_points"]),
                "frozen_compare_readout_protocol_text": format_compare_readout_protocol(frozen_ab_protocol),
                "active_compare_readout_protocol_text": format_compare_readout_protocol(active_ab_protocol),
                "active_compare_readout_matches_frozen": bool(compare_readout_protocol_matches_frozen(active_ab_protocol)),
            }
        )

    reports: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    per_needle_rows: List[Dict[str, Any]] = []
    per_slot_rows: List[Dict[str, Any]] = []

    for row in manifest_rows:
        report = _build_case_report(
            row,
            infos_ref=infos_ref,
            centerline_cfg=centerline_cfg,
            base_ref_path=Path(tol_base_path),
            pins_ref_path=Path(tol_pins_path),
            r_match_u=r_match_u,
            tol_json=tol_json,
        )
        reports.append(report)
        summary_rows.append(_build_summary_row(report))
        per_needle_rows.extend(_flatten_per_needle_rows(report))
        per_slot_rows.extend(_flatten_per_slot_rows(report, infos_ref))

    out_dir = Path(_DAIMA_CTX.run_dir) if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None) else (REPO_ROOT / "debug_out" / "manual_round1")
    _json_dump(out_dir / "testset_detection_report.json", {"protocol_split_role": DEFAULT_SPLIT_ROLE, "samples": reports})
    _write_csv(
        out_dir / "testset_detection_summary.csv",
        summary_rows,
        [
            "sample_uid",
            "sample_id",
            "source_sample_id",
            "gt_partition",
            "gt_ply",
            "ab_compare_mode",
            "ab_readout_mode",
            "formal_round2_runtime_applied",
            "formal_round2_runtime_changed_count",
            "formal_round2_runtime_artifact_json",
            "slot_total_count",
            "paired_count",
            "missing_normal_count",
            "extra_defect_count",
            "tol_pass_count",
            "tol_fail_count",
            "tol_need_rescan_count",
            "baseline_pass_count",
            "baseline_fail_count",
            "baseline_need_rescan_count",
            "ab_shadow_pass_count",
            "ab_shadow_fail_count",
            "ab_shadow_other_count",
            "formal_pass_count",
            "formal_fail_count",
            "formal_need_rescan_count",
            "formal_shadow_pass_count",
            "formal_shadow_fail_count",
            "formal_shadow_need_rescan_count",
            "formal_gate_need_rescan_count",
            "formal_gate_clear_count",
            "shadow_gate_need_rescan_count",
            "shadow_gate_clear_count",
            "formal_shadow_gate_need_rescan_count",
            "formal_shadow_gate_clear_count",
        ],
    )
    _write_csv(
        out_dir / "testset_detection_per_needle.csv",
        per_needle_rows,
        [
            "sample_uid",
            "sample_id",
            "source_sample_id",
            "gt_partition",
            "gt_ply",
            "stable_id",
            "ref_index",
            "def_index",
            "pair_dist_u",
            "fit_status",
            "final_used_method",
            "tol_model_status",
            "tol_status",
            "tangent_status",
            "baseline_verdict",
            "baseline_reason",
            "baseline_confidence_bucket",
            "uncertainty_flags",
            "gate_status_shadow",
            "gate_reason_shadow",
            "gate_hit_text",
            "gate_hard_hit_text",
            "gate_soft_hit_text",
            "gate_decision_rule_version",
            "gate_hard_fail_count",
            "formal_decision_rule_version",
            "formal_decision_policy",
            "formal_decision_base_source",
            "formal_decision_base_pred",
            "formal_decision_gate_rule",
            "formal_decision_gate_status",
            "formal_decision_gate_reason",
            "formal_decision_gate_hit_text",
            "formal_decision_local_rule",
            "formal_decision_local_status",
            "formal_decision_local_reason",
            "formal_decision_verdict",
            "formal_decision_reason",
            "formal_decision_rule_version_pre_round2",
            "formal_decision_policy_pre_round2",
            "formal_decision_base_source_pre_round2",
            "formal_decision_base_pred_pre_round2",
            "formal_decision_gate_rule_pre_round2",
            "formal_decision_gate_status_pre_round2",
            "formal_decision_gate_reason_pre_round2",
            "formal_decision_gate_hit_text_pre_round2",
            "formal_decision_local_rule_pre_round2",
            "formal_decision_local_status_pre_round2",
            "formal_decision_local_reason_pre_round2",
            "formal_decision_verdict_pre_round2",
            "formal_decision_reason_pre_round2",
            "formal_decision_round2_enabled",
            "formal_decision_round2_policy",
            "formal_decision_round2_base_pred",
            "formal_decision_round2_reason",
            "formal_decision_round2_score",
            "formal_decision_round2_threshold",
            "formal_decision_round2_changed",
            "formal_decision_round2_score_source",
            "formal_decision_round2_score_rank_ord",
            "formal_decision_round2_evidence_rank_sum",
            "formal_decision_round2_base_fail_count_in_sample",
            "formal_decision_round2_g1_fail_count_in_sample",
            "formal_decision_round2_artifact_json",
            "formal_decision_round2_candidate_name",
            "formal_decision_round2_threshold_tag",
            "formal_decision_policy_shadow",
            "formal_decision_base_source_shadow",
            "formal_decision_base_pred_shadow",
            "formal_decision_gate_rule_shadow",
            "formal_decision_gate_status_shadow",
            "formal_decision_gate_reason_shadow",
            "formal_decision_gate_hit_text_shadow",
            "formal_decision_local_rule_shadow",
            "formal_decision_local_status_shadow",
            "formal_decision_local_reason_shadow",
            "formal_decision_verdict_shadow",
            "formal_decision_reason_shadow",
            "tol_fail_subtype",
            "point_score_raw_max",
            "point_score_clipped_max",
            "point_score_fixed_raw_max",
            "point_score_slack_raw_max",
            "point_score_selected_raw_max",
            "needle_score_raw",
            "needle_score_clipped",
            "needle_score_q95",
            "needle_score_topk_mean",
            "needle_score_topk_count",
            "ab_station_step_mm",
            "pred_pos_bins",
            "pred_run_count",
            "pred_max_run_bins",
            "pred_max_run_span_mm",
            "pred_dominant_run_frac",
            "pred_singleton_run_count",
            "pred_short_run_bins",
            "pred_short_run_frac",
            "ab_compare_mode",
            "ab_readout_mode",
            "ab_slack_radius_steps",
            "ab_shadow_verdict",
            "ab_shadow_reason",
            "ab_slack_gain",
            "warp_severity",
            "ab_warp_p90_steps",
            "ab_warp_max_steps",
            "ab_warp_p90_mm",
            "ab_warp_max_mm",
            "ab_warp_nonzero_ratio",
            "max_lateral_ratio",
            "tip_short_ratio",
            "len_short_ratio",
            "max_tangent_ratio",
            "compare_alignment_mode",
            "gate_source_fields_text",
        ],
    )
    _write_csv(
        out_dir / "testset_detection_per_slot.csv",
        per_slot_rows,
        [
            "sample_uid",
            "sample_id",
            "source_sample_id",
            "gt_partition",
            "gt_ply",
            "stable_id",
            "ref_index",
            "slot_total_count",
            "sample_paired_count",
            "sample_missing_count",
            "sample_extra_defect_count",
            "pred_present",
            "pred_missing",
            "pred_pair_status",
            "pred_missing_reason",
            "pred_geo_eval_eligible",
            "def_index",
            "pair_dist_u",
            "fit_status",
            "tol_status",
            "tangent_status",
            "baseline_verdict",
            "baseline_reason",
            "formal_decision_verdict",
            "formal_decision_reason",
            "formal_decision_gate_status",
            "formal_decision_gate_reason",
            "formal_decision_local_rule",
            "formal_decision_local_status",
            "formal_decision_local_reason",
            "formal_decision_verdict_pre_round2",
            "formal_decision_reason_pre_round2",
            "formal_decision_round2_policy",
            "formal_decision_round2_reason",
            "formal_decision_round2_score",
            "formal_decision_round2_threshold",
            "formal_decision_round2_changed",
            "formal_decision_round2_score_source",
            "ab_shadow_verdict",
            "ab_shadow_reason",
            "tol_fail_subtype",
            "point_score_selected_raw_max",
            "needle_score_raw",
            "ab_station_step_mm",
            "pred_pos_bins",
            "pred_run_count",
            "pred_max_run_bins",
            "pred_max_run_span_mm",
            "pred_dominant_run_frac",
            "pred_singleton_run_count",
            "pred_short_run_bins",
            "pred_short_run_frac",
            "compare_alignment_mode",
            "ab_compare_mode",
            "ab_readout_mode",
        ],
    )
    print(f"[INFO] Wrote round-1 batch outputs under: {out_dir}")


if __name__ == "__main__":
    main()
