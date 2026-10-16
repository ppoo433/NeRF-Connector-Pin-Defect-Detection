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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHORITATIVE_GEO_RAW = ""
DEFAULT_DETECTION_PER_SLOT_RAW = ""


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _normalize_sample_uid(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_uid", "") or row.get("sample_id", "") or row.get("source_sample_id", "") or "").strip()


def _normalize_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return (_normalize_sample_uid(row), int(_safe_int(row.get("stable_id", -1), default=-1)))


def _latest_artifact(filename: str) -> Path:
    runs_root = REPO_ROOT / "debug_out" / "runs"
    cands = sorted(runs_root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No {filename} found under: {runs_root}")
    return cands[0].resolve()


def _resolve_existing_path(raw: str, default_path: Path) -> Path:
    text = str(raw or "").strip()
    if text:
        path = Path(text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    return default_path.resolve()


def default_authoritative_geo_csv() -> Path:
    path = Path(DEFAULT_AUTHORITATIVE_GEO_RAW)
    if path.is_file():
        return path.resolve()
    try:
        return _latest_artifact("manual_authoritative_geo_per_slot.csv")
    except FileNotFoundError:
        return _latest_artifact("authoritative_geo_per_slot.csv")


def default_detection_per_slot_csv() -> Path:
    path = Path(DEFAULT_DETECTION_PER_SLOT_RAW)
    if path.is_file():
        return path.resolve()
    try:
        return _latest_artifact("train_combined_detection_per_slot.csv")
    except FileNotFoundError:
        return _latest_artifact("testset_detection_per_slot.csv")


def _parse_sample_ids(raw: str) -> Optional[set[str]]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return None
    return set(values)


def _default_output_dir() -> Path:
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None):
        return Path(_DAIMA_CTX.run_dir).resolve()
    return (REPO_ROOT / "debug_out" / "manual_feature_bank").resolve()


def build_feature_bank_rows(
    *,
    authoritative_geo_csv: Path,
    detection_per_slot_csv: Path,
    sample_ids: Optional[Iterable[str]] = None,
    include_geo_excluded: bool = False,
) -> List[Dict[str, Any]]:
    authoritative_rows = _read_csv(authoritative_geo_csv)
    detection_rows = _read_csv(detection_per_slot_csv)
    allowed = set(sample_ids or [])
    detection_by_key = {_normalize_key(row): row for row in detection_rows}
    rows: List[Dict[str, Any]] = []
    for gt_row in authoritative_rows:
        sample_uid = _normalize_sample_uid(gt_row)
        if allowed and sample_uid not in allowed:
            continue
        geo_included = int(_safe_int(gt_row.get("geo_included", 1), default=1))
        if not include_geo_excluded and geo_included != 1:
            continue
        stable_id = int(_safe_int(gt_row.get("stable_id", -1), default=-1))
        det_row = detection_by_key.get((sample_uid, stable_id), {})
        gt_abnormal = int(_safe_int(gt_row.get("gt_abnormal", 0), default=0))
        current_geo_pred = str(gt_row.get("geo_pred", "")).strip()
        row: Dict[str, Any] = {
            "sample_uid": sample_uid,
            "sample_id": str(gt_row.get("sample_id", "")).strip(),
            "source_sample_id": str(gt_row.get("source_sample_id", "")).strip(),
            "target_id": str(gt_row.get("target_id", "")).strip(),
            "gt_partition": str(gt_row.get("gt_partition", "")).strip(),
            "is_augmented": int(_safe_int(gt_row.get("is_augmented", 0), default=0)),
            "augmentation_name": str(gt_row.get("augmentation_name", "")).strip(),
            "augmentation_source_sample_id": str(gt_row.get("augmentation_source_sample_id", "")).strip(),
            "origin_sample_uid": str(gt_row.get("origin_sample_uid", "")).strip(),
            "origin_source_sample_id": str(gt_row.get("origin_source_sample_id", "")).strip(),
            "cv_group_uid": str(gt_row.get("cv_group_uid", "") or sample_uid).strip(),
            "cv_group_reason": str(gt_row.get("cv_group_reason", "")).strip(),
            "stable_id": stable_id,
            "gt_abnormal": gt_abnormal,
            "geo_included": geo_included,
            "geo_case": str(gt_row.get("geo_case", "")).strip(),
            "geo_pred": current_geo_pred,
            "accepted": int(_safe_int(gt_row.get("accepted", 0), default=0)),
            "need_rescan": int(_safe_int(gt_row.get("need_rescan", 0), default=0)),
            "geo_reject_reason": str(gt_row.get("geo_reject_reason", "")).strip(),
            "matched_detection": int(bool(det_row)),
            "formal_decision_verdict": str(det_row.get("formal_decision_verdict", "")).strip(),
            "formal_decision_reason": str(det_row.get("formal_decision_reason", "")).strip(),
            "baseline_verdict": str(det_row.get("baseline_verdict", "")).strip(),
            "baseline_reason": str(det_row.get("baseline_reason", "")).strip(),
            "ab_shadow_verdict": str(det_row.get("ab_shadow_verdict", "")).strip(),
            "ab_shadow_reason": str(det_row.get("ab_shadow_reason", "")).strip(),
            "tol_fail_subtype": str(det_row.get("tol_fail_subtype", "")).strip(),
            "fit_status": str(det_row.get("fit_status", "")).strip(),
            "compare_alignment_mode": str(det_row.get("compare_alignment_mode", "")).strip(),
            "ab_compare_mode": str(det_row.get("ab_compare_mode", "")).strip(),
            "ab_readout_mode": str(det_row.get("ab_readout_mode", "")).strip(),
            "needle_score_raw": _safe_float(det_row.get("needle_score_raw", float("nan"))),
            "point_score_selected_raw_max": _safe_float(det_row.get("point_score_selected_raw_max", float("nan"))),
            "pred_max_run_span_mm": _safe_float(det_row.get("pred_max_run_span_mm", float("nan"))),
            "pred_dominant_run_frac": _safe_float(det_row.get("pred_dominant_run_frac", float("nan"))),
            "pred_run_count": _safe_float(det_row.get("pred_run_count", float("nan"))),
            "pred_pos_bins": _safe_float(det_row.get("pred_pos_bins", float("nan"))),
            "pred_short_run_frac": _safe_float(det_row.get("pred_short_run_frac", float("nan"))),
            "pred_short_run_bins": _safe_float(det_row.get("pred_short_run_bins", float("nan"))),
            "pred_max_run_bins": _safe_float(det_row.get("pred_max_run_bins", float("nan"))),
            "pair_dist_u": _safe_float(det_row.get("pair_dist_u", float("nan"))),
            "is_main_chain_candidate": int(
                str(det_row.get("tol_fail_subtype", "")).strip() == "lateral-only"
                and str(det_row.get("ab_shadow_reason", "")).strip() == "point_readout_positive"
            ),
            "is_current_fp": int(gt_abnormal == 0 and current_geo_pred == "fail"),
            "is_current_tp": int(gt_abnormal == 1 and current_geo_pred == "fail"),
            "is_current_fn": int(gt_abnormal == 1 and current_geo_pred != "fail"),
            "is_current_tn": int(gt_abnormal == 0 and current_geo_pred != "fail"),
        }
        rows.append(row)
    return rows


def build_feature_bank_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    unique_samples = sorted({str(row.get("sample_uid", "")) for row in rows if str(row.get("sample_uid", ""))})
    main_chain_rows = [row for row in rows if int(_safe_int(row.get("is_main_chain_candidate", 0), default=0)) == 1]
    return {
        "schema_version": 1,
        "row_count": int(len(rows)),
        "sample_count": int(len(unique_samples)),
        "sample_ids": unique_samples,
        "positive_count": int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in rows)),
        "negative_count": int(sum(1 for row in rows if int(_safe_int(row.get("gt_abnormal", 0), default=0)) == 0)),
        "main_chain_count": int(len(main_chain_rows)),
        "main_chain_positive_count": int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in main_chain_rows)),
        "main_chain_negative_count": int(sum(1 for row in main_chain_rows if int(_safe_int(row.get("gt_abnormal", 0), default=0)) == 0)),
        "current_geo_fp": int(sum(int(_safe_int(row.get("is_current_fp", 0), default=0)) for row in rows)),
        "current_geo_tp": int(sum(int(_safe_int(row.get("is_current_tp", 0), default=0)) for row in rows)),
        "current_geo_fn": int(sum(int(_safe_int(row.get("is_current_fn", 0), default=0)) for row in rows)),
        "current_geo_tn": int(sum(int(_safe_int(row.get("is_current_tn", 0), default=0)) for row in rows)),
    }


def run_build(
    *,
    authoritative_geo_csv: Path,
    detection_per_slot_csv: Path,
    sample_ids: Optional[Iterable[str]] = None,
    include_geo_excluded: bool = False,
) -> Tuple[Path, Path]:
    rows = build_feature_bank_rows(
        authoritative_geo_csv=authoritative_geo_csv,
        detection_per_slot_csv=detection_per_slot_csv,
        sample_ids=sample_ids,
        include_geo_excluded=include_geo_excluded,
    )
    summary = build_feature_bank_summary(rows)
    out_dir = _default_output_dir()
    feature_bank_csv = (out_dir / "feature_bank_38.csv").resolve()
    summary_json = (out_dir / "feature_bank_summary.json").resolve()
    fieldnames = list(rows[0].keys()) if rows else [
        "sample_uid",
        "stable_id",
        "gt_abnormal",
        "needle_score_raw",
    ]
    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "authoritative_geo_csv": str(authoritative_geo_csv),
                "detection_per_slot_csv": str(detection_per_slot_csv),
            },
            outputs={
                "feature_bank_csv": str(feature_bank_csv),
                "summary_json": str(summary_json),
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "sample_ids": sorted(set(sample_ids or [])),
                "include_geo_excluded": bool(include_geo_excluded),
            }
        )
    _write_csv(feature_bank_csv, rows, fieldnames=fieldnames)
    _write_json(summary_json, summary)
    return feature_bank_csv, summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline feature bank for tolerance_band_v1 analysis.")
    parser.add_argument("--authoritative_geo_csv", default=DEFAULT_AUTHORITATIVE_GEO_RAW, help="Path to merged authoritative geo per-slot CSV.")
    parser.add_argument("--detection_per_slot_csv", default=DEFAULT_DETECTION_PER_SLOT_RAW, help="Path to detection per-slot CSV.")
    parser.add_argument("--sample_ids", default="", help="Optional comma-separated sample_uid allowlist.")
    parser.add_argument("--include_geo_excluded", action="store_true", help="Include rows where geo_included != 1.")
    args = parser.parse_args()

    authoritative_geo_csv = _resolve_existing_path(args.authoritative_geo_csv, default_authoritative_geo_csv())
    detection_per_slot_csv = _resolve_existing_path(args.detection_per_slot_csv, default_detection_per_slot_csv())
    sample_ids = _parse_sample_ids(args.sample_ids)
    feature_bank_csv, summary_json = run_build(
        authoritative_geo_csv=authoritative_geo_csv,
        detection_per_slot_csv=detection_per_slot_csv,
        sample_ids=sample_ids,
        include_geo_excluded=bool(args.include_geo_excluded),
    )
    print(str(feature_bank_csv))
    print(str(summary_json))


if __name__ == "__main__":
    main()
