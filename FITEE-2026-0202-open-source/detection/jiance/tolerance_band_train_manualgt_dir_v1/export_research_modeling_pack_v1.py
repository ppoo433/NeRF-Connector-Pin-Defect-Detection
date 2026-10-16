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
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from . import analyze_feature_separability_v1 as separability_v1
    from . import build_feature_lab_v1 as feature_lab_v1
    from .build_geo_shadow_feature_bank_v3 import REPO_ROOT, _resolve_existing_path, _safe_float, _safe_int
except ImportError:
    from tolerance_band_train_manualgt_dir_v1 import analyze_feature_separability_v1 as separability_v1
    from tolerance_band_train_manualgt_dir_v1 import build_feature_lab_v1 as feature_lab_v1
    from tolerance_band_train_manualgt_dir_v1.build_geo_shadow_feature_bank_v3 import (
        REPO_ROOT,
        _resolve_existing_path,
        _safe_float,
        _safe_int,
    )


DEFAULT_OUTPUT_ROOT_RAW = str(REPO_ROOT / "research_pack")
DEFAULT_TEST_GEO_GT_RAW = str(
    REPO_ROOT
    / "debug_out"
    / "runs"
    / "20260429_200347_746160_4548"
    / "authoritative_geo_per_slot.csv"
)
DEFAULT_TEST_GEO_EXCLUDED_RAW = str(
    REPO_ROOT
    / "debug_out"
    / "runs"
    / "20260429_200347_746160_4548"
    / "authoritative_geo_excluded_slots.csv"
)
DEFAULT_TOP_N = 160

LABEL_COLUMNS = {
    "gt_abnormal",
    "gt_label",
    "label_available",
    "test_gt_label_available",
    "evaluation_only",
    "do_not_use_for_training_or_tuning",
}
SPLIT_COLUMNS = {
    "split_role",
    "cv_group_uid",
    "cv_group_reason",
    "sample_uid",
    "sample_id",
    "source_sample_id",
    "target_id",
    "stable_id",
    "gt_partition",
    "is_augmented",
    "augmentation_name",
    "augmentation_source_sample_id",
    "origin_sample_uid",
    "origin_source_sample_id",
}
AUDIT_COLUMNS = {
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
    "test_gt_geo_included",
    "test_gt_geo_case",
    "test_gt_geo_exclusion_reason",
    "test_gt_join_status",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {str(k or "").replace("\ufeff", "").strip(): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _sample_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    sample_uid = str(row.get("sample_uid", "") or row.get("sample_id", "") or "").strip()
    stable_id = int(_safe_int(row.get("stable_id", -1), default=-1))
    return sample_uid, stable_id


def _stable_sort_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    sample_uid, stable_id = _sample_key(row)
    return sample_uid, stable_id, str(row.get("split_role", ""))


def _unique_output_dir(output_root: Path) -> Path:
    base = output_root / f"tolerance_band_direction_feature_modeling_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not base.exists():
        return base.resolve()
    for idx in range(2, 1000):
        cand = Path(f"{base}_{idx:03d}")
        if not cand.exists():
            return cand.resolve()
    raise RuntimeError(f"Could not allocate unique research pack directory under {output_root}")


def _fieldnames(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for key in preferred:
        seen[str(key)] = None
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen[key_s] = None
    return list(seen.keys())


def _finite_values(rows: Sequence[Mapping[str, Any]], column: str) -> np.ndarray:
    values = np.asarray([_safe_float(row.get(column, float("nan"))) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size <= 0 or b.size <= 0:
        return float("nan")
    values = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, values, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _feature_family_map(feature_columns: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for family in (
        "base_numeric",
        "event",
        "shape",
        "enhanced_shape",
        "direction_summary",
        "direction_event",
        "relative",
        "group",
        "slot_context",
        "common_mode",
        "neighbor",
        "sequence",
        "direction_sequence",
    ):
        for col in feature_columns.get(family, []) or []:
            out[str(col)] = family
    for col in feature_columns.get("hybrid_v3", []) or []:
        col_s = str(col)
        if col_s.startswith("seq_raw_"):
            out[col_s] = "sequence"
        elif col_s.startswith("dir_seq_margin_"):
            out[col_s] = "direction_sequence"
        elif col_s not in out:
            out[col_s] = "other_feature"
    return out


def _column_role(column: str, training_features: set[str]) -> str:
    if column in training_features:
        return "feature"
    if column in LABEL_COLUMNS:
        return "label"
    if column in SPLIT_COLUMNS:
        return "meta"
    if column in AUDIT_COLUMNS:
        return "audit"
    return "guard"


def _leakage_risk(column: str, family: str, role: str) -> str:
    if role == "label":
        return "high_label"
    if column in {"is_current_fp", "is_current_tp", "is_current_fn", "is_current_tn"}:
        return "high_gt_derived_audit"
    if column.startswith("test_gt_") or column in {"evaluation_only", "do_not_use_for_training_or_tuning"}:
        return "high_eval_only"
    if family in {"slot_context", "group", "common_mode", "neighbor"}:
        return "medium_context_validate_by_grouped_cv"
    if role != "feature":
        return "not_for_training"
    return "low"


def _catalog_rows(
    *,
    fieldnames: Sequence[str],
    feature_columns: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    training_features = set(str(col) for col in feature_columns.get("hybrid_v3", []) or [])
    family_by_col = _feature_family_map(feature_columns)
    rows: List[Dict[str, Any]] = []
    for col in fieldnames:
        role = _column_role(str(col), training_features)
        family = family_by_col.get(str(col), "")
        rows.append(
            {
                "column": str(col),
                "role": role,
                "family": family if family else role,
                "allowed_for_training": int(role == "feature"),
                "leakage_risk": _leakage_risk(str(col), family, role),
                "notes": _column_notes(str(col), role, family),
            }
        )
    return rows


def _column_notes(column: str, role: str, family: str) -> str:
    if column == "cv_group_uid":
        return "Use this as the grouping column for CV; never randomly split pins from the same group."
    if column == "gt_abnormal":
        return "Binary target for train rows; for test rows it is evaluation-only."
    if column == "label_available":
        return "1 for train labels; test-blind rows keep labels unavailable."
    if column == "do_not_use_for_training_or_tuning":
        return "Rows marked 1 must not affect feature/model/threshold/postprocess selection."
    if role == "feature" and family == "sequence":
        return "Raw point-score sequence bin feature."
    if role == "feature" and family == "direction_sequence":
        return "Raw tangent-margin sequence bin feature."
    if role == "feature" and family in {"direction_summary", "direction_event"}:
        return "Direction tolerance-band feature derived from tangent cone residuals; validate by grouped CV."
    if role == "feature" and family in {"common_mode", "group", "relative"}:
        return "Sample-relative feature; validate with grouped CV to avoid same-sample leakage."
    if role == "feature" and family == "neighbor":
        return "Adjacent-pin context feature; validate with grouped CV."
    if role == "audit":
        return "Audit or detector-status column; not part of the default training feature set."
    if role == "guard":
        return "Conservative non-feature column unless you explicitly audit it."
    return ""


def _add_train_flags(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        label = int(_safe_int(row.get("gt_abnormal", 0), default=0))
        row["gt_abnormal"] = label
        row["gt_label"] = "abnormal" if label == 1 else "normal"
        row["label_available"] = 1
        row["evaluation_only"] = 0
        row["do_not_use_for_training_or_tuning"] = 0
        row["test_gt_label_available"] = ""
        row["test_gt_geo_included"] = ""
        row["test_gt_geo_case"] = ""
        row["test_gt_geo_exclusion_reason"] = ""
        row["test_gt_join_status"] = "not_test"
        out.append(row)
    return sorted(out, key=_stable_sort_key)


def _load_test_gt_maps(geo_gt_csv: Path, excluded_csv: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, str]], Dict[Tuple[str, int], Dict[str, str]]]:
    included = {_sample_key(row): dict(row) for row in _read_csv(geo_gt_csv)}
    excluded = {_sample_key(row): dict(row) for row in _read_csv(excluded_csv)}
    return included, excluded


def _add_test_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    included_gt: Mapping[Tuple[str, int], Mapping[str, Any]],
    excluded_gt: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labeled: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    seen_feature_keys: set[Tuple[str, int]] = set()
    for raw in rows:
        key = _sample_key(raw)
        seen_feature_keys.add(key)
        row = dict(raw)
        row["evaluation_only"] = 1
        row["do_not_use_for_training_or_tuning"] = 1
        gt_row = included_gt.get(key)
        excl_row = excluded_gt.get(key)
        if gt_row is not None:
            row["gt_abnormal"] = int(_safe_int(gt_row.get("gt_abnormal", 0), default=0))
            row["gt_label"] = str(gt_row.get("gt_label", "") or ("abnormal" if row["gt_abnormal"] == 1 else "normal"))
            row["label_available"] = 1
            row["test_gt_label_available"] = 1
            row["test_gt_geo_included"] = int(_safe_int(gt_row.get("geo_included", 1), default=1))
            row["test_gt_geo_case"] = str(gt_row.get("geo_case", ""))
            row["test_gt_geo_exclusion_reason"] = ""
            row["test_gt_join_status"] = "matched_included"
        elif excl_row is not None:
            row["gt_abnormal"] = ""
            row["gt_label"] = ""
            row["label_available"] = 0
            row["test_gt_label_available"] = 0
            row["test_gt_geo_included"] = int(_safe_int(excl_row.get("geo_included", 0), default=0))
            row["test_gt_geo_case"] = ""
            row["test_gt_geo_exclusion_reason"] = str(excl_row.get("geo_exclusion_reason", ""))
            row["test_gt_join_status"] = "matched_excluded"
        else:
            row["gt_abnormal"] = ""
            row["gt_label"] = ""
            row["label_available"] = 0
            row["test_gt_label_available"] = 0
            row["test_gt_geo_included"] = ""
            row["test_gt_geo_case"] = ""
            row["test_gt_geo_exclusion_reason"] = ""
            row["test_gt_join_status"] = "missing_gt_key"
        labeled.append(row)
        audit_rows.append(
            {
                "sample_uid": key[0],
                "stable_id": key[1],
                "feature_row_present": 1,
                "gt_included_present": int(gt_row is not None),
                "gt_excluded_present": int(excl_row is not None),
                "join_status": row["test_gt_join_status"],
                "gt_abnormal": row.get("gt_abnormal", ""),
                "gt_label": row.get("gt_label", ""),
                "geo_exclusion_reason": row.get("test_gt_geo_exclusion_reason", ""),
            }
        )

    for key, gt_row in included_gt.items():
        if key not in seen_feature_keys:
            audit_rows.append(
                {
                    "sample_uid": key[0],
                    "stable_id": key[1],
                    "feature_row_present": 0,
                    "gt_included_present": 1,
                    "gt_excluded_present": 0,
                    "join_status": "gt_included_without_feature",
                    "gt_abnormal": gt_row.get("gt_abnormal", ""),
                    "gt_label": gt_row.get("gt_label", ""),
                    "geo_exclusion_reason": "",
                }
            )
    for key, gt_row in excluded_gt.items():
        if key not in seen_feature_keys:
            audit_rows.append(
                {
                    "sample_uid": key[0],
                    "stable_id": key[1],
                    "feature_row_present": 0,
                    "gt_included_present": 0,
                    "gt_excluded_present": 1,
                    "join_status": "gt_excluded_without_feature",
                    "gt_abnormal": "",
                    "gt_label": "",
                    "geo_exclusion_reason": gt_row.get("geo_exclusion_reason", ""),
                }
            )
    return sorted(labeled, key=_stable_sort_key), sorted(audit_rows, key=lambda r: (str(r.get("sample_uid", "")), int(_safe_int(r.get("stable_id", -1), -1))))


def _test_blind_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["gt_abnormal"] = ""
        row["gt_label"] = ""
        row["label_available"] = 0
        row["evaluation_only"] = 0
        row["do_not_use_for_training_or_tuning"] = 1
        row["test_gt_label_available"] = 0
        row["test_gt_geo_included"] = ""
        row["test_gt_geo_case"] = ""
        row["test_gt_geo_exclusion_reason"] = ""
        row["test_gt_join_status"] = "blind_no_gt_join"
        out.append(row)
    return sorted(out, key=_stable_sort_key)


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("cv_group_uid", "") or row.get("sample_uid", "")), []).append(row)
    out: List[Dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        positives = int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in group_rows))
        sample_count = len({str(row.get("sample_uid", "")) for row in group_rows})
        out.append(
            {
                "cv_group_uid": group,
                "row_count": int(len(group_rows)),
                "sample_count": int(sample_count),
                "positive_count": positives,
                "negative_count": int(len(group_rows) - positives),
                "positive_rate": float(positives / len(group_rows)) if group_rows else 0.0,
            }
        )
    return out


def _stable_id_summary(rows: Sequence[Mapping[str, Any]], *, split_name: str) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        if int(_safe_int(row.get("label_available", 0), default=0)) != 1:
            continue
        grouped.setdefault(int(_safe_int(row.get("stable_id", -1), default=-1)), []).append(row)
    out: List[Dict[str, Any]] = []
    for stable_id, group_rows in sorted(grouped.items()):
        positives = int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in group_rows))
        out.append(
            {
                "split": split_name,
                "stable_id": int(stable_id),
                "row_count": int(len(group_rows)),
                "positive_count": positives,
                "negative_count": int(len(group_rows) - positives),
                "positive_rate": float(positives / len(group_rows)) if group_rows else 0.0,
            }
        )
    return out


def _missingness_report(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for col in feature_columns:
        train_vals = _finite_values(train_rows, str(col))
        test_vals = _finite_values(test_rows, str(col))
        train_missing = float(1.0 - (train_vals.size / max(len(train_rows), 1)))
        test_missing = float(1.0 - (test_vals.size / max(len(test_rows), 1)))
        train_unique = int(np.unique(train_vals).size) if train_vals.size else 0
        test_unique = int(np.unique(test_vals).size) if test_vals.size else 0
        train_std = float(np.std(train_vals)) if train_vals.size else float("nan")
        test_std = float(np.std(test_vals)) if test_vals.size else float("nan")
        drift_ks = _ks_stat(train_vals, test_vals)
        out.append(
            {
                "feature": str(col),
                "train_finite_count": int(train_vals.size),
                "train_missing_frac": train_missing,
                "train_unique_count": train_unique,
                "train_is_constant": int(train_unique <= 1),
                "train_std": train_std,
                "test_finite_count": int(test_vals.size),
                "test_missing_frac": test_missing,
                "test_unique_count": test_unique,
                "test_is_constant": int(test_unique <= 1),
                "test_std": test_std,
                "train_test_ks": drift_ks,
                "train_median": float(np.median(train_vals)) if train_vals.size else float("nan"),
                "test_median": float(np.median(test_vals)) if test_vals.size else float("nan"),
            }
        )
    return sorted(out, key=lambda r: (float(_safe_float(r.get("train_missing_frac", 0.0))), -float(_safe_float(r.get("train_test_ks", 0.0)))), reverse=True)


def _copy_text_csv(src: Path, dst: Path) -> None:
    _write_text(dst, src.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256(path),
    }


def _manifest_file_record(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": "self_referential_manifest_not_hashed",
    }


def _write_readme(
    path: Path,
    *,
    pack_dir: Path,
    manifest: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    test_summary: Mapping[str, Any],
    feature_count: int,
) -> None:
    text = f"""# tolerance_band feature modeling research pack

Generated: {manifest.get("generated_at", "")}

This pack is for feature exploration and model selection outside the current HGB/postprocess pipeline.

## Files

- `train_labeled_features.csv`: supervised training table. Use this for feature selection, model selection, threshold tuning, and grouped CV.
- `test_labeled_features_eval_only.csv`: test feature table joined with authoritative geo GT. This is evaluation-only and must not be used for feature/model/threshold/postprocess selection.
- `test_blind_features.csv`: test feature table without GT labels. Use this for future blind inference.
- `combined_labeled_features_for_audit.csv`: train plus eval-only test rows in one schema-aligned table for visualization/audit only.
- `feature_column_catalog.csv`: column roles, feature families, training eligibility, and leakage risk.
- `feature_separability_train.csv`: training-label single-feature AP/AUC/KS/F1 diagnostics.
- `feature_separability_top.csv`: top single-feature diagnostics.
- `feature_family_summary.csv`: feature-family-level separability summary.
- `feature_missingness_report.csv`: train/test missingness, constant-column, low-variance, and drift hints.
- `train_group_summary.csv`: `cv_group_uid` distribution for grouped CV.
- `train_label_balance_by_stable_id.csv`: training label distribution by stable slot.
- `test_label_balance_by_stable_id_eval_only.csv`: test label distribution by stable slot, eval-only.
- `test_gt_join_audit.csv`: join audit between test features and authoritative geo GT/excluded slots.
- `manifest.json`: input/output paths, counts, hashes, and generation config.

## Row Counts

- Train rows: {train_summary.get("row_count")}
- Train samples: {train_summary.get("sample_count")}
- Train positives: {train_summary.get("positive_count")}
- Train negatives: {train_summary.get("negative_count")}
- Test feature rows: {test_summary.get("row_count")}
- Test samples: {test_summary.get("sample_count")}
- Test labeled geo-included rows: {test_summary.get("labeled_included_count")}
- Test excluded rows: {test_summary.get("excluded_count")}
- Feature columns allowed by default: {feature_count}

## Target

The target is `gt_abnormal`:

- `1`: abnormal needle
- `0`: normal needle

Use `gt_label` only as a human-readable copy.

## Grouped CV Requirement

For all training-side experiments, split by `cv_group_uid`. Do not randomly split individual needles from the same point cloud or augmented family across train/validation folds.

Recommended split object: `StratifiedGroupKFold`.

## Training Columns

Use only columns where `feature_column_catalog.csv` has:

- `role = feature`
- `allowed_for_training = 1`

Do not train on label, audit, guard, or eval-only columns.

## Test-Set Rule

`test_labeled_features_eval_only.csv` is provided because you requested a complete research table, but it must remain evaluation-only. It may be used for manual error analysis and final reporting. It must not affect feature selection, model choice, threshold choice, postprocess design, or hyperparameter tuning.

## Suggested Modeling Workflow

1. Load `train_labeled_features.csv`.
2. Load `feature_column_catalog.csv` and keep `allowed_for_training == 1`.
3. Use `gt_abnormal` as `y`.
4. Use `cv_group_uid` as `groups`.
5. Compare simple models first: Logistic Regression, ExtraTrees, RandomForest, HGB.
6. Select thresholds from grouped OOF scores, not training in-sample scores.
7. Use `test_blind_features.csv` for blind inference.
8. Use `test_labeled_features_eval_only.csv` only for final evaluation/audit.

## Pack Directory

`{pack_dir}`
"""
    _write_text(path, text)


def _default_pack_output_root(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path(DEFAULT_OUTPUT_ROOT_RAW).resolve()
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def run_export(
    *,
    output_root: Path,
    test_geo_gt_csv: Path,
    test_geo_excluded_csv: Path,
    top_n: int,
) -> Dict[str, Any]:
    pack_dir = _unique_output_dir(output_root)
    pack_dir.mkdir(parents=True, exist_ok=False)

    feature_lab_v1._DAIMA_CTX = _DAIMA_CTX
    separability_v1._DAIMA_CTX = _DAIMA_CTX

    lab_outputs = feature_lab_v1.run_build(
        train_feature_bank_csv=feature_lab_v1.default_train_feature_bank_csv(),
        train_summary_json=feature_lab_v1.default_train_summary_json(),
        test_detection_per_slot_csv=feature_lab_v1.default_test_detection_per_slot_csv(),
        test_detection_report_json=feature_lab_v1.default_test_detection_report_json(),
        max_train_samples=0,
        max_test_samples=0,
    )
    feature_lab_summary_json = Path(lab_outputs["summary_json"]).resolve()
    feature_lab_summary = _load_json(feature_lab_summary_json)
    train_rows = _add_train_flags(_read_csv(Path(lab_outputs["train_csv"]).resolve()))
    test_blind_source_rows = _test_blind_rows(_read_csv(Path(lab_outputs["test_blind_csv"]).resolve()))

    included_gt, excluded_gt = _load_test_gt_maps(test_geo_gt_csv, test_geo_excluded_csv)
    test_labeled_rows, test_join_audit = _add_test_labels(
        test_blind_source_rows,
        included_gt=included_gt,
        excluded_gt=excluded_gt,
    )

    feature_columns = dict(feature_lab_summary.get("feature_columns", {}) or {})
    train_feature_cols = [str(col) for col in feature_columns.get("hybrid_v3", []) or []]
    combined_rows = train_rows + test_labeled_rows
    preferred = list(dict.fromkeys(list(feature_lab_v1.META_COLUMNS) + [
        "gt_label",
        "test_gt_label_available",
        "evaluation_only",
        "do_not_use_for_training_or_tuning",
        "test_gt_geo_included",
        "test_gt_geo_case",
        "test_gt_geo_exclusion_reason",
        "test_gt_join_status",
    ] + train_feature_cols))
    fields = _fieldnames(combined_rows, preferred)

    train_csv = pack_dir / "train_labeled_features.csv"
    test_labeled_csv = pack_dir / "test_labeled_features_eval_only.csv"
    test_blind_csv = pack_dir / "test_blind_features.csv"
    combined_csv = pack_dir / "combined_labeled_features_for_audit.csv"
    catalog_csv = pack_dir / "feature_column_catalog.csv"
    missingness_csv = pack_dir / "feature_missingness_report.csv"
    train_group_csv = pack_dir / "train_group_summary.csv"
    train_stable_csv = pack_dir / "train_label_balance_by_stable_id.csv"
    test_stable_csv = pack_dir / "test_label_balance_by_stable_id_eval_only.csv"
    join_audit_csv = pack_dir / "test_gt_join_audit.csv"
    readme_md = pack_dir / "README.md"
    manifest_json = pack_dir / "manifest.json"

    _write_csv(train_csv, train_rows, fields)
    _write_csv(test_labeled_csv, test_labeled_rows, fields)
    _write_csv(test_blind_csv, test_blind_source_rows, fields)
    _write_csv(combined_csv, combined_rows, fields)
    catalog_rows = _catalog_rows(fieldnames=fields, feature_columns=feature_columns)
    _write_csv(catalog_csv, catalog_rows, ["column", "role", "family", "allowed_for_training", "leakage_risk", "notes"])
    _write_csv(missingness_csv, _missingness_report(train_rows=train_rows, test_rows=test_blind_source_rows, feature_columns=train_feature_cols), [
        "feature",
        "train_finite_count",
        "train_missing_frac",
        "train_unique_count",
        "train_is_constant",
        "train_std",
        "test_finite_count",
        "test_missing_frac",
        "test_unique_count",
        "test_is_constant",
        "test_std",
        "train_test_ks",
        "train_median",
        "test_median",
    ])
    _write_csv(train_group_csv, _group_summary(train_rows), ["cv_group_uid", "row_count", "sample_count", "positive_count", "negative_count", "positive_rate"])
    _write_csv(train_stable_csv, _stable_id_summary(train_rows, split_name="train"), ["split", "stable_id", "row_count", "positive_count", "negative_count", "positive_rate"])
    _write_csv(test_stable_csv, _stable_id_summary(test_labeled_rows, split_name="test_eval_only"), ["split", "stable_id", "row_count", "positive_count", "negative_count", "positive_rate"])
    _write_csv(join_audit_csv, test_join_audit, [
        "sample_uid",
        "stable_id",
        "feature_row_present",
        "gt_included_present",
        "gt_excluded_present",
        "join_status",
        "gt_abnormal",
        "gt_label",
        "geo_exclusion_reason",
    ])

    sep_outputs = separability_v1.run_analysis(feature_lab_summary_json=feature_lab_summary_json, top_n=int(top_n))
    sep_all_csv = pack_dir / "feature_separability_train.csv"
    sep_top_csv = pack_dir / "feature_separability_top.csv"
    sep_family_csv = pack_dir / "feature_family_summary.csv"
    sep_report_json = pack_dir / "feature_separability_report.json"
    _copy_text_csv(Path(sep_outputs["feature_separability_train_csv"]).resolve(), sep_all_csv)
    _copy_text_csv(Path(sep_outputs["feature_separability_top_csv"]).resolve(), sep_top_csv)
    _copy_text_csv(Path(sep_outputs["feature_family_summary_csv"]).resolve(), sep_family_csv)
    _copy_text_csv(Path(sep_outputs["feature_separability_report_json"]).resolve(), sep_report_json)

    train_positive = int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in train_rows))
    test_labeled = [row for row in test_labeled_rows if int(_safe_int(row.get("test_gt_label_available", 0), default=0)) == 1]
    test_positive = int(sum(int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in test_labeled))
    test_excluded = [row for row in test_labeled_rows if str(row.get("test_gt_join_status", "")) == "matched_excluded"]
    missing_gt = [row for row in test_labeled_rows if str(row.get("test_gt_join_status", "")) == "missing_gt_key"]
    train_summary = {
        "row_count": int(len(train_rows)),
        "sample_count": int(len({str(row.get("sample_uid", "")) for row in train_rows})),
        "positive_count": train_positive,
        "negative_count": int(len(train_rows) - train_positive),
    }
    test_summary = {
        "row_count": int(len(test_labeled_rows)),
        "sample_count": int(len({str(row.get("sample_uid", "")) for row in test_labeled_rows})),
        "labeled_included_count": int(len(test_labeled)),
        "excluded_count": int(len(test_excluded)),
        "missing_gt_key_count": int(len(missing_gt)),
        "positive_count_eval_only": test_positive,
        "negative_count_eval_only": int(len(test_labeled) - test_positive),
    }
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "research_modeling_pack",
        "test_gt_used_for_feature_selection": False,
        "test_labels_are_eval_only": True,
        "pack_dir": str(pack_dir),
        "inputs": {
            "feature_lab_summary_json": str(feature_lab_summary_json),
            "feature_lab_train_csv": str(Path(lab_outputs["train_csv"]).resolve()),
            "feature_lab_test_blind_csv": str(Path(lab_outputs["test_blind_csv"]).resolve()),
            "test_geo_gt_csv": str(test_geo_gt_csv),
            "test_geo_excluded_csv": str(test_geo_excluded_csv),
        },
        "counts": {
            "train": train_summary,
            "test_eval_only": test_summary,
            "training_feature_count": int(len(train_feature_cols)),
            "total_column_count": int(len(fields)),
        },
        "grouping": {
            "required_group_column": "cv_group_uid",
            "recommendation": "Use grouped CV; do not randomly split needles from the same cv_group_uid.",
        },
        "outputs": {
            "train_labeled_features_csv": str(train_csv),
            "test_labeled_features_eval_only_csv": str(test_labeled_csv),
            "test_blind_features_csv": str(test_blind_csv),
            "combined_labeled_features_for_audit_csv": str(combined_csv),
            "feature_column_catalog_csv": str(catalog_csv),
            "feature_missingness_report_csv": str(missingness_csv),
            "train_group_summary_csv": str(train_group_csv),
            "train_label_balance_by_stable_id_csv": str(train_stable_csv),
            "test_label_balance_by_stable_id_eval_only_csv": str(test_stable_csv),
            "test_gt_join_audit_csv": str(join_audit_csv),
            "feature_separability_train_csv": str(sep_all_csv),
            "feature_separability_top_csv": str(sep_top_csv),
            "feature_family_summary_csv": str(sep_family_csv),
            "feature_separability_report_json": str(sep_report_json),
            "README_md": str(readme_md),
            "manifest_json": str(manifest_json),
        },
    }
    _write_readme(
        readme_md,
        pack_dir=pack_dir,
        manifest=manifest,
        train_summary=train_summary,
        test_summary=test_summary,
        feature_count=len(train_feature_cols),
    )
    _write_json(manifest_json, manifest)

    file_records = []
    for key, raw_path in manifest["outputs"].items():
        path = Path(raw_path)
        if key == "manifest_json":
            file_records.append(_manifest_file_record(path))
        else:
            file_records.append(_file_record(path))
    manifest["files"] = file_records
    _write_json(manifest_json, manifest)

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "test_geo_gt_csv": str(test_geo_gt_csv),
                "test_geo_excluded_csv": str(test_geo_excluded_csv),
                "feature_lab_summary_json": str(feature_lab_summary_json),
            },
            outputs=manifest["outputs"],
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "output_root": str(output_root),
                "pack_dir": str(pack_dir),
                "top_n": int(top_n),
                "test_gt_used_for_feature_selection": False,
                "test_labels_are_eval_only": True,
                "training_feature_count": int(len(train_feature_cols)),
            }
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete research pack for tolerance-band feature/model selection.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT_RAW, help="Root directory for timestamped research pack output.")
    parser.add_argument("--test_geo_gt_csv", default=DEFAULT_TEST_GEO_GT_RAW, help="Authoritative test geo per-slot CSV.")
    parser.add_argument("--test_geo_excluded_csv", default=DEFAULT_TEST_GEO_EXCLUDED_RAW, help="Authoritative test geo excluded slots CSV.")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N, help="Top-N feature separability rows to copy into the pack.")
    args = parser.parse_args()

    output_root = _default_pack_output_root(args.output_root)
    test_geo_gt_csv = _resolve_existing_path(args.test_geo_gt_csv, Path(DEFAULT_TEST_GEO_GT_RAW).resolve())
    test_geo_excluded_csv = _resolve_existing_path(args.test_geo_excluded_csv, Path(DEFAULT_TEST_GEO_EXCLUDED_RAW).resolve())
    manifest = run_export(
        output_root=output_root,
        test_geo_gt_csv=test_geo_gt_csv,
        test_geo_excluded_csv=test_geo_excluded_csv,
        top_n=int(args.top_n),
    )
    print(str(manifest["pack_dir"]))
    print(str(manifest["outputs"]["manifest_json"]))
    print(str(manifest["outputs"]["README_md"]))


if __name__ == "__main__":
    main()
