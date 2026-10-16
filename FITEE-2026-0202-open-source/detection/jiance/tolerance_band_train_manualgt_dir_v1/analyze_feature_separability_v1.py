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
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .build_feature_lab_v1 import REPO_ROOT
    from .build_geo_shadow_feature_bank_v3 import _latest_artifact, _resolve_existing_path, _safe_float, _safe_int, _write_csv, _write_json
except ImportError:
    from tolerance_band_train_manualgt_dir_v1.build_feature_lab_v1 import REPO_ROOT
    from tolerance_band_train_manualgt_dir_v1.build_geo_shadow_feature_bank_v3 import (
        _latest_artifact,
        _resolve_existing_path,
        _safe_float,
        _safe_int,
        _write_csv,
        _write_json,
    )


DEFAULT_FEATURE_LAB_SUMMARY_RAW = ""
TARGET_RECALL = 0.85
TARGET_PRECISION = 0.85
TOP_N = 80


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {str(k or "").replace("\ufeff", "").strip(): str(v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _default_output_dir() -> Path:
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None):
        return Path(_DAIMA_CTX.run_dir).resolve()
    return (REPO_ROOT / "debug_out" / "feature_lab_separability_v1").resolve()


def default_feature_lab_summary_json() -> Path:
    path = Path(DEFAULT_FEATURE_LAB_SUMMARY_RAW)
    if path.is_file():
        return path.resolve()
    return _latest_artifact("feature_lab_summary.json")


def _float_array(rows: Sequence[Mapping[str, Any]], column: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(column, float("nan"))) for row in rows], dtype=np.float64)


def _label_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(_safe_int(row.get("gt_abnormal", 0), default=0)) for row in rows], dtype=np.int32)


def _finite_values(rows: Sequence[Mapping[str, Any]], column: str) -> np.ndarray:
    values = _float_array(rows, column)
    return values[np.isfinite(values)]


def _ks_stat(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.sort(pos[np.isfinite(pos)])
    neg = np.sort(neg[np.isfinite(neg)])
    if pos.size <= 0 or neg.size <= 0:
        return float("nan")
    values = np.unique(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, values, side="right") / float(pos.size)
    cdf_neg = np.searchsorted(neg, values, side="right") / float(neg.size)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _cohen_d(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size <= 1 or neg.size <= 1:
        return float("nan")
    pooled = np.sqrt(((pos.size - 1) * np.var(pos) + (neg.size - 1) * np.var(neg)) / float(pos.size + neg.size - 2))
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0
    return float((np.mean(pos) - np.mean(neg)) / pooled)


def _safe_metric_pair(y: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(scores)
    y2 = y[mask]
    s2 = scores[mask]
    if y2.size <= 0 or np.unique(y2).size < 2 or np.unique(s2).size < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y2, s2)), float(roc_auc_score(y2, s2))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": float(0.5 * (recall + specificity)),
    }


def _frontier_summary(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(scores)
    y2 = y[mask]
    s2 = scores[mask]
    if y2.size <= 0 or np.unique(y2).size < 2 or np.unique(s2).size < 2:
        return {}
    thresholds = sorted({float(v) for v in s2.tolist()}, reverse=True)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        pred = (s2 >= threshold).astype(np.int32)
        row = {"threshold": float(threshold)}
        row.update(_binary_metrics(y2, pred))
        rows.append(row)
    best_f1 = max(rows, key=lambda r: (float(r["f1"]), float(r["precision"]), float(r["recall"])))
    recall_rows = [row for row in rows if float(row["recall"]) >= TARGET_RECALL]
    precision_rows = [row for row in rows if float(row["precision"]) >= TARGET_PRECISION]
    best_precision_at_recall = (
        max(recall_rows, key=lambda r: (float(r["precision"]), float(r["f1"]), float(r["recall"])))
        if recall_rows
        else None
    )
    best_recall_at_precision = (
        max(precision_rows, key=lambda r: (float(r["recall"]), float(r["f1"]), float(r["precision"])))
        if precision_rows
        else None
    )
    return {
        "single_feature_best_f1": float(best_f1["f1"]),
        "single_feature_best_f1_threshold": float(best_f1["threshold"]),
        "single_feature_best_f1_precision": float(best_f1["precision"]),
        "single_feature_best_f1_recall": float(best_f1["recall"]),
        "best_precision_at_recall_ge_085": float(best_precision_at_recall["precision"]) if best_precision_at_recall else float("nan"),
        "best_precision_at_recall_ge_085_threshold": float(best_precision_at_recall["threshold"]) if best_precision_at_recall else float("nan"),
        "best_recall_at_precision_ge_085": float(best_recall_at_precision["recall"]) if best_recall_at_precision else float("nan"),
        "best_recall_at_precision_ge_085_threshold": float(best_recall_at_precision["threshold"]) if best_recall_at_precision else float("nan"),
    }


def _feature_family(column: str, feature_columns: Mapping[str, Sequence[str]]) -> str:
    for family in ("base_numeric", "event", "shape", "enhanced_shape", "relative", "group", "slot_context", "common_mode", "neighbor", "sequence"):
        if column in set(feature_columns.get(family, []) or []):
            return family
    if column.startswith("seq_raw_"):
        return "sequence"
    return "other"


def _group_direction_consistency(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    *,
    direction: int,
    group_column: str,
) -> Tuple[int, float]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_column, "") or row.get("sample_uid", "")), []).append(row)
    total = 0
    agree = 0
    for group_rows in grouped.values():
        y = _label_array(group_rows)
        values = _float_array(group_rows, column)
        mask = np.isfinite(values)
        if np.unique(y[mask]).size < 2:
            continue
        pos = values[mask & (y == 1)]
        neg = values[mask & (y == 0)]
        if pos.size <= 0 or neg.size <= 0:
            continue
        gap = float(np.nanmedian(pos) - np.nanmedian(neg))
        total += 1
        if direction * gap >= 0.0:
            agree += 1
    return int(total), float(agree / total) if total > 0 else float("nan")


def _drift_stats(train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], column: str) -> Dict[str, Any]:
    train = _finite_values(train_rows, column)
    test = _finite_values(test_rows, column)
    if train.size <= 0 or test.size <= 0:
        return {
            "train_test_ks": float("nan"),
            "train_test_median_shift": float("nan"),
            "train_test_std_shift": float("nan"),
            "test_missing_frac": 1.0 if len(test_rows) else float("nan"),
        }
    train_std = float(np.std(train))
    median_shift = float(np.median(test) - np.median(train))
    return {
        "train_test_ks": _ks_stat(train, test),
        "train_test_median_shift": median_shift,
        "train_test_std_shift": float(median_shift / train_std) if train_std > 1e-12 else 0.0,
        "test_missing_frac": float(1.0 - (test.size / max(len(test_rows), 1))),
        "test_median": float(np.median(test)),
        "test_q90": float(np.quantile(test, 0.90)),
    }


def _analyze_feature(
    *,
    column: str,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature_columns: Mapping[str, Sequence[str]],
    group_column: str,
) -> Optional[Dict[str, Any]]:
    y = _label_array(train_rows)
    values = _float_array(train_rows, column)
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 5 or np.unique(y[mask]).size < 2 or np.unique(values[mask]).size < 2:
        return None
    pos = values[mask & (y == 1)]
    neg = values[mask & (y == 0)]
    ap_pos, auc_pos = _safe_metric_pair(y, values)
    ap_neg, auc_neg = _safe_metric_pair(y, -values)
    if not np.isfinite(ap_pos) and not np.isfinite(ap_neg):
        return None
    direction = 1 if (np.nan_to_num(ap_pos, nan=-1.0) >= np.nan_to_num(ap_neg, nan=-1.0)) else -1
    best_scores = values if direction > 0 else -values
    group_count, group_consistency = _group_direction_consistency(
        train_rows,
        column,
        direction=direction,
        group_column=group_column,
    )
    row: Dict[str, Any] = {
        "feature": column,
        "family": _feature_family(column, feature_columns),
        "direction": "higher_is_abnormal" if direction > 0 else "lower_is_abnormal",
        "train_finite_count": int(np.count_nonzero(mask)),
        "train_missing_frac": float(1.0 - (np.count_nonzero(mask) / max(len(train_rows), 1))),
        "pos_median": float(np.nanmedian(pos)),
        "neg_median": float(np.nanmedian(neg)),
        "median_gap_pos_minus_neg": float(np.nanmedian(pos) - np.nanmedian(neg)),
        "pos_q25": float(np.nanquantile(pos, 0.25)),
        "pos_q75": float(np.nanquantile(pos, 0.75)),
        "neg_q25": float(np.nanquantile(neg, 0.25)),
        "neg_q75": float(np.nanquantile(neg, 0.75)),
        "ap_higher": float(ap_pos),
        "ap_lower": float(ap_neg),
        "ap_best": float(max(np.nan_to_num(ap_pos, nan=-1.0), np.nan_to_num(ap_neg, nan=-1.0))),
        "roc_auc_higher": float(auc_pos),
        "roc_auc_lower": float(auc_neg),
        "roc_auc_best": float(max(np.nan_to_num(auc_pos, nan=-1.0), np.nan_to_num(auc_neg, nan=-1.0))),
        "ks_pos_neg": _ks_stat(pos, neg),
        "cohen_d_pos_minus_neg": _cohen_d(pos, neg),
        "group_valid_count": int(group_count),
        "group_direction_consistency": float(group_consistency),
    }
    row.update(_frontier_summary(y, best_scores))
    row.update(_drift_stats(train_rows, test_rows, column))
    return row


def _family_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("family", "other")), []).append(row)
    out: List[Dict[str, Any]] = []
    for family, family_rows in sorted(grouped.items()):
        top_ap = sorted([float(_safe_float(row.get("ap_best", float("nan")))) for row in family_rows if np.isfinite(_safe_float(row.get("ap_best", float("nan"))))], reverse=True)
        top_ks = sorted([float(_safe_float(row.get("ks_pos_neg", float("nan")))) for row in family_rows if np.isfinite(_safe_float(row.get("ks_pos_neg", float("nan"))))], reverse=True)
        out.append(
            {
                "family": family,
                "feature_count": int(len(family_rows)),
                "best_ap": float(top_ap[0]) if top_ap else float("nan"),
                "mean_top5_ap": float(np.mean(top_ap[:5])) if top_ap else float("nan"),
                "best_ks": float(top_ks[0]) if top_ks else float("nan"),
                "mean_top5_ks": float(np.mean(top_ks[:5])) if top_ks else float("nan"),
            }
        )
    return sorted(out, key=lambda r: (float(_safe_float(r.get("best_ap", 0.0))), float(_safe_float(r.get("best_ks", 0.0)))), reverse=True)


def run_analysis(*, feature_lab_summary_json: Path, top_n: int) -> Dict[str, Path]:
    feature_lab_summary = _load_json(feature_lab_summary_json)
    outputs = dict(feature_lab_summary.get("outputs", {}) or {})
    train_csv = Path(str(outputs.get("train_csv", ""))).resolve()
    test_csv = Path(str(outputs.get("test_blind_csv", ""))).resolve()
    train_rows = _read_csv(train_csv)
    test_rows = _read_csv(test_csv)
    feature_columns = dict(feature_lab_summary.get("feature_columns", {}) or {})
    analysis_columns = list(feature_columns.get("hybrid_v3", []) or [])
    group_column = "cv_group_uid" if all(str(row.get("cv_group_uid", "")).strip() for row in train_rows) else "sample_uid"

    rows: List[Dict[str, Any]] = []
    for column in analysis_columns:
        analyzed = _analyze_feature(
            column=str(column),
            train_rows=train_rows,
            test_rows=test_rows,
            feature_columns=feature_columns,
            group_column=group_column,
        )
        if analyzed is not None:
            rows.append(analyzed)
    rows = sorted(
        rows,
        key=lambda r: (
            float(_safe_float(r.get("ap_best", 0.0))),
            float(_safe_float(r.get("ks_pos_neg", 0.0))),
            float(_safe_float(r.get("single_feature_best_f1", 0.0))),
        ),
        reverse=True,
    )
    top_rows = rows[: max(int(top_n), 1)]
    family_rows = _family_summary(rows)

    out_dir = _default_output_dir()
    all_csv = (out_dir / "feature_separability_train.csv").resolve()
    top_csv = (out_dir / "feature_separability_top.csv").resolve()
    family_csv = (out_dir / "feature_family_summary.csv").resolve()
    report_json = (out_dir / "feature_separability_report.json").resolve()
    fields = list(rows[0].keys()) if rows else ["feature", "family", "ap_best"]
    family_fields = list(family_rows[0].keys()) if family_rows else ["family", "feature_count"]
    report = {
        "schema_version": 1,
        "mode": "train_supervised_feature_separability_and_test_blind_drift",
        "test_gt_used_for_feature_selection": False,
        "feature_lab_summary_json": str(feature_lab_summary_json),
        "train_csv": str(train_csv),
        "test_blind_csv": str(test_csv),
        "group_column": group_column,
        "train_row_count": int(len(train_rows)),
        "test_blind_row_count": int(len(test_rows)),
        "feature_count_requested": int(len(analysis_columns)),
        "feature_count_analyzed": int(len(rows)),
        "target_recall_for_single_feature_hint": float(TARGET_RECALL),
        "target_precision_for_single_feature_hint": float(TARGET_PRECISION),
        "top_features_preview": top_rows[:20],
        "family_summary": family_rows,
        "outputs": {
            "feature_separability_train_csv": str(all_csv),
            "feature_separability_top_csv": str(top_csv),
            "feature_family_summary_csv": str(family_csv),
            "feature_separability_report_json": str(report_json),
        },
    }

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "feature_lab_summary_json": str(feature_lab_summary_json),
                "train_csv": str(train_csv),
                "test_blind_csv": str(test_csv),
            },
            outputs=report["outputs"],
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "top_n": int(top_n),
                "group_column": group_column,
                "test_gt_used_for_feature_selection": False,
                "feature_count_requested": int(len(analysis_columns)),
            }
        )

    _write_csv(all_csv, rows, fields)
    _write_csv(top_csv, top_rows, fields)
    _write_csv(family_csv, family_rows, family_fields)
    _write_json(report_json, report)
    return {
        "feature_separability_train_csv": all_csv,
        "feature_separability_top_csv": top_csv,
        "feature_family_summary_csv": family_csv,
        "feature_separability_report_json": report_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze single-feature separability on train labels and blind train/test drift.")
    parser.add_argument("--feature_lab_summary_json", default=DEFAULT_FEATURE_LAB_SUMMARY_RAW)
    parser.add_argument("--top_n", type=int, default=TOP_N)
    args = parser.parse_args()

    outputs = run_analysis(
        feature_lab_summary_json=_resolve_existing_path(args.feature_lab_summary_json, default_feature_lab_summary_json()),
        top_n=int(args.top_n),
    )
    for key in (
        "feature_separability_train_csv",
        "feature_separability_top_csv",
        "feature_family_summary_csv",
        "feature_separability_report_json",
    ):
        print(str(outputs[key]))


if __name__ == "__main__":
    main()
