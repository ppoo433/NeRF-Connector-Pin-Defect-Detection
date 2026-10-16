from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, precision_recall_curve

from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

SEED = 42

def strip_raw_prefix(col: str) -> str:
    return col[5:] if col.startswith("raw__") else col

def source_cols_from_config(config: Mapping[str, Any]) -> List[str]:
    raw = config.get("source_feature_cols", config.get("feature_cols", []))
    return [strip_raw_prefix(str(c)) for c in raw]

def filter_test_geo_included(test: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    if "test_gt_geo_included" in out.columns:
        inc = pd.to_numeric(out["test_gt_geo_included"], errors="coerce").fillna(0).astype(int)
        out = out.loc[inc == 1].copy()
    out = out.loc[out["gt_abnormal"].notna()].copy()
    return out.reset_index(drop=True)

def numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    source_cols = [strip_raw_prefix(c) for c in cols]
    return df.loc[:, source_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

def metrics_from_pred(y_true: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "ap": float(average_precision_score(y_true, scores)),
        "ba": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def make_model(model_name: str, seed: int=SEED, params: Optional[Mapping[str, Any]]=None) -> Pipeline:
    params = dict(params or {})
    if model_name == 'lgbm':
        if not HAS_LIGHTGBM or LGBMClassifier is None:
            raise RuntimeError('LightGBM is not available in this Python environment')
        clf = LGBMClassifier(n_estimators=int(params.get('n_estimators', 160)), learning_rate=float(params.get('learning_rate', 0.035)), num_leaves=int(params.get('num_leaves', 7)), max_depth=int(params.get('max_depth', -1)), min_child_samples=int(params.get('min_child_samples', 8)), subsample=float(params.get('subsample', 0.9)), colsample_bytree=float(params.get('colsample_bytree', 0.9)), reg_alpha=float(params.get('reg_alpha', 0.0)), reg_lambda=float(params.get('reg_lambda', 0.5)), class_weight=params.get('class_weight', 'balanced'), random_state=seed, n_jobs=-1, verbosity=-1)
        return Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', clf)])
    raise ValueError(f'Unsupported model: {model_name}')

def fit_full_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    cols: Sequence[str],
    model_name: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    seed_offset: int = 0,
) -> np.ndarray:
    missing_train = [c for c in cols if c not in train.columns]
    missing_test = [c for c in cols if c not in test.columns]
    if missing_train or missing_test:
        raise RuntimeError(f"Configured columns missing: train={missing_train[:10]} test={missing_test[:10]}")
    model = make_model(model_name, seed=SEED + seed_offset, params=params)
    model.fit(numeric_frame(train, cols), y_train)
    return model.predict_proba(numeric_frame(test, cols))[:, 1]
