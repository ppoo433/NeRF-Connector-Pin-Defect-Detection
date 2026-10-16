from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _safe_name(name: str) -> str:
    # Keep filesystem paths predictable for algo names / tags.
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return s or "unnamed"


def pin_mask_to_prob(pred_pin: np.ndarray) -> np.ndarray:
    """Convert a boolean mask (True=pin) into float32 probabilities {0,1}."""
    if pred_pin.dtype != bool:
        raise TypeError(f"pred_pin must be bool, got dtype={pred_pin.dtype}")
    if pred_pin.ndim != 1:
        raise ValueError(f"pred_pin must be 1D, got shape={pred_pin.shape}")
    return pred_pin.astype(np.float32, copy=False)


def identity_mapping_meta(*, n_gt: int) -> Dict[str, Any]:
    n = int(n_gt)
    return {
        "mapping": "identity",
        "k": 0,
        "gt_n": n,
        "src_n": n,
        "coverage_max_dist": None,
        "coverage_frac": 1.0,
        "dist_min": 0.0,
        "dist_p50": 0.0,
        "dist_p95": 0.0,
        "dist_max": 0.0,
    }


def map_scores_to_gt(
    *,
    points_gt: np.ndarray,
    points_src: np.ndarray,
    scores_src: np.ndarray,
    k: int = 1,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Map per-point scores from a processed/resampled point set back to GT point order.

    - points_gt: (N,3) float
    - points_src: (M,3) float
    - scores_src: (M,) float
    - k:
        - 1 => nearest-neighbor assignment
        - >1 => inverse-distance weighted kNN average
    - max_dist:
        - if provided, we compute coverage_frac = mean(nn_dist <= max_dist)
    """
    pts_gt = np.asarray(points_gt, dtype=np.float32)
    pts_src = np.asarray(points_src, dtype=np.float32)
    sc_src = np.asarray(scores_src)
    if pts_gt.ndim != 2 or pts_gt.shape[1] != 3:
        raise ValueError(f"points_gt must be (N,3), got shape={pts_gt.shape}")
    if pts_src.ndim != 2 or pts_src.shape[1] != 3:
        raise ValueError(f"points_src must be (M,3), got shape={pts_src.shape}")
    if sc_src.ndim != 1 or sc_src.shape[0] != pts_src.shape[0]:
        raise ValueError(f"scores_src must be (M,), got shape={sc_src.shape} for M={pts_src.shape[0]}")
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be >=1, got {k}")
    if pts_src.shape[0] == 0:
        raise ValueError("points_src must be non-empty.")

    tree = cKDTree(pts_src)

    # SciPy's cKDTree supports parallel query via workers in newer versions.
    try:
        d, idx = tree.query(pts_gt, k=k, workers=-1)
    except TypeError:
        d, idx = tree.query(pts_gt, k=k)

    if k == 1:
        nn_dist = d.astype(np.float64, copy=False)
        out = sc_src[idx].astype(np.float32, copy=False)
        mapping = "kdtree_nn"
    else:
        d = np.asarray(d, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
        nn_dist = d[:, 0]
        w = 1.0 / (d + 1e-12)
        w = w / np.sum(w, axis=1, keepdims=True)
        out = np.sum(w * sc_src[idx], axis=1).astype(np.float32, copy=False)
        mapping = "kdtree_knn_idw"

    dist_min = float(np.min(nn_dist)) if nn_dist.size else 0.0
    dist_p50 = float(np.median(nn_dist)) if nn_dist.size else 0.0
    dist_p95 = float(np.quantile(nn_dist, 0.95)) if nn_dist.size else 0.0
    dist_max = float(np.max(nn_dist)) if nn_dist.size else 0.0

    cov = 1.0
    if max_dist is not None:
        md = float(max_dist)
        cov = float(np.mean(nn_dist <= md)) if nn_dist.size else 0.0
    meta: Dict[str, Any] = {
        "mapping": mapping,
        "k": k,
        "gt_n": int(pts_gt.shape[0]),
        "src_n": int(pts_src.shape[0]),
        "coverage_max_dist": float(max_dist) if max_dist is not None else None,
        "coverage_frac": float(cov),
        "dist_min": dist_min,
        "dist_p50": dist_p50,
        "dist_p95": dist_p95,
        "dist_max": dist_max,
    }
    return out, meta


def save_pred_npz(
    *,
    out_dir: Path,
    algo_name: str,
    sample_id: int,
    pin_prob: np.ndarray,
    meta: Dict[str, Any],
    compress: bool = True,
) -> Tuple[Path, Path]:
    """
    Save a prediction aligned to GT point order into debug_out.

    Files:
      - <run_dir>/pred_npz/<algo>/<sid>.npz  (pin_prob float32, shape [N])
      - <run_dir>/pred_npz/<algo>/<sid>.json (mapping/coverage meta)
    """
    out_dir = Path(out_dir)
    algo_dir = out_dir / "pred_npz" / _safe_name(algo_name)
    algo_dir.mkdir(parents=True, exist_ok=True)

    sid = int(sample_id)
    npz_path = algo_dir / f"{sid:02d}.npz"
    json_path = algo_dir / f"{sid:02d}.json"

    a = np.asarray(pin_prob, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(f"pin_prob must be 1D, got shape={a.shape}")

    if compress:
        np.savez_compressed(npz_path, pin_prob=a)
    else:
        np.savez(npz_path, pin_prob=a)

    meta_out: Dict[str, Any] = dict(meta or {})
    meta_out.update(
        {
            "sample_id": sid,
            "algo": str(algo_name),
            "pin_prob_len": int(a.shape[0]),
            "pin_prob_dtype": str(a.dtype),
        }
    )
    json_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, json_path

