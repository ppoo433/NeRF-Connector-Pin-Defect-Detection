from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from color_constancy import ColorConstancyConfig, apply_color_constancy
from ply_utils import read_ply_vertex


try:
    import torch
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch + PyG are required for dl_pn2_msg/common.py. "
        "Run inside the fenge_tp3d docker image on the server."
    ) from e


logger = logging.getLogger("dl_pn2_msg.common")


def load_split_csv(path: Path) -> Dict[int, Tuple[str, int]]:
    """
    Returns: {sample_id: (role, cv_fold)}
    """
    out: Dict[int, Tuple[str, int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["sample_id"])
            role = str(row["role"])
            cv_fold = int(row["cv_fold"])
            out[sid] = (role, cv_fold)
    return out


def _as_xyz(v: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32, copy=False)


def _as_rgb_u8(v: Dict[str, np.ndarray]) -> np.ndarray:
    if "red" not in v or "green" not in v or "blue" not in v:
        raise ValueError("PLY missing required RGB properties: red/green/blue")
    return np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8, copy=False)


def _as_normals_f32(v: Dict[str, np.ndarray]) -> np.ndarray:
    if "nx" not in v or "ny" not in v or "nz" not in v:
        raise ValueError("PLY missing required normal properties: nx/ny/nz")
    return np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float32, copy=False)


def _parse_gt_labels(v: Dict[str, np.ndarray]) -> np.ndarray:
    pin = v.get("scalar_pin")
    base = v.get("scalar_base")
    if pin is None or base is None:
        raise ValueError("GT PLY missing required properties: scalar_pin / scalar_base")
    gt_pin = np.isfinite(pin)
    gt_base = np.isfinite(base)
    if np.any(gt_pin & gt_base):
        raise ValueError("Invalid GT: scalar_pin and scalar_base overlap on some points.")
    if not np.all(gt_pin | gt_base):
        missing = int(np.sum(~(gt_pin | gt_base)))
        raise ValueError(f"Invalid GT: {missing} points are unlabeled (neither pin nor base).")
    return gt_pin.astype(np.int64, copy=False)  # 1=pin, 0=base


def xyz_normalize_block(xyz: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    xyz = np.asarray(xyz, dtype=np.float32)
    mn = np.min(xyz, axis=0)
    mx = np.max(xyz, axis=0)
    center = (mn + mx) * 0.5
    diag = float(np.linalg.norm(mx - mn))
    scale = diag if diag > 1e-12 else 1.0
    xyz_n = (xyz - center[None, :]).astype(np.float32, copy=False) / float(scale)
    return xyz_n, {"block_xyz_diag": float(diag), "block_xyz_scale": float(scale)}


def _estimate_nn_median(xyz: np.ndarray, *, max_points: int, rng: np.random.Generator) -> float:
    n = int(xyz.shape[0])
    m = int(min(max_points, n))
    if m < 10:
        return float("nan")
    idx = rng.choice(n, size=m, replace=False).astype(np.int64, copy=False)
    pts = xyz[idx].astype(np.float32, copy=False)
    tree = cKDTree(xyz.astype(np.float32, copy=False))
    try:
        d, _ = tree.query(pts, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(pts, k=2)
    nn1 = d[:, 1].astype(np.float64, copy=False)
    return float(np.median(nn1)) if nn1.size else float("nan")


def density_features(
    xyz: np.ndarray,
    *,
    mode: str,
    k: int,
    k2: int,
    nn_median: float,
) -> np.ndarray:
    """
    Per-point density proxy, scaled by per-cloud nn_median.

    Supported modes:
      - "knn_mean_std": mean/std of the first-k neighbor distances (rel to nn_median)
      - "knn_median_iqr": median/IQR of the first-k neighbor distances (rel)
      - "knn_p50_p90": p50/p90 of the first-k neighbor distances (rel)
      - "knn_mean_std_log": log(mean)/log(std) of the first-k neighbor distances (rel)
      - "knn_inv_mean_std": 1/(mean+eps), 1/(std+eps) of the first-k neighbor distances (rel)
      - "multi_knn_mean_std": [mean/std @k] + [mean/std @k2] (rel)
      - "rank": per-cloud percentile rank of [mean,std] (rel); reduces cross-cloud scale/shift

    Returns float32 array of shape [N,C] (C depends on mode).
    """
    mode = str(mode).strip().lower()
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    k2 = int(k2)
    if k2 <= 0:
        k2 = k
    if not np.isfinite(nn_median) or nn_median <= 1e-12:
        nn_median = 1.0

    k_max = int(max(k, k2))
    tree = cKDTree(xyz.astype(np.float32, copy=False))
    try:
        d, _ = tree.query(xyz.astype(np.float32, copy=False), k=k_max + 1, workers=-1)
    except TypeError:
        d, _ = tree.query(xyz.astype(np.float32, copy=False), k=k_max + 1)
    d = np.asarray(d, dtype=np.float64)
    d = d[:, 1:]  # drop self
    d_rel = d / float(nn_median)

    d1 = d_rel[:, : int(k)]
    def _rank01(a: np.ndarray) -> np.ndarray:
        # Percentile ranks in [0,1]. Stable sort for determinism.
        a = np.asarray(a, dtype=np.float64)
        n = int(a.shape[0])
        if n <= 1:
            return np.full((n,), 0.5, dtype=np.float32)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty((n,), dtype=np.int64)
        ranks[order] = np.arange(n, dtype=np.int64)
        return (ranks.astype(np.float32) / float(n - 1)).astype(np.float32, copy=False)

    if mode == "knn_mean_std":
        feats = [np.mean(d1, axis=1), np.std(d1, axis=1)]
    elif mode == "knn_median_iqr":
        q25 = np.quantile(d1, 0.25, axis=1)
        q50 = np.quantile(d1, 0.50, axis=1)
        q75 = np.quantile(d1, 0.75, axis=1)
        feats = [q50, q75 - q25]
    elif mode == "knn_p50_p90":
        q50 = np.quantile(d1, 0.50, axis=1)
        q90 = np.quantile(d1, 0.90, axis=1)
        feats = [q50, q90]
    elif mode == "knn_mean_std_log":
        m = np.mean(d1, axis=1)
        s = np.std(d1, axis=1)
        feats = [np.log(np.clip(m, 1e-8, None)), np.log(np.clip(s, 1e-8, None))]
    elif mode == "knn_inv_mean_std":
        m = np.mean(d1, axis=1)
        s = np.std(d1, axis=1)
        feats = [1.0 / (m + 1e-8), 1.0 / (s + 1e-8)]
    elif mode in {"rank", "knn_mean_std_rank", "knn_rank"}:
        m = np.mean(d1, axis=1)
        s = np.std(d1, axis=1)
        feats = [_rank01(m), _rank01(s)]
    elif mode == "multi_knn_mean_std":
        k2_eff = int(max(k, min(k2, k_max)))
        d2 = d_rel[:, :k2_eff]
        feats = [np.mean(d1, axis=1), np.std(d1, axis=1), np.mean(d2, axis=1), np.std(d2, axis=1)]
    else:
        raise ValueError(f"Unknown density mode: {mode!r}")

    out = np.stack(feats, axis=1).astype(np.float32, copy=False)
    if out.ndim != 2 or out.shape[0] != int(xyz.shape[0]):
        raise RuntimeError(f"Invalid density feature shape: {out.shape} (mode={mode!r})")
    return out


def build_cloud(
    *,
    ply_path: Path,
    use_rgb: bool,
    use_density: bool,
    use_color_constancy: bool,
    use_normals: bool,
    use_global_xyz: bool,
    color_cc_cfg: ColorConstancyConfig,
    density_k: int,
    density_k2: int,
    density_mode: str,
    density_nn_median_mode: str,
    density_max_points_for_nn_median: int,
    rng: np.random.Generator,
    need_nn_median: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      - xyz: float32 [N,3] (original coordinates)
      - x: float32 [N,C] (features for network; at least 1 channel)
      - y: int64 [N] (1=pin, 0=base)
      - dbg: dict of per-cloud stats
    """
    v = read_ply_vertex(Path(ply_path))
    xyz = _as_xyz(v)
    y = _parse_gt_labels(v)

    dbg: Dict[str, float] = {}
    feats: List[np.ndarray] = []

    if use_normals:
        n = _as_normals_f32(v)
        bad = ~np.isfinite(n)
        if np.any(bad):
            dbg["normals_nonfinite_n"] = float(int(np.sum(bad)))
            n = n.copy()
            n[bad] = 0.0
        feats.append(n.astype(np.float32, copy=False))

    if use_global_xyz:
        mn = np.min(xyz, axis=0)
        mx = np.max(xyz, axis=0)
        center = (mn + mx) * 0.5
        diag = float(np.linalg.norm(mx - mn))
        scale = diag if diag > 1e-12 else 1.0
        dbg["cloud_xyz_diag"] = float(diag)
        dbg["cloud_xyz_scale"] = float(scale)
        xyz_g = (xyz - center[None, :]).astype(np.float32, copy=False) / float(scale)
        feats.append(xyz_g.astype(np.float32, copy=False))

    if use_rgb:
        rgb_u8 = _as_rgb_u8(v)
        if use_color_constancy:
            rgb_u8, cc_dbg = apply_color_constancy(rgb_u8, cfg=color_cc_cfg)
            dbg.update({f"cc_{k}": float(v) for k, v in cc_dbg.items()})
        feats.append((rgb_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False))

    nn_med = float("nan")
    if bool(use_density) or bool(need_nn_median):
        nn_rng = rng
        if str(density_nn_median_mode).strip().lower() in {"det", "deterministic", "fixed"}:
            # Make nn_median estimation stable across training seeds/runs.
            nn_rng = np.random.default_rng(0)
        nn_med = _estimate_nn_median(xyz, max_points=int(density_max_points_for_nn_median), rng=nn_rng)
        dbg["nn_median"] = float(nn_med)

    if use_density:
        feats.append(
            density_features(
                xyz,
                mode=str(density_mode),
                k=int(density_k),
                k2=int(density_k2),
                nn_median=float(nn_med),
            )
        )

    if not feats:
        # Keep the model signature stable even when rgb/density are disabled:
        # coordinates are already provided via `pos`.
        feats.append(np.ones((xyz.shape[0], 1), dtype=np.float32))

    x = np.concatenate(feats, axis=1).astype(np.float32, copy=False)
    return xyz, x, y, dbg


def choose_seeds(n: int, *, num_seeds: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    num_seeds = int(num_seeds)
    if num_seeds <= 0:
        raise ValueError("num_seeds must be > 0")
    if n <= num_seeds:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=num_seeds, replace=False).astype(np.int64, copy=False)


def knn_block_indices(tree: cKDTree, xyz: np.ndarray, *, seed_idx: int, k: int) -> np.ndarray:
    n = int(xyz.shape[0])
    k_eff = int(min(k, n))
    seed_idx = int(seed_idx)
    if seed_idx < 0 or seed_idx >= n:
        raise IndexError(f"seed_idx out of range: {seed_idx} for n={n}")
    q = xyz[seed_idx].astype(np.float32, copy=False)
    try:
        _, idx = tree.query(q, k=k_eff, workers=-1)
    except TypeError:
        _, idx = tree.query(q, k=k_eff)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 0:
        idx = idx.reshape(1)
    return idx


def build_block_data(
    *,
    xyz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
) -> Tuple[Data, Dict[str, float]]:
    idx = np.asarray(idx, dtype=np.int64)
    xyz_b = xyz[idx]
    x_b = x[idx]
    y_b = y[idx]

    pos_n, dbg = xyz_normalize_block(xyz_b)
    data = Data(
        pos=torch.from_numpy(pos_n),
        x=torch.from_numpy(x_b.astype(np.float32, copy=False)),
        y=torch.from_numpy(y_b.astype(np.int64, copy=False)),
    )
    return data, dbg


def safe_cc_cfg_dict(cfg: ColorConstancyConfig) -> Dict[str, float]:
    # Ensure we only log numeric config (avoid accidental sensitive fields).
    d = asdict(cfg)
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out
