# -*- coding: utf-8 -*-
"""
Build golden centerline tolerance band (V1) from a point-cloud folder.

Input:
  - A folder that contains many samples (base + pins point clouds). The script auto-discovers:
    A) Flat files: <id>_base.ply + <id>_pins.ply
    B) Subfolders: <sample_dir>/base.ply + <sample_dir>/pins.ply (or needle.ply)

Output (user-requested):
  - <INPUT_DIR>/output/<run_id>/...
  - <INPUT_DIR>/output/last_output.json
  - <INPUT_DIR>/output/<run_id>/plots/boxplot_*.png  (optional; requires matplotlib)
  - <INPUT_DIR>/output/<run_id>/analysis_summary.json

Run logs / snapshots:
  - Still written to debug_out/ via daima_runtime.runlog (AGENTS.md requirement).

V1 tolerance model:
  - Lateral tolerance tube: tol_lateral_mm[i]
  - Axial shortness (one-sided): tol_tip_short_mm, tol_len_short_mm
  - Rule A: fit_status=high_risk needles are excluded from tolerance statistics (need_rescan), but still reported.
"""

from __future__ import annotations

_DAIMA_CTX = None
if __name__ == "__main__":
    # Enable unified console logging when running this file directly (PyCharm Run).
    try:
        import sys
        from pathlib import Path

        # Avoid permission issues when Python tries to write `__pycache__` under `jiance/`.
        sys.dont_write_bytecode = True

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

        # `pin_cloud2center_line.py` lives under `jiance/` (not a package).
        _JIANCE_DIR = (_REPO_ROOT / "jiance").resolve()
        if str(_JIANCE_DIR) not in sys.path:
            sys.path.insert(0, str(_JIANCE_DIR))

        from daima_runtime.runlog import setup_run

        _DAIMA_CTX = setup_run(__file__)
    except Exception:
        _DAIMA_CTX = None

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    from pin_cloud2center_line import UNIT_SCALE as _UNIT_SCALE_DEFAULT  # type: ignore
except Exception:
    _UNIT_SCALE_DEFAULT = 1.0


# -----------------------------
# User-editable defaults
# -----------------------------

# Input folder that contains many base/pins point clouds.
# If empty, auto-pick latest under <repo_root>/point_cloud/fenge/split_out/<latest_run_id>/.
INPUT_DIR = ""

# Optional: choose a specific reference sample_id. If empty, use first (natural-sorted) sample_id.
REFERENCE_SAMPLE_ID = ""

# Output folder name created under INPUT_DIR.
OUTPUT_DIRNAME = "output"

# Tolerance parameters (position-band V1).
RESAMPLE_N = 240

# Legacy alias retained for backward-compatible metadata / plotting labels.
# Active mainline-compatible default is 1.0; wider bands should be explored via
# the explicit lateral/axial knobs below instead of silently changing this alias.
INFLATE_FACTOR = 1.0

# Position-band sweep knobs.
# They can also be overridden by env:
#   DAIMA_TOL_LATERAL_QUANTILE
#   DAIMA_TOL_LATERAL_INFLATE_FACTOR
#   DAIMA_TOL_LATERAL_FLOOR_MM
#   DAIMA_TOL_LATERAL_SMOOTH_WINDOW
#   DAIMA_TOL_AXIAL_QUANTILE
#   DAIMA_TOL_AXIAL_INFLATE_FACTOR
LATERAL_QUANTILE = 0.95
LATERAL_INFLATE_FACTOR = 1.0
LATERAL_FLOOR_MM = 0.0
LATERAL_SMOOTH_WINDOW = 0

AXIAL_QUANTILE = 0.95
AXIAL_INFLATE_FACTOR = 1.0

MIN_OK_SAMPLES_PER_NEEDLE = 6

# Matching parameters.
MATCH_RADIUS_MM: object = "auto"  # "auto" or float (in mm)

# Exclude these fit_status from tolerance statistics (Rule A).
EXCLUDE_FIT_STATUS = {"high_risk"}

# Optional speed knob (0 => no limit). Can also be set via env DAIMA_TOL_MAX_SAMPLES.
MAX_SAMPLES = 0

# Registration (alignment) method.
# Default is the robust v1 pipeline validated by `jiance/eval_registration_pipeline.py`:
#   overall coarse (FPFH+RANSAC+ICP) -> base ICP refine.
REG_METHOD = "overall_then_base_v1"  # or: "base_legacy"

# Optional: override overall point-cloud folder (where <id>.ply lives).
# If empty, auto-infer from INPUT_DIR (expects .../point_cloud/fenge/split_out/<run_id>/).
OVERALL_DIR = ""

# Plotting
# - New default: visualize the tolerance band directly (p95 * inflate_factor).
# - Legacy: per-sample distributions via boxplots can be enabled for debugging.
PLOT_ENABLE_TOLERANCE_BAND_PLOTS = True
PLOT_ENABLE_BOXPLOTS = False
PLOT_SHOW_PRE_INFLATE_P95 = True
PLOT_HEATMAP_CMAP = "viridis"

# Registration knobs (can also be overridden by env; see _reg_config()).
REG_TARGET_GRID_POINTS = 50000
REG_MIN_VOXEL_MM = 0.2
REG_MAX_VOXEL_MM = 5.0
REG_RANSAC_MAX_ITER = 400000
REG_RANSAC_CONFIDENCE = 0.999

# Keep extraction behavior consistent with validated pin_cloud2center_line defaults.
CENTERLINE_CFG_DENOISE_ENABLE = False
CENTERLINE_CFG_ENDPOINT_CENTERING_ITERS = 2
CENTERLINE_CFG_FORCE_FINAL_STAGE13 = True


@dataclass(frozen=True)
class SamplePair:
    sample_id: str
    base_ply: Path
    pins_ply: Path


def _repo_root() -> Path:
    if _DAIMA_CTX is not None:
        return Path(getattr(_DAIMA_CTX, "repo_root", Path(__file__).resolve().parents[1]))
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _unit_scale() -> float:
    try:
        return float(_UNIT_SCALE_DEFAULT)
    except Exception:
        return 1.0


def _centerline_extract_config():
    from pin_cloud2center_line import CenterlineConfig  # type: ignore

    base_kwargs = {
        "denoise_enable": bool(CENTERLINE_CFG_DENOISE_ENABLE),
        "endpoint_centering_iterations": int(CENTERLINE_CFG_ENDPOINT_CENTERING_ITERS),
    }
    try:
        return CenterlineConfig(
            force_final_curve_stage13=bool(CENTERLINE_CFG_FORCE_FINAL_STAGE13),
            **base_kwargs,
        )
    except TypeError:
        return CenterlineConfig(**base_kwargs)


def _mm_from_u(dist_u: float, unit_scale: float) -> float:
    return float(dist_u) / float(unit_scale + 1e-12)


def _u_from_mm(dist_mm: float, unit_scale: float) -> float:
    return float(dist_mm) * float(unit_scale)


def _natural_key(s: str) -> List[object]:
    out: List[object] = []
    for part in re.split(r"(\d+)", str(s)):
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return out


def _get_env_float(env_name: str, default: float) -> float:
    v = str(os.getenv(env_name, "")).strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _get_env_int(env_name: str, default: int) -> int:
    v = str(os.getenv(env_name, "")).strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _resolve_tolerance_cfg() -> Dict[str, Any]:
    legacy_inflate = _get_env_float("DAIMA_TOL_INFLATE_FACTOR", float(INFLATE_FACTOR))
    lateral_q = float(np.clip(_get_env_float("DAIMA_TOL_LATERAL_QUANTILE", float(LATERAL_QUANTILE)), 0.0, 1.0))
    axial_q = float(np.clip(_get_env_float("DAIMA_TOL_AXIAL_QUANTILE", float(AXIAL_QUANTILE)), 0.0, 1.0))
    lateral_inflate = _get_env_float("DAIMA_TOL_LATERAL_INFLATE_FACTOR", float(LATERAL_INFLATE_FACTOR))
    axial_inflate = _get_env_float("DAIMA_TOL_AXIAL_INFLATE_FACTOR", float(AXIAL_INFLATE_FACTOR))
    if "DAIMA_TOL_LATERAL_INFLATE_FACTOR" not in os.environ:
        lateral_inflate = float(legacy_inflate if "DAIMA_TOL_INFLATE_FACTOR" in os.environ else lateral_inflate)
    if "DAIMA_TOL_AXIAL_INFLATE_FACTOR" not in os.environ:
        axial_inflate = float(legacy_inflate if "DAIMA_TOL_INFLATE_FACTOR" in os.environ else axial_inflate)

    lateral_floor_mm = max(0.0, _get_env_float("DAIMA_TOL_LATERAL_FLOOR_MM", float(LATERAL_FLOOR_MM)))
    smooth_window = max(0, _get_env_int("DAIMA_TOL_LATERAL_SMOOTH_WINDOW", int(LATERAL_SMOOTH_WINDOW)))
    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1

    return {
        "lateral_quantile": float(lateral_q),
        "lateral_inflate_factor": float(max(lateral_inflate, 0.0)),
        "lateral_floor_mm": float(lateral_floor_mm),
        "lateral_smooth_window": int(smooth_window),
        "axial_quantile": float(axial_q),
        "axial_inflate_factor": float(max(axial_inflate, 0.0)),
        # Keep the legacy field aligned with the lateral band, because the detect
        # side historically treats `inflate_factor` as the position-band scalar.
        "inflate_factor": float(max(lateral_inflate, 0.0)),
    }


def _pick_default_input_dir(repo_root: Path) -> Optional[Path]:
    base = (repo_root / "point_cloud" / "fenge" / "split_out").resolve()
    if not base.is_dir():
        return None
    subdirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        return None
    return subdirs[-1]


def _resolve_input_dir() -> Path:
    env = str(os.getenv("DAIMA_TOL_INPUT_DIR", "")).strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = _repo_root() / p
        return p.resolve()

    raw = str(INPUT_DIR).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = _repo_root() / p
        return p.resolve()

    picked = _pick_default_input_dir(_repo_root())
    if picked is not None:
        return picked.resolve()

    # Fallback: repo root (keeps PyCharm runnable even if discovery fails).
    return _repo_root().resolve()


def _add_unique_path(map_sid_to_path: Dict[str, Path], sid: str, p: Path, kind: str, errors: List[str]) -> None:
    if sid in map_sid_to_path:
        if map_sid_to_path[sid].resolve() != p.resolve():
            errors.append(f"duplicate {kind} for sample_id={sid}: {map_sid_to_path[sid]} vs {p}")
        return
    map_sid_to_path[sid] = p


def _discover_samples(input_dir: Path) -> Tuple[List[SamplePair], List[str]]:
    """
    Returns:
      samples: list of SamplePair
      errors: list of discovery issues (duplicates/missing pairs)
    """
    errors: List[str] = []
    bases: Dict[str, Path] = {}
    pins: Dict[str, Path] = {}

    # Layout A: <id>_base.ply / <id>_pins.ply
    for p in input_dir.rglob("*_base.ply"):
        sid = p.stem[: -len("_base")]
        if sid:
            _add_unique_path(bases, sid, p, "base", errors)
    for p in input_dir.rglob("*_pins.ply"):
        sid = p.stem[: -len("_pins")]
        if sid:
            _add_unique_path(pins, sid, p, "pins", errors)

    # Layout B: <sample_dir>/base.ply / pins.ply (or needle.ply)
    for p in input_dir.rglob("base.ply"):
        sid = p.parent.name
        if sid:
            _add_unique_path(bases, sid, p, "base", errors)
    for name in ["pins.ply", "needle.ply"]:
        for p in input_dir.rglob(name):
            sid = p.parent.name
            if sid:
                _add_unique_path(pins, sid, p, "pins", errors)

    # Fallback: *_needle*.ply (id-prefix naming)
    for p in input_dir.rglob("*_needle*.ply"):
        m = re.match(r"(.+)_needle", p.stem)
        if m:
            sid = str(m.group(1))
            if sid and sid not in pins:
                _add_unique_path(pins, sid, p, "pins", errors)

    all_ids = sorted(set(bases.keys()) | set(pins.keys()), key=_natural_key)
    pairs: List[SamplePair] = []
    for sid in all_ids:
        b = bases.get(sid)
        n = pins.get(sid)
        if b is None:
            errors.append(f"missing base for sample_id={sid}: pins={n}")
            continue
        if n is None:
            errors.append(f"missing pins for sample_id={sid}: base={b}")
            continue
        pairs.append(SamplePair(sample_id=sid, base_ply=b, pins_ply=n))

    # Optional limit for smoke tests.
    max_s = MAX_SAMPLES
    env_max = str(os.getenv("DAIMA_TOL_MAX_SAMPLES", "")).strip()
    if env_max:
        try:
            max_s = int(env_max)
        except Exception:
            max_s = max_s
    if int(max_s) > 0:
        pairs = pairs[: int(max_s)]

    return pairs, errors


def _remove_outliers(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 50, std_ratio: float = 1.5) -> o3d.geometry.PointCloud:
    # Same defaults as jiance/1.11.py.
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return pcd.select_by_index(ind)


def _estimate_voxel_size(pcd: o3d.geometry.PointCloud, target_downsample_points: int = 50000) -> float:
    # Same idea as jiance/1.11.py:estimate_voxel_size().
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
    n = len(pcd.points)
    if n <= 0:
        return 0.1
    voxel_size = float(diag) * float(n / max(int(target_downsample_points), 1)) ** (1.0 / 3.0)
    return float(voxel_size)


def _compute_board_transform_1p11(base_ref_path: Path, base_sample_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Base registration (sample -> ref), following jiance/1.11.py:compute_board_transform().
    Returns:
      T: 4x4 transformation matrix that maps sample into ref
      meta: dict with fitness/rmse/voxel_size/etc (best-effort)
    """
    t0 = time.time()
    meta: Dict[str, Any] = {"status": "ok"}

    base_ref = o3d.io.read_point_cloud(str(base_ref_path))
    base_sample = o3d.io.read_point_cloud(str(base_sample_path))
    base_ref = _remove_outliers(base_ref)
    base_sample = _remove_outliers(base_sample)

    voxel_size = _estimate_voxel_size(base_ref)
    meta["voxel_size_u"] = float(voxel_size)

    def preprocess(pcd: o3d.geometry.PointCloud, size: float):
        pcd_down = pcd.voxel_down_sample(size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(size) * 2.0, max_nn=30)
        )
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(size) * 5.0, max_nn=100),
        )
        return pcd_down, pcd_fpfh

    s_down, s_fpfh = preprocess(base_sample, voxel_size)
    t_down, t_fpfh = preprocess(base_ref, voxel_size)

    res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down,
        t_down,
        s_fpfh,
        t_fpfh,
        True,
        float(voxel_size) * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(float(voxel_size) * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )
    meta["ransac_fitness"] = float(getattr(res_ransac, "fitness", float("nan")))
    meta["ransac_inlier_rmse_u"] = float(getattr(res_ransac, "inlier_rmse", float("nan")))

    res_icp = o3d.pipelines.registration.registration_icp(
        base_sample,
        base_ref,
        float(voxel_size) * 1.5,
        res_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    meta["icp_fitness"] = float(getattr(res_icp, "fitness", float("nan")))
    meta["icp_inlier_rmse_u"] = float(getattr(res_icp, "inlier_rmse", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        meta["status"] = "failed"
        raise RuntimeError(f"Invalid ICP transformation for base={base_sample_path}")
    return T, meta


def _resolve_overall_dir(input_dir: Path) -> Optional[Path]:
    """
    Resolve the folder that contains overall clouds: <overall_dir>/<sample_id>.ply

    Priority:
      1) env DAIMA_TOL_OVERALL_DIR
      2) file constant OVERALL_DIR
      3) infer from input_dir: find a parent named "split_out", then use its parent (fenge)
    """
    env = str(os.getenv("DAIMA_TOL_OVERALL_DIR", "")).strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = _repo_root() / p
        return p.resolve()

    raw = str(OVERALL_DIR).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = _repo_root() / p
        return p.resolve()

    for p in [input_dir] + list(input_dir.parents):
        if p.name == "split_out":
            return p.parent.resolve()
    return None


def _reg_config(input_dir: Path, unit_scale: float) -> Dict[str, Any]:
    method = str(os.getenv("DAIMA_TOL_REG_METHOD", "")).strip() or str(REG_METHOD).strip()
    overall_dir = _resolve_overall_dir(input_dir)

    def _get_int(env_name: str, default: int) -> int:
        v = str(os.getenv(env_name, "")).strip()
        if not v:
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _get_float(env_name: str, default: float) -> float:
        v = str(os.getenv(env_name, "")).strip()
        if not v:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    target_grid_points = _get_int("DAIMA_TOL_REG_TARGET_GRID_POINTS", int(REG_TARGET_GRID_POINTS))
    min_voxel_mm = _get_float("DAIMA_TOL_REG_MIN_VOXEL_MM", float(REG_MIN_VOXEL_MM))
    max_voxel_mm = _get_float("DAIMA_TOL_REG_MAX_VOXEL_MM", float(REG_MAX_VOXEL_MM))
    ransac_max_iter = _get_int("DAIMA_TOL_REG_RANSAC_MAX_ITER", int(REG_RANSAC_MAX_ITER))
    ransac_conf = _get_float("DAIMA_TOL_REG_RANSAC_CONFIDENCE", float(REG_RANSAC_CONFIDENCE))

    return {
        "method": str(method),
        "overall_dir": overall_dir,
        "target_grid_points": int(target_grid_points),
        "min_voxel_u": _u_from_mm(float(min_voxel_mm), float(unit_scale)),
        "max_voxel_u": _u_from_mm(float(max_voxel_mm), float(unit_scale)),
        "ransac_max_iter": int(ransac_max_iter),
        "ransac_confidence": float(ransac_conf),
    }


def _estimate_voxel_u_fixed(
    pcd: o3d.geometry.PointCloud,
    *,
    target_grid_points: int,
    min_voxel_u: float,
    max_voxel_u: float,
) -> float:
    """
    Robust voxel heuristic (fixed version):
      voxel ~= diag / cbrt(target_grid_points)
    then clamp to [min_voxel_u, max_voxel_u].
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound())))
    if not np.isfinite(diag) or diag <= 1e-12:
        return float(min_voxel_u)
    v = float(diag) / float(max(int(target_grid_points), 1) ** (1.0 / 3.0))
    v = float(np.clip(v, float(min_voxel_u), float(max_voxel_u)))
    return float(v)


def _ensure_normals(pcd: o3d.geometry.PointCloud, *, radius_u: float) -> None:
    if pcd.has_normals():
        return
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius_u), max_nn=30))


def _preprocess_fpfh(pcd: o3d.geometry.PointCloud, voxel_u: float):
    pcd_down = pcd.voxel_down_sample(float(voxel_u))
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 2.0, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 5.0, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def _register_fpfh_ransac_then_icp_v1(
    *,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_u: float,
    ransac_max_iter: int,
    ransac_confidence: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    V1 registration (same spirit as eval_registration_pipeline.py):
      - voxel downsample + normals + FPFH
      - RANSAC feature matching (with edge-length + distance checkers)
      - ICP point-to-plane refine (stricter threshold)
    """
    t0 = time.time()
    meta: Dict[str, Any] = {"status": "ok"}

    s_down, s_fpfh = _preprocess_fpfh(source, voxel_u)
    t_down, t_fpfh = _preprocess_fpfh(target, voxel_u)

    res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down,
        t_down,
        s_fpfh,
        t_fpfh,
        True,
        float(voxel_u) * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(float(voxel_u) * 1.5),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(int(ransac_max_iter), float(ransac_confidence)),
    )
    meta["ransac_fitness"] = float(getattr(res_ransac, "fitness", float("nan")))
    meta["ransac_inlier_rmse_u"] = float(getattr(res_ransac, "inlier_rmse", float("nan")))

    # Ensure normals for point-to-plane.
    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * 0.75,
        res_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    meta["icp_fitness"] = float(getattr(res_icp, "fitness", float("nan")))
    meta["icp_inlier_rmse_u"] = float(getattr(res_icp, "inlier_rmse", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        meta["status"] = "failed"
        raise RuntimeError("Invalid transformation from ICP")
    return T, meta


def _refine_icp_point_to_plane(
    *,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    voxel_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    meta: Dict[str, Any] = {"status": "ok"}

    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * 0.75,
        np.asarray(init_T, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    meta["icp_fitness"] = float(getattr(res_icp, "fitness", float("nan")))
    meta["icp_inlier_rmse_u"] = float(getattr(res_icp, "inlier_rmse", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        meta["status"] = "failed"
        raise RuntimeError("Invalid transformation from ICP refine")
    return T, meta


def _compute_board_transform_overall_then_base_v1(
    *,
    overall_ref_path: Path,
    overall_sample_path: Path,
    base_ref_path: Path,
    base_sample_path: Path,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Robust alignment (sample -> ref):
      1) overall coarse: FPFH+RANSAC+ICP on overall clouds -> T0
      2) base refine: ICP point-to-plane on base clouds (init=T0) -> T
    """
    t0 = time.time()
    meta: Dict[str, Any] = {"status": "ok", "method": "overall_then_base_v1"}

    # Load and clean.
    overall_ref = o3d.io.read_point_cloud(str(overall_ref_path))
    overall_sample = o3d.io.read_point_cloud(str(overall_sample_path))
    overall_ref = _remove_outliers(overall_ref)
    overall_sample = _remove_outliers(overall_sample)

    voxel_overall_u = _estimate_voxel_u_fixed(
        overall_ref,
        target_grid_points=int(cfg["target_grid_points"]),
        min_voxel_u=float(cfg["min_voxel_u"]),
        max_voxel_u=float(cfg["max_voxel_u"]),
    )
    meta["voxel_overall_u"] = float(voxel_overall_u)

    T0, meta_overall = _register_fpfh_ransac_then_icp_v1(
        source=overall_sample,
        target=overall_ref,
        voxel_u=float(voxel_overall_u),
        ransac_max_iter=int(cfg["ransac_max_iter"]),
        ransac_confidence=float(cfg["ransac_confidence"]),
    )
    meta["overall_stage"] = dict(meta_overall)

    base_ref = o3d.io.read_point_cloud(str(base_ref_path))
    base_sample = o3d.io.read_point_cloud(str(base_sample_path))
    base_ref = _remove_outliers(base_ref)
    base_sample = _remove_outliers(base_sample)

    voxel_base_u = _estimate_voxel_u_fixed(
        base_ref,
        target_grid_points=int(cfg["target_grid_points"]),
        min_voxel_u=float(cfg["min_voxel_u"]),
        max_voxel_u=float(cfg["max_voxel_u"]),
    )
    meta["voxel_base_u"] = float(voxel_base_u)

    T, meta_base = _refine_icp_point_to_plane(
        source=base_sample,
        target=base_ref,
        init_T=T0,
        voxel_u=float(voxel_base_u),
    )
    meta["base_refine_stage"] = dict(meta_base)

    # Promote base refine ICP stats to top-level for easier consumption.
    meta["icp_fitness"] = float(meta_base.get("icp_fitness", float("nan")))
    meta["icp_inlier_rmse_u"] = float(meta_base.get("icp_inlier_rmse_u", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        meta["status"] = "failed"
        raise RuntimeError("Invalid overall_then_base transformation")
    return T, meta


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape((0, 3))
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    out = (T @ pts_h.T).T[:, :3]
    return out.astype(np.float64)


def _curve_length_u(curve: np.ndarray) -> float:
    c = np.asarray(curve, dtype=np.float64)
    if len(c) < 2:
        return 0.0
    seg = np.linalg.norm(c[1:] - c[:-1], axis=1)
    return float(np.sum(seg))


def _resample_by_count(curve: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(curve, dtype=np.float64)
    if len(pts) == 0:
        return pts.reshape((0, 3))
    if len(pts) == 1:
        return np.repeat(pts, repeats=max(1, int(n)), axis=0)
    n = max(2, int(n))

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], repeats=n, axis=0)

    sample_s = np.linspace(0.0, total, n)
    idx = np.searchsorted(s, sample_s, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (sample_s - s[idx]) / (seg[idx] + 1e-12)
    out = pts[idx] + (pts[idx + 1] - pts[idx]) * t[:, None]
    return out


def _station_s_mm(curve: np.ndarray, unit_scale: float) -> np.ndarray:
    c = np.asarray(curve, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3 or len(c) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(c) == 1:
        return np.zeros((1,), dtype=np.float64)
    seg_u = np.linalg.norm(c[1:] - c[:-1], axis=1)
    s_u = np.concatenate([[0.0], np.cumsum(seg_u)])
    return s_u / float(unit_scale + 1e-12)


def _smooth_1d_edge(arr: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size == 0 or int(window) <= 1:
        return x.copy()
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return x.copy()
    pad = win // 2
    kernel = np.ones((win,), dtype=np.float64) / float(win)
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid").astype(np.float64)


def _fill_nan_1d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()
    mask = np.isfinite(x)
    if np.all(mask):
        return x.copy()
    if not np.any(mask):
        return np.zeros_like(x)
    idx = np.arange(x.size, dtype=np.float64)
    filled = x.copy()
    filled[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return filled


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _tangents(curve: np.ndarray) -> np.ndarray:
    c = np.asarray(curve, dtype=np.float64)
    if len(c) < 2:
        return np.zeros((len(c), 3), dtype=np.float64)
    t = np.zeros_like(c)
    t[0] = c[1] - c[0]
    t[-1] = c[-1] - c[-2]
    if len(c) > 2:
        t[1:-1] = c[2:] - c[:-2]
    tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return t / tn


def _orient_curve_to_base_start(curve: np.ndarray, base_end_idx_local: Any) -> np.ndarray:
    # Same as jiance/1.11.py:orient_curve_to_base_start().
    pts = np.asarray(curve, dtype=np.float64)
    if len(pts) < 2:
        return pts
    try:
        idx = int(base_end_idx_local)
    except Exception:
        return pts
    return pts[::-1] if idx == -1 else pts


def _orient_curve_to_match_endpoints(curve: np.ndarray, ref_curve: np.ndarray) -> np.ndarray:
    pts = np.asarray(curve, dtype=np.float64)
    ref = np.asarray(ref_curve, dtype=np.float64)
    if len(pts) < 2 or len(ref) < 2:
        return pts
    err_fwd = float(np.linalg.norm(pts[0] - ref[0]) + np.linalg.norm(pts[-1] - ref[-1]))
    err_rev = float(np.linalg.norm(pts[-1] - ref[0]) + np.linalg.norm(pts[0] - ref[-1]))
    return pts[::-1] if err_rev < err_fwd else pts


def _choose_reference(samples: Sequence[SamplePair]) -> str:
    env = str(os.getenv("DAIMA_TOL_REFERENCE_ID", "")).strip()
    if env:
        return env
    raw = str(REFERENCE_SAMPLE_ID).strip()
    if raw:
        return raw
    return samples[0].sample_id if samples else ""


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _finite(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size <= 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def _maybe_write_per_stable_id_boxplots(
    *,
    per_needle_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    stable_ids: Sequence[int],
) -> Dict[str, Any]:
    """
    Write per-stable_id boxplots for key per-needle metrics.

    Files:
      - boxplot_max_lateral_mm.png / _zoom.png
      - boxplot_tip_short_mm.png / _zoom.png
      - boxplot_len_short_mm.png / _zoom.png

    Returns:
      A JSON-serializable dict with plot file list and plotting meta.
    """
    info: Dict[str, Any] = {"status": "skipped", "reason": "", "plots": [], "zoom": {}}
    if not per_needle_rows:
        info["reason"] = "no per_needle_rows"
        return info

    try:
        import matplotlib

        # Prefer a non-interactive backend for headless runs. If pyplot is already imported
        # (e.g., by IDE helpers), backend switching may fail; we'll proceed best-effort.
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        info["reason"] = f"matplotlib not available: {e}"
        return info

    out_dir.mkdir(parents=True, exist_ok=True)

    def _group_values(metric_key: str) -> List[List[float]]:
        grouped: Dict[int, List[float]] = {int(sid): [] for sid in stable_ids}
        for r in per_needle_rows:
            try:
                sid = int(r.get("stable_id", -1))
            except Exception:
                continue
            if sid not in grouped:
                continue
            v = r.get(metric_key, float("nan"))
            if _finite(v):
                grouped[sid].append(float(v))

        # Matplotlib boxplot does not accept empty lists. Use [nan] placeholder.
        out: List[List[float]] = []
        for sid in stable_ids:
            vals = grouped.get(int(sid), [])
            out.append(vals if vals else [float("nan")])
        return out

    def _labels(metric_data: List[List[float]]) -> List[str]:
        labels: List[str] = []
        for sid, vals in zip(stable_ids, metric_data):
            n = int(np.sum(np.isfinite(np.asarray(vals, dtype=np.float64))))
            labels.append(f"{int(sid)}\n(n={n})")
        return labels

    def _save_boxplot(
        metric_key: str, *, title: str, out_name: str, ylim: Optional[Tuple[float, float]] = None
    ) -> None:
        data = _group_values(metric_key)
        labels = _labels(data)

        # Keep the figure wide enough for ~20 stable_ids.
        fig_w = max(12.0, float(len(stable_ids)) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, 4.2), dpi=200)
        # Matplotlib 3.9 renamed `labels` -> `tick_labels` (kept for backward compatibility).
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=True)  # type: ignore[call-arg]
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title(title)
        ax.set_ylabel("mm")
        ax.grid(True, axis="y", alpha=0.25)
        if ylim is not None and np.isfinite(ylim[1]) and float(ylim[1]) > float(ylim[0]):
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        fig.tight_layout()
        fig.savefig(str(out_dir / out_name), bbox_inches="tight")
        plt.close(fig)

        info["plots"].append(out_name)

    def _compute_zoom_ylim(metric_key: str, *, q: float) -> Tuple[float, float]:
        vals: List[float] = []
        for r in per_needle_rows:
            v = r.get(metric_key, float("nan"))
            if _finite(v):
                vals.append(float(v))
        p = _percentile(vals, q)
        if not np.isfinite(p):
            return (0.0, float("nan"))
        # Add a small headroom so whiskers/median are not glued to the top boundary.
        ymax = float(max(p * 1.10, 1e-6))
        return (0.0, ymax)

    # Zoom rule: clip y-axis to global p95 range (consistent with V1 tolerance p95 spirit).
    zoom_q = 0.95
    zooms = {
        "max_lateral_mm": _compute_zoom_ylim("max_lateral_mm", q=zoom_q),
        "tip_short_mm": _compute_zoom_ylim("tip_short_mm", q=zoom_q),
        "len_short_mm": _compute_zoom_ylim("len_short_mm", q=zoom_q),
    }
    info["zoom"] = {
        "quantile": float(zoom_q),
        "ymax_by_metric": {k: float(v[1]) for k, v in zooms.items()},
    }

    _save_boxplot(
        "max_lateral_mm",
        title="Per-stable_id distribution: max_lateral_mm",
        out_name="boxplot_max_lateral_mm.png",
        ylim=None,
    )
    _save_boxplot(
        "max_lateral_mm",
        title=f"Per-stable_id distribution: max_lateral_mm (zoom p{int(zoom_q*100)})",
        out_name="boxplot_max_lateral_mm_zoom.png",
        ylim=zooms["max_lateral_mm"],
    )
    _save_boxplot(
        "tip_short_mm",
        title="Per-stable_id distribution: tip_short_mm",
        out_name="boxplot_tip_short_mm.png",
        ylim=None,
    )
    _save_boxplot(
        "tip_short_mm",
        title=f"Per-stable_id distribution: tip_short_mm (zoom p{int(zoom_q*100)})",
        out_name="boxplot_tip_short_mm_zoom.png",
        ylim=zooms["tip_short_mm"],
    )
    _save_boxplot(
        "len_short_mm",
        title="Per-stable_id distribution: len_short_mm",
        out_name="boxplot_len_short_mm.png",
        ylim=None,
    )
    _save_boxplot(
        "len_short_mm",
        title=f"Per-stable_id distribution: len_short_mm (zoom p{int(zoom_q*100)})",
        out_name="boxplot_len_short_mm_zoom.png",
        ylim=zooms["len_short_mm"],
    )

    info["status"] = "ok"
    info["reason"] = ""
    return info


def _maybe_write_tolerance_band_plots(*, tolerance: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Visualize the V1 tolerance band directly (p95 * inflate_factor), rather than per-sample distributions.

    Files (under out_dir):
      - tol_lateral_band_grid.png: per-stable_id tol_lateral_mm curves as small multiples
      - tol_lateral_heatmap.png: stable_id x arc-position heatmap of tol_lateral_mm
      - tol_axial_thresholds.png: per-stable_id bars for tol_tip_short_mm / tol_len_short_mm

    Returns:
      A JSON-serializable dict with plot file list and plotting meta.
    """
    info: Dict[str, Any] = {"status": "skipped", "reason": "", "plots": [], "meta": {}}

    needles = tolerance.get("needles", [])
    if not isinstance(needles, list) or not needles:
        info["reason"] = "no needles in tolerance model"
        return info

    inflate_factor = float(
        tolerance.get("lateral_inflate_factor", tolerance.get("inflate_factor", float(INFLATE_FACTOR)))
    )
    resample_n = int(tolerance.get("resample_n", 0) or int(RESAMPLE_N))
    show_pre_inflate = bool(PLOT_SHOW_PRE_INFLATE_P95) and np.isfinite(inflate_factor) and inflate_factor > 1e-12

    recs: List[Dict[str, Any]] = []
    for nd in needles:
        if not isinstance(nd, dict):
            continue
        try:
            sid = int(nd.get("stable_id", -1))
        except Exception:
            continue
        if sid < 0:
            continue

        tol_lat = np.asarray(nd.get("tol_lateral_mm", []), dtype=np.float64).reshape(-1)
        if tol_lat.size < 2 or not np.any(np.isfinite(tol_lat)):
            continue

        tol_tip = nd.get("tol_tip_short_mm", float("nan"))
        tol_len = nd.get("tol_len_short_mm", float("nan"))
        stats = nd.get("stats", {}) if isinstance(nd.get("stats", {}), dict) else {}
        nominal_len = stats.get("nominal_length_mm", float("nan"))

        recs.append(
            {
                "stable_id": int(sid),
                "tol_lateral_mm": tol_lat.astype(np.float64),
                "tol_tip_short_mm": float(tol_tip) if _finite(tol_tip) else float("nan"),
                "tol_len_short_mm": float(tol_len) if _finite(tol_len) else float("nan"),
                "nominal_length_mm": float(nominal_len) if _finite(nominal_len) else float("nan"),
            }
        )

    if not recs:
        info["reason"] = "no valid tol_lateral_mm arrays"
        return info

    recs.sort(key=lambda r: int(r["stable_id"]))
    stable_ids = [int(r["stable_id"]) for r in recs]

    try:
        import matplotlib

        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        info["reason"] = f"matplotlib not available: {e}"
        return info

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Grid plot (small multiples) ---
    # Use a common y-limit so the tubes are comparable across stable_ids.
    all_vals = np.concatenate([r["tol_lateral_mm"] for r in recs], axis=0)
    all_vals = all_vals[np.isfinite(all_vals)]
    ymax = float(np.max(all_vals)) if all_vals.size else float("nan")
    ymax = float(max(ymax * 1.05, 1e-6)) if np.isfinite(ymax) else float("nan")

    n = len(recs)
    ncols = min(5, max(1, n))
    nrows = int(np.ceil(float(n) / float(ncols)))
    fig_w = max(12.0, float(ncols) * 3.0)
    fig_h = max(6.0, float(nrows) * 2.2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=200, sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for ax_i, ax in enumerate(axes_arr):
        if ax_i >= n:
            ax.axis("off")
            continue
        r = recs[ax_i]
        sid = int(r["stable_id"])
        tol_lat = np.asarray(r["tol_lateral_mm"], dtype=np.float64).reshape(-1)
        x = np.linspace(0.0, 100.0, len(tol_lat))
        ax.plot(x, tol_lat, color="#1f77b4", lw=1.5, label="tol (p95*inflate)")
        if show_pre_inflate:
            ax.plot(x, tol_lat / float(inflate_factor), color="#ff7f0e", lw=1.0, ls="--", alpha=0.95, label="p95")
        ttl = f"sid={sid}"
        if _finite(r.get("nominal_length_mm", float("nan"))):
            ttl += f" | L={float(r['nominal_length_mm']):.1f}mm"
        ax.set_title(ttl, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlim(0.0, 100.0)
        if np.isfinite(ymax):
            ax.set_ylim(0.0, ymax)
        ax.set_xticks([0.0, 50.0, 100.0])

    # Common labels.
    for ax in axes_arr[:n]:
        ax.tick_params(labelsize=8)
    fig.suptitle("V1 tolerance band: tol_lateral_mm along arc (mm)", fontsize=12)
    fig.text(0.5, 0.02, "Arc position (%)", ha="center", va="center")
    fig.text(0.015, 0.5, "mm", ha="center", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.04, 1.0, 0.95])
    grid_name = "tol_lateral_band_grid.png"
    fig.savefig(str(out_dir / grid_name), bbox_inches="tight")
    plt.close(fig)
    info["plots"].append(grid_name)

    # --- 2) Heatmap plot ---
    # Resample each tol vector to a common length for a clean heatmap.
    resample_n_hm = int(max(2, resample_n))
    mat = np.full((n, resample_n_hm), np.nan, dtype=np.float64)
    for i, r in enumerate(recs):
        tol_lat = np.asarray(r["tol_lateral_mm"], dtype=np.float64).reshape(-1)
        x0 = np.linspace(0.0, 1.0, len(tol_lat))
        x1 = np.linspace(0.0, 1.0, resample_n_hm)
        mat[i] = np.interp(x1, x0, tol_lat)

    vmax = float(np.nanmax(mat)) if np.any(np.isfinite(mat)) else float("nan")
    vmax = float(max(vmax, 1e-6)) if np.isfinite(vmax) else float("nan")

    fig, ax = plt.subplots(figsize=(12.0, max(4.0, float(n) * 0.35)), dpi=200)
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=str(PLOT_HEATMAP_CMAP),
        vmin=0.0,
        vmax=vmax if np.isfinite(vmax) else None,
        extent=[0.0, 100.0, -0.5, float(n) - 0.5],
    )
    ax.set_title("V1 tolerance band heatmap: tol_lateral_mm (mm)")
    ax.set_xlabel("Arc position (%)")
    ax.set_ylabel("stable_id")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([str(s) for s in stable_ids], fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("tol_lateral_mm (mm)")
    fig.tight_layout()
    heatmap_name = "tol_lateral_heatmap.png"
    fig.savefig(str(out_dir / heatmap_name), bbox_inches="tight")
    plt.close(fig)
    info["plots"].append(heatmap_name)

    # --- 3) Axial thresholds (tip/len shortness) ---
    tip_vals = np.asarray([r.get("tol_tip_short_mm", float("nan")) for r in recs], dtype=np.float64)
    len_vals = np.asarray([r.get("tol_len_short_mm", float("nan")) for r in recs], dtype=np.float64)
    x = np.arange(n, dtype=int)

    fig_w = max(12.0, float(n) * 0.55)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 6.0), dpi=200, sharex=True)
    ax1.bar(x, tip_vals, color="#2ca02c", alpha=0.9)
    ax1.set_title("tol_tip_short_mm (p95*inflate)")
    ax1.set_ylabel("mm")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.bar(x, len_vals, color="#d62728", alpha=0.9)
    ax2.set_title("tol_len_short_mm (p95*inflate)")
    ax2.set_ylabel("mm")
    ax2.set_xlabel("stable_id")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in stable_ids], fontsize=8)

    fig.tight_layout()
    axial_name = "tol_axial_thresholds.png"
    fig.savefig(str(out_dir / axial_name), bbox_inches="tight")
    plt.close(fig)
    info["plots"].append(axial_name)

    info["status"] = "ok"
    info["reason"] = ""
    info["meta"] = {
        "n_needles": int(n),
        "stable_ids": [int(s) for s in stable_ids],
        "inflate_factor": float(inflate_factor),
        "show_pre_inflate_p95": bool(show_pre_inflate),
        "resample_n_heatmap": int(resample_n_hm),
        "ymax_grid_mm": float(ymax) if np.isfinite(ymax) else None,
        "vmax_heatmap_mm": float(vmax) if np.isfinite(vmax) else None,
        "cmap": str(PLOT_HEATMAP_CMAP),
    }
    return info


def main() -> int:
    unit_scale = _unit_scale()
    input_dir = _resolve_input_dir()
    out_root = (input_dir / OUTPUT_DIRNAME).resolve()
    centerline_cfg = _centerline_extract_config()
    tol_cfg = _resolve_tolerance_cfg()

    run_id = str(getattr(_DAIMA_CTX, "run_id", "")) if _DAIMA_CTX is not None else ""
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = (out_root / run_id).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # Snapshot IO paths/config for reproducibility.
    if _DAIMA_CTX is not None:
        _reg_cfg0 = _reg_config(input_dir, unit_scale)
        _DAIMA_CTX.snapshot_paths(
            inputs={"input_dir": str(input_dir)},
            outputs={"output_root": str(out_root), "output_dir": str(out_dir)},
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "unit_scale": float(unit_scale),
                "resample_n": int(RESAMPLE_N),
                "inflate_factor": float(tol_cfg["inflate_factor"]),
                "lateral_quantile": float(tol_cfg["lateral_quantile"]),
                "lateral_inflate_factor": float(tol_cfg["lateral_inflate_factor"]),
                "lateral_floor_mm": float(tol_cfg["lateral_floor_mm"]),
                "lateral_smooth_window": int(tol_cfg["lateral_smooth_window"]),
                "axial_quantile": float(tol_cfg["axial_quantile"]),
                "axial_inflate_factor": float(tol_cfg["axial_inflate_factor"]),
                "min_ok_samples_per_needle": int(MIN_OK_SAMPLES_PER_NEEDLE),
                "match_radius_mm": MATCH_RADIUS_MM,
                "exclude_fit_status": sorted(list(EXCLUDE_FIT_STATUS)),
                "output_dirname": str(OUTPUT_DIRNAME),
                "max_samples": int(MAX_SAMPLES),
                "centerline_cfg": {
                    "denoise_enable": bool(CENTERLINE_CFG_DENOISE_ENABLE),
                    "endpoint_centering_iterations": int(CENTERLINE_CFG_ENDPOINT_CENTERING_ITERS),
                    "force_final_curve_stage13": bool(CENTERLINE_CFG_FORCE_FINAL_STAGE13),
                },
                "registration_method": str(_reg_cfg0.get("method", "")),
                "overall_dir": _reg_cfg0.get("overall_dir", None),
                "registration_cfg": {
                    "target_grid_points": int(_reg_cfg0.get("target_grid_points", 0)),
                    "min_voxel_mm": _mm_from_u(float(_reg_cfg0.get("min_voxel_u", float("nan"))), unit_scale),
                    "max_voxel_mm": _mm_from_u(float(_reg_cfg0.get("max_voxel_u", float("nan"))), unit_scale),
                    "ransac_max_iter": int(_reg_cfg0.get("ransac_max_iter", 0)),
                    "ransac_confidence": float(_reg_cfg0.get("ransac_confidence", float("nan"))),
                },
            }
        )

    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_dir={out_dir}")
    print(
        "[INFO] tolerance_cfg: "
        f"lateral_q={float(tol_cfg['lateral_quantile']):.3f} "
        f"lateral_inflate={float(tol_cfg['lateral_inflate_factor']):.3f} "
        f"lateral_floor_mm={float(tol_cfg['lateral_floor_mm']):.3f} "
        f"lateral_smooth_window={int(tol_cfg['lateral_smooth_window'])} "
        f"axial_q={float(tol_cfg['axial_quantile']):.3f} "
        f"axial_inflate={float(tol_cfg['axial_inflate_factor']):.3f}"
    )

    samples, discover_errors = _discover_samples(input_dir)
    print(f"[INFO] discovered samples={len(samples)}")
    if discover_errors:
        print(f"[WARN] discovery issues={len(discover_errors)} (will be written to matching_report.json)")

    if not samples:
        msg = f"No valid (base,pins) sample pairs found under input_dir={input_dir}"
        print(f"[ERROR] {msg}")
        (out_dir / "matching_report.json").write_text(
            json.dumps(
                {"input_dir": str(input_dir), "output_dir": str(out_dir), "errors": [msg] + discover_errors},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2

    ref_id = _choose_reference(samples)
    ref = next((s for s in samples if s.sample_id == ref_id), None)
    if ref is None:
        ref = samples[0]
        ref_id = ref.sample_id
        print(f"[WARN] reference_sample_id not found; fallback to {ref_id}")
    print(f"[INFO] reference_sample_id={ref_id}")

    # 1) Registration transforms (sample -> ref).
    reg_cfg = _reg_config(input_dir, unit_scale)
    reg_method = str(reg_cfg.get("method", "base_legacy"))
    overall_dir = reg_cfg.get("overall_dir", None)
    print(f"[INFO] registration_method={reg_method}")
    if overall_dir is not None:
        print(f"[INFO] overall_dir={overall_dir}")

    reg_rows: List[Dict[str, Any]] = []
    T_map: Dict[str, np.ndarray] = {}
    reg_meta_map: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        overall_ply = ""
        if isinstance(overall_dir, Path):
            cand = (overall_dir / f"{s.sample_id}.ply").resolve()
            overall_ply = str(cand) if cand.exists() else str(cand)
        row: Dict[str, Any] = {
            "sample_id": s.sample_id,
            "status": "",
            "reg_method": reg_method,
            "overall_ply": overall_ply,
            "base_ply": str(s.base_ply),
            "pins_ply": str(s.pins_ply),
        }
        if s.sample_id == ref_id:
            T_map[s.sample_id] = np.eye(4, dtype=float)
            row.update(
                {
                    "status": "ref",
                    "voxel_size_mm": "",
                    "voxel_overall_mm": "",
                    "voxel_base_mm": "",
                    "icp_fitness": 1.0,
                    "icp_rmse_mm": 0.0,
                    "elapsed_s": 0.0,
                    "error": "",
                }
            )
            reg_meta_map[s.sample_id] = {"status": "ref", "method": reg_method}
            reg_rows.append(row)
            continue

        try:
            if str(reg_method).strip().lower() in ("overall_then_base_v1", "overall+base_v1", "overall_then_base"):
                if not isinstance(overall_dir, Path):
                    raise RuntimeError(
                        "overall_then_base_v1 requires overall_dir; set DAIMA_TOL_OVERALL_DIR or OVERALL_DIR"
                    )
                overall_ref = (overall_dir / f"{ref_id}.ply").resolve()
                overall_sam = (overall_dir / f"{s.sample_id}.ply").resolve()
                if not overall_ref.exists():
                    raise RuntimeError(f"missing overall ref ply: {overall_ref}")
                if not overall_sam.exists():
                    raise RuntimeError(f"missing overall sample ply: {overall_sam}")
                T, meta = _compute_board_transform_overall_then_base_v1(
                    overall_ref_path=overall_ref,
                    overall_sample_path=overall_sam,
                    base_ref_path=ref.base_ply,
                    base_sample_path=s.base_ply,
                    cfg=reg_cfg,
                )
            elif str(reg_method).strip().lower() in ("base_legacy", "base_1p11", "base_only_legacy"):
                T, meta = _compute_board_transform_1p11(ref.base_ply, s.base_ply)
                meta = dict(meta)
                meta["method"] = "base_legacy"
            else:
                raise RuntimeError(f"Unknown registration method: {reg_method}")

            T_map[s.sample_id] = T
            reg_meta_map[s.sample_id] = meta
            row["status"] = str(meta.get("status", "ok"))
            # Report voxel in mm (best-effort across different methods).
            if "voxel_base_u" in meta:
                row["voxel_base_mm"] = _mm_from_u(float(meta.get("voxel_base_u", float("nan"))), unit_scale)
                row["voxel_size_mm"] = row["voxel_base_mm"]
            else:
                row["voxel_base_mm"] = ""
                row["voxel_size_mm"] = _mm_from_u(float(meta.get("voxel_size_u", float("nan"))), unit_scale)
            if "voxel_overall_u" in meta:
                row["voxel_overall_mm"] = _mm_from_u(float(meta.get("voxel_overall_u", float("nan"))), unit_scale)
            else:
                row["voxel_overall_mm"] = ""
            row["icp_fitness"] = float(meta.get("icp_fitness", float("nan")))
            row["icp_rmse_mm"] = _mm_from_u(float(meta.get("icp_inlier_rmse_u", float("nan"))), unit_scale)
            row["elapsed_s"] = float(meta.get("elapsed_s", float("nan")))
            row["error"] = ""
        except Exception as e:
            row["status"] = "failed"
            row["voxel_size_mm"] = ""
            row["voxel_overall_mm"] = ""
            row["voxel_base_mm"] = ""
            row["icp_fitness"] = float("nan")
            row["icp_rmse_mm"] = float("nan")
            row["elapsed_s"] = float("nan")
            row["error"] = str(e)
            reg_meta_map[s.sample_id] = {"status": "failed", "method": reg_method, "error": str(e)}
        reg_rows.append(row)

    _write_csv(
        out_dir / "registration_report.csv",
        reg_rows,
        fieldnames=[
            "sample_id",
            "status",
            "reg_method",
            "overall_ply",
            "base_ply",
            "pins_ply",
            "voxel_size_mm",
            "voxel_overall_mm",
            "voxel_base_mm",
            "icp_fitness",
            "icp_rmse_mm",
            "elapsed_s",
            "error",
        ],
    )

    # 2) Extract centerlines using pin_cloud2center_line.py logic.
    from pin_cloud2center_line import extract_centerlines_from_pcd  # type: ignore

    extracted: Dict[str, List[Dict[str, Any]]] = {}
    extract_errors: List[str] = []
    ref_infos: Optional[List[Dict[str, Any]]] = None

    for s in samples:
        if reg_meta_map.get(s.sample_id, {}).get("status") == "failed":
            extracted[s.sample_id] = []
            extract_errors.append(f"skip extraction due to failed registration: sample_id={s.sample_id}")
            continue

        try:
            needle_infos = extract_centerlines_from_pcd(str(s.pins_ply), config=centerline_cfg)
            if not isinstance(needle_infos, list):
                raise RuntimeError(f"extract_centerlines_from_pcd returned non-list: {type(needle_infos)}")
        except Exception as e:
            extracted[s.sample_id] = []
            extract_errors.append(f"extraction failed: sample_id={s.sample_id} err={e}")
            continue

        T = T_map.get(s.sample_id, np.eye(4, dtype=float))
        out_infos: List[Dict[str, Any]] = []
        for info in needle_infos:
            try:
                curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
                center = np.asarray(info.get("center_point", []), dtype=np.float64).reshape(-1)
                if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
                    continue
                if center.size != 3 or not np.all(np.isfinite(center)):
                    center = np.mean(curve, axis=0)

                curve_r = _transform_points(T, curve)
                center_r = _transform_points(T, center.reshape(1, 3)).reshape(3)
                out_infos.append(
                    {
                        "needle_id": int(info.get("needle_id", -1)),
                        "stable_id_src": int(info.get("stable_id", -1)),
                        "fit_status": str(info.get("overall_status", "ok")),
                        "base_end_idx_local": info.get("base_end_idx_local", None),
                        "curve_u": curve_r,
                        "center_u": center_r,
                    }
                )
            except Exception:
                continue

        extracted[s.sample_id] = out_infos
        if s.sample_id == ref_id:
            ref_infos = out_infos

    if not ref_infos:
        msg = f"reference extraction failed or produced no curves: ref_id={ref_id}"
        print(f"[ERROR] {msg}")
        (out_dir / "matching_report.json").write_text(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "output_dir": str(out_dir),
                    "reference_sample_id": ref_id,
                    "errors": [msg] + discover_errors + extract_errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 3

    # Build reference anchors by stable_id from reference sample.
    ref_by_sid: Dict[int, Dict[str, Any]] = {}
    for info in ref_infos:
        sid = int(info.get("stable_id_src", -1))
        if sid < 0:
            continue
        if sid not in ref_by_sid:
            ref_by_sid[sid] = info

    ref_sids = sorted(ref_by_sid.keys())
    ref_centers = np.array([ref_by_sid[sid]["center_u"] for sid in ref_sids], dtype=np.float64)
    if len(ref_centers) < 2:
        msg = f"reference has too few anchors for matching: n={len(ref_centers)}"
        print(f"[ERROR] {msg}")
        (out_dir / "matching_report.json").write_text(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "output_dir": str(out_dir),
                    "reference_sample_id": ref_id,
                    "errors": [msg] + discover_errors + extract_errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 4

    # Auto match radius unless user overrides.
    match_radius_u: float
    if isinstance(MATCH_RADIUS_MM, (int, float)) and float(MATCH_RADIUS_MM) > 0:
        match_radius_u = _u_from_mm(float(MATCH_RADIUS_MM), unit_scale)
        match_mode = "manual"
    else:
        tree_tmp = cKDTree(ref_centers)
        d, _ = tree_tmp.query(ref_centers, k=min(2, len(ref_centers)))
        nn = d[:, 1] if d.ndim == 2 and d.shape[1] >= 2 else np.array([np.nan], dtype=np.float64)
        nn_med = float(np.nanmedian(nn)) if np.any(np.isfinite(nn)) else float("nan")
        if not np.isfinite(nn_med) or nn_med <= 1e-9:
            nn_med = float(np.linalg.norm(ref_centers.max(axis=0) - ref_centers.min(axis=0))) * 0.1
        match_radius_u = float(nn_med) * 0.5
        match_mode = "auto"
    match_radius_mm = _mm_from_u(match_radius_u, unit_scale)
    print(f"[INFO] match_radius={match_radius_mm:.3f}mm ({match_mode})")

    # 3) Match each sample needle to reference stable_id anchors.
    tree_ref = cKDTree(ref_centers)
    curves_by_sid: Dict[int, List[Dict[str, Any]]] = {sid: [] for sid in ref_sids}

    matching_report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "reference_sample_id": str(ref_id),
        "registration_method": str(reg_method),
        "overall_dir": str(overall_dir) if isinstance(overall_dir, Path) else None,
        "registration_cfg": {
            "target_grid_points": int(reg_cfg.get("target_grid_points", 0)),
            "min_voxel_mm": _mm_from_u(float(reg_cfg.get("min_voxel_u", float("nan"))), unit_scale),
            "max_voxel_mm": _mm_from_u(float(reg_cfg.get("max_voxel_u", float("nan"))), unit_scale),
            "ransac_max_iter": int(reg_cfg.get("ransac_max_iter", 0)),
            "ransac_confidence": float(reg_cfg.get("ransac_confidence", float("nan"))),
        },
        "unit_scale": float(unit_scale),
        "resample_n": int(RESAMPLE_N),
        "inflate_factor": float(tol_cfg["inflate_factor"]),
        "lateral_quantile": float(tol_cfg["lateral_quantile"]),
        "lateral_inflate_factor": float(tol_cfg["lateral_inflate_factor"]),
        "lateral_floor_mm": float(tol_cfg["lateral_floor_mm"]),
        "lateral_smooth_window": int(tol_cfg["lateral_smooth_window"]),
        "axial_quantile": float(tol_cfg["axial_quantile"]),
        "axial_inflate_factor": float(tol_cfg["axial_inflate_factor"]),
        "min_ok_samples_per_needle": int(MIN_OK_SAMPLES_PER_NEEDLE),
        "match_radius_mm": float(match_radius_mm),
        "exclude_fit_status": sorted(list(EXCLUDE_FIT_STATUS)),
        "discovery_errors": discover_errors,
        "extraction_errors": extract_errors,
        "samples": [],
    }

    for s in samples:
        sample_entry: Dict[str, Any] = {
            "sample_id": s.sample_id,
            "base_ply": str(s.base_ply),
            "pins_ply": str(s.pins_ply),
            "registration": reg_meta_map.get(s.sample_id, {}),
            "pairs": [],
            "missing_stable_ids": [],
            "extra_needle_ids": [],
        }

        infos = extracted.get(s.sample_id, [])
        if not infos:
            sample_entry["missing_stable_ids"] = list(map(int, ref_sids))
            matching_report["samples"].append(sample_entry)
            continue

        cand: List[Dict[str, Any]] = []
        for k, info in enumerate(infos):
            center = np.asarray(info["center_u"], dtype=np.float64).reshape(3)
            d_u, j = tree_ref.query(center, k=1)
            if np.isfinite(d_u) and float(d_u) <= float(match_radius_u):
                cand.append({"dist_u": float(d_u), "stable_id": int(ref_sids[int(j)]), "needle_idx": int(k)})

        cand.sort(key=lambda x: float(x["dist_u"]))
        used_sid: set[int] = set()
        used_idx: set[int] = set()
        pairs: List[Tuple[int, int, float]] = []
        for c in cand:
            sid = int(c["stable_id"])
            idx = int(c["needle_idx"])
            if sid in used_sid or idx in used_idx:
                continue
            used_sid.add(sid)
            used_idx.add(idx)
            pairs.append((sid, idx, float(c["dist_u"])))

        sample_entry["missing_stable_ids"] = [int(sid) for sid in ref_sids if int(sid) not in used_sid]
        sample_entry["extra_needle_ids"] = [
            int(infos[i]["needle_id"]) for i in range(len(infos)) if int(i) not in used_idx
        ]

        for sid, idx, dist_u in pairs:
            info = infos[int(idx)]
            fit_status = str(info.get("fit_status", "ok"))
            used_for_stats = fit_status not in EXCLUDE_FIT_STATUS
            sample_entry["pairs"].append(
                {
                    "stable_id": int(sid),
                    "needle_id": int(info.get("needle_id", -1)),
                    "dist_mm": _mm_from_u(float(dist_u), unit_scale),
                    "fit_status": fit_status,
                    "used_for_stats": bool(used_for_stats),
                }
            )
            curves_by_sid[int(sid)].append(
                {
                    "sample_id": s.sample_id,
                    "stable_id": int(sid),
                    "needle_id": int(info.get("needle_id", -1)),
                    "fit_status": fit_status,
                    "base_end_idx_local": info.get("base_end_idx_local", None),
                    "curve_u": info["curve_u"],
                }
            )

        matching_report["samples"].append(sample_entry)

    (out_dir / "matching_report.json").write_text(
        json.dumps(matching_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) Build nominal (median) per stable_id and compute tolerances.
    needle_out: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    per_needle_rows: List[Dict[str, Any]] = []

    for sid in ref_sids:
        recs = curves_by_sid.get(int(sid), [])
        n_total = len(recs)
        ok_recs = [r for r in recs if str(r.get("fit_status", "ok")) not in EXCLUDE_FIT_STATUS]
        n_ok = len(ok_recs)
        n_excl = n_total - n_ok

        if n_ok <= 0:
            needle_out.append(
                {
                    "stable_id": int(sid),
                    "status": "no_nominal",
                    "n_samples_total": int(n_total),
                    "n_samples_ok_used": 0,
                    "n_samples_excluded_fit": int(n_excl),
                    "nominal_curve_u": [],
                    "tol_lateral_mm": [],
                    "tol_tip_short_mm": None,
                    "tol_len_short_mm": None,
                    "stats": {},
                }
            )
            summary_rows.append(
                {
                    "stable_id": int(sid),
                    "status": "no_nominal",
                    "n_samples_total": int(n_total),
                    "n_samples_ok_used": 0,
                    "n_samples_excluded_fit": int(n_excl),
                    "nominal_length_mm": "",
                    "tol_tip_short_mm": "",
                    "tol_len_short_mm": "",
                    "tol_lateral_max_mm": "",
                    "tol_lateral_mean_mm": "",
                }
            )
            continue

        # Orientation reference curve.
        ref_curve = _orient_curve_to_base_start(
            np.asarray(ok_recs[0]["curve_u"], dtype=np.float64),
            ok_recs[0].get("base_end_idx_local", None),
        )
        ref_curve = _resample_by_count(ref_curve, int(RESAMPLE_N))

        stack: List[np.ndarray] = []
        for r in ok_recs:
            c = np.asarray(r["curve_u"], dtype=np.float64)
            c = _orient_curve_to_base_start(c, r.get("base_end_idx_local", None))
            c = _orient_curve_to_match_endpoints(c, ref_curve)
            c = _resample_by_count(c, int(RESAMPLE_N))
            stack.append(c)

        nominal = np.median(np.stack(stack, axis=0), axis=0)
        nominal = _resample_by_count(nominal, int(RESAMPLE_N))

        tan = _tangents(nominal)
        axis = _normalize(nominal[-1] - nominal[0])
        len_nom_mm = _mm_from_u(_curve_length_u(nominal), unit_scale)

        station_s_mm = _station_s_mm(nominal, unit_scale)
        station_step_mm = float(np.nanmedian(np.diff(station_s_mm))) if len(station_s_mm) >= 2 else 0.0
        coverage_slack_mm = max(0.0, 0.5 * station_step_mm)

        lateral_all: List[np.ndarray] = []
        coverage_masks: List[np.ndarray] = []
        tip_short_all: List[float] = []
        len_short_all: List[float] = []

        for r in ok_recs:
            c = np.asarray(r["curve_u"], dtype=np.float64)
            c = _orient_curve_to_base_start(c, r.get("base_end_idx_local", None))
            c = _orient_curve_to_match_endpoints(c, nominal)
            len_s_mm = _mm_from_u(_curve_length_u(c), unit_scale)
            c = _resample_by_count(c, int(RESAMPLE_N))

            delta = c - nominal
            axial = np.sum(delta * tan, axis=1, keepdims=True) * tan
            lateral_u = np.linalg.norm(delta - axial, axis=1)
            lateral_mm = lateral_u / float(unit_scale + 1e-12)
            coverage_mask = station_s_mm <= (float(len_s_mm) + float(coverage_slack_mm))
            coverage_masks.append(coverage_mask.astype(bool))
            lateral_masked = lateral_mm.astype(np.float64)
            lateral_masked[~coverage_mask] = np.nan
            lateral_all.append(lateral_masked)

            tip_axial_mm = float(np.dot((c[-1] - nominal[-1]), axis) / float(unit_scale + 1e-12))
            tip_short_mm = max(0.0, -tip_axial_mm)
            tip_short_all.append(float(tip_short_mm))

            len_short_mm = max(0.0, float(len_nom_mm - len_s_mm))
            len_short_all.append(float(len_short_mm))

            per_needle_rows.append(
                {
                    "sample_id": str(r.get("sample_id", "")),
                    "stable_id": int(sid),
                    "needle_id": int(r.get("needle_id", -1)),
                    "fit_status": str(r.get("fit_status", "")),
                    "tip_short_mm": float(tip_short_mm),
                    "len_short_mm": float(len_short_mm),
                    "max_lateral_mm": float(np.max(lateral_mm)) if lateral_mm.size else float("nan"),
                }
            )

        lateral_mat = np.stack(lateral_all, axis=0) if lateral_all else np.zeros((0, int(RESAMPLE_N)), dtype=np.float64)
        if lateral_mat.size:
            with np.errstate(all="ignore"):
                tol_lateral = np.nanquantile(lateral_mat, float(tol_cfg["lateral_quantile"]), axis=0)
            tol_lateral = _fill_nan_1d(np.asarray(tol_lateral, dtype=np.float64).reshape(-1))
            tol_lateral = tol_lateral * float(tol_cfg["lateral_inflate_factor"])
            tol_lateral = _smooth_1d_edge(tol_lateral, int(tol_cfg["lateral_smooth_window"]))
            if float(tol_cfg["lateral_floor_mm"]) > 0.0:
                tol_lateral = np.maximum(tol_lateral, float(tol_cfg["lateral_floor_mm"]))
        else:
            tol_lateral = np.array([], dtype=np.float64)
        tol_tip_short = float(
            np.quantile(np.asarray(tip_short_all, dtype=np.float64), float(tol_cfg["axial_quantile"]))
            * float(tol_cfg["axial_inflate_factor"])
        )
        tol_len_short = float(
            np.quantile(np.asarray(len_short_all, dtype=np.float64), float(tol_cfg["axial_quantile"]))
            * float(tol_cfg["axial_inflate_factor"])
        )

        coverage_count = (
            np.sum(np.stack(coverage_masks, axis=0).astype(np.int32), axis=0)
            if coverage_masks
            else np.zeros((len(station_s_mm),), dtype=np.int32)
        )
        coverage_rate = (
            coverage_count.astype(np.float64) / float(max(int(n_ok), 1))
            if int(n_ok) > 0
            else np.zeros((len(station_s_mm),), dtype=np.float64)
        )

        status = "ok" if n_ok >= int(MIN_OK_SAMPLES_PER_NEEDLE) else "insufficient_data"
        stats = {
            "nominal_length_mm": float(len_nom_mm),
            "tol_lateral_max_mm": float(np.max(tol_lateral)) if tol_lateral.size else 0.0,
            "tol_lateral_mean_mm": float(np.mean(tol_lateral)) if tol_lateral.size else 0.0,
            "coverage_min": int(np.min(coverage_count)) if coverage_count.size else 0,
            "coverage_tip": int(coverage_count[-1]) if coverage_count.size else 0,
        }

        needle_out.append(
            {
                "stable_id": int(sid),
                "status": status,
                "n_samples_total": int(n_total),
                "n_samples_ok_used": int(n_ok),
                "n_samples_excluded_fit": int(n_excl),
                "nominal_curve_u": nominal.astype(float).tolist(),
                "station_s_mm": station_s_mm.astype(float).tolist(),
                "tol_lateral_mm": tol_lateral.astype(float).tolist(),
                "coverage_count": coverage_count.astype(int).tolist(),
                "coverage_rate": coverage_rate.astype(float).tolist(),
                "tol_tip_short_mm": float(tol_tip_short),
                "tol_len_short_mm": float(tol_len_short),
                "stats": stats,
            }
        )

        summary_rows.append(
            {
                "stable_id": int(sid),
                "status": status,
                "n_samples_total": int(n_total),
                "n_samples_ok_used": int(n_ok),
                "n_samples_excluded_fit": int(n_excl),
                "nominal_length_mm": float(len_nom_mm),
                "tol_tip_short_mm": float(tol_tip_short),
                "tol_len_short_mm": float(tol_len_short),
                "tol_lateral_max_mm": float(stats["tol_lateral_max_mm"]),
                "tol_lateral_mean_mm": float(stats["tol_lateral_mean_mm"]),
                "coverage_min": int(stats["coverage_min"]),
                "coverage_tip": int(stats["coverage_tip"]),
            }
        )

    tolerance = {
        "meta": {"run_id": run_id, "created_at": _now_iso(), "input_dir": str(input_dir), "output_dir": str(out_dir)},
        "unit_scale": float(unit_scale),
        "reference_sample_id": str(ref_id),
        "inflate_factor": float(tol_cfg["inflate_factor"]),
        "lateral_quantile": float(tol_cfg["lateral_quantile"]),
        "lateral_inflate_factor": float(tol_cfg["lateral_inflate_factor"]),
        "lateral_floor_mm": float(tol_cfg["lateral_floor_mm"]),
        "lateral_smooth_window": int(tol_cfg["lateral_smooth_window"]),
        "axial_quantile": float(tol_cfg["axial_quantile"]),
        "axial_inflate_factor": float(tol_cfg["axial_inflate_factor"]),
        "resample_n": int(RESAMPLE_N),
        "match_radius_mm": float(match_radius_mm),
        "min_ok_samples_per_needle": int(MIN_OK_SAMPLES_PER_NEEDLE),
        "needles": needle_out,
    }

    (out_dir / "centerline_tolerance.json").write_text(
        json.dumps(tolerance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "centerline_tolerance_summary.csv",
        summary_rows,
        fieldnames=[
            "stable_id",
            "status",
            "n_samples_total",
            "n_samples_ok_used",
            "n_samples_excluded_fit",
            "nominal_length_mm",
            "tol_tip_short_mm",
            "tol_len_short_mm",
            "tol_lateral_max_mm",
            "tol_lateral_mean_mm",
            "coverage_min",
            "coverage_tip",
        ],
    )
    if per_needle_rows:
        _write_csv(
            out_dir / "golden_per_needle_metrics.csv",
            per_needle_rows,
            fieldnames=[
                "sample_id",
                "stable_id",
                "needle_id",
                "fit_status",
                "tip_short_mm",
                "len_short_mm",
                "max_lateral_mm",
            ],
        )

    # 5) Optional plots (boxplots) + analysis summary.
    analysis_summary: Dict[str, Any] = {
            "meta": {
                "run_id": run_id,
                "created_at": _now_iso(),
                "input_dir": str(input_dir),
                "output_dir": str(out_dir),
                "reference_sample_id": str(ref_id),
                "registration_method": str(reg_method),
                "lateral_quantile": float(tol_cfg["lateral_quantile"]),
                "lateral_inflate_factor": float(tol_cfg["lateral_inflate_factor"]),
                "lateral_floor_mm": float(tol_cfg["lateral_floor_mm"]),
                "lateral_smooth_window": int(tol_cfg["lateral_smooth_window"]),
                "axial_quantile": float(tol_cfg["axial_quantile"]),
                "axial_inflate_factor": float(tol_cfg["axial_inflate_factor"]),
            },
        "plots": {},
        "tolerance_band_plots": {},
    }

    plots_dir = (out_dir / "plots").resolve()

    # Legacy: per-sample distributions (boxplots).
    if bool(PLOT_ENABLE_BOXPLOTS) and per_needle_rows and ref_sids:
        analysis_summary["plots"] = _maybe_write_per_stable_id_boxplots(
            per_needle_rows=per_needle_rows,
            out_dir=plots_dir,
            stable_ids=ref_sids,
        )
    else:
        analysis_summary["plots"] = {
            "status": "skipped",
            "reason": "disabled" if not bool(PLOT_ENABLE_BOXPLOTS) else "no per_needle_rows or no reference stable_ids",
            "plots": [],
        }

    # New: visualize the tolerance band directly (p95 * inflate_factor).
    if bool(PLOT_ENABLE_TOLERANCE_BAND_PLOTS):
        analysis_summary["tolerance_band_plots"] = _maybe_write_tolerance_band_plots(
            tolerance=tolerance,
            out_dir=plots_dir,
        )
    else:
        analysis_summary["tolerance_band_plots"] = {"status": "skipped", "reason": "disabled", "plots": [], "meta": {}}

    (out_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_root / "last_output.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": _now_iso(),
                "input_dir": str(input_dir),
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Wrote: {out_dir / 'centerline_tolerance.json'}")
    print(f"[INFO] Wrote: {out_dir / 'centerline_tolerance_summary.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'registration_report.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'matching_report.json'}")
    if per_needle_rows:
        print(f"[INFO] Wrote: {out_dir / 'golden_per_needle_metrics.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'analysis_summary.json'}")
    if (
        isinstance(analysis_summary.get("plots"), dict)
        and analysis_summary["plots"].get("status") == "ok"
        or isinstance(analysis_summary.get("tolerance_band_plots"), dict)
        and analysis_summary["tolerance_band_plots"].get("status") == "ok"
    ):
        print(f"[INFO] Wrote plots under: {out_dir / 'plots'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
