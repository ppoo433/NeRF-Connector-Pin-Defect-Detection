from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


try:
    from pin_cloud2center_line import UNIT_SCALE as _UNIT_SCALE_DEFAULT  # type: ignore
except Exception:
    _UNIT_SCALE_DEFAULT = 1.0


_PLY_TYPE_TO_DTYPE: Dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "long": "<i8",
    "int64": "<i8",
    "ulong": "<u8",
    "uint64": "<u8",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class LabeledCloud:
    sample_id: str
    ply_path: Path
    overall_points: np.ndarray
    overall_colors: np.ndarray
    base_points: np.ndarray
    base_colors: np.ndarray
    pins_points: np.ndarray
    pins_colors: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class RegistrationConfig:
    method: str
    cloud_dir: Path
    ref_id: str
    source_id: str
    unit_scale: float
    target_grid_points: int
    min_voxel_mm: float
    max_voxel_mm: float
    ransac_max_iter: int
    ransac_confidence: float
    write_overlay: bool
    overlay_downsample_mm: float
    pin_inlier_thresholds_mm: List[float]
    rgb_voxel_scales_mm: List[float]
    rgb_max_iters: List[int]
    rgb_lambda_geometric: float
    rgb_max_corr_factor: float
    rgb_final_base_refine: bool
    rgb_gate_enable: bool
    rgb_gate_min_init_fitness: float
    rgb_gate_min_color_std: float
    rgb_gate_max_pin_chamfer_degrade_mm: float
    rgb_gate_max_pin_bidir_p95_degrade_mm: float
    rgb_gate_max_base_p95_degrade_mm: float
    rgb_gate_min_improve_mm: float
    norgb_enable_fgr_init: bool
    norgb_enable_gicp_refine: bool
    norgb_robust_kernel_k_mm: float
    norgb_score_w_pin_chamfer: float
    norgb_score_w_pin_bidir_p95: float
    norgb_score_w_base_p95: float
    init_backend: str
    init_backend_allow_fallback: bool
    local_refine_multiscale_enable: bool
    local_refine_voxel_scales_mm: List[float]
    local_refine_max_iters: List[int]
    local_refine_max_corr_factor: float
    local_refine_robust_enable: bool
    local_refine_robust_kernel: str
    local_refine_robust_k_mm: float
    bcjr_enable: bool
    bcjr_voxel_scales_mm: List[float]
    bcjr_max_iters: List[int]
    bcjr_lambda_base: List[float]
    bcjr_lambda_overall: List[float]
    bcjr_prior_mu: List[float]
    bcjr_weak_mode_count: int
    bcjr_q_min: float
    bcjr_lm_damping: float
    bcjr_exclude_base_overlap_factor: float
    bcjr_beta_bias: float
    bcjr_beta_r: float
    bcjr_beta_n: float
    bcjr_beta_u: float
    bcjr_beta_g: float
    bcjr_clip_translation_factor: float
    bcjr_clip_rotation_deg: List[float]
    bcjr_rescue_enable: bool
    bcjr_rescue_rotation_deg: float
    bcjr_rescue_verify_voxel_mm: float
    bcjr_rescue_verify_max_corr_factor: float
    bcjr_rescue_verify_base_guard_mm: float
    bcjr_rescue_detail_rmse_tie_mm: float
    pair_id: str
    require_route_tag: bool
    route_tag: str


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return str(default)
    t = str(v).strip()
    return t if t else str(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    t = str(v).strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _parse_float_csv(raw: str, default: List[float]) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            v = float(t)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            vals.append(float(v))
    if not vals:
        return list(default)
    out = sorted(set(float(v) for v in vals))
    return out if out else list(default)


def _parse_float_csv_ordered(raw: str, default: List[float]) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            v = float(t)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            vals.append(float(v))
    return vals if vals else list(default)


def _parse_int_csv_ordered(raw: str, default: List[int]) -> List[int]:
    vals: List[int] = []
    for tok in str(raw).replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            v = int(t)
        except Exception:
            continue
        if v > 0:
            vals.append(int(v))
    return vals if vals else list(default)


def _u_from_mm(x_mm: float, unit_scale: float) -> float:
    return float(x_mm) * float(unit_scale)


def _mm_from_u(x_u: float, unit_scale: float) -> float:
    return float(x_u) / float(unit_scale + 1e-12)


def _read_ply_header(fp) -> Tuple[int, List[Tuple[str, str]]]:
    vertex_count: Optional[int] = None
    vertex_props: List[Tuple[str, str]] = []
    in_vertex = False

    while True:
        line = fp.readline()
        if not line:
            raise RuntimeError("EOF before end_header")
        s = line.decode("ascii", errors="ignore").strip()
        if s.startswith("element "):
            parts = s.split()
            if len(parts) >= 3 and parts[1] == "vertex":
                vertex_count = int(parts[2])
                vertex_props = []
                in_vertex = True
            else:
                in_vertex = False
        elif in_vertex and s.startswith("property "):
            parts = s.split()
            if len(parts) >= 3 and parts[1] != "list":
                vertex_props.append((parts[1], parts[2]))
            else:
                raise RuntimeError(f"unsupported PLY property: {s}")

        if s == "end_header":
            break

    if vertex_count is None:
        raise RuntimeError("missing vertex count in PLY header")
    if not vertex_props:
        raise RuntimeError("missing vertex properties in PLY header")
    return int(vertex_count), vertex_props


def _build_dtype(vertex_props: List[Tuple[str, str]]) -> np.dtype:
    fields = []
    for ply_type, name in vertex_props:
        key = str(ply_type).strip().lower()
        dt = _PLY_TYPE_TO_DTYPE.get(key)
        if dt is None:
            raise RuntimeError(f"unsupported PLY dtype: {ply_type} ({name})")
        fields.append((str(name), dt))
    return np.dtype(fields)


def _read_labeled_cloud(path: Path) -> LabeledCloud:
    with open(path, "rb") as f:
        n, props = _read_ply_header(f)
        dt = _build_dtype(props)
        arr = np.fromfile(f, dtype=dt, count=int(n))

    required = ["x", "y", "z", "scalar_pin", "scalar_base"]
    for key in required:
        if key not in arr.dtype.names:
            raise RuntimeError(f"{path.name} missing required field: {key}")

    xyz = np.stack(
        [
            np.asarray(arr["x"], dtype=np.float64),
            np.asarray(arr["y"], dtype=np.float64),
            np.asarray(arr["z"], dtype=np.float64),
        ],
        axis=1,
    )
    mask_pin = np.isfinite(arr["scalar_pin"])
    mask_base = np.isfinite(arr["scalar_base"])

    color_fields: Optional[Tuple[str, str, str]] = None
    for cand in [("red", "green", "blue"), ("r", "g", "b")]:
        if all(k in arr.dtype.names for k in cand):
            color_fields = cand
            break

    if color_fields is not None:
        rgb = np.stack(
            [
                np.asarray(arr[color_fields[0]], dtype=np.float64),
                np.asarray(arr[color_fields[1]], dtype=np.float64),
                np.asarray(arr[color_fields[2]], dtype=np.float64),
            ],
            axis=1,
        )
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        p99 = float(np.quantile(rgb, 0.99)) if rgb.size > 0 else 0.0
        if p99 > 1.5:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.zeros((len(arr), 3), dtype=np.float64)

    meta = {
        "n_total": int(len(arr)),
        "n_pins": int(np.sum(mask_pin)),
        "n_base": int(np.sum(mask_base)),
        "both_count": int(np.sum(mask_pin & mask_base)),
        "neither_count": int(np.sum((~mask_pin) & (~mask_base))),
        "has_color": bool(color_fields is not None),
        "color_fields": list(color_fields) if color_fields is not None else [],
        "color_range_min": [float(np.min(rgb[:, i])) for i in range(3)] if rgb.size > 0 else [0.0, 0.0, 0.0],
        "color_range_max": [float(np.max(rgb[:, i])) for i in range(3)] if rgb.size > 0 else [0.0, 0.0, 0.0],
    }

    return LabeledCloud(
        sample_id=str(path.stem),
        ply_path=path,
        overall_points=xyz,
        overall_colors=rgb,
        base_points=xyz[mask_base],
        base_colors=rgb[mask_base],
        pins_points=xyz[mask_pin],
        pins_colors=rgb[mask_pin],
        meta=meta,
    )


def _points_to_pcd(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _points_to_pcd_with_colors(points: np.ndarray, colors: np.ndarray):
    pcd = _points_to_pcd(points)
    c = np.asarray(colors, dtype=np.float64).reshape((-1, 3))
    if len(c) != len(pcd.points):
        raise RuntimeError(f"points/colors length mismatch: {len(pcd.points)} vs {len(c)}")
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd


def _colorize_pcd(points: np.ndarray, rgb: Tuple[float, float, float]):
    pcd = _points_to_pcd(points)
    c = np.asarray(rgb, dtype=np.float64).reshape((1, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(c, repeats=max(len(pcd.points), 1), axis=0)[: len(pcd.points)])
    return pcd


def _remove_outliers(pcd, nb_neighbors: int = 50, std_ratio: float = 1.5):
    _cl, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return pcd.select_by_index(ind)


def _aabb_diag_u(pcd) -> float:
    bbox = pcd.get_axis_aligned_bounding_box()
    d = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
    return float(d)


def _estimate_voxel_u_fixed(
    pcd,
    *,
    target_grid_points: int,
    min_voxel_u: float,
    max_voxel_u: float,
) -> float:
    diag = _aabb_diag_u(pcd)
    if not np.isfinite(diag) or diag <= 1e-12:
        return float(min_voxel_u)
    v = float(diag) / float(max(int(target_grid_points), 1) ** (1.0 / 3.0))
    return float(np.clip(v, float(min_voxel_u), float(max_voxel_u)))


def _ensure_normals(pcd, radius_u: float) -> None:
    if pcd.has_normals():
        return
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius_u), max_nn=30))


def _preprocess_fpfh(pcd, voxel_u: float):
    down = pcd.voxel_down_sample(float(voxel_u))
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 5.0, max_nn=100),
    )
    return down, fpfh


def _register_fpfh_ransac_then_icp_v1(
    *,
    source,
    target,
    voxel_u: float,
    ransac_max_iter: int,
    ransac_confidence: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    meta: Dict[str, Any] = {"status": "ok"}

    src_down, src_fpfh = _preprocess_fpfh(source, voxel_u)
    tgt_down, tgt_fpfh = _preprocess_fpfh(target, voxel_u)

    res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
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

    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * 0.75,
        np.asarray(res_ransac.transformation, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    meta["ransac_fitness"] = float(getattr(res_ransac, "fitness", float("nan")))
    meta["ransac_inlier_rmse_u"] = float(getattr(res_ransac, "inlier_rmse", float("nan")))
    meta["icp_fitness"] = float(getattr(res_icp, "fitness", float("nan")))
    meta["icp_inlier_rmse_u"] = float(getattr(res_icp, "inlier_rmse", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        meta["status"] = "failed"
        raise RuntimeError("invalid transformation from ICP")
    return T, meta


def _register_fpfh_fgr_then_icp_v1(
    *,
    source,
    target,
    voxel_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    meta: Dict[str, Any] = {"status": "ok"}

    src_down, src_fpfh = _preprocess_fpfh(source, voxel_u)
    tgt_down, tgt_fpfh = _preprocess_fpfh(target, voxel_u)

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=float(voxel_u) * 1.5,
    )
    res_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        option,
    )

    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)
    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * 0.75,
        np.asarray(res_fgr.transformation, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    meta["fgr_fitness"] = float(getattr(res_fgr, "fitness", float("nan")))
    meta["fgr_inlier_rmse_u"] = float(getattr(res_fgr, "inlier_rmse", float("nan")))
    meta["icp_fitness"] = float(getattr(res_icp, "fitness", float("nan")))
    meta["icp_inlier_rmse_u"] = float(getattr(res_icp, "inlier_rmse", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        meta["status"] = "failed"
        raise RuntimeError("invalid transformation from FGR+ICP")
    return T, meta


def _register_fpfh_ransac_global_only(
    *,
    source,
    target,
    voxel_u: float,
    ransac_max_iter: int,
    ransac_confidence: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    src_down, src_fpfh = _preprocess_fpfh(source, voxel_u)
    tgt_down, tgt_fpfh = _preprocess_fpfh(target, voxel_u)
    res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
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
    T = np.asarray(res_ransac.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid transformation from global RANSAC")
    return T, {
        "status": "ok",
        "backend": "ransac",
        "ransac_fitness": float(getattr(res_ransac, "fitness", float("nan"))),
        "ransac_inlier_rmse_u": float(getattr(res_ransac, "inlier_rmse", float("nan"))),
        "elapsed_s": float(time.time() - t0),
    }


def _register_fpfh_fgr_global_only(
    *,
    source,
    target,
    voxel_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    src_down, src_fpfh = _preprocess_fpfh(source, voxel_u)
    tgt_down, tgt_fpfh = _preprocess_fpfh(target, voxel_u)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=float(voxel_u) * 1.5,
    )
    res_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        option,
    )
    T = np.asarray(res_fgr.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid transformation from global FGR")
    return T, {
        "status": "ok",
        "backend": "fgr",
        "fgr_fitness": float(getattr(res_fgr, "fitness", float("nan"))),
        "fgr_inlier_rmse_u": float(getattr(res_fgr, "inlier_rmse", float("nan"))),
        "elapsed_s": float(time.time() - t0),
    }


def _feature_correspondences_from_fpfh(
    *,
    src_down,
    tgt_down,
    src_fpfh,
    tgt_fpfh,
    mutual_filter: bool,
    max_pair_dist_u: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    corr = o3d.pipelines.registration.correspondences_from_features(
        src_fpfh,
        tgt_fpfh,
        mutual_filter=bool(mutual_filter),
    )
    corr_np = np.asarray(corr, dtype=np.int64)
    if corr_np.ndim != 2 or corr_np.shape[1] != 2 or corr_np.shape[0] < 3:
        raise RuntimeError("insufficient FPFH correspondences for global initialization")

    src_pts = np.asarray(src_down.points, dtype=np.float64)
    tgt_pts = np.asarray(tgt_down.points, dtype=np.float64)

    src_idx = np.clip(corr_np[:, 0], 0, max(src_pts.shape[0] - 1, 0))
    tgt_idx = np.clip(corr_np[:, 1], 0, max(tgt_pts.shape[0] - 1, 0))
    src_corr = src_pts[src_idx]
    tgt_corr = tgt_pts[tgt_idx]

    kept = int(src_corr.shape[0])
    if np.isfinite(float(max_pair_dist_u)) and float(max_pair_dist_u) > 0:
        pair_d = np.linalg.norm(src_corr - tgt_corr, axis=1)
        keep_mask = pair_d <= float(max_pair_dist_u)
        if int(np.count_nonzero(keep_mask)) >= 3:
            src_corr = src_corr[keep_mask]
            tgt_corr = tgt_corr[keep_mask]
            kept = int(src_corr.shape[0])

    if src_corr.shape[0] < 3:
        raise RuntimeError("too few filtered correspondences for global initialization")

    meta = {
        "corr_count_raw": int(corr_np.shape[0]),
        "corr_count_used": int(kept),
        "corr_mutual_filter": bool(mutual_filter),
        "corr_max_pair_dist_u": float(max_pair_dist_u),
    }
    return src_corr.T, tgt_corr.T, meta


def _safe_setattr(obj: Any, key: str, value: Any) -> None:
    if hasattr(obj, key):
        setattr(obj, key, value)


def _register_teaserpp_global_only(
    *,
    source,
    target,
    voxel_u: float,
    mutual_filter: bool,
    corr_max_pair_dist_u: float,
    noise_bound_u: float,
    cbar2: float,
    estimate_scaling: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()

    try:
        import teaserpp_python  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"teaserpp_python import failed: {exc}")

    src_down, src_fpfh = _preprocess_fpfh(source, voxel_u)
    tgt_down, tgt_fpfh = _preprocess_fpfh(target, voxel_u)
    src_corr, tgt_corr, corr_meta = _feature_correspondences_from_fpfh(
        src_down=src_down,
        tgt_down=tgt_down,
        src_fpfh=src_fpfh,
        tgt_fpfh=tgt_fpfh,
        mutual_filter=bool(mutual_filter),
        max_pair_dist_u=float(corr_max_pair_dist_u),
    )

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = float(max(float(noise_bound_u), 1e-12))
    params.cbar2 = float(max(float(cbar2), 1e-6))
    params.estimate_scaling = bool(estimate_scaling)

    algo_enum = getattr(teaserpp_python.RobustRegistrationSolver, "ROTATION_ESTIMATION_ALGORITHM", None)
    if algo_enum is not None and hasattr(algo_enum, "GNC_TLS"):
        params.rotation_estimation_algorithm = algo_enum.GNC_TLS

    _safe_setattr(params, "rotation_gnc_factor", 1.4)
    _safe_setattr(params, "rotation_max_iterations", 100)
    _safe_setattr(params, "rotation_cost_threshold", 1e-12)

    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(src_corr, tgt_corr)
    sol = solver.getSolution()

    R = np.asarray(sol.rotation, dtype=np.float64)
    t = np.asarray(sol.translation, dtype=np.float64).reshape(-1)
    if R.shape != (3, 3) or t.shape[0] < 3:
        raise RuntimeError("invalid TEASER++ solution shape")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t[:3]
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid transformation from TEASER++")

    meta = {
        "status": "ok",
        "backend": "teaserpp",
        **corr_meta,
        "teaser_noise_bound_u": float(params.noise_bound),
        "teaser_cbar2": float(params.cbar2),
        "teaser_estimate_scaling": bool(params.estimate_scaling),
        "elapsed_s": float(time.time() - t0),
    }
    return T, meta


def _parse_transform_4x4_from_text(raw: str) -> np.ndarray:
    lines = [str(line).strip() for line in str(raw).splitlines()]
    start = 0
    for idx, line in enumerate(lines):
        if "MATRIX" in line.upper():
            start = idx + 1
            break

    rows: List[List[float]] = []
    fpat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    for line in lines[start:]:
        nums = [float(x) for x in fpat.findall(line)]
        if len(nums) >= 4:
            rows.append(nums[:4])
            if len(rows) == 4:
                break

    if len(rows) != 4:
        all_nums = [float(x) for x in fpat.findall(str(raw))]
        if len(all_nums) >= 16:
            rows = [all_nums[i * 4 : (i + 1) * 4] for i in range(4)]

    T = np.asarray(rows, dtype=np.float64)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("failed to parse 4x4 transform")
    return T


def _write_xyz_ply_ascii_float(path: Path, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError("invalid point array for PLY writing")
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {int(pts.shape[0])}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("end_header\n")
        for p in pts:
            fp.write(f"{float(p[0]):.9f} {float(p[1]):.9f} {float(p[2]):.9f}\n")


def _register_opengr_super4pcs_global_only(
    *,
    source,
    target,
    voxel_u: float,
    bin_path: str,
    overlap: float,
    max_time_s: int,
    sample_points: int,
    delta_factor: float,
    max_input_points: int,
    timeout_s: int,
    input_voxel_factor: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    bin_cmd = str(bin_path).strip() or "Super4PCS"
    if shutil.which(bin_cmd) is None and (not Path(bin_cmd).exists()):
        raise RuntimeError(f"OpenGR binary not found: {bin_cmd}")

    overlap_eff = float(np.clip(float(overlap), 0.01, 0.99))
    max_time_eff = int(max(int(max_time_s), 1))
    sample_eff = int(max(int(sample_points), 200))
    delta_u = float(max(float(delta_factor) * float(voxel_u), 1e-9))
    timeout_eff = int(max(int(timeout_s), max_time_eff + 5))
    input_max_pts = int(max(int(max_input_points), 2000))
    input_voxel_u = float(max(float(input_voxel_factor) * float(voxel_u), 1e-12))

    src_in = source
    tgt_in = target
    if input_voxel_u > float(voxel_u) + 1e-12:
        src_tmp = src_in.voxel_down_sample(input_voxel_u)
        tgt_tmp = tgt_in.voxel_down_sample(input_voxel_u)
        if len(src_tmp.points) >= 200 and len(tgt_tmp.points) >= 200:
            src_in = src_tmp
            tgt_in = tgt_tmp

    def _cap_points(pcd, max_points: int):
        n = int(np.asarray(pcd.points).shape[0])
        if n <= max_points:
            return pcd, n, n
        step = int(max(2, int(np.ceil(float(n) / float(max_points)))))
        pcd2 = pcd.uniform_down_sample(step)
        n2 = int(np.asarray(pcd2.points).shape[0])
        return pcd2, n, n2

    src_use, src_n0, src_n1 = _cap_points(src_in, input_max_pts)
    tgt_use, tgt_n0, tgt_n1 = _cap_points(tgt_in, input_max_pts)
    if src_n1 < 200 or tgt_n1 < 200:
        raise RuntimeError(
            f"OpenGR input too small after downsample: src={src_n1}, tgt={tgt_n1}"
        )

    with tempfile.TemporaryDirectory(prefix="opengr_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        src_path = tmp_root / "source.ply"
        tgt_path = tmp_root / "target.ply"
        matrix_path = tmp_root / "transform.txt"

        _write_xyz_ply_ascii_float(src_path, np.asarray(src_use.points, dtype=np.float64))
        _write_xyz_ply_ascii_float(tgt_path, np.asarray(tgt_use.points, dtype=np.float64))

        cmd = [
            bin_cmd,
            "-i",
            str(src_path),
            str(tgt_path),
            "-o",
            f"{overlap_eff:.6f}",
            "-d",
            f"{delta_u:.9f}",
            "-n",
            str(sample_eff),
            "-t",
            str(max_time_eff),
            "-m",
            str(matrix_path),
        ]
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout_eff),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"OpenGR Super4PCS timeout: timeout_s={timeout_eff}, "
                f"cmd={' '.join(cmd[:6])}..."
            ) from exc
        std_out = str(cp.stdout or "")
        std_err = str(cp.stderr or "")

        if cp.returncode != 0:
            raise RuntimeError(
                "OpenGR Super4PCS failed: "
                f"returncode={cp.returncode}, stderr_tail={std_err[-600:]}"
            )

        if matrix_path.exists():
            matrix_text = matrix_path.read_text(encoding="utf-8", errors="ignore")
        else:
            matrix_text = std_out

        T = _parse_transform_4x4_from_text(matrix_text)
        if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
            raise RuntimeError("invalid transformation from OpenGR")

    meta = {
        "status": "ok",
        "backend": "opengr",
        "opengr_bin": str(bin_cmd),
        "opengr_overlap": float(overlap_eff),
        "opengr_delta_u": float(delta_u),
        "opengr_sample_points": int(sample_eff),
        "opengr_max_time_s": int(max_time_eff),
        "opengr_timeout_s": int(timeout_eff),
        "opengr_input_voxel_factor": float(input_voxel_factor),
        "opengr_input_max_points": int(input_max_pts),
        "opengr_input_src_points_before_cap": int(src_n0),
        "opengr_input_src_points_used": int(src_n1),
        "opengr_input_tgt_points_before_cap": int(tgt_n0),
        "opengr_input_tgt_points_used": int(tgt_n1),
        "opengr_stdout_tail": std_out[-600:],
        "opengr_stderr_tail": std_err[-600:],
        "elapsed_s": float(time.time() - t0),
    }
    return T, meta


def _register_global_init_by_backend(
    *,
    source,
    target,
    voxel_u: float,
    backend: str,
    allow_fallback: bool,
    ransac_max_iter: int,
    ransac_confidence: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    requested = str(backend).strip().lower()
    alias = {
        "teaser++": "teaserpp",
        "super4pcs": "opengr",
    }
    requested = alias.get(requested, requested)
    if requested in ("", "default"):
        requested = "ransac"

    effective = requested
    fallback_reason = ""

    def _run_backend(name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if name == "fgr":
            return _register_fpfh_fgr_global_only(
                source=source,
                target=target,
                voxel_u=float(voxel_u),
            )
        if name == "ransac":
            return _register_fpfh_ransac_global_only(
                source=source,
                target=target,
                voxel_u=float(voxel_u),
                ransac_max_iter=int(ransac_max_iter),
                ransac_confidence=float(ransac_confidence),
            )
        if name == "teaserpp":
            return _register_teaserpp_global_only(
                source=source,
                target=target,
                voxel_u=float(voxel_u),
                mutual_filter=_env_bool("DAIMA_TEASER_MUTUAL_FILTER", True),
                corr_max_pair_dist_u=_env_float("DAIMA_TEASER_CORR_MAX_PAIR_DIST_U", float(voxel_u) * 3.0),
                noise_bound_u=_env_float("DAIMA_TEASER_NOISE_BOUND_U", float(voxel_u) * 1.5),
                cbar2=_env_float("DAIMA_TEASER_CBAR2", 1.0),
                estimate_scaling=_env_bool("DAIMA_TEASER_ESTIMATE_SCALING", False),
            )
        if name == "opengr":
            opengr_max_time_s = _env_int("DAIMA_OPENGR_MAX_TIME_S", 90)
            return _register_opengr_super4pcs_global_only(
                source=source,
                target=target,
                voxel_u=float(voxel_u),
                bin_path=_env_str("DAIMA_OPENGR_BIN", "Super4PCS"),
                overlap=_env_float("DAIMA_OPENGR_OVERLAP", 0.6),
                max_time_s=opengr_max_time_s,
                sample_points=_env_int("DAIMA_OPENGR_SAMPLE_POINTS", 1000),
                delta_factor=_env_float("DAIMA_OPENGR_DELTA_FACTOR", 1.5),
                max_input_points=_env_int("DAIMA_OPENGR_MAX_INPUT_POINTS", 30000),
                timeout_s=_env_int("DAIMA_OPENGR_TIMEOUT_S", int(max(opengr_max_time_s + 10, 30))),
                input_voxel_factor=_env_float("DAIMA_OPENGR_INPUT_VOXEL_FACTOR", 1.5),
            )
        raise RuntimeError(f"unknown init backend: {name}")

    try:
        T, meta = _run_backend(effective)
    except Exception as exc:
        if not allow_fallback or effective == "ransac":
            raise
        effective = "ransac"
        fallback_reason = f"backend_{requested}_failed_fallback_to_ransac: {exc}"
        T, meta = _run_backend(effective)

    out = dict(meta)
    out["backend_requested"] = str(requested)
    out["backend_effective"] = str(effective)
    out["fallback_reason"] = str(fallback_reason)
    return T, out


def _build_robust_loss(kernel: str, k_u: float):
    kk = float(max(float(k_u), 1e-12))
    name = str(kernel).strip().lower()
    if name in ("", "none", "off"):
        return None
    if name == "cauchy":
        return o3d.pipelines.registration.CauchyLoss(k=kk)
    if name == "tukey":
        return o3d.pipelines.registration.TukeyLoss(k=kk)
    raise RuntimeError(f"unknown robust kernel: {kernel}")


def _refine_icp_point_to_plane_multiscale(
    *,
    source,
    target,
    init_T: np.ndarray,
    voxel_scales_u: List[float],
    max_iters: List[int],
    max_corr_factor: float,
    robust_enable: bool,
    robust_kernel: str,
    robust_k_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    T = np.asarray(init_T, dtype=np.float64)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid init transformation for multiscale ICP")

    scales = [float(v) for v in voxel_scales_u if np.isfinite(float(v)) and float(v) > 1e-12]
    if not scales:
        raise RuntimeError("empty voxel scales for multiscale ICP")

    iters = [int(v) for v in max_iters if int(v) > 0]
    if not iters:
        raise RuntimeError("empty max_iters for multiscale ICP")
    if len(iters) < len(scales):
        iters = iters + [iters[-1]] * (len(scales) - len(iters))
    elif len(iters) > len(scales):
        iters = iters[: len(scales)]

    stage_meta: List[Dict[str, Any]] = []
    corr_factor = float(max(float(max_corr_factor), 1e-3))
    use_robust = bool(robust_enable)
    loss = _build_robust_loss(robust_kernel, float(robust_k_u)) if use_robust else None

    for idx, (voxel_u, max_iter) in enumerate(zip(scales, iters), start=1):
        src_d = source.voxel_down_sample(float(voxel_u))
        tgt_d = target.voxel_down_sample(float(voxel_u))
        if len(src_d.points) < 30 or len(tgt_d.points) < 30:
            stage_meta.append(
                {
                    "stage": int(idx),
                    "status": "skipped_insufficient_points",
                    "voxel_u": float(voxel_u),
                    "max_iter": int(max_iter),
                    "n_src_down": int(len(src_d.points)),
                    "n_tgt_down": int(len(tgt_d.points)),
                }
            )
            continue

        normal_radius = float(voxel_u) * 2.0
        src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

        max_corr = float(voxel_u) * corr_factor
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
        if loss is None:
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        res = o3d.pipelines.registration.registration_icp(
            src_d,
            tgt_d,
            max_corr,
            T,
            est,
            criteria,
        )
        T = np.asarray(res.transformation, dtype=np.float64)
        if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
            raise RuntimeError(f"invalid transformation from multiscale ICP stage{idx}")

        stage_meta.append(
            {
                "stage": int(idx),
                "status": "ok",
                "voxel_u": float(voxel_u),
                "max_corr_u": float(max_corr),
                "max_iter": int(max_iter),
                "n_src_down": int(len(src_d.points)),
                "n_tgt_down": int(len(tgt_d.points)),
                "fitness": float(getattr(res, "fitness", float("nan"))),
                "inlier_rmse_u": float(getattr(res, "inlier_rmse", float("nan"))),
            }
        )

    ok_stages = [x for x in stage_meta if str(x.get("status", "")) == "ok"]
    if not ok_stages:
        raise RuntimeError("multiscale ICP has no valid stage")

    last = ok_stages[-1]
    return T, {
        "status": "ok",
        "kernel": "none" if loss is None else str(robust_kernel).strip().lower(),
        "kernel_k_u": float(max(float(robust_k_u), 1e-12)),
        "max_corr_factor": float(corr_factor),
        "stages": stage_meta,
        "elapsed_s": float(time.time() - t0),
        "final_fitness": float(last.get("fitness", float("nan"))),
        "final_inlier_rmse_u": float(last.get("inlier_rmse_u", float("nan"))),
    }


def _refine_icp_point_to_plane_robust(
    *,
    source,
    target,
    init_T: np.ndarray,
    voxel_u: float,
    robust_k_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    loss = o3d.pipelines.registration.CauchyLoss(k=float(max(robust_k_u, 1e-12)))
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * 0.75,
        np.asarray(init_T, dtype=float),
        est,
    )

    meta = {
        "status": "ok",
        "kernel": "cauchy",
        "kernel_k_u": float(max(robust_k_u, 1e-12)),
        "icp_fitness": float(getattr(res_icp, "fitness", float("nan"))),
        "icp_inlier_rmse_u": float(getattr(res_icp, "inlier_rmse", float("nan"))),
        "elapsed_s": float(time.time() - t0),
    }
    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        meta["status"] = "failed"
        raise RuntimeError("invalid transformation from robust ICP refine")
    return T, meta


def _refine_gicp(
    *,
    source,
    target,
    init_T: np.ndarray,
    voxel_u: float,
    robust_k_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    loss = o3d.pipelines.registration.CauchyLoss(k=float(max(robust_k_u, 1e-12)))
    est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    res = o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        float(voxel_u) * 0.9,
        np.asarray(init_T, dtype=float),
        est,
        criteria,
    )
    T = np.asarray(res.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid transformation from GICP refine")
    return T, {
        "status": "ok",
        "kernel": "cauchy",
        "kernel_k_u": float(max(robust_k_u, 1e-12)),
        "gicp_fitness": float(getattr(res, "fitness", float("nan"))),
        "gicp_inlier_rmse_u": float(getattr(res, "inlier_rmse", float("nan"))),
        "elapsed_s": float(time.time() - t0),
    }


def _refine_icp_point_to_plane(
    *,
    source,
    target,
    init_T: np.ndarray,
    voxel_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * 0.75,
        np.asarray(init_T, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    meta = {
        "status": "ok",
        "icp_fitness": float(getattr(res_icp, "fitness", float("nan"))),
        "icp_inlier_rmse_u": float(getattr(res_icp, "inlier_rmse", float("nan"))),
        "elapsed_s": float(time.time() - t0),
    }
    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        meta["status"] = "failed"
        raise RuntimeError("invalid transformation from ICP refine")
    return T, meta


def _refine_colored_icp_multiscale(
    *,
    source,
    target,
    init_T: np.ndarray,
    voxel_scales_u: List[float],
    max_iters: List[int],
    lambda_geometric: float,
    max_corr_factor: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    T = np.asarray(init_T, dtype=np.float64)
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid init transformation for colored ICP")

    scales = [float(v) for v in voxel_scales_u if np.isfinite(float(v)) and float(v) > 1e-12]
    if not scales:
        raise RuntimeError("empty voxel scales for colored ICP")

    iters = [int(v) for v in max_iters if int(v) > 0]
    if not iters:
        raise RuntimeError("empty max_iters for colored ICP")
    if len(iters) < len(scales):
        iters = iters + [iters[-1]] * (len(scales) - len(iters))
    elif len(iters) > len(scales):
        iters = iters[: len(scales)]

    stage_meta: List[Dict[str, Any]] = []
    l_geo = float(np.clip(lambda_geometric, 1e-6, 1.0 - 1e-6))
    corr_factor = float(max(float(max_corr_factor), 1e-3))

    for idx, (voxel_u, max_iter) in enumerate(zip(scales, iters), start=1):
        src_d = source.voxel_down_sample(float(voxel_u))
        tgt_d = target.voxel_down_sample(float(voxel_u))
        if len(src_d.points) < 50 or len(tgt_d.points) < 50:
            raise RuntimeError(
                f"colored ICP stage{idx}: insufficient downsampled points src={len(src_d.points)} tgt={len(tgt_d.points)}"
            )

        normal_radius = float(voxel_u) * 2.0
        src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

        max_corr = float(voxel_u) * corr_factor
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
        est = o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=l_geo)

        res = o3d.pipelines.registration.registration_colored_icp(
            src_d,
            tgt_d,
            max_corr,
            T,
            est,
            criteria,
        )
        T = np.asarray(res.transformation, dtype=np.float64)
        if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
            raise RuntimeError(f"invalid transformation from colored ICP stage{idx}")

        stage_meta.append(
            {
                "stage": int(idx),
                "voxel_u": float(voxel_u),
                "max_corr_u": float(max_corr),
                "max_iter": int(max_iter),
                "lambda_geometric": float(l_geo),
                "n_src_down": int(len(src_d.points)),
                "n_tgt_down": int(len(tgt_d.points)),
                "fitness": float(getattr(res, "fitness", float("nan"))),
                "inlier_rmse_u": float(getattr(res, "inlier_rmse", float("nan"))),
            }
        )

    return T, {
        "status": "ok",
        "stages": stage_meta,
        "elapsed_s": float(time.time() - t0),
        "final_fitness": float(stage_meta[-1].get("fitness", float("nan"))) if stage_meta else float("nan"),
        "final_inlier_rmse_u": float(stage_meta[-1].get("inlier_rmse_u", float("nan"))) if stage_meta else float("nan"),
        "lambda_geometric": float(l_geo),
        "max_corr_factor": float(corr_factor),
    }


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape((-1, 3))
    if p.size == 0:
        return p.reshape((0, 3))
    ph = np.hstack([p, np.ones((len(p), 1), dtype=np.float64)])
    return (np.asarray(T, dtype=np.float64) @ ph.T).T[:, :3]


def _copy_point_cloud(pcd):
    return copy.deepcopy(pcd)


def _build_overall_detail_cloud(overall_pcd, base_pcd, exclude_radius_u: float):
    overall_pts = np.asarray(overall_pcd.points, dtype=np.float64)
    if overall_pts.shape[0] == 0:
        return _points_to_pcd(np.zeros((0, 3), dtype=np.float64))
    if float(exclude_radius_u) <= 1e-12 or len(base_pcd.points) == 0:
        return _copy_point_cloud(overall_pcd)

    tree = o3d.geometry.KDTreeFlann(base_pcd)
    keep_idx: List[int] = []
    radius = float(max(exclude_radius_u, 1e-12))
    for idx, pt in enumerate(overall_pts):
        k, _, dist2 = tree.search_knn_vector_3d(pt, 1)
        if int(k) < 1:
            keep_idx.append(int(idx))
            continue
        nearest_u = float(np.sqrt(max(float(dist2[0]), 0.0)))
        if nearest_u > radius:
            keep_idx.append(int(idx))
    if not keep_idx:
        return _points_to_pcd(np.zeros((0, 3), dtype=np.float64))
    return overall_pcd.select_by_index(keep_idx)


def _robust_weights_from_residuals(residual_u: np.ndarray, *, kernel: str, k_u: float) -> np.ndarray:
    r = np.abs(np.asarray(residual_u, dtype=np.float64).reshape(-1))
    if r.size == 0:
        return np.zeros((0,), dtype=np.float64)
    kk = float(max(float(k_u), 1e-12))
    name = str(kernel).strip().lower()
    if name in ("", "none", "off"):
        return np.ones_like(r, dtype=np.float64)
    x = r / kk
    if name == "cauchy":
        return 1.0 / (1.0 + x * x)
    if name == "tukey":
        w = np.zeros_like(r, dtype=np.float64)
        keep = x < 1.0
        z = 1.0 - np.square(x[keep])
        w[keep] = np.square(z)
        return w
    return np.ones_like(r, dtype=np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(w)
    k = w / theta
    K = _skew(k)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _so3_log(R: np.ndarray) -> np.ndarray:
    mat = np.asarray(R, dtype=np.float64).reshape((3, 3))
    cos_theta = float(np.clip((np.trace(mat) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta <= 1e-9:
        return np.asarray(
            [
                mat[2, 1] - mat[1, 2],
                mat[0, 2] - mat[2, 0],
                mat[1, 0] - mat[0, 1],
            ],
            dtype=np.float64,
        ) * 0.5
    if np.pi - theta <= 1e-5:
        axis = np.sqrt(np.maximum((np.diag(mat) + 1.0) * 0.5, 0.0))
        axis = np.asarray(axis, dtype=np.float64)
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis[1] = np.copysign(axis[1], mat[0, 1] + mat[1, 0])
            axis[2] = np.copysign(axis[2], mat[0, 2] + mat[2, 0])
        elif axis[1] >= axis[0] and axis[1] >= axis[2]:
            axis[0] = np.copysign(axis[0], mat[0, 1] + mat[1, 0])
            axis[2] = np.copysign(axis[2], mat[1, 2] + mat[2, 1])
        else:
            axis[0] = np.copysign(axis[0], mat[0, 2] + mat[2, 0])
            axis[1] = np.copysign(axis[1], mat[1, 2] + mat[2, 1])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return np.zeros((3,), dtype=np.float64)
        return axis / axis_norm * theta
    w_hat = (mat - mat.T) * (0.5 / np.sin(theta))
    return np.asarray([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64) * theta


def _se3_exp(delta6: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta6, dtype=np.float64).reshape(6)
    omega = delta[:3]
    upsilon = delta[3:]
    theta = float(np.linalg.norm(omega))
    Omega = _skew(omega)
    R = _so3_exp(omega)
    if theta <= 1e-12:
        V = np.eye(3, dtype=np.float64) + 0.5 * Omega + (1.0 / 6.0) * (Omega @ Omega)
    else:
        theta2 = theta * theta
        V = (
            np.eye(3, dtype=np.float64)
            + ((1.0 - np.cos(theta)) / theta2) * Omega
            + ((theta - np.sin(theta)) / (theta2 * theta)) * (Omega @ Omega)
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ upsilon
    return T


def _se3_log(T: np.ndarray) -> np.ndarray:
    mat = np.asarray(T, dtype=np.float64).reshape((4, 4))
    omega = _so3_log(mat[:3, :3])
    theta = float(np.linalg.norm(omega))
    Omega = _skew(omega)
    if theta <= 1e-12:
        V_inv = np.eye(3, dtype=np.float64) - 0.5 * Omega + (1.0 / 12.0) * (Omega @ Omega)
    else:
        theta2 = theta * theta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        beta = (1.0 / theta2) - ((1.0 + cos_theta) / (2.0 * theta * max(sin_theta, 1e-12)))
        V_inv = np.eye(3, dtype=np.float64) - 0.5 * Omega + beta * (Omega @ Omega)
    upsilon = V_inv @ np.asarray(mat[:3, 3], dtype=np.float64).reshape(3)
    return np.hstack([omega, upsilon]).astype(np.float64)


def _rotation_angle_deg(R: np.ndarray) -> float:
    mat = np.asarray(R, dtype=np.float64).reshape((3, 3))
    cos_theta = float(np.clip((np.trace(mat) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _pose_delta_mm_deg(T_from: np.ndarray, T_to: np.ndarray, unit_scale: float) -> Dict[str, float]:
    rel = np.linalg.inv(np.asarray(T_from, dtype=np.float64)) @ np.asarray(T_to, dtype=np.float64)
    trans_u = float(np.linalg.norm(rel[:3, 3]))
    rot_deg = _rotation_angle_deg(rel[:3, :3])
    return {
        "translation_u": float(trans_u),
        "translation_mm": _mm_from_u(float(trans_u), unit_scale),
        "rotation_deg": float(rot_deg),
    }


def _compute_base_information_matrix(source_base, target_base, T_anchor: np.ndarray, max_corr_u: float) -> np.ndarray:
    try:
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_base,
            target_base,
            float(max(max_corr_u, 1e-12)),
            np.asarray(T_anchor, dtype=np.float64),
        )
        info_np = np.asarray(info, dtype=np.float64).reshape((6, 6))
        if np.all(np.isfinite(info_np)):
            return 0.5 * (info_np + info_np.T)
    except Exception:
        pass

    corres = _build_correspondences_knn2(
        source_pcd=source_base,
        target_pcd=target_base,
        T=np.asarray(T_anchor, dtype=np.float64),
        max_corr_u=float(max_corr_u),
    )
    H, _, _ = _accumulate_point_to_plane_system(
        corres=corres,
        weights=np.ones((int(corres.get("count", 0)),), dtype=np.float64),
        T=np.asarray(T_anchor, dtype=np.float64),
    )
    if np.all(np.isfinite(H)) and float(np.trace(H)) > 0.0:
        return 0.5 * (H + H.T)
    return np.eye(6, dtype=np.float64)


def _normalize_information_matrix(info6x6: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    info = 0.5 * (np.asarray(info6x6, dtype=np.float64).reshape((6, 6)) + np.asarray(info6x6, dtype=np.float64).reshape((6, 6)).T)
    if not np.all(np.isfinite(info)):
        return np.eye(6, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(info)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    max_eig = float(max(np.max(eigvals), float(eps)))
    min_eig = float(max(max_eig * 0.02, float(eps)))
    clipped = np.clip(eigvals, min_eig, max_eig) / max_eig
    out = eigvecs @ np.diag(clipped) @ eigvecs.T
    return 0.5 * (out + out.T)


def _weak_mode_projector(info6x6: np.ndarray, weak_mode_count: int = 2, eps: float = 1e-9) -> Tuple[np.ndarray, List[float]]:
    info = 0.5 * (np.asarray(info6x6, dtype=np.float64).reshape((6, 6)) + np.asarray(info6x6, dtype=np.float64).reshape((6, 6)).T)
    eigvals, eigvecs = np.linalg.eigh(info)
    eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), float(eps), None)
    weak_n = int(np.clip(int(weak_mode_count), 1, 6))
    weak_basis = np.asarray(eigvecs[:, :weak_n], dtype=np.float64)
    projector = weak_basis @ weak_basis.T
    projector = 0.5 * (projector + projector.T)
    eig_desc = [float(v) for v in eigvals[::-1]]
    return projector, eig_desc


def _rotation_about_pivot(axis: np.ndarray, angle_rad: float, pivot_u: np.ndarray) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis_arr))
    if axis_norm <= 1e-12 or (not np.isfinite(axis_norm)):
        return np.eye(4, dtype=np.float64)
    rot_vec = (axis_arr / axis_norm) * float(angle_rad)
    R = _so3_exp(rot_vec)
    pivot = np.asarray(pivot_u, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pivot - (R @ pivot)
    return T


def _extract_bcjr_rescue_rotation_axes(info6x6: np.ndarray, max_axes: int = 2) -> List[Dict[str, Any]]:
    info = 0.5 * (np.asarray(info6x6, dtype=np.float64).reshape((6, 6)) + np.asarray(info6x6, dtype=np.float64).reshape((6, 6)).T)
    eigvals, eigvecs = np.linalg.eigh(info)
    order = np.argsort(np.asarray(eigvals, dtype=np.float64))
    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order.tolist(), start=1):
        vec6 = np.asarray(eigvecs[:, int(idx)], dtype=np.float64).reshape(6)
        rot = np.asarray(vec6[:3], dtype=np.float64)
        rot_norm = float(np.linalg.norm(rot))
        if rot_norm <= 1e-9 or (not np.isfinite(rot_norm)):
            continue
        axis = rot / rot_norm
        duplicate = False
        for item in out:
            prev = np.asarray(item.get("axis", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
            if float(abs(np.dot(axis, prev))) >= 0.98:
                duplicate = True
                break
        if duplicate:
            continue
        out.append(
            {
                "rank": int(rank),
                "eig_index": int(idx),
                "eig_value": float(eigvals[int(idx)]),
                "axis": axis.astype(np.float64),
                "axis_norm": float(rot_norm),
                "mode_vector": vec6.astype(np.float64),
            }
        )
        if len(out) >= int(max(max_axes, 0)):
            break
    return out


def _compute_verify_residual_stats(
    *,
    source_pcd,
    target_pcd,
    T: np.ndarray,
    max_corr_u: float,
    robust_kernel: str,
    robust_k_u: float,
    unit_scale: float,
) -> Dict[str, Any]:
    corres = _build_correspondences_knn2(
        source_pcd=source_pcd,
        target_pcd=target_pcd,
        T=np.asarray(T, dtype=np.float64),
        max_corr_u=float(max_corr_u),
    )
    residual_u = _compute_correspondence_point_to_plane_residuals(corres)
    count = int(corres.get("count", 0))
    if count <= 0 or residual_u.size == 0:
        return {
            "count": int(count),
            "rmse_u": float("inf"),
            "rmse_mm": float("inf"),
            "residual_p95_u": float("inf"),
            "residual_p95_mm": float("inf"),
            "weight_sum": 0.0,
        }

    w = _robust_weights_from_residuals(
        residual_u,
        kernel=str(robust_kernel),
        k_u=float(robust_k_u),
    )
    weight_sum = float(np.sum(w))
    if weight_sum <= 1e-12 or (not np.isfinite(weight_sum)):
        rmse_u = float("inf")
    else:
        rmse_u = float(np.sqrt(np.sum(w * np.square(residual_u)) / max(weight_sum, 1e-12)))

    residual_abs = np.abs(np.asarray(residual_u, dtype=np.float64))
    p95_u = float(np.quantile(residual_abs, 0.95)) if residual_abs.size > 0 else float("inf")
    return {
        "count": int(count),
        "rmse_u": float(rmse_u),
        "rmse_mm": _mm_from_u(float(rmse_u), unit_scale) if np.isfinite(rmse_u) else float("inf"),
        "residual_p95_u": float(p95_u),
        "residual_p95_mm": _mm_from_u(float(p95_u), unit_scale) if np.isfinite(p95_u) else float("inf"),
        "weight_sum": float(weight_sum),
    }


def _verify_bcjr_candidate(
    *,
    source_base,
    target_base,
    source_detail,
    target_detail,
    T: np.ndarray,
    cfg: RegistrationConfig,
) -> Dict[str, Any]:
    voxel_mm = float(max(float(cfg.bcjr_rescue_verify_voxel_mm), 1e-6))
    voxel_u = _u_from_mm(voxel_mm, cfg.unit_scale)
    max_corr_u = float(max(float(cfg.bcjr_rescue_verify_max_corr_factor) * float(voxel_u), 1e-9))
    normal_radius = float(voxel_u) * 2.0
    robust_kernel = str(cfg.local_refine_robust_kernel) if bool(cfg.local_refine_robust_enable) else "none"
    robust_k_u = _u_from_mm(cfg.local_refine_robust_k_mm, cfg.unit_scale)

    src_base_ds = source_base.voxel_down_sample(float(voxel_u))
    tgt_base_ds = target_base.voxel_down_sample(float(voxel_u))
    src_detail_ds = source_detail.voxel_down_sample(float(voxel_u))
    tgt_detail_ds = target_detail.voxel_down_sample(float(voxel_u))

    if len(src_base_ds.points) >= 6 and len(tgt_base_ds.points) >= 6:
        _ensure_normals(src_base_ds, radius_u=normal_radius)
        _ensure_normals(tgt_base_ds, radius_u=normal_radius)
        base_stats = _compute_verify_residual_stats(
            source_pcd=src_base_ds,
            target_pcd=tgt_base_ds,
            T=np.asarray(T, dtype=np.float64),
            max_corr_u=float(max_corr_u),
            robust_kernel=robust_kernel,
            robust_k_u=float(robust_k_u),
            unit_scale=float(cfg.unit_scale),
        )
    else:
        base_stats = {
            "count": 0,
            "rmse_u": float("inf"),
            "rmse_mm": float("inf"),
            "residual_p95_u": float("inf"),
            "residual_p95_mm": float("inf"),
            "weight_sum": 0.0,
        }

    if len(src_detail_ds.points) >= 6 and len(tgt_detail_ds.points) >= 6:
        _ensure_normals(src_detail_ds, radius_u=normal_radius)
        _ensure_normals(tgt_detail_ds, radius_u=normal_radius)
        detail_stats = _compute_verify_residual_stats(
            source_pcd=src_detail_ds,
            target_pcd=tgt_detail_ds,
            T=np.asarray(T, dtype=np.float64),
            max_corr_u=float(max_corr_u),
            robust_kernel=robust_kernel,
            robust_k_u=float(robust_k_u),
            unit_scale=float(cfg.unit_scale),
        )
    else:
        detail_stats = {
            "count": 0,
            "rmse_u": float("inf"),
            "rmse_mm": float("inf"),
            "residual_p95_u": float("inf"),
            "residual_p95_mm": float("inf"),
            "weight_sum": 0.0,
        }

    return {
        "voxel_u": float(voxel_u),
        "voxel_mm": float(voxel_mm),
        "max_corr_u": float(max_corr_u),
        "max_corr_mm": _mm_from_u(float(max_corr_u), cfg.unit_scale),
        "detail_verify_corr_count": int(detail_stats.get("count", 0)),
        "detail_verify_rmse_u": float(detail_stats.get("rmse_u", float("inf"))),
        "detail_verify_rmse_mm": float(detail_stats.get("rmse_mm", float("inf"))),
        "detail_verify_residual_p95_mm": float(detail_stats.get("residual_p95_mm", float("inf"))),
        "base_verify_corr_count": int(base_stats.get("count", 0)),
        "base_verify_rmse_u": float(base_stats.get("rmse_u", float("inf"))),
        "base_verify_rmse_mm": float(base_stats.get("rmse_mm", float("inf"))),
        "base_verify_residual_p95_mm": float(base_stats.get("residual_p95_mm", float("inf"))),
    }


def _select_bcjr_candidate(candidates: List[Dict[str, Any]], cfg: RegistrationConfig) -> Dict[str, Any]:
    valid = []
    for cand in candidates:
        verify = dict(cand.get("verify", {}))
        detail_rmse_mm = float(verify.get("detail_verify_rmse_mm", float("inf")))
        base_rmse_mm = float(verify.get("base_verify_rmse_mm", float("inf")))
        if np.isfinite(detail_rmse_mm) and np.isfinite(base_rmse_mm):
            valid.append(cand)

    if not valid:
        return dict(candidates[0]) if candidates else {}

    best_base = min(float(c.get("verify", {}).get("base_verify_rmse_mm", float("inf"))) for c in valid)
    base_guard_mm = float(max(float(cfg.bcjr_rescue_verify_base_guard_mm), 0.0))
    guarded = [
        c
        for c in valid
        if float(c.get("verify", {}).get("base_verify_rmse_mm", float("inf"))) <= float(best_base) + float(base_guard_mm)
    ]
    pool = guarded if guarded else valid

    best_detail = min(float(c.get("verify", {}).get("detail_verify_rmse_mm", float("inf"))) for c in pool)
    tie_mm = float(max(float(cfg.bcjr_rescue_detail_rmse_tie_mm), 0.0))
    close = [
        c
        for c in pool
        if float(c.get("verify", {}).get("detail_verify_rmse_mm", float("inf"))) <= float(best_detail) + float(tie_mm)
    ]
    pool2 = close if close else pool

    def _rank_key(cand: Dict[str, Any]) -> Tuple[float, float, float, int]:
        verify = dict(cand.get("verify", {}))
        detail_rmse_mm = float(verify.get("detail_verify_rmse_mm", float("inf")))
        detail_count = int(verify.get("detail_verify_corr_count", 0))
        base_rmse_mm = float(verify.get("base_verify_rmse_mm", float("inf")))
        candidate_index = int(cand.get("candidate_index", 999))
        return (
            -float(detail_count),
            float(detail_rmse_mm),
            float(base_rmse_mm),
            int(candidate_index),
        )

    return dict(sorted(pool2, key=_rank_key)[0])


def _empty_correspondence_dict() -> Dict[str, Any]:
    empty_f = np.zeros((0,), dtype=np.float64)
    empty_p = np.zeros((0, 3), dtype=np.float64)
    empty_i = np.zeros((0,), dtype=np.int64)
    return {
        "count": 0,
        "source_indices": empty_i,
        "target_indices": empty_i,
        "source_points": empty_p,
        "source_points_transformed": empty_p,
        "source_normals_rot": empty_p,
        "target_points": empty_p,
        "target_normals": empty_p,
        "d1_u": empty_f,
        "d2_u": empty_f,
    }


def _build_correspondences_knn2(source_pcd, target_pcd, T: np.ndarray, max_corr_u: float) -> Dict[str, Any]:
    src_pts = np.asarray(source_pcd.points, dtype=np.float64)
    tgt_pts = np.asarray(target_pcd.points, dtype=np.float64)
    if src_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
        return _empty_correspondence_dict()

    radius_u = float(max(float(max_corr_u) * 2.0, 1e-6))
    _ensure_normals(source_pcd, radius_u=radius_u)
    _ensure_normals(target_pcd, radius_u=radius_u)

    src_normals = np.asarray(source_pcd.normals, dtype=np.float64)
    tgt_normals = np.asarray(target_pcd.normals, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64).reshape((4, 4))
    src_trans = _transform_points(T_arr, src_pts)
    R = T_arr[:3, :3]
    src_normals_rot = (R @ src_normals.T).T if src_normals.shape == src_pts.shape else np.zeros_like(src_pts)

    tree = o3d.geometry.KDTreeFlann(target_pcd)
    src_ids: List[int] = []
    tgt_ids: List[int] = []
    d1_vals: List[float] = []
    d2_vals: List[float] = []
    for idx, pt in enumerate(src_trans):
        k, nn_idx, nn_dist2 = tree.search_knn_vector_3d(pt, 2)
        if int(k) < 1:
            continue
        d1_u = float(np.sqrt(max(float(nn_dist2[0]), 0.0)))
        if d1_u > float(max_corr_u):
            continue
        d2_u = float(np.sqrt(max(float(nn_dist2[1]), 0.0))) if int(k) >= 2 else d1_u
        src_ids.append(int(idx))
        tgt_ids.append(int(nn_idx[0]))
        d1_vals.append(float(d1_u))
        d2_vals.append(float(d2_u))

    if not src_ids:
        return _empty_correspondence_dict()

    src_idx_arr = np.asarray(src_ids, dtype=np.int64)
    tgt_idx_arr = np.asarray(tgt_ids, dtype=np.int64)
    return {
        "count": int(len(src_ids)),
        "source_indices": src_idx_arr,
        "target_indices": tgt_idx_arr,
        "source_points": src_pts[src_idx_arr],
        "source_points_transformed": src_trans[src_idx_arr],
        "source_normals_rot": src_normals_rot[src_idx_arr],
        "target_points": tgt_pts[tgt_idx_arr],
        "target_normals": tgt_normals[tgt_idx_arr],
        "d1_u": np.asarray(d1_vals, dtype=np.float64),
        "d2_u": np.asarray(d2_vals, dtype=np.float64),
    }


def _compute_correspondence_point_to_plane_residuals(corres: Dict[str, Any]) -> np.ndarray:
    src_t = np.asarray(corres.get("source_points_transformed", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    tgt = np.asarray(corres.get("target_points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    normals = np.asarray(corres.get("target_normals", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    if src_t.shape[0] == 0 or tgt.shape[0] == 0 or normals.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.sum(normals * (src_t - tgt), axis=1)


def _accumulate_point_to_plane_system(corres: Dict[str, Any], weights: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    del T
    src_t = np.asarray(corres.get("source_points_transformed", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    tgt = np.asarray(corres.get("target_points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    normals = np.asarray(corres.get("target_normals", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
    n = int(src_t.shape[0])
    H = np.zeros((6, 6), dtype=np.float64)
    g = np.zeros((6,), dtype=np.float64)
    if n == 0:
        return H, g, {"count": 0, "weight_sum": 0.0, "cost": 0.0}

    residuals = np.sum(normals * (src_t - tgt), axis=1)
    J_rot = np.cross(src_t, normals)
    J = np.hstack([J_rot, normals])
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        w = np.ones((n,), dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    H = J.T @ (w[:, None] * J)
    g = J.T @ (w * residuals)
    return H, g, {
        "count": int(n),
        "weight_sum": float(np.sum(w)),
        "cost": float(np.sum(w * np.square(residuals))),
        "mean_abs_residual_u": float(np.mean(np.abs(residuals))) if residuals.size > 0 else float("nan"),
    }


def _normalize_overall_mass(raw_lambda: float, base_weight_sum: float, detail_weight_sum: float) -> float:
    lam = float(max(float(raw_lambda), 0.0))
    if lam <= 0.0 or float(detail_weight_sum) <= 1e-12:
        return 0.0
    ratio = float(max(float(base_weight_sum), 1e-6) / max(float(detail_weight_sum), 1e-6))
    eff = lam * ratio
    return float(np.clip(eff, 0.35 * lam, 2.50 * lam))


def _linearize_bcjr_prior(T: np.ndarray, T_anchor: np.ndarray, info_prior: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    H = np.zeros((6, 6), dtype=np.float64)
    g = np.zeros((6,), dtype=np.float64)
    mu_eff = float(max(float(mu), 0.0))
    if mu_eff <= 0.0:
        return H, g, {"cost": 0.0, "xi_norm": 0.0}
    rel = np.linalg.inv(np.asarray(T_anchor, dtype=np.float64)) @ np.asarray(T, dtype=np.float64)
    xi = _se3_log(rel)
    info = 0.5 * (np.asarray(info_prior, dtype=np.float64).reshape((6, 6)) + np.asarray(info_prior, dtype=np.float64).reshape((6, 6)).T)
    H = mu_eff * info
    g = mu_eff * (info @ xi)
    cost = float(mu_eff * (xi.T @ info @ xi))
    return H, g, {"cost": cost, "xi_norm": float(np.linalg.norm(xi))}


def _clip_bcjr_delta(delta6: np.ndarray, scale_u: float, clip_translation_factor: float, clip_rotation_deg: float) -> np.ndarray:
    delta = np.asarray(delta6, dtype=np.float64).reshape(6).copy()
    rot = delta[:3]
    trans = delta[3:]
    rot_norm = float(np.linalg.norm(rot))
    rot_max = float(np.deg2rad(max(float(clip_rotation_deg), 1e-6)))
    if rot_norm > rot_max:
        rot *= rot_max / rot_norm
    trans_norm = float(np.linalg.norm(trans))
    trans_max = float(max(float(scale_u) * float(max(clip_translation_factor, 1e-6)), 1e-9))
    if trans_norm > trans_max:
        trans *= trans_max / trans_norm
    return np.hstack([rot, trans]).astype(np.float64)


def _compute_bcjr_detail_weights(
    *,
    corres: Dict[str, Any],
    T_anchor: np.ndarray,
    weak_projector: np.ndarray,
    cfg: RegistrationConfig,
    unit_scale: float,
    voxel_u: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(corres.get("count", 0))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), {
            "count": 0,
            "q_mean": float("nan"),
            "q_std": float("nan"),
            "q_min": float("nan"),
            "q_max": float("nan"),
            "active_ratio": float("nan"),
        }

    src = np.asarray(corres["source_points"], dtype=np.float64)
    src_t = np.asarray(corres["source_points_transformed"], dtype=np.float64)
    src_n = np.asarray(corres["source_normals_rot"], dtype=np.float64)
    tgt = np.asarray(corres["target_points"], dtype=np.float64)
    tgt_n = np.asarray(corres["target_normals"], dtype=np.float64)
    d1_u = np.asarray(corres["d1_u"], dtype=np.float64)
    d2_u = np.asarray(corres["d2_u"], dtype=np.float64)

    anchor_pts = _transform_points(np.asarray(T_anchor, dtype=np.float64), src)
    r_anchor_u = np.sum(tgt_n * (anchor_pts - tgt), axis=1)
    r_norm = np.clip(np.abs(r_anchor_u) / max(0.75 * float(voxel_u), 1e-9), 0.0, 2.0)
    norm_cos = np.clip(np.abs(np.sum(src_n * tgt_n, axis=1)), 0.0, 1.0)

    voxel_mm = _mm_from_u(float(voxel_u), unit_scale)
    d1_mm = np.asarray([_mm_from_u(v, unit_scale) for v in d1_u], dtype=np.float64)
    d2_mm = np.asarray([_mm_from_u(v, unit_scale) for v in d2_u], dtype=np.float64)
    ambiguity_gap = np.clip((d2_mm - d1_mm) / max(0.5 * float(voxel_mm), 1e-6), 0.0, 1.0)

    J_rot = np.cross(src_t, tgt_n)
    J = np.hstack([J_rot, tgt_n])
    gain = np.einsum("bi,ij,bj->b", J, np.asarray(weak_projector, dtype=np.float64), J)
    gain = np.clip(gain, 0.0, None)
    positive_gain = gain[gain > 1e-12]
    gain_med = float(np.median(positive_gain)) if positive_gain.size > 0 else 1.0
    gain_norm = np.clip(gain / max(gain_med, 1e-9), 0.0, 2.0)

    score = (
        float(cfg.bcjr_beta_bias)
        - float(cfg.bcjr_beta_r) * r_norm
        + float(cfg.bcjr_beta_n) * norm_cos
        + float(cfg.bcjr_beta_u) * ambiguity_gap
        + float(cfg.bcjr_beta_g) * gain_norm
    )
    score = np.clip(score, -40.0, 40.0)
    sigmoid = 1.0 / (1.0 + np.exp(-score))
    q_min = float(np.clip(float(cfg.bcjr_q_min), 0.0, 0.95))
    q = q_min + (1.0 - q_min) * sigmoid

    return q.astype(np.float64), {
        "count": int(n),
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
        "q_mean": float(np.mean(q)),
        "q_std": float(np.std(q)),
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "active_ratio": float(np.mean(q >= 0.50)),
        "anchor_residual_p50_mm": _mm_from_u(float(np.quantile(np.abs(r_anchor_u), 0.50)), unit_scale),
        "anchor_residual_p95_mm": _mm_from_u(float(np.quantile(np.abs(r_anchor_u), 0.95)), unit_scale),
        "norm_cos_mean": float(np.mean(norm_cos)),
        "ambiguity_gap_mean": float(np.mean(ambiguity_gap)),
        "gain_norm_mean": float(np.mean(gain_norm)),
    }


def _run_base_conditioned_joint_refinement(
    *,
    source_base,
    target_base,
    source_detail,
    target_detail,
    T_anchor: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    T = np.asarray(T_anchor, dtype=np.float64).copy()
    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("invalid anchor transform for BCJR")

    def _fit_float_list(values: List[float], n: int, default: float) -> List[float]:
        seq = [float(v) for v in list(values) if np.isfinite(float(v))]
        if not seq:
            seq = [float(default)]
        if len(seq) < n:
            seq = seq + [float(seq[-1])] * (n - len(seq))
        return seq[:n]

    def _fit_int_list(values: List[int], n: int, default: int) -> List[int]:
        seq = [int(v) for v in list(values) if int(v) > 0]
        if not seq:
            seq = [int(default)]
        if len(seq) < n:
            seq = seq + [int(seq[-1])] * (n - len(seq))
        return seq[:n]

    if not bool(cfg.bcjr_enable):
        return T, {
            "status": "disabled",
            "anchor_transform": np.asarray(T_anchor, dtype=np.float64),
            "elapsed_s": float(time.time() - t0),
            "stages": [],
        }

    scale_mm = [float(v) for v in cfg.bcjr_voxel_scales_mm if np.isfinite(float(v)) and float(v) > 0.0]
    if not scale_mm:
        return T, {
            "status": "anchor_only_no_scales",
            "anchor_transform": np.asarray(T_anchor, dtype=np.float64),
            "elapsed_s": float(time.time() - t0),
            "stages": [],
        }

    scales_u = [_u_from_mm(v, cfg.unit_scale) for v in scale_mm]
    n_scales = len(scales_u)
    max_iters = _fit_int_list(cfg.bcjr_max_iters, n_scales, 10)
    lambda_base = _fit_float_list(cfg.bcjr_lambda_base, n_scales, 1.0)
    lambda_overall = _fit_float_list(cfg.bcjr_lambda_overall, n_scales, 1.0)
    prior_mu = _fit_float_list(cfg.bcjr_prior_mu, n_scales, 0.2)
    clip_rot_deg = _fit_float_list(cfg.bcjr_clip_rotation_deg, n_scales, 0.3)

    robust_kernel = str(cfg.local_refine_robust_kernel) if bool(cfg.local_refine_robust_enable) else "none"
    robust_k_u = _u_from_mm(cfg.local_refine_robust_k_mm, cfg.unit_scale)
    stages: List[Dict[str, Any]] = []
    step_history: List[Dict[str, Any]] = []

    for scale_idx, (voxel_u, voxel_mm, iter_n, lam_b, lam_o, mu_prior, rot_clip_deg) in enumerate(
        zip(scales_u, scale_mm, max_iters, lambda_base, lambda_overall, prior_mu, clip_rot_deg),
        start=1,
    ):
        src_base_ds = source_base.voxel_down_sample(float(voxel_u))
        tgt_base_ds = target_base.voxel_down_sample(float(voxel_u))
        src_det_ds = source_detail.voxel_down_sample(float(voxel_u))
        tgt_det_ds = target_detail.voxel_down_sample(float(voxel_u))

        stage_entry: Dict[str, Any] = {
            "stage": int(scale_idx),
            "status": "ok",
            "voxel_u": float(voxel_u),
            "voxel_mm": float(voxel_mm),
            "max_iter": int(iter_n),
            "lambda_base": float(lam_b),
            "lambda_overall_raw": float(lam_o),
            "prior_mu": float(mu_prior),
            "clip_rotation_deg": float(rot_clip_deg),
            "n_src_base_down": int(len(src_base_ds.points)),
            "n_tgt_base_down": int(len(tgt_base_ds.points)),
            "n_src_detail_down": int(len(src_det_ds.points)),
            "n_tgt_detail_down": int(len(tgt_det_ds.points)),
        }

        if len(src_base_ds.points) < 20 or len(tgt_base_ds.points) < 20:
            stage_entry["status"] = "skipped_insufficient_base_points"
            stage_entry["iterations"] = []
            stages.append(stage_entry)
            continue

        normal_radius = float(voxel_u) * 2.0
        _ensure_normals(src_base_ds, radius_u=normal_radius)
        _ensure_normals(tgt_base_ds, radius_u=normal_radius)
        if len(src_det_ds.points) >= 10 and len(tgt_det_ds.points) >= 10:
            _ensure_normals(src_det_ds, radius_u=normal_radius)
            _ensure_normals(tgt_det_ds, radius_u=normal_radius)

        max_corr_u = float(max(float(cfg.local_refine_max_corr_factor) * float(voxel_u), 1e-9))
        info_b = _compute_base_information_matrix(
            source_base=src_base_ds,
            target_base=tgt_base_ds,
            T_anchor=np.asarray(T_anchor, dtype=np.float64),
            max_corr_u=float(max_corr_u),
        )
        info_prior = _normalize_information_matrix(info_b)
        weak_projector, eigvals_desc = _weak_mode_projector(
            info6x6=info_b,
            weak_mode_count=int(cfg.bcjr_weak_mode_count),
        )
        stage_entry["max_corr_u"] = float(max_corr_u)
        stage_entry["max_corr_mm"] = _mm_from_u(float(max_corr_u), cfg.unit_scale)
        stage_entry["base_info_eigvals_desc"] = [float(v) for v in eigvals_desc]

        iter_logs: List[Dict[str, Any]] = []
        prev_cost: Optional[float] = None
        stage_used = False
        for iter_idx in range(1, int(iter_n) + 1):
            corres_b = _build_correspondences_knn2(
                source_pcd=src_base_ds,
                target_pcd=tgt_base_ds,
                T=T,
                max_corr_u=float(max_corr_u),
            )
            if int(corres_b.get("count", 0)) < 6:
                stage_entry["status"] = "stopped_no_base_correspondences"
                break

            if len(src_det_ds.points) >= 10 and len(tgt_det_ds.points) >= 10:
                corres_d = _build_correspondences_knn2(
                    source_pcd=src_det_ds,
                    target_pcd=tgt_det_ds,
                    T=T,
                    max_corr_u=float(max_corr_u),
                )
            else:
                corres_d = _empty_correspondence_dict()

            r_b = _compute_correspondence_point_to_plane_residuals(corres_b)
            w_b = _robust_weights_from_residuals(
                r_b,
                kernel=robust_kernel,
                k_u=float(robust_k_u),
            )
            q_d, q_stats = _compute_bcjr_detail_weights(
                corres=corres_d,
                T_anchor=np.asarray(T_anchor, dtype=np.float64),
                weak_projector=weak_projector,
                cfg=cfg,
                unit_scale=float(cfg.unit_scale),
                voxel_u=float(voxel_u),
            )
            r_d = _compute_correspondence_point_to_plane_residuals(corres_d)
            w_d_robust = _robust_weights_from_residuals(
                r_d,
                kernel=robust_kernel,
                k_u=float(robust_k_u),
            )
            w_d = q_d * w_d_robust

            H_b, g_b, s_b = _accumulate_point_to_plane_system(corres_b, w_b, T)
            H_d, g_d, s_d = _accumulate_point_to_plane_system(corres_d, w_d, T)
            H_prior, g_prior, s_prior = _linearize_bcjr_prior(
                T=T,
                T_anchor=np.asarray(T_anchor, dtype=np.float64),
                info_prior=info_prior,
                mu=float(mu_prior),
            )
            lambda_o_eff = _normalize_overall_mass(
                raw_lambda=float(lam_o),
                base_weight_sum=float(s_b.get("weight_sum", 0.0)),
                detail_weight_sum=float(s_d.get("weight_sum", 0.0)),
            )

            H = float(lam_b) * H_b + float(lambda_o_eff) * H_d + H_prior
            H = H + np.eye(6, dtype=np.float64) * float(max(cfg.bcjr_lm_damping, 1e-9))
            g = float(lam_b) * g_b + float(lambda_o_eff) * g_d + g_prior

            try:
                delta = -np.linalg.solve(H, g)
            except Exception:
                delta = -np.linalg.lstsq(H, g, rcond=None)[0]
            delta = _clip_bcjr_delta(
                delta6=delta,
                scale_u=float(voxel_u),
                clip_translation_factor=float(cfg.bcjr_clip_translation_factor),
                clip_rotation_deg=float(rot_clip_deg),
            )

            trans_step_u = float(np.linalg.norm(np.asarray(delta[3:], dtype=np.float64)))
            rot_step_deg = float(np.degrees(np.linalg.norm(np.asarray(delta[:3], dtype=np.float64))))
            total_cost = float(lam_b) * float(s_b.get("cost", 0.0)) + float(lambda_o_eff) * float(s_d.get("cost", 0.0)) + float(s_prior.get("cost", 0.0))

            iter_log = {
                "iter": int(iter_idx),
                "base_corr_count": int(s_b.get("count", 0)),
                "detail_corr_count": int(s_d.get("count", 0)),
                "base_weight_sum": float(s_b.get("weight_sum", 0.0)),
                "detail_weight_sum": float(s_d.get("weight_sum", 0.0)),
                "lambda_overall_eff": float(lambda_o_eff),
                "cost": float(total_cost),
                "delta_translation_u": float(trans_step_u),
                "delta_translation_mm": _mm_from_u(float(trans_step_u), cfg.unit_scale),
                "delta_rotation_deg": float(rot_step_deg),
                "detail_q_mean": float(q_stats.get("q_mean", float("nan"))),
                "detail_active_ratio": float(q_stats.get("active_ratio", float("nan"))),
            }
            iter_logs.append(iter_log)
            step_history.append(
                {
                    "stage": int(scale_idx),
                    "iter": int(iter_idx),
                    "translation_mm": _mm_from_u(float(trans_step_u), cfg.unit_scale),
                    "rotation_deg": float(rot_step_deg),
                }
            )

            T = _se3_exp(delta) @ T
            stage_used = True

            trans_tol_u = float(max(0.02 * float(voxel_u), 1e-9))
            rot_tol_deg = float(max(0.05 * float(rot_clip_deg), 0.01))
            cost_rel = abs(float(total_cost) - float(prev_cost)) / max(abs(float(prev_cost)), 1e-9) if prev_cost is not None and np.isfinite(float(prev_cost)) else float("inf")
            prev_cost = float(total_cost)
            if trans_step_u <= trans_tol_u and rot_step_deg <= rot_tol_deg and cost_rel <= 1e-3:
                stage_entry["status"] = "converged"
                break

        if (not stage_used) and str(stage_entry.get("status")) == "ok":
            stage_entry["status"] = "anchor_only_no_valid_iteration"

        stage_entry["iterations"] = iter_logs
        last_iter = iter_logs[-1] if iter_logs else {}
        stage_entry["detail_q_stats_last"] = q_stats if iter_logs else {}
        stage_entry["detail_active_ratio_last"] = float(last_iter.get("detail_active_ratio", float("nan"))) if last_iter else float("nan")
        stage_entry["effective_weight_mass_last"] = {
            "base_weight_sum": float(last_iter.get("base_weight_sum", 0.0)) if last_iter else 0.0,
            "detail_weight_sum": float(last_iter.get("detail_weight_sum", 0.0)) if last_iter else 0.0,
            "lambda_overall_eff": float(last_iter.get("lambda_overall_eff", 0.0)) if last_iter else 0.0,
        }
        stages.append(stage_entry)

    final_meta = {
        "status": "ok" if any(str(s.get("status", "")).startswith(("ok", "converged")) for s in stages) else "anchor_only",
        "anchor_transform": np.asarray(T_anchor, dtype=np.float64),
        "stages": stages,
        "bcjr_base_info_eigvals_per_scale": [s.get("base_info_eigvals_desc", []) for s in stages],
        "bcjr_prior_mu_per_scale": [float(s.get("prior_mu", float("nan"))) for s in stages],
        "bcjr_lambda_base_per_scale": [float(s.get("lambda_base", float("nan"))) for s in stages],
        "bcjr_lambda_overall_per_scale": [float(s.get("lambda_overall_raw", float("nan"))) for s in stages],
        "bcjr_detail_active_ratio_per_scale": [float(s.get("detail_active_ratio_last", float("nan"))) for s in stages],
        "bcjr_detail_q_stats_per_scale": [s.get("detail_q_stats_last", {}) for s in stages],
        "bcjr_effective_weight_mass_per_scale": [s.get("effective_weight_mass_last", {}) for s in stages],
        "bcjr_anchor_to_final_delta_mm_deg": _pose_delta_mm_deg(np.asarray(T_anchor, dtype=np.float64), T, cfg.unit_scale),
        "bcjr_step_norm_history": step_history,
        "elapsed_s": float(time.time() - t0),
    }
    return T, final_meta


def _distance_stats(source_pts: np.ndarray, target_pts: np.ndarray, T: np.ndarray, unit_scale: float) -> Dict[str, Any]:
    def _empty() -> Dict[str, Any]:
        return {
            "n": 0,
            "mean_u": float("nan"),
            "p50_u": float("nan"),
            "p95_u": float("nan"),
            "p99_u": float("nan"),
            "max_u": float("nan"),
            "mean_mm": float("nan"),
            "p50_mm": float("nan"),
            "p95_mm": float("nan"),
            "p99_mm": float("nan"),
            "max_mm": float("nan"),
        }

    def _summary(d: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(d, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return _empty()
        mean_u = float(np.mean(x))
        p50_u = float(np.quantile(x, 0.50))
        p95_u = float(np.quantile(x, 0.95))
        p99_u = float(np.quantile(x, 0.99))
        max_u = float(np.max(x))
        return {
            "n": int(x.size),
            "mean_u": mean_u,
            "p50_u": p50_u,
            "p95_u": p95_u,
            "p99_u": p99_u,
            "max_u": max_u,
            "mean_mm": _mm_from_u(mean_u, unit_scale),
            "p50_mm": _mm_from_u(p50_u, unit_scale),
            "p95_mm": _mm_from_u(p95_u, unit_scale),
            "p99_mm": _mm_from_u(p99_u, unit_scale),
            "max_mm": _mm_from_u(max_u, unit_scale),
        }

    if len(source_pts) == 0 or len(target_pts) == 0:
        return _empty()
    src = _points_to_pcd(_transform_points(T, source_pts))
    tgt = _points_to_pcd(target_pts)
    d = np.asarray(src.compute_point_cloud_distance(tgt), dtype=np.float64)
    return _summary(d)


def _pin_alignment_stats(
    source_pin_pts: np.ndarray,
    ref_pin_pts: np.ndarray,
    T: np.ndarray,
    unit_scale: float,
    inlier_thresholds_mm: List[float],
) -> Dict[str, Any]:
    src_stats = _distance_stats(source_pin_pts, ref_pin_pts, T, unit_scale)
    src_aligned = _transform_points(T, source_pin_pts)
    ref_stats = _distance_stats(ref_pin_pts, src_aligned, np.eye(4, dtype=np.float64), unit_scale)

    mean_u_src = float(src_stats.get("mean_u", float("nan")))
    mean_u_ref = float(ref_stats.get("mean_u", float("nan")))
    p95_u_src = float(src_stats.get("p95_u", float("nan")))
    p95_u_ref = float(ref_stats.get("p95_u", float("nan")))

    if np.isfinite(mean_u_src) and np.isfinite(mean_u_ref):
        chamfer_mean_u = 0.5 * (mean_u_src + mean_u_ref)
    else:
        chamfer_mean_u = float("nan")

    if np.isfinite(p95_u_src) and np.isfinite(p95_u_ref):
        bidir_p95_u = float(max(p95_u_src, p95_u_ref))
    else:
        bidir_p95_u = float("nan")

    src_aligned_pcd = _points_to_pcd(src_aligned)
    ref_pcd = _points_to_pcd(ref_pin_pts)
    d_src = np.asarray(src_aligned_pcd.compute_point_cloud_distance(ref_pcd), dtype=np.float64)
    d_ref = np.asarray(ref_pcd.compute_point_cloud_distance(src_aligned_pcd), dtype=np.float64)

    ratio_src: Dict[str, float] = {}
    ratio_ref: Dict[str, float] = {}
    ratio_bi: Dict[str, float] = {}
    for th_mm in list(inlier_thresholds_mm):
        th_u = _u_from_mm(float(th_mm), unit_scale)
        key = f"{float(th_mm):g}mm"
        ratio_src[key] = float(np.mean(d_src <= th_u)) if d_src.size > 0 else float("nan")
        ratio_ref[key] = float(np.mean(d_ref <= th_u)) if d_ref.size > 0 else float("nan")
        if np.isfinite(ratio_src[key]) and np.isfinite(ratio_ref[key]):
            ratio_bi[key] = 0.5 * (float(ratio_src[key]) + float(ratio_ref[key]))
        else:
            ratio_bi[key] = float("nan")

    return {
        "source_to_ref": src_stats,
        "ref_to_source": ref_stats,
        "chamfer_mean_u": float(chamfer_mean_u),
        "chamfer_mean_mm": _mm_from_u(float(chamfer_mean_u), unit_scale) if np.isfinite(chamfer_mean_u) else float("nan"),
        "bidirectional_p95_u": float(bidir_p95_u),
        "bidirectional_p95_mm": _mm_from_u(float(bidir_p95_u), unit_scale) if np.isfinite(bidir_p95_u) else float("nan"),
        "inlier_thresholds_mm": [float(v) for v in inlier_thresholds_mm],
        "inlier_ratio_source_to_ref": ratio_src,
        "inlier_ratio_ref_to_source": ratio_ref,
        "inlier_ratio_bidirectional": ratio_bi,
    }


def _compute_alignment_metrics(
    *,
    T: np.ndarray,
    ref_cloud: LabeledCloud,
    src_cloud: LabeledCloud,
    unit_scale: float,
    pin_inlier_thresholds_mm: List[float],
) -> Dict[str, Any]:
    base_stats = _distance_stats(src_cloud.base_points, ref_cloud.base_points, T, unit_scale)
    overall_stats = _distance_stats(src_cloud.overall_points, ref_cloud.overall_points, T, unit_scale)
    pin_stats = _pin_alignment_stats(
        src_cloud.pins_points,
        ref_cloud.pins_points,
        T,
        unit_scale,
        pin_inlier_thresholds_mm,
    )
    return {
        "base": base_stats,
        "overall": overall_stats,
        "pins": pin_stats,
    }


def _alignment_score_mm(
    alignment_stats: Dict[str, Any],
    *,
    w_pin_chamfer: float,
    w_pin_bidir_p95: float,
    w_base_p95: float,
) -> float:
    pins = alignment_stats.get("pins", {})
    base = alignment_stats.get("base", {})
    pin_ch = float(pins.get("chamfer_mean_mm", float("nan")))
    pin_p95 = float(pins.get("bidirectional_p95_mm", float("nan")))
    base_p95 = float(base.get("p95_mm", float("nan")))
    if not (np.isfinite(pin_ch) and np.isfinite(pin_p95) and np.isfinite(base_p95)):
        return float("inf")
    return float(w_pin_chamfer) * pin_ch + float(w_pin_bidir_p95) * pin_p95 + float(w_base_p95) * base_p95


def _attach_transform_contract(meta: Dict[str, Any], *, init_T: Any, final_T: Any) -> None:
    init_arr = np.asarray(init_T, dtype=np.float64)
    final_arr = np.asarray(final_T, dtype=np.float64)
    if init_arr.shape == (4, 4) and np.all(np.isfinite(init_arr)):
        meta["init_transform"] = init_arr
    if final_arr.shape == (4, 4) and np.all(np.isfinite(final_arr)):
        meta["final_transform"] = final_arr


def _is_color_informative(cloud: LabeledCloud, min_std: float) -> Tuple[bool, Dict[str, Any]]:
    c = np.asarray(cloud.overall_colors, dtype=np.float64).reshape((-1, 3))
    if c.size == 0:
        return False, {"status": "no_color_points"}
    std_vec = np.std(c, axis=0)
    std_mean = float(np.mean(std_vec))
    ok = bool(cloud.meta.get("has_color", False)) and np.isfinite(std_mean) and (std_mean >= float(min_std))
    return ok, {
        "status": "ok" if ok else "low_color_std",
        "has_color": bool(cloud.meta.get("has_color", False)),
        "std_r": float(std_vec[0]),
        "std_g": float(std_vec[1]),
        "std_b": float(std_vec[2]),
        "std_mean": float(std_mean),
        "min_std": float(min_std),
    }


def _merge_pcds(*pcds):
    out = o3d.geometry.PointCloud()
    for p in pcds:
        if p is not None:
            out += p
    return out


def _write_overlay(
    out_path: Path,
    *,
    target_pts: np.ndarray,
    source_pts: np.ndarray,
    T: np.ndarray,
    downsample_u: float,
    target_color: Tuple[float, float, float],
    source_color: Tuple[float, float, float],
) -> None:
    tgt = _colorize_pcd(target_pts, target_color)
    src = _colorize_pcd(source_pts, source_color)
    src = copy.deepcopy(src)
    src.transform(np.asarray(T, dtype=float))
    if float(downsample_u) > 1e-12:
        tgt = tgt.voxel_down_sample(float(downsample_u))
        src = src.voxel_down_sample(float(downsample_u))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(out_path), _merge_pcds(tgt, src), write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"failed to write overlay: {out_path}")


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_config(method: str) -> RegistrationConfig:
    default_cloud = (Path(__file__).resolve().parent / "cloud").resolve()
    unit_scale_default = float(_UNIT_SCALE_DEFAULT) if np.isfinite(float(_UNIT_SCALE_DEFAULT)) else 1.0
    source_default = _env_str("DAIMA_REG_SOURCE_ID", _env_str("DAIMA_REG_SAMPLE_ID", "bad1"))
    pin_inlier_thresholds_mm = _parse_float_csv(
        _env_str("DAIMA_PIN_INLIER_THRESH_MM", "0.2,0.5,1.0"),
        default=[0.2, 0.5, 1.0],
    )
    rgb_voxel_scales_mm = _parse_float_csv_ordered(
        _env_str("DAIMA_RGB_VOXEL_SCALES_MM", "1.2,0.8,0.4"),
        default=[1.2, 0.8, 0.4],
    )
    rgb_max_iters = _parse_int_csv_ordered(
        _env_str("DAIMA_RGB_MAX_ITERS", "60,40,25"),
        default=[60, 40, 25],
    )
    local_refine_voxel_scales_mm = _parse_float_csv_ordered(
        _env_str("DAIMA_LOCAL_REFINE_VOXEL_SCALES_MM", "1.2,0.8,0.4"),
        default=[1.2, 0.8, 0.4],
    )
    local_refine_max_iters = _parse_int_csv_ordered(
        _env_str("DAIMA_LOCAL_REFINE_MAX_ITERS", "60,40,25"),
        default=[60, 40, 25],
    )
    bcjr_voxel_scales_mm = _parse_float_csv_ordered(
        _env_str("DAIMA_BCJR_VOXEL_SCALES_MM", "1.2,0.8,0.4"),
        default=[1.2, 0.8, 0.4],
    )
    bcjr_max_iters = _parse_int_csv_ordered(
        _env_str("DAIMA_BCJR_MAX_ITERS", "15,10,6"),
        default=[15, 10, 6],
    )
    bcjr_lambda_base = _parse_float_csv_ordered(
        _env_str("DAIMA_BCJR_LAMBDA_BASE", "1.0,0.85,0.70"),
        default=[1.0, 0.85, 0.70],
    )
    bcjr_lambda_overall = _parse_float_csv_ordered(
        _env_str("DAIMA_BCJR_LAMBDA_OVERALL", "0.70,1.00,1.20"),
        default=[0.70, 1.00, 1.20],
    )
    bcjr_prior_mu = _parse_float_csv_ordered(
        _env_str("DAIMA_BCJR_PRIOR_MU", "0.80,0.35,0.12"),
        default=[0.80, 0.35, 0.12],
    )
    bcjr_clip_rotation_deg = _parse_float_csv_ordered(
        _env_str("DAIMA_BCJR_CLIP_ROTATION_DEG", "0.60,0.30,0.15"),
        default=[0.60, 0.30, 0.15],
    )
    rescue_verify_voxel_mm_default = float(
        bcjr_voxel_scales_mm[1] if len(bcjr_voxel_scales_mm) >= 2 else bcjr_voxel_scales_mm[0]
    )

    return RegistrationConfig(
        method=str(method).strip().lower(),
        cloud_dir=Path(_env_str("DAIMA_REG_CLOUD_DIR", str(default_cloud))).resolve(),
        ref_id=_env_str("DAIMA_REG_REF_ID", "10"),
        source_id=source_default,
        unit_scale=_env_float("DAIMA_UNIT_SCALE", unit_scale_default),
        target_grid_points=_env_int("DAIMA_REG_TARGET_GRID_POINTS", 50000),
        min_voxel_mm=_env_float("DAIMA_REG_MIN_VOXEL_MM", 0.2),
        max_voxel_mm=_env_float("DAIMA_REG_MAX_VOXEL_MM", 5.0),
        ransac_max_iter=_env_int("DAIMA_REG_RANSAC_MAX_ITER", 400000),
        ransac_confidence=_env_float("DAIMA_REG_RANSAC_CONFIDENCE", 0.999),
        write_overlay=_env_bool("DAIMA_REG_WRITE_OVERLAY", True),
        overlay_downsample_mm=_env_float("DAIMA_REG_OVERLAY_DOWNSAMPLE_MM", 0.8),
        pin_inlier_thresholds_mm=pin_inlier_thresholds_mm,
        rgb_voxel_scales_mm=rgb_voxel_scales_mm,
        rgb_max_iters=rgb_max_iters,
        rgb_lambda_geometric=_env_float("DAIMA_RGB_LAMBDA_GEOMETRIC", 0.968),
        rgb_max_corr_factor=_env_float("DAIMA_RGB_MAX_CORR_FACTOR", 1.4),
        rgb_final_base_refine=_env_bool("DAIMA_RGB_FINAL_BASE_REFINE", True),
        rgb_gate_enable=_env_bool("DAIMA_RGB_GATE_ENABLE", True),
        rgb_gate_min_init_fitness=_env_float("DAIMA_RGB_GATE_MIN_INIT_FITNESS", 0.25),
        rgb_gate_min_color_std=_env_float("DAIMA_RGB_GATE_MIN_COLOR_STD", 0.03),
        rgb_gate_max_pin_chamfer_degrade_mm=_env_float("DAIMA_RGB_GATE_MAX_PIN_CHAMFER_DEGRADE_MM", 0.02),
        rgb_gate_max_pin_bidir_p95_degrade_mm=_env_float("DAIMA_RGB_GATE_MAX_PIN_BIDIR_P95_DEGRADE_MM", 0.04),
        rgb_gate_max_base_p95_degrade_mm=_env_float("DAIMA_RGB_GATE_MAX_BASE_P95_DEGRADE_MM", 0.04),
        rgb_gate_min_improve_mm=_env_float("DAIMA_RGB_GATE_MIN_IMPROVE_MM", 0.003),
        norgb_enable_fgr_init=_env_bool("DAIMA_NORGB_ENABLE_FGR_INIT", True),
        norgb_enable_gicp_refine=_env_bool("DAIMA_NORGB_ENABLE_GICP_REFINE", True),
        norgb_robust_kernel_k_mm=_env_float("DAIMA_NORGB_ROBUST_KERNEL_K_MM", 0.6),
        norgb_score_w_pin_chamfer=_env_float("DAIMA_NORGB_SCORE_W_PIN_CHAMFER", 1.0),
        norgb_score_w_pin_bidir_p95=_env_float("DAIMA_NORGB_SCORE_W_PIN_BIDIR_P95", 0.25),
        norgb_score_w_base_p95=_env_float("DAIMA_NORGB_SCORE_W_BASE_P95", 0.15),
        init_backend=_env_str("DAIMA_INIT_BACKEND", "ransac").strip().lower(),
        init_backend_allow_fallback=_env_bool("DAIMA_INIT_BACKEND_ALLOW_FALLBACK", False),
        local_refine_multiscale_enable=_env_bool("DAIMA_LOCAL_REFINE_MULTISCALE_ENABLE", True),
        local_refine_voxel_scales_mm=local_refine_voxel_scales_mm,
        local_refine_max_iters=local_refine_max_iters,
        local_refine_max_corr_factor=_env_float("DAIMA_LOCAL_REFINE_MAX_CORR_FACTOR", 1.5),
        local_refine_robust_enable=_env_bool("DAIMA_LOCAL_REFINE_ROBUST_ENABLE", True),
        local_refine_robust_kernel=_env_str("DAIMA_LOCAL_REFINE_ROBUST_KERNEL", "cauchy").strip().lower(),
        local_refine_robust_k_mm=_env_float("DAIMA_LOCAL_REFINE_ROBUST_K_MM", 0.6),
        bcjr_enable=_env_bool("DAIMA_BCJR_ENABLE", True),
        bcjr_voxel_scales_mm=bcjr_voxel_scales_mm,
        bcjr_max_iters=bcjr_max_iters,
        bcjr_lambda_base=bcjr_lambda_base,
        bcjr_lambda_overall=bcjr_lambda_overall,
        bcjr_prior_mu=bcjr_prior_mu,
        bcjr_weak_mode_count=_env_int("DAIMA_BCJR_WEAK_MODE_COUNT", 2),
        bcjr_q_min=_env_float("DAIMA_BCJR_Q_MIN", 0.05),
        bcjr_lm_damping=_env_float("DAIMA_BCJR_LM_DAMPING", 1e-4),
        bcjr_exclude_base_overlap_factor=_env_float("DAIMA_BCJR_EXCLUDE_BASE_OVERLAP_FACTOR", 0.60),
        bcjr_beta_bias=_env_float("DAIMA_BCJR_BETA_BIAS", -0.20),
        bcjr_beta_r=_env_float("DAIMA_BCJR_BETA_R", 1.80),
        bcjr_beta_n=_env_float("DAIMA_BCJR_BETA_N", 1.10),
        bcjr_beta_u=_env_float("DAIMA_BCJR_BETA_U", 1.20),
        bcjr_beta_g=_env_float("DAIMA_BCJR_BETA_G", 1.50),
        bcjr_clip_translation_factor=_env_float("DAIMA_BCJR_CLIP_TRANSLATION_FACTOR", 0.50),
        bcjr_clip_rotation_deg=bcjr_clip_rotation_deg,
        bcjr_rescue_enable=_env_bool("DAIMA_BCJR_RESCUE_ENABLE", True),
        bcjr_rescue_rotation_deg=_env_float("DAIMA_BCJR_RESCUE_ROTATION_DEG", 180.0),
        bcjr_rescue_verify_voxel_mm=_env_float("DAIMA_BCJR_RESCUE_VERIFY_VOXEL_MM", rescue_verify_voxel_mm_default),
        bcjr_rescue_verify_max_corr_factor=_env_float("DAIMA_BCJR_RESCUE_VERIFY_MAX_CORR_FACTOR", 1.5),
        bcjr_rescue_verify_base_guard_mm=_env_float("DAIMA_BCJR_RESCUE_VERIFY_BASE_GUARD_MM", 0.20),
        bcjr_rescue_detail_rmse_tie_mm=_env_float("DAIMA_BCJR_RESCUE_DETAIL_RMSE_TIE_MM", 0.02),
        pair_id=_env_str("DAIMA_REG_PAIR_ID", "").strip(),
        require_route_tag=_env_bool("DAIMA_REG_REQUIRE_ROUTE_TAG", False),
        route_tag=_env_str("DAIMA_ROUTE_TAG", "").strip(),
    )


def _load_pcd_sets(cloud: LabeledCloud) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["overall"] = _remove_outliers(_points_to_pcd(cloud.overall_points))
    out["base"] = _remove_outliers(_points_to_pcd(cloud.base_points))
    out["overall_color"] = _remove_outliers(_points_to_pcd_with_colors(cloud.overall_points, cloud.overall_colors))
    out["base_color"] = _remove_outliers(_points_to_pcd_with_colors(cloud.base_points, cloud.base_colors))
    return out


def _compute_transform(cfg: RegistrationConfig, ref_cloud: LabeledCloud, src_cloud: LabeledCloud) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"status": "ok", "method": cfg.method}

    min_voxel_u = _u_from_mm(cfg.min_voxel_mm, cfg.unit_scale)
    max_voxel_u = _u_from_mm(cfg.max_voxel_mm, cfg.unit_scale)

    ref_pcds = _load_pcd_sets(ref_cloud)
    src_pcds = _load_pcd_sets(src_cloud)

    def _voxel_for(pcd) -> float:
        return _estimate_voxel_u_fixed(
            pcd,
            target_grid_points=int(cfg.target_grid_points),
            min_voxel_u=float(min_voxel_u),
            max_voxel_u=float(max_voxel_u),
        )

    method = str(cfg.method).strip().lower()

    local_scales_u = [_u_from_mm(v, cfg.unit_scale) for v in cfg.local_refine_voxel_scales_mm]
    local_robust_k_u = _u_from_mm(cfg.local_refine_robust_k_mm, cfg.unit_scale)

    def _run_local_refine(source, target, init_T: np.ndarray, voxel_u: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        if bool(cfg.local_refine_multiscale_enable):
            return _refine_icp_point_to_plane_multiscale(
                source=source,
                target=target,
                init_T=np.asarray(init_T, dtype=np.float64),
                voxel_scales_u=local_scales_u,
                max_iters=cfg.local_refine_max_iters,
                max_corr_factor=float(cfg.local_refine_max_corr_factor),
                robust_enable=bool(cfg.local_refine_robust_enable),
                robust_kernel=str(cfg.local_refine_robust_kernel),
                robust_k_u=float(local_robust_k_u),
            )
        if bool(cfg.local_refine_robust_enable):
            return _refine_icp_point_to_plane_robust(
                source=source,
                target=target,
                init_T=np.asarray(init_T, dtype=np.float64),
                voxel_u=float(voxel_u),
                robust_k_u=float(local_robust_k_u),
            )
        return _refine_icp_point_to_plane(
            source=source,
            target=target,
            init_T=np.asarray(init_T, dtype=np.float64),
            voxel_u=float(voxel_u),
        )

    if method in ("base_v1", "base_only_v1"):
        voxel_u = _voxel_for(ref_pcds["base"])
        T0, m0 = _register_global_init_by_backend(
            source=src_pcds["base"],
            target=ref_pcds["base"],
            voxel_u=float(voxel_u),
            backend=str(cfg.init_backend),
            allow_fallback=bool(cfg.init_backend_allow_fallback),
            ransac_max_iter=int(cfg.ransac_max_iter),
            ransac_confidence=float(cfg.ransac_confidence),
        )
        T, m1 = _run_local_refine(
            source=src_pcds["base"],
            target=ref_pcds["base"],
            init_T=T0,
            voxel_u=float(voxel_u),
        )
        meta["init_stage"] = dict(m0)
        meta["local_refine_stage"] = dict(m1)
        meta["voxel_base_u"] = float(voxel_u)
        meta["voxel_base_mm"] = _mm_from_u(float(voxel_u), cfg.unit_scale)
        meta["local_refine_multiscale_enable"] = bool(cfg.local_refine_multiscale_enable)
        meta["local_refine_voxel_scales_mm"] = [float(v) for v in cfg.local_refine_voxel_scales_mm]
        meta["local_refine_max_iters"] = [int(v) for v in cfg.local_refine_max_iters]
        meta["local_refine_max_corr_factor"] = float(cfg.local_refine_max_corr_factor)
        meta["local_refine_robust_enable"] = bool(cfg.local_refine_robust_enable)
        meta["local_refine_robust_kernel"] = str(cfg.local_refine_robust_kernel)
        meta["local_refine_robust_k_mm"] = float(cfg.local_refine_robust_k_mm)
        meta["icp_fitness"] = float(m1.get("final_fitness", m1.get("icp_fitness", float("nan"))))
        meta["icp_inlier_rmse_u"] = float(m1.get("final_inlier_rmse_u", m1.get("icp_inlier_rmse_u", float("nan"))))
        meta["elapsed_s"] = float(m0.get("elapsed_s", 0.0)) + float(m1.get("elapsed_s", 0.0))
        _attach_transform_contract(meta, init_T=T0, final_T=T)
        return T, meta

    if method in ("overall_v1", "overall_only_v1"):
        voxel_u = _voxel_for(ref_pcds["overall"])
        T0, m0 = _register_global_init_by_backend(
            source=src_pcds["overall"],
            target=ref_pcds["overall"],
            voxel_u=float(voxel_u),
            backend=str(cfg.init_backend),
            allow_fallback=bool(cfg.init_backend_allow_fallback),
            ransac_max_iter=int(cfg.ransac_max_iter),
            ransac_confidence=float(cfg.ransac_confidence),
        )
        T, m1 = _run_local_refine(
            source=src_pcds["overall"],
            target=ref_pcds["overall"],
            init_T=T0,
            voxel_u=float(voxel_u),
        )
        meta["init_stage"] = dict(m0)
        meta["local_refine_stage"] = dict(m1)
        meta["voxel_overall_u"] = float(voxel_u)
        meta["voxel_overall_mm"] = _mm_from_u(float(voxel_u), cfg.unit_scale)
        meta["local_refine_multiscale_enable"] = bool(cfg.local_refine_multiscale_enable)
        meta["local_refine_voxel_scales_mm"] = [float(v) for v in cfg.local_refine_voxel_scales_mm]
        meta["local_refine_max_iters"] = [int(v) for v in cfg.local_refine_max_iters]
        meta["local_refine_max_corr_factor"] = float(cfg.local_refine_max_corr_factor)
        meta["local_refine_robust_enable"] = bool(cfg.local_refine_robust_enable)
        meta["local_refine_robust_kernel"] = str(cfg.local_refine_robust_kernel)
        meta["local_refine_robust_k_mm"] = float(cfg.local_refine_robust_k_mm)
        meta["icp_fitness"] = float(m1.get("final_fitness", m1.get("icp_fitness", float("nan"))))
        meta["icp_inlier_rmse_u"] = float(m1.get("final_inlier_rmse_u", m1.get("icp_inlier_rmse_u", float("nan"))))
        meta["elapsed_s"] = float(m0.get("elapsed_s", 0.0)) + float(m1.get("elapsed_s", 0.0))
        _attach_transform_contract(meta, init_T=T0, final_T=T)
        return T, meta

    if method in ("overall_then_base_v1", "overall+base_v1", "overall_then_base"):
        voxel_overall_u = _voxel_for(ref_pcds["overall"])
        T0, m0 = _register_global_init_by_backend(
            source=src_pcds["overall"],
            target=ref_pcds["overall"],
            voxel_u=float(voxel_overall_u),
            backend=str(cfg.init_backend),
            allow_fallback=bool(cfg.init_backend_allow_fallback),
            ransac_max_iter=int(cfg.ransac_max_iter),
            ransac_confidence=float(cfg.ransac_confidence),
        )

        voxel_base_u = _voxel_for(ref_pcds["base"])
        T, m1 = _run_local_refine(
            source=src_pcds["base"],
            target=ref_pcds["base"],
            init_T=T0,
            voxel_u=float(voxel_base_u),
        )

        meta["overall_stage"] = dict(m0)
        meta["init_stage"] = dict(m0)
        meta["base_refine_stage"] = dict(m1)
        meta["local_refine_stage"] = dict(m1)
        meta["voxel_overall_u"] = float(voxel_overall_u)
        meta["voxel_overall_mm"] = _mm_from_u(float(voxel_overall_u), cfg.unit_scale)
        meta["voxel_base_u"] = float(voxel_base_u)
        meta["voxel_base_mm"] = _mm_from_u(float(voxel_base_u), cfg.unit_scale)
        meta["local_refine_multiscale_enable"] = bool(cfg.local_refine_multiscale_enable)
        meta["local_refine_voxel_scales_mm"] = [float(v) for v in cfg.local_refine_voxel_scales_mm]
        meta["local_refine_max_iters"] = [int(v) for v in cfg.local_refine_max_iters]
        meta["local_refine_max_corr_factor"] = float(cfg.local_refine_max_corr_factor)
        meta["local_refine_robust_enable"] = bool(cfg.local_refine_robust_enable)
        meta["local_refine_robust_kernel"] = str(cfg.local_refine_robust_kernel)
        meta["local_refine_robust_k_mm"] = float(cfg.local_refine_robust_k_mm)
        meta["icp_fitness"] = float(m1.get("final_fitness", m1.get("icp_fitness", float("nan"))))
        meta["icp_inlier_rmse_u"] = float(m1.get("final_inlier_rmse_u", m1.get("icp_inlier_rmse_u", float("nan"))))
        meta["elapsed_s"] = float(m0.get("elapsed_s", 0.0)) + float(m1.get("elapsed_s", 0.0))
        _attach_transform_contract(meta, init_T=T0, final_T=T)
        return T, meta

    if method in ("overall_then_base_gate_v2", "bcjr_v1", "bcjr"):
        voxel_overall_u = _voxel_for(ref_pcds["overall"])
        T0, m0 = _register_global_init_by_backend(
            source=src_pcds["overall"],
            target=ref_pcds["overall"],
            voxel_u=float(voxel_overall_u),
            backend=str(cfg.init_backend),
            allow_fallback=bool(cfg.init_backend_allow_fallback),
            ransac_max_iter=int(cfg.ransac_max_iter),
            ransac_confidence=float(cfg.ransac_confidence),
        )

        voxel_base_u = _voxel_for(ref_pcds["base"])
        Tb, m1 = _run_local_refine(
            source=src_pcds["base"],
            target=ref_pcds["base"],
            init_T=T0,
            voxel_u=float(voxel_base_u),
        )

        exclude_radius_u = float(max(float(cfg.bcjr_exclude_base_overlap_factor) * float(voxel_overall_u), 1e-9))
        src_detail = _build_overall_detail_cloud(src_pcds["overall"], src_pcds["base"], exclude_radius_u)
        ref_detail = _build_overall_detail_cloud(ref_pcds["overall"], ref_pcds["base"], exclude_radius_u)
        t_joint0 = time.time()
        rescue_max_corr_u = float(max(float(cfg.local_refine_max_corr_factor) * float(voxel_base_u), 1e-9))
        src_base_rescue = src_pcds["base"].voxel_down_sample(float(voxel_base_u))
        tgt_base_rescue = ref_pcds["base"].voxel_down_sample(float(voxel_base_u))
        rescue_info = _compute_base_information_matrix(
            source_base=src_base_rescue,
            target_base=tgt_base_rescue,
            T_anchor=np.asarray(Tb, dtype=np.float64),
            max_corr_u=float(rescue_max_corr_u),
        )
        rescue_axes = _extract_bcjr_rescue_rotation_axes(
            rescue_info,
            max_axes=2,
        )
        src_base_pts = np.asarray(src_pcds["base"].points, dtype=np.float64)
        if src_base_pts.shape[0] > 0:
            rescue_pivot = _transform_points(
                np.asarray(Tb, dtype=np.float64),
                np.mean(src_base_pts, axis=0, keepdims=True),
            )[0]
        else:
            rescue_pivot = np.asarray(Tb, dtype=np.float64)[:3, 3].copy()

        candidate_specs: List[Dict[str, Any]] = [
            {
                "candidate_index": 0,
                "id": "anchor",
                "name": "anchor",
                "seed_transform": np.asarray(Tb, dtype=np.float64).copy(),
                "axis": None,
                "axis_rank": None,
                "axis_eig_value": None,
                "seed_rotation_deg": 0.0,
            }
        ]
        if bool(cfg.bcjr_rescue_enable):
            for axis_meta in rescue_axes:
                axis_vec = np.asarray(axis_meta.get("axis", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
                T_seed = _rotation_about_pivot(
                    axis=axis_vec,
                    angle_rad=float(np.deg2rad(float(cfg.bcjr_rescue_rotation_deg))),
                    pivot_u=np.asarray(rescue_pivot, dtype=np.float64),
                ) @ np.asarray(Tb, dtype=np.float64)
                candidate_specs.append(
                    {
                        "candidate_index": int(len(candidate_specs)),
                        "id": f"flip_axis_{int(axis_meta.get('rank', 0))}",
                        "name": f"flip_axis_{int(axis_meta.get('rank', 0))}",
                        "seed_transform": np.asarray(T_seed, dtype=np.float64),
                        "axis": axis_vec.astype(np.float64),
                        "axis_rank": int(axis_meta.get("rank", 0)),
                        "axis_eig_value": float(axis_meta.get("eig_value", float("nan"))),
                        "seed_rotation_deg": float(cfg.bcjr_rescue_rotation_deg),
                    }
                )

        candidates: List[Dict[str, Any]] = []
        for spec in candidate_specs:
            T_seed = np.asarray(spec.get("seed_transform"), dtype=np.float64)
            T_c, m_c = _run_base_conditioned_joint_refinement(
                source_base=src_pcds["base"],
                target_base=ref_pcds["base"],
                source_detail=src_detail,
                target_detail=ref_detail,
                T_anchor=T_seed,
                cfg=cfg,
            )
            verify = _verify_bcjr_candidate(
                source_base=src_pcds["base"],
                target_base=ref_pcds["base"],
                source_detail=src_detail,
                target_detail=ref_detail,
                T=np.asarray(T_c, dtype=np.float64),
                cfg=cfg,
            )
            candidates.append(
                {
                    "candidate_index": int(spec.get("candidate_index", 0)),
                    "id": str(spec.get("id", "")),
                    "name": str(spec.get("name", "")),
                    "seed_transform": T_seed,
                    "final_transform": np.asarray(T_c, dtype=np.float64),
                    "axis": None if spec.get("axis") is None else np.asarray(spec.get("axis"), dtype=np.float64),
                    "axis_rank": spec.get("axis_rank"),
                    "axis_eig_value": spec.get("axis_eig_value"),
                    "pivot_u": np.asarray(rescue_pivot, dtype=np.float64),
                    "seed_rotation_deg": float(spec.get("seed_rotation_deg", 0.0)),
                    "anchor_to_seed_delta_mm_deg": _pose_delta_mm_deg(np.asarray(Tb, dtype=np.float64), T_seed, cfg.unit_scale),
                    "anchor_to_final_delta_mm_deg": _pose_delta_mm_deg(np.asarray(Tb, dtype=np.float64), np.asarray(T_c, dtype=np.float64), cfg.unit_scale),
                    "seed_to_final_delta_mm_deg": _pose_delta_mm_deg(T_seed, np.asarray(T_c, dtype=np.float64), cfg.unit_scale),
                    "verify": verify,
                    "joint_refine_stage": dict(m_c),
                }
            )

        selected = _select_bcjr_candidate(candidates, cfg)
        T = np.asarray(selected.get("final_transform"), dtype=np.float64)
        m2 = dict(selected.get("joint_refine_stage", {}))
        seed_to_final_delta = dict(m2.get("bcjr_anchor_to_final_delta_mm_deg", {}))
        m2["bcjr_candidate_seed_to_final_delta_mm_deg"] = seed_to_final_delta
        m2["bcjr_anchor_to_final_delta_mm_deg"] = _pose_delta_mm_deg(np.asarray(Tb, dtype=np.float64), T, cfg.unit_scale)
        m2["bcjr_candidate_id"] = str(selected.get("id", ""))
        m2["bcjr_candidate_verify"] = dict(selected.get("verify", {}))
        m2["bcjr_selection_elapsed_total_s"] = float(time.time() - t_joint0)

        meta["overall_stage"] = dict(m0)
        meta["init_stage"] = dict(m0)
        meta["base_refine_stage"] = dict(m1)
        meta["local_refine_stage"] = dict(m1)
        meta["joint_refine_stage"] = dict(m2)
        meta["anchor_transform"] = np.asarray(Tb, dtype=np.float64)
        meta["voxel_overall_u"] = float(voxel_overall_u)
        meta["voxel_overall_mm"] = _mm_from_u(float(voxel_overall_u), cfg.unit_scale)
        meta["voxel_base_u"] = float(voxel_base_u)
        meta["voxel_base_mm"] = _mm_from_u(float(voxel_base_u), cfg.unit_scale)
        meta["detail_overlap_exclude_radius_u"] = float(exclude_radius_u)
        meta["detail_overlap_exclude_radius_mm"] = _mm_from_u(float(exclude_radius_u), cfg.unit_scale)
        meta["detail_point_counts"] = {
            "source_overall": int(len(src_pcds["overall"].points)),
            "source_base": int(len(src_pcds["base"].points)),
            "source_detail": int(len(src_detail.points)),
            "target_overall": int(len(ref_pcds["overall"].points)),
            "target_base": int(len(ref_pcds["base"].points)),
            "target_detail": int(len(ref_detail.points)),
        }
        meta["local_refine_multiscale_enable"] = bool(cfg.local_refine_multiscale_enable)
        meta["local_refine_voxel_scales_mm"] = [float(v) for v in cfg.local_refine_voxel_scales_mm]
        meta["local_refine_max_iters"] = [int(v) for v in cfg.local_refine_max_iters]
        meta["local_refine_max_corr_factor"] = float(cfg.local_refine_max_corr_factor)
        meta["local_refine_robust_enable"] = bool(cfg.local_refine_robust_enable)
        meta["local_refine_robust_kernel"] = str(cfg.local_refine_robust_kernel)
        meta["local_refine_robust_k_mm"] = float(cfg.local_refine_robust_k_mm)
        meta["bcjr_enable"] = bool(cfg.bcjr_enable)
        meta["bcjr_voxel_scales_mm"] = [float(v) for v in cfg.bcjr_voxel_scales_mm]
        meta["bcjr_max_iters"] = [int(v) for v in cfg.bcjr_max_iters]
        meta["bcjr_lambda_base"] = [float(v) for v in cfg.bcjr_lambda_base]
        meta["bcjr_lambda_overall"] = [float(v) for v in cfg.bcjr_lambda_overall]
        meta["bcjr_prior_mu"] = [float(v) for v in cfg.bcjr_prior_mu]
        meta["bcjr_weak_mode_count"] = int(cfg.bcjr_weak_mode_count)
        meta["bcjr_q_min"] = float(cfg.bcjr_q_min)
        meta["bcjr_lm_damping"] = float(cfg.bcjr_lm_damping)
        meta["bcjr_beta_bias"] = float(cfg.bcjr_beta_bias)
        meta["bcjr_beta_r"] = float(cfg.bcjr_beta_r)
        meta["bcjr_beta_n"] = float(cfg.bcjr_beta_n)
        meta["bcjr_beta_u"] = float(cfg.bcjr_beta_u)
        meta["bcjr_beta_g"] = float(cfg.bcjr_beta_g)
        meta["bcjr_clip_translation_factor"] = float(cfg.bcjr_clip_translation_factor)
        meta["bcjr_clip_rotation_deg"] = [float(v) for v in cfg.bcjr_clip_rotation_deg]
        meta["bcjr_rescue_enable"] = bool(cfg.bcjr_rescue_enable)
        meta["bcjr_rescue_rotation_deg"] = float(cfg.bcjr_rescue_rotation_deg)
        meta["bcjr_rescue_verify_voxel_mm"] = float(cfg.bcjr_rescue_verify_voxel_mm)
        meta["bcjr_rescue_verify_max_corr_factor"] = float(cfg.bcjr_rescue_verify_max_corr_factor)
        meta["bcjr_rescue_verify_base_guard_mm"] = float(cfg.bcjr_rescue_verify_base_guard_mm)
        meta["bcjr_rescue_detail_rmse_tie_mm"] = float(cfg.bcjr_rescue_detail_rmse_tie_mm)
        meta["bcjr_anchor_candidates"] = candidates
        meta["bcjr_anchor_candidate_axis"] = [
            {
                "id": str(c.get("id", "")),
                "rank": c.get("axis_rank"),
                "axis": [] if c.get("axis") is None else [float(v) for v in np.asarray(c.get("axis"), dtype=np.float64).reshape(3)],
                "eig_value": float(c.get("axis_eig_value", float("nan"))) if c.get("axis_eig_value") is not None else float("nan"),
            }
            for c in candidates
        ]
        meta["bcjr_anchor_candidate_pivot"] = [float(v) for v in np.asarray(rescue_pivot, dtype=np.float64).reshape(3)]
        meta["bcjr_candidate_verify_detail_rmse_mm"] = {
            str(c.get("id", "")): float(dict(c.get("verify", {})).get("detail_verify_rmse_mm", float("inf")))
            for c in candidates
        }
        meta["bcjr_candidate_verify_detail_corr_count"] = {
            str(c.get("id", "")): int(dict(c.get("verify", {})).get("detail_verify_corr_count", 0))
            for c in candidates
        }
        meta["bcjr_candidate_verify_base_rmse_mm"] = {
            str(c.get("id", "")): float(dict(c.get("verify", {})).get("base_verify_rmse_mm", float("inf")))
            for c in candidates
        }
        meta["bcjr_candidate_chosen_id"] = str(selected.get("id", ""))
        meta["bcjr_rescue_base_info_eigvals_desc"] = [float(v) for v in np.asarray(np.linalg.eigvalsh(0.5 * (rescue_info + rescue_info.T)), dtype=np.float64)[::-1]]
        meta["icp_fitness"] = float(m1.get("final_fitness", m1.get("icp_fitness", float("nan"))))
        meta["icp_inlier_rmse_u"] = float(m1.get("final_inlier_rmse_u", m1.get("icp_inlier_rmse_u", float("nan"))))
        meta["elapsed_s"] = float(m0.get("elapsed_s", 0.0)) + float(m1.get("elapsed_s", 0.0)) + float(m2.get("bcjr_selection_elapsed_total_s", m2.get("elapsed_s", 0.0)))
        _attach_transform_contract(meta, init_T=T0, final_T=T)
        return T, meta

    if method in ("rgb_guided_v1", "overall_then_rgb_v1", "overall_rgb_v1"):
        voxel_overall_u = _voxel_for(ref_pcds["overall"])
        T0, m0 = _register_fpfh_ransac_then_icp_v1(
            source=src_pcds["overall"],
            target=ref_pcds["overall"],
            voxel_u=float(voxel_overall_u),
            ransac_max_iter=int(cfg.ransac_max_iter),
            ransac_confidence=float(cfg.ransac_confidence),
        )

        gate_reason = "gate_disabled"
        gate_meta: Dict[str, Any] = {
            "enabled": bool(cfg.rgb_gate_enable),
            "min_init_fitness": float(cfg.rgb_gate_min_init_fitness),
            "min_color_std": float(cfg.rgb_gate_min_color_std),
            "max_pin_chamfer_degrade_mm": float(cfg.rgb_gate_max_pin_chamfer_degrade_mm),
            "max_pin_bidir_p95_degrade_mm": float(cfg.rgb_gate_max_pin_bidir_p95_degrade_mm),
            "max_base_p95_degrade_mm": float(cfg.rgb_gate_max_base_p95_degrade_mm),
            "min_improve_mm": float(cfg.rgb_gate_min_improve_mm),
        }

        color_ok, color_meta = _is_color_informative(src_cloud, float(cfg.rgb_gate_min_color_std))
        gate_meta["color_check"] = color_meta

        init_fit = float(m0.get("icp_fitness", float("nan")))
        init_ok = np.isfinite(init_fit) and (init_fit >= float(cfg.rgb_gate_min_init_fitness))
        gate_meta["init_fit"] = float(init_fit)
        gate_meta["init_fit_ok"] = bool(init_ok)

        use_rgb = bool(cfg.rgb_gate_enable) and bool(color_ok) and bool(init_ok)
        if not bool(cfg.rgb_gate_enable):
            gate_reason = "disabled"
        elif not bool(color_ok):
            gate_reason = "color_not_informative"
        elif not bool(init_ok):
            gate_reason = "init_quality_low"
        else:
            gate_reason = "passed_precheck"

        metrics_before = _compute_alignment_metrics(
            T=np.asarray(T0, dtype=np.float64),
            ref_cloud=ref_cloud,
            src_cloud=src_cloud,
            unit_scale=float(cfg.unit_scale),
            pin_inlier_thresholds_mm=cfg.pin_inlier_thresholds_mm,
        )

        rgb_voxel_scales_u = [_u_from_mm(v, cfg.unit_scale) for v in cfg.rgb_voxel_scales_mm]
        T = np.asarray(T0, dtype=np.float64)
        m1: Dict[str, Any] = {"status": "skipped", "reason": gate_reason}
        if use_rgb:
            T1, m1_run = _refine_colored_icp_multiscale(
                source=src_pcds["overall_color"],
                target=ref_pcds["overall_color"],
                init_T=T0,
                voxel_scales_u=rgb_voxel_scales_u,
                max_iters=cfg.rgb_max_iters,
                lambda_geometric=float(cfg.rgb_lambda_geometric),
                max_corr_factor=float(cfg.rgb_max_corr_factor),
            )
            m1 = dict(m1_run)

            metrics_after = _compute_alignment_metrics(
                T=np.asarray(T1, dtype=np.float64),
                ref_cloud=ref_cloud,
                src_cloud=src_cloud,
                unit_scale=float(cfg.unit_scale),
                pin_inlier_thresholds_mm=cfg.pin_inlier_thresholds_mm,
            )

            pin_ch_before = float(metrics_before.get("pins", {}).get("chamfer_mean_mm", float("nan")))
            pin_ch_after = float(metrics_after.get("pins", {}).get("chamfer_mean_mm", float("nan")))
            pin_p95_before = float(metrics_before.get("pins", {}).get("bidirectional_p95_mm", float("nan")))
            pin_p95_after = float(metrics_after.get("pins", {}).get("bidirectional_p95_mm", float("nan")))
            base_p95_before = float(metrics_before.get("base", {}).get("p95_mm", float("nan")))
            base_p95_after = float(metrics_after.get("base", {}).get("p95_mm", float("nan")))

            d_pin_ch = pin_ch_after - pin_ch_before
            d_pin_p95 = pin_p95_after - pin_p95_before
            d_base_p95 = base_p95_after - base_p95_before

            accept = True
            if np.isfinite(d_pin_ch) and d_pin_ch > float(cfg.rgb_gate_max_pin_chamfer_degrade_mm):
                accept = False
            if np.isfinite(d_pin_p95) and d_pin_p95 > float(cfg.rgb_gate_max_pin_bidir_p95_degrade_mm):
                accept = False
            if np.isfinite(d_base_p95) and d_base_p95 > float(cfg.rgb_gate_max_base_p95_degrade_mm):
                accept = False

            if np.isfinite(d_pin_ch) and np.isfinite(d_pin_p95):
                improved = (d_pin_ch <= -float(cfg.rgb_gate_min_improve_mm)) or (
                    d_pin_p95 <= -float(cfg.rgb_gate_min_improve_mm)
                )
            else:
                improved = False

            if accept and improved:
                T = np.asarray(T1, dtype=np.float64)
                gate_reason = "accepted"
            else:
                T = np.asarray(T0, dtype=np.float64)
                gate_reason = "rollback_to_geometry"

            gate_meta["delta_pin_chamfer_mm"] = float(d_pin_ch)
            gate_meta["delta_pin_bidir_p95_mm"] = float(d_pin_p95)
            gate_meta["delta_base_p95_mm"] = float(d_base_p95)
            gate_meta["accept"] = bool(accept)
            gate_meta["improved"] = bool(improved)
        else:
            gate_meta["accept"] = False
            gate_meta["improved"] = False

        meta["overall_stage"] = dict(m0)
        meta["rgb_refine_stage"] = dict(m1)
        meta["rgb_gate"] = dict(gate_meta)
        meta["rgb_gate_reason"] = str(gate_reason)
        meta["voxel_overall_u"] = float(voxel_overall_u)
        meta["voxel_overall_mm"] = _mm_from_u(float(voxel_overall_u), cfg.unit_scale)
        meta["rgb_voxel_scales_u"] = [float(v) for v in rgb_voxel_scales_u]
        meta["rgb_voxel_scales_mm"] = [float(v) for v in cfg.rgb_voxel_scales_mm]
        meta["rgb_max_iters"] = [int(v) for v in cfg.rgb_max_iters]
        meta["rgb_lambda_geometric"] = float(cfg.rgb_lambda_geometric)
        meta["rgb_max_corr_factor"] = float(cfg.rgb_max_corr_factor)

        elapsed_acc = float(m0.get("elapsed_s", 0.0)) + float(m1.get("elapsed_s", 0.0))
        if bool(cfg.rgb_final_base_refine):
            voxel_base_u = _voxel_for(ref_pcds["base"])
            T2, m2 = _refine_icp_point_to_plane(
                source=src_pcds["base"],
                target=ref_pcds["base"],
                init_T=T,
                voxel_u=float(voxel_base_u),
            )
            T = np.asarray(T2, dtype=np.float64)
            meta["base_refine_stage"] = dict(m2)
            meta["voxel_base_u"] = float(voxel_base_u)
            meta["voxel_base_mm"] = _mm_from_u(float(voxel_base_u), cfg.unit_scale)
            meta["icp_fitness"] = float(m2.get("icp_fitness", float("nan")))
            meta["icp_inlier_rmse_u"] = float(m2.get("icp_inlier_rmse_u", float("nan")))
            elapsed_acc += float(m2.get("elapsed_s", 0.0))
        else:
            meta["icp_fitness"] = float(m1.get("final_fitness", float("nan")))
            meta["icp_inlier_rmse_u"] = float(m1.get("final_inlier_rmse_u", float("nan")))

        meta["elapsed_s"] = float(elapsed_acc)
        _attach_transform_contract(meta, init_T=T0, final_T=T)
        return T, meta

    if method in ("norgb_enhanced_v1", "robust_geom_v1"):
        voxel_overall_u = _voxel_for(ref_pcds["overall"])
        voxel_base_u = _voxel_for(ref_pcds["base"])
        robust_k_u = _u_from_mm(cfg.norgb_robust_kernel_k_mm, cfg.unit_scale)

        candidates: List[Dict[str, Any]] = []

        def _add_candidate(name: str, T_init: np.ndarray, stage_meta: Dict[str, Any], stage_elapsed: float) -> None:
            T_cur = np.asarray(T_init, dtype=np.float64)
            local_steps: List[Dict[str, Any]] = []

            if bool(cfg.norgb_enable_gicp_refine):
                T_cur, m_gicp = _refine_gicp(
                    source=src_pcds["overall"],
                    target=ref_pcds["overall"],
                    init_T=T_cur,
                    voxel_u=float(voxel_overall_u),
                    robust_k_u=float(robust_k_u),
                )
                local_steps.append({"name": "gicp_overall", **dict(m_gicp)})

            T_cur, m_base = _refine_icp_point_to_plane_robust(
                source=src_pcds["base"],
                target=ref_pcds["base"],
                init_T=T_cur,
                voxel_u=float(voxel_base_u),
                robust_k_u=float(robust_k_u),
            )
            local_steps.append({"name": "robust_icp_base", **dict(m_base)})

            align = _compute_alignment_metrics(
                T=T_cur,
                ref_cloud=ref_cloud,
                src_cloud=src_cloud,
                unit_scale=float(cfg.unit_scale),
                pin_inlier_thresholds_mm=cfg.pin_inlier_thresholds_mm,
            )
            score = _alignment_score_mm(
                align,
                w_pin_chamfer=float(cfg.norgb_score_w_pin_chamfer),
                w_pin_bidir_p95=float(cfg.norgb_score_w_pin_bidir_p95),
                w_base_p95=float(cfg.norgb_score_w_base_p95),
            )

            elapsed_local = stage_elapsed + sum(float(s.get("elapsed_s", 0.0)) for s in local_steps)
            candidates.append(
                {
                    "name": str(name),
                    "init_transform": np.asarray(T_init, dtype=np.float64),
                    "init_stage": dict(stage_meta),
                    "refine_stages": local_steps,
                    "T": T_cur,
                    "alignment": align,
                    "score_mm": float(score),
                    "elapsed_s": float(elapsed_local),
                }
            )

        T_ransac, m_ransac = _register_fpfh_ransac_then_icp_v1(
            source=src_pcds["overall"],
            target=ref_pcds["overall"],
            voxel_u=float(voxel_overall_u),
            ransac_max_iter=int(cfg.ransac_max_iter),
            ransac_confidence=float(cfg.ransac_confidence),
        )
        _add_candidate("ransac_init", T_ransac, {"name": "ransac_overall", **dict(m_ransac)}, float(m_ransac.get("elapsed_s", 0.0)))

        if bool(cfg.norgb_enable_fgr_init):
            try:
                T_fgr, m_fgr = _register_fpfh_fgr_then_icp_v1(
                    source=src_pcds["overall"],
                    target=ref_pcds["overall"],
                    voxel_u=float(voxel_overall_u),
                )
                _add_candidate(
                    "fgr_init",
                    T_fgr,
                    {"name": "fgr_overall", **dict(m_fgr)},
                    float(m_fgr.get("elapsed_s", 0.0)),
                )
            except Exception as e:
                candidates.append(
                    {
                        "name": "fgr_init",
                        "status": "failed",
                        "error": str(e),
                        "score_mm": float("inf"),
                    }
                )

        valid = [c for c in candidates if np.isfinite(float(c.get("score_mm", float("inf"))))]
        if not valid:
            raise RuntimeError("norgb_enhanced_v1: no valid candidate")

        best = min(valid, key=lambda c: float(c.get("score_mm", float("inf"))) )
        T_best = np.asarray(best.get("T"), dtype=np.float64)
        if T_best.shape != (4, 4) or (not np.all(np.isfinite(T_best))):
            raise RuntimeError("norgb_enhanced_v1: invalid best transformation")

        meta["candidates"] = [
            {
                "name": str(c.get("name", "")),
                "status": str(c.get("status", "ok")),
                "score_mm": float(c.get("score_mm", float("inf"))),
                "elapsed_s": float(c.get("elapsed_s", 0.0)) if np.isfinite(float(c.get("elapsed_s", 0.0))) else float("nan"),
                "alignment": c.get("alignment", {}),
                "init_stage": c.get("init_stage", {}),
                "refine_stages": c.get("refine_stages", []),
                "error": c.get("error", ""),
            }
            for c in candidates
        ]
        meta["selected_candidate"] = str(best.get("name", ""))
        meta["selected_score_mm"] = float(best.get("score_mm", float("inf")))
        meta["voxel_overall_u"] = float(voxel_overall_u)
        meta["voxel_overall_mm"] = _mm_from_u(float(voxel_overall_u), cfg.unit_scale)
        meta["voxel_base_u"] = float(voxel_base_u)
        meta["voxel_base_mm"] = _mm_from_u(float(voxel_base_u), cfg.unit_scale)
        meta["robust_kernel_k_mm"] = float(cfg.norgb_robust_kernel_k_mm)
        meta["robust_kernel_k_u"] = float(robust_k_u)
        meta["enable_fgr_init"] = bool(cfg.norgb_enable_fgr_init)
        meta["enable_gicp_refine"] = bool(cfg.norgb_enable_gicp_refine)
        meta["score_weights"] = {
            "pin_chamfer": float(cfg.norgb_score_w_pin_chamfer),
            "pin_bidir_p95": float(cfg.norgb_score_w_pin_bidir_p95),
            "base_p95": float(cfg.norgb_score_w_base_p95),
        }
        meta["elapsed_s"] = float(sum(float(c.get("elapsed_s", 0.0)) for c in valid))
        _attach_transform_contract(meta, init_T=best.get("init_transform", T_best), final_T=T_best)
        return T_best, meta

    raise RuntimeError(f"unknown method: {cfg.method}")


def run_registration(method: str, daima_ctx: Any = None) -> int:
    cfg = _build_config(method)

    if not cfg.cloud_dir.exists():
        raise RuntimeError(f"cloud_dir not found: {cfg.cloud_dir}")

    ref_ply = (cfg.cloud_dir / f"{cfg.ref_id}.ply").resolve()
    src_ply = (cfg.cloud_dir / f"{cfg.source_id}.ply").resolve()

    if not ref_ply.exists():
        raise RuntimeError(f"ref ply not found: {ref_ply}")
    if not src_ply.exists():
        raise RuntimeError(f"source ply not found: {src_ply}")

    route_tag = str(cfg.route_tag).strip()
    if bool(cfg.require_route_tag) and (not route_tag):
        raise RuntimeError("route_tag_required_but_missing: set DAIMA_ROUTE_TAG")

    run_dir: Optional[Path] = None
    if daima_ctx is not None:
        raw_run_dir = str(getattr(daima_ctx, "run_dir", "")).strip()
        if raw_run_dir:
            run_dir = Path(raw_run_dir)

    fallback_runs_root = Path("debug_out").resolve() / "runs"
    if run_dir is not None:
        run_root = run_dir
    elif bool(cfg.require_route_tag):
        run_root = fallback_runs_root / "ROUTE_TAG_ONLY_FALLBACK"
    else:
        run_root = fallback_runs_root / "NO_RUN_DIR"

    out_root_base = run_root / "registration"
    if route_tag:
        out_root_base = out_root_base / route_tag
    out_root_base_text = str(out_root_base).replace("\\", "/")
    if bool(cfg.require_route_tag) and ("NO_RUN_DIR" in out_root_base_text):
        raise RuntimeError(f"route_tag_required_but_output_in_no_run_dir: out_root_base={out_root_base}")
    if bool(cfg.require_route_tag) and (route_tag not in out_root_base_text):
        raise RuntimeError(f"route_tag_required_but_not_in_output_path: out_root_base={out_root_base}")
    out_root = out_root_base / str(cfg.method) / f"ref_{cfg.ref_id}__src_{cfg.source_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    if daima_ctx is not None:
        daima_ctx.snapshot_paths(
            inputs={
                "cloud_dir": str(cfg.cloud_dir),
                "ref_ply": str(ref_ply),
                "source_ply": str(src_ply),
            },
            outputs={"out_root": str(out_root)},
        )
        daima_ctx.snapshot_config(
            config={
                "method": str(cfg.method),
                "ref_id": str(cfg.ref_id),
                "source_id": str(cfg.source_id),
                "unit_scale": float(cfg.unit_scale),
                "target_grid_points": int(cfg.target_grid_points),
                "min_voxel_mm": float(cfg.min_voxel_mm),
                "max_voxel_mm": float(cfg.max_voxel_mm),
                "ransac_max_iter": int(cfg.ransac_max_iter),
                "ransac_confidence": float(cfg.ransac_confidence),
                "write_overlay": bool(cfg.write_overlay),
                "overlay_downsample_mm": float(cfg.overlay_downsample_mm),
                "pin_inlier_thresholds_mm": [float(v) for v in cfg.pin_inlier_thresholds_mm],
                "rgb_voxel_scales_mm": [float(v) for v in cfg.rgb_voxel_scales_mm],
                "rgb_max_iters": [int(v) for v in cfg.rgb_max_iters],
                "rgb_lambda_geometric": float(cfg.rgb_lambda_geometric),
                "rgb_max_corr_factor": float(cfg.rgb_max_corr_factor),
                "rgb_final_base_refine": bool(cfg.rgb_final_base_refine),
                "rgb_gate_enable": bool(cfg.rgb_gate_enable),
                "rgb_gate_min_init_fitness": float(cfg.rgb_gate_min_init_fitness),
                "rgb_gate_min_color_std": float(cfg.rgb_gate_min_color_std),
                "rgb_gate_max_pin_chamfer_degrade_mm": float(cfg.rgb_gate_max_pin_chamfer_degrade_mm),
                "rgb_gate_max_pin_bidir_p95_degrade_mm": float(cfg.rgb_gate_max_pin_bidir_p95_degrade_mm),
                "rgb_gate_max_base_p95_degrade_mm": float(cfg.rgb_gate_max_base_p95_degrade_mm),
                "rgb_gate_min_improve_mm": float(cfg.rgb_gate_min_improve_mm),
                "norgb_enable_fgr_init": bool(cfg.norgb_enable_fgr_init),
                "norgb_enable_gicp_refine": bool(cfg.norgb_enable_gicp_refine),
                "norgb_robust_kernel_k_mm": float(cfg.norgb_robust_kernel_k_mm),
                "norgb_score_w_pin_chamfer": float(cfg.norgb_score_w_pin_chamfer),
                "norgb_score_w_pin_bidir_p95": float(cfg.norgb_score_w_pin_bidir_p95),
                "norgb_score_w_base_p95": float(cfg.norgb_score_w_base_p95),
                "init_backend": str(cfg.init_backend),
                "init_backend_allow_fallback": bool(cfg.init_backend_allow_fallback),
                "local_refine_multiscale_enable": bool(cfg.local_refine_multiscale_enable),
                "local_refine_voxel_scales_mm": [float(v) for v in cfg.local_refine_voxel_scales_mm],
                "local_refine_max_iters": [int(v) for v in cfg.local_refine_max_iters],
                "local_refine_max_corr_factor": float(cfg.local_refine_max_corr_factor),
                "local_refine_robust_enable": bool(cfg.local_refine_robust_enable),
                "local_refine_robust_kernel": str(cfg.local_refine_robust_kernel),
                "local_refine_robust_k_mm": float(cfg.local_refine_robust_k_mm),
                "bcjr_enable": bool(cfg.bcjr_enable),
                "bcjr_voxel_scales_mm": [float(v) for v in cfg.bcjr_voxel_scales_mm],
                "bcjr_max_iters": [int(v) for v in cfg.bcjr_max_iters],
                "bcjr_lambda_base": [float(v) for v in cfg.bcjr_lambda_base],
                "bcjr_lambda_overall": [float(v) for v in cfg.bcjr_lambda_overall],
                "bcjr_prior_mu": [float(v) for v in cfg.bcjr_prior_mu],
                "bcjr_weak_mode_count": int(cfg.bcjr_weak_mode_count),
                "bcjr_q_min": float(cfg.bcjr_q_min),
                "bcjr_lm_damping": float(cfg.bcjr_lm_damping),
                "bcjr_exclude_base_overlap_factor": float(cfg.bcjr_exclude_base_overlap_factor),
                "bcjr_beta_bias": float(cfg.bcjr_beta_bias),
                "bcjr_beta_r": float(cfg.bcjr_beta_r),
                "bcjr_beta_n": float(cfg.bcjr_beta_n),
                "bcjr_beta_u": float(cfg.bcjr_beta_u),
                "bcjr_beta_g": float(cfg.bcjr_beta_g),
                "bcjr_clip_translation_factor": float(cfg.bcjr_clip_translation_factor),
                "bcjr_clip_rotation_deg": [float(v) for v in cfg.bcjr_clip_rotation_deg],
                "bcjr_rescue_enable": bool(cfg.bcjr_rescue_enable),
                "bcjr_rescue_rotation_deg": float(cfg.bcjr_rescue_rotation_deg),
                "bcjr_rescue_verify_voxel_mm": float(cfg.bcjr_rescue_verify_voxel_mm),
                "bcjr_rescue_verify_max_corr_factor": float(cfg.bcjr_rescue_verify_max_corr_factor),
                "bcjr_rescue_verify_base_guard_mm": float(cfg.bcjr_rescue_verify_base_guard_mm),
                "bcjr_rescue_detail_rmse_tie_mm": float(cfg.bcjr_rescue_detail_rmse_tie_mm),
                "pair_id": str(cfg.pair_id),
                "require_route_tag": bool(cfg.require_route_tag),
                "route_tag": str(cfg.route_tag),
            }
        )

    print(f"[INFO] method={cfg.method}")
    print(f"[INFO] ref={cfg.ref_id} source={cfg.source_id}")
    print(f"[INFO] cloud_dir={cfg.cloud_dir}")
    if route_tag:
        print(f"[INFO] route_tag={route_tag}")
    if str(cfg.pair_id).strip():
        print(f"[INFO] pair_id={cfg.pair_id}")

    ref_cloud = _read_labeled_cloud(ref_ply)
    src_cloud = _read_labeled_cloud(src_ply)

    t0 = time.time()
    T, reg_meta = _compute_transform(cfg, ref_cloud, src_cloud)
    elapsed = float(time.time() - t0)

    alignment_stats = _compute_alignment_metrics(
        T=T,
        ref_cloud=ref_cloud,
        src_cloud=src_cloud,
        unit_scale=float(cfg.unit_scale),
        pin_inlier_thresholds_mm=cfg.pin_inlier_thresholds_mm,
    )
    base_stats = alignment_stats.get("base", {})
    overall_stats = alignment_stats.get("overall", {})
    pin_stats = alignment_stats.get("pins", {})

    overlay_paths: Dict[str, str] = {}
    if bool(cfg.write_overlay):
        down_u = _u_from_mm(cfg.overlay_downsample_mm, cfg.unit_scale)
        try:
            p_base = out_root / "overlay_base.ply"
            _write_overlay(
                p_base,
                target_pts=ref_cloud.base_points,
                source_pts=src_cloud.base_points,
                T=T,
                downsample_u=float(down_u),
                target_color=(0.55, 0.55, 0.55),
                source_color=(0.10, 0.85, 1.00),
            )
            overlay_paths["base"] = str(p_base)
        except Exception as e:
            overlay_paths["base_error"] = str(e)

        try:
            p_overall = out_root / "overlay_overall.ply"
            _write_overlay(
                p_overall,
                target_pts=ref_cloud.overall_points,
                source_pts=src_cloud.overall_points,
                T=T,
                downsample_u=float(down_u),
                target_color=(0.40, 0.40, 0.40),
                source_color=(1.00, 0.15, 0.15),
            )
            overlay_paths["overall"] = str(p_overall)
        except Exception as e:
            overlay_paths["overall_error"] = str(e)

        try:
            p_pins = out_root / "overlay_pins.ply"
            _write_overlay(
                p_pins,
                target_pts=ref_cloud.pins_points,
                source_pts=src_cloud.pins_points,
                T=T,
                downsample_u=float(down_u),
                target_color=(1.00, 0.85, 0.10),
                source_color=(0.10, 0.95, 0.35),
            )
            overlay_paths["pins"] = str(p_pins)
        except Exception as e:
            overlay_paths["pins_error"] = str(e)

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": str(cfg.method),
        "ref_id": str(cfg.ref_id),
        "source_id": str(cfg.source_id),
        "pair_id": str(cfg.pair_id),
        "route_tag": str(route_tag),
        "artifact_identity": {
            "method": str(cfg.method),
            "pair_id": str(cfg.pair_id),
            "ref_id": str(cfg.ref_id),
            "source_id": str(cfg.source_id),
            "route_tag": str(route_tag),
            "cloud_dir": str(cfg.cloud_dir),
        },
        "paths": {
            "cloud_dir": str(cfg.cloud_dir),
            "ref_ply": str(ref_ply),
            "source_ply": str(src_ply),
            "out_root": str(out_root),
        },
        "config": {
            "unit_scale": float(cfg.unit_scale),
            "target_grid_points": int(cfg.target_grid_points),
            "min_voxel_mm": float(cfg.min_voxel_mm),
            "max_voxel_mm": float(cfg.max_voxel_mm),
            "ransac_max_iter": int(cfg.ransac_max_iter),
            "ransac_confidence": float(cfg.ransac_confidence),
            "overlay_downsample_mm": float(cfg.overlay_downsample_mm),
            "pin_inlier_thresholds_mm": [float(v) for v in cfg.pin_inlier_thresholds_mm],
            "rgb_voxel_scales_mm": [float(v) for v in cfg.rgb_voxel_scales_mm],
            "rgb_max_iters": [int(v) for v in cfg.rgb_max_iters],
            "rgb_lambda_geometric": float(cfg.rgb_lambda_geometric),
            "rgb_max_corr_factor": float(cfg.rgb_max_corr_factor),
            "rgb_final_base_refine": bool(cfg.rgb_final_base_refine),
            "rgb_gate_enable": bool(cfg.rgb_gate_enable),
            "rgb_gate_min_init_fitness": float(cfg.rgb_gate_min_init_fitness),
            "rgb_gate_min_color_std": float(cfg.rgb_gate_min_color_std),
            "rgb_gate_max_pin_chamfer_degrade_mm": float(cfg.rgb_gate_max_pin_chamfer_degrade_mm),
            "rgb_gate_max_pin_bidir_p95_degrade_mm": float(cfg.rgb_gate_max_pin_bidir_p95_degrade_mm),
            "rgb_gate_max_base_p95_degrade_mm": float(cfg.rgb_gate_max_base_p95_degrade_mm),
            "rgb_gate_min_improve_mm": float(cfg.rgb_gate_min_improve_mm),
            "norgb_enable_fgr_init": bool(cfg.norgb_enable_fgr_init),
            "norgb_enable_gicp_refine": bool(cfg.norgb_enable_gicp_refine),
            "norgb_robust_kernel_k_mm": float(cfg.norgb_robust_kernel_k_mm),
            "norgb_score_w_pin_chamfer": float(cfg.norgb_score_w_pin_chamfer),
            "norgb_score_w_pin_bidir_p95": float(cfg.norgb_score_w_pin_bidir_p95),
            "norgb_score_w_base_p95": float(cfg.norgb_score_w_base_p95),
            "init_backend": str(cfg.init_backend),
            "init_backend_allow_fallback": bool(cfg.init_backend_allow_fallback),
            "local_refine_multiscale_enable": bool(cfg.local_refine_multiscale_enable),
            "local_refine_voxel_scales_mm": [float(v) for v in cfg.local_refine_voxel_scales_mm],
            "local_refine_max_iters": [int(v) for v in cfg.local_refine_max_iters],
            "local_refine_max_corr_factor": float(cfg.local_refine_max_corr_factor),
            "local_refine_robust_enable": bool(cfg.local_refine_robust_enable),
            "local_refine_robust_kernel": str(cfg.local_refine_robust_kernel),
            "local_refine_robust_k_mm": float(cfg.local_refine_robust_k_mm),
            "bcjr_enable": bool(cfg.bcjr_enable),
            "bcjr_voxel_scales_mm": [float(v) for v in cfg.bcjr_voxel_scales_mm],
            "bcjr_max_iters": [int(v) for v in cfg.bcjr_max_iters],
            "bcjr_lambda_base": [float(v) for v in cfg.bcjr_lambda_base],
            "bcjr_lambda_overall": [float(v) for v in cfg.bcjr_lambda_overall],
            "bcjr_prior_mu": [float(v) for v in cfg.bcjr_prior_mu],
            "bcjr_weak_mode_count": int(cfg.bcjr_weak_mode_count),
            "bcjr_q_min": float(cfg.bcjr_q_min),
            "bcjr_lm_damping": float(cfg.bcjr_lm_damping),
            "bcjr_exclude_base_overlap_factor": float(cfg.bcjr_exclude_base_overlap_factor),
            "bcjr_beta_bias": float(cfg.bcjr_beta_bias),
            "bcjr_beta_r": float(cfg.bcjr_beta_r),
            "bcjr_beta_n": float(cfg.bcjr_beta_n),
            "bcjr_beta_u": float(cfg.bcjr_beta_u),
            "bcjr_beta_g": float(cfg.bcjr_beta_g),
            "bcjr_clip_translation_factor": float(cfg.bcjr_clip_translation_factor),
            "bcjr_clip_rotation_deg": [float(v) for v in cfg.bcjr_clip_rotation_deg],
            "bcjr_rescue_enable": bool(cfg.bcjr_rescue_enable),
            "bcjr_rescue_rotation_deg": float(cfg.bcjr_rescue_rotation_deg),
            "bcjr_rescue_verify_voxel_mm": float(cfg.bcjr_rescue_verify_voxel_mm),
            "bcjr_rescue_verify_max_corr_factor": float(cfg.bcjr_rescue_verify_max_corr_factor),
            "bcjr_rescue_verify_base_guard_mm": float(cfg.bcjr_rescue_verify_base_guard_mm),
            "bcjr_rescue_detail_rmse_tie_mm": float(cfg.bcjr_rescue_detail_rmse_tie_mm),
            "pair_id": str(cfg.pair_id),
            "require_route_tag": bool(cfg.require_route_tag),
            "route_tag": str(cfg.route_tag),
        },
        "input_meta": {
            "ref": ref_cloud.meta,
            "source": src_cloud.meta,
        },
        "registration": reg_meta,
        "alignment_stats": alignment_stats,
        "elapsed_total_s": elapsed,
        "transform_source_to_ref": np.asarray(T, dtype=float),
        "overlays": overlay_paths,
    }

    _write_json(out_root / "metrics.json", result)
    _write_json(
        out_root / "transform.json",
        {
            "method": str(cfg.method),
            "pair_id": str(cfg.pair_id),
            "ref_id": str(cfg.ref_id),
            "source_id": str(cfg.source_id),
            "route_tag": str(route_tag),
            "artifact_identity": {
                "method": str(cfg.method),
                "pair_id": str(cfg.pair_id),
                "ref_id": str(cfg.ref_id),
                "source_id": str(cfg.source_id),
                "route_tag": str(route_tag),
                "cloud_dir": str(cfg.cloud_dir),
            },
            "transform_source_to_ref": np.asarray(T, dtype=float),
        },
    )

    print(f"[INFO] wrote: {out_root / 'metrics.json'}")
    print(f"[INFO] wrote: {out_root / 'transform.json'}")
    print(
        "[INFO] pin_metrics: "
        f"src2ref_p95_mm={float(pin_stats.get('source_to_ref', {}).get('p95_mm', float('nan'))):.4f}, "
        f"ref2src_p95_mm={float(pin_stats.get('ref_to_source', {}).get('p95_mm', float('nan'))):.4f}, "
        f"chamfer_mean_mm={float(pin_stats.get('chamfer_mean_mm', float('nan'))):.4f}"
    )
    if overlay_paths:
        print(f"[INFO] overlays: {overlay_paths}")
    return 0
