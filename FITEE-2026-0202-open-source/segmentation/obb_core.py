from __future__ import annotations

import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None  # type: ignore[assignment]

try:
    from ply_utils import read_ply_vertex
except Exception:  # pragma: no cover
    read_ply_vertex = None  # type: ignore[assignment]

_B4_MM_TO_UNIT = 1e-3

_B4_REG_TARGET_GRID_POINTS = 50000

_B4_REG_MIN_VOXEL_MM = 0.2

_B4_REG_MAX_VOXEL_MM = 1.0

_B4_REG_RANSAC_MAX_ITER = 400000

_B4_REG_RANSAC_CONFIDENCE = 0.999

_B4_REG_RANSAC_DIST_MULT = 1.5

_B4_REG_ICP_DIST_MULT = 0.4

_B4_TEMPLATE_CACHE: Dict[str, np.ndarray] = {}

def _as_float(name: str, v: object, default: float) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Invalid float param {name}={v!r}") from e

def _as_int(name: str, v: object, default: int) -> int:
    if v is None:
        return int(default)
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Invalid int param {name}={v!r}") from e

def _as_str(name: str, v: object, default: str) -> str:
    if v is None:
        return str(default)
    return str(v)

def _as_bool(name: str, v: object, default: bool) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return bool(int(v) != 0)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool param {name}={v!r}")

def _threshold_quantile(pin_prob: np.ndarray, *, target_pin_frac: float) -> Tuple[float, Dict[str, float]]:
    p = np.asarray(pin_prob, dtype=np.float64)
    target_pin_frac = float(target_pin_frac)
    target_pin_frac = float(np.clip(target_pin_frac, 1e-6, 1.0 - 1e-6))
    q = 1.0 - target_pin_frac
    thr = float(np.quantile(p, q))
    dbg = {
        "pp_target_pin_frac": float(target_pin_frac),
        "pp_quantile_q": float(q),
        "pp_thr": float(thr),
    }
    return thr, dbg

def _b4_rot_matrix_xyz_deg(rot_deg: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot_deg, dtype=np.float64).reshape(3)
    rx, ry, rz = np.deg2rad(rot)
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    rx_m = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m

def _b4_estimate_voxel_size(points_xyz: np.ndarray) -> float:
    pts = np.asarray(points_xyz, dtype=np.float64)
    min_u = float(_B4_REG_MIN_VOXEL_MM * _B4_MM_TO_UNIT)
    max_u = float(_B4_REG_MAX_VOXEL_MM * _B4_MM_TO_UNIT)
    if pts.ndim != 2 or int(pts.shape[1]) != 3 or int(pts.shape[0]) <= 1:
        return min_u
    diag = float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)))
    voxel = float(diag * (float(int(pts.shape[0])) / float(_B4_REG_TARGET_GRID_POINTS)) ** (1.0 / 3.0))
    if not np.isfinite(voxel) or voxel <= 0:
        voxel = min_u
    return float(np.clip(voxel, min_u, max_u))

def _b4_attempt_voxel(base_voxel_u: float, attempt_idx: int) -> float:
    factors = [1.00, 0.92, 1.08, 0.85, 1.15, 0.96, 1.04, 0.88, 1.12, 1.00]
    factor = float(factors[(int(attempt_idx) - 1) % len(factors)])
    min_u = float(_B4_REG_MIN_VOXEL_MM * _B4_MM_TO_UNIT)
    max_u = float(_B4_REG_MAX_VOXEL_MM * _B4_MM_TO_UNIT)
    return float(np.clip(float(base_voxel_u) * factor, min_u, max_u))

def _b4_ensure_normals(pcd: "o3d.geometry.PointCloud", radius_u: float) -> None:
    if o3d is None:
        return
    if len(pcd.points) <= 0:
        return
    if (not pcd.has_normals()) or (len(pcd.normals) != len(pcd.points)):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius_u), max_nn=30)
        )

def _b4_preprocess_fpfh(
    pcd: "o3d.geometry.PointCloud",
    voxel_u: float,
) -> Tuple["o3d.geometry.PointCloud", "o3d.pipelines.registration.Feature"]:
    pcd_down = pcd.voxel_down_sample(float(voxel_u))
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 2.0, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 5.0, max_nn=100),
    )
    return pcd_down, pcd_fpfh

def _b4_icp_gpu_point_to_plane(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    init_T: np.ndarray,
    max_corr: float,
    voxel_u: float,
) -> Tuple[np.ndarray, float, float]:
    if o3d is None:
        raise RuntimeError("Open3D unavailable")
    if not hasattr(o3d, "core") or not hasattr(o3d.core, "cuda") or (not o3d.core.cuda.is_available()):
        raise RuntimeError("CUDA not available")
    if not hasattr(o3d, "t") or (not hasattr(o3d.t, "pipelines")) or (not hasattr(o3d.t.pipelines, "registration")):
        raise RuntimeError("Open3D tensor registration API missing")

    _b4_ensure_normals(source, radius_u=float(voxel_u) * 2.0)
    _b4_ensure_normals(target, radius_u=float(voxel_u) * 2.0)

    device = o3d.core.Device("CUDA:0")
    t_source = o3d.t.geometry.PointCloud.from_legacy(source, o3d.core.Dtype.Float32, device)
    t_target = o3d.t.geometry.PointCloud.from_legacy(target, o3d.core.Dtype.Float32, device)
    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    init_tensor = o3d.core.Tensor(np.asarray(init_T, dtype=np.float32), device=device)

    res = o3d.t.pipelines.registration.icp(
        t_source,
        t_target,
        float(max_corr),
        init_tensor,
        estimation,
        criteria,
    )
    T = np.asarray(res.transformation.cpu().numpy(), dtype=np.float64)
    fit = float(res.fitness)
    rmse = float(res.inlier_rmse)
    return T, fit, rmse

def _b4_ransac_init_once(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    voxel_u: float,
) -> Dict[str, object]:
    if o3d is None:
        raise RuntimeError("Open3D unavailable")
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.asarray(source_pts, dtype=np.float64))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(target_pts, dtype=np.float64))

    s_down, s_fpfh = _b4_preprocess_fpfh(source, float(voxel_u))
    t_down, t_fpfh = _b4_preprocess_fpfh(target, float(voxel_u))
    if len(s_down.points) < 20 or len(t_down.points) < 20:
        raise RuntimeError("Insufficient downsampled points for RANSAC")

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down,
        t_down,
        s_fpfh,
        t_fpfh,
        True,
        float(voxel_u) * float(_B4_REG_RANSAC_DIST_MULT),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                float(voxel_u) * float(_B4_REG_RANSAC_DIST_MULT)
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(_B4_REG_RANSAC_MAX_ITER),
            float(_B4_REG_RANSAC_CONFIDENCE),
        ),
    )
    init_T = np.asarray(ransac.transformation, dtype=np.float64)
    if init_T.shape != (4, 4) or (not np.all(np.isfinite(init_T))):
        raise RuntimeError("Invalid RANSAC transformation matrix")

    return {
        "transformation": init_T,
        "ransac_fitness": float(getattr(ransac, "fitness", float("nan"))),
        "ransac_inlier_rmse": float(getattr(ransac, "inlier_rmse", float("nan"))),
    }

def _b4_refine_icp_once(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    voxel_u: float,
    init_T: np.ndarray,
    use_gpu_icp: bool,
) -> Dict[str, object]:
    if o3d is None:
        raise RuntimeError("Open3D unavailable")
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.asarray(source_pts, dtype=np.float64))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(target_pts, dtype=np.float64))
    max_corr = float(voxel_u) * float(_B4_REG_ICP_DIST_MULT)

    used_gpu = False
    gpu_fallback = ""
    if use_gpu_icp:
        try:
            T, fit, rmse = _b4_icp_gpu_point_to_plane(
                source=source,
                target=target,
                init_T=np.asarray(init_T, dtype=np.float64),
                max_corr=float(max_corr),
                voxel_u=float(voxel_u),
            )
            used_gpu = True
        except Exception as e:
            gpu_fallback = str(e)
            used_gpu = False
            T = np.eye(4, dtype=np.float64)
            fit = float("nan")
            rmse = float("nan")

    if (not use_gpu_icp) or (not used_gpu):
        _b4_ensure_normals(source, radius_u=float(voxel_u) * 2.0)
        _b4_ensure_normals(target, radius_u=float(voxel_u) * 2.0)
        icp = o3d.pipelines.registration.registration_icp(
            source,
            target,
            float(max_corr),
            np.asarray(init_T, dtype=np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        T = np.asarray(icp.transformation, dtype=np.float64)
        fit = float(icp.fitness)
        rmse = float(icp.inlier_rmse)

    if T.shape != (4, 4) or (not np.all(np.isfinite(T))):
        raise RuntimeError("Invalid ICP transformation matrix")

    return {
        "transformation": T,
        "icp_fitness": float(fit),
        "icp_inlier_rmse": float(rmse),
        "used_gpu_icp": 1 if used_gpu else 0,
        "gpu_fallback_error": gpu_fallback,
    }

def _b4_resolve_template_path(path_raw: str) -> Path:
    p = Path(str(path_raw).strip())
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((Path(__file__).resolve().parents[1] / p).resolve())
    for c in candidates:
        if c.is_file():
            return c
    return candidates[-1] if candidates else p

def _b4_load_template_base_points(
    template_path: Path,
    *,
    base_key: str,
    pin_key: str,
) -> np.ndarray:
    cache_key = f"{str(template_path.resolve())}|{base_key}|{pin_key}"
    cached = _B4_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)

    xyz: Optional[np.ndarray] = None
    base_mask: Optional[np.ndarray] = None
    if read_ply_vertex is not None:
        v = read_ply_vertex(template_path)
        xyz = np.stack(
            [
                np.asarray(v["x"], dtype=np.float64),
                np.asarray(v["y"], dtype=np.float64),
                np.asarray(v["z"], dtype=np.float64),
            ],
            axis=1,
        )
        if base_key in v:
            base_mask = np.isfinite(np.asarray(v[base_key], dtype=np.float64))
        elif pin_key in v:
            pin_mask = np.isfinite(np.asarray(v[pin_key], dtype=np.float64))
            base_mask = ~pin_mask
    elif o3d is not None:
        pcd = o3d.io.read_point_cloud(str(template_path))
        xyz = np.asarray(pcd.points, dtype=np.float64)

    if xyz is None or xyz.ndim != 2 or int(xyz.shape[1]) != 3 or int(xyz.shape[0]) <= 0:
        raise RuntimeError(f"Failed to load template points: {template_path}")

    if base_mask is not None and int(np.sum(base_mask)) > 0:
        pts = xyz[base_mask]
    else:
        pts = xyz
    if int(pts.shape[0]) <= 0:
        raise RuntimeError(f"Template has no usable points: {template_path}")
    _B4_TEMPLATE_CACHE[cache_key] = np.asarray(pts, dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)

def _b4_register_template_to_pred(
    *,
    template_base_pts: np.ndarray,
    target_base_pts: np.ndarray,
    max_retries: int,
    rmse_gate_mm: float,
    min_fitness: float,
    use_gpu_icp: bool,
    cpu_workers: int,
    parallel_ransac: bool,
) -> Dict[str, object]:
    attempts: List[Dict[str, object]] = []
    best_row: Optional[Dict[str, object]] = None
    best_T: Optional[np.ndarray] = None
    gpu_used_any = 0

    base_voxel = _b4_estimate_voxel_size(np.asarray(target_base_pts, dtype=np.float64))
    attempt_specs: List[Tuple[int, float]] = [
        (int(i), float(_b4_attempt_voxel(base_voxel, int(i))))
        for i in range(1, int(max_retries) + 1)
    ]

    init_rows: List[Dict[str, object]] = []
    if parallel_ransac and int(cpu_workers) > 1 and len(attempt_specs) > 1:
        max_workers = int(min(int(cpu_workers), len(attempt_specs)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(_b4_ransac_init_once, template_base_pts, target_base_pts, float(voxel_u)): (idx, voxel_u)
                for idx, voxel_u in attempt_specs
            }
            for fut in as_completed(fut_map):
                idx, voxel_u = fut_map[fut]
                row: Dict[str, object] = {
                    "attempt_idx": int(idx),
                    "voxel_u": float(voxel_u),
                    "status": "ok",
                    "error": "",
                    "init_T": None,
                    "ransac_fitness": float("nan"),
                    "ransac_inlier_rmse": float("nan"),
                }
                try:
                    meta = fut.result()
                    row["init_T"] = np.asarray(meta.get("transformation"), dtype=np.float64)
                    row["ransac_fitness"] = float(meta.get("ransac_fitness", float("nan")))
                    row["ransac_inlier_rmse"] = float(meta.get("ransac_inlier_rmse", float("nan")))
                except Exception as e:
                    row["status"] = "error"
                    row["error"] = str(e)
                init_rows.append(row)
    else:
        for idx, voxel_u in attempt_specs:
            row = {
                "attempt_idx": int(idx),
                "voxel_u": float(voxel_u),
                "status": "ok",
                "error": "",
                "init_T": None,
                "ransac_fitness": float("nan"),
                "ransac_inlier_rmse": float("nan"),
            }
            try:
                meta = _b4_ransac_init_once(template_base_pts, target_base_pts, float(voxel_u))
                row["init_T"] = np.asarray(meta.get("transformation"), dtype=np.float64)
                row["ransac_fitness"] = float(meta.get("ransac_fitness", float("nan")))
                row["ransac_inlier_rmse"] = float(meta.get("ransac_inlier_rmse", float("nan")))
            except Exception as e:
                row["status"] = "error"
                row["error"] = str(e)
            init_rows.append(row)

    init_rows.sort(key=lambda r: int(r.get("attempt_idx", 0)))

    def _rank_key(reg_row: Dict[str, object]) -> Tuple[int, float, float, float, int]:
        success = int(reg_row.get("is_success", 0))
        fitness = float(reg_row.get("icp_fitness", float("nan")))
        if not np.isfinite(fitness):
            fitness = float("-inf")
        rmse_mm = float(reg_row.get("icp_rmse_mm", float("nan")))
        if not np.isfinite(rmse_mm):
            rmse_mm = float("inf")
        # Maximize success + fitness, then minimize rmse; prefer stronger RANSAC and earlier attempt on ties.
        ransac_fit = float(reg_row.get("ransac_fitness", float("nan")))
        if not np.isfinite(ransac_fit):
            ransac_fit = float("-inf")
        attempt_idx_rank = -int(reg_row.get("attempt_idx", 10**9))
        return (success, fitness, -rmse_mm, ransac_fit, attempt_idx_rank)

    for row in init_rows:
        attempt_idx = int(row.get("attempt_idx", 0))
        voxel_u = float(row.get("voxel_u", float("nan")))
        out_row: Dict[str, object] = {
            "attempt_idx": attempt_idx,
            "voxel_u": voxel_u,
            "status": str(row.get("status", "error")),
            "error": str(row.get("error", "")),
            "ransac_fitness": float(row.get("ransac_fitness", float("nan"))),
            "ransac_inlier_rmse": float(row.get("ransac_inlier_rmse", float("nan"))),
            "icp_fitness": float("nan"),
            "icp_rmse_u": float("nan"),
            "icp_rmse_mm": float("nan"),
            "is_success": 0,
            "used_gpu_icp": 0,
            "gpu_fallback": "",
            "transformation": None,
        }
        if str(row.get("status", "error")) == "ok":
            try:
                icp_meta = _b4_refine_icp_once(
                    source_pts=template_base_pts,
                    target_pts=target_base_pts,
                    voxel_u=float(voxel_u),
                    init_T=np.asarray(row.get("init_T"), dtype=np.float64),
                    use_gpu_icp=bool(use_gpu_icp),
                )
                T = np.asarray(icp_meta.get("transformation"), dtype=np.float64)
                rmse_u = float(icp_meta.get("icp_inlier_rmse", float("nan")))
                rmse_mm = float(rmse_u / _B4_MM_TO_UNIT)
                fit = float(icp_meta.get("icp_fitness", float("nan")))
                rmse_ok = bool(np.isfinite(rmse_mm) and (rmse_mm <= float(rmse_gate_mm)))
                if float(min_fitness) > 0.0:
                    fit_ok = bool(np.isfinite(fit) and (fit >= float(min_fitness)))
                else:
                    fit_ok = True
                out_row["icp_fitness"] = fit
                out_row["icp_rmse_u"] = rmse_u
                out_row["icp_rmse_mm"] = rmse_mm
                out_row["used_gpu_icp"] = int(icp_meta.get("used_gpu_icp", 0))
                out_row["gpu_fallback"] = str(icp_meta.get("gpu_fallback_error", "") or "")
                out_row["is_success"] = 1 if (rmse_ok and fit_ok) else 0
                out_row["transformation"] = T
                gpu_used_any = max(gpu_used_any, int(out_row["used_gpu_icp"]))
            except Exception as e:
                out_row["status"] = "error"
                out_row["error"] = str(e)

        attempts.append(out_row)

        if out_row["status"] == "ok":
            if best_row is None:
                best_row = dict(out_row)
                best_T = np.asarray(out_row.get("transformation"), dtype=np.float64)
            else:
                if _rank_key(out_row) > _rank_key(best_row):
                    best_row = dict(out_row)
                    best_T = np.asarray(out_row.get("transformation"), dtype=np.float64)

    success = bool(best_row is not None and int(best_row.get("is_success", 0)) == 1)
    return {
        "attempt_rows": attempts,
        "best_row": best_row,
        "best_T": best_T,
        "success": 1 if success else 0,
        "gpu_used_any": int(gpu_used_any),
    }

def _b4_apply_manual_base_boxes(
    *,
    pred_pin: np.ndarray,
    points_xyz: Optional[np.ndarray],
    params: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, float]]:
    dbg: Dict[str, float] = {}
    pred = np.asarray(pred_pin, dtype=bool)

    enable = _as_bool("b4_base_box_enable", params.get("b4_base_box_enable"), False)
    dbg["pp_b4_base_box_enable"] = 1.0 if enable else 0.0
    if not enable:
        return pred, dbg
    if points_xyz is None:
        dbg["pp_b4_base_box_skip_no_points_xyz"] = 1.0
        return pred, dbg

    xyz = np.asarray(points_xyz, dtype=np.float64)
    if xyz.ndim != 2 or int(xyz.shape[1]) != 3:
        dbg["pp_b4_base_box_skip_invalid_points_shape"] = 1.0
        return pred, dbg
    if int(xyz.shape[0]) != int(pred.shape[0]):
        dbg["pp_b4_base_box_skip_points_pred_mismatch"] = 1.0
        dbg["pp_b4_base_box_points_n"] = float(int(xyz.shape[0]))
        dbg["pp_b4_base_box_pred_n"] = float(int(pred.shape[0]))
        return pred, dbg

    boxes_raw = params.get("b4_base_boxes_m", params.get("box_configs_m"))
    if boxes_raw is None:
        dbg["pp_b4_base_box_skip_no_boxes"] = 1.0
        return pred, dbg
    if not isinstance(boxes_raw, (list, tuple)):
        dbg["pp_b4_base_box_skip_invalid_boxes_type"] = 1.0
        return pred, dbg

    margin_m = float(max(0.0, _as_float("b4_base_box_margin_m", params.get("b4_base_box_margin_m"), 0.0)))
    reg_enable = _as_bool("b4_reg_enable", params.get("b4_reg_enable"), True)
    reg_require_success = _as_bool("b4_reg_require_success", params.get("b4_reg_require_success"), True)
    reg_max_retries = max(1, _as_int("b4_reg_max_retries", params.get("b4_reg_max_retries"), 20))
    reg_rmse_gate_mm = max(0.0, _as_float("b4_reg_rmse_gate_mm", params.get("b4_reg_rmse_gate_mm"), 0.6))
    reg_min_fitness = max(0.0, _as_float("b4_reg_min_fitness", params.get("b4_reg_min_fitness"), 0.0))
    reg_use_gpu_icp = _as_bool("b4_reg_use_gpu_icp", params.get("b4_reg_use_gpu_icp"), True)
    reg_cpu_workers = max(1, _as_int("b4_reg_cpu_workers", params.get("b4_reg_cpu_workers"), min(8, os.cpu_count() or 8)))
    reg_parallel_ransac = _as_bool("b4_reg_parallel_ransac", params.get("b4_reg_parallel_ransac"), True)
    reg_min_base_points = max(30, _as_int("b4_reg_min_base_points", params.get("b4_reg_min_base_points"), 200))
    reg_max_target_points = max(reg_min_base_points, _as_int("b4_reg_max_target_points", params.get("b4_reg_max_target_points"), 60000))
    reg_seed = _as_int("b4_reg_seed", params.get("b4_reg_seed"), 20260214)
    reg_template_path_raw = _as_str("b4_reg_template_path", params.get("b4_reg_template_path"), "point_cloud/70.ply").strip()
    reg_template_base_key = _as_str("b4_reg_template_base_key", params.get("b4_reg_template_base_key"), "scalar_base").strip()
    reg_template_pin_key = _as_str("b4_reg_template_pin_key", params.get("b4_reg_template_pin_key"), "scalar_pin").strip()

    dbg["pp_b4_reg_enable"] = 1.0 if reg_enable else 0.0
    dbg["pp_b4_reg_require_success"] = 1.0 if reg_require_success else 0.0
    dbg["pp_b4_reg_max_retries"] = float(reg_max_retries)
    dbg["pp_b4_reg_rmse_gate_mm"] = float(reg_rmse_gate_mm)
    dbg["pp_b4_reg_min_fitness"] = float(reg_min_fitness)
    dbg["pp_b4_reg_use_gpu_icp"] = 1.0 if reg_use_gpu_icp else 0.0
    dbg["pp_b4_reg_cpu_workers"] = float(reg_cpu_workers)
    dbg["pp_b4_reg_parallel_ransac"] = 1.0 if reg_parallel_ransac else 0.0

    reg_R = np.eye(3, dtype=np.float64)
    reg_t = np.zeros((3,), dtype=np.float64)
    if reg_enable:
        if o3d is None:
            dbg["pp_b4_reg_skip_no_open3d"] = 1.0
            if reg_require_success:
                return pred, dbg
        else:
            target_base_mask = ~pred
            n_target_base = int(np.sum(target_base_mask))
            dbg["pp_b4_reg_target_base_n_raw"] = float(n_target_base)
            if n_target_base < reg_min_base_points:
                dbg["pp_b4_reg_skip_target_too_few"] = 1.0
                if reg_require_success:
                    return pred, dbg
            else:
                target_base_pts = np.asarray(xyz[target_base_mask], dtype=np.float64)
                if int(target_base_pts.shape[0]) > reg_max_target_points:
                    rng = np.random.default_rng(int(reg_seed))
                    sel = rng.choice(int(target_base_pts.shape[0]), size=int(reg_max_target_points), replace=False)
                    target_base_pts = target_base_pts[np.asarray(sel, dtype=np.int64)]
                dbg["pp_b4_reg_target_base_n_used"] = float(int(target_base_pts.shape[0]))
                template_path = _b4_resolve_template_path(reg_template_path_raw)
                dbg["pp_b4_reg_template_path_exists"] = 1.0 if template_path.is_file() else 0.0
                try:
                    template_base_pts = _b4_load_template_base_points(
                        template_path,
                        base_key=reg_template_base_key,
                        pin_key=reg_template_pin_key,
                    )
                    dbg["pp_b4_reg_template_base_n"] = float(int(template_base_pts.shape[0]))
                    reg_meta = _b4_register_template_to_pred(
                        template_base_pts=np.asarray(template_base_pts, dtype=np.float64),
                        target_base_pts=np.asarray(target_base_pts, dtype=np.float64),
                        max_retries=int(reg_max_retries),
                        rmse_gate_mm=float(reg_rmse_gate_mm),
                        min_fitness=float(reg_min_fitness),
                        use_gpu_icp=bool(reg_use_gpu_icp),
                        cpu_workers=int(reg_cpu_workers),
                        parallel_ransac=bool(reg_parallel_ransac),
                    )
                    attempts = list(reg_meta.get("attempt_rows", []))
                    best_row = reg_meta.get("best_row")
                    best_T = reg_meta.get("best_T")
                    dbg["pp_b4_reg_attempts_used"] = float(len(attempts))
                    dbg["pp_b4_reg_gpu_used_any"] = float(int(reg_meta.get("gpu_used_any", 0)))
                    dbg["pp_b4_reg_success"] = float(int(reg_meta.get("success", 0)))
                    if best_row is not None:
                        dbg["pp_b4_reg_best_rmse_mm"] = float(best_row.get("icp_rmse_mm", float("nan")))
                        dbg["pp_b4_reg_best_fitness"] = float(best_row.get("icp_fitness", float("nan")))
                        dbg["pp_b4_reg_best_attempt_idx"] = float(int(best_row.get("attempt_idx", -1)))
                    if best_T is not None and np.asarray(best_T).shape == (4, 4):
                        T = np.asarray(best_T, dtype=np.float64)
                        reg_R = T[:3, :3]
                        reg_t = T[:3, 3]
                    else:
                        dbg["pp_b4_reg_skip_invalid_best_T"] = 1.0
                        if reg_require_success:
                            return pred, dbg

                    if reg_require_success and int(reg_meta.get("success", 0)) != 1:
                        dbg["pp_b4_reg_skip_gate_fail"] = 1.0
                        return pred, dbg
                except Exception as e:
                    dbg["pp_b4_reg_error"] = 1.0
                    dbg["pp_b4_reg_error_hash"] = float(abs(hash(str(e))) % 1000000)
                    if reg_require_success:
                        return pred, dbg

    eps = 1e-12
    inside_union = np.zeros((int(xyz.shape[0]),), dtype=bool)
    n_valid = 0
    n_invalid = 0
    for item in boxes_raw:
        center: Optional[np.ndarray] = None
        extent: Optional[np.ndarray] = None
        rotation: Optional[np.ndarray] = None
        try:
            if isinstance(item, dict):
                center = np.asarray(item.get("center"), dtype=np.float64).reshape(3)
                extent = np.asarray(item.get("extent"), dtype=np.float64).reshape(3)
                rotation = np.asarray(item.get("rotation"), dtype=np.float64).reshape(3)
            else:
                arr = np.asarray(item, dtype=np.float64).reshape(-1)
                if int(arr.size) >= 9:
                    center = arr[0:3]
                    extent = arr[3:6]
                    rotation = arr[6:9]
            if center is None or extent is None or rotation is None:
                n_invalid += 1
                continue
            if np.any(~np.isfinite(center)) or np.any(~np.isfinite(extent)) or np.any(~np.isfinite(rotation)):
                n_invalid += 1
                continue
            half = np.maximum(0.5 * np.asarray(extent, dtype=np.float64) + float(margin_m), 0.0)
            if float(np.max(half)) <= 0.0:
                n_invalid += 1
                continue
            rot_t = _b4_rot_matrix_xyz_deg(np.asarray(rotation, dtype=np.float64))
            center_scene = np.asarray(center, dtype=np.float64) @ reg_R.T + reg_t
            rot_scene = np.asarray(reg_R @ rot_t, dtype=np.float64)
            local = (xyz - center_scene[None, :]) @ rot_scene
            inside = np.all(np.abs(local) <= (half[None, :] + eps), axis=1)
            inside_union |= inside
            n_valid += 1
        except Exception:
            n_invalid += 1
            continue

    dbg["pp_b4_base_box_n_boxes_total"] = float(len(boxes_raw))
    dbg["pp_b4_base_box_n_boxes_valid"] = float(int(n_valid))
    dbg["pp_b4_base_box_n_boxes_invalid"] = float(int(n_invalid))
    dbg["pp_b4_base_box_margin_m"] = float(margin_m)

    if int(n_valid) <= 0:
        dbg["pp_b4_base_box_skip_no_valid_box"] = 1.0
        return pred, dbg

    pred_before_n = int(np.sum(pred))
    forced_pin2base_mask = inside_union & pred
    pred_out = pred.copy()
    pred_out[inside_union] = False

    force_outside_pin = _as_bool("b4_box_force_outside_pin", params.get("b4_box_force_outside_pin"), False)
    forced_base2pin = 0
    if force_outside_pin:
        outside = ~inside_union
        forced_base2pin = int(np.sum(outside & (~pred)))
        pred_out[outside] = True
        dbg["pp_b4_base_box_force_outside_pin"] = 1.0
    else:
        dbg["pp_b4_base_box_force_outside_pin"] = 0.0

    dbg["pp_b4_base_box_inside_points"] = float(int(np.sum(inside_union)))
    dbg["pp_b4_base_box_inside_ratio"] = float(np.mean(inside_union)) if inside_union.size else float("nan")
    dbg["pp_b4_base_box_forced_pin2base_abs"] = float(int(np.sum(forced_pin2base_mask)))
    dbg["pp_b4_base_box_forced_base2pin_abs"] = float(int(forced_base2pin))
    dbg["pp_b4_base_box_pred_pin_n_before"] = float(int(pred_before_n))
    dbg["pp_b4_base_box_pred_pin_n_after"] = float(int(np.sum(pred_out)))
    return pred_out, dbg
