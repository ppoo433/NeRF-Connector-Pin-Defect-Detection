# -*- coding: utf-8 -*-

_DAIMA_CTX = None
if __name__ == "__main__":
    # Enable unified console logging when running this file directly (PyCharm Run).
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

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
# 假设 pin_cloud2center_line.py 在同一目录
from baseline.common import (
    compare_readout_protocol_matches_frozen,
    compute_baseline_verdict_v1,
    compute_formal_decision_shadow_v1,
    compute_selected_score_run_features_v1,
    evaluate_centerline_tolerance_with_direction,
    format_compare_readout_protocol,
    finite,
    get_frozen_compare_readout_protocol_v1,
    resolve_compare_readout_protocol_from_env,
)
from pin_cloud2center_line import extract_centerlines_from_pcd, create_tube_from_points


if "_REPO_ROOT" not in globals():
    _from_path = Path(__file__).resolve()
    _repo_root = _from_path.parent
    for _d in [_from_path] + list(_from_path.parents):
        if (_d / ".release-root").is_file():
            _repo_root = _d
            break
    else:
        for _d in [_from_path] + list(_from_path.parents):
            if (_d / ".git").exists():
                _repo_root = _d
                break
    _REPO_ROOT = _repo_root

# =========================================================
#                   Units / Scaling (IMPORTANT)
# =========================================================
# pin_cloud2center_line.py works in a scaled coordinate system and prints UNIT_SCALE.
# In this project, UNIT_SCALE = 1/1000, meaning:
#   1 mm  ->  UNIT_SCALE (coordinate units)
# This file uses "step_mm" in millimeters for centerline resampling / tangent comparison, so we MUST convert:
#   step_u = step_mm * UNIT_SCALE
# Otherwise, resampling degenerates to only a few points (e.g. 4 points), and np.gradient() becomes unstable,
# causing massive false positives in normal regions.
try:
    from pin_cloud2center_line import UNIT_SCALE as PIN_UNIT_SCALE
except Exception:
    PIN_UNIT_SCALE = 1.0


def mm_to_unit_len(mm: float) -> float:
    return float(mm) * float(PIN_UNIT_SCALE)


# Keep extraction behavior consistent with validated pin_cloud2center_line defaults.
CENTERLINE_CFG_DENOISE_ENABLE = False
CENTERLINE_CFG_ENDPOINT_CENTERING_ITERS = 2
CENTERLINE_CFG_FORCE_FINAL_STAGE13 = True

# Registration defaults (aligned with repo's overall_then_base_v1 idea).
REG_TARGET_GRID_POINTS = 50000
REG_MIN_VOXEL_MM = 0.2
REG_MAX_VOXEL_MM = 1.0
REG_RANSAC_MAX_ITER = 400000
REG_RANSAC_CONFIDENCE = 0.999
REG_RANSAC_DIST_MULT = 1.5
REG_ICP_DIST_MULT = 0.4
REG_N_TRIALS = 20  # Default: run multi-trial registration and pick the best result.

# Baseline verdict / uncertainty gate (interactive readout only; heat source stays on tol_scores in A/B).
BASELINE_VERDICT_MODE = "baseline_v1"
BASELINE_PASS_GUARD_RATIO = 0.05
BASELINE_FAIL_GUARD_RATIO = 0.10
BASELINE_HARD_FAIL_MARGIN_RATIO = 0.20
BASELINE_MIN_PAIRED = 18
BASELINE_MIN_ICP_FITNESS = 0.0
BASELINE_MAX_ICP_RMSE_MM = 0.50
BASELINE_MAX_BASE_BIAS_MM = 0.05

# Detection policy (2026-03-31):
# - Keep tip/length short ratios for debug/export only.
# - Do not let shortness or rescan change the final detection verdict.
DETECTION_DISABLE_SHORT_TOLERANCE = True
DETECTION_DISABLE_RESCAN = True


def _centerline_extract_config():
    from pin_cloud2center_line import CenterlineConfig

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


def orient_curve_to_base_start(curve, base_end_idx_local):
    """
    Try to make the curve parameterization consistent across reference/defect needles.

    If base_end_idx_local == -1, the truncated/base end is at the last point -> reverse the curve so base is at index 0.
    If base_end_idx_local == 0 or None/unknown, keep as-is.
    """
    pts = np.asarray(curve, dtype=np.float64)
    if len(pts) < 2:
        return pts
    if base_end_idx_local is None:
        return pts
    try:
        idx = int(base_end_idx_local)
    except Exception:
        return pts
    return pts[::-1] if idx == -1 else pts

# =========================================================
#                   可视化配置 (VISUALIZATION CONFIG)
# =========================================================
# 您可以在这里自由开关各个部分的显示状态


# 1. 底座相关
VIS_SHOW_BASE = False  # 是否显示底座点云 (灰色)

# 2. 模板针 (Template/Reference) 相关 - 通常对应"漏针"或"标准参考"
VIS_SHOW_TEMPLATE_CL = False  # 默认不显示模板中心线；当前评判标准以容差带为准
VIS_SHOW_TEMPLATE_PC = True  # [修复] 是否显示模板原始点云 (黄色) -> 用于查看漏针实体

# 3. 检测针 (Defect/Target) 相关 - 对应"检测结果"、"多针"、"热力图"
VIS_SHOW_DEFECT_CL = True  # 是否显示检测针中心线管道 (带热力图颜色/红色)
VIS_SHOW_DEFECT_PC = True  # 是否显示检测针原始点云 (带热力图颜色/红色)

# 4. 样式参数
TUBE_RADIUS_TEMPLATE = 0.1  # 模板线粗细
TUBE_RADIUS_DEFECT = 0.1  # 待测线粗细
VIS_SHOW_PCA_ARROWS = True  # [新增] 是否显示 PCA 主轴方向箭头

# 5. [新增] 粗粒度视图中的模板点云显示
VIS_SHOW_TEMPLATE_PC_IN_COARSE = True  # 在粗粒度视图中显示模板点云

# 6. [新增] 可视化降载（解决 Open3D 白屏/无响应）
# 大量 TriangleMesh（tube/arrow）+ 大点云同时渲染时，draw_geometries 容易白屏/卡死。
# 开启快速模式后：
# - 中心线热力图用 LineSet 替代 tube mesh
# - 箭头用 LineSet 替代 3D arrow mesh
# - 热力点云默认只在最终视图显示，并对每根针限点
VIS_FAST_MODE = True
VIS_HEATMAP_PC_ONLY_FINAL = True
VIS_MAX_POINTS_PER_NEEDLE = 20000
VIS_SHOW_TOLERANCE_BAND = True
VIS_TOLERANCE_BAND_ALPHA = 0.55
VIS_TOLERANCE_BAND_COLOR = [0.0, 0.55, 0.95]
VIS_TOLERANCE_BAND_MIN_RADIUS_MM = 0.02
VIS_SHOW_DIRECTION_CONE_MODULE = True
VIS_SCENE_COLUMN_GAP_MM = 60.0
VIS_DIRECTION_MODULE_GAP_MM = 30.0
VIS_DIRECTION_HEAT_GAP_MM = 30.0
VIS_DIRECTION_CONE_STRIDE = 4
VIS_DIRECTION_CONE_SKIP_END_POINTS = 2
VIS_DIRECTION_CONE_LENGTH_SCALE = 3.0
VIS_DIRECTION_CONE_RING_SEGMENTS = 10
VIS_DIRECTION_CONE_COLOR = [1.0, 1.0, 1.0]
VIS_DIRECTION_CONE_FAIL_COLOR = [1.0, 0.35, 0.20]
VIS_DIRECTION_ARROW_OK_COLOR = [0.15, 0.95, 0.20]
VIS_DIRECTION_ARROW_FAIL_COLOR = [1.0, 0.20, 0.20]
VIS_DIRECTION_DEFECT_BASELINE_COLOR = [0.45, 0.45, 0.45]
VIS_DIRECTION_OUT_SEG_COLOR = [1.0, 0.18, 0.18]
VIS_DIRECTION_SHOW_TEMPLATE_CL = False
VIS_VERDICT_MODULE_GAP_MM = 30.0
VIS_VERDICT_PASS_COLOR = [0.10, 0.35, 1.00]
VIS_VERDICT_FAIL_COLOR = [1.00, 0.00, 0.00]
VIS_VERDICT_MISSING_COLOR = [1.00, 1.00, 0.00]


# =========================================================
#                   Final Decision Method
# =========================================================
# NOTE:
# - "tangent": keep the legacy tangent-angle based method (default).
# - "tolerance_band": use centerline_tolerance.json as the primary decision rule.
FINAL_METHOD = "tolerance_band"

# Only used when FINAL_METHOD == "tolerance_band".
TOLERANCE_JSON_PATH = str(_REPO_ROOT / "private_inputs" / "centerline_tolerance.json")
_ENV_TOLERANCE_JSON_PATH = str(os.getenv("DAIMA_TOLERANCE_JSON_PATH", "")).strip()
if _ENV_TOLERANCE_JSON_PATH:
    TOLERANCE_JSON_PATH = _ENV_TOLERANCE_JSON_PATH


# =========================================================

# ===================== 辅助工具 =====================

def _score_to_color_legacy(score):
    """
    [修改版] 颜色映射：
    0.0 (Blue) -> 0.5 (Yellow) -> 1.0 (Red)
    去掉了绿色和青色，只要有分，马上变黄/红，警示效果更强。
    """
    s = np.clip(score, 0.0, 1.0)

    # 简单的两段式插值
    if s < 0.5:
        # 0.0 - 0.5: Blue [0,0,1] -> Yellow [1,1,0]
        # 这样低分段也会迅速变亮，引起注意
        t = s / 0.5
        return [t, t, 1.0 - t]
    else:
        # 0.5 - 1.0: Yellow [1,1,0] -> Red [1,0,0]
        t = (s - 0.5) / 0.5
        return [1.0, 1.0 - t, 0.0]


def score_to_color(score):
    """Tolerance rainbow colormap: blue->cyan->green->yellow->red."""
    s = np.clip(score, 0.0, 1.0)

    if s < 0.25:
        t = s / 0.25
        return [0.0, t, 1.0]
    if s < 0.50:
        t = (s - 0.25) / 0.25
        return [0.0, 1.0, 1.0 - t]
    if s < 0.75:
        t = (s - 0.50) / 0.25
        return [t, 1.0, 0.0]
    t = (s - 0.75) / 0.25
    return [1.0, 1.0 - t, 0.0]


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    print(f"[INFO] Loaded {path}, points = {len(pcd.points)}")
    return pcd


def remove_outliers(pcd, nb_neighbors=50, std_ratio=1.5):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def estimate_voxel_size(pcd, target_downsample_points=50000):
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
    if len(pcd.points) == 0: return 0.1
    voxel_size = diag * (len(pcd.points) / max(target_downsample_points, 1)) ** (1.0 / 3.0)
    return float(voxel_size)


def _clamp_voxel_size_u(voxel_u):
    min_u = max(float(mm_to_unit_len(REG_MIN_VOXEL_MM)), 1e-6)
    max_u = max(float(mm_to_unit_len(REG_MAX_VOXEL_MM)), min_u * 2.0)
    try:
        v = float(voxel_u)
    except Exception:
        v = min_u
    if not np.isfinite(v) or v <= 0:
        v = min_u
    return float(np.clip(v, min_u, max_u))


def _estimate_voxel_size_clamped(pcd):
    v = estimate_voxel_size(pcd, target_downsample_points=int(REG_TARGET_GRID_POINTS))
    return _clamp_voxel_size_u(v)


def _ensure_normals(pcd, radius_u):
    if pcd is None or len(pcd.points) <= 0:
        return
    if (not pcd.has_normals()) or (len(pcd.normals) != len(pcd.points)):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius_u), max_nn=30)
        )


def _preprocess_fpfh(pcd, voxel_u):
    pcd_down = pcd.voxel_down_sample(float(voxel_u))
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 2.0, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_u) * 5.0, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def _register_fpfh_ransac_then_icp(source, target, voxel_u):
    t0 = time.time()
    meta = {"status": "ok"}

    if len(source.points) < 4 or len(target.points) < 4:
        raise RuntimeError("insufficient points for global registration")

    # Default: run multiple stochastic (RANSAC) trials and pick the best (fitness high, rmse low)
    # to reduce run-to-run variance and avoid occasional bad initializations.
    n_trials = int(REG_N_TRIALS)
    seed_base = 0
    try:
        n_trials_s = str(os.environ.get("DAIMA_REG_N_TRIALS", "")).strip()
        if n_trials_s:
            n_trials = int(n_trials_s)
    except Exception:
        pass
    try:
        seed_s = str(os.environ.get("DAIMA_REG_SEED_BASE", "")).strip()
        if seed_s:
            seed_base = int(seed_s)
    except Exception:
        seed_base = 0

    n_trials = int(max(1, n_trials))

    s_down, s_fpfh = _preprocess_fpfh(source, voxel_u)
    t_down, t_fpfh = _preprocess_fpfh(target, voxel_u)

    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    trials = []
    best_key = None
    best_T = None
    best_meta = None
    best_idx = -1

    for trial_idx in range(n_trials):
        t_trial0 = time.time()
        trial_seed = int(seed_base) + int(trial_idx)
        try:
            try:
                if hasattr(o3d, "utility") and hasattr(o3d.utility, "random") and hasattr(o3d.utility.random, "seed"):
                    o3d.utility.random.seed(int(trial_seed))
            except Exception:
                pass

            res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                s_down,
                t_down,
                s_fpfh,
                t_fpfh,
                True,
                float(voxel_u) * float(REG_RANSAC_DIST_MULT),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        float(voxel_u) * float(REG_RANSAC_DIST_MULT)
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(REG_RANSAC_MAX_ITER), float(REG_RANSAC_CONFIDENCE)
                ),
            )

            res_icp = o3d.pipelines.registration.registration_icp(
                source,
                target,
                float(voxel_u) * float(REG_ICP_DIST_MULT),
                res_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )

            r_fit = float(getattr(res_ransac, "fitness", float("nan")))
            r_rmse = float(getattr(res_ransac, "inlier_rmse", float("nan")))
            i_fit = float(getattr(res_icp, "fitness", float("nan")))
            i_rmse = float(getattr(res_icp, "inlier_rmse", float("nan")))

            T = np.asarray(res_icp.transformation, dtype=float)
            if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                raise RuntimeError("Invalid transformation from global+ICP stage")

            trial_meta = {
                "status": "ok",
                "trial_idx": int(trial_idx),
                "seed": int(trial_seed),
                "ransac_fitness": float(r_fit),
                "ransac_inlier_rmse": float(r_rmse),
                "icp_fitness": float(i_fit),
                "icp_inlier_rmse": float(i_rmse),
                "elapsed_s": float(time.time() - t_trial0),
            }
            trials.append(trial_meta)

            fit_cmp = float(i_fit)
            if not np.isfinite(fit_cmp):
                fit_cmp = float("-inf")
            rmse_cmp = float(i_rmse)
            if not np.isfinite(rmse_cmp):
                rmse_cmp = float("inf")
            key = (fit_cmp, -rmse_cmp)
            if best_key is None or key > best_key:
                best_key = key
                best_T = T
                best_meta = dict(trial_meta)
                best_idx = int(trial_idx)
        except Exception as e:
            trials.append(
                {
                    "status": "failed",
                    "trial_idx": int(trial_idx),
                    "seed": int(trial_seed),
                    "error": str(e),
                    "elapsed_s": float(time.time() - t_trial0),
                }
            )

    if best_T is None or best_meta is None:
        meta["status"] = "failed"
        meta["n_trials"] = int(n_trials)
        meta["trials"] = trials
        meta["elapsed_s"] = float(time.time() - t0)
        raise RuntimeError("All registration trials failed")

    meta.update(best_meta)
    meta["n_trials"] = int(n_trials)
    meta["best_trial_idx"] = int(best_idx)
    meta["trials"] = trials
    meta["elapsed_s"] = float(time.time() - t0)
    return np.asarray(best_T, dtype=float), meta


def _refine_icp_point_to_plane(source, target, init_T, voxel_u):
    t0 = time.time()
    meta = {"status": "ok"}

    _ensure_normals(target, radius_u=float(voxel_u) * 2.0)
    _ensure_normals(source, radius_u=float(voxel_u) * 2.0)

    res_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(voxel_u) * float(REG_ICP_DIST_MULT),
        np.asarray(init_T, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    meta["icp_fitness"] = float(getattr(res_icp, "fitness", float("nan")))
    meta["icp_inlier_rmse"] = float(getattr(res_icp, "inlier_rmse", float("nan")))
    meta["elapsed_s"] = float(time.time() - t0)

    T = np.asarray(res_icp.transformation, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        meta["status"] = "failed"
        raise RuntimeError("Invalid transformation from base refine stage")
    return T, meta


def _merge_point_clouds(*pcds):
    pts_all = []
    cols_all = []
    has_any_color = False

    for pcd in pcds:
        if pcd is None or len(pcd.points) <= 0:
            continue
        pts = np.asarray(pcd.points)
        if pts.size <= 0:
            continue
        pts_all.append(pts)

        cols = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((0, 3), dtype=np.float64)
        if cols.ndim == 2 and len(cols) == len(pts):
            cols_all.append(cols)
            has_any_color = True
        else:
            cols_all.append(np.zeros((len(pts), 3), dtype=np.float64))

    out = o3d.geometry.PointCloud()
    if not pts_all:
        return out

    pts_cat = np.vstack(pts_all)
    out.points = o3d.utility.Vector3dVector(pts_cat)
    if has_any_color and cols_all:
        out.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    return out


def _compute_board_transform_base_only_legacy(base_normal, base_defect):
    voxel_size = _estimate_voxel_size_clamped(base_normal)
    T, stage = _register_fpfh_ransac_then_icp(base_defect, base_normal, voxel_size)
    meta = {
        "status": "ok",
        "method": "base_legacy",
        "voxel_base": float(voxel_size),
        "base_stage": dict(stage),
        "icp_fitness": float(stage.get("icp_fitness", float("nan"))),
        "icp_inlier_rmse": float(stage.get("icp_inlier_rmse", float("nan"))),
    }
    return T, meta


# ===================== 核心几何算法 (细粒度检测专用) =====================


# 替换 detection_12_1.py 中的同名函数


# ===================== 配准与配对逻辑 =====================

def compute_board_transform(
    base_normal_path,
    base_defect_path,
    pins_normal_path=None,
    pins_defect_path=None,
    return_clean_clouds=False,
):
    print("[INFO] Computing board transform (overall_then_base_v1)...")
    t_all = time.time()

    base_normal = remove_outliers(load_point_cloud(base_normal_path))
    base_defect = remove_outliers(load_point_cloud(base_defect_path))

    T = None
    meta = {"status": "ok", "method": "overall_then_base_v1"}

    can_use_overall = bool(pins_normal_path) and bool(pins_defect_path)
    if can_use_overall:
        try:
            pins_normal = remove_outliers(load_point_cloud(pins_normal_path))
            pins_defect = remove_outliers(load_point_cloud(pins_defect_path))

            overall_normal = _merge_point_clouds(base_normal, pins_normal)
            overall_defect = _merge_point_clouds(base_defect, pins_defect)

            if len(overall_normal.points) < 100 or len(overall_defect.points) < 100:
                raise RuntimeError("overall cloud has too few points after merge")

            voxel_overall = _estimate_voxel_size_clamped(overall_normal)
            T0, meta_overall = _register_fpfh_ransac_then_icp(overall_defect, overall_normal, voxel_overall)

            voxel_base = _estimate_voxel_size_clamped(base_normal)
            T, meta_base = _refine_icp_point_to_plane(base_defect, base_normal, T0, voxel_base)

            meta["voxel_overall"] = float(voxel_overall)
            meta["voxel_base"] = float(voxel_base)
            meta["overall_stage"] = dict(meta_overall)
            meta["base_refine_stage"] = dict(meta_base)
            meta["icp_fitness"] = float(meta_base.get("icp_fitness", float("nan")))
            meta["icp_inlier_rmse"] = float(meta_base.get("icp_inlier_rmse", float("nan")))

            print(
                "[INFO] Registration overall_then_base_v1 done: "
                f"overall_icp_fit={meta_overall.get('icp_fitness', float('nan')):.4f}, "
                f"base_icp_fit={meta_base.get('icp_fitness', float('nan')):.4f}"
            )
        except Exception as e:
            print(f"[WARN] overall_then_base_v1 failed, fallback to base_legacy: {e}")
            T, legacy_meta = _compute_board_transform_base_only_legacy(base_normal, base_defect)
            meta = dict(legacy_meta)
            meta["fallback_reason"] = str(e)
    else:
        print("[WARN] pins paths not provided; fallback to base_legacy registration")
        T, meta = _compute_board_transform_base_only_legacy(base_normal, base_defect)

    if T is None:
        T, meta = _compute_board_transform_base_only_legacy(base_normal, base_defect)

    meta["elapsed_s"] = float(time.time() - t_all)

    if return_clean_clouds:
        return np.asarray(T, dtype=float), meta, base_normal, base_defect
    return np.asarray(T, dtype=float), meta


def transform_needle_infos(needle_infos, T):
    new_infos = []
    for info in needle_infos:
        new_info = info.copy()
        c = np.append(info["center_point"], 1.0)
        new_info["center_point"] = (T @ c)[:3]
        pts = info["curve_global_final"]
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        new_info["curve_global_final"] = (T @ pts_h.T).T[:, :3]
        if "points" in info:
            raw = info["points"]
            raw_h = np.hstack([raw, np.ones((len(raw), 1))])
            new_info["points"] = (T @ raw_h.T).T[:, :3]
        new_infos.append(new_info)
    return new_infos


def match_needles_by_position(needle_infos_ref, needle_infos_def, r_match=None):
    c_ref = np.array([i["center_point"] for i in needle_infos_ref])
    c_def = np.array([i["center_point"] for i in needle_infos_def])

    if len(c_ref) == 0 or len(c_def) == 0:
        return {"pairs": [], "missing_normal": [], "extra_defect": []}

    if r_match is None:
        tree = cKDTree(c_ref)
        d, _ = tree.query(c_ref, k=2)
        if len(c_ref) > 1:
            r_match = np.median(d[:, 1]) * 0.5
        else:
            r_match = 1.0

    tree_ref = cKDTree(c_ref)
    pairs = []
    used_def = set()

    for k, center in enumerate(c_def):
        d, j = tree_ref.query(center)
        if d < r_match:
            pairs.append({"normal_id": j, "defect_id": k, "dist": d})
            used_def.add(k)

    final_pairs = []
    used_ref = set()
    pairs.sort(key=lambda x: x["dist"])
    for p in pairs:
        if p["normal_id"] not in used_ref:
            final_pairs.append(p)
            used_ref.add(p["normal_id"])

    return {
        "pairs": final_pairs,
        "missing_normal": [j for j in range(len(c_ref)) if j not in used_ref],
        "extra_defect": [k for k in range(len(c_def)) if k not in used_def]
    }


def resample_curve_by_step(curve, step=0.2):
    """
    [升级版] 对中心线进行重采样。
    即使输入已经是平滑中心线，为了保证在加密切片(step变小)后
    切向量(导数)依然光滑，必须使用样条插值而不是线性插值。

    参数:
        curve: (N, 3) 原始平滑中心线
        step: 重采样步长 (mm)，越小切片越密
    """
    if len(curve) < 4:
        return curve

    # 1. 计算总弧长
    diffs = curve[1:] - curve[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    total_len = np.sum(dists)

    # 2. 计算目标点数 (增加切片数的关键)
    # 例如：针长 15mm，步长 0.2mm -> 生成约 75 个切片点
    num_pts = int(total_len / step) + 1
    if num_pts < 4: num_pts = 4

    # 3. 样条插值 (Preserving Smoothness)
    try:
        # 去除极其接近的重复点，防止 splprep 报错
        valid_idx = [0]
        for i in range(1, len(curve)):
            if np.linalg.norm(curve[i] - curve[valid_idx[-1]]) > 1e-4:
                valid_idx.append(i)
        clean_curve = curve[valid_idx]

        if len(clean_curve) < 4:
            return curve  # 点太少，直接返回

        tck, u = splprep(clean_curve.T, s=0, k=3)
        u_new = np.linspace(0, 1, num_pts)
        resampled = splev(u_new, tck)
        return np.column_stack(resampled)
    except Exception as e:
        print(f"[WARN] Spline resampling failed: {e}, using linear interpolation")
        t_cumsum = np.concatenate(([0], np.cumsum(dists)))
        t_new = np.linspace(0, total_len, num_pts)
        x_new = np.interp(t_new, t_cumsum, curve[:, 0])
        y_new = np.interp(t_new, t_cumsum, curve[:, 1])
        z_new = np.interp(t_new, t_cumsum, curve[:, 2])
        return np.column_stack([x_new, y_new, z_new])


def resample_curve_by_count(curve, num_pts=240):
    """
    Resample a 3D polyline to a fixed number of points using spline interpolation (best-effort).

    This is used by tolerance-band evaluation where the tolerance model is defined on a fixed
    resampling count (e.g. resample_n=240 in centerline_tolerance.json).
    """
    curve = np.asarray(curve, dtype=np.float64)
    try:
        num_pts = int(num_pts)
    except Exception:
        return curve

    if num_pts <= 1 or len(curve) < 2:
        return curve
    if len(curve) < 4 or num_pts < 4:
        # Linear interpolation as a fallback for short curves.
        t = np.linspace(0.0, 1.0, len(curve))
        t_new = np.linspace(0.0, 1.0, num_pts)
        out = np.column_stack([np.interp(t_new, t, curve[:, k]) for k in range(3)])
        return out

    # Remove near-duplicate points to avoid splprep errors.
    valid_idx = [0]
    for i in range(1, len(curve)):
        if np.linalg.norm(curve[i] - curve[valid_idx[-1]]) > 1e-4:
            valid_idx.append(i)
    clean_curve = curve[valid_idx]
    if len(clean_curve) < 4:
        return curve

    try:
        tck, _u = splprep(clean_curve.T, s=0, k=3)
        u_new = np.linspace(0.0, 1.0, num_pts)
        resampled = splev(u_new, tck)
        return np.column_stack(resampled)
    except Exception as e:
        print(f"[WARN] Spline resample_by_count failed: {e}, using linear interpolation")
        t = np.linspace(0.0, 1.0, len(curve))
        t_new = np.linspace(0.0, 1.0, num_pts)
        return np.column_stack([np.interp(t_new, t, curve[:, k]) for k in range(3)])


def orient_curve_to_match_endpoints(curve, ref_curve):
    """
    Ensure curve direction matches ref_curve direction (base->tip), best-effort.
    """
    c = np.asarray(curve, dtype=np.float64)
    r = np.asarray(ref_curve, dtype=np.float64)
    if len(c) < 2 or len(r) < 2:
        return c
    d_keep = np.linalg.norm(c[0] - r[0]) + np.linalg.norm(c[-1] - r[-1])
    d_flip = np.linalg.norm(c[0] - r[-1]) + np.linalg.norm(c[-1] - r[0])
    return c[::-1] if d_flip < d_keep else c


def curve_length_u(curve):
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(curve[1:] - curve[:-1], axis=1)))


def cross_section_pca(cloud_points, plane_origin, plane_normal, thickness=0.3):
    """
    [精简版] 获取切片点云的 PCA 主轴。
    """
    diff = cloud_points - plane_origin
    proj_len = np.dot(diff, plane_normal)
    mask = np.abs(proj_len) < thickness
    slice_pts = cloud_points[mask]

    if len(slice_pts) < 3:
        return np.array([1, 0, 0])

    centered = slice_pts - slice_pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    return principal_axis


def compute_relative_angle_score(tangent_ref, tangent_def, max_angle_deg=30.0):
    """
    [细粒度评分] 计算两个切向量之间的夹角，并归一化为 [0, 1]。
    角度越大 -> 分数越高 (0 = 正常, 1 = 异常)
    """
    dot_prod = np.clip(np.dot(tangent_ref, tangent_def), -1.0, 1.0)
    angle_rad = np.arccos(np.abs(dot_prod))
    angle_deg = np.degrees(angle_rad)
    score = min(angle_deg / max_angle_deg, 1.0)
    return score


def load_centerline_tolerance(json_path):
    """
    Load centerline_tolerance.json written by jiance/build_centerline_tolerance.py.
    Returns dict on success, or None on failure.
    """
    if not json_path:
        return None
    try:
        p = Path(str(json_path))
        if not p.is_file():
            print(f"[WARN] centerline_tolerance.json not found: {p}")
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to load centerline_tolerance.json: {e}")
        return None


def _load_json_text(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _has_tangent_payload(tol_json: Mapping[str, Any] | None) -> bool:
    if not isinstance(tol_json, Mapping):
        return False
    needles = tol_json.get("needles", [])
    if not isinstance(needles, list):
        return False
    for needle in needles:
        if not isinstance(needle, Mapping):
            continue
        status = str(needle.get("tol_tangent_model_status", "")).strip().lower()
        axis = needle.get("tol_tangent_axis_xyz", [])
        half_angle = needle.get("tol_tangent_half_angle_deg", [])
        if status == "ok" and isinstance(axis, list) and len(axis) >= 2 and isinstance(half_angle, list) and len(half_angle) >= 2:
            return True
    return False


def _find_tolerance_model_root(json_path: Path) -> Path:
    for parent in [json_path.parent] + list(json_path.parents):
        if (parent / "output").is_dir():
            return parent
    return json_path.parent


def _resolve_tangent_capable_tolerance_json(json_path: Path) -> tuple[Path, dict[str, Any]]:
    root_dir = _find_tolerance_model_root(json_path)
    candidate_paths: list[Path] = []
    seen: set[str] = set()
    for candidate in [json_path, root_dir / "best" / "centerline_tolerance.json"] + sorted(
        (root_dir / "output").glob("*/centerline_tolerance.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            candidate_resolved = candidate.resolve()
        except Exception:
            continue
        key = str(candidate_resolved).lower()
        if key in seen or not candidate_resolved.is_file():
            continue
        seen.add(key)
        candidate_paths.append(candidate_resolved)

    fallback_json = load_centerline_tolerance(str(json_path))
    if _has_tangent_payload(fallback_json):
        return json_path.resolve(), dict(fallback_json) if isinstance(fallback_json, dict) else {}

    for candidate_path in candidate_paths:
        candidate_json = _load_json_text(candidate_path)
        if _has_tangent_payload(candidate_json):
            return candidate_path, dict(candidate_json) if isinstance(candidate_json, dict) else {}

    return json_path.resolve(), dict(fallback_json) if isinstance(fallback_json, dict) else {}


def resolve_tolerance_reference_paths(tol_json):
    """
    Try to locate the reference sample base/pins ply files based on tolerance model metadata.

    Returns (base_ply_path, pins_ply_path) as Path, or (None, None) on failure.
    """
    try:
        meta = tol_json.get("meta", {}) if isinstance(tol_json, dict) else {}
        input_dir = meta.get("input_dir", None)
        ref_id = tol_json.get("reference_sample_id", None) if isinstance(tol_json, dict) else None
        if not input_dir or ref_id is None:
            return None, None
        input_dir = Path(str(input_dir))
        ref_id = str(ref_id).strip()
        if not ref_id:
            return None, None

        # Pattern A: <input_dir>/<id>_base.ply, <input_dir>/<id>_pins.ply
        base_a = input_dir / f"{ref_id}_base.ply"
        pins_a = input_dir / f"{ref_id}_pins.ply"
        if base_a.is_file() and pins_a.is_file():
            return base_a, pins_a

        # Pattern B: <input_dir>/<id>/base.ply, <input_dir>/<id>/pins.ply
        base_b = input_dir / ref_id / "base.ply"
        pins_b = input_dir / ref_id / "pins.ply"
        if base_b.is_file() and pins_b.is_file():
            return base_b, pins_b

        # Best-effort: return whichever exists.
        base_p = base_a if base_a.is_file() else base_b if base_b.is_file() else None
        pins_p = pins_a if pins_a.is_file() else pins_b if pins_b.is_file() else None
        return base_p, pins_p
    except Exception:
        return None, None


def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return float(default)


def _format_flag_list_text(flags):
    if isinstance(flags, (list, tuple)):
        vals = [str(x).strip() for x in flags if str(x).strip()]
        return ";".join(vals)
    return str(flags).strip()


def _compact_json_text(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _compute_point_only_shadow_verdict(tol_metrics_map):
    eps = 1e-9
    needle_score_raw = _safe_float(
        tol_metrics_map.get("needle_score_raw", float("nan")),
        default=float("nan"),
    )
    if finite(needle_score_raw) and float(needle_score_raw) > 0.0 + eps:
        return "fail", "point_readout_positive"
    if finite(needle_score_raw):
        return "pass", "clear_pass"
    return "fail", "non_finite_needle_score"


def _compute_lateral_only_fail_subtype(tol_metrics_map):
    eps = 1e-9
    lateral_ratio = _safe_float(
        tol_metrics_map.get("max_lateral_ratio", float("nan")),
        default=float("nan"),
    )
    if finite(lateral_ratio) and float(lateral_ratio) > 1.0 + eps:
        return "lateral-only"
    return ""


def _compute_lateral_only_tol_status(tol_metrics_map):
    eps = 1e-9
    lateral_ratio = _safe_float(
        tol_metrics_map.get("max_lateral_ratio", float("nan")),
        default=float("nan"),
    )
    if finite(lateral_ratio) and float(lateral_ratio) > 1.0 + eps:
        return "fail"
    if finite(lateral_ratio):
        return "pass"
    return "fail"


def _apply_detection_policy_to_tol_metrics(tol_metrics):
    tol_metrics_map = dict(tol_metrics) if isinstance(tol_metrics, dict) else {}
    if DETECTION_DISABLE_SHORT_TOLERANCE:
        ab_verdict, ab_reason = _compute_point_only_shadow_verdict(tol_metrics_map)
        tol_metrics_map["ab_shadow_verdict"] = str(ab_verdict)
        tol_metrics_map["ab_shadow_reason"] = str(ab_reason)
        tol_metrics_map["tol_fail_subtype"] = str(_compute_lateral_only_fail_subtype(tol_metrics_map))
    tol_metrics_map["short_tolerance_detection_enabled"] = bool(not DETECTION_DISABLE_SHORT_TOLERANCE)
    tol_metrics_map["rescan_detection_enabled"] = bool(not DETECTION_DISABLE_RESCAN)
    return tol_metrics_map


def _collapse_need_rescan_to_binary(verdict_text, reason_text, fallback_verdict):
    verdict_n = str(verdict_text).strip()
    reason_n = str(reason_text).strip()
    fallback_n = str(fallback_verdict).strip()
    if not DETECTION_DISABLE_RESCAN or verdict_n != "need_rescan":
        return verdict_n, reason_n
    if fallback_n not in {"pass", "fail"}:
        fallback_n = "fail"
    if reason_n:
        return fallback_n, f"{reason_n}__binary_no_rescan"
    return fallback_n, "binary_no_rescan"


def _build_shadow_gate_payload(info_def, reg_meta, paired_count, pair_dist_u):
    regrow = dict(info_def.get("regrow", {})) if isinstance(info_def, dict) else {}
    fit_status = str(info_def.get("overall_status", "")) if isinstance(info_def, dict) else ""
    source_fields = {}
    hard_hits = []
    soft_hits = []

    def _add_unique(dst, text):
        s = str(text).strip()
        if not s:
            return
        if s not in dst:
            dst.append(s)

    def _add_hard(text):
        _add_unique(hard_hits, text)

    def _add_soft(text):
        _add_unique(soft_hits, text)

    if fit_status:
        source_fields["fit_status"] = str(fit_status)
    if str(fit_status).strip() == "high_risk":
        _add_soft("fit:high_risk")

    try:
        paired_i = int(paired_count)
    except Exception:
        paired_i = 0
    source_fields["paired_count"] = int(paired_i)
    if paired_i < int(BASELINE_MIN_PAIRED):
        _add_hard("pairing:low_pairing")

    if finite(pair_dist_u):
        source_fields["pair_dist_u"] = float(pair_dist_u)

    reg_map = dict(reg_meta) if isinstance(reg_meta, dict) else {}
    reg_status = str(reg_map.get("status", "")).strip()
    if reg_status:
        source_fields["registration_status"] = str(reg_status)
        if reg_status.lower() not in ("ok", "success"):
            _add_hard(f"registration:{reg_status.lower()}")
    reg_fit = _safe_float(reg_map.get("icp_fitness", float("nan")))
    reg_rmse_mm = _safe_float(reg_map.get("icp_inlier_rmse_mm", float("nan")))
    if finite(reg_fit):
        source_fields["registration_icp_fitness"] = float(reg_fit)
        if float(BASELINE_MIN_ICP_FITNESS) > 0.0 and float(reg_fit) < float(BASELINE_MIN_ICP_FITNESS):
            _add_soft("registration:fitness_low")
    if finite(reg_rmse_mm):
        source_fields["registration_icp_inlier_rmse_mm"] = float(reg_rmse_mm)
        if float(BASELINE_MAX_ICP_RMSE_MM) > 0.0 and float(reg_rmse_mm) > float(BASELINE_MAX_ICP_RMSE_MM):
            _add_soft("registration:rmse_high")

    hit_key_map = {
        "risk_hit_min_n_slab": "min_n_slab<n_th",
        "risk_hit_min_cov": "min_coverage<cov_th",
        "risk_hit_max_delta": "max_delta_mm>delta_th",
        "risk_hit_clipped": "clipped_rate>clip_th",
        "risk_hit_max_rv": "max_radial_var>rv_th",
    }
    raw_key_list = ("min_n_slab", "min_coverage", "max_delta_mm", "clipped_rate", "max_radial_var")
    for side in ("start", "end"):
        endpoint_stats = dict(regrow.get(side, {})) if isinstance(regrow.get(side, {}), dict) else {}
        for raw_key in raw_key_list:
            raw_val = endpoint_stats.get(raw_key, float("nan"))
            if finite(raw_val):
                source_fields[f"{side}_{raw_key}"] = float(raw_val)
        fail_count_key = f"triclass_{side}_risk_fail_count"
        fail_count_val = regrow.get(fail_count_key, float("nan"))
        fail_count_f = _safe_float(fail_count_val, default=float("nan"))
        if finite(fail_count_val):
            source_fields[fail_count_key] = float(fail_count_f)
        hit_text_key = f"triclass_{side}_risk_hit_text"
        hit_text_val = str(regrow.get(hit_text_key, "")).strip()
        if hit_text_val:
            source_fields[hit_text_key] = str(hit_text_val)
        side_hits = {}
        for suffix, label in hit_key_map.items():
            full_key = f"triclass_{side}_{suffix}"
            hit_val = regrow.get(full_key, float("nan"))
            if finite(hit_val):
                hit_f = float(hit_val)
                source_fields[full_key] = hit_f
                side_hits[suffix] = bool(hit_f > 0.5)
        if side_hits.get("risk_hit_min_n_slab", False):
            _add_hard(f"{side}:min_n_slab<n_th")
        if side_hits.get("risk_hit_min_cov", False) and side_hits.get("risk_hit_max_delta", False):
            _add_hard(f"{side}:min_coverage<cov_th&max_delta_mm>delta_th")
        if finite(fail_count_f) and float(fail_count_f) >= 2.0:
            _add_hard(f"{side}:risk_fail_count>=2")

        if side_hits.get("risk_hit_min_cov", False):
            _add_soft(f"{side}:min_coverage<cov_th")
        if side_hits.get("risk_hit_max_delta", False):
            _add_soft(f"{side}:max_delta_mm>delta_th")
        if side_hits.get("risk_hit_clipped", False):
            _add_soft(f"{side}:clipped_rate>clip_th")
        if side_hits.get("risk_hit_max_rv", False):
            _add_soft(f"{side}:max_radial_var>rv_th")

    all_hits = list(hard_hits)
    for hit in soft_hits:
        if hit not in all_hits:
            all_hits.append(hit)

    gate_status = "need_rescan" if hard_hits else "clear"
    gate_reason = hard_hits[0] if hard_hits else "clear"
    return {
        "gate_decision_rule_version": "shadow_gate_v2_hard_combo",
        "gate_status_shadow": str(gate_status),
        "gate_reason_shadow": str(gate_reason),
        "gate_hit_list": list(all_hits),
        "gate_hit_text": _format_flag_list_text(all_hits),
        "gate_hard_hit_list": list(hard_hits),
        "gate_hard_hit_text": _format_flag_list_text(hard_hits),
        "gate_soft_hit_list": list(soft_hits),
        "gate_soft_hit_text": _format_flag_list_text(soft_hits),
        "gate_hard_fail_count": int(len(hard_hits)),
        "gate_source_fields": source_fields,
        "gate_source_fields_text": _compact_json_text(source_fields),
    }


def _build_baseline_verdict_payload(tol_status, tangent_status, fit_status, tol_metrics, paired_count, reg_meta=None):
    reg_map = dict(reg_meta) if isinstance(reg_meta, dict) else {}
    verdict = compute_baseline_verdict_v1(
        mode=str(BASELINE_VERDICT_MODE),
        tol_status=tol_status,
        tangent_status=tangent_status,
        fit_status=fit_status,
        metrics=tol_metrics if isinstance(tol_metrics, dict) else None,
        reg_status=str(reg_map.get("status", "")),
        paired=paired_count,
        icp_rmse_mm=_safe_float(reg_map.get("icp_inlier_rmse_mm", float("nan"))),
        icp_fitness=_safe_float(reg_map.get("icp_fitness", float("nan"))),
        base_bias_final_norm_mm=_safe_float(reg_map.get("base_bias_final_norm_mm", float("nan"))),
        pass_guard_ratio=float(BASELINE_PASS_GUARD_RATIO),
        fail_guard_ratio=float(BASELINE_FAIL_GUARD_RATIO),
        hard_fail_margin_ratio=float(BASELINE_HARD_FAIL_MARGIN_RATIO),
        min_paired=int(BASELINE_MIN_PAIRED),
        min_icp_fitness=float(BASELINE_MIN_ICP_FITNESS),
        max_icp_rmse_mm=float(BASELINE_MAX_ICP_RMSE_MM),
        max_base_bias_mm=float(BASELINE_MAX_BASE_BIAS_MM),
    )
    baseline_verdict, baseline_reason = _collapse_need_rescan_to_binary(
        verdict.get("baseline_verdict", ""),
        verdict.get("baseline_reason", ""),
        tol_status,
    )
    return {
        "baseline_verdict": str(baseline_verdict),
        "baseline_reason": str(baseline_reason),
        "uncertainty_flags": _format_flag_list_text(verdict.get("uncertainty_flags", [])),
        "baseline_confidence_bucket": str(verdict.get("baseline_confidence_bucket", "")),
        "position_margin_ratio": _safe_float(verdict.get("position_margin_ratio", float("nan"))),
        "direction_margin_ratio": _safe_float(verdict.get("direction_margin_ratio", float("nan"))),
        "compare_alignment_mode": str(verdict.get("compare_alignment_mode", "fixed_station") or "fixed_station"),
    }


def _build_formal_decision_shadow_payload(tol_metrics, gate_source_fields):
    tol_metrics_map = dict(tol_metrics) if isinstance(tol_metrics, dict) else {}
    gate_fields = dict(gate_source_fields) if isinstance(gate_source_fields, dict) else {}
    local_segment_fields = {
        "tol_fail_subtype": str(tol_metrics_map.get("tol_fail_subtype", "")),
        "ab_shadow_reason": str(tol_metrics_map.get("ab_shadow_reason", "")),
        "point_score_selected_raw_max": _safe_float(tol_metrics_map.get("point_score_selected_raw_max", float("nan"))),
        "needle_score_raw": _safe_float(tol_metrics_map.get("needle_score_raw", float("nan"))),
        "pred_pos_bins": int(_safe_float(tol_metrics_map.get("pred_pos_bins", 0.0), default=0.0)),
        "pred_run_count": int(_safe_float(tol_metrics_map.get("pred_run_count", 0.0), default=0.0)),
        "pred_max_run_span_mm": _safe_float(tol_metrics_map.get("pred_max_run_span_mm", float("nan"))),
        "pred_dominant_run_frac": _safe_float(tol_metrics_map.get("pred_dominant_run_frac", float("nan"))),
    }
    verdict = compute_formal_decision_shadow_v1(
        base_pred=str(tol_metrics_map.get("ab_shadow_verdict", "")),
        gate_source_fields=gate_fields,
        gate_rule="global_only_v1",
        min_paired=int(BASELINE_MIN_PAIRED),
        local_override_rule="p4_no_rescan_contiguous_prior_v1",
        local_segment_fields=local_segment_fields,
    )
    policy_name = str(verdict.get("formal_decision_policy_shadow", ""))
    rule_version = str(verdict.get("formal_decision_rule_version", ""))
    if DETECTION_DISABLE_RESCAN:
        base_pred_shadow = str(verdict.get("formal_decision_base_pred_shadow", "")).strip()
        local_status_shadow = str(verdict.get("formal_decision_local_status_shadow", "")).strip()
        local_reason_shadow = str(verdict.get("formal_decision_local_reason_shadow", "")).strip()
        if local_status_shadow == "pass":
            final_verdict_shadow = "pass"
            final_reason_shadow = local_reason_shadow or "local_override_pass"
        elif base_pred_shadow in {"pass", "fail"}:
            final_verdict_shadow = base_pred_shadow
            final_reason_shadow = f"base_{base_pred_shadow}"
        else:
            final_verdict_shadow = "fail"
            final_reason_shadow = "base_nonbinary_as_fail"
        if policy_name:
            policy_name = f"{policy_name}__binary_no_rescan"
        if rule_version:
            rule_version = f"{rule_version}__binary_no_rescan"
    else:
        final_verdict_shadow = str(verdict.get("formal_decision_verdict_shadow", ""))
        final_reason_shadow = str(verdict.get("formal_decision_reason_shadow", ""))
    return {
        "formal_decision_rule_version": str(rule_version),
        "formal_decision_policy": str(policy_name),
        "formal_decision_base_source": str(verdict.get("formal_decision_base_source_shadow", "")),
        "formal_decision_base_pred": str(verdict.get("formal_decision_base_pred_shadow", "")),
        "formal_decision_gate_rule": str(verdict.get("formal_decision_gate_rule_shadow", "")),
        "formal_decision_gate_status": str(verdict.get("formal_decision_gate_status_shadow", "")),
        "formal_decision_gate_reason": str(verdict.get("formal_decision_gate_reason_shadow", "")),
        "formal_decision_gate_hit_text": str(verdict.get("formal_decision_gate_hit_text_shadow", "")),
        "formal_decision_local_rule": str(verdict.get("formal_decision_local_rule_shadow", "")),
        "formal_decision_local_status": str(verdict.get("formal_decision_local_status_shadow", "")),
        "formal_decision_local_reason": str(verdict.get("formal_decision_local_reason_shadow", "")),
        "formal_decision_verdict": str(final_verdict_shadow),
        "formal_decision_reason": str(final_reason_shadow),
        "formal_decision_policy_shadow": str(policy_name),
        "formal_decision_base_source_shadow": str(verdict.get("formal_decision_base_source_shadow", "")),
        "formal_decision_base_pred_shadow": str(verdict.get("formal_decision_base_pred_shadow", "")),
        "formal_decision_gate_rule_shadow": str(verdict.get("formal_decision_gate_rule_shadow", "")),
        "formal_decision_gate_status_shadow": str(verdict.get("formal_decision_gate_status_shadow", "")),
        "formal_decision_gate_reason_shadow": str(verdict.get("formal_decision_gate_reason_shadow", "")),
        "formal_decision_gate_hit_text_shadow": str(verdict.get("formal_decision_gate_hit_text_shadow", "")),
        "formal_decision_local_rule_shadow": str(verdict.get("formal_decision_local_rule_shadow", "")),
        "formal_decision_local_status_shadow": str(verdict.get("formal_decision_local_status_shadow", "")),
        "formal_decision_local_reason_shadow": str(verdict.get("formal_decision_local_reason_shadow", "")),
        "formal_decision_verdict_shadow": str(final_verdict_shadow),
        "formal_decision_reason_shadow": str(final_reason_shadow),
    }


def evaluate_centerline_against_tolerance(
    nominal_curve_u,
    defect_curve_u,
    *,
    tol_lateral_mm,
    tol_tip_short_mm,
    tol_len_short_mm,
    unit_scale,
    tol_tangent_axis_xyz=None,
    tol_tangent_half_angle_deg=None,
    tol_tangent_window_mm=None,
    tol_tangent_model_status=None,
):
    return evaluate_centerline_tolerance_with_direction(
        nominal_curve_u,
        defect_curve_u,
        tol_lateral_mm=tol_lateral_mm,
        tol_tip_short_mm=tol_tip_short_mm,
        tol_len_short_mm=tol_len_short_mm,
        unit_scale=unit_scale,
        tol_tangent_axis_xyz=tol_tangent_axis_xyz,
        tol_tangent_half_angle_deg=tol_tangent_half_angle_deg,
        tol_tangent_window_mm=tol_tangent_window_mm,
        tol_tangent_model_status=tol_tangent_model_status,
        use_direction_score=False,
    )


def _analyze_tolerance_pair(info_ref, info_def, pair, pairing_result, *, final_method, reg_meta=None, step_mm=0.2):
    idx_ref = int(pair.get("normal_id", -1))
    idx_def = int(pair.get("defect_id", -1))
    crv_ref_raw = orient_curve_to_base_start(
        info_ref.get("curve_global_final", np.empty((0, 3))),
        info_ref.get("base_end_idx_local", None),
    )
    crv_def_raw = orient_curve_to_base_start(
        info_def.get("curve_global_final", np.empty((0, 3))),
        info_def.get("base_end_idx_local", None),
    )
    if len(crv_ref_raw) < 2 or len(crv_def_raw) < 2:
        return None, None

    pts_centerline_ref_final = resample_curve_by_step(crv_ref_raw, step=mm_to_unit_len(step_mm))
    pts_centerline_def_final = resample_curve_by_step(crv_def_raw, step=mm_to_unit_len(step_mm))
    scores_final = np.zeros(len(pts_centerline_def_final), dtype=np.float64)
    final_used_method = "tangent_fallback"
    tol_status = "unknown"
    tangent_status = "unknown"
    tol_metrics = {}

    if str(final_method) == "tolerance_band":
        model_status = str(info_ref.get("tol_model_status", info_ref.get("status", "ok")))
        fit_status = str(info_def.get("overall_status", "ok"))
        if model_status != "ok":
            tol_status = "model_insufficient"
            tangent_status = "model_insufficient"
        else:
            tol_lat = info_ref.get("tol_lateral_mm", None)
            if tol_lat is None or len(tol_lat) < 2:
                tol_status = "model_insufficient"
                tangent_status = "model_insufficient"
            else:
                tol_scores, tol_metrics, nominal_n, defect_n = evaluate_centerline_against_tolerance(
                    crv_ref_raw,
                    crv_def_raw,
                    tol_lateral_mm=tol_lat,
                    tol_tip_short_mm=info_ref.get("tol_tip_short_mm", None),
                    tol_len_short_mm=info_ref.get("tol_len_short_mm", None),
                    unit_scale=info_ref.get("tol_unit_scale", PIN_UNIT_SCALE),
                    tol_tangent_axis_xyz=info_ref.get("tol_tangent_axis_xyz", None),
                    tol_tangent_half_angle_deg=info_ref.get("tol_tangent_half_angle_deg", None),
                    tol_tangent_window_mm=info_ref.get("tol_tangent_window_mm", None),
                    tol_tangent_model_status=info_ref.get("tol_tangent_model_status", None),
                )
                tol_metrics = _apply_detection_policy_to_tol_metrics(tol_metrics)
                if tol_scores is None or len(tol_scores) < 2:
                    tol_status = "fail" if DETECTION_DISABLE_RESCAN else "need_rescan"
                    tangent_status = "model_insufficient"
                else:
                    scores_final = np.asarray(tol_scores, dtype=np.float64).reshape(-1)
                    pts_centerline_ref_final = np.asarray(nominal_n, dtype=np.float64)
                    pts_centerline_def_final = np.asarray(defect_n, dtype=np.float64)
                    final_used_method = "tolerance_band"

                    tol_status = _compute_lateral_only_tol_status(tol_metrics)
                    tangent_status = (
                        str(tol_metrics.get("tangent_status", "model_insufficient"))
                        if isinstance(tol_metrics, dict)
                        else "model_insufficient"
                    )
                    if fit_status == "high_risk" and not DETECTION_DISABLE_RESCAN:
                        tol_status = "need_rescan"
                        tangent_status = "need_rescan"
    else:
        tol_status = "unsupported_method"
        tangent_status = "unsupported_method"

    if len(pts_centerline_def_final) != len(scores_final):
        n_min = min(len(pts_centerline_def_final), len(scores_final))
        if n_min < 2:
            return None, None
        pts_centerline_ref_final = np.asarray(pts_centerline_ref_final[:n_min], dtype=np.float64)
        pts_centerline_def_final = np.asarray(pts_centerline_def_final[:n_min], dtype=np.float64)
        scores_final = np.asarray(scores_final[:n_min], dtype=np.float64)

    paired_count = len(pairing_result.get("pairs", [])) if isinstance(pairing_result, dict) else 0
    pair_dist_u = _safe_float(pair.get("dist", float("nan")))
    shadow_gate_payload = _build_shadow_gate_payload(
        info_def=info_def,
        reg_meta=reg_meta if isinstance(reg_meta, dict) else {},
        paired_count=paired_count,
        pair_dist_u=pair_dist_u,
    )
    baseline_payload = _build_baseline_verdict_payload(
        tol_status=tol_status,
        tangent_status=tangent_status,
        fit_status=str(info_def.get("overall_status", "")),
        tol_metrics=tol_metrics if isinstance(tol_metrics, dict) else {},
        paired_count=paired_count,
        reg_meta=reg_meta,
    )
    tol_metrics_map = dict(tol_metrics) if isinstance(tol_metrics, dict) else {}
    segment_feature_payload = compute_selected_score_run_features_v1(
        point_score_selected_raw_arr=tol_metrics_map.get("point_score_selected_raw_arr", []),
        station_step_mm=tol_metrics_map.get("ab_station_step_mm", 0.2),
        positive_threshold=0.0,
    )
    tol_metrics_map.update(segment_feature_payload)
    formal_shadow_payload = _build_formal_decision_shadow_payload(
        tol_metrics=tol_metrics_map,
        gate_source_fields=shadow_gate_payload.get("gate_source_fields", {}),
    )
    record = {
        "ref_index": int(idx_ref),
        "def_index": int(idx_def),
        "pair_dist_u": float(pair_dist_u),
        "stable_id": int(info_ref.get("stable_id", -1)),
        "fit_status": str(info_def.get("overall_status", "")),
        "final_used_method": str(final_used_method),
        "tol_model_status": str(info_ref.get("tol_model_status", info_ref.get("status", ""))),
        "tol_status": str(tol_status) if tol_status is not None else "",
        "tangent_status": str(tangent_status),
        "tol_fail_subtype": str(tol_metrics_map.get("tol_fail_subtype", "")),
        "point_score_raw_max": _safe_float(tol_metrics_map.get("point_score_raw_max", float("nan"))),
        "point_score_clipped_max": _safe_float(tol_metrics_map.get("point_score_clipped_max", float("nan"))),
        "point_score_fixed_raw_max": _safe_float(tol_metrics_map.get("point_score_fixed_raw_max", float("nan"))),
        "point_score_slack_raw_max": _safe_float(tol_metrics_map.get("point_score_slack_raw_max", float("nan"))),
        "point_score_selected_raw_max": _safe_float(tol_metrics_map.get("point_score_selected_raw_max", float("nan"))),
        "needle_score_raw": _safe_float(tol_metrics_map.get("needle_score_raw", float("nan"))),
        "needle_score_clipped": _safe_float(tol_metrics_map.get("needle_score_clipped", float("nan"))),
        "needle_score_q95": _safe_float(tol_metrics_map.get("needle_score_q95", float("nan"))),
        "needle_score_topk_mean": _safe_float(tol_metrics_map.get("needle_score_topk_mean", float("nan"))),
        "needle_score_topk_count": int(_safe_float(tol_metrics_map.get("needle_score_topk_count", 0.0), default=0.0)),
        "ab_station_step_mm": _safe_float(tol_metrics_map.get("ab_station_step_mm", float("nan"))),
        "pred_pos_bins": int(_safe_float(tol_metrics_map.get("pred_pos_bins", 0.0), default=0.0)),
        "pred_run_count": int(_safe_float(tol_metrics_map.get("pred_run_count", 0.0), default=0.0)),
        "pred_max_run_bins": int(_safe_float(tol_metrics_map.get("pred_max_run_bins", 0.0), default=0.0)),
        "pred_max_run_span_mm": _safe_float(tol_metrics_map.get("pred_max_run_span_mm", float("nan"))),
        "pred_dominant_run_frac": _safe_float(tol_metrics_map.get("pred_dominant_run_frac", float("nan"))),
        "pred_singleton_run_count": int(_safe_float(tol_metrics_map.get("pred_singleton_run_count", 0.0), default=0.0)),
        "pred_short_run_bins": int(_safe_float(tol_metrics_map.get("pred_short_run_bins", 0.0), default=0.0)),
        "pred_short_run_frac": _safe_float(tol_metrics_map.get("pred_short_run_frac", float("nan"))),
        "ab_compare_mode": str(tol_metrics_map.get("ab_compare_mode", "fixed_station") or "fixed_station"),
        "ab_readout_mode": str(tol_metrics_map.get("ab_readout_mode", "raw_max") or "raw_max"),
        "ab_slack_radius_steps": int(_safe_float(tol_metrics_map.get("ab_slack_radius_steps", 0.0), default=0.0)),
        "ab_shadow_verdict": str(tol_metrics_map.get("ab_shadow_verdict", "")),
        "ab_shadow_reason": str(tol_metrics_map.get("ab_shadow_reason", "")),
        "ab_slack_gain": _safe_float(tol_metrics_map.get("ab_slack_gain", float("nan"))),
        "warp_severity": _safe_float(tol_metrics_map.get("ab_warp_p90_mm", float("nan"))),
        "ab_warp_p90_steps": _safe_float(tol_metrics_map.get("ab_warp_p90_steps", float("nan"))),
        "ab_warp_max_steps": _safe_float(tol_metrics_map.get("ab_warp_max_steps", float("nan"))),
        "ab_warp_p90_mm": _safe_float(tol_metrics_map.get("ab_warp_p90_mm", float("nan"))),
        "ab_warp_max_mm": _safe_float(tol_metrics_map.get("ab_warp_max_mm", float("nan"))),
        "ab_warp_nonzero_ratio": _safe_float(tol_metrics_map.get("ab_warp_nonzero_ratio", float("nan"))),
        "max_lateral_ratio": _safe_float(tol_metrics_map.get("max_lateral_ratio", float("nan"))),
        "tip_short_ratio": _safe_float(tol_metrics_map.get("tip_short_ratio", float("nan"))),
        "len_short_ratio": _safe_float(tol_metrics_map.get("len_short_ratio", float("nan"))),
        "max_tangent_ratio": _safe_float(tol_metrics_map.get("max_tangent_ratio", float("nan"))),
        "tol_metrics": tol_metrics_map,
        **baseline_payload,
        **shadow_gate_payload,
        **formal_shadow_payload,
    }
    artifact = {
        "info_ref": info_ref,
        "info_def": info_def,
        "pts_centerline_ref_final": np.asarray(pts_centerline_ref_final, dtype=np.float64),
        "pts_centerline_def_final": np.asarray(pts_centerline_def_final, dtype=np.float64),
        "scores_final": np.asarray(scores_final, dtype=np.float64),
        "tol_metrics": tol_metrics_map,
        "tol_status": str(tol_status),
        "tangent_status": str(tangent_status),
        "final_used_method": str(final_used_method),
    }
    return record, artifact


def detect_torsion_fine_grained(curve_ref, curve_def, cloud_def, step_mm=0.2, pin_width=1.0, max_angle=30.0):
    """
    [最终版] 细粒度扭转检测：直接比较中心线切向量，不受切片形状干扰。

    参数:
        curve_ref, curve_def: (N, 3) 重采样后的平滑中心线
        cloud_def: 待测针的原始点云 (用于辅助可视化)
        step_mm: 切片步长 (越小越精细，对应 resample_curve_by_step 的 step)
        pin_width: 针宽，用于控制切片厚度
        max_angle: 最大容许角度 (度)，超过此值为 100% 异常
    返回:
        (scores, curve_resampled): 切片分数数组 + 重采样中心线
    """
    # 1) Convert step from mm -> curve coordinate units (see Units section above)
    step_u = mm_to_unit_len(step_mm)
    if step_u <= 0:
        step_u = float(step_mm)

    # 2) Resample (ensure tangents remain smooth)
    curve_ref_sampled = resample_curve_by_step(curve_ref, step=step_u)
    curve_def_sampled = resample_curve_by_step(curve_def, step=step_u)

    if len(curve_ref_sampled) < 2 or len(curve_def_sampled) < 2:
        return np.zeros(len(curve_def_sampled)), curve_def_sampled

    min_len = min(len(curve_ref_sampled), len(curve_def_sampled))
    curve_ref_sampled = curve_ref_sampled[:min_len]
    curve_def_sampled = curve_def_sampled[:min_len]

    # 3) Compute tangents (Numerical Gradient)
    def _normalize(v):
        nrm = np.linalg.norm(v, axis=1)
        nrm[nrm < 1e-8] = 1.0
        return v / nrm[:, np.newaxis]

    tangents_ref = _normalize(np.gradient(curve_ref_sampled, axis=0))

    # Evaluate both orientations for curve_def to avoid "start/end mismatch" (especially around elbows).
    # Note: abs(dot) removes tangent sign ambiguity, but curve reversal changes the parameterization order,
    # which can otherwise look like a large defect around the elbow.
    def _scores_for(def_curve_sampled):
        tangents_def = _normalize(np.gradient(def_curve_sampled, axis=0))
        dot = np.einsum("ij,ij->i", tangents_ref, tangents_def)
        dot = np.clip(dot, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(np.abs(dot)))
        return np.clip(angle_deg / float(max_angle), 0.0, 1.0)

    scores_fwd = _scores_for(curve_def_sampled)
    scores_rev = _scores_for(curve_def_sampled[::-1])

    def _mean_core(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) <= 2:
            return float(np.mean(x)) if len(x) > 0 else 0.0
        return float(np.mean(x[1:-1]))

    if _mean_core(scores_rev) + 1e-12 < _mean_core(scores_fwd):
        return np.asarray(scores_rev, dtype=np.float64), curve_def_sampled[::-1]

    return np.asarray(scores_fwd, dtype=np.float64), curve_def_sampled


def create_uniform_tube(points, radius, color):
    """
    创建单一颜色的管道
    """
    if len(points) < 2:
        return o3d.geometry.TriangleMesh()
    try:
        return create_tube_from_points(points, radius, color, num_segments=16)
    except:
        return o3d.geometry.TriangleMesh()


def downsample_points_np(points, max_points):
    if points is None:
        return points
    if max_points is None or max_points <= 0:
        return points
    n = len(points)
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return points[idx]


def create_uniform_lineset(points, color):
    if points is None or len(points) < 2:
        return o3d.geometry.LineSet()
    pts = np.asarray(points, dtype=np.float64)
    lines = np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))]).astype(np.int32)
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def create_heatmap_lineset(points, scores):
    if points is None or scores is None or len(points) < 2 or len(scores) < 2:
        return o3d.geometry.LineSet()
    pts = np.asarray(points, dtype=np.float64)
    lines = np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))]).astype(np.int32)
    seg_scores = (np.asarray(scores[:-1], dtype=np.float64) + np.asarray(scores[1:], dtype=np.float64)) / 2.0
    colors = np.asarray([score_to_color(float(s)) for s in seg_scores], dtype=np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def create_tangent_arrow_lineset(curve_points, step=5, arrow_len=2.0, color=[0, 1, 0]):
    if curve_points is None or len(curve_points) < 2:
        return o3d.geometry.LineSet()

    pts = np.asarray(curve_points, dtype=np.float64)
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-8] = 1.0
    tangents /= norms[:, np.newaxis]

    arrow_starts = []
    arrow_ends = []
    for i in range(0, len(pts), max(int(step), 1)):
        p = pts[i]
        v = tangents[i]
        arrow_starts.append(p)
        arrow_ends.append(p + v * float(arrow_len))

    if len(arrow_starts) == 0:
        return o3d.geometry.LineSet()

    arrow_starts = np.asarray(arrow_starts, dtype=np.float64)
    arrow_ends = np.asarray(arrow_ends, dtype=np.float64)
    all_points = np.vstack([arrow_starts, arrow_ends])

    n = len(arrow_starts)
    lines = np.column_stack([np.arange(n), np.arange(n) + n]).astype(np.int32)
    colors = np.tile(np.asarray(color, dtype=np.float64), (n, 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _normalize_vec(v):
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    if x.size != 3:
        return np.zeros((3,), dtype=np.float64)
    n = float(np.linalg.norm(x))
    if (not np.isfinite(n)) or n <= 1e-12:
        return np.zeros((3,), dtype=np.float64)
    return x / n


def _orthonormal_basis_from_axis(axis):
    a = _normalize_vec(axis)
    if np.linalg.norm(a) <= 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(a, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(a, ref)
    u = _normalize_vec(u)
    if np.linalg.norm(u) <= 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = _normalize_vec(np.cross(a, ref))
    v = _normalize_vec(np.cross(a, u))
    return u, v


def _make_colored_lineset(points, lines, colors):
    if points is None or lines is None or len(points) <= 0 or len(lines) <= 0:
        return o3d.geometry.LineSet()
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def create_direction_cone_lineset(
    anchor_points,
    axis_vectors,
    half_angles_deg,
    *,
    cone_len,
    stride=8,
    skip_end_points=2,
    ring_segments=10,
    out_mask=None,
    color_ok=None,
    color_fail=None,
):
    pts = np.asarray(anchor_points, dtype=np.float64).reshape(-1, 3)
    axes = np.asarray(axis_vectors, dtype=np.float64).reshape(-1, 3)
    half = np.asarray(half_angles_deg, dtype=np.float64).reshape(-1)
    n = min(len(pts), len(axes), len(half))
    if n < 2:
        return o3d.geometry.LineSet()
    stride = max(1, int(stride))
    skip = max(0, int(skip_end_points))
    ring_segments = max(6, int(ring_segments))
    cone_len = float(cone_len)
    if (not np.isfinite(cone_len)) or cone_len <= 1e-12:
        return o3d.geometry.LineSet()
    mask = np.asarray(out_mask, dtype=bool).reshape(-1) if out_mask is not None else np.zeros((n,), dtype=bool)
    color_ok = np.asarray(color_ok if color_ok is not None else [0.15, 0.75, 1.0], dtype=np.float64)
    color_fail = np.asarray(color_fail if color_fail is not None else [1.0, 0.35, 0.20], dtype=np.float64)

    all_points = []
    all_lines = []
    all_colors = []
    for idx in range(skip, max(skip, n - skip), stride):
        axis = _normalize_vec(axes[idx])
        if np.linalg.norm(axis) <= 1e-12:
            continue
        ang = float(np.clip(half[idx], 0.0, 89.0))
        radius = float(np.tan(np.deg2rad(ang)) * cone_len)
        apex = pts[idx]
        center = apex + axis * cone_len
        u, v = _orthonormal_basis_from_axis(axis)
        color = color_fail if (idx < len(mask) and bool(mask[idx])) else color_ok

        base_index = len(all_points)
        all_points.append(apex)
        all_points.append(center)
        all_lines.append([base_index, base_index + 1])
        all_colors.append(color)

        ring_ids = []
        for j in range(ring_segments):
            phi = (2.0 * np.pi * float(j)) / float(ring_segments)
            ring_pt = center + radius * (np.cos(phi) * u + np.sin(phi) * v)
            ring_ids.append(len(all_points))
            all_points.append(ring_pt)
        for rid in ring_ids:
            all_lines.append([base_index, rid])
            all_colors.append(color)
        for j in range(len(ring_ids)):
            all_lines.append([ring_ids[j], ring_ids[(j + 1) % len(ring_ids)]])
            all_colors.append(color)
    return _make_colored_lineset(all_points, all_lines, all_colors)


def create_direction_arrow_lineset(
    anchor_points,
    direction_vectors,
    *,
    arrow_len,
    stride=8,
    skip_end_points=2,
    out_mask=None,
    color_ok=None,
    color_fail=None,
):
    pts = np.asarray(anchor_points, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(direction_vectors, dtype=np.float64).reshape(-1, 3)
    n = min(len(pts), len(dirs))
    if n < 1:
        return o3d.geometry.LineSet()
    stride = max(1, int(stride))
    skip = max(0, int(skip_end_points))
    arrow_len = float(arrow_len)
    if (not np.isfinite(arrow_len)) or arrow_len <= 1e-12:
        return o3d.geometry.LineSet()
    mask = np.asarray(out_mask, dtype=bool).reshape(-1) if out_mask is not None else np.zeros((n,), dtype=bool)
    color_ok = np.asarray(color_ok if color_ok is not None else [0.15, 0.95, 0.20], dtype=np.float64)
    color_fail = np.asarray(color_fail if color_fail is not None else [1.0, 0.20, 0.20], dtype=np.float64)

    all_points = []
    all_lines = []
    all_colors = []
    head_len = arrow_len * 0.25
    head_width = arrow_len * 0.18
    for idx in range(skip, max(skip, n - skip), stride):
        direction = _normalize_vec(dirs[idx])
        if np.linalg.norm(direction) <= 1e-12:
            continue
        start = pts[idx]
        end = start + direction * arrow_len
        color = color_fail if (idx < len(mask) and bool(mask[idx])) else color_ok
        u, _ = _orthonormal_basis_from_axis(direction)
        head_a = end - direction * head_len + u * head_width
        head_b = end - direction * head_len - u * head_width
        base_index = len(all_points)
        all_points.extend([start, end, head_a, head_b])
        all_lines.extend([
            [base_index, base_index + 1],
            [base_index + 1, base_index + 2],
            [base_index + 1, base_index + 3],
        ])
        all_colors.extend([color, color, color])
    return _make_colored_lineset(all_points, all_lines, all_colors)


def create_masked_segment_lineset(points, mask, color, *, require_both=False):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    flag = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(len(pts), len(flag))
    if n < 2:
        return o3d.geometry.LineSet()
    pts = pts[:n]
    flag = flag[:n]
    lines = []
    for i in range(n - 1):
        use_seg = bool(flag[i] and flag[i + 1]) if require_both else bool(flag[i] or flag[i + 1])
        if use_seg:
            lines.append([i, i + 1])
    if not lines:
        return o3d.geometry.LineSet()
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    return _make_colored_lineset(pts, lines, colors)


def create_heatmap_tube(points, scores, radius):
    """
    创建热力图管道 (按分数分段着色)
    """
    if len(points) < 2 or len(scores) < 2:
        return o3d.geometry.TriangleMesh()

    meshes = []
    for i in range(len(points) - 1):
        seg = np.array([points[i], points[i + 1]])
        avg_score = (scores[i] + scores[i + 1]) / 2.0
        color = score_to_color(avg_score)
        try:
            m = create_tube_from_points(seg, radius, color, num_segments=16)
            if m.has_vertices():
                meshes.append(m)
        except:
            pass

    combined = o3d.geometry.TriangleMesh()
    for m in meshes:
        combined += m
    return combined


def create_tolerance_band_mesh(points, tol_lateral_mm, unit_scale):
    """
    Build a tolerance band mesh as per-segment variable-radius tubes around nominal centerline.
    """
    pts = np.asarray(points, dtype=np.float64)
    tol_mm = np.asarray(tol_lateral_mm, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return o3d.geometry.TriangleMesh()
    if tol_mm.size < 2:
        return o3d.geometry.TriangleMesh()

    n = len(pts)
    if tol_mm.size != n:
        tol_mm = np.interp(
            np.linspace(0.0, 1.0, n),
            np.linspace(0.0, 1.0, max(2, int(tol_mm.size))),
            tol_mm,
        )

    u = _safe_float(unit_scale, default=PIN_UNIT_SCALE)
    if (not np.isfinite(u)) or (u <= 1e-12):
        u = float(PIN_UNIT_SCALE) if float(PIN_UNIT_SCALE) > 1e-12 else 1.0

    min_r_u = float(mm_to_unit_len(VIS_TOLERANCE_BAND_MIN_RADIUS_MM))
    if not np.isfinite(min_r_u) or min_r_u <= 0:
        min_r_u = 1e-6

    meshes = []
    build_fail_count = 0
    for i in range(n - 1):
        seg = np.asarray([pts[i], pts[i + 1]], dtype=np.float64)
        r_mm = 0.5 * (float(tol_mm[i]) + float(tol_mm[i + 1]))
        r_u = max(float(r_mm) * float(u), min_r_u)
        try:
            m = create_tube_from_points(seg, radius=r_u, color=[1.0, 1.0, 1.0])
            if m is not None and m.has_vertices():
                meshes.append(m)
        except Exception:
            build_fail_count += 1
            continue

    if len(meshes) <= 0:
        print(
            "[WARN] Tolerance band mesh is empty: "
            f"segments={n - 1}, failed={build_fail_count}"
        )
    elif build_fail_count > 0:
        print(
            "[WARN] Tolerance band mesh partially built: "
            f"ok={len(meshes)}, failed={build_fail_count}"
        )

    band = o3d.geometry.TriangleMesh()
    for m in meshes:
        band += m
    return band


def _draw_with_optional_materials(geoms, material_records):
    """
    Prefer Open3D draw() for alpha materials; fallback to draw_geometries().
    """
    if not material_records:
        return False

    if os.environ.get("DAIMA_FORCE_LEGACY_VIS", "").strip() == "1":
        return False

    if not hasattr(o3d.visualization, "draw"):
        return False

    try:
        draw_items = []
        for i, g in enumerate(geoms):
            item = {"name": f"geom_{i}", "geometry": g}
            mid = id(g)
            if mid in material_records:
                item["material"] = material_records[mid]
            draw_items.append(item)

        o3d.visualization.draw(
            draw_items,
            title="Detection Result",
            width=1600,
            height=800,
            show_skybox=False,
            bg_color=(1.0, 1.0, 1.0, 1.0),
        )
        return True
    except Exception as e:
        # If draw() fails after opening the GUI window, explicitly quit the GUI app to avoid a lingering black window
        # that keeps the Python process alive.
        print(f"[WARN] draw(material) failed, fallback to draw_geometries: {e}")
        try:
            import open3d.visualization.gui as gui

            gui.Application.instance.quit()
        except Exception:
            pass
    return False


# ===================== 三阶段检测主流程 =====================

def visualize_with_matching_and_heatmap(base_ref, base_def,
                                        infos_ref, infos_def_aligned,
                                        pairing_result,
                                        step_mm=0.2,
                                        global_pin_width=1.0,
                                        save_path=None,
                                        heat_save_path=None,
                                        scene_save_path=None,
                                        scene_failred_save_path=None,
                                        final_method=None,
                                        reg_meta=None):
    """
    [修改版] 四模块检测 + 可视化：
    1. 粗粒度检测 (全针 PCA)
    2. 细粒度检测 (切片 PCA)
    3. 综合显示 (细粒度结果)
    4. 方向偏移容差锥视图

    新增参数:
        save_path: 兼容旧调用，等价于 heat_save_path
        heat_save_path: 热力图点云导出路径 (ply)
        scene_save_path: 多视图 scene 点云导出路径 (ply)
        scene_failred_save_path: 多视图 scene + fail 针红色 overlay 导出路径 (ply)
    """
    if heat_save_path is None and save_path is not None:
        heat_save_path = save_path
    print(f"[INFO] Visualizing tolerance-only heatmaps (step={step_mm} mm)...")
    try:
        print(
            f"[INFO] Step conversion: step_mm={float(step_mm):g} -> step_u={mm_to_unit_len(step_mm):.6g} "
            f"(UNIT_SCALE={float(PIN_UNIT_SCALE):.6g})"
        )
    except Exception:
        pass

    if final_method is None:
        final_method = str(FINAL_METHOD).strip() or "tangent"

    geoms = []
    pcds_to_save = []
    material_records = {}
    analysis_report = {
        "final_method": str(final_method),
        "step_mm": float(step_mm),
        "pin_unit_scale": float(PIN_UNIT_SCALE),
        "baseline_verdict_mode": str(BASELINE_VERDICT_MODE),
        "decision_status_source": "formal_decision_verdict",
        "display_status_source": "tol_scores",
        "display_note": "P5-C promoted: main verdict uses formal_decision_verdict; heatmap remains tol_scores explanation.",
        "missing_normal": list(pairing_result.get("missing_normal", [])) if isinstance(pairing_result, dict) else [],
        "extra_defect": list(pairing_result.get("extra_defect", [])) if isinstance(pairing_result, dict) else [],
        "needles": [],
    }

    offset_centerline = np.array([-global_pin_width * 10.0, 0.0, 0.0], dtype=np.float64)
    offset_cloud = np.array([global_pin_width * 10.0, 0.0, 0.0], dtype=np.float64)

    if VIS_SHOW_BASE and base_ref is not None:
        base_ref_vis = copy.deepcopy(base_ref)
        base_ref_vis.paint_uniform_color([0.5, 0.5, 0.5])
        for off in [offset_centerline, offset_cloud]:
            b_clone = copy.deepcopy(base_ref_vis)
            b_clone.translate(off)
            geoms.append(b_clone)

    for pair in pairing_result.get("pairs", []):
        idx_ref = int(pair.get("normal_id", -1))
        idx_def = int(pair.get("defect_id", -1))
        if idx_ref < 0 or idx_ref >= len(infos_ref) or idx_def < 0 or idx_def >= len(infos_def_aligned):
            continue

        info_ref = infos_ref[idx_ref]
        info_def = infos_def_aligned[idx_def]
        pair_record, pair_artifact = _analyze_tolerance_pair(
            info_ref,
            info_def,
            pair,
            pairing_result,
            final_method=final_method,
            reg_meta=reg_meta,
            step_mm=step_mm,
        )
        if pair_record is None or pair_artifact is None:
            continue

        analysis_report["needles"].append(pair_record)
        pts_centerline_ref_final = np.asarray(pair_artifact["pts_centerline_ref_final"], dtype=np.float64)
        pts_centerline_def_final = np.asarray(pair_artifact["pts_centerline_def_final"], dtype=np.float64)
        scores_final = np.asarray(pair_artifact["scores_final"], dtype=np.float64)
        final_used_method = str(pair_artifact.get("final_used_method", ""))
        tol_metrics = dict(pair_artifact.get("tol_metrics", {}))

        if (
            VIS_SHOW_TOLERANCE_BAND
            and final_used_method == "tolerance_band"
            and len(pts_centerline_ref_final) >= 2
            and isinstance(tol_metrics, dict)
        ):
            tol_lat_vis = info_ref.get("tol_lateral_mm", None)
            if tol_lat_vis is not None and len(np.asarray(tol_lat_vis).reshape(-1)) >= 2:
                try:
                    band_mesh = create_tolerance_band_mesh(
                        points=pts_centerline_ref_final,
                        tol_lateral_mm=tol_lat_vis,
                        unit_scale=info_ref.get("tol_unit_scale", PIN_UNIT_SCALE),
                    )
                    if band_mesh is not None and band_mesh.has_vertices():
                        rgb = np.asarray(VIS_TOLERANCE_BAND_COLOR, dtype=np.float64).reshape(-1)
                        if rgb.size < 3:
                            rgb = np.array([0.2, 0.8, 1.0], dtype=np.float64)
                        rgb = np.clip(rgb[:3], 0.0, 1.0)
                        band_mesh.paint_uniform_color(rgb.tolist())
                        band_mesh.translate(offset_centerline)
                        geoms.append(band_mesh)

                        alpha = float(np.clip(VIS_TOLERANCE_BAND_ALPHA, 0.0, 1.0))
                        if alpha < 0.999:
                            try:
                                mat = o3d.visualization.rendering.MaterialRecord()
                                mat.shader = "defaultLitTransparency"
                                mat.base_color = [float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha]
                                mat.has_alpha = True
                                material_records[id(band_mesh)] = mat
                            except Exception:
                                pass
                    else:
                        print(
                            "[WARN] Skip tolerance band overlay for stable_id="
                            f"{info_ref.get('stable_id', -1)}: empty mesh"
                        )
                except Exception as e:
                    print(f"[WARN] Failed to build tolerance band overlay: {e}")

        if VIS_SHOW_DEFECT_CL and len(pts_centerline_def_final) >= 2:
            if VIS_FAST_MODE:
                defect_ls = create_heatmap_lineset(pts_centerline_def_final, scores_final)
                if len(defect_ls.points) > 0:
                    defect_ls.translate(offset_centerline)
                    geoms.append(defect_ls)
            else:
                defect_tube = create_heatmap_tube(pts_centerline_def_final, scores_final, TUBE_RADIUS_DEFECT)
                if defect_tube.has_vertices():
                    defect_tube.translate(offset_centerline)
                    geoms.append(defect_tube)

        if "points" in info_def and VIS_SHOW_DEFECT_PC and len(pts_centerline_def_final) >= 2:
            cloud_points_np = np.asarray(info_def.get("points", np.empty((0, 3))), dtype=np.float64)
            if len(cloud_points_np) > 0:
                try:
                    cloud_points_np = downsample_points_np(cloud_points_np, VIS_MAX_POINTS_PER_NEEDLE)
                    tree_def = cKDTree(np.asarray(pts_centerline_def_final, dtype=np.float64))
                    _, nearest_idx = tree_def.query(np.asarray(cloud_points_np, dtype=np.float64), k=1)
                    nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)

                    cloud_colors = []
                    for idx_local in nearest_idx:
                        s = scores_final[idx_local] if idx_local < len(scores_final) else 0.0
                        cloud_colors.append(score_to_color(float(s)))

                    pcd_heat = o3d.geometry.PointCloud()
                    pcd_heat.points = o3d.utility.Vector3dVector(cloud_points_np)
                    pcd_heat.colors = o3d.utility.Vector3dVector(np.asarray(cloud_colors, dtype=np.float64))
                    pcd_heat.translate(offset_cloud)
                    geoms.append(pcd_heat)
                    pcds_to_save.append(pcd_heat)
                except Exception:
                    pass

    yellow_color = [1.0, 1.0, 0.0]
    for idx in pairing_result.get("missing_normal", []):
        idx = int(idx)
        if idx < 0 or idx >= len(infos_ref):
            continue
        if "points" in infos_ref[idx]:
            pts = downsample_points_np(infos_ref[idx]["points"], VIS_MAX_POINTS_PER_NEEDLE)
            if len(pts) > 0:
                pcd_miss = o3d.geometry.PointCloud()
                pcd_miss.points = o3d.utility.Vector3dVector(pts)
                pcd_miss.paint_uniform_color(yellow_color)
                for off in [offset_centerline, offset_cloud]:
                    p_clone = copy.deepcopy(pcd_miss)
                    p_clone.translate(off)
                    geoms.append(p_clone)
                    if np.array_equal(off, offset_cloud):
                        pcds_to_save.append(p_clone)

    red_color = [1.0, 0.0, 0.0]
    for idx in pairing_result.get("extra_defect", []):
        idx = int(idx)
        if idx < 0 or idx >= len(infos_def_aligned):
            continue
        if "points" in infos_def_aligned[idx]:
            pts = downsample_points_np(infos_def_aligned[idx]["points"], VIS_MAX_POINTS_PER_NEEDLE)
            if len(pts) > 0:
                pcd_ext = o3d.geometry.PointCloud()
                pcd_ext.points = o3d.utility.Vector3dVector(pts)
                pcd_ext.paint_uniform_color(red_color)
                for off in [offset_centerline, offset_cloud]:
                    p_clone = copy.deepcopy(pcd_ext)
                    p_clone.translate(off)
                    geoms.append(p_clone)
                    if np.array_equal(off, offset_cloud):
                        pcds_to_save.append(p_clone)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=global_pin_width * 5, origin=[0, 0, 0])
    geoms.append(axes)

    # Visualization-only scaling: Open3D camera near-plane / zoom becomes unusable when the whole scene is tiny
    # in world units (e.g., UNIT_SCALE=0.001). This scales the displayed geometries only (after any save_path write)
    # so users can zoom in without aggressive clipping.
    try:
        vis_scale_s = os.environ.get("DAIMA_VIS_SCALE", "").strip()
        vis_target_diag_s = os.environ.get("DAIMA_VIS_TARGET_DIAG", "").strip()

        vis_scale = None
        diag = None
        target_diag = None

        if vis_scale_s:
            try:
                vis_scale = float(vis_scale_s)
            except Exception:
                vis_scale = None
        else:
            try:
                target_diag = float(vis_target_diag_s) if vis_target_diag_s else 10.0
            except Exception:
                target_diag = 10.0

            if np.isfinite(target_diag) and float(target_diag) > 0:
                min_b = None
                max_b = None
                for g in geoms:
                    try:
                        aabb = g.get_axis_aligned_bounding_box()
                        mi = np.asarray(aabb.get_min_bound(), dtype=np.float64).reshape(3)
                        ma = np.asarray(aabb.get_max_bound(), dtype=np.float64).reshape(3)
                        if not (np.all(np.isfinite(mi)) and np.all(np.isfinite(ma))):
                            continue
                        if min_b is None:
                            min_b = mi
                            max_b = ma
                        else:
                            min_b = np.minimum(min_b, mi)
                            max_b = np.maximum(max_b, ma)
                    except Exception:
                        continue

                if min_b is not None and max_b is not None:
                    diag = float(np.linalg.norm(max_b - min_b))
                    if np.isfinite(diag) and diag > 0 and diag < float(target_diag):
                        vis_scale = float(target_diag) / float(diag)

        if vis_scale is not None and np.isfinite(vis_scale) and float(vis_scale) > 0 and abs(float(vis_scale) - 1.0) > 1e-9:
            if diag is not None and target_diag is not None:
                print(
                    "[INFO] Visualization scale applied: "
                    f"diag={float(diag):.6g} -> target_diag={float(target_diag):.6g} (x{float(vis_scale):.6g})"
                )
            else:
                print(f"[INFO] Visualization scale applied: x{float(vis_scale):.6g} (manual)")
            center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            for g in geoms:
                try:
                    g.scale(float(vis_scale), center)
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] Visualization scaling skipped: {e}")

    # Legacy two-panel renderer disabled.
    # Keep falling through to the four-module implementation below.

    geoms = []
    heat_pcds_to_save = []  # 用于 *_heat.ply 的点云列表
    scene_pcds_to_save = []  # 用于 *_scene.ply 的多视图点云列表
    scene_failred_pcds_to_save = []  # 用于 *_scene_failred.ply 的多视图点云列表
    if final_method is None:
        final_method = str(FINAL_METHOD).strip() or "tangent"
    analysis_report = {
        "final_method": str(final_method),
        "step_mm": float(step_mm),
        "pin_unit_scale": float(PIN_UNIT_SCALE),
        "baseline_verdict_mode": str(BASELINE_VERDICT_MODE),
        "decision_status_source": "formal_decision_verdict",
        "display_status_source": "tol_scores",
        "display_note": "P5-C promoted: main verdict uses formal_decision_verdict; heatmap remains tol_scores explanation.",
        "missing_normal": list(pairing_result.get("missing_normal", [])) if isinstance(pairing_result, dict) else [],
        "extra_defect": list(pairing_result.get("extra_defect", [])) if isinstance(pairing_result, dict) else [],
        "needles": [],
    }

    def _append_export_pcd(dst, pcd):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            return
        try:
            if len(pcd.points) <= 0:
                return
        except Exception:
            return
        dst.append(copy.deepcopy(pcd))

    def _append_heat_export_pcd(pcd):
        _append_export_pcd(heat_pcds_to_save, pcd)
        _append_export_pcd(scene_pcds_to_save, pcd)
        _append_export_pcd(scene_failred_pcds_to_save, pcd)

    def _append_scene_export_pcd(pcd):
        _append_export_pcd(scene_pcds_to_save, pcd)
        _append_export_pcd(scene_failred_pcds_to_save, pcd)

    def _append_scene_failred_export_pcd(pcd):
        _append_export_pcd(scene_failred_pcds_to_save, pcd)

    def _build_uniform_point_cloud(points, color, translation_vec):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) <= 0:
            return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([float(color[0]), float(color[1]), float(color[2])])
        pcd.translate(np.asarray(translation_vec, dtype=np.float64))
        return pcd

    def _build_heat_point_cloud(points, scores, translation_vec):
        pts = np.asarray(points, dtype=np.float64)
        vals = np.asarray(scores, dtype=np.float64).reshape(-1)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) <= 0 or len(vals) <= 0:
            return None
        n = min(len(pts), len(vals))
        pts = pts[:n]
        vals = vals[:n]
        colors = np.asarray([score_to_color(float(v)) for v in vals], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate(np.asarray(translation_vec, dtype=np.float64))
        return pcd

    # 分离四个区域（与 *_scene.ply 的三列语义对齐，再追加第4列方向锥）
    scene_gap_u = mm_to_unit_len(VIS_SCENE_COLUMN_GAP_MM)
    offset_centerline = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    offset_cloud = np.array([scene_gap_u, 0.0, 0.0], dtype=np.float64)
    offset_reg = np.array([-scene_gap_u, 0.0, 0.0], dtype=np.float64)
    offset_direction_cone = offset_cloud + np.array([mm_to_unit_len(VIS_DIRECTION_MODULE_GAP_MM), 0.0, 0.0], dtype=np.float64)
    offset_direction_heat = offset_direction_cone + np.array([mm_to_unit_len(VIS_DIRECTION_HEAT_GAP_MM), 0.0, 0.0], dtype=np.float64)
    offset_verdict = offset_direction_heat + np.array([mm_to_unit_len(VIS_VERDICT_MODULE_GAP_MM), 0.0, 0.0], dtype=np.float64)
    view_offsets = [offset_reg, offset_centerline, offset_cloud, offset_direction_cone, offset_direction_heat, offset_verdict]

    # 1. 注册叠加列：底座叠加（参考=白，待测=绿）
    if base_ref is not None:
        base_ref_vis = copy.deepcopy(base_ref)
        base_ref_vis.paint_uniform_color([1.0, 1.0, 1.0])
        base_ref_vis.translate(offset_reg)
        geoms.append(base_ref_vis)
        _append_scene_export_pcd(base_ref_vis)
    if base_def is not None:
        base_def_vis = copy.deepcopy(base_def)
        base_def_vis.paint_uniform_color([0.0, 1.0, 0.0])
        base_def_vis.translate(offset_reg)
        geoms.append(base_def_vis)
        _append_scene_export_pcd(base_def_vis)

    # 2. 配对针检测
    pair_visual_records: List[Dict[str, Any]] = []
    for pair in pairing_result["pairs"]:
        idx_ref = int(pair.get("normal_id", -1))
        idx_def = int(pair.get("defect_id", -1))
        if idx_ref < 0 or idx_ref >= len(infos_ref) or idx_def < 0 or idx_def >= len(infos_def_aligned):
            continue
        info_ref = infos_ref[idx_ref]
        info_def = infos_def_aligned[idx_def]
        pair_record, pair_artifact = _analyze_tolerance_pair(
            info_ref,
            info_def,
            pair,
            pairing_result,
            final_method=final_method,
            reg_meta=reg_meta,
            step_mm=step_mm,
        )
        if pair_record is None or pair_artifact is None:
            continue
        analysis_report["needles"].append(pair_record)
        pair_visual_records.append(
            {
                "info_ref": info_ref,
                "info_def": info_def,
                "pair_record": pair_record,
                "pair_artifact": pair_artifact,
            }
        )

    analysis_report["formal_round2_runtime"] = {"enabled": False, "reason": "Final decisions are produced by TGAD LightGBM."}

    for visual_item in pair_visual_records:
        info_ref = visual_item["info_ref"]
        info_def = visual_item["info_def"]
        pair_record = visual_item["pair_record"]
        pair_artifact = visual_item["pair_artifact"]
        pts_centerline_ref_final = np.asarray(pair_artifact.get("pts_centerline_ref_final", np.empty((0, 3))), dtype=np.float64)
        pts_centerline_def_final = np.asarray(pair_artifact.get("pts_centerline_def_final", np.empty((0, 3))), dtype=np.float64)
        scores_final = np.asarray(pair_artifact.get("scores_final", np.empty((0,), dtype=np.float64)), dtype=np.float64)
        final_used_method = str(pair_artifact.get("final_used_method", ""))
        tol_metrics_map = dict(pair_artifact.get("tol_metrics", {}) or {})

        def add_registration_visuals(translation_vec):
            if "points" in info_ref:
                ref_pts = np.asarray(info_ref.get("points", np.empty((0, 3))), dtype=np.float64)
                ref_pts = downsample_points_np(ref_pts, VIS_MAX_POINTS_PER_NEEDLE)
                if len(ref_pts) > 0:
                    pcd_ref = o3d.geometry.PointCloud()
                    pcd_ref.points = o3d.utility.Vector3dVector(ref_pts)
                    pcd_ref.paint_uniform_color([1.0, 1.0, 1.0])
                    pcd_ref.translate(translation_vec)
                    geoms.append(pcd_ref)
                    _append_scene_export_pcd(pcd_ref)
            if "points" in info_def:
                def_pts = np.asarray(info_def.get("points", np.empty((0, 3))), dtype=np.float64)
                def_pts = downsample_points_np(def_pts, VIS_MAX_POINTS_PER_NEEDLE)
                if len(def_pts) > 0:
                    pcd_def = o3d.geometry.PointCloud()
                    pcd_def.points = o3d.utility.Vector3dVector(def_pts)
                    pcd_def.paint_uniform_color([0.0, 1.0, 0.0])
                    pcd_def.translate(translation_vec)
                    geoms.append(pcd_def)
                    _append_scene_export_pcd(pcd_def)

        def add_centerline_band_visuals(score_arr, translation_vec, pts_def_vis, pts_ref_vis):
            band_proxy_pcd = None
            if (
                VIS_SHOW_TOLERANCE_BAND
                and final_used_method == "tolerance_band"
                and len(pts_ref_vis) >= 2
                and isinstance(tol_metrics_map, dict)
            ):
                tol_lat_vis = info_ref.get("tol_lateral_mm", None)
                if tol_lat_vis is not None and len(np.asarray(tol_lat_vis).reshape(-1)) >= 2:
                    try:
                        band_mesh = create_tolerance_band_mesh(
                            points=pts_ref_vis,
                            tol_lateral_mm=tol_lat_vis,
                            unit_scale=info_ref.get("tol_unit_scale", PIN_UNIT_SCALE),
                        )
                        if band_mesh is not None and band_mesh.has_vertices():
                            rgb = np.asarray(VIS_TOLERANCE_BAND_COLOR, dtype=np.float64).reshape(-1)
                            if rgb.size < 3:
                                rgb = np.array([0.0, 0.55, 0.95], dtype=np.float64)
                            rgb = np.clip(rgb[:3], 0.0, 1.0)
                            band_mesh.paint_uniform_color(rgb.tolist())
                            band_mesh.translate(translation_vec)
                            geoms.append(band_mesh)
                            band_proxy_points = min(12000, max(1200, int(len(np.asarray(pts_ref_vis, dtype=np.float64)) * 10)))
                            band_proxy_pcd = _sample_mesh_to_point_cloud(
                                band_mesh,
                                number_of_points=band_proxy_points,
                            )
                            if hasattr(o3d.visualization, "rendering") and hasattr(o3d.visualization.rendering, "MaterialRecord"):
                                try:
                                    mat = o3d.visualization.rendering.MaterialRecord()
                                    mat.shader = "defaultLitTransparency"
                                    mat.base_color = [float(rgb[0]), float(rgb[1]), float(rgb[2]), float(np.clip(VIS_TOLERANCE_BAND_ALPHA, 0.0, 1.0))]
                                    mat.has_alpha = True
                                    material_records[id(band_mesh)] = mat
                                except Exception:
                                    pass
                    except Exception:
                        pass
            if VIS_SHOW_DEFECT_CL:
                if VIS_FAST_MODE:
                    defect_ls = create_heatmap_lineset(pts_def_vis, score_arr)
                    if len(defect_ls.points) > 0:
                        defect_ls.translate(translation_vec)
                        geoms.append(defect_ls)
                else:
                    defect_tube = create_heatmap_tube(pts_def_vis, score_arr, TUBE_RADIUS_DEFECT)
                    if defect_tube.has_vertices():
                        defect_tube.translate(translation_vec)
                        geoms.append(defect_tube)
            if band_proxy_pcd is not None:
                _append_scene_export_pcd(band_proxy_pcd)
            def_curve_export = _build_heat_point_cloud(pts_def_vis, score_arr, translation_vec)
            if def_curve_export is not None:
                _append_scene_export_pcd(def_curve_export)

        def add_cloud_heat_visuals(score_arr, translation_vec, pts_def_vis):
            if "points" in info_def and VIS_SHOW_DEFECT_PC:
                cloud_points_np = np.asarray(info_def.get("points", np.empty((0, 3))), dtype=np.float64)
                if len(cloud_points_np) > 0:
                    try:
                        cloud_points_np = downsample_points_np(cloud_points_np, VIS_MAX_POINTS_PER_NEEDLE)
                        pts_def_np = np.asarray(pts_def_vis, dtype=np.float64)
                        if pts_def_np.ndim != 2 or len(pts_def_np) == 0:
                            raise ValueError("empty centerline points for heatmap mapping")
                        tree_def = cKDTree(pts_def_np)
                        _, nearest_idx = tree_def.query(np.asarray(cloud_points_np, dtype=np.float64), k=1)
                        nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
                        cloud_colors = []
                        for idx in nearest_idx:
                            s = score_arr[idx] if idx < len(score_arr) else 0.0
                            cloud_colors.append(score_to_color(s))
                        pcd_heat = o3d.geometry.PointCloud()
                        pcd_heat.points = o3d.utility.Vector3dVector(cloud_points_np)
                        pcd_heat.colors = o3d.utility.Vector3dVector(np.array(cloud_colors))
                        pcd_heat.translate(translation_vec)
                        geoms.append(pcd_heat)
                        _append_heat_export_pcd(pcd_heat)
                    except Exception:
                        pass

        def add_direction_cone_visuals(cone_translation_vec, heat_translation_vec, pts_def_vis, pts_ref_vis, metrics):
            if not VIS_SHOW_DIRECTION_CONE_MODULE:
                return
            if not isinstance(metrics, dict):
                return

            direction_scores = np.asarray(metrics.get("tangent_score_arr", []), dtype=np.float64).reshape(-1)
            direction_angles = np.asarray(metrics.get("tangent_angle_deg_arr", []), dtype=np.float64).reshape(-1)
            half_angles = np.asarray(metrics.get("tangent_half_angle_deg_arr", []), dtype=np.float64).reshape(-1)
            decision_mask_arr = np.asarray(metrics.get("tangent_decision_mask_arr", []), dtype=bool).reshape(-1)
            axis_arr = np.asarray(metrics.get("tangent_axis_xyz_arr", []), dtype=np.float64).reshape(-1, 3)
            defect_tangent_arr = np.asarray(metrics.get("defect_tangent_xyz_arr", []), dtype=np.float64).reshape(-1, 3)

            n = min(
                len(pts_def_vis),
                len(pts_ref_vis),
                len(direction_scores),
                len(direction_angles),
                len(half_angles),
                len(axis_arr),
                len(defect_tangent_arr),
            )
            if n < 2:
                return

            pts_def_dir = np.asarray(pts_def_vis[:n], dtype=np.float64)
            pts_ref_dir = np.asarray(pts_ref_vis[:n], dtype=np.float64)
            score_dir = np.asarray(direction_scores[:n], dtype=np.float64)
            angle_dir = np.asarray(direction_angles[:n], dtype=np.float64)
            half_dir = np.asarray(half_angles[:n], dtype=np.float64)
            axis_dir = np.asarray(axis_arr[:n], dtype=np.float64)
            defect_dir = np.asarray(defect_tangent_arr[:n], dtype=np.float64)
            raw_out_mask = np.isfinite(angle_dir) & np.isfinite(half_dir) & (angle_dir > half_dir + 1e-9)
            if len(decision_mask_arr) >= n:
                out_mask = np.asarray(decision_mask_arr[:n], dtype=bool)
            else:
                out_mask = raw_out_mask

            if VIS_SHOW_DEFECT_CL:
                if VIS_FAST_MODE:
                    defect_ls = create_uniform_lineset(pts_def_dir, VIS_DIRECTION_DEFECT_BASELINE_COLOR)
                    if len(defect_ls.points) > 0:
                        defect_ls.translate(cone_translation_vec)
                        geoms.append(defect_ls)
                else:
                    defect_tube = create_uniform_tube(pts_def_dir, TUBE_RADIUS_DEFECT * 0.85, VIS_DIRECTION_DEFECT_BASELINE_COLOR)
                    if defect_tube.has_vertices():
                        defect_tube.translate(cone_translation_vec)
                        geoms.append(defect_tube)

            out_seg_ls = create_masked_segment_lineset(
                pts_def_dir,
                out_mask,
                VIS_DIRECTION_OUT_SEG_COLOR,
                require_both=False,
            )
            if len(out_seg_ls.points) > 0:
                out_seg_ls.translate(cone_translation_vec)
                geoms.append(out_seg_ls)

            cone_ls = create_direction_cone_lineset(
                pts_ref_dir,
                axis_dir,
                half_dir,
                cone_len=global_pin_width * VIS_DIRECTION_CONE_LENGTH_SCALE,
                stride=VIS_DIRECTION_CONE_STRIDE,
                skip_end_points=VIS_DIRECTION_CONE_SKIP_END_POINTS,
                ring_segments=VIS_DIRECTION_CONE_RING_SEGMENTS,
                out_mask=out_mask,
                color_ok=VIS_DIRECTION_CONE_COLOR,
                color_fail=VIS_DIRECTION_CONE_FAIL_COLOR,
            )
            if len(cone_ls.points) > 0:
                cone_ls.translate(cone_translation_vec)
                geoms.append(cone_ls)

            arrow_ls = create_direction_arrow_lineset(
                pts_ref_dir,
                defect_dir,
                arrow_len=global_pin_width * VIS_DIRECTION_CONE_LENGTH_SCALE,
                stride=VIS_DIRECTION_CONE_STRIDE,
                skip_end_points=VIS_DIRECTION_CONE_SKIP_END_POINTS,
                out_mask=out_mask,
                color_ok=VIS_DIRECTION_ARROW_OK_COLOR,
                color_fail=VIS_DIRECTION_ARROW_FAIL_COLOR,
            )
            if len(arrow_ls.points) > 0:
                arrow_ls.translate(cone_translation_vec)
                geoms.append(arrow_ls)

            ref_dir_export = _build_uniform_point_cloud(pts_ref_dir, [1.0, 1.0, 1.0], cone_translation_vec)
            if ref_dir_export is not None:
                _append_scene_export_pcd(ref_dir_export)
            def_dir_export = _build_uniform_point_cloud(pts_def_dir, VIS_DIRECTION_DEFECT_BASELINE_COLOR, cone_translation_vec)
            if def_dir_export is not None:
                _append_scene_export_pcd(def_dir_export)

            line_export_step_u = mm_to_unit_len(0.35)
            for ls, color in [
                (out_seg_ls, VIS_DIRECTION_OUT_SEG_COLOR),
                (cone_ls, VIS_DIRECTION_CONE_COLOR),
                (arrow_ls, VIS_DIRECTION_ARROW_OK_COLOR),
            ]:
                pcd_proxy = _sample_lineset_to_point_cloud(ls, step_u=line_export_step_u, default_color=color)
                if pcd_proxy is not None:
                    _append_scene_export_pcd(pcd_proxy)

            ref_dir_heat = _build_uniform_point_cloud(pts_ref_dir, [0.95, 0.95, 0.95], heat_translation_vec)
            if ref_dir_heat is not None:
                _append_scene_export_pcd(ref_dir_heat)
                geoms.append(copy.deepcopy(ref_dir_heat))

            dir_centerline_heat = _build_heat_point_cloud(pts_def_dir, score_dir, heat_translation_vec)
            if dir_centerline_heat is not None:
                _append_scene_export_pcd(dir_centerline_heat)
                geoms.append(copy.deepcopy(dir_centerline_heat))

            if "points" in info_def:
                cloud_points_np = np.asarray(info_def.get("points", np.empty((0, 3))), dtype=np.float64)
                if len(cloud_points_np) > 0:
                    try:
                        cloud_points_np = downsample_points_np(cloud_points_np, VIS_MAX_POINTS_PER_NEEDLE)
                        tree_def = cKDTree(pts_def_dir)
                        _, nearest_idx = tree_def.query(np.asarray(cloud_points_np, dtype=np.float64), k=1)
                        nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
                        dir_cloud_colors = []
                        for idx_local in nearest_idx:
                            s = score_dir[idx_local] if idx_local < len(score_dir) else 0.0
                            dir_cloud_colors.append(score_to_color(float(s)))
                        pcd_dir_heat = o3d.geometry.PointCloud()
                        pcd_dir_heat.points = o3d.utility.Vector3dVector(cloud_points_np)
                        pcd_dir_heat.colors = o3d.utility.Vector3dVector(np.asarray(dir_cloud_colors, dtype=np.float64))
                        pcd_dir_heat.translate(heat_translation_vec)
                        geoms.append(pcd_dir_heat)
                        _append_scene_export_pcd(pcd_dir_heat)
                    except Exception:
                        pass

        def add_verdict_view_visuals(translation_vec, verdict_text):
            if "points" not in info_def:
                return
            verdict_pts = downsample_points_np(
                info_def.get("points", np.empty((0, 3))),
                VIS_MAX_POINTS_PER_NEEDLE,
            )
            if len(verdict_pts) <= 0:
                return
            verdict_color = (
                VIS_VERDICT_FAIL_COLOR
                if str(verdict_text).strip() == "fail"
                else VIS_VERDICT_PASS_COLOR
            )
            pcd_verdict = _build_uniform_point_cloud(verdict_pts, verdict_color, translation_vec)
            if pcd_verdict is not None:
                _append_scene_failred_export_pcd(pcd_verdict)

        # === C. 分别添加四组视图（注册 / 中心线带 / 点云热力 / 方向锥） ===
        add_registration_visuals(offset_reg)
        add_centerline_band_visuals(scores_final, offset_centerline, pts_centerline_def_final, pts_centerline_ref_final)
        add_cloud_heat_visuals(scores_final, offset_cloud, pts_centerline_def_final)
        add_direction_cone_visuals(offset_direction_cone, offset_direction_heat, pts_centerline_def_final, pts_centerline_ref_final, tol_metrics_map)
        add_verdict_view_visuals(
            offset_verdict,
            pair_record.get("formal_decision_verdict", ""),
        )

    # 3. 绘制漏针
    yellow_color = [1.0, 1.0, 0.0]
    for idx in pairing_result["missing_normal"]:
        if "points" in infos_ref[idx]:
            pts = downsample_points_np(infos_ref[idx]["points"], VIS_MAX_POINTS_PER_NEEDLE)
            if len(pts) > 0:
                pcd_miss = o3d.geometry.PointCloud()
                pcd_miss.points = o3d.utility.Vector3dVector(pts)
                pcd_miss.paint_uniform_color(yellow_color)
                p_clone = copy.deepcopy(pcd_miss)
                p_clone.translate(offset_cloud)
                geoms.append(p_clone)
                _append_heat_export_pcd(p_clone)
                verdict_miss = _build_uniform_point_cloud(
                    pts,
                    VIS_VERDICT_MISSING_COLOR,
                    offset_verdict,
                )
                if verdict_miss is not None:
                    _append_scene_failred_export_pcd(verdict_miss)

    # 4. 绘制多针
    red_color = [1.0, 0.0, 0.0]
    for idx in pairing_result["extra_defect"]:
        if "points" in infos_def_aligned[idx]:
            pts = downsample_points_np(infos_def_aligned[idx]["points"], VIS_MAX_POINTS_PER_NEEDLE)
            if len(pts) > 0:
                pcd_ext = o3d.geometry.PointCloud()
                pcd_ext.points = o3d.utility.Vector3dVector(pts)
                pcd_ext.paint_uniform_color(red_color)
                p_clone = copy.deepcopy(pcd_ext)
                p_clone.translate(offset_cloud)
                geoms.append(p_clone)
                _append_heat_export_pcd(p_clone)
                verdict_extra = _build_uniform_point_cloud(
                    pts,
                    VIS_VERDICT_FAIL_COLOR,
                    offset_verdict,
                )
                if verdict_extra is not None:
                    _append_scene_failred_export_pcd(verdict_extra)

    _write_combined_point_cloud(heat_save_path, heat_pcds_to_save, label="heat")
    _write_combined_point_cloud(scene_save_path, scene_pcds_to_save, label="scene")
    _write_combined_point_cloud(scene_failred_save_path, scene_failred_pcds_to_save, label="scene_failred")

    # 坐标轴
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=global_pin_width * 5, origin=[0, 0, 0])
    geoms.append(axes)

    print(f"[INFO] Open3D draw_geometries: geoms={len(geoms)} fast_mode={VIS_FAST_MODE}")
    if _truthy_env("DAIMA_SKIP_OPEN3D_DRAW"):
        print("[INFO] Skip Open3D draw_geometries because DAIMA_SKIP_OPEN3D_DRAW=1")
    else:
        try:
            o3d.visualization.draw_geometries(geoms, window_name="Detection Result", width=1600, height=800)
        except Exception as e:
            print(f"[ERROR] Open3D visualization failed: {e}")
    return analysis_report


# ===================== [替换] 新的箭头生成工具 =====================

def create_tangent_arrows_from_curve(curve_points, step=5, arrow_len=2.0, color=[0, 1, 0]):
    """
    [新版] 基于拟合中心线计算切向量箭头。
    直接计算中心线的导数，不受切片形状影响，方向极其稳定。
    """
    if len(curve_points) < 2:
        return []

    # 1. 计算切向量 (Gradient)
    tangents = np.gradient(curve_points, axis=0)

    # 2. 归一化
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-8] = 1.0
    tangents_norm = tangents / norms[:, np.newaxis]

    arrows = []

    # 3. 生成箭头 (step控制密度)
    for i in range(0, len(curve_points), step):
        p = curve_points[i]
        vec = tangents_norm[i]

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.3 * arrow_len,
            cone_radius=0.08 * arrow_len,
            cylinder_height=0.7 * arrow_len,
            cylinder_radius=0.03 * arrow_len
        )
        arrow.paint_uniform_color(color)

        # 计算旋转矩阵: 从 Open3D 默认的 (0,0,1) 旋转到 vec
        default_z = np.array([0, 0, 1])
        rot_axis = np.cross(default_z, vec)
        rot_sin = np.linalg.norm(rot_axis)
        rot_cos = np.dot(default_z, vec)

        if rot_sin < 1e-6:
            # 平行或反向
            if rot_cos < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
                arrow.rotate(R, center=[0, 0, 0])
        else:
            rot_axis /= rot_sin
            angle = np.arccos(np.clip(rot_cos, -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
            arrow.rotate(R, center=[0, 0, 0])

        # 平移到曲线上的点
        arrow.translate(p)
        arrows.append(arrow)

    return arrows


def _truthy_env(name: str) -> bool:
    val = str(os.getenv(name, "")).strip().lower()
    return val in {"1", "true", "yes", "on", "y"}


def _infer_sample_id_from_path(path_like, fallback: str = "sample") -> str:
    try:
        stem = Path(str(path_like)).stem
    except Exception:
        return str(fallback)
    prefix = str(stem).split("_", 1)[0].strip()
    return prefix or str(fallback)


def _resolve_interactive_case_paths() -> dict[str, object]:
    default_ref_base = _REPO_ROOT / "point_cloud" / "fenge" / "split_out" / "20260124_183935_626294_27528" / "1_base.ply"
    default_ref_pins = _REPO_ROOT / "point_cloud" / "fenge" / "split_out" / "20260124_183935_626294_27528" / "1_pins.ply"
    default_sample_id = str(os.getenv("DAIMA_DEFECT_SAMPLE_ID", "6") or "6").strip() or "6"
    default_base_def = _REPO_ROOT / "point_cloud" / "test" / "split" / f"{default_sample_id}_base.ply"
    default_pins_def = _REPO_ROOT / "point_cloud" / "test" / "split" / f"{default_sample_id}_pin.ply"

    base_ref_path = Path(str(os.getenv("DAIMA_REF_BASE_PATH", "")).strip() or str(default_ref_base))
    pins_ref_path = Path(str(os.getenv("DAIMA_REF_PINS_PATH", "")).strip() or str(default_ref_pins))
    base_def_path = Path(str(os.getenv("DAIMA_DEFECT_BASE_PATH", "")).strip() or str(default_base_def))
    pins_def_path = Path(str(os.getenv("DAIMA_DEFECT_PINS_PATH", "")).strip() or str(default_pins_def))

    sample_id = str(os.getenv("DAIMA_DEFECT_SAMPLE_ID", "")).strip()
    if not sample_id:
        sample_id = _infer_sample_id_from_path(base_def_path, fallback=default_sample_id)
    sample_id = sample_id or "sample"
    return {
        "sample_id": str(sample_id),
        "base_ref_path": base_ref_path,
        "pins_ref_path": pins_ref_path,
        "base_def_path": base_def_path,
        "pins_def_path": pins_def_path,
    }


def _resolve_visualization_export_paths(sample_id: str) -> dict[str, Path]:
    export_dir_env = str(os.getenv("DAIMA_VIS_EXPORT_DIR", "")).strip()
    if export_dir_env:
        export_dir = Path(export_dir_env)
    elif _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None) is not None:
        export_dir = Path(_DAIMA_CTX.run_dir) / "vis_ply"
    else:
        export_dir = _REPO_ROOT / "debug_out" / "manual_visualization" / "vis_ply"
    export_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = str(sample_id).strip() or "sample"
    return {
        "export_dir": export_dir,
        "heat_path": export_dir / f"{sample_tag}_heat.ply",
        "scene_path": export_dir / f"{sample_tag}_scene.ply",
        "scene_failred_path": export_dir / f"{sample_tag}_scene_failred.ply",
    }


def _write_combined_point_cloud(save_path, pcds, *, label: str) -> None:
    if not save_path:
        return
    valid_pcds = []
    for pcd in pcds or []:
        if not isinstance(pcd, o3d.geometry.PointCloud):
            continue
        try:
            if len(pcd.points) <= 0:
                continue
        except Exception:
            continue
        valid_pcds.append(copy.deepcopy(pcd))
    if not valid_pcds:
        print(f"[WARN] Skip {label} export: no point clouds collected")
        return
    save_path = Path(str(save_path))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving {label} to {save_path} ...")
    combined_pcd = valid_pcds[0]
    for pcd in valid_pcds[1:]:
        combined_pcd += pcd
    ok = o3d.io.write_point_cloud(str(save_path), combined_pcd, write_ascii=False, compressed=True)
    try:
        n_points = len(combined_pcd.points)
    except Exception:
        n_points = -1
    if ok:
        print(f"[INFO] Saved {label}: path={save_path} points={n_points}")
    else:
        print(f"[WARN] Failed to save {label}: path={save_path}")


def _sample_lineset_to_point_cloud(lineset, *, step_u: float, default_color=None):
    if not isinstance(lineset, o3d.geometry.LineSet):
        return None
    try:
        pts = np.asarray(lineset.points, dtype=np.float64).reshape(-1, 3)
        lines = np.asarray(lineset.lines, dtype=np.int32).reshape(-1, 2)
    except Exception:
        return None
    if len(pts) <= 0 or len(lines) <= 0:
        return None
    try:
        colors = np.asarray(lineset.colors, dtype=np.float64).reshape(-1, 3)
    except Exception:
        colors = np.empty((0, 3), dtype=np.float64)
    if default_color is None:
        default_color = [1.0, 1.0, 1.0]
    default_color = np.asarray(default_color, dtype=np.float64).reshape(3)
    sample_points = []
    sample_colors = []
    step_u = float(step_u)
    if (not np.isfinite(step_u)) or step_u <= 1e-12:
        step_u = mm_to_unit_len(0.35)
    for idx, seg in enumerate(lines):
        try:
            p0 = pts[int(seg[0])]
            p1 = pts[int(seg[1])]
        except Exception:
            continue
        length = float(np.linalg.norm(p1 - p0))
        n_samples = max(2, min(12, int(np.ceil(length / step_u)) + 1))
        ts = np.linspace(0.0, 1.0, n_samples)
        color = colors[idx] if idx < len(colors) else default_color
        for t in ts:
            sample_points.append((1.0 - t) * p0 + t * p1)
            sample_colors.append(color)
    if not sample_points:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(sample_points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(sample_colors, dtype=np.float64))
    return pcd


def _sample_mesh_to_point_cloud(mesh, *, number_of_points: int, translation_vec=None):
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        return None
    try:
        if not mesh.has_vertices() or not mesh.has_triangles():
            return None
    except Exception:
        return None
    n_points = max(256, int(number_of_points))
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    except Exception:
        return None
    try:
        if len(pcd.points) <= 0:
            return None
    except Exception:
        return None
    if translation_vec is not None:
        pcd.translate(np.asarray(translation_vec, dtype=np.float64))
    return pcd


def main():
    case_paths = _resolve_interactive_case_paths()
    sample_id = str(case_paths["sample_id"])
    base_ref_path = str(case_paths["base_ref_path"])
    base_def_path = str(case_paths["base_def_path"])
    pins_ref_path = str(case_paths["pins_ref_path"])
    pins_def_path = str(case_paths["pins_def_path"])
    export_paths = _resolve_visualization_export_paths(sample_id)
    heat_output_filename = str(export_paths["heat_path"])
    scene_output_filename = str(export_paths["scene_path"])
    scene_failred_output_filename = str(export_paths["scene_failred_path"])
    export_dir = str(export_paths["export_dir"])

    final_method = str(FINAL_METHOD).strip() or "tangent"
    tol_json = None
    tol_base_path = None
    tol_pins_path = None
    selected_tolerance_json_path = Path(TOLERANCE_JSON_PATH)
    tol_unit_scale = None
    r_match_u = None
    centerline_cfg = _centerline_extract_config()

    infos_ref = None  # build later (tangent: from pins_ref_path; tolerance: from tolerance model)

    if final_method == "tolerance_band":
        selected_tolerance_json_path, tol_json = _resolve_tangent_capable_tolerance_json(Path(TOLERANCE_JSON_PATH))
        tol_base_path, tol_pins_path = resolve_tolerance_reference_paths(tol_json or {})
        if tol_json is None or tol_base_path is None or tol_pins_path is None:
            print("[WARN] tolerance_band requested but tolerance model / reference files not available; fallback to tangent.")
            final_method = "tangent"
            tol_json = None
        else:
            if selected_tolerance_json_path.resolve() != Path(TOLERANCE_JSON_PATH).resolve():
                print(f"[INFO] Auto-preferred tangent-capable tolerance model: {selected_tolerance_json_path}")
            base_ref_path = str(tol_base_path)
            pins_ref_path = str(tol_pins_path)

            tol_unit_scale = _safe_float(tol_json.get("unit_scale", PIN_UNIT_SCALE), default=PIN_UNIT_SCALE)
            match_radius_mm = _safe_float(tol_json.get("match_radius_mm", 0.0), default=0.0)
            if np.isfinite(match_radius_mm) and match_radius_mm > 0 and np.isfinite(tol_unit_scale) and tol_unit_scale > 1e-12:
                r_match_u = float(match_radius_mm) * float(tol_unit_scale)

            # Optional consistency warning.
            try:
                rel = abs(float(tol_unit_scale) - float(PIN_UNIT_SCALE)) / max(abs(float(PIN_UNIT_SCALE)), 1e-12)
                if np.isfinite(rel) and rel > 1e-3:
                    print(
                        f"[WARN] unit_scale mismatch: tolerance_json={float(tol_unit_scale):.6g} "
                        f"vs PIN_UNIT_SCALE={float(PIN_UNIT_SCALE):.6g}"
                    )
            except Exception:
                pass

            print("[INFO] Extracting tolerance reference pins for visualization anchors...")
            ref_pins_infos = extract_centerlines_from_pcd(str(tol_pins_path), config=centerline_cfg)
            ref_by_sid = {}
            for info in ref_pins_infos:
                try:
                    sid = int(info.get("stable_id", -1))
                except Exception:
                    continue
                if sid < 0:
                    continue
                if sid not in ref_by_sid:
                    ref_by_sid[sid] = info

            infos_ref = []
            for needle in (tol_json.get("needles", []) if isinstance(tol_json, dict) else []):
                try:
                    sid = int(needle.get("stable_id", -1))
                except Exception:
                    continue
                if sid < 0:
                    continue
                nominal = np.asarray(needle.get("nominal_curve_u", []), dtype=np.float64)
                if nominal.ndim != 2 or nominal.shape[1] != 3 or len(nominal) < 2:
                    continue

                info_ref = {
                    "stable_id": int(sid),
                    "curve_global_final": nominal,
                    "center_point": np.mean(nominal, axis=0),
                    "base_end_idx_local": 0,
                    # Tolerance-band model params (consumed by visualize_with_matching_and_heatmap()).
                    "tol_model_status": str(needle.get("status", "")),
                    "tol_tangent_model_status": str(needle.get("tol_tangent_model_status", needle.get("status", ""))),
                    "tol_unit_scale": float(tol_unit_scale) if tol_unit_scale is not None else float(PIN_UNIT_SCALE),
                    "tol_lateral_mm": needle.get("tol_lateral_mm", []),
                    "tol_tip_short_mm": needle.get("tol_tip_short_mm", None),
                    "tol_len_short_mm": needle.get("tol_len_short_mm", None),
                    "tol_tangent_axis_xyz": needle.get("tol_tangent_axis_xyz", []),
                    "tol_tangent_half_angle_deg": needle.get("tol_tangent_half_angle_deg", []),
                    "tol_tangent_window_mm": needle.get("tol_tangent_window_mm", None),
                    "tol_stats": needle.get("stats", {}),
                }

                if sid in ref_by_sid:
                    src = ref_by_sid[sid]
                    if "points" in src:
                        info_ref["points"] = src["points"]
                    if "pin_width" in src:
                        info_ref["pin_width"] = src.get("pin_width", 1.0)
                infos_ref.append(info_ref)

            infos_ref.sort(key=lambda d: int(d.get("stable_id", 9999)))
            print(
                f"[INFO] tolerance_band enabled: ref_id={tol_json.get('reference_sample_id','')} "
                f"needles={len(infos_ref)} match_radius_mm={match_radius_mm:.3f}"
            )

    if _DAIMA_CTX is not None:
        active_ab_protocol = resolve_compare_readout_protocol_from_env()
        frozen_ab_protocol = get_frozen_compare_readout_protocol_v1()
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "base_ref_path": base_ref_path,
                "base_def_path": base_def_path,
                "pins_ref_path": pins_ref_path,
                "pins_def_path": pins_def_path,
                "centerline_tolerance_json": str(selected_tolerance_json_path) if final_method == "tolerance_band" else "",
            },
            outputs={
                "vis_export_dir": export_dir,
                "heat_output_filename": heat_output_filename,
                "scene_output_filename": scene_output_filename,
                "scene_failred_output_filename": scene_failred_output_filename,
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "VIS_FAST_MODE": VIS_FAST_MODE,
                "VIS_HEATMAP_PC_ONLY_FINAL": VIS_HEATMAP_PC_ONLY_FINAL,
                "VIS_MAX_POINTS_PER_NEEDLE": VIS_MAX_POINTS_PER_NEEDLE,
                "VIS_SHOW_TOLERANCE_BAND": bool(VIS_SHOW_TOLERANCE_BAND),
                "VIS_TOLERANCE_BAND_ALPHA": float(VIS_TOLERANCE_BAND_ALPHA),
                "VIS_TOLERANCE_BAND_COLOR": [
                    float(VIS_TOLERANCE_BAND_COLOR[0]),
                    float(VIS_TOLERANCE_BAND_COLOR[1]),
                    float(VIS_TOLERANCE_BAND_COLOR[2]),
                ] if isinstance(VIS_TOLERANCE_BAND_COLOR, (list, tuple)) and len(VIS_TOLERANCE_BAND_COLOR) >= 3 else [0.2, 0.8, 1.0],
                "VIS_TOLERANCE_BAND_MIN_RADIUS_MM": float(VIS_TOLERANCE_BAND_MIN_RADIUS_MM),
                "VIS_SHOW_DIRECTION_CONE_MODULE": bool(VIS_SHOW_DIRECTION_CONE_MODULE),
                "VIS_SCENE_COLUMN_GAP_MM": float(VIS_SCENE_COLUMN_GAP_MM),
                "VIS_DIRECTION_MODULE_GAP_MM": float(VIS_DIRECTION_MODULE_GAP_MM),
                "VIS_DIRECTION_CONE_STRIDE": int(VIS_DIRECTION_CONE_STRIDE),
                "VIS_DIRECTION_CONE_SKIP_END_POINTS": int(VIS_DIRECTION_CONE_SKIP_END_POINTS),
                "VIS_DIRECTION_CONE_LENGTH_SCALE": float(VIS_DIRECTION_CONE_LENGTH_SCALE),
                "VIS_DIRECTION_CONE_RING_SEGMENTS": int(VIS_DIRECTION_CONE_RING_SEGMENTS),
                "VIS_DIRECTION_HEAT_GAP_MM": float(VIS_DIRECTION_HEAT_GAP_MM),
                "VIS_VERDICT_MODULE_GAP_MM": float(VIS_VERDICT_MODULE_GAP_MM),
                "step_mm": 0.2,
                "FINAL_METHOD": str(final_method),
                "TOLERANCE_JSON_PATH": str(selected_tolerance_json_path) if final_method == "tolerance_band" else "",
                "DETECTION_DISABLE_SHORT_TOLERANCE": bool(DETECTION_DISABLE_SHORT_TOLERANCE),
                "DETECTION_DISABLE_RESCAN": bool(DETECTION_DISABLE_RESCAN),
                "DAIMA_COMPARE_MODE": str(active_ab_protocol["compare_mode"]),
                "DAIMA_READOUT_MODE": str(active_ab_protocol["readout_mode"]),
                "DAIMA_SLACK_RADIUS_STEPS": str(active_ab_protocol["slack_radius_steps"]),
                "DAIMA_READOUT_TOPK_FRAC": str(active_ab_protocol["topk_frac"]),
                "DAIMA_READOUT_TOPK_MIN_POINTS": str(active_ab_protocol["topk_min_points"]),
                "frozen_compare_readout_protocol_text": format_compare_readout_protocol(frozen_ab_protocol),
                "active_compare_readout_protocol_text": format_compare_readout_protocol(active_ab_protocol),
                "active_compare_readout_matches_frozen": bool(compare_readout_protocol_matches_frozen(active_ab_protocol)),
                "DAIMA_SKIP_OPEN3D_DRAW": bool(_truthy_env("DAIMA_SKIP_OPEN3D_DRAW")),
                "sample_id": str(sample_id),
                "vis_export_dir": str(export_dir),
                "heat_output_filename": str(heat_output_filename),
                "scene_output_filename": str(scene_output_filename),
                "scene_failred_output_filename": str(scene_failred_output_filename),
                "r_match_u": float(r_match_u) if r_match_u is not None else None,
                "registration": {
                    "REG_TARGET_GRID_POINTS": int(REG_TARGET_GRID_POINTS),
                    "REG_MIN_VOXEL_MM": float(REG_MIN_VOXEL_MM),
                    "REG_MAX_VOXEL_MM": float(REG_MAX_VOXEL_MM),
                    "REG_RANSAC_MAX_ITER": int(REG_RANSAC_MAX_ITER),
                    "REG_RANSAC_CONFIDENCE": float(REG_RANSAC_CONFIDENCE),
                    "REG_RANSAC_DIST_MULT": float(REG_RANSAC_DIST_MULT),
                    "REG_ICP_DIST_MULT": float(REG_ICP_DIST_MULT),
                },
                "centerline_cfg": {
                    "denoise_enable": bool(CENTERLINE_CFG_DENOISE_ENABLE),
                    "endpoint_centering_iterations": int(CENTERLINE_CFG_ENDPOINT_CENTERING_ITERS),
                    "force_final_curve_stage13": bool(CENTERLINE_CFG_FORCE_FINAL_STAGE13),
                },
            }
        )

    # 1. 计算板级变换
    T, reg_meta, base_ref, base_def = compute_board_transform(
        base_ref_path,
        base_def_path,
        pins_ref_path,
        pins_def_path,
        return_clean_clouds=True,
    )
    base_def_aligned = copy.deepcopy(base_def)
    base_def_aligned.transform(T)

    # 2. 提取中心线 (保留原始点云用于扭转检测)
    print("[INFO] Extracting centerlines...")
    if infos_ref is None:
        infos_ref = extract_centerlines_from_pcd(pins_ref_path, config=centerline_cfg)
    infos_def = extract_centerlines_from_pcd(pins_def_path, config=centerline_cfg)

    # 计算全局参考针宽
    ws = [i.get("pin_width", 1.0) for i in infos_ref if isinstance(i.get("pin_width", None), (int, float))]
    if not ws:
        ws = [i.get("pin_width", 1.0) for i in infos_def if isinstance(i.get("pin_width", None), (int, float))]
    global_w = float(np.median(ws)) if ws else 1.0
    print(f"[INFO] Global pin width: {global_w:.3f}")

    # 3. 变换待测针信息
    infos_def_aligned = transform_needle_infos(infos_def, T)

    # 4. 配对
    print("[INFO] Matching needles...")
    pairing = match_needles_by_position(infos_ref, infos_def_aligned, r_match=r_match_u)

    # 5. 可视化与检测
    report = visualize_with_matching_and_heatmap(
        base_ref, base_def_aligned,
        infos_ref, infos_def_aligned,
        pairing, step_mm=0.2, global_pin_width=global_w,
        heat_save_path=heat_output_filename,
        scene_save_path=scene_output_filename,
        scene_failred_save_path=scene_failred_output_filename,
        final_method=final_method,
        reg_meta=reg_meta,
    )

    # Persist a machine-readable report under debug_out/runs/<run_id>/ for later paper/eval work.
    if isinstance(report, dict):
        report.setdefault("requested_final_method", str(FINAL_METHOD))
        report.setdefault("final_method_used", str(final_method))
        report.setdefault(
            "paths",
            {
                "base_ref_path": str(base_ref_path),
                "base_def_path": str(base_def_path),
                "pins_ref_path": str(pins_ref_path),
                "pins_def_path": str(pins_def_path),
                "sample_id": str(sample_id),
                "vis_export_dir": str(export_dir),
                "heat_output_filename": str(heat_output_filename),
                "scene_output_filename": str(scene_output_filename),
                "scene_failred_output_filename": str(scene_failred_output_filename),
                "output_filename": str(scene_output_filename),
            },
        )
        report["registration"] = reg_meta if isinstance(reg_meta, dict) else {}
        if tol_json is not None:
            report["tolerance"] = {
                "tolerance_json_path": str(selected_tolerance_json_path),
                "reference_sample_id": str(tol_json.get("reference_sample_id", "")),
                "unit_scale": _safe_float(tol_json.get("unit_scale", float("nan"))),
                "match_radius_mm": _safe_float(tol_json.get("match_radius_mm", float("nan"))),
                "ref_base_ply": str(tol_base_path) if tol_base_path is not None else "",
                "ref_pins_ply": str(tol_pins_path) if tol_pins_path is not None else "",
            }

        if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None) is not None:
            try:
                out_p = Path(_DAIMA_CTX.run_dir) / "detection_report.json"
                out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[INFO] Wrote detection_report: {out_p}")
            except Exception as e:
                print(f"[WARN] Failed to write detection_report.json: {e}")


if __name__ == "__main__":
    main()
