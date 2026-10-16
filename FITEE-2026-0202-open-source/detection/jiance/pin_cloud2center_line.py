#!/usr/bin/env python
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

        from daima_runtime.runlog import setup_run

        _DAIMA_CTX = setup_run(__file__)
    except Exception:
        _DAIMA_CTX = None

import open3d as o3d
import numpy as np
import heapq
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter


# ==========================================
# 全局参数配置
# ==========================================

INPUT_PCD_PATH = 'private_inputs/6_pin.ply'

# ------------------------------------------
# GT evaluation (optional, only runs when executing this file directly)
# ------------------------------------------
# Set this to the output folder of `generate_centerline_pseudo_gt.py`.
# The folder must contain `centerline_gt.json`.
EVAL_GT_DIR = r""  # e.g. r"<private-path>"
# Output subdir under `debug_out/` (relative path). Results will be stored under:
#   debug_out/<EVAL_OUT_DIR>/<pcd_stem>/<run_id>/
EVAL_OUT_DIR = r"gt_eval"
# If True and EVAL_GT_DIR is set, only run GT evaluation and exit (no visualization).
EVAL_ONLY = True
# Resampling step for tangent/endpoint comparison (millimeters).
EVAL_STEP_MM = 0.2
# Fixed sample count for tangent comparison (arc-length parametrized).
EVAL_TANGENT_SAMPLES = 240
# Print top-K worst cases to console.
EVAL_TOPK = 10

# ------------------------------------------
# Denoise switch (important for single-needle eval)
# ------------------------------------------
# NOTE:
# - The denoise pipeline (pre-DBSCAN ROR/SOR) was introduced mainly for multi-needle NeRF clouds
#   to break "bridging" noise that merges different needles into one DBSCAN cluster.
# - For single-needle clouds, aggressive denoise can delete real surface points and cause the
#   needle to break into multiple components (centerline "segmentation").
# - Therefore, default is OFF for this single-needle GT-eval workflow; enable it explicitly
#   when you evaluate multi-needle clouds.
DEFAULT_DENOISE_ENABLE = False

# Endpoint centering (slab refinement): iterative centering of the two endpoints before the main
# refinement loop. In single-needle clouds, too many endpoint-centering iterations can amplify
# local asymmetry/noise and worsen tip/base behavior. Use a safer default for direct-run GT-eval;
# multi-needle workflows can override via CLI.
DEFAULT_ENDPOINT_CENTERING_ITERS = 2

# PyCharm direct Run defaults (no CLI args):
# - keep visualization enabled
# - enable debug dump by default so artifacts are reproducible under debug_out/
DEFAULT_VISUALIZE = True
DEFAULT_DEBUG_DUMP = True

# ------------------------------------------
# 单位 / 尺度处理（重要）
# 本文件中所有“距离类”参数（eps/radius/threshold 等）必须与点云坐标单位一致。
#
# 若你的点云做过统一缩放（例如 mm -> m，或任意比例缩放），这里也要做同样缩放，
# 否则 DBSCAN 聚类、半径可视化等会严重失真。
#
# 点云坐标单位：m
# 所有“以 mm 为单位”的阈值/步长，都会通过 UNIT_SCALE 转到点云坐标单位：
#   value_u = value_mm * UNIT_SCALE
# 其中 1 mm = 0.001 m，因此 UNIT_SCALE 应为 1/1000。
# ------------------------------------------
UNIT_SCALE = 1.0 / 1000.0

# NeRF 点云噪声量级（用户给定，单位：mm）
DEFAULT_NERF_NOISE_MM = 0.3

# 1.11.py 的默认采样步长（用户给定，单位：mm）
DEFAULT_CENTERLINE_STEP_MM = 0.2

# DBSCAN 聚类参数
DBSCAN_EPS_ORIG = 1.0
DBSCAN_EPS = DBSCAN_EPS_ORIG * UNIT_SCALE
DBSCAN_MIN_POINTS = 20

# 截面切片个数控制
TARGET_POINTS_PER_SLICE = 150
MIN_SLICES = 20
MAX_SLICES = 80

# 可视化管道半径
RAW_TUBE_RADIUS_ORIG = 0.02
FINAL_TUBE_RADIUS_ORIG = 0.1
RAW_TUBE_RADIUS = RAW_TUBE_RADIUS_ORIG * UNIT_SCALE
FINAL_TUBE_RADIUS = FINAL_TUBE_RADIUS_ORIG * UNIT_SCALE


# ==========================================
# 新算法配置（graph + slab）
# ==========================================


def mm_to_unit(mm: float, unit_scale: float = UNIT_SCALE) -> float:
    return float(mm) * float(unit_scale)


@dataclass(frozen=True)
class CenterlineConfig:
    # 统一单位：点云坐标单位与本参数保持一致；mm 参数通过 unit_scale 换算
    unit_scale: float = UNIT_SCALE

    # DBSCAN（粗聚类）
    dbscan_eps_mm: float = 1.0  # eps_max_mm
    dbscan_eps_min_mm: float = 0.6  # eps_min_mm（自动 eps 的下限，避免点云很密时被打碎）
    dbscan_eps_factor: float = 5.0  # eps_auto = factor * median_nn（再用 dbscan_eps_mm 做上限）
    dbscan_min_points: int = 20

    # 1D 曲线重采样步长（沿弧长）
    centerline_step_mm: float = DEFAULT_CENTERLINE_STEP_MM

    # NeRF 噪声量级（用于默认 slice/slab 尺度）
    nerf_noise_mm: float = DEFAULT_NERF_NOISE_MM

    # ------------------------------------------
    # NeRF 点云去噪 / 离群点移除（默认开启）
    #
    # 背景：
    # - NeRF 导出的点云常见 “floaters / sparse outliers / 噪声点”
    # - 下游 graph+slab 拟合具备一定鲁棒性，但极端离群点仍可能：
    #   - 造成 kNN 图的错误连边/分段
    #   - 干扰端点（最远点）选择
    #
    # 默认策略：DBSCAN 后按簇(per_cluster)去噪，尽量不影响 stable_id 排序。
    # ------------------------------------------
    denoise_enable: bool = bool(DEFAULT_DENOISE_ENABLE)
    denoise_stage: str = "pre_dbscan"  # "pre_dbscan" | "per_cluster" | "both"

    # Radius outlier removal (ROR): 移除半径内邻居数不足的点
    denoise_ror_enable: bool = True
    denoise_ror_nb_points: int = 16
    denoise_ror_radius_mm: float = 0.0  # <=0: auto = factor * median_nn（并带一个小的 mm 级下限）
    denoise_ror_radius_factor: float = 3.0

    # Statistical outlier removal (SOR): 移除平均邻域距离显著偏大的点
    denoise_sor_enable: bool = True
    denoise_sor_nb_neighbors: int = 20
    denoise_sor_std_ratio: float = 3.5

    # Pre-DBSCAN stage override:
    # - For sparse "bridging" noise that merges needles into one DBSCAN cluster, enabling SOR (with a mild
    #   std_ratio) can help break the bridge.
    # - NOTE: too aggressive SOR (small std_ratio) can also fragment needle clusters.
    denoise_pre_dbscan_sor_enable: bool = True

    # 图参数化（簇内分段/排序）
    graph_k: int = 12
    graph_max_edge_factor: float = 2.8  # max_edge = factor * median_nn（抑制噪声捷径边）
    min_component_points: int = 600

    # ------------------------------------------
    # Shortest-path initializer (graph ordering)
    #
    # Motivation:
    # - Pure Euclidean shortest-path may take "shortcut" edges (cut corners) on high-curvature shapes,
    #   which biases tangents and slab refinement (a feedback loop).
    # - Keep default as euclid to preserve current behavior; enable dir_penalty explicitly for debugging.
    # ------------------------------------------
    init_edge_cost_mode: str = "euclid"  # "euclid" | "dir_penalty"
    init_dir_penalty_lambda: float = 0.5  # only used when mode="dir_penalty"

    # ------------------------------------------
    # Cluster fragmentation merge (important for multi-needle boards)
    #
    # Problem:
    # - After DBSCAN, we build a kNN graph and prune long edges, then split into connected-components.
    # - On NeRF point clouds, a *single* needle cluster can be cut into 2+ components (gaps / non-uniform density),
    #   which makes the final segment count > expected (often +1).
    #
    # Approach:
    # - Identify components that likely belong to the SAME needle by:
    #   - being close on the plane perpendicular to the estimated needle axis (uv distance),
    #   - having limited axial overlap on the estimated needle axis (t overlap ratio).
    # - Reconnect them by adding a small number of bridging edges to the pruned graph,
    #   then recompute connected-components (so downstream shortest-path ordering still works).
    # ------------------------------------------
    merge_fragments_enable: bool = True
    merge_fragments_uv_mm: float = 3.1          # uv centroid distance threshold; q047 repeatability data needs >3.006mm to merge one split fragment back to 20 needles
    merge_fragments_size_ratio_max: float = 0.10 # small-fragment merge regardless of axial overlap
    merge_fragments_t_overlap_max: float = 0.60 # max axial overlap ratio to still consider "fragments"
    merge_fragments_bridge_max_mm: float = 2.0  # max allowed 3D NN distance to add a bridging edge
    merge_fragments_max_iters: int = 3          # safety cap (usually 1-2 is enough)

    # slab 截面提取
    slab_quantile_low: float = 0.10
    slab_quantile_high: float = 0.90
    min_slab_points: int = 30

    # Single-sided slab gating (optional; default OFF via action="none")
    slab_coverage_bins: int = 12
    slab_min_coverage: float = 0.35  # fraction of occupied angle bins in (u,v)
    # Default: "damp" to reduce drift on NeRF single-sided cross-sections (see `研究报告/中心线.md`).
    slab_single_side_action: str = "damp"  # "none" | "skip" | "damp"
    slab_single_side_damp: float = 0.25  # only used when action="damp"

    # 迭代收敛
    refine_iterations: int = 8
    refine_damping: float = 0.7  # 每次迭代的更新阻尼（0~1）
    max_update_mm: float = 0.8  # 单次更新最大横向位移（mm）
    anchor_endpoints: bool = True  # 端点锚定，显著抑制中心线在截断端/针尖端“缩短”
    endpoint_centering_iterations: int = int(DEFAULT_ENDPOINT_CENTERING_ITERS)  # 与脚本默认口径保持一致
    # Endpoint centering timing:
    # - early: (legacy) pre-center endpoints before body refinement
    # - late : refine body first, then (optionally) re-center endpoints, then do a short post-refine
    # - off  : disable endpoint centering (still can anchor endpoints to suppress shrinking)
    #
    # Motivation:
    # - On NeRF point clouds, endpoint cross-sections are often single-sided (coverage low), so early
    #   endpoint centering can introduce a lateral bias that later refinement cannot correct when
    #   endpoints are anchored (feedback loop). "late" reduces this risk by using a stabilized body first.
    endpoint_centering_timing: str = "late"  # "early" | "late" | "off"
    endpoint_centering_min_coverage: float = 0.60  # only used when timing="late"
    endpoint_centering_low_coverage_action: str = "damp"  # "none" | "skip" | "damp"
    endpoint_centering_low_coverage_damp: float = 0.25  # only used when action="damp"
    post_endpoint_refine_iterations: int = 1  # only used when timing="late"
    endpoint_centering_rigid_mm: float = 0.6  # rigid range for endpoint re-centering blend (mm)
    endpoint_centering_blend_mm: float = 0.6  # blend range for endpoint re-centering blend (mm)

    # “端面(cap)存在性”判别：用于区分截断端 vs 自由端（截断端无端面点；自由端有端面点）
    endpoint_cap_depth_mm: float = 0.2  # 沿切向薄片厚度（mm）
    endpoint_cap_inner_radius_mm: float = 0.15  # 截面中心占比统计半径（mm）
    endpoint_cap_min_points: int = 8  # 绝对下限：NeRF 弯头段端部可能只有 ~8-16 点，20 会导致误判 ambiguous
    endpoint_cap_min_points_rel_factor: float = 0.12  # 相对下限：min_points_rel = ceil(factor * n_med)
    endpoint_cap_min_delta: float = 0.05  # 两端 frac_inner 差值小于该阈值则视为不确定
    endpoint_cap_r10_delta_mm: float = 0.05  # frac_inner 不明显时，用 r10 差值辅助判别（mm）
    endpoint_frac_inner_trunc_th: float = 0.02  # 截断端：中心区域占比很低
    endpoint_frac_inner_free_th: float = 0.06  # 自由端：端面点会填充中心区域（CAD 长针弯头样例可低至 ~0.075）
    endpoint_tangent_inward_mm: float = 1.0  # 端点切向用“向内 1mm 的弦方向”替代端点差分，避免端点噪声/钩子误导 cap 统计

    # 截断端（底座端）修复：抑制“端点钩子”并避免中心线长度回缩（仅对截断端生效）
    trunc_end_fix_enabled: bool = True
    # trunc_end_fix_mode:
    # - axial_translate: (P0 safer) only change axial coordinate (extend/shrink) and keep lateral offsets
    # - project_to_axis: (legacy, high-risk) project rigid+blend segment onto the estimated axis
    trunc_end_fix_mode: str = "axial_translate"
    trunc_end_rigid_mm: float = 2.0  # 截断端刚性段长度（mm），将该范围内中心线投影到稳定轴线上
    trunc_end_blend_mm: float = 1.0  # 刚性段与自由段之间的过渡混合长度（mm）
    trunc_end_axis_inward1_mm: float = 2.0  # 用于估计端部轴向的内侧起点距离（mm）
    trunc_end_axis_inward2_mm: float = 4.0  # 用于估计端部轴向的内侧终点距离（mm）
    trunc_end_end_quantile: float = 0.999  # 端点外推：点云在轴向投影的高分位（越大越接近“点云最大长度”）
    trunc_end_max_extend_mm: float = 5.0  # 端点外推最大允许量（mm），防止噪声把端点拉太远
    trunc_end_axis_radius_factor: float = 0.7  # 外推时仅使用靠近轴线的点（r <= factor * pin_width）

    # ------------------------------------------
    # Truncated-end lateral pullback (center-out sweep, debug-only in tmp_test)
    #
    # Motivation (user-observed failure mode):
    # - Local slab planes can tilt on one-sided cross-sections near endpoints.
    # - This biases the lateral update and becomes a feedback loop (tangent -> plane -> update -> tangent).
    #
    # Strategy:
    # - Keep the body (middle) as a strong prior.
    # - Unlock the truncated endpoint, and sweep from the middle toward the truncated end to pull it back.
    # - Slab tangent is not a local finite-difference; it is estimated from a one-sided curve segment that
    #   extends from the current point toward the middle (PCA of the segment).
    # ------------------------------------------
    trunc_end_pullback_enabled: bool = True
    trunc_end_pullback_iters: int = 2
    trunc_end_pullback_mid_keep_mm: float = 0.6  # do not update within +/- keep range around the middle (mm)
    trunc_end_pullback_tangent_inward_mm: float = 2.0  # tangent segment length toward the middle (mm)
    trunc_end_pullback_tangent_max_points: int = 12  # cap of segment points for PCA (stability/perf)
    trunc_end_pullback_tangent_min_points: int = 4  # min points to trust PCA; fallback to chord otherwise
    # Anti-fold guards for stage10 pullback (avoid short-range 180deg twists).
    trunc_end_pullback_fold_guard_enabled: bool = True
    trunc_end_pullback_fold_hard_angle_deg: float = 150.0
    trunc_end_pullback_fold_angle_jump_deg: float = 35.0
    trunc_end_pullback_fold_short_edge_ratio: float = 0.45  # relative to centerline step
    trunc_end_pullback_fold_short_edge_angle_deg: float = 95.0
    # In the endpoint risk window, weak slab evidence should skip update instead of damping.
    trunc_end_pullback_lowcov_force_skip_enabled: bool = True
    trunc_end_pullback_lowcov_force_skip_mm: float = 2.0

    # ------------------------------------------
    # Body-guided endpoint regrowth (stage13, experiment in tmp_test)
    #
    # Motivation:
    # - End regions are often single-sided / biased on NeRF point clouds; any local center estimator
    #   (midrange/mean/PCA) can get "locked" to the visible side.
    # - Anchoring endpoints prevents shrinking but may freeze a wrong lateral position.
    #
    # Strategy:
    # 1) Keep the mid-body fit as-is (our strong prior).
    # 2) Trim unreliable end windows on both sides.
    # 3) Unanchor: regrow endpoints outward from the stable body with:
    #    - body-guided tangents (curve-only, toward-mid PCA),
    #    - slab evidence when coverage is sufficient,
    #    - a larger axial slab thickness (taller "cylinder") to increase the chance of seeing both sides,
    #    - auto stop when slab evidence disappears.
    #
    # Default policy:
    # - Be conservative on weak slab evidence (low_coverage_action="stop").
    # - Keep stage11/stage12 enabled by default as a safety net (skip_postfixes=False).
    # ------------------------------------------
    endpoint_regrow_enabled: bool = True
    endpoint_regrow_skip_postfixes: bool = False
    # Hard lock option:
    # - When True and stage13 produces `stage13_after_regrow`, force this curve as final output.
    # - stage11/stage12 are skipped once the stage13 curve is locked.
    force_final_curve_stage13: bool = True
    endpoint_regrow_trim_mm: float = 2.0
    # Adaptive endpoint-tail trim before regrow:
    # trim only when endpoint region shows persistent large-angle deflection relative to a nearby stable axis.
    endpoint_regrow_deflect_trim_enabled: bool = True
    endpoint_regrow_deflect_angle_deg: float = 20.0
    endpoint_regrow_deflect_hard_angle_deg: float = 32.0
    endpoint_regrow_deflect_lateral_mm: float = 0.10
    endpoint_regrow_deflect_min_run_points: int = 3
    endpoint_regrow_deflect_ref_span_mm: float = 2.0
    endpoint_regrow_deflect_tail_grace_points: int = 4
    endpoint_regrow_max_extend_mm: float = 6.0
    endpoint_regrow_half_thickness_factor: float = 2.0  # increase slab "cylinder height" near ends
    endpoint_regrow_min_stop_points: int = 12  # stop regrowth if slab has fewer points than this
    endpoint_regrow_min_update_points: int = 30  # require this many points to trust slab center update
    endpoint_regrow_min_coverage: float = 0.55  # require this coverage to trust slab center update
    endpoint_regrow_low_coverage_action: str = "stop"  # pred_only | damp | stop
    endpoint_regrow_low_coverage_damp: float = 0.25
    endpoint_regrow_max_update_scale: float = 1.50  # allow larger lateral corrections near endpoints
    # Stage13 candidate center estimators for endpoint regrow.
    # center_mode: midrange | trimmed_mean | geomedian | epcs_pair (EPCS-lite paired angular bins)
    endpoint_regrow_center_mode: str = "midrange"
    endpoint_regrow_center_trimmed_q_low: float = 0.10
    endpoint_regrow_center_trimmed_q_high: float = 0.90
    endpoint_regrow_center_geomedian_max_iter: int = 32
    endpoint_regrow_center_geomedian_tol_mm: float = 0.01
    endpoint_regrow_center_epcs_bins: int = 24
    endpoint_regrow_center_epcs_min_bin_points: int = 2
    endpoint_regrow_center_epcs_min_pairs: int = 3
    endpoint_regrow_accept_gate_enabled: bool = True  # conservative default: stage13 candidate must pass acceptance gate
    endpoint_regrow_accept_gate_rollback: bool = True  # rollback stage13 candidate when gate says "worse"
    endpoint_regrow_accept_lff_tol: float = 0.01
    endpoint_regrow_accept_far_tol: float = 0.02
    endpoint_regrow_accept_p99_tol_mm: float = 0.08
    endpoint_regrow_accept_endpoint_risk_gate: bool = True
    endpoint_regrow_accept_endpoint_window_mm: float = 3.0
    # Stage13 tri-class pre-gate (v1 conservative by default):
    # - extend: only when endpoint evidence is sufficient and action reason exists
    # - keep: default fallback for ambiguous/insufficient evidence
    # - shorten: only for severe endpoint risk
    endpoint_regrow_triclass_enabled: bool = True
    endpoint_regrow_triclass_variant: str = "v1"
    endpoint_regrow_triclass_extend_cov_min: float = 0.60
    endpoint_regrow_triclass_extend_support_n1_min: int = 15
    endpoint_regrow_triclass_extend_radial_var_max: float = 1.00
    endpoint_regrow_triclass_shorten_fail_count: int = 3
    endpoint_regrow_triclass_ambiguous_policy: str = "strict_keep"  # strict_keep | probe_on_gray(v2 reserved)
    endpoint_regrow_triclass_min_action_trim_mm: float = 0.05
    endpoint_regrow_triclass_backward_scan_steps: int = 6
    endpoint_regrow_triclass_backward_near_steps: int = 1
    endpoint_regrow_triclass_cap_reliability_min: float = 0.60
    endpoint_regrow_triclass_shorten_extra_mm: float = 0.40
    # Tri-class corridor evidence (multi-step forward region):
    # when one-step evidence is weak but forward corridor still has enough support,
    # allow extension in hard-axial mode (pred-only, no lateral slab pull).
    endpoint_regrow_triclass_corridor_enabled: bool = True
    endpoint_regrow_triclass_corridor_steps: int = 3
    endpoint_regrow_triclass_corridor_valid_steps_min: int = 2
    endpoint_regrow_triclass_corridor_support_sum_min: int = 36
    endpoint_regrow_triclass_corridor_support_peak_min: int = 12
    endpoint_regrow_triclass_corridor_cov_min: float = 0.45
    endpoint_regrow_triclass_corridor_rv_max: float = 1.30
    endpoint_regrow_triclass_hard_axis_bbox_grace_steps: int = 2
    # Tri-class adaptive thresholds (reduce hard-coded magic numbers by per-side robust stats).
    endpoint_regrow_triclass_adaptive_enabled: bool = True
    endpoint_regrow_triclass_adaptive_min_samples: int = 3

    # Stage13 step-level boundary guard:
    # - bbox-front support: reuse endpoint OBB sizing to ensure there are still forward points.
    # - polar boundary: build a local non-rectangular boundary in slab (u,v) and stop/clamp if crossed.
    endpoint_regrow_guard_enabled: bool = True
    endpoint_regrow_guard_bbox_depth_mm: float = 0.8
    endpoint_regrow_guard_bbox_min_points: int = 4
    endpoint_regrow_guard_use_polar_boundary: bool = True
    endpoint_regrow_guard_polar_bins: int = 24
    endpoint_regrow_guard_polar_min_bin_points: int = 2
    endpoint_regrow_guard_polar_min_valid_bins: int = 6
    endpoint_regrow_guard_polar_radius_quantile: float = 0.90
    endpoint_regrow_guard_polar_margin_mm: float = 0.08
    endpoint_regrow_guard_polar_fail_action: str = "stop"  # stop | clamp
    endpoint_regrow_guard_bbox_half_u_floor_factor: float = 0.50
    endpoint_regrow_guard_bbox_half_v_floor_factor: float = 0.50
    # Gradient-based sparse-tail mode (new): detect sudden front-support collapse by trend, not only abs threshold.
    endpoint_regrow_drop_guard_enabled: bool = True
    endpoint_regrow_drop_window_steps: int = 3
    endpoint_regrow_drop_min_baseline_points: int = 24
    endpoint_regrow_drop_ratio: float = 0.35
    endpoint_regrow_drop_abs_points: int = 20
    endpoint_regrow_sparse_confirm_points: int = 10
    endpoint_regrow_sparse_confirm_steps: int = 2
    endpoint_regrow_sparse_continue_ratio: float = 0.90
    endpoint_regrow_guard_bbox_probe_enabled: bool = True
    endpoint_regrow_guard_bbox_probe_steps: int = 1
    endpoint_regrow_guard_bbox_probe_depth_mm: float = 2.4
    endpoint_regrow_guard_bbox_probe_radius_mm: float = 1.5
    endpoint_regrow_guard_bbox_probe_min_points: int = 12
    endpoint_regrow_guard_bbox_probe_min_n_cand_scale: float = 8.0
    endpoint_regrow_sparse_dense_extra_confirm_steps: int = 0
    endpoint_regrow_sparse_dense_min_n_cand_scale: float = 8.0
    endpoint_regrow_direction_lock_enabled: bool = True
    endpoint_regrow_direction_lock_lateral_streak: int = 2
    endpoint_regrow_direction_lock_angle_deg: float = 16.0
    endpoint_regrow_triclass_borderline_hard_axial: bool = False

    # Rigid endpoint prior (material-driven): near endpoint, large relative drift / large twist is unlikely.
    endpoint_regrow_rigid_enabled: bool = True
    endpoint_regrow_rigid_anchor_mm: float = 3.0
    endpoint_regrow_rigid_max_angle_deg: float = 22.0
    endpoint_regrow_rigid_max_angle_deg_lowcov: float = 14.0
    endpoint_regrow_rigid_near_mm: float = 2.0
    endpoint_regrow_rigid_lateral_max_mm: float = 0.18
    endpoint_regrow_rigid_lateral_min_thickness_factor: float = 0.50
    endpoint_regrow_rigid_lateral_relax_per_mm: float = 0.08
    endpoint_regrow_rigid_lateral_cap_mm: float = 0.45
    endpoint_regrow_rigid_radial_var_max: float = 1.00
    endpoint_regrow_rigid_soft_mode: bool = True  # keep axial extension; clamp/pred-only on bad shape
    endpoint_regrow_rigid_badshape_patience: int = 0  # 0 disables bad-shape hard stop (axial extension stays enabled)
    endpoint_regrow_dump_steps: bool = True

    # ------------------------------------------
    # Endpoint bbox fix（通用端点长度/钩子修复）
    #
    # 背景：
    # - 迭代/平滑/重采样 + 局部切向抖动，可能导致端点“缩短”或端点附近出现钩状弯曲
    # - 仅依赖 cap 特征推断截断端时，若判别为 ambiguous（base_end_idx_local=None），截断端修复可能不执行
    #
    # 思路（用户提出的“包围盒”思想落地）：
    # - 在端点沿“稳定轴向”的外侧构建一个小的有向包围盒/圆柱（OBB in {t,u,v}），若盒外仍有点 → 端点偏短
    # - 若稳定轴向外侧有点，而端点局部切向外侧无点，且两者夹角较大 → 端点前发生弯曲/钩子（更可能是算法伪影）
    #
    # 策略：
    # - 两端都做“长度补足（extend）”
    # - “钩子修复（rigid+blend 投影到稳定轴）”默认只对截断端生效（避免抹掉自由端真实缺陷弯曲）
    # ------------------------------------------
    endpoint_bbox_fix_enabled: bool = True
    endpoint_bbox_depth_mm: float = 2.0  # “包围盒外侧”检查深度（mm）
    endpoint_bbox_min_points: int = 15  # 外侧盒中最少点数，超过则认为端点偏短（更保守，减少误触发）
    endpoint_bbox_extend_quantile: float = 0.99  # 外推目标的轴向分位（更保守，减少 length diff 回退）
    endpoint_bbox_max_extend_mm: float = 5.0  # 端点外推最大允许量（mm）
    endpoint_bbox_axis_radius_factor: float = 0.7  # 轴向圆柱半径 = factor * pin_width（pin_width 为截面跨度估计）
    endpoint_bbox_use_obb: bool = True  # True: 端点外侧检查使用局部 OBB（自适应宽/厚）；False: 退回轴向圆柱
    endpoint_bbox_size_inward_mm: float = 4.0  # 用于估计 OBB 横向尺寸：从端点向内统计的轴向范围（mm）
    endpoint_bbox_size_q_low: float = 0.10  # OBB 横向 span 的低分位（更鲁棒，抗单侧缺失/噪声）
    endpoint_bbox_size_q_high: float = 0.90  # OBB 横向 span 的高分位
    endpoint_bbox_size_margin: float = 1.10  # OBB 横向半尺寸安全系数（>1 更保守，减少漏检导致不外推）
    endpoint_bbox_min_half_u_factor: float = 0.50  # OBB 半宽下限 = factor * pin_width
    endpoint_bbox_min_half_v_factor: float = 0.50  # OBB 半厚下限 = factor * pin_thickness
    endpoint_bbox_hook_fix_enabled: bool = True
    endpoint_bbox_hook_angle_deg: float = 25.0  # 局部切向与稳定轴向夹角大于该阈值，才考虑“钩子修复”
    endpoint_bbox_hook_base_only: bool = True  # True: 只对截断端生效；False: 两端都可能修复（更激进）

    # ------------------------------------------
    # Endpoint risk gate (P0 stop-bleed / two-phase commit)
    #
    # Motivation:
    # - NeRF point clouds are often surface-only; NN distance is not a reliable inside/outside signal.
    # - When stage11/stage12 post-fix pushes the endpoint region "out of the cloud", we must auto-rollback.
    #
    # Implementation:
    # - Evaluate slab slice stats within a short endpoint window (0~endpoint_risk_window_mm).
    # - Compute a risk tuple; if post-fix becomes worse, rollback and mark high_risk.
    # ------------------------------------------
    endpoint_risk_gate_enabled: bool = True
    endpoint_risk_window_mm: float = 3.0
    endpoint_risk_min_coverage: float = 0.55
    endpoint_risk_max_delta_mm: float = 0.5
    endpoint_risk_max_clipped_rate: float = 0.30
    endpoint_risk_max_radial_var: float = 0.8

    # 调试输出：抓取中间结果定位“钩子/缩短”来源（默认关闭）
    debug_dump_dir: Optional[str] = None  # 例如 r"<private-path>"
    debug_dump_first_n_segments: int = 0  # >0 时仅输出前 N 段
    debug_dump_every_iter: bool = False  # True: 每轮迭代都输出；False: 仅输出关键阶段/第0轮

    # 终端/曲线平滑
    smooth_soft_window_points: int = 9
    smooth_strong_window_points: int = 21
    smooth_polyorder: int = 2

    # width/thickness 估计（基于 centerline 局部截面）
    width_quantile_low: float = 0.10
    width_quantile_high: float = 0.90
    width_sample_stride: int = 5  # 每隔多少个 centerline 点取一次截面统计
    # 几何先验（毫米）：针截面宽/厚范围基本固定；用于在 NeRF 单侧缺失时 clamp 估计值，避免尺度被低估
    pin_width_prior_min_mm: float = 1.3
    pin_width_prior_max_mm: float = 2.3
    pin_thickness_prior_mm: float = 0.6
    pin_thickness_prior_min_factor: float = 0.70
    pin_thickness_prior_max_factor: float = 1.40

    def dbscan_eps(self) -> float:
        return mm_to_unit(self.dbscan_eps_mm, self.unit_scale)

    def dbscan_eps_min(self) -> float:
        return mm_to_unit(self.dbscan_eps_min_mm, self.unit_scale)

    def centerline_step(self) -> float:
        return mm_to_unit(self.centerline_step_mm, self.unit_scale)

    def slab_half_thickness(self) -> float:
        # 经验：至少覆盖噪声与步长，避免切片过薄导致不稳定；同时不能太厚以免跨弯折混层
        half_mm = max(0.5 * self.centerline_step_mm, 1.5 * self.nerf_noise_mm)
        return mm_to_unit(half_mm, self.unit_scale)

    def slab_search_radius(self, pin_width_hint: Optional[float] = None) -> float:
        # 若后续有截面宽度估计，可用作 radius hint；否则给一个保守的默认值。
        # 这里默认取 ~3mm，覆盖 1.3–2.3mm 宽度 + 噪声。
        if pin_width_hint is not None and pin_width_hint > 0:
            return float(max(1.5 * pin_width_hint, mm_to_unit(2.0, self.unit_scale)))
        return mm_to_unit(3.0, self.unit_scale)

    def max_update(self) -> float:
        return mm_to_unit(self.max_update_mm, self.unit_scale)


def estimate_cloud_stats(points, sample_n=5000, seed=0):
    """
    快速估算点云尺度信息（单位与 points 一致）。

    返回:
        diag: AABB 对角线长度
        median_nn: 子采样后的最近邻距离中位数
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0, 0.0

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ext = mx - mn
    diag = float(np.linalg.norm(ext))

    if len(pts) < 2:
        return diag, 0.0

    if len(pts) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=sample_n, replace=False)
        pts_s = pts[idx]
    else:
        pts_s = pts

    tree = cKDTree(pts_s)
    d, _ = tree.query(pts_s, k=2)
    median_nn = float(np.median(d[:, 1]))
    return diag, median_nn


def stable_order_clusters(clusters: List[np.ndarray]) -> Tuple[List[int], dict]:
    """
    Stable ordering for two-row pin layouts (PCA + row split + lexicographic sort).
    Returns (order_indices, meta).
    """
    if not clusters:
        return [], {}

    centers = np.asarray([np.mean(c, axis=0) for c in clusters], dtype=np.float64)
    if len(centers) == 1:
        return [0], {"method": "single", "mean": centers[0].tolist()}

    mean = centers.mean(axis=0)
    X = centers - mean
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    u_axis = eigvecs[:, order[0]]
    v_axis = eigvecs[:, order[1]]
    u = X @ u_axis
    v = X @ v_axis
    v_median = float(np.median(v))
    rows = (v > v_median).astype(int)

    sort_idx = sorted(range(len(centers)), key=lambda i: (rows[i], u[i]))
    meta = {
        "method": "pca_row_lexsort",
        "mean": mean.tolist(),
        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),
        "v_median": v_median,
    }
    return sort_idx, meta


# ==========================================
# graph + slab：基础工具
# ==========================================


def estimate_median_nn(points: np.ndarray, sample_n: int = 8000, seed: int = 0) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0

    if len(pts) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=sample_n, replace=False)
        pts = pts[idx]

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(np.median(d[:, 1]))


def _safe_ratio(numer: float, denom: float) -> float:
    denom = float(denom)
    if denom <= 0:
        return 0.0
    return float(numer) / denom


def _denoise_points_open3d(
    points: np.ndarray,
    config: CenterlineConfig,
    *,
    tag: str,
    median_nn_hint: Optional[float] = None,
    enable_ror: Optional[bool] = None,
    enable_sor: Optional[bool] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    NeRF 点云离群点/噪声清理：non-finite -> ROR -> SOR

    Returns:
        points_filtered: (N,3) numpy array
        report: JSON-serializable dict for detailed logging/debug
    """
    pts = np.asarray(points, dtype=float)
    report: Dict[str, Any] = {"tag": str(tag), "n_in": int(len(pts))}
    if len(pts) == 0:
        report["n_out"] = 0
        return pts, report

    if enable_ror is None:
        enable_ror = bool(getattr(config, "denoise_ror_enable", True))
    if enable_sor is None:
        enable_sor = bool(getattr(config, "denoise_sor_enable", True))

    # non-finite cleanup (avoid Open3D version differences by doing it in numpy)
    finite_mask = np.isfinite(pts).all(axis=1)
    n_non_finite = int(np.sum(~finite_mask))
    if n_non_finite > 0:
        pts = pts[finite_mask]
    report["removed_non_finite"] = n_non_finite
    report["n_after_non_finite"] = int(len(pts))
    if len(pts) == 0:
        report["n_out"] = 0
        return pts, report

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # --- ROR ---
    if bool(enable_ror):
        nb_points = int(max(1, getattr(config, "denoise_ror_nb_points", 16)))
        if len(pts) > nb_points:
            radius_mm_cfg = float(getattr(config, "denoise_ror_radius_mm", 0.0))
            if radius_mm_cfg > 0:
                radius = float(mm_to_unit(radius_mm_cfg, config.unit_scale))
                radius_src = "cfg_mm"
            else:
                # Auto radius:
                # - main term: median NN spacing (robust against a small portion of floaters)
                # - floor term: avoid radius ~ 0 when the input has duplicates/degenerate spacing
                median_nn = float(median_nn_hint or 0.0)
                floor_mm = float(max(0.5 * config.nerf_noise_mm, 0.5 * config.centerline_step_mm))
                floor_u = float(mm_to_unit(floor_mm, config.unit_scale))
                radius = float(max(float(getattr(config, "denoise_ror_radius_factor", 3.0)) * median_nn, floor_u))
                radius_src = "auto"

            pcd_ror, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=float(radius))
            n_before = int(np.asarray(pcd.points).shape[0])
            n_after = int(len(ind))
            report["ror"] = {
                "enabled": True,
                "nb_points": nb_points,
                "radius": float(radius),
                "radius_mm": float(radius / (config.unit_scale + 1e-12)),
                "radius_src": str(radius_src),
                "removed": int(n_before - n_after),
                "removed_ratio": float(_safe_ratio(n_before - n_after, n_before)),
                "n_after": n_after,
            }
            pcd = pcd_ror
            pts = np.asarray(pcd.points)
        else:
            report["ror"] = {"enabled": True, "skipped": "too_few_points", "nb_points": nb_points}
    else:
        report["ror"] = {"enabled": False}

    # --- SOR ---
    if bool(enable_sor):
        nb_neighbors = int(max(1, getattr(config, "denoise_sor_nb_neighbors", 20)))
        if len(pts) > nb_neighbors:
            std_ratio = float(max(0.1, getattr(config, "denoise_sor_std_ratio", 2.0)))
            pcd_sor, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=float(std_ratio))
            n_before = int(np.asarray(pcd.points).shape[0])
            n_after = int(len(ind))
            report["sor"] = {
                "enabled": True,
                "nb_neighbors": nb_neighbors,
                "std_ratio": float(std_ratio),
                "removed": int(n_before - n_after),
                "removed_ratio": float(_safe_ratio(n_before - n_after, n_before)),
                "n_after": n_after,
            }
            pcd = pcd_sor
            pts = np.asarray(pcd.points)
        else:
            report["sor"] = {"enabled": True, "skipped": "too_few_points", "nb_neighbors": nb_neighbors}
    else:
        report["sor"] = {"enabled": False}

    report["n_out"] = int(len(pts))
    report["removed_total"] = int(report["n_in"] - report["n_out"])
    report["removed_total_ratio"] = float(_safe_ratio(report["removed_total"], report["n_in"]))
    return pts, report


def build_knn_graph(points: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []

    k_eff = int(max(1, min(k, n - 1)))
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=k_eff + 1)  # includes self

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j, w in zip(idxs[i][1:], dists[i][1:]):
            j = int(j)
            w = float(w)
            if j == i:
                continue
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj


def prune_graph_edges(adj: List[List[Tuple[int, float]]], max_edge: float) -> List[List[Tuple[int, float]]]:
    if not adj:
        return adj

    max_edge = float(max_edge)
    pruned: List[List[Tuple[int, float]]] = []
    for neighbors in adj:
        pruned.append([(j, w) for (j, w) in neighbors if w <= max_edge])
    return pruned


def connected_components(adj: List[List[Tuple[int, float]]]) -> List[np.ndarray]:
    n = len(adj)
    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
    comps: List[np.ndarray] = []
    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        comps.append(np.asarray(comp, dtype=int))
    return comps


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        return np.zeros((3,), dtype=float)
    return v / n


def _principal_axis_pca(points: np.ndarray) -> np.ndarray:
    """
    Estimate the dominant direction of a point set via PCA.
    Returns a unit vector (3,).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    mean = pts.mean(axis=0)
    X = pts - mean
    cov = np.cov(X.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return _normalize_vec(axis)
    except Exception:
        return np.array([0.0, 0.0, 1.0], dtype=float)


def _orthonormal_basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal (u,v) basis that spans the plane perpendicular to `axis`.
    """
    t = _normalize_vec(axis)
    if not np.any(np.isfinite(t)) or float(np.linalg.norm(t)) <= 1e-12:
        t = np.array([0.0, 0.0, 1.0], dtype=float)

    basis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(basis, t))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=float)

    u = _normalize_vec(np.cross(t, basis))
    v = _normalize_vec(np.cross(t, u))
    return u, v


def _t_overlap_ratio(t0: float, t1: float, s0: float, s1: float) -> float:
    """
    Overlap ratio of two 1D segments [t0,t1] and [s0,s1], normalized by the shorter length.
    Value in [0,1] (best-effort).
    """
    t0, t1 = (float(t0), float(t1)) if t0 <= t1 else (float(t1), float(t0))
    s0, s1 = (float(s0), float(s1)) if s0 <= s1 else (float(s1), float(s0))
    inter = max(0.0, min(t1, s1) - max(t0, s0))
    lt = max(1e-12, t1 - t0)
    ls = max(1e-12, s1 - s0)
    return float(inter / min(lt, ls))


def _estimate_needle_axis_from_components(
    pts_cluster: np.ndarray,
    comps: List[np.ndarray],
    *,
    min_points: int,
) -> np.ndarray:
    """
    Estimate a common needle axis for a DBSCAN cluster by averaging PCA axes from components.

    Rationale:
    - For multi-needle clusters, PCA on the whole cluster may pick a board/layout direction as the top eigenvector.
    - Per-component PCA is much more likely to align with the needle axis. Components are mostly parallel, so
      averaging yields a stable axis.
    """
    # Use the largest (most reliable) component as the reference direction, then align and
    # average other component axes with point-count weights. This prevents small fragments
    # from biasing the axis used by fragment-merge heuristics.
    keep = [c for c in comps if int(len(c)) >= int(min_points)]
    if not keep:
        return _principal_axis_pca(pts_cluster)

    keep = sorted(keep, key=lambda c: int(len(c)), reverse=True)
    ref = _principal_axis_pca(pts_cluster[keep[0]])
    if float(np.linalg.norm(ref)) <= 0.5:
        return _principal_axis_pca(pts_cluster)

    acc = np.zeros(3, dtype=float)
    wsum = 0.0
    for c in keep:
        axis = _principal_axis_pca(pts_cluster[c])
        if float(np.linalg.norm(axis)) <= 0.5:
            continue
        if float(np.dot(axis, ref)) < 0:
            axis = -axis
        w = float(len(c))
        acc += w * axis
        wsum += w

    if not np.any(np.isfinite(acc)) or float(wsum) <= 1e-9:
        return _principal_axis_pca(pts_cluster)

    mean_axis = _normalize_vec(acc / wsum)
    if float(np.linalg.norm(mean_axis)) <= 0.5:
        return _principal_axis_pca(pts_cluster)
    return mean_axis


def _merge_fragment_components_by_reconnect(
    *,
    adj: List[List[Tuple[int, float]]],
    pts_cluster: np.ndarray,
    comps: List[np.ndarray],
    config: "CenterlineConfig",
    cluster_id: int,
) -> List[np.ndarray]:
    """
    Merge likely-fragmented components by ADDING bridging edges to the pruned graph.

    This keeps the downstream shortest-path ordering valid, unlike simply concatenating index sets.
    """
    if not bool(getattr(config, "merge_fragments_enable", True)):
        return comps

    min_pts = int(getattr(config, "min_component_points", 600))
    max_iters = int(max(0, getattr(config, "merge_fragments_max_iters", 3) or 0))
    if max_iters <= 0:
        return comps

    uv_max_mm = float(getattr(config, "merge_fragments_uv_mm", 2.5))
    size_ratio_max = float(getattr(config, "merge_fragments_size_ratio_max", 0.10))
    overlap_max = float(getattr(config, "merge_fragments_t_overlap_max", 0.6))
    bridge_max_mm = float(getattr(config, "merge_fragments_bridge_max_mm", 2.0))
    unit_scale = float(getattr(config, "unit_scale", 1.0))
    uv_max_u = float(mm_to_unit(uv_max_mm, unit_scale))
    bridge_max_u = float(mm_to_unit(bridge_max_mm, unit_scale))

    comps_cur = list(comps)
    for _it in range(max_iters):
        # Only merge components that already qualify as "segments" (>=min_component_points).
        keep_ids = [i for i, c in enumerate(comps_cur) if int(len(c)) >= int(min_pts)]
        if len(keep_ids) <= 1:
            break

        t_axis = _estimate_needle_axis_from_components(pts_cluster, [comps_cur[i] for i in keep_ids], min_points=min_pts)
        u_axis, v_axis = _orthonormal_basis_from_axis(t_axis)

        # Component stats (for merge decision).
        stats = []
        for i in keep_ids:
            c = comps_cur[i]
            pts_c = pts_cluster[c]
            cent = pts_c.mean(axis=0)
            uv = np.array([float(np.dot(cent, u_axis)), float(np.dot(cent, v_axis))], dtype=float)
            proj = pts_c @ t_axis
            t0 = float(np.min(proj))
            t1 = float(np.max(proj))
            stats.append({"comp_id": int(i), "size": int(len(c)), "uv": uv, "t0": t0, "t1": t1})

        # Union-find on components that should be merged.
        m = len(stats)
        parent = list(range(m))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        cand_pairs = 0
        for a in range(m):
            for b in range(a + 1, m):
                duv = float(np.linalg.norm(stats[a]["uv"] - stats[b]["uv"]))
                if not np.isfinite(duv) or duv > uv_max_u:
                    continue
                sa = int(stats[a]["size"])
                sb = int(stats[b]["size"])
                sr = float(min(sa, sb) / max(1, max(sa, sb)))

                # Merge rule:
                # - If one component is much smaller, treat it as a fragment even if axial ranges overlap.
                # - Otherwise, require limited axial overlap (typical of a needle cut into two pieces).
                if np.isfinite(sr) and sr <= size_ratio_max:
                    union(a, b)
                    cand_pairs += 1
                    continue

                ov = _t_overlap_ratio(stats[a]["t0"], stats[a]["t1"], stats[b]["t0"], stats[b]["t1"])
                if not np.isfinite(ov) or ov > overlap_max:
                    continue
                union(a, b)
                cand_pairs += 1

        if cand_pairs == 0:
            break

        # Collect groups.
        groups: Dict[int, List[int]] = {}
        for i in range(m):
            r = find(i)
            groups.setdefault(r, []).append(i)

        edges_added = 0
        for _, members in groups.items():
            if len(members) <= 1:
                continue

            members_sorted = sorted(members, key=lambda x: int(stats[x]["size"]), reverse=True)
            root_idx = members_sorted[0]
            root_comp_id = int(stats[root_idx]["comp_id"])
            root_nodes = comps_cur[root_comp_id]
            root_pts = pts_cluster[root_nodes]
            if len(root_pts) == 0:
                continue

            tree = cKDTree(root_pts)
            for midx in members_sorted[1:]:
                comp_id = int(stats[midx]["comp_id"])
                nodes = comps_cur[comp_id]
                pts = pts_cluster[nodes]
                if len(pts) == 0:
                    continue

                d, nn_idx = tree.query(pts, k=1)
                d = np.asarray(d).reshape(-1)
                nn_idx = np.asarray(nn_idx).reshape(-1)
                if d.size == 0 or nn_idx.size == 0:
                    continue

                j = int(np.argmin(d))
                min_d = float(d[j])
                if not np.isfinite(min_d) or min_d > bridge_max_u:
                    continue

                root_local = int(nn_idx[j])
                u = int(root_nodes[root_local])
                v = int(nodes[j])
                adj[u].append((v, float(min_d)))
                adj[v].append((u, float(min_d)))
                edges_added += 1

        if edges_added <= 0:
            break

        before = int(len(comps_cur))
        comps_cur = connected_components(adj)
        after = int(len(comps_cur))
        if after < before:
            print(f"[INFO] cluster={cluster_id} merge_fragments: comps {before}->{after} | edges_added={edges_added}")
        else:
            break

    return comps_cur


def dijkstra_with_prev(adj: List[List[Tuple[int, float]]], start: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=float)
    prev = np.full(n, -1, dtype=int)
    dist[start] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, int(start))]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue

        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def approximate_farthest_pair_two_sweep(points: np.ndarray) -> Tuple[int, int]:
    """
    用两次“最远点”查询近似点集直径端点（Euclidean 2-sweep）。
    经验上对细长目标（针脚）比在有环表面图上做 diameter 更稳定。
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0

    # IMPORTANT:
    # - Previously we used a fixed seed (index=0). On NeRF clouds, point order is arbitrary; if pts[0] happens to
    #   be on a side wall / noisy region, the 2-sweep endpoints can lock onto the side surface and bias the whole
    #   initialization path (hard to fix later by local slab refinement).
    # - Use a deterministic and geometry-aware seed: the point closest to the robust center (median), which is
    #   much less sensitive to input point ordering.
    center = np.median(pts, axis=0)
    a0 = int(np.argmin(np.linalg.norm(pts - center[None, :], axis=1)))
    d0 = np.linalg.norm(pts - pts[a0], axis=1)
    a = int(np.argmax(d0))
    d1 = np.linalg.norm(pts - pts[a], axis=1)
    b = int(np.argmax(d1))
    return a, b


def shortest_path_nodes(adj: List[List[Tuple[int, float]]], start: int, end: int) -> np.ndarray:
    """返回 start->end 的最短路节点序列（基于 dijkstra_with_prev 的 prev 回溯）。"""
    start = int(start)
    end = int(end)
    if start == end:
        return np.asarray([start], dtype=int)

    dist, prev = dijkstra_with_prev(adj, start)
    if end < 0 or end >= len(prev) or not np.isfinite(dist[end]):
        return np.asarray([], dtype=int)

    path: List[int] = []
    cur = end
    while cur != -1:
        path.append(int(cur))
        if cur == start:
            break
        cur = int(prev[cur])

    if not path or path[-1] != start:
        return np.asarray([], dtype=int)

    path.reverse()
    return np.asarray(path, dtype=int)


def _reweight_adj_dir_penalty(
    adj: List[List[Tuple[int, float]]],
    points: np.ndarray,
    axis: np.ndarray,
    lambda_: float,
) -> List[List[Tuple[int, float]]]:
    """
    Build a re-weighted adjacency list for Dijkstra:
      w' = w * (1 + lambda * (1 - |dot(dir, axis)|))

    This penalizes edges that are not aligned with the estimated needle axis, reducing "shortcut" edges
    in high-curvature regions. Default behavior is unchanged unless explicitly enabled.
    """
    if not adj:
        return adj

    pts = np.asarray(points, dtype=np.float64)
    axis_u = _normalize_vec(axis)
    lam = float(max(0.0, lambda_))
    if lam <= 1e-12:
        return adj

    out: List[List[Tuple[int, float]]] = []
    for i, neigh in enumerate(adj):
        pi = pts[i]
        nn: List[Tuple[int, float]] = []
        for j, w in neigh:
            j = int(j)
            w = float(w)
            if j == i:
                continue
            v = pts[j] - pi
            nv = float(np.linalg.norm(v))
            if nv > 1e-12:
                d = v / nv
                dot = float(abs(np.dot(d, axis_u)))
                w = w * (1.0 + lam * (1.0 - dot))
            nn.append((j, float(w)))
        out.append(nn)
    return out


def approximate_diameter_path(adj: List[List[Tuple[int, float]]], component_idx: np.ndarray) -> np.ndarray:
    """
    用“两次最短路”近似图直径端点，并返回端点间最短路径（节点索引序列）。
    """
    if len(component_idx) == 0:
        return np.asarray([], dtype=int)

    a = int(component_idx[0])
    dist_a, _ = dijkstra_with_prev(adj, a)
    b = int(component_idx[np.argmax(dist_a[component_idx])])

    dist_b, prev_b = dijkstra_with_prev(adj, b)
    c = int(component_idx[np.argmax(dist_b[component_idx])])

    # reconstruct path b -> c
    path: List[int] = []
    cur = c
    while cur != -1:
        path.append(cur)
        if cur == b:
            break
        cur = int(prev_b[cur])

    path.reverse()
    return np.asarray(path, dtype=int)


def resample_polyline_by_step(points: np.ndarray, step: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()

    step = float(step)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-12:
        return pts[:1].copy()

    if step <= 0:
        return pts.copy()

    s = np.concatenate([[0.0], np.cumsum(seg)])
    sample_s = np.arange(0.0, total, step)
    if sample_s.size == 0 or sample_s[-1] < total:
        sample_s = np.append(sample_s, total)

    idx = np.searchsorted(s, sample_s, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (sample_s - s[idx]) / (seg[idx] + 1e-12)
    out = pts[idx] + (pts[idx + 1] - pts[idx]) * t[:, None]
    return out


def compute_tangents(curve: np.ndarray) -> np.ndarray:
    c = np.asarray(curve, dtype=float)
    n = len(c)
    if n == 0:
        return c.copy()
    if n == 1:
        return np.zeros_like(c)

    t = np.zeros_like(c)
    t[0] = c[1] - c[0]
    t[-1] = c[-1] - c[-2]
    if n > 2:
        t[1:-1] = c[2:] - c[:-2]

    norms = np.linalg.norm(t, axis=1) + 1e-12
    return t / norms[:, None]


def make_plane_basis(tangent: np.ndarray, prev_u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tangent, dtype=float)
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-12:
        t = np.array([0.0, 0.0, 1.0])
    else:
        t = t / t_norm

    u = None
    if prev_u is not None:
        pu = np.asarray(prev_u, dtype=float)
        pu = pu - t * float(np.dot(pu, t))
        pu_norm = np.linalg.norm(pu)
        if pu_norm > 1e-8:
            u = pu / pu_norm

    if u is None:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(tmp, t))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(t, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)

    v = np.cross(t, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def robust_midrange_2d(coords_u: np.ndarray, coords_v: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    u = np.asarray(coords_u, dtype=float)
    v = np.asarray(coords_v, dtype=float)
    if len(u) == 0:
        return 0.0, 0.0

    q_low = float(q_low)
    q_high = float(q_high)
    u_lo, u_hi = np.quantile(u, [q_low, q_high])
    v_lo, v_hi = np.quantile(v, [q_low, q_high])
    return float(0.5 * (u_lo + u_hi)), float(0.5 * (v_lo + v_hi))


def _trimmed_mean_1d(values: np.ndarray, q_low: float, q_high: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= 0:
        return 0.0
    ql = float(np.clip(float(q_low), 0.0, 0.49))
    qh = float(np.clip(float(q_high), 0.51, 1.0))
    if qh <= ql + 1e-9:
        return float(np.median(arr))
    lo, hi = np.quantile(arr, [ql, qh])
    mask = (arr >= lo) & (arr <= hi)
    if not np.any(mask):
        return float(np.median(arr))
    return float(np.mean(arr[mask]))


def _geometric_median_2d(points_uv: np.ndarray, max_iter: int = 32, tol: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=float)
    if pts.ndim != 2 or pts.shape[0] <= 0:
        return np.array([0.0, 0.0], dtype=float)
    if pts.shape[0] == 1:
        return np.asarray(pts[0], dtype=float)

    x = np.median(pts, axis=0)
    max_iter = int(max(1, max_iter))
    tol = float(max(1e-12, tol))

    for _ in range(max_iter):
        diff = pts - x
        dist = np.linalg.norm(diff, axis=1)
        mask = dist > 1e-12
        if not np.any(mask):
            break
        w = 1.0 / dist[mask]
        x_new = np.sum(pts[mask] * w[:, None], axis=0) / (np.sum(w) + 1e-12)
        if float(np.linalg.norm(x_new - x)) <= tol:
            x = x_new
            break
        x = x_new
    return np.asarray(x, dtype=float)


def _endpoint_regrow_center_uv(
    coords_u: np.ndarray,
    coords_v: np.ndarray,
    *,
    mode: str,
    q_low: float,
    q_high: float,
    geomedian_max_iter: int,
    geomedian_tol_u: float,
    epcs_bins: int,
    epcs_min_bin_points: int,
    epcs_min_pairs: int,
) -> Tuple[float, float, Dict[str, object]]:
    u = np.asarray(coords_u, dtype=float).reshape(-1)
    v = np.asarray(coords_v, dtype=float).reshape(-1)
    n = int(min(len(u), len(v)))
    if n <= 0:
        return 0.0, 0.0, {"center_mode_effective": "none", "center_fallback": "empty", "center_pairs": 0}

    u = u[:n]
    v = v[:n]
    mode_use = str(mode or "midrange").strip().lower()
    allowed = {"midrange", "trimmed_mean", "geomedian", "epcs_pair"}
    if mode_use not in allowed:
        mode_use = "midrange"

    if mode_use == "midrange":
        du, dv = robust_midrange_2d(u, v, q_low=q_low, q_high=q_high)
        return float(du), float(dv), {"center_mode_effective": "midrange", "center_fallback": "", "center_pairs": 0}

    if mode_use == "trimmed_mean":
        du = _trimmed_mean_1d(u, q_low=q_low, q_high=q_high)
        dv = _trimmed_mean_1d(v, q_low=q_low, q_high=q_high)
        return float(du), float(dv), {
            "center_mode_effective": "trimmed_mean",
            "center_fallback": "",
            "center_pairs": 0,
        }

    if mode_use == "geomedian":
        uv = np.stack([u, v], axis=1)
        gm = _geometric_median_2d(
            uv,
            max_iter=int(max(1, geomedian_max_iter)),
            tol=float(max(1e-12, geomedian_tol_u)),
        )
        if np.all(np.isfinite(gm)):
            return float(gm[0]), float(gm[1]), {
                "center_mode_effective": "geomedian",
                "center_fallback": "",
                "center_pairs": 0,
            }
        du, dv = robust_midrange_2d(u, v, q_low=q_low, q_high=q_high)
        return float(du), float(dv), {
            "center_mode_effective": "midrange",
            "center_fallback": "geomedian_nonfinite",
            "center_pairs": 0,
        }

    bins = int(max(8, int(epcs_bins)))
    if bins % 2 != 0:
        bins += 1
    min_bin = int(max(1, int(epcs_min_bin_points)))
    min_pairs = int(max(1, int(epcs_min_pairs)))

    ang = np.arctan2(v, u)
    idx = np.floor((ang + np.pi) * (bins / (2.0 * np.pi))).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)

    cnt = np.bincount(idx, minlength=bins).astype(np.int64)
    sum_u = np.bincount(idx, weights=u, minlength=bins)
    sum_v = np.bincount(idx, weights=v, minlength=bins)

    pair_centers: List[np.ndarray] = []
    pair_weights: List[float] = []
    half_bins = bins // 2
    for i in range(half_bins):
        j = i + half_bins
        ci = int(cnt[i])
        cj = int(cnt[j])
        if ci < min_bin or cj < min_bin:
            continue
        mui = float(sum_u[i] / (ci + 1e-12))
        mvi = float(sum_v[i] / (ci + 1e-12))
        muj = float(sum_u[j] / (cj + 1e-12))
        mvj = float(sum_v[j] / (cj + 1e-12))
        pair_centers.append(np.array([0.5 * (mui + muj), 0.5 * (mvi + mvj)], dtype=float))
        pair_weights.append(float(min(ci, cj)))

    n_pairs = int(len(pair_centers))
    if n_pairs >= min_pairs:
        pc = np.stack(pair_centers, axis=0)
        ww = np.asarray(pair_weights, dtype=float)
        wsum = float(np.sum(ww))
        if wsum > 1e-12:
            c_uv = np.sum(pc * ww[:, None], axis=0) / wsum
        else:
            c_uv = np.mean(pc, axis=0)
        return float(c_uv[0]), float(c_uv[1]), {
            "center_mode_effective": "epcs_pair",
            "center_fallback": "",
            "center_pairs": int(n_pairs),
        }

    du, dv = robust_midrange_2d(u, v, q_low=q_low, q_high=q_high)
    return float(du), float(dv), {
        "center_mode_effective": "midrange",
        "center_fallback": f"epcs_pairs_{n_pairs}_lt_{min_pairs}",
        "center_pairs": int(n_pairs),
    }


def slab_angular_coverage(coords_u: np.ndarray, coords_v: np.ndarray, bins: int) -> float:
    """
    Heuristic "single-sided" detection for slab cross-sections.

    We bin points by polar angle in the local (u,v) plane and compute the fraction of occupied bins.
    When the point cloud is missing one side of the cross-section (common in NeRF), the angular
    coverage tends to drop sharply, and slab centering becomes ill-posed (drifts toward the visible side).

    Returns:
      coverage in [0,1], where 1 means all bins have points.
    """
    u = np.asarray(coords_u, dtype=np.float64).reshape(-1)
    v = np.asarray(coords_v, dtype=np.float64).reshape(-1)
    n = min(len(u), len(v))
    if n <= 0:
        return 0.0

    bins = int(max(4, bins))
    ang = np.arctan2(v[:n], u[:n])  # [-pi, pi]
    idx = np.floor((ang + np.pi) * (bins / (2.0 * np.pi))).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)

    occ = np.bincount(idx, minlength=bins) > 0
    return float(np.mean(occ.astype(np.float64)))


def _radial_var_from_uv(coords_u: np.ndarray, coords_v: np.ndarray, eps: float = 1e-9) -> float:
    """A cheap "out-of-cloud / single-sided slice" risk proxy (unitless).

    Reference (internal): `研究报告/中心线.md` proposes:
      radial_var = (q90(r) - q10(r)) / (q50(r) + eps)

    When the centerline point moves toward a surface (or outside), one side radius becomes small while the
    other side remains large, which tends to increase this ratio.
    """
    u = np.asarray(coords_u, dtype=np.float64).reshape(-1)
    v = np.asarray(coords_v, dtype=np.float64).reshape(-1)
    n = min(len(u), len(v))
    if n < 8:
        return float("inf")

    r = np.sqrt(u[:n] * u[:n] + v[:n] * v[:n])
    if not np.all(np.isfinite(r)):
        r = r[np.isfinite(r)]
    if len(r) < 8:
        return float("inf")

    q10 = float(np.quantile(r, 0.10))
    q50 = float(np.quantile(r, 0.50))
    q90 = float(np.quantile(r, 0.90))
    if q50 <= 1e-12:
        return float("inf")
    return float((q90 - q10) / (q50 + float(eps)))


def _polar_boundary_radius_limit_from_uv(
    coords_u: np.ndarray,
    coords_v: np.ndarray,
    theta: float,
    *,
    bins: int = 24,
    q_radius: float = 0.90,
    min_bin_points: int = 2,
    min_valid_bins: int = 6,
) -> Tuple[float, int]:
    """Estimate a local non-rectangular boundary radius in slab (u,v) for a given angle.

    Returns:
      (radius_limit_u, n_valid_bins)
      - radius_limit_u is `inf` when boundary evidence is insufficient.
    """
    u = np.asarray(coords_u, dtype=np.float64).reshape(-1)
    v = np.asarray(coords_v, dtype=np.float64).reshape(-1)
    n = int(min(len(u), len(v)))
    if n < 8:
        return float("inf"), 0

    bins = int(max(8, bins))
    q_radius = float(np.clip(q_radius, 0.50, 0.995))
    min_bin_points = int(max(1, min_bin_points))
    min_valid_bins = int(max(1, min(min_valid_bins, bins)))

    uu = u[:n]
    vv = v[:n]
    rr = np.sqrt(uu * uu + vv * vv)
    ang = np.mod(np.arctan2(vv, uu), 2.0 * np.pi)
    idx = np.floor(ang * (float(bins) / (2.0 * np.pi))).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)

    lim = np.full(int(bins), np.nan, dtype=np.float64)
    for b in range(int(bins)):
        rb = rr[idx == b]
        if len(rb) >= min_bin_points:
            lim[b] = float(np.quantile(rb, q_radius))

    valid = np.isfinite(lim)
    n_valid = int(np.sum(valid))
    if n_valid < min_valid_bins:
        return float("inf"), int(n_valid)

    theta_n = float(np.mod(theta, 2.0 * np.pi))
    tgt = int(np.floor(theta_n * (float(bins) / (2.0 * np.pi))))
    tgt = int(np.clip(tgt, 0, bins - 1))
    if np.isfinite(lim[tgt]):
        return float(lim[tgt]), int(n_valid)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) <= 0:
        return float("inf"), int(n_valid)
    d = np.abs(valid_idx - int(tgt))
    d = np.minimum(d, int(bins) - d)
    near = int(valid_idx[int(np.argmin(d))])
    return float(lim[near]), int(n_valid)


def _endpoint_region_slice_summary(
    points: np.ndarray,
    curve_end0: np.ndarray,
    config: CenterlineConfig,
    *,
    pin_width_hint: Optional[float] = None,
    tree: Optional[cKDTree] = None,
    window_u: Optional[float] = None,
) -> Dict[str, float]:
    """Quick slab-based stats within a short endpoint window.

    `curve_end0` must be oriented so that the endpoint of interest is at index=0 and arclen increases inward.
    """
    pts = np.asarray(points, dtype=float)
    c0 = np.asarray(curve_end0, dtype=float)
    if len(pts) < 20 or len(c0) < 2:
        return {
            "n_eval": 0.0,
            "min_n_slab": 0.0,
            "min_coverage": 0.0,
            "max_delta_mm": float("inf"),
            "clipped_rate": 1.0,
            "max_radial_var": float("inf"),
        }

    unit_scale = float(config.unit_scale)
    search_r = float(config.slab_search_radius(pin_width_hint))
    half_t = float(config.slab_half_thickness())
    max_update_u = float(config.max_update())
    ql = float(getattr(config, "slab_quantile_low", 0.10))
    qh = float(getattr(config, "slab_quantile_high", 0.90))
    bins = int(getattr(config, "slab_coverage_bins", 12))

    if window_u is None:
        window_u = float(mm_to_unit(float(getattr(config, "endpoint_risk_window_mm", 3.0)), unit_scale))
    window_u = float(max(0.0, window_u))

    tangents = compute_tangents(c0)
    s = _cum_arclen(c0)
    j_end = int(np.searchsorted(s, float(window_u), side="right"))
    j_end = int(np.clip(j_end, 2, len(c0)))

    if tree is None:
        tree = cKDTree(pts)

    min_n_slab: int = 10**9
    min_cov: float = 1.0
    max_delta_mm: float = 0.0
    max_radial_var: float = 0.0
    clipped_cnt: int = 0
    n_eval: int = 0

    prev_u: Optional[np.ndarray] = None
    for i in range(j_end):
        c = c0[i]
        t = tangents[i]
        u, v = make_plane_basis(t, prev_u=prev_u)
        prev_u = u

        cand = tree.query_ball_point(c, r=search_r)
        if len(cand) < 3:
            continue

        local = pts[cand] - c
        axial = local @ t
        mask = np.abs(axial) <= half_t
        if int(np.sum(mask)) < 3:
            continue

        local = local[mask]
        coords_u = local @ u
        coords_v = local @ v

        n = int(min(len(coords_u), len(coords_v)))
        if n <= 0:
            continue

        cov = slab_angular_coverage(coords_u, coords_v, bins=bins)
        du, dv = robust_midrange_2d(coords_u, coords_v, q_low=ql, q_high=qh)
        delta_u = float(np.hypot(float(du), float(dv)))
        delta_mm = float(delta_u / (unit_scale + 1e-12))
        radial_var = _radial_var_from_uv(coords_u, coords_v)

        n_eval += 1
        min_n_slab = min(min_n_slab, n)
        min_cov = min(min_cov, float(cov))
        max_delta_mm = max(max_delta_mm, float(delta_mm))
        max_radial_var = max(max_radial_var, float(radial_var))
        if delta_u > max_update_u + 1e-12:
            clipped_cnt += 1

    if n_eval <= 0:
        return {
            "n_eval": 0.0,
            "min_n_slab": 0.0,
            "min_coverage": 0.0,
            "max_delta_mm": float("inf"),
            "clipped_rate": 1.0,
            "max_radial_var": float("inf"),
        }

    clipped_rate = float(clipped_cnt / float(max(1, n_eval)))
    return {
        "n_eval": float(n_eval),
        "min_n_slab": float(min_n_slab),
        "min_coverage": float(min_cov),
        "max_delta_mm": float(max_delta_mm),
        "clipped_rate": float(clipped_rate),
        "max_radial_var": float(max_radial_var),
    }


def _endpoint_risk_detail(stats: Dict[str, float], config: CenterlineConfig) -> Dict[str, float]:
    """Compute endpoint risk detail and per-condition hits."""
    min_n_slab = float(stats.get("min_n_slab", 0.0))
    min_cov = float(stats.get("min_coverage", 0.0))
    max_delta_mm = float(stats.get("max_delta_mm", float("inf")))
    clipped_rate = float(stats.get("clipped_rate", 1.0))
    max_radial_var = float(stats.get("max_radial_var", float("inf")))

    # Thresholds (conservative defaults, see 资料.md + 研究报告/中心线.md)
    n_th = float(max(1, int(getattr(config, "min_slab_points", 30))))
    cov_th = float(getattr(config, "endpoint_risk_min_coverage", 0.0))
    delta_th = float(getattr(config, "endpoint_risk_max_delta_mm", float("inf")))
    clip_th = float(getattr(config, "endpoint_risk_max_clipped_rate", 1.0))
    rv_th = float(getattr(config, "endpoint_risk_max_radial_var", float("inf")))

    hit_min_n_slab = 1.0 if min_n_slab < n_th else 0.0
    hit_min_cov = 1.0 if min_cov < cov_th else 0.0
    hit_max_delta = 1.0 if max_delta_mm > delta_th else 0.0
    hit_clipped = 1.0 if clipped_rate > clip_th else 0.0
    hit_max_rv = 1.0 if max_radial_var > rv_th else 0.0
    fail_count = float(hit_min_n_slab + hit_min_cov + hit_max_delta + hit_clipped + hit_max_rv)

    return {
        "fail_count": float(fail_count),
        "min_n_slab": float(min_n_slab),
        "min_cov": float(min_cov),
        "max_delta_mm": float(max_delta_mm),
        "clipped_rate": float(clipped_rate),
        "max_radial_var": float(max_radial_var),
        "n_th": float(n_th),
        "cov_th": float(cov_th),
        "delta_th": float(delta_th),
        "clip_th": float(clip_th),
        "rv_th": float(rv_th),
        "hit_min_n_slab": float(hit_min_n_slab),
        "hit_min_cov": float(hit_min_cov),
        "hit_max_delta": float(hit_max_delta),
        "hit_clipped": float(hit_clipped),
        "hit_max_rv": float(hit_max_rv),
    }


def _endpoint_risk_tuple(stats: Dict[str, float], config: CenterlineConfig) -> Tuple[float, float, float, float, float, float]:
    """Return a tuple that can be compared lexicographically; larger => worse."""
    d = _endpoint_risk_detail(stats, config)
    fail_count = float(d.get("fail_count", 0.0) or 0.0)
    max_delta_mm = float(d.get("max_delta_mm", float("inf")))
    max_radial_var = float(d.get("max_radial_var", float("inf")))
    min_cov = float(d.get("min_cov", 0.0) or 0.0)
    min_n_slab = float(d.get("min_n_slab", 0.0) or 0.0)
    clipped_rate = float(d.get("clipped_rate", 1.0) or 1.0)

    # Bigger is worse:
    # - fail_count dominates
    # - then continuous metrics (delta, radial_var, etc.)
    return (
        float(fail_count),
        float(max_delta_mm),
        float(max_radial_var),
        float(-min_cov),
        float(-min_n_slab),
        float(clipped_rate),
    )


def _endpoint_risk_worse(before: Dict[str, float], after: Dict[str, float], config: CenterlineConfig) -> bool:
    return _endpoint_risk_tuple(after, config) > _endpoint_risk_tuple(before, config)


def _endpoint_cap_features(
    points: np.ndarray,
    curve: np.ndarray,
    config: CenterlineConfig,
    pin_width_hint: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    端点“端面(cap)存在性”特征：用于区分
    - 截断端/底座端：无端面点，截面中心区域基本空
    - 自由端：有端面点，截面中心区域有较高占比

    返回：
        {
          "start": {"n":..., "frac_inner":..., "r10":...},
          "end":   {"n":..., "frac_inner":..., "r10":...},
        }
    所有距离单位与 points/curve 一致。
    """
    pts = np.asarray(points, dtype=float)
    c = np.asarray(curve, dtype=float)
    if len(pts) < 10 or len(c) < 2:
        return {
            "start": {"n": 0.0, "frac_inner": 0.0, "r10": float("inf")},
            "end": {"n": 0.0, "frac_inner": 0.0, "r10": float("inf")},
        }

    tree = cKDTree(pts)
    tangents = compute_tangents(c)
    search_r = float(config.slab_search_radius(pin_width_hint))
    cap_depth = float(mm_to_unit(config.endpoint_cap_depth_mm, config.unit_scale))
    r_inner = float(mm_to_unit(config.endpoint_cap_inner_radius_mm, config.unit_scale))

    # 端点切向更稳健：用“向内一定弧长的弦方向”代替端点差分切向（端点更易受噪声/钩子影响）。
    try:
        seg = c[1:] - c[:-1]
        ds = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        if float(s[-1]) > 1e-12 and len(c) >= 3:
            inward = float(mm_to_unit(getattr(config, "endpoint_tangent_inward_mm", 1.0), config.unit_scale))
            inward = float(max(inward, 2.0 * float(config.centerline_step())))
            inward = float(min(inward, 0.45 * float(s[-1])))

            i_s = int(np.searchsorted(s, inward, side="right"))
            i_s = max(1, min(i_s, len(c) - 1))
            v_s = c[i_s] - c[0]
            nv = float(np.linalg.norm(v_s))
            if nv > 1e-12:
                tangents[0] = v_s / nv

            target = float(s[-1] - inward)
            i_e = int(np.searchsorted(s, target, side="left"))
            i_e = max(0, min(i_e, len(c) - 2))
            v_e = c[-1] - c[i_e]
            nv = float(np.linalg.norm(v_e))
            if nv > 1e-12:
                tangents[-1] = v_e / nv
    except Exception:
        pass

    # 内部参考：用于相对化端部点数（NeRF 密度不均时更稳）
    n_med: float = 0.0
    try:
        n_curve = int(len(c))
        if n_curve >= 6:
            i0 = int(round(0.2 * (n_curve - 1)))
            i1 = int(round(0.8 * (n_curve - 1)))
            if i1 <= i0:
                i0, i1 = 1, n_curve - 2
            i0 = max(1, min(i0, n_curve - 2))
            i1 = max(i0 + 1, min(i1, n_curve - 2))

            max_samples = 25
            stride = max(1, int(round((i1 - i0) / max(1, min(max_samples, i1 - i0 + 1)))))
            counts: List[int] = []
            for i in range(i0, i1 + 1, stride):
                cc_i = c[i]
                t_i = tangents[i]
                cand_i = tree.query_ball_point(cc_i, r=search_r)
                if len(cand_i) == 0:
                    counts.append(0)
                    continue
                local_i = pts[cand_i] - cc_i
                axial_i = local_i @ t_i
                counts.append(int(np.sum(np.abs(axial_i) <= cap_depth)))
            n_med = float(np.median(counts)) if len(counts) > 0 else 0.0
    except Exception:
        n_med = 0.0

    min_points_rel = int(
        np.ceil(float(getattr(config, "endpoint_cap_min_points_rel_factor", 0.0)) * float(n_med))
    )
    min_points_abs = int(getattr(config, "endpoint_cap_min_points", 0))
    min_points = int(max(1, max(min_points_abs, min_points_rel)))

    def one(ep: int) -> Dict[str, float]:
        cc = c[ep]
        t = tangents[ep]
        u, v = make_plane_basis(t)
        cand = tree.query_ball_point(cc, r=search_r)
        if len(cand) == 0:
            return {
                "n": 0.0,
                "n_cand": 0.0,
                "frac_inner": 0.0,
                "r10": float("inf"),
                "drop_ratio": 0.0,
                "reliability": 0.0,
            }

        local = pts[cand] - cc
        axial = local @ t
        mask = np.abs(axial) <= cap_depth
        local = local[mask]
        n = int(len(local))
        if n <= 0:
            return {
                "n": 0.0,
                "n_cand": float(len(cand)),
                "frac_inner": 0.0,
                "r10": float("inf"),
                "drop_ratio": 0.0,
                "reliability": 0.0,
            }

        uu = local @ u
        vv = local @ v
        rr = np.sqrt(uu * uu + vv * vv)
        frac_inner = float(np.mean(rr < r_inner))
        r10 = float(np.quantile(rr, 0.10)) if n >= 10 else float(np.min(rr))
        drop_ratio = float(n / (float(n_med) + 1e-12)) if float(n_med) > 0 else float("nan")
        reliability = float(np.clip(float(n) / float(min_points), 0.0, 1.0))
        return {
            "n": float(n),
            "n_cand": float(len(cand)),
            "frac_inner": frac_inner,
            "r10": r10,
            "drop_ratio": drop_ratio,
            "reliability": reliability,
        }

    return {"start": one(0), "end": one(-1), "_internal": {"n_med": float(n_med), "min_points": float(min_points)}}


def _infer_truncated_end_from_cap(
    cap_feat: Dict[str, Dict[str, float]], config: CenterlineConfig
) -> Tuple[Optional[int], str]:
    """
    基于端面特征判别“截断端/底座端”。
    返回：
        (base_end_idx, reason)
    其中 base_end_idx 为 0 或 -1；若不确定则返回 (None, "ambiguous").
    """
    a = cap_feat.get("start", {})
    b = cap_feat.get("end", {})
    n0 = float(a.get("n", 0.0))
    n1 = float(b.get("n", 0.0))

    f0 = float(a.get("frac_inner", 0.0))
    f1 = float(b.get("frac_inner", 0.0))
    # 端部点数很低时 frac_inner 波动会很大，容易把“截断端”误判成“也有 cap”。
    # 因为你的场景里自由端端面几乎不会缺失点，因此可用点数(相对内部参考)作为“可靠性”对 cap 证据降权。
    r0 = float(a.get("reliability", 1.0))
    r1 = float(b.get("reliability", 1.0))
    s0 = float(np.clip(r0, 0.0, 1.0)) * f0
    s1 = float(np.clip(r1, 0.0, 1.0)) * f1
    delta = float(abs(s0 - s1))

    # 如果两端都极少点：端部特征不可靠，保守降级
    min_points_abs = float(getattr(config, "endpoint_cap_min_points", 0.0))
    if n0 < min_points_abs and n1 < min_points_abs:
        return None, "ambiguous_low_points_both"

    # 主判别：两端 frac_inner 差异明显时，截断端中心区域占比更低
    if delta >= float(config.endpoint_cap_min_delta):
        return (0, "cap_score") if s0 < s1 else (-1, "cap_score")

    # 次判别：当 frac_inner 差异不大但一端几乎为 0、另一端明显非 0 时，仍可判别
    trunc_th = float(getattr(config, "endpoint_frac_inner_trunc_th", 0.02))
    free_th = float(getattr(config, "endpoint_frac_inner_free_th", 0.06))
    if (f0 <= trunc_th and f1 >= free_th) or (f1 <= trunc_th and f0 >= free_th):
        return (0, "cap_frac_threshold") if f0 < f1 else (-1, "cap_frac_threshold")

    # frac_inner 很接近时：用 r10（截面半径分位数）辅助判别。
    # 直观：截断端中心“更空”，r10 往往更大；自由端端面填充中心，r10 更小。
    r0 = float(a.get("r10", float("inf")))
    r1 = float(b.get("r10", float("inf")))
    if not np.isfinite(r0) or not np.isfinite(r1):
        return None, "ambiguous_small_delta"

    th = float(mm_to_unit(config.endpoint_cap_r10_delta_mm, config.unit_scale))
    if float(abs(r0 - r1)) < th:
        return None, "ambiguous_small_delta"

    return (0, "cap_r10") if r0 > r1 else (-1, "cap_r10")


def _ensure_savgol_window(window: int, n: int, polyorder: int) -> int:
    w = int(window)
    if w % 2 == 0:
        w += 1
    if n % 2 == 0:
        w = min(w, n - 1)
    else:
        w = min(w, n)
    min_w = polyorder + 2
    if min_w % 2 == 0:
        min_w += 1
    w = max(w, min_w)
    if w > n:
        w = n - 1 if n % 2 == 0 else n
    if w < 3:
        w = 3
    return w


def smooth_curve_curvature_adaptive(curve: np.ndarray, config: CenterlineConfig) -> np.ndarray:
    c = np.asarray(curve, dtype=float)
    n = len(c)
    if n < 5:
        return c

    p = int(config.smooth_polyorder)
    soft_w = _ensure_savgol_window(config.smooth_soft_window_points, n, p)
    strong_w = _ensure_savgol_window(config.smooth_strong_window_points, n, p)
    if strong_w < soft_w:
        strong_w = soft_w

    def sg(points: np.ndarray, w: int) -> np.ndarray:
        xs = savgol_filter(points[:, 0], w, p)
        ys = savgol_filter(points[:, 1], w, p)
        zs = savgol_filter(points[:, 2], w, p)
        return np.stack([xs, ys, zs], axis=1)

    curve_soft = sg(c, soft_w)
    curve_strong = sg(c, strong_w)

    # 曲率用于自适应混合：曲率小处更强平滑，曲率大处更弱平滑（保留弯折）
    k_inner = compute_discrete_curvature(curve_soft)
    kappa = np.zeros(n, dtype=float)
    if len(k_inner) > 0:
        kappa[1:-1] = k_inner
        kappa[0] = k_inner[0]
        kappa[-1] = k_inner[-1]

    k_med = float(np.median(kappa))
    k_hi = float(np.quantile(kappa, 0.9))
    if k_hi <= k_med + 1e-12:
        blend = np.ones(n, dtype=float)
    else:
        x = (kappa - k_med) / (k_hi - k_med + 1e-12)
        x = np.clip(x, 0.0, 1.0)
        blend = 1.0 - x  # 低曲率 -> 1.0(强平滑)，高曲率 -> 0.0(弱平滑)

    out = blend[:, None] * curve_strong + (1.0 - blend)[:, None] * curve_soft
    out[0] = curve_soft[0]
    out[-1] = curve_soft[-1]
    return out


def _debug_enabled(config: CenterlineConfig, segment_id: Optional[int]) -> bool:
    if not getattr(config, "debug_dump_dir", None):
        return False
    n = int(getattr(config, "debug_dump_first_n_segments", 0) or 0)
    if n > 0 and segment_id is not None and int(segment_id) >= n:
        return False
    return True


def _debug_segment_dir(config: CenterlineConfig, debug_tag: str) -> Path:
    root = Path(str(getattr(config, "debug_dump_dir"))).expanduser()
    seg_dir = root / str(debug_tag)
    seg_dir.mkdir(parents=True, exist_ok=True)
    return seg_dir


def _debug_write_ply_points(path: Path, points: np.ndarray, color: Optional[List[float]] = None) -> None:
    pts = np.asarray(points, dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color is not None and len(color) == 3:
        pcd.paint_uniform_color([float(color[0]), float(color[1]), float(color[2])])
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def _cum_arclen(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=float)
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = float(np.dot(a, b) / (na * nb + 1e-12))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _endpoint_tangent_inward(curve_view: np.ndarray, config: CenterlineConfig) -> Optional[np.ndarray]:
    """
    Compute a robust inward tangent at curve_view[0] by using a chord to a point at a small inward arclen.

    curve_view: curve oriented so that the endpoint of interest is at index 0 and inward direction is +index.
    """
    c0 = np.asarray(curve_view, dtype=float)
    if len(c0) < 2:
        return None

    try:
        seg = c0[1:] - c0[:-1]
        ds = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        if float(s[-1]) < 1e-12:
            return None

        inward_u = float(mm_to_unit(float(getattr(config, "endpoint_tangent_inward_mm", 1.0)), config.unit_scale))
        inward_u = float(max(inward_u, 2.0 * float(config.centerline_step())))
        inward_u = float(min(inward_u, 0.45 * float(s[-1])))
        i = int(np.searchsorted(s, inward_u, side="right"))
        i = max(1, min(i, len(c0) - 1))
        v = c0[i] - c0[0]
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            v = c0[1] - c0[0]
            nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            return None
        return v / nv
    except Exception:
        return None


def _endpoint_stable_axis_inward(curve_view: np.ndarray, config: CenterlineConfig) -> Optional[np.ndarray]:
    """
    Compute a "stable" inward axis near curve_view[0] by using a chord between two inward arclen positions.

    This is more robust than the immediate endpoint tangent when the endpoint suffers from hook-like artifacts.
    """
    c0 = np.asarray(curve_view, dtype=float)
    if len(c0) < 2:
        return None

    try:
        s = _cum_arclen(c0)
        unit_scale = float(config.unit_scale)
        in1_u = float(mm_to_unit(float(getattr(config, "trunc_end_axis_inward1_mm", 2.0)), unit_scale))
        in2_u = float(mm_to_unit(float(getattr(config, "trunc_end_axis_inward2_mm", 4.0)), unit_scale))

        def pick_index(dist_u: float) -> int:
            idx = int(np.searchsorted(s, float(dist_u), side="left"))
            return int(np.clip(idx, 0, len(c0) - 1))

        i1 = pick_index(in1_u)
        i2 = pick_index(in2_u)
        if i1 >= len(c0) - 1:
            i1 = max(0, len(c0) - 2)
        if i2 <= i1:
            i2 = min(len(c0) - 1, i1 + 1)

        v = c0[i2] - c0[i1]
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            v = c0[min(len(c0) - 1, i1 + 1)] - c0[i1]
            nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            return None
        return v / nv
    except Exception:
        return None


def _clamp_pin_width_thickness_prior(
    pin_width: float,
    pin_thickness: float,
    config: CenterlineConfig,
) -> Tuple[float, float]:
    """
    Clamp (width, thickness) by the physical priors of the pin cross-section.

    NOTE: Inputs/outputs are in the same unit as the point cloud (not mm).
    """
    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))

    w_min = float(mm_to_unit(float(getattr(config, "pin_width_prior_min_mm", 1.3)), unit_scale))
    w_max = float(mm_to_unit(float(getattr(config, "pin_width_prior_max_mm", 2.3)), unit_scale))
    w_min, w_max = (min(w_min, w_max), max(w_min, w_max))

    t_nom = float(mm_to_unit(float(getattr(config, "pin_thickness_prior_mm", 0.6)), unit_scale))
    t_min_factor = float(getattr(config, "pin_thickness_prior_min_factor", 0.70))
    t_max_factor = float(getattr(config, "pin_thickness_prior_max_factor", 1.40))
    t_min = float(min(t_nom * t_min_factor, t_nom * t_max_factor))
    t_max = float(max(t_nom * t_min_factor, t_nom * t_max_factor))

    w = float(pin_width) if np.isfinite(pin_width) else 0.0
    t = float(pin_thickness) if np.isfinite(pin_thickness) else 0.0

    if w <= 0.0:
        w = w_min
    w = float(np.clip(w, w_min, w_max))

    if t <= 0.0:
        t = t_nom
    t = float(np.clip(t, t_min, t_max))
    return w, t


def _endpoint_estimate_bbox_obb_axes_and_half_sizes(
    local: np.ndarray,
    t_axis_in: np.ndarray,
    config: CenterlineConfig,
    pin_width_hint: Optional[float] = None,
    pin_thickness_hint: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, float]]:
    """
    Estimate a local OBB (u,v half-sizes) around an endpoint for the "outside points" check.

    Returns: (u_axis, v_axis, half_u, half_v, stats)
    - u_axis/v_axis: orthonormal basis perpendicular to t_axis_in (stable inward axis).
    - half_u/half_v: half sizes along u/v (same unit as points).
    """
    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))

    # Always start from a valid in-plane basis; PCA will refine u/v if enough data exists.
    u0, v0 = make_plane_basis(np.asarray(t_axis_in, dtype=float))

    w_hint = float(pin_width_hint) if (pin_width_hint is not None and np.isfinite(pin_width_hint)) else 0.0
    t_hint = float(pin_thickness_hint) if (pin_thickness_hint is not None and np.isfinite(pin_thickness_hint)) else 0.0
    w_hint, t_hint = _clamp_pin_width_thickness_prior(w_hint, t_hint, config)

    # Inward slice for OBB sizing (avoid leaking into the bend / far field).
    inward_u = float(mm_to_unit(float(getattr(config, "endpoint_bbox_size_inward_mm", 4.0)), unit_scale))
    inward_u = float(max(inward_u, 2.0 * float(config.centerline_step())))

    ql = float(getattr(config, "endpoint_bbox_size_q_low", getattr(config, "width_quantile_low", 0.10)))
    qh = float(getattr(config, "endpoint_bbox_size_q_high", getattr(config, "width_quantile_high", 0.90)))
    ql = float(np.clip(ql, 0.0, 0.49))
    qh = float(np.clip(qh, 0.51, 1.0))

    margin = float(getattr(config, "endpoint_bbox_size_margin", 1.20))
    margin = float(max(margin, 1.0))

    # Conservative lateral gate for collecting the inward slice points.
    search_r = float(config.slab_search_radius(w_hint))

    t_in = np.asarray(t_axis_in, dtype=float)
    axial_in = local @ t_in
    # local points are relative to endpoint; inward points have axial_in >= 0
    mask_in = (axial_in >= 0.0) & (axial_in <= inward_u)
    if int(np.sum(mask_in)) >= 3:
        radial = local - axial_in[:, None] * t_in[None, :]
        rr = np.linalg.norm(radial, axis=1)
        mask_in = mask_in & (rr <= search_r)

    n_in = int(np.sum(mask_in))
    stats: Dict[str, float] = {
        "obb_used": 0.0,
        "obb_n_in": float(n_in),
        "obb_half_u_mm": 0.0,
        "obb_half_v_mm": 0.0,
    }

    # Fallback: use priors/hints only (still returns an oriented box in u0/v0).
    span_u = float(w_hint)
    span_v = float(t_hint)
    u_axis = np.asarray(u0, dtype=float)
    v_axis = np.asarray(v0, dtype=float)

    if n_in >= 30 and bool(getattr(config, "endpoint_bbox_use_obb", True)):
        try:
            local_in = local[mask_in]
            uv = np.stack([local_in @ u0, local_in @ v0], axis=1)
            uv0 = uv - uv.mean(axis=0)
            cov = np.cov(uv0.T)
            eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
            axes2 = eigvecs[:, ::-1]  # descending (cols)

            u2 = axes2[:, 0]
            u_axis = float(u2[0]) * u0 + float(u2[1]) * v0
            nu = float(np.linalg.norm(u_axis))
            if nu > 1e-12:
                u_axis = u_axis / nu
            else:
                u_axis = np.asarray(u0, dtype=float)

            v_axis = np.cross(t_in, u_axis)
            nv = float(np.linalg.norm(v_axis))
            if nv > 1e-12:
                v_axis = v_axis / nv
            else:
                v_axis = np.asarray(v0, dtype=float)

            uu = local_in @ u_axis
            vv = local_in @ v_axis
            span_u = float(np.quantile(uu, qh) - np.quantile(uu, ql))
            span_v = float(np.quantile(vv, qh) - np.quantile(vv, ql))

            # Ensure u is the major axis (width), v is the minor axis (thickness).
            if span_v > span_u:
                span_u, span_v = span_v, span_u
                u_axis, v_axis = v_axis, u_axis

            if np.isfinite(span_u) and np.isfinite(span_v) and span_u > 0.0 and span_v > 0.0:
                span_u, span_v = _clamp_pin_width_thickness_prior(span_u, span_v, config)
                stats["obb_used"] = 1.0
        except Exception:
            # keep fallback
            pass

    bbox_min_half_u_factor = float(
        max(0.0, float(getattr(config, "endpoint_bbox_min_half_u_factor", 0.50) or 0.50))
    )
    bbox_min_half_v_factor = float(
        max(0.0, float(getattr(config, "endpoint_bbox_min_half_v_factor", 0.50) or 0.50))
    )
    half_u = 0.5 * float(span_u) * margin
    half_v = 0.5 * float(span_v) * margin
    min_half_u = float(max(0.0, bbox_min_half_u_factor * float(w_hint)))
    min_half_v = float(max(0.0, bbox_min_half_v_factor * float(t_hint)))
    half_u = float(max(half_u, min_half_u))
    half_v = float(max(half_v, min_half_v))

    stats["obb_half_u_mm"] = float(half_u / (unit_scale + 1e-12))
    stats["obb_half_v_mm"] = float(half_v / (unit_scale + 1e-12))
    return np.asarray(u_axis, dtype=float), np.asarray(v_axis, dtype=float), float(half_u), float(half_v), stats


def fix_endpoints_by_bbox(
    points: np.ndarray,
    curve: np.ndarray,
    base_end_idx_local: Optional[int],
    config: CenterlineConfig,
    pin_width_hint: Optional[float] = None,
    pin_thickness_hint: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Endpoint length/hook stabilization using an oriented bbox (implemented as an axis-aligned cylinder in the
    local {t,u,v} frame).

    - "too short": if there are enough points in a small outward bbox, extend the endpoint outward (quantile-based)
    - "hook": if stable-axis bbox sees points but local-tangent bbox sees few points AND the angle is large, project
      a short rigid segment to the stable axis (base-end only by default).

    NOTE:
    - This fix is data-driven: it only extends when "outside points" exist.
    - The hook projection is intentionally conservative to avoid hiding real defects on the free end.
    """
    stats: Dict[str, Dict[str, float]] = {
        "start": {
            "applied": 0.0,
            "extend_mm": 0.0,
            "n_out_axis": 0.0,
            "n_out_local": 0.0,
            "angle_deg": 0.0,
            "hook": 0.0,
            "rolled_back": 0.0,
        },
        "end": {
            "applied": 0.0,
            "extend_mm": 0.0,
            "n_out_axis": 0.0,
            "n_out_local": 0.0,
            "angle_deg": 0.0,
            "hook": 0.0,
            "rolled_back": 0.0,
        },
    }

    if not bool(getattr(config, "endpoint_bbox_fix_enabled", False)):
        return np.asarray(curve, dtype=float), stats

    pts = np.asarray(points, dtype=float)
    c = np.asarray(curve, dtype=float)
    if len(pts) < 20 or len(c) < 2:
        return c, stats

    unit_scale = float(config.unit_scale)
    depth_u = float(mm_to_unit(float(getattr(config, "endpoint_bbox_depth_mm", 2.0)), unit_scale))
    min_points = int(getattr(config, "endpoint_bbox_min_points", 10))
    q_ext = float(getattr(config, "endpoint_bbox_extend_quantile", 0.995))
    q_ext = float(np.clip(q_ext, 0.90, 1.0))
    max_extend_u = float(mm_to_unit(float(getattr(config, "endpoint_bbox_max_extend_mm", 5.0)), unit_scale))
    axis_r_factor = float(getattr(config, "endpoint_bbox_axis_radius_factor", 0.7))

    hook_enabled = bool(getattr(config, "endpoint_bbox_hook_fix_enabled", False))
    hook_angle_th = float(getattr(config, "endpoint_bbox_hook_angle_deg", 25.0))
    hook_base_only = bool(getattr(config, "endpoint_bbox_hook_base_only", True))

    # Use one KDTree for both endpoints.
    tree = cKDTree(pts)

    out_curve = c.copy()
    for end_name, end_idx_local in (("start", 0), ("end", -1)):
        # Work in a view where the target end is at index 0.
        reversed_view = end_idx_local == -1
        c0 = out_curve[::-1].copy() if reversed_view else out_curve.copy()
        if len(c0) < 2:
            continue
        c0_before = c0.copy()

        is_base_end = base_end_idx_local is not None and int(base_end_idx_local) == int(end_idx_local)
        allow_hook = bool(hook_enabled and (not hook_base_only or is_base_end))

        t_axis_in = _endpoint_stable_axis_inward(c0, config)
        if t_axis_in is None:
            t_axis_in = _endpoint_tangent_inward(c0, config)
        if t_axis_in is None:
            t_axis_in = c0[1] - c0[0]
            nt = float(np.linalg.norm(t_axis_in))
            if nt > 1e-12:
                t_axis_in = t_axis_in / nt
            else:
                continue

        t_local_in = _endpoint_tangent_inward(c0, config)
        if t_local_in is None:
            t_local_in = t_axis_in

        t_axis_out = -np.asarray(t_axis_in, dtype=float)
        t_local_out = -np.asarray(t_local_in, dtype=float)
        ang = _angle_deg(t_axis_out, t_local_out)

        # Clamp width/thickness by physical priors to avoid NeRF single-side under-estimation.
        w_hint, t_hint = _clamp_pin_width_thickness_prior(
            float(pin_width_hint) if pin_width_hint is not None else 0.0,
            float(pin_thickness_hint) if pin_thickness_hint is not None else 0.0,
            config,
        )

        window_u = float(mm_to_unit(float(getattr(config, "endpoint_risk_window_mm", 3.0)), unit_scale))
        risk_before: Optional[Dict[str, float]] = None
        if bool(getattr(config, "endpoint_risk_gate_enabled", True)):
            try:
                risk_before = _endpoint_region_slice_summary(pts, c0, config, pin_width_hint=w_hint, tree=tree, window_u=window_u)
            except Exception:
                risk_before = None

        # Axis-cylinder radius (used for query envelope / fallback mode).
        r_max = float(max(float(axis_r_factor) * float(w_hint), float(mm_to_unit(0.6, unit_scale))))

        # Query radius must cover potential outward extension.
        search_r = float(config.slab_search_radius(w_hint))
        r_query = float(
            max(
                search_r,
                float(np.sqrt(r_max * r_max + max_extend_u * max_extend_u)) + float(mm_to_unit(0.3, unit_scale)),
            )
        )
        cand = tree.query_ball_point(c0[0], r=r_query)
        if len(cand) == 0:
            continue

        local = pts[cand] - c0[0]  # (M,3)

        use_obb = bool(getattr(config, "endpoint_bbox_use_obb", True))
        if use_obb:
            u_axis, v_axis, half_u, half_v, obb_stats = _endpoint_estimate_bbox_obb_axes_and_half_sizes(
                local,
                t_axis_in=np.asarray(t_axis_in, dtype=float),
                config=config,
                pin_width_hint=w_hint,
                pin_thickness_hint=t_hint,
            )
            stats[end_name].update(obb_stats)

            def count_outside(t_out: np.ndarray) -> Tuple[int, np.ndarray]:
                t_out = np.asarray(t_out, dtype=float)
                axial = local @ t_out
                uu = local @ u_axis
                vv = local @ v_axis
                mask_lat = (np.abs(uu) <= half_u) & (np.abs(vv) <= half_v)
                out_mask = mask_lat & (axial > 0.0) & (axial <= depth_u)
                axial_pos = axial[mask_lat & (axial > 0.0)]
                return int(np.sum(out_mask)), np.asarray(axial_pos, dtype=float)

        else:

            def count_outside(t_out: np.ndarray) -> Tuple[int, np.ndarray]:
                t_out = np.asarray(t_out, dtype=float)
                axial = local @ t_out
                radial = local - axial[:, None] * t_out[None, :]
                rr = np.linalg.norm(radial, axis=1)
                mask_r = rr <= r_max
                out_mask = mask_r & (axial > 0.0) & (axial <= depth_u)
                axial_pos = axial[mask_r & (axial > 0.0)]
                return int(np.sum(out_mask)), np.asarray(axial_pos, dtype=float)

        n_out_axis, axial_pos_axis = count_outside(t_axis_out)
        n_out_local, _axial_pos_local = count_outside(t_local_out)

        stats[end_name]["n_out_axis"] = float(n_out_axis)
        stats[end_name]["n_out_local"] = float(n_out_local)
        stats[end_name]["angle_deg"] = float(ang)

        extend_u = 0.0
        if n_out_axis >= int(max(1, min_points)) and len(axial_pos_axis) > 0:
            try:
                s_end = float(np.quantile(axial_pos_axis, q_ext))
            except Exception:
                s_end = float(np.max(axial_pos_axis))
            extend_u = float(np.clip(s_end, 0.0, max_extend_u))

        # Hook heuristic: stable axis sees enough "outside" points but local tangent doesn't, and angle is large.
        hook_low = max(2, int(round(0.25 * float(max(1, min_points)))))
        hook = bool(allow_hook and (ang >= hook_angle_th) and (n_out_axis >= min_points) and (n_out_local <= hook_low))
        stats[end_name]["hook"] = 1.0 if hook else 0.0

        applied = False
        if extend_u > float(0.25 * float(config.centerline_step())):
            c0[0] = c0[0] + extend_u * t_axis_out
            applied = True

        # If we suspect a hook artifact (mostly for the truncated end), project a short rigid+blend range to the
        # stable axis to suppress the curl before reaching the true end.
        if hook and len(c0) >= 3:
            rigid_u = float(mm_to_unit(float(getattr(config, "trunc_end_rigid_mm", 2.0)), unit_scale))
            blend_u = float(mm_to_unit(float(getattr(config, "trunc_end_blend_mm", 1.0)), unit_scale))
            rigid_u = float(max(rigid_u, 0.0))
            blend_u = float(max(blend_u, 0.0))
            total_u = rigid_u + blend_u

            s = _cum_arclen(c0)
            end_new = c0[0].copy()
            t_in = np.asarray(t_axis_in, dtype=float)

            for j in range(1, len(c0)):
                dist_u = float(s[j])
                if dist_u > total_u + 1e-12:
                    break

                alpha = float((c0[j] - end_new) @ t_in)
                if alpha < 0.0:
                    alpha = 0.0
                proj = end_new + alpha * t_in

                if dist_u <= rigid_u + 1e-12 or blend_u <= 1e-12:
                    w = 1.0
                else:
                    w = float(np.clip(1.0 - (dist_u - rigid_u) / (blend_u + 1e-12), 0.0, 1.0))
                c0[j] = w * proj + (1.0 - w) * c0[j]
            applied = True

        if applied:
            stats[end_name]["applied"] = 1.0
            stats[end_name]["extend_mm"] = float(extend_u / (unit_scale + 1e-12))

        if applied and risk_before is not None and bool(getattr(config, "endpoint_risk_gate_enabled", True)):
            try:
                risk_after = _endpoint_region_slice_summary(
                    pts, c0, config, pin_width_hint=w_hint, tree=tree, window_u=window_u
                )
                if _endpoint_risk_worse(risk_before, risk_after, config):
                    stats[end_name]["rolled_back"] = 1.0
                    stats[end_name]["applied"] = 0.0
                    stats[end_name]["extend_mm"] = 0.0
                    # Rollback the whole endpoint fix for this end.
                    c0 = c0_before
                    print(
                        f"[WARN] endpoint_bbox_fix rolled back ({end_name}): "
                        f"delta_mm {float(risk_before.get('max_delta_mm', 0.0)):.3f}->{float(risk_after.get('max_delta_mm', 0.0)):.3f}, "
                        f"cov {float(risk_before.get('min_coverage', 0.0)):.2f}->{float(risk_after.get('min_coverage', 0.0)):.2f}, "
                        f"n {int(risk_before.get('min_n_slab', 0.0))}->{int(risk_after.get('min_n_slab', 0.0))}, "
                        f"rv {float(risk_before.get('max_radial_var', 0.0)):.2f}->{float(risk_after.get('max_radial_var', 0.0)):.2f}, "
                        f"clip {float(risk_before.get('clipped_rate', 0.0)):.2f}->{float(risk_after.get('clipped_rate', 0.0)):.2f}"
                    )
            except Exception:
                pass

        out_curve = c0[::-1] if reversed_view else c0

    return out_curve, stats


def fix_truncated_end_hook_and_shrink(
    points: np.ndarray,
    curve: np.ndarray,
    base_end_idx_local: Optional[int],
    config: CenterlineConfig,
    pin_width_hint: Optional[float] = None,
    pin_thickness_hint: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    针对“截断端/底座端”做端点修复：
    - 解决端点处因开口边界/密度不均导致的“钩子”
    - 解决中心线端点在迭代/平滑中向内收缩导致的“缩短”

    思路：
    1) 在截断端向内若干 mm 处取稳定轴向（用两点弦方向，避开端点局部不稳定）
    2) 沿该轴向，用点云在轴向投影的高分位确定“点云可见长度边界”，把端点外推到该位置
    3) 对截断端前 trunc_end_rigid_mm+blend 的小段做“低自由度”处理（P0 safer）：
       - 默认：仅做轴向平移（保持横向/perp 不变），抑制 shrink 且避免侧向跳变；
       - legacy：将该段投影到轴线上（高风险，可能在弯针处把曲线推到点云外）。

    返回：
        (curve_fixed, stats)
    """
    stats: Dict[str, float] = {
        "applied": 0.0,
        "extend_mm": 0.0,
        "rigid_mm": float(getattr(config, "trunc_end_rigid_mm", 0.0)),
        # If we applied the fix but then rolled back for safety, this becomes 1.0.
        "rolled_back": 0.0,
    }

    if not bool(getattr(config, "trunc_end_fix_enabled", False)):
        return np.asarray(curve, dtype=float), stats
    if base_end_idx_local is None:
        return np.asarray(curve, dtype=float), stats

    n = int(len(curve))
    if n < 2:
        return np.asarray(curve, dtype=float), stats

    base_end_idx_local = int(base_end_idx_local)
    if base_end_idx_local not in (0, -1):
        return np.asarray(curve, dtype=float), stats

    pts = np.asarray(points, dtype=float)
    c = np.asarray(curve, dtype=float)
    if len(pts) < 20:
        return c, stats

    # 将曲线统一成“截断端在 index=0，向内方向 index 增大”的方向，方便处理
    reversed_view = base_end_idx_local == -1
    c0 = c[::-1] if reversed_view else c

    s = _cum_arclen(c0)
    unit_scale = float(config.unit_scale)
    rigid_u = float(mm_to_unit(float(getattr(config, "trunc_end_rigid_mm", 2.0)), unit_scale))
    blend_u = float(mm_to_unit(float(getattr(config, "trunc_end_blend_mm", 1.0)), unit_scale))
    in1_u = float(mm_to_unit(float(getattr(config, "trunc_end_axis_inward1_mm", 2.0)), unit_scale))
    in2_u = float(mm_to_unit(float(getattr(config, "trunc_end_axis_inward2_mm", 4.0)), unit_scale))
    q_end = float(getattr(config, "trunc_end_end_quantile", 0.999))
    q_end = float(np.clip(q_end, 0.90, 1.0))

    def pick_index(dist_u: float) -> int:
        idx = int(np.searchsorted(s, float(dist_u), side="left"))
        return int(np.clip(idx, 0, n - 1))

    i1 = pick_index(in1_u)
    i2 = pick_index(in2_u)
    if i1 >= n - 1:
        i1 = max(0, n - 2)
    if i2 <= i1:
        i2 = min(n - 1, i1 + 1)

    p1 = c0[i1]
    p2 = c0[i2]
    t_in = p2 - p1
    nt = float(np.linalg.norm(t_in))
    if nt < 1e-12:
        # fallback: 用更近的点
        j2 = min(n - 1, i1 + 1)
        t_in = c0[j2] - p1
        nt = float(np.linalg.norm(t_in))
    if nt < 1e-12:
        return c, stats
    t_in = t_in / (nt + 1e-12)  # 指向“向内”
    t_out = -t_in  # 指向“向外/截断端”

    # 用“靠近轴线”的点做外推，抑制噪声点把端点拉长
    rel = pts - p1
    axial_out = rel @ t_out
    radial = rel - (rel @ t_in)[:, None] * t_in[None, :]
    rr = np.linalg.norm(radial, axis=1)
    if pin_width_hint is not None and float(pin_width_hint) > 0:
        r_max = float(getattr(config, "trunc_end_axis_radius_factor", 0.7)) * float(pin_width_hint)
    else:
        # fallback: 用半径的高分位作为尺度
        r_max = float(np.quantile(rr, 0.90))
    r_floor = float(mm_to_unit(0.6, unit_scale))
    r_max = float(max(r_max, r_floor))

    mask = rr <= r_max
    if int(np.sum(mask)) < 50:
        mask = np.ones(len(pts), dtype=bool)

    # 端点外推：取轴向投影高分位，尽量贴近点云长度（避免缩短）
    s_end = float(np.quantile(axial_out[mask], q_end))
    # 防止比当前端点更“向内”导致更短：至少不小于当前端点的轴向位置
    s_end = max(s_end, float((c0[0] - p1) @ t_out))

    # 防止噪声把端点拉太远：限制最大外推量
    max_extend_u = float(mm_to_unit(float(getattr(config, "trunc_end_max_extend_mm", 5.0)), unit_scale))
    cur_s0 = float((c0[0] - p1) @ t_out)
    s_end = min(s_end, cur_s0 + max_extend_u)

    end_new = p1 + s_end * t_out

    # Low-DOF post-fix on a short endpoint segment (rigid + blend)
    out = c0.copy()
    out[0] = end_new
    s_from_end = s  # s[0]=0 表示端点处
    rigid_u = max(rigid_u, 0.0)
    blend_u = max(blend_u, 0.0)
    total_u = rigid_u + blend_u

    mode = str(getattr(config, "trunc_end_fix_mode", "axial_translate") or "axial_translate").strip().lower()
    if mode not in ("axial_translate", "project_to_axis"):
        mode = "axial_translate"

    delta_end = end_new - c0[0]  # by construction this is mostly axial (t_out) with non-negative magnitude

    for j in range(1, n):
        dist_u = float(s_from_end[j])
        if dist_u > total_u + 1e-12:
            break

        if dist_u <= rigid_u + 1e-12 or blend_u <= 1e-12:
            w = 1.0
        else:
            # 线性衰减：rigid->blend->free
            w = float(np.clip(1.0 - (dist_u - rigid_u) / (blend_u + 1e-12), 0.0, 1.0))

        if mode == "project_to_axis":
            # High-risk (legacy): project to axis (kills perp). Keep as an option for rollback/debug only.
            alpha = float((out[j] - end_new) @ t_in)
            if alpha < 0.0:
                alpha = 0.0
            proj = end_new + alpha * t_in
            out[j] = w * proj + (1.0 - w) * out[j]
        else:
            # P0 safer: axial-only translation (keeps perp/lateral offset).
            out[j] = out[j] + w * delta_end

    # Safety gate (IMPORTANT):
    # On curved needles, the "project-to-axis" step can cut corners and push the curve outside the point cloud.
    # If the local endpoint region becomes obviously worse after this fix, roll it back automatically.
    try:
        pts_tree = cKDTree(pts)

        # P0 primary gate: slice-based endpoint risk (coverage/delta/radial_var), robust for surface-only points.
        if bool(getattr(config, "endpoint_risk_gate_enabled", True)):
            window_u = float(
                max(
                    total_u,
                    mm_to_unit(float(getattr(config, "endpoint_risk_window_mm", 3.0)), unit_scale),
                )
            )
            before_sum = _endpoint_region_slice_summary(
                pts, c0, config, pin_width_hint=pin_width_hint, tree=pts_tree, window_u=window_u
            )
            after_sum = _endpoint_region_slice_summary(
                pts, out, config, pin_width_hint=pin_width_hint, tree=pts_tree, window_u=window_u
            )

            if _endpoint_risk_worse(before_sum, after_sum, config):
                stats["rolled_back"] = 1.0
                stats["applied"] = 0.0
                stats["extend_mm"] = 0.0
                print(
                    "[WARN] trunc_end_fix rolled back by endpoint risk gate: "
                    f"delta_mm {float(before_sum.get('max_delta_mm', 0.0)):.3f}->{float(after_sum.get('max_delta_mm', 0.0)):.3f}, "
                    f"cov {float(before_sum.get('min_coverage', 0.0)):.2f}->{float(after_sum.get('min_coverage', 0.0)):.2f}, "
                    f"n {int(before_sum.get('min_n_slab', 0.0))}->{int(after_sum.get('min_n_slab', 0.0))}, "
                    f"rv {float(before_sum.get('max_radial_var', 0.0)):.2f}->{float(after_sum.get('max_radial_var', 0.0)):.2f}, "
                    f"clip {float(before_sum.get('clipped_rate', 0.0)):.2f}->{float(after_sum.get('clipped_rate', 0.0)):.2f}"
                )
                return np.asarray(curve, dtype=float), stats

        # Evaluate only the affected region (rigid + blend). Use the same arclen reference `s` (pre-fix) to pick
        # the prefix indices; this matches exactly which points were eligible for projection in the loop above.
        j_eval = int(pick_index(total_u) + 1)
        j_eval = int(np.clip(j_eval, 2, n))

        def _dist_stats_mm(curve_seg_u: np.ndarray) -> Dict[str, float]:
            if len(curve_seg_u) <= 0:
                return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
            d_u, _ = pts_tree.query(curve_seg_u)
            d_mm = np.asarray(d_u, dtype=float) / (unit_scale + 1e-12)
            return {
                "p50": float(np.quantile(d_mm, 0.50)),
                "p90": float(np.quantile(d_mm, 0.90)),
                "p99": float(np.quantile(d_mm, 0.99)),
                "max": float(np.max(d_mm)),
            }

        before_stats = _dist_stats_mm(c0[:j_eval])
        after_stats = _dist_stats_mm(out[:j_eval])

        # Use thickness-derived radius if possible. This is consistent with evaluate_centerline_distance().
        typical_radius_mm: Optional[float] = None
        if pin_thickness_hint is not None and float(pin_thickness_hint) > 0:
            typical_radius_mm = float(0.5 * float(pin_thickness_hint) / (unit_scale + 1e-12))
        elif pin_width_hint is not None and float(pin_width_hint) > 0:
            # Fallback: width is usually larger than thickness, so use a conservative fraction.
            typical_radius_mm = float(0.20 * float(pin_width_hint) / (unit_scale + 1e-12))

        rollback = False
        if typical_radius_mm is not None and typical_radius_mm > 1e-9:
            far_ratio = 1.3
            far_th_mm = float(far_ratio * typical_radius_mm)

            good_before = float(before_stats.get("p99", 0.0)) <= far_th_mm + 1e-6
            bad_after = float(after_stats.get("p99", 0.0)) > far_th_mm + 1e-6
            # Roll back only when the fix INTRODUCES a new "far/outside" condition.
            rollback = bool(good_before and bad_after)
        else:
            # Absolute fallback (rare): roll back if it becomes much worse.
            abs_jump_mm = max(0.20, 1.5 * float(getattr(config, "nerf_noise_mm", 0.3)))
            rollback = bool(
                float(after_stats.get("max", 0.0)) > float(before_stats.get("max", 0.0)) + abs_jump_mm
                and float(after_stats.get("max", 0.0)) > 1.3 * float(before_stats.get("max", 0.0))
            )

        if rollback:
            stats["rolled_back"] = 1.0
            # Leave `applied=0` so downstream logic doesn't treat this as a committed fix.
            stats["applied"] = 0.0
            stats["extend_mm"] = 0.0
            print(
                "[WARN] trunc_end_fix rolled back: local endpoint region becomes worse "
                f"(before_p99={float(before_stats.get('p99', 0.0)):.3f}mm -> after_p99={float(after_stats.get('p99', 0.0)):.3f}mm, "
                f"before_max={float(before_stats.get('max', 0.0)):.3f}mm -> after_max={float(after_stats.get('max', 0.0)):.3f}mm)"
            )
            return np.asarray(curve, dtype=float), stats
    except Exception:
        # Never let the safety gate block the main pipeline.
        pass

    stats["applied"] = 1.0
    stats["extend_mm"] = float(np.linalg.norm((end_new - c0[0])) / (unit_scale + 1e-12))

    out_final = out[::-1] if reversed_view else out
    return out_final, stats


def refine_centerline_by_slabs(
    points: np.ndarray,
    initial_curve: np.ndarray,
    config: CenterlineConfig,
    pin_width_hint: Optional[float] = None,
    debug_tag: Optional[str] = None,
    segment_id: Optional[int] = None,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 5:
        return np.asarray(initial_curve, dtype=float).copy()

    curve = resample_polyline_by_step(np.asarray(initial_curve, dtype=float), config.centerline_step())
    if len(curve) < 2:
        return curve

    dbg_on = _debug_enabled(config, segment_id) and bool(debug_tag)
    dbg_dir: Optional[Path] = None
    if dbg_on:
        dbg_dir = _debug_segment_dir(config, str(debug_tag))
        # points 一次性写出
        try:
            _debug_write_ply_points(dbg_dir / "points.ply", pts, color=[0.6, 0.6, 0.6])
            _debug_write_ply_points(dbg_dir / "stage00_init_curve.ply", curve, color=[0.0, 0.0, 1.0])
        except Exception:
            pass

    tree = cKDTree(pts)
    search_r = float(config.slab_search_radius(pin_width_hint))
    half_t = float(config.slab_half_thickness())
    max_update = float(config.max_update())
    slab_ql = float(getattr(config, "slab_quantile_low", 0.10))
    slab_qh = float(getattr(config, "slab_quantile_high", 0.90))
    slab_bins = int(getattr(config, "slab_coverage_bins", 12))
    slab_min_cov = float(getattr(config, "slab_min_coverage", 0.0))
    slab_action = str(getattr(config, "slab_single_side_action", "none") or "none").strip().lower()
    slab_damp = float(getattr(config, "slab_single_side_damp", 0.25))

    endpoint_timing = str(getattr(config, "endpoint_centering_timing", "early") or "early").strip().lower()
    if endpoint_timing in ("disable", "disabled", "none", "false", "0"):
        endpoint_timing = "off"
    if endpoint_timing not in ("early", "late", "off"):
        endpoint_timing = "late"

    # 端点在“无端面点”的情况下很容易在迭代/平滑中逐步向内部收缩；
    # 先用 slab 做端点横向居中（不依赖 90° 先验），再在迭代中锚定端点以抑制 shrinking。
    # NOTE: legacy "early" endpoint centering is preserved for rollback/debug via endpoint_centering_timing="early".
    if bool(getattr(config, "anchor_endpoints", False)) and len(curve) >= 2 and endpoint_timing == "early":
        # 用整段点云的主方向作为“参考轴”，避免端点局部 PCA 因端面/侧壁对称性选错方向。
        t_global: Optional[np.ndarray]
        slender: bool = False
        try:
            A0 = pts - pts.mean(axis=0)
            _, s0, vh0 = np.linalg.svd(A0, full_matrices=False)
            tg = np.asarray(vh0[0], dtype=float)
            t_global = tg / (float(np.linalg.norm(tg)) + 1e-12)
            # slender_ratio 用于判断“整体是否明显细长”，细长段用全局轴更可靠
            if len(s0) >= 2 and float(s0[1]) > 1e-12:
                slender = float(s0[0] / s0[1]) >= 2.0
        except Exception:
            t_global = None
            slender = False

        # Endpoint centering is iterative. If we query neighborhoods around the UPDATED endpoints every
        # iteration, the neighborhood itself can drift (especially near caps or sparse tips), which may
        # amplify noise and hurt endpoint stability. Use a fixed neighborhood anchored at the initial
        # endpoint positions when possible.
        ep_cand_fixed: Dict[int, Optional[List[int]]] = {}
        for ep in (0, len(curve) - 1):
            try:
                c_anchor = curve[int(ep)]
                cand0 = tree.query_ball_point(c_anchor, r=search_r)
                if len(cand0) >= max(20, int(config.min_slab_points)):
                    ep_cand_fixed[int(ep)] = list(cand0)
                else:
                    ep_cand_fixed[int(ep)] = None
            except Exception:
                ep_cand_fixed[int(ep)] = None

        def pca_axis(center: np.ndarray, cand_idx: Optional[List[int]] = None) -> Optional[np.ndarray]:
            if slender and t_global is not None:
                # 细长直段：直接用全局轴，避免端面导致局部 PCA 轴倾斜而产生“轴向回缩”
                return t_global

            cand = cand_idx if cand_idx is not None else tree.query_ball_point(center, r=search_r)
            if len(cand) < max(20, int(config.min_slab_points)):
                return None
            P = pts[cand]
            mu = P.mean(axis=0)
            A = P - mu
            try:
                _, _, vh = np.linalg.svd(A, full_matrices=False)
            except np.linalg.LinAlgError:
                return None

            # 选择与全局主方向最一致的轴，提升“轴向”稳定性
            if t_global is not None and vh.shape[0] >= 3:
                dots = np.abs(vh @ t_global)
                axis = np.asarray(vh[int(np.argmax(dots))], dtype=float)
            else:
                axis = np.asarray(vh[0], dtype=float)
            nn = float(np.linalg.norm(axis))
            if nn < 1e-12:
                return None
            return axis / nn

        for _ in range(int(max(0, config.endpoint_centering_iterations))):
            tangents0 = compute_tangents(curve)
            for ep in (0, len(curve) - 1):
                c = curve[ep]
                cand_ep = ep_cand_fixed.get(int(ep))
                t = pca_axis(c, cand_ep)
                if t is None:
                    t = tangents0[ep]
                u, v = make_plane_basis(t)

                cand = cand_ep if cand_ep is not None else tree.query_ball_point(c, r=search_r)
                if len(cand) < int(config.min_slab_points):
                    continue

                local = pts[cand] - c
                axial = local @ t
                mask = np.abs(axial) <= half_t
                if int(np.sum(mask)) < int(config.min_slab_points):
                    continue

                local = local[mask]
                coords_u = local @ u
                coords_v = local @ v
                local_damp = 1.0
                if slab_action != "none" and slab_min_cov > 0:
                    cov = slab_angular_coverage(coords_u, coords_v, bins=slab_bins)
                    if cov < slab_min_cov:
                        if slab_action == "skip":
                            continue
                        if slab_action == "damp":
                            local_damp = float(np.clip(slab_damp, 0.0, 1.0))
                du, dv = robust_midrange_2d(coords_u, coords_v, q_low=slab_ql, q_high=slab_qh)

                delta = du * u + dv * v
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > max_update:
                    delta = delta / (delta_norm + 1e-12) * max_update
                curve[ep] = c + float(config.refine_damping) * local_damp * delta

        if dbg_dir is not None:
            try:
                _debug_write_ply_points(dbg_dir / "stage01_after_endpoint_centering.ply", curve, color=[0.0, 0.5, 1.0])
            except Exception:
                pass

    for _ in range(int(config.refine_iterations)):
        it = int(_)
        tangents = compute_tangents(curve)
        curve_new = curve.copy()
        prev_u: Optional[np.ndarray] = None
        max_delta = 0.0

        for i, (c, t) in enumerate(zip(curve, tangents)):
            if bool(getattr(config, "anchor_endpoints", False)):
                # 端点只做“预居中”，后续迭代中固定端点，避免在噪声/绕圈切向下发生长度回缩
                if i == 0 or i == len(curve) - 1:
                    continue

            u, v = make_plane_basis(t, prev_u=prev_u)
            prev_u = u

            cand = tree.query_ball_point(c, r=search_r)
            if len(cand) < int(config.min_slab_points):
                continue

            local = pts[cand] - c
            axial = local @ t
            mask = np.abs(axial) <= half_t
            if int(np.sum(mask)) < int(config.min_slab_points):
                continue

            local = local[mask]
            coords_u = local @ u
            coords_v = local @ v
            local_damp = 1.0
            if slab_action != "none" and slab_min_cov > 0:
                cov = slab_angular_coverage(coords_u, coords_v, bins=slab_bins)
                if cov < slab_min_cov:
                    if slab_action == "skip":
                        continue
                    if slab_action == "damp":
                        local_damp = float(np.clip(slab_damp, 0.0, 1.0))
            du, dv = robust_midrange_2d(coords_u, coords_v, q_low=slab_ql, q_high=slab_qh)

            delta = du * u + dv * v
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_update:
                delta = delta / (delta_norm + 1e-12) * max_update
                delta_norm = max_update
            max_delta = max(max_delta, delta_norm)

            curve_new[i] = c + float(config.refine_damping) * local_damp * delta

        if dbg_dir is not None and (bool(getattr(config, "debug_dump_every_iter", False)) or it == 0):
            try:
                _debug_write_ply_points(dbg_dir / f"stage02_iter{it:02d}_curve_new.ply", curve_new, color=[1.0, 0.5, 0.0])
            except Exception:
                pass

        curve_sm = smooth_curve_curvature_adaptive(curve_new, config)
        if bool(getattr(config, "anchor_endpoints", False)) and len(curve_sm) >= 2:
            # 关键：端点允许做横向 slab 居中更新，但不允许在平滑/重采样中“回缩”。
            # 因此这里锚定到本轮更新后的端点（curve_new），而不是上一轮的 curve。
            curve_sm[0] = curve_new[0]
            curve_sm[-1] = curve_new[-1]

        if dbg_dir is not None and (bool(getattr(config, "debug_dump_every_iter", False)) or it == 0):
            try:
                _debug_write_ply_points(dbg_dir / f"stage03_iter{it:02d}_curve_sm.ply", curve_sm, color=[1.0, 0.0, 1.0])
            except Exception:
                pass

        curve = resample_polyline_by_step(curve_sm, config.centerline_step())
        if bool(getattr(config, "anchor_endpoints", False)) and len(curve) >= 2:
            # 再次强制端点精确，抑制重采样造成的端点漂移
            curve[0] = curve_new[0]
            curve[-1] = curve_new[-1]

        if dbg_dir is not None and (bool(getattr(config, "debug_dump_every_iter", False)) or it == 0):
            try:
                _debug_write_ply_points(dbg_dir / f"stage04_iter{it:02d}_resampled.ply", curve, color=[0.0, 1.0, 0.0])
            except Exception:
                pass

        # 收敛判据：整体更新很小
        if max_delta < 0.05 * config.centerline_step():
            break

    # Phase B/C (new default): refine body first, then (optionally) center endpoints late with a coverage gate,
    # then do a short post-refine to blend the endpoint update into the body. This reduces the risk of an
    # early endpoint update drifting to a wrong lateral position on single-sided NeRF cross-sections.
    if bool(getattr(config, "anchor_endpoints", False)) and endpoint_timing == "late" and len(curve) >= 2:
        if dbg_dir is not None:
            try:
                _debug_write_ply_points(
                    dbg_dir / "stage05_body_refined_before_endpoint_centering.ply", curve, color=[0.2, 1.0, 0.2]
                )
            except Exception:
                pass

        ep_iters = int(max(0, int(getattr(config, "endpoint_centering_iterations", 0) or 0)))
        if ep_iters > 0:
            ep_min_cov = float(getattr(config, "endpoint_centering_min_coverage", 0.0) or 0.0)
            ep_action = str(getattr(config, "endpoint_centering_low_coverage_action", "none") or "none").strip().lower()
            ep_damp = float(getattr(config, "endpoint_centering_low_coverage_damp", 0.25))

            # Fixed neighborhoods anchored at the (pre-centering) endpoint positions for stability.
            ep_cand_fixed: Dict[int, Optional[List[int]]] = {}
            for ep in (0, len(curve) - 1):
                try:
                    c_anchor = curve[int(ep)]
                    cand0 = tree.query_ball_point(c_anchor, r=search_r)
                    if len(cand0) >= max(20, int(config.min_slab_points)):
                        ep_cand_fixed[int(ep)] = list(cand0)
                    else:
                        ep_cand_fixed[int(ep)] = None
                except Exception:
                    ep_cand_fixed[int(ep)] = None

            def endpoint_body_tangent(ep_idx: int) -> np.ndarray:
                # Use the stabilized body direction (not endpoint local PCA) to define the slab plane.
                n = int(len(curve))
                if n < 2:
                    return np.array([0.0, 0.0, 1.0], dtype=float)
                if int(ep_idx) == 0:
                    if n >= 4:
                        t_ep = curve[3] - curve[1]
                    elif n >= 3:
                        t_ep = curve[2] - curve[1]
                    else:
                        t_ep = curve[1] - curve[0]
                else:
                    if n >= 4:
                        t_ep = curve[-2] - curve[-4]
                    elif n >= 3:
                        t_ep = curve[-2] - curve[-3]
                    else:
                        t_ep = curve[-1] - curve[-2]

                nt = float(np.linalg.norm(t_ep))
                if nt < 1e-12:
                    return np.array([0.0, 0.0, 1.0], dtype=float)
                return np.asarray(t_ep, dtype=float) / nt

            for _ in range(ep_iters):
                for ep in (0, len(curve) - 1):
                    c = curve[ep]
                    t_ep = endpoint_body_tangent(int(ep))
                    u, v = make_plane_basis(t_ep)

                    cand = ep_cand_fixed.get(int(ep))
                    if cand is None:
                        cand = tree.query_ball_point(c, r=search_r)
                    if len(cand) < int(config.min_slab_points):
                        continue

                    local = pts[cand] - c
                    axial = local @ t_ep
                    mask = np.abs(axial) <= half_t
                    if int(np.sum(mask)) < int(config.min_slab_points):
                        continue

                    local = local[mask]
                    coords_u = local @ u
                    coords_v = local @ v

                    local_damp = 1.0
                    if ep_action != "none" and ep_min_cov > 0:
                        cov = slab_angular_coverage(coords_u, coords_v, bins=slab_bins)
                        if cov < ep_min_cov:
                            if ep_action == "skip":
                                continue
                            if ep_action == "damp":
                                local_damp = float(np.clip(ep_damp, 0.0, 1.0))

                    du, dv = robust_midrange_2d(coords_u, coords_v, q_low=slab_ql, q_high=slab_qh)
                    delta = du * u + dv * v
                    delta_norm = float(np.linalg.norm(delta))
                    if delta_norm > max_update:
                        delta = delta / (delta_norm + 1e-12) * max_update
                    curve[ep] = c + float(config.refine_damping) * local_damp * delta

            if dbg_dir is not None:
                try:
                    _debug_write_ply_points(dbg_dir / "stage06_after_endpoint_centering_late.ply", curve, color=[0.0, 0.5, 1.0])
                except Exception:
                    pass

            post_iters = int(max(0, int(getattr(config, "post_endpoint_refine_iterations", 0) or 0)))
            if post_iters > 0:
                for _post in range(post_iters):
                    tangents = compute_tangents(curve)
                    curve_new = curve.copy()
                    prev_u: Optional[np.ndarray] = None
                    max_delta = 0.0

                    for i, (c, t) in enumerate(zip(curve, tangents)):
                        # Endpoints remain fixed in the post-refine phase.
                        if i == 0 or i == len(curve) - 1:
                            continue

                        u, v = make_plane_basis(t, prev_u=prev_u)
                        prev_u = u

                        cand = tree.query_ball_point(c, r=search_r)
                        if len(cand) < int(config.min_slab_points):
                            continue

                        local = pts[cand] - c
                        axial = local @ t
                        mask = np.abs(axial) <= half_t
                        if int(np.sum(mask)) < int(config.min_slab_points):
                            continue

                        local = local[mask]
                        coords_u = local @ u
                        coords_v = local @ v
                        local_damp = 1.0
                        if slab_action != "none" and slab_min_cov > 0:
                            cov = slab_angular_coverage(coords_u, coords_v, bins=slab_bins)
                            if cov < slab_min_cov:
                                if slab_action == "skip":
                                    continue
                                if slab_action == "damp":
                                    local_damp = float(np.clip(slab_damp, 0.0, 1.0))
                        du, dv = robust_midrange_2d(coords_u, coords_v, q_low=slab_ql, q_high=slab_qh)

                        delta = du * u + dv * v
                        delta_norm = float(np.linalg.norm(delta))
                        if delta_norm > max_update:
                            delta = delta / (delta_norm + 1e-12) * max_update
                            delta_norm = max_update
                        max_delta = max(max_delta, delta_norm)
                        curve_new[i] = c + float(config.refine_damping) * local_damp * delta

                    curve_sm = smooth_curve_curvature_adaptive(curve_new, config)
                    if len(curve_sm) >= 2:
                        curve_sm[0] = curve_new[0]
                        curve_sm[-1] = curve_new[-1]

                    curve = resample_polyline_by_step(curve_sm, config.centerline_step())
                    if len(curve) >= 2:
                        curve[0] = curve_new[0]
                        curve[-1] = curve_new[-1]

                    if max_delta < 0.05 * config.centerline_step():
                        break

                if dbg_dir is not None:
                    try:
                        _debug_write_ply_points(
                            dbg_dir / "stage07_post_refined_after_endpoint_centering.ply", curve, color=[0.2, 1.0, 0.2]
                        )
                    except Exception:
                        pass

            # Final endpoint re-centering with a rigid+blend translation (to avoid endpoint-only kinks).
            # This runs AFTER post-refine so endpoint centering remains valid for the final curve.
            try:
                rigid_mm = float(getattr(config, "endpoint_centering_rigid_mm", 0.0) or 0.0)
                blend_mm = float(getattr(config, "endpoint_centering_blend_mm", 0.0) or 0.0)

                unit_scale = float(getattr(config, "unit_scale", 0.0) or 0.0)
                rigid_u = float(mm_to_unit(max(0.0, rigid_mm), unit_scale))
                blend_u = float(mm_to_unit(max(0.0, blend_mm), unit_scale))
                total_u = float(rigid_u + blend_u)

                if total_u > 1e-12 and len(curve) >= 2:
                    s = _cum_arclen(curve)
                    try:
                        cand0 = tree.query_ball_point(curve[0], r=search_r)
                        cand1 = tree.query_ball_point(curve[-1], r=search_r)
                    except Exception:
                        cand0 = None
                        cand1 = None

                    shifts: Dict[int, np.ndarray] = {}
                    for ep, cand in ((0, cand0), (len(curve) - 1, cand1)):
                        c = curve[ep]
                        t_ep = endpoint_body_tangent(int(ep))
                        u, v = make_plane_basis(t_ep)

                        cand_use = cand if cand is not None else tree.query_ball_point(c, r=search_r)
                        if len(cand_use) < int(config.min_slab_points):
                            continue

                        local = pts[cand_use] - c
                        axial = local @ t_ep
                        mask = np.abs(axial) <= half_t
                        if int(np.sum(mask)) < int(config.min_slab_points):
                            continue

                        local = local[mask]
                        coords_u = local @ u
                        coords_v = local @ v

                        local_damp = 1.0
                        if ep_action != "none" and ep_min_cov > 0:
                            cov = slab_angular_coverage(coords_u, coords_v, bins=slab_bins)
                            if cov < ep_min_cov:
                                if ep_action == "skip":
                                    local_damp = 0.0
                                elif ep_action == "damp":
                                    local_damp = float(np.clip(ep_damp, 0.0, 1.0))

                        du, dv = robust_midrange_2d(coords_u, coords_v, q_low=slab_ql, q_high=slab_qh)
                        delta = du * u + dv * v
                        delta_norm = float(np.linalg.norm(delta))
                        if delta_norm > max_update:
                            delta = delta / (delta_norm + 1e-12) * max_update

                        shift_vec = float(config.refine_damping) * local_damp * delta
                        shifts[int(ep)] = np.asarray(shift_vec, dtype=float)

                    if shifts:
                        curve_blend = np.asarray(curve, dtype=float).copy()
                        for ep, shift_vec in shifts.items():
                            if float(np.linalg.norm(shift_vec)) < 1e-12:
                                continue
                            dist = s if int(ep) == 0 else (float(s[-1]) - s)
                            for j in range(len(curve_blend)):
                                du = float(dist[j])
                                if du > total_u + 1e-12:
                                    continue
                                if du <= rigid_u + 1e-12 or blend_u <= 1e-12:
                                    w = 1.0
                                else:
                                    w = float(np.clip(1.0 - (du - rigid_u) / (blend_u + 1e-12), 0.0, 1.0))
                                curve_blend[j] = curve_blend[j] + w * shift_vec
                        curve = curve_blend

                        if dbg_dir is not None:
                            try:
                                _debug_write_ply_points(
                                    dbg_dir / "stage08_after_endpoint_recenter_blend.ply",
                                    curve,
                                    color=[0.2, 0.8, 0.8],
                                )
                            except Exception:
                                pass
            except Exception:
                # Never let optional endpoint blend centering block the pipeline.
                pass

    if dbg_dir is not None:
        try:
            _debug_write_ply_points(dbg_dir / "stage09_refined_before_trunc_fix.ply", curve, color=[0.2, 1.0, 0.2])
        except Exception:
            pass

    return curve


def tangent_toward_mid_pca(
    curve: np.ndarray,
    idx: int,
    mid_idx: int,
    *,
    s: Optional[np.ndarray] = None,
    inward_u: float = 0.0,
    max_points: int = 12,
    min_points: int = 4,
) -> np.ndarray:
    """
    One-sided tangent estimator: use a curve segment from the current point toward the middle, then take its PCA axis.

    This is intentionally designed to be more stable near endpoints than local finite-difference tangents on
    NeRF-like point clouds (often single-sided caps -> tangent/plane feedback loop).
    """
    c = np.asarray(curve, dtype=float)
    n = int(len(c))
    if n < 2:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    i = int(np.clip(int(idx), 0, n - 1))
    m = int(np.clip(int(mid_idx), 0, n - 1))

    if i == m:
        # Fallback to a symmetric chord around the middle.
        j0 = max(0, m - 1)
        j1 = min(n - 1, m + 1)
        t = c[j1] - c[j0]
        nt = float(np.linalg.norm(t))
        if nt < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return np.asarray(t / nt, dtype=float)

    if s is None:
        s = _cum_arclen(c)
    ss = np.asarray(s, dtype=float)
    if ss.ndim != 1 or int(len(ss)) != n:
        ss = _cum_arclen(c)

    # Choose a segment [i..j] (or [j..i]) that goes from the current point toward the middle.
    max_points = int(max(2, max_points))
    min_points = int(max(2, min_points))
    inward_u = float(max(0.0, inward_u))

    if i < m:
        # toward middle: +index
        target = float(ss[i] + inward_u)
        j = int(np.searchsorted(ss, target, side="left"))
        j = min(j, m)
        j = max(j, min(m, i + (min_points - 1)))
        j = min(j, i + (max_points - 1), m)
        seg = c[i : j + 1]
        dir_in = c[j] - c[i]
    else:
        # toward middle: -index
        target = float(ss[i] - inward_u)
        j = int(np.searchsorted(ss, target, side="right") - 1)
        j = max(j, m)
        j = min(j, max(m, i - (min_points - 1)))
        j = max(j, i - (max_points - 1), m)
        seg = c[j : i + 1]
        dir_in = c[j] - c[i]

    nd = float(np.linalg.norm(dir_in))
    if nd < 1e-12:
        # fallback: local tangent
        t = c[min(n - 1, i + 1)] - c[max(0, i - 1)]
        nt = float(np.linalg.norm(t))
        if nt < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return np.asarray(t / nt, dtype=float)

    axis = dir_in / (nd + 1e-12)
    if int(len(seg)) >= min_points:
        try:
            A = seg - np.mean(seg, axis=0)
            _, _, vh = np.linalg.svd(A, full_matrices=False)
            if vh.shape[0] >= 1:
                axis = np.asarray(vh[0], dtype=float)
                na = float(np.linalg.norm(axis))
                if na > 1e-12:
                    axis = axis / na
        except Exception:
            axis = dir_in / (nd + 1e-12)

    # Make it point inward (toward the middle) for consistency.
    if float(np.dot(axis, dir_in)) < 0.0:
        axis = -axis
    return np.asarray(axis, dtype=float)


def pullback_truncated_end_center_out(
    points: np.ndarray,
    curve: np.ndarray,
    base_end_idx_local: int,
    config: CenterlineConfig,
    *,
    pin_width_hint: Optional[float] = None,
    debug_suffix: Optional[str] = None,
    debug_tag: Optional[str] = None,
    segment_id: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Lateral "pullback" for the truncated end by a center-out slab sweep.

    - Only updates points on the truncated-end side (base_end_idx_local is 0 or -1).
    - Keeps the middle region as a prior (no updates within trunc_end_pullback_mid_keep_mm around the middle).
    - Uses a one-sided tangent (PCA of a segment toward the middle) to define slab planes, mitigating the
      endpoint feedback loop (local tangent bias -> slab plane bias -> update bias).
    """
    pts = np.asarray(points, dtype=float)
    c0 = np.asarray(curve, dtype=float).copy()
    n = int(len(c0))
    if len(pts) < 10 or n < 4:
        return c0, {"applied": 0.0, "iters": 0.0, "moved_mm": 0.0, "end_shift_mm": 0.0}

    end_idx = 0 if int(base_end_idx_local) == 0 else (n - 1)
    mid = int(n // 2)

    iters = int(max(0, int(getattr(config, "trunc_end_pullback_iters", 0) or 0)))
    if iters <= 0:
        return c0, {"applied": 0.0, "iters": 0.0, "moved_mm": 0.0, "end_shift_mm": 0.0}

    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    keep_u = float(mm_to_unit(float(getattr(config, "trunc_end_pullback_mid_keep_mm", 0.0)), unit_scale))
    inward_u = float(mm_to_unit(float(getattr(config, "trunc_end_pullback_tangent_inward_mm", 0.0)), unit_scale))
    max_pts = int(getattr(config, "trunc_end_pullback_tangent_max_points", 12))
    min_pts = int(getattr(config, "trunc_end_pullback_tangent_min_points", 4))
    step_u = float(config.centerline_step())

    # Anti-fold guards: reject steps that introduce sharp local foldbacks in a tiny arc-length window.
    fold_guard_enabled = bool(getattr(config, "trunc_end_pullback_fold_guard_enabled", True))
    fold_hard_angle_deg = float(
        max(60.0, float(getattr(config, "trunc_end_pullback_fold_hard_angle_deg", 150.0) or 150.0))
    )
    fold_angle_jump_deg = float(
        max(1.0, float(getattr(config, "trunc_end_pullback_fold_angle_jump_deg", 35.0) or 35.0))
    )
    fold_short_edge_ratio = float(
        np.clip(float(getattr(config, "trunc_end_pullback_fold_short_edge_ratio", 0.45) or 0.45), 0.05, 2.0)
    )
    fold_short_edge_angle_deg = float(
        max(60.0, float(getattr(config, "trunc_end_pullback_fold_short_edge_angle_deg", 95.0) or 95.0))
    )

    lowcov_force_skip_enabled = bool(getattr(config, "trunc_end_pullback_lowcov_force_skip_enabled", True))
    lowcov_force_skip_u = float(
        mm_to_unit(float(getattr(config, "trunc_end_pullback_lowcov_force_skip_mm", 2.0) or 2.0), unit_scale)
    )

    tree = cKDTree(pts)
    search_r = float(config.slab_search_radius(pin_width_hint))
    half_t = float(config.slab_half_thickness())
    max_update = float(config.max_update())
    slab_ql = float(getattr(config, "slab_quantile_low", 0.10))
    slab_qh = float(getattr(config, "slab_quantile_high", 0.90))
    slab_bins = int(getattr(config, "slab_coverage_bins", 12))
    slab_min_cov = float(getattr(config, "slab_min_coverage", 0.0))
    slab_action = str(getattr(config, "slab_single_side_action", "none") or "none").strip().lower()
    slab_damp = float(getattr(config, "slab_single_side_damp", 0.25))

    dbg_on = _debug_enabled(config, segment_id) and bool(debug_tag)
    dbg_dir: Optional[Path] = None
    if dbg_on:
        try:
            dbg_dir = _debug_segment_dir(config, str(debug_tag))
        except Exception:
            dbg_dir = None
    suffix = ""
    if debug_suffix:
        suffix = f"_{str(debug_suffix).strip()}"

    end_before = np.asarray(c0[end_idx], dtype=float).copy()
    moved_sum = 0.0
    fold_reject_count = 0
    lowcov_force_skip_count = 0
    fold_max_angle_seen = 0.0
    fold_min_edge_seen = float("inf")

    def _local_turn_metrics(curve_arr: np.ndarray, idx: int) -> Tuple[float, float]:
        max_ang = 0.0
        min_edge = float("inf")

        # Edges adjacent to idx.
        for j in (idx - 1, idx):
            if 0 <= j < n - 1:
                e = float(np.linalg.norm(curve_arr[j + 1] - curve_arr[j]))
                if np.isfinite(e):
                    min_edge = min(min_edge, e)

        # Angles affected by changing point idx.
        for j in (idx - 1, idx):
            if 0 < j < n - 1:
                a = np.asarray(curve_arr[j] - curve_arr[j - 1], dtype=float)
                b = np.asarray(curve_arr[j + 1] - curve_arr[j], dtype=float)
                ang = float(_angle_deg(a, b))
                if np.isfinite(ang):
                    max_ang = max(max_ang, ang)

        if not np.isfinite(min_edge):
            min_edge = 0.0
        return float(max_ang), float(min_edge)

    # Sweep indices from the middle toward the truncated end.
    if end_idx == 0:
        sweep = list(range(mid - 1, -1, -1))
    else:
        sweep = list(range(mid + 1, n))

    for it in range(iters):
        s = _cum_arclen(c0)
        if len(s) != n:
            s = _cum_arclen(c0)
        prev_u: Optional[np.ndarray] = None
        for i in sweep:
            if keep_u > 1e-12 and abs(float(s[i] - s[mid])) <= keep_u:
                continue

            t = tangent_toward_mid_pca(
                c0,
                int(i),
                int(mid),
                s=s,
                inward_u=float(inward_u),
                max_points=int(max_pts),
                min_points=int(min_pts),
            )
            u, v = make_plane_basis(t, prev_u=prev_u)
            prev_u = np.asarray(u, dtype=float)

            cand = tree.query_ball_point(c0[i], r=search_r)
            if len(cand) < int(getattr(config, "min_slab_points", 30)):
                continue

            local = pts[cand] - c0[i]
            axial = local @ t
            mask = np.abs(axial) <= half_t
            if int(np.sum(mask)) < int(getattr(config, "min_slab_points", 30)):
                continue

            local = local[mask]
            coords_u = local @ u
            coords_v = local @ v
            local_damp = 1.0
            if slab_action != "none" and slab_min_cov > 0.0:
                cov = float(slab_angular_coverage(coords_u, coords_v, bins=slab_bins))
                if cov < slab_min_cov:
                    dist_to_end_u = float(s[i]) if end_idx == 0 else float(s[-1] - s[i])
                    if lowcov_force_skip_enabled and dist_to_end_u <= lowcov_force_skip_u + 1e-12:
                        lowcov_force_skip_count += 1
                        continue
                    if slab_action == "skip":
                        continue
                    if slab_action == "damp":
                        local_damp = float(np.clip(slab_damp, 0.0, 1.0))

            du, dv = robust_midrange_2d(coords_u, coords_v, q_low=slab_ql, q_high=slab_qh)
            delta = du * u + dv * v  # lateral-only by construction
            dn = float(np.linalg.norm(delta))
            if dn < 1e-12:
                continue
            if dn > max_update:
                delta = delta / (dn + 1e-12) * max_update

            step = float(config.refine_damping) * float(local_damp) * delta
            if fold_guard_enabled and (0 < i < n - 1):
                pre_ang, pre_min_edge = _local_turn_metrics(c0, int(i))
                old_pi = np.asarray(c0[i], dtype=float).copy()
                cand_pi = np.asarray(old_pi + step, dtype=float)

                c0[i] = cand_pi
                post_ang, post_min_edge = _local_turn_metrics(c0, int(i))
                c0[i] = old_pi

                fold_max_angle_seen = max(fold_max_angle_seen, float(pre_ang), float(post_ang))
                fold_min_edge_seen = min(fold_min_edge_seen, float(pre_min_edge), float(post_min_edge))

                hard_turn = bool(post_ang >= fold_hard_angle_deg and post_ang > pre_ang + fold_angle_jump_deg)
                short_edge_u = float(max(1e-12, fold_short_edge_ratio * step_u))
                short_edge_fold = bool(post_min_edge <= short_edge_u and post_ang >= fold_short_edge_angle_deg)

                if hard_turn or short_edge_fold:
                    fold_reject_count += 1
                    continue

            c0[i] = c0[i] + step
            moved_sum += float(np.linalg.norm(step))

        if dbg_dir is not None and (bool(getattr(config, "debug_dump_every_iter", False)) or it == 0):
            try:
                _debug_write_ply_points(
                    dbg_dir / f"stage10_iter{it:02d}_pullback{suffix}_after.ply",
                    c0,
                    color=[0.10, 0.10, 0.10],
                )
            except Exception:
                pass

    end_after = np.asarray(c0[end_idx], dtype=float)
    end_shift_u = float(np.linalg.norm(end_after - end_before))
    moved_mm = float(moved_sum / (unit_scale + 1e-12))
    end_shift_mm = float(end_shift_u / (unit_scale + 1e-12))
    fold_min_edge_mm = float((fold_min_edge_seen if np.isfinite(fold_min_edge_seen) else 0.0) / (unit_scale + 1e-12))

    applied = 1.0 if moved_sum > 1e-12 else 0.0
    stats = {
        "applied": float(applied),
        "iters": float(iters),
        "moved_mm": float(moved_mm),
        "end_shift_mm": float(end_shift_mm),
        "fold_guard_enabled": float(1.0 if fold_guard_enabled else 0.0),
        "fold_reject_count": float(fold_reject_count),
        "fold_max_angle_deg": float(fold_max_angle_seen),
        "fold_min_edge_mm": float(fold_min_edge_mm),
        "lowcov_force_skip_count": float(lowcov_force_skip_count),
    }
    return c0, stats


def _trim_curve_by_arclen(
    curve: np.ndarray,
    *,
    trim_start_u: float,
    trim_end_u: float,
) -> Tuple[np.ndarray, float, float, int, int]:
    """
    Trim curve endpoints by arc length.

    Returns: (body_curve, trimmed_start_u, trimmed_end_u, i0, i1)
    - trimmed_* are the actual removed arc lengths (unit of the point cloud).
    """
    c = np.asarray(curve, dtype=float)
    n = int(len(c))
    trim_s = float(max(0.0, trim_start_u))
    trim_e = float(max(0.0, trim_end_u))
    if n < 3 or (trim_s <= 1e-12 and trim_e <= 1e-12):
        return c.copy(), 0.0, 0.0, 0, max(0, n - 1)

    s = _cum_arclen(c)
    total = float(s[-1]) if len(s) else 0.0
    if total <= trim_s + trim_e + 1e-12:
        # Too short to trim meaningfully.
        return c.copy(), 0.0, 0.0, 0, max(0, n - 1)

    i0 = int(np.searchsorted(s, float(trim_s), side="left"))
    i1 = int(np.searchsorted(s, float(total - trim_e), side="right") - 1)
    i0 = int(np.clip(i0, 0, n - 2))
    i1 = int(np.clip(i1, i0 + 1, n - 1))

    body = c[i0 : i1 + 1].copy()
    trimmed_start_u = float(s[i0])
    trimmed_end_u = float(total - s[i1])
    return body, trimmed_start_u, trimmed_end_u, i0, i1


def _unit_tangent_at_index(curve: np.ndarray, idx: int) -> Optional[np.ndarray]:
    c = np.asarray(curve, dtype=float)
    n = int(len(c))
    if n < 2:
        return None
    i = int(np.clip(idx, 0, n - 1))
    if i <= 0:
        v = c[1] - c[0]
    elif i >= n - 1:
        v = c[n - 1] - c[n - 2]
    else:
        v = c[i + 1] - c[i - 1]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return None
    return v / nv


def _detect_endpoint_deflection_trim(
    curve_view: np.ndarray,
    config: CenterlineConfig,
    *,
    max_trim_u: float,
) -> Dict[str, float]:
    """
    Detect endpoint-tail deflection in a curve view where endpoint is at index 0.

    Returns dict with:
    - detected: 1/0
    - trim_u, trim_mm
    - onset_idx, onset_mm
    - angle_peak_deg, lateral_peak_mm
    - run_len
    """
    c = np.asarray(curve_view, dtype=float)
    n = int(len(c))
    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    denom = float(unit_scale + 1e-12)
    out: Dict[str, float] = {
        "detected": 0.0,
        "trim_u": 0.0,
        "trim_mm": 0.0,
        "scan_mm": 0.0,
        "onset_idx": -1.0,
        "onset_mm": 0.0,
        "angle_peak_deg": 0.0,
        "lateral_peak_mm": 0.0,
        "run_len": 0.0,
    }
    trim_cap_u = float(max(0.0, max_trim_u))
    if n < 6 or trim_cap_u <= 1e-12:
        return out

    s = _cum_arclen(c)
    total = float(s[-1]) if len(s) else 0.0
    if total <= 1e-12:
        return out

    step_u = float(max(1e-12, float(config.centerline_step())))
    scan_u = float(min(trim_cap_u, 0.45 * total))
    out["scan_mm"] = float(scan_u / denom)
    if scan_u <= 1.5 * step_u:
        return out

    scan_idx = int(np.searchsorted(s, scan_u, side="right") - 1)
    scan_idx = int(np.clip(scan_idx, 2, n - 2))
    if scan_idx < 3:
        return out

    ref_span_mm = float(getattr(config, "endpoint_regrow_deflect_ref_span_mm", 2.0) or 2.0)
    ref_span_u = float(mm_to_unit(max(0.5, ref_span_mm), unit_scale))
    ref_lo_u = float(min(scan_u, max(0.0, total - 2.0 * step_u)))
    ref_hi_u = float(min(total, ref_lo_u + max(ref_span_u, 2.0 * step_u)))
    i_ref0 = int(np.searchsorted(s, ref_lo_u, side="left"))
    i_ref1 = int(np.searchsorted(s, ref_hi_u, side="right") - 1)
    i_ref0 = int(np.clip(i_ref0, 0, n - 2))
    i_ref1 = int(np.clip(i_ref1, i_ref0 + 1, n - 1))

    t_ref_vec = c[i_ref1] - c[i_ref0]
    nt_ref = float(np.linalg.norm(t_ref_vec))
    if nt_ref < 1e-12:
        t_ref = _unit_tangent_at_index(c, i_ref0)
        if t_ref is None:
            return out
    else:
        t_ref = t_ref_vec / nt_ref
    p_ref = c[i_ref0]

    angle_th = float(max(1.0, float(getattr(config, "endpoint_regrow_deflect_angle_deg", 20.0) or 20.0)))
    hard_angle_th = float(
        max(angle_th + 1.0, float(getattr(config, "endpoint_regrow_deflect_hard_angle_deg", 32.0) or 32.0))
    )
    lateral_th_mm = float(max(0.0, float(getattr(config, "endpoint_regrow_deflect_lateral_mm", 0.10) or 0.10)))
    lateral_th_u = float(mm_to_unit(lateral_th_mm, unit_scale))
    k = int(max(2, int(getattr(config, "endpoint_regrow_deflect_min_run_points", 3) or 3)))

    angles = np.zeros(scan_idx + 1, dtype=float)
    laterals = np.zeros(scan_idx + 1, dtype=float)
    flags = np.zeros(scan_idx + 1, dtype=np.int32)
    for i in range(scan_idx + 1):
        t_i = _unit_tangent_at_index(c, i)
        if t_i is None:
            t_i = t_ref
        ang = float(_angle_deg(t_i, t_ref))
        q = c[i] - p_ref
        axial = float(np.dot(q, t_ref))
        lateral_vec = q - axial * t_ref
        lat = float(np.linalg.norm(lateral_vec))

        angles[i] = ang
        laterals[i] = lat
        hit_soft = bool((ang >= angle_th) and (lat >= lateral_th_u))
        hit_hard = bool(ang >= hard_angle_th)
        flags[i] = 1 if (hit_soft or hit_hard) else 0

    if len(flags) < k:
        return out
    tail_grace = int(max(1, int(getattr(config, "endpoint_regrow_deflect_tail_grace_points", 4) or 4)))

    kernel = np.ones(k, dtype=np.int32)
    win_sum = np.convolve(flags, kernel, mode="valid")
    run_starts = np.where(win_sum >= k)[0]

    if len(run_starts) == 0:
        # Fallback for sharp endpoint flips: allow a short hard-angle trigger near the tail.
        hard_hits = np.where(angles[: scan_idx + 1] >= hard_angle_th)[0]
        if len(hard_hits) == 0:
            return out
        first_hard = int(hard_hits[0])
        if first_hard > tail_grace + 1:
            return out
        first_start = int(first_hard)
        end_idx = int(min(scan_idx, first_hard + 1))
    else:
        first_start = int(run_starts[0])
        if first_start > tail_grace:
            # Deflection does not start at endpoint tail; avoid trimming interior bends.
            return out

        last_start = first_start
        for rs in run_starts[1:]:
            rs_i = int(rs)
            if rs_i <= (last_start + 1):
                last_start = rs_i
            else:
                break
        end_idx = int(min(scan_idx, last_start + k - 1))
    trim_u = float(min(scan_u, float(s[end_idx])))
    if trim_u <= 1e-12:
        return out

    out["detected"] = 1.0
    out["trim_u"] = float(trim_u)
    out["trim_mm"] = float(trim_u / denom)
    out["onset_idx"] = float(first_start)
    out["onset_mm"] = float(s[first_start] / denom)
    out["angle_peak_deg"] = float(np.max(angles[: end_idx + 1])) if end_idx >= 0 else 0.0
    out["lateral_peak_mm"] = float(np.max(laterals[: end_idx + 1]) / denom) if end_idx >= 0 else 0.0
    out["run_len"] = float(max(0, end_idx - first_start + 1))
    return out


def _regrow_one_end_body_guided(
    pts: np.ndarray,
    tree: cKDTree,
    body_view: np.ndarray,
    config: CenterlineConfig,
    *,
    pin_width_hint: Optional[float] = None,
    half_t_regrow: float,
    search_r: float,
    step_u: float,
    max_steps: int,
    force_pred_only: bool = False,
    guard_bbox_grace_steps: int = 0,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """
    Regrow ONE end outward from a trimmed body in a view where the target end is at index 0.

    Returns:
    - grown_points: list of points in outward order (near -> far), excluding body_view[0].
    - stats: dict (JSON-serializable)
    """
    c_body = np.asarray(body_view, dtype=float)
    if len(c_body) < 4 or len(pts) < 10:
        return [], {"applied": 0.0, "steps": 0, "grown_mm": 0.0, "stop": "too_short", "steps_detail": []}

    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    denom = float(unit_scale + 1e-12)
    w_hint_u, t_hint_u = _clamp_pin_width_thickness_prior(
        float(pin_width_hint) if (pin_width_hint is not None and np.isfinite(pin_width_hint)) else 0.0,
        0.0,
        config,
    )
    t_hint_mm = float(t_hint_u / denom)

    # Tangent estimation params: reuse the "toward-mid PCA" settings (already validated for stability).
    inward_u = float(mm_to_unit(float(getattr(config, "trunc_end_pullback_tangent_inward_mm", 2.0)), unit_scale))
    max_pts = int(getattr(config, "trunc_end_pullback_tangent_max_points", 12))
    min_pts = int(getattr(config, "trunc_end_pullback_tangent_min_points", 4))

    slab_ql = float(getattr(config, "slab_quantile_low", 0.10))
    slab_qh = float(getattr(config, "slab_quantile_high", 0.90))
    slab_bins = int(getattr(config, "slab_coverage_bins", 12))

    min_stop = int(max(3, int(getattr(config, "endpoint_regrow_min_stop_points", 12) or 12)))
    min_update = int(max(min_stop, int(getattr(config, "endpoint_regrow_min_update_points", 30) or 30)))
    min_cov = float(getattr(config, "endpoint_regrow_min_coverage", 0.0) or 0.0)
    low_cov_action = str(getattr(config, "endpoint_regrow_low_coverage_action", "stop") or "stop").strip().lower()
    low_cov_damp = float(getattr(config, "endpoint_regrow_low_coverage_damp", 0.25) or 0.25)

    center_mode = str(getattr(config, "endpoint_regrow_center_mode", "midrange") or "midrange").strip().lower()
    if center_mode not in {"midrange", "trimmed_mean", "geomedian", "epcs_pair"}:
        center_mode = "midrange"
    center_trim_q_low = float(
        np.clip(float(getattr(config, "endpoint_regrow_center_trimmed_q_low", slab_ql) or slab_ql), 0.0, 0.49)
    )
    center_trim_q_high = float(
        np.clip(float(getattr(config, "endpoint_regrow_center_trimmed_q_high", slab_qh) or slab_qh), 0.51, 1.0)
    )
    if center_trim_q_high <= center_trim_q_low + 1e-6:
        center_trim_q_low, center_trim_q_high = 0.10, 0.90
    center_geomedian_max_iter = int(
        max(8, int(getattr(config, "endpoint_regrow_center_geomedian_max_iter", 32) or 32))
    )
    center_geomedian_tol_u = float(
        mm_to_unit(
            float(max(0.0001, float(getattr(config, "endpoint_regrow_center_geomedian_tol_mm", 0.01) or 0.01))),
            unit_scale,
        )
    )
    center_epcs_bins = int(max(8, int(getattr(config, "endpoint_regrow_center_epcs_bins", 24) or 24)))
    center_epcs_min_bin_points = int(
        max(1, int(getattr(config, "endpoint_regrow_center_epcs_min_bin_points", 2) or 2))
    )
    center_epcs_min_pairs = int(max(1, int(getattr(config, "endpoint_regrow_center_epcs_min_pairs", 3) or 3)))

    max_update = float(config.max_update())
    max_update *= float(max(1.0, float(getattr(config, "endpoint_regrow_max_update_scale", 1.0) or 1.0)))

    # Strong prior: near endpoint, large relative drift and large tangent twist are unlikely.
    rigid_enabled = bool(getattr(config, "endpoint_regrow_rigid_enabled", True))
    rigid_anchor_u = float(
        mm_to_unit(float(max(0.2, float(getattr(config, "endpoint_regrow_rigid_anchor_mm", 3.0) or 3.0))), unit_scale)
    )
    rigid_angle_deg = float(max(1.0, float(getattr(config, "endpoint_regrow_rigid_max_angle_deg", 22.0) or 22.0)))
    rigid_angle_lowcov_deg = float(
        max(1.0, float(getattr(config, "endpoint_regrow_rigid_max_angle_deg_lowcov", 14.0) or 14.0))
    )
    rigid_near_mm = float(max(0.0, float(getattr(config, "endpoint_regrow_rigid_near_mm", 2.0) or 2.0)))
    rigid_lateral_max_mm = float(max(0.0, float(getattr(config, "endpoint_regrow_rigid_lateral_max_mm", 0.18) or 0.18)))
    rigid_lateral_min_thickness_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_rigid_lateral_min_thickness_factor", 0.50) or 0.50))
    )
    rigid_lateral_min_mm = float(rigid_lateral_min_thickness_factor * t_hint_mm)
    rigid_lateral_relax = float(
        max(0.0, float(getattr(config, "endpoint_regrow_rigid_lateral_relax_per_mm", 0.08) or 0.08))
    )
    rigid_lateral_cap_mm = float(max(rigid_lateral_max_mm, float(getattr(config, "endpoint_regrow_rigid_lateral_cap_mm", 0.45) or 0.45)))
    rigid_lateral_cap_mm = float(max(rigid_lateral_cap_mm, rigid_lateral_min_mm))
    rigid_radial_var_max = float(max(0.0, float(getattr(config, "endpoint_regrow_rigid_radial_var_max", 1.0) or 1.0)))
    rigid_soft_mode = bool(getattr(config, "endpoint_regrow_rigid_soft_mode", True))
    badshape_patience = int(max(0, int(getattr(config, "endpoint_regrow_rigid_badshape_patience", 0) or 0)))

    guard_enabled = bool(getattr(config, "endpoint_regrow_guard_enabled", True))
    guard_bbox_depth_u = float(
        mm_to_unit(float(max(0.05, float(getattr(config, "endpoint_regrow_guard_bbox_depth_mm", 0.8) or 0.8))), unit_scale)
    )
    guard_bbox_min_points = int(max(1, int(getattr(config, "endpoint_regrow_guard_bbox_min_points", 4) or 4)))
    guard_bbox_half_u_floor_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_half_u_floor_factor", 0.50) or 0.50))
    )
    guard_bbox_half_v_floor_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_half_v_floor_factor", 0.50) or 0.50))
    )
    guard_bbox_half_u_floor_u = float(max(0.0, guard_bbox_half_u_floor_factor * float(w_hint_u)))
    guard_bbox_half_v_floor_u = float(max(0.0, guard_bbox_half_v_floor_factor * float(t_hint_u)))
    guard_use_polar = bool(getattr(config, "endpoint_regrow_guard_use_polar_boundary", True))
    guard_polar_bins = int(max(8, int(getattr(config, "endpoint_regrow_guard_polar_bins", 24) or 24)))
    guard_polar_min_bin_points = int(
        max(1, int(getattr(config, "endpoint_regrow_guard_polar_min_bin_points", 2) or 2))
    )
    guard_polar_min_valid_bins = int(
        max(3, int(getattr(config, "endpoint_regrow_guard_polar_min_valid_bins", 6) or 6))
    )
    guard_polar_q = float(
        np.clip(
            float(getattr(config, "endpoint_regrow_guard_polar_radius_quantile", 0.90) or 0.90),
            0.50,
            0.995,
        )
    )
    guard_polar_margin_u = float(
        mm_to_unit(
            float(max(0.0, float(getattr(config, "endpoint_regrow_guard_polar_margin_mm", 0.08) or 0.08))),
            unit_scale,
        )
    )
    guard_polar_fail_action = str(
        getattr(config, "endpoint_regrow_guard_polar_fail_action", "stop") or "stop"
    ).strip().lower()
    if guard_polar_fail_action not in {"stop", "clamp"}:
        guard_polar_fail_action = "stop"

    drop_guard_enabled = bool(getattr(config, "endpoint_regrow_drop_guard_enabled", True))
    drop_window_steps = int(max(2, int(getattr(config, "endpoint_regrow_drop_window_steps", 3) or 3)))
    drop_min_baseline_points = int(
        max(int(guard_bbox_min_points) + 2, int(getattr(config, "endpoint_regrow_drop_min_baseline_points", 24) or 24))
    )
    drop_ratio = float(
        np.clip(float(getattr(config, "endpoint_regrow_drop_ratio", 0.35) or 0.35), 0.05, 0.95)
    )
    drop_abs_points = int(max(1, int(getattr(config, "endpoint_regrow_drop_abs_points", 20) or 20)))
    sparse_confirm_points = int(max(1, int(getattr(config, "endpoint_regrow_sparse_confirm_points", 10) or 10)))
    sparse_confirm_steps = int(max(1, int(getattr(config, "endpoint_regrow_sparse_confirm_steps", 2) or 2)))
    sparse_continue_ratio = float(
        np.clip(float(getattr(config, "endpoint_regrow_sparse_continue_ratio", 0.90) or 0.90), 0.10, 1.0)
    )
    bbox_probe_enabled = bool(getattr(config, "endpoint_regrow_guard_bbox_probe_enabled", True))
    bbox_probe_steps_cfg = getattr(config, "endpoint_regrow_guard_bbox_probe_steps", 1)
    if bbox_probe_steps_cfg is None:
        bbox_probe_steps_cfg = 1
    bbox_probe_steps = int(max(0, int(bbox_probe_steps_cfg)))
    bbox_probe_depth_mm = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_probe_depth_mm", 2.4) or 2.4))
    )
    bbox_probe_radius_mm = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_probe_radius_mm", 1.5) or 1.5))
    )
    bbox_probe_min_points = int(
        max(1, int(getattr(config, "endpoint_regrow_guard_bbox_probe_min_points", 12) or 12))
    )
    bbox_probe_depth_u = float(mm_to_unit(float(bbox_probe_depth_mm), unit_scale))
    bbox_probe_radius_u = float(mm_to_unit(float(bbox_probe_radius_mm), unit_scale))
    guard_bbox_half_lat_floor_u = float(
        max(guard_bbox_half_u_floor_u, guard_bbox_half_v_floor_u, bbox_probe_radius_u)
    )
    bbox_probe_min_n_cand_scale = float(
        max(1.0, float(getattr(config, "endpoint_regrow_guard_bbox_probe_min_n_cand_scale", 8.0) or 8.0))
    )
    bbox_probe_min_n_cand = int(max(min_update, int(np.ceil(float(min_update) * bbox_probe_min_n_cand_scale))))
    sparse_dense_extra_confirm_steps = int(
        max(0, int(getattr(config, "endpoint_regrow_sparse_dense_extra_confirm_steps", 1) or 1))
    )
    sparse_dense_min_n_cand_scale = float(
        max(1.0, float(getattr(config, "endpoint_regrow_sparse_dense_min_n_cand_scale", 8.0) or 8.0))
    )
    sparse_dense_min_n_cand = int(max(min_update, int(np.ceil(float(min_update) * sparse_dense_min_n_cand_scale))))
    direction_lock_enabled = bool(getattr(config, "endpoint_regrow_direction_lock_enabled", True))
    direction_lock_lateral_streak = int(
        max(1, int(getattr(config, "endpoint_regrow_direction_lock_lateral_streak", 2) or 2))
    )
    direction_lock_angle_deg = float(
        max(1.0, float(getattr(config, "endpoint_regrow_direction_lock_angle_deg", rigid_angle_lowcov_deg) or rigid_angle_lowcov_deg))
    )
    guard_bbox_grace_steps = int(max(0, int(guard_bbox_grace_steps)))

    # Stable reference axis from the trimmed body endpoint toward the body center.
    s_body = _cum_arclen(c_body)
    t_ref_in = tangent_toward_mid_pca(
        c_body,
        idx=0,
        mid_idx=int(len(c_body) - 1),
        s=s_body,
        inward_u=float(max(inward_u, rigid_anchor_u)),
        max_points=int(max_pts),
        min_points=int(min_pts),
    )
    t_ref = -np.asarray(t_ref_in, dtype=float)
    n_ref = float(np.linalg.norm(t_ref))
    if not np.isfinite(n_ref) or n_ref <= 1e-12:
        v01 = np.asarray(c_body[0] - c_body[min(1, len(c_body) - 1)], dtype=float)
        n01 = float(np.linalg.norm(v01))
        if n01 > 1e-12:
            t_ref = v01 / n01
        else:
            t_ref = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        t_ref = t_ref / n_ref
    p_ref0 = np.asarray(c_body[0], dtype=float)

    def _rigid_lateral_limit_mm(grown_mm: float) -> float:
        extra = max(0.0, float(grown_mm) - rigid_near_mm)
        lim = rigid_lateral_max_mm + extra * rigid_lateral_relax
        lim = float(min(rigid_lateral_cap_mm, lim))
        return float(max(rigid_lateral_min_mm, lim))

    grown: List[np.ndarray] = []
    p_curr = np.asarray(c_body[0], dtype=float)
    prev_u: Optional[np.ndarray] = None
    stop_reason = "max_steps"
    step_logs: List[Dict[str, object]] = []
    badshape_streak = 0
    guard_hits = 0
    guard_bbox_fail = 0
    guard_polar_fail = 0

    n_slab = 0
    cov = 0.0

    support_hist: List[int] = []
    sparse_mode = False
    sparse_anchor_len: Optional[int] = None
    sparse_anchor_step = 0
    sparse_decline_count = 0
    sparse_last_support: Optional[int] = None
    sparse_lock_dir: Optional[np.ndarray] = None
    sparse_triggered = 0
    sparse_confirmed = 0
    bbox_probe_used = 0
    direction_lock_active = bool(force_pred_only)
    direction_lock_triggered = int(1 if direction_lock_active else 0)
    lateral_clamp_streak = 0

    for _k in range(int(max_steps)):
        step_idx = int(_k + 1)
        grown_mm_next = float(step_idx * step_u / denom)
        rec: Dict[str, object] = {
            "step": step_idx,
            "grown_mm": grown_mm_next,
            "n_cand": 0,
            "n_slab": 0,
            "coverage": 0.0,
            "radial_var": float("inf"),
            "do_update": 0,
            "damp": 0.0,
            "angle_ref_deg": 0.0,
            "angle_limit_deg": 0.0,
            "delta_mm": 0.0,
            "delta_clipped": 0,
            "lateral_mm": 0.0,
            "lateral_limit_mm": _rigid_lateral_limit_mm(grown_mm_next),
            "badshape_streak": 0,
            "soft_action": "",
            "guard_hit": 0,
            "guard_reason": "",
            "guard_n_out_axis": 0,
            "guard_bbox_grace": 0,
            "guard_polar_bins_valid": 0,
            "guard_polar_r_limit_mm": 0.0,
            "guard_polar_r_next_mm": 0.0,
            "center_mode": center_mode,
            "center_mode_effective": center_mode,
            "center_fallback": "",
            "center_pairs": 0,
            "drop_trigger": 0,
            "sparse_mode": 1 if sparse_mode else 0,
            "sparse_baseline": 0.0,
            "sparse_anchor_step": int(sparse_anchor_step),
            "sparse_decline_count": int(sparse_decline_count),
            "sparse_confirm_steps_eff": int(sparse_confirm_steps),
            "dense_cand": 0,
            "pred_only_lock": 1 if direction_lock_active else 0,
            "bbox_probe": 0,
            "stop": "",
        }

        # Build a curve view from current point toward the mid-body to compute a stable inward tangent.
        if grown:
            # inward order: [curr, prev, ..., first] + body (starting from the trimmed end)
            tan_curve = [np.asarray(p_curr, dtype=float)] + [np.asarray(p, dtype=float) for p in reversed(grown[:-1])] + [
                np.asarray(p, dtype=float) for p in c_body
            ]
            c_tan = np.stack(tan_curve, axis=0)
        else:
            c_tan = c_body

        s_tan = _cum_arclen(c_tan)
        t_in = tangent_toward_mid_pca(
            c_tan,
            idx=0,
            mid_idx=int(len(c_tan) - 1),
            s=s_tan,
            inward_u=float(inward_u),
            max_points=int(max_pts),
            min_points=int(min_pts),
        )
        t_out = -np.asarray(t_in, dtype=float)
        t_out_n = float(np.linalg.norm(t_out))
        if not np.isfinite(t_out_n) or t_out_n <= 1e-12:
            rec["stop"] = "bad_tangent"
            step_logs.append(rec)
            stop_reason = "bad_tangent"
            break
        t_out = t_out / t_out_n
        if sparse_mode and sparse_lock_dir is not None:
            t_out = np.asarray(sparse_lock_dir, dtype=float)
        elif direction_lock_active:
            t_out = np.asarray(t_ref, dtype=float)

        # Predict one step outward along the body-guided tangent.
        p_pred = p_curr + float(step_u) * t_out

        # Slab evidence around the predicted point (acts as both "existence check" and optional lateral correction).
        cand = tree.query_ball_point(p_pred, r=float(search_r))
        rec["n_cand"] = int(len(cand))
        if len(cand) < int(min_stop):
            rec["stop"] = "no_candidates"
            step_logs.append(rec)
            stop_reason = "no_candidates"
            break

        local = pts[cand] - p_pred
        axial = local @ t_out
        mask = np.abs(axial) <= float(half_t_regrow)
        n_slab = int(np.sum(mask))
        rec["n_slab"] = int(n_slab)
        if n_slab < int(min_stop):
            rec["stop"] = "no_slab_points"
            step_logs.append(rec)
            stop_reason = "no_slab_points"
            break

        local = local[mask]
        u, v = make_plane_basis(t_out, prev_u=prev_u)
        prev_u = np.asarray(u, dtype=float)

        coords_u = local @ u
        coords_v = local @ v
        cov = float(slab_angular_coverage(coords_u, coords_v, bins=slab_bins)) if slab_bins > 0 else 1.0
        rec["coverage"] = float(cov)

        radial_var = float(_radial_var_from_uv(coords_u, coords_v))
        rec["radial_var"] = float(radial_var)

        # Default: keep p_pred unless slab evidence is strong enough to justify lateral updates.
        step_pred_only = bool(force_pred_only or direction_lock_active)
        do_update = bool(n_slab >= int(min_update) and (min_cov <= 0.0 or cov >= float(min_cov)))
        if step_pred_only:
            do_update = False
        rec["do_update"] = 1 if do_update else 0
        if not do_update:
            if (not sparse_mode) and (not step_pred_only) and low_cov_action in {"stop", "break"}:
                rec["stop"] = "low_evidence"
                step_logs.append(rec)
                stop_reason = "low_evidence"
                break
            if low_cov_action in {"damp"}:
                damp = float(np.clip(low_cov_damp, 0.0, 1.0))
            else:
                # pred_only/skip: no lateral correction
                damp = 0.0
        else:
            damp = 1.0
        if sparse_mode:
            do_update = False
            rec["do_update"] = 0
            damp = 0.0
        if step_pred_only:
            do_update = False
            rec["do_update"] = 0
            damp = 0.0
        rec["damp"] = float(damp)

        # Relative twist guard: with weak evidence, require stricter tangent agreement to the endpoint reference.
        dot_ref = float(np.clip(np.dot(t_out, t_ref), -1.0, 1.0))
        angle_ref_deg = float(np.degrees(np.arccos(dot_ref)))
        angle_limit_deg = float(rigid_angle_deg if do_update else min(rigid_angle_deg, rigid_angle_lowcov_deg))
        rec["angle_ref_deg"] = float(angle_ref_deg)
        rec["angle_limit_deg"] = float(angle_limit_deg)

        soft_actions: List[str] = []
        lateral_clamped_this_step = False
        if sparse_mode:
            soft_actions.append("sparse_pred_only")
        angle_bad = bool(rigid_enabled and angle_ref_deg > angle_limit_deg + 1e-9)
        radial_bad = bool(rigid_enabled and np.isfinite(radial_var) and radial_var > rigid_radial_var_max + 1e-9)

        if angle_bad:
            if rigid_soft_mode:
                # Keep axial growth but disable lateral updates for this risky step.
                damp = 0.0
                rec["damp"] = float(damp)
                soft_actions.append("soft_angle_pred_only")
            else:
                rec["stop"] = "rigid_angle"
                step_logs.append(rec)
                stop_reason = "rigid_angle"
                break

        if radial_bad:
            if rigid_soft_mode:
                # Single-sided / biased slabs often raise radial_var; keep outward growth and avoid lateral pull.
                damp = 0.0
                rec["damp"] = float(damp)
                soft_actions.append("soft_radial_pred_only")
            else:
                rec["stop"] = "rigid_radial_var"
                step_logs.append(rec)
                stop_reason = "rigid_radial_var"
                break

        if rigid_soft_mode and soft_actions:
            badshape_streak += 1
            rec["badshape_streak"] = int(badshape_streak)
            if badshape_patience > 0 and badshape_streak >= badshape_patience:
                rec["soft_action"] = "|".join(soft_actions)
                rec["stop"] = "rigid_badshape_patience"
                step_logs.append(rec)
                stop_reason = "rigid_badshape_patience"
                break
        else:
            badshape_streak = 0
            rec["badshape_streak"] = 0

        du, dv, center_diag = _endpoint_regrow_center_uv(
            coords_u,
            coords_v,
            mode=center_mode,
            q_low=center_trim_q_low,
            q_high=center_trim_q_high,
            geomedian_max_iter=center_geomedian_max_iter,
            geomedian_tol_u=center_geomedian_tol_u,
            epcs_bins=center_epcs_bins,
            epcs_min_bin_points=center_epcs_min_bin_points,
            epcs_min_pairs=center_epcs_min_pairs,
        )
        rec["center_mode_effective"] = str(center_diag.get("center_mode_effective", center_mode))
        rec["center_fallback"] = str(center_diag.get("center_fallback", ""))
        rec["center_pairs"] = int(center_diag.get("center_pairs", 0) or 0)
        delta = float(du) * u + float(dv) * v  # lateral-only by construction
        dn = float(np.linalg.norm(delta))
        clipped = 0
        if dn > 1e-12 and dn > max_update:
            delta = delta / (dn + 1e-12) * float(max_update)
            clipped = 1

        rec["delta_mm"] = float(np.linalg.norm(delta) / denom)
        rec["delta_clipped"] = int(clipped)

        p_next = p_pred + float(config.refine_damping) * float(damp) * delta

        # Relative lateral drift guard against endpoint reference axis.
        rel = np.asarray(p_next, dtype=float) - p_ref0
        rel_ax = float(np.dot(rel, t_ref))
        rel_lat = rel - rel_ax * t_ref
        lateral_u = float(np.linalg.norm(rel_lat))
        lateral_mm = float(lateral_u / denom)
        rec["lateral_mm"] = float(lateral_mm)

        if rigid_enabled:
            lat_lim_mm = _rigid_lateral_limit_mm(grown_mm_next)
            rec["lateral_limit_mm"] = float(lat_lim_mm)
            lat_lim_u = float(mm_to_unit(lat_lim_mm, unit_scale))
            if lateral_u > lat_lim_u + 1e-12:
                if rigid_soft_mode:
                    # Clamp only the normal-plane drift; keep axial extension.
                    rel_lat_dir = rel_lat / (lateral_u + 1e-12)
                    rel_lat = rel_lat_dir * lat_lim_u
                    p_next = p_ref0 + rel_ax * t_ref + rel_lat
                    lateral_u = float(np.linalg.norm(rel_lat))
                    rec["lateral_mm"] = float(lateral_u / denom)
                    soft_actions.append("soft_lateral_clamp")
                    lateral_clamped_this_step = True
                    badshape_streak += 1
                    rec["badshape_streak"] = int(badshape_streak)
                    if badshape_patience > 0 and badshape_streak >= badshape_patience:
                        rec["soft_action"] = "|".join(soft_actions)
                        rec["stop"] = "rigid_badshape_patience"
                        step_logs.append(rec)
                        stop_reason = "rigid_badshape_patience"
                        break
                else:
                    rec["stop"] = "rigid_lateral"
                    step_logs.append(rec)
                    stop_reason = "rigid_lateral"
                    break

        if direction_lock_enabled:
            if sparse_mode and lateral_clamped_this_step and angle_ref_deg >= direction_lock_angle_deg:
                lateral_clamp_streak += 1
            else:
                lateral_clamp_streak = 0
            if (not direction_lock_active) and lateral_clamp_streak >= int(direction_lock_lateral_streak):
                direction_lock_active = True
                direction_lock_triggered += 1
                sparse_lock_dir = np.asarray(t_ref, dtype=float)
                soft_actions.append("dir_lock_pred_only")
            rec["pred_only_lock"] = 1 if direction_lock_active else 0

        if guard_enabled:
            # 1) Forward support check (bbox-like) in local {t,u,v} frame.
            local_next = pts[cand] - p_next
            axial_next = local_next @ t_out
            uu_next = local_next @ u
            vv_next = local_next @ v

            # Lateral envelope from current slab (non-rectangular boundary falls back to this support scale).
            u_lo, u_hi = np.quantile(coords_u, [0.10, 0.90])
            v_lo, v_hi = np.quantile(coords_v, [0.10, 0.90])
            half_u = float(max(0.5 * (u_hi - u_lo), guard_bbox_half_lat_floor_u))
            half_v = float(max(0.5 * (v_hi - v_lo), guard_bbox_half_lat_floor_u))
            mask_lat = (np.abs(uu_next) <= half_u) & (np.abs(vv_next) <= half_v)
            n_out_axis = int(np.sum(mask_lat & (axial_next > 0.0) & (axial_next <= guard_bbox_depth_u)))
            rec["guard_n_out_axis"] = int(n_out_axis)

            baseline_support = 0.0
            if drop_guard_enabled and (not sparse_mode) and len(support_hist) >= int(drop_window_steps):
                baseline_support = float(np.median(np.asarray(support_hist[-drop_window_steps:], dtype=float)))
                rec["sparse_baseline"] = float(baseline_support)
                rel_ratio = float(n_out_axis / (baseline_support + 1e-12))
                rel_drop = float(baseline_support - float(n_out_axis))
                if (
                    baseline_support >= float(drop_min_baseline_points)
                    and rel_ratio <= float(drop_ratio)
                    and rel_drop >= float(drop_abs_points)
                ):
                    sparse_mode = True
                    sparse_triggered += 1
                    sparse_anchor_len = int(len(grown) + 1)
                    sparse_anchor_step = int(step_idx)
                    sparse_lock_dir = np.asarray(t_ref, dtype=float)
                    sparse_last_support = int(n_out_axis)
                    sparse_decline_count = 0
                    rec["drop_trigger"] = 1
                    rec["sparse_mode"] = 1
                    rec["sparse_anchor_step"] = int(sparse_anchor_step)

            bbox_fail_now = bool(n_out_axis < int(guard_bbox_min_points))
            if bbox_fail_now:
                rec["guard_hit"] = 1
                grace_ok = bool(step_idx <= int(guard_bbox_grace_steps))
                rec["guard_bbox_grace"] = 1 if grace_ok else 0
                rec["guard_reason"] = "bbox_front_support_grace" if grace_ok else "bbox_front_support"
                guard_hits += 1
                guard_bbox_fail += 1
                if (not sparse_mode) and (not grace_ok):
                    probe_ok = False
                    if bbox_probe_enabled and step_pred_only and bbox_probe_used < int(bbox_probe_steps):
                        dense_local = bool(
                            len(cand) >= int(bbox_probe_min_n_cand)
                            and n_slab >= int(min_update)
                            and (min_cov <= 0.0 or cov >= float(min_cov))
                        )
                        shape_ok = bool(
                            np.isfinite(radial_var)
                            and radial_var <= float(max(1.20, 1.25 * float(rigid_radial_var_max)))
                        )
                        probe_depth_eff_u = float(max(guard_bbox_depth_u, bbox_probe_depth_u))
                        probe_radius_eff_u = float(max(guard_bbox_half_lat_floor_u, bbox_probe_radius_u))
                        probe_n = 0
                        if probe_depth_eff_u > 1e-12 and probe_radius_eff_u > 1e-12:
                            rr2_next = uu_next * uu_next + vv_next * vv_next
                            probe_mask = (
                                (axial_next > 0.0)
                                & (axial_next <= probe_depth_eff_u)
                                & (rr2_next <= probe_radius_eff_u * probe_radius_eff_u)
                            )
                            probe_n = int(np.sum(probe_mask))
                        rec["bbox_probe_n"] = int(probe_n)
                        rec["bbox_probe_depth_mm"] = float(probe_depth_eff_u / denom)
                        rec["bbox_probe_radius_mm"] = float(probe_radius_eff_u / denom)
                        probe_geom_ok = bool(probe_n >= int(bbox_probe_min_points))
                        probe_ok = bool(dense_local and shape_ok and probe_geom_ok)
                    if probe_ok:
                        sparse_mode = True
                        sparse_triggered += 1
                        sparse_anchor_len = int(len(grown) + 1)
                        sparse_anchor_step = int(step_idx)
                        sparse_lock_dir = np.asarray(t_ref, dtype=float)
                        sparse_last_support = int(n_out_axis)
                        sparse_decline_count = 0
                        bbox_probe_used += 1
                        rec["bbox_probe"] = 1
                        rec["guard_reason"] = "bbox_front_support_probe"
                        rec["sparse_mode"] = 1
                        rec["sparse_anchor_step"] = int(sparse_anchor_step)
                        if not direction_lock_active:
                            direction_lock_active = True
                            direction_lock_triggered += 1
                        rec["pred_only_lock"] = 1
                        soft_actions.append("bbox_probe_sparse")
                    else:
                        rec["stop"] = "guard_bbox_front_support"
                        step_logs.append(rec)
                        stop_reason = "guard_bbox_front_support"
                        break

            if sparse_mode:
                if sparse_anchor_step <= 0:
                    sparse_anchor_step = int(step_idx)
                dense_cand = bool(len(cand) >= int(sparse_dense_min_n_cand))
                confirm_steps_eff = int(
                    sparse_confirm_steps + (sparse_dense_extra_confirm_steps if dense_cand else 0)
                )
                rec["dense_cand"] = 1 if dense_cand else 0
                rec["sparse_confirm_steps_eff"] = int(confirm_steps_eff)
                confirm_now = False
                if step_idx > int(sparse_anchor_step):
                    low_points = bool(n_out_axis <= int(sparse_confirm_points))
                    keep_dropping = False
                    if sparse_last_support is not None:
                        keep_dropping = bool(
                            n_out_axis <= max(0.0, float(sparse_last_support) * float(sparse_continue_ratio))
                        )
                    if low_points or keep_dropping:
                        sparse_decline_count += 1
                    else:
                        sparse_decline_count = 0
                    if sparse_decline_count >= int(confirm_steps_eff):
                        confirm_now = True
                rec["sparse_mode"] = 1
                rec["sparse_anchor_step"] = int(sparse_anchor_step)
                rec["sparse_decline_count"] = int(sparse_decline_count)
                sparse_last_support = int(n_out_axis)
                if confirm_now:
                    rec["guard_hit"] = 1
                    rec["guard_reason"] = "sparse_confirm_endpoint"
                    rec["stop"] = "sparse_confirm_endpoint"
                    step_logs.append(rec)
                    stop_reason = "sparse_confirm_endpoint"
                    sparse_confirmed += 1
                    break

            # 2) Local non-rectangular boundary (polar in slab plane).
            if guard_use_polar:
                rel_next = np.asarray(p_next, dtype=float) - np.asarray(p_pred, dtype=float)
                du_next = float(np.dot(rel_next, u))
                dv_next = float(np.dot(rel_next, v))
                r_next = float(np.hypot(du_next, dv_next))
                theta_next = float(np.arctan2(dv_next, du_next))
                r_lim, n_valid_bins = _polar_boundary_radius_limit_from_uv(
                    coords_u,
                    coords_v,
                    theta_next,
                    bins=int(guard_polar_bins),
                    q_radius=float(guard_polar_q),
                    min_bin_points=int(guard_polar_min_bin_points),
                    min_valid_bins=int(guard_polar_min_valid_bins),
                )
                rec["guard_polar_bins_valid"] = int(n_valid_bins)
                rec["guard_polar_r_limit_mm"] = float(r_lim / denom) if np.isfinite(r_lim) else 0.0
                rec["guard_polar_r_next_mm"] = float(r_next / denom)
                if np.isfinite(r_lim):
                    r_allow = float(max(0.0, r_lim + guard_polar_margin_u))
                    if r_next > r_allow + 1e-12:
                        if guard_polar_fail_action == "clamp" and r_next > 1e-12:
                            scale = float(r_allow / (r_next + 1e-12))
                            du_c = float(du_next * scale)
                            dv_c = float(dv_next * scale)
                            rel_ax_only = float(np.dot(np.asarray(p_next, dtype=float) - np.asarray(p_pred, dtype=float), t_out))
                            p_next = np.asarray(p_pred, dtype=float) + rel_ax_only * t_out + du_c * u + dv_c * v
                            rel = np.asarray(p_next, dtype=float) - p_ref0
                            rel_ax = float(np.dot(rel, t_ref))
                            rel_lat = rel - rel_ax * t_ref
                            lateral_u = float(np.linalg.norm(rel_lat))
                            rec["lateral_mm"] = float(lateral_u / denom)
                            rec["guard_hit"] = 1
                            rec["guard_reason"] = "polar_clamp"
                            soft_actions.append("guard_polar_clamp")
                            guard_hits += 1
                            guard_polar_fail += 1
                        else:
                            rec["guard_hit"] = 1
                            guard_hits += 1
                            guard_polar_fail += 1
                            if sparse_mode:
                                rec["guard_reason"] = "polar_cross_soft"
                                soft_actions.append("guard_polar_cross_soft")
                            else:
                                rec["guard_reason"] = "polar_cross"
                                rec["stop"] = "guard_polar_cross"
                                step_logs.append(rec)
                                stop_reason = "guard_polar_cross"
                                break

        if guard_enabled:
            support_hist.append(int(rec.get("guard_n_out_axis", 0) or 0))

        if soft_actions:
            rec["soft_action"] = "|".join(soft_actions)

        step_logs.append(rec)
        grown.append(np.asarray(p_next, dtype=float))
        p_curr = np.asarray(p_next, dtype=float)

    sparse_rollback_applied = 0
    if stop_reason == "sparse_confirm_endpoint" and sparse_anchor_len is not None and len(grown) > int(sparse_anchor_len):
        grown = grown[: int(sparse_anchor_len)]
        sparse_rollback_applied = 1

    grown_u = float(len(grown)) * float(step_u)
    soft_events = int(sum(1 for r in step_logs if str(r.get("soft_action", "") or "").strip()))
    stats = {
        "applied": 1.0 if grown else 0.0,
        "center_mode": str(center_mode),
        "force_pred_only": 1.0 if bool(force_pred_only) else 0.0,
        "guard_bbox_grace_steps": int(guard_bbox_grace_steps),
        "steps": int(len(grown)),
        "grown_mm": float(grown_u / denom),
        "stop": str(stop_reason),
        "last_n_slab": int(n_slab),
        "last_cov": float(cov),
        "soft_events": int(soft_events),
        "guard_hits": int(guard_hits),
        "guard_bbox_fail": int(guard_bbox_fail),
        "guard_polar_fail": int(guard_polar_fail),
        "sparse_triggered": int(sparse_triggered),
        "sparse_confirmed": int(sparse_confirmed),
        "sparse_anchor_step": int(sparse_anchor_step),
        "bbox_probe_used": int(bbox_probe_used),
        "direction_lock_triggered": int(direction_lock_triggered),
        "sparse_rollback_applied": int(sparse_rollback_applied),
        "steps_detail": step_logs,
    }
    return grown, stats


def _debug_write_regrow_steps(dbg_dir: Path, side: str, steps: List[Dict[str, object]]) -> None:
    rows = list(steps or [])
    if not rows:
        return

    json_path = dbg_dir / f"stage13_regrow_steps_{side}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames: List[str] = []
    keys = set()
    for r in rows:
        keys.update(r.keys())
    preferred = [
        "step",
        "grown_mm",
        "n_cand",
        "n_slab",
        "coverage",
        "radial_var",
        "do_update",
        "damp",
        "angle_ref_deg",
        "angle_limit_deg",
        "delta_mm",
        "delta_clipped",
        "lateral_mm",
        "lateral_limit_mm",
        "guard_hit",
        "guard_reason",
        "guard_n_out_axis",
        "drop_trigger",
        "sparse_mode",
        "sparse_baseline",
        "sparse_anchor_step",
        "sparse_decline_count",
        "guard_polar_bins_valid",
        "guard_polar_r_limit_mm",
        "guard_polar_r_next_mm",
        "center_mode",
        "center_mode_effective",
        "center_fallback",
        "center_pairs",
        "stop",
    ]
    for k in preferred:
        if k in keys:
            fieldnames.append(k)
    for k in sorted(keys):
        if k not in fieldnames:
            fieldnames.append(k)

    csv_path = dbg_dir / f"stage13_regrow_steps_{side}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _endpoint_regrow_one_step_lookahead(
    points: np.ndarray,
    body_view: np.ndarray,
    config: CenterlineConfig,
    *,
    tree: Optional[cKDTree] = None,
    pin_width_hint: Optional[float] = None,
    half_t_regrow: Optional[float] = None,
    search_r: Optional[float] = None,
    step_u: Optional[float] = None,
) -> Dict[str, object]:
    """Estimate one-step endpoint evidence before regrow (support/cov/radial_var)."""
    pts = np.asarray(points, dtype=float)
    c_body = np.asarray(body_view, dtype=float)
    out: Dict[str, object] = {
        "valid": 0.0,
        "reason": "too_short",
        "n_cand1": 0.0,
        "n_slab1": 0.0,
        "support_n1": 0.0,
        "cov1": 0.0,
        "radial_var1": float("inf"),
    }
    if len(pts) < 10 or len(c_body) < 4:
        return out

    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    w_hint_u, t_hint_u = _clamp_pin_width_thickness_prior(
        float(pin_width_hint) if (pin_width_hint is not None and np.isfinite(pin_width_hint)) else 0.0,
        0.0,
        config,
    )
    guard_bbox_half_u_floor_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_half_u_floor_factor", 0.50) or 0.50))
    )
    guard_bbox_half_v_floor_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_half_v_floor_factor", 0.50) or 0.50))
    )
    guard_bbox_half_u_floor_u = float(max(0.0, guard_bbox_half_u_floor_factor * float(w_hint_u)))
    guard_bbox_half_v_floor_u = float(max(0.0, guard_bbox_half_v_floor_factor * float(t_hint_u)))
    bbox_probe_radius_mm = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_probe_radius_mm", 1.5) or 1.5))
    )
    bbox_probe_radius_u = float(mm_to_unit(float(bbox_probe_radius_mm), unit_scale))
    guard_bbox_half_lat_floor_u = float(
        max(guard_bbox_half_u_floor_u, guard_bbox_half_v_floor_u, bbox_probe_radius_u)
    )
    if step_u is None:
        step_u = float(config.centerline_step())
    if search_r is None:
        search_r = float(config.slab_search_radius(pin_width_hint))
    if half_t_regrow is None:
        half_t_regrow = float(config.slab_half_thickness())

    min_stop = int(max(3, int(getattr(config, "endpoint_regrow_min_stop_points", 12) or 12)))

    inward_u = float(mm_to_unit(float(getattr(config, "trunc_end_pullback_tangent_inward_mm", 2.0)), unit_scale))
    max_pts = int(getattr(config, "trunc_end_pullback_tangent_max_points", 12))
    min_pts = int(getattr(config, "trunc_end_pullback_tangent_min_points", 4))

    s_body = _cum_arclen(c_body)
    t_in = tangent_toward_mid_pca(
        c_body,
        idx=0,
        mid_idx=int(len(c_body) - 1),
        s=s_body,
        inward_u=float(inward_u),
        max_points=int(max_pts),
        min_points=int(min_pts),
    )
    t_out = -np.asarray(t_in, dtype=float)
    t_out_n = float(np.linalg.norm(t_out))
    if not np.isfinite(t_out_n) or t_out_n <= 1e-12:
        v01 = np.asarray(c_body[0] - c_body[min(1, len(c_body) - 1)], dtype=float)
        n01 = float(np.linalg.norm(v01))
        if not np.isfinite(n01) or n01 <= 1e-12:
            out["reason"] = "bad_tangent"
            return out
        t_out = v01 / n01
    else:
        t_out = t_out / t_out_n

    p_pred = np.asarray(c_body[0], dtype=float) + float(step_u) * t_out
    if tree is None:
        tree = cKDTree(pts)

    cand = tree.query_ball_point(p_pred, r=float(search_r))
    out["n_cand1"] = float(len(cand))
    if len(cand) < int(min_stop):
        out["reason"] = "no_candidates"
        return out

    local = pts[cand] - p_pred
    axial = local @ t_out
    mask = np.abs(axial) <= float(half_t_regrow)
    n_slab = int(np.sum(mask))
    out["n_slab1"] = float(n_slab)
    if n_slab < int(min_stop):
        out["reason"] = "no_slab_points"
        return out

    local = local[mask]
    u, v = make_plane_basis(t_out, prev_u=None)
    coords_u = local @ u
    coords_v = local @ v

    slab_bins = int(getattr(config, "slab_coverage_bins", 12))
    cov1 = float(slab_angular_coverage(coords_u, coords_v, bins=slab_bins)) if slab_bins > 0 else 1.0
    rv1 = float(_radial_var_from_uv(coords_u, coords_v))

    try:
        u_lo, u_hi = np.quantile(coords_u, [0.10, 0.90])
        v_lo, v_hi = np.quantile(coords_v, [0.10, 0.90])
        half_u = float(max(0.5 * (u_hi - u_lo), guard_bbox_half_lat_floor_u))
        half_v = float(max(0.5 * (v_hi - v_lo), guard_bbox_half_lat_floor_u))
    except Exception:
        out["reason"] = "quantile_failed"
        out["cov1"] = float(cov1)
        out["radial_var1"] = float(rv1)
        return out

    local_next = pts[cand] - p_pred
    axial_next = local_next @ t_out
    uu_next = local_next @ u
    vv_next = local_next @ v
    mask_lat = (np.abs(uu_next) <= half_u) & (np.abs(vv_next) <= half_v)
    guard_bbox_depth_u = float(
        mm_to_unit(float(max(0.05, float(getattr(config, "endpoint_regrow_guard_bbox_depth_mm", 0.8) or 0.8))), unit_scale)
    )
    n_out_axis = int(np.sum(mask_lat & (axial_next > 0.0) & (axial_next <= guard_bbox_depth_u)))

    out.update(
        {
            "valid": 1.0,
            "reason": "ok",
            "support_n1": float(n_out_axis),
            "cov1": float(cov1),
            "radial_var1": float(rv1),
        }
    )
    return out


def _endpoint_regrow_corridor_lookahead(
    points: np.ndarray,
    body_view: np.ndarray,
    config: CenterlineConfig,
    *,
    tree: Optional[cKDTree] = None,
    pin_width_hint: Optional[float] = None,
    half_t_regrow: Optional[float] = None,
    search_r: Optional[float] = None,
    step_u: Optional[float] = None,
) -> Dict[str, object]:
    """Estimate multi-step forward corridor evidence for endpoint extension.

    This complements one-step lookahead: even when the immediate next slice is sparse,
    we can still decide to extend if a short forward corridor has stable support.
    """
    pts = np.asarray(points, dtype=float)
    c_body = np.asarray(body_view, dtype=float)
    out: Dict[str, object] = {
        "valid": 0.0,
        "reason": "too_short",
        "steps_eval": 0.0,
        "valid_steps": 0.0,
        "support_sum": 0.0,
        "support_peak": 0.0,
        "cov_mean": 0.0,
        "rv_max": float("inf"),
        "step_rows": [],
    }
    if len(pts) < 10 or len(c_body) < 4:
        return out

    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    w_hint_u, t_hint_u = _clamp_pin_width_thickness_prior(
        float(pin_width_hint) if (pin_width_hint is not None and np.isfinite(pin_width_hint)) else 0.0,
        0.0,
        config,
    )
    guard_bbox_half_u_floor_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_half_u_floor_factor", 0.50) or 0.50))
    )
    guard_bbox_half_v_floor_factor = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_half_v_floor_factor", 0.50) or 0.50))
    )
    guard_bbox_half_u_floor_u = float(max(0.0, guard_bbox_half_u_floor_factor * float(w_hint_u)))
    guard_bbox_half_v_floor_u = float(max(0.0, guard_bbox_half_v_floor_factor * float(t_hint_u)))
    bbox_probe_radius_mm = float(
        max(0.0, float(getattr(config, "endpoint_regrow_guard_bbox_probe_radius_mm", 1.5) or 1.5))
    )
    bbox_probe_radius_u = float(mm_to_unit(float(bbox_probe_radius_mm), unit_scale))
    guard_bbox_half_lat_floor_u = float(
        max(guard_bbox_half_u_floor_u, guard_bbox_half_v_floor_u, bbox_probe_radius_u)
    )
    if step_u is None:
        step_u = float(config.centerline_step())
    if search_r is None:
        search_r = float(config.slab_search_radius(pin_width_hint))
    if half_t_regrow is None:
        half_t_regrow = float(config.slab_half_thickness())

    min_stop = int(max(3, int(getattr(config, "endpoint_regrow_min_stop_points", 12) or 12)))
    corridor_steps = int(max(1, int(getattr(config, "endpoint_regrow_triclass_corridor_steps", 3) or 3)))
    out["steps_eval"] = float(corridor_steps)

    inward_u = float(mm_to_unit(float(getattr(config, "trunc_end_pullback_tangent_inward_mm", 2.0)), unit_scale))
    max_pts = int(getattr(config, "trunc_end_pullback_tangent_max_points", 12))
    min_pts = int(getattr(config, "trunc_end_pullback_tangent_min_points", 4))

    s_body = _cum_arclen(c_body)
    t_in = tangent_toward_mid_pca(
        c_body,
        idx=0,
        mid_idx=int(len(c_body) - 1),
        s=s_body,
        inward_u=float(inward_u),
        max_points=int(max_pts),
        min_points=int(min_pts),
    )
    t_out = -np.asarray(t_in, dtype=float)
    t_out_n = float(np.linalg.norm(t_out))
    if not np.isfinite(t_out_n) or t_out_n <= 1e-12:
        v01 = np.asarray(c_body[0] - c_body[min(1, len(c_body) - 1)], dtype=float)
        n01 = float(np.linalg.norm(v01))
        if not np.isfinite(n01) or n01 <= 1e-12:
            out["reason"] = "bad_tangent"
            return out
        t_out = v01 / n01
    else:
        t_out = t_out / t_out_n

    if tree is None:
        tree = cKDTree(pts)

    slab_bins = int(getattr(config, "slab_coverage_bins", 12))
    guard_bbox_depth_u = float(
        mm_to_unit(float(max(0.05, float(getattr(config, "endpoint_regrow_guard_bbox_depth_mm", 0.8) or 0.8))), unit_scale)
    )
    cov_vals: List[float] = []
    rv_vals: List[float] = []
    support_vals: List[float] = []
    step_rows: List[Dict[str, float]] = []

    for k in range(1, corridor_steps + 1):
        rec: Dict[str, float] = {
            "step": float(k),
            "valid": 0.0,
            "n_cand": 0.0,
            "n_slab": 0.0,
            "support": 0.0,
            "cov": 0.0,
            "rv": float("inf"),
        }
        p_pred = np.asarray(c_body[0], dtype=float) + float(k * step_u) * t_out
        cand = tree.query_ball_point(p_pred, r=float(search_r))
        rec["n_cand"] = float(len(cand))
        if len(cand) < int(min_stop):
            step_rows.append(rec)
            continue

        local = pts[cand] - p_pred
        axial = local @ t_out
        mask = np.abs(axial) <= float(half_t_regrow)
        n_slab = int(np.sum(mask))
        rec["n_slab"] = float(n_slab)
        if n_slab < int(min_stop):
            step_rows.append(rec)
            continue

        local = local[mask]
        u, v = make_plane_basis(t_out, prev_u=None)
        coords_u = local @ u
        coords_v = local @ v
        cov_k = float(slab_angular_coverage(coords_u, coords_v, bins=slab_bins)) if slab_bins > 0 else 1.0
        rv_k = float(_radial_var_from_uv(coords_u, coords_v))
        rec["cov"] = float(cov_k)
        rec["rv"] = float(rv_k)

        try:
            u_lo, u_hi = np.quantile(coords_u, [0.10, 0.90])
            v_lo, v_hi = np.quantile(coords_v, [0.10, 0.90])
            half_u = float(max(0.5 * (u_hi - u_lo), guard_bbox_half_lat_floor_u))
            half_v = float(max(0.5 * (v_hi - v_lo), guard_bbox_half_lat_floor_u))
        except Exception:
            step_rows.append(rec)
            continue

        local_next = pts[cand] - p_pred
        axial_next = local_next @ t_out
        uu_next = local_next @ u
        vv_next = local_next @ v
        mask_lat = (np.abs(uu_next) <= half_u) & (np.abs(vv_next) <= half_v)
        n_out_axis = int(np.sum(mask_lat & (axial_next > 0.0) & (axial_next <= guard_bbox_depth_u)))

        rec["support"] = float(n_out_axis)
        rec["valid"] = 1.0
        step_rows.append(rec)

        cov_vals.append(float(cov_k))
        rv_vals.append(float(rv_k))
        support_vals.append(float(n_out_axis))

    out["step_rows"] = step_rows
    if not support_vals:
        out["reason"] = "no_valid_steps"
        return out

    out.update(
        {
            "valid": 1.0,
            "reason": "ok",
            "valid_steps": float(len(support_vals)),
            "support_sum": float(np.sum(np.asarray(support_vals, dtype=float))),
            "support_peak": float(np.max(np.asarray(support_vals, dtype=float))),
            "cov_mean": float(np.mean(np.asarray(cov_vals, dtype=float))) if cov_vals else 0.0,
            "rv_max": float(np.max(np.asarray(rv_vals, dtype=float))) if rv_vals else float("inf"),
        }
    )
    return out


def _endpoint_regrow_backward_evidence(
    points: np.ndarray,
    body_view: np.ndarray,
    config: CenterlineConfig,
    *,
    tree: Optional[cKDTree] = None,
    pin_width_hint: Optional[float] = None,
    half_t_regrow: Optional[float] = None,
    search_r: Optional[float] = None,
    step_u: Optional[float] = None,
) -> Dict[str, object]:
    """Probe inward support to classify keep(short distance) vs shorten(far distance)."""
    pts = np.asarray(points, dtype=float)
    c_body = np.asarray(body_view, dtype=float)
    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    denom = float(unit_scale + 1e-12)
    out: Dict[str, object] = {
        "valid": 0.0,
        "reason": "too_short",
        "scan_steps": 0.0,
        "near_steps": 0.0,
        "support_min": 0.0,
        "hit": 0.0,
        "near_hit": 0.0,
        "hit_step": 0.0,
        "hit_distance_mm": float("inf"),
        "hit_support": 0.0,
        "step_rows": [],
    }
    if len(pts) < 10 or len(c_body) < 4:
        return out

    if step_u is None:
        step_u = float(config.centerline_step())
    if search_r is None:
        search_r = float(config.slab_search_radius(pin_width_hint))
    if half_t_regrow is None:
        half_t_regrow = float(config.slab_half_thickness())

    min_stop = int(max(3, int(getattr(config, "endpoint_regrow_min_stop_points", 12) or 12)))
    support_min = int(max(1, int(getattr(config, "endpoint_regrow_guard_bbox_min_points", 4) or 4)))
    near_steps = int(max(1, int(getattr(config, "endpoint_regrow_triclass_backward_near_steps", 1) or 1)))
    scan_steps = int(max(near_steps, int(getattr(config, "endpoint_regrow_triclass_backward_scan_steps", 6) or 6)))

    inward_u = float(mm_to_unit(float(getattr(config, "trunc_end_pullback_tangent_inward_mm", 2.0)), unit_scale))
    max_pts = int(getattr(config, "trunc_end_pullback_tangent_max_points", 12))
    min_pts = int(getattr(config, "trunc_end_pullback_tangent_min_points", 4))
    s_body = _cum_arclen(c_body)
    t_in = tangent_toward_mid_pca(
        c_body,
        idx=0,
        mid_idx=int(len(c_body) - 1),
        s=s_body,
        inward_u=float(inward_u),
        max_points=int(max_pts),
        min_points=int(min_pts),
    )
    t_in = np.asarray(t_in, dtype=float)
    n_in = float(np.linalg.norm(t_in))
    if (not np.isfinite(n_in)) or n_in <= 1e-12:
        out["reason"] = "bad_tangent"
        return out
    t_in = t_in / n_in

    if tree is None:
        tree = cKDTree(pts)

    step_rows: List[Dict[str, object]] = []
    hit_step = 0
    hit_support = 0
    for k in range(1, int(scan_steps) + 1):
        p_probe = np.asarray(c_body[0], dtype=float) + float(k) * float(step_u) * t_in
        cand = tree.query_ball_point(p_probe, r=float(search_r))
        n_cand = int(len(cand))
        n_slab = 0
        if n_cand >= int(min_stop):
            local = pts[cand] - p_probe
            axial = local @ t_in
            n_slab = int(np.sum(np.abs(axial) <= float(half_t_regrow)))
        rec = {"step": int(k), "n_cand": int(n_cand), "n_slab": int(n_slab), "hit": 1 if n_slab >= int(support_min) else 0}
        step_rows.append(rec)
        if hit_step <= 0 and n_slab >= int(support_min):
            hit_step = int(k)
            hit_support = int(n_slab)

    out["valid"] = 1.0
    out["reason"] = "ok"
    out["scan_steps"] = float(scan_steps)
    out["near_steps"] = float(near_steps)
    out["support_min"] = float(support_min)
    out["step_rows"] = step_rows
    if hit_step > 0:
        out["hit"] = 1.0
        out["near_hit"] = 1.0 if hit_step <= int(near_steps) else 0.0
        out["hit_step"] = float(hit_step)
        out["hit_distance_mm"] = float((float(hit_step) * float(step_u)) / denom)
        out["hit_support"] = float(hit_support)
    else:
        out["reason"] = "no_backward_support"
    return out


def _endpoint_regrow_triclass_decide(
    *,
    side_name: str,
    endpoint_before: Optional[Dict[str, float]],
    lookahead: Dict[str, object],
    backward: Optional[Dict[str, object]],
    corridor: Optional[Dict[str, object]],
    cap_side: Optional[Dict[str, float]],
    base_end_idx_local: Optional[int],
    trim_target_mm: float,
    config: CenterlineConfig,
) -> Dict[str, object]:
    """Tri-class decision with simplified forward/backward evidence semantics."""
    support_n1 = float(lookahead.get("support_n1", 0.0) or 0.0)
    cov1 = float(lookahead.get("cov1", 0.0) or 0.0)
    radial_var1 = float(lookahead.get("radial_var1", float("inf")))
    lookahead_valid = bool(float(lookahead.get("valid", 0.0) or 0.0) > 0.5)
    lookahead_reason = str(lookahead.get("reason", "") or "")

    backward_d = dict(backward or {})
    backward_valid = bool(float(backward_d.get("valid", 0.0) or 0.0) > 0.5)
    backward_reason = str(backward_d.get("reason", "") or "")
    backward_hit = bool(float(backward_d.get("hit", 0.0) or 0.0) > 0.5)
    backward_near_hit = bool(float(backward_d.get("near_hit", 0.0) or 0.0) > 0.5)
    backward_hit_step = float(backward_d.get("hit_step", 0.0) or 0.0)
    backward_hit_distance_mm = float(backward_d.get("hit_distance_mm", float("inf")))
    backward_support_n = float(backward_d.get("hit_support", 0.0) or 0.0)
    backward_near_steps = float(backward_d.get("near_steps", 1.0) or 1.0)

    forward_min_points = float(max(1, int(getattr(config, "endpoint_regrow_guard_bbox_min_points", 4) or 4)))
    forward_hit = bool(lookahead_valid and support_n1 >= forward_min_points)

    corridor_d = dict(corridor or {})
    corridor_valid = bool(float(corridor_d.get("valid", 0.0) or 0.0) > 0.5)
    corridor_reason = str(corridor_d.get("reason", "") or "")
    corridor_valid_steps = float(corridor_d.get("valid_steps", 0.0) or 0.0)
    corridor_support_sum = float(corridor_d.get("support_sum", 0.0) or 0.0)
    corridor_support_peak = float(corridor_d.get("support_peak", 0.0) or 0.0)
    corridor_cov_mean = float(corridor_d.get("cov_mean", 0.0) or 0.0)
    corridor_rv_max = float(corridor_d.get("rv_max", float("inf")))

    risk_stats = dict(endpoint_before or {})
    try:
        risk_detail = _endpoint_risk_detail(risk_stats, config) if risk_stats else {}
        risk_fail_count = float(risk_detail.get("fail_count", 0.0) or 0.0)
    except Exception:
        risk_detail = {}
        risk_fail_count = 0.0

    risk_hit_min_n_slab = float(risk_detail.get("hit_min_n_slab", 0.0) or 0.0)
    risk_hit_min_cov = float(risk_detail.get("hit_min_cov", 0.0) or 0.0)
    risk_hit_max_delta = float(risk_detail.get("hit_max_delta", 0.0) or 0.0)
    risk_hit_clipped = float(risk_detail.get("hit_clipped", 0.0) or 0.0)
    risk_hit_max_rv = float(risk_detail.get("hit_max_rv", 0.0) or 0.0)
    risk_hits: List[str] = []
    if risk_hit_min_n_slab > 0.5:
        risk_hits.append("min_n_slab<n_th")
    if risk_hit_min_cov > 0.5:
        risk_hits.append("min_coverage<cov_th")
    if risk_hit_max_delta > 0.5:
        risk_hits.append("max_delta_mm>delta_th")
    if risk_hit_clipped > 0.5:
        risk_hits.append("clipped_rate>clip_th")
    if risk_hit_max_rv > 0.5:
        risk_hits.append("max_radial_var>rv_th")
    risk_hit_text = "|".join(risk_hits) if risk_hits else "none"

    extend_cov_min = float(getattr(config, "endpoint_regrow_triclass_extend_cov_min", 0.60) or 0.60)
    extend_support_min = float(getattr(config, "endpoint_regrow_triclass_extend_support_n1_min", 15) or 15)
    extend_rv_max = float(getattr(config, "endpoint_regrow_triclass_extend_radial_var_max", 1.00) or 1.00)
    corridor_enabled = bool(getattr(config, "endpoint_regrow_triclass_corridor_enabled", True))
    corridor_valid_steps_min = float(
        max(1, int(getattr(config, "endpoint_regrow_triclass_corridor_valid_steps_min", 2) or 2))
    )
    corridor_support_sum_min = float(
        max(0, int(getattr(config, "endpoint_regrow_triclass_corridor_support_sum_min", 36) or 36))
    )
    corridor_support_peak_min = float(
        max(0, int(getattr(config, "endpoint_regrow_triclass_corridor_support_peak_min", 12) or 12))
    )
    corridor_cov_min = float(getattr(config, "endpoint_regrow_triclass_corridor_cov_min", 0.45) or 0.45)
    corridor_rv_max_th = float(getattr(config, "endpoint_regrow_triclass_corridor_rv_max", 1.30) or 1.30)
    shorten_fail_count = float(getattr(config, "endpoint_regrow_triclass_shorten_fail_count", 3) or 3)
    trim_min_action_mm = float(getattr(config, "endpoint_regrow_triclass_min_action_trim_mm", 0.05) or 0.05)
    cap_reliability_min = float(getattr(config, "endpoint_regrow_triclass_cap_reliability_min", 0.60) or 0.60)

    adaptive_enabled = bool(getattr(config, "endpoint_regrow_triclass_adaptive_enabled", True))
    adaptive_min_samples = int(max(2, int(getattr(config, "endpoint_regrow_triclass_adaptive_min_samples", 3) or 3)))

    support_obs: List[float] = [float(max(0.0, support_n1))] if np.isfinite(support_n1) else []
    cov_obs: List[float] = [float(cov1)] if np.isfinite(cov1) else []
    rv_obs: List[float] = [float(radial_var1)] if np.isfinite(radial_var1) else []

    extend_support_min_eff = float(extend_support_min)
    extend_cov_min_eff = float(extend_cov_min)
    extend_rv_max_eff = float(extend_rv_max)
    corridor_valid_steps_min_eff = float(corridor_valid_steps_min)
    corridor_support_sum_min_eff = float(corridor_support_sum_min)
    corridor_support_peak_min_eff = float(corridor_support_peak_min)
    corridor_cov_min_eff = float(corridor_cov_min)
    corridor_rv_max_th_eff = float(corridor_rv_max_th)
    adaptive_used = False

    corridor_pass = bool(
        corridor_enabled
        and corridor_valid
        and corridor_valid_steps >= corridor_valid_steps_min_eff
        and corridor_support_sum >= corridor_support_sum_min_eff
        and corridor_support_peak >= corridor_support_peak_min_eff
        and corridor_cov_mean >= corridor_cov_min_eff
        and np.isfinite(corridor_rv_max)
        and corridor_rv_max <= corridor_rv_max_th_eff
    )

    side_end_idx = 0 if str(side_name).strip().lower() == "start" else -1
    is_base_side = bool(base_end_idx_local is not None and int(base_end_idx_local) == int(side_end_idx))
    cap_d = dict(cap_side or {})
    cap_reliability = float(cap_d.get("reliability", 0.0) or 0.0)

    action = "keep"
    hard_reason = "ambiguous"
    extend_mode = "none"

    if forward_hit:
        action = "extend"
        hard_reason = "forward_support"
        extend_mode = "normal"
    elif backward_hit and backward_near_hit:
        action = "keep"
        hard_reason = "backward_near_keep"
    elif backward_hit and trim_target_mm > trim_min_action_mm + 1e-12:
        action = "shorten"
        hard_reason = "backward_far_shorten"
    elif backward_hit:
        action = "keep"
        hard_reason = "backward_far_but_trim_small_keep"
    elif (not is_base_side) and cap_reliability >= cap_reliability_min and trim_target_mm <= trim_min_action_mm + 1e-12:
        action = "keep"
        hard_reason = "free_end_trim_small"
    else:
        action = "keep"
        hard_reason = "forward_empty_backward_empty_keep"

    if action == "extend" and risk_fail_count >= shorten_fail_count:
        extend_mode = "hard_axial"
        hard_reason = f"{hard_reason}|risk_hard_axial"

    explain = (
        f"risk_fail={risk_fail_count:.0f} hits={risk_hit_text}, trim={trim_target_mm:.2f}mm, "
        f"fwd(hit={int(forward_hit)},n1={support_n1:.0f},th={forward_min_points:.0f}), "
        f"back(hit={int(backward_hit)},near={int(backward_near_hit)},step={backward_hit_step:.0f},dist={backward_hit_distance_mm:.2f}mm,n={backward_support_n:.0f}), "
        f"look(cov1={cov1:.2f},rv1={radial_var1:.2f},valid={int(lookahead_valid)}), "
        f"thr(sup>={extend_support_min_eff:.1f},cov>={extend_cov_min_eff:.2f},rv<={extend_rv_max_eff:.2f},adapt={int(adaptive_used)}), "
        f"corridor(valid={int(corridor_valid)},steps={corridor_valid_steps:.0f},sum={corridor_support_sum:.0f},"
        f"peak={corridor_support_peak:.0f},cov={corridor_cov_mean:.2f},rv={corridor_rv_max:.2f};"
        f"thr:steps>={corridor_valid_steps_min_eff:.0f},sum>={corridor_support_sum_min_eff:.0f},"
        f"peak>={corridor_support_peak_min_eff:.0f},cov>={corridor_cov_min_eff:.2f},rv<={corridor_rv_max_th_eff:.2f}), "
        f"cap_rel={cap_reliability:.2f}, base_side={int(is_base_side)}"
    )

    return {
        "action": str(action),
        "extend_mode": str(extend_mode),
        "hard_reason": str(hard_reason),
        "explain": str(explain),
        "lookahead_valid": 1.0 if lookahead_valid else 0.0,
        "lookahead_reason": str(lookahead_reason),
        "forward_hit": 1.0 if forward_hit else 0.0,
        "forward_min_points": float(forward_min_points),
        "forward_support_n": float(support_n1),
        "backward_valid": 1.0 if backward_valid else 0.0,
        "backward_reason": str(backward_reason),
        "backward_hit": 1.0 if backward_hit else 0.0,
        "backward_near_hit": 1.0 if backward_near_hit else 0.0,
        "backward_hit_step": float(backward_hit_step),
        "backward_hit_distance_mm": float(backward_hit_distance_mm),
        "backward_support_n": float(backward_support_n),
        "backward_near_steps": float(backward_near_steps),
        "support_n1": float(support_n1),
        "cov1": float(cov1),
        "radial_var1": float(radial_var1),
        "corridor_valid": 1.0 if corridor_valid else 0.0,
        "corridor_reason": str(corridor_reason),
        "corridor_valid_steps": float(corridor_valid_steps),
        "corridor_support_sum": float(corridor_support_sum),
        "corridor_support_peak": float(corridor_support_peak),
        "corridor_cov_mean": float(corridor_cov_mean),
        "corridor_rv_max": float(corridor_rv_max),
        "corridor_pass": 1.0 if corridor_pass else 0.0,
        "adaptive_enabled": 1.0 if adaptive_enabled else 0.0,
        "adaptive_used": 1.0 if adaptive_used else 0.0,
        "adaptive_min_samples": float(adaptive_min_samples),
        "adaptive_support_obs_n": float(len(support_obs)),
        "adaptive_cov_obs_n": float(len(cov_obs)),
        "adaptive_rv_obs_n": float(len(rv_obs)),
        "extend_support_min_eff": float(extend_support_min_eff),
        "extend_cov_min_eff": float(extend_cov_min_eff),
        "extend_rv_max_eff": float(extend_rv_max_eff),
        "corridor_valid_steps_min_eff": float(corridor_valid_steps_min_eff),
        "corridor_support_sum_min_eff": float(corridor_support_sum_min_eff),
        "corridor_support_peak_min_eff": float(corridor_support_peak_min_eff),
        "corridor_cov_min_eff": float(corridor_cov_min_eff),
        "corridor_rv_max_th_eff": float(corridor_rv_max_th_eff),
        "risk_fail_count": float(risk_fail_count),
        "risk_hit_min_n_slab": float(risk_hit_min_n_slab),
        "risk_hit_min_cov": float(risk_hit_min_cov),
        "risk_hit_max_delta": float(risk_hit_max_delta),
        "risk_hit_clipped": float(risk_hit_clipped),
        "risk_hit_max_rv": float(risk_hit_max_rv),
        "risk_hit_text": str(risk_hit_text),
        "trim_target_mm": float(trim_target_mm),
        "cap_reliability": float(cap_reliability),
        "is_base_side": 1.0 if is_base_side else 0.0,
    }


def regrow_endpoints_body_guided(

    points: np.ndarray,
    curve: np.ndarray,
    config: CenterlineConfig,
    *,
    pin_width_hint: Optional[float] = None,
    pin_thickness_hint: Optional[float] = None,
    debug_tag: Optional[str] = None,
    segment_id: Optional[int] = None,
    cap_feat: Optional[Dict[str, Dict[str, float]]] = None,
    base_end_idx_local: Optional[int] = None,
    endpoint0_before: Optional[Dict[str, float]] = None,
    endpoint1_before: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Stage13: Body-guided endpoint regrowth.

    - Trim both ends by `endpoint_regrow_trim_mm` to remove unreliable drifted end windows.
    - Regrow outward from the trimmed body using curve-only tangents (toward-mid PCA) and slab evidence when reliable.
    - Stops automatically when slab evidence disappears.

    Returns: (curve_new, stats)
    """
    enabled = bool(getattr(config, "endpoint_regrow_enabled", False))
    if not enabled:
        return np.asarray(curve, dtype=float), {"applied": 0.0}

    pts = np.asarray(points, dtype=float)
    c0 = np.asarray(curve, dtype=float).copy()
    if len(pts) < 10 or len(c0) < 6:
        return c0, {"applied": 0.0, "reason": "too_short"}

    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    denom = float(unit_scale + 1e-12)

    trim_mm = float(getattr(config, "endpoint_regrow_trim_mm", 0.0) or 0.0)
    trim_u = float(mm_to_unit(trim_mm, unit_scale))

    deflect_trim_enabled = bool(getattr(config, "endpoint_regrow_deflect_trim_enabled", True))
    start_deflect: Dict[str, float] = {
        "detected": 0.0,
        "trim_u": float(trim_u),
        "trim_mm": float(trim_mm),
        "scan_mm": float(trim_mm),
        "onset_idx": -1.0,
        "onset_mm": 0.0,
        "angle_peak_deg": 0.0,
        "lateral_peak_mm": 0.0,
        "run_len": 0.0,
    }
    end_deflect: Dict[str, float] = dict(start_deflect)
    trim_start_req_u = float(trim_u)
    trim_end_req_u = float(trim_u)
    if deflect_trim_enabled:
        start_deflect = _detect_endpoint_deflection_trim(c0, config, max_trim_u=float(trim_u))
        end_deflect = _detect_endpoint_deflection_trim(c0[::-1].copy(), config, max_trim_u=float(trim_u))
        trim_start_req_u = float(start_deflect.get("trim_u", 0.0) or 0.0)
        trim_end_req_u = float(end_deflect.get("trim_u", 0.0) or 0.0)

    body, trim_s_u, trim_e_u, _i0, _i1 = _trim_curve_by_arclen(
        c0,
        trim_start_u=float(trim_start_req_u),
        trim_end_u=float(trim_end_req_u),
    )
    if len(body) < 4:
        return c0, {"applied": 0.0, "reason": "no_trim"}

    # Slab params for regrowth.
    search_r = float(config.slab_search_radius(pin_width_hint))
    half_t = float(config.slab_half_thickness())
    ht_factor = float(max(1.0, float(getattr(config, "endpoint_regrow_half_thickness_factor", 1.0) or 1.0)))
    half_t_regrow = float(half_t * ht_factor)
    step_u = float(config.centerline_step())
    max_extend_mm = float(getattr(config, "endpoint_regrow_max_extend_mm", 0.0) or 0.0)
    max_extend_u = float(mm_to_unit(max_extend_mm, unit_scale))
    max_steps = int(max(0, int(np.ceil(max_extend_u / (step_u + 1e-12))))) if max_extend_u > 1e-12 else 0
    max_steps = int(max(0, max_steps))

    tree = cKDTree(pts)

    tri_enabled_cfg = bool(getattr(config, "endpoint_regrow_triclass_enabled", True))
    tri_variant = str(getattr(config, "endpoint_regrow_triclass_variant", "v1") or "v1").strip().lower()
    tri_active = bool(tri_enabled_cfg and tri_variant == "v1")
    tri_ambiguous_policy = str(
        getattr(config, "endpoint_regrow_triclass_ambiguous_policy", "strict_keep") or "strict_keep"
    ).strip().lower()
    tri_shorten_extra_mm = float(getattr(config, "endpoint_regrow_triclass_shorten_extra_mm", 0.40) or 0.40)
    tri_shorten_extra_u = float(mm_to_unit(max(0.0, tri_shorten_extra_mm), unit_scale))
    tri_hard_axis_bbox_grace_steps = int(
        max(0, int(getattr(config, "endpoint_regrow_triclass_hard_axis_bbox_grace_steps", 2) or 2))
    )

    def _empty_end_stats(stop_reason: str) -> Dict[str, object]:
        return {
            "applied": 0.0,
            "center_mode": str(getattr(config, "endpoint_regrow_center_mode", "midrange") or "midrange"),
            "force_pred_only": 0.0,
            "steps": 0,
            "grown_mm": 0.0,
            "stop": str(stop_reason),
            "last_n_slab": 0,
            "last_cov": 0.0,
            "soft_events": 0,
            "guard_hits": 0,
            "guard_bbox_fail": 0,
            "guard_polar_fail": 0,
            "sparse_triggered": 0,
            "sparse_confirmed": 0,
            "sparse_anchor_step": 0,
            "bbox_probe_used": 0,
            "direction_lock_triggered": 0,
            "sparse_rollback_applied": 0,
            "steps_detail": [],
        }

    trim_start_req_u_raw = float(trim_start_req_u)
    trim_end_req_u_raw = float(trim_end_req_u)
    trim_start_req_u_eff = float(trim_start_req_u)
    trim_end_req_u_eff = float(trim_end_req_u)

    tri_start: Dict[str, object] = {
        "action": "extend",
        "extend_mode": "normal",
        "hard_reason": "disabled",
        "explain": "",
        "support_n1": 0.0,
        "cov1": 0.0,
        "radial_var1": float("inf"),
        "corridor_valid": 0.0,
        "corridor_reason": "disabled",
        "corridor_valid_steps": 0.0,
        "corridor_support_sum": 0.0,
        "corridor_support_peak": 0.0,
        "corridor_cov_mean": 0.0,
        "corridor_rv_max": float("inf"),
        "corridor_pass": 0.0,
        "risk_fail_count": 0.0,
        "lookahead_valid": 0.0,
        "lookahead_reason": "disabled",
        "trim_target_mm": float(trim_start_req_u / denom),
        "cap_reliability": 0.0,
        "is_base_side": 0.0,
    }
    tri_end: Dict[str, object] = {
        "action": "extend",
        "extend_mode": "normal",
        "hard_reason": "disabled",
        "explain": "",
        "support_n1": 0.0,
        "cov1": 0.0,
        "radial_var1": float("inf"),
        "corridor_valid": 0.0,
        "corridor_reason": "disabled",
        "corridor_valid_steps": 0.0,
        "corridor_support_sum": 0.0,
        "corridor_support_peak": 0.0,
        "corridor_cov_mean": 0.0,
        "corridor_rv_max": float("inf"),
        "corridor_pass": 0.0,
        "risk_fail_count": 0.0,
        "lookahead_valid": 0.0,
        "lookahead_reason": "disabled",
        "trim_target_mm": float(trim_end_req_u / denom),
        "cap_reliability": 0.0,
        "is_base_side": 0.0,
    }
    tri_shorten_start_applied = 0.0
    tri_shorten_end_applied = 0.0

    if tri_active:
        cap_d = dict(cap_feat or {})
        look_start = _endpoint_regrow_one_step_lookahead(
            pts,
            body,
            config,
            tree=tree,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
        )
        look_end = _endpoint_regrow_one_step_lookahead(
            pts,
            body[::-1].copy(),
            config,
            tree=tree,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
        )
        back_start = _endpoint_regrow_backward_evidence(
            pts,
            body,
            config,
            tree=tree,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
        )
        back_end = _endpoint_regrow_backward_evidence(
            pts,
            body[::-1].copy(),
            config,
            tree=tree,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
        )
        corridor_start = _endpoint_regrow_corridor_lookahead(
            pts,
            body,
            config,
            tree=tree,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
        )
        corridor_end = _endpoint_regrow_corridor_lookahead(
            pts,
            body[::-1].copy(),
            config,
            tree=tree,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
        )

        tri_start = _endpoint_regrow_triclass_decide(
            side_name="start",
            endpoint_before=endpoint0_before,
            lookahead=look_start,
            backward=back_start,
            corridor=corridor_start,
            cap_side=(cap_d.get("start") if isinstance(cap_d.get("start"), dict) else {}),
            base_end_idx_local=base_end_idx_local,
            trim_target_mm=float(trim_start_req_u / denom),
            config=config,
        )
        tri_end = _endpoint_regrow_triclass_decide(
            side_name="end",
            endpoint_before=endpoint1_before,
            lookahead=look_end,
            backward=back_end,
            corridor=corridor_end,
            cap_side=(cap_d.get("end") if isinstance(cap_d.get("end"), dict) else {}),
            base_end_idx_local=base_end_idx_local,
            trim_target_mm=float(trim_end_req_u / denom),
            config=config,
        )

        if str(tri_start.get("action", "keep")) == "shorten":
            trim_start_req_u_eff = float(trim_start_req_u_eff + tri_shorten_extra_u)
            tri_shorten_start_applied = 1.0
        if str(tri_end.get("action", "keep")) == "shorten":
            trim_end_req_u_eff = float(trim_end_req_u_eff + tri_shorten_extra_u)
            tri_shorten_end_applied = 1.0

        if tri_shorten_start_applied > 0.5 or tri_shorten_end_applied > 0.5:
            body2, trim_s_u2, trim_e_u2, _j0, _j1 = _trim_curve_by_arclen(
                c0,
                trim_start_u=float(trim_start_req_u_eff),
                trim_end_u=float(trim_end_req_u_eff),
            )
            if len(body2) >= 4:
                body = np.asarray(body2, dtype=float)
                trim_s_u = float(trim_s_u2)
                trim_e_u = float(trim_e_u2)
                trim_start_req_u = float(trim_start_req_u_eff)
                trim_end_req_u = float(trim_end_req_u_eff)
            else:
                if tri_shorten_start_applied > 0.5:
                    tri_start["hard_reason"] = str(tri_start.get("hard_reason", "")) + "|shorten_rejected_short_body"
                    tri_start["action"] = "keep"
                if tri_shorten_end_applied > 0.5:
                    tri_end["hard_reason"] = str(tri_end.get("hard_reason", "")) + "|shorten_rejected_short_body"
                    tri_end["action"] = "keep"
                tri_shorten_start_applied = 0.0
                tri_shorten_end_applied = 0.0
                trim_start_req_u_eff = float(trim_start_req_u_raw)
                trim_end_req_u_eff = float(trim_end_req_u_raw)
                trim_start_req_u = float(trim_start_req_u_raw)
                trim_end_req_u = float(trim_end_req_u_raw)
    else:
        tri_start["hard_reason"] = "disabled_or_variant"
        tri_end["hard_reason"] = "disabled_or_variant"

    start_enable_extend = bool(str(tri_start.get("action", "extend")).strip().lower() == "extend")
    end_enable_extend = bool(str(tri_end.get("action", "extend")).strip().lower() == "extend")
    start_force_pred_only = bool(str(tri_start.get("extend_mode", "normal")).strip().lower() == "hard_axial")
    end_force_pred_only = bool(str(tri_end.get("extend_mode", "normal")).strip().lower() == "hard_axial")

    # Debug dump: body seed only (stable prior).
    dbg_on = _debug_enabled(config, segment_id) and bool(debug_tag)
    if dbg_on and debug_tag:
        try:
            dbg_dir = _debug_segment_dir(config, str(debug_tag))
            _debug_write_ply_points(dbg_dir / "stage13_seed_body.ply", body, color=[0.10, 0.10, 0.10])
        except Exception:
            pass

    if start_enable_extend:
        start_grown, start_stats = _regrow_one_end_body_guided(
            pts,
            tree,
            body_view=body,
            config=config,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
            max_steps=int(max_steps),
            force_pred_only=bool(start_force_pred_only),
            guard_bbox_grace_steps=int(tri_hard_axis_bbox_grace_steps if start_force_pred_only else 0),
        )
    else:
        start_grown = []
        start_stats = _empty_end_stats(
            "tri_shorten" if str(tri_start.get("action", "")) == "shorten" else "tri_keep"
        )

    if end_enable_extend:
        end_grown, end_stats = _regrow_one_end_body_guided(
            pts,
            tree,
            body_view=body[::-1].copy(),
            config=config,
            pin_width_hint=pin_width_hint,
            half_t_regrow=float(half_t_regrow),
            search_r=float(search_r),
            step_u=float(step_u),
            max_steps=int(max_steps),
            force_pred_only=bool(end_force_pred_only),
            guard_bbox_grace_steps=int(tri_hard_axis_bbox_grace_steps if end_force_pred_only else 0),
        )
    else:
        end_grown = []
        end_stats = _empty_end_stats(
            "tri_shorten" if str(tri_end.get("action", "")) == "shorten" else "tri_keep"
        )

    if dbg_on and bool(getattr(config, "endpoint_regrow_dump_steps", True)) and debug_tag:
        try:
            dbg_dir = _debug_segment_dir(config, str(debug_tag))
            _debug_write_regrow_steps(dbg_dir, "start", list(start_stats.get("steps_detail", []) or []))
            _debug_write_regrow_steps(dbg_dir, "end", list(end_stats.get("steps_detail", []) or []))
        except Exception:
            pass

    # Combine (keep ordering): start outward points (far->near) + body + end outward points (near->far).
    parts = []
    if start_grown:
        parts.append(np.stack(list(reversed(start_grown)), axis=0))
    parts.append(np.asarray(body, dtype=float))
    if end_grown:
        parts.append(np.stack(end_grown, axis=0))

    out = np.concatenate(parts, axis=0) if parts else np.asarray(c0, dtype=float)
    # Light resample to enforce step consistency (prevents tiny kinks at the trim boundary).
    try:
        out = resample_polyline_by_step(out, step=float(step_u))
    except Exception:
        out = np.asarray(out, dtype=float)

    applied = 1.0 if (start_grown or end_grown) else 0.0
    curve_changed = 0.0
    if tri_active:
        try:
            if len(out) != len(c0):
                curve_changed = 1.0
            elif len(out) >= 2 and len(c0) >= 2:
                d0 = float(np.linalg.norm(np.asarray(out[0], dtype=float) - np.asarray(c0[0], dtype=float)))
                d1 = float(np.linalg.norm(np.asarray(out[-1], dtype=float) - np.asarray(c0[-1], dtype=float)))
                if d0 > 1e-12 or d1 > 1e-12:
                    curve_changed = 1.0
        except Exception:
            curve_changed = 1.0 if len(out) != len(c0) else 0.0

    stats: Dict[str, object] = {
        "applied": float(applied),
        "curve_changed": float(curve_changed),
        "center_mode": str(getattr(config, "endpoint_regrow_center_mode", "midrange") or "midrange"),
        "trim_mm_target": float(trim_mm),
        "trim_start_target_mm_raw": float(trim_start_req_u_raw / denom),
        "trim_end_target_mm_raw": float(trim_end_req_u_raw / denom),
        "trim_start_target_mm": float(trim_start_req_u / denom),
        "trim_end_target_mm": float(trim_end_req_u / denom),
        "trim_start_mm": float(trim_s_u / denom),
        "trim_end_mm": float(trim_e_u / denom),
        "deflect_trim_enabled": float(1.0 if deflect_trim_enabled else 0.0),
        "deflect_start_detected": float(start_deflect.get("detected", 0.0) or 0.0),
        "deflect_end_detected": float(end_deflect.get("detected", 0.0) or 0.0),
        "deflect_start_onset_mm": float(start_deflect.get("onset_mm", 0.0) or 0.0),
        "deflect_end_onset_mm": float(end_deflect.get("onset_mm", 0.0) or 0.0),
        "deflect_start_angle_peak_deg": float(start_deflect.get("angle_peak_deg", 0.0) or 0.0),
        "deflect_end_angle_peak_deg": float(end_deflect.get("angle_peak_deg", 0.0) or 0.0),
        "deflect_start_lateral_peak_mm": float(start_deflect.get("lateral_peak_mm", 0.0) or 0.0),
        "deflect_end_lateral_peak_mm": float(end_deflect.get("lateral_peak_mm", 0.0) or 0.0),
        "deflect_start_run_len": float(start_deflect.get("run_len", 0.0) or 0.0),
        "deflect_end_run_len": float(end_deflect.get("run_len", 0.0) or 0.0),
        "half_thickness_factor": float(ht_factor),
        "max_extend_mm": float(max_extend_mm),
        "triclass_enabled": float(1.0 if tri_active else 0.0),
        "triclass_variant": str(tri_variant),
        "triclass_ambiguous_policy": str(tri_ambiguous_policy),
        "triclass_shorten_extra_mm": float(tri_shorten_extra_mm),
        "triclass_hard_axis_bbox_grace_steps": int(tri_hard_axis_bbox_grace_steps),
        "triclass_start_adaptive_used": float(tri_start.get("adaptive_used", 0.0) or 0.0),
        "triclass_end_adaptive_used": float(tri_end.get("adaptive_used", 0.0) or 0.0),
        "triclass_start_extend_support_min_eff": float(tri_start.get("extend_support_min_eff", 0.0) or 0.0),
        "triclass_end_extend_support_min_eff": float(tri_end.get("extend_support_min_eff", 0.0) or 0.0),
        "triclass_start_extend_cov_min_eff": float(tri_start.get("extend_cov_min_eff", 0.0) or 0.0),
        "triclass_end_extend_cov_min_eff": float(tri_end.get("extend_cov_min_eff", 0.0) or 0.0),
        "triclass_start_extend_rv_max_eff": float(tri_start.get("extend_rv_max_eff", 0.0) or 0.0),
        "triclass_end_extend_rv_max_eff": float(tri_end.get("extend_rv_max_eff", 0.0) or 0.0),
        "triclass_start_action": str(tri_start.get("action", "keep")),
        "triclass_start_extend_mode": str(tri_start.get("extend_mode", "none")),
        "triclass_start_hard_reason": str(tri_start.get("hard_reason", "")),
        "triclass_start_explain": str(tri_start.get("explain", "")),
        "triclass_start_lookahead_valid": float(tri_start.get("lookahead_valid", 0.0) or 0.0),
        "triclass_start_lookahead_reason": str(tri_start.get("lookahead_reason", "")),
        "triclass_start_forward_hit": float(tri_start.get("forward_hit", 0.0) or 0.0),
        "triclass_start_forward_min_points": float(tri_start.get("forward_min_points", 0.0) or 0.0),
        "triclass_start_forward_support_n": float(tri_start.get("forward_support_n", 0.0) or 0.0),
        "triclass_start_backward_valid": float(tri_start.get("backward_valid", 0.0) or 0.0),
        "triclass_start_backward_reason": str(tri_start.get("backward_reason", "")),
        "triclass_start_backward_hit": float(tri_start.get("backward_hit", 0.0) or 0.0),
        "triclass_start_backward_near_hit": float(tri_start.get("backward_near_hit", 0.0) or 0.0),
        "triclass_start_backward_hit_step": float(tri_start.get("backward_hit_step", 0.0) or 0.0),
        "triclass_start_backward_hit_distance_mm": float(tri_start.get("backward_hit_distance_mm", 0.0) or 0.0),
        "triclass_start_backward_support_n": float(tri_start.get("backward_support_n", 0.0) or 0.0),
        "triclass_start_backward_near_steps": float(tri_start.get("backward_near_steps", 0.0) or 0.0),
        "triclass_start_support_n1": float(tri_start.get("support_n1", 0.0) or 0.0),
        "triclass_start_cov1": float(tri_start.get("cov1", 0.0) or 0.0),
        "triclass_start_radial_var1": float(tri_start.get("radial_var1", float("inf"))),
        "triclass_start_corridor_valid": float(tri_start.get("corridor_valid", 0.0) or 0.0),
        "triclass_start_corridor_reason": str(tri_start.get("corridor_reason", "")),
        "triclass_start_corridor_valid_steps": float(tri_start.get("corridor_valid_steps", 0.0) or 0.0),
        "triclass_start_corridor_support_sum": float(tri_start.get("corridor_support_sum", 0.0) or 0.0),
        "triclass_start_corridor_support_peak": float(tri_start.get("corridor_support_peak", 0.0) or 0.0),
        "triclass_start_corridor_cov_mean": float(tri_start.get("corridor_cov_mean", 0.0) or 0.0),
        "triclass_start_corridor_rv_max": float(tri_start.get("corridor_rv_max", float("inf"))),
        "triclass_start_corridor_pass": float(tri_start.get("corridor_pass", 0.0) or 0.0),
        "triclass_start_risk_fail_count": float(tri_start.get("risk_fail_count", 0.0) or 0.0),
        "triclass_start_risk_hit_min_n_slab": float(tri_start.get("risk_hit_min_n_slab", 0.0) or 0.0),
        "triclass_start_risk_hit_min_cov": float(tri_start.get("risk_hit_min_cov", 0.0) or 0.0),
        "triclass_start_risk_hit_max_delta": float(tri_start.get("risk_hit_max_delta", 0.0) or 0.0),
        "triclass_start_risk_hit_clipped": float(tri_start.get("risk_hit_clipped", 0.0) or 0.0),
        "triclass_start_risk_hit_max_rv": float(tri_start.get("risk_hit_max_rv", 0.0) or 0.0),
        "triclass_start_risk_hit_text": str(tri_start.get("risk_hit_text", "")),
        "triclass_start_trim_target_mm": float(tri_start.get("trim_target_mm", 0.0) or 0.0),
        "triclass_start_cap_reliability": float(tri_start.get("cap_reliability", 0.0) or 0.0),
        "triclass_start_is_base_side": float(tri_start.get("is_base_side", 0.0) or 0.0),
        "triclass_end_action": str(tri_end.get("action", "keep")),
        "triclass_end_extend_mode": str(tri_end.get("extend_mode", "none")),
        "triclass_end_hard_reason": str(tri_end.get("hard_reason", "")),
        "triclass_end_explain": str(tri_end.get("explain", "")),
        "triclass_end_lookahead_valid": float(tri_end.get("lookahead_valid", 0.0) or 0.0),
        "triclass_end_lookahead_reason": str(tri_end.get("lookahead_reason", "")),
        "triclass_end_forward_hit": float(tri_end.get("forward_hit", 0.0) or 0.0),
        "triclass_end_forward_min_points": float(tri_end.get("forward_min_points", 0.0) or 0.0),
        "triclass_end_forward_support_n": float(tri_end.get("forward_support_n", 0.0) or 0.0),
        "triclass_end_backward_valid": float(tri_end.get("backward_valid", 0.0) or 0.0),
        "triclass_end_backward_reason": str(tri_end.get("backward_reason", "")),
        "triclass_end_backward_hit": float(tri_end.get("backward_hit", 0.0) or 0.0),
        "triclass_end_backward_near_hit": float(tri_end.get("backward_near_hit", 0.0) or 0.0),
        "triclass_end_backward_hit_step": float(tri_end.get("backward_hit_step", 0.0) or 0.0),
        "triclass_end_backward_hit_distance_mm": float(tri_end.get("backward_hit_distance_mm", 0.0) or 0.0),
        "triclass_end_backward_support_n": float(tri_end.get("backward_support_n", 0.0) or 0.0),
        "triclass_end_backward_near_steps": float(tri_end.get("backward_near_steps", 0.0) or 0.0),
        "triclass_end_support_n1": float(tri_end.get("support_n1", 0.0) or 0.0),
        "triclass_end_cov1": float(tri_end.get("cov1", 0.0) or 0.0),
        "triclass_end_radial_var1": float(tri_end.get("radial_var1", float("inf"))),
        "triclass_end_corridor_valid": float(tri_end.get("corridor_valid", 0.0) or 0.0),
        "triclass_end_corridor_reason": str(tri_end.get("corridor_reason", "")),
        "triclass_end_corridor_valid_steps": float(tri_end.get("corridor_valid_steps", 0.0) or 0.0),
        "triclass_end_corridor_support_sum": float(tri_end.get("corridor_support_sum", 0.0) or 0.0),
        "triclass_end_corridor_support_peak": float(tri_end.get("corridor_support_peak", 0.0) or 0.0),
        "triclass_end_corridor_cov_mean": float(tri_end.get("corridor_cov_mean", 0.0) or 0.0),
        "triclass_end_corridor_rv_max": float(tri_end.get("corridor_rv_max", float("inf"))),
        "triclass_end_corridor_pass": float(tri_end.get("corridor_pass", 0.0) or 0.0),
        "triclass_end_risk_fail_count": float(tri_end.get("risk_fail_count", 0.0) or 0.0),
        "triclass_end_risk_hit_min_n_slab": float(tri_end.get("risk_hit_min_n_slab", 0.0) or 0.0),
        "triclass_end_risk_hit_min_cov": float(tri_end.get("risk_hit_min_cov", 0.0) or 0.0),
        "triclass_end_risk_hit_max_delta": float(tri_end.get("risk_hit_max_delta", 0.0) or 0.0),
        "triclass_end_risk_hit_clipped": float(tri_end.get("risk_hit_clipped", 0.0) or 0.0),
        "triclass_end_risk_hit_max_rv": float(tri_end.get("risk_hit_max_rv", 0.0) or 0.0),
        "triclass_end_risk_hit_text": str(tri_end.get("risk_hit_text", "")),
        "triclass_end_trim_target_mm": float(tri_end.get("trim_target_mm", 0.0) or 0.0),
        "triclass_end_cap_reliability": float(tri_end.get("cap_reliability", 0.0) or 0.0),
        "triclass_end_is_base_side": float(tri_end.get("is_base_side", 0.0) or 0.0),
        "triclass_shorten_start_applied": float(tri_shorten_start_applied),
        "triclass_shorten_end_applied": float(tri_shorten_end_applied),
        "start": dict(start_stats),
        "end": dict(end_stats),
    }
    return np.asarray(out, dtype=float), stats


def estimate_width_thickness_from_centerline_slabs(
    points: np.ndarray,
    centerline: np.ndarray,
    config: CenterlineConfig,
) -> Tuple[float, float]:
    """
    基于 centerline 的局部截面统计估计 pin_width / pin_thickness，避免被整体跨度（弯头腿长等）污染。
    """
    pts = np.asarray(points, dtype=float)
    curve = np.asarray(centerline, dtype=float)
    if len(pts) < 10 or len(curve) < 2:
        return 0.0, 0.0

    tree = cKDTree(pts)
    tangents = compute_tangents(curve)
    half_t = float(config.slab_half_thickness())
    search_r = float(config.slab_search_radius(None))

    widths: List[float] = []
    thicknesses: List[float] = []

    prev_u: Optional[np.ndarray] = None
    stride = int(max(1, config.width_sample_stride))
    for i in range(0, len(curve), stride):
        c = curve[i]
        t = tangents[i]
        u, v = make_plane_basis(t, prev_u=prev_u)
        prev_u = u

        # 先用一个保守半径抓局部点，再在平面内做 PCA 求截面主轴
        cand = tree.query_ball_point(c, r=search_r)
        if len(cand) < int(config.min_slab_points):
            continue

        local = pts[cand] - c
        axial = local @ t
        mask = np.abs(axial) <= half_t
        if int(np.sum(mask)) < int(config.min_slab_points):
            continue

        local = local[mask]
        uv = np.stack([local @ u, local @ v], axis=1)  # (M,2)
        uv_mean = uv.mean(axis=0)
        uv0 = uv - uv_mean
        cov = np.cov(uv0.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        except Exception:
            continue

        axes = eigvecs[:, ::-1]  # descending
        proj = uv0 @ axes  # (M,2)
        ql, qh = float(config.width_quantile_low), float(config.width_quantile_high)
        span0 = float(np.quantile(proj[:, 0], qh) - np.quantile(proj[:, 0], ql))
        span1 = float(np.quantile(proj[:, 1], qh) - np.quantile(proj[:, 1], ql))
        if span0 <= 0 or span1 <= 0:
            continue

        w = float(max(span0, span1))
        t = float(min(span0, span1))
        if w > 2.2 * search_r or t > 2.2 * search_r:
            continue

        widths.append(w)
        thicknesses.append(t)

    if not widths:
        return 0.0, 0.0

    w = float(np.median(widths))
    t = float(np.median(thicknesses))
    return _clamp_pin_width_thickness_prior(w, t, config)


def evaluate_centerline_basic(points: np.ndarray, curve: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=float)
    c = np.asarray(curve, dtype=float)
    if len(pts) == 0 or len(c) == 0:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }

    tree = cKDTree(pts)
    d, _ = tree.query(c)
    return {
        "p50": float(np.quantile(d, 0.50)),
        "p90": float(np.quantile(d, 0.90)),
        "p99": float(np.quantile(d, 0.99)),
        "mean": float(np.mean(d)),
        "max": float(np.max(d)),
    }


# ==========================================
# 评估函数
# ==========================================
def compute_slice_features(layer_pts, center_point):
    """
    计算单个切片的几何特征
    输入:
        layer_pts: (N, 3) 该切片内的点云 (局部坐标系)
        center_point: (3,) 该切片的拟合中心
    输出:
        feats: dict, 包含半径统计和PCA特征
    """
    # 1. 半径统计 (只看XY平面的距离)
    xy = layer_pts[:, :2]
    center_xy = center_point[:2]
    dists = np.linalg.norm(xy - center_xy, axis=1)

    r_mean = np.mean(dists)
    r_std = np.std(dists)
    r_max = np.max(dists)

    # 2. 局部 PCA (检测压扁、非圆形变)
    # 计算协方差矩阵的特征值
    if len(layer_pts) >= 3:
        # 中心化
        pts_centered = layer_pts - np.mean(layer_pts, axis=0)
        cov = np.cov(pts_centered.T)
        eigvals = np.linalg.eigvalsh(cov)  # 升序排列 [lambda1, lambda2, lambda3]
        # 归一化特征值，描述形状因子
        sum_eigs = np.sum(eigvals) + 1e-8
        linearity = (eigvals[2] - eigvals[1]) / sum_eigs
        planarity = (eigvals[1] - eigvals[0]) / sum_eigs
    else:
        linearity, planarity = 0.0, 0.0

    return {
        "r_mean": r_mean,
        "r_std": r_std,
        "r_max": r_max,
        "linearity": linearity,
        "planarity": planarity
    }

def evaluate_centerline_distance(
    pts_local,
    curve_local,
    pin_thickness,
    far_ratio=1.3,
    max_longest_far_fraction=0.07,
):
    """
    利用“厚度一半”的几何先验做拟合风险评估：
    - 对每个中心线点，计算到最近点云的距离 d_i；
    - 归一化为 rel_d = d_i / (0.5 * 厚度)；
    - 如果有一段连续很长区间，rel_d 一直 > far_ratio，
      则认为这段大概率已经跑出针外 → 拟合高风险。

    参数：
    - pts_local: (N, 3) 原始点云，局部坐标系
    - curve_local: (M, 3) 中心线，局部坐标系
    - pin_thickness: 针截面的厚度估计
    - far_ratio: 多远算“明显离开针体”，默认为 1.3
    - max_longest_far_fraction: “连续远离段”占比超过该阈值则判为高风险

    返回：
    - dict: {
        "status": "ok" / "high_risk",
        "median_dist": 中位距离,
        "mean_dist": 平均距离,
        "far_fraction": 所有远离段长度占总长度的比例,
        "longest_far_fraction": 最长连续远离段长度占总长度的比例,
      }
    """
    pts_tree = cKDTree(pts_local)

    dists, _ = pts_tree.query(curve_local)
    typical_radius = 0.5 * pin_thickness
    rel_d = dists / (typical_radius + 1e-8)

    far_mask = rel_d > far_ratio

    # 计算每一段中心线弧长
    seg_lengths = np.linalg.norm(curve_local[1:] - curve_local[:-1], axis=1)
    total_len = seg_lengths.sum() + 1e-8

    far_edge = far_mask[:-1] & far_mask[1:]
    far_len = seg_lengths[far_edge].sum()

    # 最长连续远离段
    longest_far = 0.0
    curr_far = 0.0
    for i in range(len(seg_lengths)):
        if i < len(far_edge) and far_edge[i]:
            curr_far += seg_lengths[i]
        else:
            if curr_far > longest_far:
                longest_far = curr_far
            curr_far = 0.0
    if curr_far > longest_far:
        longest_far = curr_far

    far_fraction = far_len / total_len
    longest_far_fraction = longest_far / total_len

    status_legacy = "ok"
    if longest_far_fraction > max_longest_far_fraction:
        status_legacy = "high_risk"

    return {
        # Keep reporting geometric metrics, but disable high_risk label due frequent false positives.
        "status": "ok",
        "status_legacy": status_legacy,
        "median_dist": float(np.median(dists)),
        "mean_dist": float(np.mean(dists)),
        "far_fraction": float(far_fraction),
        "longest_far_fraction": float(longest_far_fraction),
    }


# ==========================================
# Stage13 acceptance gate helpers
# ==========================================


def _regrow_quality_worse(
    before_basic: Dict[str, float],
    before_dist: Dict[str, object],
    after_basic: Dict[str, float],
    after_dist: Dict[str, object],
    config: CenterlineConfig,
    *,
    endpoint0_before: Optional[Dict[str, float]] = None,
    endpoint0_after: Optional[Dict[str, float]] = None,
    endpoint1_before: Optional[Dict[str, float]] = None,
    endpoint1_after: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str]:
    """Conservative stage13 acceptance gate: flag if regrow clearly worsens fit risk."""
    before_status = str(before_dist.get("status", "") or "")
    after_status = str(after_dist.get("status", "") or "")

    before_lff = float(before_dist.get("longest_far_fraction", 0.0) or 0.0)
    after_lff = float(after_dist.get("longest_far_fraction", 0.0) or 0.0)
    before_far = float(before_dist.get("far_fraction", 0.0) or 0.0)
    after_far = float(after_dist.get("far_fraction", 0.0) or 0.0)

    before_p99 = float(before_basic.get("p99", 0.0) or 0.0)
    after_p99 = float(after_basic.get("p99", 0.0) or 0.0)

    tol_lff = float(max(0.0, float(getattr(config, "endpoint_regrow_accept_lff_tol", 0.01) or 0.01)))
    tol_far = float(max(0.0, float(getattr(config, "endpoint_regrow_accept_far_tol", 0.02) or 0.02)))
    p99_tol_u = float(
        mm_to_unit(
            float(max(0.0, float(getattr(config, "endpoint_regrow_accept_p99_tol_mm", 0.08) or 0.08))),
            float(getattr(config, "unit_scale", UNIT_SCALE)),
        )
    )

    reasons: List[str] = []
    if after_status == "high_risk" and before_status != "high_risk":
        reasons.append("status_ok_to_high_risk")
    if after_lff > before_lff + tol_lff:
        reasons.append(f"longest_far_fraction {before_lff:.4f}->{after_lff:.4f}")
    if after_far > before_far + tol_far:
        reasons.append(f"far_fraction {before_far:.4f}->{after_far:.4f}")
    if after_p99 > before_p99 + p99_tol_u:
        reasons.append(f"p99 {before_p99:.6g}->{after_p99:.6g}")

    if bool(getattr(config, "endpoint_regrow_accept_endpoint_risk_gate", True)):
        if endpoint0_before is not None and endpoint0_after is not None:
            if _endpoint_risk_worse(endpoint0_before, endpoint0_after, config):
                reasons.append(
                    "endpoint0_risk "
                    f"cov {float(endpoint0_before.get('min_coverage', 0.0)):.2f}->{float(endpoint0_after.get('min_coverage', 0.0)):.2f}, "
                    f"rv {float(endpoint0_before.get('max_radial_var', float('inf'))):.2f}->{float(endpoint0_after.get('max_radial_var', float('inf'))):.2f}"
                )
        if endpoint1_before is not None and endpoint1_after is not None:
            if _endpoint_risk_worse(endpoint1_before, endpoint1_after, config):
                reasons.append(
                    "endpoint1_risk "
                    f"cov {float(endpoint1_before.get('min_coverage', 0.0)):.2f}->{float(endpoint1_after.get('min_coverage', 0.0)):.2f}, "
                    f"rv {float(endpoint1_before.get('max_radial_var', float('inf'))):.2f}->{float(endpoint1_after.get('max_radial_var', float('inf'))):.2f}"
                )

    return bool(reasons), "; ".join(reasons)


# ==========================================
# GT evaluation (stable_id-aligned)
# ==========================================


def _gt_eval_find_repo_root(from_path: Path) -> Path:
    from_path = from_path.resolve()
    for d in [from_path] + list(from_path.parents):
        if (d / ".release-root").is_file():
            return d
    for d in [from_path] + list(from_path.parents):
        if (d / ".git").exists():
            return d
    return from_path.parent


def _gt_eval_debug_out_dir() -> Path:
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "debug_out_dir", None) is not None:
        return Path(_DAIMA_CTX.debug_out_dir)
    return _gt_eval_find_repo_root(Path(__file__)) / "debug_out"


def _gt_eval_run_id() -> str:
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_id", None):
        return str(_DAIMA_CTX.run_id)
    # Fallback: stable enough for manual imports (avoid collisions).
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _gt_eval_resolve_output_dir(out_subdir: str, pcd_path: str) -> Path:
    debug_out_dir = _gt_eval_debug_out_dir()
    sub = str(out_subdir or "gt_eval").strip()
    sub_path = Path(sub)
    if sub_path.is_absolute():
        # Keep artifacts inside debug_out (project rule).
        sub_path = Path("gt_eval")
    stem = Path(pcd_path).stem if pcd_path else "pcd"
    out_dir = debug_out_dir / sub_path / stem / _gt_eval_run_id()

    # Best-effort guard against path traversal outside debug_out.
    try:
        out_dir_r = out_dir.resolve(strict=False)
        debug_r = debug_out_dir.resolve(strict=False)
        try:
            ok = out_dir_r.is_relative_to(debug_r)  # py>=3.9
        except Exception:
            ok = debug_r in out_dir_r.parents or out_dir_r == debug_r
        if not ok:
            out_dir = debug_out_dir / "gt_eval" / stem / _gt_eval_run_id()
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _gt_eval_polyline_length(curve: np.ndarray) -> float:
    pts = np.asarray(curve, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return float(np.sum(seg))


def _gt_eval_resample_by_count(curve: np.ndarray, n: int) -> np.ndarray:
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


def _gt_eval_distance_stats(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if len(d) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p95": float(np.quantile(d, 0.95)),
        "max": float(np.max(d)),
    }


def _gt_eval_curve_to_curve_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return _gt_eval_distance_stats(d)


def _gt_eval_orient_pred_to_gt(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, bool, float, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(pred) < 2 or len(gt) < 2:
        return pred, False, 0.0, 0.0
    err_fwd = float(np.linalg.norm(pred[0] - gt[0]) + np.linalg.norm(pred[-1] - gt[-1]))
    err_rev = float(np.linalg.norm(pred[-1] - gt[0]) + np.linalg.norm(pred[0] - gt[-1]))
    if err_rev + 1e-12 < err_fwd:
        return pred[::-1], True, float(np.linalg.norm(pred[-1] - gt[0])), float(np.linalg.norm(pred[0] - gt[-1]))
    return pred, False, float(np.linalg.norm(pred[0] - gt[0])), float(np.linalg.norm(pred[-1] - gt[-1]))


def _gt_eval_angles_deg(t_ref: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
    t_ref = np.asarray(t_ref, dtype=np.float64)
    t_pred = np.asarray(t_pred, dtype=np.float64)
    n = min(len(t_ref), len(t_pred))
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    t_ref = t_ref[:n]
    t_pred = t_pred[:n]
    dot = np.einsum("ij,ij->i", t_ref, t_pred)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _gt_eval_point_to_polyline_closest(
    p: np.ndarray,
    polyline: np.ndarray,
    *,
    seg_a: np.ndarray,
    seg_ab: np.ndarray,
    seg_ab2: np.ndarray,
    seg_len: np.ndarray,
    seg_t: np.ndarray,
    seg_s0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Closest point query from a point to a 3D polyline (piecewise linear).

    Returns:
        q: closest point on the polyline
        t: unit tangent of the closest segment (fallback: [0,0,1] if degenerate)
        s: arc-length position of q along the polyline (same units as polyline)

    Notes:
        - For a polyline, the closest point is the closest point among all segments.
          If the perpendicular projection falls outside a segment, the closest endpoint is used.
        - This implements the standard "point-to-polyline distance" rule used by many GIS/proximity tools.
          We then decompose (p - q) into along-track (tangent) and cross-track (normal) components.
    """
    p = np.asarray(p, dtype=np.float64).reshape(3)
    polyline = np.asarray(polyline, dtype=np.float64)
    if len(polyline) == 0:
        return np.zeros((3,), dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    if len(polyline) == 1 or len(seg_a) == 0:
        return polyline[0].copy(), np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0

    # Project p to each segment (vectorized over segments), then pick the closest projection.
    #   q_j = a_j + u_j * ab_j,  u_j = clamp( dot(p-a_j, ab_j) / ||ab_j||^2, 0..1 )
    #   choose j minimizing ||p - q_j||.
    diff = p[None, :] - seg_a  # (S,3)
    u = np.einsum("ij,ij->i", diff, seg_ab) / (seg_ab2 + 1e-12)
    u = np.clip(u, 0.0, 1.0)
    q_all = seg_a + seg_ab * u[:, None]
    d_all = p[None, :] - q_all
    dist2 = np.einsum("ij,ij->i", d_all, d_all)
    j = int(np.argmin(dist2))

    q = q_all[j]
    t = seg_t[j].copy()
    if float(seg_len[j]) <= 1e-12:
        t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    s = float(seg_s0[j] + u[j] * seg_len[j])
    return q, t, s


def _gt_eval_cross_along_track_errors(
    points: np.ndarray,
    gt_polyline: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-track (normal) / along-track (tangent) offsets from `points` to a GT polyline.

    - Cross-track error: magnitude of the component perpendicular to the local GT tangent.
    - Along-track error: signed component along the local GT tangent.

    Returns:
        along_u: (N,) signed along-track offset, in the same unit as inputs
        cross_u: (N,) cross-track magnitude, in the same unit as inputs
    """
    pts = np.asarray(points, dtype=np.float64)
    gt = np.asarray(gt_polyline, dtype=np.float64)
    if len(pts) == 0 or len(gt) < 2:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    seg_a = gt[:-1]
    seg_b = gt[1:]
    seg_ab = seg_b - seg_a
    seg_ab2 = np.einsum("ij,ij->i", seg_ab, seg_ab)
    seg_len = np.sqrt(np.maximum(seg_ab2, 0.0))
    seg_t = np.zeros_like(seg_ab)
    nz = seg_len > 1e-12
    seg_t[nz] = seg_ab[nz] / seg_len[nz, None]
    seg_s0 = np.concatenate([[0.0], np.cumsum(seg_len[:-1])]) if len(seg_len) else np.zeros((0,), dtype=np.float64)

    along_u = np.zeros((len(pts),), dtype=np.float64)
    cross_u = np.zeros((len(pts),), dtype=np.float64)
    for i, p in enumerate(pts):
        q, t, _ = _gt_eval_point_to_polyline_closest(
            p,
            gt,
            seg_a=seg_a,
            seg_ab=seg_ab,
            seg_ab2=seg_ab2,
            seg_len=seg_len,
            seg_t=seg_t,
            seg_s0=seg_s0,
        )
        d = p - q
        dt = float(d @ t)  # along-track (signed)
        dn2 = float(d @ d) - dt * dt
        dn = float(np.sqrt(max(0.0, dn2)))  # cross-track magnitude
        along_u[i] = dt
        cross_u[i] = dn

    return along_u, cross_u


def _gt_eval_load_centerline_gt(gt_dir: str) -> Tuple[dict, Dict[int, np.ndarray], Path, Dict[int, int]]:
    gt_path = Path(gt_dir)
    gt_json = gt_path if gt_path.is_file() else (gt_path / "centerline_gt.json")
    if not gt_json.exists():
        raise FileNotFoundError(f"centerline_gt.json not found: {gt_json}")

    data = json.loads(gt_json.read_text(encoding="utf-8"))
    needles = data.get("needles", [])

    gt_map: Dict[int, np.ndarray] = {}
    dup_counts: Dict[int, int] = {}
    for n in needles:
        try:
            stable_id = int(n.get("stable_id", -1))
        except Exception:
            continue
        if stable_id < 0:
            continue

        segs = n.get("segments", []) or []
        best = np.zeros((0, 3), dtype=np.float64)
        for seg in segs:
            arr = np.asarray(seg, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3:
                continue
            if len(arr) > len(best):
                best = arr

        dup_counts[stable_id] = int(dup_counts.get(stable_id, 0) + 1)
        if stable_id not in gt_map or len(best) > len(gt_map[stable_id]):
            gt_map[stable_id] = best

    return data, gt_map, gt_json, dup_counts


def evaluate_centerlines_against_gt(
    pcd_path: str,
    gt_dir: str,
    *,
    out_subdir: str = "",
    method: str = "graph_slab",
    config: Optional[CenterlineConfig] = None,
    step_mm: float = EVAL_STEP_MM,
    tangent_samples: int = EVAL_TANGENT_SAMPLES,
) -> dict:
    """
    Evaluate centerlines produced by `pin_cloud2center_line.py` against a manually-made centerline GT.

    Assumption: GT and prediction are in the same coordinate system (same source point cloud), so no registration
    is applied here. Matching is performed by `stable_id`.
    """
    if config is None:
        config = CenterlineConfig()

    pcd_path = str(pcd_path)
    gt_dir = str(gt_dir)

    gt_data, gt_map, gt_json_path, gt_dup = _gt_eval_load_centerline_gt(gt_dir)
    gt_ids = sorted(gt_map.keys())

    # Extract predicted centerlines (no visualization).
    infos = extract_centerlines_from_pcd(pcd_path, method=method, config=config)
    pred_groups: Dict[int, List[np.ndarray]] = {}
    pred_dup: Dict[int, int] = {}
    for info in infos:
        try:
            sid = int(info.get("stable_id", -1))
        except Exception:
            continue
        if sid < 0:
            continue
        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        pred_groups.setdefault(sid, []).append(curve)

    pred_map: Dict[int, np.ndarray] = {}
    for sid, curves in pred_groups.items():
        pred_dup[sid] = int(len(curves))
        # Prefer the longest curve (most samples). This is stable for typical "one component per cluster" cases.
        best = max(curves, key=lambda c: int(len(c)))
        pred_map[sid] = np.asarray(best, dtype=np.float64)

    pred_ids = sorted(pred_map.keys())
    missing_pred = [sid for sid in gt_ids if sid not in pred_map]
    extra_pred = [sid for sid in pred_ids if sid not in gt_map]

    # stable_id sanity threshold (based on GT centers spacing).
    center_warn_dist = float("inf")
    if len(gt_ids) >= 2:
        centers = np.asarray([np.mean(gt_map[sid], axis=0) for sid in gt_ids if len(gt_map[sid])], dtype=np.float64)
        if len(centers) >= 2:
            tree = cKDTree(centers)
            d, _ = tree.query(centers, k=2)
            nn = np.asarray(d[:, 1], dtype=np.float64)
            center_warn_dist = float(0.6 * np.median(nn))

    unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
    to_mm = lambda x: float(x) / (unit_scale + 1e-12)

    per = []
    warnings: List[str] = []

    # Basic source consistency check (helps catch comparing against a wrong GT folder).
    src_cloud = gt_data.get("source_cloud_path", None)
    if src_cloud:
        try:
            a = str(Path(src_cloud).resolve(strict=False)).casefold()
            b = str(Path(pcd_path).resolve(strict=False)).casefold()
            if a != b:
                warnings.append(f"GT source_cloud_path != pcd_path: {src_cloud} vs {pcd_path}")
        except Exception:
            pass

    out_dir = _gt_eval_resolve_output_dir(out_subdir, pcd_path)

    for sid in gt_ids:
        gt_curve = gt_map.get(sid, np.zeros((0, 3), dtype=np.float64))
        pred_curve = pred_map.get(sid, np.zeros((0, 3), dtype=np.float64))

        pred_curve, reversed_flag, end0_u, end1_u = _gt_eval_orient_pred_to_gt(pred_curve, gt_curve)

        gt_rs = _gt_eval_resample_by_count(gt_curve, tangent_samples)
        pred_rs = _gt_eval_resample_by_count(pred_curve, tangent_samples)

        # Curve-to-curve (discrete) distances (orientation invariant).
        pred_to_gt = _gt_eval_curve_to_curve_stats(pred_rs, gt_rs)
        gt_to_pred = _gt_eval_curve_to_curve_stats(gt_rs, pred_rs)
        chamfer_mean = 0.5 * (float(pred_to_gt["mean"]) + float(gt_to_pred["mean"]))
        hausdorff = float(max(float(pred_to_gt["max"]), float(gt_to_pred["max"])))

        # Endpoints (orientation sensitive).
        end_sum_u = float(end0_u + end1_u)

        # Length.
        len_gt_u = _gt_eval_polyline_length(gt_curve)
        len_pred_u = _gt_eval_polyline_length(pred_curve)
        len_diff_u = float(len_pred_u - len_gt_u)

        # Tangent angles (orientation aligned above). Resample by physical step (mm) for stable comparisons.
        ang_stats = {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        step_u = float(mm_to_unit(step_mm, unit_scale))
        if step_u > 1e-12 and len(gt_curve) >= 2 and len(pred_curve) >= 2:
            gt_t = resample_polyline_by_step(gt_curve, step=step_u)
            pr_t = resample_polyline_by_step(pred_curve, step=step_u)
        else:
            gt_t = gt_rs
            pr_t = pred_rs
        n_t = min(len(gt_t), len(pr_t))
        if n_t >= 2:
            t_gt = compute_tangents(gt_t[:n_t])
            t_pr = compute_tangents(pr_t[:n_t])
            ang = _gt_eval_angles_deg(t_gt, t_pr)
            ang_stats = _gt_eval_distance_stats(ang)

        # Cross-track (normal) offset: decompose the point-to-GT error into tangent/normal components.
        # This quantifies "pure lateral deviation" and is much less sensitive to along-axis sliding.
        pred_step = pr_t if step_u > 1e-12 else pred_rs
        _, cross_u = _gt_eval_cross_along_track_errors(pred_step, gt_curve)
        cross_mm_stats = _gt_eval_distance_stats(cross_u / (unit_scale + 1e-12)) if len(cross_u) else {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

        center_u = float(np.linalg.norm(np.mean(pred_rs, axis=0) - np.mean(gt_rs, axis=0))) if len(pred_rs) and len(gt_rs) else 0.0
        sid_warnings: List[str] = []
        if center_u > center_warn_dist:
            sid_warnings.append(f"center_shift>{to_mm(center_warn_dist):.3f}mm")

        per.append(
            {
                "stable_id": int(sid),
                "gt_points": int(len(gt_curve)),
                "pred_points": int(len(pred_curve)),
                "pred_duplicate_components": int(max(0, pred_dup.get(sid, 1) - 1)),
                "pred_reversed_to_match_gt": bool(reversed_flag),
                "center_shift": {"u": float(center_u), "mm": to_mm(center_u)},
                "normal_offset_mm": cross_mm_stats,
                "length": {
                    "gt_u": float(len_gt_u),
                    "pred_u": float(len_pred_u),
                    "diff_u": float(len_diff_u),
                    "gt_mm": to_mm(len_gt_u),
                    "pred_mm": to_mm(len_pred_u),
                    "diff_mm": to_mm(len_diff_u),
                },
                "endpoints": {
                    "start_u": float(end0_u),
                    "end_u": float(end1_u),
                    "sum_u": float(end_sum_u),
                    "start_mm": to_mm(end0_u),
                    "end_mm": to_mm(end1_u),
                    "sum_mm": to_mm(end_sum_u),
                },
                "curve_distance": {
                    "pred_to_gt": pred_to_gt,
                    "gt_to_pred": gt_to_pred,
                    "chamfer_mean_u": float(chamfer_mean),
                    "chamfer_mean_mm": to_mm(chamfer_mean),
                    "hausdorff_u": float(hausdorff),
                    "hausdorff_mm": to_mm(hausdorff),
                },
                "tangent_angle_deg": ang_stats,
                "warnings": sid_warnings,
            }
        )

    # Aggregate summary.
    chamfers_mm = [float(p["curve_distance"]["chamfer_mean_mm"]) for p in per]
    hausdorffs_mm = [float(p["curve_distance"]["hausdorff_mm"]) for p in per]
    end_sums_mm = [float(p["endpoints"]["sum_mm"]) for p in per]
    len_abs_diff_mm = [abs(float(p["length"]["diff_mm"])) for p in per]
    normal_p95_mm = [float(p.get("normal_offset_mm", {}).get("p95", 0.0)) for p in per]
    summary = {
        "needles_gt": int(len(gt_ids)),
        "needles_pred": int(len(pred_ids)),
        "missing_pred_stable_ids": missing_pred,
        "extra_pred_stable_ids": extra_pred,
        "gt_duplicate_stable_id_counts": {str(k): int(v) for k, v in gt_dup.items() if int(v) > 1},
        "pred_duplicate_component_counts": {str(k): int(v) for k, v in pred_dup.items() if int(v) > 1},
        "chamfer_mean_mm": _gt_eval_distance_stats(np.asarray(chamfers_mm, dtype=np.float64)),
        "hausdorff_mm": _gt_eval_distance_stats(np.asarray(hausdorffs_mm, dtype=np.float64)),
        "endpoint_sum_mm": _gt_eval_distance_stats(np.asarray(end_sums_mm, dtype=np.float64)),
        "abs_length_diff_mm": _gt_eval_distance_stats(np.asarray(len_abs_diff_mm, dtype=np.float64)),
        # A scalar summary for "pure lateral deviation". Per-needle uses p95 (robust to local bumps).
        "normal_offset_p95_mm": _gt_eval_distance_stats(np.asarray(normal_p95_mm, dtype=np.float64)),
    }

    # Worst-case list for quick iteration.
    worst = sorted(per, key=lambda p: float(p["curve_distance"]["chamfer_mean_mm"]), reverse=True)
    worst_topk = worst[: max(0, int(EVAL_TOPK))]
    summary["worst_by_chamfer_mean_mm"] = [
        {"stable_id": int(p["stable_id"]), "chamfer_mean_mm": float(p["curve_distance"]["chamfer_mean_mm"])}
        for p in worst_topk
    ]

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pcd_path": pcd_path,
        "gt_dir": gt_dir,
        "gt_json_path": str(gt_json_path),
        "gt_source_cloud_path": gt_data.get("source_cloud_path", None),
        "method": str(method),
        "unit_scale": float(unit_scale),
        "eval": {"step_mm": float(step_mm), "tangent_samples": int(tangent_samples)},
        "summary": summary,
        "warnings": warnings,
        "per_needle": per,
        "out_dir": str(out_dir),
    }

    # Write outputs (inside debug_out/).
    out_json = out_dir / "gt_eval_report.json"
    out_csv = out_dir / "gt_eval_summary.csv"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stable_id",
                "chamfer_mean_mm",
                "hausdorff_mm",
                "endpoint_sum_mm",
                "abs_length_diff_mm",
                "center_shift_mm",
                "normal_offset_mm_p95",
                "pred_points",
                "gt_points",
                "pred_reversed",
                "pred_duplicate_components",
                "warnings",
            ]
        )
        for p in per:
            w.writerow(
                [
                    int(p["stable_id"]),
                    float(p["curve_distance"]["chamfer_mean_mm"]),
                    float(p["curve_distance"]["hausdorff_mm"]),
                    float(p["endpoints"]["sum_mm"]),
                    abs(float(p["length"]["diff_mm"])),
                    float(p["center_shift"]["mm"]),
                    float(p.get("normal_offset_mm", {}).get("p95", 0.0)),
                    int(p["pred_points"]),
                    int(p["gt_points"]),
                    int(bool(p["pred_reversed_to_match_gt"])),
                    int(p["pred_duplicate_components"]),
                    "|".join(p.get("warnings", []) or []),
                ]
            )

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={"pcd_path": pcd_path, "gt_dir": gt_dir, "gt_json_path": str(gt_json_path)},
            outputs={"eval_out_dir": str(out_dir), "eval_report_json": str(out_json), "eval_summary_csv": str(out_csv)},
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "eval_method": str(method),
                "eval_step_mm": float(step_mm),
                "eval_tangent_samples": int(tangent_samples),
                "eval_unit_scale": float(unit_scale),
            }
        )

    # Console hints for fast iteration (optional).
    print(f"[GT-EVAL] Output: {out_dir}")
    print(
        "[GT-EVAL] Summary: "
        f"gt={summary['needles_gt']} pred={summary['needles_pred']} "
        f"missing_pred={len(missing_pred)} extra_pred={len(extra_pred)} "
        f"chamfer_mean_mm.p95={summary['chamfer_mean_mm']['p95']:.3f} "
        f"hausdorff_mm.max={summary['hausdorff_mm']['max']:.3f}"
    )
    if worst_topk:
        print("[GT-EVAL] Worst by chamfer_mean_mm:")
        for p in worst_topk:
            print(f"  - stable_id={int(p['stable_id'])} chamfer_mean_mm={float(p['curve_distance']['chamfer_mean_mm']):.3f}")

    return report


# ==========================================
# 椭圆拟合相关函数
# ==========================================







# 修改 pin_cloud2center_line.py 中的这个函数

def get_robust_slice_center(pts_layer, prev_center=None, max_jump=0.5):
    """
    针对长方形截面点云的鲁棒中心提取。
    利用中位数和截断均值抵抗边缘缺失和飞点。

    输入:
        pts_layer: (N, 3) 局部坐标系下的点，Z轴基本一致
    """
    # 1. 解耦 XY 平面
    x = pts_layer[:, 0]
    y = pts_layer[:, 1]

    # 2. 核心改进：使用截断均值 (Trimmed Mean) 代替椭圆拟合
    # 逻辑：长方形是对称的，去掉两头 10% 的边缘噪声（NeRF的飞点），
    # 剩下的中间部分的均值就是极其稳定的几何中心。
    def robust_axis_center(data):
        # 过滤掉极端的飞点
        q10 = np.percentile(data, 10)
        q90 = np.percentile(data, 90)
        mask = (data >= q10) & (data <= q90)
        clean_data = data[mask]
        if len(clean_data) == 0:
            return np.median(data)
        return np.mean(clean_data)

    cx = robust_axis_center(x)
    cy = robust_axis_center(y)
    cz = np.median(pts_layer[:, 2])

    curr_center = np.array([cx, cy, cz])

    # 3. 连续性约束 (防止突变)
    # 如果这一层算出来的中心跟上一层差太远，强制拉回来
    if prev_center is not None:
        dist = np.linalg.norm(curr_center[:2] - prev_center[:2])
        if dist > max_jump:
            # 这是一个简单的卡尔曼滤波思想：测量值太离谱时，信任预测值(上一帧)
            # 融合系数 0.2 表示只取 20% 的新测量值，80% 沿用旧值
            curr_center[:2] = 0.8 * prev_center[:2] + 0.2 * curr_center[:2]

    return curr_center


# ==========================================
# 曲率与权重相关函数
# ==========================================

def compute_discrete_curvature(points):
    """
    对 3D 多段线的离散曲率进行估计：
    对每个中间点 i，利用 i-1, i, i+1 构成的三角形面积与边长估计曲率。
    """
    pts = np.asarray(points)
    n = pts.shape[0]
    if n < 3:
        return np.zeros(n - 2)

    v1 = pts[1:-1] - pts[:-2]
    v2 = pts[1:-1] - pts[2:]

    cross = np.cross(v1, v2)
    area2 = np.linalg.norm(cross, axis=1)
    L1 = np.linalg.norm(v1, axis=1)
    L2 = np.linalg.norm(v2, axis=1)
    L3 = np.linalg.norm(pts[2:] - pts[:-2], axis=1)
    # 经典离散曲率近似：k ~ 4 * area / (L1 * L2 * L3)
    kappa = 4 * area2 / (L1 * L2 * L3 + 1e-8)
    return kappa


def compute_weights_from_curvature(kappa, window=5):
    """
    根据曲率序列 kappa 构造权重：
    - 先对曲率做滑动中值滤波得到平滑版本 kappa_smooth；
    - 对偏离较大的点给予较小权重，抑制局部异常弯折。
    """
    kappa = np.asarray(kappa)
    n = len(kappa)
    if n == 0:
        return np.array([])

    # 滑动中值滤波
    kappa_smooth = np.copy(kappa)
    half = max(1, window // 2)
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        kappa_smooth[i] = np.median(kappa[i0:i1])

    # 以 (kappa - kappa_smooth) 的绝对值做“异常度”
    delta = np.abs(kappa - kappa_smooth)
    med = np.median(delta)
    mad = np.median(np.abs(delta - med)) + 1e-8

    z = (delta - med) / (mad + 1e-8)
    # z 越大，说明偏离平滑曲率越严重 → 权重越小
    w = np.exp(-np.clip(z, 0, 5))
    # 归一化到 [0.2, 1.0]，防止某些点权重过小
    w = 0.2 + 0.8 * (w - w.min()) / (w.max() - w.min() + 1e-8)
    return w


def compute_weights_from_density(densities):
    """
    根据截面点数（局部密度）构造权重：
    - 密度越高，权重越大；
    - 低密度截面权重减小，但不会变成 0。
    """
    dens = np.asarray(densities, dtype=float)
    if len(dens) == 0:
        return np.array([])

    med = np.median(dens)
    if med <= 0:
        return np.ones_like(dens)

    rel = dens / med
    rel = np.clip(rel, 0.1, 3.0)

    # 归一化到 [0.3, 1.0]
    w = 0.3 + 0.7 * (rel - rel.min()) / (rel.max() - rel.min() + 1e-8)
    return w


def smooth_polyline(points, weights=None, window_frac=0.03, min_window=5):
    """
    对 3D 多段线进行加权平滑：
    - 使用滑动窗口加权平均；
    - window_frac 控制窗口相对大小；
    - weights 是每个点的权重（如密度权重或曲率权重）。
    """
    pts = np.asarray(points)
    n = pts.shape[0]
    if n == 0:
        return pts

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n:
            raise ValueError("weights 长度必须与 points 一致")

    window = max(min_window, int(n * window_frac))
    half = max(1, window // 2)

    out = np.zeros_like(pts)
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        w_local = weights[i0:i1]
        p_local = pts[i0:i1]
        w_sum = w_local.sum() + 1e-8
        out[i] = np.sum(p_local * w_local[:, None], axis=0) / w_sum

    return out


# ==========================================
# 密度场风格评估
# ==========================================

def evaluate_centerline_density(
    pts_local,
    curve_local,
    typical_radius,
    axial_window,
    low_density_ratio=0.35,
    max_low_fraction=0.12,
):
    """
    利用“中心线附近局部点云密度”评估拟合质量：

    对于中心线上的每个点 c_i：
      1) 在球半径 typical_radius 内搜索近邻点；
      2) 再根据 z 坐标差筛选出一段轴向窗口 (axial_window) 内的点；
      3) 计数作为该点附近的局部密度。

    最终统计：
      - median_density：全局中位密度；
      - low_fraction：低密度（< low_density_ratio * median）区域的长度占比；
      - longest_low_fraction：最长连续低密度段长度占比，
        若 longest_low_fraction > max_low_fraction，则判为高风险。
    """
    pts_local = np.asarray(pts_local)
    curve_local = np.asarray(curve_local)

    if len(curve_local) < 2:
        return {
            "status": "high_risk",
            "median_density": 0.0,
            "mean_density": 0.0,
            "low_fraction": 1.0,
            "longest_low_fraction": 1.0,
        }

    tree = cKDTree(pts_local)
    counts = []

    for c in curve_local:
        neighbors_idx = tree.query_ball_point(c, r=typical_radius)
        if not neighbors_idx:
            counts.append(0)
            continue

        local_pts = pts_local[neighbors_idx]
        dz = np.abs(local_pts[:, 2] - c[2])
        mask_axial = dz <= axial_window
        counts.append(int(np.sum(mask_axial)))

    counts = np.array(counts, dtype=float)
    median_density = float(np.median(counts))
    mean_density = float(np.mean(counts))

    if median_density < 3:
        # 整体密度过低，直接判定为高风险
        return {
            "status": "high_risk",
            "median_density": median_density,
            "mean_density": mean_density,
            "low_fraction": 1.0,
            "longest_low_fraction": 1.0,
        }

    rel = counts / (median_density + 1e-8)
    low_mask = rel < low_density_ratio

    seg_lengths = np.linalg.norm(curve_local[1:] - curve_local[:-1], axis=1)
    total_len = seg_lengths.sum() + 1e-8

    low_edge = low_mask[:-1] & low_mask[1:]
    low_len = seg_lengths[low_edge].sum()

    longest_low = 0.0
    curr_low = 0.0
    for i in range(len(seg_lengths)):
        if i < len(low_edge) and low_edge[i]:
            curr_low += seg_lengths[i]
        else:
            if curr_low > longest_low:
                longest_low = curr_low
            curr_low = 0.0
    if curr_low > longest_low:
        longest_low = curr_low

    low_fraction = low_len / total_len
    longest_low_fraction = longest_low / total_len

    status = "ok"
    if longest_low_fraction > max_low_fraction:
        status = "high_risk"

    return {
        "status": status,
        "median_density": median_density,
        "mean_density": mean_density,
        "low_fraction": float(low_fraction),
        "longest_low_fraction": float(longest_low_fraction),
    }


# ==========================================
# 带权中心线构造
# ==========================================

def build_weighted_centerline(centroids, densities):
    """
    输入：
      - centroids: (K, 3) 截面中心点序列（局部坐标系）
      - densities: (K,) 每个截面对应的点数（密度）

    输出：
      - curve_final: (M, 3) 高密度采样后平滑、带权融合的中心线（局部坐标系）
    """
    pts = np.asarray(centroids, dtype=float)
    dens = np.asarray(densities, dtype=float)

    if len(pts) < 4:
        raise ValueError("截面数量太少，无法构造带权中心线")

    # 1) 基于密度的截面权重
    w_section = compute_weights_from_density(dens)

    # 2) 构造纵向参数 t（弧长参数化再归一化）
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = s[-1] if s[-1] > 1e-8 else 1.0
    t = s / total_len

    # 3) 计算几何曲率权重
    kappa_center = compute_discrete_curvature(pts)
    kappa_full = np.zeros(len(pts), dtype=float)
    kappa_full[1:-1] = kappa_center
    kappa_full[0] = kappa_center[0]
    kappa_full[-1] = kappa_center[-1]

    w_kink = compute_weights_from_curvature(kappa_full)

    # 4) 组合截面权重和曲率权重（当前仅用于保持兼容计算路径）
    _ = np.clip(w_section * w_kink, 0.2, 1.0)

    # 5) 样条插值 & 重采样
    try:
        tck, u = splprep(pts.T, u=t, s=0)
    except Exception as e:
        raise RuntimeError(f"样条拟合失败: {e}")

    u_dense = np.linspace(0.0, 1.0, 200)
    x_s, y_s, z_s = splev(u_dense, tck)
    curve_dense = np.vstack([x_s, y_s, z_s]).T

    # 6) 后续平滑直接使用 component-wise 滤波。
    curve_final = smooth_centerline_component_wise(curve_dense, polyorder=2)
    return curve_final


# ==========================================
# Tube mesh 可视化工具
# ==========================================

def create_tube_from_points(points, radius=0.02, color=[0, 0, 1]):
    """
    使用一系列小圆柱拼接方式，把一条 3D 多段线渲染为管状 mesh。
    - points: (N, 3)
    - radius: 管道半径
    - color: RGB 颜色 [r, g, b]
    """
    points = np.asarray(points)
    meshes = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-8:
            continue

        direction = direction / length

        # 在 z 轴方向创建一个高度为 1 的圆柱，然后缩放到 length 并旋转到 direction
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                              height=length,
                                                              resolution=16,
                                                              split=4)
        cylinder.compute_vertex_normals()

        # 初始圆柱是沿 z 轴从 [0, 0, -length/2] 到 [0, 0, length/2]
        # 我们先让它沿 z 轴，从 0 到 length
        R_align_z = np.eye(3)
        t_align_z = np.array([0, 0, length / 2.0])

        cylinder.rotate(R_align_z, center=np.array([0, 0, 0]))
        cylinder.translate(t_align_z)

        # 计算从 z 轴到 direction 的旋转矩阵
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        if np.linalg.norm(v) < 1e-8:
            R = np.eye(3)
            if c < 0:
                R = -np.eye(3)
        else:
            s = np.linalg.norm(v)
            k = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + k + k.dot(k) * ((1 - c) / (s**2 + 1e-8))

        cylinder.rotate(R, center=np.array([0, 0, 0]))

        # 把圆柱移到 p1 的位置
        cylinder.translate(p1)

        cylinder.paint_uniform_color(color)
        meshes.append(cylinder)

    if not meshes:
        return o3d.geometry.TriangleMesh()

    tube = meshes[0]
    for m in meshes[1:]:
        tube += m

    return tube


# ==========================================
# 主流程：鲁棒中心线拟合
# ==========================================

def fit_curve_graph_slab(
    pcd_path: str,
    visualize: bool = True,
    show_raw_cloud: bool = True,
    config: Optional[CenterlineConfig] = None,
):
    """
    新版：面向缺陷检测的通用中心线拟合（不把 90°弯头当强先验）。

    核心思路：
    - DBSCAN 粗聚类（通常得到针段簇）
    - 簇内用 kNN 图 + 剪枝求 connected components（支持断裂/间隙/误连）
    - 每个 component：图直径路径做 1D 顺序 → slab 截面鲁棒居中迭代收敛
    - 基于 centerline 截面重估 pin_width/pin_thickness（避免弯头腿长污染）

    返回：List[dict]（方案 A：segment 级返回，兼容 1.11.py）
    """
    if config is None:
        config = CenterlineConfig()

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(pcd)

    pts_all = np.asarray(pcd.points)
    diag, median_nn = estimate_cloud_stats(pts_all)
    print(
        "[INFO] Units: "
        f"UNIT_SCALE={config.unit_scale:g} | "
        f"DBSCAN_EPS_MM_MIN={config.dbscan_eps_min_mm:g} | "
        f"DBSCAN_EPS_MM_MAX={config.dbscan_eps_mm:g} | "
        f"DBSCAN_EPS_FACTOR={config.dbscan_eps_factor:g} | "
        f"CENTERLINE_STEP_MM={config.centerline_step_mm:g} | "
        f"NERF_NOISE_MM={config.nerf_noise_mm:g} | "
        f"RAW_TUBE_RADIUS={RAW_TUBE_RADIUS:g} | "
        f"FINAL_TUBE_RADIUS={FINAL_TUBE_RADIUS:g}"
    )
    print(f"[INFO] Cloud stats: diag={diag:.6g} | median_nn={median_nn:.6g}")

    # Pre-DBSCAN denoise: break NeRF "bridging" noise that can merge multiple needles into one DBSCAN cluster.
    denoise_stage = str(getattr(config, "denoise_stage", "")).lower().strip()
    if bool(getattr(config, "denoise_enable", False)) and denoise_stage in {"pre_dbscan", "both"}:
        pts_all_d, denoise_report = _denoise_points_open3d(
            pts_all,
            config,
            tag="pre_dbscan",
            median_nn_hint=median_nn,
            enable_sor=bool(getattr(config, "denoise_pre_dbscan_sor_enable", False)),
        )
        print("[INFO] Denoise(pre_dbscan): " + json.dumps(denoise_report, ensure_ascii=False))
        if _DAIMA_CTX is not None and _DAIMA_CTX.run_dir is not None:
            try:
                (_DAIMA_CTX.run_dir / "denoise_pre_dbscan.json").write_text(
                    json.dumps(denoise_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_all_d)
        pts_all = np.asarray(pcd.points)
        diag2, median_nn2 = estimate_cloud_stats(pts_all)
        print(f"[INFO] Cloud stats (after denoise): diag={diag2:.6g} | median_nn={median_nn2:.6g}")
        diag, median_nn = diag2, median_nn2

    eps_max = float(config.dbscan_eps())
    eps_min_mm = float(max(config.dbscan_eps_min_mm, 2.0 * config.nerf_noise_mm, 2.0 * config.centerline_step_mm))
    eps_min = float(mm_to_unit(eps_min_mm, config.unit_scale))
    if eps_max > 0:
        eps_min = min(eps_min, eps_max)

    if float(config.dbscan_eps_factor) > 0 and median_nn > 0:
        eps_auto = float(config.dbscan_eps_factor) * float(median_nn)
    else:
        eps_auto = eps_max if eps_max > 0 else eps_min

    eps = eps_auto
    if eps_max > 0:
        eps = min(eps, eps_max)
    eps = max(eps, eps_min)

    print("[INFO] Running DBSCAN clustering...")
    labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(config.dbscan_min_points)))
    max_label = int(labels.max()) if len(labels) else -1
    n_clusters = max_label + 1
    n_noise = int(np.sum(labels < 0))
    eps_mm = float(eps / (config.unit_scale + 1e-12))
    eps_auto_mm = float(eps_auto / (config.unit_scale + 1e-12))
    print(
        f"[INFO] DBSCAN clusters={n_clusters} | noise={n_noise} | "
        f"eps={eps:.6g} ({eps_mm:.3f}mm) | "
        f"eps_auto={eps_auto_mm:.3f}mm | eps_min={eps_min_mm:.3f}mm | eps_max={config.dbscan_eps_mm:.3f}mm"
    )

    geometries = []
    needle_infos = []
    next_needle_id = 0

    for cluster_id in range(n_clusters):
        idx = np.where(labels == cluster_id)[0]
        sub_pcd = pcd.select_by_index(idx)
        pts_cluster = np.asarray(sub_pcd.points)
        if len(pts_cluster) < int(config.min_component_points):
            continue

        # Optional: per-cluster denoise (after DBSCAN) to further reduce floaters inside a needle cluster.
        if bool(getattr(config, "denoise_enable", False)) and denoise_stage in {"per_cluster", "both"}:
            med_nn_c0 = estimate_median_nn(pts_cluster)
            pts_cluster_d, denoise_report_c = _denoise_points_open3d(
                pts_cluster,
                config,
                tag=f"cluster_{cluster_id}",
                median_nn_hint=med_nn_c0,
            )
            print(
                f"[INFO] Denoise(per_cluster cluster={cluster_id}): "
                + json.dumps(denoise_report_c, ensure_ascii=False)
            )
            if _DAIMA_CTX is not None and _DAIMA_CTX.run_dir is not None:
                try:
                    (_DAIMA_CTX.run_dir / f"denoise_cluster_{cluster_id:02d}.json").write_text(
                        json.dumps(denoise_report_c, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            sub_pcd.points = o3d.utility.Vector3dVector(pts_cluster_d)
            pts_cluster = np.asarray(sub_pcd.points)
            if len(pts_cluster) < int(config.min_component_points):
                continue

        # 初始颜色设为中性灰（可视化时再按质量变色）
        sub_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if show_raw_cloud:
            geometries.append(sub_pcd)

        # 簇内分段：kNN 图 + 剪枝 + connected components
        adj = build_knn_graph(pts_cluster, k=int(config.graph_k))
        med_nn_c = estimate_median_nn(pts_cluster)
        max_edge = float(config.graph_max_edge_factor) * float(med_nn_c) if med_nn_c > 0 else np.inf
        adj = prune_graph_edges(adj, max_edge=max_edge)
        comps = connected_components(adj)
        # Merge likely-fragmented components (same needle cut into pieces) by reconnecting the pruned graph.
        # This prevents the common "+1 segment" artifact in multi-needle board clouds.
        comps = _merge_fragment_components_by_reconnect(
            adj=adj,
            pts_cluster=pts_cluster,
            comps=comps,
            config=config,
            cluster_id=int(cluster_id),
        )
        comps.sort(key=lambda a: len(a), reverse=True)

        for comp_id, comp_idx in enumerate(comps):
            if len(comp_idx) < int(config.min_component_points):
                continue

            pts_comp = pts_cluster[comp_idx]
            a_local, b_local = approximate_farthest_pair_two_sweep(pts_comp)
            start = int(comp_idx[int(a_local)])
            end = int(comp_idx[int(b_local)])

            # 主路径：使用欧氏“最远点”端点 + 图最短路排序（比在有环表面图上直接做 diameter 更稳）
            adj_path = adj
            init_mode = str(getattr(config, "init_edge_cost_mode", "euclid") or "euclid").strip().lower()
            if init_mode in {"dir_penalty", "dir", "axis_penalty"}:
                try:
                    axis = _principal_axis_pca(pts_comp)
                    lam = float(getattr(config, "init_dir_penalty_lambda", 0.0))
                    adj_path = _reweight_adj_dir_penalty(adj, pts_cluster, axis=axis, lambda_=lam)
                except Exception:
                    adj_path = adj

            path_idx = shortest_path_nodes(adj_path, start, end)
            if len(path_idx) < 2:
                # fallback：仍保留旧逻辑，避免极端情况下无路可走
                path_idx = approximate_diameter_path(adj, comp_idx)
                if len(path_idx) < 2:
                    continue

            path_pts = pts_cluster[path_idx]
            init_curve = resample_polyline_by_step(path_pts, step=float(config.centerline_step()))

            debug_tag: Optional[str] = None
            if _debug_enabled(config, next_needle_id):
                debug_tag = f"segment_{next_needle_id:03d}_cluster{cluster_id:02d}_comp{comp_id:02d}"

            curve = refine_centerline_by_slabs(
                pts_comp,
                init_curve,
                config=config,
                debug_tag=debug_tag,
                segment_id=next_needle_id,
            )
            if debug_tag is not None:
                try:
                    dbg_dir = _debug_segment_dir(config, debug_tag)
                    _debug_write_ply_points(dbg_dir / "stage10_final_before_trunc_fix.ply", curve, color=[0.0, 1.0, 1.0])
                except Exception:
                    pass

            # width/thickness：优先 slab 截面统计；失败则退化为点到中心线距离分位数估计
            pin_width, pin_thickness = estimate_width_thickness_from_centerline_slabs(pts_comp, curve, config)
            if pin_width <= 0 or pin_thickness <= 0:
                curve_tree = cKDTree(curve)
                d_pt, _ = curve_tree.query(pts_comp)
                pin_thickness = float(2.0 * np.quantile(d_pt, 0.50))
                pin_width = float(2.0 * np.quantile(d_pt, 0.95))
            pin_width, pin_thickness = _clamp_pin_width_thickness_prior(pin_width, pin_thickness, config)

            quality_basic = evaluate_centerline_basic(pts_comp, curve)
            if pin_thickness > 0:
                quality_dist = evaluate_centerline_distance(
                    pts_comp,
                    curve,
                    pin_thickness=float(pin_thickness),
                    far_ratio=1.3,
                    max_longest_far_fraction=0.07,
                )
            else:
                quality_dist = {"status": "unknown"}

            overall_status = "ok"

            # 端面特征：用于判别截断端/自由端（仅做记录/调试，不对中心线施加硬约束）
            cap_feat = _endpoint_cap_features(pts_comp, curve, config, pin_width_hint=pin_width)
            base_end_idx_local, base_end_reason_local = _infer_truncated_end_from_cap(cap_feat, config)

            # 端点横向拉回（中心向外）：利用中部拟合较好的先验，从中点向两端扫描并拉回端部漂移。
            # 注意：此处只修复横向偏移（u/v），不改轴向长度（t）。
            pullback_end0_stats = {"applied": 0.0, "iters": 0.0, "moved_mm": 0.0, "end_shift_mm": 0.0}
            pullback_end1_stats = {"applied": 0.0, "iters": 0.0, "moved_mm": 0.0, "end_shift_mm": 0.0}
            pullback_stats = {"applied": 0.0, "iters": 0.0, "moved_mm": 0.0, "end_shift_mm": 0.0}  # compat: trunc end
            if bool(getattr(config, "trunc_end_pullback_enabled", False)):
                # End 0 (curve[0])
                curve_pb, pullback_end0_stats = pullback_truncated_end_center_out(
                    pts_comp,
                    curve,
                    base_end_idx_local=0,
                    config=config,
                    pin_width_hint=float(pin_width),
                    debug_suffix="start",
                    debug_tag=debug_tag,
                    segment_id=next_needle_id,
                )
                if float(pullback_end0_stats.get("applied", 0.0)) > 0.5:
                    curve = curve_pb
                    if debug_tag is not None:
                        try:
                            dbg_dir = _debug_segment_dir(config, debug_tag)
                            _debug_write_ply_points(dbg_dir / "stage10_pullback_start_after.ply", curve, color=[0.05, 0.05, 0.05])
                        except Exception:
                            pass

                # End 1 (curve[-1])
                curve_pb, pullback_end1_stats = pullback_truncated_end_center_out(
                    pts_comp,
                    curve,
                    base_end_idx_local=-1,
                    config=config,
                    pin_width_hint=float(pin_width),
                    debug_suffix="end",
                    debug_tag=debug_tag,
                    segment_id=next_needle_id,
                )
                if float(pullback_end1_stats.get("applied", 0.0)) > 0.5:
                    curve = curve_pb
                    if debug_tag is not None:
                        try:
                            dbg_dir = _debug_segment_dir(config, debug_tag)
                            _debug_write_ply_points(dbg_dir / "stage10_pullback_end_after.ply", curve, color=[0.05, 0.05, 0.05])
                        except Exception:
                            pass

                # Keep the legacy "trunc_pullback" stats as the inferred truncated end side (if available).
                if base_end_idx_local is not None:
                    try:
                        pullback_stats = (
                            dict(pullback_end0_stats) if int(base_end_idx_local) == 0 else dict(pullback_end1_stats)
                        )
                    except Exception:
                        pullback_stats = {"applied": 0.0, "iters": 0.0, "moved_mm": 0.0, "end_shift_mm": 0.0}

                # Update quality evaluation after BOTH ends are processed.
                if float(pullback_end0_stats.get("applied", 0.0)) > 0.5 or float(pullback_end1_stats.get("applied", 0.0)) > 0.5:
                    if debug_tag is not None:
                        try:
                            dbg_dir = _debug_segment_dir(config, debug_tag)
                            _debug_write_ply_points(dbg_dir / "stage10_pullback_after.ply", curve, color=[0.05, 0.05, 0.05])
                        except Exception:
                            pass

                    quality_basic = evaluate_centerline_basic(pts_comp, curve)
                    if pin_thickness > 0:
                        quality_dist = evaluate_centerline_distance(
                            pts_comp,
                            curve,
                            pin_thickness=float(pin_thickness),
                            far_ratio=1.3,
                            max_longest_far_fraction=0.07,
                        )
                    else:
                        quality_dist = {"status": "unknown"}

                    overall_status = "ok"

            # Stage13: body-guided regrowth (trim end windows, then regrow from the stable body outward).
            regrow_stats: Dict[str, object] = {
                "applied": 0.0,
                "rolled_back": 0.0,
                "rollback_reason": "",
                "accept_gate_warn": 0.0,
                "accept_gate_reason": "",
            }
            regrow_applied = False
            force_stage13_final = bool(getattr(config, "force_final_curve_stage13", False))
            stage13_curve_locked: Optional[np.ndarray] = None
            force_stage13_locked_applied = False
            if bool(getattr(config, "endpoint_regrow_enabled", False)):
                curve_before_regrow = np.asarray(curve, dtype=float).copy()
                quality_basic_before_regrow = dict(quality_basic)
                quality_dist_before_regrow = dict(quality_dist)

                endpoint0_before_regrow = None
                endpoint1_before_regrow = None
                endpoint0_after_regrow = None
                endpoint1_after_regrow = None
                try:
                    regrow_window_mm = float(
                        getattr(
                            config,
                            "endpoint_regrow_accept_endpoint_window_mm",
                            float(getattr(config, "endpoint_risk_window_mm", 3.0)),
                        )
                        or float(getattr(config, "endpoint_risk_window_mm", 3.0))
                    )
                    regrow_window_u = float(mm_to_unit(max(0.0, regrow_window_mm), float(config.unit_scale)))
                    regrow_tree = cKDTree(pts_comp)
                    endpoint0_before_regrow = _endpoint_region_slice_summary(
                        pts_comp,
                        curve_before_regrow,
                        config,
                        pin_width_hint=pin_width,
                        tree=regrow_tree,
                        window_u=regrow_window_u,
                    )
                    endpoint1_before_regrow = _endpoint_region_slice_summary(
                        pts_comp,
                        curve_before_regrow[::-1].copy(),
                        config,
                        pin_width_hint=pin_width,
                        tree=regrow_tree,
                        window_u=regrow_window_u,
                    )
                except Exception:
                    endpoint0_before_regrow = None
                    endpoint1_before_regrow = None

                curve_rg, regrow_stats = regrow_endpoints_body_guided(
                    pts_comp,
                    curve,
                    config=config,
                    pin_width_hint=float(pin_width),
                    pin_thickness_hint=float(pin_thickness),
                    debug_tag=debug_tag,
                    segment_id=next_needle_id,
                    cap_feat=cap_feat,
                    base_end_idx_local=base_end_idx_local,
                    endpoint0_before=endpoint0_before_regrow,
                    endpoint1_before=endpoint1_before_regrow,
                )
                regrow_stats = dict(regrow_stats or {})
                regrow_stats.setdefault("applied", 0.0)
                regrow_stats.setdefault("rolled_back", 0.0)
                regrow_stats.setdefault("rollback_reason", "")
                regrow_stats.setdefault("accept_gate_warn", 0.0)
                regrow_stats.setdefault("accept_gate_reason", "")

                if (len(curve_rg) >= 2) and (
                    float(regrow_stats.get("applied", 0.0) or 0.0) > 0.5
                    or float(regrow_stats.get("curve_changed", 0.0) or 0.0) > 0.5
                ):
                    curve = curve_rg
                    stage13_curve_locked = np.asarray(curve_rg, dtype=float).copy()
                    regrow_applied = bool(float(regrow_stats.get("applied", 0.0) or 0.0) > 0.5)
                    if debug_tag is not None:
                        try:
                            dbg_dir = _debug_segment_dir(config, debug_tag)
                            _debug_write_ply_points(dbg_dir / "stage13_after_regrow.ply", curve, color=[0.85, 0.65, 0.10])
                        except Exception:
                            pass

                    # Re-evaluate endpoint cap features / base-end inference on the regrown curve (endpoints changed).
                    cap_feat = _endpoint_cap_features(pts_comp, curve, config, pin_width_hint=pin_width)
                    base_end_idx_local, base_end_reason_local = _infer_truncated_end_from_cap(cap_feat, config)

                    # Update quality evaluation (final curve for this stage).
                    quality_basic = evaluate_centerline_basic(pts_comp, curve)
                    if pin_thickness > 0:
                        quality_dist = evaluate_centerline_distance(
                            pts_comp,
                            curve,
                            pin_thickness=float(pin_thickness),
                            far_ratio=1.3,
                            max_longest_far_fraction=0.07,
                        )
                    else:
                        quality_dist = {"status": "unknown"}

                    overall_status = "ok"

                    try:
                        regrow_window_mm = float(
                            getattr(
                                config,
                                "endpoint_regrow_accept_endpoint_window_mm",
                                float(getattr(config, "endpoint_risk_window_mm", 3.0)),
                            )
                            or float(getattr(config, "endpoint_risk_window_mm", 3.0))
                        )
                        regrow_window_u = float(mm_to_unit(max(0.0, regrow_window_mm), float(config.unit_scale)))
                        regrow_tree = cKDTree(pts_comp)
                        endpoint0_after_regrow = _endpoint_region_slice_summary(
                            pts_comp,
                            curve,
                            config,
                            pin_width_hint=pin_width,
                            tree=regrow_tree,
                            window_u=regrow_window_u,
                        )
                        endpoint1_after_regrow = _endpoint_region_slice_summary(
                            pts_comp,
                            curve[::-1].copy(),
                            config,
                            pin_width_hint=pin_width,
                            tree=regrow_tree,
                            window_u=regrow_window_u,
                        )
                    except Exception:
                        endpoint0_after_regrow = None
                        endpoint1_after_regrow = None

                    # Acceptance gate for stage13 candidate (supports rollback when configured).
                    if bool(getattr(config, "endpoint_regrow_accept_gate_enabled", False)):
                        worse, reason = _regrow_quality_worse(
                            quality_basic_before_regrow,
                            quality_dist_before_regrow,
                            quality_basic,
                            quality_dist,
                            config,
                            endpoint0_before=endpoint0_before_regrow,
                            endpoint0_after=endpoint0_after_regrow,
                            endpoint1_before=endpoint1_before_regrow,
                            endpoint1_after=endpoint1_after_regrow,
                        )
                        if worse:
                            regrow_stats["accept_gate_warn"] = 1.0
                            regrow_stats["accept_gate_reason"] = str(reason)
                            if bool(getattr(config, "endpoint_regrow_accept_gate_rollback", True)) and (not force_stage13_final):
                                regrow_stats["applied"] = 0.0
                                regrow_stats["rolled_back"] = 1.0
                                regrow_stats["rollback_reason"] = str(reason)
                                curve = curve_before_regrow
                                regrow_applied = False
                                quality_basic = dict(quality_basic_before_regrow)
                                quality_dist = dict(quality_dist_before_regrow)
                                overall_status = "ok"
                                try:
                                    cap_feat = _endpoint_cap_features(pts_comp, curve, config, pin_width_hint=pin_width)
                                    base_end_idx_local, base_end_reason_local = _infer_truncated_end_from_cap(cap_feat, config)
                                except Exception:
                                    pass
                                print(f"[WARN] regrow acceptance gate rollback: {reason}")
                            elif bool(getattr(config, "endpoint_regrow_accept_gate_rollback", True)) and force_stage13_final:
                                regrow_stats["rollback_blocked_by_force_stage13"] = 1.0
                                print(
                                    "[WARN] regrow acceptance gate flagged but force_final_curve_stage13 keeps "
                                    f"stage13 curve: {reason}"
                                )
                            else:
                                print(f"[WARN] regrow acceptance gate flagged (no rollback): {reason}")

            # 截断端修复：抑制“钩子/缩短”（仅对截断端生效，不影响自由端）
            if force_stage13_final and stage13_curve_locked is not None and len(stage13_curve_locked) >= 2:
                curve = np.asarray(stage13_curve_locked, dtype=float).copy()
                regrow_applied = True
                force_stage13_locked_applied = True
                regrow_stats["force_final_curve_stage13_applied"] = 1.0
                regrow_stats["force_final_curve_source"] = "stage13_after_regrow"
                print("[INFO] force_final_curve_stage13: lock stage13 curve and skip stage11/stage12.")

            trunc_stats = {"applied": 0.0, "extend_mm": 0.0, "rigid_mm": float(getattr(config, "trunc_end_rigid_mm", 0.0))}
            skip_postfix = bool(
                force_stage13_locked_applied
                or (regrow_applied and bool(getattr(config, "endpoint_regrow_skip_postfixes", True)))
            )
            if (not skip_postfix) and bool(getattr(config, "trunc_end_fix_enabled", False)) and base_end_idx_local is not None:
                curve_fixed, trunc_stats = fix_truncated_end_hook_and_shrink(
                    pts_comp,
                    curve,
                    base_end_idx_local=base_end_idx_local,
                    config=config,
                    pin_width_hint=pin_width,
                    pin_thickness_hint=pin_thickness,
                )
                if float(trunc_stats.get("applied", 0.0)) > 0.5:
                    curve = curve_fixed
                    if debug_tag is not None:
                        try:
                            dbg_dir = _debug_segment_dir(config, debug_tag)
                            _debug_write_ply_points(
                                dbg_dir / "stage11_after_trunc_fix.ply", curve, color=[1.0, 1.0, 0.0]
                            )
                        except Exception:
                            pass

                    # 更新质量评估（以最终中心线为准）
                    quality_basic = evaluate_centerline_basic(pts_comp, curve)
                    if pin_thickness > 0:
                        quality_dist = evaluate_centerline_distance(
                            pts_comp,
                            curve,
                            pin_thickness=float(pin_thickness),
                            far_ratio=1.3,
                            max_longest_far_fraction=0.07,
                        )
                    else:
                        quality_dist = {"status": "unknown"}

                    overall_status = "ok"

            # Endpoint bbox fix: ensure "not too short" on both ends, and optionally suppress hook-like artifacts on
            # the truncated end (data-driven; only triggers when outside points exist).
            bbox_stats = {
                "start": {"applied": 0.0, "extend_mm": 0.0, "n_out_axis": 0.0, "n_out_local": 0.0, "angle_deg": 0.0, "hook": 0.0},
                "end": {"applied": 0.0, "extend_mm": 0.0, "n_out_axis": 0.0, "n_out_local": 0.0, "angle_deg": 0.0, "hook": 0.0},
            }
            if (not skip_postfix) and bool(getattr(config, "endpoint_bbox_fix_enabled", False)):
                curve_fixed2, bbox_stats = fix_endpoints_by_bbox(
                    pts_comp,
                    curve,
                    base_end_idx_local=base_end_idx_local,
                    config=config,
                    pin_width_hint=pin_width,
                    pin_thickness_hint=pin_thickness,
                )
                if float(bbox_stats.get("start", {}).get("applied", 0.0)) > 0.5 or float(
                    bbox_stats.get("end", {}).get("applied", 0.0)
                ) > 0.5:
                    curve = curve_fixed2
                    if debug_tag is not None:
                        try:
                            dbg_dir = _debug_segment_dir(config, debug_tag)
                            _debug_write_ply_points(dbg_dir / "stage12_after_bbox_fix.ply", curve, color=[0.4, 0.9, 0.2])
                        except Exception:
                            pass

                    # 更新质量评估（以最终中心线为准）
                    quality_basic = evaluate_centerline_basic(pts_comp, curve)
                    if pin_thickness > 0:
                        quality_dist = evaluate_centerline_distance(
                            pts_comp,
                            curve,
                            pin_thickness=float(pin_thickness),
                            far_ratio=1.3,
                            max_longest_far_fraction=0.07,
                        )
                    else:
                        quality_dist = {"status": "unknown"}

                    overall_status = "ok"

            width_mm = float(pin_width / (config.unit_scale + 1e-12))
            thick_mm = float(pin_thickness / (config.unit_scale + 1e-12))
            pb_note = ""
            if float(pullback_end0_stats.get("applied", 0.0)) > 0.5 or float(pullback_end1_stats.get("applied", 0.0)) > 0.5:
                s_shift = float(pullback_end0_stats.get("end_shift_mm", 0.0) or 0.0)
                e_shift = float(pullback_end1_stats.get("end_shift_mm", 0.0) or 0.0)
                moved_total = float(pullback_end0_stats.get("moved_mm", 0.0) or 0.0) + float(pullback_end1_stats.get("moved_mm", 0.0) or 0.0)
                pb_note = (
                    f" | pullback=s{s_shift:.2f}mm e{e_shift:.2f}mm"
                    f" moved{moved_total:.2f}mm"
                )
            trunc_note = ""
            if float(trunc_stats.get("applied", 0.0)) > 0.5:
                trunc_note = f" | trunc_fix=+{float(trunc_stats.get('extend_mm', 0.0)):.2f}mm@{float(trunc_stats.get('rigid_mm', 0.0)):.1f}mm"
            bbox_note = ""
            if float(bbox_stats.get("start", {}).get("applied", 0.0)) > 0.5 or float(bbox_stats.get("end", {}).get("applied", 0.0)) > 0.5:
                s_ext = float(bbox_stats.get("start", {}).get("extend_mm", 0.0))
                e_ext = float(bbox_stats.get("end", {}).get("extend_mm", 0.0))
                s_hook = int(float(bbox_stats.get("start", {}).get("hook", 0.0)) > 0.5)
                e_hook = int(float(bbox_stats.get("end", {}).get("hook", 0.0)) > 0.5)
                bbox_note = f" | bbox_fix=s+{s_ext:.2f}mm(h{s_hook}) e+{e_ext:.2f}mm(h{e_hook})"
            regrow_note = ""
            if regrow_applied:
                try:
                    s_g = float(((regrow_stats.get("start") or {}).get("grown_mm", 0.0)) or 0.0)
                    e_g = float(((regrow_stats.get("end") or {}).get("grown_mm", 0.0)) or 0.0)
                    t_s = float(regrow_stats.get("trim_start_mm", 0.0) or 0.0)
                    t_e = float(regrow_stats.get("trim_end_mm", 0.0) or 0.0)
                    regrow_note = f" | regrow=+{s_g:.1f}/{e_g:.1f}mm trim{t_s:.1f}/{t_e:.1f}mm"
                    if float(regrow_stats.get("accept_gate_warn", 0.0) or 0.0) > 0.5:
                        regrow_note += " [gate-flag]"
                except Exception:
                    regrow_note = " | regrow=1"
            final_note = ""
            if force_stage13_locked_applied:
                final_note = " | final=stage13_locked"
            print(
                f"[INFO] segment {next_needle_id} (cluster={cluster_id}, comp={comp_id})"
                f" | points={len(pts_comp)} | width~{width_mm:.3f}mm | thickness~{thick_mm:.3f}mm"
                f" | p90_dist={quality_basic['p90']:.6g} | status={overall_status}{pb_note}{regrow_note}{trunc_note}"
                f"{bbox_note}"
                f"{final_note}"
            )

            needle_infos.append(
                {
                    "needle_id": int(next_needle_id),
                    "cluster_id": int(cluster_id),
                    # NOTE: stable_id must be unique per final component. It will be assigned after all components
                    # are extracted (since one DBSCAN cluster can be split into multiple connected-components).
                    "stable_id": -1,
                    "component_id": int(comp_id),
                    "center_point": np.mean(curve, axis=0),
                    "curve_global_final": curve,
                    "pin_width": float(pin_width),
                    "pin_thickness": float(pin_thickness),
                    "overall_status": overall_status,
                    "quality_basic": quality_basic,
                    "quality_dist": quality_dist,
                    "endpoint_cap": cap_feat,
                    "base_end_idx_local": base_end_idx_local,
                    "base_end_reason_local": base_end_reason_local,
                    "end_pullback": {"start": pullback_end0_stats, "end": pullback_end1_stats},
                    "trunc_pullback": pullback_stats,
                    "regrow": regrow_stats,
                    "trunc_fix": trunc_stats,
                    "bbox_fix": bbox_stats,
                    "final_curve_source": "stage13_after_regrow" if force_stage13_locked_applied else "pipeline_postfix",
                    "points": pts_comp,
                }
            )
            next_needle_id += 1

    # Assign stable_id on the FINAL extracted components (not on DBSCAN clusters).
    stable_order_meta = {}
    if needle_infos:
        centers_as_clusters: List[np.ndarray] = []
        center_to_info_idx: List[int] = []
        for i, info in enumerate(needle_infos):
            c = np.asarray(info.get("center_point", []), dtype=float).reshape(-1)
            if c.size != 3 or not np.all(np.isfinite(c)):
                continue
            centers_as_clusters.append(c.reshape(1, 3))
            center_to_info_idx.append(int(i))

        if centers_as_clusters:
            order_idx, stable_order_meta = stable_order_clusters(centers_as_clusters)
            for stable_id, local_idx in enumerate(order_idx):
                info_idx = center_to_info_idx[int(local_idx)]
                needle_infos[info_idx]["stable_id"] = int(stable_id)

        # Always print a quick sanity summary (helps GT-eval debug and regression checks).
        sids = [
            int(info.get("stable_id", -1))
            for info in needle_infos
            if int(info.get("stable_id", -1)) >= 0
        ]
        uniq = len(set(sids))
        total = len(sids)
        dup = total - uniq
        print(f"[INFO] stable_id assignment: total={total} unique={uniq} dup={dup}")

        if _DAIMA_CTX is not None and _DAIMA_CTX.run_dir is not None:
            try:
                (_DAIMA_CTX.run_dir / "stable_order_meta.json").write_text(
                    json.dumps(stable_order_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # Optional: emit a per-segment summary table under debug_out (for quick regression checks).
        # This is only enabled when debug_dump_dir is set (typically via CLI: --debug_dump).
        if getattr(config, "debug_dump_dir", None):
            try:
                out_root = Path(str(getattr(config, "debug_dump_dir"))).expanduser()
                out_root.mkdir(parents=True, exist_ok=True)

                unit_scale = float(getattr(config, "unit_scale", UNIT_SCALE))
                denom = float(unit_scale + 1e-12)

                rows = []
                for info in needle_infos:
                    # Avoid dumping large arrays (e.g. "points") into JSON/CSV.
                    qb = dict(info.get("quality_basic") or {})
                    qd = dict(info.get("quality_dist") or {})
                    pullback = dict(info.get("trunc_pullback") or {})
                    end_pullback = dict(info.get("end_pullback") or {})
                    pb_start = dict((end_pullback.get("start") or {}))
                    pb_end = dict((end_pullback.get("end") or {}))
                    trunc_fix = dict(info.get("trunc_fix") or {})
                    bbox_fix = dict(info.get("bbox_fix") or {})
                    regrow = dict(info.get("regrow") or {})
                    rg_start = dict((regrow.get("start") or {}))
                    rg_end = dict((regrow.get("end") or {}))

                    def _mm(x: float) -> float:
                        return float(x / denom)

                    row = {
                        "stable_id": int(info.get("stable_id", -1)),
                        "needle_id": int(info.get("needle_id", -1)),
                        "cluster_id": int(info.get("cluster_id", -1)),
                        "component_id": int(info.get("component_id", -1)),
                        "n_points": int(len(info.get("points"))) if info.get("points", None) is not None else 0,
                        "overall_status": str(info.get("overall_status", "")),
                        "pin_width_mm": _mm(float(info.get("pin_width", 0.0) or 0.0)),
                        "pin_thickness_mm": _mm(float(info.get("pin_thickness", 0.0) or 0.0)),
                        "dist_p50_mm": _mm(float(qb.get("p50", 0.0) or 0.0)),
                        "dist_p90_mm": _mm(float(qb.get("p90", 0.0) or 0.0)),
                        "dist_p99_mm": _mm(float(qb.get("p99", 0.0) or 0.0)),
                        "dist_mean_mm": _mm(float(qb.get("mean", 0.0) or 0.0)),
                        "dist_max_mm": _mm(float(qb.get("max", 0.0) or 0.0)),
                        "far_fraction": float(qd.get("far_fraction", 0.0) or 0.0),
                        "longest_far_fraction": float(qd.get("longest_far_fraction", 0.0) or 0.0),
                        "pullback_start_applied": float(pb_start.get("applied", 0.0) or 0.0),
                        "pullback_start_moved_mm": float(pb_start.get("moved_mm", 0.0) or 0.0),
                        "pullback_start_end_shift_mm": float(pb_start.get("end_shift_mm", 0.0) or 0.0),
                        "pullback_start_fold_reject_count": float(pb_start.get("fold_reject_count", 0.0) or 0.0),
                        "pullback_start_fold_max_angle_deg": float(pb_start.get("fold_max_angle_deg", 0.0) or 0.0),
                        "pullback_start_lowcov_force_skip_count": float(
                            pb_start.get("lowcov_force_skip_count", 0.0) or 0.0
                        ),
                        "pullback_end_applied": float(pb_end.get("applied", 0.0) or 0.0),
                        "pullback_end_moved_mm": float(pb_end.get("moved_mm", 0.0) or 0.0),
                        "pullback_end_end_shift_mm": float(pb_end.get("end_shift_mm", 0.0) or 0.0),
                        "pullback_end_fold_reject_count": float(pb_end.get("fold_reject_count", 0.0) or 0.0),
                        "pullback_end_fold_max_angle_deg": float(pb_end.get("fold_max_angle_deg", 0.0) or 0.0),
                        "pullback_end_lowcov_force_skip_count": float(
                            pb_end.get("lowcov_force_skip_count", 0.0) or 0.0
                        ),
                        "trunc_pullback_applied": float(pullback.get("applied", 0.0) or 0.0),
                        "trunc_pullback_moved_mm": float(pullback.get("moved_mm", 0.0) or 0.0),
                        "trunc_pullback_end_shift_mm": float(pullback.get("end_shift_mm", 0.0) or 0.0),
                        "trunc_fix_applied": float(trunc_fix.get("applied", 0.0) or 0.0),
                        "trunc_fix_rolled_back": float(trunc_fix.get("rolled_back", 0.0) or 0.0),
                        "trunc_fix_extend_mm": float(trunc_fix.get("extend_mm", 0.0) or 0.0),
                        "regrow_applied": float(regrow.get("applied", 0.0) or 0.0),
                        "regrow_curve_changed": float(regrow.get("curve_changed", 0.0) or 0.0),
                        "regrow_center_mode": str(regrow.get("center_mode", "")),
                        "regrow_rolled_back": float(regrow.get("rolled_back", 0.0) or 0.0),
                        "regrow_rollback_reason": str(regrow.get("rollback_reason", "")),
                        "regrow_accept_gate_warn": float(regrow.get("accept_gate_warn", 0.0) or 0.0),
                        "regrow_accept_gate_reason": str(regrow.get("accept_gate_reason", "")),
                        "regrow_triclass_enabled": float(regrow.get("triclass_enabled", 0.0) or 0.0),
                        "regrow_triclass_variant": str(regrow.get("triclass_variant", "")),
                        "regrow_tri_start_action": str(regrow.get("triclass_start_action", "")),
                        "regrow_tri_start_extend_mode": str(regrow.get("triclass_start_extend_mode", "")),
                        "regrow_tri_end_action": str(regrow.get("triclass_end_action", "")),
                        "regrow_tri_end_extend_mode": str(regrow.get("triclass_end_extend_mode", "")),
                        "regrow_tri_start_reason": str(regrow.get("triclass_start_hard_reason", "")),
                        "regrow_tri_end_reason": str(regrow.get("triclass_end_hard_reason", "")),
                        "regrow_tri_start_forward_hit": float(regrow.get("triclass_start_forward_hit", 0.0) or 0.0),
                        "regrow_tri_end_forward_hit": float(regrow.get("triclass_end_forward_hit", 0.0) or 0.0),
                        "regrow_tri_start_forward_min_points": float(
                            regrow.get("triclass_start_forward_min_points", 0.0) or 0.0
                        ),
                        "regrow_tri_end_forward_min_points": float(
                            regrow.get("triclass_end_forward_min_points", 0.0) or 0.0
                        ),
                        "regrow_tri_start_forward_support_n": float(
                            regrow.get("triclass_start_forward_support_n", 0.0) or 0.0
                        ),
                        "regrow_tri_end_forward_support_n": float(
                            regrow.get("triclass_end_forward_support_n", 0.0) or 0.0
                        ),
                        "regrow_tri_start_backward_valid": float(regrow.get("triclass_start_backward_valid", 0.0) or 0.0),
                        "regrow_tri_end_backward_valid": float(regrow.get("triclass_end_backward_valid", 0.0) or 0.0),
                        "regrow_tri_start_backward_reason": str(regrow.get("triclass_start_backward_reason", "")),
                        "regrow_tri_end_backward_reason": str(regrow.get("triclass_end_backward_reason", "")),
                        "regrow_tri_start_backward_hit": float(regrow.get("triclass_start_backward_hit", 0.0) or 0.0),
                        "regrow_tri_end_backward_hit": float(regrow.get("triclass_end_backward_hit", 0.0) or 0.0),
                        "regrow_tri_start_backward_near_hit": float(
                            regrow.get("triclass_start_backward_near_hit", 0.0) or 0.0
                        ),
                        "regrow_tri_end_backward_near_hit": float(
                            regrow.get("triclass_end_backward_near_hit", 0.0) or 0.0
                        ),
                        "regrow_tri_start_backward_hit_step": float(
                            regrow.get("triclass_start_backward_hit_step", 0.0) or 0.0
                        ),
                        "regrow_tri_end_backward_hit_step": float(
                            regrow.get("triclass_end_backward_hit_step", 0.0) or 0.0
                        ),
                        "regrow_tri_start_backward_hit_distance_mm": float(
                            regrow.get("triclass_start_backward_hit_distance_mm", 0.0) or 0.0
                        ),
                        "regrow_tri_end_backward_hit_distance_mm": float(
                            regrow.get("triclass_end_backward_hit_distance_mm", 0.0) or 0.0
                        ),
                        "regrow_tri_start_backward_support_n": float(
                            regrow.get("triclass_start_backward_support_n", 0.0) or 0.0
                        ),
                        "regrow_tri_end_backward_support_n": float(
                            regrow.get("triclass_end_backward_support_n", 0.0) or 0.0
                        ),
                        "regrow_tri_start_support_n1": float(regrow.get("triclass_start_support_n1", 0.0) or 0.0),
                        "regrow_tri_end_support_n1": float(regrow.get("triclass_end_support_n1", 0.0) or 0.0),
                        "regrow_tri_start_cov1": float(regrow.get("triclass_start_cov1", 0.0) or 0.0),
                        "regrow_tri_end_cov1": float(regrow.get("triclass_end_cov1", 0.0) or 0.0),
                        "regrow_tri_start_rv1": float(regrow.get("triclass_start_radial_var1", 0.0) or 0.0),
                        "regrow_tri_end_rv1": float(regrow.get("triclass_end_radial_var1", 0.0) or 0.0),
                        "regrow_tri_start_corridor_valid": float(
                            regrow.get("triclass_start_corridor_valid", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_valid": float(
                            regrow.get("triclass_end_corridor_valid", 0.0) or 0.0
                        ),
                        "regrow_tri_start_corridor_reason": str(
                            regrow.get("triclass_start_corridor_reason", "")
                        ),
                        "regrow_tri_end_corridor_reason": str(
                            regrow.get("triclass_end_corridor_reason", "")
                        ),
                        "regrow_tri_start_corridor_steps": float(
                            regrow.get("triclass_start_corridor_valid_steps", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_steps": float(
                            regrow.get("triclass_end_corridor_valid_steps", 0.0) or 0.0
                        ),
                        "regrow_tri_start_corridor_support_sum": float(
                            regrow.get("triclass_start_corridor_support_sum", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_support_sum": float(
                            regrow.get("triclass_end_corridor_support_sum", 0.0) or 0.0
                        ),
                        "regrow_tri_start_corridor_support_peak": float(
                            regrow.get("triclass_start_corridor_support_peak", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_support_peak": float(
                            regrow.get("triclass_end_corridor_support_peak", 0.0) or 0.0
                        ),
                        "regrow_tri_start_corridor_cov_mean": float(
                            regrow.get("triclass_start_corridor_cov_mean", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_cov_mean": float(
                            regrow.get("triclass_end_corridor_cov_mean", 0.0) or 0.0
                        ),
                        "regrow_tri_start_corridor_rv_max": float(
                            regrow.get("triclass_start_corridor_rv_max", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_rv_max": float(
                            regrow.get("triclass_end_corridor_rv_max", 0.0) or 0.0
                        ),
                        "regrow_tri_start_corridor_pass": float(
                            regrow.get("triclass_start_corridor_pass", 0.0) or 0.0
                        ),
                        "regrow_tri_end_corridor_pass": float(
                            regrow.get("triclass_end_corridor_pass", 0.0) or 0.0
                        ),
                        "regrow_tri_start_adaptive_used": float(
                            regrow.get("triclass_start_adaptive_used", 0.0) or 0.0
                        ),
                        "regrow_tri_end_adaptive_used": float(
                            regrow.get("triclass_end_adaptive_used", 0.0) or 0.0
                        ),
                        "regrow_tri_start_extend_support_min_eff": float(
                            regrow.get("triclass_start_extend_support_min_eff", 0.0) or 0.0
                        ),
                        "regrow_tri_end_extend_support_min_eff": float(
                            regrow.get("triclass_end_extend_support_min_eff", 0.0) or 0.0
                        ),
                        "regrow_tri_start_extend_cov_min_eff": float(
                            regrow.get("triclass_start_extend_cov_min_eff", 0.0) or 0.0
                        ),
                        "regrow_tri_end_extend_cov_min_eff": float(
                            regrow.get("triclass_end_extend_cov_min_eff", 0.0) or 0.0
                        ),
                        "regrow_tri_start_extend_rv_max_eff": float(
                            regrow.get("triclass_start_extend_rv_max_eff", 0.0) or 0.0
                        ),
                        "regrow_tri_end_extend_rv_max_eff": float(
                            regrow.get("triclass_end_extend_rv_max_eff", 0.0) or 0.0
                        ),
                        "regrow_tri_start_fail_count": float(regrow.get("triclass_start_risk_fail_count", 0.0) or 0.0),
                        "regrow_tri_end_fail_count": float(regrow.get("triclass_end_risk_fail_count", 0.0) or 0.0),
                        "regrow_tri_start_risk_hit_min_n_slab": float(
                            regrow.get("triclass_start_risk_hit_min_n_slab", 0.0) or 0.0
                        ),
                        "regrow_tri_end_risk_hit_min_n_slab": float(
                            regrow.get("triclass_end_risk_hit_min_n_slab", 0.0) or 0.0
                        ),
                        "regrow_tri_start_risk_hit_min_cov": float(
                            regrow.get("triclass_start_risk_hit_min_cov", 0.0) or 0.0
                        ),
                        "regrow_tri_end_risk_hit_min_cov": float(
                            regrow.get("triclass_end_risk_hit_min_cov", 0.0) or 0.0
                        ),
                        "regrow_tri_start_risk_hit_max_delta": float(
                            regrow.get("triclass_start_risk_hit_max_delta", 0.0) or 0.0
                        ),
                        "regrow_tri_end_risk_hit_max_delta": float(
                            regrow.get("triclass_end_risk_hit_max_delta", 0.0) or 0.0
                        ),
                        "regrow_tri_start_risk_hit_clipped": float(
                            regrow.get("triclass_start_risk_hit_clipped", 0.0) or 0.0
                        ),
                        "regrow_tri_end_risk_hit_clipped": float(
                            regrow.get("triclass_end_risk_hit_clipped", 0.0) or 0.0
                        ),
                        "regrow_tri_start_risk_hit_max_rv": float(
                            regrow.get("triclass_start_risk_hit_max_rv", 0.0) or 0.0
                        ),
                        "regrow_tri_end_risk_hit_max_rv": float(
                            regrow.get("triclass_end_risk_hit_max_rv", 0.0) or 0.0
                        ),
                        "regrow_tri_start_risk_hit_text": str(regrow.get("triclass_start_risk_hit_text", "")),
                        "regrow_tri_end_risk_hit_text": str(regrow.get("triclass_end_risk_hit_text", "")),
                        "regrow_tri_start_shorten_applied": float(
                            regrow.get("triclass_shorten_start_applied", 0.0) or 0.0
                        ),
                        "regrow_tri_end_shorten_applied": float(
                            regrow.get("triclass_shorten_end_applied", 0.0) or 0.0
                        ),
                        "regrow_tri_shorten_extra_mm": float(regrow.get("triclass_shorten_extra_mm", 0.0) or 0.0),
                        "regrow_trim_start_mm": float(regrow.get("trim_start_mm", 0.0) or 0.0),
                        "regrow_trim_end_mm": float(regrow.get("trim_end_mm", 0.0) or 0.0),
                        "regrow_trim_start_target_mm": float(regrow.get("trim_start_target_mm", 0.0) or 0.0),
                        "regrow_trim_end_target_mm": float(regrow.get("trim_end_target_mm", 0.0) or 0.0),
                        "regrow_trim_start_target_mm_raw": float(
                            regrow.get("trim_start_target_mm_raw", 0.0) or 0.0
                        ),
                        "regrow_trim_end_target_mm_raw": float(
                            regrow.get("trim_end_target_mm_raw", 0.0) or 0.0
                        ),
                        "regrow_deflect_trim_enabled": float(regrow.get("deflect_trim_enabled", 0.0) or 0.0),
                        "regrow_deflect_start_detected": float(regrow.get("deflect_start_detected", 0.0) or 0.0),
                        "regrow_deflect_end_detected": float(regrow.get("deflect_end_detected", 0.0) or 0.0),
                        "regrow_deflect_start_onset_mm": float(regrow.get("deflect_start_onset_mm", 0.0) or 0.0),
                        "regrow_deflect_end_onset_mm": float(regrow.get("deflect_end_onset_mm", 0.0) or 0.0),
                        "regrow_deflect_start_angle_peak_deg": float(
                            regrow.get("deflect_start_angle_peak_deg", 0.0) or 0.0
                        ),
                        "regrow_deflect_end_angle_peak_deg": float(
                            regrow.get("deflect_end_angle_peak_deg", 0.0) or 0.0
                        ),
                        "regrow_deflect_start_lateral_peak_mm": float(
                            regrow.get("deflect_start_lateral_peak_mm", 0.0) or 0.0
                        ),
                        "regrow_deflect_end_lateral_peak_mm": float(
                            regrow.get("deflect_end_lateral_peak_mm", 0.0) or 0.0
                        ),
                        "regrow_deflect_start_run_len": float(regrow.get("deflect_start_run_len", 0.0) or 0.0),
                        "regrow_deflect_end_run_len": float(regrow.get("deflect_end_run_len", 0.0) or 0.0),
                        "regrow_half_thickness_factor": float(regrow.get("half_thickness_factor", 0.0) or 0.0),
                        "regrow_start_steps": int(rg_start.get("steps", 0) or 0),
                        "regrow_end_steps": int(rg_end.get("steps", 0) or 0),
                        "regrow_start_grown_mm": float(rg_start.get("grown_mm", 0.0) or 0.0),
                        "regrow_end_grown_mm": float(rg_end.get("grown_mm", 0.0) or 0.0),
                        "regrow_start_soft_events": int(rg_start.get("soft_events", 0) or 0),
                        "regrow_end_soft_events": int(rg_end.get("soft_events", 0) or 0),
                        "regrow_start_guard_hits": int(rg_start.get("guard_hits", 0) or 0),
                        "regrow_end_guard_hits": int(rg_end.get("guard_hits", 0) or 0),
                        "regrow_start_guard_bbox_fail": int(rg_start.get("guard_bbox_fail", 0) or 0),
                        "regrow_end_guard_bbox_fail": int(rg_end.get("guard_bbox_fail", 0) or 0),
                        "regrow_start_guard_polar_fail": int(rg_start.get("guard_polar_fail", 0) or 0),
                        "regrow_end_guard_polar_fail": int(rg_end.get("guard_polar_fail", 0) or 0),
                        "regrow_start_force_pred_only": float(rg_start.get("force_pred_only", 0.0) or 0.0),
                        "regrow_end_force_pred_only": float(rg_end.get("force_pred_only", 0.0) or 0.0),
                        "regrow_start_guard_bbox_grace_steps": int(
                            rg_start.get("guard_bbox_grace_steps", 0) or 0
                        ),
                        "regrow_end_guard_bbox_grace_steps": int(
                            rg_end.get("guard_bbox_grace_steps", 0) or 0
                        ),
                        "regrow_start_bbox_probe_used": int(rg_start.get("bbox_probe_used", 0) or 0),
                        "regrow_end_bbox_probe_used": int(rg_end.get("bbox_probe_used", 0) or 0),
                        "regrow_start_direction_lock_triggered": int(
                            rg_start.get("direction_lock_triggered", 0) or 0
                        ),
                        "regrow_end_direction_lock_triggered": int(
                            rg_end.get("direction_lock_triggered", 0) or 0
                        ),
                        "regrow_start_stop": str(rg_start.get("stop", "")),
                        "regrow_end_stop": str(rg_end.get("stop", "")),
                        "bbox_fix_start_extend_mm": float((bbox_fix.get("start") or {}).get("extend_mm", 0.0) or 0.0),
                        "bbox_fix_end_extend_mm": float((bbox_fix.get("end") or {}).get("extend_mm", 0.0) or 0.0),
                        "base_end_reason_local": str(info.get("base_end_reason_local", "")),
                    }
                    rows.append(row)

                rows.sort(key=lambda r: (int(r.get("stable_id", -1)), int(r.get("needle_id", -1))))

                csv_path = out_root / "centerline_summary.csv"
                json_path = out_root / "centerline_summary.json"

                fieldnames = list(rows[0].keys()) if rows else []
                if fieldnames:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        for r in rows:
                            w.writerow(r)

                payload = {
                    "meta": {
                        "pcd_path": str(pcd_path),
                        "run_id": str(getattr(_DAIMA_CTX, "run_id", "")) if _DAIMA_CTX is not None else "",
                        "init_edge_cost_mode": str(getattr(config, "init_edge_cost_mode", "")),
                        "init_dir_penalty_lambda": float(getattr(config, "init_dir_penalty_lambda", 0.0)),
                        "slab_single_side_action": str(getattr(config, "slab_single_side_action", "")),
                        "slab_min_coverage": float(getattr(config, "slab_min_coverage", 0.0)),
                        "slab_coverage_bins": int(getattr(config, "slab_coverage_bins", 0)),
                    },
                    "rows": rows,
                }
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                if _DAIMA_CTX is not None:
                    _DAIMA_CTX.snapshot_paths(outputs={"centerline_summary_csv": str(csv_path), "centerline_summary_json": str(json_path)})
            except Exception:
                pass

    if visualize:
        # centerlines 可视化：使用最终（后处理后的）曲线
        for info in needle_infos:
            curve = np.asarray(info.get("curve_global_final", []), dtype=float)
            if len(curve) < 2:
                continue
            status = str(info.get("overall_status", "ok"))
            final_color = [0.0, 1.0, 0.0] if status == "ok" else [1.0, 0.0, 0.0]
            tube_final = create_tube_from_points(curve, radius=FINAL_TUBE_RADIUS, color=final_color)
            geometries.append(tube_final)

        for g in geometries:
            if isinstance(g, o3d.geometry.TriangleMesh):
                g.compute_vertex_normals()

        print("[INFO] Visualizing: raw cloud + final centerlines (green/red)")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Needle centerlines (green/red: final)",
            width=1280,
            height=800,
        )

    return needle_infos


def fit_curve_legacy(pcd_path, visualize=True, show_raw_cloud=True):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(pcd)

    pts_all = np.asarray(pcd.points)
    diag, median_nn = estimate_cloud_stats(pts_all)
    print(
        "[INFO] Units: "
        f"UNIT_SCALE={UNIT_SCALE:g} | "
        f"DBSCAN_EPS={DBSCAN_EPS:g} | "
        f"RAW_TUBE_RADIUS={RAW_TUBE_RADIUS:g} | "
        f"FINAL_TUBE_RADIUS={FINAL_TUBE_RADIUS:g}"
    )
    print(f"[INFO] Cloud stats: diag={diag:.6g} | median_nn={median_nn:.6g}")

    print("[INFO] 进行 DBSCAN 聚类...")
    labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_EPS,
                                         min_points=DBSCAN_MIN_POINTS))
    max_label = labels.max()
    print(f"[INFO] 检测到 {max_label + 1} 根针")

    cluster_points = []
    for cluster_id in range(max_label + 1):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            cluster_points.append(np.zeros((0, 3), dtype=float))
        else:
            cluster_points.append(pts_all[idx])
    order_idx, _ = stable_order_clusters(cluster_points)
    stable_id_map = {int(cluster_idx): int(stable_id) for stable_id, cluster_idx in enumerate(order_idx)}

    geometries = []
    needle_infos = []  # 收集每根针的中心线与相关信息

    for i in range(max_label + 1):
        idx = np.where(labels == i)[0]
        sub_pcd = pcd.select_by_index(idx)

        pts = np.asarray(sub_pcd.points)
        if len(pts) < 50:
            print(f"[WARN] 针 {i} 点数过少，跳过")
            continue

        # 初始颜色设为中性灰
        sub_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if show_raw_cloud:
            geometries.append(sub_pcd)

        # 使用 PCA 找到主轴，作为局部坐标系的 z 轴
        mean = pts.mean(axis=0)
        pts_centered = pts - mean
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        z_axis = eigvecs[:, 0]
        # 确保 z 轴大致向上（也可以根据具体设备约定）
        if z_axis[2] < 0:
            z_axis = -z_axis

        # 构造与 z 正交的 x, y 轴
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, z_axis)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(tmp, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-8)

        R_local = np.stack([x_axis, y_axis, z_axis], axis=1)
        pts_local = pts_centered @ R_local

        xy_data = pts_local[:, :2]
        cov_2d = np.cov(xy_data.T)
        eigvals_2d, eigvecs_2d = np.linalg.eig(cov_2d)

        # 构造 2D 旋转矩阵 (扩展为 3x3)
        R_2d = np.eye(3)
        R_2d[:2, :2] = eigvecs_2d  # 这里的特征向量就是主轴方向

        # 更新 pts_local 和 全局旋转矩阵
        pts_local = pts_local @ R_2d
        R_local = R_local @ R_2d

        z_vals = pts_local[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()

        x_extent = np.percentile(pts_local[:, 0], 95) - np.percentile(pts_local[:, 0], 5)
        y_extent = np.percentile(pts_local[:, 1], 95) - np.percentile(pts_local[:, 1], 5)
        pin_width = max(x_extent, y_extent)
        pin_thickness = min(x_extent, y_extent)

        rough_slices = int(len(pts) / TARGET_POINTS_PER_SLICE)
        n_slices = max(MIN_SLICES, min(MAX_SLICES, rough_slices))
        if n_slices < 3:
            print(f"[WARN] 针 {i} 截面数过少（估计 {n_slices}），跳过")
            continue

        z_grids = np.linspace(z_min, z_max, n_slices + 1)
        step_height = (z_max - z_min) / max(1, n_slices)

        centroids = []
        densities = []
        last_valid_centroid = None
        profile_z = []
        profile_r_mean = []
        profile_r_std = []
        profile_r_max = []
        profile_planarity = []

        for k in range(n_slices):
            z0, z1 = z_grids[k], z_grids[k + 1]
            mask_layer = (z_vals >= z0) & (z_vals < z1)
            layer_pts = pts_local[mask_layer]
            if len(layer_pts) < 3:
                continue

            curr_centroid = get_robust_slice_center(
                layer_pts,
                prev_center=last_valid_centroid,
                max_jump=pin_width * 0.5
            )
            slice_feats = compute_slice_features(layer_pts, curr_centroid)
            # 存储特征
            profile_z.append(curr_centroid[2])
            profile_r_mean.append(slice_feats["r_mean"])
            profile_r_std.append(slice_feats["r_std"])
            profile_r_max.append(slice_feats["r_max"])
            profile_planarity.append(slice_feats["linearity"])

            if last_valid_centroid is not None:
                dist_jump = np.linalg.norm(curr_centroid - last_valid_centroid)
                if dist_jump > max(pin_width * 3.0, step_height * 5.0):
                    # 认为当前截面中心跳变过大，是异常截面，跳过
                    continue

            centroids.append(curr_centroid)
            densities.append(len(layer_pts))
            last_valid_centroid = curr_centroid

        if len(centroids) < 4:
            print(f"[WARN] 针 {i} 有效截面数过少（{len(centroids)}），跳过")
            continue

        centroids = np.array(centroids)
        densities = np.array(densities)

        # 拟合原始样条中心线（蓝线）
        try:
            tck_raw, u_raw = splprep(centroids.T, s=0)
            u_fine = np.linspace(0.0, 1.0, 200)
            x_s, y_s, z_s = splev(u_fine, tck_raw)
            curve_local_raw = np.vstack([x_s, y_s, z_s]).T
            curve_global_raw = curve_local_raw @ R_local.T + mean

            tube_raw = create_tube_from_points(
                curve_global_raw,
                radius=RAW_TUBE_RADIUS,
                color=[0.0, 0.0, 1.0],
            )
            geometries.append(tube_raw)
        except Exception as e:
            print(f"[WARN] 针 {i} 原始样条拟合失败: {e}")

        # 最终中心线 + 风险评估
        try:
            curve_local_final = build_weighted_centerline(centroids, densities)
            curve_global_final = curve_local_final @ R_local.T + mean

            typical_radius = 0.5 * pin_width
            axial_window = step_height * 1.2

            quality_density = evaluate_centerline_density(
                pts_local,
                curve_local_final,
                typical_radius=typical_radius,
                axial_window=axial_window,
            )

            quality_dist = evaluate_centerline_distance(
                pts_local,
                curve_local_final,
                pin_thickness=pin_thickness,
                far_ratio=1.3,
                max_longest_far_fraction=0.07,
            )

            # Keep diagnostic metrics, but disable high_risk reporting/visual coloring.
            overall_status = "ok"
            final_color = [0.0, 1.0, 0.0]
            print(
                f"[INFO] ??{i} ???????????K"
                f" | median_density={quality_density['median_density']:.1f}"
                f" | longest_low_fraction={quality_density['longest_low_fraction']:.2f}"
                f" | median_dist={quality_dist['median_dist']:.3f}"
                f" | longest_far_fraction={quality_dist['longest_far_fraction']:.2f}"
            )

            tube_final = create_tube_from_points(
                curve_global_final,
                radius=FINAL_TUBE_RADIUS,
                color=final_color,
            )
            geometries.append(tube_final)

            # ===== 新增：构建本根针的中心线记录 =====
            needle_info = {
                "needle_id": int(i),
                "stable_id": int(stable_id_map.get(i, i)),
                "center_point": mean.copy(),                # 当前坐标系下的针质心
                "curve_global_final": curve_global_final,   # (M,3) 的 numpy 数组
                "pin_width": float(pin_width),
                "pin_thickness": float(pin_thickness),
                "step_height": float(step_height),
                "overall_status": overall_status,
                "quality_density": quality_density,
                "quality_dist": quality_dist,
                "profile_z": np.array(profile_z),  # z坐标 (用于对齐)
                "profile_r_mean": np.array(profile_r_mean),  # 平均半径 (检测粗细/缺肉)
                "profile_r_max": np.array(profile_r_max),  # 最大半径 (检测毛刺/凸起)
                "profile_planarity": np.array(profile_planarity),  # 平面度 (检测是否被压扁)
                "points": pts,
                # "R_local": R_local.copy(),
            }
            needle_infos.append(needle_info)
            # ===== 新增结束 =====

            print(f"[INFO] 针 {i} 拟合成功：截面 {len(centroids)} 个")

        except Exception as e:
            print(f"[ERROR] 针 {i} 带权样条拟合失败: {e}")

    if visualize:
        for g in geometries:
            if isinstance(g, o3d.geometry.TriangleMesh):
                g.compute_vertex_normals()

        print("[INFO] 可视化：原始点云 + 蓝色原始中心线 + 绿色/红色最终中心线")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Needle centerlines (blue: raw, green/ red: final)",
            width=1280,
            height=800,
        )

    return needle_infos


def fit_curve_robust(
    pcd_path: str,
    visualize: bool = True,
    show_raw_cloud: bool = True,
    method: str = "graph_slab",
    config: Optional[CenterlineConfig] = None,
):
    """
    统一入口：默认走新版 graph+slab；可通过 method="legacy" 回退到旧版实现。
    """
    method = str(method).lower().strip()
    if method in {"graph_slab", "graph", "slab"}:
        return fit_curve_graph_slab(pcd_path, visualize=visualize, show_raw_cloud=show_raw_cloud, config=config)
    if method in {"legacy", "old"}:
        return fit_curve_legacy(pcd_path, visualize=visualize, show_raw_cloud=show_raw_cloud)
    raise ValueError(f"Unknown method: {method!r}")


def smooth_centerline_component_wise(points, window_length=None, polyorder=3):
    """
    修改建议：
    1. 增大 window_frac 系数 (0.15 -> 0.35) 让线更"硬"
    2. 保持 polyorder=2 (适合针这种刚体，允许整体弯曲但不允许局部抖动)
    """
    if len(points) < 5:
        return points

    # [修改点 1]：增大窗口比例
    # 原来是 0.15，现在改为 0.30 或 0.35
    # 意味着每个点的坐标会参考整根针 30% 长度内的所有点来决定
    window_frac = 0.35

    if window_length is None:
        window_length = int(len(points) * window_frac)
        if window_length % 2 == 0: window_length += 1  # 必须是奇数
        # 设定一个较大的最小窗口，比如至少包含 9 个切片点
        window_length = max(9, window_length)

    # 防止窗口超过数组长度
    if window_length >= len(points):
        window_length = len(points) - 1 if len(points) % 2 == 0 else len(points)

    # [修改点 2 (可选)]：如果针非常直，可以强制 polyorder=1
    # 但建议保持 2，因为 2 足够硬且能表现整体弯曲
    current_polyorder = polyorder

    # 安全检查：窗口长度必须大于阶数
    if window_length <= current_polyorder:
        current_polyorder = window_length - 1

    x_smooth = savgol_filter(points[:, 0], window_length, current_polyorder)
    y_smooth = savgol_filter(points[:, 1], window_length, current_polyorder)
    z_smooth = savgol_filter(points[:, 2], window_length, current_polyorder)

    return np.stack([x_smooth, y_smooth, z_smooth], axis=1)

def extract_centerlines_from_pcd(pcd_path, method: str = "graph_slab", config: Optional[CenterlineConfig] = None):
    """仅提取中心线信息，不进行可视化。"""
    return fit_curve_robust(pcd_path, visualize=False, show_raw_cloud=False, method=method, config=config)


if __name__ == "__main__":

    # Optional CLI overrides for safer batch evaluation. Defaults preserve PyCharm "Run" behavior.
    _pcd_path = INPUT_PCD_PATH
    _eval_gt_dir = EVAL_GT_DIR
    _denoise_enable = bool(DEFAULT_DENOISE_ENABLE)
    _endpoint_centering_iters = int(DEFAULT_ENDPOINT_CENTERING_ITERS)
    _endpoint_centering_timing = None
    _endpoint_centering_min_coverage = None
    _post_endpoint_refine_iters = None
    _trunc_end_fix = None
    _endpoint_bbox_fix = None
    _endpoint_bbox_depth_mm = None
    _endpoint_bbox_min_points = None
    _endpoint_bbox_extend_quantile = None
    _endpoint_bbox_max_extend_mm = None
    _endpoint_bbox_axis_radius_factor = None
    _endpoint_bbox_use_obb = None
    _endpoint_bbox_size_inward_mm = None
    _endpoint_bbox_size_q_low = None
    _endpoint_bbox_size_q_high = None
    _endpoint_bbox_size_margin = None
    _endpoint_bbox_min_half_u_factor = None
    _endpoint_bbox_min_half_v_factor = None
    _endpoint_bbox_hook_fix = None
    _endpoint_bbox_hook_angle_deg = None
    _endpoint_bbox_hook_base_only = None
    _visualize = bool(DEFAULT_VISUALIZE)
    _debug_dump = bool(DEFAULT_DEBUG_DUMP)
    _debug_dump_first_n_segments = 0
    _debug_dump_every_iter = False
    _init_edge_cost_mode = None
    _init_dir_penalty_lambda = None
    _slab_single_side_action = None
    _slab_min_coverage = None
    _slab_coverage_bins = None
    _endpoint_regrow = None
    _endpoint_regrow_skip_postfixes = None
    _force_final_curve_stage13 = None
    _endpoint_regrow_trim_mm = None
    _endpoint_regrow_max_extend_mm = None
    _endpoint_regrow_half_thickness_factor = None
    _endpoint_regrow_center_mode = None
    _endpoint_regrow_center_trimmed_q_low = None
    _endpoint_regrow_center_trimmed_q_high = None
    _endpoint_regrow_center_geomedian_max_iter = None
    _endpoint_regrow_center_geomedian_tol_mm = None
    _endpoint_regrow_center_epcs_bins = None
    _endpoint_regrow_center_epcs_min_bin_points = None
    _endpoint_regrow_center_epcs_min_pairs = None
    _endpoint_regrow_rigid = None
    _endpoint_regrow_rigid_max_angle_deg = None
    _endpoint_regrow_rigid_max_angle_deg_lowcov = None
    _endpoint_regrow_rigid_near_mm = None
    _endpoint_regrow_rigid_lateral_max_mm = None
    _endpoint_regrow_rigid_lateral_min_thickness_factor = None
    _endpoint_regrow_rigid_lateral_relax_per_mm = None
    _endpoint_regrow_rigid_lateral_cap_mm = None
    _endpoint_regrow_rigid_radial_var_max = None
    _endpoint_regrow_rigid_soft_mode = None
    _endpoint_regrow_rigid_badshape_patience = None
    _endpoint_regrow_accept_gate = None
    _endpoint_regrow_accept_gate_rollback = None
    _endpoint_regrow_accept_endpoint_risk_gate = None
    _endpoint_regrow_accept_endpoint_window_mm = None
    _endpoint_regrow_guard = None
    _endpoint_regrow_guard_bbox_depth_mm = None
    _endpoint_regrow_guard_bbox_min_points = None
    _endpoint_regrow_guard_bbox_half_u_floor_factor = None
    _endpoint_regrow_guard_bbox_half_v_floor_factor = None
    _endpoint_regrow_guard_bbox_probe_depth_mm = None
    _endpoint_regrow_guard_bbox_probe_radius_mm = None
    _endpoint_regrow_guard_bbox_probe_min_points = None
    _endpoint_regrow_guard_use_polar_boundary = None
    _endpoint_regrow_guard_polar_bins = None
    _endpoint_regrow_guard_polar_min_bin_points = None
    _endpoint_regrow_guard_polar_min_valid_bins = None
    _endpoint_regrow_guard_polar_radius_quantile = None
    _endpoint_regrow_guard_polar_margin_mm = None
    _endpoint_regrow_guard_polar_fail_action = None
    _endpoint_regrow_deflect_trim = None
    _endpoint_regrow_deflect_angle_deg = None
    _endpoint_regrow_deflect_hard_angle_deg = None
    _endpoint_regrow_deflect_lateral_mm = None
    _endpoint_regrow_deflect_min_run_points = None
    _endpoint_regrow_deflect_ref_span_mm = None
    _endpoint_regrow_triclass = None
    _endpoint_regrow_triclass_variant = None
    _endpoint_regrow_triclass_extend_cov_min = None
    _endpoint_regrow_triclass_extend_support_n1_min = None
    _endpoint_regrow_triclass_extend_radial_var_max = None
    _endpoint_regrow_triclass_shorten_fail_count = None
    _endpoint_regrow_triclass_ambiguous_policy = None
    _endpoint_regrow_triclass_min_action_trim_mm = None
    _endpoint_regrow_triclass_backward_scan_steps = None
    _endpoint_regrow_triclass_backward_near_steps = None
    _endpoint_regrow_triclass_cap_reliability_min = None
    _endpoint_regrow_triclass_shorten_extra_mm = None
    _endpoint_regrow_triclass_corridor_enabled = None
    _endpoint_regrow_triclass_corridor_steps = None
    _endpoint_regrow_triclass_corridor_valid_steps_min = None
    _endpoint_regrow_triclass_corridor_support_sum_min = None
    _endpoint_regrow_triclass_corridor_support_peak_min = None
    _endpoint_regrow_triclass_corridor_cov_min = None
    _endpoint_regrow_triclass_corridor_rv_max = None
    _endpoint_regrow_triclass_hard_axis_bbox_grace_steps = None
    _endpoint_regrow_triclass_adaptive_enabled = None
    _endpoint_regrow_triclass_adaptive_min_samples = None
    try:
        ap = argparse.ArgumentParser(add_help=True)
        ap.add_argument(
            "--pcd",
            default=None,
            help="Input point cloud path. Overrides INPUT_PCD_PATH (default in-file constant).",
        )
        ap.add_argument(
            "--gt_dir",
            default=None,
            help="GT folder that contains centerline_gt.json. Overrides EVAL_GT_DIR (default in-file constant).",
        )
        ap.add_argument(
            "--denoise",
            action=argparse.BooleanOptionalAction,
            default=bool(DEFAULT_DENOISE_ENABLE),
            help=(
                "Enable/disable the denoise pipeline (ROR/SOR). "
                "Default is OFF for single-needle GT eval; enable for multi-needle clouds."
            ),
        )
        ap.add_argument(
            "--endpoint_centering_iters",
            type=int,
            default=int(DEFAULT_ENDPOINT_CENTERING_ITERS),
            help=(
                "Endpoint centering iterations (slab refinement). "
                "Lower is safer for single-needle GT eval; increase for multi-needle clouds if endpoints look unstable."
            ),
        )
        ap.add_argument(
            "--endpoint_centering_timing",
            type=str,
            default=None,
            help=(
                "Endpoint centering timing: early | late | off. "
                "Default uses in-file config default (recommended: late)."
            ),
        )
        ap.add_argument(
            "--endpoint_centering_min_coverage",
            type=float,
            default=None,
            help=(
                "Min angular coverage (0~1) to apply late endpoint centering updates. "
                "Default uses in-file config default."
            ),
        )
        ap.add_argument(
            "--post_endpoint_refine_iters",
            type=int,
            default=None,
            help=(
                "Post-refine iterations after late endpoint centering (blends endpoint update into the body). "
                "Default uses in-file config default."
            ),
        )
        ap.add_argument(
            "--endpoint_bbox_fix",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable endpoint bbox fix (length/hook stabilization). Default uses in-file config default.",
        )
        ap.add_argument(
            "--trunc_end_fix",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Enable/disable trunc-end-only fix (stage11). "
                "If enabled, the fix is still guarded by a safety rollback when it makes the endpoint region worse."
            ),
        )
        ap.add_argument(
            "--endpoint_bbox_depth_mm",
            type=float,
            default=None,
            help="Endpoint bbox outward depth in mm. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_min_points",
            type=int,
            default=None,
            help="Min points inside bbox outward region to consider the endpoint too short. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_extend_quantile",
            type=float,
            default=None,
            help="Quantile (0.9~1.0) for outward extension target. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_max_extend_mm",
            type=float,
            default=None,
            help="Max endpoint extension in mm. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_axis_radius_factor",
            type=float,
            default=None,
            help="Axis-cylinder radius factor * pin_width. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_use_obb",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable OBB sizing for endpoint bbox fix. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_size_inward_mm",
            type=float,
            default=None,
            help="OBB sizing: inward axial range in mm (from endpoint). Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_size_q_low",
            type=float,
            default=None,
            help="OBB sizing: low quantile for span estimation. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_size_q_high",
            type=float,
            default=None,
            help="OBB sizing: high quantile for span estimation. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_size_margin",
            type=float,
            default=None,
            help="OBB sizing: half-size margin factor (>1). Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_min_half_u_factor",
            type=float,
            default=None,
            help="OBB min half-u floor factor * pin_width.",
        )
        ap.add_argument(
            "--endpoint_bbox_min_half_v_factor",
            type=float,
            default=None,
            help="OBB min half-v floor factor * pin_thickness.",
        )
        ap.add_argument(
            "--endpoint_bbox_hook_fix",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable bbox-based hook fix (rigid+blend projection). Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_hook_angle_deg",
            type=float,
            default=None,
            help="Angle threshold (deg) between local tangent and stable axis to consider a hook artifact. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_bbox_hook_base_only",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="If true, only apply bbox hook fix on truncated(base) end. Default uses in-file config default.",
        )
        ap.add_argument(
            "--visualize",
            action=argparse.BooleanOptionalAction,
            default=bool(DEFAULT_VISUALIZE),
            help="Enable/disable Open3D visualization windows for this run.",
        )
        ap.add_argument(
            "--debug_dump",
            action=argparse.BooleanOptionalAction,
            default=bool(DEFAULT_DEBUG_DUMP),
            help=(
                "Dump intermediate centerline PLYs and a per-segment summary under debug_out/runs/<run_id>/centerline_debug/..."
            ),
        )
        ap.add_argument(
            "--debug_dump_first_n_segments",
            type=int,
            default=0,
            help="If >0, only dump the first N extracted segments (helps reduce artifacts size).",
        )
        ap.add_argument(
            "--debug_dump_every_iter",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="If true, dump intermediate curves for every refinement iteration (may generate many files).",
        )
        ap.add_argument(
            "--init_edge_cost_mode",
            type=str,
            default=None,
            help="Shortest-path initializer edge cost: euclid | dir_penalty (optional override).",
        )
        ap.add_argument(
            "--init_dir_penalty_lambda",
            type=float,
            default=None,
            help="Direction penalty lambda for init_edge_cost_mode=dir_penalty (optional override).",
        )
        ap.add_argument(
            "--slab_single_side_action",
            type=str,
            default=None,
            help="Single-sided slab action: none | skip | damp (optional override).",
        )
        ap.add_argument(
            "--slab_min_coverage",
            type=float,
            default=None,
            help="Min angular coverage (0~1) in slab cross-section to apply updates (optional override).",
        )
        ap.add_argument(
            "--slab_coverage_bins",
            type=int,
            default=None,
            help="Angular bins used for slab coverage estimation (optional override).",
        )
        ap.add_argument(
            "--endpoint_regrow",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Enable/disable stage13 body-guided endpoint regrowth (trim ends then regrow from the stable body). "
                "Default uses in-file config default."
            ),
        )
        ap.add_argument(
            "--endpoint_regrow_skip_postfixes",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="If true, skip stage11/stage12 when regrowth is applied. Default uses in-file config default.",
        )
        ap.add_argument(
            "--force_final_curve_stage13",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="If true, force final centerline to stage13_after_regrow and skip stage11/stage12 once stage13 exists.",
        )
        ap.add_argument(
            "--endpoint_regrow_trim_mm",
            type=float,
            default=None,
            help="Trim length (mm) on EACH end before regrowth. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_regrow_max_extend_mm",
            type=float,
            default=None,
            help="Max regrow extension (mm) per end. Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_regrow_half_thickness_factor",
            type=float,
            default=None,
            help="Scale factor for slab half thickness during regrowth (taller cylinder near ends). Default uses in-file config default.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_mode",
            type=str,
            default=None,
            help="Endpoint slab center estimator: midrange | trimmed_mean | geomedian | epcs_pair.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_trimmed_q_low",
            type=float,
            default=None,
            help="Lower quantile for trimmed_mean endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_trimmed_q_high",
            type=float,
            default=None,
            help="Upper quantile for trimmed_mean endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_geomedian_max_iter",
            type=int,
            default=None,
            help="Max iterations for geometric-median endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_geomedian_tol_mm",
            type=float,
            default=None,
            help="Convergence tolerance (mm) for geometric-median endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_epcs_bins",
            type=int,
            default=None,
            help="Angular bins for EPCS-lite paired-bin endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_epcs_min_bin_points",
            type=int,
            default=None,
            help="Minimum points per bin for EPCS-lite paired-bin endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_center_epcs_min_pairs",
            type=int,
            default=None,
            help="Minimum opposite-bin pairs required by EPCS-lite endpoint center estimator.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable rigid endpoint prior during regrowth (angle/lateral guards).",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_max_angle_deg",
            type=float,
            default=None,
            help="Max tangent deviation (deg) from endpoint reference under good evidence.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_max_angle_deg_lowcov",
            type=float,
            default=None,
            help="Stricter max tangent deviation (deg) under low evidence.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_near_mm",
            type=float,
            default=None,
            help="Near-end rigid zone length (mm) for lateral drift limit.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_lateral_max_mm",
            type=float,
            default=None,
            help="Base max lateral drift (mm) relative to endpoint reference axis.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_lateral_min_thickness_factor",
            type=float,
            default=None,
            help="Rigid lateral drift min floor factor * pin_thickness.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_lateral_relax_per_mm",
            type=float,
            default=None,
            help="Lateral limit relaxation per mm of regrow distance.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_lateral_cap_mm",
            type=float,
            default=None,
            help="Absolute cap (mm) for lateral drift during regrow.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_radial_var_max",
            type=float,
            default=None,
            help="Max allowed endpoint slab radial_var during regrow before stop.",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_soft_mode",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable soft rigid guards (keep axial extension; clamp/pred-only instead of hard stop).",
        )
        ap.add_argument(
            "--endpoint_regrow_rigid_badshape_patience",
            type=int,
            default=None,
            help="Stop after this many consecutive bad-shape soft events.",
        )
        ap.add_argument(
            "--endpoint_regrow_accept_gate",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable stage13 acceptance gate.",
        )
        ap.add_argument(
            "--endpoint_regrow_accept_gate_rollback",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="If true, rollback stage13 candidate when acceptance gate flags worse quality.",
        )
        ap.add_argument(
            "--endpoint_regrow_accept_endpoint_risk_gate",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable endpoint-window risk gate in stage13 acceptance rollback.",
        )
        ap.add_argument(
            "--endpoint_regrow_accept_endpoint_window_mm",
            type=float,
            default=None,
            help="Endpoint window size (mm) used by stage13 endpoint risk acceptance gate.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable step-level stage13 boundary guard.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_depth_mm",
            type=float,
            default=None,
            help="Forward support check depth (mm) for stage13 step guard.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_min_points",
            type=int,
            default=None,
            help="Min forward support points required by stage13 step guard.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_half_u_floor_factor",
            type=float,
            default=None,
            help="Guard bbox half-u floor factor * pin_width.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_half_v_floor_factor",
            type=float,
            default=None,
            help="Guard bbox half-v floor factor * pin_thickness.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_probe_depth_mm",
            type=float,
            default=None,
            help="Sparse probe depth along axis (mm) when one-step guard support is low.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_probe_radius_mm",
            type=float,
            default=None,
            help="Sparse probe lateral radius (mm) when one-step guard support is low.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_bbox_probe_min_points",
            type=int,
            default=None,
            help="Sparse probe minimum forward points required to keep extending.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_use_polar_boundary",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable local non-rectangular (polar) boundary guard in stage13.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_polar_bins",
            type=int,
            default=None,
            help="Polar angular bins for local boundary estimation in stage13 guard.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_polar_min_bin_points",
            type=int,
            default=None,
            help="Min points per polar bin to consider it valid.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_polar_min_valid_bins",
            type=int,
            default=None,
            help="Min valid polar bins to trust local non-rectangular boundary.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_polar_radius_quantile",
            type=float,
            default=None,
            help="Radius quantile for polar boundary envelope.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_polar_margin_mm",
            type=float,
            default=None,
            help="Extra radial margin (mm) for polar boundary guard.",
        )
        ap.add_argument(
            "--endpoint_regrow_guard_polar_fail_action",
            type=str,
            default=None,
            help="Action when crossing local polar boundary: stop | clamp.",
        )
        ap.add_argument(
            "--endpoint_regrow_deflect_trim",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable deflection-based endpoint-tail trim before stage13 regrow.",
        )
        ap.add_argument(
            "--endpoint_regrow_deflect_angle_deg",
            type=float,
            default=None,
            help="Angle threshold (deg) for endpoint deflection trim detection.",
        )
        ap.add_argument(
            "--endpoint_regrow_deflect_hard_angle_deg",
            type=float,
            default=None,
            help="Hard angle threshold (deg) for endpoint deflection trim detection.",
        )
        ap.add_argument(
            "--endpoint_regrow_deflect_lateral_mm",
            type=float,
            default=None,
            help="Lateral offset threshold (mm) for endpoint deflection trim detection.",
        )
        ap.add_argument(
            "--endpoint_regrow_deflect_min_run_points",
            type=int,
            default=None,
            help="Consecutive points required to confirm endpoint deflection.",
        )
        ap.add_argument(
            "--endpoint_regrow_deflect_ref_span_mm",
            type=float,
            default=None,
            help="Reference span (mm) used to estimate stable axis for deflection detection.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable conservative tri-class pre-gate (shorten/keep/extend) before Stage13 regrow.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_variant",
            type=str,
            default=None,
            help="Tri-class variant: v1 (conservative).",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_extend_cov_min",
            type=float,
            default=None,
            help="Tri-class extend gate: minimum one-step coverage.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_extend_support_n1_min",
            type=int,
            default=None,
            help="Tri-class extend gate: minimum one-step forward support points.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_extend_radial_var_max",
            type=float,
            default=None,
            help="Tri-class extend gate: maximum one-step radial_var.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_shorten_fail_count",
            type=int,
            default=None,
            help="Tri-class shorten trigger: endpoint risk fail_count threshold.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_ambiguous_policy",
            type=str,
            default=None,
            help="Tri-class ambiguous policy: strict_keep | probe_on_gray(v2 reserved).",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_min_action_trim_mm",
            type=float,
            default=None,
            help="Tri-class min trim target (mm) required before shorten/extend actions.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_backward_scan_steps",
            type=int,
            default=None,
            help="Tri-class backward evidence: inward scan steps.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_backward_near_steps",
            type=int,
            default=None,
            help="Tri-class backward evidence: near-hit threshold in inward steps.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_cap_reliability_min",
            type=float,
            default=None,
            help="Tri-class free-end keep veto: minimum cap reliability.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_shorten_extra_mm",
            type=float,
            default=None,
            help="Tri-class shorten action: extra trim length (mm) added on that side.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable multi-step corridor evidence for tri-class extend decision.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor_steps",
            type=int,
            default=None,
            help="Tri-class corridor lookahead steps.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor_valid_steps_min",
            type=int,
            default=None,
            help="Tri-class corridor gate: minimum valid corridor steps.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor_support_sum_min",
            type=int,
            default=None,
            help="Tri-class corridor gate: minimum summed forward support.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor_support_peak_min",
            type=int,
            default=None,
            help="Tri-class corridor gate: minimum peak forward support among steps.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor_cov_min",
            type=float,
            default=None,
            help="Tri-class corridor gate: minimum mean coverage.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_corridor_rv_max",
            type=float,
            default=None,
            help="Tri-class corridor gate: maximum rv max across corridor steps.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_hard_axis_bbox_grace_steps",
            type=int,
            default=None,
            help="Hard-axial extend mode: initial bbox-front grace steps before stop.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_adaptive",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable adaptive tri-class thresholds estimated from one-step + corridor evidence.",
        )
        ap.add_argument(
            "--endpoint_regrow_triclass_adaptive_min_samples",
            type=int,
            default=None,
            help="Minimum finite samples required to enable adaptive tri thresholds.",
        )
        args, _unknown = ap.parse_known_args()
        if args.pcd:
            _pcd_path = str(args.pcd)
        if args.gt_dir:
            _eval_gt_dir = str(args.gt_dir)
        _denoise_enable = bool(getattr(args, "denoise", _denoise_enable))
        _endpoint_centering_iters = int(getattr(args, "endpoint_centering_iters", _endpoint_centering_iters))
        _endpoint_centering_timing = getattr(args, "endpoint_centering_timing", _endpoint_centering_timing)
        _endpoint_centering_min_coverage = getattr(args, "endpoint_centering_min_coverage", _endpoint_centering_min_coverage)
        _post_endpoint_refine_iters = getattr(args, "post_endpoint_refine_iters", _post_endpoint_refine_iters)
        _endpoint_bbox_fix = getattr(args, "endpoint_bbox_fix", None)
        _trunc_end_fix = getattr(args, "trunc_end_fix", None)
        _endpoint_bbox_depth_mm = getattr(args, "endpoint_bbox_depth_mm", None)
        _endpoint_bbox_min_points = getattr(args, "endpoint_bbox_min_points", None)
        _endpoint_bbox_extend_quantile = getattr(args, "endpoint_bbox_extend_quantile", None)
        _endpoint_bbox_max_extend_mm = getattr(args, "endpoint_bbox_max_extend_mm", None)
        _endpoint_bbox_axis_radius_factor = getattr(args, "endpoint_bbox_axis_radius_factor", None)
        _endpoint_bbox_use_obb = getattr(args, "endpoint_bbox_use_obb", None)
        _endpoint_bbox_size_inward_mm = getattr(args, "endpoint_bbox_size_inward_mm", None)
        _endpoint_bbox_size_q_low = getattr(args, "endpoint_bbox_size_q_low", None)
        _endpoint_bbox_size_q_high = getattr(args, "endpoint_bbox_size_q_high", None)
        _endpoint_bbox_size_margin = getattr(args, "endpoint_bbox_size_margin", None)
        _endpoint_bbox_min_half_u_factor = getattr(args, "endpoint_bbox_min_half_u_factor", None)
        _endpoint_bbox_min_half_v_factor = getattr(args, "endpoint_bbox_min_half_v_factor", None)
        _endpoint_bbox_hook_fix = getattr(args, "endpoint_bbox_hook_fix", None)
        _endpoint_bbox_hook_angle_deg = getattr(args, "endpoint_bbox_hook_angle_deg", None)
        _endpoint_bbox_hook_base_only = getattr(args, "endpoint_bbox_hook_base_only", None)
        _visualize = bool(getattr(args, "visualize", _visualize))
        _debug_dump = bool(getattr(args, "debug_dump", _debug_dump))
        _debug_dump_first_n_segments = int(getattr(args, "debug_dump_first_n_segments", _debug_dump_first_n_segments))
        _debug_dump_every_iter = bool(getattr(args, "debug_dump_every_iter", _debug_dump_every_iter))
        _init_edge_cost_mode = getattr(args, "init_edge_cost_mode", _init_edge_cost_mode)
        _init_dir_penalty_lambda = getattr(args, "init_dir_penalty_lambda", _init_dir_penalty_lambda)
        _slab_single_side_action = getattr(args, "slab_single_side_action", _slab_single_side_action)
        _slab_min_coverage = getattr(args, "slab_min_coverage", _slab_min_coverage)
        _slab_coverage_bins = getattr(args, "slab_coverage_bins", _slab_coverage_bins)
        _endpoint_regrow = getattr(args, "endpoint_regrow", _endpoint_regrow)
        _endpoint_regrow_skip_postfixes = getattr(args, "endpoint_regrow_skip_postfixes", _endpoint_regrow_skip_postfixes)
        _force_final_curve_stage13 = getattr(args, "force_final_curve_stage13", _force_final_curve_stage13)
        _endpoint_regrow_trim_mm = getattr(args, "endpoint_regrow_trim_mm", _endpoint_regrow_trim_mm)
        _endpoint_regrow_max_extend_mm = getattr(args, "endpoint_regrow_max_extend_mm", _endpoint_regrow_max_extend_mm)
        _endpoint_regrow_half_thickness_factor = getattr(
            args, "endpoint_regrow_half_thickness_factor", _endpoint_regrow_half_thickness_factor
        )
        _endpoint_regrow_center_mode = getattr(args, "endpoint_regrow_center_mode", _endpoint_regrow_center_mode)
        _endpoint_regrow_center_trimmed_q_low = getattr(
            args,
            "endpoint_regrow_center_trimmed_q_low",
            _endpoint_regrow_center_trimmed_q_low,
        )
        _endpoint_regrow_center_trimmed_q_high = getattr(
            args,
            "endpoint_regrow_center_trimmed_q_high",
            _endpoint_regrow_center_trimmed_q_high,
        )
        _endpoint_regrow_center_geomedian_max_iter = getattr(
            args,
            "endpoint_regrow_center_geomedian_max_iter",
            _endpoint_regrow_center_geomedian_max_iter,
        )
        _endpoint_regrow_center_geomedian_tol_mm = getattr(
            args,
            "endpoint_regrow_center_geomedian_tol_mm",
            _endpoint_regrow_center_geomedian_tol_mm,
        )
        _endpoint_regrow_center_epcs_bins = getattr(
            args,
            "endpoint_regrow_center_epcs_bins",
            _endpoint_regrow_center_epcs_bins,
        )
        _endpoint_regrow_center_epcs_min_bin_points = getattr(
            args,
            "endpoint_regrow_center_epcs_min_bin_points",
            _endpoint_regrow_center_epcs_min_bin_points,
        )
        _endpoint_regrow_center_epcs_min_pairs = getattr(
            args,
            "endpoint_regrow_center_epcs_min_pairs",
            _endpoint_regrow_center_epcs_min_pairs,
        )
        _endpoint_regrow_rigid = getattr(args, "endpoint_regrow_rigid", _endpoint_regrow_rigid)
        _endpoint_regrow_rigid_max_angle_deg = getattr(
            args, "endpoint_regrow_rigid_max_angle_deg", _endpoint_regrow_rigid_max_angle_deg
        )
        _endpoint_regrow_rigid_max_angle_deg_lowcov = getattr(
            args, "endpoint_regrow_rigid_max_angle_deg_lowcov", _endpoint_regrow_rigid_max_angle_deg_lowcov
        )
        _endpoint_regrow_rigid_near_mm = getattr(args, "endpoint_regrow_rigid_near_mm", _endpoint_regrow_rigid_near_mm)
        _endpoint_regrow_rigid_lateral_max_mm = getattr(
            args, "endpoint_regrow_rigid_lateral_max_mm", _endpoint_regrow_rigid_lateral_max_mm
        )
        _endpoint_regrow_rigid_lateral_min_thickness_factor = getattr(
            args,
            "endpoint_regrow_rigid_lateral_min_thickness_factor",
            _endpoint_regrow_rigid_lateral_min_thickness_factor,
        )
        _endpoint_regrow_rigid_lateral_relax_per_mm = getattr(
            args, "endpoint_regrow_rigid_lateral_relax_per_mm", _endpoint_regrow_rigid_lateral_relax_per_mm
        )
        _endpoint_regrow_rigid_lateral_cap_mm = getattr(
            args, "endpoint_regrow_rigid_lateral_cap_mm", _endpoint_regrow_rigid_lateral_cap_mm
        )
        _endpoint_regrow_rigid_radial_var_max = getattr(
            args, "endpoint_regrow_rigid_radial_var_max", _endpoint_regrow_rigid_radial_var_max
        )
        _endpoint_regrow_rigid_soft_mode = getattr(args, "endpoint_regrow_rigid_soft_mode", _endpoint_regrow_rigid_soft_mode)
        _endpoint_regrow_rigid_badshape_patience = getattr(
            args, "endpoint_regrow_rigid_badshape_patience", _endpoint_regrow_rigid_badshape_patience
        )
        _endpoint_regrow_accept_gate = getattr(args, "endpoint_regrow_accept_gate", _endpoint_regrow_accept_gate)
        _endpoint_regrow_accept_gate_rollback = getattr(
            args, "endpoint_regrow_accept_gate_rollback", _endpoint_regrow_accept_gate_rollback
        )
        _endpoint_regrow_accept_endpoint_risk_gate = getattr(
            args, "endpoint_regrow_accept_endpoint_risk_gate", _endpoint_regrow_accept_endpoint_risk_gate
        )
        _endpoint_regrow_accept_endpoint_window_mm = getattr(
            args, "endpoint_regrow_accept_endpoint_window_mm", _endpoint_regrow_accept_endpoint_window_mm
        )
        _endpoint_regrow_guard = getattr(args, "endpoint_regrow_guard", _endpoint_regrow_guard)
        _endpoint_regrow_guard_bbox_depth_mm = getattr(
            args, "endpoint_regrow_guard_bbox_depth_mm", _endpoint_regrow_guard_bbox_depth_mm
        )
        _endpoint_regrow_guard_bbox_min_points = getattr(
            args, "endpoint_regrow_guard_bbox_min_points", _endpoint_regrow_guard_bbox_min_points
        )
        _endpoint_regrow_guard_bbox_half_u_floor_factor = getattr(
            args,
            "endpoint_regrow_guard_bbox_half_u_floor_factor",
            _endpoint_regrow_guard_bbox_half_u_floor_factor,
        )
        _endpoint_regrow_guard_bbox_half_v_floor_factor = getattr(
            args,
            "endpoint_regrow_guard_bbox_half_v_floor_factor",
            _endpoint_regrow_guard_bbox_half_v_floor_factor,
        )
        _endpoint_regrow_guard_bbox_probe_depth_mm = getattr(
            args,
            "endpoint_regrow_guard_bbox_probe_depth_mm",
            _endpoint_regrow_guard_bbox_probe_depth_mm,
        )
        _endpoint_regrow_guard_bbox_probe_radius_mm = getattr(
            args,
            "endpoint_regrow_guard_bbox_probe_radius_mm",
            _endpoint_regrow_guard_bbox_probe_radius_mm,
        )
        _endpoint_regrow_guard_bbox_probe_min_points = getattr(
            args,
            "endpoint_regrow_guard_bbox_probe_min_points",
            _endpoint_regrow_guard_bbox_probe_min_points,
        )
        _endpoint_regrow_guard_use_polar_boundary = getattr(
            args, "endpoint_regrow_guard_use_polar_boundary", _endpoint_regrow_guard_use_polar_boundary
        )
        _endpoint_regrow_guard_polar_bins = getattr(
            args, "endpoint_regrow_guard_polar_bins", _endpoint_regrow_guard_polar_bins
        )
        _endpoint_regrow_guard_polar_min_bin_points = getattr(
            args,
            "endpoint_regrow_guard_polar_min_bin_points",
            _endpoint_regrow_guard_polar_min_bin_points,
        )
        _endpoint_regrow_guard_polar_min_valid_bins = getattr(
            args,
            "endpoint_regrow_guard_polar_min_valid_bins",
            _endpoint_regrow_guard_polar_min_valid_bins,
        )
        _endpoint_regrow_guard_polar_radius_quantile = getattr(
            args,
            "endpoint_regrow_guard_polar_radius_quantile",
            _endpoint_regrow_guard_polar_radius_quantile,
        )
        _endpoint_regrow_guard_polar_margin_mm = getattr(
            args, "endpoint_regrow_guard_polar_margin_mm", _endpoint_regrow_guard_polar_margin_mm
        )
        _endpoint_regrow_guard_polar_fail_action = getattr(
            args, "endpoint_regrow_guard_polar_fail_action", _endpoint_regrow_guard_polar_fail_action
        )
        _endpoint_regrow_deflect_trim = getattr(args, "endpoint_regrow_deflect_trim", _endpoint_regrow_deflect_trim)
        _endpoint_regrow_deflect_angle_deg = getattr(
            args, "endpoint_regrow_deflect_angle_deg", _endpoint_regrow_deflect_angle_deg
        )
        _endpoint_regrow_deflect_hard_angle_deg = getattr(
            args, "endpoint_regrow_deflect_hard_angle_deg", _endpoint_regrow_deflect_hard_angle_deg
        )
        _endpoint_regrow_deflect_lateral_mm = getattr(
            args, "endpoint_regrow_deflect_lateral_mm", _endpoint_regrow_deflect_lateral_mm
        )
        _endpoint_regrow_deflect_min_run_points = getattr(
            args, "endpoint_regrow_deflect_min_run_points", _endpoint_regrow_deflect_min_run_points
        )
        _endpoint_regrow_deflect_ref_span_mm = getattr(
            args, "endpoint_regrow_deflect_ref_span_mm", _endpoint_regrow_deflect_ref_span_mm
        )
        _endpoint_regrow_triclass = getattr(args, "endpoint_regrow_triclass", _endpoint_regrow_triclass)
        _endpoint_regrow_triclass_variant = getattr(
            args, "endpoint_regrow_triclass_variant", _endpoint_regrow_triclass_variant
        )
        _endpoint_regrow_triclass_extend_cov_min = getattr(
            args,
            "endpoint_regrow_triclass_extend_cov_min",
            _endpoint_regrow_triclass_extend_cov_min,
        )
        _endpoint_regrow_triclass_extend_support_n1_min = getattr(
            args,
            "endpoint_regrow_triclass_extend_support_n1_min",
            _endpoint_regrow_triclass_extend_support_n1_min,
        )
        _endpoint_regrow_triclass_extend_radial_var_max = getattr(
            args,
            "endpoint_regrow_triclass_extend_radial_var_max",
            _endpoint_regrow_triclass_extend_radial_var_max,
        )
        _endpoint_regrow_triclass_shorten_fail_count = getattr(
            args,
            "endpoint_regrow_triclass_shorten_fail_count",
            _endpoint_regrow_triclass_shorten_fail_count,
        )
        _endpoint_regrow_triclass_ambiguous_policy = getattr(
            args,
            "endpoint_regrow_triclass_ambiguous_policy",
            _endpoint_regrow_triclass_ambiguous_policy,
        )
        _endpoint_regrow_triclass_min_action_trim_mm = getattr(
            args,
            "endpoint_regrow_triclass_min_action_trim_mm",
            _endpoint_regrow_triclass_min_action_trim_mm,
        )
        _endpoint_regrow_triclass_backward_scan_steps = getattr(
            args,
            "endpoint_regrow_triclass_backward_scan_steps",
            _endpoint_regrow_triclass_backward_scan_steps,
        )
        _endpoint_regrow_triclass_backward_near_steps = getattr(
            args,
            "endpoint_regrow_triclass_backward_near_steps",
            _endpoint_regrow_triclass_backward_near_steps,
        )
        _endpoint_regrow_triclass_cap_reliability_min = getattr(
            args,
            "endpoint_regrow_triclass_cap_reliability_min",
            _endpoint_regrow_triclass_cap_reliability_min,
        )
        _endpoint_regrow_triclass_shorten_extra_mm = getattr(
            args,
            "endpoint_regrow_triclass_shorten_extra_mm",
            _endpoint_regrow_triclass_shorten_extra_mm,
        )
        _endpoint_regrow_triclass_corridor_enabled = getattr(
            args,
            "endpoint_regrow_triclass_corridor",
            _endpoint_regrow_triclass_corridor_enabled,
        )
        _endpoint_regrow_triclass_corridor_steps = getattr(
            args,
            "endpoint_regrow_triclass_corridor_steps",
            _endpoint_regrow_triclass_corridor_steps,
        )
        _endpoint_regrow_triclass_corridor_valid_steps_min = getattr(
            args,
            "endpoint_regrow_triclass_corridor_valid_steps_min",
            _endpoint_regrow_triclass_corridor_valid_steps_min,
        )
        _endpoint_regrow_triclass_corridor_support_sum_min = getattr(
            args,
            "endpoint_regrow_triclass_corridor_support_sum_min",
            _endpoint_regrow_triclass_corridor_support_sum_min,
        )
        _endpoint_regrow_triclass_corridor_support_peak_min = getattr(
            args,
            "endpoint_regrow_triclass_corridor_support_peak_min",
            _endpoint_regrow_triclass_corridor_support_peak_min,
        )
        _endpoint_regrow_triclass_corridor_cov_min = getattr(
            args,
            "endpoint_regrow_triclass_corridor_cov_min",
            _endpoint_regrow_triclass_corridor_cov_min,
        )
        _endpoint_regrow_triclass_corridor_rv_max = getattr(
            args,
            "endpoint_regrow_triclass_corridor_rv_max",
            _endpoint_regrow_triclass_corridor_rv_max,
        )
        _endpoint_regrow_triclass_hard_axis_bbox_grace_steps = getattr(
            args,
            "endpoint_regrow_triclass_hard_axis_bbox_grace_steps",
            _endpoint_regrow_triclass_hard_axis_bbox_grace_steps,
        )
        _endpoint_regrow_triclass_adaptive_enabled = getattr(
            args,
            "endpoint_regrow_triclass_adaptive",
            _endpoint_regrow_triclass_adaptive_enabled,
        )
        _endpoint_regrow_triclass_adaptive_min_samples = getattr(
            args,
            "endpoint_regrow_triclass_adaptive_min_samples",
            _endpoint_regrow_triclass_adaptive_min_samples,
        )
    except SystemExit:
        raise
    except Exception:
        # Never block execution due to CLI parsing issues in PyCharm / embedded runs.
        pass

    _config_kwargs: Dict[str, object] = {
        "denoise_enable": bool(_denoise_enable),
        "endpoint_centering_iterations": int(_endpoint_centering_iters),
    }
    if _endpoint_centering_timing is not None and str(_endpoint_centering_timing).strip():
        _config_kwargs["endpoint_centering_timing"] = str(_endpoint_centering_timing).strip()
    if _endpoint_centering_min_coverage is not None:
        _config_kwargs["endpoint_centering_min_coverage"] = float(_endpoint_centering_min_coverage)
    if _post_endpoint_refine_iters is not None:
        _config_kwargs["post_endpoint_refine_iterations"] = int(_post_endpoint_refine_iters)
    if _init_edge_cost_mode is not None and str(_init_edge_cost_mode).strip():
        _config_kwargs["init_edge_cost_mode"] = str(_init_edge_cost_mode).strip()
    if _init_dir_penalty_lambda is not None:
        _config_kwargs["init_dir_penalty_lambda"] = float(_init_dir_penalty_lambda)
    if _slab_single_side_action is not None and str(_slab_single_side_action).strip():
        _config_kwargs["slab_single_side_action"] = str(_slab_single_side_action).strip()
    if _slab_min_coverage is not None:
        _config_kwargs["slab_min_coverage"] = float(_slab_min_coverage)
    if _slab_coverage_bins is not None:
        _config_kwargs["slab_coverage_bins"] = int(_slab_coverage_bins)
    if _endpoint_regrow is not None:
        _config_kwargs["endpoint_regrow_enabled"] = bool(_endpoint_regrow)
    if _endpoint_regrow_skip_postfixes is not None:
        _config_kwargs["endpoint_regrow_skip_postfixes"] = bool(_endpoint_regrow_skip_postfixes)
    if _force_final_curve_stage13 is not None:
        _config_kwargs["force_final_curve_stage13"] = bool(_force_final_curve_stage13)
    if _endpoint_regrow_trim_mm is not None:
        _config_kwargs["endpoint_regrow_trim_mm"] = float(_endpoint_regrow_trim_mm)
    if _endpoint_regrow_max_extend_mm is not None:
        _config_kwargs["endpoint_regrow_max_extend_mm"] = float(_endpoint_regrow_max_extend_mm)
    if _endpoint_regrow_half_thickness_factor is not None:
        _config_kwargs["endpoint_regrow_half_thickness_factor"] = float(_endpoint_regrow_half_thickness_factor)
    if _endpoint_regrow_center_mode is not None and str(_endpoint_regrow_center_mode).strip():
        _config_kwargs["endpoint_regrow_center_mode"] = str(_endpoint_regrow_center_mode).strip()
    if _endpoint_regrow_center_trimmed_q_low is not None:
        _config_kwargs["endpoint_regrow_center_trimmed_q_low"] = float(_endpoint_regrow_center_trimmed_q_low)
    if _endpoint_regrow_center_trimmed_q_high is not None:
        _config_kwargs["endpoint_regrow_center_trimmed_q_high"] = float(_endpoint_regrow_center_trimmed_q_high)
    if _endpoint_regrow_center_geomedian_max_iter is not None:
        _config_kwargs["endpoint_regrow_center_geomedian_max_iter"] = int(_endpoint_regrow_center_geomedian_max_iter)
    if _endpoint_regrow_center_geomedian_tol_mm is not None:
        _config_kwargs["endpoint_regrow_center_geomedian_tol_mm"] = float(_endpoint_regrow_center_geomedian_tol_mm)
    if _endpoint_regrow_center_epcs_bins is not None:
        _config_kwargs["endpoint_regrow_center_epcs_bins"] = int(_endpoint_regrow_center_epcs_bins)
    if _endpoint_regrow_center_epcs_min_bin_points is not None:
        _config_kwargs["endpoint_regrow_center_epcs_min_bin_points"] = int(_endpoint_regrow_center_epcs_min_bin_points)
    if _endpoint_regrow_center_epcs_min_pairs is not None:
        _config_kwargs["endpoint_regrow_center_epcs_min_pairs"] = int(_endpoint_regrow_center_epcs_min_pairs)
    if _endpoint_regrow_rigid is not None:
        _config_kwargs["endpoint_regrow_rigid_enabled"] = bool(_endpoint_regrow_rigid)
    if _endpoint_regrow_rigid_max_angle_deg is not None:
        _config_kwargs["endpoint_regrow_rigid_max_angle_deg"] = float(_endpoint_regrow_rigid_max_angle_deg)
    if _endpoint_regrow_rigid_max_angle_deg_lowcov is not None:
        _config_kwargs["endpoint_regrow_rigid_max_angle_deg_lowcov"] = float(_endpoint_regrow_rigid_max_angle_deg_lowcov)
    if _endpoint_regrow_rigid_near_mm is not None:
        _config_kwargs["endpoint_regrow_rigid_near_mm"] = float(_endpoint_regrow_rigid_near_mm)
    if _endpoint_regrow_rigid_lateral_max_mm is not None:
        _config_kwargs["endpoint_regrow_rigid_lateral_max_mm"] = float(_endpoint_regrow_rigid_lateral_max_mm)
    if _endpoint_regrow_rigid_lateral_min_thickness_factor is not None:
        _config_kwargs["endpoint_regrow_rigid_lateral_min_thickness_factor"] = float(
            _endpoint_regrow_rigid_lateral_min_thickness_factor
        )
    if _endpoint_regrow_rigid_lateral_relax_per_mm is not None:
        _config_kwargs["endpoint_regrow_rigid_lateral_relax_per_mm"] = float(_endpoint_regrow_rigid_lateral_relax_per_mm)
    if _endpoint_regrow_rigid_lateral_cap_mm is not None:
        _config_kwargs["endpoint_regrow_rigid_lateral_cap_mm"] = float(_endpoint_regrow_rigid_lateral_cap_mm)
    if _endpoint_regrow_rigid_radial_var_max is not None:
        _config_kwargs["endpoint_regrow_rigid_radial_var_max"] = float(_endpoint_regrow_rigid_radial_var_max)
    if _endpoint_regrow_rigid_soft_mode is not None:
        _config_kwargs["endpoint_regrow_rigid_soft_mode"] = bool(_endpoint_regrow_rigid_soft_mode)
    if _endpoint_regrow_rigid_badshape_patience is not None:
        _config_kwargs["endpoint_regrow_rigid_badshape_patience"] = int(_endpoint_regrow_rigid_badshape_patience)
    if _endpoint_regrow_accept_gate is not None:
        _config_kwargs["endpoint_regrow_accept_gate_enabled"] = bool(_endpoint_regrow_accept_gate)
    if _endpoint_regrow_accept_gate_rollback is not None:
        _config_kwargs["endpoint_regrow_accept_gate_rollback"] = bool(_endpoint_regrow_accept_gate_rollback)
    if _endpoint_regrow_accept_endpoint_risk_gate is not None:
        _config_kwargs["endpoint_regrow_accept_endpoint_risk_gate"] = bool(_endpoint_regrow_accept_endpoint_risk_gate)
    if _endpoint_regrow_accept_endpoint_window_mm is not None:
        _config_kwargs["endpoint_regrow_accept_endpoint_window_mm"] = float(_endpoint_regrow_accept_endpoint_window_mm)
    if _endpoint_regrow_guard is not None:
        _config_kwargs["endpoint_regrow_guard_enabled"] = bool(_endpoint_regrow_guard)
    if _endpoint_regrow_guard_bbox_depth_mm is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_depth_mm"] = float(_endpoint_regrow_guard_bbox_depth_mm)
    if _endpoint_regrow_guard_bbox_min_points is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_min_points"] = int(_endpoint_regrow_guard_bbox_min_points)
    if _endpoint_regrow_guard_bbox_half_u_floor_factor is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_half_u_floor_factor"] = float(
            _endpoint_regrow_guard_bbox_half_u_floor_factor
        )
    if _endpoint_regrow_guard_bbox_half_v_floor_factor is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_half_v_floor_factor"] = float(
            _endpoint_regrow_guard_bbox_half_v_floor_factor
        )
    if _endpoint_regrow_guard_bbox_probe_depth_mm is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_probe_depth_mm"] = float(
            _endpoint_regrow_guard_bbox_probe_depth_mm
        )
    if _endpoint_regrow_guard_bbox_probe_radius_mm is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_probe_radius_mm"] = float(
            _endpoint_regrow_guard_bbox_probe_radius_mm
        )
    if _endpoint_regrow_guard_bbox_probe_min_points is not None:
        _config_kwargs["endpoint_regrow_guard_bbox_probe_min_points"] = int(
            _endpoint_regrow_guard_bbox_probe_min_points
        )
    if _endpoint_regrow_guard_use_polar_boundary is not None:
        _config_kwargs["endpoint_regrow_guard_use_polar_boundary"] = bool(_endpoint_regrow_guard_use_polar_boundary)
    if _endpoint_regrow_guard_polar_bins is not None:
        _config_kwargs["endpoint_regrow_guard_polar_bins"] = int(_endpoint_regrow_guard_polar_bins)
    if _endpoint_regrow_guard_polar_min_bin_points is not None:
        _config_kwargs["endpoint_regrow_guard_polar_min_bin_points"] = int(_endpoint_regrow_guard_polar_min_bin_points)
    if _endpoint_regrow_guard_polar_min_valid_bins is not None:
        _config_kwargs["endpoint_regrow_guard_polar_min_valid_bins"] = int(_endpoint_regrow_guard_polar_min_valid_bins)
    if _endpoint_regrow_guard_polar_radius_quantile is not None:
        _config_kwargs["endpoint_regrow_guard_polar_radius_quantile"] = float(_endpoint_regrow_guard_polar_radius_quantile)
    if _endpoint_regrow_guard_polar_margin_mm is not None:
        _config_kwargs["endpoint_regrow_guard_polar_margin_mm"] = float(_endpoint_regrow_guard_polar_margin_mm)
    if _endpoint_regrow_guard_polar_fail_action is not None and str(_endpoint_regrow_guard_polar_fail_action).strip():
        _config_kwargs["endpoint_regrow_guard_polar_fail_action"] = str(_endpoint_regrow_guard_polar_fail_action).strip()
    if _endpoint_regrow_deflect_trim is not None:
        _config_kwargs["endpoint_regrow_deflect_trim_enabled"] = bool(_endpoint_regrow_deflect_trim)
    if _endpoint_regrow_deflect_angle_deg is not None:
        _config_kwargs["endpoint_regrow_deflect_angle_deg"] = float(_endpoint_regrow_deflect_angle_deg)
    if _endpoint_regrow_deflect_hard_angle_deg is not None:
        _config_kwargs["endpoint_regrow_deflect_hard_angle_deg"] = float(_endpoint_regrow_deflect_hard_angle_deg)
    if _endpoint_regrow_deflect_lateral_mm is not None:
        _config_kwargs["endpoint_regrow_deflect_lateral_mm"] = float(_endpoint_regrow_deflect_lateral_mm)
    if _endpoint_regrow_deflect_min_run_points is not None:
        _config_kwargs["endpoint_regrow_deflect_min_run_points"] = int(_endpoint_regrow_deflect_min_run_points)
    if _endpoint_regrow_deflect_ref_span_mm is not None:
        _config_kwargs["endpoint_regrow_deflect_ref_span_mm"] = float(_endpoint_regrow_deflect_ref_span_mm)
    if _endpoint_regrow_triclass is not None:
        _config_kwargs["endpoint_regrow_triclass_enabled"] = bool(_endpoint_regrow_triclass)
    if _endpoint_regrow_triclass_variant is not None and str(_endpoint_regrow_triclass_variant).strip():
        _config_kwargs["endpoint_regrow_triclass_variant"] = str(_endpoint_regrow_triclass_variant).strip()
    if _endpoint_regrow_triclass_extend_cov_min is not None:
        _config_kwargs["endpoint_regrow_triclass_extend_cov_min"] = float(_endpoint_regrow_triclass_extend_cov_min)
    if _endpoint_regrow_triclass_extend_support_n1_min is not None:
        _config_kwargs["endpoint_regrow_triclass_extend_support_n1_min"] = int(
            _endpoint_regrow_triclass_extend_support_n1_min
        )
    if _endpoint_regrow_triclass_extend_radial_var_max is not None:
        _config_kwargs["endpoint_regrow_triclass_extend_radial_var_max"] = float(
            _endpoint_regrow_triclass_extend_radial_var_max
        )
    if _endpoint_regrow_triclass_shorten_fail_count is not None:
        _config_kwargs["endpoint_regrow_triclass_shorten_fail_count"] = int(
            _endpoint_regrow_triclass_shorten_fail_count
        )
    if _endpoint_regrow_triclass_ambiguous_policy is not None and str(_endpoint_regrow_triclass_ambiguous_policy).strip():
        _config_kwargs["endpoint_regrow_triclass_ambiguous_policy"] = str(
            _endpoint_regrow_triclass_ambiguous_policy
        ).strip()
    if _endpoint_regrow_triclass_min_action_trim_mm is not None:
        _config_kwargs["endpoint_regrow_triclass_min_action_trim_mm"] = float(
            _endpoint_regrow_triclass_min_action_trim_mm
        )
    if _endpoint_regrow_triclass_backward_scan_steps is not None:
        _config_kwargs["endpoint_regrow_triclass_backward_scan_steps"] = int(
            _endpoint_regrow_triclass_backward_scan_steps
        )
    if _endpoint_regrow_triclass_backward_near_steps is not None:
        _config_kwargs["endpoint_regrow_triclass_backward_near_steps"] = int(
            _endpoint_regrow_triclass_backward_near_steps
        )
    if _endpoint_regrow_triclass_cap_reliability_min is not None:
        _config_kwargs["endpoint_regrow_triclass_cap_reliability_min"] = float(
            _endpoint_regrow_triclass_cap_reliability_min
        )
    if _endpoint_regrow_triclass_shorten_extra_mm is not None:
        _config_kwargs["endpoint_regrow_triclass_shorten_extra_mm"] = float(_endpoint_regrow_triclass_shorten_extra_mm)
    if _endpoint_regrow_triclass_corridor_enabled is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_enabled"] = bool(_endpoint_regrow_triclass_corridor_enabled)
    if _endpoint_regrow_triclass_corridor_steps is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_steps"] = int(_endpoint_regrow_triclass_corridor_steps)
    if _endpoint_regrow_triclass_corridor_valid_steps_min is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_valid_steps_min"] = int(
            _endpoint_regrow_triclass_corridor_valid_steps_min
        )
    if _endpoint_regrow_triclass_corridor_support_sum_min is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_support_sum_min"] = int(
            _endpoint_regrow_triclass_corridor_support_sum_min
        )
    if _endpoint_regrow_triclass_corridor_support_peak_min is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_support_peak_min"] = int(
            _endpoint_regrow_triclass_corridor_support_peak_min
        )
    if _endpoint_regrow_triclass_corridor_cov_min is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_cov_min"] = float(
            _endpoint_regrow_triclass_corridor_cov_min
        )
    if _endpoint_regrow_triclass_corridor_rv_max is not None:
        _config_kwargs["endpoint_regrow_triclass_corridor_rv_max"] = float(_endpoint_regrow_triclass_corridor_rv_max)
    if _endpoint_regrow_triclass_hard_axis_bbox_grace_steps is not None:
        _config_kwargs["endpoint_regrow_triclass_hard_axis_bbox_grace_steps"] = int(
            _endpoint_regrow_triclass_hard_axis_bbox_grace_steps
        )
    if _endpoint_regrow_triclass_adaptive_enabled is not None:
        _config_kwargs["endpoint_regrow_triclass_adaptive_enabled"] = bool(
            _endpoint_regrow_triclass_adaptive_enabled
        )
    if _endpoint_regrow_triclass_adaptive_min_samples is not None:
        _config_kwargs["endpoint_regrow_triclass_adaptive_min_samples"] = int(
            _endpoint_regrow_triclass_adaptive_min_samples
        )

    # When debug_dump is enabled, force dumps to be stored under debug_out/ (project rule).
    _centerline_debug_root: Optional[Path] = None
    if bool(_debug_dump):
        try:
            stem = Path(_pcd_path).stem if _pcd_path else "pcd"
            if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None) is not None:
                _centerline_debug_root = Path(_DAIMA_CTX.run_dir) / "centerline_debug" / stem
            else:
                # Fallback (should be rare): still keep dumps under <repo_root>/debug_out/runs/<run_id>/...
                repo_root = _gt_eval_find_repo_root(Path(__file__))
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                _centerline_debug_root = repo_root / "debug_out" / "runs" / run_id / "centerline_debug" / stem
            _centerline_debug_root.mkdir(parents=True, exist_ok=True)
            _config_kwargs["debug_dump_dir"] = str(_centerline_debug_root)
            _config_kwargs["debug_dump_first_n_segments"] = int(_debug_dump_first_n_segments)
            _config_kwargs["debug_dump_every_iter"] = bool(_debug_dump_every_iter)
        except Exception:
            _centerline_debug_root = None
    if _endpoint_bbox_fix is not None:
        _config_kwargs["endpoint_bbox_fix_enabled"] = bool(_endpoint_bbox_fix)
    if _trunc_end_fix is not None:
        _config_kwargs["trunc_end_fix_enabled"] = bool(_trunc_end_fix)
    if _endpoint_bbox_depth_mm is not None:
        _config_kwargs["endpoint_bbox_depth_mm"] = float(_endpoint_bbox_depth_mm)
    if _endpoint_bbox_min_points is not None:
        _config_kwargs["endpoint_bbox_min_points"] = int(_endpoint_bbox_min_points)
    if _endpoint_bbox_extend_quantile is not None:
        _config_kwargs["endpoint_bbox_extend_quantile"] = float(_endpoint_bbox_extend_quantile)
    if _endpoint_bbox_max_extend_mm is not None:
        _config_kwargs["endpoint_bbox_max_extend_mm"] = float(_endpoint_bbox_max_extend_mm)
    if _endpoint_bbox_axis_radius_factor is not None:
        _config_kwargs["endpoint_bbox_axis_radius_factor"] = float(_endpoint_bbox_axis_radius_factor)
    if _endpoint_bbox_use_obb is not None:
        _config_kwargs["endpoint_bbox_use_obb"] = bool(_endpoint_bbox_use_obb)
    if _endpoint_bbox_size_inward_mm is not None:
        _config_kwargs["endpoint_bbox_size_inward_mm"] = float(_endpoint_bbox_size_inward_mm)
    if _endpoint_bbox_size_q_low is not None:
        _config_kwargs["endpoint_bbox_size_q_low"] = float(_endpoint_bbox_size_q_low)
    if _endpoint_bbox_size_q_high is not None:
        _config_kwargs["endpoint_bbox_size_q_high"] = float(_endpoint_bbox_size_q_high)
    if _endpoint_bbox_size_margin is not None:
        _config_kwargs["endpoint_bbox_size_margin"] = float(_endpoint_bbox_size_margin)
    if _endpoint_bbox_min_half_u_factor is not None:
        _config_kwargs["endpoint_bbox_min_half_u_factor"] = float(_endpoint_bbox_min_half_u_factor)
    if _endpoint_bbox_min_half_v_factor is not None:
        _config_kwargs["endpoint_bbox_min_half_v_factor"] = float(_endpoint_bbox_min_half_v_factor)
    if _endpoint_bbox_hook_fix is not None:
        _config_kwargs["endpoint_bbox_hook_fix_enabled"] = bool(_endpoint_bbox_hook_fix)
    if _endpoint_bbox_hook_angle_deg is not None:
        _config_kwargs["endpoint_bbox_hook_angle_deg"] = float(_endpoint_bbox_hook_angle_deg)
    if _endpoint_bbox_hook_base_only is not None:
        _config_kwargs["endpoint_bbox_hook_base_only"] = bool(_endpoint_bbox_hook_base_only)

    _config = CenterlineConfig(**_config_kwargs)

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "pcd_path": _pcd_path,
                "eval_gt_dir": _eval_gt_dir,
            }
        )
        if _centerline_debug_root is not None:
            _DAIMA_CTX.snapshot_paths(outputs={"centerline_debug_root": str(_centerline_debug_root)})
        _DAIMA_CTX.snapshot_config(
            config={
                "UNIT_SCALE": UNIT_SCALE,
                "EVAL_OUT_DIR": EVAL_OUT_DIR,
                "EVAL_ONLY": EVAL_ONLY,
                "EVAL_STEP_MM": EVAL_STEP_MM,
                "EVAL_TANGENT_SAMPLES": EVAL_TANGENT_SAMPLES,
                "EVAL_TOPK": EVAL_TOPK,
                "DENOISE_ENABLE": bool(_denoise_enable),
                "DENOISE_STAGE": str(getattr(_config, "denoise_stage", "")),
                "VISUALIZE": bool(_visualize),
                "DEBUG_DUMP": bool(_debug_dump),
                "DEBUG_DUMP_FIRST_N_SEGMENTS": int(_debug_dump_first_n_segments),
                "DEBUG_DUMP_EVERY_ITER": bool(_debug_dump_every_iter),
                "ENDPOINT_CENTERING_ITERS": int(_endpoint_centering_iters),
                "ENDPOINT_CENTERING_TIMING": str(getattr(_config, "endpoint_centering_timing", "")),
                "ENDPOINT_CENTERING_MIN_COVERAGE": float(getattr(_config, "endpoint_centering_min_coverage", 0.0)),
                "POST_ENDPOINT_REFINE_ITERS": int(getattr(_config, "post_endpoint_refine_iterations", 0)),
                "INIT_EDGE_COST_MODE": str(getattr(_config, "init_edge_cost_mode", "")),
                "INIT_DIR_PENALTY_LAMBDA": float(getattr(_config, "init_dir_penalty_lambda", 0.0)),
                "SLAB_SINGLE_SIDE_ACTION": str(getattr(_config, "slab_single_side_action", "")),
                "SLAB_MIN_COVERAGE": float(getattr(_config, "slab_min_coverage", 0.0)),
                "SLAB_COVERAGE_BINS": int(getattr(_config, "slab_coverage_bins", 0)),
                "SLAB_SINGLE_SIDE_DAMP": float(getattr(_config, "slab_single_side_damp", 0.0)),
                "TRUNC_END_FIX_ENABLED": bool(getattr(_config, "trunc_end_fix_enabled", False)),
                "TRUNC_END_PULLBACK_FOLD_GUARD_ENABLED": bool(
                    getattr(_config, "trunc_end_pullback_fold_guard_enabled", True)
                ),
                "TRUNC_END_PULLBACK_FOLD_HARD_ANGLE_DEG": float(
                    getattr(_config, "trunc_end_pullback_fold_hard_angle_deg", 0.0)
                ),
                "TRUNC_END_PULLBACK_FOLD_ANGLE_JUMP_DEG": float(
                    getattr(_config, "trunc_end_pullback_fold_angle_jump_deg", 0.0)
                ),
                "TRUNC_END_PULLBACK_FOLD_SHORT_EDGE_RATIO": float(
                    getattr(_config, "trunc_end_pullback_fold_short_edge_ratio", 0.0)
                ),
                "TRUNC_END_PULLBACK_FOLD_SHORT_EDGE_ANGLE_DEG": float(
                    getattr(_config, "trunc_end_pullback_fold_short_edge_angle_deg", 0.0)
                ),
                "TRUNC_END_PULLBACK_LOWCOV_FORCE_SKIP_ENABLED": bool(
                    getattr(_config, "trunc_end_pullback_lowcov_force_skip_enabled", True)
                ),
                "TRUNC_END_PULLBACK_LOWCOV_FORCE_SKIP_MM": float(
                    getattr(_config, "trunc_end_pullback_lowcov_force_skip_mm", 0.0)
                ),
                "ENDPOINT_BBOX_FIX_ENABLED": bool(getattr(_config, "endpoint_bbox_fix_enabled", False)),
                "ENDPOINT_REGROW_ENABLED": bool(getattr(_config, "endpoint_regrow_enabled", False)),
                "ENDPOINT_REGROW_SKIP_POSTFIXES": bool(getattr(_config, "endpoint_regrow_skip_postfixes", False)),
                "FORCE_FINAL_CURVE_STAGE13": bool(getattr(_config, "force_final_curve_stage13", False)),
                "ENDPOINT_REGROW_TRIM_MM": float(getattr(_config, "endpoint_regrow_trim_mm", 0.0)),
                "ENDPOINT_REGROW_MAX_EXTEND_MM": float(getattr(_config, "endpoint_regrow_max_extend_mm", 0.0)),
                "ENDPOINT_REGROW_HALF_THICKNESS_FACTOR": float(getattr(_config, "endpoint_regrow_half_thickness_factor", 1.0)),
                "ENDPOINT_REGROW_MIN_STOP_POINTS": int(getattr(_config, "endpoint_regrow_min_stop_points", 0)),
                "ENDPOINT_REGROW_MIN_UPDATE_POINTS": int(getattr(_config, "endpoint_regrow_min_update_points", 0)),
                "ENDPOINT_REGROW_MIN_COVERAGE": float(getattr(_config, "endpoint_regrow_min_coverage", 0.0)),
                "ENDPOINT_REGROW_LOW_COVER_ACTION": str(getattr(_config, "endpoint_regrow_low_coverage_action", "")),
                "ENDPOINT_REGROW_MAX_UPDATE_SCALE": float(getattr(_config, "endpoint_regrow_max_update_scale", 1.0)),
                "ENDPOINT_REGROW_CENTER_MODE": str(getattr(_config, "endpoint_regrow_center_mode", "")),
                "ENDPOINT_REGROW_CENTER_TRIMMED_Q_LOW": float(
                    getattr(_config, "endpoint_regrow_center_trimmed_q_low", 0.0)
                ),
                "ENDPOINT_REGROW_CENTER_TRIMMED_Q_HIGH": float(
                    getattr(_config, "endpoint_regrow_center_trimmed_q_high", 0.0)
                ),
                "ENDPOINT_REGROW_CENTER_GEOMEDIAN_MAX_ITER": int(
                    getattr(_config, "endpoint_regrow_center_geomedian_max_iter", 0)
                ),
                "ENDPOINT_REGROW_CENTER_GEOMEDIAN_TOL_MM": float(
                    getattr(_config, "endpoint_regrow_center_geomedian_tol_mm", 0.0)
                ),
                "ENDPOINT_REGROW_CENTER_EPCS_BINS": int(
                    getattr(_config, "endpoint_regrow_center_epcs_bins", 0)
                ),
                "ENDPOINT_REGROW_CENTER_EPCS_MIN_BIN_POINTS": int(
                    getattr(_config, "endpoint_regrow_center_epcs_min_bin_points", 0)
                ),
                "ENDPOINT_REGROW_CENTER_EPCS_MIN_PAIRS": int(
                    getattr(_config, "endpoint_regrow_center_epcs_min_pairs", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_triclass_enabled", False)
                ),
                "ENDPOINT_REGROW_TRICLASS_VARIANT": str(
                    getattr(_config, "endpoint_regrow_triclass_variant", "")
                ),
                "ENDPOINT_REGROW_TRICLASS_EXTEND_COV_MIN": float(
                    getattr(_config, "endpoint_regrow_triclass_extend_cov_min", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_EXTEND_SUPPORT_N1_MIN": int(
                    getattr(_config, "endpoint_regrow_triclass_extend_support_n1_min", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_EXTEND_RADIAL_VAR_MAX": float(
                    getattr(_config, "endpoint_regrow_triclass_extend_radial_var_max", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_SHORTEN_FAIL_COUNT": int(
                    getattr(_config, "endpoint_regrow_triclass_shorten_fail_count", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_AMBIGUOUS_POLICY": str(
                    getattr(_config, "endpoint_regrow_triclass_ambiguous_policy", "")
                ),
                "ENDPOINT_REGROW_TRICLASS_MIN_ACTION_TRIM_MM": float(
                    getattr(_config, "endpoint_regrow_triclass_min_action_trim_mm", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_BACKWARD_SCAN_STEPS": int(
                    getattr(_config, "endpoint_regrow_triclass_backward_scan_steps", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_BACKWARD_NEAR_STEPS": int(
                    getattr(_config, "endpoint_regrow_triclass_backward_near_steps", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CAP_RELIABILITY_MIN": float(
                    getattr(_config, "endpoint_regrow_triclass_cap_reliability_min", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_SHORTEN_EXTRA_MM": float(
                    getattr(_config, "endpoint_regrow_triclass_shorten_extra_mm", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_triclass_corridor_enabled", True)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_STEPS": int(
                    getattr(_config, "endpoint_regrow_triclass_corridor_steps", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_VALID_STEPS_MIN": int(
                    getattr(_config, "endpoint_regrow_triclass_corridor_valid_steps_min", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_SUPPORT_SUM_MIN": int(
                    getattr(_config, "endpoint_regrow_triclass_corridor_support_sum_min", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_SUPPORT_PEAK_MIN": int(
                    getattr(_config, "endpoint_regrow_triclass_corridor_support_peak_min", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_COV_MIN": float(
                    getattr(_config, "endpoint_regrow_triclass_corridor_cov_min", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_CORRIDOR_RV_MAX": float(
                    getattr(_config, "endpoint_regrow_triclass_corridor_rv_max", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_HARD_AXIS_BBOX_GRACE_STEPS": int(
                    getattr(_config, "endpoint_regrow_triclass_hard_axis_bbox_grace_steps", 0)
                ),
                "ENDPOINT_REGROW_TRICLASS_ADAPTIVE_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_triclass_adaptive_enabled", True)
                ),
                "ENDPOINT_REGROW_TRICLASS_ADAPTIVE_MIN_SAMPLES": int(
                    getattr(_config, "endpoint_regrow_triclass_adaptive_min_samples", 0)
                ),
                "ENDPOINT_REGROW_ACCEPT_GATE_ENABLED": bool(getattr(_config, "endpoint_regrow_accept_gate_enabled", True)),
                "ENDPOINT_REGROW_ACCEPT_GATE_ROLLBACK": bool(
                    getattr(_config, "endpoint_regrow_accept_gate_rollback", True)
                ),
                "ENDPOINT_REGROW_ACCEPT_LFF_TOL": float(getattr(_config, "endpoint_regrow_accept_lff_tol", 0.0)),
                "ENDPOINT_REGROW_ACCEPT_FAR_TOL": float(getattr(_config, "endpoint_regrow_accept_far_tol", 0.0)),
                "ENDPOINT_REGROW_ACCEPT_P99_TOL_MM": float(getattr(_config, "endpoint_regrow_accept_p99_tol_mm", 0.0)),
                "ENDPOINT_REGROW_ACCEPT_ENDPOINT_RISK_GATE": bool(
                    getattr(_config, "endpoint_regrow_accept_endpoint_risk_gate", True)
                ),
                "ENDPOINT_REGROW_ACCEPT_ENDPOINT_WINDOW_MM": float(
                    getattr(_config, "endpoint_regrow_accept_endpoint_window_mm", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_ENABLED": bool(getattr(_config, "endpoint_regrow_rigid_enabled", True)),
                "ENDPOINT_REGROW_RIGID_ANCHOR_MM": float(getattr(_config, "endpoint_regrow_rigid_anchor_mm", 0.0)),
                "ENDPOINT_REGROW_RIGID_MAX_ANGLE_DEG": float(getattr(_config, "endpoint_regrow_rigid_max_angle_deg", 0.0)),
                "ENDPOINT_REGROW_RIGID_MAX_ANGLE_DEG_LOWCOV": float(
                    getattr(_config, "endpoint_regrow_rigid_max_angle_deg_lowcov", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_NEAR_MM": float(getattr(_config, "endpoint_regrow_rigid_near_mm", 0.0)),
                "ENDPOINT_REGROW_RIGID_LATERAL_MAX_MM": float(
                    getattr(_config, "endpoint_regrow_rigid_lateral_max_mm", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_LATERAL_MIN_THICKNESS_FACTOR": float(
                    getattr(_config, "endpoint_regrow_rigid_lateral_min_thickness_factor", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_LATERAL_RELAX_PER_MM": float(
                    getattr(_config, "endpoint_regrow_rigid_lateral_relax_per_mm", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_LATERAL_CAP_MM": float(
                    getattr(_config, "endpoint_regrow_rigid_lateral_cap_mm", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_RADIAL_VAR_MAX": float(
                    getattr(_config, "endpoint_regrow_rigid_radial_var_max", 0.0)
                ),
                "ENDPOINT_REGROW_RIGID_SOFT_MODE": bool(getattr(_config, "endpoint_regrow_rigid_soft_mode", True)),
                "ENDPOINT_REGROW_RIGID_BADSHAPE_PATIENCE": int(
                    getattr(_config, "endpoint_regrow_rigid_badshape_patience", 0)
                ),
                "ENDPOINT_REGROW_DEFLECT_TRIM_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_deflect_trim_enabled", True)
                ),
                "ENDPOINT_REGROW_DEFLECT_ANGLE_DEG": float(
                    getattr(_config, "endpoint_regrow_deflect_angle_deg", 0.0)
                ),
                "ENDPOINT_REGROW_DEFLECT_HARD_ANGLE_DEG": float(
                    getattr(_config, "endpoint_regrow_deflect_hard_angle_deg", 0.0)
                ),
                "ENDPOINT_REGROW_DEFLECT_LATERAL_MM": float(
                    getattr(_config, "endpoint_regrow_deflect_lateral_mm", 0.0)
                ),
                "ENDPOINT_REGROW_DEFLECT_MIN_RUN_POINTS": int(
                    getattr(_config, "endpoint_regrow_deflect_min_run_points", 0)
                ),
                "ENDPOINT_REGROW_DEFLECT_REF_SPAN_MM": float(
                    getattr(_config, "endpoint_regrow_deflect_ref_span_mm", 0.0)
                ),
                "ENDPOINT_REGROW_DEFLECT_TAIL_GRACE_POINTS": int(
                    getattr(_config, "endpoint_regrow_deflect_tail_grace_points", 0)
                ),
                "ENDPOINT_REGROW_GUARD_ENABLED": bool(getattr(_config, "endpoint_regrow_guard_enabled", True)),
                "ENDPOINT_REGROW_GUARD_BBOX_DEPTH_MM": float(
                    getattr(_config, "endpoint_regrow_guard_bbox_depth_mm", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_MIN_POINTS": int(
                    getattr(_config, "endpoint_regrow_guard_bbox_min_points", 0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_HALF_U_FLOOR_FACTOR": float(
                    getattr(_config, "endpoint_regrow_guard_bbox_half_u_floor_factor", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_HALF_V_FLOOR_FACTOR": float(
                    getattr(_config, "endpoint_regrow_guard_bbox_half_v_floor_factor", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_USE_POLAR_BOUNDARY": bool(
                    getattr(_config, "endpoint_regrow_guard_use_polar_boundary", True)
                ),
                "ENDPOINT_REGROW_GUARD_POLAR_BINS": int(
                    getattr(_config, "endpoint_regrow_guard_polar_bins", 0)
                ),
                "ENDPOINT_REGROW_GUARD_POLAR_MIN_BIN_POINTS": int(
                    getattr(_config, "endpoint_regrow_guard_polar_min_bin_points", 0)
                ),
                "ENDPOINT_REGROW_GUARD_POLAR_MIN_VALID_BINS": int(
                    getattr(_config, "endpoint_regrow_guard_polar_min_valid_bins", 0)
                ),
                "ENDPOINT_REGROW_GUARD_POLAR_RADIUS_QUANTILE": float(
                    getattr(_config, "endpoint_regrow_guard_polar_radius_quantile", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_POLAR_MARGIN_MM": float(
                    getattr(_config, "endpoint_regrow_guard_polar_margin_mm", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_POLAR_FAIL_ACTION": str(
                    getattr(_config, "endpoint_regrow_guard_polar_fail_action", "")
                ),
                "ENDPOINT_REGROW_DROP_GUARD_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_drop_guard_enabled", True)
                ),
                "ENDPOINT_REGROW_DROP_WINDOW_STEPS": int(
                    getattr(_config, "endpoint_regrow_drop_window_steps", 0)
                ),
                "ENDPOINT_REGROW_DROP_MIN_BASELINE_POINTS": int(
                    getattr(_config, "endpoint_regrow_drop_min_baseline_points", 0)
                ),
                "ENDPOINT_REGROW_DROP_RATIO": float(
                    getattr(_config, "endpoint_regrow_drop_ratio", 0.0)
                ),
                "ENDPOINT_REGROW_DROP_ABS_POINTS": int(
                    getattr(_config, "endpoint_regrow_drop_abs_points", 0)
                ),
                "ENDPOINT_REGROW_SPARSE_CONFIRM_POINTS": int(
                    getattr(_config, "endpoint_regrow_sparse_confirm_points", 0)
                ),
                "ENDPOINT_REGROW_SPARSE_CONFIRM_STEPS": int(
                    getattr(_config, "endpoint_regrow_sparse_confirm_steps", 0)
                ),
                "ENDPOINT_REGROW_SPARSE_CONTINUE_RATIO": float(
                    getattr(_config, "endpoint_regrow_sparse_continue_ratio", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_PROBE_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_guard_bbox_probe_enabled", True)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_PROBE_STEPS": int(
                    getattr(_config, "endpoint_regrow_guard_bbox_probe_steps", 0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_PROBE_DEPTH_MM": float(
                    getattr(_config, "endpoint_regrow_guard_bbox_probe_depth_mm", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_PROBE_RADIUS_MM": float(
                    getattr(_config, "endpoint_regrow_guard_bbox_probe_radius_mm", 0.0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_PROBE_MIN_POINTS": int(
                    getattr(_config, "endpoint_regrow_guard_bbox_probe_min_points", 0)
                ),
                "ENDPOINT_REGROW_GUARD_BBOX_PROBE_MIN_N_CAND_SCALE": float(
                    getattr(_config, "endpoint_regrow_guard_bbox_probe_min_n_cand_scale", 0.0)
                ),
                "ENDPOINT_REGROW_SPARSE_DENSE_EXTRA_CONFIRM_STEPS": int(
                    getattr(_config, "endpoint_regrow_sparse_dense_extra_confirm_steps", 0)
                ),
                "ENDPOINT_REGROW_SPARSE_DENSE_MIN_N_CAND_SCALE": float(
                    getattr(_config, "endpoint_regrow_sparse_dense_min_n_cand_scale", 0.0)
                ),
                "ENDPOINT_REGROW_DIRECTION_LOCK_ENABLED": bool(
                    getattr(_config, "endpoint_regrow_direction_lock_enabled", True)
                ),
                "ENDPOINT_REGROW_DIRECTION_LOCK_LATERAL_STREAK": int(
                    getattr(_config, "endpoint_regrow_direction_lock_lateral_streak", 0)
                ),
                "ENDPOINT_REGROW_DIRECTION_LOCK_ANGLE_DEG": float(
                    getattr(_config, "endpoint_regrow_direction_lock_angle_deg", 0.0)
                ),
                "ENDPOINT_REGROW_TRICLASS_BORDERLINE_HARD_AXIAL": bool(
                    getattr(_config, "endpoint_regrow_triclass_borderline_hard_axial", False)
                ),
                "ENDPOINT_BBOX_DEPTH_MM": float(getattr(_config, "endpoint_bbox_depth_mm", 0.0)),
                "ENDPOINT_BBOX_MIN_POINTS": int(getattr(_config, "endpoint_bbox_min_points", 0)),
                "ENDPOINT_BBOX_EXTEND_QUANTILE": float(getattr(_config, "endpoint_bbox_extend_quantile", 0.0)),
                "ENDPOINT_BBOX_MAX_EXTEND_MM": float(getattr(_config, "endpoint_bbox_max_extend_mm", 0.0)),
                "ENDPOINT_BBOX_AXIS_RADIUS_FACTOR": float(getattr(_config, "endpoint_bbox_axis_radius_factor", 0.0)),
                "ENDPOINT_BBOX_USE_OBB": bool(getattr(_config, "endpoint_bbox_use_obb", True)),
                "ENDPOINT_BBOX_SIZE_INWARD_MM": float(getattr(_config, "endpoint_bbox_size_inward_mm", 0.0)),
                "ENDPOINT_BBOX_SIZE_Q_LOW": float(getattr(_config, "endpoint_bbox_size_q_low", 0.0)),
                "ENDPOINT_BBOX_SIZE_Q_HIGH": float(getattr(_config, "endpoint_bbox_size_q_high", 0.0)),
                "ENDPOINT_BBOX_SIZE_MARGIN": float(getattr(_config, "endpoint_bbox_size_margin", 0.0)),
                "ENDPOINT_BBOX_MIN_HALF_U_FACTOR": float(getattr(_config, "endpoint_bbox_min_half_u_factor", 0.0)),
                "ENDPOINT_BBOX_MIN_HALF_V_FACTOR": float(getattr(_config, "endpoint_bbox_min_half_v_factor", 0.0)),
                "ENDPOINT_BBOX_HOOK_FIX_ENABLED": bool(getattr(_config, "endpoint_bbox_hook_fix_enabled", False)),
                "ENDPOINT_BBOX_HOOK_ANGLE_DEG": float(getattr(_config, "endpoint_bbox_hook_angle_deg", 0.0)),
                "ENDPOINT_BBOX_HOOK_BASE_ONLY": bool(getattr(_config, "endpoint_bbox_hook_base_only", True)),
            }
        )

    # Optional: GT evaluation. Only runs in direct execution mode, never on import (e.g. from 1.11.py).
    if str(_eval_gt_dir).strip():
        _ = evaluate_centerlines_against_gt(
            _pcd_path,
            _eval_gt_dir,
            out_subdir=EVAL_OUT_DIR,
            method="graph_slab",
            config=_config,
            step_mm=EVAL_STEP_MM,
            tangent_samples=EVAL_TANGENT_SAMPLES,
        )
        if bool(EVAL_ONLY):
            raise SystemExit(0)

    fit_curve_robust(_pcd_path, visualize=bool(_visualize), show_raw_cloud=True, config=_config)
