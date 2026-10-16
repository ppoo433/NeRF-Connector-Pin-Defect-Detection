# -*- coding: utf-8 -*-
"""
Export `reference_centerline_v1` artifacts from the existing centerline extractor.

P0 scope only:
- candidate project JSON (editable form)
- final centerline JSON (reference backbone form)
- preview PLY
- manifest JSON

Defaults are PyCharm-friendly: no required CLI args.
"""

from __future__ import annotations

_DAIMA_CTX = None
if __name__ == "__main__":
    try:
        import sys
        from pathlib import Path

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
        for p in [_REPO_ROOT, (_REPO_ROOT / "jiance").resolve()]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        from daima_runtime.runlog import setup_run

        _DAIMA_CTX = setup_run(__file__)
    except Exception:
        _DAIMA_CTX = None

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from pin_cloud2center_line import CenterlineConfig, UNIT_SCALE, extract_centerlines_from_pcd
import pin_cloud2center_line as _pcl
from reference_ply_io import (
    DEFAULT_GUI_CTRL_POINTS,
    DEFAULT_GUI_SMOOTH,
    write_ply_xyz_rgb_per_point,
)
from reference_slot_matching import ObservedNeedle, load_reference_slots, match_needles_to_slots
from registration.reg_common import LabeledCloud, _build_config as _reg_build_config, _compute_transform as _reg_compute_transform


# -----------------------------
# User-editable defaults
# -----------------------------

INPUT_ROOT = ""  # default: <repo_root>/point_cloud/test/split
OUTPUT_ROOT = ""  # default: <repo_root>/annotations/reference_centerlines_v1
SLOTS_JSON = ""  # default: <repo_root>/annotations/reference_slots_v1/reference_slots.json
SAMPLE_IDS: List[str] = ["1"]  # default smoke sample for PyCharm Run
INPUT_PARTITION = "bad"
CENTERLINE_METHOD = "graph_slab"
BACKBONE_VERSION = "reference_centerline_v1"
WRITE_PREVIEW_PLY = True
ALIGNMENT_METHOD = "overall_then_base_v1"
ALIGNMENT_MAX_RETRIES = 20
ALIGNMENT_GATE_MIN_FITNESS = 0.9
ALIGNMENT_GATE_MAX_RMSE_MM = 1.0


def _find_repo_root(from_path: Path) -> Path:
    from_path = from_path.resolve()
    for d in [from_path] + list(from_path.parents):
        if (d / ".release-root").is_file():
            return d
    for d in [from_path] + list(from_path.parents):
        if (d / ".git").exists():
            return d
    return from_path.parent


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): row.get(str(k), "") for k in fieldnames})


def _natural_key(text: str) -> List[object]:
    import re

    out: List[object] = []
    for part in re.split(r"(\d+)", str(text)):
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return out


def _resolve_input_root(repo_root: Path, cli_value: str) -> Path:
    raw = str(cli_value or INPUT_ROOT).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    return (repo_root / "point_cloud" / "test" / "split").resolve()


def _resolve_partitioned_input_root(input_root: Path, input_partition: str) -> Path:
    part = str(input_partition or INPUT_PARTITION).strip().lower()
    candidate = (input_root / part).resolve()
    if part in {"bad", "normal"} and candidate.is_dir():
        return candidate
    return input_root.resolve()


def _resolve_payload_path(repo_root: Path, raw_value: Any) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_output_root(repo_root: Path, cli_value: str) -> Path:
    raw = str(cli_value or OUTPUT_ROOT).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    return (repo_root / "annotations" / "reference_centerlines_v1").resolve()


def _resolve_slots_json(repo_root: Path, cli_value: str) -> Path:
    raw = str(cli_value or SLOTS_JSON).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    return (repo_root / "annotations" / "reference_slots_v1" / "reference_slots.json").resolve()


def _resolve_sample_ids(input_root: Path, cli_ids: Optional[Sequence[str]]) -> List[str]:
    if cli_ids:
        return [str(x).strip() for x in cli_ids if str(x).strip()]
    if SAMPLE_IDS:
        return [str(x).strip() for x in SAMPLE_IDS if str(x).strip()]
    import re

    ids: List[str] = []
    for p in input_root.glob("*.ply"):
        stem = str(p.stem)
        if re.fullmatch(r"\d+_base", stem):
            continue
        m = re.fullmatch(r"(\d+)_pins?", stem)
        if m:
            ids.append(str(m.group(1)))
            continue
        if re.fullmatch(r"\d+", stem):
            ids.append(stem)
    ids = sorted(set(ids), key=_natural_key)
    return ids[:1]


def _read_point_cloud_basic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = np.zeros((0, 3), dtype=np.float64)
    if cols.ndim != 2 or cols.shape[1] != 3 or len(cols) != len(pts):
        cols = np.zeros((len(pts), 3), dtype=np.float64)
    cols = np.nan_to_num(cols, nan=0.0, posinf=1.0, neginf=0.0)
    cols = np.clip(cols, 0.0, 1.0)
    return pts, cols


def _stack_cloud_parts(parts: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    pts_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []
    for pts, cols in parts:
        p = np.asarray(pts, dtype=np.float64)
        c = np.asarray(cols, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
            continue
        if c.ndim != 2 or c.shape[1] != 3 or len(c) != len(p):
            c = np.zeros((len(p), 3), dtype=np.float64)
        pts_list.append(p)
        cols_list.append(c)
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.vstack(pts_list), np.vstack(cols_list)


def _bundle_candidate_paths(input_root: Path, sample_id: str) -> Dict[str, List[Path]]:
    return {
        "pins": [
            (input_root / f"{sample_id}_pin.ply").resolve(),
            (input_root / f"{sample_id}_pins.ply").resolve(),
            (input_root / f"{sample_id}.ply").resolve(),
        ],
        "base": [
            (input_root / f"{sample_id}_base.ply").resolve(),
        ],
        "overall": [
            (input_root / f"{sample_id}_overall.ply").resolve(),
            (input_root / f"{sample_id}_cloud.ply").resolve(),
            (input_root / f"{sample_id}_all.ply").resolve(),
        ],
    }


def _pick_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _resolve_sample_cloud_bundle(input_root: Path, sample_id: str) -> Dict[str, Any]:
    cands = _bundle_candidate_paths(input_root, sample_id)
    pins_path = _pick_first_existing(cands["pins"])
    base_path = _pick_first_existing(cands["base"])
    overall_path = _pick_first_existing(cands["overall"])
    if pins_path is None:
        raise FileNotFoundError(str(cands["pins"][0]))
    if base_path is None:
        raise FileNotFoundError(str(cands["base"][0]))
    pins_points, pins_colors = _read_point_cloud_basic(pins_path)
    base_points, base_colors = _read_point_cloud_basic(base_path)
    overall_source = "file"
    if overall_path is not None:
        overall_points, overall_colors = _read_point_cloud_basic(overall_path)
    else:
        overall_source = "synthetic_pins_plus_base"
        overall_points, overall_colors = _stack_cloud_parts([(pins_points, pins_colors), (base_points, base_colors)])
    return {
        "sample_id": str(sample_id),
        "pins_path": str(pins_path),
        "base_path": str(base_path),
        "overall_path": ("" if overall_path is None else str(overall_path)),
        "overall_source": str(overall_source),
        "pins_points": pins_points,
        "pins_colors": pins_colors,
        "base_points": base_points,
        "base_colors": base_colors,
        "overall_points": overall_points,
        "overall_colors": overall_colors,
    }


def _bundle_to_labeled_cloud(bundle: Dict[str, Any]) -> LabeledCloud:
    return LabeledCloud(
        sample_id=str(bundle.get("sample_id", "")),
        ply_path=Path(str(bundle.get("overall_path", "") or bundle.get("pins_path", ""))).resolve(),
        overall_points=np.asarray(bundle.get("overall_points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64),
        overall_colors=np.asarray(bundle.get("overall_colors", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64),
        base_points=np.asarray(bundle.get("base_points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64),
        base_colors=np.asarray(bundle.get("base_colors", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64),
        pins_points=np.asarray(bundle.get("pins_points", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64),
        pins_colors=np.asarray(bundle.get("pins_colors", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64),
        meta={
            "pins_path": str(bundle.get("pins_path", "") or ""),
            "base_path": str(bundle.get("base_path", "") or ""),
            "overall_path": str(bundle.get("overall_path", "") or ""),
            "overall_source": str(bundle.get("overall_source", "") or ""),
            "n_total": int(len(np.asarray(bundle.get("overall_points", []), dtype=np.float64))),
            "n_pins": int(len(np.asarray(bundle.get("pins_points", []), dtype=np.float64))),
            "n_base": int(len(np.asarray(bundle.get("base_points", []), dtype=np.float64))),
        },
    )


def _build_reference_frame_contract(slots_payload: Dict[str, Any], reference_bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reference_frame_version": "reference_frame_v1",
        "frame_basis": "reference_slots_coordinate_system",
        "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
        "reference_sample_id": str(reference_bundle.get("sample_id", slots_payload.get("source_sample_id", "")) or ""),
        "reference_overall_source": str(reference_bundle.get("overall_source", "") or ""),
        "reference_bundle_root": str(reference_bundle.get("bundle_root", "") or ""),
        "reference_bundle_root_source": str(reference_bundle.get("bundle_root_source", "") or ""),
        "numbering_frame_contract": "sample_must_be_aligned_to_reference_slots_frame_before_slot_matching",
    }


def _resolve_reference_cloud_root(repo_root: Path, input_root: Path, slots_payload: Dict[str, Any]) -> Tuple[Path, str]:
    source_sample_id = str(slots_payload.get("source_sample_id", "") or "").strip()
    if not source_sample_id:
        raise ValueError("reference_slots.json missing source_sample_id")

    source_centerline_path = _resolve_payload_path(repo_root, slots_payload.get("source_centerline_path", ""))
    if source_centerline_path.is_file():
        try:
            centerline_payload = _read_json(source_centerline_path)
        except Exception:
            centerline_payload = {}
        source_cloud_path = _resolve_payload_path(repo_root, centerline_payload.get("source_cloud_path", ""))
        if source_cloud_path.is_file():
            return source_cloud_path.parent.resolve(), "source_centerline_source_cloud_path"

    current_root = input_root.resolve()
    if str(current_root.name).strip().lower() == "normal":
        sibling_bad_root = (current_root.parent / "bad").resolve()
        if _pick_first_existing(_bundle_candidate_paths(sibling_bad_root, source_sample_id)["pins"]) is not None:
            return sibling_bad_root, "sibling_bad_partition"

    if _pick_first_existing(_bundle_candidate_paths(current_root, source_sample_id)["pins"]) is not None:
        return current_root, "input_root_fallback"
    return current_root, "input_root_missing_reference_bundle"


def _resolve_reference_cloud_bundle(repo_root: Path, input_root: Path, slots_payload: Dict[str, Any]) -> Dict[str, Any]:
    source_sample_id = str(slots_payload.get("source_sample_id", "") or "").strip()
    reference_root, reference_root_source = _resolve_reference_cloud_root(repo_root, input_root, slots_payload)
    bundle = _resolve_sample_cloud_bundle(reference_root, source_sample_id)
    bundle["bundle_root"] = str(reference_root)
    bundle["bundle_root_source"] = str(reference_root_source)
    return bundle


def _mm_from_u(value_u: float, unit_scale: float) -> float:
    return float(value_u) / (float(unit_scale) + 1e-12)


def _alignment_attempt_configs(base_cfg) -> List[Any]:
    attempts: List[Any] = []
    voxel_scales_variants = [
        [1.2, 0.8, 0.4],
        [1.6, 1.0, 0.5],
        [1.0, 0.7, 0.35],
        [2.0, 1.2, 0.6],
        [0.8, 0.6, 0.3],
    ]
    corr_variants = [1.5, 1.7, 1.3, 1.9]
    robust_k_variants = [0.6, 0.8, 0.5, 1.0]
    target_grid_variants = [50000, 65000, 40000, 80000, 30000]
    backend_variants = [
        ("det_mh_base_pca_v1", False),
        ("det_mh_base_pca_v1", True),
        ("ransac", False),
        ("ransac", True),
    ]
    for idx in range(int(ALIGNMENT_MAX_RETRIES)):
        scales = voxel_scales_variants[idx % len(voxel_scales_variants)]
        corr = corr_variants[idx % len(corr_variants)]
        rk = robust_k_variants[idx % len(robust_k_variants)]
        grid = target_grid_variants[idx % len(target_grid_variants)]
        backend, allow_fb = backend_variants[idx % len(backend_variants)]
        attempts.append(
            replace(
                base_cfg,
                method=str(ALIGNMENT_METHOD),
                target_grid_points=int(grid),
                local_refine_voxel_scales_mm=[float(v) for v in scales],
                local_refine_max_corr_factor=float(corr),
                local_refine_robust_k_mm=float(rk),
                init_backend=str(backend),
                init_backend_allow_fallback=bool(allow_fb),
            )
        )
    return attempts


def _alignment_passes_gate(meta: Dict[str, Any], unit_scale: float) -> Tuple[bool, Dict[str, Any]]:
    fitness = _safe_float(meta.get("icp_fitness", float("nan")), float("nan"))
    rmse_u = _safe_float(meta.get("icp_inlier_rmse_u", float("nan")), float("nan"))
    rmse_mm = _mm_from_u(rmse_u, unit_scale) if np.isfinite(rmse_u) else float("nan")
    checks = {
        "status_ok": str(meta.get("status", "ok")) == "ok",
        "fitness_ok": bool(np.isfinite(fitness) and fitness >= float(ALIGNMENT_GATE_MIN_FITNESS)),
        "rmse_ok": bool(np.isfinite(rmse_mm) and rmse_mm <= float(ALIGNMENT_GATE_MAX_RMSE_MM)),
    }
    return bool(all(checks.values())), {
        "icp_fitness": float(fitness) if np.isfinite(fitness) else None,
        "icp_rmse_mm": float(rmse_mm) if np.isfinite(rmse_mm) else None,
        "gate_min_fitness": float(ALIGNMENT_GATE_MIN_FITNESS),
        "gate_max_rmse_mm": float(ALIGNMENT_GATE_MAX_RMSE_MM),
        "checks": checks,
    }


def _align_sample_to_reference_frame_with_retries(sample_bundle: Dict[str, Any], reference_bundle: Dict[str, Any], unit_scale: float) -> Dict[str, Any]:
    sample_cloud = _bundle_to_labeled_cloud(sample_bundle)
    reference_cloud = _bundle_to_labeled_cloud(reference_bundle)
    cfg_base = _reg_build_config(str(ALIGNMENT_METHOD))
    cfg_base = replace(
        cfg_base,
        method=str(ALIGNMENT_METHOD),
        unit_scale=float(unit_scale),
        ref_id=str(reference_bundle.get("sample_id", "ref") or "ref"),
        source_id=str(sample_bundle.get("sample_id", "src") or "src"),
    )
    attempts_meta: List[Dict[str, Any]] = []
    best_idx = -1
    best_score: Tuple[float, float] = (-1.0, float("inf"))
    best_T = np.eye(4, dtype=np.float64)
    best_meta: Dict[str, Any] = {"status": "failed", "reason": "no_attempts_passed"}

    for idx, cfg in enumerate(_alignment_attempt_configs(cfg_base), start=1):
        try:
            T, meta = _reg_compute_transform(cfg, reference_cloud, sample_cloud)
            gate_ok, gate_meta = _alignment_passes_gate(meta, unit_scale)
            attempt_meta = {
                "attempt_index": int(idx),
                "status": str(meta.get("status", "ok")),
                "init_backend": str(cfg.init_backend),
                "init_backend_allow_fallback": bool(cfg.init_backend_allow_fallback),
                "target_grid_points": int(cfg.target_grid_points),
                "local_refine_voxel_scales_mm": [float(v) for v in cfg.local_refine_voxel_scales_mm],
                "local_refine_max_corr_factor": float(cfg.local_refine_max_corr_factor),
                "local_refine_robust_k_mm": float(cfg.local_refine_robust_k_mm),
                "icp_fitness": gate_meta.get("icp_fitness", None),
                "icp_rmse_mm": gate_meta.get("icp_rmse_mm", None),
                "gate": gate_meta,
                "passed": bool(gate_ok),
            }
            attempts_meta.append(attempt_meta)
            score = (
                float(gate_meta.get("icp_fitness", -1.0) if gate_meta.get("icp_fitness", None) is not None else -1.0),
                float(gate_meta.get("icp_rmse_mm", float("inf")) if gate_meta.get("icp_rmse_mm", None) is not None else float("inf")),
            )
            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_idx = int(idx)
                best_score = score
                best_T = np.asarray(T, dtype=np.float64)
                best_meta = dict(meta)
                best_meta["gate"] = gate_meta
            if gate_ok:
                return {
                    "status": "ok",
                    "method": str(ALIGNMENT_METHOD),
                    "attempts": attempts_meta,
                    "attempt_count": int(idx),
                    "best_attempt_index": int(idx),
                    "transform": np.asarray(T, dtype=np.float64),
                    "best_meta": dict(best_meta),
                }
        except Exception as exc:
            attempts_meta.append(
                {
                    "attempt_index": int(idx),
                    "status": "exception",
                    "error": str(exc),
                    "passed": False,
                }
            )
            continue

    return {
        "status": "failed",
        "method": str(ALIGNMENT_METHOD),
        "attempts": attempts_meta,
        "attempt_count": int(len(attempts_meta)),
        "best_attempt_index": int(best_idx),
        "transform": np.asarray(best_T, dtype=np.float64),
        "best_meta": dict(best_meta),
    }


def _transform_xyz(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return pts.copy().reshape((-1, 3))
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    out = (np.asarray(T, dtype=np.float64) @ homo.T).T[:, :3]
    return out.astype(np.float64)


def _transform_infos_to_reference_frame(infos: Sequence[Dict[str, Any]], T: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    TT = np.asarray(T, dtype=np.float64)
    for info in infos:
        row = dict(info)
        try:
            src_sid = int(row.get("source_stable_id", row.get("stable_id", -1)))
        except Exception:
            src_sid = -1
        row["source_stable_id"] = int(src_sid)
        curve = np.asarray(row.get("curve_global_final", []), dtype=np.float64)
        if curve.ndim == 2 and curve.shape[1] == 3 and len(curve) > 0:
            row["curve_global_final"] = _transform_xyz(curve, TT)
        center = np.asarray(row.get("center_point", []), dtype=np.float64).reshape(-1)
        if center.size == 3 and np.all(np.isfinite(center)):
            row["center_point"] = _transform_xyz(center.reshape(1, 3), TT)[0]
        pts = np.asarray(row.get("points", []), dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0:
            row["points"] = _transform_xyz(pts, TT)
        row["alignment_method"] = str(ALIGNMENT_METHOD)
        out.append(row)
    return out


def _resample_polyline_by_arclength(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) <= 1:
        return pts.copy()
    n = int(max(2, n))
    if len(pts) == n:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        idx = np.linspace(0, len(pts) - 1, n)
        lo = np.floor(idx).astype(int)
        hi = np.ceil(idx).astype(int)
        t = (idx - lo).reshape(-1, 1)
        return pts[lo] * (1.0 - t) + pts[hi] * t
    q = np.linspace(0.0, total, n)
    out = np.zeros((n, 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(q, s, pts[:, k])
    return out


def _curve_axis(curve: np.ndarray) -> List[float]:
    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return [0.0, 0.0, 1.0]
    vec = np.asarray(pts[-1] - pts[0], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return [0.0, 0.0, 1.0]
    return (vec / norm).astype(float).tolist()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _curve_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 1:
        z = np.zeros((3,), dtype=np.float64)
        return z.copy(), z.copy()
    return pts[0].reshape(3), pts[-1].reshape(3)


def _polyline_length_mm(points: np.ndarray, unit_scale: float) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.sum(seg) / (float(unit_scale) + 1e-12))


def _build_observed_needles_from_infos(infos: Sequence[Dict[str, Any]], unit_scale: float) -> List[ObservedNeedle]:
    needles: List[ObservedNeedle] = []
    rows = sorted(infos, key=lambda d: (int(d.get("stable_id", 10**9)), int(d.get("needle_id", 10**9))))
    for obs_idx, info in enumerate(rows):
        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        base_u, tip_u = _curve_endpoints(curve)
        needles.append(
            ObservedNeedle(
                needle_id=int(info.get("needle_id", obs_idx)),
                observed_order_id=int(obs_idx),
                source_stable_id=int(info.get("stable_id", -1)) if int(info.get("stable_id", -1)) >= 0 else None,
                center_u=np.mean(curve, axis=0).reshape(3),
                curve_u=curve,
                length_mm=_polyline_length_mm(curve, unit_scale),
                base_point_u=base_u,
                tip_point_u=tip_u,
            )
        )
    return needles


def _assignment_from_match_result(match_result: Dict[str, Any], slots_json_path: Path, reference_slots_version: str) -> Dict[str, Any]:
    return {
        "reference_slots_version": str(reference_slots_version),
        "slots_json_path": str(slots_json_path),
        "assignment_version": str(match_result.get("assignment_version", "reference_slot_matching_v1")),
        "center_radius_mm": float(match_result.get("center_radius_mm", float("nan"))),
        "angle_gate_deg": float(match_result.get("angle_gate_deg", float("nan"))),
        "length_gate_mm": float(match_result.get("length_gate_mm", float("nan"))),
        "ambiguous_cost_delta": float(match_result.get("ambiguous_cost_delta", float("nan"))),
        "matched_pairs": list(match_result.get("matched_pairs", [])),
        "missing_stable_ids": list(match_result.get("missing_stable_ids", [])),
        "extra_needle_ids": list(match_result.get("extra_needle_ids", [])),
        "ambiguous_matches": list(match_result.get("ambiguous_matches", [])),
        "n_slots": int(match_result.get("n_slots", 0)),
        "n_needles": int(match_result.get("n_needles", 0)),
        "n_matched": int(match_result.get("n_matched", 0)),
        "n_missing": int(match_result.get("n_missing", 0)),
        "n_extra": int(match_result.get("n_extra", 0)),
        "n_ambiguous": int(match_result.get("n_ambiguous", 0)),
    }


def _apply_slot_matching(infos: Sequence[Dict[str, Any]], slots_json_path: Path, unit_scale: float) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    slots, slots_payload = load_reference_slots(slots_json_path)
    observed_needles = _build_observed_needles_from_infos(infos, unit_scale)
    match_result = match_needles_to_slots(slots, observed_needles, unit_scale=unit_scale)
    assigned_infos = _assigned_infos_from_pairs(infos, match_result.get("matched_pairs", []) or [])
    assignment = _assignment_from_match_result(match_result, slots_json_path, str(slots_payload.get("reference_slots_version", "reference_slots_v1")))
    return assigned_infos, assignment


def _apply_slot_matching_with_relative_rescue(
    infos: Sequence[Dict[str, Any]],
    slots_json_path: Path,
    unit_scale: float,
    *,
    slots_payload: Optional[Dict[str, Any]] = None,
    observed_needles: Optional[Sequence[ObservedNeedle]] = None,
    center_radius_mm: Any = "auto",
    angle_gate_deg: float = 35.0,
    length_gate_mm: float = 4.0,
    ambiguous_cost_delta: float = 0.12,
) -> Dict[str, Any]:
    slots, loaded_slots_payload = load_reference_slots(slots_json_path)
    payload = dict(slots_payload) if isinstance(slots_payload, dict) else dict(loaded_slots_payload)
    observed = list(observed_needles) if observed_needles is not None else _build_observed_needles_from_infos(infos, unit_scale)
    match_result = match_needles_to_slots(
        slots,
        observed,
        unit_scale=unit_scale,
        center_radius_mm=center_radius_mm,
        angle_gate_deg=float(angle_gate_deg),
        length_gate_mm=float(length_gate_mm),
        ambiguous_cost_delta=float(ambiguous_cost_delta),
    )
    assignment_baseline = _assignment_from_match_result(
        match_result,
        slots_json_path,
        str(payload.get("reference_slots_version", "reference_slots_v1")),
    )
    assigned_infos_baseline = _assigned_infos_from_pairs(infos, assignment_baseline.get("matched_pairs", []) or [])
    rescue_candidates = _build_relative_rescue_candidates(
        infos=infos,
        assignment=assignment_baseline,
        assigned_infos=assigned_infos_baseline,
        slots_payload=payload,
        unit_scale=unit_scale,
    )
    rescue_result = _run_relative_rescue_matching(rescue_candidates, assignment_baseline)
    assignment = _merge_assignment_with_rescue(assignment_baseline, rescue_result)
    assigned_infos = _assigned_infos_from_pairs(infos, assignment.get("matched_pairs", []) or [])
    return {
        "assigned_infos_baseline": assigned_infos_baseline,
        "assignment_baseline": assignment_baseline,
        "rescue_candidates": rescue_candidates,
        "rescue_result": rescue_result,
        "assignment": assignment,
        "assigned_infos": assigned_infos,
    }


def _assigned_infos_from_pairs(infos: Sequence[Dict[str, Any]], matched_pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_needle: Dict[int, Dict[str, Any]] = {int(info.get("needle_id", -1)): dict(info) for info in infos}
    assigned_infos: List[Dict[str, Any]] = []
    for pair in matched_pairs:
        needle_id = int(pair.get("needle_id", -1))
        if needle_id not in by_needle:
            continue
        info = dict(by_needle[needle_id])
        info["source_stable_id"] = int(info.get("stable_id", -1))
        info["stable_id"] = int(pair.get("stable_id", -1))
        info["slot_match_status"] = str(pair.get("match_status", "matched") or "matched")
        info["slot_match_cost"] = _safe_float(pair.get("match_cost", float("nan")), float("nan"))
        info["slot_center_dist_mm"] = _safe_float(pair.get("center_dist_mm", float("nan")), float("nan"))
        info["slot_direction_angle_deg"] = _safe_float(pair.get("direction_angle_deg", float("nan")), float("nan"))
        info["slot_length_diff_mm"] = _safe_float(pair.get("length_diff_mm", float("nan")), float("nan"))
        if "relative_score" in pair:
            info["relative_rescue_score"] = _safe_float(pair.get("relative_score", float("nan")), float("nan"))
        if "relative_offset_error_mm" in pair:
            info["relative_offset_error_mm"] = _safe_float(pair.get("relative_offset_error_mm", float("nan")), float("nan"))
        if "anchor_count_used" in pair:
            info["anchor_count_used"] = int(pair.get("anchor_count_used", 0) or 0)
        assigned_infos.append(info)
    assigned_infos = sorted(assigned_infos, key=lambda d: int(d.get("stable_id", 10**9)))
    return assigned_infos


def _baseline_assignment_maps(assignment: Dict[str, Any], assigned_infos: Sequence[Dict[str, Any]], infos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    info_by_needle = {int(info.get("needle_id", -1)): dict(info) for info in infos if int(info.get("needle_id", -1)) >= 0}
    assigned_by_needle = {int(info.get("needle_id", -1)): dict(info) for info in assigned_infos if int(info.get("needle_id", -1)) >= 0}
    matched_pairs = [dict(row) for row in (assignment.get("matched_pairs", []) or [])]
    matched_by_slot = {int(row.get("stable_id", -1)): dict(row) for row in matched_pairs if int(row.get("stable_id", -1)) >= 0}
    matched_by_needle = {int(row.get("needle_id", -1)): dict(row) for row in matched_pairs if int(row.get("needle_id", -1)) >= 0}
    return {
        "matched_pairs": matched_pairs,
        "matched_by_slot": matched_by_slot,
        "matched_by_needle": matched_by_needle,
        "assigned_by_needle": assigned_by_needle,
        "info_by_needle": info_by_needle,
        "missing_slot_ids": [int(x) for x in (assignment.get("missing_stable_ids", []) or [])],
        "extra_needle_ids": [int(x) for x in (assignment.get("extra_needle_ids", []) or [])],
    }


def _build_slot_topology_index(slots_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = [dict(row) for row in (slots_payload.get("slots", []) or []) if int(row.get("stable_id", -1)) >= 0]
    by_sid = {int(row.get("stable_id", -1)): dict(row) for row in rows}
    row_groups: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        row_groups.setdefault(int(row.get("row_id", -1)), []).append(row)
    for rid in list(row_groups.keys()):
        row_groups[rid] = sorted(row_groups[rid], key=lambda d: int(d.get("u_order", 10**9)))
    out: Dict[int, Dict[str, Any]] = {}
    for sid, row in by_sid.items():
        rid = int(row.get("row_id", -1))
        uo = int(row.get("u_order", -1))
        neighbors: List[int] = []
        for cand in row_groups.get(rid, []):
            csid = int(cand.get("stable_id", -1))
            if csid == sid:
                continue
            if abs(int(cand.get("u_order", -1)) - uo) <= 2:
                neighbors.append(csid)
        for orid, group in row_groups.items():
            if int(orid) == rid:
                continue
            for cand in group:
                csid = int(cand.get("stable_id", -1))
                if abs(int(cand.get("u_order", -1)) - uo) <= 1:
                    neighbors.append(csid)
        out[int(sid)] = {
            "stable_id": int(sid),
            "row_id": rid,
            "u_order": uo,
            "v_order": int(row.get("v_order", -1)),
            "nominal_center_u": np.asarray(row.get("nominal_center_u", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)[:3],
            "nominal_curve_u": np.asarray(row.get("nominal_curve_u", []), dtype=np.float64),
            "nominal_length_mm": _safe_float(row.get("nominal_length_mm", 0.0), 0.0),
            "neighbors": sorted(set(int(x) for x in neighbors if int(x) >= 0)),
        }
    return out


def _select_anchor_matches(assignment: Dict[str, Any], assigned_infos: Sequence[Dict[str, Any]], slots_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    baseline = _baseline_assignment_maps(assignment, assigned_infos, assigned_infos)
    pairs = list(baseline.get("matched_pairs", []))
    if not pairs:
        return []
    costs = [float(row.get("match_cost", float("nan"))) for row in pairs if np.isfinite(float(row.get("match_cost", float("nan"))))]
    center_vals = [float(row.get("center_dist_mm", float("nan"))) for row in pairs if np.isfinite(float(row.get("center_dist_mm", float("nan"))))]
    cost_thr = float(np.median(np.asarray(costs, dtype=np.float64))) if costs else float("inf")
    center_thr = float(np.percentile(np.asarray(center_vals, dtype=np.float64), 75)) if center_vals else float("inf")
    topo = _build_slot_topology_index(slots_payload)
    anchors: List[Dict[str, Any]] = []
    for pair in pairs:
        sid = int(pair.get("stable_id", -1))
        nid = int(pair.get("needle_id", -1))
        if sid < 0 or nid < 0:
            continue
        info = baseline.get("assigned_by_needle", {}).get(nid, {})
        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        pair_cost = _safe_float(pair.get("match_cost", float("nan")), float("nan"))
        pair_center = _safe_float(pair.get("center_dist_mm", float("nan")), float("nan"))
        if np.isfinite(pair_cost) and np.isfinite(cost_thr) and pair_cost > max(cost_thr, 0.45):
            continue
        if np.isfinite(pair_center) and np.isfinite(center_thr) and pair_center > max(center_thr, 1.25):
            continue
        anchors.append(
            {
                "stable_id": sid,
                "needle_id": nid,
                "center_u": np.mean(curve, axis=0).reshape(3).astype(float),
                "row_id": int(topo.get(sid, {}).get("row_id", -1)),
                "u_order": int(topo.get(sid, {}).get("u_order", -1)),
                "match_cost": pair_cost,
                "center_dist_mm": pair_center,
                "direction_angle_deg": _safe_float(pair.get("direction_angle_deg", float("nan")), float("nan")),
            }
        )
    if len(anchors) < 4:
        anchors = []
        for pair in pairs:
            sid = int(pair.get("stable_id", -1))
            nid = int(pair.get("needle_id", -1))
            info = baseline.get("assigned_by_needle", {}).get(nid, {})
            curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
            if sid < 0 or nid < 0 or curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
                continue
            anchors.append(
                {
                    "stable_id": sid,
                    "needle_id": nid,
                    "center_u": np.mean(curve, axis=0).reshape(3).astype(float),
                    "row_id": int(topo.get(sid, {}).get("row_id", -1)),
                    "u_order": int(topo.get(sid, {}).get("u_order", -1)),
                    "match_cost": _safe_float(pair.get("match_cost", float("nan")), float("nan")),
                    "center_dist_mm": _safe_float(pair.get("center_dist_mm", float("nan")), float("nan")),
                    "direction_angle_deg": _safe_float(pair.get("direction_angle_deg", float("nan")), float("nan")),
                }
            )
    return anchors


def _neighbor_anchor_slots(missing_sid: int, anchor_matches: Sequence[Dict[str, Any]], topology: Dict[int, Dict[str, Any]], max_anchor_count: int = 6) -> List[Dict[str, Any]]:
    target = topology.get(int(missing_sid), {})
    if not target:
        return []
    rid = int(target.get("row_id", -1))
    uo = int(target.get("u_order", -1))
    ranked = []
    for anchor in anchor_matches:
        asid = int(anchor.get("stable_id", -1))
        topo_a = topology.get(asid, {})
        if not topo_a:
            continue
        du = abs(int(topo_a.get("u_order", -1)) - uo)
        dr = abs(int(topo_a.get("row_id", -1)) - rid)
        ranked.append((dr * 10 + du, float(anchor.get("match_cost", float("inf"))), dict(anchor)))
    ranked = sorted(ranked, key=lambda x: (x[0], x[1]))
    return [dict(x[2]) for x in ranked[: max(1, int(max_anchor_count))]]


def _score_relative_slot_consistency(extra_info: Dict[str, Any], missing_sid: int, anchor_matches: Sequence[Dict[str, Any]], topology: Dict[int, Dict[str, Any]], unit_scale: float, center_radius_mm: float, angle_gate_deg: float, length_gate_mm: float) -> Optional[Dict[str, Any]]:
    curve = np.asarray(extra_info.get("curve_global_final", []), dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
        return None
    target = topology.get(int(missing_sid), {})
    target_center = np.asarray(target.get("nominal_center_u", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    if target_center.size != 3:
        return None
    obs_center = np.mean(curve, axis=0).reshape(3)
    used = _neighbor_anchor_slots(int(missing_sid), anchor_matches, topology)
    if len(used) < 2:
        return None
    rel_errors: List[float] = []
    same_row_count = 0
    target_row = int(target.get("row_id", -1))
    for anchor in used:
        asid = int(anchor.get("stable_id", -1))
        anchor_slot = topology.get(asid, {})
        anchor_slot_center = np.asarray(anchor_slot.get("nominal_center_u", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        anchor_obs_center = np.asarray(anchor.get("center_u", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if anchor_slot_center.size != 3 or anchor_obs_center.size != 3:
            continue
        d_ref = target_center - anchor_slot_center
        d_obs = obs_center - anchor_obs_center
        rel_errors.append(float(np.linalg.norm(d_obs - d_ref) / (float(unit_scale) + 1e-12)))
        if int(anchor.get("row_id", -999)) == target_row:
            same_row_count += 1
    if len(rel_errors) < 2:
        return None
    rel_error = float(np.mean(np.asarray(rel_errors, dtype=np.float64)))
    abs_center_dist = float(np.linalg.norm(obs_center - target_center) / (float(unit_scale) + 1e-12))
    target_curve = np.asarray(target.get("nominal_curve_u", []), dtype=np.float64)
    direction_angle = float("nan")
    if target_curve.ndim == 2 and target_curve.shape[1] == 3 and len(target_curve) >= 2:
        va = np.asarray(curve[-1] - curve[0], dtype=np.float64)
        vb = np.asarray(target_curve[-1] - target_curve[0], dtype=np.float64)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na > 1e-12 and nb > 1e-12:
            dot = float(np.clip(np.abs(np.dot(va / na, vb / nb)), 0.0, 1.0))
            direction_angle = float(np.degrees(np.arccos(dot)))
    length_diff = abs(float(_polyline_length_mm(curve, unit_scale)) - float(target.get("nominal_length_mm", 0.0)))
    row_consistency = float(same_row_count) / max(len(used), 1)
    score = 0.65 * (rel_error / max(center_radius_mm, 1e-9)) + 0.15 * (abs_center_dist / max(center_radius_mm, 1e-9))
    if np.isfinite(direction_angle):
        score += 0.10 * (direction_angle / max(angle_gate_deg, 1e-9))
    score += 0.10 * (length_diff / max(length_gate_mm, 1e-9))
    score -= 0.05 * row_consistency
    return {
        "needle_id": int(extra_info.get("needle_id", -1)),
        "candidate_stable_id": int(missing_sid),
        "relative_score": float(score),
        "anchor_count_used": int(len(used)),
        "relative_offset_error_mm": float(rel_error),
        "absolute_center_dist_mm": float(abs_center_dist),
        "direction_angle_deg": (None if not np.isfinite(direction_angle) else float(direction_angle)),
        "length_diff_mm": float(length_diff),
        "row_consistency": float(row_consistency),
        "anchor_stable_ids": [int(a.get("stable_id", -1)) for a in used],
    }


def _build_relative_rescue_candidates(infos: Sequence[Dict[str, Any]], assignment: Dict[str, Any], assigned_infos: Sequence[Dict[str, Any]], slots_payload: Dict[str, Any], unit_scale: float) -> List[Dict[str, Any]]:
    baseline = _baseline_assignment_maps(assignment, assigned_infos, infos)
    if not baseline.get("extra_needle_ids") or not baseline.get("missing_slot_ids"):
        return []
    topology = _build_slot_topology_index(slots_payload)
    anchors = _select_anchor_matches(assignment, assigned_infos, slots_payload)
    if len(anchors) < 2:
        return []
    candidates: List[Dict[str, Any]] = []
    for needle_id in baseline.get("extra_needle_ids", []):
        extra_info = baseline.get("info_by_needle", {}).get(int(needle_id), None)
        if not extra_info:
            continue
        for sid in baseline.get("missing_slot_ids", []):
            scored = _score_relative_slot_consistency(
                extra_info=extra_info,
                missing_sid=int(sid),
                anchor_matches=anchors,
                topology=topology,
                unit_scale=unit_scale,
                center_radius_mm=float(assignment.get("center_radius_mm", 1.0) or 1.0),
                angle_gate_deg=float(assignment.get("angle_gate_deg", 35.0) or 35.0),
                length_gate_mm=float(assignment.get("length_gate_mm", 4.0) or 4.0),
            )
            if scored is not None:
                candidates.append(scored)
    return sorted(candidates, key=lambda d: (float(d.get("relative_score", float("inf"))), float(d.get("relative_offset_error_mm", float("inf"))), float(d.get("absolute_center_dist_mm", float("inf")))))


def _run_relative_rescue_matching(candidates: Sequence[Dict[str, Any]], assignment: Dict[str, Any]) -> Dict[str, Any]:
    by_needle: Dict[int, List[Dict[str, Any]]] = {}
    for row in candidates:
        by_needle.setdefault(int(row.get("needle_id", -1)), []).append(dict(row))
    accepted: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    used_slots: set[int] = set()
    used_needles: set[int] = set()
    center_radius_mm = float(assignment.get("center_radius_mm", 1.0) or 1.0)
    angle_gate_deg = float(assignment.get("angle_gate_deg", 35.0) or 35.0)
    length_gate_mm = float(assignment.get("length_gate_mm", 4.0) or 4.0)
    total_extra = int(len(assignment.get("extra_needle_ids", []) or []))
    for needle_id in sorted(by_needle):
        rows = by_needle.get(int(needle_id), [])
        ranked = sorted(
            rows,
            key=lambda d: (
                float(d.get("relative_score", float("inf"))),
                float(d.get("relative_offset_error_mm", float("inf"))),
                float(d.get("absolute_center_dist_mm", float("inf"))),
                int(d.get("candidate_stable_id", 10**9)),
            ),
        )
        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None
        gap_ok = bool(top2 is None or (float(top2.get("relative_score", float("inf"))) - float(top1.get("relative_score", float("inf")))) >= 0.08)
        override_gap_ok = bool(top2 is None or (float(top2.get("relative_score", float("inf"))) - float(top1.get("relative_score", float("inf")))) >= 0.25)
        strict_accept = (
            int(top1.get("anchor_count_used", 0)) >= 2
            and float(top1.get("relative_offset_error_mm", float("inf"))) <= float(center_radius_mm)
            and float(top1.get("absolute_center_dist_mm", float("inf"))) <= float(center_radius_mm * 1.6)
            and (top1.get("direction_angle_deg", None) is None or float(top1.get("direction_angle_deg", 0.0)) <= float(angle_gate_deg * 1.2))
            and float(top1.get("length_diff_mm", float("inf"))) <= float(length_gate_mm * 1.25)
            and gap_ok
        )
        singleton_override = (
            total_extra == 1
            and override_gap_ok
            and int(top1.get("anchor_count_used", 0)) >= 4
            and float(top1.get("row_consistency", 0.0) or 0.0) >= 0.95
            and float(top1.get("relative_offset_error_mm", float("inf"))) <= float(center_radius_mm * 1.1)
            and float(top1.get("absolute_center_dist_mm", float("inf"))) <= float(center_radius_mm * 1.1)
            and (top1.get("direction_angle_deg", None) is None or float(top1.get("direction_angle_deg", 0.0)) <= float(angle_gate_deg * 1.8))
            and float(top1.get("length_diff_mm", float("inf"))) <= float(length_gate_mm * 2.5)
        )
        accept = bool(strict_accept or singleton_override)
        if not accept:
            ambiguous.append(
                {
                    "needle_id": int(needle_id),
                    "candidate_stable_ids": [int(r.get("candidate_stable_id", -1)) for r in ranked[:2]],
                    "candidate_relative_scores": [float(r.get("relative_score", float("inf"))) for r in ranked[:2]],
                    "strict_accept": bool(strict_accept),
                    "singleton_override": bool(singleton_override),
                    "reason": "relative_rescue_not_confident",
                }
            )
            continue
        sid = int(top1.get("candidate_stable_id", -1))
        if sid in used_slots or int(needle_id) in used_needles:
            continue
        used_slots.add(sid)
        used_needles.add(int(needle_id))
        accepted.append(
            {
                "stable_id": int(sid),
                "needle_id": int(needle_id),
                "match_status": "rescued_relative",
                "match_cost": float(top1.get("relative_score", float("inf"))),
                "center_dist_mm": float(top1.get("absolute_center_dist_mm", float("inf"))),
                "direction_angle_deg": (None if top1.get("direction_angle_deg", None) is None else float(top1.get("direction_angle_deg", 0.0))),
                "length_diff_mm": float(top1.get("length_diff_mm", float("inf"))),
                "relative_score": float(top1.get("relative_score", float("inf"))),
                "relative_offset_error_mm": float(top1.get("relative_offset_error_mm", float("inf"))),
                "anchor_count_used": int(top1.get("anchor_count_used", 0)),
                "anchor_stable_ids": list(top1.get("anchor_stable_ids", []) or []),
                "rescue_rule": ("strict" if strict_accept else "singleton_override"),
            }
        )
    return {
        "accepted_pairs": sorted(accepted, key=lambda d: int(d.get("stable_id", 10**9))),
        "ambiguous": ambiguous,
        "n_accepted": int(len(accepted)),
        "n_ambiguous": int(len(ambiguous)),
    }


def _merge_assignment_with_rescue(assignment: Dict[str, Any], rescue_result: Dict[str, Any]) -> Dict[str, Any]:
    rescue_pairs = [dict(row) for row in (rescue_result.get("accepted_pairs", []) or [])]
    rescue_by_slot = {int(r.get("stable_id", -1)): dict(r) for r in rescue_pairs if int(r.get("stable_id", -1)) >= 0}
    rescue_by_needle = {int(r.get("needle_id", -1)): dict(r) for r in rescue_pairs if int(r.get("needle_id", -1)) >= 0}
    baseline_pairs = [dict(row) for row in (assignment.get("matched_pairs", []) or [])]
    merged = dict(assignment)
    merged["assignment_version"] = "reference_slot_matching_v1+relative_rescue_v1"
    merged["matched_pairs"] = sorted(baseline_pairs + rescue_pairs, key=lambda d: int(d.get("stable_id", 10**9)))
    merged["missing_stable_ids"] = sorted(int(x) for x in (assignment.get("missing_stable_ids", []) or []) if int(x) not in rescue_by_slot)
    merged["extra_needle_ids"] = sorted(int(x) for x in (assignment.get("extra_needle_ids", []) or []) if int(x) not in rescue_by_needle)
    merged["rescue_pairs"] = sorted(rescue_pairs, key=lambda d: int(d.get("stable_id", 10**9)))
    merged["rescue_ambiguous"] = list(rescue_result.get("ambiguous", []) or [])
    merged["n_matched"] = int(len(merged["matched_pairs"]))
    merged["n_missing"] = int(len(merged["missing_stable_ids"]))
    merged["n_extra"] = int(len(merged["extra_needle_ids"]))
    merged["n_rescued"] = int(len(rescue_pairs))
    return merged


def _build_matching_comparison_report(sample_id: str, baseline_assignment: Dict[str, Any], merged_assignment: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "report_version": "matching_comparison_v1",
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "baseline": {
            "n_matched": int(baseline_assignment.get("n_matched", 0)),
            "n_missing": int(baseline_assignment.get("n_missing", 0)),
            "n_extra": int(baseline_assignment.get("n_extra", 0)),
            "missing_slot_ids": list(baseline_assignment.get("missing_stable_ids", []) or []),
            "extra_needle_ids": list(baseline_assignment.get("extra_needle_ids", []) or []),
        },
        "after_rescue": {
            "n_matched": int(merged_assignment.get("n_matched", 0)),
            "n_missing": int(merged_assignment.get("n_missing", 0)),
            "n_extra": int(merged_assignment.get("n_extra", 0)),
            "missing_slot_ids": list(merged_assignment.get("missing_stable_ids", []) or []),
            "extra_needle_ids": list(merged_assignment.get("extra_needle_ids", []) or []),
            "rescue_pairs": list(merged_assignment.get("rescue_pairs", []) or []),
        },
        "relative_rescue_candidates": int(len(candidates)),
    }


def _matched_info_by_sid(infos: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for info in infos:
        try:
            sid = int(info.get("stable_id", -1))
        except Exception:
            continue
        if sid < 0 or sid in out:
            continue
        out[sid] = dict(info)
    return out


def _matched_pair_by_sid(assignment: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for pair in assignment.get("matched_pairs", []) or []:
        try:
            sid = int(pair.get("stable_id", -1))
        except Exception:
            continue
        if sid < 0 or sid in out:
            continue
        out[sid] = dict(pair)
    return out


def _ambiguous_details_by_sid(assignment: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in assignment.get("ambiguous_matches", []) or []:
        try:
            needle_id = int(row.get("needle_id", -1))
        except Exception:
            needle_id = -1
        sids = row.get("candidate_stable_ids", []) or []
        costs = row.get("candidate_costs", []) or []
        for idx, sid_raw in enumerate(sids):
            try:
                sid = int(sid_raw)
            except Exception:
                continue
            if sid < 0:
                continue
            details = out.setdefault(sid, [])
            cost = None
            if idx < len(costs):
                try:
                    val = float(costs[idx])
                    if np.isfinite(val):
                        cost = float(val)
                except Exception:
                    cost = None
            details.append(
                {
                    "needle_id": needle_id,
                    "candidate_cost": cost,
                    "reason": str(row.get("reason", "") or ""),
                }
            )
    return out


def _build_slot_ledger_experiment(
    sample_id: str,
    slots_payload: Dict[str, Any],
    assigned_infos: Sequence[Dict[str, Any]],
    assignment: Dict[str, Any],
) -> Dict[str, Any]:
    matched_info_by_sid = _matched_info_by_sid(assigned_infos)
    matched_pair_by_sid = _matched_pair_by_sid(assignment)
    ambiguous_details = _ambiguous_details_by_sid(assignment)
    ambiguous_ids = set(int(sid) for sid in ambiguous_details.keys())
    missing_ids_all = {int(x) for x in (assignment.get("missing_stable_ids", []) or [])}
    missing_ids_resolved = set(int(x) for x in missing_ids_all if int(x) not in ambiguous_ids)

    rows: List[Dict[str, Any]] = []
    for slot in sorted(slots_payload.get("slots", []) or [], key=lambda d: int(d.get("stable_id", 10**9))):
        sid = int(slot.get("stable_id", -1))
        if sid < 0:
            continue
        info = matched_info_by_sid.get(sid, {})
        pair = matched_pair_by_sid.get(sid, {})
        amb_rows = ambiguous_details.get(sid, [])
        if sid in matched_info_by_sid:
            slot_presence_status = "matched"
        elif sid in ambiguous_ids:
            slot_presence_status = "ambiguous"
        elif sid in missing_ids_all:
            slot_presence_status = "missing"
        else:
            slot_presence_status = "unresolved"

        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        has_geometry = bool(curve.ndim == 2 and curve.shape[1] == 3 and len(curve) >= 2)
        row = {
            "stable_id": int(sid),
            "row_id": int(slot.get("row_id", -1)),
            "u_order": int(slot.get("u_order", -1)),
            "v_order": int(slot.get("v_order", -1)),
            "slot_label": str(slot.get("slot_label", "") or ""),
            "slot_status": str(slot.get("slot_status", "") or ""),
            "nominal_center_u": np.asarray(slot.get("nominal_center_u", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1).astype(float).tolist()[:3],
            "slot_presence_status": str(slot_presence_status),
            "matched_needle_id": (None if int(info.get("needle_id", -1)) < 0 else int(info.get("needle_id", -1))) if info else None,
            "source_stable_id": (None if int(info.get("source_stable_id", -1)) < 0 else int(info.get("source_stable_id", -1))) if info else None,
            "slot_match_status": str(info.get("slot_match_status", slot_presence_status) or slot_presence_status),
            "slot_match_cost": (None if not pair else _safe_float(pair.get("match_cost", float("nan")), float("nan"))),
            "slot_center_dist_mm": (None if not pair else _safe_float(pair.get("center_dist_mm", float("nan")), float("nan"))),
            "slot_direction_angle_deg": (None if not pair else _safe_float(pair.get("direction_angle_deg", float("nan")), float("nan"))),
            "slot_length_diff_mm": (None if not pair else _safe_float(pair.get("length_diff_mm", float("nan")), float("nan"))),
            "has_geometry": bool(has_geometry),
            "curve_points": int(len(curve)) if has_geometry else 0,
            "segment_count": int(1 if has_geometry else 0),
            "segments": ([curve.astype(float).tolist()] if has_geometry else []),
            "ambiguous_candidate_needle_ids": [int(r.get("needle_id", -1)) for r in amb_rows if int(r.get("needle_id", -1)) >= 0],
            "ambiguous_candidate_costs": [r.get("candidate_cost", None) for r in amb_rows],
        }
        rows.append(row)

    matched_ids = sorted(int(r["stable_id"]) for r in rows if str(r.get("slot_presence_status", "")) == "matched")
    ambiguous_slot_ids = sorted(int(r["stable_id"]) for r in rows if str(r.get("slot_presence_status", "")) == "ambiguous")
    missing_slot_ids = sorted(int(r["stable_id"]) for r in rows if str(r.get("slot_presence_status", "")) == "missing")
    unresolved_slot_ids = sorted(int(r["stable_id"]) for r in rows if str(r.get("slot_presence_status", "")) == "unresolved")
    return {
        "schema_version": 1,
        "experiment_version": "slot_ledger_experiment_v1",
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
        "assignment_version": str(assignment.get("assignment_version", "reference_slot_matching_v1")),
        "n_slots_expected": int(slots_payload.get("n_slots", len(rows)) or len(rows)),
        "n_slots_written": int(len(rows)),
        "matched_slot_ids": matched_ids,
        "ambiguous_slot_ids": ambiguous_slot_ids,
        "missing_slot_ids": missing_slot_ids,
        "missing_slot_ids_raw": sorted(int(x) for x in missing_ids_all),
        "missing_slot_ids_resolved": sorted(int(x) for x in missing_ids_resolved),
        "unresolved_slot_ids": unresolved_slot_ids,
        "slots": rows,
    }


def _build_alignment_failed_slot_ledger(sample_id: str, slots_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for slot in sorted(slots_payload.get("slots", []) or [], key=lambda d: int(d.get("stable_id", 10**9))):
        sid = int(slot.get("stable_id", -1))
        if sid < 0:
            continue
        rows.append(
            {
                "stable_id": int(sid),
                "row_id": int(slot.get("row_id", -1)),
                "u_order": int(slot.get("u_order", -1)),
                "v_order": int(slot.get("v_order", -1)),
                "slot_label": str(slot.get("slot_label", "") or ""),
                "slot_status": str(slot.get("slot_status", "") or ""),
                "nominal_center_u": np.asarray(slot.get("nominal_center_u", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1).astype(float).tolist()[:3],
                "slot_presence_status": "alignment_failed",
                "matched_needle_id": None,
                "source_stable_id": None,
                "slot_match_status": "alignment_failed",
                "slot_match_cost": None,
                "slot_center_dist_mm": None,
                "slot_direction_angle_deg": None,
                "slot_length_diff_mm": None,
                "has_geometry": False,
                "curve_points": 0,
                "segment_count": 0,
                "segments": [],
                "ambiguous_candidate_needle_ids": [],
                "ambiguous_candidate_costs": [],
            }
        )
    return {
        "schema_version": 1,
        "experiment_version": "slot_ledger_experiment_v1",
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
        "assignment_version": "reference_slot_matching_v1",
        "n_slots_expected": int(slots_payload.get("n_slots", len(rows)) or len(rows)),
        "n_slots_written": int(len(rows)),
        "matched_slot_ids": [],
        "ambiguous_slot_ids": [],
        "missing_slot_ids": [],
        "missing_slot_ids_raw": [],
        "missing_slot_ids_resolved": [],
        "unresolved_slot_ids": sorted(int(r["stable_id"]) for r in rows),
        "slots": rows,
    }


def _build_numbering_health(slot_ledger: Dict[str, Any], assignment: Dict[str, Any], alignment_result: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(slot_ledger.get("slots", []) or [])
    matched = [int(r.get("stable_id", -1)) for r in rows if str(r.get("slot_presence_status", "")) == "matched"]
    missing = [int(r.get("stable_id", -1)) for r in rows if str(r.get("slot_presence_status", "")) == "missing"]
    ambiguous = [int(r.get("stable_id", -1)) for r in rows if str(r.get("slot_presence_status", "")) == "ambiguous"]
    unresolved = [int(r.get("stable_id", -1)) for r in rows if str(r.get("slot_presence_status", "")) not in {"matched", "missing", "ambiguous"}]
    center_vals = [float(r.get("slot_center_dist_mm")) for r in rows if r.get("slot_center_dist_mm", None) is not None and np.isfinite(float(r.get("slot_center_dist_mm")))]
    angle_vals = [float(r.get("slot_direction_angle_deg")) for r in rows if r.get("slot_direction_angle_deg", None) is not None and np.isfinite(float(r.get("slot_direction_angle_deg")))]
    return {
        "schema_version": 1,
        "status": ("ok" if str(alignment_result.get("status", "")) == "ok" else "blocked_by_alignment"),
        "n_slots_expected": int(slot_ledger.get("n_slots_expected", len(rows)) or len(rows)),
        "n_slots_written": int(len(rows)),
        "n_matched": int(len(matched)),
        "n_missing": int(len(missing)),
        "n_ambiguous": int(len(ambiguous)),
        "n_unresolved": int(len(unresolved)),
        "n_extra": int(len(assignment.get("extra_needle_ids", []) or [])),
        "matched_slot_ids": sorted(matched),
        "missing_slot_ids": sorted(missing),
        "ambiguous_slot_ids": sorted(ambiguous),
        "unresolved_slot_ids": sorted(unresolved),
        "max_center_dist_mm": (None if not center_vals else float(np.max(center_vals))),
        "p95_center_dist_mm": (None if not center_vals else float(np.percentile(np.asarray(center_vals, dtype=np.float64), 95))),
        "max_direction_angle_deg": (None if not angle_vals else float(np.max(angle_vals))),
        "alignment_status": str(alignment_result.get("status", "")),
        "alignment_attempts": int(alignment_result.get("attempt_count", 0) or 0),
    }


def _build_slot_ledger_consistency_report(
    sample_id: str,
    slots_payload: Dict[str, Any],
    assignment: Dict[str, Any],
    slot_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    template_slot_ids = sorted(
        int(row.get("stable_id", -1))
        for row in (slots_payload.get("slots", []) or [])
        if int(row.get("stable_id", -1)) >= 0
    )
    ledger_slot_ids = sorted(
        int(row.get("stable_id", -1))
        for row in (slot_ledger.get("slots", []) or [])
        if int(row.get("stable_id", -1)) >= 0
    )
    ledger_dups = sorted({sid for sid in ledger_slot_ids if ledger_slot_ids.count(sid) > 1})
    assignment_matched = sorted(int(row.get("stable_id", -1)) for row in (assignment.get("matched_pairs", []) or []) if int(row.get("stable_id", -1)) >= 0)
    assignment_ambiguous = sorted(int(x) for x in _ambiguous_details_by_sid(assignment).keys())
    assignment_missing_all = sorted(int(x) for x in (assignment.get("missing_stable_ids", []) or []))
    assignment_missing_resolved = sorted(int(x) for x in assignment_missing_all if int(x) not in set(assignment_ambiguous))

    ledger_matched = sorted(int(row.get("stable_id", -1)) for row in (slot_ledger.get("slots", []) or []) if str(row.get("slot_presence_status", "")) == "matched")
    ledger_ambiguous = sorted(int(row.get("stable_id", -1)) for row in (slot_ledger.get("slots", []) or []) if str(row.get("slot_presence_status", "")) == "ambiguous")
    ledger_missing = sorted(int(row.get("stable_id", -1)) for row in (slot_ledger.get("slots", []) or []) if str(row.get("slot_presence_status", "")) == "missing")
    ledger_unresolved = sorted(int(row.get("stable_id", -1)) for row in (slot_ledger.get("slots", []) or []) if str(row.get("slot_presence_status", "")) == "unresolved")

    checks = {
        "template_slot_ids_match": bool(template_slot_ids == ledger_slot_ids),
        "ledger_unique_stable_ids": bool(len(ledger_dups) == 0),
        "matched_slot_ids_match_assignment": bool(assignment_matched == ledger_matched),
        "ambiguous_slot_ids_match_assignment": bool(assignment_ambiguous == ledger_ambiguous),
        "missing_slot_ids_match_assignment_resolved": bool(assignment_missing_resolved == ledger_missing),
        "missing_plus_ambiguous_cover_assignment_missing": bool(sorted(ledger_missing + ledger_ambiguous) == assignment_missing_all),
        "no_unresolved_slots": bool(len(ledger_unresolved) == 0),
        "matched_slots_nonempty": bool(len(template_slot_ids) == 0 or len(assignment_matched) > 0),
        "not_all_slots_missing": bool(len(template_slot_ids) == 0 or len(assignment_missing_all) < len(template_slot_ids)),
    }
    status = "pass" if all(bool(v) for v in checks.values()) else "fail"
    matched_slot_coverage = (
        float(len(assignment_matched)) / float(len(template_slot_ids))
        if template_slot_ids
        else 0.0
    )
    return {
        "schema_version": 1,
        "report_version": "slot_ledger_consistency_v1",
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "status": str(status),
        "checks": checks,
        "matched_slot_coverage": float(matched_slot_coverage),
        "template_slot_ids": template_slot_ids,
        "ledger_slot_ids": ledger_slot_ids,
        "ledger_duplicate_stable_ids": ledger_dups,
        "assignment_matched_slot_ids": assignment_matched,
        "ledger_matched_slot_ids": ledger_matched,
        "assignment_ambiguous_slot_ids": assignment_ambiguous,
        "ledger_ambiguous_slot_ids": ledger_ambiguous,
        "assignment_missing_slot_ids_raw": assignment_missing_all,
        "assignment_missing_slot_ids_resolved": assignment_missing_resolved,
        "ledger_missing_slot_ids": ledger_missing,
        "ledger_unresolved_slot_ids": ledger_unresolved,
    }


def _build_drop_one_invariance_report(
    sample_id: str,
    slots_json_path: Path,
    infos: Sequence[Dict[str, Any]],
    assignment: Dict[str, Any],
    unit_scale: float,
) -> Dict[str, Any]:
    _slots, slots_payload = load_reference_slots(slots_json_path)
    observed_needles = _build_observed_needles_from_infos(infos, unit_scale)
    matched_pairs = sorted(
        [dict(row) for row in (assignment.get("matched_pairs", []) or [])],
        key=lambda d: (
            _safe_float(d.get("match_cost", float("inf")), float("inf")),
            _safe_float(d.get("center_dist_mm", float("inf")), float("inf")),
            int(d.get("stable_id", 10**9)),
        ),
    )
    if not matched_pairs:
        return {
            "schema_version": 1,
            "report_version": "drop_one_invariance_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "no_matched_pairs_available",
        }

    drop_pair = matched_pairs[0]
    dropped_needle_id = int(drop_pair.get("needle_id", -1))
    dropped_stable_id = int(drop_pair.get("stable_id", -1))
    if dropped_needle_id < 0 or dropped_stable_id < 0:
        return {
            "schema_version": 1,
            "report_version": "drop_one_invariance_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "invalid_drop_pair",
            "drop_pair": drop_pair,
        }

    needles_after = [n for n in observed_needles if int(getattr(n, "needle_id", -1)) != dropped_needle_id]
    post_bundle = _apply_slot_matching_with_relative_rescue(
        infos=infos,
        slots_json_path=slots_json_path,
        unit_scale=unit_scale,
        slots_payload=slots_payload,
        observed_needles=needles_after,
        center_radius_mm=assignment.get("center_radius_mm", "auto"),
        angle_gate_deg=float(assignment.get("angle_gate_deg", 35.0) or 35.0),
        length_gate_mm=float(assignment.get("length_gate_mm", 4.0) or 4.0),
        ambiguous_cost_delta=float(assignment.get("ambiguous_cost_delta", 0.12) or 0.12),
    )
    post = dict(post_bundle.get("assignment", {}) or {})

    baseline_by_needle = {int(row.get("needle_id", -1)): int(row.get("stable_id", -1)) for row in matched_pairs if int(row.get("needle_id", -1)) >= 0 and int(row.get("stable_id", -1)) >= 0}
    post_by_needle = {int(row.get("needle_id", -1)): int(row.get("stable_id", -1)) for row in (post.get("matched_pairs", []) or []) if int(row.get("needle_id", -1)) >= 0 and int(row.get("stable_id", -1)) >= 0}

    changed_non_dropped: List[Dict[str, Any]] = []
    disappeared_non_dropped: List[Dict[str, Any]] = []
    for needle_id, before_sid in sorted(baseline_by_needle.items()):
        if int(needle_id) == dropped_needle_id:
            continue
        if int(needle_id) not in post_by_needle:
            disappeared_non_dropped.append({"needle_id": int(needle_id), "before_stable_id": int(before_sid)})
            continue
        after_sid = int(post_by_needle[needle_id])
        if after_sid != int(before_sid):
            changed_non_dropped.append({"needle_id": int(needle_id), "before_stable_id": int(before_sid), "after_stable_id": int(after_sid)})

    baseline_missing = {int(x) for x in (assignment.get("missing_stable_ids", []) or [])}
    post_missing = {int(x) for x in (post.get("missing_stable_ids", []) or [])}
    new_missing_slot_ids = sorted(int(x) for x in (post_missing - baseline_missing))
    checks = {
        "exactly_one_new_missing_slot": bool(new_missing_slot_ids == [dropped_stable_id]),
        "no_changed_non_dropped_slots": bool(len(changed_non_dropped) == 0),
        "no_disappeared_non_dropped_needles": bool(len(disappeared_non_dropped) == 0),
    }
    status = "pass" if all(bool(v) for v in checks.values()) else "fail"
    return {
        "schema_version": 1,
        "report_version": "drop_one_invariance_v1",
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
        "assignment_version": str(assignment.get("assignment_version", "reference_slot_matching_v1")),
        "status": str(status),
        "checks": checks,
        "dropped_needle_id": int(dropped_needle_id),
        "dropped_stable_id": int(dropped_stable_id),
        "baseline_missing_slot_ids": sorted(int(x) for x in baseline_missing),
        "post_drop_missing_slot_ids": sorted(int(x) for x in post_missing),
        "new_missing_slot_ids": new_missing_slot_ids,
        "changed_non_dropped_slots": changed_non_dropped,
        "disappeared_non_dropped_needles": disappeared_non_dropped,
        "baseline_matched_pairs": matched_pairs,
        "post_drop_matched_pairs": list(post.get("matched_pairs", []) or []),
    }


def _experiment_output_paths(sample_id: str) -> Dict[str, Path]:
    repo_root = _find_repo_root(Path(__file__))
    if _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None):
        root = Path(_DAIMA_CTX.run_dir).resolve()
    else:
        root = (repo_root / "debug_out" / "adhoc_export_reference_centerline_v1").resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"sample_{sample_id}"
    return {
        "slot_ledger_json": (root / f"{prefix}_slot_ledger_experiment.json").resolve(),
        "slot_ledger_csv": (root / f"{prefix}_slot_ledger_summary.csv").resolve(),
        "slot_ledger_consistency_json": (root / f"{prefix}_slot_ledger_consistency_report.json").resolve(),
        "drop_one_invariance_json": (root / f"{prefix}_drop_one_invariance_report.json").resolve(),
        "observed_order_invariance_json": (root / f"{prefix}_observed_order_invariance_report.json").resolve(),
        "relative_rescue_report_json": (root / f"{prefix}_relative_rescue_report.json").resolve(),
        "relative_rescue_candidates_csv": (root / f"{prefix}_relative_rescue_candidates.csv").resolve(),
        "matching_comparison_report_json": (root / f"{prefix}_matching_comparison_report.json").resolve(),
    }


def _batch_output_paths(output_root: Path) -> Dict[str, Path]:
    batch_root = Path(output_root).resolve().parent
    batch_root.mkdir(parents=True, exist_ok=True)
    return {
        "reference_frame_json": (batch_root / "reference_frame_v1.json").resolve(),
        "batch_alignment_summary_json": (batch_root / "batch_alignment_summary.json").resolve(),
        "batch_alignment_summary_csv": (batch_root / "batch_alignment_summary.csv").resolve(),
        "cross_sample_slot_consistency_json": (batch_root / "cross_sample_slot_consistency_report.json").resolve(),
        "cross_sample_slot_consistency_csv": (batch_root / "cross_sample_slot_consistency.csv").resolve(),
        "batch_gate_summary_json": (batch_root / "batch_gate_summary.json").resolve(),
    }


def _write_reference_frame_artifact(path: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "schema_version": 1,
        "created_at": _now_iso(),
        **dict(contract),
    }
    _write_json(path, payload)
    return payload


def _build_observed_order_invariance_report(
    sample_id: str,
    infos_aligned: Sequence[Dict[str, Any]],
    slots_json_path: Path,
    assignment: Dict[str, Any],
    unit_scale: float,
) -> Dict[str, Any]:
    observed = _build_observed_needles_from_infos(infos_aligned, unit_scale)
    if not observed:
        return {
            "schema_version": 1,
            "report_version": "observed_order_invariance_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "no_observed_needles",
        }
    rng = np.random.default_rng(0)
    order = np.arange(len(observed), dtype=np.int64)
    rng.shuffle(order)
    observed_shuffled = [observed[int(i)] for i in order]
    _slots, slots_payload = load_reference_slots(slots_json_path)
    shuffled_bundle = _apply_slot_matching_with_relative_rescue(
        infos=infos_aligned,
        slots_json_path=slots_json_path,
        unit_scale=unit_scale,
        slots_payload=slots_payload,
        observed_needles=observed_shuffled,
        center_radius_mm=assignment.get("center_radius_mm", "auto"),
        angle_gate_deg=float(assignment.get("angle_gate_deg", 35.0) or 35.0),
        length_gate_mm=float(assignment.get("length_gate_mm", 4.0) or 4.0),
        ambiguous_cost_delta=float(assignment.get("ambiguous_cost_delta", 0.12) or 0.12),
    )
    shuffled = dict(shuffled_bundle.get("assignment", {}) or {})
    baseline_map = {
        int(row.get("needle_id", -1)): int(row.get("stable_id", -1))
        for row in (assignment.get("matched_pairs", []) or [])
        if int(row.get("needle_id", -1)) >= 0 and int(row.get("stable_id", -1)) >= 0
    }
    shuffled_map = {
        int(row.get("needle_id", -1)): int(row.get("stable_id", -1))
        for row in (shuffled.get("matched_pairs", []) or [])
        if int(row.get("needle_id", -1)) >= 0 and int(row.get("stable_id", -1)) >= 0
    }
    changed: List[Dict[str, Any]] = []
    for needle_id, sid_before in sorted(baseline_map.items()):
        sid_after = shuffled_map.get(int(needle_id), None)
        if sid_after is None:
            changed.append({"needle_id": int(needle_id), "before_stable_id": int(sid_before), "after_stable_id": None})
        elif int(sid_after) != int(sid_before):
            changed.append({"needle_id": int(needle_id), "before_stable_id": int(sid_before), "after_stable_id": int(sid_after)})
    checks = {
        "matched_mapping_preserved": bool(len(changed) == 0),
        "missing_slot_ids_preserved": bool(sorted(int(x) for x in (assignment.get("missing_stable_ids", []) or [])) == sorted(int(x) for x in (shuffled.get("missing_stable_ids", []) or []))),
        "ambiguous_count_preserved": bool(len(assignment.get("ambiguous_matches", []) or []) == len(shuffled.get("ambiguous_matches", []) or [])),
    }
    status = "pass" if all(bool(v) for v in checks.values()) else "fail"
    return {
        "schema_version": 1,
        "report_version": "observed_order_invariance_v1",
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "status": str(status),
        "checks": checks,
        "shuffle_order": [int(x) for x in order.tolist()],
        "changed_matches": changed,
        "baseline_missing_slot_ids": sorted(int(x) for x in (assignment.get("missing_stable_ids", []) or [])),
        "shuffled_missing_slot_ids": sorted(int(x) for x in (shuffled.get("missing_stable_ids", []) or [])),
    }


def _sample_summary_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": str(entry.get("sample_id", "")),
        "status": str(entry.get("status", "")),
        "alignment_status": str(entry.get("alignment_status", "")),
        "alignment_attempts": int(entry.get("alignment_attempts", 0) or 0),
        "sample_overall_source": str(((entry.get("sample_bundle", {}) or {}).get("overall_source", "")) or ""),
        "reference_overall_source": str(((entry.get("reference_bundle", {}) or {}).get("overall_source", "")) or ""),
        "n_needles": int(entry.get("n_needles", 0) or 0),
        "n_ambiguous": int(entry.get("n_ambiguous", 0) or 0),
        "slot_ledger_status": str(entry.get("slot_ledger_status", "")),
        "drop_one_invariance_status": str(entry.get("drop_one_invariance_status", "")),
        "observed_order_invariance_status": str(entry.get("observed_order_invariance_status", "")),
        "missing_stable_ids": ";".join(str(int(x)) for x in (entry.get("missing_stable_ids", []) or [])),
        "extra_needle_ids": ";".join(str(int(x)) for x in (entry.get("extra_needle_ids", []) or [])),
        "matched_slot_ids": ";".join(str(int(x)) for x in (entry.get("matched_slot_ids", []) or [])),
        "ambiguous_slot_ids": ";".join(str(int(x)) for x in (entry.get("ambiguous_slot_ids", []) or [])),
        "unresolved_slot_ids": ";".join(str(int(x)) for x in (entry.get("unresolved_slot_ids", []) or [])),
    }


def _build_batch_alignment_summary(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [_sample_summary_row(e) for e in entries]
    alignment_ok = sum(1 for e in entries if str(e.get("alignment_status", "")) == "ok")
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "n_samples": int(len(entries)),
        "alignment_ok_count": int(alignment_ok),
        "alignment_fail_count": int(len(entries) - alignment_ok),
        "rows": rows,
    }


def _collect_slot_status_matrix(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matrix: List[Dict[str, Any]] = []
    for entry in entries:
        sample_id = str(entry.get("sample_id", ""))
        final_path = Path(str(entry.get("final_json_path", "") or ""))
        if not final_path.is_file():
            continue
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        for row in payload.get("slots", []) or []:
            try:
                sid = int(row.get("stable_id", -1))
            except Exception:
                continue
            if sid < 0:
                continue
            matrix.append(
                {
                    "sample_id": str(sample_id),
                    "alignment_status": str(entry.get("alignment_status", "")),
                    "stable_id": int(sid),
                    "slot_presence_status": str(row.get("slot_presence_status", "") or ""),
                    "matched_needle_id": row.get("matched_needle_id", None),
                    "source_stable_id": row.get("source_stable_id", None),
                    "slot_match_cost": row.get("slot_match_cost", None),
                }
            )
    return matrix


def _build_cross_sample_slot_consistency_report(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    matrix = _collect_slot_status_matrix(entries)
    by_slot: Dict[int, List[Dict[str, Any]]] = {}
    for row in matrix:
        by_slot.setdefault(int(row.get("stable_id", -1)), []).append(dict(row))
    slot_summaries: List[Dict[str, Any]] = []
    for sid in sorted(by_slot.keys()):
        rows = by_slot[sid]
        matched = sum(1 for r in rows if str(r.get("slot_presence_status", "")) == "matched")
        missing = sum(1 for r in rows if str(r.get("slot_presence_status", "")) == "missing")
        ambiguous = sum(1 for r in rows if str(r.get("slot_presence_status", "")) == "ambiguous")
        failed_align = sum(1 for r in rows if str(r.get("alignment_status", "")) != "ok")
        slot_summaries.append(
            {
                "stable_id": int(sid),
                "sample_count": int(len(rows)),
                "matched_count": int(matched),
                "missing_count": int(missing),
                "ambiguous_count": int(ambiguous),
                "alignment_failed_count": int(failed_align),
                "matched_rate": float(matched / len(rows)) if rows else 0.0,
                "missing_rate": float(missing / len(rows)) if rows else 0.0,
                "ambiguous_rate": float(ambiguous / len(rows)) if rows else 0.0,
            }
        )
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "n_samples": int(len({str(r.get('sample_id', '')) for r in matrix})),
        "n_slots": int(len(slot_summaries)),
        "slot_summaries": slot_summaries,
        "matrix": matrix,
    }


def _build_batch_gate_summary(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = int(len(entries))
    alignment_ok = sum(1 for e in entries if str(e.get("alignment_status", "")) == "ok")
    ledger_ok = sum(1 for e in entries if str(e.get("slot_ledger_status", "")) == "pass")
    drop_ok = sum(1 for e in entries if str(e.get("drop_one_invariance_status", "")) == "pass")
    order_ok = sum(1 for e in entries if str(e.get("observed_order_invariance_status", "")) == "pass")
    pilot_mode = bool(n <= 5)
    if pilot_mode:
        passed = bool(n > 0 and ledger_ok == n and drop_ok == n and order_ok == n and alignment_ok >= max(1, n - 1))
    else:
        passed = bool(n > 0 and ledger_ok == n and drop_ok == n and order_ok == n and alignment_ok / max(n, 1) >= 0.9)
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "mode": ("pilot" if pilot_mode else "full_batch"),
        "n_samples": int(n),
        "alignment_ok_count": int(alignment_ok),
        "slot_ledger_pass_count": int(ledger_ok),
        "drop_one_pass_count": int(drop_ok),
        "observed_order_pass_count": int(order_ok),
        "passed": bool(passed),
    }


def _palette_rgb(sid: int) -> List[int]:
    palette = [
        [255, 80, 80],
        [80, 180, 255],
        [80, 220, 120],
        [255, 200, 90],
        [180, 120, 255],
        [255, 120, 200],
        [120, 220, 220],
        [230, 230, 110],
    ]
    return palette[int(sid) % len(palette)]


def _build_candidate_project(
    sample_id: str,
    source_cloud_path: Path,
    output_dir: Path,
    infos: Sequence[Dict[str, Any]],
    unit_scale: float,
    reference_frame_contract: Optional[Dict[str, Any]] = None,
    alignment_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    rows = sorted(infos, key=lambda d: int(d.get("stable_id", 10**9)))
    for info in rows:
        sid = int(info.get("stable_id", -1))
        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        if sid < 0 or curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        ctrl_n = int(max(4, min(int(DEFAULT_GUI_CTRL_POINTS), int(len(curve)))))
        ctrl_points = _resample_polyline_by_arclength(curve, ctrl_n)
        clusters.append(
            {
                "tag": f"stable_{sid:02d}",
                "stable_id": int(sid),
                "source_stable_id": int(info.get("source_stable_id", -1)),
                "needle_id": int(info.get("needle_id", -1)),
                "slot_match_status": str(info.get("slot_match_status", "matched")),
                "axis": _curve_axis(curve),
                "ctrl_points": ctrl_points.astype(float).tolist(),
                "smooth_strength": float(DEFAULT_GUI_SMOOTH),
                "selected_idx": 0,
                "upper_component": None,
                "pin_width": _safe_float(info.get("pin_width", 0.0), 0.0),
                "pin_thickness": _safe_float(info.get("pin_thickness", 0.0), 0.0),
                "straight_ranges": [],
            }
        )
    return {
        "schema_version": 1,
        "input_ply": str(source_cloud_path),
        "output_dir": str(output_dir),
        "unit_scale": float(unit_scale),
        "reference_frame_version": str((reference_frame_contract or {}).get("reference_frame_version", "")),
        "alignment_method": str((alignment_result or {}).get("method", "")),
        "alignment_status": str((alignment_result or {}).get("status", "")),
        "current_idx": 0,
        "move_step": 0.0,
        "clusters": clusters,
    }


def _build_final_json(
    sample_id: str,
    source_cloud_path: Path,
    infos: Sequence[Dict[str, Any]],
    unit_scale: float,
    method: str,
    assignment: Dict[str, Any],
    slot_ledger: Dict[str, Any],
    reference_frame_contract: Dict[str, Any],
    alignment_result: Dict[str, Any],
) -> Dict[str, Any]:
    needles: List[Dict[str, Any]] = []
    rows = sorted(infos, key=lambda d: int(d.get("stable_id", 10**9)))
    for info in rows:
        sid = int(info.get("stable_id", -1))
        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        if sid < 0 or curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        needles.append(
            {
                "stable_id": int(sid),
                "needle_id": int(info.get("needle_id", -1)),
                "source_stable_id": int(info.get("source_stable_id", -1)),
                "source": "pin_cloud2center_line.extract_centerlines_from_pcd",
                "pin_width": _safe_float(info.get("pin_width", 0.0), 0.0),
                "pin_thickness": _safe_float(info.get("pin_thickness", 0.0), 0.0),
                "slot_match_status": str(info.get("slot_match_status", "matched")),
                "slot_match_cost": _safe_float(info.get("slot_match_cost", float("nan")), float("nan")),
                "slot_center_dist_mm": _safe_float(info.get("slot_center_dist_mm", float("nan")), float("nan")),
                "slot_direction_angle_deg": _safe_float(info.get("slot_direction_angle_deg", float("nan")), float("nan")),
                "slot_length_diff_mm": _safe_float(info.get("slot_length_diff_mm", float("nan")), float("nan")),
                "segments": [curve.astype(float).tolist()],
            }
        )
    numbering_health = _build_numbering_health(slot_ledger, assignment, alignment_result)
    return {
        "schema_version": 1,
        "packed_at": _now_iso(),
        "sample_id": str(sample_id),
        "source_cloud_path": str(source_cloud_path),
        "unit_scale": float(unit_scale),
        "backbone_version": BACKBONE_VERSION,
        "reference_frame_version": str(reference_frame_contract.get("reference_frame_version", "reference_frame_v1")),
        "frame_basis": str(reference_frame_contract.get("frame_basis", "reference_slots_coordinate_system")),
        "numbering_frame_contract": str(reference_frame_contract.get("numbering_frame_contract", "")),
        "reference_slots_version": str(assignment.get("reference_slots_version", "reference_slots_v1")),
        "assignment_version": str(assignment.get("assignment_version", "reference_slot_matching_v1")),
        "assignment_source": str(assignment.get("slots_json_path", "")),
        "alignment_method": str(alignment_result.get("method", ALIGNMENT_METHOD)),
        "alignment_status": str(alignment_result.get("status", "failed")),
        "alignment_attempts": int(alignment_result.get("attempt_count", 0) or 0),
        "alignment_best_attempt_index": int(alignment_result.get("best_attempt_index", -1) or -1),
        "sample_to_reference_transform": np.asarray(alignment_result.get("transform", np.eye(4, dtype=np.float64)), dtype=np.float64).astype(float).tolist(),
        "alignment_meta": dict(alignment_result.get("best_meta", {}) or {}),
        "missing_stable_ids": list(assignment.get("missing_stable_ids", [])),
        "extra_needle_ids": list(assignment.get("extra_needle_ids", [])),
        "ambiguous_matches": list(assignment.get("ambiguous_matches", [])),
        "source_artifacts": {
            "extraction_script": "jiance/pin_cloud2center_line.py",
            "method": str(method),
            "run_id": str(getattr(_DAIMA_CTX, "run_id", "")) if _DAIMA_CTX is not None else "",
        },
        "slots": list(slot_ledger.get("slots", []) or []),
        "numbering_health": numbering_health,
        "needles": needles,
    }


def _write_preview_ply(path: Path, infos: Sequence[Dict[str, Any]]) -> Optional[Path]:
    pts_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    for info in sorted(infos, key=lambda d: int(d.get("stable_id", 10**9))):
        sid = int(info.get("stable_id", -1))
        curve = np.asarray(info.get("curve_global_final", []), dtype=np.float64)
        if sid < 0 or curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        rgb = np.asarray(_palette_rgb(sid), dtype=np.int32).reshape(1, 3)
        cols = np.repeat(rgb, len(curve), axis=0)
        pts_all.append(curve)
        cols_all.append(cols)
    if not pts_all:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ply_xyz_rgb_per_point(str(path), np.vstack(pts_all), np.vstack(cols_all))
    return path


def _resolve_sample_cloud_path(input_root: Path, sample_id: str) -> Path:
    candidates = [
        input_root / f"{sample_id}_pin.ply",
        input_root / f"{sample_id}_pins.ply",
        input_root / f"{sample_id}.ply",
    ]
    for cand in candidates:
        cand = cand.resolve()
        if cand.is_file():
            return cand
    raise FileNotFoundError(str(candidates[0].resolve()))


def _export_one_sample(sample_id: str, input_root: Path, output_root: Path, method: str, write_preview: bool) -> Dict[str, Any]:
    sample_bundle = _resolve_sample_cloud_bundle(input_root, str(sample_id))
    source_cloud_path = Path(str(sample_bundle.get("pins_path", ""))).resolve()

    config = CenterlineConfig()
    infos = extract_centerlines_from_pcd(str(source_cloud_path), method=method, config=config)
    infos = [dict(x) for x in infos if isinstance(x, dict)]

    sample_dir = (output_root / str(sample_id)).resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = (sample_dir / "centerline_candidate_project.json").resolve()
    final_path = (sample_dir / "centerline_final.json").resolve()
    preview_path = (sample_dir / "preview_auto.ply").resolve()
    assignment_report_path = (sample_dir / "slot_assignment_report.json").resolve()
    alignment_report_path = (sample_dir / "alignment_report.json").resolve()

    slots_json_path = _resolve_slots_json(_find_repo_root(Path(__file__)), "")
    _, slots_payload = load_reference_slots(slots_json_path)
    reference_bundle = _resolve_reference_cloud_bundle(_find_repo_root(Path(__file__)), input_root, slots_payload)
    reference_frame_contract = _build_reference_frame_contract(slots_payload, reference_bundle)
    alignment_result = _align_sample_to_reference_frame_with_retries(sample_bundle, reference_bundle, float(getattr(config, "unit_scale", UNIT_SCALE)))

    if str(alignment_result.get("status", "failed")) == "ok":
        infos_aligned = _transform_infos_to_reference_frame(infos, np.asarray(alignment_result.get("transform", np.eye(4, dtype=np.float64)), dtype=np.float64))
        assignment_bundle = _apply_slot_matching_with_relative_rescue(
            infos=infos_aligned,
            slots_json_path=slots_json_path,
            unit_scale=float(getattr(config, "unit_scale", UNIT_SCALE)),
            slots_payload=slots_payload,
        )
        assigned_infos_baseline = list(assignment_bundle.get("assigned_infos_baseline", []) or [])
        assignment_baseline = dict(assignment_bundle.get("assignment_baseline", {}) or {})
        rescue_candidates = list(assignment_bundle.get("rescue_candidates", []) or [])
        rescue_result = dict(assignment_bundle.get("rescue_result", {}) or {})
        assignment = dict(assignment_bundle.get("assignment", {}) or {})
        assigned_infos = list(assignment_bundle.get("assigned_infos", []) or [])
        slot_ledger = _build_slot_ledger_experiment(str(sample_id), slots_payload, assigned_infos, assignment)
        slot_ledger_consistency = _build_slot_ledger_consistency_report(str(sample_id), slots_payload, assignment, slot_ledger)
        drop_one_report = _build_drop_one_invariance_report(
            str(sample_id),
            slots_json_path,
            infos_aligned,
            assignment,
            float(getattr(config, "unit_scale", UNIT_SCALE)),
        )
        observed_order_report = _build_observed_order_invariance_report(
            str(sample_id),
            infos_aligned,
            slots_json_path,
            assignment,
            float(getattr(config, "unit_scale", UNIT_SCALE)),
        )
        matching_comparison_report = _build_matching_comparison_report(
            str(sample_id),
            assignment_baseline,
            assignment,
            rescue_candidates,
        )
        final_status = "ok"
    else:
        infos_aligned = []
        assigned_infos = []
        assignment = {
            "assignment_version": "reference_slot_matching_v1",
            "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
            "slots_json_path": str(slots_json_path),
            "matched_pairs": [],
            "missing_stable_ids": [],
            "extra_needle_ids": [],
            "ambiguous_matches": [],
            "n_slots": int(slots_payload.get("n_slots", len(slots_payload.get("slots", []) or [])) or len(slots_payload.get("slots", []) or [])),
            "n_needles": int(len(infos)),
            "n_matched": 0,
            "n_missing": 0,
            "n_extra": 0,
            "n_ambiguous": 0,
            "status": "alignment_failed",
        }
        slot_ledger = _build_alignment_failed_slot_ledger(str(sample_id), slots_payload)
        slot_ledger_consistency = {
            "schema_version": 1,
            "report_version": "slot_ledger_consistency_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "alignment_failed",
        }
        drop_one_report = {
            "schema_version": 1,
            "report_version": "drop_one_invariance_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "alignment_failed",
        }
        observed_order_report = {
            "schema_version": 1,
            "report_version": "observed_order_invariance_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "alignment_failed",
        }
        rescue_result = {"accepted_pairs": [], "ambiguous": [], "n_accepted": 0, "n_ambiguous": 0}
        rescue_candidates = []
        matching_comparison_report = {
            "schema_version": 1,
            "report_version": "matching_comparison_v1",
            "created_at": _now_iso(),
            "sample_id": str(sample_id),
            "status": "skipped",
            "reason": "alignment_failed",
        }
        final_status = "alignment_failed"

    candidate = _build_candidate_project(
        str(sample_id),
        source_cloud_path,
        sample_dir,
        assigned_infos,
        float(getattr(config, "unit_scale", UNIT_SCALE)),
        reference_frame_contract=reference_frame_contract,
        alignment_result=alignment_result,
    )
    final = _build_final_json(
        str(sample_id),
        source_cloud_path,
        assigned_infos,
        float(getattr(config, "unit_scale", UNIT_SCALE)),
        str(method),
        assignment,
        slot_ledger,
        reference_frame_contract,
        alignment_result,
    )

    _write_json(candidate_path, candidate)
    _write_json(final_path, final)
    _write_json(assignment_report_path, assignment)
    _write_json(alignment_report_path, alignment_result)

    exp_paths = _experiment_output_paths(str(sample_id))
    _write_json(exp_paths["slot_ledger_json"], slot_ledger)
    _write_json(exp_paths["slot_ledger_consistency_json"], slot_ledger_consistency)
    _write_json(exp_paths["drop_one_invariance_json"], drop_one_report)
    _write_json(exp_paths["observed_order_invariance_json"], observed_order_report)
    _write_json(exp_paths["relative_rescue_report_json"], rescue_result)
    _write_json(exp_paths["matching_comparison_report_json"], matching_comparison_report)
    slot_rows_csv = []
    for row in slot_ledger.get("slots", []) or []:
        slot_rows_csv.append(
            {
                "stable_id": int(row.get("stable_id", -1)),
                "row_id": int(row.get("row_id", -1)),
                "u_order": int(row.get("u_order", -1)),
                "v_order": int(row.get("v_order", -1)),
                "slot_label": str(row.get("slot_label", "") or ""),
                "slot_presence_status": str(row.get("slot_presence_status", "") or ""),
                "matched_needle_id": row.get("matched_needle_id", None),
                "source_stable_id": row.get("source_stable_id", None),
                "slot_match_status": str(row.get("slot_match_status", "") or ""),
                "slot_match_cost": row.get("slot_match_cost", None),
                "slot_center_dist_mm": row.get("slot_center_dist_mm", None),
                "slot_direction_angle_deg": row.get("slot_direction_angle_deg", None),
                "slot_length_diff_mm": row.get("slot_length_diff_mm", None),
                "has_geometry": int(bool(row.get("has_geometry", False))),
                "curve_points": int(row.get("curve_points", 0) or 0),
                "segment_count": int(row.get("segment_count", 0) or 0),
                "ambiguous_candidate_needle_ids": ";".join(str(int(x)) for x in (row.get("ambiguous_candidate_needle_ids", []) or [])),
            }
        )
    _write_csv(
        exp_paths["slot_ledger_csv"],
        slot_rows_csv,
        [
            "stable_id",
            "row_id",
            "u_order",
            "v_order",
            "slot_label",
            "slot_presence_status",
            "matched_needle_id",
            "source_stable_id",
            "slot_match_status",
            "slot_match_cost",
            "slot_center_dist_mm",
            "slot_direction_angle_deg",
            "slot_length_diff_mm",
            "has_geometry",
            "curve_points",
            "segment_count",
            "ambiguous_candidate_needle_ids",
        ],
    )
    rescue_rows_csv: List[Dict[str, Any]] = []
    for cand in rescue_candidates:
        rescue_rows_csv.append(
            {
                "needle_id": int(cand.get("needle_id", -1)),
                "candidate_stable_id": int(cand.get("candidate_stable_id", -1)),
                "relative_score": cand.get("relative_score", None),
                "anchor_count_used": int(cand.get("anchor_count_used", 0) or 0),
                "relative_offset_error_mm": cand.get("relative_offset_error_mm", None),
                "absolute_center_dist_mm": cand.get("absolute_center_dist_mm", None),
                "direction_angle_deg": cand.get("direction_angle_deg", None),
                "length_diff_mm": cand.get("length_diff_mm", None),
                "row_consistency": cand.get("row_consistency", None),
                "anchor_stable_ids": ";".join(str(int(x)) for x in (cand.get("anchor_stable_ids", []) or [])),
            }
        )
    _write_csv(
        exp_paths["relative_rescue_candidates_csv"],
        rescue_rows_csv,
        [
            "needle_id",
            "candidate_stable_id",
            "relative_score",
            "anchor_count_used",
            "relative_offset_error_mm",
            "absolute_center_dist_mm",
            "direction_angle_deg",
            "length_diff_mm",
            "row_consistency",
            "anchor_stable_ids",
        ],
    )
    print(f"[INFO] sample_id={sample_id} slot-ledger experiment wrote: {exp_paths['slot_ledger_json']}")
    print(
        f"[INFO] sample_id={sample_id} alignment: {alignment_result.get('status', 'unknown')}"
        f" | attempts={int(alignment_result.get('attempt_count', 0) or 0)}"
    )
    print(
        f"[INFO] sample_id={sample_id} slot-ledger consistency: {slot_ledger_consistency.get('status', 'unknown')}"
        f" | drop-one invariance: {drop_one_report.get('status', 'unknown')}"
        f" | observed-order invariance: {observed_order_report.get('status', 'unknown')}"
    )
    print(
        f"[INFO] sample_id={sample_id} relative rescue: accepted={int(rescue_result.get('n_accepted', 0) or 0)}"
        f" ambiguous={int(rescue_result.get('n_ambiguous', 0) or 0)}"
        f" | baseline_missing={int(assignment_baseline.get('n_missing', assignment.get('n_missing', 0)) or 0) if 'assignment_baseline' in locals() else int(assignment.get('n_missing', 0) or 0)}"
        f" -> final_missing={int(assignment.get('n_missing', 0) or 0)}"
    )
    if final_status == "ok" and str(slot_ledger_consistency.get("status", "")) != "pass":
        final_status = "slot_matching_failed"

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            outputs={
                f"sample_{sample_id}_alignment_report_json": str(alignment_report_path),
                f"sample_{sample_id}_slot_ledger_json": str(exp_paths["slot_ledger_json"]),
                f"sample_{sample_id}_slot_ledger_csv": str(exp_paths["slot_ledger_csv"]),
                f"sample_{sample_id}_slot_ledger_consistency_json": str(exp_paths["slot_ledger_consistency_json"]),
                f"sample_{sample_id}_drop_one_invariance_json": str(exp_paths["drop_one_invariance_json"]),
                f"sample_{sample_id}_observed_order_invariance_json": str(exp_paths["observed_order_invariance_json"]),
                f"sample_{sample_id}_relative_rescue_report_json": str(exp_paths["relative_rescue_report_json"]),
                f"sample_{sample_id}_relative_rescue_candidates_csv": str(exp_paths["relative_rescue_candidates_csv"]),
                f"sample_{sample_id}_matching_comparison_report_json": str(exp_paths["matching_comparison_report_json"]),
            }
        )

    preview_written = None
    if write_preview:
        preview_written = _write_preview_ply(preview_path, assigned_infos)

    stable_ids = sorted([int(n.get("stable_id", -1)) for n in final.get("needles", []) if int(n.get("stable_id", -1)) >= 0])
    return {
        "sample_id": str(sample_id),
        "status": str(final_status),
        "source_cloud_path": str(source_cloud_path),
        "sample_bundle": {
            "pins_path": str(sample_bundle.get("pins_path", "") or ""),
            "base_path": str(sample_bundle.get("base_path", "") or ""),
            "overall_path": str(sample_bundle.get("overall_path", "") or ""),
            "overall_source": str(sample_bundle.get("overall_source", "") or ""),
        },
        "reference_bundle": {
            "sample_id": str(reference_bundle.get("sample_id", "") or ""),
            "pins_path": str(reference_bundle.get("pins_path", "") or ""),
            "base_path": str(reference_bundle.get("base_path", "") or ""),
            "overall_path": str(reference_bundle.get("overall_path", "") or ""),
            "overall_source": str(reference_bundle.get("overall_source", "") or ""),
        },
        "candidate_project_path": str(candidate_path),
        "final_json_path": str(final_path),
        "assignment_report_path": str(assignment_report_path),
        "alignment_report_path": str(alignment_report_path),
        "preview_ply_path": str(preview_written) if preview_written is not None else "",
        "n_needles": int(len(final.get("needles", []))),
        "stable_ids": stable_ids,
        "missing_stable_ids": list(final.get("missing_stable_ids", [])),
        "extra_needle_ids": list(final.get("extra_needle_ids", [])),
        "n_ambiguous": int(len(final.get("ambiguous_matches", []))),
        "alignment_status": str(alignment_result.get("status", "")),
        "alignment_attempts": int(alignment_result.get("attempt_count", 0) or 0),
        "matched_slot_ids": list(final.get("numbering_health", {}).get("matched_slot_ids", []) or []),
        "ambiguous_slot_ids": list(final.get("numbering_health", {}).get("ambiguous_slot_ids", []) or []),
        "unresolved_slot_ids": list(final.get("numbering_health", {}).get("unresolved_slot_ids", []) or []),
        "experiment_artifacts": {k: str(v) for k, v in exp_paths.items()},
        "slot_ledger_status": str(slot_ledger_consistency.get("status", "unknown")),
        "drop_one_invariance_status": str(drop_one_report.get("status", "unknown")),
        "observed_order_invariance_status": str(observed_order_report.get("status", "unknown")),
        "relative_rescue_accepted": int(rescue_result.get("n_accepted", 0) or 0),
        "relative_rescue_ambiguous": int(rescue_result.get("n_ambiguous", 0) or 0),
    }


def main() -> int:
    repo_root = _find_repo_root(Path(__file__))

    parser = argparse.ArgumentParser(description="Export reference_centerline_v1 artifacts from existing extractor.")
    parser.add_argument("--input_root", default="", help="Optional input root. Defaults to point_cloud/pingtai/bad.")
    parser.add_argument("--input_partition", default=INPUT_PARTITION, choices=["bad", "normal"], help="When input_root is tolerance_anomaly_GT root, choose which partition subdir to read.")
    parser.add_argument("--output_root", default="", help="Optional output root. Defaults to annotations/reference_centerlines_v1.")
    parser.add_argument("--sample_ids", nargs="*", default=None, help="Optional sample ids. Defaults to in-file SAMPLE_IDS.")
    parser.add_argument("--method", default=CENTERLINE_METHOD, help="Centerline extraction method.")
    parser.add_argument("--no_preview", action="store_true", help="Disable preview_auto.ply export.")
    parser.add_argument("--cross_sample_consistency", action="store_true", help="Write cross-sample slot consistency reports.")
    parser.add_argument("--order_invariance", action="store_true", help="Reserved compatibility flag; order invariance runs by default in stage 2.")
    args = parser.parse_args()

    input_root = _resolve_partitioned_input_root(_resolve_input_root(repo_root, str(args.input_root or "")), str(args.input_partition or INPUT_PARTITION))
    output_root = _resolve_output_root(repo_root, str(args.output_root or ""))
    sample_ids = _resolve_sample_ids(input_root, args.sample_ids)
    write_preview = not bool(args.no_preview) and bool(WRITE_PREVIEW_PLY)
    manifest_path = (output_root.parent / "reference_centerlines_v1_manifest.json").resolve()
    batch_paths = _batch_output_paths(output_root)
    slots_json_path = _resolve_slots_json(repo_root, "")
    _, slots_payload = load_reference_slots(slots_json_path)
    reference_bundle = _resolve_reference_cloud_bundle(repo_root, input_root, slots_payload)
    reference_frame_contract = _build_reference_frame_contract(slots_payload, reference_bundle)
    reference_frame_payload = _write_reference_frame_artifact(batch_paths["reference_frame_json"], reference_frame_contract)

    if hasattr(_pcl, "_DAIMA_CTX"):
        _pcl._DAIMA_CTX = _DAIMA_CTX

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "input_root": str(input_root),
                "sample_clouds": [str(_resolve_sample_cloud_path(input_root, str(sid))) for sid in sample_ids],
            },
            outputs={
                "output_root": str(output_root),
                "manifest_json": str(manifest_path),
                "reference_frame_json": str(batch_paths["reference_frame_json"]),
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "sample_ids": list(sample_ids),
                "method": str(args.method),
                "write_preview": bool(write_preview),
                "cross_sample_consistency": bool(args.cross_sample_consistency),
                "backbone_version": BACKBONE_VERSION,
                "candidate_ctrl_points": int(DEFAULT_GUI_CTRL_POINTS),
                "candidate_smooth_strength": float(DEFAULT_GUI_SMOOTH),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        try:
            ent = _export_one_sample(str(sample_id), input_root, output_root, str(args.method), write_preview)
            entries.append(ent)
            status_tag = "OK" if str(ent.get("status", "")) == "ok" else "WARN"
            print(
                f"[{status_tag}] sample_id={sample_id} status={ent.get('status', '')}"
                f" n_needles={ent['n_needles']} out={ent['final_json_path']}"
            )
        except Exception as exc:
            msg = str(exc)
            errors.append({"sample_id": str(sample_id), "error": msg})
            print(f"[ERROR] sample_id={sample_id} err={msg}")

    batch_alignment_summary = _build_batch_alignment_summary(entries)
    _write_json(batch_paths["batch_alignment_summary_json"], batch_alignment_summary)
    _write_csv(
        batch_paths["batch_alignment_summary_csv"],
        list(batch_alignment_summary.get("rows", []) or []),
        [
            "sample_id",
            "status",
            "alignment_status",
            "alignment_attempts",
            "sample_overall_source",
            "reference_overall_source",
            "n_needles",
            "n_ambiguous",
            "slot_ledger_status",
            "drop_one_invariance_status",
            "observed_order_invariance_status",
            "missing_stable_ids",
            "extra_needle_ids",
            "matched_slot_ids",
            "ambiguous_slot_ids",
            "unresolved_slot_ids",
        ],
    )

    cross_sample_report = _build_cross_sample_slot_consistency_report(entries)
    _write_json(batch_paths["cross_sample_slot_consistency_json"], cross_sample_report)
    _write_csv(
        batch_paths["cross_sample_slot_consistency_csv"],
        list(cross_sample_report.get("matrix", []) or []),
        [
            "sample_id",
            "alignment_status",
            "stable_id",
            "slot_presence_status",
            "matched_needle_id",
            "source_stable_id",
            "slot_match_cost",
        ],
    )

    batch_gate_summary = _build_batch_gate_summary(entries)
    _write_json(batch_paths["batch_gate_summary_json"], batch_gate_summary)

    manifest = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "run_id": str(getattr(_DAIMA_CTX, "run_id", "")) if _DAIMA_CTX is not None else "",
        "version": BACKBONE_VERSION,
        "reference_frame": reference_frame_payload,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "sample_ids": list(sample_ids),
        "entries": entries,
        "errors": errors,
        "batch_alignment_summary_path": str(batch_paths["batch_alignment_summary_json"]),
        "cross_sample_slot_consistency_path": str(batch_paths["cross_sample_slot_consistency_json"]),
        "batch_gate_summary_path": str(batch_paths["batch_gate_summary_json"]),
    }
    _write_json(manifest_path, manifest)

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            outputs={
                "manifest_json": str(manifest_path),
                "batch_alignment_summary_json": str(batch_paths["batch_alignment_summary_json"]),
                "cross_sample_slot_consistency_json": str(batch_paths["cross_sample_slot_consistency_json"]),
                "batch_gate_summary_json": str(batch_paths["batch_gate_summary_json"]),
            }
        )

    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Wrote batch summary: {batch_paths['batch_alignment_summary_json']}")
    print(f"[INFO] Wrote cross-sample consistency: {batch_paths['cross_sample_slot_consistency_json']}")
    print(f"[INFO] Batch gate passed={bool(batch_gate_summary.get('passed', False))}")
    print(f"[INFO] Success={len(entries)} Error={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
