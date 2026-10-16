# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from authoritative_gt_contract_v1 import (
    authoritative_gt_record_from_manifest_row,
    canonical_authoritative_gt_base_path,
    canonical_authoritative_gt_pin_path,
    legacy_authoritative_gt_pin_path,
    make_sample_uid,
    resolve_authoritative_gt_record,
    resolve_authoritative_gt_pair,
    sample_uid_from_manifest_row,
    source_sample_id_from_manifest_row,
)

try:
    from .detect_tolerance_interactive_1_11 import (
        PIN_UNIT_SCALE,
        TOLERANCE_JSON_PATH,
        _centerline_extract_config,
        _safe_float,
        compute_board_transform,
        extract_centerlines_from_pcd,
        load_centerline_tolerance,
        match_needles_by_position,
        resolve_tolerance_reference_paths,
        transform_needle_infos,
    )
except ImportError:
    from detect_tolerance_interactive_1_11 import (
        PIN_UNIT_SCALE,
        TOLERANCE_JSON_PATH,
        _centerline_extract_config,
        _safe_float,
        compute_board_transform,
        extract_centerlines_from_pcd,
        load_centerline_tolerance,
        match_needles_by_position,
        resolve_tolerance_reference_paths,
        transform_needle_infos,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
GT_ROOT_DEFAULT = REPO_ROOT / "point_cloud" / "tolerance_anomaly_GT"
GT_LABEL_FIELD = "scalar_A"
GT_LABEL_MATCH_MAX_DIST_MM = float(str(os.getenv("DAIMA_GT_LABEL_MATCH_MAX_DIST_MM", "0.5") or "0.5"))
LOGIC_GT_SAMPLE_CACHE_SCHEMA = "logic_gt_native_sample_cache_v2"
LOGIC_GT_SAMPLE_CACHE_ROOT = REPO_ROOT / "debug_out" / "cache" / "logic_gt_native_sample_v2"
_PLY_SCALAR_TYPE_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _ply_dtype_from_token(token: str, fmt: str) -> str:
    base = _PLY_SCALAR_TYPE_TO_DTYPE.get(str(token).strip().lower())
    if not base:
        raise ValueError(f"unsupported PLY property type: {token}")
    if fmt == "ascii":
        return base
    if base[0] in {"i", "u", "f"} and len(base) == 2 and base[1].isdigit():
        endian = "<" if fmt == "binary_little_endian" else ">"
        return f"{endian}{base}"
    return base


def _parse_ply_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        format_name = ""
        vertex_count: Optional[int] = None
        vertex_props: List[Tuple[str, str]] = []
        current_element = ""
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF in PLY header: {path}")
            text = line.decode("ascii", errors="replace").rstrip("\n")
            header_lines.append(text)
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 3:
                format_name = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and len(parts) >= 3:
                if parts[1] == "list":
                    if current_element == "vertex":
                        raise ValueError(f"vertex list properties are unsupported: {path}")
                    continue
                if current_element == "vertex":
                    vertex_props.append((parts[1], parts[2]))
            elif parts[0] == "end_header":
                break
        data_start = f.tell()
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {format_name} | {path}")
    if vertex_count is None:
        raise ValueError(f"missing vertex element in PLY header: {path}")
    if not vertex_props:
        raise ValueError(f"missing vertex properties in PLY header: {path}")
    return {
        "format": format_name,
        "vertex_count": int(vertex_count),
        "vertex_props": list(vertex_props),
        "header_lines": list(header_lines),
        "data_start": int(data_start),
    }


def _vertex_dtype(header: Dict[str, Any]) -> np.dtype:
    fields = []
    for type_token, name in header["vertex_props"]:
        fields.append((str(name), _ply_dtype_from_token(str(type_token), str(header["format"]))))
    return np.dtype(fields, align=False)


def _numeric_matrix_to_structured(matrix: np.ndarray, dtype: np.dtype) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    out = np.zeros(int(matrix.shape[0]), dtype=dtype)
    for idx, name in enumerate(dtype.names or []):
        out[name] = matrix[:, idx]
    return out


def _load_vertex_table(path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    header = _parse_ply_header(path)
    dtype = _vertex_dtype(header)
    n_rows = int(header["vertex_count"])
    n_cols = len(dtype.names or [])
    if header["format"] == "ascii":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in header["header_lines"]:
                f.readline()
            if n_rows <= 0:
                data = np.zeros(0, dtype=dtype)
            else:
                matrix = np.loadtxt(f, dtype=np.float64, ndmin=2)
                if matrix.shape[0] != n_rows:
                    raise ValueError(f"vertex count mismatch in ASCII PLY: {path}")
                if matrix.shape[1] != n_cols:
                    raise ValueError(f"vertex column count mismatch in ASCII PLY: {path}")
                data = _numeric_matrix_to_structured(matrix, dtype)
        return header, data
    with path.open("rb") as f:
        f.seek(int(header["data_start"]))
        data = np.fromfile(f, dtype=dtype, count=n_rows)
    if int(data.shape[0]) != n_rows:
        raise ValueError(f"vertex count mismatch in binary PLY: {path}")
    return header, data


def _require_fields(data: np.ndarray, names: Sequence[str], context: str) -> None:
    available = set(data.dtype.names or [])
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")


def _xyz_f32(data: np.ndarray) -> np.ndarray:
    _require_fields(data, ["x", "y", "z"], "PLY vertex table")
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32, copy=False)
    return np.ascontiguousarray(xyz)


def _sample_id_from_manifest_row(manifest_row: Mapping[str, Any]) -> str:
    sample_id = source_sample_id_from_manifest_row(manifest_row)
    if sample_id:
        return str(sample_id)
    for key in ("base_ply", "pin_ply", "gt_ply"):
        value = str(manifest_row.get(key, "")).strip()
        if not value:
            continue
        stem = Path(value).stem
        token = str(stem).split("_", 1)[0].strip()
        if token:
            return token
    raise ValueError(f"manifest row missing sample_id and cannot infer from paths: {dict(manifest_row)}")


def authoritative_gt_paths(
    gt_root: Path,
    sample_id: str,
    *,
    gt_partition: str = "bad",
    gt_ply: str = "",
    gt_base_ply: str = "",
) -> Tuple[Path, Path]:
    sample_id_n = str(sample_id or "").strip()
    if not sample_id_n and not str(gt_ply or "").strip():
        raise ValueError("sample_id or gt_ply is required to resolve authoritative GT paths")
    return resolve_authoritative_gt_pair(
        sample_id_n,
        Path(gt_root).resolve(),
        allow_legacy=True,
        gt_partition=str(gt_partition or "bad"),
        gt_ply=str(gt_ply or ""),
        gt_base_ply=str(gt_base_ply or ""),
    )


def _norm_path_text(path_like: str) -> str:
    p = Path(str(path_like).strip())
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return os.path.normcase(str(p))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_signature(path_like: Any) -> Dict[str, Any]:
    path = Path(str(path_like or "")).resolve()
    sig: Dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
    except FileNotFoundError:
        sig["missing"] = True
        return sig
    sig["mtime_ns"] = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    sig["size"] = int(stat.st_size)
    return sig


def _logic_gt_cache_reference_signatures() -> Dict[str, Any]:
    tol_json = load_centerline_tolerance(TOLERANCE_JSON_PATH)
    tol_base_path, tol_pins_path = resolve_tolerance_reference_paths(tol_json or {})
    return {
        "schema_version": LOGIC_GT_SAMPLE_CACHE_SCHEMA,
        "loader_script": _file_signature(Path(__file__)),
        "extractor_script": _file_signature(Path(compute_board_transform.__code__.co_filename)),
        "tolerance_json": _file_signature(Path(TOLERANCE_JSON_PATH)),
        "tolerance_base": _file_signature(Path(tol_base_path)) if tol_base_path else {},
        "tolerance_pins": _file_signature(Path(tol_pins_path)) if tol_pins_path else {},
    }


def _logic_gt_cache_sample_file_signatures(
    *,
    authoritative_gt_ply: str,
    authoritative_gt_base_ply: str,
) -> Dict[str, Any]:
    return {
        "gt_pin": _file_signature(Path(authoritative_gt_ply)) if str(authoritative_gt_ply).strip() else {},
        "gt_base": _file_signature(Path(authoritative_gt_base_ply)) if str(authoritative_gt_base_ply).strip() else {},
    }


def _logic_gt_cache_key(
    *,
    gt_root: Path,
    sample_id: str,
    gt_partition: str,
    gt_sample_key: str,
    authoritative_gt_ply: str,
    authoritative_gt_base_ply: str,
    reference_signatures: Mapping[str, Any],
    sample_file_signatures: Mapping[str, Any],
) -> str:
    payload = {
        "gt_root": _norm_path_text(str(gt_root)),
        "sample_id": str(sample_id),
        "gt_partition": str(gt_partition),
        "gt_sample_key": str(gt_sample_key),
        "authoritative_gt_ply": _norm_path_text(authoritative_gt_ply),
        "authoritative_gt_base_ply": _norm_path_text(authoritative_gt_base_ply),
        "reference_signatures": dict(reference_signatures),
        "sample_file_signatures": dict(sample_file_signatures),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _logic_gt_cache_path(cache_root: Path, *, sample_id: str, gt_partition: str, cache_key: str) -> Path:
    safe_sample_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(sample_id))
    safe_partition = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(gt_partition or "unknown"))
    return (cache_root / f"{safe_partition}_{safe_sample_id}_{cache_key[:16]}.json").resolve()


def _slot_rows_from_map(slot_map: Mapping[Tuple[str, int], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(row)
        for _key, row in sorted(
            ((key, dict(value)) for key, value in slot_map.items()),
            key=lambda item: (str(item[0][0]), int(item[0][1])),
        )
    ]


def _slot_map_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    slot_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(row)
        sample_id = str(row_dict.get("sample_uid", "") or row_dict.get("sample_id", "") or "").strip()
        stable_id = int(_safe_float(row_dict.get("stable_id", -1), default=-1))
        if not sample_id or stable_id < 0:
            continue
        slot_map[(sample_id, stable_id)] = row_dict
    return slot_map


def _read_logic_gt_sample_cache(cache_path: Path, cache_key: str) -> Optional[Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[str, Any]]]:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(payload.get("schema_version", "") or "") != LOGIC_GT_SAMPLE_CACHE_SCHEMA:
        return None
    if str(payload.get("cache_key", "") or "") != str(cache_key):
        return None
    slot_rows = list(payload.get("slot_rows", []) or [])
    summary = dict(payload.get("summary", {}) or {})
    return _slot_map_from_rows(slot_rows), summary


def _write_logic_gt_sample_cache(
    cache_path: Path,
    *,
    cache_key: str,
    sample_row: Mapping[str, Any],
    slot_map: Mapping[Tuple[str, int], Mapping[str, Any]],
    summary: Mapping[str, Any],
    reference_signatures: Mapping[str, Any],
    sample_file_signatures: Mapping[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LOGIC_GT_SAMPLE_CACHE_SCHEMA,
        "created_at": _now_iso(),
        "cache_key": str(cache_key),
        "sample_row": dict(sample_row),
        "reference_signatures": dict(reference_signatures),
        "sample_file_signatures": dict(sample_file_signatures),
        "slot_rows": _slot_rows_from_map(slot_map),
        "summary": dict(summary),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)


def validate_manifest_row_against_authoritative_gt(
    manifest_row: Mapping[str, Any],
    gt_root: Path,
) -> Dict[str, Any]:
    resolved = authoritative_gt_record_from_manifest_row(
        manifest_row,
        gt_root,
        allow_legacy=True,
        default_partition="bad",
    )
    sample_id = str(resolved.get("sample_id", "") or _sample_id_from_manifest_row(manifest_row))
    gt_partition = str(resolved.get("gt_partition", "bad") or "bad")
    sample_uid = sample_uid_from_manifest_row(
        {
            **dict(manifest_row or {}),
            "gt_partition": str(gt_partition),
            "source_sample_id": str(sample_id),
        },
        default_partition=str(gt_partition),
    )
    authoritative_gt_ply = Path(str(resolved.get("pin_path", ""))).resolve()
    authoritative_gt_base_ply = Path(str(resolved.get("base_path", ""))).resolve()
    canonical_gt_ply = authoritative_gt_ply
    canonical_gt_base_ply = authoritative_gt_base_ply
    legacy_gt_ply = legacy_authoritative_gt_pin_path(sample_id, gt_root, gt_partition=gt_partition)
    manifest_gt_ply = str(manifest_row.get("gt_ply", "")).strip()
    del manifest_gt_ply
    return {
        "sample_uid": str(sample_uid),
        "sample_id": str(sample_id),
        "source_sample_id": str(sample_id),
        "gt_partition": str(gt_partition),
        "gt_sample_key": f"{gt_partition}:{sample_id}",
        "authoritative_gt_ply": str(canonical_gt_ply),
        "authoritative_gt_base_ply": str(canonical_gt_base_ply),
        "authoritative_gt_resolved_ply": str(authoritative_gt_ply),
        "authoritative_gt_resolved_base_ply": str(authoritative_gt_base_ply),
        "authoritative_gt_pin_variant": str(resolved.get("pin_variant", "")),
        "authoritative_gt_legacy_ply": str(legacy_gt_ply),
    }


def validate_manifest_rows_against_authoritative_gt(
    manifest_rows: Iterable[Mapping[str, Any]],
    gt_root: Path,
) -> List[Dict[str, Any]]:
    validated_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        row_copy = dict(row)
        row_copy.update(validate_manifest_row_against_authoritative_gt(row, gt_root))
        validated_rows.append(row_copy)
    return validated_rows


def _normalize_gt_sample_record(
    gt_root: Path,
    sample_record: Any,
) -> Dict[str, Any]:
    if isinstance(sample_record, Mapping):
        row = dict(sample_record)
        if str(row.get("authoritative_gt_ply", "")).strip() and str(row.get("authoritative_gt_base_ply", "")).strip():
            sample_id = str(row.get("sample_id", "") or _sample_id_from_manifest_row(row))
            gt_partition = str(row.get("gt_partition", "bad") or "bad")
            gt_sample_key = str(row.get("gt_sample_key", f"{gt_partition}:{sample_id}") or f"{gt_partition}:{sample_id}")
            sample_uid = str(row.get("sample_uid", "") or make_sample_uid(row.get("source_sample_id", sample_id), gt_partition))
            out = dict(row)
            out.setdefault("sample_uid", str(sample_uid))
            out.setdefault("sample_id", str(sample_id))
            out.setdefault("source_sample_id", str(row.get("source_sample_id", sample_id) or sample_id))
            out.setdefault("gt_partition", str(gt_partition))
            out.setdefault("gt_sample_key", str(gt_sample_key))
            out.setdefault("authoritative_gt_resolved_ply", str(row.get("authoritative_gt_ply", "")))
            out.setdefault("authoritative_gt_resolved_base_ply", str(row.get("authoritative_gt_base_ply", "")))
            return out
        validated = validate_manifest_row_against_authoritative_gt(row, gt_root)
        out = dict(row)
        out.update(validated)
        return out

    record = resolve_authoritative_gt_record(sample_record, gt_root, allow_legacy=True, gt_partition="bad")
    sample_id = str(record.get("sample_id", "") or "")
    gt_partition = str(record.get("gt_partition", "bad") or "bad")
    gt_sample_key = str(record.get("gt_sample_key", f"{gt_partition}:{sample_id}") or f"{gt_partition}:{sample_id}")
    return {
        "sample_uid": str(make_sample_uid(sample_id, gt_partition)),
        "sample_id": str(sample_id),
        "source_sample_id": str(sample_id),
        "gt_partition": str(gt_partition),
        "gt_sample_key": str(gt_sample_key),
        "authoritative_gt_ply": str(record.get("pin_path", "")),
        "authoritative_gt_base_ply": str(record.get("base_path", "")),
        "authoritative_gt_resolved_ply": str(record.get("pin_path", "")),
        "authoritative_gt_resolved_base_ply": str(record.get("base_path", "")),
        "authoritative_gt_pin_variant": str(record.get("pin_variant", "")),
    }


def _coord_key(bits_row: np.ndarray) -> Tuple[int, int, int]:
    return int(bits_row[0]), int(bits_row[1]), int(bits_row[2])


def _label_mask_from_gt(gt_data: np.ndarray) -> Tuple[np.ndarray, str]:
    field_by_lower = {str(name).lower(): str(name) for name in (gt_data.dtype.names or [])}
    resolved = field_by_lower.get(str(GT_LABEL_FIELD).lower(), "")
    if not resolved:
        return np.zeros(len(gt_data), dtype=bool), ""
    values = np.asarray(gt_data[resolved], dtype=np.float32)
    return np.isfinite(values) & (values != 0), resolved


def _transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    pts_aligned = (np.asarray(T, dtype=np.float64) @ pts_h.T).T[:, :3]
    return np.asarray(pts_aligned, dtype=np.float32)


def _build_reference_infos() -> Tuple[Dict[str, Any], Path, Path, Dict[str, Any]]:
    tol_json = load_centerline_tolerance(TOLERANCE_JSON_PATH)
    tol_base_path, tol_pins_path = resolve_tolerance_reference_paths(tol_json or {})
    if tol_json is None or tol_base_path is None or tol_pins_path is None:
        raise RuntimeError(f"tolerance model / reference files not available: {TOLERANCE_JSON_PATH}")

    centerline_cfg = _centerline_extract_config()
    tol_unit_scale = _safe_float(tol_json.get("unit_scale", PIN_UNIT_SCALE), default=PIN_UNIT_SCALE)
    match_radius_mm = _safe_float(tol_json.get("match_radius_mm", 0.0), default=0.0)
    r_match_u = None
    if np.isfinite(match_radius_mm) and match_radius_mm > 0 and np.isfinite(tol_unit_scale) and tol_unit_scale > 1e-12:
        r_match_u = float(match_radius_mm) * float(tol_unit_scale)

    ref_pins_infos = extract_centerlines_from_pcd(str(tol_pins_path), config=centerline_cfg)
    ref_by_sid: Dict[int, Dict[str, Any]] = {}
    for info in ref_pins_infos:
        try:
            sid = int(info.get("stable_id", -1))
        except Exception:
            continue
        if sid >= 0 and sid not in ref_by_sid:
            ref_by_sid[sid] = info

    infos_ref: List[Dict[str, Any]] = []
    for needle in (tol_json.get("needles", []) if isinstance(tol_json, dict) else []):
        try:
            sid = int(needle.get("stable_id", -1))
        except Exception:
            continue
        nominal = np.asarray(needle.get("nominal_curve_u", []), dtype=np.float64)
        if sid < 0 or nominal.ndim != 2 or nominal.shape[1] != 3 or len(nominal) < 2:
            continue
        info_ref = {
            "stable_id": int(sid),
            "curve_global_final": nominal,
            "center_point": np.mean(nominal, axis=0),
            "base_end_idx_local": 0,
            "tol_model_status": str(needle.get("status", "")),
            "tol_tangent_model_status": str(needle.get("tol_tangent_model_status", needle.get("status", ""))),
            "tol_unit_scale": float(tol_unit_scale),
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
    ref_pack = {
        "infos_ref": infos_ref,
        "centerline_cfg": centerline_cfg,
        "tol_unit_scale": float(tol_unit_scale),
        "r_match_u": r_match_u,
        "match_radius_mm": float(match_radius_mm),
        "tolerance_json_path": str(TOLERANCE_JSON_PATH),
        "reference_sample_id": str(tol_json.get("reference_sample_id", "")),
    }
    return ref_pack, Path(tol_base_path), Path(tol_pins_path), dict(tol_json or {})


def _defect_point_owner_map(infos_def: Iterable[Mapping[str, Any]]) -> Tuple[Dict[Tuple[int, int, int], int], int]:
    owner_by_coord: Dict[Tuple[int, int, int], int] = {}
    conflict_count = 0
    for defect_id, info in enumerate(infos_def):
        pts = np.asarray(info.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) <= 0:
            continue
        bits = np.ascontiguousarray(pts).view(np.int32).reshape(-1, 3)
        for bits_row in bits:
            key = _coord_key(bits_row)
            prev = owner_by_coord.get(key)
            if prev is None:
                owner_by_coord[key] = int(defect_id)
            elif int(prev) != int(defect_id):
                conflict_count += 1
    return owner_by_coord, int(conflict_count)


def _defect_point_cloud(infos_def: Iterable[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    point_blocks: List[np.ndarray] = []
    owner_blocks: List[np.ndarray] = []
    for defect_id, info in enumerate(infos_def):
        pts = np.asarray(info.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) <= 0:
            continue
        point_blocks.append(pts)
        owner_blocks.append(np.full(len(pts), int(defect_id), dtype=np.int32))
    if not point_blocks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.concatenate(point_blocks, axis=0), np.concatenate(owner_blocks, axis=0)


def _assign_gt_points_to_defects(
    gt_xyz: np.ndarray,
    infos_def: Iterable[Mapping[str, Any]],
    *,
    tol_unit_scale: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32).reshape(-1, 3)
    owner_ids = np.full(len(gt_xyz), -1, dtype=np.int32)
    assignment_source = np.full(len(gt_xyz), -1, dtype=np.int8)
    exact_unmatched_indices: List[int] = []

    owner_by_coord, coord_owner_conflicts = _defect_point_owner_map(infos_def)
    gt_bits = np.ascontiguousarray(gt_xyz).view(np.int32).reshape(-1, 3)
    for idx, bits_row in enumerate(gt_bits):
        defect_id = owner_by_coord.get(_coord_key(bits_row))
        if defect_id is None:
            exact_unmatched_indices.append(int(idx))
            continue
        owner_ids[idx] = int(defect_id)
        assignment_source[idx] = 0

    fallback_match_count = 0
    fallback_dist_mm_max = 0.0
    fallback_dist_mm_p95 = 0.0
    unresolved_points = 0
    if exact_unmatched_indices:
        defect_points, defect_owner_ids = _defect_point_cloud(infos_def)
        if len(defect_points) > 0:
            tree = cKDTree(np.asarray(defect_points, dtype=np.float64))
            threshold_u = float(tol_unit_scale) * float(GT_LABEL_MATCH_MAX_DIST_MM)
            query_points = np.asarray(gt_xyz[np.asarray(exact_unmatched_indices, dtype=np.int64)], dtype=np.float64)
            dists, nn_indices = tree.query(query_points, k=1, workers=-1)
            accepted_dists_mm: List[float] = []
            for gt_idx, dist_u, nn_idx in zip(exact_unmatched_indices, dists.tolist(), nn_indices.tolist()):
                if float(dist_u) <= float(threshold_u):
                    owner_ids[int(gt_idx)] = int(defect_owner_ids[int(nn_idx)])
                    assignment_source[int(gt_idx)] = 1
                    fallback_match_count += 1
                    accepted_dists_mm.append(float(dist_u) / float(tol_unit_scale))
                else:
                    unresolved_points += 1
        else:
            unresolved_points = int(len(exact_unmatched_indices))

        if fallback_match_count > 0:
            accepted_dists = [
                float(dist_u) / float(tol_unit_scale)
                for dist_u, src in zip(dists.tolist(), assignment_source[np.asarray(exact_unmatched_indices, dtype=np.int64)].tolist())
                if int(src) == 1
            ]
            if accepted_dists:
                accepted_np = np.asarray(accepted_dists, dtype=np.float64)
                fallback_dist_mm_max = float(np.max(accepted_np))
                fallback_dist_mm_p95 = float(np.quantile(accepted_np, 0.95))

    summary = {
        "coord_owner_conflicts": int(coord_owner_conflicts),
        "exact_unmatched_points": int(len(exact_unmatched_indices)),
        "fallback_match_count": int(fallback_match_count),
        "fallback_match_max_dist_mm": float(fallback_dist_mm_max),
        "fallback_match_p95_dist_mm": float(fallback_dist_mm_p95),
        "fallback_match_threshold_mm": float(GT_LABEL_MATCH_MAX_DIST_MM),
        "unmatched_points": int(unresolved_points),
    }
    return owner_ids, assignment_source, summary


def _sample_gt_authoritative(
    *,
    sample_id: str,
    sample_uid: str = "",
    gt_root: Path,
    ref_pack: Mapping[str, Any],
    tol_base_path: Path,
    tol_pins_path: Path,
    gt_partition: str = "bad",
    gt_ply: str = "",
    gt_base_ply: str = "",
    gt_sample_key: str = "",
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[str, Any], Dict[Tuple[str, int], Dict[str, Any]]]:
    gt_record = resolve_authoritative_gt_record(
        sample_id,
        gt_root,
        allow_legacy=True,
        gt_partition=str(gt_partition or "bad"),
        gt_ply=str(gt_ply or ""),
        gt_base_ply=str(gt_base_ply or ""),
    )
    gt_partition = str(gt_record.get("gt_partition", "bad") or "bad")
    gt_sample_key = str(gt_sample_key or gt_record.get("gt_sample_key", f"{gt_partition}:{sample_id}") or f"{gt_partition}:{sample_id}")
    sample_uid_n = str(sample_uid or make_sample_uid(sample_id, gt_partition)).strip()
    pin_gt_path = Path(str(gt_record.get("pin_path", ""))).resolve()
    base_gt_path = Path(str(gt_record.get("base_path", ""))).resolve()
    canonical_pin_gt_path = pin_gt_path
    canonical_base_gt_path = base_gt_path
    if not pin_gt_path.is_file():
        raise FileNotFoundError(f"missing pin GT PLY: {pin_gt_path}")
    if not base_gt_path.is_file():
        raise FileNotFoundError(f"missing base GT PLY: {base_gt_path}")

    _gt_header, gt_data = _load_vertex_table(pin_gt_path)
    _require_fields(gt_data, ["x", "y", "z"], "GT pin PLY")
    gt_xyz = _xyz_f32(gt_data)
    label_mask, label_field_name = _label_mask_from_gt(gt_data)

    infos_ref = list(ref_pack["infos_ref"])
    centerline_cfg = ref_pack["centerline_cfg"]
    r_match_u = ref_pack["r_match_u"]

    T, reg_meta, _, _ = compute_board_transform(
        str(tol_base_path),
        str(base_gt_path),
        str(tol_pins_path),
        str(pin_gt_path),
        return_clean_clouds=True,
    )
    infos_def = extract_centerlines_from_pcd(str(pin_gt_path), config=centerline_cfg)
    infos_def_aligned = transform_needle_infos(infos_def, T)
    pairing = match_needles_by_position(infos_ref, infos_def_aligned, r_match=r_match_u)
    owner_ids, assignment_source, assignment_summary = _assign_gt_points_to_defects(
        gt_xyz,
        infos_def,
        tol_unit_scale=float(ref_pack["tol_unit_scale"]),
    )
    gt_xyz_aligned = _transform_points(gt_xyz, np.asarray(T, dtype=np.float64))
    labeled_counts_by_defect: Counter[int] = Counter()
    total_counts_by_defect: Counter[int] = Counter()
    assignment_counts_by_defect: Dict[int, Dict[str, int]] = {}
    for idx, defect_id in enumerate(owner_ids.tolist()):
        if int(defect_id) < 0:
            continue
        defect_i = int(defect_id)
        total_counts_by_defect[defect_i] += 1
        if bool(label_mask[idx]):
            labeled_counts_by_defect[defect_i] += 1
        src_key = "exact" if int(assignment_source[idx]) == 0 else "fallback"
        slot_stat = assignment_counts_by_defect.setdefault(defect_i, {"exact": 0, "fallback": 0})
        slot_stat[src_key] = int(slot_stat.get(src_key, 0)) + 1

    paired_defect_ids = {
        int(pair.get("defect_id", -1))
        for pair in pairing.get("pairs", [])
        if int(pair.get("defect_id", -1)) >= 0
    }
    unpaired_labeled_defect_ids = sorted(
        defect_id for defect_id, count in labeled_counts_by_defect.items() if int(count) > 0 and int(defect_id) not in paired_defect_ids
    )

    pair_by_ref_index: Dict[int, Dict[str, Any]] = {}
    for pair in pairing.get("pairs", []):
        ref_index = int(pair.get("normal_id", -1))
        defect_index = int(pair.get("defect_id", -1))
        if ref_index < 0 or ref_index >= len(infos_ref) or defect_index < 0 or defect_index >= len(infos_def):
            continue
        pair_by_ref_index[int(ref_index)] = {
            "pair": dict(pair),
            "defect_id": int(defect_index),
            "stable_id": int(infos_ref[ref_index].get("stable_id", defect_index)),
        }

    unresolved_labeled_points = int(
        np.count_nonzero(np.asarray(label_mask, dtype=bool) & np.asarray(owner_ids < 0, dtype=bool))
    )
    all_unmatched_points = int(np.count_nonzero(np.asarray(owner_ids < 0, dtype=bool)))
    pin_qc_pass = bool(
        int(assignment_summary.get("coord_owner_conflicts", 0)) == 0
        and len(unpaired_labeled_defect_ids) == 0
    )
    point_qc_pass = bool(pin_qc_pass and int(unresolved_labeled_points) == 0)

    slot_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    point_support_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    missing_stable_ids: List[int] = []
    for ref_index, info_ref in enumerate(infos_ref):
        stable_id = int(info_ref.get("stable_id", ref_index))
        pair_meta = pair_by_ref_index.get(int(ref_index))
        gt_present = int(1 if pair_meta is not None else 0)
        gt_missing = int(1 - gt_present)
        gt_abnormal = 0
        labeled_points = 0
        assigned_points = 0
        defect_id = -1
        pair_dist_u = float("nan")
        gt_point_eval_eligible = 0
        gt_point_global_eval_eligible = 0
        gt_point_support_present = 0
        gt_geo_eligible = 0
        gt_exclusion_reason = "gt_missing"
        assignment_stats = {"exact": 0, "fallback": 0}
        if pair_meta is not None:
            defect_id = int(pair_meta.get("defect_id", -1))
            pair_dist_u = _safe_float(dict(pair_meta.get("pair", {})).get("dist", float("nan")), default=float("nan"))
            labeled_points = int(labeled_counts_by_defect.get(defect_id, 0))
            assigned_points = int(total_counts_by_defect.get(defect_id, 0))
            gt_abnormal = int(labeled_points > 0)
            gt_geo_eligible = int(1 if pin_qc_pass else 0)
            gt_point_support_present = int(1 if assigned_points > 0 else 0)
            gt_point_global_eval_eligible = int(1 if point_qc_pass and assigned_points > 0 else 0)
            gt_point_eval_eligible = int(1 if point_qc_pass and gt_abnormal and assigned_points > 0 else 0)
            gt_exclusion_reason = "" if int(gt_geo_eligible) == 1 else "sample_pin_qc_failed"
            assignment_stats = dict(assignment_counts_by_defect.get(defect_id, {"exact": 0, "fallback": 0}))
        else:
            missing_stable_ids.append(int(stable_id))
        slot_row = {
            "sample_uid": str(sample_uid_n),
            "sample_id": str(sample_id),
            "source_sample_id": str(sample_id),
            "gt_partition": str(gt_partition),
            "gt_sample_key": str(gt_sample_key),
            "stable_id": int(stable_id),
            "gt_missing": int(gt_missing),
            "gt_present": int(gt_present),
            "gt_logic_label": "missing" if int(gt_missing) == 1 else "present",
            "gt_geo_eligible": int(gt_geo_eligible),
            "gt_abnormal": int(gt_abnormal),
            "gt_label": "abnormal" if int(gt_abnormal) == 1 else ("normal" if int(gt_present) == 1 else ""),
            "dominant_label": "A" if int(gt_abnormal) == 1 else "",
            "gt_point_support_present": int(gt_point_support_present),
            "gt_point_global_eval_eligible": int(gt_point_global_eval_eligible),
            "gt_point_eval_eligible": int(gt_point_eval_eligible),
            "gt_exclusion_reason": str(gt_exclusion_reason),
            "labeled_points": int(labeled_points),
            "A_points": int(labeled_points),
            "assigned_points": int(assigned_points),
            "gt_source": "tolerance_anomaly_GT",
            "gt_label_field": str(label_field_name or ""),
            "gt_pin_path": str(canonical_pin_gt_path),
            "gt_base_path": str(canonical_base_gt_path),
            "gt_pair_ref_index": int(ref_index),
            "gt_pair_defect_id": int(defect_id),
            "gt_pair_dist_u": float(pair_dist_u),
            "gt_qc_pass": int(1 if pin_qc_pass else 0),
            "gt_pin_qc_pass": int(1 if pin_qc_pass else 0),
            "gt_point_qc_pass": int(1 if point_qc_pass else 0),
            "gt_assignment_exact_count": int(assignment_stats.get("exact", 0)),
            "gt_assignment_fallback_count": int(assignment_stats.get("fallback", 0)),
        }
        slot_key = (str(sample_uid_n), int(stable_id))
        slot_map[slot_key] = slot_row

        if pair_meta is not None and assigned_points > 0:
            mask = np.asarray(owner_ids == int(defect_id), dtype=bool)
            point_support_by_key[slot_key] = {
                "sample_uid": str(sample_uid_n),
                "sample_id": str(sample_id),
                "source_sample_id": str(sample_id),
                "gt_partition": str(gt_partition),
                "gt_sample_key": str(gt_sample_key),
                "stable_id": int(stable_id),
                "points_xyz_aligned": np.asarray(gt_xyz_aligned[mask], dtype=np.float32),
                "gt_positive_mask": np.asarray(label_mask[mask], dtype=np.uint8),
                "assignment_source": np.asarray(assignment_source[mask], dtype=np.int8),
                "gt_pin_path": str(canonical_pin_gt_path),
                "gt_base_path": str(canonical_base_gt_path),
                "gt_label_field": str(label_field_name or ""),
                "gt_qc_pass": int(1 if pin_qc_pass else 0),
                "gt_pin_qc_pass": int(1 if pin_qc_pass else 0),
                "gt_point_qc_pass": int(1 if point_qc_pass else 0),
                "gt_point_support_present": int(gt_point_support_present),
                "gt_point_global_eval_eligible": int(gt_point_global_eval_eligible),
                "gt_point_eval_eligible": int(gt_point_eval_eligible),
            }

    summary = {
        "sample_uid": str(sample_uid_n),
        "sample_id": str(sample_id),
        "source_sample_id": str(sample_id),
        "gt_partition": str(gt_partition),
        "gt_sample_key": str(gt_sample_key),
        "pin_gt_path": str(canonical_pin_gt_path),
        "base_gt_path": str(canonical_base_gt_path),
        "resolved_pin_gt_path": str(pin_gt_path),
        "resolved_base_gt_path": str(base_gt_path),
        "label_field_name": str(label_field_name or ""),
        "label_field_present": int(1 if label_field_name else 0),
        "raw_labeled_points": int(label_mask.sum()),
        "paired_count": int(len(pairing.get("pairs", []))),
        "missing_normal_count": int(len(pairing.get("missing_normal", []))),
        "extra_defect_count": int(len(pairing.get("extra_defect", []))),
        "coord_owner_conflicts": int(assignment_summary.get("coord_owner_conflicts", 0)),
        "exact_unmatched_points": int(assignment_summary.get("exact_unmatched_points", 0)),
        "exact_unmatched_labeled_points": int(
            np.count_nonzero(np.asarray(label_mask, dtype=bool) & np.asarray(assignment_source < 0, dtype=bool))
        ),
        "fallback_match_count": int(assignment_summary.get("fallback_match_count", 0)),
        "fallback_match_max_dist_mm": float(assignment_summary.get("fallback_match_max_dist_mm", 0.0)),
        "fallback_match_p95_dist_mm": float(assignment_summary.get("fallback_match_p95_dist_mm", 0.0)),
        "fallback_match_threshold_mm": float(assignment_summary.get("fallback_match_threshold_mm", GT_LABEL_MATCH_MAX_DIST_MM)),
        "unmatched_labeled_points": int(unresolved_labeled_points),
        "unmatched_points_total": int(all_unmatched_points),
        "unpaired_labeled_defect_ids": [int(x) for x in unpaired_labeled_defect_ids],
        "used_stable_ids": [int(k[1]) for k in sorted(slot_map.keys(), key=lambda item: item[1]) if int(slot_map[k]["gt_present"]) == 1],
        "missing_stable_ids": [int(x) for x in missing_stable_ids],
        "registration_status": str(reg_meta.get("status", "")) if isinstance(reg_meta, dict) else "",
        "registration_method": str(reg_meta.get("method", "")) if isinstance(reg_meta, dict) else "",
        "gt_slot_total": int(len(infos_ref)),
        "gt_present_count": int(sum(int(row.get("gt_present", 0)) for row in slot_map.values())),
        "gt_missing_count": int(sum(int(row.get("gt_missing", 0)) for row in slot_map.values())),
        "gt_geo_eligible_count": int(sum(int(row.get("gt_geo_eligible", 0)) for row in slot_map.values())),
        "gt_abnormal_count": int(sum(int(row.get("gt_abnormal", 0)) for row in slot_map.values())),
        "gt_point_support_present_count": int(sum(int(row.get("gt_point_support_present", 0)) for row in slot_map.values())),
        "gt_point_global_eval_eligible_count": int(sum(int(row.get("gt_point_global_eval_eligible", 0)) for row in slot_map.values())),
        "gt_point_eval_eligible_count": int(sum(int(row.get("gt_point_eval_eligible", 0)) for row in slot_map.values())),
        "qc_pass": bool(pin_qc_pass),
        "pin_qc_pass": bool(pin_qc_pass),
        "point_qc_pass": bool(point_qc_pass),
    }
    return slot_map, summary, point_support_by_key


def load_gt_authoritative_by_sample(
    gt_root: Path,
    sample_ids: Iterable[str],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    gt_root = Path(gt_root).resolve()
    ref_pack, tol_base_path, tol_pins_path, _tol_json = _build_reference_infos()
    gt_slot_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []
    sample_summaries: Dict[str, Dict[str, Any]] = {}
    point_support_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for sample_id in sample_ids:
        sample_id = str(sample_id or "").strip()
        if not sample_id:
            continue
        try:
            sample_slot_map, summary, sample_point_support = _sample_gt_authoritative(
                sample_id=sample_id,
                sample_uid=make_sample_uid(sample_id, "bad"),
                gt_root=gt_root,
                ref_pack=ref_pack,
                tol_base_path=tol_base_path,
                tol_pins_path=tol_pins_path,
                gt_partition="bad",
            )
        except FileNotFoundError as exc:
            skipped.append({"sample_uid": str(make_sample_uid(sample_id, "bad")), "sample_id": str(sample_id), "reason": "missing_gt_file", "error": str(exc)})
            continue
        except Exception as exc:
            skipped.append({"sample_uid": str(make_sample_uid(sample_id, "bad")), "sample_id": str(sample_id), "reason": "gt_native_parse_failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not bool(summary.get("pin_qc_pass", summary.get("qc_pass", False))):
            skipped.append({"sample_uid": str(summary.get("sample_uid", make_sample_uid(sample_id, 'bad'))), "sample_id": str(sample_id), "reason": "gt_native_qc_failed", **summary})
            continue
        gt_slot_map.update(sample_slot_map)
        sample_summaries[str(summary.get("sample_uid", make_sample_uid(sample_id, 'bad')))] = dict(summary)
        point_support_by_key.update(sample_point_support)
    return gt_slot_map, skipped, sample_summaries, point_support_by_key


def load_gt_authoritative_by_manifest_rows(
    gt_root: Path,
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    include_point_support: bool = True,
    use_sample_cache: bool = False,
    cache_root: Optional[Path] = None,
    require_sample_cache_hit: bool = False,
    write_sample_cache_on_miss: bool = True,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    gt_root = Path(gt_root).resolve()
    gt_slot_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []
    sample_summaries: Dict[str, Dict[str, Any]] = {}
    point_support_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    validated_rows = validate_manifest_rows_against_authoritative_gt(manifest_rows, gt_root)
    cache_root_path = Path(cache_root or LOGIC_GT_SAMPLE_CACHE_ROOT).resolve()
    sample_cache_enabled = bool(use_sample_cache) and not bool(include_point_support)
    reference_signatures = _logic_gt_cache_reference_signatures() if sample_cache_enabled else {}
    ref_pack: Optional[Dict[str, Any]] = None
    tol_base_path: Optional[Path] = None
    tol_pins_path: Optional[Path] = None
    seen_sample_uids: Dict[str, str] = {}
    for row in validated_rows:
        sample_uid = str(row.get("sample_uid", "") or "").strip()
        sample_id = str(row.get("source_sample_id", row.get("sample_id", "")) or "").strip()
        if not sample_uid or not sample_id:
            continue
        resolved_gt_ply = str(row.get("authoritative_gt_ply", "") or "").strip()
        prev_gt_ply = seen_sample_uids.get(sample_uid, "")
        if prev_gt_ply and _norm_path_text(prev_gt_ply) != _norm_path_text(resolved_gt_ply):
            raise RuntimeError(
                "manifest mixes multiple GT partitions/files for the same sample_uid within one run: "
                f"sample_uid={sample_uid} prev_gt_ply={prev_gt_ply} new_gt_ply={resolved_gt_ply}"
            )
        seen_sample_uids[sample_uid] = resolved_gt_ply
        sample_cache_hit = False
        sample_cache_path = None
        cache_key = ""
        sample_slot_map: Dict[Tuple[str, int], Dict[str, Any]]
        summary: Dict[str, Any]
        sample_point_support: Dict[Tuple[str, int], Dict[str, Any]]
        sample_file_signatures = (
            _logic_gt_cache_sample_file_signatures(
                authoritative_gt_ply=str(row.get("authoritative_gt_ply", "") or ""),
                authoritative_gt_base_ply=str(row.get("authoritative_gt_base_ply", "") or ""),
            )
            if sample_cache_enabled
            else {}
        )
        if sample_cache_enabled:
            cache_key = _logic_gt_cache_key(
                gt_root=gt_root,
                sample_id=sample_id,
                gt_partition=str(row.get("gt_partition", "bad") or "bad"),
                gt_sample_key=str(row.get("gt_sample_key", "") or ""),
                authoritative_gt_ply=str(row.get("authoritative_gt_ply", "") or ""),
                authoritative_gt_base_ply=str(row.get("authoritative_gt_base_ply", "") or ""),
                reference_signatures=reference_signatures,
                sample_file_signatures=sample_file_signatures,
            )
            sample_cache_path = _logic_gt_cache_path(
                cache_root_path,
                sample_id=sample_id,
                gt_partition=str(row.get("gt_partition", "bad") or "bad"),
                cache_key=cache_key,
            )
            cached = _read_logic_gt_sample_cache(sample_cache_path, cache_key)
            if cached is not None:
                sample_slot_map, summary = cached
                sample_point_support = {}
                sample_cache_hit = True
            elif bool(require_sample_cache_hit):
                raise RuntimeError(
                    "Prepared logic/geo GT cache missing or stale for "
                    f"sample_uid={sample_uid} sample_id={sample_id} cache_path={sample_cache_path}"
                )
        if not sample_cache_hit:
            if ref_pack is None or tol_base_path is None or tol_pins_path is None:
                ref_pack, tol_base_path, tol_pins_path, _tol_json = _build_reference_infos()
            assert ref_pack is not None and tol_base_path is not None and tol_pins_path is not None
            try:
                sample_slot_map, summary, sample_point_support = _sample_gt_authoritative(
                    sample_id=sample_id,
                    sample_uid=sample_uid,
                    gt_root=gt_root,
                    ref_pack=ref_pack,
                    tol_base_path=tol_base_path,
                    tol_pins_path=tol_pins_path,
                    gt_partition=str(row.get("gt_partition", "bad") or "bad"),
                    gt_ply=str(row.get("authoritative_gt_ply", "") or ""),
                    gt_base_ply=str(row.get("authoritative_gt_base_ply", "") or ""),
                    gt_sample_key=str(row.get("gt_sample_key", "") or ""),
                )
            except FileNotFoundError as exc:
                skipped.append(
                    {
                        "sample_id": str(sample_id),
                        "sample_uid": str(sample_uid),
                        "gt_partition": str(row.get("gt_partition", "bad") or "bad"),
                        "gt_sample_key": str(row.get("gt_sample_key", "") or ""),
                        "reason": "missing_gt_file",
                        "error": str(exc),
                    }
                )
                continue
            except Exception as exc:
                skipped.append(
                    {
                        "sample_id": str(sample_id),
                        "sample_uid": str(sample_uid),
                        "gt_partition": str(row.get("gt_partition", "bad") or "bad"),
                        "gt_sample_key": str(row.get("gt_sample_key", "") or ""),
                        "reason": "gt_native_parse_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if sample_cache_enabled and sample_cache_path is not None and bool(write_sample_cache_on_miss):
                _write_logic_gt_sample_cache(
                    sample_cache_path,
                    cache_key=cache_key,
                    sample_row=row,
                    slot_map=sample_slot_map,
                    summary=summary,
                    reference_signatures=reference_signatures,
                    sample_file_signatures=sample_file_signatures,
                )
        if not bool(include_point_support):
            sample_point_support = {}
        summary = dict(summary)
        summary["logic_gt_cache_hit_yes_no"] = "Y" if bool(sample_cache_hit) else "N"
        summary["logic_gt_cache_path"] = str(sample_cache_path) if sample_cache_path is not None else ""
        summary["logic_gt_cache_key"] = str(cache_key)
        summary["logic_gt_cache_status"] = (
            "hit"
            if bool(sample_cache_hit)
            else ("miss_written" if sample_cache_enabled and bool(write_sample_cache_on_miss) else ("miss_live" if sample_cache_enabled else "disabled"))
        )
        if not bool(summary.get("pin_qc_pass", summary.get("qc_pass", False))):
            skipped.append(
                {
                    "sample_id": str(sample_id),
                    "sample_uid": str(sample_uid),
                    "gt_partition": str(summary.get("gt_partition", row.get("gt_partition", "bad") or "bad")),
                    "gt_sample_key": str(summary.get("gt_sample_key", row.get("gt_sample_key", "") or "")),
                    "reason": "gt_native_qc_failed",
                    **summary,
                }
            )
            continue
        gt_slot_map.update(sample_slot_map)
        sample_summaries[str(sample_uid)] = dict(summary)
        point_support_by_key.update(sample_point_support)
    return gt_slot_map, skipped, sample_summaries, point_support_by_key


def load_gt_binary_by_sample(gt_root: Path, sample_ids: Iterable[str]) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    gt_slot_map, skipped, _sample_summaries, _point_support_by_key = load_gt_authoritative_by_sample(gt_root, sample_ids)
    gt_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, row in gt_slot_map.items():
        if int(row.get("gt_geo_eligible", 0)) != 1:
            continue
        gt_map[key] = {
            "gt_abnormal": int(row.get("gt_abnormal", 0)),
            "gt_label": str(row.get("gt_label", "")),
            "dominant_label": str(row.get("dominant_label", "")),
            "labeled_points": int(row.get("labeled_points", 0)),
            "A_points": int(row.get("A_points", 0)),
            "gt_source": str(row.get("gt_source", "tolerance_anomaly_GT")),
            "gt_label_field": str(row.get("gt_label_field", "")),
            "gt_pin_path": str(row.get("gt_pin_path", "")),
            "gt_base_path": str(row.get("gt_base_path", "")),
            "gt_present": int(row.get("gt_present", 0)),
            "gt_missing": int(row.get("gt_missing", 0)),
            "gt_logic_label": str(row.get("gt_logic_label", "")),
            "gt_geo_eligible": int(row.get("gt_geo_eligible", 0)),
            "gt_point_eval_eligible": int(row.get("gt_point_eval_eligible", 0)),
            "gt_exclusion_reason": str(row.get("gt_exclusion_reason", "")),
        }
    return gt_map, skipped


def load_gt_binary_by_manifest_rows(
    gt_root: Path,
    manifest_rows: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    gt_slot_map, skipped, _sample_summaries, _point_support_by_key = load_gt_authoritative_by_manifest_rows(gt_root, manifest_rows)
    gt_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, row in gt_slot_map.items():
        if int(row.get("gt_geo_eligible", 0)) != 1:
            continue
        gt_map[key] = {
            "sample_id": str(row.get("sample_id", "")),
            "gt_partition": str(row.get("gt_partition", "")),
            "gt_sample_key": str(row.get("gt_sample_key", "")),
            "gt_abnormal": int(row.get("gt_abnormal", 0)),
            "gt_label": str(row.get("gt_label", "")),
            "dominant_label": str(row.get("dominant_label", "")),
            "labeled_points": int(row.get("labeled_points", 0)),
            "A_points": int(row.get("A_points", 0)),
            "gt_source": str(row.get("gt_source", "tolerance_anomaly_GT")),
            "gt_label_field": str(row.get("gt_label_field", "")),
            "gt_pin_path": str(row.get("gt_pin_path", "")),
            "gt_base_path": str(row.get("gt_base_path", "")),
            "gt_present": int(row.get("gt_present", 0)),
            "gt_missing": int(row.get("gt_missing", 0)),
            "gt_logic_label": str(row.get("gt_logic_label", "")),
            "gt_geo_eligible": int(row.get("gt_geo_eligible", 0)),
            "gt_point_eval_eligible": int(row.get("gt_point_eval_eligible", 0)),
            "gt_exclusion_reason": str(row.get("gt_exclusion_reason", "")),
        }
    return gt_map, skipped
