# -*- coding: utf-8 -*-
"""
Export `reference_windows_v1` artifacts from frozen reference centerlines.

P3 scope only:
- windows.json
- root manifest

No label projection in this round.
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
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# User-editable defaults
# -----------------------------

BACKBONE_ROOT = ""  # default: <repo_root>/annotations/reference_centerlines_v1
SLOTS_JSON = ""  # default: <repo_root>/annotations/reference_slots_v1/reference_slots.json
OUTPUT_ROOT = ""  # default: <repo_root>/annotations/reference_windows_v1
SAMPLE_IDS: List[str] = ["1"]
WINDOW_SPEC_VERSION = "reference_windows_v1"
DEFAULT_WINDOW_SPAN_MM = 1.0
DEFAULT_WINDOW_STEP_MM = 0.5


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _natural_key(text: str) -> List[object]:
    import re

    out: List[object] = []
    for part in re.split(r"(\d+)", str(text)):
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return out


def _resolve_root(repo_root: Path, raw: str, fallback_parts: Sequence[str]) -> Path:
    value = str(raw or "").strip()
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    p = repo_root
    for part in fallback_parts:
        p = p / part
    return p.resolve()


def _resolve_slots_json(repo_root: Path, raw: str) -> Path:
    value = str(raw or SLOTS_JSON).strip()
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    return (repo_root / "annotations" / "reference_slots_v1" / "reference_slots.json").resolve()


def _resolve_sample_ids(backbone_root: Path, cli_ids: Optional[Sequence[str]]) -> List[str]:
    if cli_ids:
        return [str(x).strip() for x in cli_ids if str(x).strip()]
    if SAMPLE_IDS:
        return [str(x).strip() for x in SAMPLE_IDS if str(x).strip()]
    ids = [p.name for p in backbone_root.iterdir() if p.is_dir()]
    return sorted(ids, key=_natural_key)[:1]


def _polyline_arclength_mm(points: np.ndarray, unit_scale: float) -> Tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return np.zeros((max(len(pts), 1),), dtype=np.float64), 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_u = np.concatenate([[0.0], np.cumsum(seg)])
    s_mm = s_u / (float(unit_scale) + 1e-12)
    return s_mm.astype(np.float64), float(s_mm[-1])


def _window_role(idx: int, total: int) -> str:
    if total <= 1:
        return "middle"
    if idx == 0:
        return "start"
    if idx == total - 1:
        return "end"
    return "middle"


def _slot_ids(slots_payload: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for row in slots_payload.get("slots", []) or []:
        try:
            sid = int(row.get("stable_id", -1))
        except Exception:
            continue
        if sid >= 0:
            out.append(int(sid))
    return sorted(set(out))


def _slot_curve_map(slots_payload: Dict[str, Any]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for row in slots_payload.get("slots", []) or []:
        try:
            sid = int(row.get("stable_id", -1))
        except Exception:
            continue
        curve = np.asarray(row.get("nominal_curve_u", []), dtype=np.float64)
        if sid >= 0 and curve.ndim == 2 and curve.shape[1] == 3 and len(curve) >= 2:
            out[sid] = curve
    return out


def _matched_needles_by_sid(final_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in final_json.get("needles", []) or []:
        try:
            sid = int(row.get("stable_id", -1))
        except Exception:
            continue
        if sid >= 0 and sid not in out:
            out[sid] = dict(row)
    return out


def _ambiguous_slot_ids(final_json: Dict[str, Any]) -> List[int]:
    slot_ids: set[int] = set()
    for row in final_json.get("ambiguous_matches", []) or []:
        for sid in row.get("candidate_stable_ids", []) or []:
            try:
                slot_ids.add(int(sid))
            except Exception:
                continue
    return sorted(slot_ids)


def _slot_presence_status(slot_id: int, matched_map: Dict[int, Dict[str, Any]], missing_ids: set[int], ambiguous_ids: set[int]) -> str:
    if int(slot_id) in matched_map:
        return "matched"
    if int(slot_id) in ambiguous_ids:
        return "ambiguous"
    if int(slot_id) in missing_ids:
        return "missing"
    return "unresolved"


def _build_windows_for_curve(curve: np.ndarray, unit_scale: float, window_span_mm: float, window_step_mm: float) -> Tuple[List[Dict[str, Any]], float]:
    s_mm, nominal_length_mm = _polyline_arclength_mm(curve, unit_scale)
    if nominal_length_mm <= 0.0:
        return [], 0.0

    span = float(max(window_span_mm, 1e-6))
    step = float(max(window_step_mm, 1e-6))
    half = 0.5 * span

    centers: List[float] = []
    if nominal_length_mm <= span:
        centers = [0.5 * nominal_length_mm]
    else:
        c = half
        last_center = nominal_length_mm - half
        while c <= last_center + 1e-9:
            centers.append(float(c))
            c += step
        if not centers:
            centers = [0.5 * nominal_length_mm]
        elif centers[-1] < last_center - 1e-9:
            centers.append(float(last_center))

    centers = sorted({round(float(x), 9) for x in centers})

    windows: List[Dict[str, Any]] = []
    for idx, center in enumerate(centers):
        start = max(0.0, center - half)
        end = min(nominal_length_mm, center + half)
        windows.append(
            {
                "window_id": int(idx),
                "station_s_mm": round(float(center), 6),
                "window_center_s_mm": round(float(center), 6),
                "window_start_s_mm": round(float(start), 6),
                "window_end_s_mm": round(float(end), 6),
                "window_span_mm": round(float(end - start), 6),
                "window_step_mm": round(float(step), 6),
                "window_role": "",
            }
        )
    total = len(windows)
    for idx, row in enumerate(windows):
        row["window_role"] = _window_role(idx, total)
    return windows, nominal_length_mm


def _build_sample_windows(sample_id: str, final_json: Dict[str, Any], slots_payload: Dict[str, Any], source_centerline_path: Path, source_slots_path: Path, window_span_mm: float, window_step_mm: float) -> Dict[str, Any]:
    unit_scale = float(final_json.get("unit_scale", slots_payload.get("unit_scale", 0.001)) or 0.001)
    slot_curve_map = _slot_curve_map(slots_payload)
    matched_map = _matched_needles_by_sid(final_json)
    missing_ids = {int(x) for x in (final_json.get("missing_stable_ids", []) or [])}
    ambiguous_ids = set(_ambiguous_slot_ids(final_json))

    needles_out: List[Dict[str, Any]] = []
    for sid in _slot_ids(slots_payload):
        curve = slot_curve_map.get(int(sid), np.zeros((0, 3), dtype=np.float64))
        if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            windows = []
            nominal_length_mm = 0.0
        else:
            windows, nominal_length_mm = _build_windows_for_curve(curve, unit_scale, window_span_mm, window_step_mm)
        matched_row = matched_map.get(int(sid), {})
        needles_out.append(
            {
                "stable_id": int(sid),
                "slot_presence_status": _slot_presence_status(int(sid), matched_map, missing_ids, ambiguous_ids),
                "matched_needle_id": (None if int(matched_row.get("needle_id", -1)) < 0 else int(matched_row.get("needle_id", -1))) if matched_row else None,
                "source_stable_id": (None if int(matched_row.get("source_stable_id", -1)) < 0 else int(matched_row.get("source_stable_id", -1))) if matched_row else None,
                "n_windows": int(len(windows)),
                "nominal_length_mm": round(float(nominal_length_mm), 6),
                "windows": windows,
            }
        )
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "sample_id": str(sample_id),
        "backbone_version": str(final_json.get("backbone_version", "reference_centerline_v1")),
        "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
        "window_spec_version": WINDOW_SPEC_VERSION,
        "source_centerline_path": str(source_centerline_path),
        "source_slots_path": str(source_slots_path),
        "unit_scale": unit_scale,
        "window_span_mm": float(window_span_mm),
        "window_step_mm": float(window_step_mm),
        "missing_stable_ids": list(final_json.get("missing_stable_ids", []) or []),
        "extra_needle_ids": list(final_json.get("extra_needle_ids", []) or []),
        "needles": needles_out,
    }


def _validate_windows_payload(payload: Dict[str, Any]) -> None:
    for needle in payload.get("needles", []) or []:
        seen = set()
        prev_s = -1e18
        for row in needle.get("windows", []) or []:
            wid = int(row.get("window_id", -1))
            if wid in seen:
                raise ValueError(f"duplicate window_id: {wid}")
            seen.add(wid)
            s = float(row.get("station_s_mm", -1.0))
            if s < prev_s:
                raise ValueError("window_id / station_s_mm order mismatch")
            prev_s = s


def _export_one_sample(backbone_root: Path, output_root: Path, slots_payload: Dict[str, Any], source_slots_path: Path, sample_id: str, window_span_mm: float, window_step_mm: float) -> Dict[str, Any]:
    final_path = (backbone_root / str(sample_id) / "centerline_final.json").resolve()
    if not final_path.is_file():
        raise FileNotFoundError(str(final_path))
    final_json = _read_json(final_path)
    payload = _build_sample_windows(str(sample_id), final_json, slots_payload, final_path, source_slots_path, window_span_mm, window_step_mm)
    _validate_windows_payload(payload)

    out_dir = (output_root / str(sample_id)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / "windows.json").resolve()
    _write_json(out_path, payload)

    n_needles = int(len(payload.get("needles", [])))
    n_windows = int(sum(int(n.get("n_windows", 0)) for n in payload.get("needles", [])))
    return {
        "sample_id": str(sample_id),
        "status": "ok",
        "windows_json_path": str(out_path),
        "n_needles": n_needles,
        "n_windows": n_windows,
    }


def main() -> int:
    repo_root = _find_repo_root(Path(__file__))

    parser = argparse.ArgumentParser(description="Export reference_windows_v1 from frozen reference centerlines.")
    parser.add_argument("--backbone_root", default="", help="Optional backbone root. Defaults to annotations/reference_centerlines_v1.")
    parser.add_argument("--slots_json", default="", help="Optional reference_slots.json path.")
    parser.add_argument("--output_root", default="", help="Optional output root. Defaults to annotations/reference_windows_v1.")
    parser.add_argument("--sample_ids", nargs="*", default=None, help="Optional sample ids. Defaults to in-file SAMPLE_IDS.")
    parser.add_argument("--window_span_mm", type=float, default=DEFAULT_WINDOW_SPAN_MM, help="Window span in mm.")
    parser.add_argument("--window_step_mm", type=float, default=DEFAULT_WINDOW_STEP_MM, help="Window step in mm.")
    args = parser.parse_args()

    backbone_root = _resolve_root(repo_root, str(args.backbone_root or BACKBONE_ROOT), ["annotations", "reference_centerlines_v1"])
    slots_json_path = _resolve_slots_json(repo_root, str(args.slots_json or SLOTS_JSON))
    output_root = _resolve_root(repo_root, str(args.output_root or OUTPUT_ROOT), ["annotations", "reference_windows_v1"])
    sample_ids = _resolve_sample_ids(backbone_root, args.sample_ids)
    manifest_path = (output_root.parent / "reference_windows_v1_manifest.json").resolve()
    slots_payload = _read_json(slots_json_path)

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "backbone_root": str(backbone_root),
                "slots_json": str(slots_json_path),
                "sample_ids": list(sample_ids),
            },
            outputs={
                "output_root": str(output_root),
                "manifest_json": str(manifest_path),
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "window_spec_version": WINDOW_SPEC_VERSION,
                "reference_slots_version": str(slots_payload.get("reference_slots_version", "reference_slots_v1")),
                "window_span_mm": float(args.window_span_mm),
                "window_step_mm": float(args.window_step_mm),
                "sample_ids": list(sample_ids),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        try:
            ent = _export_one_sample(backbone_root, output_root, slots_payload, slots_json_path, str(sample_id), float(args.window_span_mm), float(args.window_step_mm))
            entries.append(ent)
            print(f"[OK] sample_id={sample_id} needles={ent['n_needles']} windows={ent['n_windows']}")
        except Exception as exc:
            msg = str(exc)
            errors.append({"sample_id": str(sample_id), "error": msg})
            print(f"[ERROR] sample_id={sample_id} err={msg}")

    manifest = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "run_id": str(getattr(_DAIMA_CTX, "run_id", "")) if _DAIMA_CTX is not None else "",
        "version": WINDOW_SPEC_VERSION,
        "backbone_root": str(backbone_root),
        "output_root": str(output_root),
        "sample_ids": list(sample_ids),
        "window_span_mm": float(args.window_span_mm),
        "window_step_mm": float(args.window_step_mm),
        "entries": entries,
        "errors": errors,
    }
    _write_json(manifest_path, manifest)

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(outputs={"manifest_json": str(manifest_path)})

    print(f"[INFO] Wrote windows manifest: {manifest_path}")
    print(f"[INFO] Success={len(entries)} Error={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
