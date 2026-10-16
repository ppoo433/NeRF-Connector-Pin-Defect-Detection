# -*- coding: utf-8 -*-
"""
Reference-slot matching core for stable slot-centric numbering.

Phase B scope:
- load fixed reference slots
- load observed needles from a backbone-like JSON
- compute conservative one-to-one matching
- support self-match and simulated missing-needle smoke
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
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# User-editable defaults
# -----------------------------

SLOTS_JSON = ""  # default: <repo_root>/annotations/reference_slots_v1/reference_slots.json
OBSERVED_JSON = ""  # default: <repo_root>/annotations/reference_centerlines_v1/1/centerline_final.json
OUTPUT_JSON = ""  # default: debug_out run_dir / reference_slot_matching_report.json
SOURCE_SAMPLE_ID = "1"
CENTER_RADIUS_MM: float | str = "auto"
ANGLE_GATE_DEG = 35.0
LENGTH_GATE_MM = 4.0
AMBIGUOUS_COST_DELTA = 0.12
DROP_STABLE_IDS: List[int] = []


@dataclass
class SlotAnchor:
    stable_id: int
    nominal_center_u: np.ndarray
    nominal_curve_u: np.ndarray
    nominal_length_mm: float
    base_point_u: np.ndarray
    tip_point_u: np.ndarray


@dataclass
class ObservedNeedle:
    needle_id: int
    observed_order_id: int
    source_stable_id: Optional[int]
    center_u: np.ndarray
    curve_u: np.ndarray
    length_mm: float
    base_point_u: np.ndarray
    tip_point_u: np.ndarray


@dataclass
class MatchCandidate:
    stable_id: int
    needle_id: int
    center_dist_mm: float
    direction_angle_deg: float
    length_diff_mm: float
    cost: float


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


def _resolve_path(repo_root: Path, raw: str, fallback_parts: Sequence[str]) -> Path:
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


def _safe_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return np.zeros((3,), dtype=np.float64)
    return arr.reshape(3)


def _polyline_length_mm(points: np.ndarray, unit_scale: float) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.sum(seg) / (float(unit_scale) + 1e-12))


def _curve_endpoints(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 1:
        z = np.zeros((3,), dtype=np.float64)
        return z.copy(), z.copy()
    return pts[0].reshape(3), pts[-1].reshape(3)


def _curve_direction(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    vec = np.asarray(pts[-1] - pts[0], dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / n


def _undirected_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    va = _curve_direction(np.vstack([np.zeros((1, 3)), a.reshape(1, 3)])) if a.ndim == 1 else _curve_direction(a)
    vb = _curve_direction(np.vstack([np.zeros((1, 3)), b.reshape(1, 3)])) if b.ndim == 1 else _curve_direction(b)
    dot = float(np.clip(np.abs(np.dot(va.reshape(3), vb.reshape(3))), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _median_slot_spacing_mm(slots: Sequence[SlotAnchor], unit_scale: float) -> float:
    centers = np.asarray([s.nominal_center_u for s in slots], dtype=np.float64)
    if len(centers) < 2:
        return 5.0
    dists = []
    for i in range(len(centers)):
        diff = centers - centers[i].reshape(1, 3)
        dd = np.linalg.norm(diff, axis=1)
        dd = dd[np.isfinite(dd) & (dd > 1e-12)]
        if dd.size:
            dists.append(float(np.min(dd)) / (float(unit_scale) + 1e-12))
    if not dists:
        return 5.0
    return float(np.median(np.asarray(dists, dtype=np.float64)))


def _resolve_center_radius_mm(slots: Sequence[SlotAnchor], unit_scale: float, user_value: Any) -> float:
    if isinstance(user_value, (int, float)):
        v = float(user_value)
        if np.isfinite(v) and v > 0:
            return v
    spacing = _median_slot_spacing_mm(slots, unit_scale)
    return max(0.5 * spacing, 1.0)


def load_reference_slots(slots_json_path: Path) -> Tuple[List[SlotAnchor], Dict[str, Any]]:
    payload = _read_json(slots_json_path)
    slots: List[SlotAnchor] = []
    for row in payload.get("slots", []) or []:
        sid = int(row.get("stable_id", -1))
        if sid < 0:
            continue
        curve = np.asarray(row.get("nominal_curve_u", []), dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        slots.append(
            SlotAnchor(
                stable_id=int(sid),
                nominal_center_u=_safe_vec3(row.get("nominal_center_u", [0.0, 0.0, 0.0])),
                nominal_curve_u=curve,
                nominal_length_mm=float(row.get("nominal_length_mm", 0.0) or 0.0),
                base_point_u=_safe_vec3(row.get("base_point_u", [0.0, 0.0, 0.0])),
                tip_point_u=_safe_vec3(row.get("tip_point_u", [0.0, 0.0, 0.0])),
            )
        )
    slots = sorted(slots, key=lambda s: int(s.stable_id))
    return slots, payload


def load_observed_needles(observed_json_path: Path, drop_stable_ids: Optional[Sequence[int]] = None) -> Tuple[List[ObservedNeedle], Dict[str, Any]]:
    payload = _read_json(observed_json_path)
    unit_scale = float(payload.get("unit_scale", 0.001) or 0.001)
    drop = {int(x) for x in (drop_stable_ids or [])}
    needles: List[ObservedNeedle] = []
    for idx, row in enumerate(payload.get("needles", []) or []):
        source_sid = row.get("stable_id", None)
        if source_sid is not None and int(source_sid) in drop:
            continue
        segments = row.get("segments", []) or []
        curve = np.asarray(segments[0], dtype=np.float64) if segments else np.zeros((0, 3), dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            continue
        base_u, tip_u = _curve_endpoints(curve)
        needles.append(
            ObservedNeedle(
                needle_id=int(idx),
                observed_order_id=int(idx),
                source_stable_id=(None if source_sid is None else int(source_sid)),
                center_u=np.mean(curve, axis=0).reshape(3),
                curve_u=curve,
                length_mm=_polyline_length_mm(curve, unit_scale),
                base_point_u=base_u,
                tip_point_u=tip_u,
            )
        )
    return needles, payload


def _candidate_cost(slot: SlotAnchor, needle: ObservedNeedle, unit_scale: float, center_radius_mm: float, angle_gate_deg: float, length_gate_mm: float) -> Optional[MatchCandidate]:
    center_dist_mm = float(np.linalg.norm(slot.nominal_center_u - needle.center_u) / (float(unit_scale) + 1e-12))
    direction_angle_deg = _undirected_angle_deg(slot.nominal_curve_u, needle.curve_u)
    length_diff_mm = abs(float(slot.nominal_length_mm) - float(needle.length_mm))

    if center_dist_mm > float(center_radius_mm):
        return None
    if direction_angle_deg > float(angle_gate_deg):
        return None
    if length_diff_mm > float(length_gate_mm):
        return None

    c_norm = center_dist_mm / max(float(center_radius_mm), 1e-9)
    a_norm = direction_angle_deg / max(float(angle_gate_deg), 1e-9)
    l_norm = length_diff_mm / max(float(length_gate_mm), 1e-9)
    cost = 0.55 * c_norm + 0.25 * a_norm + 0.20 * l_norm
    return MatchCandidate(
        stable_id=int(slot.stable_id),
        needle_id=int(needle.needle_id),
        center_dist_mm=float(center_dist_mm),
        direction_angle_deg=float(direction_angle_deg),
        length_diff_mm=float(length_diff_mm),
        cost=float(cost),
    )


def match_needles_to_slots(slots: Sequence[SlotAnchor], needles: Sequence[ObservedNeedle], *, unit_scale: float, center_radius_mm: float | str = CENTER_RADIUS_MM, angle_gate_deg: float = ANGLE_GATE_DEG, length_gate_mm: float = LENGTH_GATE_MM, ambiguous_cost_delta: float = AMBIGUOUS_COST_DELTA) -> Dict[str, Any]:
    resolved_center_radius_mm = _resolve_center_radius_mm(slots, unit_scale, center_radius_mm)

    by_needle: Dict[int, List[MatchCandidate]] = {}
    all_candidates: List[MatchCandidate] = []
    ambiguous_matches: List[Dict[str, Any]] = []

    for needle in needles:
        cand: List[MatchCandidate] = []
        for slot in slots:
            c = _candidate_cost(slot, needle, unit_scale, resolved_center_radius_mm, angle_gate_deg, length_gate_mm)
            if c is not None:
                cand.append(c)
                all_candidates.append(c)
        cand = sorted(cand, key=lambda x: x.cost)
        by_needle[int(needle.needle_id)] = cand
        if len(cand) >= 2 and (float(cand[1].cost) - float(cand[0].cost)) <= float(ambiguous_cost_delta):
            ambiguous_matches.append(
                {
                    "needle_id": int(needle.needle_id),
                    "candidate_stable_ids": [int(cand[0].stable_id), int(cand[1].stable_id)],
                    "candidate_costs": [float(cand[0].cost), float(cand[1].cost)],
                    "reason": "top_two_costs_too_close",
                }
            )

    used_slots: set[int] = set()
    used_needles: set[int] = set()
    matched_pairs: List[Dict[str, Any]] = []

    for cand in sorted(all_candidates, key=lambda c: (float(c.cost), float(c.center_dist_mm), float(c.direction_angle_deg), float(c.length_diff_mm))):
        sid = int(cand.stable_id)
        nid = int(cand.needle_id)
        if sid in used_slots or nid in used_needles:
            continue
        if any(int(a.get("needle_id", -1)) == nid for a in ambiguous_matches):
            continue
        used_slots.add(sid)
        used_needles.add(nid)
        matched_pairs.append(
            {
                "stable_id": sid,
                "needle_id": nid,
                "match_status": "matched",
                "match_cost": float(cand.cost),
                "center_dist_mm": float(cand.center_dist_mm),
                "direction_angle_deg": float(cand.direction_angle_deg),
                "length_diff_mm": float(cand.length_diff_mm),
            }
        )

    missing_stable_ids = [int(s.stable_id) for s in slots if int(s.stable_id) not in used_slots]
    extra_needle_ids = [int(n.needle_id) for n in needles if int(n.needle_id) not in used_needles and not any(int(a.get("needle_id", -1)) == int(n.needle_id) for a in ambiguous_matches)]

    return {
        "assignment_version": "reference_slot_matching_v1",
        "center_radius_mm": float(resolved_center_radius_mm),
        "angle_gate_deg": float(angle_gate_deg),
        "length_gate_mm": float(length_gate_mm),
        "ambiguous_cost_delta": float(ambiguous_cost_delta),
        "matched_pairs": sorted(matched_pairs, key=lambda d: int(d.get("stable_id", 10**9))),
        "missing_stable_ids": sorted(missing_stable_ids),
        "extra_needle_ids": sorted(extra_needle_ids),
        "ambiguous_matches": sorted(ambiguous_matches, key=lambda d: int(d.get("needle_id", 10**9))),
        "n_slots": int(len(slots)),
        "n_needles": int(len(needles)),
        "n_matched": int(len(matched_pairs)),
        "n_missing": int(len(missing_stable_ids)),
        "n_extra": int(len(extra_needle_ids)),
        "n_ambiguous": int(len(ambiguous_matches)),
    }


def _report(slots_json_path: Path, observed_json_path: Path, slots_payload: Dict[str, Any], observed_payload: Dict[str, Any], result: Dict[str, Any], drop_stable_ids: Sequence[int]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _now_iso(),
        "run_id": str(getattr(_DAIMA_CTX, "run_id", "")) if _DAIMA_CTX is not None else "",
        "slots_json_path": str(slots_json_path),
        "observed_json_path": str(observed_json_path),
        "slots_version": str(slots_payload.get("reference_slots_version", "")),
        "observed_sample_id": str(observed_payload.get("sample_id", "")),
        "drop_stable_ids": [int(x) for x in drop_stable_ids],
        "result": result,
    }


def main() -> int:
    repo_root = _find_repo_root(Path(__file__))

    parser = argparse.ArgumentParser(description="Reference-slot matching smoke / core runner.")
    parser.add_argument("--slots_json", default="", help="Optional reference_slots.json path.")
    parser.add_argument("--observed_json", default="", help="Optional observed centerline_final.json path.")
    parser.add_argument("--source_sample_id", default=SOURCE_SAMPLE_ID, help="Used only for default observed path resolution.")
    parser.add_argument("--center_radius_mm", default=str(CENTER_RADIUS_MM), help="Center-distance gate in mm or 'auto'.")
    parser.add_argument("--angle_gate_deg", type=float, default=ANGLE_GATE_DEG, help="Direction gate in deg.")
    parser.add_argument("--length_gate_mm", type=float, default=LENGTH_GATE_MM, help="Length-difference gate in mm.")
    parser.add_argument("--ambiguous_cost_delta", type=float, default=AMBIGUOUS_COST_DELTA, help="Ambiguity threshold on top-2 cost gap.")
    parser.add_argument("--drop_stable_ids", nargs="*", default=None, help="Optional stable_ids to drop from observed needles for missing-needle smoke.")
    parser.add_argument("--output_json", default="", help="Optional output report path.")
    args = parser.parse_args()

    slots_json_path = _resolve_path(repo_root, str(args.slots_json or SLOTS_JSON), ["annotations", "reference_slots_v1", "reference_slots.json"])
    observed_json_path = _resolve_path(repo_root, str(args.observed_json or OBSERVED_JSON), ["annotations", "reference_centerlines_v1", str(args.source_sample_id or SOURCE_SAMPLE_ID), "centerline_final.json"])
    output_json_path = _resolve_path(repo_root, str(args.output_json or OUTPUT_JSON), ["debug_out", "_tmp_placeholder.json"])
    if not str(args.output_json or OUTPUT_JSON).strip() and _DAIMA_CTX is not None and getattr(_DAIMA_CTX, "run_dir", None):
        output_json_path = (Path(_DAIMA_CTX.run_dir) / "reference_slot_matching_report.json").resolve()

    drop_stable_ids = [int(x) for x in (args.drop_stable_ids or DROP_STABLE_IDS or [])]

    slots, slots_payload = load_reference_slots(slots_json_path)
    needles, observed_payload = load_observed_needles(observed_json_path, drop_stable_ids=drop_stable_ids)
    unit_scale = float(slots_payload.get("unit_scale", observed_payload.get("unit_scale", 0.001)) or 0.001)

    center_radius_arg: Any = str(args.center_radius_mm)
    if str(center_radius_arg).strip().lower() != "auto":
        center_radius_arg = float(center_radius_arg)

    if _DAIMA_CTX is not None:
        _DAIMA_CTX.snapshot_paths(
            inputs={
                "slots_json": str(slots_json_path),
                "observed_json": str(observed_json_path),
            },
            outputs={
                "report_json": str(output_json_path),
            },
        )
        _DAIMA_CTX.snapshot_config(
            config={
                "source_sample_id": str(args.source_sample_id or SOURCE_SAMPLE_ID),
                "drop_stable_ids": list(drop_stable_ids),
                "center_radius_mm": center_radius_arg,
                "angle_gate_deg": float(args.angle_gate_deg),
                "length_gate_mm": float(args.length_gate_mm),
                "ambiguous_cost_delta": float(args.ambiguous_cost_delta),
            }
        )

    result = match_needles_to_slots(
        slots,
        needles,
        unit_scale=unit_scale,
        center_radius_mm=center_radius_arg,
        angle_gate_deg=float(args.angle_gate_deg),
        length_gate_mm=float(args.length_gate_mm),
        ambiguous_cost_delta=float(args.ambiguous_cost_delta),
    )
    report = _report(slots_json_path, observed_json_path, slots_payload, observed_payload, result, drop_stable_ids)
    _write_json(output_json_path, report)

    print(f"[INFO] Wrote: {output_json_path}")
    print(f"[INFO] matched={result['n_matched']} missing={result['n_missing']} extra={result['n_extra']} ambiguous={result['n_ambiguous']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
