"""Automatic anchor frame selection for robust scale recovery.

Anchors are intended to be a small subset of high-quality marker detections
that are less affected by:
  - extreme viewing angles (~85 deg),
  - partial occlusion,
  - image boundary cropping.

The selected anchors can then be used in a strict "anchors-only" mode to
prefer failure over accepting a biased scale estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnchorSelectorConfig:
    """Config for automatic anchor selection."""

    images_path: Path
    allowed_marker_ids: list[int]
    registered_image_names: set[str] | None = None

    # Hard gates (0 disables the gate)
    min_side_px: float = 0.0
    min_border_dist_px: float = 0.0

    # Selection budget
    max_both_markers: int = 30
    max_per_marker: int = 30
    max_total: int = 80

    # Optional scoring signal
    use_sharpness: bool = True


def _quad_area_xy(pts: np.ndarray) -> float:
    # Shoelace formula (abs area).
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_convex_quad(pts: np.ndarray) -> bool:
    # Check consistent turning direction for consecutive edges.
    if pts.shape != (4, 2):
        return False
    s = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        ab = b - a
        bc = c - b
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        if cross == 0:
            continue
        s.append(np.sign(cross))
    return bool(s) and (all(v >= 0 for v in s) or all(v <= 0 for v in s))


def _corner_angles_deg(pts: np.ndarray) -> list[float]:
    # Angle at each vertex (0..180).
    out: list[float] = []
    for i in range(4):
        p = pts[i]
        p_prev = pts[(i - 1) % 4]
        p_next = pts[(i + 1) % 4]
        v1 = p_prev - p
        v2 = p_next - p
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 0 or n2 <= 0:
            out.append(float("nan"))
            continue
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
        out.append(ang)
    return out


def _min_corner_border_dist_px(pts: np.ndarray, width: int, height: int) -> float:
    # Distance to nearest image border for each corner; take the minimum.
    x = pts[:, 0]
    y = pts[:, 1]
    d = np.minimum.reduce([x, (width - 1) - x, y, (height - 1) - y])
    return float(np.min(d))


def _laplacian_var(gray: np.ndarray) -> float:
    import cv2  # local import to keep module importable without cv2 in some environments

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


class AnchorSelector:
    """Select high-quality anchor images from detections."""

    def __init__(self, cfg: AnchorSelectorConfig) -> None:
        self.cfg = cfg
        self._allowed = set(int(x) for x in (cfg.allowed_marker_ids or []))

    def select(self, detections: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """
        Args:
            detections: mapping image_name -> {"markers":[{"id": int, "corners": [[x,y]x4]}, ...]}

        Returns:
            anchors: sorted list of selected image names
            report: detailed report for debugging
        """
        images_path = Path(self.cfg.images_path)
        registered = self.cfg.registered_image_names

        per_image: dict[str, Any] = {}
        for image_name, det in detections.items():
            if registered is not None and image_name not in registered:
                continue

            markers = det.get("markers", []) if isinstance(det, dict) else []
            # Gather marker candidates in this image.
            m_entries: dict[int, Any] = {}
            for m in markers:
                try:
                    mid = int(m.get("id"))
                except Exception:
                    continue
                if self._allowed and mid not in self._allowed:
                    continue
                corners = m.get("corners")
                if not isinstance(corners, list) or len(corners) != 4:
                    continue
                pts = np.asarray(corners, dtype=float).reshape(4, 2)

                # Basic geometry.
                side_lengths = [
                    float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)
                ]
                min_side = float(np.min(side_lengths)) if side_lengths else 0.0
                max_side = float(np.max(side_lengths)) if side_lengths else 0.0
                aspect = (max_side / min_side) if min_side > 0 else float("inf")
                area = _quad_area_xy(pts)
                convex = _is_convex_quad(pts)
                angles = _corner_angles_deg(pts)
                ang_dev = [abs(a - 90.0) for a in angles if np.isfinite(a)]
                ang_dev_mean = float(np.mean(ang_dev)) if ang_dev else float("inf")
                ang_dev_max = float(np.max(ang_dev)) if ang_dev else float("inf")

                # Image-dependent metrics (border distance, optional sharpness).
                border_dist = None
                sharpness = None
                img_path = images_path / image_name
                if img_path.is_file():
                    import cv2  # local import

                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        h, w = img.shape[:2]
                        border_dist = _min_corner_border_dist_px(pts, w, h)

                        if self.cfg.use_sharpness:
                            x0 = int(np.floor(float(np.min(pts[:, 0]))))
                            x1 = int(np.ceil(float(np.max(pts[:, 0]))))
                            y0 = int(np.floor(float(np.min(pts[:, 1]))))
                            y1 = int(np.ceil(float(np.max(pts[:, 1]))))
                            # Expand ROI a bit.
                            pad = int(max(8, 0.05 * max(x1 - x0, y1 - y0)))
                            x0 = max(0, x0 - pad)
                            y0 = max(0, y0 - pad)
                            x1 = min(w, x1 + pad)
                            y1 = min(h, y1 + pad)
                            if x1 > x0 and y1 > y0:
                                gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                                sharpness = _laplacian_var(gray)

                # Hard gates.
                rejected_reasons: list[str] = []
                if not convex:
                    rejected_reasons.append("non_convex")
                if area <= 1.0:
                    rejected_reasons.append("degenerate_area")
                if self.cfg.min_side_px and min_side < float(self.cfg.min_side_px):
                    rejected_reasons.append("min_side_px")
                if (
                    self.cfg.min_border_dist_px
                    and border_dist is not None
                    and border_dist < float(self.cfg.min_border_dist_px)
                ):
                    rejected_reasons.append("min_border_dist_px")

                # Score: larger & more fronto-parallel is better; closer to border is worse.
                score = float("-inf")
                if not rejected_reasons:
                    score = (
                        1.0 * min_side
                        + 0.2 * float(border_dist if border_dist is not None else 0.0)
                        - 20.0 * max(0.0, aspect - 1.0)
                        - 0.2 * ang_dev_mean
                    )
                    if sharpness is not None and sharpness > 0:
                        score += 0.02 * float(np.log1p(sharpness))

                m_entries[mid] = {
                    "score": score,
                    "rejected": bool(rejected_reasons),
                    "rejected_reasons": rejected_reasons,
                    "metrics": {
                        "min_side_px": min_side,
                        "max_side_px": max_side,
                        "aspect_ratio": aspect,
                        "area_px2": area,
                        "angle_dev_mean_deg": ang_dev_mean,
                        "angle_dev_max_deg": ang_dev_max,
                        "min_corner_border_dist_px": border_dist,
                        "sharpness_lap_var": sharpness,
                        "convex": convex,
                    },
                }

            if m_entries:
                per_image[image_name] = {
                    "markers": m_entries,
                    "has_both": all(mid in m_entries and not m_entries[mid]["rejected"] for mid in self._allowed)
                    if self._allowed
                    else False,
                }

        # Rank candidates.
        both: list[tuple[float, str]] = []
        per_marker: dict[int, list[tuple[float, str]]] = {mid: [] for mid in self._allowed}

        for image_name, rec in per_image.items():
            markers: dict[int, Any] = rec.get("markers", {})
            # per-marker ranking
            for mid in self._allowed:
                m = markers.get(mid)
                if not m or m.get("rejected"):
                    continue
                per_marker[mid].append((float(m["score"]), image_name))

            # both-markers ranking
            if rec.get("has_both"):
                # Conservative: the weaker marker dominates.
                s = min(float(markers[mid]["score"]) for mid in self._allowed)
                both.append((s, image_name))

        def _top_unique(items: list[tuple[float, str]], k: int, selected: set[str]) -> list[str]:
            out: list[str] = []
            for score, name in sorted(items, key=lambda x: x[0], reverse=True):
                if name in selected:
                    continue
                out.append(name)
                selected.add(name)
                if len(out) >= k:
                    break
            return out

        selected: set[str] = set()
        chosen_both = _top_unique(both, int(max(self.cfg.max_both_markers, 0)), selected)

        chosen_per_marker: dict[int, list[str]] = {}
        for mid in sorted(self._allowed):
            chosen_per_marker[mid] = _top_unique(
                per_marker[mid], int(max(self.cfg.max_per_marker, 0)), selected
            )

        anchors = sorted(list(selected))
        if self.cfg.max_total and len(anchors) > int(self.cfg.max_total):
            anchors = anchors[: int(self.cfg.max_total)]

        report: dict[str, Any] = {
            "config": {
                "images_path": str(images_path),
                "allowed_marker_ids": sorted(list(self._allowed)),
                "registered_image_names": None if registered is None else "provided",
                "min_side_px": float(self.cfg.min_side_px),
                "min_border_dist_px": float(self.cfg.min_border_dist_px),
                "max_both_markers": int(self.cfg.max_both_markers),
                "max_per_marker": int(self.cfg.max_per_marker),
                "max_total": int(self.cfg.max_total),
                "use_sharpness": bool(self.cfg.use_sharpness),
            },
            "counts": {
                "images_considered": int(len(per_image)),
                "anchors_selected": int(len(anchors)),
                "anchors_both": int(len(chosen_both)),
                "anchors_by_marker": {int(k): int(len(v)) for k, v in chosen_per_marker.items()},
            },
            "selection": {
                "both_markers": chosen_both,
                "per_marker": {int(k): v for k, v in chosen_per_marker.items()},
                "anchors": anchors,
            },
            "per_image": per_image,
        }

        return anchors, report

