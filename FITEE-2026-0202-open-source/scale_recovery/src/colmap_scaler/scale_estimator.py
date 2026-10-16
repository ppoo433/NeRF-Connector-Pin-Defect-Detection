"""Estimate scale factor from triangulated marker corners."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ScaleEstimator:
    """Estimate scale factor from markers with known size."""

    def __init__(self, marker_size_meters: float) -> None:
        """Initialize scale estimator.

        :param marker_size_meters: Physical marker size in meters.
        """
        self.marker_size = marker_size_meters

    def estimate_scale(
        self,
        triangulated_markers: dict[int, dict],
        method: str = "robust_mean"
    ) -> tuple[float, dict]:
        """Estimate scale factor from triangulated markers.

        :param triangulated_markers: Dictionary of triangulated marker data.
        :param method: Method to use ("simple", "robust_mean", "median").
        :returns: Tuple of (scale factor, statistics dict).
        """
        if not triangulated_markers:
            raise ValueError("No valid markers to estimate scale from")

        # Calculate scale for each marker (weighted by triangulation quality where available)
        marker_scales = {}
        marker_weights = {}
        for marker_id, data in triangulated_markers.items():
            if not data.get("valid", False):
                continue

            corners_3d = np.array(data["corners_3d"])
            corner_stats = data.get("corner_stats")
            scale, details = self._estimate_scale_from_corners(corners_3d, corner_stats=corner_stats)

            if scale is not None:
                marker_scales[marker_id] = scale
                marker_weights[marker_id] = details.get("marker_weight", 1.0)

        if not marker_scales:
            raise ValueError("Failed to estimate scale from any marker")

        logger.info("\nScale estimates from individual markers:")
        for marker_id, scale in marker_scales.items():
            logger.info(f"  Marker {marker_id}: {scale:.4f}")

        # Combine scales using specified method (supporting optional weights).
        scales = list(marker_scales.values())
        weights = [marker_weights[mid] for mid in marker_scales.keys()]
        final_scale = self._combine_scales(scales, method, weights=weights)

        # Calculate statistics
        stats = {
            "scale_factor": final_scale,
            "num_markers": len(marker_scales),
            "marker_scales": marker_scales,
            "marker_weights": marker_weights,
            "mean": np.mean(scales),
            "median": np.median(scales),
            "std": np.std(scales),
            "min": np.min(scales),
            "max": np.max(scales),
            "method": method,
        }

        return final_scale, stats

    def _estimate_scale_from_corners(
        self, corners_3d: np.ndarray, corner_stats: list[dict] | None = None
    ) -> tuple[float | None, dict]:
        """Estimate scale from a single marker's 3D corners.

        :param corners_3d: 4x3 array of corner positions.
        :param corner_stats: Optional triangulation quality stats (per corner).
        :returns: (scale factor or None, details dict).
        """
        if corners_3d.shape != (4, 3):
            return None, {}

        # Corner-quality weights (higher = more trustworthy).
        corner_weights = np.ones(4, dtype=np.float64)
        if corner_stats and len(corner_stats) == 4:
            # Use a conservative weight: more inliers & lower reprojection error -> higher weight.
            # Errors are in pixels; we keep weights bounded to avoid numerical blowups.
            for i, st in enumerate(corner_stats):
                n_in = float(st.get("num_inliers", 0))
                err = float(st.get("inlier_error_median", float("inf")))
                if not np.isfinite(err) or err <= 0 or n_in <= 0:
                    corner_weights[i] = 0.0
                else:
                    corner_weights[i] = n_in / max(err, 1e-3)

        # Calculate edge lengths in 3D
        edge_pairs = [
            (0, 1),  # Top edge
            (1, 2),  # Right edge
            (2, 3),  # Bottom edge
            (3, 0),  # Left edge
        ]

        # Calculate diagonal lengths
        diag_pairs = [
            (0, 2),  # Diagonal 1
            (1, 3),  # Diagonal 2
        ]

        # Expected lengths
        edge_expected = self.marker_size
        diag_expected = self.marker_size * np.sqrt(2)

        # Calculate scale for each measurement + weight derived from corner weights.
        scales: list[float] = []
        weights: list[float] = []

        for i, j in edge_pairs:
            length_3d = float(np.linalg.norm(corners_3d[i] - corners_3d[j]))
            if length_3d > 0:
                scales.append(edge_expected / length_3d)
                weights.append(float(min(corner_weights[i], corner_weights[j])))

        for i, j in diag_pairs:
            length_3d = float(np.linalg.norm(corners_3d[i] - corners_3d[j]))
            if length_3d > 0:
                scales.append(diag_expected / length_3d)
                weights.append(float(min(corner_weights[i], corner_weights[j])))

        if not scales:
            return None, {}

        # If all weights are zero (no stats), fall back to unweighted median.
        if not any(w > 0 for w in weights):
            scale = float(np.median(scales))
            return scale, {"marker_weight": float(len(scales))}

        scale = float(self._weighted_median(scales, weights))
        return scale, {"marker_weight": float(np.sum(weights))}

    def _weighted_median(self, values: list[float], weights: list[float]) -> float:
        """Compute weighted median of values."""
        v = np.asarray(values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)

        if v.size == 0:
            raise ValueError("Empty values for weighted median")

        w = np.maximum(w, 0.0)
        if float(np.sum(w)) <= 0:
            return float(np.median(v))

        idx = np.argsort(v)
        v = v[idx]
        w = w[idx]
        cum = np.cumsum(w)
        cutoff = float(np.sum(w)) / 2.0
        return float(v[int(np.searchsorted(cum, cutoff))])

    def _combine_scales(self, scales: list[float], method: str, weights: list[float] | None = None) -> float:
        """Combine multiple scale estimates.

        :param scales: List of scale estimates.
        :param method: Combination method.
        :param weights: Optional weights (same length as scales).
        :returns: Combined scale factor.
        """
        if weights is not None and len(weights) != len(scales):
            raise ValueError("weights must have the same length as scales")

        if method == "simple":
            return float(np.mean(scales))
        elif method == "median":
            return float(np.median(scales)) if weights is None else self._weighted_median(scales, weights)
        elif method == "robust_mean":
            # Remove outliers using IQR method, then compute (optionally weighted) mean.
            q1, q3 = np.percentile(scales, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if weights is None:
                filtered = [s for s in scales if lower <= s <= upper]
                return float(np.mean(filtered)) if filtered else float(np.mean(scales))

            filtered_scales = []
            filtered_weights = []
            for s, w in zip(scales, weights):
                if lower <= s <= upper:
                    filtered_scales.append(s)
                    filtered_weights.append(w)

            if not filtered_scales:
                filtered_scales = scales
                filtered_weights = weights

            w = np.asarray(filtered_weights, dtype=np.float64)
            w = np.maximum(w, 0.0)
            if float(np.sum(w)) <= 0:
                return float(np.mean(filtered_scales))

            return float(np.average(filtered_scales, weights=w))
        else:
            raise ValueError(f"Unknown method: {method}")

    def validate_scale(
        self,
        scale_factor: float,
        stats: dict,
        max_std_ratio: float = 0.1
    ) -> bool:
        """Validate estimated scale factor.

        :param scale_factor: Estimated scale factor.
        :param stats: Scale statistics.
        :param max_std_ratio: Maximum allowed std/mean ratio.
        :returns: True if scale is valid.
        """
        std_ratio = stats["std"] / stats["mean"]

        logger.info("\nScale Validation:")
        logger.info(f"  Scale factor: {scale_factor:.4f}")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std: {stats['std']:.4f}")
        logger.info(f"  Std/Mean ratio: {std_ratio:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        if std_ratio > max_std_ratio:
            logger.warning(
                f"High variance in scale estimates "
                f"(std/mean = {std_ratio:.2%} > {max_std_ratio:.2%})"
            )
            return False

        if scale_factor <= 0:
            logger.error("Invalid scale factor (must be positive)")
            return False

        if scale_factor < 0.01 or scale_factor > 100:
            logger.warning(
                f"Unusual scale factor {scale_factor:.4f} "
                f"(expected range: 0.01-100)"
            )

        return True
