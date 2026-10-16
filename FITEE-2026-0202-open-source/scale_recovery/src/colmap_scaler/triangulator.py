"""Triangulate marker corners from 2D detections to 3D points."""

import logging
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


class CornerTriangulator:
    """Triangulate marker corners from 2D detections."""

    def __init__(self, reconstruction: pycolmap.Reconstruction) -> None:
        """Initialize triangulator.

        :param reconstruction: COLMAP reconstruction.
        """
        self.reconstruction = reconstruction
        self.image_name_to_id = {
            img.name: img_id for img_id, img in reconstruction.images.items()
        }

    def triangulate_marker_corners(
        self,
        detections: dict,
        min_num_views: int = 2,
        max_reprojection_error: float = 3.0,
        allowed_marker_ids: list[int] | None = None,
        ransac_max_trials: int = 2000,
    ) -> dict[int, dict]:
        """Triangulate all marker corners from 2D detections.

        :param detections: Detection results from JSON (detections dict).
        :param min_num_views: Minimum number of views required to triangulate.
        :param max_reprojection_error: Maximum reprojection error in pixels for inlier selection.
        :param allowed_marker_ids: If provided, only these marker IDs are used.
        :param ransac_max_trials: Maximum RANSAC trials per corner.
        :returns: Dictionary mapping marker_id to corner data.
        """
        allowed_ids = set(allowed_marker_ids) if allowed_marker_ids is not None else None

        # Organize observations by marker ID and corner index
        marker_observations = {}

        for image_name, detection_data in detections.items():
            if image_name not in self.image_name_to_id:
                logger.warning(f"Image {image_name} not found in reconstruction")
                continue

            image_id = self.image_name_to_id[image_name]

            for marker in detection_data.get("markers", []):
                marker_id = marker["id"]
                if allowed_ids is not None and marker_id not in allowed_ids:
                    continue
                corners = marker["corners"]

                if marker_id not in marker_observations:
                    marker_observations[marker_id] = {i: [] for i in range(4)}

                # Add observation for each corner
                for corner_idx, corner_2d in enumerate(corners):
                    marker_observations[marker_id][corner_idx].append(
                        {
                            "image_id": image_id,
                            "point2D": np.array(corner_2d, dtype=np.float64),
                        }
                    )

        # Triangulate each marker's corners
        results = {}
        for marker_id, corners_obs in marker_observations.items():
            corner_3d_points = []
            corner_stats = []
            valid = True
            failure: dict | None = None

            for corner_idx in range(4):
                observations = corners_obs[corner_idx]

                if len(observations) < min_num_views:
                    logger.warning(
                        f"Marker {marker_id} corner {corner_idx} "
                        f"has only {len(observations)} views (need {min_num_views})"
                    )
                    valid = False
                    failure = {
                        "corner_idx": corner_idx,
                        "reason": "insufficient_views",
                        "num_observations": int(len(observations)),
                        "min_num_views": int(min_num_views),
                    }
                    break

                prepared_obs = self._prepare_observations(observations)

                # Deterministic per-corner RNG for reproducibility.
                seed = (int(marker_id) * 1000003 + int(corner_idx) * 9176) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)

                # Triangulate this corner robustly (RANSAC + inlier-only refinement).
                point3D, stats = self._triangulate_point_robust(
                    prepared_obs,
                    min_num_views=min_num_views,
                    max_reprojection_error=max_reprojection_error,
                    ransac_max_trials=ransac_max_trials,
                    rng=rng,
                )

                if point3D is None:
                    logger.warning(
                        f"Failed to triangulate marker {marker_id} corner {corner_idx}"
                    )
                    valid = False
                    failure = {
                        "corner_idx": corner_idx,
                        "reason": "robust_triangulation_failed",
                        "stats": stats,
                    }
                    break

                corner_3d_points.append(point3D.tolist())
                corner_stats.append(stats)

            if valid:
                results[marker_id] = {
                    "corners_3d": corner_3d_points,
                    "corner_stats": corner_stats,
                    "num_observations": [s["num_observations"] for s in corner_stats],
                    "num_inliers": [s["num_inliers"] for s in corner_stats],
                    "reprojection_error_mean": [s["inlier_error_mean"] for s in corner_stats],
                    "reprojection_error_median": [s["inlier_error_median"] for s in corner_stats],
                    "reprojection_error_p90": [s["inlier_error_p90"] for s in corner_stats],
                    "reprojection_error_max": [s["inlier_error_max"] for s in corner_stats],
                    "valid": True,
                }
            else:
                results[marker_id] = {
                    "valid": False,
                    "failure": failure,
                    "partial_corner_stats": corner_stats,
                }

        return results

    def _prepare_observations(self, observations: list[dict]) -> list[dict]:
        """Prepare observations for fast robust triangulation.

        :param observations: List of observations with image_id and point2D.
        :returns: List of prepared observation dicts.
        """
        prepared: list[dict] = []
        for obs in observations:
            image_id = obs["image_id"]
            image = self.reconstruction.images[image_id]
            camera = self.reconstruction.cameras[image.camera_id]

            point2D_distorted = obs["point2D"]
            point2D_normalized = camera.cam_from_img(point2D_distorted)

            cam_from_world = image.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            Rt = np.hstack([R, t.reshape(3, 1)])

            prepared.append(
                {
                    "image_id": image_id,
                    "image_name": image.name,
                    "camera": camera,
                    "cam_from_world": cam_from_world,
                    "proj_matrix_normalized": Rt,
                    "point2D_distorted": point2D_distorted,
                    "point2D_normalized": point2D_normalized,
                }
            )

        return prepared

    def _triangulate_dlt(
        self, proj_matrices: list[np.ndarray], points2D: list[np.ndarray]
    ) -> np.ndarray | None:
        """Triangulate using Direct Linear Transform.

        :param proj_matrices: List of 3x4 projection matrices.
        :param points2D: List of 2D points.
        :returns: 3D point as numpy array or None if failed.
        """
        # Build matrix A for homogeneous linear system
        A = []
        for P, point2D in zip(proj_matrices, points2D):
            x, y = point2D
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])

        A = np.array(A)

        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]  # Convert from homogeneous to 3D
            return X[:3]
        except np.linalg.LinAlgError:
            return None

    def _triangulate_dlt_weighted(
        self,
        proj_matrices: list[np.ndarray],
        points2D: list[np.ndarray],
        weights: list[float] | np.ndarray,
    ) -> np.ndarray | None:
        """Weighted DLT triangulation in normalized coordinates.

        Each observation contributes two rows to A, both scaled by sqrt(weight).
        """
        if len(proj_matrices) != len(points2D) or len(points2D) != len(weights):
            return None

        A = []
        for P, point2D, w in zip(proj_matrices, points2D, weights):
            x, y = point2D
            s = float(np.sqrt(max(float(w), 0.0)))
            A.append(s * (x * P[2] - P[0]))
            A.append(s * (y * P[2] - P[1]))

        A = np.array(A)
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]
            return X[:3]
        except np.linalg.LinAlgError:
            return None

    def _calculate_reprojection_errors_with_distortion(
        self, point3D: np.ndarray, prepared_observations: list[dict]
    ) -> list[float]:
        """Calculate reprojection error per observation (pixels), using camera distortion models."""
        errors: list[float] = []
        for obs in prepared_observations:
            camera = obs["camera"]
            cam_from_world = obs["cam_from_world"]
            point2D_obs = obs["point2D_distorted"]

            point3D_cam = cam_from_world * point3D
            if point3D_cam[2] <= 0:
                errors.append(float("inf"))
                continue

            point2D_proj = camera.img_from_cam(point3D_cam)
            errors.append(float(np.linalg.norm(point2D_proj - point2D_obs)))

        return errors

    def _triangulate_point_robust(
        self,
        prepared_observations: list[dict],
        min_num_views: int,
        max_reprojection_error: float,
        ransac_max_trials: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, dict]:
        """Robustly triangulate a 3D point from multiple views using RANSAC and inlier refinement.

        This is designed for the common "many blurry frames + some sharp frames" scenario:
        it should succeed as long as there exists a consistent inlier set of size >= min_num_views
        under the given max_reprojection_error threshold.
        """
        num_obs = len(prepared_observations)
        stats: dict = {
            "num_observations": num_obs,
            "num_inliers": 0,
            "inlier_error_mean": float("inf"),
            "inlier_error_median": float("inf"),
            "inlier_error_p90": float("inf"),
            "inlier_error_max": float("inf"),
            "worst_outliers": [],
        }

        if num_obs < min_num_views:
            return None, stats

        proj_mats = [o["proj_matrix_normalized"] for o in prepared_observations]
        pts_norm = [o["point2D_normalized"] for o in prepared_observations]

        # RANSAC over 2-view hypotheses.
        best_inliers: list[int] | None = None
        best_inlier_count = 0
        best_inlier_mean = float("inf")

        if num_obs == min_num_views:
            best_inliers = list(range(num_obs))
            best_inlier_count = num_obs
        else:
            max_pairs = num_obs * (num_obs - 1) // 2
            trials = int(min(max(ransac_max_trials, 0), max_pairs))
            if trials <= 0:
                trials = min(2000, max_pairs)

            for _ in range(trials):
                i, j = rng.choice(num_obs, size=2, replace=False).tolist()
                point = self._triangulate_dlt(
                    [proj_mats[i], proj_mats[j]], [pts_norm[i], pts_norm[j]]
                )
                if point is None:
                    continue

                errors = self._calculate_reprojection_errors_with_distortion(
                    point, prepared_observations
                )
                inliers = [k for k, e in enumerate(errors) if e <= max_reprojection_error]
                if len(inliers) < min_num_views:
                    continue

                inlier_mean = float(np.mean([errors[k] for k in inliers]))
                if len(inliers) > best_inlier_count or (
                    len(inliers) == best_inlier_count and inlier_mean < best_inlier_mean
                ):
                    best_inliers = inliers
                    best_inlier_count = len(inliers)
                    best_inlier_mean = inlier_mean
                    if best_inlier_count == num_obs:
                        break

        if best_inliers is None or best_inlier_count < min_num_views:
            return None, stats

        # Re-triangulate on the consensus set, then re-evaluate inliers on *all* observations.
        inliers = sorted(best_inliers)
        point = self._triangulate_dlt(
            [proj_mats[i] for i in inliers], [pts_norm[i] for i in inliers]
        )
        if point is None:
            return None, stats

        # Iteratively (1) recompute inliers under the strict threshold, (2) re-triangulate,
        # and (3) optionally apply a weighted DLT refinement to improve precision.
        eps = 1e-3  # px, prevents exploding weights for ~0 reprojection error
        for _ in range(3):
            errors_all = self._calculate_reprojection_errors_with_distortion(
                point, prepared_observations
            )
            inliers = [k for k, e in enumerate(errors_all) if e <= max_reprojection_error]
            if len(inliers) < min_num_views:
                return None, stats

            proj_in = [proj_mats[i] for i in inliers]
            pts_in = [pts_norm[i] for i in inliers]

            # Base re-triangulation.
            point_new = self._triangulate_dlt(proj_in, pts_in)
            if point_new is None:
                return None, stats

            # One IRLS-style weighted DLT step using pixel-domain reprojection errors as weights.
            errors_in = np.array([errors_all[i] for i in inliers], dtype=np.float64)
            weights = (1.0 / np.maximum(errors_in, eps)).tolist()
            point_w = self._triangulate_dlt_weighted(proj_in, pts_in, weights)
            if point_w is not None:
                point_new = point_w

            if np.linalg.norm(point_new - point) < 1e-9:
                point = point_new
                break
            point = point_new

        # Final stats (IMPORTANT: re-evaluate inliers for the *final* point).
        #
        # Without this, the last inlier set may correspond to a slightly different point
        # than the final refined one, which can make the reported max inlier error exceed
        # the threshold by a small margin and incorrectly reject an otherwise valid marker.
        errors_all = self._calculate_reprojection_errors_with_distortion(point, prepared_observations)
        inliers_final = [k for k, e in enumerate(errors_all) if e <= max_reprojection_error]
        if len(inliers_final) < min_num_views:
            return None, stats

        inliers = inliers_final
        inlier_errors = [errors_all[i] for i in inliers]

        errors_sorted = sorted(
            [
                (prepared_observations[i]["image_name"], errors_all[i])
                for i in range(num_obs)
                if np.isfinite(errors_all[i]) and (i not in inliers)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        stats.update(
            {
                "num_inliers": int(len(inliers)),
                "inlier_error_mean": float(np.mean(inlier_errors)),
                "inlier_error_median": float(np.median(inlier_errors)),
                "inlier_error_p90": float(np.percentile(inlier_errors, 90)),
                "inlier_error_max": float(np.max(inlier_errors)),
                "worst_outliers": [
                    {"image_name": name, "error_px": float(err)}
                    for name, err in errors_sorted[:5]
                ],
            }
        )

        return point, stats

    def validate_triangulated_corners(
        self, triangulated_markers: dict[int, dict], max_reprojection_error: float = 3.0
    ) -> dict[int, dict]:
        """Validate triangulated corners and filter by reprojection error.

        :param triangulated_markers: Triangulation results.
        :param max_reprojection_error: Maximum allowed reprojection error in pixels.
        :returns: Filtered triangulation results.
        """
        validated = {}

        for marker_id, data in triangulated_markers.items():
            if not data.get("valid", False):
                continue

            # Check inlier reprojection errors (per corner).
            max_error = float(max(data.get("reprojection_error_max", [float("inf")])))
            mean_error = float(
                np.mean(data.get("reprojection_error_mean", [float("inf")]))
            )

            if max_error > max_reprojection_error:
                logger.warning(
                    f"Marker {marker_id} has high reprojection error: "
                    f"max={max_error:.2f}px, mean={mean_error:.2f}px"
                )
                continue

            validated[marker_id] = data

        return validated
