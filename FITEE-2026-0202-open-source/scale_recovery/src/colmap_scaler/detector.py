"""Marker detection using OpenCV."""

import logging
from typing import Any
import cv2
import numpy as np
from pathlib import Path

from .config import DetectionConfig, DetectionResult, MarkerType

logger = logging.getLogger(__name__)


class MarkerDetector:
    """Detect ArUco markers in images using OpenCV."""

    def __init__(self, config: DetectionConfig) -> None:
        """Initialize marker detector.

        :param config: Detection configuration.
        """
        self.config = config
        self.detector = None
        self.dictionary = None
        self.dictionary_name = None
        # Cache detectors per (dictionary, image_max_dim). Use Any to avoid runtime
        # type issues across OpenCV builds.
        self._detector_cache: dict[tuple[str, int], Any] = {}
        self._allowed_marker_ids = (
            set(config.allowed_marker_ids) if config.allowed_marker_ids is not None else None
        )

        # Initialize detector (both ArUco and AprilTag use OpenCV ArUco detector)
        if config.marker_type in (MarkerType.ARUCO, MarkerType.APRILTAG):
            self._setup_aruco_detector()
        else:
            raise NotImplementedError(f"Marker type {config.marker_type} not yet implemented")

    def _setup_aruco_detector(self) -> None:
        """Setup ArUco detector with specified or auto-detected dictionary."""
        if self.config.dictionary:
            # Use specified dictionary
            self.dictionary_name = self.config.dictionary
            self.dictionary = self._get_aruco_dictionary(self.dictionary_name)
            # Detector is created lazily per image size to correctly interpret
            # min/max perimeter values (configured in pixels) as OpenCV "rate" parameters.
            self.detector = None
        else:
            # Auto-detection will be done per image
            self.dictionary = None
            self.detector = None

    def _get_aruco_dictionary(self, dict_name: str):
        """Get OpenCV ArUco dictionary by name.

        :param dict_name: Name of the dictionary.
        :returns: OpenCV ArUco dictionary object.
        """
        # Map string names to OpenCV constants
        dict_mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        }

        if dict_name not in dict_mapping:
            raise ValueError(f"Unknown dictionary: {dict_name}")

        return cv2.aruco.getPredefinedDictionary(dict_mapping[dict_name])

    def _create_aruco_detector(self, dictionary, dictionary_name: str, image_max_dim: int):
        """Create ArUco detector with parameters.

        :param dictionary: OpenCV ArUco dictionary.
        :param dictionary_name: Dictionary name (used to detect AprilTag variants).
        :param image_max_dim: max(image_width, image_height) for converting pixel perimeter to rate.
        :returns: Configured ArUco detector.
        """
        parameters = cv2.aruco.DetectorParameters()

        # Check if this is an AprilTag dictionary
        is_apriltag = "APRILTAG" in dictionary_name.upper()

        # Configure corner refinement
        if self.config.refine_corners:
            # Use APRILTAG refinement for AprilTags
            if is_apriltag:
                parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            else:
                parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            parameters.cornerRefinementWinSize = 5
            parameters.cornerRefinementMaxIterations = 30
            parameters.cornerRefinementMinAccuracy = 0.1

        # Set perimeter constraints.
        #
        # OpenCV uses *rates* relative to max(image_width, image_height):
        #   minMarkerPerimeterRate = min_perimeter_px / max_dim_px
        #   maxMarkerPerimeterRate = max_perimeter_px / max_dim_px
        # (see OpenCV aruco::DetectorParameters docs).
        #
        # Our CLI/config values are expressed in pixels, so we convert them per image size.
        if image_max_dim <= 0:
            # Fallback to OpenCV defaults
            min_rate = parameters.minMarkerPerimeterRate
            max_rate = parameters.maxMarkerPerimeterRate
        else:
            min_rate = float(self.config.min_marker_perimeter) / float(image_max_dim)
            max_rate = float(self.config.max_marker_perimeter) / float(image_max_dim)

        # Clamp to a sensible range for stability (a marker fully spanning the image has ~4*max_dim perimeter).
        min_rate = float(np.clip(min_rate, 0.0, 4.0))
        max_rate = float(np.clip(max_rate, 0.0, 4.0))
        if max_rate < min_rate:
            logger.warning(
                "max_perimeter < min_perimeter after normalization; "
                f"clamping max_rate to min_rate ({max_rate:.4f} -> {min_rate:.4f})"
            )
            max_rate = min_rate

        parameters.minMarkerPerimeterRate = min_rate
        parameters.maxMarkerPerimeterRate = max_rate

        # AprilTag-specific parameters
        if is_apriltag:
            parameters.useAruco3Detection = False  # Better for AprilTags
            parameters.adaptiveThreshConstant = 7

        # Create detector
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    def _get_aruco_detector_for_image(self, dict_name: str, image_max_dim: int):
        """Get (or create) an ArUco detector tuned for a specific image size."""
        key = (dict_name, int(image_max_dim))
        detector = self._detector_cache.get(key)
        if detector is not None:
            return detector

        dictionary = self._get_aruco_dictionary(dict_name)
        detector = self._create_aruco_detector(dictionary, dict_name, image_max_dim)
        self._detector_cache[key] = detector
        return detector

    def auto_detect_dictionary(self, image: np.ndarray) -> str | None:
        """Auto-detect which ArUco dictionary to use.

        :param image: Input image as numpy array.
        :returns: Name of detected dictionary or None if no markers found.
        """
        # Try common dictionaries in order of likelihood
        # Put AprilTag dictionaries first since user mentioned AprilTag 25h9
        common_dicts = [
            "DICT_APRILTAG_25h9",
            "DICT_APRILTAG_36h11",
            "DICT_APRILTAG_36h10",
            "DICT_APRILTAG_16h5",
            "DICT_4X4_50",
            "DICT_5X5_100",
            "DICT_6X6_250",
            "DICT_ARUCO_ORIGINAL",
        ]

        max_detections = 0
        best_dict = None

        image_max_dim = int(max(image.shape[:2])) if image is not None else 0

        for dict_name in common_dicts:
            try:
                detector = self._get_aruco_detector_for_image(dict_name, image_max_dim)
                corners, ids, _ = detector.detectMarkers(image)

                if ids is not None and len(ids) > max_detections:
                    max_detections = len(ids)
                    best_dict = dict_name
            except Exception:
                continue

        if best_dict:
            logger.info(f"Auto-detected dictionary: {best_dict} (found {max_detections} markers)")

        return best_dict

    def detect_in_image(self, image_path: Path) -> DetectionResult:
        """Detect markers in a single image.

        :param image_path: Path to the image file.
        :returns: Detection result for the image.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return DetectionResult(image_name=image_path.name)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_max_dim = int(max(gray.shape[:2]))

        # Auto-detect dictionary if needed, and always build a detector for the current image size.
        dict_name = self.dictionary_name
        if dict_name is None:
            dict_name = self.auto_detect_dictionary(gray)
            if dict_name is None:
                return DetectionResult(image_name=image_path.name)
            self.dictionary_name = dict_name

        detector = self._get_aruco_detector_for_image(dict_name, image_max_dim)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        # Create result
        result = DetectionResult(image_name=image_path.name)

        if ids is not None:
            for i, marker_id in enumerate(ids):
                marker_id_int = int(marker_id[0])
                if self._allowed_marker_ids is not None and marker_id_int not in self._allowed_marker_ids:
                    continue
                # Convert corners to list format
                # corners[i][0] has shape (4, 2) with corners in order: TL, TR, BR, BL
                corner_list = corners[i][0].tolist()
                result.add_marker(marker_id_int, corner_list)

        return result

    def visualize_detection(self, image_path: Path, result: DetectionResult) -> np.ndarray | None:
        """Create visualization of detected markers.

        :param image_path: Path to the original image.
        :param result: Detection result.
        :returns: Annotated image or None if failed.
        """
        if not self.config.visualize:
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # Draw each detected marker
        for marker in result.markers:
            corners = np.array(marker["corners"], dtype=np.float32).reshape((1, 4, 2))
            marker_id = marker["id"]

            # Draw marker outline
            cv2.polylines(image, [corners.astype(np.int32)], True, (0, 255, 0), 2)

            # Draw corner points
            for corner in corners[0]:
                cv2.circle(image, tuple(corner.astype(int)), 5, (255, 0, 0), -1)

            # Draw ID
            center = corners[0].mean(axis=0).astype(int)
            cv2.putText(image, f"ID: {marker_id}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return image

    def save_visualization(self, image_path: Path, result: DetectionResult) -> None:
        """Save visualization to file.

        :param image_path: Path to the original image.
        :param result: Detection result.
        """
        if not self.config.visualize:
            return

        annotated = self.visualize_detection(image_path, result)
        if annotated is None:
            return

        # Create parallel 'tags' directory next to 'images' directory
        # If images are in /path/to/dataset/images/, save to /path/to/dataset/tags/
        tags_dir = self.config.images_path.parent / "tags"

        # Get relative path from images directory
        rel_path = image_path.relative_to(self.config.images_path)
        output_path = tags_dir / rel_path

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save annotated image
        cv2.imwrite(str(output_path), annotated)

    def process_image_list(self, image_paths: list[Path]) -> dict[str, DetectionResult]:
        """Process a list of images.

        :param image_paths: List of image paths to process.
        :returns: Dictionary mapping image names to detection results.
        """
        results = {}

        logger.info(f"Detecting markers in {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths, 1):
            result = self.detect_in_image(image_path)
            results[image_path.name] = result

            if self.config.visualize and result.has_markers():
                self.save_visualization(image_path, result)

            if i % 10 == 0:
                logger.info(f"  Processed {i}/{len(image_paths)} images")

        return results
