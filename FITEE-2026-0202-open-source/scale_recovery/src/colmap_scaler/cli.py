"""Command-line interface for COLMAP scaler."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from .pipeline import ScalingPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class ScalerArgs:
    """COLMAP Scaler - Rescale COLMAP reconstructions using fiducial markers.

    This tool detects ArUco or AprilTag markers in images and uses them
    to rescale COLMAP reconstructions to real-world units.
    """

    input: Path
    """Path to dataset folder or images directory. If path ends with /images, parent folder is used."""

    marker_size: float
    """Physical marker size in meters (e.g., 0.05 for 50mm)."""

    reconstruction: Path | None = None
    """Path to COLMAP reconstruction. Default: dataset_folder/sparse/0"""

    dictionary: str | None = None
    """Marker dictionary (e.g., DICT_4X4_50, DICT_APRILTAG_25h9). Auto-detect if not specified."""

    num_images: int | None = None
    """Process only N randomly selected images. Process all if not specified."""

    image_list: Path | None = None
    """Text file with specific image names to process (one per line)."""

    visualize: bool = False
    """Save visualization images with detected markers in dataset_folder/tags/."""

    refine_corners: bool = True
    """Apply subpixel corner refinement for better accuracy."""

    min_perimeter: int = 50
    """Minimum marker perimeter in pixels."""

    max_perimeter: int = 10000
    """Maximum marker perimeter in pixels."""

    output_name: str | None = None
    """Output folder name under sparse/. Default: 0"""

    backup_name: str | None = None
    """Backup folder name. Default: {output_name}_before_scale"""

    min_num_views: int = 2
    """Minimum number of views required to triangulate a corner."""

    max_reprojection_error: float = 3.0
    """Maximum reprojection error in pixels."""

    allowed_marker_ids: list[int] = field(default_factory=lambda: [0, 1])
    """Only use these marker IDs. Default: [0, 1]. Set to an empty list to allow all IDs."""

    ransac_max_trials: int = 2000
    """Maximum RANSAC trials per corner for robust triangulation."""

    export_json: bool = False
    """Export scale_report.json and detections.json files."""

    # Anchor selection (optional, strict by design)
    auto_anchors: bool = False
    """Auto-select high-quality anchor frames from detections (recommended for low-angle/occlusion datasets)."""

    anchors_only: bool = False
    """If set, use ONLY anchor frames for triangulation/scale (prefer failure over biased scale)."""

    anchors_file: Path | None = None
    """Optional text file with anchor image names (one per line). Overrides auto_anchors selection."""

    anchors_max_both: int = 30
    """Max number of anchors that contain BOTH marker IDs (if available)."""

    anchors_max_per_marker: int = 30
    """Max number of anchors per marker ID (top-K by quality)."""

    anchors_max_total: int = 80
    """Max total anchors after combining both-markers and per-marker lists."""

    anchors_min_side_px: float = 0.0
    """Hard gate: reject detections where the shortest quad side is smaller than this (0 disables)."""

    anchors_min_border_px: float = 0.0
    """Hard gate: reject detections with corners too close to image border (0 disables)."""

    anchors_use_sharpness: bool = True
    """Include a Laplacian-variance sharpness term in anchor scoring."""


def main() -> None:
    """Main entry point for COLMAP Scaler."""
    args = tyro.cli(ScalerArgs)

    # Triangulation requires at least two views.
    if args.min_num_views < 2:
        raise ValueError("--min-num-views must be >= 2 for triangulation")

    # Empty list means "no filtering".
    if args.allowed_marker_ids == []:
        args.allowed_marker_ids = None

    # Create and run pipeline
    pipeline = ScalingPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
