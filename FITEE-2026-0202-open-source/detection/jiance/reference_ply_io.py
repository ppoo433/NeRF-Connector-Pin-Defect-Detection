from __future__ import annotations

import numpy as np

DEFAULT_GUI_SAMPLES = 240

DEFAULT_GUI_CTRL_POINTS = max(4, DEFAULT_GUI_SAMPLES // 10)

DEFAULT_GUI_SMOOTH = 0.4

def write_ply_xyz_rgb_per_point(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if cols.shape != (len(pts), 3):
        raise ValueError("colors must be (N,3)")

    cols = np.clip(cols, 0, 255).astype(np.int32)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(
                f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
