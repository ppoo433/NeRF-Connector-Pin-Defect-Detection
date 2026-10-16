from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ColorConstancyConfig:
    """
    Lightweight per-cloud color constancy for point clouds (RGB only).

    We keep this intentionally simple and fully CPU/numpy so it can run both locally
    and on the server without extra deps.

    References:
      - Shades-of-Gray / Minkowski framework: 资料.md R81
      - OpenCV GrayworldWB saturation definition & exclusion idea: 资料.md R87/R88
    """

    method: str = "shades_of_gray"  # "gray_world" | "shades_of_gray"
    p: float = 6.0  # Minkowski p (p=1 -> Gray-World)
    sat_threshold: float = 0.90  # exclude points with saturation > thr when estimating illuminant
    eps: float = 1e-6
    max_gain: float = 4.0  # clamp per-channel gain for safety

    # Reliability / soft switch (derived from dataset stats; see plan.md evidence).
    soft_cast_ratio_mean: float = 1.08
    hard_cast_ratio_mean: float = 1.12
    soft_sat_p95: float = 0.30
    hard_sat_p95: float = 0.45
    soft_alpha: float = 0.5  # blend strength under "uncertain" conditions


def _rgb_f32(rgb_u8: np.ndarray) -> np.ndarray:
    return (rgb_u8.astype(np.float32, copy=False) / 255.0).astype(np.float32, copy=False)


def compute_saturation(rgb_u8: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """
    OpenCV GrayworldWB saturation definition:
      saturation = 1 - min(R,G,B) / max(R,G,B), if max != 0 else 0.
    """
    rgb = _rgb_f32(rgb_u8)
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    sat = np.zeros((rgb.shape[0],), dtype=np.float32)
    nz = mx > float(eps)
    sat[nz] = 1.0 - (mn[nz] / mx[nz])
    return sat


def _estimate_illuminant(rgb_u8: np.ndarray, *, cfg: ColorConstancyConfig) -> Tuple[np.ndarray, float, float, float]:
    rgb = _rgb_f32(rgb_u8)
    sat = compute_saturation(rgb_u8, eps=cfg.eps)
    mask = sat <= float(cfg.sat_threshold)
    used_frac = float(np.mean(mask)) if mask.size > 0 else 0.0

    # Fallback if everything is excluded.
    if not np.any(mask):
        illum = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return illum, used_frac, float(np.nan), float(np.nan)

    rgb_sel = rgb[mask]
    p = 1.0 if cfg.method == "gray_world" else float(max(cfg.p, 1e-3))

    if p == 1.0:
        illum = np.mean(rgb_sel, axis=0).astype(np.float32, copy=False)
    else:
        illum = (np.mean(np.power(np.clip(rgb_sel, 0.0, 1.0), p), axis=0) + float(cfg.eps)) ** (1.0 / p)
        illum = illum.astype(np.float32, copy=False)

    mean_illum = float(np.mean(illum))
    # Gains that push the estimated illuminant towards neutral grey.
    gains = (mean_illum / np.maximum(illum, float(cfg.eps))).astype(np.float32, copy=False)
    gains = np.clip(gains, 1.0 / float(cfg.max_gain), float(cfg.max_gain)).astype(np.float32, copy=False)

    cast_ratio = float(np.max(gains) / max(float(np.min(gains)), float(cfg.eps)))
    sat_p95 = float(np.quantile(sat, 0.95)) if sat.size > 0 else float("nan")
    return gains, used_frac, cast_ratio, sat_p95


def apply_color_constancy(rgb_u8: np.ndarray, *, cfg: ColorConstancyConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply per-cloud color constancy with a soft reliability switch.

    Returns:
      - rgb_out_u8: uint8[N,3]
      - debug: float-only dict (so callers can log it into metrics)
    """
    rgb_u8 = np.asarray(rgb_u8)
    if rgb_u8.ndim != 2 or rgb_u8.shape[1] != 3:
        raise ValueError(f"rgb_u8 must have shape [N,3], got {rgb_u8.shape}")
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = rgb_u8.astype(np.uint8, copy=False)

    # Basic stats for the reliability gate (cheap, deterministic).
    mean_rgb = np.mean(rgb_u8.astype(np.float32), axis=0)
    mean_min = float(np.min(mean_rgb))
    mean_max = float(np.max(mean_rgb))
    cast_ratio_mean = (mean_max / max(mean_min, float(cfg.eps))) if mean_min > 0 else float("inf")

    gains, used_frac, gain_ratio, sat_p95 = _estimate_illuminant(rgb_u8, cfg=cfg)

    # Decide blend alpha (0 => disabled, 1 => fully applied).
    alpha = 1.0
    if cast_ratio_mean >= float(cfg.hard_cast_ratio_mean) or sat_p95 >= float(cfg.hard_sat_p95):
        alpha = 0.0
    else:
        if cast_ratio_mean >= float(cfg.soft_cast_ratio_mean):
            alpha = min(alpha, float(cfg.soft_alpha))
        if sat_p95 >= float(cfg.soft_sat_p95):
            alpha = min(alpha, float(cfg.soft_alpha))

    rgb_f = _rgb_f32(rgb_u8)
    rgb_corr = np.clip(rgb_f * gains[None, :], 0.0, 1.0)
    if alpha <= 0.0:
        rgb_out = rgb_u8
    else:
        # Soft blend to reduce the risk of over-correction.
        rgb_blend = np.clip((1.0 - alpha) * rgb_f + alpha * rgb_corr, 0.0, 1.0)
        rgb_out = (rgb_blend * 255.0 + 0.5).astype(np.uint8, copy=False)

    debug = {
        "cc_alpha": float(alpha),
        "cc_method_is_gray_world": float(cfg.method == "gray_world"),
        "cc_p": float(1.0 if cfg.method == "gray_world" else float(cfg.p)),
        "cc_sat_threshold": float(cfg.sat_threshold),
        "cc_used_frac": float(used_frac),
        "cc_cast_ratio_mean": float(cast_ratio_mean),
        "cc_sat_p95": float(sat_p95),
        "cc_gain_r": float(gains[0]),
        "cc_gain_g": float(gains[1]),
        "cc_gain_b": float(gains[2]),
        "cc_gain_ratio": float(gain_ratio),
    }
    return rgb_out, debug

