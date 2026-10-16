from __future__ import annotations

import sys
import os
import csv
import copy
import math
from pathlib import Path

# Allow running this file directly (python dl_pn2_msg/train_pn2_msg.py) while importing repo-root modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from daima_runtime.runlog import setup_run
from daima_runtime.split_utils import resolve_split_csv
from color_constancy import ColorConstancyConfig
from dl_pn2_msg.common import (
    build_block_data,
    build_cloud,
    choose_seeds,
    knn_block_indices,
    load_split_csv,
    safe_cc_cfg_dict,
)
from dl_pn2_msg.model import PointNet2MSG, PointNet2MSGConfig


try:
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch + PyG are required for dl_pn2_msg/train_pn2_msg.py. "
        "Run inside the server docker image fenge_tp3d."
    ) from e


_DAIMA_CTX = setup_run(__file__)
logger = logging.getLogger("dl_pn2_msg.train_pn2_msg")


def _get_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(str(v).strip())
    except Exception as e:
        raise ValueError(f"Invalid env {name}={v!r} (expected int)") from e


def _get_env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None:
        return str(default)
    return str(v)


def _get_env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except Exception as e:
        raise ValueError(f"Invalid env {name}={v!r} (expected float)") from e


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid env {name}={v!r} (expected bool like 1/0/true/false)")


# ------------------------------
# Editable defaults (PyCharm-run friendly)
# ------------------------------
DATA_DIR = Path(os.environ.get("DAIMA_POINT_CLOUD_DIR") or str(_DAIMA_CTX.repo_root / "private_inputs")).expanduser()
SPLIT_CSV = resolve_split_csv(_DAIMA_CTX.repo_root)

FOLD_ID = 0
RANDOM_SEED = 20260126

# Feature ablation (2x2): density on/off 脳 rgb on/off
USE_RGB = True
USE_DENSITY = True
USE_COLOR_CONSTANCY = False  # only meaningful when USE_RGB=True
USE_NORMALS = False
USE_GLOBAL_XYZ = False  # global xyz normalized per-cloud (pos is block-normalized)

# Color augmentation (train-only; Step 4). Defaults keep behavior unchanged.
# PointNeXt reports random color drop improves robustness; here we implement simple drop + jitter.
COLOR_DROPOUT_BLOCK_P = 0.0  # with prob p, set all rgb in the block to 0
COLOR_DROPOUT_POINT_P = 0.0  # with prob p per point, set rgb to 0
COLOR_JITTER_SCALE = 0.0  # multiplicative jitter: rgb *= (1 + u[-s, +s])
COLOR_JITTER_SHIFT = 0.0  # additive jitter: rgb += u[-b, +b]
COLOR_JITTER_NOISE_STD = 0.0  # gaussian noise std (added to rgb)

# Density features (per-cloud)
DENSITY_K = 16
DENSITY_K2 = 32
DENSITY_MODE = "knn_mean_std"  # see dl_pn2_msg/common.py:density_features
DENSITY_NN_MEDIAN_MODE = "random"  # random | deterministic
DENSITY_MAX_POINTS_FOR_NN_MED = 50000

# Density usage (train-only; P2-9 density prior variants)
# Note: these are meant to improve robustness to non-uniform sampling density (PointConv/PointNorm style),
# not to turn density into a brittle class cue.
DENSITY_SEED_SAMPLING_MODE = "uniform"  # uniform | inv_density
DENSITY_SEED_SAMPLING_POWER = 1.0

DENSITY_LOSS_REWEIGHT_MODE = "none"  # none | inv_density
DENSITY_LOSS_REWEIGHT_K = 16
DENSITY_LOSS_REWEIGHT_POWER = 1.0
DENSITY_LOSS_REWEIGHT_CLIP_MIN = 0.5
DENSITY_LOSS_REWEIGHT_CLIP_MAX = 2.0

# Block training
BLOCK_POINTS = 4096
TRAIN_BLOCKS_PER_CLOUD = 8
VAL_BLOCKS_PER_CLOUD = 8

# Model
MODEL_CFG = PointNet2MSGConfig()

# Optim
EPOCHS = 1  # short smoke by default; increase later
BATCH_SIZE_BLOCKS = 4

# DataLoader (perf; train-only). Defaults keep the original behavior (single-process).
DL_NUM_WORKERS = 0
DL_PIN_MEMORY = False
DL_PERSISTENT_WORKERS = False
DL_PREFETCH_FACTOR = 2  # only used when DL_NUM_WORKERS > 0

LR = 1e-3
WEIGHT_DECAY = 1e-4

# EMA (exponential moving average of model weights; Step 3C stability).
# 0 disables EMA and keeps the original behavior.
EMA_DECAY = 0.0

# Cloud-gate validation & checkpoint selection (Step 3C-4a; cloud-level metrics).
# Off by default to avoid slowing down baseline training.
ENABLE_CLOUD_GATE_EVAL = False
CLOUD_GATE_EVAL_EVERY = 5  # when enabled, evaluate every N epochs (and always at last epoch)
CLOUD_GATE_PIN_IOU_THR = 0.90
CLOUD_GATE_BASE_FP_THR = 0.05

# Loss (Step 3B; default keeps the existing behavior)
LOSS_TYPE = "bce_pos_weight"  # bce_pos_weight | bce_asymm | tversky | focal_tversky
POS_WEIGHT_MODE = "auto"  # auto | fixed
POS_WEIGHT_FIXED = 1.0

# Asymmetric BCE weights: FP is controlled via w_neg (penalize negatives predicted as positive).
BCE_ASYMM_W_POS = 1.0
BCE_ASYMM_W_NEG = 1.0

# Tversky: TI = TP / (TP + alpha*FP + beta*FN). Larger alpha => FP more expensive.
TVERSKY_ALPHA = 0.5
TVERSKY_BETA = 0.5
FOCAL_TVERSKY_GAMMA = 1.0

# NP/low-FPR alignment: Top-α negative tail loss (engineering proxy for pAUC@low-FPR; 资料.md:R129/R130).
# Set lambda=0 to disable (default keeps behavior unchanged).
TAIL_NEG_ALPHA = 0.05
TAIL_NEG_LAMBDA = 0.0

# NP/low-FPR alignment (P2-11; 研究报告/gpt5pro调研报告.md Top-1; 资料.md:R134/R129~R131):
# Soft-FN penalty at the per-group threshold t_s(beta) (beta≈target FPR), plus optional group-wise tail-neg.
#
# All coefficients are hyperparameters (teacher requirement R134): keep defaults disabled to preserve baseline.
NP_BETA_FPR = 0.05
NP_LAMBDA = 0.0
NP_TAU = 0.1
NP_MIN_BASE_PTS = 64
NP_MIN_PIN_PTS = 64

# Grouping for NP and tail losses:
# - global: across the whole batch (fast; but mixes clouds)
# - block: per block/graph in the batch (batch_data.batch)
# - sid: per cloud id when available (batch_data.sid); falls back to block
NP_GROUP_MODE = "sid"
TAIL_NEG_GROUP_MODE = "global"

# Env overrides (no code edit). Useful for Step3C/P2-3 automation on server.
_ENV_FOLD_ID = "DAIMA_PN2_MSG_FOLD_ID"
_ENV_RANDOM_SEED = "DAIMA_PN2_MSG_RANDOM_SEED"
_ENV_EPOCHS = "DAIMA_PN2_MSG_EPOCHS"
_ENV_USE_RGB = "DAIMA_PN2_MSG_USE_RGB"
_ENV_USE_DENSITY = "DAIMA_PN2_MSG_USE_DENSITY"
_ENV_USE_COLOR_CONSTANCY = "DAIMA_PN2_MSG_USE_COLOR_CONSTANCY"
_ENV_USE_NORMALS = "DAIMA_PN2_MSG_USE_NORMALS"
_ENV_USE_GLOBAL_XYZ = "DAIMA_PN2_MSG_USE_GLOBAL_XYZ"
_ENV_COLOR_DROPOUT_BLOCK_P = "DAIMA_PN2_MSG_COLOR_DROPOUT_BLOCK_P"
_ENV_COLOR_DROPOUT_POINT_P = "DAIMA_PN2_MSG_COLOR_DROPOUT_POINT_P"
_ENV_COLOR_JITTER_SCALE = "DAIMA_PN2_MSG_COLOR_JITTER_SCALE"
_ENV_COLOR_JITTER_SHIFT = "DAIMA_PN2_MSG_COLOR_JITTER_SHIFT"
_ENV_COLOR_JITTER_NOISE_STD = "DAIMA_PN2_MSG_COLOR_JITTER_NOISE_STD"
_ENV_DENSITY_K = "DAIMA_PN2_MSG_DENSITY_K"
_ENV_DENSITY_K2 = "DAIMA_PN2_MSG_DENSITY_K2"
_ENV_DENSITY_MODE = "DAIMA_PN2_MSG_DENSITY_MODE"
_ENV_DENSITY_NN_MEDIAN_MODE = "DAIMA_PN2_MSG_DENSITY_NN_MEDIAN_MODE"
_ENV_DENSITY_SEED_SAMPLING_MODE = "DAIMA_PN2_MSG_DENSITY_SEED_SAMPLING_MODE"
_ENV_DENSITY_SEED_SAMPLING_POWER = "DAIMA_PN2_MSG_DENSITY_SEED_SAMPLING_POWER"
_ENV_DENSITY_LOSS_REWEIGHT_MODE = "DAIMA_PN2_MSG_DENSITY_LOSS_REWEIGHT_MODE"
_ENV_DENSITY_LOSS_REWEIGHT_K = "DAIMA_PN2_MSG_DENSITY_LOSS_REWEIGHT_K"
_ENV_DENSITY_LOSS_REWEIGHT_POWER = "DAIMA_PN2_MSG_DENSITY_LOSS_REWEIGHT_POWER"
_ENV_DENSITY_LOSS_REWEIGHT_CLIP_MIN = "DAIMA_PN2_MSG_DENSITY_LOSS_REWEIGHT_CLIP_MIN"
_ENV_DENSITY_LOSS_REWEIGHT_CLIP_MAX = "DAIMA_PN2_MSG_DENSITY_LOSS_REWEIGHT_CLIP_MAX"
_ENV_BLOCK_POINTS = "DAIMA_PN2_MSG_BLOCK_POINTS"
_ENV_TRAIN_BLOCKS_PER_CLOUD = "DAIMA_PN2_MSG_TRAIN_BLOCKS_PER_CLOUD"
_ENV_VAL_BLOCKS_PER_CLOUD = "DAIMA_PN2_MSG_VAL_BLOCKS_PER_CLOUD"
_ENV_BATCH_SIZE_BLOCKS = "DAIMA_PN2_MSG_BATCH_SIZE_BLOCKS"
_ENV_DL_NUM_WORKERS = "DAIMA_PN2_MSG_DL_NUM_WORKERS"
_ENV_DL_PIN_MEMORY = "DAIMA_PN2_MSG_DL_PIN_MEMORY"
_ENV_DL_PERSISTENT_WORKERS = "DAIMA_PN2_MSG_DL_PERSISTENT_WORKERS"
_ENV_DL_PREFETCH_FACTOR = "DAIMA_PN2_MSG_DL_PREFETCH_FACTOR"
_ENV_LR = "DAIMA_PN2_MSG_LR"
_ENV_WEIGHT_DECAY = "DAIMA_PN2_MSG_WEIGHT_DECAY"
_ENV_EMA_DECAY = "DAIMA_PN2_MSG_EMA_DECAY"
_ENV_ENABLE_CLOUD_GATE_EVAL = "DAIMA_PN2_MSG_ENABLE_CLOUD_GATE_EVAL"
_ENV_CLOUD_GATE_EVAL_EVERY = "DAIMA_PN2_MSG_CLOUD_GATE_EVAL_EVERY"
_ENV_CLOUD_GATE_PIN_IOU_THR = "DAIMA_PN2_MSG_CLOUD_GATE_PIN_IOU_THR"
_ENV_CLOUD_GATE_BASE_FP_THR = "DAIMA_PN2_MSG_CLOUD_GATE_BASE_FP_THR"
_ENV_LOSS_TYPE = "DAIMA_PN2_MSG_LOSS_TYPE"
_ENV_POS_WEIGHT_MODE = "DAIMA_PN2_MSG_POS_WEIGHT_MODE"
_ENV_POS_WEIGHT_FIXED = "DAIMA_PN2_MSG_POS_WEIGHT_FIXED"
_ENV_BCE_ASYMM_W_POS = "DAIMA_PN2_MSG_BCE_ASYMM_W_POS"
_ENV_BCE_ASYMM_W_NEG = "DAIMA_PN2_MSG_BCE_ASYMM_W_NEG"
_ENV_TVERSKY_ALPHA = "DAIMA_PN2_MSG_TVERSKY_ALPHA"
_ENV_TVERSKY_BETA = "DAIMA_PN2_MSG_TVERSKY_BETA"
_ENV_FOCAL_TVERSKY_GAMMA = "DAIMA_PN2_MSG_FOCAL_TVERSKY_GAMMA"
_ENV_TAIL_NEG_ALPHA = "DAIMA_PN2_MSG_TAIL_NEG_ALPHA"
_ENV_TAIL_NEG_LAMBDA = "DAIMA_PN2_MSG_TAIL_NEG_LAMBDA"
_ENV_TAIL_NEG_GROUP_MODE = "DAIMA_PN2_MSG_TAIL_NEG_GROUP_MODE"

_ENV_NP_BETA_FPR = "DAIMA_PN2_MSG_NP_BETA_FPR"
_ENV_NP_LAMBDA = "DAIMA_PN2_MSG_NP_LAMBDA"
_ENV_NP_TAU = "DAIMA_PN2_MSG_NP_TAU"
_ENV_NP_MIN_BASE_PTS = "DAIMA_PN2_MSG_NP_MIN_BASE_PTS"
_ENV_NP_MIN_PIN_PTS = "DAIMA_PN2_MSG_NP_MIN_PIN_PTS"
_ENV_NP_GROUP_MODE = "DAIMA_PN2_MSG_NP_GROUP_MODE"

FOLD_ID = _get_env_int(_ENV_FOLD_ID, int(FOLD_ID))
RANDOM_SEED = _get_env_int(_ENV_RANDOM_SEED, int(RANDOM_SEED))
EPOCHS = _get_env_int(_ENV_EPOCHS, int(EPOCHS))
USE_RGB = _get_env_bool(_ENV_USE_RGB, bool(USE_RGB))
USE_DENSITY = _get_env_bool(_ENV_USE_DENSITY, bool(USE_DENSITY))
USE_COLOR_CONSTANCY = _get_env_bool(_ENV_USE_COLOR_CONSTANCY, bool(USE_COLOR_CONSTANCY))
USE_NORMALS = _get_env_bool(_ENV_USE_NORMALS, bool(USE_NORMALS))
USE_GLOBAL_XYZ = _get_env_bool(_ENV_USE_GLOBAL_XYZ, bool(USE_GLOBAL_XYZ))
COLOR_DROPOUT_BLOCK_P = _get_env_float(_ENV_COLOR_DROPOUT_BLOCK_P, float(COLOR_DROPOUT_BLOCK_P))
COLOR_DROPOUT_POINT_P = _get_env_float(_ENV_COLOR_DROPOUT_POINT_P, float(COLOR_DROPOUT_POINT_P))
COLOR_JITTER_SCALE = _get_env_float(_ENV_COLOR_JITTER_SCALE, float(COLOR_JITTER_SCALE))
COLOR_JITTER_SHIFT = _get_env_float(_ENV_COLOR_JITTER_SHIFT, float(COLOR_JITTER_SHIFT))
COLOR_JITTER_NOISE_STD = _get_env_float(_ENV_COLOR_JITTER_NOISE_STD, float(COLOR_JITTER_NOISE_STD))

DENSITY_K = _get_env_int(_ENV_DENSITY_K, int(DENSITY_K))
DENSITY_K2 = _get_env_int(_ENV_DENSITY_K2, int(DENSITY_K2))
DENSITY_MODE = _get_env_str(_ENV_DENSITY_MODE, str(DENSITY_MODE)).strip()
DENSITY_NN_MEDIAN_MODE = _get_env_str(_ENV_DENSITY_NN_MEDIAN_MODE, str(DENSITY_NN_MEDIAN_MODE)).strip()
DENSITY_SEED_SAMPLING_MODE = _get_env_str(_ENV_DENSITY_SEED_SAMPLING_MODE, str(DENSITY_SEED_SAMPLING_MODE)).strip()
DENSITY_SEED_SAMPLING_POWER = _get_env_float(_ENV_DENSITY_SEED_SAMPLING_POWER, float(DENSITY_SEED_SAMPLING_POWER))
DENSITY_LOSS_REWEIGHT_MODE = _get_env_str(_ENV_DENSITY_LOSS_REWEIGHT_MODE, str(DENSITY_LOSS_REWEIGHT_MODE)).strip()
DENSITY_LOSS_REWEIGHT_K = _get_env_int(_ENV_DENSITY_LOSS_REWEIGHT_K, int(DENSITY_LOSS_REWEIGHT_K))
DENSITY_LOSS_REWEIGHT_POWER = _get_env_float(_ENV_DENSITY_LOSS_REWEIGHT_POWER, float(DENSITY_LOSS_REWEIGHT_POWER))
DENSITY_LOSS_REWEIGHT_CLIP_MIN = _get_env_float(
    _ENV_DENSITY_LOSS_REWEIGHT_CLIP_MIN, float(DENSITY_LOSS_REWEIGHT_CLIP_MIN)
)
DENSITY_LOSS_REWEIGHT_CLIP_MAX = _get_env_float(
    _ENV_DENSITY_LOSS_REWEIGHT_CLIP_MAX, float(DENSITY_LOSS_REWEIGHT_CLIP_MAX)
)
BLOCK_POINTS = _get_env_int(_ENV_BLOCK_POINTS, int(BLOCK_POINTS))
TRAIN_BLOCKS_PER_CLOUD = _get_env_int(_ENV_TRAIN_BLOCKS_PER_CLOUD, int(TRAIN_BLOCKS_PER_CLOUD))
VAL_BLOCKS_PER_CLOUD = _get_env_int(_ENV_VAL_BLOCKS_PER_CLOUD, int(VAL_BLOCKS_PER_CLOUD))
BATCH_SIZE_BLOCKS = _get_env_int(_ENV_BATCH_SIZE_BLOCKS, int(BATCH_SIZE_BLOCKS))
DL_NUM_WORKERS = _get_env_int(_ENV_DL_NUM_WORKERS, int(DL_NUM_WORKERS))
DL_PIN_MEMORY = _get_env_bool(_ENV_DL_PIN_MEMORY, bool(DL_PIN_MEMORY))
DL_PERSISTENT_WORKERS = _get_env_bool(_ENV_DL_PERSISTENT_WORKERS, bool(DL_PERSISTENT_WORKERS))
DL_PREFETCH_FACTOR = _get_env_int(_ENV_DL_PREFETCH_FACTOR, int(DL_PREFETCH_FACTOR))
LR = _get_env_float(_ENV_LR, float(LR))
WEIGHT_DECAY = _get_env_float(_ENV_WEIGHT_DECAY, float(WEIGHT_DECAY))
EMA_DECAY = _get_env_float(_ENV_EMA_DECAY, float(EMA_DECAY))
ENABLE_CLOUD_GATE_EVAL = _get_env_bool(_ENV_ENABLE_CLOUD_GATE_EVAL, bool(ENABLE_CLOUD_GATE_EVAL))
CLOUD_GATE_EVAL_EVERY = _get_env_int(_ENV_CLOUD_GATE_EVAL_EVERY, int(CLOUD_GATE_EVAL_EVERY))
CLOUD_GATE_PIN_IOU_THR = _get_env_float(_ENV_CLOUD_GATE_PIN_IOU_THR, float(CLOUD_GATE_PIN_IOU_THR))
CLOUD_GATE_BASE_FP_THR = _get_env_float(_ENV_CLOUD_GATE_BASE_FP_THR, float(CLOUD_GATE_BASE_FP_THR))
LOSS_TYPE = _get_env_str(_ENV_LOSS_TYPE, str(LOSS_TYPE)).strip()
POS_WEIGHT_MODE = _get_env_str(_ENV_POS_WEIGHT_MODE, str(POS_WEIGHT_MODE)).strip()
POS_WEIGHT_FIXED = _get_env_float(_ENV_POS_WEIGHT_FIXED, float(POS_WEIGHT_FIXED))
BCE_ASYMM_W_POS = _get_env_float(_ENV_BCE_ASYMM_W_POS, float(BCE_ASYMM_W_POS))
BCE_ASYMM_W_NEG = _get_env_float(_ENV_BCE_ASYMM_W_NEG, float(BCE_ASYMM_W_NEG))
TVERSKY_ALPHA = _get_env_float(_ENV_TVERSKY_ALPHA, float(TVERSKY_ALPHA))
TVERSKY_BETA = _get_env_float(_ENV_TVERSKY_BETA, float(TVERSKY_BETA))
FOCAL_TVERSKY_GAMMA = _get_env_float(_ENV_FOCAL_TVERSKY_GAMMA, float(FOCAL_TVERSKY_GAMMA))
TAIL_NEG_ALPHA = _get_env_float(_ENV_TAIL_NEG_ALPHA, float(TAIL_NEG_ALPHA))
TAIL_NEG_LAMBDA = _get_env_float(_ENV_TAIL_NEG_LAMBDA, float(TAIL_NEG_LAMBDA))
TAIL_NEG_GROUP_MODE = _get_env_str(_ENV_TAIL_NEG_GROUP_MODE, str(TAIL_NEG_GROUP_MODE)).strip()

NP_BETA_FPR = _get_env_float(_ENV_NP_BETA_FPR, float(NP_BETA_FPR))
NP_LAMBDA = _get_env_float(_ENV_NP_LAMBDA, float(NP_LAMBDA))
NP_TAU = _get_env_float(_ENV_NP_TAU, float(NP_TAU))
NP_MIN_BASE_PTS = _get_env_int(_ENV_NP_MIN_BASE_PTS, int(NP_MIN_BASE_PTS))
NP_MIN_PIN_PTS = _get_env_int(_ENV_NP_MIN_PIN_PTS, int(NP_MIN_PIN_PTS))
NP_GROUP_MODE = _get_env_str(_ENV_NP_GROUP_MODE, str(NP_GROUP_MODE)).strip()


def _topk_negative_tail_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
    group_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Engineering proxy for low-FPR alignment:
      - take negative logits z_neg (y==0)
      - penalize the top-α fraction (largest logits) with softplus(z)

    softplus(z) = log(1+exp(z)) equals the standard BCE loss term for a negative label.

    If group_ids is provided, we compute the tail loss *within each group* and then
    average across groups (avoids one large cloud dominating the tail set).
    """
    a = float(alpha)
    if a <= 0.0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    if a > 1.0:
        raise ValueError(f"alpha must be in (0,1], got {alpha}")

    z = logits.reshape(-1)
    yb = y.to(dtype=torch.float32).reshape(-1)
    if z.numel() != yb.numel():
        raise ValueError(f"logits/y shape mismatch: logits={tuple(z.shape)} y={tuple(yb.shape)}")

    if group_ids is None:
        neg = z[yb < 0.5]
        if neg.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        k = int(math.ceil(a * float(neg.numel())))
        k = max(1, min(k, int(neg.numel())))
        top = torch.topk(neg, k=k, largest=True).values
        return torch.nn.functional.softplus(top).mean()

    gids = group_ids.reshape(-1).to(dtype=torch.int64)
    if gids.numel() != yb.numel():
        raise ValueError(f"group_ids shape mismatch: group_ids={tuple(gids.shape)} y={tuple(yb.shape)}")

    losses: List[torch.Tensor] = []
    for gid in torch.unique(gids):
        mask = (gids == gid) & (yb < 0.5)
        neg = z[mask]
        if neg.numel() == 0:
            continue
        k = int(math.ceil(a * float(neg.numel())))
        k = max(1, min(k, int(neg.numel())))
        top = torch.topk(neg, k=k, largest=True).values
        losses.append(torch.nn.functional.softplus(top).mean())

    if not losses:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.stack(losses, dim=0).mean()


def _get_point_group_ids(batch_data: "Batch", *, mode: str) -> Optional[torch.Tensor]:
    """
    Return a per-point group id tensor (shape [N]) for NP/tail losses.

    mode:
      - global: None (single group)
      - block: per-graph id (batch_data.batch)
      - sid: per cloud id when available (batch_data.sid), else falls back to block
    """
    m = str(mode).strip().lower()
    if m in {"", "none", "global"}:
        return None
    if m in {"block", "batch", "graph"}:
        return batch_data.batch.to(dtype=torch.int64)
    if m in {"sid", "sample_id", "cloud"}:
        sid = getattr(batch_data, "sid", None)
        if sid is None:
            return batch_data.batch.to(dtype=torch.int64)
        sid_t = sid if torch.is_tensor(sid) else torch.as_tensor(sid, device=batch_data.batch.device)
        sid_t = sid_t.to(dtype=torch.int64).reshape(-1)
        # Best-effort validation: sid is per-graph; map points via batch indices.
        try:
            n_graphs = int(getattr(batch_data, "num_graphs"))
            if int(sid_t.numel()) != n_graphs:
                return batch_data.batch.to(dtype=torch.int64)
        except Exception:
            return batch_data.batch.to(dtype=torch.int64)
        return sid_t[batch_data.batch.to(dtype=torch.int64)]
    raise ValueError(f"Invalid group mode: {mode!r}. Expected one of: global, block, sid.")


def _np_soft_fn_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor],
    beta_fpr: float,
    tau: float,
    min_base_pts: int,
    min_pin_pts: int,
) -> torch.Tensor:
    """
    Soft-FN loss under a target FPR constraint (NP style).

    For each group g:
      - compute threshold t_g(beta) as the (1-beta) quantile of base logits (y==0)
      - penalize pin logits below this threshold with sigmoid((t - z_pos)/tau)

    Notes:
    - t_g is treated as stop-grad (computed on detached logits).
    - tau is in *logit* units (keep it hyperparameterized).
    """
    beta = float(beta_fpr)
    if beta <= 0.0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    if beta >= 1.0:
        raise ValueError(f"beta_fpr must be in (0,1), got {beta_fpr!r}")
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau!r}")

    z = logits.reshape(-1)
    yb = y.to(dtype=torch.float32).reshape(-1)
    if z.numel() != yb.numel():
        raise ValueError(f"logits/y shape mismatch: logits={tuple(z.shape)} y={tuple(yb.shape)}")

    if group_ids is None:
        gids = torch.zeros_like(yb, dtype=torch.int64)
    else:
        gids = group_ids.reshape(-1).to(dtype=torch.int64)
        if gids.numel() != yb.numel():
            raise ValueError(f"group_ids shape mismatch: group_ids={tuple(gids.shape)} y={tuple(yb.shape)}")

    min_base = int(min_base_pts)
    min_pin = int(min_pin_pts)
    if min_base < 1 or min_pin < 1:
        raise ValueError(f"min_base_pts/min_pin_pts must be >=1, got {min_base_pts!r}/{min_pin_pts!r}")

    q = float(1.0 - beta)
    losses: List[torch.Tensor] = []
    for gid in torch.unique(gids):
        idx = gids == gid
        z_g = z[idx]
        y_g = yb[idx]
        z_base = z_g[y_g < 0.5]
        z_pin = z_g[y_g >= 0.5]
        if z_base.numel() < min_base or z_pin.numel() < min_pin:
            continue
        t = torch.quantile(z_base.detach(), q=q)
        losses.append(torch.sigmoid((t - z_pin) / float(tau)).mean())

    if not losses:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.stack(losses, dim=0).mean()


def _bce_asymm_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    w_pos: float,
    w_neg: float,
) -> torch.Tensor:
    """
    Weighted BCE with independent weights for positives/negatives:
      loss = -w_pos * y * log(sigmoid(logit)) - w_neg * (1-y) * log(1-sigmoid(logit))
    This directly penalizes FP via w_neg (negatives predicted as positives).
    """
    y = y.to(dtype=torch.float32)
    w_pos_t = torch.as_tensor(float(w_pos), device=logits.device, dtype=torch.float32)
    w_neg_t = torch.as_tensor(float(w_neg), device=logits.device, dtype=torch.float32)
    # log(sigmoid(x)) and log(1-sigmoid(x)) are computed stably via logsigmoid.
    loss_pos = -w_pos_t * y * torch.nn.functional.logsigmoid(logits)
    loss_neg = -w_neg_t * (1.0 - y) * torch.nn.functional.logsigmoid(-logits)
    return (loss_pos + loss_neg).mean()


def _tversky_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Tversky loss (binary): (1 - TI)^gamma where
      TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    """
    y = y.to(dtype=torch.float32)
    p = torch.sigmoid(logits)
    tp = torch.sum(p * y)
    fp = torch.sum(p * (1.0 - y))
    fn = torch.sum((1.0 - p) * y)
    ti = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return torch.pow(1.0 - ti, float(gamma))


def _eval_iou_from_logits(logits: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-12) -> float:
    prob = torch.sigmoid(logits)
    pred = prob >= 0.5
    gt = y >= 0.5
    tp = float(torch.sum(pred & gt).detach().cpu().item())
    fp = float(torch.sum(pred & (~gt)).detach().cpu().item())
    fn = float(torch.sum((~pred) & gt).detach().cpu().item())
    return float(tp / max(tp + fp + fn, eps))


def _eval_cloud_metrics_from_logits_np(logits: np.ndarray, y: np.ndarray, *, eps: float = 1e-12) -> Dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if logits.shape[0] != y.shape[0]:
        raise ValueError(f"logits/y length mismatch: {logits.shape} vs {y.shape}")
    pred = logits >= 0.0  # logit>=0 <=> prob>=0.5
    gt = y >= 0.5
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & (~gt)))
    fn = float(np.sum((~pred) & gt))
    tn = float(np.sum((~pred) & (~gt)))
    pin_iou = float(tp / max(tp + fp + fn, eps))
    base_fp_rate = float(fp / max(fp + tn, eps))
    return {
        "n": float(int(logits.shape[0])),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "pin_iou": float(pin_iou),
        "base_fp_rate": float(base_fp_rate),
    }


def _infer_full_cloud_logits_stitch(
    *,
    xyz: np.ndarray,
    x: np.ndarray,
    model: PointNet2MSG,
    device: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Full-coverage inference (cloud-level) via kNN blocks + logit_sum/count stitch-back.
    Mirrors dl_pn2_msg/infer_pn2_msg.py to keep selection aligned with inference behavior.
    """
    n = int(xyz.shape[0])
    tree = cKDTree(xyz.astype(np.float32, copy=False))

    logit_sum = np.zeros((n,), dtype=np.float32)
    count = np.zeros((n,), dtype=np.int32)

    # Safety: in worst case each block covers only 1 new point.
    max_blocks = int(n)
    n_blocks = 0

    t0 = time.perf_counter()
    while int(np.sum(count > 0)) < n:
        uncovered = np.flatnonzero(count == 0)
        seed_idx = int(uncovered[0])
        idx = knn_block_indices(tree, xyz, seed_idx=seed_idx, k=int(BLOCK_POINTS))

        data, _dbg = build_block_data(xyz=xyz, x=x, y=np.zeros((n,), dtype=np.int64), idx=idx)
        data = data.to(device)

        with torch.no_grad():
            logits_b = model(
                x=data.x,
                pos=data.pos,
                batch=torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device),
            )
        logits_np = logits_b.detach().float().cpu().numpy().astype(np.float32, copy=False)
        logit_sum[idx] += logits_np
        count[idx] += 1

        n_blocks += 1
        if n_blocks >= max_blocks:
            raise RuntimeError(f"Exceeded max_blocks={max_blocks} while trying to cover all points (n={n}).")

    dt = time.perf_counter() - t0
    logits = logit_sum / np.maximum(count.astype(np.float32), 1.0)
    dbg = {
        "n_blocks": float(int(n_blocks)),
        "count_min": float(int(np.min(count))) if count.size else 0.0,
        "count_p50": float(np.median(count)) if count.size else 0.0,
        "count_max": float(int(np.max(count))) if count.size else 0.0,
        "time_s": float(dt),
    }
    return logits, dbg


def _write_csv(path: Path, rows: List[Dict[str, object]], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _append_csv(path: Path, row: Dict[str, object], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = Path(path).is_file()
    with Path(path).open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _eval_val_cloud_gate(
    *,
    epoch: int,
    tag: str,
    sids: List[int],
    model: PointNet2MSG,
    device: str,
) -> Optional[Dict[str, float]]:
    """
    Cloud-level gate metrics on val clouds (no augmentation).

    Returns a small summary dict for checkpoint selection; also writes per-cloud CSV into debug_out.
    """
    if not sids:
        return None

    thr_pin_iou = float(CLOUD_GATE_PIN_IOU_THR)
    thr_base_fp = float(CLOUD_GATE_BASE_FP_THR)

    t0 = time.perf_counter()
    rows: List[Dict[str, object]] = []
    for sid in sids:
        ply_path = Path(DATA_DIR) / f"{int(sid)}.ply"
        xyz, x, y, cloud_dbg = build_cloud(
            ply_path=ply_path,
            use_rgb=bool(USE_RGB),
            use_density=bool(USE_DENSITY),
            use_color_constancy=bool(USE_RGB and USE_COLOR_CONSTANCY),
            use_normals=bool(USE_NORMALS),
            use_global_xyz=bool(USE_GLOBAL_XYZ),
            color_cc_cfg=_COLOR_CC_CFG,
            density_k=int(DENSITY_K),
            density_k2=int(DENSITY_K2),
            density_mode=str(DENSITY_MODE),
            density_nn_median_mode=str(DENSITY_NN_MEDIAN_MODE),
            density_max_points_for_nn_median=int(DENSITY_MAX_POINTS_FOR_NN_MED),
            # Keep eval deterministic; training randomness should not leak into selection.
            rng=np.random.default_rng(0),
        )
        logits, stitch_dbg = _infer_full_cloud_logits_stitch(xyz=xyz, x=x, model=model, device=device)
        m = _eval_cloud_metrics_from_logits_np(logits, y)
        ok = (float(m["pin_iou"]) >= thr_pin_iou) and (float(m["base_fp_rate"]) <= thr_base_fp)
        rows.append(
            {
                "epoch": int(epoch),
                "tag": str(tag),
                "sample_id": int(sid),
                "n": int(m["n"]),
                "pin_iou": float(m["pin_iou"]),
                "base_fp_rate": float(m["base_fp_rate"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
                "tn": int(m["tn"]),
                "gate_ok": int(1 if ok else 0),
                "n_blocks": int(stitch_dbg.get("n_blocks", 0.0)),
                "count_min": int(stitch_dbg.get("count_min", 0.0)),
                "count_p50": float(stitch_dbg.get("count_p50", 0.0)),
                "count_max": int(stitch_dbg.get("count_max", 0.0)),
                "infer_time_s": float(stitch_dbg.get("time_s", 0.0)),
                "cloud_nn_median": (
                    float(cloud_dbg.get("nn_median"))
                    if np.isfinite(float(cloud_dbg.get("nn_median", float("nan"))))
                    else float("nan")
                ),
                "cloud_cloud_xyz_diag": (
                    float(cloud_dbg.get("cloud_xyz_diag"))
                    if np.isfinite(float(cloud_dbg.get("cloud_xyz_diag", float("nan"))))
                    else float("nan")
                ),
                "cloud_cloud_xyz_scale": (
                    float(cloud_dbg.get("cloud_xyz_scale"))
                    if np.isfinite(float(cloud_dbg.get("cloud_xyz_scale", float("nan"))))
                    else float("nan")
                ),
            }
        )

    dt = time.perf_counter() - t0
    pin_ious = [float(r["pin_iou"]) for r in rows]
    base_fps = [float(r["base_fp_rate"]) for r in rows]
    gate_ok = [bool(int(r["gate_ok"])) for r in rows]
    summary = {
        "epoch": float(int(epoch)),
        "fail_n": float(int(np.sum([not x for x in gate_ok]))),
        "min_pin_iou": float(np.min(pin_ious)) if pin_ious else float("nan"),
        "max_base_fp_rate": float(np.max(base_fps)) if base_fps else float("nan"),
        "mean_pin_iou": float(np.mean(pin_ious)) if pin_ious else float("nan"),
        "mean_base_fp_rate": float(np.mean(base_fps)) if base_fps else float("nan"),
        "thr_pin_iou": float(thr_pin_iou),
        "thr_base_fp_rate": float(thr_base_fp),
        "time_s": float(dt),
    }

    if _DAIMA_CTX.run_dir is not None:
        metrics_dir = _DAIMA_CTX.run_dir / "metrics"
        per_cloud_path = metrics_dir / f"val_cloud_metrics_epoch_{int(epoch):03d}_{str(tag)}.csv"
        _write_csv(
            per_cloud_path,
            rows,
            fieldnames=[
                "epoch",
                "tag",
                "sample_id",
                "n",
                "pin_iou",
                "base_fp_rate",
                "tp",
                "fp",
                "fn",
                "tn",
                "gate_ok",
                "n_blocks",
                "count_min",
                "count_p50",
                "count_max",
                "infer_time_s",
                "cloud_nn_median",
                "cloud_cloud_xyz_diag",
                "cloud_cloud_xyz_scale",
            ],
        )
        hist_path = metrics_dir / "val_cloud_gate_history.csv"
        _append_csv(
            hist_path,
            {
                "epoch": int(epoch),
                "tag": str(tag),
                "fail_n": int(summary["fail_n"]),
                "min_pin_iou": float(summary["min_pin_iou"]),
                "max_base_fp_rate": float(summary["max_base_fp_rate"]),
                "mean_pin_iou": float(summary["mean_pin_iou"]),
                "mean_base_fp_rate": float(summary["mean_base_fp_rate"]),
                "thr_pin_iou": float(thr_pin_iou),
                "thr_base_fp_rate": float(thr_base_fp),
                "time_s": float(summary["time_s"]),
            },
            fieldnames=[
                "epoch",
                "tag",
                "fail_n",
                "min_pin_iou",
                "max_base_fp_rate",
                "mean_pin_iou",
                "mean_base_fp_rate",
                "thr_pin_iou",
                "thr_base_fp_rate",
                "time_s",
            ],
        )

    logger.info(
        "CloudGate[%s] epoch=%s fail_n=%s min_pin_iou=%.4f max_base_fp=%.4f (%.2fs)",
        str(tag),
        int(epoch),
        int(summary["fail_n"]),
        float(summary["min_pin_iou"]),
        float(summary["max_base_fp_rate"]),
        float(summary["time_s"]),
    )
    return {k: float(v) for k, v in summary.items()}


def _build_blocks_for_sids(
    sids: List[int],
    *,
    num_blocks_per_cloud: int,
    rng: np.random.Generator,
    augment_color: bool,
) -> Tuple[List["torch_geometric.data.Data"], Dict[str, float], int]:
    """
    Returns: (blocks, dbg_stats, in_channels)
    """
    blocks: List["torch_geometric.data.Data"] = []
    dbg: Dict[str, float] = {"clouds": float(len(sids)), "blocks": 0.0}
    in_channels: Optional[int] = None

    seed_sampling_mode = str(DENSITY_SEED_SAMPLING_MODE).strip().lower()
    if seed_sampling_mode in {"", "none"}:
        seed_sampling_mode = "uniform"
    if seed_sampling_mode not in {"uniform", "inv_density"}:
        raise ValueError(
            f"Invalid {_ENV_DENSITY_SEED_SAMPLING_MODE}={DENSITY_SEED_SAMPLING_MODE!r}. Expected 'uniform' or 'inv_density'."
        )

    loss_reweight_mode = str(DENSITY_LOSS_REWEIGHT_MODE).strip().lower()
    if loss_reweight_mode in {"", "off"}:
        loss_reweight_mode = "none"
    if loss_reweight_mode not in {"none", "inv_density"}:
        raise ValueError(
            f"Invalid {_ENV_DENSITY_LOSS_REWEIGHT_MODE}={DENSITY_LOSS_REWEIGHT_MODE!r}. Expected 'none' or 'inv_density'."
        )
    if float(DENSITY_LOSS_REWEIGHT_CLIP_MIN) <= 0.0:
        raise ValueError(
            f"Invalid {_ENV_DENSITY_LOSS_REWEIGHT_CLIP_MIN}={DENSITY_LOSS_REWEIGHT_CLIP_MIN!r}. Expected > 0."
        )
    if float(DENSITY_LOSS_REWEIGHT_CLIP_MAX) < float(DENSITY_LOSS_REWEIGHT_CLIP_MIN):
        raise ValueError(
            f"Invalid clip range: {DENSITY_LOSS_REWEIGHT_CLIP_MIN}..{DENSITY_LOSS_REWEIGHT_CLIP_MAX} "
            f"({_ENV_DENSITY_LOSS_REWEIGHT_CLIP_MIN}/{_ENV_DENSITY_LOSS_REWEIGHT_CLIP_MAX})"
        )

    # We need per-cloud nn_median whenever we re-use density as a training stabilizer (sampling/reweight),
    # even if density is not part of the feature channels.
    need_nn_median = bool(USE_DENSITY) or (seed_sampling_mode != "uniform") or (loss_reweight_mode != "none")

    for sid in sids:
        ply_path = Path(DATA_DIR) / f"{int(sid)}.ply"
        xyz, x, y, cloud_dbg = build_cloud(
            ply_path=ply_path,
            use_rgb=bool(USE_RGB),
            use_density=bool(USE_DENSITY),
            use_color_constancy=bool(USE_RGB and USE_COLOR_CONSTANCY),
            use_normals=bool(USE_NORMALS),
            use_global_xyz=bool(USE_GLOBAL_XYZ),
            color_cc_cfg=_COLOR_CC_CFG,
            density_k=int(DENSITY_K),
            density_k2=int(DENSITY_K2),
            density_mode=str(DENSITY_MODE),
            density_nn_median_mode=str(DENSITY_NN_MEDIAN_MODE),
            density_max_points_for_nn_median=int(DENSITY_MAX_POINTS_FOR_NN_MED),
            rng=rng,
            need_nn_median=bool(need_nn_median),
        )
        if in_channels is None:
            in_channels = int(x.shape[1])
        elif int(x.shape[1]) != int(in_channels):
            raise RuntimeError(f"Inconsistent feature dim across clouds: got {x.shape[1]} vs {in_channels}")

        tree = cKDTree(xyz.astype(np.float32, copy=False))
        d_mean_rel: Optional[np.ndarray] = None
        w_cloud: Optional[np.ndarray] = None
        if (seed_sampling_mode != "uniform") or (loss_reweight_mode != "none"):
            nn_med = float(cloud_dbg.get("nn_median", float("nan")))
            if not np.isfinite(nn_med) or nn_med <= 1e-12:
                nn_med = 1.0
            k = int(max(1, int(DENSITY_LOSS_REWEIGHT_K)))
            try:
                d, _ = tree.query(xyz.astype(np.float32, copy=False), k=k + 1, workers=-1)
            except TypeError:
                d, _ = tree.query(xyz.astype(np.float32, copy=False), k=k + 1)
            d = np.asarray(d, dtype=np.float64)[:, 1:]
            d_mean_rel = (np.mean(d, axis=1) / float(nn_med)).astype(np.float32, copy=False)

        if seed_sampling_mode == "uniform":
            seeds = choose_seeds(int(xyz.shape[0]), num_seeds=int(num_blocks_per_cloud), rng=rng)
        else:
            if d_mean_rel is None:
                raise RuntimeError("seed_sampling_mode requires d_mean_rel, but it is None.")
            w = np.clip(d_mean_rel, 1e-8, None) ** float(DENSITY_SEED_SAMPLING_POWER)
            w64 = w.astype(np.float64, copy=False)
            s = float(np.sum(w64))
            if (not np.isfinite(s)) or s <= 0.0:
                p = None
            else:
                p = w64 / s
                # numpy.random.Generator.choice is strict about sum(p)==1.
                p_sum = float(np.sum(p))
                if (not np.isfinite(p_sum)) or p_sum <= 0.0:
                    p = None
                else:
                    p = p / p_sum
                    if p.size > 0:
                        p[-1] = 1.0 - float(np.sum(p[:-1]))
                    p = np.clip(p, 0.0, 1.0)
                    p_sum2 = float(np.sum(p))
                    p = (p / p_sum2) if (np.isfinite(p_sum2) and p_sum2 > 0.0) else None
            seeds = rng.choice(int(xyz.shape[0]), size=int(num_blocks_per_cloud), replace=False, p=p).astype(
                np.int64, copy=False
            )

        if loss_reweight_mode == "none":
            w_cloud = None
        else:
            if d_mean_rel is None:
                raise RuntimeError("loss_reweight_mode requires d_mean_rel, but it is None.")
            w = np.clip(d_mean_rel, 1e-8, None) ** float(DENSITY_LOSS_REWEIGHT_POWER)
            w = np.clip(w, float(DENSITY_LOSS_REWEIGHT_CLIP_MIN), float(DENSITY_LOSS_REWEIGHT_CLIP_MAX))
            w_mean = float(np.mean(w)) if w.size else 1.0
            if not np.isfinite(w_mean) or w_mean <= 1e-12:
                w_mean = 1.0
            w_cloud = (w / w_mean).astype(np.float32, copy=False)

        rgb_slice = None
        if bool(USE_RGB):
            start = 0
            if bool(USE_NORMALS):
                start += 3
            if bool(USE_GLOBAL_XYZ):
                start += 3
            rgb_slice = slice(int(start), int(start + 3))
        for seed_idx in seeds.tolist():
            idx = knn_block_indices(tree, xyz, seed_idx=int(seed_idx), k=int(BLOCK_POINTS))
            data, block_dbg = build_block_data(xyz=xyz, x=x, y=y, idx=idx)
            if w_cloud is not None:
                data.w = torch.from_numpy(w_cloud[idx].astype(np.float32, copy=False))  # type: ignore[attr-defined]
            if augment_color and rgb_slice is not None:
                rgb_t = data.x[:, rgb_slice]
                if float(COLOR_DROPOUT_BLOCK_P) > 0.0 and float(rng.random()) < float(COLOR_DROPOUT_BLOCK_P):
                    rgb_t.zero_()
                if float(COLOR_DROPOUT_POINT_P) > 0.0:
                    m = rng.random(int(rgb_t.shape[0])) < float(COLOR_DROPOUT_POINT_P)
                    if np.any(m):
                        rgb_t[torch.from_numpy(m)] = 0.0
                if float(COLOR_JITTER_SCALE) > 0.0:
                    s = 1.0 + float(rng.uniform(-float(COLOR_JITTER_SCALE), float(COLOR_JITTER_SCALE)))
                    rgb_t.mul_(float(s))
                if float(COLOR_JITTER_SHIFT) > 0.0:
                    b = float(rng.uniform(-float(COLOR_JITTER_SHIFT), float(COLOR_JITTER_SHIFT)))
                    rgb_t.add_(float(b))
                if float(COLOR_JITTER_NOISE_STD) > 0.0:
                    noise = rng.normal(0.0, float(COLOR_JITTER_NOISE_STD), size=tuple(rgb_t.shape)).astype(
                        np.float32, copy=False
                    )
                    rgb_t.add_(torch.from_numpy(noise))
                rgb_t.clamp_(0.0, 1.0)
            # Lightweight provenance for debugging:
            data.sid = int(sid)  # type: ignore[attr-defined]
            blocks.append(data)
            dbg["blocks"] = float(len(blocks))
            # Keep only a few numeric stats (avoid logging arrays).
            dbg.setdefault("block_xyz_scale_sum", 0.0)
            dbg["block_xyz_scale_sum"] += float(block_dbg.get("block_xyz_scale", 0.0))

        # Per-cloud stats (optional)
        for k, v in cloud_dbg.items():
            if np.isfinite(v):
                dbg.setdefault(f"cloud_{k}_sum", 0.0)
                dbg[f"cloud_{k}_sum"] += float(v)

    if in_channels is None:
        raise RuntimeError("Empty dataset: no blocks built.")
    return blocks, dbg, int(in_channels)


def main() -> None:
    logger.info("Repo root: %s", _DAIMA_CTX.repo_root)
    logger.info("Data dir : %s", DATA_DIR)
    logger.info("Split csv: %s", SPLIT_CSV)
    logger.info("Run dir  : %s", _DAIMA_CTX.run_dir)

    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Split file not found: {SPLIT_CSV}")
    split_map = load_split_csv(SPLIT_CSV)

    trainval_sids = sorted([sid for sid, (role, _f) in split_map.items() if role == "trainval"])
    if not trainval_sids:
        raise RuntimeError("No trainval samples found in split file.")

    fold_id = int(FOLD_ID)
    train_sids = [sid for sid in trainval_sids if int(split_map[sid][1]) != fold_id]
    val_sids = [sid for sid in trainval_sids if int(split_map[sid][1]) == fold_id]
    logger.info("Train/val split: fold=%s train=%s val=%s", fold_id, len(train_sids), len(val_sids))

    seed = int(RANDOM_SEED)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _DAIMA_CTX.snapshot_paths(
        inputs={"data_dir": str(DATA_DIR), "split_csv": str(SPLIT_CSV)},
        outputs={"run_dir": str(_DAIMA_CTX.run_dir)},
    )
    _DAIMA_CTX.snapshot_config(
        config={
            "fold_id": float(fold_id),
            "random_seed": float(int(RANDOM_SEED)),
            "features": {
                "use_rgb": float(1.0 if USE_RGB else 0.0),
                "use_density": float(1.0 if USE_DENSITY else 0.0),
                "use_color_constancy": float(1.0 if (USE_RGB and USE_COLOR_CONSTANCY) else 0.0),
                "use_normals": float(1.0 if USE_NORMALS else 0.0),
                "use_global_xyz": float(1.0 if USE_GLOBAL_XYZ else 0.0),
                "density_k": float(int(DENSITY_K)),
                "density_k2": float(int(DENSITY_K2)),
                "density_mode": str(DENSITY_MODE),
                "density_nn_median_mode": str(DENSITY_NN_MEDIAN_MODE),
                "density_seed_sampling_mode": str(DENSITY_SEED_SAMPLING_MODE),
                "density_seed_sampling_power": float(DENSITY_SEED_SAMPLING_POWER),
                "density_loss_reweight_mode": str(DENSITY_LOSS_REWEIGHT_MODE),
                "density_loss_reweight_k": float(int(DENSITY_LOSS_REWEIGHT_K)),
                "density_loss_reweight_power": float(DENSITY_LOSS_REWEIGHT_POWER),
                "density_loss_reweight_clip_min": float(DENSITY_LOSS_REWEIGHT_CLIP_MIN),
                "density_loss_reweight_clip_max": float(DENSITY_LOSS_REWEIGHT_CLIP_MAX),
                "color_cc_cfg": safe_cc_cfg_dict(_COLOR_CC_CFG),
            },
            "blocks": {
                "block_points": float(int(BLOCK_POINTS)),
                "train_blocks_per_cloud": float(int(TRAIN_BLOCKS_PER_CLOUD)),
                "val_blocks_per_cloud": float(int(VAL_BLOCKS_PER_CLOUD)),
            },
            "augment": {
                "color_dropout_block_p": float(COLOR_DROPOUT_BLOCK_P),
                "color_dropout_point_p": float(COLOR_DROPOUT_POINT_P),
                "color_jitter_scale": float(COLOR_JITTER_SCALE),
                "color_jitter_shift": float(COLOR_JITTER_SHIFT),
                "color_jitter_noise_std": float(COLOR_JITTER_NOISE_STD),
            },
            "model": {
                "type": "PointNet2MSG",
                "cfg": {
                    "sa1_ratio": float(MODEL_CFG.sa1_ratio),
                    "sa2_ratio": float(MODEL_CFG.sa2_ratio),
                    "sa1_radii": list(float(r) for r in MODEL_CFG.sa1_radii),
                    "sa2_radii": list(float(r) for r in MODEL_CFG.sa2_radii),
                    "max_num_neighbors": float(int(MODEL_CFG.max_num_neighbors)),
                    "fp_knn_k": float(int(MODEL_CFG.fp_knn_k)),
                    "dropout": float(MODEL_CFG.dropout),
                },
            },
            "train": {
                "epochs": float(int(EPOCHS)),
                "batch_size_blocks": float(int(BATCH_SIZE_BLOCKS)),
                "lr": float(LR),
                "weight_decay": float(WEIGHT_DECAY),
                "ema_decay": float(EMA_DECAY),
                "dataloader": {
                    "num_workers": float(int(DL_NUM_WORKERS)),
                    "pin_memory": float(1.0 if DL_PIN_MEMORY else 0.0),
                    "persistent_workers": float(1.0 if DL_PERSISTENT_WORKERS else 0.0),
                    "prefetch_factor": float(int(DL_PREFETCH_FACTOR)),
                },
                "cloud_gate": {
                    "enabled": float(1.0 if ENABLE_CLOUD_GATE_EVAL else 0.0),
                    "eval_every": float(int(CLOUD_GATE_EVAL_EVERY)),
                    "thr_pin_iou": float(CLOUD_GATE_PIN_IOU_THR),
                    "thr_base_fp_rate": float(CLOUD_GATE_BASE_FP_THR),
                },
                "loss": {
                    "type": str(LOSS_TYPE),
                    "pos_weight_mode": str(POS_WEIGHT_MODE),
                    "pos_weight_fixed": float(POS_WEIGHT_FIXED),
                    "bce_asymm": {"w_pos": float(BCE_ASYMM_W_POS), "w_neg": float(BCE_ASYMM_W_NEG)},
                    "tversky": {
                        "alpha": float(TVERSKY_ALPHA),
                        "beta": float(TVERSKY_BETA),
                        "focal_gamma": float(FOCAL_TVERSKY_GAMMA),
                    },
                    "tail_neg": {
                        "alpha": float(TAIL_NEG_ALPHA),
                        "lambda": float(TAIL_NEG_LAMBDA),
                        "group_mode": str(TAIL_NEG_GROUP_MODE),
                    },
                    "np_soft_fn": {
                        "beta_fpr": float(NP_BETA_FPR),
                        "tau": float(NP_TAU),
                        "lambda": float(NP_LAMBDA),
                        "group_mode": str(NP_GROUP_MODE),
                        "min_base_pts": float(int(NP_MIN_BASE_PTS)),
                        "min_pin_pts": float(int(NP_MIN_PIN_PTS)),
                    },
                },
            },
        }
    )

    t0 = time.perf_counter()
    enable_color_aug = bool(
        USE_RGB
        and (
            float(COLOR_DROPOUT_BLOCK_P) > 0.0
            or float(COLOR_DROPOUT_POINT_P) > 0.0
            or float(COLOR_JITTER_SCALE) > 0.0
            or float(COLOR_JITTER_SHIFT) > 0.0
            or float(COLOR_JITTER_NOISE_STD) > 0.0
        )
    )
    train_blocks, dbg_train, in_channels = _build_blocks_for_sids(
        train_sids, num_blocks_per_cloud=int(TRAIN_BLOCKS_PER_CLOUD), rng=rng, augment_color=enable_color_aug
    )
    val_blocks, dbg_val, _ = _build_blocks_for_sids(
        val_sids, num_blocks_per_cloud=int(VAL_BLOCKS_PER_CLOUD), rng=rng, augment_color=False
    )
    dt_load = time.perf_counter() - t0
    logger.info(
        "Dataset ready: train_blocks=%s val_blocks=%s in_channels=%s (%.2fs)",
        len(train_blocks),
        len(val_blocks),
        int(in_channels),
        dt_load,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    model = PointNet2MSG(in_channels=int(in_channels), cfg=MODEL_CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))

    ema_decay = float(EMA_DECAY)
    ema_model: Optional[PointNet2MSG] = None
    if ema_decay > 0.0:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info("EMA enabled: decay=%.6f", float(ema_decay))

    # Class balance is computed on the (block-sampled) training set.
    # It is used for logging and optionally for pos_weight-based BCE.
    y_train = torch.cat([d.y for d in train_blocks], dim=0).numpy()
    pos = float(np.sum(y_train >= 0.5))
    neg = float(np.sum(y_train < 0.5))
    pos_weight_auto = (neg / max(pos, 1.0)) if (pos > 0 and neg > 0) else 1.0

    loss_type = str(LOSS_TYPE).strip().lower()
    pos_weight_mode = str(POS_WEIGHT_MODE).strip().lower()
    pos_weight_used: Optional[float] = None
    tail_alpha = float(TAIL_NEG_ALPHA)
    tail_lambda = float(TAIL_NEG_LAMBDA)
    tail_group_mode = str(TAIL_NEG_GROUP_MODE).strip().lower()

    np_beta = float(NP_BETA_FPR)
    np_lambda = float(NP_LAMBDA)
    np_tau = float(NP_TAU)
    np_min_base_pts = int(NP_MIN_BASE_PTS)
    np_min_pin_pts = int(NP_MIN_PIN_PTS)
    np_group_mode = str(NP_GROUP_MODE).strip().lower()

    if loss_type in {"bce_pos_weight", "bce", "bcewithlogits"}:
        if pos_weight_mode in {"auto", ""}:
            pos_weight_used = float(pos_weight_auto)
        elif pos_weight_mode in {"fixed", "manual"}:
            pos_weight_used = float(POS_WEIGHT_FIXED)
        else:
            raise ValueError(
                f"Invalid { _ENV_POS_WEIGHT_MODE }={POS_WEIGHT_MODE!r}. Expected 'auto' or 'fixed'."
            )
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(pos_weight_used)], device=device, dtype=torch.float32)
        )
        logger.info(
            "Loss: bce_pos_weight (mode=%s pos_weight=%.4f; auto=%.4f)",
            pos_weight_mode,
            float(pos_weight_used),
            float(pos_weight_auto),
        )
    elif loss_type in {"bce_asymm", "asymm_bce"}:
        loss_fn = lambda logits, yb: _bce_asymm_loss(  # noqa: E731
            logits, yb, w_pos=float(BCE_ASYMM_W_POS), w_neg=float(BCE_ASYMM_W_NEG)
        )
        logger.info(
            "Loss: bce_asymm (w_pos=%.4f w_neg=%.4f; pos_weight_auto=%.4f)",
            float(BCE_ASYMM_W_POS),
            float(BCE_ASYMM_W_NEG),
            float(pos_weight_auto),
        )
    elif loss_type in {"tversky"}:
        loss_fn = lambda logits, yb: _tversky_loss(  # noqa: E731
            logits, yb, alpha=float(TVERSKY_ALPHA), beta=float(TVERSKY_BETA), gamma=1.0
        )
        logger.info(
            "Loss: tversky (alpha=%.4f beta=%.4f gamma=1.0; pos_weight_auto=%.4f)",
            float(TVERSKY_ALPHA),
            float(TVERSKY_BETA),
            float(pos_weight_auto),
        )
    elif loss_type in {"focal_tversky", "focal-tversky"}:
        loss_fn = lambda logits, yb: _tversky_loss(  # noqa: E731
            logits,
            yb,
            alpha=float(TVERSKY_ALPHA),
            beta=float(TVERSKY_BETA),
            gamma=float(FOCAL_TVERSKY_GAMMA),
        )
        logger.info(
            "Loss: focal_tversky (alpha=%.4f beta=%.4f gamma=%.4f; pos_weight_auto=%.4f)",
            float(TVERSKY_ALPHA),
            float(TVERSKY_BETA),
            float(FOCAL_TVERSKY_GAMMA),
            float(pos_weight_auto),
        )
    else:
        raise ValueError(
            f"Invalid { _ENV_LOSS_TYPE }={LOSS_TYPE!r}. Expected one of: bce_pos_weight, bce_asymm, tversky, focal_tversky."
        )

    if tail_lambda < 0.0:
        raise ValueError(f"Invalid {_ENV_TAIL_NEG_LAMBDA}={TAIL_NEG_LAMBDA!r}. Expected >= 0.")
    if tail_lambda > 0.0:
        if tail_alpha <= 0.0 or tail_alpha > 1.0:
            raise ValueError(f"Invalid {_ENV_TAIL_NEG_ALPHA}={TAIL_NEG_ALPHA!r}. Expected 0<alpha<=1.")
        if tail_group_mode not in {"global", "block", "batch", "graph", "sid", "sample_id", "cloud"}:
            raise ValueError(
                f"Invalid {_ENV_TAIL_NEG_GROUP_MODE}={TAIL_NEG_GROUP_MODE!r}. Expected one of: global, block, sid."
            )
        logger.info(
            "Tail loss: top-alpha negative (alpha=%.4f lambda=%.4f group_mode=%s)",
            float(tail_alpha),
            float(tail_lambda),
            str(tail_group_mode),
        )
    else:
        if tail_group_mode not in {"", "global", "none", "block", "batch", "graph", "sid", "sample_id", "cloud"}:
            raise ValueError(
                f"Invalid {_ENV_TAIL_NEG_GROUP_MODE}={TAIL_NEG_GROUP_MODE!r}. Expected one of: global, block, sid."
            )
        logger.info(
            "Tail loss: disabled (alpha=%.4f lambda=%.4f group_mode=%s)",
            float(tail_alpha),
            float(tail_lambda),
            str(tail_group_mode),
        )

    if np_lambda < 0.0:
        raise ValueError(f"Invalid {_ENV_NP_LAMBDA}={NP_LAMBDA!r}. Expected >= 0.")
    if np_lambda > 0.0:
        if np_beta <= 0.0 or np_beta >= 1.0:
            raise ValueError(f"Invalid {_ENV_NP_BETA_FPR}={NP_BETA_FPR!r}. Expected 0<beta<1.")
        if np_tau <= 0.0:
            raise ValueError(f"Invalid {_ENV_NP_TAU}={NP_TAU!r}. Expected tau>0.")
        if np_min_base_pts < 1 or np_min_pin_pts < 1:
            raise ValueError(
                f"Invalid min pts: {_ENV_NP_MIN_BASE_PTS}={NP_MIN_BASE_PTS!r} {_ENV_NP_MIN_PIN_PTS}={NP_MIN_PIN_PTS!r}. Expected >=1."
            )
        if np_group_mode not in {"global", "block", "batch", "graph", "sid", "sample_id", "cloud"}:
            raise ValueError(
                f"Invalid {_ENV_NP_GROUP_MODE}={NP_GROUP_MODE!r}. Expected one of: global, block, sid."
            )
        logger.info(
            "NP soft-FN: enabled (beta=%.4f tau=%.4f lambda=%.4f group_mode=%s min_base=%s min_pin=%s)",
            float(np_beta),
            float(np_tau),
            float(np_lambda),
            str(np_group_mode),
            int(np_min_base_pts),
            int(np_min_pin_pts),
        )
    else:
        logger.info(
            "NP soft-FN: disabled (beta=%.4f tau=%.4f lambda=%.4f group_mode=%s)",
            float(np_beta),
            float(np_tau),
            float(np_lambda),
            str(np_group_mode),
        )

    logger.info(
        "Class balance (block-sampled): pos=%s neg=%s pos_weight_auto=%.4f",
        int(pos),
        int(neg),
        float(pos_weight_auto),
    )

    dl_num_workers = int(DL_NUM_WORKERS)
    if dl_num_workers < 0:
        raise ValueError(f"Invalid {_ENV_DL_NUM_WORKERS}={DL_NUM_WORKERS!r}. Expected >= 0.")
    dl_pin_memory = bool(DL_PIN_MEMORY)
    dl_persistent_workers = bool(DL_PERSISTENT_WORKERS) and dl_num_workers > 0
    dl_prefetch_factor = int(DL_PREFETCH_FACTOR)
    if dl_num_workers > 0 and dl_prefetch_factor < 1:
        raise ValueError(f"Invalid {_ENV_DL_PREFETCH_FACTOR}={DL_PREFETCH_FACTOR!r}. Expected >= 1 when workers > 0.")

    dl_common_kwargs: Dict[str, object] = {
        "batch_size": int(BATCH_SIZE_BLOCKS),
        "num_workers": int(dl_num_workers),
        "drop_last": False,
        "pin_memory": bool(dl_pin_memory),
        "persistent_workers": bool(dl_persistent_workers),
    }
    if dl_num_workers > 0:
        dl_common_kwargs["prefetch_factor"] = int(dl_prefetch_factor)

    dl_train = DataLoader(train_blocks, shuffle=True, **dl_common_kwargs)
    dl_val = DataLoader(val_blocks, shuffle=False, **dl_common_kwargs)

    best_val_iou = -1.0
    best_val_iou_ema = -1.0
    best_path: Optional[Path] = None
    best_ema_path: Optional[Path] = None
    best_gate_path: Optional[Path] = None
    best_gate_ema_path: Optional[Path] = None
    last_path: Optional[Path] = None
    last_ema_path: Optional[Path] = None
    best_gate_rank: Optional[Tuple[int, float, float]] = None
    best_gate_ema_rank: Optional[Tuple[int, float, float]] = None

    ckpt_dir = (_DAIMA_CTX.run_dir / "ckpt" / "pn2_msg") if _DAIMA_CTX.run_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(EPOCHS) + 1):
        model.train()
        t_ep = time.perf_counter()
        losses_total: List[float] = []
        losses_base: List[float] = []
        losses_tail: List[float] = []
        losses_np: List[float] = []

        for batch_data in dl_train:
            batch_data = batch_data.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x=batch_data.x, pos=batch_data.pos, batch=batch_data.batch)
            yb = batch_data.y.to(dtype=torch.float32)
            w = getattr(batch_data, "w", None)
            if w is None:
                base_loss = loss_fn(logits, yb)
            else:
                wb = w.to(dtype=torch.float32)
                if loss_type in {"bce_pos_weight", "bce", "bcewithlogits"}:
                    pos_w_t = (
                        torch.tensor([float(pos_weight_used)], device=logits.device, dtype=torch.float32)
                        if pos_weight_used is not None
                        else None
                    )
                    per = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, yb, pos_weight=pos_w_t, reduction="none"
                    )
                    base_loss = (per * wb).mean()
                elif loss_type in {"bce_asymm", "asymm_bce"}:
                    w_pos_t = torch.as_tensor(float(BCE_ASYMM_W_POS), device=logits.device, dtype=torch.float32)
                    w_neg_t = torch.as_tensor(float(BCE_ASYMM_W_NEG), device=logits.device, dtype=torch.float32)
                    loss_pos = -w_pos_t * yb * torch.nn.functional.logsigmoid(logits)
                    loss_neg = -w_neg_t * (1.0 - yb) * torch.nn.functional.logsigmoid(-logits)
                    base_loss = ((loss_pos + loss_neg) * wb).mean()
                else:
                    # For global losses (e.g. tversky), we keep the original behavior.
                    base_loss = loss_fn(logits, yb)
            tail_group_ids = None
            if float(tail_lambda) > 0.0:
                tail_group_ids = _get_point_group_ids(batch_data, mode=str(tail_group_mode))
            tail_loss = (
                _topk_negative_tail_loss(logits, yb, alpha=float(tail_alpha), group_ids=tail_group_ids)
                if float(tail_lambda) > 0.0
                else torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
            )

            np_group_ids = None
            if float(np_lambda) > 0.0:
                np_group_ids = _get_point_group_ids(batch_data, mode=str(np_group_mode))
            np_loss = (
                _np_soft_fn_loss(
                    logits,
                    yb,
                    group_ids=np_group_ids,
                    beta_fpr=float(np_beta),
                    tau=float(np_tau),
                    min_base_pts=int(np_min_base_pts),
                    min_pin_pts=int(np_min_pin_pts),
                )
                if float(np_lambda) > 0.0
                else torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
            )

            loss = base_loss + float(tail_lambda) * tail_loss + float(np_lambda) * np_loss
            loss.backward()
            opt.step()
            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
                    # Keep buffers (e.g., BatchNorm running stats) in sync with the current model.
                    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
                        ema_b.copy_(b)
            losses_total.append(float(loss.detach().cpu().item()))
            losses_base.append(float(base_loss.detach().cpu().item()))
            if float(tail_lambda) > 0.0:
                losses_tail.append(float(tail_loss.detach().cpu().item()))
            if float(np_lambda) > 0.0:
                losses_np.append(float(np_loss.detach().cpu().item()))

        train_loss = float(np.mean(losses_total)) if losses_total else float("nan")
        train_loss_base = float(np.mean(losses_base)) if losses_base else float("nan")
        train_loss_tail = float(np.mean(losses_tail)) if losses_tail else 0.0
        train_loss_np = float(np.mean(losses_np)) if losses_np else 0.0

        # Validation (block-level)
        model.eval()
        val_iou = float("nan")
        val_iou_ema = float("nan")
        if len(val_blocks) > 0:
            ious: List[float] = []
            with torch.no_grad():
                for batch_data in dl_val:
                    batch_data = batch_data.to(device)
                    logits = model(x=batch_data.x, pos=batch_data.pos, batch=batch_data.batch)
                    ious.append(_eval_iou_from_logits(logits, batch_data.y.to(dtype=torch.float32)))
            val_iou = float(np.mean(ious)) if ious else float("nan")

            if ema_model is not None:
                ious_ema: List[float] = []
                with torch.no_grad():
                    for batch_data in dl_val:
                        batch_data = batch_data.to(device)
                        logits = ema_model(x=batch_data.x, pos=batch_data.pos, batch=batch_data.batch)
                        ious_ema.append(_eval_iou_from_logits(logits, batch_data.y.to(dtype=torch.float32)))
                val_iou_ema = float(np.mean(ious_ema)) if ious_ema else float("nan")

        dt_ep = time.perf_counter() - t_ep
        if ema_model is None:
            if float(tail_lambda) > 0.0 and float(np_lambda) > 0.0:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f (base=%.5f tail=%.5f np=%.5f)  val_iou=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(train_loss_base),
                    float(train_loss_tail),
                    float(train_loss_np),
                    float(val_iou),
                    dt_ep,
                )
            elif float(tail_lambda) > 0.0:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f (base=%.5f tail=%.5f)  val_iou=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(train_loss_base),
                    float(train_loss_tail),
                    float(val_iou),
                    dt_ep,
                )
            elif float(np_lambda) > 0.0:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f (base=%.5f np=%.5f)  val_iou=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(train_loss_base),
                    float(train_loss_np),
                    float(val_iou),
                    dt_ep,
                )
            else:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f  val_iou=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(val_iou),
                    dt_ep,
                )
        else:
            if float(tail_lambda) > 0.0 and float(np_lambda) > 0.0:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f (base=%.5f tail=%.5f np=%.5f)  val_iou=%.4f  val_iou_ema=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(train_loss_base),
                    float(train_loss_tail),
                    float(train_loss_np),
                    float(val_iou),
                    float(val_iou_ema),
                    dt_ep,
                )
            elif float(tail_lambda) > 0.0:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f (base=%.5f tail=%.5f)  val_iou=%.4f  val_iou_ema=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(train_loss_base),
                    float(train_loss_tail),
                    float(val_iou),
                    float(val_iou_ema),
                    dt_ep,
                )
            elif float(np_lambda) > 0.0:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f (base=%.5f np=%.5f)  val_iou=%.4f  val_iou_ema=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(train_loss_base),
                    float(train_loss_np),
                    float(val_iou),
                    float(val_iou_ema),
                    dt_ep,
                )
            else:
                logger.info(
                    "Epoch %02d/%02d  loss=%.5f  val_iou=%.4f  val_iou_ema=%.4f  (%.2fs)",
                    epoch,
                    int(EPOCHS),
                    float(train_loss),
                    float(val_iou),
                    float(val_iou_ema),
                    dt_ep,
                )

        cloud_gate: Optional[Dict[str, float]] = None
        cloud_gate_ema: Optional[Dict[str, float]] = None
        if bool(ENABLE_CLOUD_GATE_EVAL) and ckpt_dir is not None and len(val_sids) > 0:
            every = int(CLOUD_GATE_EVAL_EVERY)
            if every > 0 and (int(epoch) % every == 0 or int(epoch) == int(EPOCHS)):
                cloud_gate = _eval_val_cloud_gate(
                    epoch=int(epoch), tag="model", sids=val_sids, model=model, device=device
                )
                if ema_model is not None:
                    cloud_gate_ema = _eval_val_cloud_gate(
                        epoch=int(epoch), tag="ema", sids=val_sids, model=ema_model, device=device
                    )

        if ckpt_dir is not None:
            last_path = ckpt_dir / "last.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": int(in_channels),
                    "model_cfg": MODEL_CFG,
                    "features": {
                        "use_rgb": bool(USE_RGB),
                        "use_density": bool(USE_DENSITY),
                        "use_color_constancy": bool(USE_COLOR_CONSTANCY),
                        "use_normals": bool(USE_NORMALS),
                        "use_global_xyz": bool(USE_GLOBAL_XYZ),
                        "density_k": int(DENSITY_K),
                        "density_k2": int(DENSITY_K2),
                        "density_mode": str(DENSITY_MODE),
                        "density_nn_median_mode": str(DENSITY_NN_MEDIAN_MODE),
                        "density_seed_sampling_mode": str(DENSITY_SEED_SAMPLING_MODE),
                        "density_seed_sampling_power": float(DENSITY_SEED_SAMPLING_POWER),
                        "density_loss_reweight_mode": str(DENSITY_LOSS_REWEIGHT_MODE),
                        "density_loss_reweight_k": int(DENSITY_LOSS_REWEIGHT_K),
                        "density_loss_reweight_power": float(DENSITY_LOSS_REWEIGHT_POWER),
                        "density_loss_reweight_clip": {
                            "min": float(DENSITY_LOSS_REWEIGHT_CLIP_MIN),
                            "max": float(DENSITY_LOSS_REWEIGHT_CLIP_MAX),
                        },
                    },
                    "split": {"fold_id": int(fold_id), "train_sids": train_sids, "val_sids": val_sids},
                    "train": {
                        "epoch": int(epoch),
                        "loss": float(train_loss),
                        "loss_base": float(train_loss_base) if np.isfinite(train_loss_base) else None,
                        "loss_tail": float(train_loss_tail) if np.isfinite(train_loss_tail) else None,
                        "loss_np": float(train_loss_np) if np.isfinite(train_loss_np) else None,
                        "loss_type": str(LOSS_TYPE),
                        "pos_weight_mode": str(POS_WEIGHT_MODE),
                        "pos_weight_auto": float(pos_weight_auto),
                        "pos_weight_used": float(pos_weight_used) if pos_weight_used is not None else None,
                        "ema_decay": float(ema_decay),
                        "tail_neg": {
                            "alpha": float(tail_alpha),
                            "lambda": float(tail_lambda),
                            "group_mode": str(tail_group_mode),
                        },
                        "np_soft_fn": {
                            "beta_fpr": float(np_beta),
                            "tau": float(np_tau),
                            "lambda": float(np_lambda),
                            "group_mode": str(np_group_mode),
                            "min_base_pts": int(np_min_base_pts),
                            "min_pin_pts": int(np_min_pin_pts),
                        },
                        "bce_asymm": {"w_pos": float(BCE_ASYMM_W_POS), "w_neg": float(BCE_ASYMM_W_NEG)},
                        "tversky": {
                            "alpha": float(TVERSKY_ALPHA),
                            "beta": float(TVERSKY_BETA),
                            "focal_gamma": float(FOCAL_TVERSKY_GAMMA),
                        },
                    },
                    "val": {
                        "iou": float(val_iou) if np.isfinite(val_iou) else None,
                        "iou_ema": float(val_iou_ema) if np.isfinite(val_iou_ema) else None,
                        "cloud_gate": cloud_gate,
                        "cloud_gate_cfg": {
                            "enabled": bool(ENABLE_CLOUD_GATE_EVAL),
                            "eval_every": int(CLOUD_GATE_EVAL_EVERY),
                            "thr_pin_iou": float(CLOUD_GATE_PIN_IOU_THR),
                            "thr_base_fp_rate": float(CLOUD_GATE_BASE_FP_THR),
                        },
                    },
                    "dbg": {"train": dbg_train, "val": dbg_val},
                    "run_id": str(_DAIMA_CTX.run_id),
                },
                last_path,
            )

            score = float(val_iou) if np.isfinite(val_iou) else float(-1.0)
            if score >= best_val_iou:
                best_val_iou = score
                best_path = ckpt_dir / "best.pt"
                if last_path is not None:
                    best_path.write_bytes(last_path.read_bytes())

            if cloud_gate is not None:
                rank = (
                    int(cloud_gate.get("fail_n", 1e9)),
                    float(cloud_gate.get("max_base_fp_rate", float("inf"))),
                    -float(cloud_gate.get("min_pin_iou", float("-inf"))),
                )
                if best_gate_rank is None or rank < best_gate_rank:
                    best_gate_rank = rank
                    best_gate_path = ckpt_dir / "best_gate.pt"
                    if last_path is not None:
                        best_gate_path.write_bytes(last_path.read_bytes())

            if ema_model is not None:
                last_ema_path = ckpt_dir / "last_ema.pt"
                torch.save(
                    {
                        "model": ema_model.state_dict(),
                        "in_channels": int(in_channels),
                        "model_cfg": MODEL_CFG,
                        "features": {
                            "use_rgb": bool(USE_RGB),
                            "use_density": bool(USE_DENSITY),
                            "use_color_constancy": bool(USE_COLOR_CONSTANCY),
                            "use_normals": bool(USE_NORMALS),
                            "use_global_xyz": bool(USE_GLOBAL_XYZ),
                            "density_k": int(DENSITY_K),
                            "density_k2": int(DENSITY_K2),
                            "density_mode": str(DENSITY_MODE),
                            "density_nn_median_mode": str(DENSITY_NN_MEDIAN_MODE),
                            "density_seed_sampling_mode": str(DENSITY_SEED_SAMPLING_MODE),
                            "density_seed_sampling_power": float(DENSITY_SEED_SAMPLING_POWER),
                            "density_loss_reweight_mode": str(DENSITY_LOSS_REWEIGHT_MODE),
                            "density_loss_reweight_k": int(DENSITY_LOSS_REWEIGHT_K),
                            "density_loss_reweight_power": float(DENSITY_LOSS_REWEIGHT_POWER),
                            "density_loss_reweight_clip": {
                                "min": float(DENSITY_LOSS_REWEIGHT_CLIP_MIN),
                                "max": float(DENSITY_LOSS_REWEIGHT_CLIP_MAX),
                            },
                        },
                        "split": {"fold_id": int(fold_id), "train_sids": train_sids, "val_sids": val_sids},
                        "train": {
                            "epoch": int(epoch),
                            "loss": float(train_loss),
                            "loss_base": float(train_loss_base) if np.isfinite(train_loss_base) else None,
                            "loss_tail": float(train_loss_tail) if np.isfinite(train_loss_tail) else None,
                            "loss_np": float(train_loss_np) if np.isfinite(train_loss_np) else None,
                            "loss_type": str(LOSS_TYPE),
                            "pos_weight_mode": str(POS_WEIGHT_MODE),
                            "pos_weight_auto": float(pos_weight_auto),
                            "pos_weight_used": float(pos_weight_used) if pos_weight_used is not None else None,
                            "ema_decay": float(ema_decay),
                            "tail_neg": {
                                "alpha": float(tail_alpha),
                                "lambda": float(tail_lambda),
                                "group_mode": str(tail_group_mode),
                            },
                            "np_soft_fn": {
                                "beta_fpr": float(np_beta),
                                "tau": float(np_tau),
                                "lambda": float(np_lambda),
                                "group_mode": str(np_group_mode),
                                "min_base_pts": int(np_min_base_pts),
                                "min_pin_pts": int(np_min_pin_pts),
                            },
                            "bce_asymm": {"w_pos": float(BCE_ASYMM_W_POS), "w_neg": float(BCE_ASYMM_W_NEG)},
                            "tversky": {
                                "alpha": float(TVERSKY_ALPHA),
                                "beta": float(TVERSKY_BETA),
                                "focal_gamma": float(FOCAL_TVERSKY_GAMMA),
                            },
                        },
                        "val": {
                            "iou": float(val_iou) if np.isfinite(val_iou) else None,
                            "iou_ema": float(val_iou_ema) if np.isfinite(val_iou_ema) else None,
                            "cloud_gate_ema": cloud_gate_ema,
                            "cloud_gate_cfg": {
                                "enabled": bool(ENABLE_CLOUD_GATE_EVAL),
                                "eval_every": int(CLOUD_GATE_EVAL_EVERY),
                                "thr_pin_iou": float(CLOUD_GATE_PIN_IOU_THR),
                                "thr_base_fp_rate": float(CLOUD_GATE_BASE_FP_THR),
                            },
                        },
                        "dbg": {"train": dbg_train, "val": dbg_val},
                        "run_id": str(_DAIMA_CTX.run_id),
                        "ema": {"enabled": True, "decay": float(ema_decay)},
                    },
                    last_ema_path,
                )

                score_ema = float(val_iou_ema) if np.isfinite(val_iou_ema) else float(-1.0)
                if score_ema >= best_val_iou_ema:
                    best_val_iou_ema = score_ema
                    best_ema_path = ckpt_dir / "best_ema.pt"
                    best_ema_path.write_bytes(last_ema_path.read_bytes())

                if cloud_gate_ema is not None:
                    rank_ema = (
                        int(cloud_gate_ema.get("fail_n", 1e9)),
                        float(cloud_gate_ema.get("max_base_fp_rate", float("inf"))),
                        -float(cloud_gate_ema.get("min_pin_iou", float("-inf"))),
                    )
                    if best_gate_ema_rank is None or rank_ema < best_gate_ema_rank:
                        best_gate_ema_rank = rank_ema
                        best_gate_ema_path = ckpt_dir / "best_gate_ema.pt"
                        if last_ema_path is not None:
                            best_gate_ema_path.write_bytes(last_ema_path.read_bytes())

    if _DAIMA_CTX.run_dir is not None:
        _DAIMA_CTX.snapshot_paths(
            outputs={
                "ckpt_best": str(best_path) if best_path is not None else "",
                "ckpt_best_ema": str(best_ema_path) if best_ema_path is not None else "",
                "ckpt_best_gate": str(best_gate_path) if best_gate_path is not None else "",
                "ckpt_best_gate_ema": str(best_gate_ema_path) if best_gate_ema_path is not None else "",
                "ckpt_last": str(last_path) if last_path is not None else "",
                "ckpt_last_ema": str(last_ema_path) if last_ema_path is not None else "",
            },
            extra={"train_sids": train_sids, "val_sids": val_sids},
        )

    if ema_model is None:
        logger.info("Done. best_val_iou=%.4f ckpt_best=%s", float(best_val_iou), str(best_path) if best_path else "")
    else:
        logger.info(
            "Done. best_val_iou=%.4f ckpt_best=%s | best_val_iou_ema=%.4f ckpt_best_ema=%s",
            float(best_val_iou),
            str(best_path) if best_path else "",
            float(best_val_iou_ema),
            str(best_ema_path) if best_ema_path else "",
        )


_COLOR_CC_CFG = ColorConstancyConfig()


if __name__ == "__main__":
    main()
