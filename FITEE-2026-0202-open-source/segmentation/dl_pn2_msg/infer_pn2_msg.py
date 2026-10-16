from __future__ import annotations

import sys
import os
import re
from pathlib import Path

# Allow running this file directly (python dl_pn2_msg/infer_pn2_msg.py) while importing repo-root modules.
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
from dl_pn2_msg.common import build_block_data, build_cloud, knn_block_indices, load_split_csv, safe_cc_cfg_dict
from dl_pn2_msg.model import PointNet2MSG, PointNet2MSGConfig
from pred_io import identity_mapping_meta, save_pred_npz


try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for dl_pn2_msg/infer_pn2_msg.py. "
        "Run inside the server docker image fenge_tp3d."
    ) from e

try:
    # Used for optional micro-batched block inference (faster inference throughput).
    from torch_geometric.data import Batch  # type: ignore
except Exception:
    Batch = None  # type: ignore


_DAIMA_CTX = setup_run(__file__)
logger = logging.getLogger("dl_pn2_msg.infer_pn2_msg")


def _get_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(str(v).strip())
    except Exception as e:
        raise ValueError(f"Invalid env {name}={v!r} (expected int)") from e


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


def _get_env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return str(default)
    return str(v)


# ------------------------------
# Editable defaults (PyCharm-run friendly)
# ------------------------------
DATA_DIR = Path(os.environ.get("DAIMA_POINT_CLOUD_DIR") or str(_DAIMA_CTX.repo_root / "private_inputs")).expanduser()
SPLIT_CSV = resolve_split_csv(_DAIMA_CTX.repo_root)

# If set, use this training run_id to locate ckpt; else auto-pick latest pn2_msg ckpt.
TRAIN_RUN_ID = ""  # e.g. "20260126_235959_123456_99999"

# Which checkpoint variant to load from ckpt/pn2_msg/:
#   - best      -> best.pt (default; backward compatible)
#   - best_gate -> best_gate.pt (cloud-gate selection; optional)
#   - best_ema  -> best_ema.pt (if training saved EMA checkpoints)
#   - best_gate_ema -> best_gate_ema.pt (cloud-gate selection on EMA; optional)
#   - last      -> last.pt
#   - last_ema  -> last_ema.pt
CKPT_NAME = "best"

# Which samples to run inference on:
# - if empty: use role=test from split csv (default = current split's test set)
SAMPLE_IDS: List[int] = []

# Must match the checkpoint features (we verify).
USE_RGB = True
USE_DENSITY = True
USE_COLOR_CONSTANCY = False
USE_NORMALS = False
USE_GLOBAL_XYZ = False  # global xyz normalized per-cloud (pos is block-normalized)

DENSITY_K = 16
DENSITY_K2 = 32
DENSITY_MODE = "knn_mean_std"  # see dl_pn2_msg/common.py:density_features
DENSITY_NN_MEDIAN_MODE = "random"  # random | deterministic
DENSITY_MAX_POINTS_FOR_NN_MED = 50000

# Block inference (full coverage via kNN blocks + logit_sum/count stitch-back).
BLOCK_POINTS = 4096
# How many blocks to run in one forward pass (micro-batch). 1 keeps the original behavior.
INFER_BLOCK_BATCH_SIZE = 1

# Env overrides (no code edit). Useful for Step3C/P2-3 automation on server.
_ENV_TRAIN_RUN_ID = "DAIMA_PN2_MSG_TRAIN_RUN_ID"
_ENV_INFER_SAMPLE_IDS = "DAIMA_PN2_MSG_INFER_SAMPLE_IDS"  # "all"/"test"/"trainval" or "1,2,3"
_ENV_CKPT_NAME = "DAIMA_PN2_MSG_CKPT_NAME"
_ENV_USE_RGB = "DAIMA_PN2_MSG_USE_RGB"
_ENV_USE_DENSITY = "DAIMA_PN2_MSG_USE_DENSITY"
_ENV_USE_COLOR_CONSTANCY = "DAIMA_PN2_MSG_USE_COLOR_CONSTANCY"
_ENV_USE_NORMALS = "DAIMA_PN2_MSG_USE_NORMALS"
_ENV_USE_GLOBAL_XYZ = "DAIMA_PN2_MSG_USE_GLOBAL_XYZ"
_ENV_DENSITY_K = "DAIMA_PN2_MSG_DENSITY_K"
_ENV_DENSITY_K2 = "DAIMA_PN2_MSG_DENSITY_K2"
_ENV_DENSITY_MODE = "DAIMA_PN2_MSG_DENSITY_MODE"
_ENV_DENSITY_NN_MEDIAN_MODE = "DAIMA_PN2_MSG_DENSITY_NN_MEDIAN_MODE"
_ENV_BLOCK_POINTS = "DAIMA_PN2_MSG_BLOCK_POINTS"
_ENV_INFER_BLOCK_BATCH_SIZE = "DAIMA_PN2_MSG_INFER_BLOCK_BATCH_SIZE"
# Safety: micro-batched inference can change outputs (not just speed) due to non-deterministic graph ops.
_ENV_INFER_ALLOW_DRIFT = "DAIMA_PN2_MSG_INFER_ALLOW_DRIFT"

TRAIN_RUN_ID = _get_env_str(_ENV_TRAIN_RUN_ID, str(TRAIN_RUN_ID)).strip()
CKPT_NAME = _get_env_str(_ENV_CKPT_NAME, str(CKPT_NAME)).strip()
USE_RGB = _get_env_bool(_ENV_USE_RGB, bool(USE_RGB))
USE_DENSITY = _get_env_bool(_ENV_USE_DENSITY, bool(USE_DENSITY))
USE_COLOR_CONSTANCY = _get_env_bool(_ENV_USE_COLOR_CONSTANCY, bool(USE_COLOR_CONSTANCY))
USE_NORMALS = _get_env_bool(_ENV_USE_NORMALS, bool(USE_NORMALS))
USE_GLOBAL_XYZ = _get_env_bool(_ENV_USE_GLOBAL_XYZ, bool(USE_GLOBAL_XYZ))
DENSITY_K = _get_env_int(_ENV_DENSITY_K, int(DENSITY_K))
DENSITY_K2 = _get_env_int(_ENV_DENSITY_K2, int(DENSITY_K2))
DENSITY_MODE = _get_env_str(_ENV_DENSITY_MODE, str(DENSITY_MODE)).strip()
DENSITY_NN_MEDIAN_MODE = _get_env_str(_ENV_DENSITY_NN_MEDIAN_MODE, str(DENSITY_NN_MEDIAN_MODE)).strip()
BLOCK_POINTS = _get_env_int(_ENV_BLOCK_POINTS, int(BLOCK_POINTS))
INFER_BLOCK_BATCH_SIZE = _get_env_int(_ENV_INFER_BLOCK_BATCH_SIZE, int(INFER_BLOCK_BATCH_SIZE))
INFER_ALLOW_DRIFT = _get_env_bool(_ENV_INFER_ALLOW_DRIFT, False)
if int(INFER_BLOCK_BATCH_SIZE) < 1:
    raise ValueError(f"Invalid {_ENV_INFER_BLOCK_BATCH_SIZE}={INFER_BLOCK_BATCH_SIZE!r}. Expected >= 1.")
if int(INFER_BLOCK_BATCH_SIZE) > 1 and not bool(INFER_ALLOW_DRIFT):
    raise RuntimeError(
        f"{_ENV_INFER_BLOCK_BATCH_SIZE}={int(INFER_BLOCK_BATCH_SIZE)} may change outputs (not just speed). "
        f"For any rigorous eval, keep {_ENV_INFER_BLOCK_BATCH_SIZE}=1. "
        f"To proceed anyway, set {_ENV_INFER_ALLOW_DRIFT}=1."
    )


_COLOR_CC_CFG = ColorConstancyConfig()


def _find_ckpt(repo_root: Path) -> Path:
    runs_dir = Path(repo_root) / "debug_out" / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"debug_out/runs not found: {runs_dir}")

    ckpt_name = str(CKPT_NAME).strip().lower()
    if ckpt_name in {"best", "best.pt"}:
        ckpt_file = "best.pt"
    elif ckpt_name in {"best_gate", "gate", "best_gate.pt"}:
        ckpt_file = "best_gate.pt"
    elif ckpt_name in {"best_ema", "ema", "best_ema.pt"}:
        ckpt_file = "best_ema.pt"
    elif ckpt_name in {"best_gate_ema", "gate_ema", "best_gate_ema.pt"}:
        ckpt_file = "best_gate_ema.pt"
    elif ckpt_name in {"last", "last.pt"}:
        ckpt_file = "last.pt"
    elif ckpt_name in {"last_ema", "last_ema.pt"}:
        ckpt_file = "last_ema.pt"
    else:
        raise ValueError(
            f"Invalid {_ENV_CKPT_NAME}={CKPT_NAME!r}. Expected one of: best, best_gate, best_ema, best_gate_ema, last, last_ema."
        )

    if TRAIN_RUN_ID:
        p = runs_dir / str(TRAIN_RUN_ID) / "ckpt" / "pn2_msg" / str(ckpt_file)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found for TRAIN_RUN_ID={TRAIN_RUN_ID}: {p}")
        return p

    cands = list(runs_dir.glob(f"*/ckpt/pn2_msg/{ckpt_file}"))
    if not cands:
        raise FileNotFoundError(f"No pn2_msg checkpoints '{ckpt_file}' found under: {runs_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_sample_ids(split_csv: Path) -> List[int]:
    env_sel = os.environ.get(_ENV_INFER_SAMPLE_IDS)
    if env_sel is not None and str(env_sel).strip() != "":
        v = str(env_sel).strip()
        v_norm = v.lower()
        split_map = load_split_csv(split_csv)
        if v_norm in {"all", "*"}:
            return sorted([int(s) for s in split_map.keys()])
        if v_norm in {"test"}:
            return sorted([int(s) for s, (role, _f) in split_map.items() if role == "test"])
        if v_norm in {"trainval", "train"}:
            return sorted([int(s) for s, (role, _f) in split_map.items() if role == "trainval"])
        parts = [p for p in re.split(r"[,\s]+", v) if p.strip() != ""]
        try:
            return [int(p) for p in parts]
        except Exception as e:
            raise ValueError(
                f"Invalid env {_ENV_INFER_SAMPLE_IDS}={v!r}. Expected 'all'/'test'/'trainval' or a comma/space-separated list of ints."
            ) from e

    if SAMPLE_IDS:
        return [int(x) for x in SAMPLE_IDS]
    split_map = load_split_csv(split_csv)
    return sorted([sid for sid, (role, _f) in split_map.items() if role == "test"])


def _infer_one(
    *,
    sid: int,
    model: PointNet2MSG,
    device: str,
) -> Tuple[Path, Path]:
    ply_path = Path(DATA_DIR) / f"{int(sid)}.ply"
    xyz, x, _y, cloud_dbg = build_cloud(
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
        rng=np.random.default_rng(0),
    )

    n = int(xyz.shape[0])
    tree = cKDTree(xyz.astype(np.float32, copy=False))

    logit_sum = np.zeros((n,), dtype=np.float32)
    count = np.zeros((n,), dtype=np.int32)

    # Safety: in worst case each block covers only 1 new point.
    max_blocks = int(n)
    n_blocks = 0

    t0 = time.perf_counter()
    while int(np.sum(count > 0)) < n:
        # Deterministic: pick the earliest uncovered seed, same as the original code.
        # We can still micro-batch by *building* multiple blocks sequentially (updating `count` for coverage),
        # then doing a single forward pass on the concatenated Batch. This preserves block selection behavior.
        batch_n_target = int(INFER_BLOCK_BATCH_SIZE)
        idx_list: List[np.ndarray] = []
        data_list: List["torch_geometric.data.Data"] = []
        sizes: List[int] = []
        for _ in range(int(batch_n_target)):
            uncovered = np.flatnonzero(count == 0)
            if int(uncovered.shape[0]) <= 0:
                break
            seed_idx = int(uncovered[0])
            idx = knn_block_indices(tree, xyz, seed_idx=int(seed_idx), k=int(BLOCK_POINTS))
            data, _dbg = build_block_data(xyz=xyz, x=x, y=np.zeros((n,), dtype=np.int64), idx=idx)
            idx_list.append(idx)
            data_list.append(data)
            sizes.append(int(data.pos.shape[0]))
            # Update coverage/count immediately so subsequent seeds match the sequential logic.
            count[idx] += 1
            n_blocks += 1
            if n_blocks >= max_blocks:
                raise RuntimeError(f"Exceeded max_blocks={max_blocks} while trying to cover all points (n={n}).")
        if not data_list:
            break

        if len(data_list) == 1:
            data = data_list[0].to(device)
            with torch.no_grad():
                logits_b = model(
                    x=data.x,
                    pos=data.pos,
                    batch=torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device),
                )
            logits_np = logits_b.detach().float().cpu().numpy().astype(np.float32, copy=False)
            idx0 = idx_list[0]
            logit_sum[idx0] += logits_np
        else:
            if Batch is None:
                raise RuntimeError(
                    "torch_geometric is required for micro-batched inference. "
                    f"Set {_ENV_INFER_BLOCK_BATCH_SIZE}=1 or run inside fenge_tp3d."
                )
            batch = Batch.from_data_list(data_list).to(device)
            with torch.no_grad():
                logits_b = model(x=batch.x, pos=batch.pos, batch=batch.batch)
            logits_np = logits_b.detach().float().cpu().numpy().astype(np.float32, copy=False)
            off = 0
            for i, idx in enumerate(idx_list):
                n_i = int(sizes[i])
                logit_sum[idx] += logits_np[off : off + n_i]
                off += n_i
            if int(off) != int(logits_np.shape[0]):
                raise RuntimeError(f"Unexpected Batch split: off={off} vs logits={int(logits_np.shape[0])}")

    dt = time.perf_counter() - t0
    logits = logit_sum / np.maximum(count.astype(np.float32), 1.0)
    pin_prob = 1.0 / (1.0 + np.exp(-logits))

    meta = identity_mapping_meta(n_gt=int(n))
    meta.update(
        {
            "algo": "pn2_msg",
            "sample_id": int(sid),
            "block_points": int(BLOCK_POINTS),
            "infer_block_batch_size": int(INFER_BLOCK_BATCH_SIZE),
            "n_blocks": int(n_blocks),
            "count_min": int(np.min(count)),
            "count_p50": float(np.median(count)),
            "count_max": int(np.max(count)),
            "time_s": float(dt),
            "use_rgb": bool(USE_RGB),
            "use_density": bool(USE_DENSITY),
            "use_color_constancy": bool(USE_COLOR_CONSTANCY),
            "density_k": int(DENSITY_K),
            "density_k2": int(DENSITY_K2),
            "density_mode": str(DENSITY_MODE),
            "density_nn_median_mode": str(DENSITY_NN_MEDIAN_MODE),
            **{f"cloud_{k}": float(v) for k, v in cloud_dbg.items() if np.isfinite(v)},
        }
    )

    if _DAIMA_CTX.run_dir is None:
        raise RuntimeError("DAIMA run_dir is None; cannot write outputs.")
    return save_pred_npz(out_dir=_DAIMA_CTX.run_dir, algo_name="pn2_msg", sample_id=int(sid), pin_prob=pin_prob, meta=meta)


def main() -> None:
    logger.info("Repo root: %s", _DAIMA_CTX.repo_root)
    logger.info("Data dir : %s", DATA_DIR)
    logger.info("Split csv: %s", SPLIT_CSV)
    logger.info("Run dir  : %s", _DAIMA_CTX.run_dir)

    ckpt_path = _find_ckpt(_DAIMA_CTX.repo_root)
    logger.info("Checkpoint: %s", ckpt_path)

    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Split file not found: {SPLIT_CSV}")
    sids = _resolve_sample_ids(SPLIT_CSV)
    if not sids:
        raise RuntimeError("No sample ids selected for inference.")
    logger.info("Infer sample_ids=%s", sids)

    _DAIMA_CTX.snapshot_paths(
        inputs={"data_dir": str(DATA_DIR), "split_csv": str(SPLIT_CSV), "ckpt": str(ckpt_path)},
        outputs={"run_dir": str(_DAIMA_CTX.run_dir)},
    )
    _DAIMA_CTX.snapshot_config(
        config={
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
                "color_cc_cfg": safe_cc_cfg_dict(_COLOR_CC_CFG),
            },
            "blocks": {
                "block_points": float(int(BLOCK_POINTS)),
                "infer_block_batch_size": float(int(INFER_BLOCK_BATCH_SIZE)),
                "infer_allow_drift": float(1.0 if INFER_ALLOW_DRIFT else 0.0),
            },
            "ckpt": {"path": str(ckpt_path), "name": str(CKPT_NAME)},
            "sample_ids": [float(int(s)) for s in sids],
        }
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_channels = int(ckpt.get("in_channels", 0))
    if in_channels <= 0:
        raise RuntimeError(f"Invalid checkpoint in_channels={in_channels}: {ckpt_path}")

    ckpt_features = ckpt.get("features", {})
    want = {
        "use_rgb": bool(USE_RGB),
        "use_density": bool(USE_DENSITY),
        "use_color_constancy": bool(USE_COLOR_CONSTANCY),
        "use_normals": bool(USE_NORMALS),
        "use_global_xyz": bool(USE_GLOBAL_XYZ),
    }
    have = {
        "use_rgb": bool(ckpt_features.get("use_rgb", False)),
        "use_density": bool(ckpt_features.get("use_density", False)),
        "use_color_constancy": bool(ckpt_features.get("use_color_constancy", False)),
        "use_normals": bool(ckpt_features.get("use_normals", False)),
        "use_global_xyz": bool(ckpt_features.get("use_global_xyz", False)),
    }
    if bool(USE_DENSITY):
        # Keep inference strictly aligned with the density feature spec used in training.
        want.update(
            {
                "density_k": int(DENSITY_K),
                "density_k2": int(DENSITY_K2),
                "density_mode": str(DENSITY_MODE),
                "density_nn_median_mode": str(DENSITY_NN_MEDIAN_MODE),
            }
        )
        have.update(
            {
                "density_k": int(ckpt_features.get("density_k", int(DENSITY_K))),
                "density_k2": int(ckpt_features.get("density_k2", int(DENSITY_K2))),
                "density_mode": str(ckpt_features.get("density_mode", "knn_mean_std")),
                "density_nn_median_mode": str(ckpt_features.get("density_nn_median_mode", "random")),
            }
        )
    if have != want:
        raise RuntimeError(f"Feature flags mismatch: ckpt={have} current={want}. Update USE_* constants to match.")

    model_cfg = ckpt.get("model_cfg", None)
    if model_cfg is None:
        model_cfg = PointNet2MSGConfig()
    model = PointNet2MSG(in_channels=int(in_channels), cfg=model_cfg).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info("Device: %s", device)

    t_all = time.perf_counter()
    out_paths: List[Tuple[Path, Path]] = []
    for sid in sids:
        logger.info("Infer sid=%s", int(sid))
        npz_path, json_path = _infer_one(sid=int(sid), model=model, device=device)
        out_paths.append((npz_path, json_path))
        logger.info("Wrote: %s", npz_path)
    dt_all = time.perf_counter() - t_all

    if _DAIMA_CTX.run_dir is not None:
        _DAIMA_CTX.snapshot_paths(
            outputs={
                "pred_npz_dir": str(_DAIMA_CTX.run_dir / "pred_npz" / "pn2_msg"),
                "pred_files": [str(p[0]) for p in out_paths],
            },
            extra={"time_s": float(dt_all)},
        )

    logger.info("Done. n=%s time_s=%.2f", len(out_paths), float(dt_all))


if __name__ == "__main__":
    main()
