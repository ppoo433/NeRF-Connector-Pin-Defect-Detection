from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for dl_pn2_msg. Run inside a torch-enabled environment "
        "(e.g. server docker fenge_tp3d)."
    ) from e


try:
    from torch_cluster import radius
    from torch_geometric.nn import PointNetConv, knn_interpolate
    from torch_geometric.nn.pool import fps
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch Geometric (PyG) is required for dl_pn2_msg. "
        "Make sure torch_geometric + torch_cluster are installed."
    ) from e


def _mlp(channels: Sequence[int], *, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(channels) - 1):
        in_ch = int(channels[i])
        out_ch = int(channels[i + 1])
        layers.append(nn.Linear(in_ch, out_ch))
        if i != len(channels) - 2:
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class PointNet2MSGConfig:
    # Set abstraction sampling ratios.
    sa1_ratio: float = 0.25
    sa2_ratio: float = 0.25

    # MSG radii are expressed in the same coordinate system as `pos`.
    # Our pipeline normalizes each block to roughly [-0.5, 0.5]^3.
    sa1_radii: tuple[float, ...] = (0.05, 0.10)
    sa2_radii: tuple[float, ...] = (0.10, 0.20)

    # Limit neighbor count for radius graph construction.
    max_num_neighbors: int = 64

    # Interpolation k for feature propagation.
    fp_knn_k: int = 3

    # Dropout (only used in classifier MLPs).
    dropout: float = 0.1


class _SAModuleMSG(nn.Module):
    def __init__(
        self,
        *,
        ratio: float,
        radii: Sequence[float],
        in_channels: int,
        out_channels: int,
        max_num_neighbors: int,
    ) -> None:
        super().__init__()
        self.ratio = float(ratio)
        self.radii = [float(r) for r in radii]
        self.max_num_neighbors = int(max_num_neighbors)

        convs: List[nn.Module] = []
        for _r in self.radii:
            # PointNetConv internally concatenates relative positions (dx,dy,dz),
            # hence `in_channels + 3`.
            local_nn = _mlp([int(in_channels) + 3, out_channels, out_channels])
            global_nn = _mlp([out_channels, out_channels])
            convs.append(PointNetConv(local_nn=local_nn, global_nn=global_nn))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep FPS stochastic during training, but deterministic during eval/inference to avoid metric drift.
        idx = fps(pos, batch=batch, ratio=self.ratio, random_start=bool(self.training))
        pos_s = pos[idx]
        batch_s = batch[idx]

        xs: List[torch.Tensor] = []
        for conv, r in zip(self.convs, self.radii):
            # torch_cluster.radius(x, y, r) returns (row, col) with:
            #   row in [0, |y|), col in [0, |x|)
            # i.e. edges from x[col] -> y[row].
            row, col = radius(pos, pos_s, r, batch_x=batch, batch_y=batch_s, max_num_neighbors=self.max_num_neighbors)
            edge_index = torch.stack([col, row], dim=0)
            xs.append(conv((x, x[idx]), (pos, pos_s), edge_index))
        x_s = torch.cat(xs, dim=1)
        return x_s, pos_s, batch_s


class _FPModule(nn.Module):
    def __init__(self, *, k: int, mlp_channels: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.k = int(k)
        self.mlp = _mlp(mlp_channels, dropout=float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        x_skip: torch.Tensor,
        pos_skip: torch.Tensor,
        batch_skip: torch.Tensor,
    ) -> torch.Tensor:
        x_interp = knn_interpolate(x, pos, pos_skip, batch_x=batch, batch_y=batch_skip, k=self.k)
        x_cat = torch.cat([x_interp, x_skip], dim=1)
        return self.mlp(x_cat)


class PointNet2MSG(nn.Module):
    """
    Minimal PointNet++ MSG-style backbone for per-point binary segmentation.

    - Uses PyG PointNetConv + fps + radius + knn_interpolate.
    - Assumes `pos` is normalized per-block to a comparable scale.
    """

    def __init__(self, *, in_channels: int, cfg: PointNet2MSGConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PointNet2MSGConfig()
        self.cfg = cfg

        # Set abstraction: two levels, MSG at each level.
        self.sa1 = _SAModuleMSG(
            ratio=cfg.sa1_ratio,
            radii=cfg.sa1_radii,
            in_channels=in_channels,
            out_channels=64,
            max_num_neighbors=cfg.max_num_neighbors,
        )
        sa1_out = 64 * len(cfg.sa1_radii)

        self.sa2 = _SAModuleMSG(
            ratio=cfg.sa2_ratio,
            radii=cfg.sa2_radii,
            in_channels=sa1_out,
            out_channels=128,
            max_num_neighbors=cfg.max_num_neighbors,
        )
        sa2_out = 128 * len(cfg.sa2_radii)

        # Feature propagation back to original points.
        self.fp2 = _FPModule(k=cfg.fp_knn_k, mlp_channels=[sa2_out + sa1_out, 256, 256], dropout=cfg.dropout)
        self.fp1 = _FPModule(k=cfg.fp_knn_k, mlp_channels=[256 + in_channels, 256, 128], dropout=cfg.dropout)

        self.classifier = _mlp([128, 128, 64, 1], dropout=cfg.dropout)

    def forward(self, *, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x0, pos0, batch0 = x, pos, batch

        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)

        x1_up = self.fp2(x=x2, pos=pos2, batch=batch2, x_skip=x1, pos_skip=pos1, batch_skip=batch1)
        x0_up = self.fp1(x=x1_up, pos=pos1, batch=batch1, x_skip=x0, pos_skip=pos0, batch_skip=batch0)

        logits = self.classifier(x0_up).squeeze(-1)
        return logits
