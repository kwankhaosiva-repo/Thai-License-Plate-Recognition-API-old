"""
src/models_plate_quad.py

PlateOBBNet: 1-Model End-to-End License Plate Rotated Bounding Box (OBB) Detector
License: 100% BSD-3 / Apache-2.0 Compatible (No YOLO / No AGPL dependencies)

Architecture:
- Backbone: Torchvision ResNet18 (Pretrained, BSD-3)
- Decoder: Feature Pyramid Network / Deconvolution to Stride 4 (160x160 for 640x640 input)
- Heads:
  1. Heatmap Head (1 channel): Predicts plate center existence (Focal Loss)
  2. Subpixel Offset Head (2 channels): Refines discretization quantization (L1 Loss)
  3. WH Size Head (2 channels): Predicts width and height (w, h) in normalized image space
  4. Direction Vector Head (2 channels): Predicts continuous orientation unit vector u = [cos θ, sin θ]
- Deployment Wrapper:
  Reconstructs strictly parallel, rectangular 4-corner coordinates in standard ONNX operations:
    P0, P1, P2, P3 = Center ± (w/2)*u ± (h/2)*v
  Outputs `corners: (B, K, 4, 2)` and `scores: (B, K)` in a single pass.
  Zero custom C++ operators; runs directly in C# Microsoft.ML.OnnxRuntime.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PlateQuadNet(nn.Module):
    """
    PlateOBBNet Core Network.
    Predicts:
      - heatmap: (B, 1, H/4, W/4)
      - offset:  (B, 2, H/4, W/4) subpixel center offset
      - wh:      (B, 2, H/4, W/4) width, height normalized in [0, 1]
      - vec:     (B, 2, H/4, W/4) directional vector [ux, uy] along plate width
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        # ResNet18 Backbone
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)  # Stride 4 (H/4, W/4, 64)
        self.layer1 = base.layer1  # Stride 4, 64 ch
        self.layer2 = base.layer2  # Stride 8, 128 ch
        self.layer3 = base.layer3  # Stride 16, 256 ch
        self.layer4 = base.layer4  # Stride 32, 512 ch

        # Lateral / Up-sampling connections (Decoder to Stride 4)
        self.lat4 = nn.Conv2d(512, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat2 = nn.Conv2d(128, 64, kernel_size=1)
        self.lat1 = nn.Conv2d(64, 64, kernel_size=1)

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.smooth = ConvBNReLU(64, 64, kernel_size=3, padding=1)

        # Prediction Heads (Stride 4: 160x160 for 640x640 input)
        # 1. Heatmap Head: 1 channel, plate center probability
        self.heatmap_head = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 2. Subpixel Center Offset Head: 2 channels (gx - ix, gy - iy)
        self.offset_head = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        # 3. WH Size Head: 2 channels (width, height)
        self.wh_head = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(),  # Enforce positive dimensions
        )

        # 4. Direction Vector Head: 2 channels [ux, uy] pointing along width
        self.vec_head = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self._init_heads()

    def _init_heads(self):
        # Initialize heatmap bias to -2.19 (prob ~= 0.1) for focal loss stability
        nn.init.constant_(self.heatmap_head[1].bias, -2.19)
        nn.init.normal_(self.offset_head[1].weight, std=0.001)
        nn.init.constant_(self.offset_head[1].bias, 0.0)
        nn.init.normal_(self.wh_head[1].weight, std=0.001)
        nn.init.constant_(self.wh_head[1].bias, 0.1)
        nn.init.normal_(self.vec_head[1].weight, std=0.001)
        # Default direction points horizontal: [ux=1.0, uy=0.0]
        nn.init.constant_(self.vec_head[1].bias[0], 1.0)
        nn.init.constant_(self.vec_head[1].bias[1], 0.0)

    def forward(self, x: torch.Tensor):
        # Backbone forward
        c1 = self.layer1(self.stem(x))  # H/4, 64
        c2 = self.layer2(c1)             # H/8, 128
        c3 = self.layer3(c2)             # H/16, 256
        c4 = self.layer4(c3)             # H/32, 512

        # Decoder FPN upsampling
        p4 = self.lat4(c4)
        p3 = self.up4(p4) + self.lat3(c3)
        p2 = self.up3(p3) + self.lat2(c2)
        p1 = self.up2(p2) + self.lat1(c1)

        feat = self.smooth(p1)

        heatmap = self.heatmap_head(feat)
        offset = self.offset_head(feat)
        wh = self.wh_head(feat) + 1e-4  # Ensure strictly positive
        vec = self.vec_head(feat)
        # Normalize direction vector to unit norm
        vec = F.normalize(vec, p=2, dim=1, eps=1e-6)

        return heatmap, offset, wh, vec


class PlateOBBLoss(nn.Module):
    """
    CenterNet + Rotated Rectangle (OBB) Loss:
    - Focal Loss for heatmap
    - L1 Loss for subpixel center offset
    - Smooth L1 for (w, h)
    - Cosine / L1 Loss for orientation unit vector
    - Direct Geometric Corner Loss on reconstructed rectangular corners
    """
    def __init__(
        self,
        wh_weight: float = 2.0,
        vec_weight: float = 2.0,
        corner_weight: float = 6.0,
        offset_weight: float = 1.0,
    ):
        super().__init__()
        self.wh_weight = wh_weight
        self.vec_weight = vec_weight
        self.corner_weight = corner_weight
        self.offset_weight = offset_weight

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_inds = target.eq(1.0).float()
        neg_inds = target.lt(1.0).float()

        neg_weights = torch.pow(1.0 - target, 4.0)

        loss = torch.tensor(0.0, device=pred.device)
        pos_pred = pred.clamp(1e-6, 1.0 - 1e-6)

        pos_loss = torch.log(pos_pred) * torch.pow(1.0 - pos_pred, 2.0) * pos_inds
        neg_loss = torch.log(1.0 - pos_pred) * torch.pow(pos_pred, 2.0) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def forward(
        self,
        preds: tuple,
        target_hm: torch.Tensor,
        target_offset: torch.Tensor,
        target_wh: torch.Tensor,
        target_vec: torch.Tensor,
        target_corners: torch.Tensor,
        mask: torch.Tensor,
    ):
        pred_hm, pred_offset, pred_wh, pred_vec = preds

        # 1. Heatmap Focal Loss
        loss_hm = self.focal_loss(pred_hm, target_hm)

        # 2. Positive Mask
        pos_mask = mask.bool()
        num_pos = pos_mask.sum().float().clamp(min=1.0)

        if pos_mask.sum() > 0:
            # Subpixel offset
            p_off = pred_offset.permute(0, 2, 3, 1)[pos_mask]
            t_off = target_offset.permute(0, 2, 3, 1)[pos_mask]
            loss_offset = F.l1_loss(p_off, t_off, reduction="sum") / num_pos

            # Size (w, h)
            p_wh = pred_wh.permute(0, 2, 3, 1)[pos_mask]
            t_wh = target_wh.permute(0, 2, 3, 1)[pos_mask]
            loss_wh = F.smooth_l1_loss(p_wh, t_wh, reduction="sum") / num_pos

            # Direction unit vector
            p_vec = pred_vec.permute(0, 2, 3, 1)[pos_mask]
            t_vec = target_vec.permute(0, 2, 3, 1)[pos_mask]
            loss_vec = F.smooth_l1_loss(p_vec, t_vec, reduction="sum") / num_pos

            # Geometric Corner Loss (Enforces that the rigid rectangle tightly bounds GT)
            # Reconstruct corners from predicted (cx, cy, w, h, u)
            B, _, H, W = pred_hm.shape
            # Target center coordinates in normalized space
            # target_corners: (B, 8, H, W)
            t_corn = target_corners.permute(0, 2, 3, 1)[pos_mask]  # (N, 8)
            cx = (t_corn[:, 0] + t_corn[:, 2] + t_corn[:, 4] + t_corn[:, 6]) / 4.0
            cy = (t_corn[:, 1] + t_corn[:, 3] + t_corn[:, 5] + t_corn[:, 7]) / 4.0

            hw = p_wh[:, 0] * 0.5
            hh = p_wh[:, 1] * 0.5
            ux = p_vec[:, 0]
            uy = p_vec[:, 1]
            vx = -uy
            vy = ux

            # 4 Corners (TL, TR, BR, BL)
            p0_x = cx - hw * ux - hh * vx
            p0_y = cy - hw * uy - hh * vy
            p1_x = cx + hw * ux - hh * vx
            p1_y = cy + hw * uy - hh * vy
            p2_x = cx + hw * ux + hh * vx
            p2_y = cy + hw * uy + hh * vy
            p3_x = cx - hw * ux + hh * vx
            p3_y = cy - hw * uy + hh * vy

            p_corn = torch.stack([p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y], dim=-1)
            loss_corners = F.smooth_l1_loss(p_corn, t_corn, reduction="sum") / num_pos
        else:
            loss_offset = torch.tensor(0.0, device=pred_hm.device)
            loss_wh = torch.tensor(0.0, device=pred_hm.device)
            loss_vec = torch.tensor(0.0, device=pred_hm.device)
            loss_corners = torch.tensor(0.0, device=pred_hm.device)

        total_loss = (
            loss_hm
            + self.offset_weight * loss_offset
            + self.wh_weight * loss_wh
            + self.vec_weight * loss_vec
            + self.corner_weight * loss_corners
        )
        return total_loss, loss_hm, loss_corners, loss_wh, loss_vec


class PlateQuadNetDeploy(nn.Module):
    """
    Self-Decoding ONNX Deployment Wrapper.
    Reconstructs strictly rectangular 4-corner coordinates in [0, 1] normalized image space
    using pure addition/multiplication (No custom C++ operators).

    Outputs:
      corners: (B, K, 4, 2) [Ordered clockwise: Top-Left, Top-Right, Bottom-Right, Bottom-Left]
      scores:  (B, K)       [Confidence scores in (0, 1)]
    """
    def __init__(self, core_model: nn.Module, topk: int = 3):
        super().__init__()
        self.core = core_model
        self.topk = topk

    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W)
        hm, offset, wh, vec = self.core(x)
        B, _, H, W = hm.shape

        # 3x3 Maxpool NMS on heatmap
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        keep = (hm == hmax).float()
        nms_hm = hm * keep

        # Flatten heatmap to find Top-K peaks
        flat_hm = nms_hm.view(B, -1)
        topk_scores, topk_inds = torch.topk(flat_hm, self.topk, dim=1)  # (B, K), (B, K)

        gy = (topk_inds // W).long()  # (B, K)
        gx = (topk_inds % W).long()   # (B, K)

        # Batch indexing
        b_idx = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, self.topk)  # (B, K)

        # Gather predictions at top-K peaks
        offset_perm = offset.permute(0, 2, 3, 1)  # (B, H, W, 2)
        wh_perm = wh.permute(0, 2, 3, 1)          # (B, H, W, 2)
        vec_perm = vec.permute(0, 2, 3, 1)        # (B, H, W, 2)

        k_offset = offset_perm[b_idx, gy, gx, :]  # (B, K, 2)
        k_wh = wh_perm[b_idx, gy, gx, :]          # (B, K, 2)
        k_vec = vec_perm[b_idx, gy, gx, :]        # (B, K, 2)

        # Plate center normalized in [0, 1]
        cx = (gx.float() + k_offset[:, :, 0]) / float(W)
        cy = (gy.float() + k_offset[:, :, 1]) / float(H)

        # Half width & height
        hw = k_wh[:, :, 0] * 0.5
        hh = k_wh[:, :, 1] * 0.5

        # Orthogonal directional unit vectors
        ux = k_vec[:, :, 0]
        uy = k_vec[:, :, 1]
        vx = -uy
        vy = ux

        # 4 Strictly Rectangular Corners in normalized [0, 1]
        # TL (P0)
        c1_x = (cx - hw * ux - hh * vx).clamp(0.0, 1.0)
        c1_y = (cy - hw * uy - hh * vy).clamp(0.0, 1.0)

        # TR (P1)
        c2_x = (cx + hw * ux - hh * vx).clamp(0.0, 1.0)
        c2_y = (cy + hw * uy - hh * vy).clamp(0.0, 1.0)

        # BR (P2)
        c3_x = (cx + hw * ux + hh * vx).clamp(0.0, 1.0)
        c3_y = (cy + hw * uy + hh * vy).clamp(0.0, 1.0)

        # BL (P3)
        c4_x = (cx - hw * ux + hh * vx).clamp(0.0, 1.0)
        c4_y = (cy - hw * uy + hh * vy).clamp(0.0, 1.0)

        # Stack into (B, K, 4, 2)
        p1 = torch.stack([c1_x, c1_y], dim=-1)
        p2 = torch.stack([c2_x, c2_y], dim=-1)
        p3 = torch.stack([c3_x, c3_y], dim=-1)
        p4 = torch.stack([c4_x, c4_y], dim=-1)

        quads = torch.stack([p1, p2, p3, p4], dim=2)  # (B, K, 4, 2)

        return quads, topk_scores
