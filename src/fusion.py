"""
跨模态融合模型: 2D 外观 + 3D 几何 → 3D bbox 残差回归.

模型三兄弟:
  ImageOnlyRefiner (C1):  纯 2D — Lightweight2DExtractor → pool → MLP
  LidarOnlyRefiner (C2):  纯 3D — PointNet2Encoder → pool → MLP
  CrossModalFusion  (C3): 融合 — 2D + 3D + 三阶段 cross-attention → MLP

C3 流程:
  RGB crop → Lightweight2D (Conv Stem + Self-Attn) → 2D tokens (64, 256)
  LiDAR   → PointNet++ 3×SA → 3D tokens (8, 256) → proj → (8, 256)
  阶段1: 3D Self-Attn  — 建模点云内部的几何关系
  阶段2: 2D ← 3D Cross — 2D token 查询 3D 几何
  阶段3: 3D ← 2D Cross — 3D token 查询几何感知后的视觉特征
  pool → MLP → (8,) 残差

设计原则:
  - 3D 坐标是主干信息 (~149K), 2D 外观是辅助信号 (~41K)
  - 辅助:主干 = 0.27x
  - 2D/3D 通过内外参投影在空间上耦合 (YOLO bbox ↔ LiDAR 点)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import SetAbstraction, farthest_point_sample


# ==============================================================================
# 2D 特征提取: 轻量 Conv Stem + Self-Attention → 2D tokens
# ==============================================================================

class Lightweight2DExtractor(nn.Module):
    """轻量 2D 特征提取: 2 层 Conv + Self-Attention → 2D tokens.

    检测框内物体外观多样性远小于 ImageNet, 无需深层网络.
    无残差连接 + 输入域窄 → 2 层 Conv + 小通道足够.

    输入:  (B, 3, 128, 128)  RGB crop (YOLO 检测框区域)
    输出:  (B, 64, d_model)   2D tokens

    下采样: 128 → 16 → 8 (stride=8, 2)
    通道:    3 → 16 → 48
    参数量: ~9K (conv) + ~18K (self-attn, d=48) + ~12K (proj) ≈ 39K
    """

    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        inner_dim = 48  # 仅为 d_model 的 19%, 窄域输入不需要高维

        # ---- Conv Stem: 2 层 (检测框内物体结构简单) ----
        self.stem = nn.Sequential(
            # 128 → 16 (stride=8, 大核看全局)
            nn.Conv2d(3, 16, 7, stride=8, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            # 16 → 8
            nn.Conv2d(16, inner_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inner_dim), nn.ReLU(inplace=True),
        )

        # ---- Self-Attention (在 inner_dim 空间, 极轻) ----
        self.self_attn = nn.MultiheadAttention(
            inner_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.ffn = nn.Sequential(
            nn.Linear(inner_dim, inner_dim * 2),   # ffn_expand=2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim * 2, inner_dim),
            nn.Dropout(dropout),
        )

        # ---- 投影到 d_model ----
        self.proj = nn.Linear(inner_dim, d_model)

    def forward(self, x):
        x = self.stem(x)                               # (B, 48, 8, 8)
        B, C, H, W = x.shape
        tokens = x.reshape(B, C, -1).permute(0, 2, 1) # (B, 64, 48)

        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)           # (B, 64, 48)

        tokens = self.proj(tokens)                      # (B, 64, d_model)
        return tokens


# ==============================================================================
# 注意力块: Self-Attention 和 Cross-Attention
# ==============================================================================

class SelfAttentionBlock(nn.Module):
    """自注意力块: MultiheadAttention + residual + LayerNorm + FFN(GELU).

    用于阶段1/2: 在每个模态内部建模 token 之间的关系.
    - 3D self-attn: 点与点之间的空间关系 (前轮 ↔ 后轮 ↔ 车顶)
    - 2D self-attn: 像素块之间的外观关系 (车窗 ↔ 车灯 ↔ 保险杠)
    """

    def __init__(self, d_model=256, n_heads=8, ffn_expand=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, d_model) — 同模态内的 token 序列
        attn_out, _ = self.attn(x, x, x)    # Q=K=V=x (self-attention)
        x = self.norm1(x + attn_out)         # residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)          # residual + norm
        return x


class CrossAttentionBlock(nn.Module):
    """交叉注意力块: Q 查询 K,V, 加 residual + FFN.

    用于阶段3/4: 跨模态信息交换.
    - 阶段3: Q=2D, K=V=3D → 图像特征获得几何感知
    - 阶段4: Q=3D, K=V=2D → 点云特征获得外观感知 (这是最终预测的依据)
    """

    def __init__(self, d_model=256, n_heads=8, ffn_expand=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        """q 查询 kv: Q=q, K=kv, V=kv.

        Args:
            q:  (B, N_q,  d_model) — 查询方 token
            kv: (B, N_kv, d_model) — 被查询方 token (提供 K 和 V)
        Returns:
            (B, N_q, d_model) — 查询方被 kv 信息增强后的 token
        """
        attn_out, _ = self.attn(q, kv, kv)   # Q=q, K=kv, V=kv
        q = self.norm1(q + attn_out)          # residual + norm
        ffn_out = self.ffn(q)
        q = self.norm2(q + ffn_out)           # residual + norm
        return q


# ==============================================================================
# 3D 特征提取: PointNet++ 4 层 Set Abstraction
# ==============================================================================

class PointNet2Encoder(nn.Module):
    """PointNet++ 编码器: 3 层 Set Abstraction, 逐层降采样+扩大感受野.

    输入:  (B, N, 5)  [x, y, z, intensity, log_scale]
    输出:  (B, 8, 3) centroids, (B, 256, 8) features

    SA 半径工作在"相对单位" (数据集已按物体物理 extent 归一化):
      SA1: npoint=32, radius=0.3  → 局部几何 (30% extent)
      SA2: npoint=16, radius=0.7  → 部件结构 (70% extent)
      SA3: npoint=8,  radius=1.5  → 全局上下文 (150% extent, 覆盖全物体)
    """

    def __init__(self, sa_configs):
        super().__init__()
        self.sa_layers = nn.ModuleList()
        in_dim = 5   # xyz + intensity + log_scale
        for cfg in sa_configs:
            self.sa_layers.append(SetAbstraction(
                npoint=cfg["npoint"], radius=cfg["radius"],
                nsample=cfg["nsample"], in_dim=in_dim + 3,
                mlp_dims=cfg["mlp"],
            ))
            in_dim = cfg["mlp"][-1]

    def forward(self, xyz):
        coords = xyz[..., :3]                  # (B, N, 3)  — FPS + Ball Query
        feats = xyz.permute(0, 2, 1)          # (B, 5, N)  — 初始特征
        c, f = coords, feats
        for sa in self.sa_layers:
            c, f = sa(c, f)                    # c: (B, npoint, 3), f: (B, D_out, npoint)
        return c, f


# ==============================================================================
# CrossModalFusion (C3): 跨模态注意力融合
# ==============================================================================

class CrossModalFusion(nn.Module):
    """C3: 跨模态融合 — 2D 外观 + 3D 几何 → 3D bbox 残差预测.
    三阶段注意力: 3D self-attn → 2D←3D cross → 3D←2D cross.

    输入:
        rgb_crop:  (B, 3, 128, 128)  YOLO 检测框内的 RGB 区域 (与 LiDAR 内外参耦合)
        lidar_pts: (B, N, 5)         bbox 内的归一化 LiDAR 点云 [xyz+intensity+log_scale]

    输出:
        (B, 8) [dx, dy, dz, dw, dh, dl, sin(dθ), cos(dθ)]

    架构: Lightweight2D (~41K) + PointNet++ (~149K) + 双向 cross-attn + MLP head.
    2D 分支自带 self-attn, 无需重复的阶段 2.
    """

    def __init__(self, sa_configs=None, d_model=256, n_heads=8,
                 num_layers=1, dropout=0.1):
        super().__init__()

        # ---- 2D 分支: 轻量 Conv + Self-Attn ----
        self.extractor_2d = Lightweight2DExtractor(
            d_model=d_model, n_heads=4, dropout=dropout)

        # ---- 3D 分支: 3 层 SA, 输出 256 通道 ----
        if sa_configs is None:
            sa_configs = [
                dict(npoint=32, radius=0.3, nsample=32, mlp=[32, 32, 64]),
                dict(npoint=16, radius=0.7, nsample=16, mlp=[64, 128, 128]),
                dict(npoint=8,  radius=1.5, nsample=8,  mlp=[128, 256, 256]),
            ]
        self.encoder_3d = PointNet2Encoder(sa_configs)
        self.proj_3d = nn.Linear(256, d_model)  # 256 → d_model

        # ---- 阶段 1: 3D 模态内自注意力 ----
        self.self_attn_3d = SelfAttentionBlock(d_model, n_heads, dropout=dropout)

        # ---- 阶段 2: 2D ← 3D Cross (图像学习几何) ----
        self.cross_2d_from_3d = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ---- 阶段 3: 3D ← 2D Cross (点云吸收外观感知) ----
        self.cross_3d_from_2d = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ---- 位置编码 ----
        self.pos_embed_3d = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # 2D 位置编码 (learnable, 64 个 token 各一个)
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)

        # ---- 回归头: d_model → 128 → 64 → 8 ----
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 8),
        )

    def forward(self, rgb_crop, lidar_pts):
        B = rgb_crop.shape[0]

        # ---- 2D 特征 (Conv Stem + Self-Attn, 已输出 tokens) ----
        tokens_2d = self.extractor_2d(rgb_crop)                 # (B, 64, d_model)
        tokens_2d = tokens_2d + self.pos_embed_2d

        # ---- 3D 特征 ----
        centroids, feat_3d = self.encoder_3d(lidar_pts)        # (B, 8, 3), (B, 256, 8)
        tokens_3d = feat_3d.permute(0, 2, 1)                    # (B, 8, 256)
        tokens_3d = self.proj_3d(tokens_3d)                     # (B, 8, d_model)
        tokens_3d = tokens_3d + self.pos_embed_3d(centroids)

        # ---- 阶段 1: 3D 自注意力 ----
        tokens_3d = self.self_attn_3d(tokens_3d)

        # ---- 阶段 2: 2D ← 3D (图像学习几何) ----
        for block in self.cross_2d_from_3d:
            tokens_2d = block(tokens_2d, tokens_3d)

        # ---- 阶段 3: 3D ← 2D (点云吸收外观) ----
        for block in self.cross_3d_from_2d:
            tokens_3d = block(tokens_3d, tokens_2d)

        # ---- 全局池化 → 回归 ----
        obj_token = tokens_3d.mean(dim=1)
        return self.head(obj_token)


# ==============================================================================
# 消融对比模型: ImageOnlyRefiner (C1), LidarOnlyRefiner (C2)
# ==============================================================================

class ImageOnlyRefiner(nn.Module):
    """C1: 纯 2D 基线 — 只用 Lightweight2DExtractor, 不看 LiDAR.
    用途: 验证 3D 几何信息是否必要.
    """

    def __init__(self, d_model=256, dropout=0.3):
        super().__init__()
        self.extractor = Lightweight2DExtractor(d_model=d_model, n_heads=4)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 8),
        )

    def forward(self, rgb_crop, lidar_pts=None):
        tokens = self.extractor(rgb_crop)         # (B, 64, d_model)
        pooled = tokens.mean(dim=1)               # (B, d_model)
        return self.head(pooled)


class LidarOnlyRefiner(nn.Module):
    """C2: 纯 3D 基线 — 只用 PointNet++, 不看 RGB.
    用途: 验证 2D 外观是否必要. 对所有 token 做 mean pooling.
    """

    def __init__(self, sa_configs=None, dropout=0.3):
        super().__init__()
        if sa_configs is None:
            sa_configs = [
                dict(npoint=32, radius=0.3, nsample=32, mlp=[32, 32, 64]),
                dict(npoint=16, radius=0.7, nsample=16, mlp=[64, 128, 128]),
                dict(npoint=8,  radius=1.5, nsample=8,  mlp=[128, 256, 256]),
            ]
        self.encoder = PointNet2Encoder(sa_configs)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 8),
        )

    def forward(self, rgb_crop, lidar_pts):
        _, feat = self.encoder(lidar_pts)        # (B, 256, npoint)
        pooled = feat.mean(dim=-1)                # (B, 256)  — 全局平均池化
        return self.head(pooled)
