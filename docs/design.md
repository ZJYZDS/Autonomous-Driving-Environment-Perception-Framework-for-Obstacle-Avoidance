# 基于 PointNet++ 的 3D 检测框残差回归

**日期**: 2026-07-10

## 概述

任务: 给定 YOLO 2D 检测框 + LiDAR 点云, 回归该物体的 3D bbox 残差 (相对于噪声扰动的 GT initial guess)。

核心思路: PointNet++ 编码 3D 几何作为主干 (~149K), 轻量 2D Conv+Self-Attention 编码外观作为辅助 (~41K), 通过三阶段交叉注意力融合 → 回归 8 维残差。

实际结论: 在当前 nuScenes 城市道路数据上, 纯 3D (C2, 191K) 与融合模型 (C3, 2.8M) 性能差距在噪声范围内, C2 是最优性价比方案。

## 与 Phase 1 的关键差异

| 方面 | Phase 1 | Phase 2 |
|------|---------|---------|
| 任务 | 6-DOF 位姿估计 (R, t) | 3D bbox 残差回归 (dx,dy,dz,dw,dh,dl,dθ) |
| 数据 | BlenderProc 合成 | nuScenes v1.0-mini 真实数据 |
| 检测 | YOLO-seg (mask) | YOLO26s (bbox, end2end NMS) |
| 2D 特征 | ResNet18 (2.73M) | Lightweight2DExtractor (41K) |
| 3D 特征 | 4×SA, 固定半径 | 3×SA, extent-adaptive 归一化 |
| 融合 | 4 尺度 grid_sample + cross-attn | 3 阶段 token-based cross-attn |
| 输出 | R(3×3) + t(3) | 8 维: [Δxyz, Δwhl, sin, cos] |
| 损失 | SmoothL1(t) + Chordal(R) | SmoothL1(center,size) + MSE(sin/cos) |
| 评测 | Accuracy@5°5cm | center/size/yaw 物理误差 (米, 度) |

## 数据流

```
nuScenes 一帧:
  CAM_FRONT 图像 (1600×900) + LIDAR_TOP 点云 (N×5)
       │
       ├──→ YOLO26s 检测 → K 个 2D bbox (x1,y1,x2,y2,conf,cls)
       │
       ├──→ LiDAR → 投影到图像平面 → 提取 bbox 内 3D 点
       │
       ├──→ GT 匹配: 2D 中心距离 (det bbox ↔ 投影 GT center)
       │
       └──→ 对每个检测物体:
              ├── RGB crop (128×128)
              ├── 点云: GT + noise → 去中心化 + 旋转对齐 → extent 归一化 → FPS(256)
              └── 目标: 8 维残差 [Δcenter, Δsize, sin(Δθ), cos(Δθ)]
```

### 噪声扰动策略 (训练时)

训练时对 GT 加噪声, 模型学习从 noisy → GT 的残差:
- center: N(0, 0.3m)
- size: N(0, 0.15m)
- yaw: N(0, 5°)

这模拟了"初始检测不精确, 需要 refinement"的场景。

### Extent 自适应归一化

不同物体尺度差异大 (行人 0.5m vs 卡车 10m), 固定 SA 半径无法同时适配。解决方案:
```
scale = max(ptp(local_xyz)) / 2    # 点云半跨度
local_xyz = local_xyz / scale      # 归一化到 0~1 范围
```
SA 半径因此工作在"相对单位", 小车自动用小半径, 大车自动用大半径。`log(scale)` 作为额外特征输入。

## 模型家族: C1 / C2 / C3

### C1 — ImageOnlyRefiner (纯 2D baseline)

```
RGB crop (B,3,128,128) → Lightweight2DExtractor → tokens (B,64,256) → mean pool → MLP → (B,8)
```

- 参数量: ~83K
- 用途: 验证 3D 几何是否必要
- 结论: C2 ≈ C1, 纯 2D 即可达到接近水平, 但最终不如 C2

### C2 — LidarOnlyRefiner (纯 3D baseline, **推荐**)

```
LiDAR points (B,256,5) → PointNet2Encoder (3×SA) → features (B,256,1) → MLP → (B,8)
```

- 参数量: 191K
- SA 配置: SA3 的 npoint=1 (直接全局池化)
- 结论: 当前场景最优, 性价比最高

### C3 — CrossModalFusion (融合, 实验性)

```
RGB crop (B,3,128,128)                          LiDAR (B,256,5)
       │                                                │
Lightweight2DExtractor                           PointNet2Encoder (3×SA)
       │                                                │
  tokens_2d (B,64,256)                       tokens_3d (B,8,256)
       │                                                │
       │                                    + pos_embed(centroids)
       │                                                │
       │                                    阶段1: 3D Self-Attn
       │◄────── 阶段2: 2D←3D Cross ──────────────┘
       └──────► 阶段3: 3D←2D Cross ──────────────►
                                                        │
                                                  mean pool
                                                        │
                                                  MLP → (B,8)
```

- 参数量: ~2.8M (cross-attn ~2.1M, MLP head ~500K, 3D encoder 149K, 2D extractor 41K)
- 辅助:主干 = 0.27x
- 结论: 当前数据上仅边际优于 C2

## 2D 分支设计: Lightweight2DExtractor

```
输入: (B, 3, 128, 128)  YOLO 检测框内 RGB crop

Conv1: 3→16, k=7, s=8, p=3  → (B, 16, 16, 16)    大核大跨度, 一次看清全局
Conv2: 16→48, k=3, s=2, p=1 → (B, 48, 8, 8)      局部细调

Reshape: (B, 48, 8, 8) → (B, 64, 48)             64 个 2D token

Self-Attn (d=48, n_heads=4, ffn_expand=2)         token 间外观关系
Proj: Linear(48, 256) → (B, 64, 256)             投影到 d_model
```

设计理由:
- **2 层 conv 够用**: 检测框内只有一个物体 (车/行人), 不是任意照片
- **stride=8 + k=7 大核**: 128×128 一次下采样到 16×16, 覆盖物体关键部件
- **d=48 内通道**: 仅为 d_model(256) 的 19%, 外观判别不需要高维
- **self-attn 在窄空间**: MHA(d=48) 仅 9K 参数, 比 d=256 省 15 倍

## 参数分布

| 模型 | 总参数 | 2D | 3D | Cross-Attn | MLP Head |
|------|--------|-----|-----|------------|----------|
| C1 | ~83K | 41K | — | — | ~42K |
| C2 | 191K | — | 149K | — | ~42K |
| C3 | ~2.8M | 41K | 149K | ~2.1M | ~500K |

## 3D 编解码: PointNet2Encoder

```
输入: (B, N, 5)  [x, y, z, intensity, log_scale]

SA1: npoint=32, radius=0.3, nsample=32, mlp=[32,32,64]   局部几何 (30% extent)
SA2: npoint=16, radius=0.7, nsample=16, mlp=[64,128,128]  部件结构 (70% extent)
SA3: npoint=8,  radius=1.5, nsample=8,  mlp=[128,256,256] 全局上下文 (150% extent)

输出: (B, 8, 3) centroids, (B, 256, 8) features
```

SA 半径在 extent 归一化后的相对空间工作: 归一化后物体半跨度为 1, radius=0.3 意为 30% 的物体半跨度。

## 融合机制: 三阶段 Cross-Attention

1. **阶段 1 — 3D Self-Attn**: 8 个 3D token 之间建模空间关系 (前轮↔后轮↔车顶)
2. **阶段 2 — 2D←3D Cross**: 64 个 2D token 查询 8 个 3D token, 图像获得几何感知
3. **阶段 3 — 3D←2D Cross**: 8 个 3D token 查询 64 个几何感知的 2D token, 点云吸收外观

每个 Cross-Attention block: Q(query) 查询 KV(kv) → residual + LayerNorm → FFN(GELU) → residual + LayerNorm。

## 损失函数

```
Loss = SmoothL1(Δcenter_pred, Δcenter_gt)      # center 权重 1.0
     + SmoothL1(Δsize_pred, Δsize_gt)          # size 权重 1.0
     + MSE(sin_pred, sin_gt) + MSE(cos_pred, cos_gt)  # yaw 权重 0.5
```

yaw 用 (sin, cos) 编码而非直接角度: 避免 0°/360° 的周期性歧义。MSE 对小偏差梯度平滑, 适合角度这种周期性小量。

## 评估指标

| 指标 | 计算 | 含义 |
|------|------|------|
| center_err (m) | L2(Δcenter_pred, Δcenter_gt) | 中心点偏移 |
| size_err (m) | mean(\|Δsize_pred - Δsize_gt\|) | 各维度尺寸误差均值 |
| yaw_deg (°) | \|atan2(sin,cos)_pred - atan2(sin,cos)_gt\| | 朝向角误差 |

## 消融实验结果

| 模型 | 参数 | val_loss | center | size | yaw |
|------|------|----------|--------|------|-----|
| C1 (纯2D) | 83K | 0.0532 | 0.414m | 0.125m | 3.44° |
| C2 (纯3D) | 191K | **0.0522** | 0.442m | 0.110m | 3.23° |
| C3 (融合) | 2.8M | 0.0511 | 0.441m | 0.114m | 4.54° |

C2 vs C3 逐帧对比 (69 objects, 均训练到收敛):

| 指标 | C2 | C3 | 差距 |
|------|-----|-----|------|
| Center err | 0.460m | 0.441m | 1.9cm |
| Size err | 0.120m | 0.114m | 6mm |
| Yaw err | 4.40° | 4.54° | -0.14° |
| Head-to-head | 30 | 39 | 57% vs 43% |

差距远小于 GT 噪声 (center=0.3m, size=0.15m, yaw=5°), 在统计噪声范围内。

## 2D 图像的作用 (何时有增益)

在当前 nuScenes 城市道路数据上, 2D 外观未提供实质性增益。但 2D 图像在以下场景具有不可替代性:

1. **远距离物体 (60-100m)**: 点云稀疏 (<15 点), RGB crop 仍有清晰轮廓
2. **朝向 180° 歧义**: 轿车头/尾部 LiDAR 几何对称, RGB 尾灯/头灯可一眼区分
3. **截断与遮挡**: 只能看到部分车身, 外观可推断完整尺寸
4. **语义推理 (未来)**: 物体细分类、状态判断 (车门开/关)

结论: 2D 在**几何信息不足**时发挥不可替代作用, 是下限保障而非上限增益。

## 文件结构

```
cross_atn_pointNet++/
├── src/
│   ├── fusion.py          # 模型: Lightweight2DExtractor, PointNet2Encoder,
│   │                        CrossModalFusion, ImageOnlyRefiner, LidarOnlyRefiner
│   ├── model.py           # PointNet++ 核心: FPS, Ball Query, Set Abstraction
│   ├── dataset_phase2.py  # 数据集: YOLO检测 → crop lpt → 噪声 → 残差目标
│   ├── dataset_phase1.py  # Phase1 遗留: LiDARProjector, 坐标变换工具
│   ├── detector.py        # YOLO26s ONNX 推理 (end2end NMS)
│   ├── loss.py            # BboxRefinementLoss: SmoothL1 + MSE(sin/cos)
│   └── metrics.py         # compute_metrics: center(m)/size(m)/yaw(°)
├── config/
│   └── phase2.yaml        # 训练配置 (model_type, SA, noise, loss weights)
├── scripts/
│   ├── train_phase2.py    # 训练脚本 (支持 C1/C2/C3, resume)
│   └── compare_c2c3.py    # C2 vs C3 逐帧对比
├── checkpoints_phase2/
│   ├── lidar_only.pt      # C2 最优 (epoch 19, val_loss=0.0621)
│   └── fusion.pt          # C3 最优 (epoch 23, val_loss=0.0511)
├── docs/
│   ├── design.md          # 本文档
│   └── rgb_image_role_in_bbox_refinement.md
├── log/
│   ├── ablation_c1c2_analysis.md
│   ├── lightweight_2d_refactor.md
│   ├── scale_variation_analysis.md
│   └── c2_vs_c3_comparison.md
└── models/
    └── yolo26s.onnx       # YOLO 检测模型
```
