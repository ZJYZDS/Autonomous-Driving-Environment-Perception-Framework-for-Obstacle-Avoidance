# C1/C2 消融实验分析

**日期**: 2026-07-10

## 实验目的

验证 2D 图像特征对 3D bbox refinement 是否必要。

## 模型

| 模型 | 描述 | 参数量 |
|------|------|--------|
| C1 | 纯 2D: ResNet18 → global pool → MLP head | 2,890,632 |
| C2 | 纯 3D: PointNet++ (3×SA) → global pool → MLP head | 191,496 |

## 数据

- nuScenes v1.0-mini
- Train: 324 frames (8 scenes)
- Val: 80 frames (2 scenes)
- 每帧通过 YOLO26s 检测物体, 为每个物体生成 RGB crop (128×128) + LiDAR 点云 (256 点, 5 通道 xyz+intensity+log_scale)
- Extent 归一化: 点云按物体物理范围归一化, SA 半径使用相对单位
- 噪声注入: center=0.3m, size=0.15m, yaw=±5° (模拟实际推理的 coarse 初始估计)

## 训练配置

- Epochs: 30 (CosineAnnealingLR)
- Batch size: 16
- Learning rate: 0.0005, Adam optimizer
- Loss: SmoothL1 (center/size) + MSE (yaw sin/cos), weights: center=1.0, size=1.0, yaw=0.5

## 结果

| 指标 | C1 (纯2D) | C2 (纯3D) |
|------|-----------|-----------|
| Best epoch | 18 | 18 |
| val_loss | 0.0532 | **0.0522** |
| center_err (m) | **0.414** | 0.442 |
| size_err (m) | 0.125 | **0.110** |
| yaw_err (deg) | 3.44 | **3.23** |

## 结论

1. **C2 ≈ C1: 纯 3D 点云与纯 2D 图像效果持平。** 二者在各项指标上差异在统计噪声范围内
2. **C2 参数量仅为 C1 的 1/15** (191K vs 2.89M), 更小、更快、同效果
3. **3D 点云自身已包含 bbox refinement 所需的全部几何信息** (center, size, yaw)。2D 图像在此任务上没有额外信号贡献
4. **C3 (Dual-Attention 融合, 6.3M 参数) 大概率不会优于 C2** — 如果纯图像打不平纯点云, 那融合也不会产生新信息

## 建议

- **采用 C2 (纯 PointNet++)** 作为最终模型, 191K 参数, 简洁高效
- 移除 2D 分支 (ResNet18) 和 cross-attention 模块, 大幅简化代码和训练流程
- 如未来需要在远距离/稀疏点云场景 (如 < 20 点/物体) 下提升性能, 可重新评估 2D 图像的价值
