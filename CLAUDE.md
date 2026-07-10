# Cross-Modal 3D BBox Refinement (PointNet++)

## 项目概述

给定 CAM_FRONT + LIDAR_TOP，YOLO 2D检测 → PointNet++ 回归 3D bbox 残差 (center/size/yaw)。
主线模型 C2 (LidarOnlyRefiner, 191K)，纯3D。C3 (CrossModalFusion, 2.8M) 实验性保留。

## 关键文件

- `src/fusion.py` — C1/C2/C3 模型定义
- `src/model.py` — PointNet++ FPS/Ball Query/Set Abstraction
- `src/dataset_phase2.py` — 数据集: YOLO检测 → crop → 残差目标
- `src/detector.py` — YOLO26s ONNX推理
- `src/loss.py` — SmoothL1(center,size) + MSE(sin/cos), 权重 1.0/2.0/1.5
- `scripts/train_phase2.py` — 训练 (--model_type c2_3d)
- `scripts/visualize_c2.py` — 推理可视化 (PLY + 渲染PNG)

## 坐标帧 (重要!)

- 训练: GT从global转LiDAR帧 (via `_global_to_lidar`), 与obj_points统一坐标系
- obj_points: LiDAR sensor frame (直接从LIDAR_TOP .bin文件加载)
- 推理: noisy_center由点云均值算得, 自然在LiDAR帧, 与训练一致
- nuScenes: size=(宽,长,高), yaw=0°时长度沿x轴(前)

## 训练配置

- C2: `python scripts/train_phase2.py --model_type c2_3d`
- 数据集: nuScenes v1.0-mini, 8 train + 2 val scenes
- 噪声: center±0.3m, size±0.15m, yaw±5°
- pc_init已禁用 (DBSCAN不稳定且收益不明显), 100% GT+noise
- 残差截断: MAX_DELTA_CENTER=2m, MAX_DELTA_SIZE=1m, MAX_DELTA_YAW=20°

## 已知问题与修复

1. **坐标帧不匹配 (已修复)**: _build_target 中 obj_points (LiDAR帧) 减去 noisy_center (Ego帧), 导致1-2m系统性偏置. 修复: 统一用LiDAR帧, GT通过 _global_to_lidar 转换, 移除pc_init分支的LiDAR→Ego转换.
2. **Loss权重失衡 (已修复)**: center/size/yaw 默认 1.0/1.0/0.5 导致center主导梯度. 修复: 调整为 1.0/2.0/1.5.
3. **pc_init/DBSCAN冗余 (已修复)**: PC_INIT_PROB=0.7 引入DBSCAN不稳定性和额外开销. 修复: 禁用pc_init, 移除 _extract_core_points 和DBSCAN依赖.
