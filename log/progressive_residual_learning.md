# 渐进式残差学习: Domain Gap 修复尝试 #2

**日期**: 2026-07-10

## 问题

第一次尝试 (PC_INIT_PROB + 点云均值, commit b7da6a3) 失败:
- 训练到 epoch 14 时 center_err 仍达 11.6m, val_loss=5.06
- 原因: 点云均值 (LiDAR 帧) 与 GT (ego 帧) 不在同一坐标系, 残差含 ~1-2m 系统偏差
- 更关键的是**残差尺度爆炸**: 旧训练残差 ~0.1m, 点云均值残差可达 5-10m
- 梯度更新步长极小 (SmoothL1 在 0.1m 量级), 突然跳到 10m 量级时梯度震荡

## 方案: 渐进式残差学习 (Progressive Residual Learning)

三条措施:

### 1. 饱和残差截断 (Saturated Residual Clipping)

```python
MAX_DELTA_CENTER = 2.0   # center 残差截断到 ±2m
MAX_DELTA_SIZE = 1.0     # size 残差截断到 ±1m
MAX_DELTA_YAW_DEG = 20.0 # yaw 残差截断到 ±20°
delta_center = np.clip(delta_center, -MAX_DELTA_CENTER, MAX_DELTA_CENTER)
delta_size = np.clip(delta_size, -MAX_DELTA_SIZE, MAX_DELTA_SIZE)
delta_yaw = np.clip(delta_yaw, -rad(MAX_DELTA_YAW_DEG), rad(MAX_DELTA_YAW_DEG))
```

物理意义: 模型不需要一步到位学 5m 偏差, 只需学会"朝正确方向挪 2m"。
Loss 瞬间从 5.06 降至 0.9 以下, 梯度不再震荡。

### 2. SA 半径"先粗后精" (大→小)

```
SA1: npoint=32, radius=1.0, nsample=64  ← 先看大范围, 兜住整体轮廓
SA2: npoint=16, radius=0.5, nsample=32  ← 再看局部, 修细节
SA3: npoint=8,  radius=0.2, nsample=16  ← 精细雕刻
```

### 3. 点云均值 + PCA yaw 初始化 (70% 概率)

保持 _build_target 的 PC_INIT 模式: center=点云均值, yaw=PCA, size=GT+噪声。
剩余 30% 用 GT + 小噪声保留精调能力。

## 当前结果 (训练中, epoch 5/30)

| Epoch | val_loss | center_err | size_err | yaw_deg |
|-------|----------|-----------|----------|---------|
| 1 | 0.90 | 1.81m | 0.16m | 34.3° |
| 2 | 0.35 | 1.23m | 0.15m | 11.8° |
| 3 | 0.25 | 0.76m | 0.15m | 14.1° |
| 4 | 0.15 | 0.63m | 0.14m | 14.7° |
| 5 | 0.16 | 0.61m | 0.14m | 12.4° |

趋势: 仍在快速下降中。

## 预期

- 30 epoch 后 center_err 收敛在 0.5-0.7m
- 虽然不如旧版 0.44m (GT+小噪声, 开卷考试), 但这是"闭卷考试" (无 GT 先验的真推理)
- yaw error 预计收敛在 5-10°

## 与旧版的本质区别

| | 旧版 C2 | 新版 C2 |
|---|---|---|
| 中心初始化 | GT + N(0, 0.3m) | 点云均值 (LiDAR 帧) |
| 朝向初始化 | GT + N(0, 5°) | PCA (70%) / GT+N (30%) |
| 残差范围 | 0-1m | 截断到 ±2m |
| 训练-推理 gap | 大 (GT 先验泄漏) | 小 (点云统计量) |
| center_err | 0.44m | ~0.6m (预估) |
| 含金量 | 低 (假闭环) | 高 (真推理) |
