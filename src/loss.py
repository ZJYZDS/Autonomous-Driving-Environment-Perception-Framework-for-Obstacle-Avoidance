"""
损失函数: SmoothL1 (center, size) + MSE (yaw sin/cos).

为什么 yaw 用 MSE 而不是 SmoothL1?
  sin/cos 是归一化到 [-1, 1] 的有界值, 不需要 SmoothL1 的大误差截断.
  MSE 对小偏差的梯度更平滑, 适合角度这种周期性小量.

为什么 yaw 预测 (sin, cos) 而不是直接预测角度?
  角度 0° 和 360° 是同一个方向, 但数值上差距很大.
  (sin, cos) 编码天然处理了这个周期性, 且 atan2 恢复角度无歧义.
"""

import torch.nn as nn


class BboxRefinementLoss(nn.Module):
    """3D bbox 残差回归的组合损失.

    损失 = center_weight * SmoothL1(dx,dy,dz)
         + size_weight  * SmoothL1(dw,dh,dl)
         + yaw_weight   * MSE(sin(dθ), cos(dθ))

    权重默认 center=1.0, size=2.0, yaw=1.5.
    size/yaw 高权重补偿其较小的数值量级, 防止 center 主导梯度.
    """

    def __init__(self, center_weight=1.0, size_weight=2.0, yaw_weight=1.5):
        super().__init__()
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.yaw_weight = yaw_weight
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """Args:
            pred:   (B, 8)  [dx,dy,dz, dw,dh,dl, sin,cos]
            target: (B, 8)  同上
        Returns:
            total_loss, {"loss": float, "center": float, "size": float, "yaw": float}
        """
        loss_center = self.smooth_l1(pred[:, :3], target[:, :3])
        loss_size = self.smooth_l1(pred[:, 3:6], target[:, 3:6])

        # sin 和 cos 分别算 MSE, 让模型同时学角度值和象限
        loss_yaw = (self.mse(pred[:, 6], target[:, 6])
                    + self.mse(pred[:, 7], target[:, 7]))

        total = (
            self.center_weight * loss_center
            + self.size_weight * loss_size
            + self.yaw_weight * loss_yaw
        )
        return total, {
            "loss": total.item(),
            "center": loss_center.item(),
            "size": loss_size.item(),
            "yaw": loss_yaw.item(),
        }
