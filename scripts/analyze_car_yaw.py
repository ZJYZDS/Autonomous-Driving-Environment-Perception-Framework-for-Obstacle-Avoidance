"""
Car 朝向判别专项分析: C2 (纯3D) vs C3 (3D+2D).

用车载 LiDAR 看三厢轿车 → 车头/车尾几何高度对称 → 180° 歧义.
2D 图像有尾灯(红)/头灯(白)/进气格栅 → 一眼区分方向.

用法: python scripts/analyze_car_yaw.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset_phase2 import Phase2Dataset, phase2_collate
from src.fusion import LidarOnlyRefiner, CrossModalFusion
from src.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cls, ckpt_path, **kwargs):
    model = cls(**kwargs).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    # 加载数据 (return_category=True 获取 GT 类别)
    ds = Phase2Dataset("data/nuscenes", split="val", cfg={
        "min_lidar_pts": 10, "num_points": 256, "crop_size": 128,
        "bbox_margin": 0.3, "noise_center": 0.3, "noise_size": 0.15,
        "noise_yaw_deg": 5.0, "match_max_dist_px": 80, "val_scene_ids": 2,
        "return_category": True,
    }, detector_path="models/yolo26s.onnx")
    loader = DataLoader(ds, batch_size=16, collate_fn=phase2_collate, shuffle=False)

    c2 = load_model(LidarOnlyRefiner, "checkpoints_phase2/lidar_only.pt")
    c3 = load_model(CrossModalFusion, "checkpoints_phase2/fusion.pt")

    # 分类统计
    car_c2, car_c3 = [], []
    other_c2, other_c3 = [], []
    categories_seen = {}

    print(f"{'#':>4s} | {'cat':>18s} | {'C2 yaw':>7s} {'C2 ctr':>7s} | {'C3 yaw':>7s} {'C3 ctr':>7s} | note")
    print("-" * 78)

    obj_idx = 0
    with torch.no_grad():
        for batch in loader:
            # 解包 batch (支持 4-tuple with categories)
            if len(batch) == 4:
                rgb, lidar, target, categories = batch
            else:
                rgb, lidar, target = batch
                categories = ["unknown"] * len(target)

            rgb, lidar, target = rgb.to(DEVICE), lidar.to(DEVICE), target.to(DEVICE)

            pred_c2 = c2(None, lidar)
            pred_c3 = c3(rgb, lidar)

            for i in range(len(target)):
                m2 = compute_metrics(pred_c2[i:i+1], target[i:i+1])
                m3 = compute_metrics(pred_c3[i:i+1], target[i:i+1])

                cat = categories[i] if i < len(categories) else "unknown"
                categories_seen[cat] = categories_seen.get(cat, 0) + 1
                is_car = cat == "vehicle.car"

                if is_car:
                    car_c2.append(m2)
                    car_c3.append(m3)
                else:
                    other_c2.append(m2)
                    other_c3.append(m3)

                # 打印详情
                note = ""
                if is_car and m2["yaw_deg"] > 30:
                    note = "YAW_BAD"
                elif is_car:
                    note = ""

                if obj_idx < 50 or (is_car and note):
                    print(f"{obj_idx:4d} | {cat:>18s} | {m2['yaw_deg']:6.2f}d {m2['center_err']:6.3f}m | "
                          f"{m3['yaw_deg']:6.2f}d {m3['center_err']:6.3f}m | {note}")

                obj_idx += 1

    # =========================================================================
    # 汇总
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Categories: {categories_seen}")
    print(f"Car: {len(car_c2)} samples, Other: {len(other_c2)} samples")

    for label, c2_list, c3_list in [
        ("vehicle.car", car_c2, car_c3),
        ("other", other_c2, other_c3),
    ]:
        if not c2_list:
            continue
        print(f"\n--- {label} ({len(c2_list)} samples) ---")
        for name, results in [("C2", c2_list), ("C3", c3_list)]:
            avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
            print(f"  {name}: center={avg['center_err']:.3f}m  size={avg['size_err']:.3f}m  yaw={avg['yaw_deg']:.2f}d")

        # Yaw 分层分析 (关键: 看 C2 犯错时 C3 的挽救)
        for lo, hi, tag in [(0, 3, "easy   <3d"), (3, 8, "medium 3-8d"), (8, 30, "hard  8-30d"), (30, 999, "severe >30d")]:
            pairs = [(a, b) for a, b in zip(c2_list, c3_list) if lo <= a["yaw_deg"] < hi]
            if not pairs:
                continue
            c2_yaw = np.mean([a["yaw_deg"] for a, _ in pairs])
            c3_yaw = np.mean([b["yaw_deg"] for _, b in pairs])
            c2_ctr = np.mean([a["center_err"] for a, _ in pairs])
            c3_ctr = np.mean([b["center_err"] for _, b in pairs])
            print(f"  {tag:>14s}: {len(pairs):3d}s | C2→C3 yaw: {c2_yaw:.2f}→{c3_yaw:.2f}d ({c2_yaw-c3_yaw:+.1f}d)  "
                  f"ctr: {c2_ctr:.3f}→{c3_ctr:.3f}m")

    # =========================================================================
    # 每个 car 样本的详细信息
    # =========================================================================
    if car_c2:
        print(f"\n{'='*60}")
        print("Per-car detail (sorted by C2 yaw error):")
        car_pairs = sorted(zip(car_c2, car_c3), key=lambda x: x[0]["yaw_deg"], reverse=True)
        for i, (m2, m3) in enumerate(car_pairs):
            delta = m2["yaw_deg"] - m3["yaw_deg"]
            tag = "<<< FLIP RESCUE" if m2["yaw_deg"] > 30 and delta > 10 else ""
            print(f"  car#{i:2d}: C2_yaw={m2['yaw_deg']:5.1f}d → C3_yaw={m3['yaw_deg']:5.1f}d | "
                  f"Δ={delta:+5.1f}d  {tag}")


if __name__ == "__main__":
    main()
