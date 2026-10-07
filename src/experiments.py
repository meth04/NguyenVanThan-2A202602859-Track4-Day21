"""Thiết kế và chạy các thí nghiệm có kiểm soát cho topic A.

Nguyên tắc: mỗi lần chạy chỉ thay đổi **một** yếu tố (mức perturb extrinsic), mọi thứ
khác giữ nguyên (frame, vùng range, cách đo). Kết quả trả về dạng `pandas.DataFrame`
để ghi CSV và vẽ biểu đồ.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from starter.datasets import load_frame
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

from src.geometry import object_points_projected
from src.metrics import bucket_of, fov_ratio, in_box_ratio, nmi_at_pixels

# Mức perturb mặc định: 0 (chuẩn) rồi tăng dần. Đơn vị độ (yaw/pitch/roll) và cm (dịch).
YAW_LEVELS_DEG = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
TRANSLATION_LEVELS_CM = (0.0, 2.0, 5.0, 10.0)
# Các mức yaw dùng riêng cho việc đo ngưỡng phát hiện drift (bonus B3).
DETECTION_LEVELS_DEG = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class FrameConfig:
    """Một frame được đưa vào thí nghiệm, kèm tên dataset để tách nhóm khi phân tích."""
    dataset: str
    data_root: str
    frame_id: str


def _project_frame(frame: dict, perturb: dict) -> dict:
    """Chiếu một frame với calibration đã perturb và trả về mọi chỉ số của một lần chạy."""
    calib = perturb_extrinsic(frame["calib"], **perturb)
    cam = velo_to_cam(frame["points"][:, :3], calib)
    uv, depth, mask = cam_to_image(cam, calib.P2, frame["image"].shape)

    out: dict[str, object] = {
        "fov_ratio": fov_ratio(mask),
        "n_points_in_fov": int(mask.sum()),
        "n_points": int(len(mask)),
        # Chỉ số label-free: NMI giữa cường độ LiDAR và độ sáng ảnh tại điểm chiếu.
        "nmi_intensity": nmi_at_pixels(frame["points"][mask, 3], uv, frame["image"]),
    }

    # Chỉ số theo vật thể: tỉ lệ điểm LiDAR của vật rơi vào box 2D ground truth.
    objects = []
    for obj in frame["labels"]:
        ouv, odepth = object_points_projected(frame["points"], calib, obj, frame["image"].shape)
        if len(ouv) == 0:
            continue
        ratio = in_box_ratio(ouv, obj.bbox)
        mean_depth = float(np.mean(odepth))
        objects.append({
            "type": obj.type, "in_box_ratio": ratio,
            "mean_depth_m": mean_depth, "n_obj_points": int(len(ouv)),
            "bucket": bucket_of(mean_depth),
        })
    out["objects"] = objects
    if objects:
        out["in_box_mean"] = float(np.mean([o["in_box_ratio"] for o in objects]))
        for bucket in ("near_lt15m", "mid_15_30m", "far_gt30m"):
            vals = [o["in_box_ratio"] for o in objects if o["bucket"] == bucket]
            out[f"in_box_{bucket}"] = float(np.mean(vals)) if vals else np.nan
    else:
        out["in_box_mean"] = np.nan
    return out


def _row_base(cfg: FrameConfig, perturb_type: str, level: float, unit: str) -> dict:
    return {"dataset": cfg.dataset, "frame_id": cfg.frame_id,
            "perturb_type": perturb_type, "level": level, "unit": unit}


def _add_metrics(row: dict, m: dict) -> None:
    for key in ("fov_ratio", "in_box_mean", "nmi_intensity",
                "in_box_near_lt15m", "in_box_mid_15_30m", "in_box_far_gt30m",
                "n_points_in_fov", "n_points"):
        row[key] = m.get(key, np.nan)


def yaw_sweep(configs: list[FrameConfig], yaw_levels_deg=YAW_LEVELS_DEG) -> pd.DataFrame:
    """Thí nghiệm chính: quét lệch yaw, đo tỉ lệ điểm trong FOV, tỉ lệ điểm trong box 2D
    của từng vật thể (tách theo khoảng cách), và chỉ số label-free NMI."""
    rows = []
    for cfg in configs:
        frame = load_frame(cfg.data_root, cfg.frame_id)
        for yaw in yaw_levels_deg:
            m = _project_frame(frame, {"yaw_deg": yaw})
            row = _row_base(cfg, "yaw", yaw, "deg")
            _add_metrics(row, m)
            rows.append(row)
    return pd.DataFrame(rows)


def object_detail(configs: list[FrameConfig], yaw_deg: float = 1.0) -> pd.DataFrame:
    """Bảng chi tiết theo từng vật thể ở một mức yaw (dùng để phân tích theo khoảng cách)."""
    rows = []
    for cfg in configs:
        frame = load_frame(cfg.data_root, cfg.frame_id)
        for yaw in (0.0, yaw_deg):
            m = _project_frame(frame, {"yaw_deg": yaw})
            for i, o in enumerate(m["objects"]):
                rows.append({"dataset": cfg.dataset, "frame_id": cfg.frame_id, "yaw_deg": yaw,
                             "obj_index": i, **o})
    return pd.DataFrame(rows)


def translation_sweep(configs: list[FrameConfig], levels_cm=TRANSLATION_LEVELS_CM) -> pd.DataFrame:
    """Quét dịch chuyển dọc trục x của LiDAR (cm) - dùng để so sánh với lệch xoay (bonus B1)."""
    rows = []
    for cfg in configs:
        frame = load_frame(cfg.data_root, cfg.frame_id)
        for t_cm in levels_cm:
            m = _project_frame(frame, {"t_xyz_m": (t_cm / 100.0, 0.0, 0.0)})
            row = _row_base(cfg, "translation_x", t_cm, "cm")
            _add_metrics(row, m)
            rows.append(row)
    return pd.DataFrame(rows)


def latency_benchmark(configs: list[FrameConfig], repeats: int = 30,
                      warmup: int = 3) -> pd.DataFrame:
    """Đo thời gian chạy phép chiếu: bỏ `warmup` lần đầu, lặp `repeats` lần, báo p50/p95."""
    rows = []
    for cfg in configs:
        frame = load_frame(cfg.data_root, cfg.frame_id)
        calib, shape = frame["calib"], frame["image"].shape
        points = frame["points"]
        for _ in range(warmup):                      # làm nóng, không tính vào kết quả
            cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2, shape)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2, shape)
            times.append((time.perf_counter() - t0) * 1e3)  # ms
        times = np.asarray(times)
        rows.append({
            "dataset": cfg.dataset, "frame_id": cfg.frame_id, "n_points": len(points),
            "repeats": repeats,
            "latency_p50_ms": float(np.percentile(times, 50)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "latency_mean_ms": float(times.mean()),
        })
    return pd.DataFrame(rows)
