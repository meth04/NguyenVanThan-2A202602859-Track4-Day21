"""Chỉ số đánh giá sai khớp calibration cho topic A.

Bài lab cần hai loại chỉ số, vì mỗi loại trả lời một câu hỏi khác nhau:

1. **Chỉ số cần nhãn (label-based)** — `in_box_ratio`
   Tỉ lệ điểm LiDAR của một vật thể rơi vào box 2D ground truth của chính vật đó.
   Nhạy và dễ hiểu, nhưng chỉ đo được ở frame đã gán nhãn.

2. **Chỉ số không cần nhãn (label-free)** — `nmi_at_pixels`
   Thông tin tương hỗ chuẩn hoá (NMI) giữa cường độ phản xạ LiDAR và độ sáng ảnh tại
   đúng vị trí chiếu. Khi calibration đúng, hai tín hiệu "khớp" nhau; khi lệch, chúng
   trôi khỏi nhau và NMI giảm. Đây là chỉ số có thể chạy trên xe mà không cần nhãn.

Ngoài ra có `fov_ratio` (tỉ lệ điểm vào khung ảnh) — chỉ số thô, dùng để so sánh.
"""
from __future__ import annotations

import cv2
import numpy as np

# Ngưỡng khoảng cách (mét) để chia nhóm khi phân tích độ nhạy theo khoảng cách.
DISTANCE_BUCKETS = {"near_lt15m": (0.0, 15.0), "mid_15_30m": (15.0, 30.0), "far_gt30m": (30.0, np.inf)}


def fov_ratio(mask: np.ndarray) -> float:
    """Tỉ lệ điểm LiDAR nằm trong khung ảnh (mask bool (N,) từ `cam_to_image`)."""
    return float(np.mean(mask)) if len(mask) else 0.0


def in_box_ratio(uv: np.ndarray, bbox: np.ndarray) -> float:
    """Tỉ lệ điểm pixel rơi vào box 2D [x1, y1, x2, y2]."""
    if len(uv) == 0:
        return 0.0
    inside = ((uv[:, 0] >= bbox[0]) & (uv[:, 0] <= bbox[2]) &
              (uv[:, 1] >= bbox[1]) & (uv[:, 1] <= bbox[3]))
    return float(inside.mean())


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 16) -> float:
    """NMI giữa hai tín hiệu 1 chiều (giá trị trong [0, 1]; 0 = độc lập hoàn toàn)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 20 or a.std() == 0 or b.std() == 0:
        return 0.0
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    p = hist / hist.sum()
    px, py = p.sum(axis=1), p.sum(axis=0)
    nz = p > 0
    mi = float(np.sum(p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return 2.0 * mi / (hx + hy) if (hx + hy) > 0 else 0.0


def nmi_at_pixels(intensity: np.ndarray, uv: np.ndarray, image: np.ndarray, bins: int = 16) -> float:
    """NMI giữa cường độ LiDAR và độ sáng ảnh tại các toạ độ pixel `uv` (M, 2)."""
    if len(uv) < 20:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = gray.shape
    u = np.clip(uv[:, 0].astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].astype(int), 0, H - 1)
    return normalized_mutual_information(intensity, gray[v, u], bins)


def bucket_of(depth: float) -> str:
    """Xếp một giá trị độ sâu vào nhóm gần / trung bình / xa."""
    for name, (lo, hi) in DISTANCE_BUCKETS.items():
        if lo <= depth < hi:
            return name
    return "far_gt30m"
