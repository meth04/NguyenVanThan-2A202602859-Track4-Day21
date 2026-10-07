"""Hình học cho kiểm tra calibration LiDAR-camera (topic A).

Toàn bộ hàm ở đây làm việc trên khung dữ liệu chung do `starter.datasets.load_frame`
trả về (dict gồm points, calib, image, labels), nên dùng được cho cả KITTI và nuScenes.

Quy ước cần nhớ
---------------
- `points` (N, 4): x, y, z (mét) + intensity, trong hệ trục LiDAR.
- `calib`: P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4); `calib.T_cam_velo` là ma trận
  4x4 đưa điểm từ LiDAR sang rectified camera frame.
- Box 3D của label (`KittiObject`) nằm trong rectified camera frame; `location` là tâm
  đáy box, `dimensions` là (h, w, l), `rotation_y` quay quanh trục y của camera.
"""
from __future__ import annotations

import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import box3d_corners_cam, cam_to_image, velo_to_cam


def object_points_velo(points_velo: np.ndarray, calib: KittiCalib, obj: KittiObject) -> np.ndarray:
    """Lọc các điểm LiDAR nằm trong box 3D của một vật thể.

    Cách làm: đưa box 3D về hệ LiDAR bằng nghịch đảo `T_cam_velo`, rồi kiểm tra điểm
    có nằm trong hình hộp chữ nhật xoay (oriented bounding box - OBB) hay không.

    Tham số
    ------
    points_velo : (N, >=3) điểm LiDAR trong hệ LiDAR.
    calib       : calibration của frame.
    obj         : một `KittiObject` (box 3D trong rectified camera frame).

    Trả về
    ------
    (M, 3) các điểm nằm trong box (M <= N).
    """
    pts = np.asarray(points_velo, dtype=np.float64)[:, :3]
    T_velo_cam = np.linalg.inv(calib.T_cam_velo)  # camera -> LiDAR

    # Tâm và trục của box, đưa từ camera frame về LiDAR frame.
    R_cam = _ry_matrix(obj.rotation_y)
    center_cam = obj.location + R_cam @ np.array([0.0, -obj.dimensions[0] / 2.0, 0.0])  # tâm hình học
    center_velo = (T_velo_cam @ np.r_[center_cam, 1.0])[:3]

    # Ba trục đơn vị của box trong camera frame (l: x, h: y, w: z).
    axes_cam = R_cam  # cột 0 = trục dài, cột 1 = trục cao, cột 2 = trục rộng
    R_vc = T_velo_cam[:3, :3]
    axes_velo = R_vc @ axes_cam
    half = np.array([obj.dimensions[2], obj.dimensions[0], obj.dimensions[1]]) / 2.0  # (l, h, w)/2

    # Chiếu vector (điểm - tâm) lên 3 trục, so với nửa kích thước.
    rel = pts - center_velo
    local = rel @ axes_velo  # (N, 3)
    inside = np.all(np.abs(local) <= half + 1e-9, axis=1)
    return pts[inside]


def _ry_matrix(ry: float) -> np.ndarray:
    """Ma trận quay quanh trục y của camera (đúng quy ước KITTI)."""
    c, s = np.cos(ry), np.sin(ry)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def project_object_corners(obj: KittiObject, calib: KittiCalib,
                           image_shape: tuple[int, ...]) -> tuple[np.ndarray | None, np.ndarray]:
    """Chiếu 8 góc box 3D lên ảnh.

    Trả về (bbox2d, valid_corners):
        bbox2d        - (4,) [x1, y1, x2, y2] khung bao của 8 góc đã chiếu, hoặc None nếu
                        toàn bộ box nằm sau camera / không chiếu được.
        valid_corners - (8,) bool, True nếu góc tương ứng nằm trước camera (z_cam > 0).
    """
    corners = box3d_corners_cam(obj)                     # (8, 3) camera frame
    P2 = np.asarray(calib.P2, dtype=np.float64)
    s = corners @ P2[:, :3].T + P2[:, 3]                 # [s*u, s*v, s]
    depth = s[:, 2]
    valid = np.isfinite(corners).all(axis=1) & (depth > 0.1)
    if not valid.any():
        return None, valid
    uv_all = np.full((8, 2), np.nan)
    uv_all[valid] = s[valid, :2] / depth[valid, None]
    x1, y1 = np.nanmin(uv_all[:, 0]), np.nanmin(uv_all[:, 1])
    x2, y2 = np.nanmax(uv_all[:, 0]), np.nanmax(uv_all[:, 1])
    return np.array([x1, y1, x2, y2]), valid


def object_points_projected(points_velo: np.ndarray, calib: KittiCalib, obj: KittiObject,
                            image_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu riêng các điểm LiDAR thuộc một vật thể lên ảnh.

    Trả về (uv, depth): toạ độ pixel và độ sâu của các điểm thuộc box 3D. Các điểm nằm
    **sau** camera bị loại, nhưng điểm nằm trước camera mà rơi **ra ngoài khung ảnh** vẫn
    được giữ lại (với toạ độ âm hoặc vượt biên). Đây là điều cần thiết để đo đúng hiện
    tượng "điểm trôi ra khỏi box 2D": nếu lọc theo khung ảnh trước, những điểm đã trôi đi
    sẽ bị loại và che mất sai số của calibration lệch.
    """
    pts = object_points_velo(points_velo, calib, obj)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    cam = velo_to_cam(pts, calib)                    # (M, 3) camera frame
    P2 = np.asarray(calib.P2, dtype=np.float64)
    s = cam @ P2[:, :3].T + P2[:, 3]                 # [s*u, s*v, s]
    depth = s[:, 2]
    valid = np.isfinite(s).all(axis=1) & (depth > 0.1)
    uv = s[valid, :2] / depth[valid, None]
    return uv, depth[valid]


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU của hai box 2D dạng [x1, y1, x2, y2]."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def points_in_bbox(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """(M,) bool: điểm pixel nào nằm trong box 2D [x1, y1, x2, y2]."""
    if len(uv) == 0:
        return np.zeros(0, dtype=bool)
    return ((uv[:, 0] >= bbox[0]) & (uv[:, 0] <= bbox[2]) &
            (uv[:, 1] >= bbox[1]) & (uv[:, 1] <= bbox[3]))
