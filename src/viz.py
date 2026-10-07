"""Vẽ biểu đồ (matplotlib) và ảnh minh hoạ (OpenCV) cho topic A.

Quy ước thị giác: nền sáng, lưới mảnh, nhãn trực tiếp chọn lọc, bảng màu phân loại cố
định thứ tự (xanh dương, cam, xanh ngọc, vàng...). Bảng màu này đã được kiểm tra về khả
năng phân biệt với người mù màu (CVD) ở chế độ sáng.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.geometry import object_points_projected  # noqa: E402
from starter.projection import perturb_extrinsic, project_velo_to_image  # noqa: E402

# --- Token thị giác (lấy từ bảng màu chuẩn, đã validate CVD) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#86b6ef", "#2a78d6", "#184f95"]  # ordinal 3 mức (đủ tối để đọc trên nền sáng)
STATUS_CRIT = "#d03b3b"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_yaw_sweep(yaw: pd.DataFrame, out: str | Path) -> Path:
    """Hai panel: (a) chỉ số label-free NMI và (b) tỉ lệ điểm trong box 2D, theo mức lệch yaw."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    groups = list(yaw.groupby(["dataset", "frame_id"], sort=False))
    for i, ((dataset, frame_id), g) in enumerate(groups):
        g = g.sort_values("level")
        c = CAT[i % len(CAT)]
        label = f"{dataset} · {frame_id}"
        for ax, col in ((axes[0], "nmi_intensity"), (axes[1], "in_box_mean")):
            ax.plot(g["level"], g[col], color=c, lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        # Nhãn trực tiếp ở điểm cuối (chọn lọc, không ghi mọi điểm).
        axes[0].annotate(label, (g["level"].iloc[-1], g["nmi_intensity"].iloc[-1]),
                         xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=7.5, va="center")
    axes[0].set_title("Chỉ số label-free: NMI(cường độ LiDAR, ảnh)", color=INK, fontsize=10, loc="left")
    axes[0].set_xlabel("Lệch yaw (độ)", color=INK2)
    axes[0].set_ylabel("NMI", color=INK2)
    axes[1].set_title("Tỉ lệ điểm trong box 2D ground truth", color=INK, fontsize=10, loc="left")
    axes[1].set_xlabel("Lệch yaw (độ)", color=INK2)
    axes[1].set_ylabel("Tỉ lệ điểm trong box", color=INK2)
    for ax in axes:
        _style(ax)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="best")
    fig.suptitle("Độ nhạy calibration theo lệch yaw (0–3°)", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out)


def plot_distance_sensitivity(detail: pd.DataFrame, out: str | Path) -> Path:
    """Cột nhóm: tỉ lệ điểm trong box theo nhóm khoảng cách, ở yaw 0° và yaw 1°."""
    order = ["near_lt15m", "mid_15_30m", "far_gt30m"]
    agg = (detail.groupby(["bucket", "yaw_deg"])["in_box_ratio"].mean().reset_index())
    yaws = sorted(detail["yaw_deg"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    x = np.arange(len(order))
    w = 0.36
    for j, yv in enumerate(yaws):
        vals = [agg[(agg.bucket == b) & (agg.yaw_deg == yv)]["in_box_ratio"].mean() for b in order]
        vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
        off = (j - (len(yaws) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w - 0.02, color=CAT[j], label=f"yaw = {yv:g}°", zorder=3)
        for b, v in zip(bars, vals):        # nhãn trực tiếp trên đỉnh cột
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, ["< 15 m", "15–30 m", "> 30 m"])
    ax.set_xlabel("Nhóm khoảng cách tới vật thể", color=INK2)
    ax.set_ylabel("Tỉ lệ điểm trong box 2D", color=INK2)
    ax.set_title("Sai khớp tăng theo khoảng cách", color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, 1.05)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    return _save(fig, out)


def plot_perturbation_compare(yaw: pd.DataFrame, trans: pd.DataFrame, out: str | Path) -> Path:
    """So sánh độ giảm tương đối của in_box khi lệch xoay (yaw) và khi dịch chuyển (cm)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, df, xlabel, title in (
        (axes[0], yaw, "Lệch yaw (độ)", "(a) Lệch xoay"),
        (axes[1], trans, "Dịch chuyển (cm)", "(b) Dịch chuyển dọc trục"),
    ):
        for i, ((dataset, frame_id), g) in enumerate(df.groupby(["dataset", "frame_id"], sort=False)):
            g = g.sort_values("level")
            base = g["in_box_mean"].iloc[0]
            drop = 100.0 * (base - g["in_box_mean"]) / base if base > 0 else g["in_box_mean"] * 0
            ax.plot(g["level"], drop, color=CAT[i % len(CAT)], lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{dataset} · {frame_id}")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_ylabel("Suy giảm in_box so với chuẩn (%)", color=INK2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="best")
    fig.suptitle("Xoay hay dịch chuyển gây sai khớp nhiều hơn?", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out)


def plot_latency(lat: pd.DataFrame, out: str | Path) -> Path:
    """Cột nhóm: latency p50/p95 của phép chiếu trên từng frame."""
    labels = [f"{r.dataset}\n{r.frame_id}" for r in lat.itertuples()]
    x = np.arange(len(lat))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for j, col in enumerate(("latency_p50_ms", "latency_p95_ms")):
        vals = lat[col].to_numpy()
        off = (j - 0.5) * w
        bars = ax.bar(x + off, vals, w - 0.02, color=CAT[j],
                      label="p50" if j == 0 else "p95", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Thời gian (ms)", color=INK2)
    ax.set_title("Latency phép chiếu LiDAR→ảnh (CPU, bỏ 3 lần làm nóng, 30 lần lặp)",
                 color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Ảnh minh hoạ (OpenCV)
# ---------------------------------------------------------------------------
def _overlay(frame: dict, calib, draw_boxes: bool = True) -> np.ndarray:
    from starter.projection import draw_box2d, overlay_points
    uv, depth, mask = project_velo_to_image(frame["points"], calib, frame["image"].shape)
    vis = overlay_points(frame["image"], uv, depth)
    if draw_boxes:
        for obj in frame["labels"]:
            vis = draw_box2d(vis, obj.bbox, label=obj.type)
    return vis


def save_demo_overlays(configs, out_dir: str | Path) -> list[Path]:
    """Lưu ảnh overlay calibration đúng cho từng frame (sản phẩm demo bắt buộc)."""
    from starter.datasets import load_frame
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cfg in configs:
        frame = load_frame(cfg.data_root, cfg.frame_id)
        vis = _overlay(frame, frame["calib"])
        p = out_dir / f"demo_overlay_{cfg.dataset}_{cfg.frame_id}.png"
        cv2.imwrite(str(p), vis)
        paths.append(p)
    return paths


def save_failure_images(configs, out_dir: str | Path) -> list[Path]:
    """Tạo các ảnh failure case:
      fail_01: nuScenes - chỉ số label-free không phát hiện được drift.
      fail_02: KITTI 000011 - tỉ lệ trong box không đơn điệu (metric artifact).
      fail_03: vật thể ở gần làm tỉ lệ trong box bão hoà (synthetic).
    """
    from starter.datasets import load_frame
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # fail_01: nuScenes, yaw 0 vs 2 độ, kèm chú thích NMI/in_box.
    cfg = next(c for c in configs if c.dataset == "nuscenes")
    frame = load_frame(cfg.data_root, cfg.frame_id)
    panels = []
    for yaw in (0.0, 2.0):
        calib = perturb_extrinsic(frame["calib"], yaw_deg=yaw)
        panels.append(_overlay(frame, calib))
    top = np.hstack([_pad_label(p, f"yaw = 0 deg (chuan)") for p in panels[:1]] +
                    [_pad_label(panels[1], "yaw = 2 deg: in_box giam, NMI gan nhu khong doi")])
    p = out_dir / "fail_01_nuscenes_labelfree_miss.png"
    cv2.imwrite(str(p), top)
    paths.append(p)

    # fail_02: KITTI 000011, yaw 0 / 2 / 3 - in_box không đơn điệu.
    cfg = next(c for c in configs if c.dataset == "kitti" and c.frame_id == "000011")
    frame = load_frame(cfg.data_root, cfg.frame_id)
    panels = []
    for yaw, tag in ((0.0, "yaw=0: in_box=0.84"), (2.0, "yaw=2: in_box=0.59"),
                     (3.0, "yaw=3: in_box=0.75 (tang lai)")):
        calib = perturb_extrinsic(frame["calib"], yaw_deg=yaw)
        panels.append(_pad_label(_overlay(frame, calib), tag))
    p = out_dir / "fail_02_kitti_nonmonotonic_inbox.png"
    cv2.imwrite(str(p), np.hstack(panels))
    paths.append(p)

    # fail_03: synthetic - lệch yaw 2 độ, box gần như vẫn khớp (bão hoà).
    cfg = next(c for c in configs if c.dataset == "synthetic")
    frame = load_frame(cfg.data_root, cfg.frame_id)
    calib = perturb_extrinsic(frame["calib"], yaw_deg=2.0)
    p = out_dir / "fail_03_synthetic_near_saturation.png"
    cv2.imwrite(str(p), _pad_label(_overlay(frame, calib), "yaw=2: vat gan nen in_box van ~0.92"))
    paths.append(p)
    return paths


def _pad_label(img: np.ndarray, text: str) -> np.ndarray:
    """Thêm dải nhãn phía trên ảnh để chú thích."""
    bar = np.full((26, img.shape[1], 3), 252, dtype=np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    return np.vstack([bar, img])
