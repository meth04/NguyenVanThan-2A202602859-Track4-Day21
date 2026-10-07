"""CLI chạy toàn bộ thí nghiệm số liệu của topic A và ghi CSV vào `results/`.

Ví dụ
-----
    python -m src.run_experiment --help
    python -m src.run_experiment                       # cấu hình mặc định
    python -m src.run_experiment --frames 000011 000049 --repeats 30

Mọi bước đều tất định (không dùng phép ngẫu nhiên), nên chạy lại cho ra đúng cùng số liệu.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import (FrameConfig, latency_benchmark, object_detail, translation_sweep,
                             yaw_sweep)

# Bộ frame mặc định: 1 synthetic để dễ đối chiếu, 2 KITTI (nhiều người / có vật bị che),
# 1 nuScenes để so sánh hai sensor (bonus B5).
DEFAULT_CONFIGS = [
    FrameConfig("synthetic", "data/synthetic", "000000"),
    FrameConfig("kitti", "data/kitti_mini", "000011"),
    FrameConfig("kitti", "data/kitti_mini", "000049"),
    FrameConfig("nuscenes", "data/nuscenes_mini_subset", "scene-0103_010"),
]


def build_configs(frames: list[str] | None, data_root: str) -> list[FrameConfig]:
    if not frames:
        return DEFAULT_CONFIGS
    return [FrameConfig("custom", data_root, f) for f in frames]


def detection_thresholds(yaw: pd.DataFrame, rel_drop: float = 0.05) -> pd.DataFrame:
    """Với mỗi frame, tìm mức yaw nhỏ nhất mà chỉ số label-free (NMI) giảm tương đối
    ít nhất `rel_drop` so với mức chuẩn (yaw = 0). Dùng để định ngưỡng phát hiện drift.
    """
    rows = []
    for (dataset, frame_id), g in yaw.groupby(["dataset", "frame_id"], sort=False):
        g = g.sort_values("level")
        clean = g.iloc[0]
        base = clean["nmi_intensity"]
        for _, r in g.iterrows():
            drop = (base - r["nmi_intensity"]) / base if base > 0 else 0.0
            if drop >= rel_drop:
                rows.append({"dataset": dataset, "frame_id": frame_id,
                             "clean_nmi": base, "detect_yaw_deg": r["level"],
                             "rel_drop_at_detect": drop, "detected": True})
                break
        else:
            rows.append({"dataset": dataset, "frame_id": frame_id,
                         "clean_nmi": base, "detect_yaw_deg": np.nan,
                         "rel_drop_at_detect": np.nan, "detected": False})
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="Chạy sweep calibration (yaw/dịch chuyển) và đo latency cho topic A.")
    ap.add_argument("--out-dir", default="results", help="thư mục ghi CSV (mặc định: results)")
    ap.add_argument("--data-root", default="data/kitti_mini",
                    help="dataset dùng khi truyền --frames (mặc định: data/kitti_mini)")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="chỉ chạy các frame này (mặc định: bộ frame chuẩn của cả 3 dataset)")
    ap.add_argument("--repeats", type=int, default=30, help="số lần lặp đo latency (>= 20)")
    ap.add_argument("--skip-latency", action="store_true", help="bỏ phần đo latency")
    args = ap.parse_args()

    configs = build_configs(args.frames, args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Yaw sweep trên {len(configs)} frame ...")
    yaw = yaw_sweep(configs)
    yaw.to_csv(out / "yaw_perturb_sweep.csv", index=False, float_format="%.6f")
    print(f"      -> {out / 'yaw_perturb_sweep.csv'} ({len(yaw)} dòng)")

    print("[2/5] Chi tiết theo vật thể ở yaw = 1 độ ...")
    detail = object_detail(configs, yaw_deg=1.0)
    detail.to_csv(out / "object_distance_detail.csv", index=False, float_format="%.6f")
    print(f"      -> {out / 'object_distance_detail.csv'} ({len(detail)} dòng)")

    print("[3/5] Ngưỡng phát hiện drift từ chỉ số label-free ...")
    thr = detection_thresholds(yaw)
    thr.to_csv(out / "detection_threshold.csv", index=False, float_format="%.6f")
    print(f"      -> {out / 'detection_threshold.csv'} ({len(thr)} dòng)")

    print("[4/5] Translation sweep (so sánh với lệch xoay, bonus B1) ...")
    trans = translation_sweep(configs)
    trans.to_csv(out / "translation_perturb_sweep.csv", index=False, float_format="%.6f")
    print(f"      -> {out / 'translation_perturb_sweep.csv'} ({len(trans)} dòng)")

    if not args.skip_latency:
        print(f"[5/5] Latency benchmark ({args.repeats} lần lặp, bỏ 3 lần làm nóng) ...")
        lat = latency_benchmark(configs, repeats=max(args.repeats, 20))
        lat.to_csv(out / "latency_benchmark.csv", index=False, float_format="%.4f")
        print(f"      -> {out / 'latency_benchmark.csv'} ({len(lat)} dòng)")

    print("\nTóm tắt yaw sweep (fov_ratio, in_box_mean, nmi_intensity):")
    show = yaw[["dataset", "frame_id", "level", "fov_ratio", "in_box_mean", "nmi_intensity"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNgưỡng phát hiện drift:")
    print(thr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
