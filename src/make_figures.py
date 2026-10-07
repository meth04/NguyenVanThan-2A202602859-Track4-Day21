"""CLI tạo toàn bộ biểu đồ và ảnh minh hoạ cho topic A.

Chạy sau `python -m src.run_experiment` (cần các file CSV trong `results/`):

    python -m src.make_figures --help
    python -m src.make_figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.experiments import FrameConfig
from src.run_experiment import DEFAULT_CONFIGS, build_configs
from src import viz


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Tạo biểu đồ và ảnh minh hoạ cho topic A")
    ap.add_argument("--results-dir", default="results", help="nơi chứa CSV kết quả")
    ap.add_argument("--fig-dir", default="results/figures", help="nơi lưu ảnh/biểu đồ")
    ap.add_argument("--data-root", default="data/kitti_mini",
                    help="dataset dùng khi truyền --frames")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="chỉ tạo ảnh demo cho các frame này")
    args = ap.parse_args()

    res, figs = Path(args.results_dir), Path(args.fig_dir)
    configs = build_configs(args.frames, args.data_root)

    yaw = pd.read_csv(res / "yaw_perturb_sweep.csv")
    trans = pd.read_csv(res / "translation_perturb_sweep.csv")
    detail = pd.read_csv(res / "object_distance_detail.csv")
    lat_path = res / "latency_benchmark.csv"

    print("Vẽ biểu đồ ...")
    print(" ->", viz.plot_yaw_sweep(yaw, figs / "yaw_sweep_metrics.png"))
    print(" ->", viz.plot_distance_sensitivity(detail, figs / "distance_sensitivity.png"))
    print(" ->", viz.plot_perturbation_compare(yaw, trans, figs / "rotation_vs_translation.png"))
    if lat_path.exists():
        print(" ->", viz.plot_latency(pd.read_csv(lat_path), figs / "latency_benchmark.png"))

    print("Lưu ảnh overlay demo ...")
    for p in viz.save_demo_overlays(configs, figs):
        print(" ->", p)

    print("Lưu ảnh failure case ...")
    for p in viz.save_failure_images(configs, figs):
        print(" ->", p)
    print("Xong.")


if __name__ == "__main__":
    main()
