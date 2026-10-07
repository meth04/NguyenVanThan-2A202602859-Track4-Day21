"""Chẩn đoán sức khoẻ point cloud: tự động tìm frame bất thường và lỗi cài sẵn.

Đây là bước "kiểm tra dữ liệu trước khi dùng model" (mục tiêu học tập 1). Module này
tính một tập chỉ số cho mỗi frame, áp các quy tắc cảnh báo có ngưỡng, và in ra bảng
frame → cờ cảnh báo → lý do. Chạy được trên cả KITTI, nuScenes và bộ dữ liệu synthetic.

    python -m src.diagnose --data-root data/synthetic
    python -m src.diagnose --data-root data/kitti_mini --out results/data_health_kitti.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points


def frame_health(points: np.ndarray, timestamp_s: float | None = None,
                 n_azimuth_bins: int = 72) -> dict[str, float]:
    """Tính chỉ số sức khoẻ cho một frame point cloud (N, >=3)."""
    pts = np.asarray(points)
    finite = np.isfinite(pts[:, :3]).all(axis=1)
    p = pts[finite][:, :3]
    rng = np.linalg.norm(p[:, :2], axis=1)
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    az_hist, _ = np.histogram(az, bins=n_azimuth_bins, range=(-180, 180))
    # Bỏ 4 ô sát sau xe (±180°) vì vùng đó thường thưa tự nhiên với LiDAR quay.
    body = az_hist[4:-4] if len(az_hist) > 8 else az_hist
    med = np.median(body) if len(body) else 0.0

    # Tỉ lệ điểm trùng nhau (theo toạ độ làm tròn cm) - dấu hiệu dữ liệu bị nhân bản.
    if len(p):
        keys = np.round(p * 100).astype(np.int64)
        _, counts = np.unique(keys, axis=0, return_counts=True)
        dup_ratio = float((counts > 1).sum() / len(p))
    else:
        dup_ratio = np.nan

    return {
        "n_points": int(len(pts)),
        "n_finite": int(finite.sum()),
        "invalid_ratio": float(1 - finite.mean()) if len(pts) else 1.0,
        "range_p50": float(np.percentile(rng, 50)) if len(rng) else np.nan,
        "range_p95": float(np.percentile(rng, 95)) if len(rng) else np.nan,
        "range_max": float(rng.max()) if len(rng) else np.nan,
        "z_min": float(p[:, 2].min()) if len(p) else np.nan,
        "z_max": float(p[:, 2].max()) if len(p) else np.nan,
        "intensity_mean": float(pts[finite][:, 3].mean()) if finite.any() else np.nan,
        "intensity_std": float(pts[finite][:, 3].std()) if finite.any() else np.nan,
        "empty_az_bins": int((az_hist == 0).sum()),
        "min_sector_ratio": float(body.min() / med) if med > 0 else np.nan,
        "dup_ratio": dup_ratio,
        "timestamp_s": timestamp_s if timestamp_s is not None else np.nan,
    }


def _robust_z(series: pd.Series) -> pd.Series:
    """Điểm z bền vững theo MAD (median absolute deviation).

    Dùng MAD thay cho độ lệch chuẩn vì chính các frame bất thường sẽ kéo lệch mean/std,
    làm ngưỡng bị "nhiễm" và bỏ sót lỗi. MAD chịu được tối đa ~50% ngoại lai.
    """
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        mad = series.std() if series.std() > 0 else 1.0
    return (series - med).abs() / (1.4826 * mad)


def flag_rules(rows: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
    """Áp quy tắc cảnh báo lên bảng chỉ số của nhiều frame.

    Hai loại quy tắc:
      - Tuyệt đối (luôn là lỗi): có điểm NaN/Inf, có ô azimuth hoàn toàn trống,
        khoảng timestamp bất thường.
      - Bền vững (so với chính phân bố của bộ dữ liệu): chỉ số liên tục vừa lệch quá
        `z_thresh` lần MAD **vừa** lệch tương đối quá `min_rel` so với trung vị. Cần cả
        hai điều kiện vì với phân bố rất chụm, một khác biệt nhỏ (0.4 m tầm xa) vẫn cho
        z lớn nhưng không có ý nghĩa; ngược lại chỉ dùng ngưỡng tuyệt đối sẽ bỏ sót.

    `range_max` và `intensity_mean` **không** dùng để gắn cờ: chúng phụ thuộc hình học
    cảnh và điều kiện ngày/đêm một cách hợp pháp, nên chỉ được báo cáo chứ không phải lỗi.
    """
    checks = {"n_points": 0.05, "min_sector_ratio": 0.25, "dup_ratio": 0.50}
    z = pd.DataFrame({c: _robust_z(rows[c]) for c in checks if c in rows})
    gaps = rows["timestamp_s"].diff()
    med_gap = gaps[gaps > 0].median() if (gaps > 0).any() else np.nan

    flags = []
    for i, r in rows.iterrows():
        reasons = []
        if r["invalid_ratio"] > 0.0005:
            reasons.append(f"NaN/Inf {r['invalid_ratio']:.2%}")
        if r["empty_az_bins"] > 0:
            reasons.append(f"{int(r['empty_az_bins'])} ô azimuth trống")
        if not np.isnan(med_gap) and i > 0 and not np.isnan(r["timestamp_s"]):
            gap = r["timestamp_s"] - rows.loc[i - 1, "timestamp_s"]
            if gap > 1.5 * med_gap:
                reasons.append(f"khoảng timestamp {gap:.2f}s (chuẩn {med_gap:.2f}s)")
        for c, min_rel in checks.items():
            med = rows[c].median()
            rel_signed = (r[c] - med) / med if med else 0.0
            if z.loc[i, c] > z_thresh and abs(rel_signed) > min_rel:
                direction = "thấp" if rel_signed < 0 else "cao"
                reasons.append(f"{c} {direction} bất thường ({100*rel_signed:+.0f}% so với trung vị)")
        flags.append({"flagged": bool(reasons), "reasons": "; ".join(reasons) or "-"})
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(flags)], axis=1)


def _timestamps(data_root: str | Path, frames: list[str]) -> dict[str, float]:
    """Đọc timestamps.txt nếu có (chỉ bộ synthetic). KITTI 3D Object không phát hành timestamp."""
    path = Path(data_root) / "training" / "timestamps.txt"
    if not path.exists():
        return {}
    vals = [float(x) for x in path.read_text().split()]
    return {fid: vals[i] for i, fid in enumerate(frames) if i < len(vals)}


def diagnose(data_root: str | Path) -> pd.DataFrame:
    frames = list_frames(data_root)
    ts = _timestamps(data_root, frames)
    rows = [{"frame_id": fid, **frame_health(load_points(data_root, fid), ts.get(fid))}
            for fid in frames]
    return flag_rules(pd.DataFrame(rows))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Chẩn đoán sức khoẻ point cloud và cảnh báo frame bất thường")
    ap.add_argument("--data-root", default="data/synthetic", help="thư mục KITTI hoặc nuScenes")
    ap.add_argument("--out", default=None, help="file CSV kết quả (mặc định: results/data_health_<ten>.csv)")
    args = ap.parse_args()

    table = diagnose(args.data_root)
    # Đặt tên theo thư mục dữ liệu (synthetic / kitti_mini / nuscenes_mini_subset) để
    # không ghi đè lẫn nhau giữa các bộ.
    tag = Path(args.data_root).name
    out = Path(args.out) if args.out else Path("results") / f"data_health_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.6f")

    print(f"Sức khoẻ dữ liệu: {args.data_root} ({len(table)} frame)")
    print(f"{'frame':>12} | {'n_points':>9} | {'invalid':>7} | {'min_sector':>10} | cảnh báo")
    for _, r in table.iterrows():
        mark = "!!" if r["flagged"] else "  "
        print(f"{mark}{r['frame_id']:>10} | {r['n_points']:>9d} | {r['invalid_ratio']:>7.2%} | "
              f"{r['min_sector_ratio']:>10.2f} | {r['reasons']}")
    print(f"-> {out}")
    n_flag = int(table["flagged"].sum())
    print(f"Tổng: {n_flag}/{len(table)} frame bị gắn cờ.")


if __name__ == "__main__":
    main()
