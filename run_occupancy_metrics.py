#!/usr/bin/env python3
"""
Occupancy-Centric Perception Evaluation — CLI entry point (Phase 1).

Usage examples
--------------
# Use a preset config:
  python run_occupancy_metrics.py --config configs/occupancy_default.yaml

# Override parameters:
  python run_occupancy_metrics.py --config configs/occupancy_default.yaml \\
      --resolution 0.25 --range 60

# No config file:
  python run_occupancy_metrics.py \\
      --csv path/to/current.csv \\
      --output-dir output/occupancy/run_01 \\
      --resolution 0.5 --range 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from occupancy_metrics import OccupancyConfig, run_all


def _sep(title: str = "") -> str:
    bar = "=" * 70
    if title:
        pad = (70 - len(title) - 2) // 2
        return f"\n{'=' * pad} {title} {'=' * (70 - pad - len(title) - 2)}"
    return f"\n{bar}"


def print_results(results: dict, cfg: OccupancyConfig) -> None:
    print(_sep(f"OCCUPANCY METRICS  [{cfg.label}]"))
    print(f"  csv:        {cfg.csv_path}")
    print(f"  classes:    {cfg.classes}")
    print(f"  grid:       {cfg.resolution_m}m resolution, {cfg.range_m}m range")
    print(f"  weighting:  {cfg.distance_weight_type}")
    print(f"  timestamps: {results['n_timestamps']:,}")
    print(f"  GT objects:  {results['n_gt_objects_total']:,}    EST objects: {results['n_est_objects_total']:,}")

    g = results["grid"]
    print(_sep("1  Grid Occupancy Risk"))
    print(f"  GT occupied cells (total):   {g['gt_occupied_cells']:,}")
    print(f"  Pred occupied cells (total): {g['pred_occupied_cells']:,}")
    print(f"  True positive cells:         {g['true_positive_cells']:,}")
    print()
    print(f"  Safety Risk (false free):           {g['safety_risk_cells']:>10,} cells")
    print(f"  Availability Risk (false occupied): {g['availability_risk_cells']:>10,} cells")
    print(f"  Safety miss rate:                   {g['safety_miss_rate']:>10.4%}")
    print()
    print(f"  Safety Risk (1/r² weighted):        {g['safety_risk_weighted']:>10.2f}")
    print(f"  Availability Risk (1/r² weighted):  {g['availability_risk_weighted']:>10.2f}")

    print(_sep("2  Per-Class Breakdown"))
    print(f"  {'class':>12s}  {'safety_cells':>14s}  {'avail_cells':>12s}  {'miss_rate':>10s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*10}")
    for cls in cfg.classes:
        c = results["per_class"].get(cls, {})
        print(
            f"  {cls:>12s}  {c.get('safety_risk_cells', 0):>14,}"
            f"  {c.get('availability_risk_cells', 0):>12,}"
            f"  {c.get('safety_miss_rate', 0):>10.4%}"
        )

    p = results["polar"]
    print(_sep("3  Polar / Stixel Metrics"))
    print(f"  Safety (obstacle seen farther):    mean {p['safety_mean_m']:.4f} m,  max {p['safety_max_m']:.4f} m")
    print(f"  Availability (obstacle seen closer): mean {p['availability_mean_m']:.4f} m,  max {p['availability_max_m']:.4f} m")

    iv = results["polar_interval"]
    print(_sep("4  Polar Interval Risk"))
    print(f"  False free (missed occupancy):      {iv['false_free_m']:.2f} m (total across rays)")
    print(f"  False occupied (hallucinated):      {iv['false_occupied_m']:.2f} m (total across rays)")

    rc = results["ray_collision"]
    print(_sep("5  Ray-based Min-Collision-Distance TP/FP/FN (forward only)"))
    print(f"  forward window: ±{cfg.ray_collision_forward_angle_deg / 2:.0f}°   dist threshold: {cfg.ray_collision_dist_threshold_m} m")
    print(f"  {'class':>12s}  {'TP':>8s}  {'FP':>8s}  {'FN':>8s}  {'precision':>10s}  {'recall':>10s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for cls in cfg.classes:
        c = rc.get(cls, {"TP": 0, "FP": 0, "FN": 0, "precision": 0.0, "recall": 0.0})
        print(
            f"  {cls:>12s}  {c['TP']:>8,}  {c['FP']:>8,}  {c['FN']:>8,}"
            f"  {c['precision']:>10.4f}  {c['recall']:>10.4f}"
        )
    unk = rc.get("__unknown__", {"TP": 0, "FP": 0, "FN": 0, "precision": 0.0, "recall": 0.0})
    if unk["TP"] or unk["FP"] or unk["FN"]:
        print(
            f"  {'(unknown)':>12s}  {unk['TP']:>8,}  {unk['FP']:>8,}  {unk['FN']:>8,}"
            f"  {unk['precision']:>10.4f}  {unk['recall']:>10.4f}"
        )

    print(_sep())


def save_results(results: dict, cfg: OccupancyConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "label": cfg.label,
        "csv_path": cfg.csv_path,
        "config": {
            "resolution_m": cfg.resolution_m,
            "range_m": cfg.range_m,
            "classes": cfg.classes,
            "visibility_exclude": cfg.visibility_exclude,
            "distance_weight_type": cfg.distance_weight_type,
            "n_rays": cfg.n_rays,
            "ray_collision_dist_threshold_m": cfg.ray_collision_dist_threshold_m,
            "ray_collision_forward_angle_deg": cfg.ray_collision_forward_angle_deg,
        },
        "n_timestamps": results["n_timestamps"],
        "n_gt_objects_total": results["n_gt_objects_total"],
        "n_est_objects_total": results["n_est_objects_total"],
        "grid": results["grid"],
        "polar": results["polar"],
        "polar_interval": results["polar_interval"],
        "per_class": results["per_class"],
        "ray_collision": results["ray_collision"],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    cfg.to_yaml(out / "config_used.yaml")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Occupancy-centric perception evaluation (Phase 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    OccupancyConfig.add_args(parser)
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()
    cfg = OccupancyConfig.from_args(args)

    print(f"Loading {cfg.csv_path} ...", flush=True)
    df = pd.read_csv(cfg.csv_path)
    print(f"  {len(df):,} rows  |  {df['unix_time'].nunique():,} timestamps")

    print("Computing occupancy metrics ...", flush=True)
    results = run_all(df, cfg)

    if not args.quiet:
        print_results(results, cfg)

    out_dir = save_results(results, cfg)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
