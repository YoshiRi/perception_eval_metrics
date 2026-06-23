"""Core occupancy-centric evaluation metrics (Phase 1)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import OccupancyConfig
from .grid import OccupancyGrid, EgoState
from .converters import bboxes_to_grid, bboxes_to_polar
from .polar import compute_polar_risk, compute_polar_interval_risk


def compute_occupancy_risk(
    gt_grid: OccupancyGrid,
    pred_grid: OccupancyGrid,
    ego_state: EgoState | None = None,
    distance_weight_type: str = "inverse_square",
    gt_full_grid: OccupancyGrid | None = None,
    distance_map: np.ndarray | None = None,
) -> Dict:
    """
    Compute occupancy-based safety and availability risk.

    Safety Risk:       GT Occupied ∩ Pred Free  (false free)
    Availability Risk: GT_full Free ∩ Pred Occupied  (false occupied)

    gt_full_grid: GT grid including visibility-excluded objects.
        When provided, availability risk uses this grid so that EST
        detections matched to excluded-visibility GTs are not counted
        as false occupied.
    distance_map: pre-computed distance map (same shape as grids).
        Avoids recomputing every call when resolution/range are fixed.
    """
    gt = gt_grid.data
    pred = pred_grid.data
    gt_full = gt_full_grid.data if gt_full_grid is not None else gt

    false_free = gt & ~pred
    false_occupied = ~gt_full & pred
    true_positive = gt & pred

    if distance_map is not None:
        dist = distance_map
    else:
        dist = gt_grid.distance_map()
    in_range = dist <= gt_grid.range_m

    eps = 1e-6

    if distance_weight_type == "inverse_square":
        w = 1.0 / (dist ** 2 + eps)
    else:
        w = np.ones_like(dist)

    gt_occupied_count = int(gt[in_range].sum())
    pred_occupied_count = int(pred[in_range].sum())

    safety_raw = int(false_free[in_range].sum())
    avail_raw = int(false_occupied[in_range].sum())
    tp_raw = int(true_positive[in_range].sum())

    safety_weighted = float((w * false_free)[in_range].sum())
    avail_weighted = float((w * false_occupied)[in_range].sum())

    safety_rate = safety_raw / gt_occupied_count if gt_occupied_count > 0 else 0.0
    gt_full_occupied = int(gt_full[in_range].sum())
    free_cells = int((~gt_full)[in_range].sum())
    avail_rate = avail_raw / free_cells if free_cells > 0 else 0.0

    return {
        "safety_risk_cells": safety_raw,
        "availability_risk_cells": avail_raw,
        "true_positive_cells": tp_raw,
        "safety_risk_weighted": round(safety_weighted, 4),
        "availability_risk_weighted": round(avail_weighted, 4),
        "safety_miss_rate": round(safety_rate, 6),
        "availability_false_alarm_rate": round(avail_rate, 6),
        "gt_occupied_cells": gt_occupied_count,
        "pred_occupied_cells": pred_occupied_count,
        "total_cells_in_range": int(in_range.sum()),
    }


def _filter_objects(
    df: pd.DataFrame, cfg: OccupancyConfig, source: str,
    *, skip_visibility_filter: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame to objects within evaluation scope."""
    mask = (
        (df["source"] == source)
        & (df["label"].isin(cfg.classes))
    )

    dist = np.sqrt(df[cfg.col_x] ** 2 + df[cfg.col_y] ** 2)
    mask = mask & (dist <= cfg.range_m)

    if source == "GT" and not skip_visibility_filter:
        mask = mask & (~df["visibility"].isin(cfg.visibility_exclude))
        mask = mask & (df["visibility"].notna())

    return df[mask].copy()


def evaluate_timestamp(
    ts_df: pd.DataFrame, cfg: OccupancyConfig,
    distance_map: np.ndarray | None = None,
) -> Dict:
    """Evaluate a single timestamp: grid + polar metrics."""
    gt_objs = _filter_objects(ts_df, cfg, "GT")
    gt_all_objs = _filter_objects(ts_df, cfg, "GT", skip_visibility_filter=True)
    est_objs = _filter_objects(ts_df, cfg, "EST")

    gt_grid = bboxes_to_grid(gt_objs, cfg)
    gt_full_grid = bboxes_to_grid(gt_all_objs, cfg)
    pred_grid = bboxes_to_grid(est_objs, cfg)

    grid_risk = compute_occupancy_risk(
        gt_grid, pred_grid,
        distance_weight_type=cfg.distance_weight_type,
        gt_full_grid=gt_full_grid,
        distance_map=distance_map,
    )

    gt_polar = bboxes_to_polar(gt_objs, cfg)
    pred_polar = bboxes_to_polar(est_objs, cfg)

    polar_risk = compute_polar_risk(gt_polar, pred_polar)
    interval_risk = compute_polar_interval_risk(gt_polar, pred_polar)

    return {
        "grid": grid_risk,
        "polar": {
            "safety_mean": polar_risk["safety_mean"],
            "safety_max": polar_risk["safety_max"],
            "availability_mean": polar_risk["availability_mean"],
            "availability_max": polar_risk["availability_max"],
            "n_valid_rays": polar_risk["n_valid_rays"],
        },
        "polar_interval": interval_risk,
        "n_gt_objects": len(gt_objs),
        "n_est_objects": len(est_objs),
    }


def _per_class_grid_risk(
    ts_df: pd.DataFrame, cfg: OccupancyConfig,
    distance_map: np.ndarray | None = None,
) -> Dict[str, Dict]:
    """Compute grid risk broken down by class."""
    results = {}
    for cls in cfg.classes:
        cls_df = ts_df[ts_df["label"] == cls]
        if cls_df.empty:
            continue

        gt_objs = _filter_objects(cls_df, cfg, "GT")
        gt_all_objs = _filter_objects(cls_df, cfg, "GT", skip_visibility_filter=True)
        est_objs = _filter_objects(cls_df, cfg, "EST")

        gt_grid = bboxes_to_grid(gt_objs, cfg)
        gt_full_grid = bboxes_to_grid(gt_all_objs, cfg)
        pred_grid = bboxes_to_grid(est_objs, cfg)

        risk = compute_occupancy_risk(
            gt_grid, pred_grid,
            distance_weight_type=cfg.distance_weight_type,
            gt_full_grid=gt_full_grid,
            distance_map=distance_map,
        )
        results[cls] = risk
    return results


def run_all(df: pd.DataFrame, cfg: OccupancyConfig) -> Dict:
    """
    Full Phase 1 pipeline: for each timestamp, compute occupancy risk.
    Returns aggregated results.
    """
    timestamps = sorted(df["unix_time"].unique())
    n = len(timestamps)

    ref_grid = OccupancyGrid(cfg.resolution_m, cfg.range_m)
    cached_distance_map = ref_grid.distance_map()

    # Accumulators
    grid_totals = {
        "safety_risk_cells": 0,
        "availability_risk_cells": 0,
        "true_positive_cells": 0,
        "safety_risk_weighted": 0.0,
        "availability_risk_weighted": 0.0,
        "gt_occupied_cells": 0,
        "pred_occupied_cells": 0,
    }
    polar_totals = {
        "safety_sum": 0.0,
        "availability_sum": 0.0,
        "safety_max": 0.0,
        "availability_max": 0.0,
        "n_valid_rays_sum": 0,
    }
    interval_totals = {
        "false_free_m": 0.0,
        "false_occupied_m": 0.0,
    }
    per_class_totals: Dict[str, Dict] = {}
    total_gt_objects = 0
    total_est_objects = 0

    for i, ts in enumerate(timestamps):
        if i % 100 == 0:
            print(f"  evaluating {i}/{n} timestamps ...", end="\r", flush=True)

        ts_df = df[df["unix_time"] == ts]
        result = evaluate_timestamp(ts_df, cfg, distance_map=cached_distance_map)

        g = result["grid"]
        for k in grid_totals:
            grid_totals[k] += g[k]

        p = result["polar"]
        polar_totals["safety_sum"] += p["safety_mean"] * p["n_valid_rays"]
        polar_totals["availability_sum"] += p["availability_mean"] * p["n_valid_rays"]
        polar_totals["safety_max"] = max(polar_totals["safety_max"], p["safety_max"])
        polar_totals["availability_max"] = max(polar_totals["availability_max"], p["availability_max"])
        polar_totals["n_valid_rays_sum"] += p["n_valid_rays"]

        iv = result["polar_interval"]
        interval_totals["false_free_m"] += iv["interval_false_free_m"]
        interval_totals["false_occupied_m"] += iv["interval_false_occupied_m"]

        total_gt_objects += result["n_gt_objects"]
        total_est_objects += result["n_est_objects"]

        cls_risks = _per_class_grid_risk(ts_df, cfg, distance_map=cached_distance_map)
        for cls, risk in cls_risks.items():
            if cls not in per_class_totals:
                per_class_totals[cls] = {k: 0 for k in grid_totals}
            for k in grid_totals:
                per_class_totals[cls][k] += risk[k]

    print(f"  evaluated {n}/{n} timestamps.      ")

    # Aggregate
    gt_occ = grid_totals["gt_occupied_cells"]
    safety_miss_rate = grid_totals["safety_risk_cells"] / gt_occ if gt_occ > 0 else 0.0

    n_valid = polar_totals["n_valid_rays_sum"]
    polar_summary = {
        "safety_mean_m": round(polar_totals["safety_sum"] / n_valid, 4) if n_valid > 0 else 0.0,
        "safety_max_m": round(polar_totals["safety_max"], 4),
        "availability_mean_m": round(polar_totals["availability_sum"] / n_valid, 4) if n_valid > 0 else 0.0,
        "availability_max_m": round(polar_totals["availability_max"], 4),
    }

    per_class_summary = {}
    for cls in cfg.classes:
        if cls not in per_class_totals:
            per_class_summary[cls] = {"safety_risk_cells": 0, "availability_risk_cells": 0, "safety_miss_rate": 0.0}
            continue
        ct = per_class_totals[cls]
        cls_gt = ct["gt_occupied_cells"]
        per_class_summary[cls] = {
            "safety_risk_cells": ct["safety_risk_cells"],
            "availability_risk_cells": ct["availability_risk_cells"],
            "safety_miss_rate": round(ct["safety_risk_cells"] / cls_gt, 6) if cls_gt > 0 else 0.0,
        }

    return {
        "n_timestamps": n,
        "n_gt_objects_total": total_gt_objects,
        "n_est_objects_total": total_est_objects,
        "grid": {
            "safety_risk_cells": grid_totals["safety_risk_cells"],
            "availability_risk_cells": grid_totals["availability_risk_cells"],
            "true_positive_cells": grid_totals["true_positive_cells"],
            "safety_risk_weighted": round(grid_totals["safety_risk_weighted"], 4),
            "availability_risk_weighted": round(grid_totals["availability_risk_weighted"], 4),
            "safety_miss_rate": round(safety_miss_rate, 6),
            "gt_occupied_cells": gt_occ,
            "pred_occupied_cells": grid_totals["pred_occupied_cells"],
        },
        "polar": polar_summary,
        "polar_interval": {
            "false_free_m": round(interval_totals["false_free_m"], 4),
            "false_occupied_m": round(interval_totals["false_occupied_m"], 4),
        },
        "per_class": per_class_summary,
    }
