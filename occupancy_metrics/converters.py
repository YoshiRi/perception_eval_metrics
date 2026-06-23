"""Convert object lists (DataFrames) to OccupancyGrid / PolarOccupancy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OccupancyConfig
from .grid import OccupancyGrid
from .polar import PolarOccupancy


def _require_bbox_columns(df: pd.DataFrame, cfg: OccupancyConfig) -> None:
    missing = [c for c in (cfg.col_w, cfg.col_l, cfg.col_yaw) if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required bbox columns: {missing}. "
            f"Occupancy evaluation requires '{cfg.col_w}', '{cfg.col_l}', "
            f"and '{cfg.col_yaw}' columns."
        )


def bboxes_to_grid(
    objects: pd.DataFrame, cfg: OccupancyConfig
) -> OccupancyGrid:
    """Rasterise object bboxes onto an occupancy grid."""
    grid = OccupancyGrid(cfg.resolution_m, cfg.range_m)
    if objects.empty:
        return grid

    _require_bbox_columns(objects, cfg)

    for _, row in objects.iterrows():
        cx = float(row[cfg.col_x])
        cy = float(row[cfg.col_y])

        if abs(cx) > cfg.range_m or abs(cy) > cfg.range_m:
            continue

        w = float(row[cfg.col_w])
        l = float(row[cfg.col_l])
        yaw = float(row[cfg.col_yaw])
        if np.isnan(w) or np.isnan(l) or np.isnan(yaw):
            raise ValueError(
                f"Object at ({cx}, {cy}) has NaN bbox dimensions "
                f"(w={w}, l={l}, yaw={yaw}). All objects must have valid bbox values."
            )
        grid.fill_rotated_box(cx, cy, w, l, yaw)

    return grid


def bboxes_to_polar(
    objects: pd.DataFrame, cfg: OccupancyConfig
) -> PolarOccupancy:
    """Build polar occupancy from object bboxes."""
    polar = PolarOccupancy(cfg.n_rays, cfg.polar_max_range_m)
    if objects.empty:
        return polar

    _require_bbox_columns(objects, cfg)

    for _, row in objects.iterrows():
        cx = float(row[cfg.col_x])
        cy = float(row[cfg.col_y])
        dist = np.sqrt(cx ** 2 + cy ** 2)

        if dist > cfg.polar_max_range_m:
            continue

        w = float(row[cfg.col_w])
        l = float(row[cfg.col_l])
        yaw = float(row[cfg.col_yaw])
        if np.isnan(w) or np.isnan(l) or np.isnan(yaw):
            raise ValueError(
                f"Object at ({cx}, {cy}) has NaN bbox dimensions "
                f"(w={w}, l={l}, yaw={yaw}). All objects must have valid bbox values."
            )
        _add_bbox_to_polar(polar, cx, cy, w, l, yaw)

    return polar


def _add_bbox_to_polar(
    polar: PolarOccupancy,
    cx: float, cy: float, w: float, l: float, yaw: float,
) -> None:
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    hw, hl = w / 2.0, l / 2.0
    corners = np.array([
        [cx + cos_y * hl - sin_y * hw, cy + sin_y * hl + cos_y * hw],
        [cx + cos_y * hl + sin_y * hw, cy + sin_y * hl - cos_y * hw],
        [cx - cos_y * hl + sin_y * hw, cy - sin_y * hl - cos_y * hw],
        [cx - cos_y * hl - sin_y * hw, cy - sin_y * hl + cos_y * hw],
    ])

    corner_dists = np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
    corner_angles = np.arctan2(corners[:, 1], corners[:, 0])

    r_near = float(corner_dists.min())
    r_far = float(corner_dists.max())

    angle_min = float(corner_angles.min())
    angle_max = float(corner_angles.max())

    # Handle wrap-around when bbox straddles the ±π boundary
    angle_spread = angle_max - angle_min
    if angle_spread > np.pi:
        angle_min, angle_max = angle_max, angle_min + 2 * np.pi

    ray_start = int(angle_min / (2 * np.pi) * polar.n_rays) % polar.n_rays
    ray_end = int(angle_max / (2 * np.pi) * polar.n_rays) % polar.n_rays

    if ray_start <= ray_end:
        rays = range(ray_start, ray_end + 1)
    else:
        rays = list(range(ray_start, polar.n_rays)) + list(range(0, ray_end + 1))

    for ri in rays:
        polar.add_interval(ri, r_near, r_far)
