"""Convert object lists (DataFrames) to OccupancyGrid / PolarOccupancy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OccupancyConfig
from .grid import OccupancyGrid
from .polar import PolarOccupancy


def _has_bbox_columns(df: pd.DataFrame, cfg: OccupancyConfig) -> bool:
    return all(c in df.columns for c in (cfg.col_w, cfg.col_l, cfg.col_yaw))


def _get_object_dims(
    row, cfg: OccupancyConfig, use_bbox: bool,
) -> tuple[float, float, float] | None:
    """Return (w, l, yaw) for an object, or None to fall back to circle."""
    if use_bbox:
        w = float(row[cfg.col_w])
        l = float(row[cfg.col_l])
        yaw = float(row[cfg.col_yaw])
        if not (np.isnan(w) or np.isnan(l) or np.isnan(yaw)):
            return w, l, yaw

    label = str(row.get("label", ""))
    if label in cfg.default_bbox_sizes:
        dims = cfg.default_bbox_sizes[label]
        l_def, w_def = float(dims[0]), float(dims[1])
        yaw = float(row[cfg.col_yaw]) if use_bbox and cfg.col_yaw in row.index and not np.isnan(row[cfg.col_yaw]) else 0.0
        return w_def, l_def, yaw

    return None


def bboxes_to_grid(
    objects: pd.DataFrame, cfg: OccupancyConfig
) -> OccupancyGrid:
    """Rasterise object bboxes (or circles) onto an occupancy grid."""
    grid = OccupancyGrid(cfg.resolution_m, cfg.range_m)
    if objects.empty:
        return grid

    use_bbox = _has_bbox_columns(objects, cfg)

    for _, row in objects.iterrows():
        cx = float(row[cfg.col_x])
        cy = float(row[cfg.col_y])

        if abs(cx) > cfg.range_m or abs(cy) > cfg.range_m:
            continue

        dims = _get_object_dims(row, cfg, use_bbox)
        if dims is not None:
            w, l, yaw = dims
            grid.fill_rotated_box(cx, cy, w, l, yaw)
        else:
            grid.fill_circle(cx, cy, cfg.point_radius_m)

    return grid


def bboxes_to_polar(
    objects: pd.DataFrame, cfg: OccupancyConfig
) -> PolarOccupancy:
    """Build polar occupancy from object positions / bboxes."""
    polar = PolarOccupancy(cfg.n_rays, cfg.polar_max_range_m)
    if objects.empty:
        return polar

    use_bbox = _has_bbox_columns(objects, cfg)

    for _, row in objects.iterrows():
        cx = float(row[cfg.col_x])
        cy = float(row[cfg.col_y])
        dist = np.sqrt(cx ** 2 + cy ** 2)

        if dist > cfg.polar_max_range_m:
            continue

        dims = _get_object_dims(row, cfg, use_bbox)
        if dims is not None:
            w, l, yaw = dims
            _add_bbox_to_polar(polar, cx, cy, w, l, yaw)
        else:
            _add_point_to_polar(polar, cx, cy, cfg.point_radius_m)

    return polar


def _add_point_to_polar(
    polar: PolarOccupancy, cx: float, cy: float, radius: float
) -> None:
    dist = np.sqrt(cx ** 2 + cy ** 2)
    angle = np.arctan2(cy, cx)
    half_angle = np.arctan2(radius, max(dist, 0.1))

    ray_start = int((angle - half_angle) / (2 * np.pi) * polar.n_rays) % polar.n_rays
    ray_end = int((angle + half_angle) / (2 * np.pi) * polar.n_rays) % polar.n_rays

    r_near = max(0.0, dist - radius)
    r_far = dist + radius

    if ray_start <= ray_end:
        rays = range(ray_start, ray_end + 1)
    else:
        rays = list(range(ray_start, polar.n_rays)) + list(range(0, ray_end + 1))

    for ri in rays:
        polar.add_interval(ri, r_near, r_far)


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
