"""Occupancy grid and ego state data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EgoState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


class OccupancyGrid:
    """2D BEV occupancy grid centred on the ego vehicle."""

    def __init__(self, resolution: float, range_m: float):
        self.resolution = resolution
        self.range_m = range_m
        n = int(2 * range_m / resolution)
        self.data = np.zeros((n, n), dtype=bool)

    @property
    def shape(self):
        return self.data.shape

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x + self.range_m) / self.resolution)
        gy = int((y + self.range_m) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        x = (gx + 0.5) * self.resolution - self.range_m
        y = (gy + 0.5) * self.resolution - self.range_m
        return x, y

    def distance_map(self) -> np.ndarray:
        """Distance from ego (grid centre) to each cell centre."""
        n = self.data.shape[0]
        half = n / 2.0
        idx = np.arange(n) + 0.5
        gx, gy = np.meshgrid(idx, idx)
        return np.sqrt(
            ((gx - half) * self.resolution) ** 2
            + ((gy - half) * self.resolution) ** 2
        )

    def in_range_mask(self) -> np.ndarray:
        return self.distance_map() <= self.range_m

    def fill_rotated_box(
        self, cx: float, cy: float, w: float, l: float, yaw: float
    ) -> None:
        """Rasterise a rotated bbox onto the grid."""
        res = self.resolution
        n = self.data.shape[0]
        rm = self.range_m

        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        half_diag = np.sqrt(w * w + l * l) / 2.0

        # Axis-aligned bounding box in grid coords
        min_gx = max(0, int((cx - half_diag + rm) / res))
        max_gx = min(n - 1, int((cx + half_diag + rm) / res) + 1)
        min_gy = max(0, int((cy - half_diag + rm) / res))
        max_gy = min(n - 1, int((cy + half_diag + rm) / res) + 1)

        if min_gx > max_gx or min_gy > max_gy:
            return

        gx_range = np.arange(min_gx, max_gx + 1)
        gy_range = np.arange(min_gy, max_gy + 1)
        wx = (gx_range + 0.5) * res - rm
        wy = (gy_range + 0.5) * res - rm
        WX, WY = np.meshgrid(wx, wy)

        dx = WX - cx
        dy = WY - cy
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y

        inside = (np.abs(local_x) <= l / 2.0) & (np.abs(local_y) <= w / 2.0)
        self.data[min_gy : max_gy + 1, min_gx : max_gx + 1] |= inside

    def fill_circle(self, cx: float, cy: float, radius: float) -> None:
        """Rasterise a circle (fallback when bbox dims unavailable)."""
        res = self.resolution
        n = self.data.shape[0]
        rm = self.range_m

        min_gx = max(0, int((cx - radius + rm) / res))
        max_gx = min(n - 1, int((cx + radius + rm) / res) + 1)
        min_gy = max(0, int((cy - radius + rm) / res))
        max_gy = min(n - 1, int((cy + radius + rm) / res) + 1)

        if min_gx > max_gx or min_gy > max_gy:
            return

        gx_range = np.arange(min_gx, max_gx + 1)
        gy_range = np.arange(min_gy, max_gy + 1)
        wx = (gx_range + 0.5) * res - rm
        wy = (gy_range + 0.5) * res - rm
        WX, WY = np.meshgrid(wx, wy)

        dist_sq = (WX - cx) ** 2 + (WY - cy) ** 2
        inside = dist_sq <= radius ** 2
        self.data[min_gy : max_gy + 1, min_gx : max_gx + 1] |= inside
