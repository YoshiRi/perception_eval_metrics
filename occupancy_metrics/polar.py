"""Polar / Stixel occupancy representation and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


class PolarOccupancy:
    """Per-ray occupied intervals in polar coordinates."""

    def __init__(self, n_rays: int, max_range: float):
        self.n_rays = n_rays
        self.max_range = max_range
        # intervals[i] = sorted list of (r_near, r_far)
        self.intervals: List[List[Tuple[float, float]]] = [
            [] for _ in range(n_rays)
        ]

    def add_interval(self, ray_idx: int, r_near: float, r_far: float) -> None:
        r_near = max(0.0, r_near)
        r_far = min(self.max_range, r_far)
        if r_near >= r_far:
            return
        self.intervals[ray_idx].append((r_near, r_far))

    def nearest_distance(self, ray_idx: int) -> float:
        """Nearest occupied distance along a ray. max_range if empty."""
        if not self.intervals[ray_idx]:
            return self.max_range
        return min(iv[0] for iv in self.intervals[ray_idx])

    def nearest_distances(self) -> np.ndarray:
        return np.array([self.nearest_distance(i) for i in range(self.n_rays)])

    def merged_intervals(self, ray_idx: int) -> List[Tuple[float, float]]:
        """Return merged (non-overlapping) intervals for a ray."""
        ivs = sorted(self.intervals[ray_idx])
        if not ivs:
            return []
        merged = [ivs[0]]
        for near, far in ivs[1:]:
            if near <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], far))
            else:
                merged.append((near, far))
        return merged


def compute_polar_risk(
    gt_polar: PolarOccupancy,
    pred_polar: PolarOccupancy,
) -> Dict:
    """
    Stixel / Polar metrics.

    Per ray θ:
      safety:       max(0, r_pred(θ) - r_gt(θ))  — obstacle seen farther than GT
      availability: max(0, r_gt(θ) - r_pred(θ))  — obstacle seen closer than GT
    """
    gt_dists = gt_polar.nearest_distances()
    pred_dists = pred_polar.nearest_distances()

    safety_per_ray = np.maximum(0.0, pred_dists - gt_dists)
    avail_per_ray = np.maximum(0.0, gt_dists - pred_dists)

    # Exclude rays where both have no obstacle
    max_r = gt_polar.max_range
    both_empty = (gt_dists >= max_r) & (pred_dists >= max_r)
    valid = ~both_empty

    return {
        "safety_per_ray": safety_per_ray,
        "availability_per_ray": avail_per_ray,
        "safety_mean": float(safety_per_ray[valid].mean()) if valid.any() else 0.0,
        "safety_max": float(safety_per_ray[valid].max()) if valid.any() else 0.0,
        "availability_mean": float(avail_per_ray[valid].mean()) if valid.any() else 0.0,
        "availability_max": float(avail_per_ray[valid].max()) if valid.any() else 0.0,
        "n_valid_rays": int(valid.sum()),
    }


def compute_polar_interval_risk(
    gt_polar: PolarOccupancy,
    pred_polar: PolarOccupancy,
) -> Dict:
    """
    Polar Occupancy Interval comparison.

    Per ray, compare merged occupied intervals to measure
    how much occupied space is missed (safety) or hallucinated (availability).
    """
    total_false_free = 0.0
    total_false_occupied = 0.0

    for i in range(gt_polar.n_rays):
        gt_merged = gt_polar.merged_intervals(i)
        pred_merged = pred_polar.merged_intervals(i)

        gt_set = _intervals_to_set(gt_merged)
        pred_set = _intervals_to_set(pred_merged)

        # false_free: GT occupied but pred free
        false_free = _interval_diff_length(gt_merged, pred_merged)
        # false_occupied: pred occupied but GT free
        false_occupied = _interval_diff_length(pred_merged, gt_merged)

        total_false_free += false_free
        total_false_occupied += false_occupied

    return {
        "interval_false_free_m": round(total_false_free, 4),
        "interval_false_occupied_m": round(total_false_occupied, 4),
    }


def _intervals_to_set(intervals):
    """Convert intervals to a sorted list for set operations."""
    return intervals


def _interval_diff_length(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> float:
    """Total length of A that is NOT covered by B."""
    if not a:
        return 0.0
    if not b:
        return sum(far - near for near, far in a)

    total = 0.0
    for a_near, a_far in a:
        uncovered = _subtract_intervals_from_segment(a_near, a_far, b)
        total += uncovered
    return total


def _subtract_intervals_from_segment(
    seg_near: float, seg_far: float,
    subtract: List[Tuple[float, float]],
) -> float:
    """Length of [seg_near, seg_far] not covered by any interval in subtract."""
    cursor = seg_near
    length = 0.0
    for s_near, s_far in sorted(subtract):
        if s_near > cursor:
            length += min(s_near, seg_far) - cursor
        cursor = max(cursor, s_far)
        if cursor >= seg_far:
            break
    if cursor < seg_far:
        length += seg_far - cursor
    return length
