"""Forward-facing, Ray-based minimum-collision-distance TP/FP/FN evaluation.

For each ray theta (restricted to a forward angular window around the ego's
+x axis), the nearest occupied distance is compared between GT and Pred:

  - both empty                                   -> excluded (nothing to evaluate)
  - GT occupied, Pred empty or |r_gt-r_pred| > thr -> FN
  - Pred occupied, GT empty or |r_gt-r_pred| > thr  -> FP
  - both occupied and |r_gt-r_pred| <= thr          -> TP

When both sides are occupied but the distance differs by more than the
threshold, the ray contributes one FN (GT's obstacle missed at its true
distance) *and* one FP (Pred's obstacle at the wrong distance) — mirroring
how an unmatched GT/EST pair is counted in object-level matching.

Each ray's status is labelled with the class of its nearest occupier (GT's
class for TP/FN, Pred's class for FP), which is how per-class breakdowns are
derived from a single combined (multi-class) ray sweep.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .polar import PolarOccupancy

UNKNOWN_LABEL = "__unknown__"


def forward_ray_mask(n_rays: int, forward_angle_deg: float) -> np.ndarray:
    """Boolean mask selecting rays within +/-forward_angle_deg/2 of the ego's +x axis."""
    half = np.deg2rad(forward_angle_deg) / 2.0
    ray_angles = (np.arange(n_rays) + 0.5) / n_rays * 2 * np.pi
    ray_angles = np.where(ray_angles > np.pi, ray_angles - 2 * np.pi, ray_angles)
    return np.abs(ray_angles) <= half


def compute_ray_collision_records(
    gt_polar: PolarOccupancy,
    pred_polar: PolarOccupancy,
    dist_threshold_m: float,
    forward_angle_deg: float = 120.0,
) -> List[Dict]:
    """Per forward-ray TP/FP/FN records. See module docstring for the logic."""
    n_rays = gt_polar.n_rays
    mask = forward_ray_mask(n_rays, forward_angle_deg)

    gt_dists = gt_polar.nearest_distances()
    pred_dists = pred_polar.nearest_distances()

    max_r = gt_polar.max_range
    gt_occ = gt_dists < max_r
    pred_occ = pred_dists < max_r

    records: List[Dict] = []

    for ray_idx in np.nonzero(mask)[0]:
        g_occ = bool(gt_occ[ray_idx])
        p_occ = bool(pred_occ[ray_idx])
        if not g_occ and not p_occ:
            continue

        gt_label = gt_polar.nearest_label(int(ray_idx)) if g_occ else None
        pred_label = pred_polar.nearest_label(int(ray_idx)) if p_occ else None

        if g_occ and p_occ:
            diff = float(abs(gt_dists[ray_idx] - pred_dists[ray_idx]))
            if diff <= dist_threshold_m:
                records.append(
                    {"ray": int(ray_idx), "status": "TP", "label": gt_label, "dist_error_m": diff}
                )
                continue
            records.append(
                {"ray": int(ray_idx), "status": "FN", "label": gt_label, "dist_error_m": diff}
            )
            records.append(
                {"ray": int(ray_idx), "status": "FP", "label": pred_label, "dist_error_m": diff}
            )
        elif g_occ:
            records.append(
                {"ray": int(ray_idx), "status": "FN", "label": gt_label, "dist_error_m": None}
            )
        else:
            records.append(
                {"ray": int(ray_idx), "status": "FP", "label": pred_label, "dist_error_m": None}
            )

    return records


def empty_counts() -> Dict:
    return {"TP": 0, "FP": 0, "FN": 0}


def count_by_class(records: List[Dict], classes: List[str]) -> Dict[str, Dict]:
    """Raw per-class (+ '__unknown__') TP/FP/FN counts for one timestamp's records."""
    counts: Dict[str, Dict] = {cls: empty_counts() for cls in classes}
    counts[UNKNOWN_LABEL] = empty_counts()

    for rec in records:
        cls = rec["label"] if rec["label"] in counts else UNKNOWN_LABEL
        counts[cls][rec["status"]] += 1

    return counts


def merge_counts(totals: Dict[str, Dict], counts: Dict[str, Dict]) -> None:
    """In-place accumulate raw TP/FP/FN counts from `counts` into `totals`."""
    for cls, c in counts.items():
        if cls not in totals:
            totals[cls] = empty_counts()
        for k in ("TP", "FP", "FN"):
            totals[cls][k] += c[k]


def finalise_counts(totals: Dict[str, Dict]) -> Dict[str, Dict]:
    """Add precision/recall to accumulated raw TP/FP/FN counts."""
    out = {}
    for cls, counts in totals.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[cls] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }
    return out
