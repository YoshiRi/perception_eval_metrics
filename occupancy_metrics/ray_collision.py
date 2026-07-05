"""Forward-corridor, Ray-based minimum-collision-distance TP/FP/FN evaluation.

Rather than a fixed *angular* forward window (which widens without bound at
long range — +/-60 deg covers +/-69m laterally at 40m out), the forward zone
here is a constant-width *corridor*: a ray's occupier is in-scope only if its
lateral offset from the ego's centerline (r * sin(theta)) is within a
per-class half-width, and it is ahead of the ego (r * cos(theta) > 0). This
keeps the evaluated zone lane-shaped at every range.

For each ray theta, the nearest occupied distance is compared between GT and
Pred, each tested against the corridor sized for *its own* class:

  - both out of corridor / empty                  -> excluded
  - GT in corridor, Pred not (or empty, or |diff| > thr) -> FN
  - Pred in corridor, GT not (or empty, or |diff| > thr)  -> FP
  - both in corridor and |r_gt - r_pred| <= thr           -> TP

When both sides are in-corridor but the distance differs by more than the
threshold, the ray contributes one FN (GT's obstacle missed at its true
distance) *and* one FP (Pred's obstacle at the wrong distance).

Each ray's status is labelled with the class of its nearest occupier (GT's
class for TP/FN, Pred's class for FP). A second, narrower and shorter-range
corridor ("critical") marks FN/FP records that fall directly in front of the
ego at close range — these are operationally unacceptable regardless of
class and are surfaced separately from the per-class precision/recall.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .polar import PolarOccupancy

UNKNOWN_LABEL = "__unknown__"


def _ray_center_angle(ray_idx: int, n_rays: int) -> float:
    """Angle (radians, wrapped to (-pi, pi]) at the center of a ray bucket."""
    angle = (ray_idx + 0.5) / n_rays * 2 * math.pi
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


def _corridor_half_width(
    label: Optional[str], widths: Dict[str, float], default_half_width_m: float
) -> float:
    if label is None:
        return default_half_width_m
    return widths.get(label, default_half_width_m)


def _in_corridor(distance: float, angle: float, half_width_m: float, max_range: float) -> bool:
    if distance >= max_range:
        return False
    longitudinal = distance * math.cos(angle)
    if longitudinal <= 0.0:
        return False
    lateral = distance * math.sin(angle)
    return abs(lateral) <= half_width_m


def compute_ray_collision_records(
    gt_polar: PolarOccupancy,
    pred_polar: PolarOccupancy,
    dist_threshold_m: float,
    corridor_half_width_m: Dict[str, float],
    corridor_default_half_width_m: float,
    critical_half_width_m: float,
    critical_range_m: float,
) -> List[Dict]:
    """Per forward-corridor-ray TP/FP/FN records. See module docstring for the logic.

    Each FN/FP record carries a `critical` bool: True if it also falls
    inside the narrow, near-range corridor (`critical_half_width_m`,
    `critical_range_m`) where a miss/false-alarm is unconditionally
    unacceptable. TP records have no `critical` key.
    """
    n_rays = gt_polar.n_rays
    max_range = gt_polar.max_range
    gt_dists = gt_polar.nearest_distances()
    pred_dists = pred_polar.nearest_distances()

    records: List[Dict] = []

    for ray_idx in range(n_rays):
        angle = _ray_center_angle(ray_idx, n_rays)

        gt_occ = gt_dists[ray_idx] < max_range
        pred_occ = pred_dists[ray_idx] < max_range
        gt_label = gt_polar.nearest_label(ray_idx) if gt_occ else None
        pred_label = pred_polar.nearest_label(ray_idx) if pred_occ else None

        gt_width = _corridor_half_width(gt_label, corridor_half_width_m, corridor_default_half_width_m)
        pred_width = _corridor_half_width(pred_label, corridor_half_width_m, corridor_default_half_width_m)

        gt_in = gt_occ and _in_corridor(float(gt_dists[ray_idx]), angle, gt_width, max_range)
        pred_in = pred_occ and _in_corridor(float(pred_dists[ray_idx]), angle, pred_width, max_range)

        if not gt_in and not pred_in:
            continue

        def _is_critical(distance: float) -> bool:
            return distance <= critical_range_m and _in_corridor(
                distance, angle, critical_half_width_m, max_range
            )

        if gt_in and pred_in:
            diff = float(abs(gt_dists[ray_idx] - pred_dists[ray_idx]))
            if diff <= dist_threshold_m:
                records.append(
                    {"ray": ray_idx, "status": "TP", "label": gt_label, "dist_error_m": diff}
                )
                continue
            records.append({
                "ray": ray_idx, "status": "FN", "label": gt_label,
                "dist_error_m": diff, "critical": _is_critical(float(gt_dists[ray_idx])),
            })
            records.append({
                "ray": ray_idx, "status": "FP", "label": pred_label,
                "dist_error_m": diff, "critical": _is_critical(float(pred_dists[ray_idx])),
            })
        elif gt_in:
            records.append({
                "ray": ray_idx, "status": "FN", "label": gt_label,
                "dist_error_m": None, "critical": _is_critical(float(gt_dists[ray_idx])),
            })
        else:
            records.append({
                "ray": ray_idx, "status": "FP", "label": pred_label,
                "dist_error_m": None, "critical": _is_critical(float(pred_dists[ray_idx])),
            })

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


def empty_critical_counts() -> Dict:
    return {"FN": 0, "FP": 0}


def count_critical(records: List[Dict]) -> Dict:
    """Raw critical-zone FN/FP counts for one timestamp's records."""
    counts = empty_critical_counts()
    for rec in records:
        if rec.get("critical"):
            counts[rec["status"]] += 1
    return counts


def merge_critical_counts(totals: Dict, counts: Dict) -> None:
    """In-place accumulate raw critical-zone FN/FP counts from `counts` into `totals`."""
    for k in ("FN", "FP"):
        totals[k] += counts[k]
