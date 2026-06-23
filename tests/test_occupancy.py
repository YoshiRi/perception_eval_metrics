"""Tests for occupancy-centric evaluation (Phase 1)."""

import numpy as np
import pandas as pd
import pytest

from occupancy_metrics.grid import OccupancyGrid, EgoState
from occupancy_metrics.polar import PolarOccupancy, compute_polar_risk, compute_polar_interval_risk
from occupancy_metrics.config import OccupancyConfig
from occupancy_metrics.converters import bboxes_to_grid, bboxes_to_polar
from occupancy_metrics.compute import compute_occupancy_risk, evaluate_timestamp, run_all


class TestOccupancyGrid:
    def test_grid_shape(self):
        grid = OccupancyGrid(resolution=0.5, range_m=10.0)
        assert grid.shape == (40, 40)

    def test_world_to_grid_center(self):
        grid = OccupancyGrid(resolution=0.5, range_m=10.0)
        gx, gy = grid.world_to_grid(0.0, 0.0)
        assert gx == 20
        assert gy == 20

    def test_fill_circle(self):
        grid = OccupancyGrid(resolution=0.5, range_m=10.0)
        grid.fill_circle(0.0, 0.0, 1.0)
        assert grid.data.sum() > 0
        center_gx, center_gy = grid.world_to_grid(0.0, 0.0)
        assert grid.data[center_gy, center_gx]

    def test_fill_rotated_box_axis_aligned(self):
        grid = OccupancyGrid(resolution=0.5, range_m=10.0)
        grid.fill_rotated_box(5.0, 0.0, 2.0, 4.0, 0.0)
        assert grid.data.sum() > 0
        gx, gy = grid.world_to_grid(5.0, 0.0)
        assert grid.data[gy, gx]

    def test_fill_rotated_box_rotated(self):
        grid = OccupancyGrid(resolution=0.5, range_m=10.0)
        grid.fill_rotated_box(5.0, 0.0, 2.0, 4.0, np.pi / 4)
        assert grid.data.sum() > 0

    def test_fill_box_outside_range(self):
        grid = OccupancyGrid(resolution=0.5, range_m=10.0)
        grid.fill_rotated_box(50.0, 50.0, 2.0, 4.0, 0.0)
        assert grid.data.sum() == 0

    def test_distance_map(self):
        grid = OccupancyGrid(resolution=1.0, range_m=5.0)
        dm = grid.distance_map()
        assert dm.shape == (10, 10)
        assert dm[5, 5] < 1.0  # near center


class TestComputeOccupancyRisk:
    def test_perfect_match(self):
        gt = OccupancyGrid(resolution=0.5, range_m=10.0)
        pred = OccupancyGrid(resolution=0.5, range_m=10.0)
        gt.fill_circle(5.0, 0.0, 1.0)
        pred.fill_circle(5.0, 0.0, 1.0)

        result = compute_occupancy_risk(gt, pred)
        assert result["safety_risk_cells"] == 0
        assert result["availability_risk_cells"] == 0
        assert result["true_positive_cells"] > 0

    def test_total_miss(self):
        gt = OccupancyGrid(resolution=0.5, range_m=10.0)
        pred = OccupancyGrid(resolution=0.5, range_m=10.0)
        gt.fill_circle(5.0, 0.0, 1.0)
        # pred is empty

        result = compute_occupancy_risk(gt, pred)
        assert result["safety_risk_cells"] > 0
        assert result["availability_risk_cells"] == 0
        assert result["safety_miss_rate"] == 1.0

    def test_phantom_detection(self):
        gt = OccupancyGrid(resolution=0.5, range_m=10.0)
        pred = OccupancyGrid(resolution=0.5, range_m=10.0)
        # gt is empty
        pred.fill_circle(5.0, 0.0, 1.0)

        result = compute_occupancy_risk(gt, pred)
        assert result["safety_risk_cells"] == 0
        assert result["availability_risk_cells"] > 0

    def test_distance_weighting(self):
        gt = OccupancyGrid(resolution=0.5, range_m=20.0)
        pred_near = OccupancyGrid(resolution=0.5, range_m=20.0)
        pred_far = OccupancyGrid(resolution=0.5, range_m=20.0)

        # Miss near ego vs miss far from ego
        gt.fill_circle(3.0, 0.0, 1.0)
        near_result = compute_occupancy_risk(gt, pred_near, distance_weight_type="inverse_square")

        gt2 = OccupancyGrid(resolution=0.5, range_m=20.0)
        gt2.fill_circle(15.0, 0.0, 1.0)
        far_result = compute_occupancy_risk(gt2, pred_far, distance_weight_type="inverse_square")

        assert near_result["safety_risk_weighted"] > far_result["safety_risk_weighted"]


class TestPolarOccupancy:
    def test_nearest_distance_empty(self):
        polar = PolarOccupancy(n_rays=360, max_range=80.0)
        assert polar.nearest_distance(0) == 80.0

    def test_nearest_distance(self):
        polar = PolarOccupancy(n_rays=360, max_range=80.0)
        polar.add_interval(0, 5.0, 10.0)
        polar.add_interval(0, 3.0, 6.0)
        assert polar.nearest_distance(0) == 3.0

    def test_merged_intervals(self):
        polar = PolarOccupancy(n_rays=360, max_range=80.0)
        polar.add_interval(0, 5.0, 10.0)
        polar.add_interval(0, 8.0, 15.0)
        merged = polar.merged_intervals(0)
        assert len(merged) == 1
        assert merged[0] == (5.0, 15.0)

    def test_separate_intervals(self):
        polar = PolarOccupancy(n_rays=360, max_range=80.0)
        polar.add_interval(0, 5.0, 7.0)
        polar.add_interval(0, 10.0, 15.0)
        merged = polar.merged_intervals(0)
        assert len(merged) == 2


class TestPolarRisk:
    def test_perfect_match(self):
        gt = PolarOccupancy(n_rays=4, max_range=80.0)
        pred = PolarOccupancy(n_rays=4, max_range=80.0)
        gt.add_interval(0, 5.0, 10.0)
        pred.add_interval(0, 5.0, 10.0)

        result = compute_polar_risk(gt, pred)
        assert result["safety_mean"] == 0.0
        assert result["availability_mean"] == 0.0

    def test_obstacle_seen_farther(self):
        gt = PolarOccupancy(n_rays=4, max_range=80.0)
        pred = PolarOccupancy(n_rays=4, max_range=80.0)
        gt.add_interval(0, 5.0, 10.0)
        pred.add_interval(0, 8.0, 12.0)

        result = compute_polar_risk(gt, pred)
        assert result["safety_mean"] > 0  # pred sees farther → safety risk

    def test_obstacle_seen_closer(self):
        gt = PolarOccupancy(n_rays=4, max_range=80.0)
        pred = PolarOccupancy(n_rays=4, max_range=80.0)
        gt.add_interval(0, 8.0, 12.0)
        pred.add_interval(0, 5.0, 10.0)

        result = compute_polar_risk(gt, pred)
        assert result["availability_mean"] > 0  # pred sees closer → avail risk


class TestPolarIntervalRisk:
    def test_perfect_match(self):
        gt = PolarOccupancy(n_rays=4, max_range=80.0)
        pred = PolarOccupancy(n_rays=4, max_range=80.0)
        gt.add_interval(0, 5.0, 10.0)
        pred.add_interval(0, 5.0, 10.0)

        result = compute_polar_interval_risk(gt, pred)
        assert result["interval_false_free_m"] == 0.0
        assert result["interval_false_occupied_m"] == 0.0

    def test_partial_miss(self):
        gt = PolarOccupancy(n_rays=4, max_range=80.0)
        pred = PolarOccupancy(n_rays=4, max_range=80.0)
        gt.add_interval(0, 5.0, 10.0)
        pred.add_interval(0, 6.0, 9.0)

        result = compute_polar_interval_risk(gt, pred)
        assert result["interval_false_free_m"] == pytest.approx(2.0, abs=0.01)
        assert result["interval_false_occupied_m"] == 0.0


class TestConverters:
    def _make_objects(self, rows):
        return pd.DataFrame(rows)

    def test_bbox_to_grid(self):
        cfg = OccupancyConfig(csv_path="", output_dir="", resolution_m=0.5, range_m=10.0)
        objs = self._make_objects([
            {"x": 5.0, "y": 0.0, "w": 2.0, "l": 4.0, "yaw": 0.0},
        ])
        grid = bboxes_to_grid(objs, cfg)
        assert grid.data.sum() > 0

    def test_missing_bbox_columns_raises(self):
        cfg = OccupancyConfig(csv_path="", output_dir="", resolution_m=0.5, range_m=10.0)
        objs = self._make_objects([
            {"x": 5.0, "y": 0.0},
        ])
        with pytest.raises(ValueError, match="missing required bbox columns"):
            bboxes_to_grid(objs, cfg)

    def test_nan_bbox_raises(self):
        cfg = OccupancyConfig(csv_path="", output_dir="", resolution_m=0.5, range_m=10.0)
        objs = self._make_objects([
            {"x": 5.0, "y": 0.0, "w": np.nan, "l": 4.0, "yaw": 0.0},
        ])
        with pytest.raises(ValueError, match="NaN bbox dimensions"):
            bboxes_to_grid(objs, cfg)

    def test_bbox_to_polar(self):
        cfg = OccupancyConfig(csv_path="", output_dir="", resolution_m=0.5, range_m=10.0, n_rays=36)
        objs = self._make_objects([
            {"x": 5.0, "y": 0.0, "w": 2.0, "l": 4.0, "yaw": 0.0},
        ])
        polar = bboxes_to_polar(objs, cfg)
        dists = polar.nearest_distances()
        assert dists.min() < 10.0

    def test_polar_missing_bbox_columns_raises(self):
        cfg = OccupancyConfig(csv_path="", output_dir="", resolution_m=0.5, range_m=10.0, n_rays=36)
        objs = self._make_objects([
            {"x": 5.0, "y": 0.0},
        ])
        with pytest.raises(ValueError, match="missing required bbox columns"):
            bboxes_to_polar(objs, cfg)


class TestVisibilityOrphan:
    """Verify that EST TPs paired with visibility-excluded GTs
    are not incorrectly counted as availability risk."""

    _bbox = {"w": 2.0, "l": 4.0, "yaw": 0.0}

    def test_excluded_gt_tp_not_counted_as_false_occupied(self):
        rows = [
            {"unix_time": 1000, "source": "GT", "status": "TP", "label": "car",
             "x": 10.0, "y": 0.0, "confidence": np.nan, "visibility": "NONE",
             **self._bbox},
            {"unix_time": 1000, "source": "EST", "status": "TP", "label": "car",
             "x": 10.0, "y": 0.0, "confidence": 0.9, "visibility": np.nan,
             **self._bbox},
        ]
        df = pd.DataFrame(rows)
        cfg = OccupancyConfig(
            csv_path="", output_dir="",
            resolution_m=0.5, range_m=20.0, classes=["car"],
        )
        result = evaluate_timestamp(df, cfg)
        assert result["grid"]["availability_risk_cells"] == 0
        assert result["grid"]["safety_risk_cells"] == 0

    def test_visible_gt_fn_still_counted_as_safety_risk(self):
        rows = [
            {"unix_time": 1000, "source": "GT", "status": "FN", "label": "car",
             "x": 10.0, "y": 0.0, "confidence": np.nan, "visibility": "FULL",
             **self._bbox},
        ]
        df = pd.DataFrame(rows)
        cfg = OccupancyConfig(
            csv_path="", output_dir="",
            resolution_m=0.5, range_m=20.0, classes=["car"],
        )
        result = evaluate_timestamp(df, cfg)
        assert result["grid"]["safety_risk_cells"] > 0
        assert result["grid"]["availability_risk_cells"] == 0

    def test_mixed_visibility_correct_risk(self):
        rows = [
            {"unix_time": 1000, "source": "GT", "status": "TP", "label": "car",
             "x": 10.0, "y": 0.0, "confidence": np.nan, "visibility": "NONE",
             **self._bbox},
            {"unix_time": 1000, "source": "GT", "status": "FN", "label": "car",
             "x": 20.0, "y": 0.0, "confidence": np.nan, "visibility": "FULL",
             **self._bbox},
            {"unix_time": 1000, "source": "EST", "status": "TP", "label": "car",
             "x": 10.0, "y": 0.0, "confidence": 0.9, "visibility": np.nan,
             **self._bbox},
        ]
        df = pd.DataFrame(rows)
        cfg = OccupancyConfig(
            csv_path="", output_dir="",
            resolution_m=0.5, range_m=30.0, classes=["car"],
        )
        result = evaluate_timestamp(df, cfg)
        assert result["grid"]["safety_risk_cells"] > 0
        assert result["grid"]["availability_risk_cells"] == 0


class TestDistanceMapCache:
    """Verify that pre-computed distance_map produces identical results."""

    def test_cached_matches_computed(self):
        from occupancy_metrics.grid import OccupancyGrid
        gt = OccupancyGrid(resolution=0.5, range_m=10.0)
        pred = OccupancyGrid(resolution=0.5, range_m=10.0)
        gt.fill_circle(5.0, 0.0, 1.0)

        result_no_cache = compute_occupancy_risk(gt, pred)
        cached_dm = gt.distance_map()
        result_cached = compute_occupancy_risk(gt, pred, distance_map=cached_dm)

        assert result_no_cache == result_cached


class TestEndToEnd:
    def _make_csv_data(self):
        rows = []
        # GT objects
        for i in range(5):
            rows.append({
                "unix_time": 1000,
                "source": "GT",
                "status": "TP" if i < 4 else "FN",
                "label": "car",
                "x": 10.0 + i * 5,
                "y": 0.0,
                "w": 4.0,
                "l": 2.0,
                "yaw": 0.0,
                "confidence": np.nan,
                "visibility": "FULL",
                "r": "0-20" if i < 2 else "20-40",
                "x_error": 0.1 if i < 4 else np.nan,
                "y_error": 0.1 if i < 4 else np.nan,
                "yaw_error": 0.01 if i < 4 else np.nan,
                "speed_error": 0.5 if i < 4 else np.nan,
            })
        # EST objects (match 4 of 5 GTs)
        for i in range(4):
            rows.append({
                "unix_time": 1000,
                "source": "EST",
                "status": "TP",
                "label": "car",
                "x": 10.1 + i * 5,
                "y": 0.1,
                "w": 4.0,
                "l": 2.0,
                "yaw": 0.0,
                "confidence": 0.9,
                "visibility": np.nan,
                "r": "0-20" if i < 2 else "20-40",
                "x_error": 0.1,
                "y_error": 0.1,
                "yaw_error": 0.01,
                "speed_error": 0.5,
            })
        return pd.DataFrame(rows)

    def test_evaluate_timestamp(self):
        df = self._make_csv_data()
        cfg = OccupancyConfig(
            csv_path="", output_dir="",
            resolution_m=0.5, range_m=40.0,
            classes=["car"],
        )
        result = evaluate_timestamp(df, cfg)
        assert result["n_gt_objects"] == 5
        assert result["n_est_objects"] == 4
        assert result["grid"]["safety_risk_cells"] > 0  # one GT missed
        assert result["grid"]["true_positive_cells"] > 0

    def test_run_all(self):
        df = self._make_csv_data()
        cfg = OccupancyConfig(
            csv_path="", output_dir="",
            resolution_m=0.5, range_m=40.0,
            classes=["car"],
        )
        result = run_all(df, cfg)
        assert result["n_timestamps"] == 1
        assert result["grid"]["safety_risk_cells"] > 0
        assert "car" in result["per_class"]
