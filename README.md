# perception-eval-metrics

NuScenes-inspired safety-critical detection metrics and GT-EST re-matching for [`perception_eval`](https://github.com/tier4/autoware_perception_evaluation) CSV output.

## What it does

Given a `current.csv` from `perception_eval`, this tool computes:

1. **Safety-critical recall** — per-class and per-distance-bin recall, filtered to detectable objects (visibility ≠ NONE) and safety-relevant classes (car / truck / bus / pedestrian)
2. **nuScenes-style mAP** — multi-threshold average precision (0.5 / 1 / 2 / 4 m) using ATE to re-judge TP/FP
3. **NDS_safety** — a NuScenes Detection Score variant weighted toward recall and positional accuracy
4. **GT-EST re-matching** — re-runs the bipartite assignment from raw positions with configurable thresholds, allowing what-if analysis without re-running `perception_eval`

See [design.md](design.md) for the full formula rationale and comparison with official NuScenes NDS.

## Quick start

```bash
git clone https://github.com/YoshiRi/perception_eval_metrics
cd perception_eval_metrics
pip install .   # or: pip install pandas scipy numpy pyyaml

# Safety metrics
python run_safety_metrics.py \
    --csv path/to/current.csv \
    --config configs/safety_default.yaml

# Re-match with custom threshold, then re-compute metrics
python run_rematch.py \
    --csv path/to/current.csv \
    --config configs/rematch_default.yaml
python run_safety_metrics.py \
    --csv output/rematch/default/rematched.csv \
    --config configs/safety_default.yaml \
    --label rematch_default

# Compare two runs side by side
python run_safety_metrics.py --compare \
    output/safety_metrics/default/summary.json \
    output/safety_metrics/near_strict/summary.json
```

## Input format

`current.csv` produced by `perception_eval`. Each row is one object at one timestamp.

| column | description |
|--------|-------------|
| `unix_time` | timestamp (μs) |
| `source` | `GT` or `EST` |
| `status` | `TP` / `FP` / `FN` |
| `label` | object class (car, truck, bus, pedestrian, …) |
| `x`, `y` | BEV position (m) |
| `confidence` | detection confidence (EST only) |
| `visibility` | `FULL` / `MOST` / `PARTIAL` / `NONE` (GT only) |
| `r` | distance bin string e.g. `"20-40"` |
| `x_error`, `y_error` | position error (TP only, m) |
| `yaw_error` | heading error (TP only, rad) |
| `speed_error` | speed error (TP only, m/s) |

## Configuration

All parameters are in YAML. The `csv_path` field can be set in the file or overridden via `--csv`:

```bash
python run_safety_metrics.py --config configs/safety_default.yaml --csv my_data/current.csv
```

Key parameters:

| parameter | default | description |
|-----------|---------|-------------|
| `classes` | car/truck/bus/pedestrian | Classes to evaluate |
| `dist_max_m` | 100 | Distance range upper bound (m) |
| `visibility_exclude` | `[NONE]` | GT visibility values excluded from denominator |
| `match_thresholds_m` | `[0.5,1,2,4]` | ATE thresholds for AP computation |
| `ate_norm_m` | 2.0 | ATE normalization for NDS score |

See [design.md](design.md) for the full parameter reference and NDS formula details.

## Dependencies

- Python ≥ 3.8
- pandas, numpy, scipy, pyyaml
