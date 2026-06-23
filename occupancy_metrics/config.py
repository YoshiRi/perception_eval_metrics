"""Configuration for occupancy-centric perception evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class OccupancyConfig:
    # ── Required ──────────────────────────────────────────────────────────
    csv_path: str
    output_dir: str

    # ── Run identity ──────────────────────────────────────────────────────
    label: str = "occupancy_eval"

    # ── Grid settings ─────────────────────────────────────────────────────
    resolution_m: float = 0.5
    range_m: float = 80.0

    # ── Evaluation scope ──────────────────────────────────────────────────
    classes: List[str] = field(
        default_factory=lambda: ["car", "truck", "bus", "pedestrian"]
    )
    visibility_exclude: List[str] = field(default_factory=lambda: ["NONE"])

    # ── Distance weighting ────────────────────────────────────────────────
    # "inverse_square": w = 1/r², "uniform": w = 1
    distance_weight_type: str = "inverse_square"

    # ── Polar / Stixel settings ───────────────────────────────────────────
    n_rays: int = 360
    polar_max_range_m: float = 80.0

    # ── CSV column mapping ────────────────────────────────────────────────
    col_x: str = "x"
    col_y: str = "y"
    col_w: str = "w"
    col_l: str = "l"
    col_yaw: str = "yaw"

    # ── Required bbox columns ────────────────────────────────────────────────
    # The CSV must contain w, l, yaw columns (or the names set above).
    # If they are missing, bboxes_to_grid / bboxes_to_polar will raise.

    # ── Serialisation ─────────────────────────────────────────────────────

    @property
    def grid_size(self) -> int:
        return int(2 * self.range_m / self.resolution_m)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OccupancyConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, allow_unicode=True, sort_keys=False)

    # ── CLI override ──────────────────────────────────────────────────────

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument("--config", help="Path to YAML config file")
        parser.add_argument("--csv", dest="csv_path", help="Override csv_path")
        parser.add_argument("--output-dir", help="Override output_dir")
        parser.add_argument("--label", help="Override label")
        parser.add_argument(
            "--classes",
            help="Comma-separated class list",
        )
        parser.add_argument(
            "--resolution", dest="resolution_m", type=float,
            help="Grid resolution in metres (default: 0.5)",
        )
        parser.add_argument(
            "--range", dest="range_m", type=float,
            help="Grid range in metres (default: 80.0)",
        )
        parser.add_argument(
            "--n-rays", dest="n_rays", type=int,
            help="Number of polar rays (default: 360)",
        )
        parser.add_argument(
            "--vis-exclude",
            help="Comma-separated visibility values to exclude",
        )

    @classmethod
    def from_args(cls, args) -> "OccupancyConfig":
        if args.config:
            cfg = cls.from_yaml(args.config)
        else:
            if not args.csv_path or not args.output_dir:
                raise ValueError(
                    "Either --config or both --csv and --output-dir are required"
                )
            cfg = cls(csv_path=args.csv_path, output_dir=args.output_dir)

        if args.csv_path:
            cfg.csv_path = args.csv_path
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.label:
            cfg.label = args.label
        if args.classes:
            cfg.classes = [c.strip() for c in args.classes.split(",")]
        if args.resolution_m is not None:
            cfg.resolution_m = args.resolution_m
        if args.range_m is not None:
            cfg.range_m = args.range_m
        if args.n_rays is not None:
            cfg.n_rays = args.n_rays
        if args.vis_exclude:
            cfg.visibility_exclude = [
                v.strip() for v in args.vis_exclude.split(",")
            ]
        return cfg
