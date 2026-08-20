"""Render a clean square SLAM map/trajectory image from the saved source data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "cell035_slam_map_with_map_frame_trajectory_clean_600x600.png"
UNKNOWN_GRAY = 205 / 255
TRAJECTORY_COLOR = "#5185c0"
START_COLOR = "#55966b"
END_COLOR = "#c96144"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Cell035 map and map-frame trajectory without annotations."
    )
    parser.add_argument("--size", type=int, default=600, help="Square output size in pixels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be greater than zero")

    map_yaml = (ROOT / "cell035_slam_map.yaml").read_text(encoding="utf-8")
    resolution = float(re.search(r"resolution:\s*([0-9.eE+-]+)", map_yaml).group(1))
    origin_text = re.search(r"origin:\s*\[([^\]]+)\]", map_yaml).group(1)
    origin_x, origin_y = (float(value.strip()) for value in origin_text.split(",")[:2])

    map_array = np.asarray(Image.open(ROOT / "cell035_slam_map.pgm").convert("L"))
    map_height, map_width = map_array.shape

    xs: list[float] = []
    ys: list[float] = []
    with (ROOT / "map_base_link_trajectory.csv").open(
        "r", encoding="utf-8", newline=""
    ) as trajectory_file:
        for row in csv.DictReader(trajectory_file):
            if row.get("x") and row.get("y"):
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))

    if not xs:
        raise RuntimeError("No x/y samples found in map_base_link_trajectory.csv")

    trajectory_x = [(x - origin_x) / resolution for x in xs]
    trajectory_y = [map_height - (y - origin_y) / resolution for y in ys]

    dpi = 100
    figure = plt.figure(
        figsize=(args.size / dpi, args.size / dpi),
        dpi=dpi,
        facecolor=(UNKNOWN_GRAY,) * 3,
    )
    axes = figure.add_axes((0, 0, 1, 1), frameon=False)
    axes.set_facecolor((UNKNOWN_GRAY,) * 3)
    axes.imshow(
        map_array,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        interpolation="nearest",
    )
    axes.plot(trajectory_x, trajectory_y, color=TRAJECTORY_COLOR, linewidth=1.5)
    axes.scatter(
        trajectory_x[0],
        trajectory_y[0],
        color=START_COLOR,
        edgecolors="white",
        linewidths=1.0,
        marker="o",
        s=50,
        zorder=3,
    )
    axes.scatter(
        trajectory_x[-1],
        trajectory_y[-1],
        color=END_COLOR,
        linewidths=2.0,
        marker="x",
        s=70,
        zorder=3,
    )
    axes.set_xlim(-0.5, map_width - 0.5)
    axes.set_ylim(map_height - 0.5, -0.5)
    axes.set_aspect("equal", adjustable="box")
    axes.set_axis_off()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor=figure.get_facecolor(), pad_inches=0)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
