#!/usr/bin/env python
import os
import json

import polars as pl


def coordinate_correction(input_dir, parameter_path: str):
    with open(parameter_path, "r") as f:
        config = json.load(f)
    config_x = config["x"]
    config_y = config["y"]
    for i in sorted(i for i in os.listdir(input_dir) if i.endswith("_picking.csv")):
        input_path = os.path.join(input_dir, i)
        coor = pl.read_csv(input_path, has_header=False, new_columns=["x", "y"])
        coor = coor.with_columns(
            x=pl.col("x")
            - (
                config_x["alpha_x2"] * pl.col("x") ** 2
                + config_x["alpha_x"] * pl.col("x")
                + config_x["alpha_y"] * pl.col("y")
                + config_x["beta"]
            ),
            y=pl.col("y")
            - (
                config_y["alpha_y2"] * pl.col("y") ** 2
                + config_y["alpha_y"] * pl.col("y")
                + config_y["alpha_x"] * pl.col("x")
                + config_y["beta"]
            ),
        ).with_columns(pl.col("x").round().cast(pl.Int32), pl.col("y").round().cast(pl.Int32))
        coor.write_csv(
            os.path.splitext(input_path)[0] + "_corrected.csv", has_header=False
        )


if __name__ == "__main__":
    coordinate_correction(
        "test_data/output_03",
        "test_data/parameters/correction_params.json",
    )
