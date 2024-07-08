#!/usr/bin/env python3

import argparse
import os
import glob
import json

import polars as pl


def coordinate_correction(input_dir, parameter_path: str):
    with open(parameter_path, "r") as f:
        config = json.load(f)
    config_x = config["x"]
    config_y = config["y"]
    for i in sorted(glob.glob(os.path.join(input_dir, "*_picking.csv"))):
        try:
            coor = pl.read_csv(i, has_header=False, new_columns=["x", "y"])
        except pl.exceptions.NoDataError:
            continue
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
        ).with_columns(
            pl.col("x").round().cast(pl.Int32), pl.col("y").round().cast(pl.Int32)
        )
        barcode = os.path.splitext(os.path.basename(i))[0].split("_")[0]
        out_name = os.path.join(os.path.dirname(i), f"{barcode}_Coordinates.csv")
        coor.write_csv(out_name, include_header=False)


if __name__ == "__main__":
    # coordinate_correction(
    #     "test_data/output_03",
    #     "test_data/parameters/correction_params.json",
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="path to the input dir")
    parser.add_argument(
        "-p", "--parameter_path", type=str, help="path to the parameter file"
    )
    args = parser.parse_args()
    coordinate_correction(args.input_dir, args.parameter_path)
