#!/usr/bin/env python3

import argparse
import imaging_picking_function as ipf
import time
import os
import sys
import pickle
import threading
import pandas as pd


def main():
    # configure_path = sys.argv[1]
    # input_dir = sys.argv[2]
    # output_dir = sys.argv[3]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Configure file of parameters for general colony segmentation and filtering",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The picking coordinates output by the last step.",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Corrected picking coordinates."
    )
    args = parser.parse_args()
    configure_path = args.config
    input_dir = args.input
    output_dir = args.output

    configure_pool = ipf.readConfigureFile(configure_path)
    ipf.modifyOSconfigure(configure_pool)
    totalSamples = []
    for each in os.listdir(input_dir):
        if "_Coordinates.csv" in each:
            tmp = each.split("_Coordinates.csv")
            totalSamples.append(tmp[0])
    if not os.path.isdir(output_dir):
        os.system("mkdir -p " + output_dir)
    for eachSample in totalSamples:
        os.system(
            "Rscript "
            + configure_pool["house_bin"]
            + "/correction.R "
            + input_dir
            + "/"
            + eachSample
            + "_Coordinates.csv "
            + output_dir
            + "/"
            + eachSample
            + "_Coordinates.csv "
            + configure_pool["parameters_dir"]
        )


if __name__ == "__main__":
    main()
