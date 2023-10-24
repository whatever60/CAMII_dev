#!/usr/bin/env python

import argparse
import imaging_picking_function as ipf
import time
import os
import sys
import threading


def main():
    # configure_path = sys.argv[1]
    # input_dir = sys.argv[2]
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Configure file of parameters for general colony segmentation and filtering",
    )
    parser.add_argument("-i", "--input", type=str, help="Input calibration plates")
    args = parser.parse_args()
    configure_path = args.config
    input_dir = args.input

    configure_pool = ipf.readConfigureFile(configure_path)
    ipf.modifyOSconfigure(configure_pool)
    total_image, image_label_list, image_trans_list, image_epi_list = ipf.readFileList(
        input_dir
    )

    ipf.calculate_calib_image(image_trans_list, image_epi_list, configure_pool)


if __name__ == "__main__":
    main()
