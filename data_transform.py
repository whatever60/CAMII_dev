#!/usr/bin/env python3
import os
import argparse
import glob
import sys

import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from hyperspectral_processing import process_bil2npz, process_npz2png


def process_bmp(input_dir: str, output_dir: str, time_point: int | None = None) -> None:
    """Raw data comes in 32-bit bmp format, with alpha channel all 255. Two images are
    taken for each plate, one with red light from bottom and one with white light from
    above along the top edge.

    We first save them as png files with better naming, then save a gray scale image
    from the red light one.

    Convert bmp images to png images.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, f in enumerate(tqdm(sorted(glob.glob(f"{input_dir}/*.bmp")))):
        barcode = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        if time_point is not None:
            if time_point < 0:
                raise ValueError("Time point should be a non-negative integer")
            barcode += f"_d{time_point}"
        width, height = np.fromfile(f, dtype=np.uint32, offset=18, count=2)
        image = np.fromfile(f, dtype=np.uint8, offset=54).reshape(height, width, 4)[
            ::-1
        ]
        os.makedirs(os.path.join(output_dir, "white_rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "white_grayscale"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "red_rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "red_grayscale"), exist_ok=True)
        if i % 2:  # white light
            cv.imwrite(os.path.join(output_dir, "white_rgb", f"{barcode}.png"), image)
            cv.imwrite(
                os.path.join(output_dir, "white_grayscale", f"{barcode}.png"),
                cv.cvtColor(image, cv.COLOR_BGR2GRAY),
            )
        else:  # red light
            cv.imwrite(os.path.join(output_dir, "red_rgb", f"{barcode}.png"), image)
            cv.imwrite(
                os.path.join(output_dir, "red_grayscale", f"{barcode}.png"),
                cv.cvtColor(image, cv.COLOR_BGR2GRAY),
            )
            # image_gray = cv.imread(
            #     os.path.join(output_dir, f"{barcode}_rgb_red.png"), cv.IMREAD_GRAYSCALE
            # )


def _add_pca_arguments(parser: argparse.ArgumentParser) -> None:
    """Add to a parser arguments related to PCA.

    PCA is performed by first cropping the image, and fitting PCA using a subset of all
        pixels in the cropped image, transforming using all pixels in the cropped
        image, and round to 0~255 for saving as png by clipping at given lower and
        upper quantiles.

    Both cropping and subsetting could be done using the first bounding box (or
        contour) given in a json file of coco format, or with rectangular coordinates.
        For cropping, it's with a top-left and bottom right xy coordinates (a tuple
            with 4 elements).
        For subsetting, it's with a zoom factor. For example, if zoom factor is k, the
            marginal 1 / k (from all 4 boundaries) of the image will be masked out and
            the rest will be used for PCA.
    """
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Whether to save the first 3 PCs as RGB image",
    )
    parser.add_argument(
        "-op",
        "--output_dir_pca",
        type=str,
        default=None,
        help="Output directory for PCA images",
    )
    # all the following will only be used if pca is True
    parser.add_argument(
        "-qp",
        "--quantile_pca",
        type=float,
        default=0.005,
        help="quantile for converting hyperspectral data to rgb",
    )
    parser.add_argument(
        "-ms",
        "--mask_subset",
        type=str,
        default=None,
        help="Mask file path, should be in coco json format",
    )
    parser.add_argument(
        "--zoom_f",
        type=int,
        default=6,
        help="Zoom factor for cropping the hyperspectral image",
    )


if __name__ == "__main__":
    # example usages:
    # 1. Convert one hyperspectral image from .bil to .npy
    # python3 bil2numpy.py bil2npy --metadata <hdr_path> bil2npy <bil_path> --output_dir <output_dir>
    # 2. Convert one hyperspectral npy file to png with RGB channels.

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description="Utility for data processing")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    bil2npz_parser = subparsers.add_parser("bil2npz", help="Convert BIL to NPZ")
    bil2npz_parser.add_argument(
        "--input_dir", type=str, help="directory containing .bil files"
    )
    bil2npz_parser.add_argument("-on", "--output_dir_npz", type=str, default=None)
    bil2npz_parser.add_argument("-or", "--output_dir_rgb", type=str, default=None)
    bil2npz_parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=["top", "bottom"],
        help="configuration string",
    )
    bil2npz_parser.add_argument(
        "-qr",
        "--quantile_rgb",
        type=float,
        default=0.999,
        help="quantile for converting hyperspectral data to rgb",
    )
    bil2npz_parser.add_argument(
        "-t",
        "--time_point",
        type=int,
        default=None,
        help="time point suffix to add to output file name",
    )
    _add_pca_arguments(bil2npz_parser)
    bil2npz_parser.add_argument(
        "-c",
        "--cropping",
        type=float,
        nargs=4,
        default=None,
        help="Cropping the hyperspectral image",
    )
    bil2npz_parser.add_argument(
        "-mc",
        "--mask_crop_dir",
        type=str,
        default=None,
        help="Mask file dir, should be in coco json format or yolo txt format",
    )

    npz2png_parser = subparsers.add_parser("npz2png", help="Convert NPZ to png")
    npz2png_parser.add_argument(
        "-i", "--input_dir", type=str, help="directory containing .npz files"
    )
    npz2png_parser.add_argument("-or", "--output_dir_rgb", type=str, default=None)
    npz2png_parser.add_argument(
        "-qr",
        "--quantile_rgb",
        type=float,
        default=0.999,
        help="quantile for converting hyperspectral data to rgb",
    )
    npz2png_parser.add_argument(
        "-c",
        "--cropping",
        type=float,
        nargs=4,
        default=None,
        help="Cropping the hyperspectral image",
    )
    npz2png_parser.add_argument(
        "-mc",
        "--mask_crop_dir",
        type=str,
        default=None,
        help="Mask file dir, should be in coco json format or yolo txt format",
    )
    # npz2png_parser.add_argument(
    #     "-t",
    #     "--time_point",
    #     type=int,
    #     default=None,
    #     help="time point suffix to add to output file name",
    # )
    _add_pca_arguments(npz2png_parser)

    process_bmp_parser = subparsers.add_parser("process_bmp", help="Process BMP files")
    process_bmp_parser.add_argument(
        "-i", "--input_dir", type=str, help="path to the BMP files"
    )
    process_bmp_parser.add_argument(
        "-o", "--output_dir", type=str, help="path to the output dir"
    )
    process_bmp_parser.add_argument(
        "-t",
        "--time_point",
        type=int,
        default=None,
        help="time point suffix to add to output file name",
    )

    args = parser.parse_args()

    if args.subcommand == "bil2npz":
        process_bil2npz(
            input_dir=args.input_dir,
            output_dir_npz=args.output_dir_npz,
            output_dir_rgb=args.output_dir_rgb,
            time_point=args.time_point,
            config=args.config,
            cropping=args.cropping,
            mask_crop_dir=args.mask_crop_dir,
            quantile_rgb=args.quantile_rgb,
            pca=args.pca,
            output_dir_pca=args.output_dir_pca,
            zoom_f=args.zoom_f,
            mask_subset=args.mask_subset,
            quantile_pca=args.quantile_pca,
        )
    elif args.subcommand == "npz2png":
        process_npz2png(
            input_dir=args.input_dir,
            output_dir_rgb=args.output_dir_rgb,
            time_point=None,
            cropping=args.cropping,
            mask_crop_dir=args.mask_crop_dir,
            quantile_rgb=args.quantile_rgb,
            pca=args.pca,
            output_dir_pca=args.output_dir_pca,
            zoom_f=args.zoom_f,
            mask_subset=args.mask_subset,
            quantile_pca=args.quantile_pca,
        )
    elif args.subcommand == "process_bmp":
        process_bmp(args.input_dir, args.output_dir, args.time_point)
    else:
        parser.print_help()
        sys.exit(1)
