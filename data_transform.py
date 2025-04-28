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


def load_bmp_as_rgba(filepath: str) -> np.ndarray:
    """Loads a 24-bit or 32-bit uncompressed BMP file and returns an RGBA image.

    Ensures shape is (H, W, 4) with alpha channel set to 255 if not present.

    Args:
        filepath: Path to the BMP image.

    Returns:
        NumPy array of shape (height, width, 4) with dtype=np.uint8
    """
    with open(filepath, "rb") as f:
        f.seek(10)
        pixel_data_offset = int.from_bytes(f.read(4), byteorder="little")

        f.seek(18)
        width = int.from_bytes(f.read(4), byteorder="little")
        height = int.from_bytes(f.read(4), byteorder="little")
        f.seek(28)
        bits_per_pixel = int.from_bytes(f.read(2), byteorder="little")
        channels = bits_per_pixel // 8

        if bits_per_pixel not in (24, 32):
            raise ValueError(f"Unsupported BMP bit depth: {bits_per_pixel}")

        row_raw = width * channels
        row_padded = ((row_raw + 3) // 4) * 4
        padding = row_padded - row_raw

        f.seek(pixel_data_offset)
        rows = []
        for _ in range(height):
            row = f.read(row_raw)
            f.read(padding)  # skip padding
            row_array = np.frombuffer(row, dtype=np.uint8).reshape((width, channels))
            rows.append(row_array)

        img = np.stack(rows[::-1], axis=0)  # bottom-up to top-down

        if channels == 3:
            alpha = np.full((height, width, 1), 255, dtype=np.uint8)
            img = np.concatenate((img, alpha), axis=-1)

        return img


def process_bmp(input_dir: str, output_dir: str, time_point: int | None = None) -> None:
    """Convert 24-bit or 32-bit BMP images to PNG with alpha channel.

    Raw data comes as BMP images with either 3 or 4 channels. We ensure all saved
    images have 4 channels (BGRA), with alpha set to 255 if missing.

    Images are captured in pairs: red light (bottom) and white light (top).
    This function separates them by index and saves both RGB and grayscale versions.

    Args:
        input_dir: Directory containing BMP images.
        output_dir: Directory to save PNG images.
        time_point: Optional time point label to append to filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    bmp_paths = sorted(glob.glob(f"{input_dir}/*.bmp"))

    for i, f in enumerate(tqdm(bmp_paths)):
        barcode = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        if time_point is not None:
            if time_point < 0:
                raise ValueError("Time point should be a non-negative integer")
            barcode += f"_d{time_point}"

        # Prepare subdirectories
        for subdir in ("white_rgb", "white_grayscale", "red_rgb", "red_grayscale"):
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        image = load_bmp_as_rgba(f)
        if i % 2:  # white light
            cv.imwrite(os.path.join(output_dir, "white_rgb", f"{barcode}.png"), image)
            cv.imwrite(
                os.path.join(output_dir, "white_grayscale", f"{barcode}.png"),
                cv.cvtColor(image, cv.COLOR_BGRA2GRAY),
            )
        else:  # red light
            cv.imwrite(os.path.join(output_dir, "red_rgb", f"{barcode}.png"), image)
            cv.imwrite(
                os.path.join(output_dir, "red_grayscale", f"{barcode}.png"),
                cv.cvtColor(image, cv.COLOR_BGRA2GRAY),
            )


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
