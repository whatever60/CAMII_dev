#!/usr/bin/env python3
import struct
import json
import ast
import os
import argparse
import glob
import sys

import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import yaml


# R_WL, G_WL, B_WL = 700, 550, 450
R_WL_LOW, R_WL_HIGH, G_WL_LOW, G_WL_HIGH, B_WL_LOW, B_WL_HIGH = (
    680,
    720,
    530,
    570,
    430,
    470,
)


def parse_metadata(metadata_path: str) -> dict:
    metadata = {}
    with open(metadata_path, "r") as f:
        # the lines in the metadata file looks like key = value
        for line in f:
            line = line.strip()
            if line:
                try:
                    key, value = line.split("=")
                except ValueError:  # not enough values to unpack
                    pass
                else:
                    key = key.strip()
                    try:
                        value = ast.literal_eval(value.strip())
                    except (ValueError, SyntaxError):  # not a valid python literal
                        value = value.strip()  # string
                    metadata[key] = value
                    if key == "wavelength":
                        metadata[key] = list(value)
                    elif key == "rotation":
                        metadata[key] = list(map(list, value))
    return metadata


def bil2np(bil_path: str, samples: int, lines: int, channels: int) -> np.ndarray:
    arr = np.fromfile(bil_path, dtype=np.uint16)
    assert arr.shape[0] == samples * lines * channels, "Invalid data shape"
    return arr.reshape(lines, channels, samples).transpose(2, 0, 1)


def np2png(
    arr: np.ndarray, wls: list, ceiling: int, quantile: float | None = 0.999
) -> Image:
    if not arr.shape[2] == len(wls):
        raise ValueError(
            "Number of channels in data and number of wavelengths do not match"
        )
    wls = np.array(wls)
    image_data_r = arr[:, :, wls.searchsorted(R_WL_LOW) : wls.searchsorted(R_WL_HIGH)]
    image_data_g = arr[:, :, wls.searchsorted(G_WL_LOW) : wls.searchsorted(G_WL_HIGH)]
    image_data_b = arr[:, :, wls.searchsorted(B_WL_LOW) : wls.searchsorted(B_WL_HIGH)]
    # image = Image.fromarray((image_data / (ceiling / 255)).astype(np.uint8))
    image_data = np.stack([image_data_r, image_data_g, image_data_b], axis=-1)
    if quantile is not None:
        ceiling = np.quantile(image_data, quantile)
    image_data = np.clip(image_data.mean(-2), a_min=0, a_max=ceiling) / ceiling * 255
    image = Image.fromarray(image_data.astype(np.uint8))
    return image


def bil2np_old(
    bil_path: str,
    sample_diff: int,
    line_diff: int,
    top_left_line: int,
    top_left_sample: int,
    max_samples: int,
    max_bands: int,
) -> np.ndarray:
    cube = np.zeros((sample_diff, line_diff, 462), dtype=np.uint16)
    with open(bil_path, mode="rb") as file:
        for i in trange(sample_diff):
            for j in trange(line_diff):
                file.seek(
                    ((top_left_line + j) * (2 * max_samples * max_bands))
                    + 2 * (top_left_sample + i)
                )  # set cursor to beginning of first value
                for k in range(max_bands):
                    data = file.read(2)
                    a = struct.unpack("H", data)
                    cube[i][j][k] = a[0]
                    file.seek(2 * max_samples - 2, 1)
                # sys.stdout.write(
                #     "\r SAMPLE: %d LINE: %d Progress: %.2f%% Time Elapsed: %f "
                #     % (
                #         i,
                #         j,
                #         (((i * line_diff) + j) / TOTAL) * 100,
                #         time.time() - start_time,
                #     )
                # )
                # sys.stdout.flush()
    return cube


def process_bmp(input_dir: str, output_dir: str) -> None:
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


def hsi_pca(
    arr: np.ndarray, mask: np.ndarray = None, quantile: float = 0.005
) -> np.ndarray:
    arr_flat = arr.reshape(-1, arr.shape[-1])
    if mask is None:
        # default mask is masking the peripheral regions of the array, only leaving the center 1/2
        # mask = np.zeros(arr.shape[:-1])
        # mask[
        #     arr.shape[0] // zoom_f : (zoom_f - 1) * arr.shape[0] // zoom_f,
        #     arr.shape[1] // zoom_f : (zoom_f - 1) * arr.shape[1] // zoom_f,
        # ] = 1
        mask = np.ones(arr.shape[:-1])
    if not mask.shape == arr.shape[:-1]:
        raise ValueError(
            "mask should have the same shape as the first two dimensions of array"
        )
    mask = mask.flatten().astype(bool)
    arr_flat_masked = arr_flat[mask]
    arr_flat_nonmask = arr_flat[~mask]
    pca = PCA(n_components=3, random_state=42).fit(arr_flat_masked)
    image_pca = pca.transform(arr_flat_masked)
    # normalize the image to [0, 1] by treating 0.005 and 0.995 quantile as 0 and 1
    qmin, qmax = np.quantile(image_pca, [quantile, 1 - quantile])
    ret = np.zeros((np.prod(arr.shape[:-1]), 3))
    ret[mask] = image_pca
    if not mask.all():
        ret[~mask] = pca.transform(arr_flat_nonmask)
    ret = ((ret - qmin) / (qmax - qmin)).clip(0, 1)
    return ret.reshape(arr.shape[:-1] + (3,)), pca.components_


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
        "-po",
        "--pca_output_dir",
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


def _crop_array(arr: np.ndarray, cropping: None | list[int | float]) -> np.ndarray:
    """Crop the hyperspectral array with given cropping coordinates.

    Args:
        arr: hyperspectral array with shape (samples, lines, bands)
        cropping: a list of 4 integers, [x1, y1, x2, y2]

    Returns:
        cropped array
    """
    h, w = arr.shape[:2]
    if cropping is None:
        return arr
    else:
        if not len(cropping) == 4:
            raise ValueError(f"Cropping should be a list of 4 numbers, got {cropping}")
    if all(int(x) == x for x in cropping):
        cropping = list(map(int, cropping))
        x1, y1, x2, y2 = cropping
    else:
        if not all(0 <= x <= 1 for x in cropping):
            raise ValueError(
                f"Cropping should be in the range of [0, 1], got {cropping}"
            )
        cropping = np.round([w, h, w, h] * np.array(cropping)).astype(int)
        x1, y1, x2, y2 = cropping
    if not (0 <= y1 < y2 <= h):
        raise ValueError(
            f"Invalid cropping coordinates {y1} and {y2} (height of the array is {h})"
        )
    if not (0 <= x1 < x2 <= w):
        raise ValueError(
            f"Invalid cropping coordinates {x1} and {x2} (width of the array is {w})"
        )
    # when cropping, swap the order indices since first dimension of a numpy array is
    # height
    return arr[y1:y2, x1:x2, :]


if __name__ == "__main__":
    # example usages:
    # 1. Convert one hyperspectral image from .bil to .npy
    # python3 bil2numpy.py bil2npy --metadata <hdr_path> bil2npy <bil_path> --output_dir <output_dir>
    # 2. Convert one hyperspectral npy file to png with RGB channels.

    # matplotlib.use("TkAgg")

    parser = argparse.ArgumentParser(description="Utility for data processing")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    bil2npy_parser = subparsers.add_parser("bil2npy", help="Convert BIL to NPY")
    bil2npy_parser.add_argument("bil_path", type=str, help="name of the data cube file")
    bil2npy_parser.add_argument(
        "--metadata", type=str, help="name of the metadata file"
    )
    bil2npy_parser.add_argument("--output_dir", type=str, default=None)
    bil2npy_parser.add_argument(
        "--config", type=str, default=None, help="configuration string"
    )
    bil2npy_parser.add_argument(
        "-qr",
        "--quantile_rgb",
        type=float,
        default=0.999,
        help="quantile for converting hyperspectral data to rgb",
    )
    _add_pca_arguments(bil2npy_parser)

    npy2png_parser = subparsers.add_parser("npy2png", help="Convert NPY to png")
    npy2png_parser.add_argument(
        "-i", "--npz_path", type=str, help="path to the NPY file"
    )
    npy2png_parser.add_argument("-m", "--metadata", type=str, help="metadata file")
    npy2png_parser.add_argument("-o", "--output_dir", type=str, default=None)
    npy2png_parser.add_argument(
        "-qr",
        "--quantile_rgb",
        type=float,
        default=0.999,
        help="quantile for converting hyperspectral data to rgb",
    )
    npy2png_parser.add_argument(
        "-mc",
        "--mask_crop",
        type=str,
        default=None,
        help="Mask file path, should be in coco json format",
    )
    npy2png_parser.add_argument(
        "-c",
        "--cropping",
        # None or a list for 4 integers or floats
        type=float,
        nargs=4,
        default=None,
        help="Cropping the hyperspectral image",
    )
    _add_pca_arguments(npy2png_parser)

    process_bmp_parser = subparsers.add_parser("process_bmp", help="Process BMP files")
    process_bmp_parser.add_argument(
        "-i", "--input_dir", type=str, help="path to the BMP files"
    )
    process_bmp_parser.add_argument(
        "-o", "--output_dir", type=str, help="path to the output dir"
    )

    args = parser.parse_args()

    if args.subcommand in ["bil2npy", "npy2png"]:
        metadata_file = args.metadata
        if metadata_file.endswith(".bil.hdr"):
            metadata = parse_metadata(metadata_file)
        elif metadata_file.endswith(".yaml"):
            with open(metadata_file) as f:
                metadata = yaml.safe_load(f)
        else:
            raise ValueError(
                "Invalid metadata file, only .bil.hdr and .yaml are supported"
            )
        wls = metadata["wavelength"]
        max_samples = metadata["samples"]
        max_lines = metadata["lines"]
        max_bands = metadata["bands"]
        ceiling = metadata["ceiling"]
        if args.subcommand == "bil2npy":
            bil_path = args.bil_path
            image_name = os.path.splitext(os.path.basename(bil_path))[0]
            output_dir = args.output_dir or os.path.dirname(bil_path)
            config = args.config

            if config == "top":
                top_left_sample = 164
                top_left_line = 97
                sample_diff = 1100
                line_diff = 1461
            elif config == "bottom":
                top_left_sample = 150
                top_left_line = 314
                sample_diff = 1100
                line_diff = 1305
            else:
                top_left_sample = 0
                top_left_line = 0
                sample_diff = max_samples
                line_diff = max_lines

            total = sample_diff * line_diff

            arr = bil2np(bil_path, max_samples, max_lines, max_bands)
            np.savez_compressed(os.path.join(output_dir, image_name + ".npz"), data=arr)
            with open(os.path.join(output_dir, image_name + ".yaml"), "w") as f:
                metadata.update(
                    {
                        "top_left_line": top_left_line,
                        "top_left_sample": top_left_sample,
                        "sample_diff": sample_diff,
                        "line_diff": line_diff,
                    }
                )
                yaml.safe_dump(metadata, f, default_flow_style=None)
        elif args.subcommand == "npy2png":
            npz_path = args.npz_path
            output_dir = args.output_dir or os.path.dirname(npz_path)
            image_name = os.path.splitext(os.path.basename(npz_path))[0]
            arr = np.load(npz_path)["data"]
        else:
            raise NotImplementedError

        # do this for both subcommands
        # save as png
        # do cropping for the image array. Cropping must be rectangular.
        cropping = args.cropping
        mask_crop = args.mask_crop
        if mask_crop is not None:
            if cropping is not None:
                raise ValueError(
                    "Only one of cropping and mask_crop should be provided"
                )
            with open(mask_crop) as f:
                masks = _coco_to_contours(json.load(f))
            # take the first mask, get rectangular bounding box, and get top-left and
            # bottom-right coordinates
            x, y, w, h = np.array(cv.boundingRect(masks[0]))
            cropping = [x, y, x + w, y + h]
        arr = _crop_array(arr, args.cropping)
        image = np2png(arr, wls, ceiling, args.quantile_rgb)
        # clip at 0.995 quantile to 255
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, image_name + "_rgb.png"))

        if args.pca is not True:
            sys.exit(0)

        # save first 3 PCs as RGB
        # get subset mask, which can be irregular contours.
        zoom_f = args.zoom_f
        mask_subset = args.mask_subset
        if zoom_f is not None:
            if mask_subset:
                raise ValueError(
                    "Only one of cropping and mask_subset should be provided"
                )
            mask_subset = np.zeros(arr.shape[:-1])
            mask_subset[
                arr.shape[0] // zoom_f : (zoom_f - 1) * arr.shape[0] // zoom_f,
                arr.shape[1] // zoom_f : (zoom_f - 1) * arr.shape[1] // zoom_f,
            ] = 1
        elif mask_subset is not None:
            from utils import _coco_to_contours

            with open(mask_subset) as f:
                masks = _coco_to_contours(json.load(f))
            mask_subset = cv.drawContours(
                np.zeros(arr.shape[:-1], dtype=np.uint8), masks, -1, 255, -1
            )
        else:
            mask_subset = None

        image_pca, loadings = hsi_pca(arr, quantile=args.quantile_pca, mask=mask_subset)
        # In the saved image, first 3 PCs are in the order of RGB.
        image_pca = Image.fromarray((image_pca * 255).astype(np.uint8))
        pca_output_dir = args.pca_output_dir or output_dir
        os.makedirs(pca_output_dir, exist_ok=True)
        image_pca.save(os.path.join(pca_output_dir, image_name + "_pc3.png"))

        # plot loading^2
        loadings_squared = np.square(loadings)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(wls, loadings_squared[0, :], color="r", label="PC1")
        ax.plot(wls, loadings_squared[1, :], color="g", label="PC2")
        ax.plot(wls, loadings_squared[2, :], color="b", label="PC3")
        ax.set_title("Squared loadings on first 3 PCs across wavelength")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Squared loadings")
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        fig.savefig(
            os.path.join(pca_output_dir, image_name + "_pc3_loading.jpg"),
            dpi=300,
            bbox_inches="tight",
        )

    elif args.subcommand == "process_bmp":
        process_bmp(args.input_dir, args.output_dir)
