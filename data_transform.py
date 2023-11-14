#!/usr/bin/env python
import struct
import ast
import numpy as np
import os
import argparse

from sklearn.decomposition import PCA
from PIL import Image
import cv2 as cv
from tqdm.auto import trange
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


def np2jpg(arr: np.ndarray, wls: list, ceiling: int) -> Image:
    if not arr.shape[2] == len(wls):
        raise ValueError(
            "Number of channels in data and number of wavelengths do not match"
        )
    wls = np.array(wls)
    image_data_r = arr[:, :, wls.searchsorted(R_WL_LOW) : wls.searchsorted(R_WL_HIGH)]
    image_data_g = arr[:, :, wls.searchsorted(G_WL_LOW) : wls.searchsorted(G_WL_HIGH)]
    image_data_b = arr[:, :, wls.searchsorted(B_WL_LOW) : wls.searchsorted(B_WL_HIGH)]
    # image = Image.fromarray((image_data / (ceiling / 255)).astype(np.uint8))
    image_data = np.stack([image_data_r, image_data_g, image_data_b], axis=-1).mean(-2)
    image = Image.fromarray((image_data / ceiling * 256).astype(np.uint8))
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
    for i, f in enumerate(
        sorted(f for f in os.listdir(input_dir) if f.endswith(".bmp"))
    ):
        barcode = os.path.splitext(f)[0].split("_")[0]
        width, height = np.fromfile(
            os.path.join(input_dir, f), dtype=np.uint32, offset=18, count=2
        )
        image = np.fromfile(
            os.path.join(input_dir, f), dtype=np.uint8, offset=54
        ).reshape(height, width, 4)[::-1]
        if i % 2:  # white light
            cv.imwrite(os.path.join(output_dir, f"{barcode}_rgb_white.png"), image)
            cv.imwrite(
                os.path.join(output_dir, f"{barcode}_gs_white.png"),
                cv.cvtColor(image, cv.COLOR_BGR2GRAY),
            )
        else:  # red light
            cv.imwrite(os.path.join(output_dir, f"{barcode}_rgb_red.png"), image)
            cv.imwrite(
                os.path.join(output_dir, f"{barcode}_gs_red.png"),
                cv.cvtColor(image, cv.COLOR_BGR2GRAY),
            )
            # image_gray = cv.imread(
            #     os.path.join(output_dir, f"{barcode}_rgb_red.png"), cv.IMREAD_GRAYSCALE
            # )


if __name__ == "__main__":
    # example usages:
    # 1. Convert one hyperspectral image from .bil to .npy
    # python3 bil2numpy.py bil2npy --metadata <hdr_path> bil2npy <bil_path> --output_dir <output_dir>
    # 2. Convert one hyperspectral npy file to jpg with RGB channels.

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

    npy2jpg_parser = subparsers.add_parser("npy2jpg", help="Convert NPY to JPG")
    npy2jpg_parser.add_argument(
        "-i", "--npz_path", type=str, help="path to the NPY file"
    )
    npy2jpg_parser.add_argument("-m", "--metadata", type=str, help="metadata file")
    npy2jpg_parser.add_argument("-o", "--output_dir", type=str, default=None)

    process_bmp_parser = subparsers.add_parser("process_bmp", help="Process BMP files")
    process_bmp_parser.add_argument(
        "-i", "--input_dir", type=str, help="path to the BMP files"
    )
    process_bmp_parser.add_argument(
        "-o", "--output_dir", type=str, help="path to the output dir"
    )

    args = parser.parse_args()

    if args.subcommand in ["bil2npy", "npy2jpg"]:
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
        elif args.subcommand == "npy2jpg":
            npz_path = args.npz_path
            output_dir = args.output_dir or os.path.dirname(npz_path)
            image_name = os.path.splitext(os.path.basename(npz_path))[0]
            if npz_path.endswith(".npz"):
                arr = np.load(npz_path)["data"]
            elif npz_path.endswith(".npy"):
                arr = np.load(npz_path)
            else:
                raise ValueError("Invalid npz/npy file")
        else:
            raise NotImplementedError

        # do this for both subcommands
        # save as jpg
        image = np2jpg(arr, wls, ceiling)
        image.save(os.path.join(output_dir, image_name + "_rgb.jpg"))

        # save first 3 PCs as RGB
        arr_flat = arr.reshape(-1, arr.shape[-1])
        image_pca = (
            PCA(n_components=3).fit_transform(arr_flat).reshape(arr.shape[:-1] + (3,))
        )
        # normalize the image to [0, 1] by treating 0.005 and 0.995 quantile as 0 and 1
        q005, q995 = np.quantile(image_pca, [0.005, 0.995])
        image_pca = ((image_pca - q005) / (q995 - q005)).clip(0, 1)
        image_pca = Image.fromarray((image_pca * 255).astype(np.uint8))
        image_pca.save(os.path.join(output_dir, image_name + "_pc3.jpg"))
    elif args.subcommand == "process_bmp":
        process_bmp(args.input_dir, args.output_dir)
