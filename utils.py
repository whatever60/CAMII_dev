import sys
import os
from collections import defaultdict

import numpy as np
import cv2 as cv
import yaml


def read_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    platform_mapping = {"darwin": "macos", "win32": "mswin10"}

    platform = platform_mapping.get(sys.platform, "mswin10")

    keys_to = [
        "plate_qc_colony_contour_pixel",
        "plate_qc_image_scale_factor",
        "plate_qc_image_width_bias",
        "plate_qc_image_height_bias",
        "plate_qc_confirm_window",
        "plate_qc_text_size_large",
        "plate_qc_text_size_small",
        "plate_qc_text_size_button",
        "colony_qc_image_scale_factor",
        "colony_qc_image_width_bias",
        "colony_qc_image_height_bias",
        "colony_qc_confirm_window",
        "colony_qc_colony_show_bias",
        "colony_qc_colony_window_bias",
        "colony_qc_colony_contour_pixel",
        "colony_qc_colony_label_size",
        "colony_qc_colony_label_thickness",
        "colony_qc_text_size_large",
        "colony_qc_text_size_mid",
        "colony_qc_text_size_small",
        "colony_qc_text_size_button",
    ]
    keys_from = []
    for k in keys_to:
        tmp = k.split("_")
        tmp.insert(2, platform)
        keys_from.append("_".join(tmp))
    ret = config.copy()
    for key_from, key_to in zip(keys_from, keys_to):
        ret[key_to] = ret[key_from]
        del ret[key_from]
    return ret


def read_file_list(input_dir: str) -> tuple[list[str], list[str], list[str]]:
    """
    Reads a list of image files from a directory and returns the file paths.

    Args:
        input_dir (str): The path to the input directory.

    Returns:
        tuple[int, list[str], list[str], list[str]]: A tuple containing the total number
            of images, a list of image labels, a list of transmission image file paths,
            and a list of epifluorescence image file paths.
    """
    image_label_list = []
    image_trans_list = []
    image_epi_list = []
    for i, image in enumerate(
        sorted(f for f in os.listdir(input_dir) if f.endswith(".png"))
    ):
        barcode = os.path.splitext(image)[0].split("_")[0]
        if i % 3 == 1:  # red light rgb image
            image_label_list.append(barcode)
            image_trans_list.append(os.path.join(input_dir, image))
        elif i % 3 == 2:  # white light rgb image
            image_epi_list.append(os.path.join(input_dir, image))
    return image_label_list, image_trans_list, image_epi_list


def add_contours(
    image: np.ndarray,
    contours: list[np.ndarray],
    centers: np.ndarray,  # [num_contours, 2]
    contour_pixel: int,
    center_pixel: int,
    border_color: tuple[int, int, int] = (0, 0, 0),
    center_color: tuple[int, int, int] = (0, 255, 0),
    annot_index: bool = False,
) -> np.ndarray:
    if image.ndim == 2:
        image_contours = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image_contours = image.copy()

    cv.drawContours(
        image_contours, contours, -1, border_color, contour_pixel
    ).astype(np.uint8)
    
    for idx, c in enumerate(np.round(centers).astype(int)):
        cv.circle(image_contours, tuple(c), center_pixel, center_color, -1)
        if annot_index:
            cv.putText(
                image_contours,
                str(idx + 1),
                tuple(c),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv.LINE_AA,
            )

    return image_contours
