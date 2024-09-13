import sys
import os
from os import path as op
from collections import defaultdict
import yaml
import glob

import numpy as np
import pandas as pd
import cv2 as cv
import biom


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
        if key_from in config:
            ret[key_to] = ret[key_from]
            del ret[key_from]
    return ret


def parse_dir_for_time_series(input_dir, ext: str = "png") -> dict[str, dict[int, str]]:
    """
    Parses a directory containing time series images.

    File name format is assumed to be "{barcode}_{time_point}.png". Time points must be
        positive integers. If time point is absent, i.e. "{barcode}.png", time point
        will be assumed to be -1. That is, this image will be the default for this
        barcode.

    Args:
        input_dir (str): The path to the input directory.

    Returns:
        dict[str, dict[int, str]]: A dictionary containing the image labels as keys
            and a dictionary containing the time points as keys and the file paths as
            values.
    """
    if not op.isdir(input_dir):
        raise ValueError(f"Directory {input_dir} not found.")
    image_time_series = defaultdict(dict)
    for image in sorted(glob.glob(f"{input_dir}/*{ext}")):
        paths = op.splitext(op.basename(image))[0].split("_")
        barcode = paths[0]
        # if not barcode in {'ABYRS855'}:
        #     continue
        if len(paths) == 1:
            time_point = -1
        else:
            try:
                time_point = int(paths[1][1:])
            except ValueError:
                time_point = -1
        image_time_series[barcode][time_point] = image
    return image_time_series


def _get_time_points(
    input_dir: dict, time: int | str | dict[str, int | str] = "default", ext: str = "png", missing_tp: str = "silence",
) -> tuple[list[str], list[str]]:
    """
    Processes an rgb dictionary to determine appropriate time points and compiles lists
    of image labels and corresponding image paths.

    Args:
        input_dir (dict): A dictionary containing image labels as keys and a dictionary
            containing the time points as keys and the file paths as values.
        time (int | str): The time point or criteria ('min' or 'max') for image
            selection.
        directory (str): The directory name ('red_rgb' or 'white_rgb') indicating the
            image type.

    Returns:
        tuple[list[str], list[str]]: Two lists, one of image labels and another of corresponding image paths.
    """
    image_label_list = []
    image_list = []
    ts_dict = parse_dir_for_time_series(input_dir, ext=ext)

    for barcode, time_points in ts_dict.items():
        if isinstance(time, dict):
            time_b = time[barcode]
        else:
            time_b = time
        if isinstance(time_b, int):
            if time_b in time_points:
                image_label_list.append(barcode)
                image_list.append(time_points[time_b])
            else:
                if missing_tp == "raise":
                    raise ValueError(
                        f"Time point {time_b} not found for barcode {barcode} in {input_dir}."
                    )
                elif missing_tp == "print":
                    print(
                        f"Time point {time_b} not found for barcode {barcode} in {input_dir}."
                    )
                elif missing_tp == "silence":
                    pass
                else:
                    raise ValueError(
                        f"Invalid missing_tp argument {missing_tp}. "
                        "Must be 'raise', 'print', or 'silence'."
                    )
        else:
            if time_points:
                # if -1 is a time point, use it as the default
                if time_b == "default":
                    if -1 in time_points:
                        selected_time = -1
                    else:
                        raise ValueError(
                            f"Invalid time argument {time_b}. Must be 'default'."
                        )
                else:
                    tps = [i for i in time_points.keys() if i >= 0]
                    if time_b == "min":
                        selected_time = min(tps)
                    elif time_b == "max":
                        selected_time = max(tps)
                    else:
                        raise ValueError(
                            f"Invalid time argument {time_b}. Must be int, 'min', or 'max'."
                        )
                image_label_list.append(barcode)
                image_list.append(time_points[selected_time])
            else:
                raise ValueError(
                    f"No time points found for barcode {barcode} in {input_dir}."
                )

    return image_label_list, image_list


def read_file_list(
    input_dir: str, time: int | str = "default"
) -> tuple[list[str], list[str], list[str]]:
    """
    Reads a list of image files from a directory, ensuring consistency between red and
        white RGB images, and returns the file paths. It provides detailed information
        on any inconsistencies found between the labels in the red_rgb and white_rgb
        directories.

    Args:
        input_dir (str): The directory from which to read images.
        time (int | str): The time point or criteria for image selection.

    Returns:
        tuple[list[str], list[str], list[str]]: Lists of image labels, transmission,
            and epifluorescence image paths.
    """
    if not op.isdir(input_dir):
        raise ValueError(f"Directory {input_dir} not found.")
    red_labels, red_images = _get_time_points(f"{input_dir}/red_rgb", time)
    white_labels, white_images = _get_time_points(f"{input_dir}/white_rgb", time)

    # Determine the differences between red and white labels
    missing_in_red = set(white_labels) - set(red_labels)
    missing_in_white = set(red_labels) - set(white_labels)

    if missing_in_red or missing_in_white:
        error_messages = []
        if missing_in_red:
            error_messages.append(
                f"{len(missing_in_red)} labels found in white_rgb but not in red_rgb: "
                f"{', '.join(sorted(missing_in_red))}"
            )
        if missing_in_white:
            error_messages.append(
                f"{len(missing_in_white)} labels found in red_rgb but not in white_rgb: "
                f"{', '.join(sorted(missing_in_white))}"
            )
        raise ValueError(
            "Inconsistent labels between red_rgb and white_rgb directories: "
            + "; ".join(error_messages)
        )

    return red_labels, red_images, white_images


# def read_file_list(
#     input_dir: str, time: int | str = "max"
# ) -> tuple[list[str], list[str], list[str]]:
#     """
#     Reads a list of image files from a directory and returns the file paths.

#     Args:
#         input_dir (str): The path to the input directory.
#         time (int | str): The time point to read images from if int. Otherwise, must be
#             "min" or "max", in which case the minimum or maximum time point will be
#             selected. By default we select the maximum time point.

#     Returns:
#         tuple[int, list[str], list[str], list[str]]: A tuple containing the total number
#             of images, a list of image labels, a list of transmission image file paths,
#             and a list of epifluorescence image file paths.
#     """
#     image_label_list = []
#     image_trans_list = []
#     image_epi_list = []
#     rgb_red_dict = parse_dir_for_time_series(f"{input_dir}/red_rgb")
#     rgb_white_dict = parse_dir_for_time_series(f"{input_dir}/white_rgb")
#     # take the union of all barcodes in the two dictionaries and raise error during loop
#     # if not all barcodes are present in both dictionaries
#     for barcode in set(rgb_red_dict.keys()) | set(rgb_white_dict.keys()):
#         rgb_red_b = rgb_red_dict.get(barcode, {})
#         rgb_white_b = rgb_white_dict.get(barcode, {})
#         if not rgb_red_b:
#             raise ValueError(f"Barcode {barcode} not found in rgb_red directory.")
#         if not rgb_white_b:
#             raise ValueError(f"Barcode {barcode} not found in rgb_white directory.")
#         if isinstance(time, int):
#             time_point_red = time
#             time_point_white = time
#         else:
#             if time == "min":
#                 func = min
#             elif time == "max":
#                 func = max
#             else:
#                 raise ValueError(
#                     f"Invalid time argument {time}. Must be int, 'min', or 'max'."
#                 )
#             time_point_red = func(rgb_red_b.keys())
#             time_point_white = func(rgb_white_b.keys())
#         # if -1 is one of the time points, use it as the default
#         if -1 in rgb_red_b:
#             time_point_red = -1
#         if -1 in rgb_white_b:
#             time_point_white = -1
#         image_label_list.append(barcode)
#         image_trans_list.append(rgb_red_b[time_point_red])
#         image_epi_list.append(rgb_white_b[time_point_white])
#     return image_label_list, image_trans_list, image_epi_list


def read_file_list_(input_dir: str) -> tuple[list[str], list[str], list[str]]:
    """
    Reads a list of image files from a directory and returns the file paths.

    DEPRECATED:
        We used to store all the images in a single directory, using file names like
        "{barcode}_{condition}.png", where condition indicates light condition and
        channel. Now we store conditions in separate directories, and incorporate
        time series label into the file names.

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
    for image in sorted(glob.glob(f"{input_dir}/*_rgb_red.png")):
        # red light rgb image
        barcode = op.splitext(op.basename(image))[0].split("_")[0]
        image_label_list.append(barcode)
        image_trans_list.append(image)
    for image in sorted(glob.glob(f"{input_dir}/*_rgb_white.png")):
        # white light rgb image
        image_epi_list.append(image)
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

    cv.drawContours(image_contours, contours, -1, border_color, contour_pixel).astype(
        np.uint8
    )

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


def _coco_to_contours(coco: dict, category_id: int = None) -> list[np.ndarray]:
    return [
        np.array(anno["segmentation"][0]).reshape(-1, 1, 2).astype(np.int32)
        for anno in coco["annotations"]
        if category_id is None or anno["category_id"] == category_id
    ]


def read_table(
    table_path: str,
    index_col: str | int = 0,
    comment: str = None,
    dtype: str = "int",
) -> pd.DataFrame:
    """Read a table from a file and return it as a DataFrame.

    Args:
        table_path: Path to the table file. Could be a tsv, csv or biom.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    if table_path.endswith(".biom"):
        df_biom = biom.load_table(table_path)
        df = df_biom.to_dataframe().astype(dtype)
        df.index.name = df_biom.table_id
        return df
    else:
        if table_path.endswith(".csv") or table_path.endswith(".csv.gz"):
            method = pd.read_csv
        elif table_path.endswith(".tsv") or table_path.endswith(".tsv.gz"):
            method = pd.read_table
        else:
            raise ValueError("Unsupported table file format.")
        return method(table_path, index_col=index_col, comment=comment)


def write_table(table: pd.DataFrame, table_path: str) -> None:
    """Write a table to a file.

    Args:
        table: DataFrame to be written to a file.
        table_path: Path to the output file. Could be a tsv, csv or biom.
    """
    if table_path.endswith(".biom"):
        data = biom.Table(
            table.to_numpy(), table.index, table.columns, table_id=table.index.name
        )
        with biom.util.biom_open(table_path, "w") as f:
            data.to_hdf5(f, "whatever60")
    elif table_path.endswith(".csv") or table_path.endswith(".csv.gz"):
        table.to_csv(table_path)
    elif table_path.endswith(".tsv") or table_path.endswith(".tsv.gz"):
        table.to_csv(table_path, sep="\t")
    else:
        raise ValueError("Unsupported table file format.")
