import ast
import glob
import json
import os
import struct

import numpy as np
from sklearn.decomposition import PCA
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import yaml


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


def _crop_array(arr: np.ndarray, cropping: list[int | float] | None) -> np.ndarray:
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
        cropping_ = list(map(int, cropping))
        x1, y1, x2, y2 = cropping_
    else:
        if not all(0 <= x <= 1 for x in cropping):
            raise ValueError(
                f"Cropping should be in the range of [0, 1], got {cropping}"
            )
        cropping_: list[int | float] = (
            np.round([w, h, w, h] * np.array(cropping)).astype(int).tolist()
        )
        x1, y1, x2, y2 = cropping_
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


def bil2np(bil_path: str, samples: int, lines: int, channels: int) -> np.ndarray:
    arr = np.fromfile(bil_path, dtype=np.uint16)
    assert arr.shape[0] == samples * lines * channels, "Invalid data shape"
    return arr.reshape(lines, channels, samples).transpose(2, 0, 1)


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


def np2png(
    arr: np.ndarray, wls: list, ceiling: int, quantile: float | None = 0.999
) -> Image.Image:
    if not arr.shape[2] == len(wls):
        raise ValueError(
            "Number of channels in data and number of wavelengths do not match"
        )
    wls: np.ndarray = np.array(wls)
    image_data_r = arr[:, :, wls.searchsorted(R_WL_LOW) : wls.searchsorted(R_WL_HIGH)]
    image_data_g = arr[:, :, wls.searchsorted(G_WL_LOW) : wls.searchsorted(G_WL_HIGH)]
    image_data_b = arr[:, :, wls.searchsorted(B_WL_LOW) : wls.searchsorted(B_WL_HIGH)]
    # image = Image.fromarray((image_data / (ceiling / 255)).astype(np.uint8))
    image_data = np.stack([image_data_r, image_data_g, image_data_b], axis=-1)
    if quantile is not None:
        ceiling: float = np.quantile(image_data, quantile).item()
    image_data = np.clip(image_data.mean(-2), a_min=0, a_max=ceiling) / ceiling * 255
    image = Image.fromarray(image_data.astype(np.uint8))
    return image


def hsi_pca(
    arr: np.ndarray, mask: np.ndarray | None = None, quantile: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
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


def process_hyperspectral_image(
    arr: np.ndarray,
    wls: list[float],
    ceiling: int,
    # cropping before pca
    cropping: list[float] | None,
    mask_crop: str | None,
    # avoid extreme values in RGB
    quantile_rgb: float,
    #
    pca: bool,
    # cropping for fitting PCA
    zoom_f: int | None,
    mask_subset: str | None,
    # avoid extreme values in PCA
    quantile_pca: float,
) -> tuple[Image.Image, np.ndarray | None, np.ndarray | None]:
    """
    Process common steps for both bil2npy and npy2png commands.

    This function performs several processing steps on a hyperspectral image array,
    including cropping, normalization, conversion to RGB, and optionally PCA analysis.

    Steps:
    1. Cropping: The image can be cropped using either direct coordinates or a mask file.
    2. Normalization and RGB Conversion: The hyperspectral data is normalized and converted to an RGB image.
    3. PCA Analysis (optional): If PCA is enabled, the first three principal components and loadings are calculated.

    Args:
        arr: The hyperspectral image array.
        wls: The wavelengths.
        ceiling: The ceiling value for normalization.
        quantile_rgb: The quantile for converting hyperspectral data to RGB.
        cropping: Cropping coordinates as a list [x_min, y_min, x_max, y_max].
        mask_crop: Path to the mask file for cropping. Mutually exclusive with cropping.
        pca: Whether to perform PCA.
        zoom_f: Zoom factor for selecting the subset mask.
        mask_subset: Path to the mask file for subset selection. Mutually exclusive with zoom_f.
        quantile_pca: Quantile for PCA scaling.

    Returns:
        A tuple containing the RGB image, PCA image, PCA loadings, and wavelengths.
    """
    if mask_crop is not None:
        if cropping is not None:
            raise ValueError("Only one of cropping and mask_crop should be provided")
        from utils import _coco_to_contours

        if mask_crop.endswith(".json"):  # coco json format
            with open(mask_crop) as f:
                masks = _coco_to_contours(json.load(f))
            x, y, w, h = np.array(cv.boundingRect(masks[0]))
            cropping = [x, y, x + w, y + h]
        # yolo format, treat the first object as the mask
        elif mask_crop.endswith(".txt"):
            with open(mask_crop) as f:
                x, y, w, h = map(float, f.readline().split()[1:])
            cropping = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    arr = _crop_array(arr, cropping)
    image_rgb = np2png(arr, wls, ceiling, quantile_rgb)

    if not pca:
        return image_rgb, None, None

    if zoom_f is not None:
        if mask_subset:
            raise ValueError("Only one of cropping and mask_subset should be provided")
        mask_subset_arr = np.zeros(arr.shape[:-1])
        mask_subset_arr[
            arr.shape[0] // zoom_f : (zoom_f - 1) * arr.shape[0] // zoom_f,
            arr.shape[1] // zoom_f : (zoom_f - 1) * arr.shape[1] // zoom_f,
        ] = 1
    elif mask_subset is not None:
        from utils import _coco_to_contours

        with open(mask_subset) as f:
            masks = _coco_to_contours(json.load(f))
        mask_subset_arr = cv.drawContours(
            np.zeros(arr.shape[:-1], dtype=np.uint8), masks, -1, (255,), -1
        )
    else:
        mask_subset_arr = None

    image_pca, loadings = hsi_pca(arr, quantile=quantile_pca, mask=mask_subset_arr)
    image_pca = (image_pca * 255).astype(np.uint8)

    return image_rgb, image_pca, loadings


def save_processed_images(
    image_name: str,
    image_rgb: Image.Image,
    output_dir_rgb: str,
    image_pca: np.ndarray | None,
    output_dir_pca: str | None,
    loadings: np.ndarray | None,
    wls: list[float] | None,
) -> None:
    """
    Save the processed images and PCA results to disk.

    Args:
        image_rgb: The RGB image.
        image_pca: The PCA image.
        loadings: The PCA loadings.
        wls: The wavelengths.
        image_name: The base name for the output files.
        output_dir: The directory to save the output files.
        output_dir_pca: Directory to save PCA outputs.
    """
    os.makedirs(output_dir_rgb, exist_ok=True)
    image_rgb.save(os.path.join(output_dir_rgb, image_name + "_rgb.png"))

    if image_pca is not None:
        # sanity check
        if output_dir_pca is None or loadings is None or wls is None:
            raise ValueError("output_dir_pca, loadings, and wls should be provided")
        if not loadings.shape[0] == 3 and loadings.shape[1] == len(wls):
            raise ValueError(
                f"Invalid loadings shape: {loadings.shape}, expecting (3, {len(wls)})"
            )
        os.makedirs(output_dir_pca, exist_ok=True)
        Image.fromarray(image_pca).save(
            os.path.join(output_dir_pca, image_name + "_pc3.png")
        )

        # Plot loading^2
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
            os.path.join(output_dir_pca, image_name + "_pc3_loading.jpg"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def process_bil2npz(
    input_dir: str,
    output_dir_npz: str,
    output_dir_rgb: str,
    time_point: int | None,
    config: str | None,
    cropping: list[float] | None,
    mask_crop_dir: str | None,
    quantile_rgb: float,
    #
    pca: bool,
    output_dir_pca: str | None,
    zoom_f: int | None,
    mask_subset: str | None,
    quantile_pca: float,
) -> None:
    """
    Process BIL files in a directory and convert them to NPZ files.

    Args:
        input_dir: Directory containing .bil files.
        output_dir: Directory to save output files.
        time_point: Time point suffix to add to output file names.
        config: Configuration string ('top', 'bottom', or None).
        quantile_rgb: Quantile for converting hyperspectral data to RGB.
        pca: Whether to perform PCA.
        zoom_f: Zoom factor for selecting the subset mask.
        mask_subset: Path to the mask file for subset selection.
        quantile_pca: Quantile for PCA scaling.
        output_dir_pca: Directory to save PCA outputs.
        cropping: Cropping coordinates as a list [x_min, y_min, x_max, y_max].
        mask_crop_dir: Path to the mask file for cropping.
    """
    bil_files = sorted(glob.glob(os.path.join(input_dir, "*.bil")))
    if not bil_files:
        raise FileNotFoundError(f"No .bil files found in directory: {input_dir}")

    for bil_path in bil_files:
        base_name = os.path.splitext(os.path.basename(bil_path))[0]
        image_name = (
            f"{base_name}_d{time_point}" if time_point is not None else base_name
        )

        hdr_path = bil_path + ".hdr"
        if not os.path.exists(hdr_path):
            raise FileNotFoundError(
                f"Metadata file {hdr_path} not found for {bil_path}"
            )

        if mask_crop_dir is not None:
            try:
                mask_crop = glob.glob(os.path.join(mask_crop_dir, image_name + "*"))[0]
            except IndexError:
                raise ValueError(f"Mask file not found for {image_name}")
        else:
            mask_crop = None

        metadata = parse_metadata(hdr_path)
        wls = metadata["wavelength"]
        max_samples = metadata["samples"]
        max_lines = metadata["lines"]
        max_bands = metadata["bands"]
        ceiling = metadata["ceiling"]

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

        arr = bil2np(bil_path, max_samples, max_lines, max_bands)

        # save npz
        os.makedirs(output_dir_npz, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir_npz, image_name + ".npz"), data=arr)
        # save metadata as yaml
        with open(os.path.join(output_dir_npz, image_name + ".yaml"), "w") as f:
            metadata.update(
                {
                    "top_left_line": top_left_line,
                    "top_left_sample": top_left_sample,
                    "sample_diff": sample_diff,
                    "line_diff": line_diff,
                }
            )
            yaml.safe_dump(metadata, f, default_flow_style=None)

        # Process data
        image_rgb, image_pca, loadings = process_hyperspectral_image(
            arr=arr,
            wls=wls,
            ceiling=ceiling,
            quantile_rgb=quantile_rgb,
            cropping=cropping,
            mask_crop=mask_crop,
            pca=pca,
            zoom_f=zoom_f,
            mask_subset=mask_subset,
            quantile_pca=quantile_pca,
        )

        # Save rgb, pca png and loading
        save_processed_images(
            image_name=image_name,
            image_rgb=image_rgb,
            output_dir_rgb=output_dir_rgb,
            image_pca=image_pca,
            output_dir_pca=output_dir_pca,
            loadings=loadings,
            wls=wls,
        )


def process_npz2png(
    input_dir: str,
    output_dir_rgb: str,
    time_point: int | None,
    cropping: list[float] | None,
    mask_crop_dir: str | None,
    quantile_rgb: float,
    #
    pca: bool,
    output_dir_pca: str | None,
    zoom_f: int | None,
    mask_subset: str | None,
    quantile_pca: float,
) -> None:
    """
    Process NPZ files in a directory and convert them to PNG images.

    Args:
        input_dir: Directory containing .npz files.
        output_dir_rgb: Directory to save output PNG images.
        time_point: Time point suffix to add to output file names.
        quantile_rgb: Quantile for converting hyperspectral data to RGB.
        pca: Whether to perform PCA.
        zoom_f: Zoom factor for selecting the subset mask.
        mask_subset: Path to the mask file for subset selection.
        quantile_pca: Quantile for PCA scaling.
        output_dir_pca: Directory to save PCA outputs.
        cropping: Cropping coordinates as a list [x_min, y_min, x_max, y_max].
        mask_crop_dir: Path to the mask file for cropping.
    """
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in directory: {input_dir}")

    for npz_path in npz_files:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        image_name = (
            f"{base_name}_d{time_point}" if time_point is not None else base_name
        )

        yaml_path = os.path.join(input_dir, f"{base_name}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Metadata file {yaml_path} not found for {npz_path}"
            )

        if mask_crop_dir is not None:
            try:
                mask_crop = glob.glob(os.path.join(mask_crop_dir, image_name + "*"))[0]
            except IndexError:
                raise ValueError(f"Mask file not found for {image_name}")
        else:
            mask_crop = None

        # Load data and metadata
        data = np.load(npz_path)["data"]
        with open(yaml_path, "r") as f:
            metadata = yaml.safe_load(f)

        wls = metadata["wavelength"]
        ceiling = metadata["ceiling"]

        # Process data
        image_rgb, image_pca, loadings = process_hyperspectral_image(
            arr=data,
            wls=wls,
            ceiling=ceiling,
            quantile_rgb=quantile_rgb,
            cropping=cropping,
            mask_crop=mask_crop,
            pca=pca,
            zoom_f=zoom_f,
            mask_subset=mask_subset,
            quantile_pca=quantile_pca,
        )

        # Save processed images
        save_processed_images(
            image_rgb=image_rgb,
            image_pca=image_pca,
            loadings=loadings,
            wls=wls,
            image_name=image_name,
            output_dir_rgb=output_dir_rgb,
            output_dir_pca=output_dir_pca,
        )
