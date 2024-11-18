import os

import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class CAMIIColonyDetector:
    def __init__(
        self,
        bg_gaussian_kernel: int = 18,
        bg_threshold_block_size: int = 501,
        bg_threshold_offset: int = 18,
        fg_gaussian_kernel: int = 5,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (4, 4),
        canny_upper_percentile: int = 80,
        calib_param: str | None | np.ndarray = None,
        calib_contrast_alpha: float = 1,
        calib_contrast_beta: float = 0,
        crop_x_min: int | None = None,
        crop_x_max: int | None = None,
        crop_y_min: int | None = None,
        crop_y_max: int | None = None,
    ):
        self.bg_gaussian_kernel = bg_gaussian_kernel
        self.bg_threshold_block_size = bg_threshold_block_size
        self.bg_threshold_offset = bg_threshold_offset
        self.fg_gaussian_kernel = fg_gaussian_kernel
        self.canny_upper_percentile = canny_upper_percentile
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.calib_param = self.load_calibration(calib_param)
        self.calib_contrast_alpha = calib_contrast_alpha
        self.calib_contrast_beta = calib_contrast_beta
        self.crop_x_min = crop_x_min
        self.crop_x_max = crop_x_max
        self.crop_y_min = crop_y_min
        self.crop_y_max = crop_y_max

    def load_image(self, image: str | np.ndarray) -> np.ndarray:
        if isinstance(image, str):
            if image.endswith(".npy"):
                return np.load(image)
            else:
                return cv.imread(image)
        return image

    def crop_image(
        self,
        image: np.ndarray,
        x_min: int | None,
        x_max: int | None,
        y_min: int | None,
        y_max: int | None,
    ) -> np.ndarray:
        pil_image = Image.fromarray(image)
        x_min = x_min if x_min is not None else 0
        x_max = x_max if x_max is not None else pil_image.width
        y_min = y_min if y_min is not None else 0
        y_max = y_max if y_max is not None else pil_image.height
        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        return np.array(cropped_image)

    def load_calibration(self, calib_param: str | np.ndarray | None) -> np.ndarray:
        if calib_param is None:
            return np.array(1)
        elif isinstance(calib_param, str):
            if not os.path.isfile(calib_param):
                raise FileNotFoundError(
                    f"Calibration parameter file '{calib_param}' not found."
                )
            if calib_param.endswith(".npy"):
                calib_param: np.ndarray = np.load(calib_param)
            elif calib_param.endswith(".npz"):
                calib_param: np.ndarray = np.load(calib_param)["arr_0"]
            else:
                raise ValueError(
                    f"Unsupported calibration parameter file format: {calib_param}"
                )
        return calib_param

    def correct_image(self, image: np.ndarray) -> np.ndarray:
        corrected_image = (
            image / self.calib_param
        ) * self.calib_contrast_alpha + self.calib_contrast_beta
        return corrected_image.astype(np.float32)

    def detect(
        self, image: str | np.ndarray, diagnose: bool = False
    ) -> list[np.ndarray] | tuple[list[np.ndarray], Figure, np.ndarray]:
        arr = self.load_image(image)
        cropped_image = self.crop_image(
            arr, self.crop_x_min, self.crop_x_max, self.crop_y_min, self.crop_y_max
        )
        self.image_cropped = cropped_image
        corrected_image = self.correct_image(cropped_image)
        self.image_corrected = corrected_image
        # Convert to grayscale if RGB
        if corrected_image.ndim == 3 and corrected_image.shape[2] == 3:
            gray_image = cv.cvtColor(corrected_image, cv.COLOR_BGR2GRAY)
        else:
            gray_image = corrected_image
        self.image_corrected_gs = gray_image

        clahe_obj = cv.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size
        )
        clahed = clahe_obj.apply(gray_image.astype(np.uint8))

        # Step 1: Filter2D
        filtered_image = cv.filter2D(
            clahed,
            -1,
            np.array([[-1, -1, -1], [-1, self.bg_gaussian_kernel, -1], [-1, -1, -1]])
            / (self.bg_gaussian_kernel - 8),
        ).astype(np.uint8)

        # Step 2: Adaptive Threshold
        arr_bg_mask = cv.adaptiveThreshold(
            filtered_image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            self.bg_threshold_block_size,
            self.bg_threshold_offset,
        )

        # Step 3: Gaussian Blur
        arr_res_bg = cv.GaussianBlur(
            src=np.where(arr_bg_mask, clahed, 0),
            ksize=(self.fg_gaussian_kernel, self.fg_gaussian_kernel),
            sigmaX=0,
        )

        # Step 4: Canny Edge Detection
        try:
            threshold2 = np.percentile(
                arr_res_bg.flatten(), self.canny_upper_percentile
            )
        except IndexError:
            threshold2 = 0

        edges = cv.Canny(
            image=arr_res_bg.astype(np.uint8), threshold1=1, threshold2=threshold2
        )

        # Step 5: Dilate
        dilated = cv.dilate(src=edges, kernel=None, iterations=3)

        # Step 6: Erode
        eroded = cv.erode(src=dilated, kernel=None, iterations=1)

        # Step 7: Find Contours
        contours, _ = cv.findContours(
            eroded,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )

        if diagnose:
            fig, axs = plt.subplots(4, 3, figsize=(15, 15))
            axs = axs.flatten()
            i = 0

            if arr.ndim == 3:
                axs[i].imshow(cv.cvtColor(arr, cv.COLOR_BGR2RGB))
                axs[i].set_title("Original Image")
                i += 1
                axs[i].imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
                axs[i].set_title("After Cropping")
                i += 1
                axs[i].imshow(
                    cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB).astype(int)
                )
                axs[i].set_title("After Calibration")
            else:
                axs[i].imshow(arr, cmap="gray")
                axs[i].set_title("Original Image")
                i += 1
                axs[i].imshow(cropped_image, cmap="gray")
                axs[i].set_title("After Cropping")
                i += 1
                axs[i].imshow(corrected_image.astype(int), cmap="gray")
                axs[i].set_title("After Calibration")
            i += 1

            axs[i].imshow(clahed, cmap="gray")
            axs[i].set_title("After CLAHE")
            i += 1

            axs[i].imshow(filtered_image, cmap="gray")
            axs[i].set_title("After Filter2D")
            i += 1

            axs[i].imshow(arr_bg_mask, cmap="gray")
            axs[i].set_title("After Adaptive Threshold")
            i += 1

            axs[i].imshow(arr_res_bg, cmap="gray")
            axs[i].set_title("After Gaussian Blur")
            i += 1

            axs[i].imshow(edges, cmap="gray")
            axs[i].set_title("After Canny Edge Detection")
            i += 1

            axs[i].imshow(dilated, cmap="gray")
            axs[i].set_title("After Dilate")
            i += 1

            axs[i].imshow(eroded, cmap="gray")
            axs[i].set_title("After Erode")
            i += 1

            result_image = cv.drawContours(
                cropped_image.copy(), contours, -1, (0, 255, 0), 1
            )

            if result_image.ndim == 2:
                axs[i].imshow(result_image, cmap="gray")
            else:
                axs[i].imshow(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
            axs[i].set_title("After Find Contours")

            for ax in axs:
                ax.axis("off")

            fig.tight_layout()
            # contours = self.contour_coords_reverse_crop(contours)
            return contours, fig, axs
        else:
            # contours = self.contour_coords_reverse_crop(contours)
            return contours

    def contour_coords_reverse_crop(self, contours: list[np.ndarray]) -> np.ndarray:
        delta_x = 0 if self.crop_x_min is None else self.crop_x_min
        delta_y = 0 if self.crop_y_min is None else self.crop_y_min
        contours = tuple(cnt + np.array([delta_x, delta_y]) for cnt in contours)
        return contours


# Example usage:
# processor = CAMIIColonyDetector(3, 11, 2, 5, 90, 'calib.npz', 1.0, 0.0, 10, 200, 10, 200)
# contours = processor.detect('path/to/image.png')
# contours, fig, axs = processor.detect('path/to/image.png', diagnose=True)
