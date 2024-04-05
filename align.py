#!/usr/bin/env python

import glob
import tempfile
import json
from typing import Callable
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import networkx as nx
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from rich import print as rprint

from data_transform import hsi_pca


class PatienceLogger:
    def __init__(self, patience, min_delta=1e-5):
        """
        Initializes the logger with a specified patience and minimum delta for improvement.

        :param patience: Number of epochs to wait after the last significant improvement.
        :param min_delta: Minimum change in the loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_params = None
        self.best_epoch = None
        self.epochs_without_improvement = 0
        self.stop_training = False

    def log(self, epoch, loss, params):
        """
        Logs the loss for a given epoch and updates the best parameters if the loss improved significantly.

        :param epoch: Current epoch number.
        :param loss: Loss for the current epoch.
        :param params: Parameters for the current epoch.
        """
        # Check if the improvement is significant
        if self.best_loss - loss < self.min_delta:
            self.epochs_without_improvement += 1
        else:
            self.epochs_without_improvement = 0
        # Update the best loss, parameters, and epoch if the current loss is lower than the best recorded loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params
            self.best_epoch = epoch

        # Check if training should stop (don't stop if loss is nan)
        if self.epochs_without_improvement >= self.patience and not torch.isnan(loss):
            self.stop_training = True


def standardize(x: torch.Tensor, return_stats: bool = False) -> torch.Tensor:
    """Given a 2d tensor, standardize both column to have mean 0, and set the distance
    to origin to 1.
    """
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    if not return_stats:
        return (x - mean) / std
    else:
        return (x - mean) / std, mean, std


# def standardize(x: torch.Tensor, return_stats: bool = False) -> torch.Tensor:
#     if not return_stats:
#         return (x - x.mean()) / x.std()
#     else:
#         return (x - x.mean()) / x.std(), x.mean(), x.std()


def loss_function(
    params: torch.Tensor,
    keypointsA: torch.Tensor,
    keypointsB: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss as average distance between transformed keypointsA and nearest
    keypointsB, with penalties. Note that in this setup `keypointB` acts as the
    reference.

    Loss components are returned as float number in tensor graph.
    """
    a, b, tx, ty, sx, sy = params
    transformedA = affine_transform(params, keypointsA)
    # [n, m], where n is the number of keypoints in A and m is the number of keypoints in B.
    cdist = torch.cdist(transformedA, keypointsB)
    min_distances, _ = torch.min(cdist, dim=1)

    # Reward correct pairing. Correct pairing is identified by mutual nearest neighbor.
    # Pairing could also be achieved by thresholding ratio of distances between nearest
    # and second nearest neighbors.
    mnn_scores = soft_mnn_consistency(cdist)

    # Regularization terms
    norm = torch.sqrt(a**2 + b**2)
    rotation_penalty = (b / norm) ** 2 + (1 - a) ** 2 + b**2  # Penalty on rotation
    # Penalty on difference between sx and sy
    stretching_penalty = (sx - sy) ** 2 + (1 - sx) ** 2 + (1 - sy) ** 2
    # min_distances: [n]
    # mnn_scores: [n]
    # rotation_penalty: float
    # stretching_penalty: float
    return (
        (min_distances * weight).mean(),
        (-mnn_scores * weight).mean(),
        rotation_penalty,
        stretching_penalty,
    )


def soft_mnn_consistency(cdist: torch.Tensor, temperature: float = 1) -> torch.Tensor:
    """Compute consistency score between two arrays of keypoints.

    The consistency score is defined as the average of the softmax of the negative
        distances between the nearest neighbor of each keypoint in the first array and
        the corresponding keypoint in the second array.

    The softmax is computed with a temperature parameter.
    """
    # cdist: [n, m], such that the first array has n data points and the second array
    # has m data points.
    # return a consistency score

    # Negative softmax to get weights (higher weight for smaller distances)
    # if cdist.shape[0] > cdist.shape[1]:
    #     cdist = cdist.t()
    weights_A_to_B = (-cdist / temperature).softmax(dim=1)  # [n, m]
    weights_B_to_A = (-cdist / temperature).softmax(dim=0)  # [n, m]

    # Get the most likely match from array 1 to array 2 for each data point in array 1
    _, max_indices_A_to_B = weights_A_to_B.max(dim=1)
    consistency_scores = weights_B_to_A[
        torch.arange(len(max_indices_A_to_B)), max_indices_A_to_B
    ]
    # consistency_scores *=
    # consistency_scores = (weights_B_to_A * weights_A_to_B)
    return consistency_scores


def affine_transform(params: torch.Tensor, keypoints: torch.Tensor):
    """Apply affine transformation to keypoints. The transformation consists of:
    - rotation and scaling around origin (hence order doesn't matter)
    - translation
    """
    a, b, tx, ty, sx, sy = params
    norm = torch.sqrt(a**2 + b**2)
    sin_t, cos_t = b / norm, a / norm
    # scaling + rotation, then translation
    # rotation_matrix = torch.stack(
    #     [torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])], dim=0
    # )
    # scaling_matrix = np.diag(torch.stack([sx, sy]))
    # transformed_keypoints = keypoints @ rotation_matrix @ scaling_matrix + torch.stack(
    #     [tx, ty]
    # )
    transformed_keypoints = keypoints @ torch.stack(
        [torch.stack([sx * cos_t, -sy * sin_t]), torch.stack([sx * sin_t, sy * cos_t])],
        dim=0,
    ) + torch.stack([tx, ty])
    # translation, rotation, then scaling
    # transformed_keypoints = (
    #     (keypoints + torch.stack([tx, ty]))
    #     @ torch.stack(
    #         [torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])],
    #         dim=0,
    #     )
    #     * torch.tensor([sx, sy])
    # )
    return transformed_keypoints


def affine_transform_rev(params: torch.Tensor, keypoints: torch.Tensor):
    """Apply the reverse affine transformation to keypoints, i.e., the reverse
    transformation of `affine_transform`.

    The transformation consistsof:
    - translation
    - rotation and scaling around origin (hence order doesn't matter)
    """
    a, b, tx, ty, sx, sy = params
    norm = torch.sqrt(a**2 + b**2)
    # Same as forward because rotation matrix is orthogonal
    sin_t, cos_t = b / norm, a / norm

    # Inverse translation
    keypoints_translated_back = keypoints - torch.stack([tx, ty])

    # Inverse rotation and scaling
    sx_inv, sy_inv = 1 / sx, 1 / sy  # Inverse scaling factors
    transformed_keypoints_rev = keypoints_translated_back @ torch.stack(
        [
            torch.stack([sx_inv * cos_t, sx_inv * sin_t]),
            torch.stack([-sy_inv * sin_t, sy_inv * cos_t]),
        ],
        dim=0,
    )
    # rotation_matrix = torch.stack(
    #     [torch.stack([cos_t, sin_t]), torch.stack([-sin_t, cos_t])], dim=0
    # )
    # scaling_matrix = torch.diag(torch.stack([sx_inv, sy_inv]))
    # transformed_keypoints_rev = (
    #     keypoints_translated_back @ scaling_matrix @ rotation_matrix
    # )

    return transformed_keypoints_rev


def _affine_transform_equation(params: torch.Tensor, flip: tuple[bool, bool]) -> str:
    # turn into string, 3 decimal places
    a, b, tx, ty, sx, sy = params

    if flip[0] and flip[1]:
        flip_str = "Flip horizontally and vertically"
    elif flip[0]:
        flip_str = "Flip horizontally"
    elif flip[1]:
        flip_str = "Flip vertically"
    else:
        flip_str = "No flip"

    # angle in degrees
    angle = torch.atan(b / a) * 180 / torch.pi
    rotation_angle = f"Rotate around origin clockwise: {angle:.3f}°"
    scale = f"Scale along axes: ({sx:.3f}, {sy:.3f})"
    translate = f"Translate: ({tx:.3f}, {ty:.3f})"
    if abs(angle) > 5:
        # warn about potential misalignment and mark with dark red color using rich
        rotation_angle += " [magenta](big rotation)[/magenta]"
    if max(np.log10(sx).abs(), np.log10(sy).abs()) > np.log10(1.3):
        scale += " [magenta](big scaling)[/magenta]"
    if np.log10(sx / sy).abs() > np.log10(1.05):
        scale += " [magenta](nonisotropic scaling)[/magenta]"
    return flip_str, rotation_angle, scale, translate


def _load_coco_to_contour(coco_annot_path: dict) -> list[np.ndarray]:
    """Convert COCO format polygon segmentation annotation to contours of opencv
    format.
    """
    with open(coco_annot_path, "r") as f:
        data = json.load(f)
    return [
        np.array(
            [
                [x, y]
                for x, y in zip(
                    annotation["segmentation"][0][::2],
                    annotation["segmentation"][0][1::2],
                )
            ]
        )
        for annotation in data["annotations"]
    ]


def get_query2target_func(
    a: torch.Tensor,
    b: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    flip: tuple[bool, bool] = (False, False),
) -> Callable:
    def ret(query: np.ndarray) -> np.ndarray:
        query_std = torch.from_numpy((query - mean_q) / std_q).float()
        query_mapped_std = affine_transform(
            torch.tensor([a, b, tx, ty, sx, sy]), flip_tensor(query_std, *flip)
        )
        mapped_std = query_mapped_std * std_t + mean_t
        return mapped_std.numpy()

    return ret


def get_query2target_func_rev(
    a: torch.Tensor,
    b: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    flip: tuple[bool, bool] = (False, False),
) -> Callable:
    """Return a function that maps target to query, i.e., the reverse transformation
    of `get_query2target_func`.
    """

    def ret(target: np.ndarray) -> np.ndarray:
        target_std = torch.from_numpy((target - mean_t) / std_t).float()
        target_mapped_std = affine_transform_rev(
            torch.tensor([a, b, tx, ty, sx, sy]), flip_tensor(target_std, *flip)
        )
        mapped_std = target_mapped_std * std_q + mean_q
        return mapped_std.numpy()

    return ret


def flip_tensor(t: torch.Tensor, flip_h: bool, flip_v: bool) -> torch.Tensor:
    t = t.clone()
    if flip_h:
        t[:, 0] *= -1
    if flip_v:
        t[:, 1] *= -1
    return t


def find_affine(
    query: np.ndarray,
    target: np.ndarray,
    weighted_by: str = "uniform",
    hparams: dict[str, float] = None,
    log: int = 1,
    flip: bool = False,
    mean_q: np.ndarray = None,
    std_q: np.ndarray = None,
    mean_t: np.ndarray = None,
    std_t: np.ndarray = None,
) -> tuple[np.ndarray, float, float, float, float]:
    query = torch.from_numpy(query).float()
    target = torch.from_numpy(target).float()

    if mean_q is None:
        query_rescaled, mean_q, std_q = standardize(query, return_stats=True)
    else:
        mean_q, std_q = (
            torch.from_numpy(mean_q).float(),
            torch.from_numpy(std_q).float(),
        )
        query_rescaled = (query - mean_q) / std_q
    if mean_t is None:
        target_rescaled, mean_t, std_t = standardize(target, return_stats=True)
    else:
        mean_t, std_t = (
            torch.from_numpy(mean_t).float(),
            torch.from_numpy(std_t).float(),
        )
        target_rescaled = (target - mean_t) / std_t

    if weighted_by == "uniform":
        weight = torch.ones_like(query_rescaled[:, 0])
    elif weighted_by == "centrality":
        # weight by distance to origin of query data points after scaling
        weight = query_rescaled[:, 0].abs() + query_rescaled[:, 1].abs()

    iterator = product([True, False], repeat=2) if flip else [(False, False)]
    # iterator = [(False, True)]
    res = {}
    for h_flip, v_flip in iterator:
        logger = _find_affine(
            flip_tensor(query_rescaled, h_flip, v_flip),
            target_rescaled,
            weight,
            log == 2,
            hparams,
        )
        res[(h_flip, v_flip)] = logger
    # find the lowest loss
    best_flip = min(res, key=lambda x: res[x].best_loss)
    logger = res[best_flip]
    epoch = logger.best_epoch
    params = logger.best_params

    if log >= 1:
        rprint(f"Optimized Parameters at Epoch {epoch}:")
        rprint(
            *_affine_transform_equation(params.detach(), best_flip),
            sep="; ",
        )

    # the transformation that goes from query to standardardized query
    return (
        params.detach().numpy(),
        mean_q.numpy(),
        std_q.numpy(),
        mean_t.numpy(),
        std_t.numpy(),
        best_flip,
    )


def _find_affine(
    query, target, weight, log, hparams: dict[str, float]
) -> PatienceLogger:
    # default hyperparameters
    # lr = 0.005
    # max_epochs = 10000
    # patience = 100
    # beta_d = 0.2
    # beta_c = 5
    # beta_t = 1
    # beta_s = 0
    if hparams is None:
        hparams = {}
    lr = hparams.get("lr", 0.002)
    max_epochs = hparams.get("max_epochs", 10000)
    patience = hparams.get("patience", 100)
    beta_d = hparams.get("beta_d", 0.2)
    beta_c = hparams.get("beta_c", 5)
    beta_t = hparams.get("beta_t", 1)
    beta_s = hparams.get("beta_s", 0)

    # Initialization
    # [a, b, tx, ty, sx, sy]
    params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], requires_grad=True)
    optimizer = optim.Adam([params], lr=lr)

    logger = PatienceLogger(patience)
    epoch = 0
    while epoch < max_epochs:
        optimizer.zero_grad()
        nn_loss, mnn_loss, rot_loss, stre_loss = loss_function(
            params, query, target, weight=weight
        )
        loss = (
            nn_loss * beta_d
            + mnn_loss * beta_c
            + rot_loss * beta_t
            + stre_loss * beta_s
        )
        loss.backward()
        optimizer.step()
        logger.log(epoch, loss, params)

        if epoch and epoch % 100 == 0 and log:
            rprint(
                f"Epoch {epoch}, Loss: {loss.item():.2f}, "
                f"Loss comps: {nn_loss.item():.2f}, {mnn_loss.item():.2f}, "
                f"{rot_loss.item():.2f}, {stre_loss.item():.2f}"
            )
        epoch += 1
        if logger.stop_training:
            break
    # else:
    #     # if the loop completes without breaking, the training is considered not
    #     # converged, but parameters at the last epoch are still returned.

    return logger


def find_mutual_pairs(array_q: np.ndarray, array_t: np.ndarray) -> np.ndarray:
    """Given two arrays of keypoints, find mutual nearest neighbors and return indices
    of mutual nearest neighbors in the second array for each keypoint in the first
    array. If no mutual nearest neighbor is found, the index is -1.
    """
    min_weight = 0.1
    max_dist = 15
    temperature = 20

    dist_q_t = torch.cdist(
        torch.from_numpy(array_q).float(), torch.from_numpy(array_t).float()
    )
    weight_q2t = (-dist_q_t / temperature).softmax(dim=1)
    weight_t2q = (-dist_q_t / temperature).softmax(dim=0)

    weights_fwd, best_pairs = weight_q2t.max(dim=1)
    weights_rev = weight_t2q.max(dim=0)[0]

    mutual_pairs = np.full(len(array_q), -1)
    for i, (w, p) in enumerate(zip(weights_fwd, best_pairs)):
        w_rev = weight_t2q[i, p]
        if (
            w > min_weight
            and w_rev > min_weight
            and dist_q_t[i, p] < max_dist
            and torch.isclose(weights_rev[p], w_rev, atol=1e-3)
        ):
            mutual_pairs[i] = p
    return mutual_pairs


def align(
    center_rgb: np.ndarray,
    center_hsi: np.ndarray,
    center_taxa: np.ndarray,
    weighted_by: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsi2rgb_param, *hsi2rgb_stats = find_affine(center_hsi, center_rgb, weighted_by)
    hsi2rgb_func = get_query2target_func(*hsi2rgb_param, *hsi2rgb_stats)
    center_hsi2rgb = hsi2rgb_func(center_hsi)
    map_hsi = find_mutual_pairs(center_rgb, center_hsi2rgb)

    taxa2rgb_param, *taxa2rgb_stats = find_affine(
        center_taxa, center_rgb, weighted_by="uniform"
    )
    taxa2rgb_func = get_query2target_func(*taxa2rgb_param, *taxa2rgb_stats)
    center_taxa2rgb = taxa2rgb_func(center_taxa)
    map_taxa = find_mutual_pairs(center_rgb, center_taxa2rgb)

    return center_hsi2rgb, center_taxa2rgb, map_hsi, map_taxa


def map_to_network(
    map_t2b: list[int], map_b2t: list[int], name_top: str = "A", name_bottom="B"
) -> nx.Graph:
    """
    Constructs a directed network from two lists of integer indices.
    map_t2b: List of target indices in B for each node in A (-1 for no target).
    map_b2t: List of target indices in A for each node in B (-1 for no target).
    """
    G = nx.DiGraph()
    G.add_nodes_from([f"{name_top}_{i}" for i in range(len(map_t2b))])
    G.add_nodes_from([f"{name_bottom}_{i}" for i in range(len(map_b2t))])
    G.add_edges_from(
        [
            (f"{name_top}_{i}", f"{name_bottom}_{target}")
            for i, target in enumerate(map_t2b)
            if target != -1
        ]
    )
    G.add_edges_from(
        [
            (f"{name_bottom}_{i}", f"{name_top}_{target}")
            for i, target in enumerate(map_b2t)
            if target != -1
        ]
    )
    G.name_top = name_top
    G.name_bottom = name_bottom
    G.num_top_nodes = len(map_t2b)
    G.num_bottom_nodes = len(map_b2t)

    return G


def network_to_map(G: nx.Graph) -> dict[str, np.ndarray]:
    """
    Converts a directed network to two lists of integer indices.
    """
    top = G.name_top
    bottom = G.name_bottom
    map_t2b = [-1] * G.num_top_nodes
    map_b2t = [-1] * G.num_bottom_nodes
    name2idx = lambda x: int(x.split("_")[-1])
    for node in G.nodes:
        if node.startswith(top):
            arr = map_t2b
        else:
            arr = map_b2t
        try:
            arr[name2idx(node)] = name2idx(next(G.successors(node)))
        except StopIteration:
            pass
    return {top: np.array(map_t2b), bottom: np.array(map_b2t)}


def remove_bad_nodes(G: nx.Graph, remove_from: str) -> tuple[nx.Graph, list[str]]:
    """
    Removes all edges connected to bad nodes from the specified list (A or B) in the network.
    A bad node is one whose target in the other list is also targeted by other nodes.
    remove_from: 'A' to remove edges connected to bad nodes from A, 'B' to remove edges from B.
    """
    G = deepcopy(G)
    top = G.name_top
    bottom = G.name_bottom
    the_other_set = [top, bottom][int(remove_from == top)]
    bad_nodes = {
        start
        for node, degree in G.in_degree()
        if node.startswith(the_other_set) and degree > 1
        for start, _ in G.in_edges(node)
    }
    # remove bad nodes and add back
    G.remove_nodes_from(bad_nodes)
    G.add_nodes_from(bad_nodes)

    return G, [int(node.split("_")[-1]) for node in bad_nodes]


class Aligner:
    modalities = ["rgb", "hsi", "isolate"]
    modality2marker = {"rgb": "o", "hsi": "x", "isolate": "^"}
    modality2color = {"rgb": "tab:orange", "hsi": "tab:green", "isolate": "tab:pink"}
    modality2label = {
        "rgb": "RGB colony center",
        "hsi": "HSI colony center",
        "isolate": "Picked isolates",
    }
    modality2mfc = {"rgb": "none", "hsi": None, "isolate": "none"}

    def __init__(
        self,
        plate_barcode: str = "default",
        rgb_png_path: str = None,
        hsi_png_path: str = None,
        hsi_npz_path: str = None,
        hsi_npz_crop: tuple[tuple[int, int], tuple[int, int]] = None,
        rgb_meta_path: str = None,  # csv
        hsi_meta_path: str = None,  # csv
        rgb_coco_contour_path: str = None,  # json
        hsi_coco_contour_path: str = None,  # json
        zotu_count_files: list[str] | str = None,  # tsv
        isolate_metadata_file: str = None,  # tsv
    ):
        self.plate_barcode = plate_barcode
        self.rgb_png_path = rgb_png_path
        self.hsi_png_path = hsi_png_path
        self.hsi_npz_path = hsi_npz_path
        self.hsi_npz_crop = hsi_npz_crop
        self.rgb_meta_path = rgb_meta_path
        self.hsi_meta_path = hsi_meta_path
        self.rgb_coco_contour_path = rgb_coco_contour_path
        self.hsi_coco_contour_path = hsi_coco_contour_path
        self.zotu_count_files = zotu_count_files
        self.isolate_metadata_file = isolate_metadata_file

    # def get_coords(self, modality: str) -> np.ndarray:
    #     if modality == "rgb":
    #         return self.rgb_meta[["center_x", "center_y"]]
    #     elif modality == "hsi":
    #         return self.hsi_meta[["center_x", "center_y"]]
    #     elif modality == "isolate":
    #         return self.isolate_meta[["src_x", "src_y"]]
    #     else:
    #         raise ValueError("modality should be one of ['rgb', 'hsi', 'isolate']")

    @property
    def pic_rgb(self) -> np.ndarray:
        if not hasattr(self, "_pic_rgb"):
            _pic_rgb = cv.imread(self.rgb_png_path)
            if _pic_rgb is None:
                raise ValueError("rgb_png_path is not specified")
            _pic_rgb = cv.cvtColor(_pic_rgb, cv.COLOR_BGR2RGB)
            self._pic_rgb = _pic_rgb
        return self._pic_rgb

    @property
    def pic_hsi(self) -> np.ndarray:
        """Retrive .png picture derived from HSI data. If not available, apply PCA on
        HSI array.
        """
        if not hasattr(self, "_pic_hsi"):
            if self.hsi_png_path is not None:
                _pic_hsi = cv.imread(self.hsi_png_path)
                _pic_hsi = cv.cvtColor(_pic_hsi, cv.COLOR_BGR2RGB)
            else:
                _pic_hsi = None

            if _pic_hsi is None:
                msg = "WARNING: hsi_png_path is not specified, applying PCA on HSI "
                "array as HSI picture"
                print(msg)
                # if self.hsi_png_path is not None:
                #     msg += f"and saving to {self.hsi_png_path}"
                #     save_path = self.hsi_png_path
                # else:
                #     save_path = tempfile.mktemp(suffix=".png")
                if self.arr_hsi is None:
                    raise ValueError("Cannot find either hsi_png_path or hsi_npz_path.")
                from data_transform import hsi_pca

                _pic_hsi, _ = hsi_pca(self.arr_hsi)
            self._pic_hsi = _pic_hsi
        return self._pic_hsi

    @property
    def arr_hsi(self) -> np.ndarray:
        if not hasattr(self, "_arr_hsi"):
            if self.hsi_npz_path is None:
                raise ValueError("hsi_npz_path is not specified")
            _arr_hsi = np.load(self.hsi_npz_path)["data"]
            if self.hsi_npz_crop is not None:
                _arr_hsi = _arr_hsi[
                    slice(*self.hsi_npz_crop[0]), slice(*self.hsi_npz_crop[1])
                ].copy()
            self._arr_hsi = _arr_hsi
        return self._arr_hsi

    @property
    def metadata_rgb(self) -> pd.DataFrame:
        if not hasattr(self, "_meta_rgb"):
            if self.rgb_meta_path is None:
                if self.rgb_png_path is None:
                    raise ValueError(
                        "Neither rgb_meta_path nor rgb_png_path is specified, "
                        "cannot load metadata."
                    )
                else:
                    print(
                        "WARNING: RGB colony metadata not available, detecting colonies "
                        "using the picture."
                    )
                    from detect_colonies import detect_colony_single

                    output_dir = tempfile.mkdtemp()
                    detect_colony_single(self.rgb_png_path, output_dir)
                    self.rgb_meta_path = glob.glob(f"{output_dir}/*_metadata.csv")[0]
            _meta_rgb = pd.read_csv(self.rgb_meta_path)
            if "picking_status" in _meta_rgb.columns:
                _meta_rgb = _meta_rgb.query("picking_status == 1")
            self._meta_rgb = _meta_rgb
        return self._meta_rgb

    @property
    def metadata_hsi(self) -> pd.DataFrame:
        """Similar to self.metadata_rgb, but for HSI data."""
        if not hasattr(self, "_meta_hsi"):
            if self.hsi_meta_path is None:
                if self.hsi_png_path is None:
                    raise ValueError(
                        "Neither hsi_meta_path nor hsi_png_path is specified, "
                        "cannot load metadata."
                    )
                else:
                    print(
                        "WARNING: HSI colony metadata not available, detecting colonies "
                        "using the picture."
                    )
                    from detect_colonies import detect_colony_single

                    output_dir = tempfile.mkdtemp()
                    detect_colony_single(self.hsi_png_path, output_dir)
                    self.hsi_meta_path = glob.glob(f"{output_dir}/*_metadata.csv")[0]
            _meta_hsi = pd.read_csv(self.hsi_meta_path)
            if "picking_status" in _meta_hsi.columns:
                _meta_hsi = _meta_hsi.query("picking_status == 1")
            self._meta_hsi = _meta_hsi
        return self._meta_hsi

    @property
    def metadata_isolate(self) -> pd.DataFrame:
        if not hasattr(self, "_meta_isolate"):
            if self.isolate_metadata_file is None:
                raise ValueError("isolate_metadata_file is not specified")
            self._meta_isolate = pd.read_table(
                self.isolate_metadata_file, index_col="sample"
            ).query("src_plate == @self.plate_barcode")
        return self._meta_isolate

    @property
    def coords_isolate(self) -> np.ndarray:
        return self.metadata_isolate[["src_x", "src_y"]].to_numpy()

    @property
    def coords_rgb(self) -> np.ndarray:
        return self.metadata_rgb[["center_x", "center_y"]].to_numpy()

    @property
    def coords_hsi(self) -> np.ndarray:
        return self.metadata_hsi[["center_x", "center_y"]].to_numpy()

    @property
    def contours_rgb(self) -> list[np.ndarray]:
        if not hasattr(self, "_contours_rgb"):
            if self.rgb_coco_contour_path is None:
                if self.rgb_png_path is None:
                    raise ValueError(
                        "Neither rgb_coco_contour_path nor rgb_png_path is specified, "
                        "cannot load contours."
                    )
                else:
                    print(
                        "WARNING: RGB colony contours not available, detecting colonies "
                        "using the picture."
                    )
                    from detect_colonies import detect_colony_single

                    output_dir = tempfile.mkdtemp()
                    detect_colony_single(self.rgb_png_path, output_dir)
                    self.rgb_coco_contour_path = glob.glob(
                        f"{output_dir}/*_annot.json"
                    )[0]
            _contours_rgb = _load_coco_to_contour(self.rgb_coco_contour_path)
            if "picking_status" in self.metadata_rgb.columns:
                _contours_rgb = [
                    c
                    for c, status in zip(
                        _contours_rgb, self.metadata_rgb["picking_status"]
                    )
                    if status == 1
                ]
            self._contours_rgb = _contours_rgb
        return self._contours_rgb

    @property
    def contours_hsi(self) -> list[np.ndarray]:
        if not hasattr(self, "_contours_hsi"):
            if self.hsi_coco_contour_path is None:
                if self.hsi_png_path is None:
                    raise ValueError(
                        "Neither hsi_coco_contour_path nor hsi_png_path is specified, "
                        "cannot load contours."
                    )
                else:
                    print(
                        "WARNING: HSI colony contours not available, detecting colonies "
                        "using the picture."
                    )
                    from detect_colonies import detect_colony_single

                    output_dir = tempfile.mkdtemp()
                    detect_colony_single(self.hsi_png_path, output_dir)
                    self.hsi_coco_contour_path = glob.glob(
                        f"{output_dir}/*_annot.json"
                    )[0]
            _contours_hsi = _load_coco_to_contour(self.hsi_coco_contour_path)
            if "picking_status" in self.metadata_hsi.columns:
                _contours_hsi = [
                    c
                    for c, status in zip(
                        _contours_hsi, self.metadata_hsi["picking_status"]
                    )
                    if status == 1
                ]
            self._contours_hsi = _contours_hsi
        return self._contours_hsi

    def fit(
        self,
        query: str,
        target: str,
        hparams: dict[str, float] = None,
        flip: bool = True,
        mean_q: np.ndarray = None,
        std_q: np.ndarray = None,
        mean_t: np.ndarray = None,
        std_t: np.ndarray = None,
        log: int = 1,
    ) -> None:
        coords_q = getattr(self, f"coords_{query}")
        coords_t = getattr(self, f"coords_{target}")
        # if query and target are rgb and hsi or the other way around, and both have
        # picture, normalize them using the picture size so that the longer axis is 3.
        if set([query, target]) == {"rgb", "hsi"}:
            pic_q = getattr(self, f"pic_{query}")
            pic_t = getattr(self, f"pic_{target}")
            mean_q = np.array(pic_q.shape[:2][::-1]) / 2
            mean_t = np.array(pic_t.shape[:2][::-1]) / 2
            std_q = np.array(pic_q.shape[:2][::-1]) / 3
            std_t = np.array(pic_t.shape[:2][::-1]) / 3
        if (query, target) == ("isolate", "rgb"):
            robot_factor = 0.066
            mean_q = coords_t.mean(axis=0) * robot_factor
            std_q = coords_t.std(axis=0) * robot_factor

        q2t_params, *q2t_stats, q2t_flip = find_affine(
            coords_q,
            coords_t,
            log=log,
            flip=flip,
            mean_q=mean_q,
            std_q=std_q,
            mean_t=mean_t,
            std_t=std_t,
            hparams=hparams,
        )
        q2t_func = get_query2target_func(*q2t_params, *q2t_stats, q2t_flip)
        t2q_func = get_query2target_func_rev(*q2t_params, *q2t_stats, q2t_flip)
        setattr(self, f"_func_{target}_{query}2{target}", q2t_func)
        setattr(self, f"_func_{target}_{target}2{query}", t2q_func)

    def transform(self, query: str, target: str) -> np.ndarray:
        coords_q = getattr(self, f"coords_{query}")
        coords_t = getattr(self, f"coords_{target}")
        q2t_func = getattr(self, f"_func_{target}_{query}2{target}")
        coords_q2t = q2t_func(coords_q)
        map_t2q = find_mutual_pairs(coords_t, coords_q2t)
        map_q2t = find_mutual_pairs(coords_q2t, coords_t)
        g = map_to_network(
            map_t2q,
            map_q2t,
            name_top=f"{target}_{target}",
            name_bottom=f"{target}_{query}",
        )
        h, bad_target_idx = remove_bad_nodes(g, g.name_top)
        map_q2t_clean = network_to_map(h)[g.name_bottom]
        setattr(self, f"_coords_{query}2{target}", coords_q2t)
        setattr(self, f"_map_{target}_{query}2{target}", map_q2t)
        setattr(self, f"_map_{target}_{target}2{query}", map_t2q)
        setattr(self, f"_graph_{target}_{query}", g)
        setattr(self, f"_graph_{target}_{query}_clean", h)
        setattr(self, f"_bad_{target}_{query}2{target}_idx", bad_target_idx)
        setattr(self, f"_map_{target}_{query}2{target}_clean", map_q2t_clean)

    def agg(self, query: str, target: str, data: pd.DataFrame) -> pd.DataFrame:
        """Take a DataFrame with rows corresponding to the query modality, and aggregate
        by adding so that in the returned dataframe, each row corresponds to the target.
        Index of the returned dataframe is retrieved from the target modality.

        Raise error indicating `transform` must be called before `agg` if necessary
        attributes are not found.
        """
        try:
            map_q2t_clean = getattr(self, f"_map_{target}_{query}2{target}_clean")
        except AttributeError:
            raise AttributeError(
                f"`transform` query modality {query} to target modality {target} before `agg`."
            )
        # reorder data according to metadata of query modality
        good_query = map_q2t_clean != -1
        data = data.loc[getattr(self, f"metadata_{query}").iloc[good_query].index]
        ind = getattr(self, f"metadata_{target}").index.to_numpy()
        data_agg = data.groupby(ind[map_q2t_clean[good_query]]).sum()
        # fill in missing rows with 0
        data_agg = data_agg.reindex(ind, fill_value=0)
        return data_agg

    def crop(
        self,
        modality: str,
        index: int,
        size: int = 0,
        padding: int | float = 0,
        modality_t: str = None,
        add_contour: str = "none",
        _use_arr: bool = False,
    ) -> np.ndarray:
        """Crop out a square image patch around the index-th contour center of the
        specified modality.

        Padding could be added to the patch when size is not given.
            If padding is 0, no padding is added, the patch is cropped such that it's
                the smallest square that includes all pixels within contour.
            When padding is int, it is the number of pixels to add to each side of the
                patch.
            When padding is float, it is the fraction of the patch size to add to each
                side of the patch.

        Contour of selected colony or all colonies could be added. Convineint for
            plotting.
        """
        if modality_t == "isolate":
            raise ValueError("Isolate has no background image or contours to crop.")
        if not modality in self.modalities:
            raise ValueError(
                f"modality should be one of {self.modalities}, got {modality}"
            )
        if modality_t is None:
            modality_t = modality
        if modality == modality_t:
            contours = getattr(self, f"contours_{modality}")
            x, y, w, h = cv.boundingRect(contours[index])
            size = max(w, h) if not size else size
            center = np.array([x + w / 2, y + h / 2])
        else:
            if not size:
                raise ValueError(
                    "size should be given when modality is not the reference modality."
                )
            else:
                center = getattr(self, f"_func_{modality_t}_{modality}2{modality_t}")(
                    getattr(self, f"coords_{modality}")[index].reshape(1, -1)
                ).flatten()

        if isinstance(padding, int):
            size += 2 * padding
        elif isinstance(padding, float):
            size *= 1 + padding
        size = int(size)
        ret = getattr(self, f"pic_{modality_t}").copy()
        if modality == modality_t:
            if add_contour == "self":
                cv.drawContours(ret, contours, index, (0, 0, 0), 1)
                mask = None
            elif add_contour == "all":
                cv.drawContours(ret, contours, -1, (0, 0, 0), 1)
                mask = None
            elif add_contour == "mask":  # mask as -1 except for the selected contour
                mask = np.zeros(ret.shape[:2], dtype=np.uint8)
                cv.fillPoly(mask, [contours[index]], 1)
                ret = ret.astype(int)
                ret[mask == 0] = -1
            else:
                mask = None
        else:
            if add_contour != "none":
                raise ValueError(
                    "add_contour should be 'none' when modality is not the reference."
                )
            mask = None
        ret = self._crop_at(ret, center, size)

        if (
            ((not modality_t) and modality == "hsi") or modality_t == "hsi"
        ) and _use_arr:
            ret_arr = self._crop_at(self.arr_hsi, center, size).copy()
            if mask is not None:
                ret_arr = ret_arr.astype(int)
                mask = self._crop_at(mask, center, size)
                ret_arr[mask == 0] = -1
            ret = ret_arr
        return ret

    @staticmethod
    def _crop_at(arr: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
        x, y = center
        x_start = max(0, x - size / 2)
        y_start = max(0, y - size / 2)
        x_start = int(np.round(x_start))
        y_start = int(np.round(y_start))
        x_end = x_start + size
        y_end = y_start + size
        x_end = min(arr.shape[1], x_end)
        y_end = min(arr.shape[0], y_end)
        return arr[y_start:y_end, x_start:x_end]

    def plot(
        self,
        modality_ref: str,
        modalities_other: list[str] | str = (),
        marker_size: int = 4,
        modality2color: dict[str, str] = None,
        modality2marker: dict[str, str] = None,
        modality2label: dict[str, str] = None,
        plot_bg: bool = True,
        plot_contours: bool = False,  # whether to plot contours of reference modality
        contour_width: int = 2,
        *,
        ax: plt.Axes,
    ) -> None:
        if isinstance(modalities_other, str):
            modalities_other = [modalities_other]
        modalities_other = list(modalities_other)
        # argument sanity check
        for m in [modality_ref] + modalities_other:
            if not m in self.modalities:
                raise ValueError(
                    f"Modalities should be a list of {self.modalities}, got {m}"
                )
        if modality_ref == "isolate":
            if plot_bg or plot_contours:
                raise ValueError("Isolate has no background image or contours to show.")
        if modality2color is None:
            modality2color = self.modality2color
        if modality2marker is None:
            modality2marker = self.modality2marker
        if modality2label is None:
            modality2label = self.modality2label

        if plot_bg:
            img = getattr(self, f"pic_{modality_ref}").copy()
            if plot_contours:
                cv.drawContours(
                    img,
                    getattr(self, f"contours_{modality_ref}"),
                    -1,
                    (0, 0, 0),
                    contour_width,
                )
            ax.imshow(img)

        coods_list = [getattr(self, f"coords_{modality_ref}")] + [
            getattr(self, f"_coords_{modality}2{modality_ref}")
            for modality in modalities_other
        ]
        for modality, coords in zip([modality_ref] + modalities_other, coods_list):
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                c=modality2color[modality],
                marker=modality2marker[modality],
                label=modality2label[modality],
                linestyle="none",
                ms=marker_size,
                mew=marker_size / 5,
                mfc=self.modality2mfc[modality],
                alpha=0.9,
            )
        ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
