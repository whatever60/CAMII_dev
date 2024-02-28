#!/usr/bin/env python

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


def soft_mnn_consistency(cdist: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
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
    log: bool = True,
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

    iterator = product([True, False], repeat=2) if flip else (False, False)
    # iterator = [(False, True)]
    res = {}
    for h_flip, v_flip in iterator:
        logger = _find_affine(
            flip_tensor(query_rescaled, h_flip, v_flip),
            target_rescaled,
            weight,
            log,
            hparams,
        )
        res[(h_flip, v_flip)] = logger
    # find the lowest loss
    best_flip = min(res, key=lambda x: res[x].best_loss)
    logger = res[best_flip]
    epoch = logger.best_epoch
    params = logger.best_params

    if log:
        rprint(f"\tOptimized Parameters at Epoch {epoch}:", end="\n\t")
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
    lr = hparams.get("lr", 0.005)
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
            params = logger.best_params
            epoch = logger.best_epoch
            loss = logger.best_loss
            break
    return logger


def find_mutual_pairs(array_q: np.ndarray, array_t: np.ndarray) -> np.ndarray:
    """Given two arrays of keypoints, find mutual nearest neighbors and return indices
    of mutual nearest neighbors in the second array for each keypoint in the first
    array. If no mutual nearest neighbor is found, the index is -1.
    """
    min_weight = 0.2
    max_dist = 15
    temperature = 20

    dist_q_t = torch.cdist(
        torch.from_numpy(array_q).float(), torch.from_numpy(array_t).float()
    )
    weight_q2t = (-dist_q_t / temperature).softmax(dim=1)
    weight_t2q = (-dist_q_t / temperature).softmax(dim=0)

    weights_fwd, best_pairs = weight_q2t.max(dim=1)
    best_pairs_rev = weight_t2q.max(dim=0)[1]

    mutual_pairs = np.full(len(array_q), -1)
    for i, (w, p) in enumerate(zip(weights_fwd, best_pairs)):
        w_rev = weight_t2q[i, p]
        p_rev = best_pairs_rev[p]
        if (
            p_rev == i
            and w > min_weight
            and w_rev > min_weight
            and dist_q_t[i, p] < max_dist
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

    return G, [int(node[1:]) for node in bad_nodes]


class Aligner:
    def __init__(
        self,
        rgb_png_path: str = None,
        hsi_npz_path: str = None,
        hsi_png_path: str = None,
        rgb_colony_meta_path: str = None,  # csv
        hsi_colony_meta_path: str = None,  # csv
        zotu_count_files: list[str] | str = None,  # tsv
        isolate_metadata_file: str = None,  # tsv
    ):
        self.rgb_png_path = rgb_png_path
        self.hsi_npz_path = hsi_npz_path
        self.hsi_png_path = hsi_png_path
        self.rgb_colony_meta_path = rgb_colony_meta_path
        self.hsi_colony_meta_path = hsi_colony_meta_path
        self.zotu_count_files = zotu_count_files
        self.isolate_metadata_file = isolate_metadata_file

    # def get_coords(self, modality: str) -> np.ndarray:
    #     if modality == "rgb":
    #         return self.rgb_colony_meta[["center_x", "center_y"]]
    #     elif modality == "hsi":
    #         return self.hsi_colony_meta[["center_x", "center_y"]]
    #     elif modality == "isolate":
    #         return self.isolate_meta[["src_x", "src_y"]]
    #     else:
    #         raise ValueError("modality should be one of ['rgb', 'hsi', 'isolate']")

    @property
    def pic_rgb(self) -> np.ndarray:
        if self._pic_rgb is None:
            _pic_rgb = cv.imread(self.rgb_png_path)
            if _pic_rgb is None:
                raise ValueError("rgb_png_path is not specified")
            _pic_rgb = cv.cvtColor(_pic_rgb, cv.COLOR_BGR2RGB)
            self._pic_rgb = _pic_rgb
        return self._pic_rgb

    @property
    def pic_hsi(self) -> np.ndarray:
        if self._pic_hsi is None:
            need_pca = False
            if self.hsi_png_path is None:
                need_pca = True
            else:
                _pic_hsi = cv.imread(self.hsi_png_path)
                if _pic_hsi is None:
                    need_pca = True
                else:
                    _pic_hsi = cv.cvtColor(_pic_hsi, cv.COLOR_BGR2RGB)

            if need_pca:
                msg = "WARNING: hsi_png_path is not specified, applying PCA on HSI "
                "array as HSI picture"
                if self.hsi_png_path is not None:
                    msg += f"and saving to {self.hsi_png_path}"
                    save_path = self.hsi_png_path
                else:
                    save_path = tempfile.mktemp(suffix=".png")
                print(msg)
                try:
                    _pic_hsi, _ = hsi_pca(self.arr_hsi)
                except ValueError:
                    raise ValueError("Cannot find either hsi_png_path or hsi_npz_path.")
                Image.fromarray((_pic_hsi * 255).astype(np.uint8)).save(save_path)

            self._pic_hsi = _pic_hsi
        return self._pic_hsi

    @property
    def arr_hsi(self) -> np.ndarray:
        if self._arr_hsi is None:
            if self.hsi_npz_path is None:
                raise ValueError("hsi_npz_path is not specified")
            self._arr_hsi = np.load(self.hsi_npz_path)["data"]
        return self._arr_hsi

    @property
    def metadata_rgb_colony(self) -> pd.DataFrame:
        if self._meta_rgb_colony is None:
            if self.rgb_colony_meta_path is None:
                raise ValueError("rgb_colony_meta_path is not specified")
            self._meta_rgb_colony = pd.read_csv(self.rgb_colony_meta_path, sep="\t")
        return self._meta_rgb_colony

    @property
    def metadata_hsi_colony(self) -> pd.DataFrame:
        if self._meta_hsi_colony is None:
            if self.hsi_colony_meta_path is None:
                raise ValueError("hsi_colony_meta_path is not specified")
            self._meta_hsi_colony = pd.read_csv(self.hsi_colony_meta_path, sep="\t")
        return self._meta_hsi_colony

    @property
    def metadata_isolate(self) -> pd.DataFrame:
        if self._meta_isolate is None:
            if self.isolate_metadata_file is None:
                raise ValueError("isolate_metadata_file is not specified")
            self._meta_isolate = pd.read_csv(self.isolate_metadata_file, sep="\t")
        return self._meta_isolate

    @property
    def coords_isolate(self) -> np.ndarray:
        return self.metadata_isolate[["src_x", "src_y"]].to_numpy()

    @property
    def coords_rgb(self) -> np.ndarray:
        return self.metadata_rgb_colony[["center_x", "center_y"]].to_numpy()

    @property
    def coords_hsi(self) -> np.ndarray:
        return self.metadata_hsi_colony[["center_x", "center_y"]].to_numpy()

    @property
    def contours_rgb(self) -> list[np.ndarray]:
        if self._contours_rgb is None:
            if self.rgb_png_path is None:
                raise ValueError("rgb_png_path is not specified")
            self._contours_rgb = _load_coco_to_contour(self.rgb_png_path)
        return self._contours_rgb

    @property
    def contours_hsi(self) -> list[np.ndarray]:
        if self._contours_hsi is None:
            if self.hsi_png_path is None:
                raise ValueError("hsi_png_path is not specified")
            self._contours_hsi = _load_coco_to_contour(self.hsi_png_path)
        return self._contours_hsi

    def align(self, query: str, target: str) -> None:
        coords_q = getattr(self, f"coords_{query}")
        coords_t = getattr(self, f"coords_{target}")
        q2t_params, *q2t_stats = find_affine(coords_q, coords_t)
        q2t_func = get_query2target_func(*q2t_params, *q2t_stats)
        coords_q2t = q2t_func(coords_q)
        map_t2q = find_mutual_pairs(coords_t, coords_q2t)
        map_q2t = find_mutual_pairs(coords_q2t, coords_t)
        g = map_to_network(
            map_t2q,
            map_q2t,
            name_top=f"{target}_{target}",
            name_bottom=f"{target}_{query}",
        )
        setattr(self, f"_coords_{query}2{target}", coords_q2t)
        setattr(self, f"_map_{target}_{query}2{target}", map_q2t)
        setattr(self, f"_map_{target}_{target}2{query}", map_t2q)
        setattr(self, f"_graph_{target}_{query}", g)

    def plot(
        self,
        modality_ref: str,
        modalities: list[str],
        marker_size: int = 6,
        modality2color: dict[str, str] = None,
        modality2marker: dict[str, str] = None,
        modality2labels: dict[str, str] = None,
        plot_bg: bool = True,
        plot_contours: bool = False,  # whether to plot contours of reference modality
        contour_width: int = 2,
        *,
        ax: plt.Axes,
    ) -> None:
        # argument sanity check
        if not modalities in ["rgb", "hsi", "isolate"]:
            raise ValueError("modalities should be a list of ['rgb', 'hsi', 'isolate']")
        if modality_ref == "isolate":
            if plot_bg or plot_contours:
                raise ValueError("Isolate has no background image or contours to show.")
        if modality2color is None:
            modality2color = {
                "rgb": "tab:blue",
                "hsi": "tab:orange",
                "isolate": "tab:green",
            }
        if modality2marker is None:
            modality2marker = {"rgb": "x", "hsi": "o", "isolate": "^"}
        if modality2labels is None:
            modality2labels = {
                "rgb": "RGB colony center",
                "hsi": "HSI colony center",
                "isolate": "Picked isolates",
            }
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
            getattr(self, f"_coords_{modality_ref}2{modality}")
            for modality in modalities
        ]
        for modality, coords in zip([modality_ref] + modalities, coods_list):
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=modality2color[modality],
                marker=modality2marker[modality],
                ms=marker_size,
                mew=marker_size / 10,
                label=modality,
            )
        ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
