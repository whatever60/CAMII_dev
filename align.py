#!/usr/bin/env python

import json
from typing import Callable

import numpy as np
import torch
import torch.optim as optim


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
        # Update the best loss, parameters, and epoch if the current loss is lower than the best recorded loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params
            self.best_epoch = epoch

        # Check if the improvement is significant
        if self.best_loss - loss > self.min_delta:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Check if training should stop (don't stop if loss is nan)
        if self.epochs_without_improvement >= self.patience and not torch.isnan(loss):
            self.stop_training = True


def standardize(x: torch.Tensor, return_stats: bool = False) -> torch.Tensor:
    if not return_stats:
        return (x - x.mean()) / x.std()
    else:
        return (x - x.mean()) / x.std(), x.mean(), x.std()


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
    stretching_penalty = (sx - sy) ** 2
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
) -> Callable:
    def ret(query: np.ndarray) -> np.ndarray:
        query_std = (query - mean_q) / std_q
        query_mapped_std = affine_transform(
            torch.tensor([a, b, tx, ty, sx, sy]), torch.from_numpy(query_std).float()
        ).numpy()
        mapped_std = query_mapped_std * std_t + mean_t
        return mapped_std

    return ret


def find_affine(
    query: np.ndarray,
    target: np.ndarray,
    weighted_by: str = "uniform",
) -> tuple[np.ndarray, float, float, float, float]:
    # hyperparameters
    lr = 0.005
    max_epochs = 10000
    patience = 100
    beta_d = 1
    beta_c = 1
    beta_t = 2
    beta_s = 2

    # Initialization
    # [a, b, tx, ty, sx, sy]
    params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], requires_grad=True)
    optimizer = optim.Adam([params], lr=lr)

    # Optimization loop
    epoch = 0

    query = torch.from_numpy(query).float()
    target = torch.from_numpy(target).float()
    # standardize
    query_rescaled, mean_q, std_q = standardize(query, return_stats=True)
    target_rescaled, mean_t, std_t = standardize(target, return_stats=True)

    # return (
    #     params.detach().numpy(),
    #     mean_q.item(),
    #     std_q.item(),
    #     mean_t.item(),
    #     std_t.item(),
    # )

    if weighted_by == "uniform":
        weight = torch.ones_like(query_rescaled[:, 0])
    elif weighted_by == "centrality":
        # weight by distance to origin of query data points after scaling
        weight = query_rescaled[:, 0].abs() + query_rescaled[:, 1].abs()

    logger = PatienceLogger(patience)
    while epoch < max_epochs:
        optimizer.zero_grad()
        nn_loss, mnn_loss, rot_loss, stre_loss = loss_function(
            params, query_rescaled, target_rescaled, weight=weight
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

        if epoch and epoch % 100 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.item():.2f}, "
                f"Loss comps: {nn_loss.item():.2f}, {mnn_loss.item():.2f}, "
                f"{rot_loss.item():.2f}, {stre_loss.item():.2f}"
            )
        epoch += 1
        if logger.stop_training:
            params = logger.best_params
            epoch = logger.best_epoch
            break
    print(f"Optimized Parameters at Epoch {epoch}: {params.detach().numpy()}")

    # the transformation that goes from query to standardardized query
    return (
        params.detach().numpy(),
        mean_q.item(),
        std_q.item(),
        mean_t.item(),
        std_t.item(),
    )


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
