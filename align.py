#!/usr/bin/env python

import json
from typing import Callable

import numpy as np
import torch
import torch.optim as optim



def standardize(x: torch.Tensor, return_stats: bool = False) -> torch.Tensor:
    if not return_stats:
        return (x - x.mean()) / x.std()
    else:
        return (x - x.mean()) / x.std(), x.mean(), x.std()


def loss_function(
    params,
    keypointsA,
    keypointsB,
):
    """Compute loss as average distance between transformed keypointsA and nearest
    keypointsB, with penalties. Note that in this setup `keypointB` acts as the
    reference.
    """
    a, b, tx, ty, sx, sy = params
    transformedA = affine_transform(params, keypointsA)
    cdist = torch.cdist(transformedA, keypointsB)  # [n, m]
    min_distances, _ = torch.min(cdist, dim=1)

    # Reward correct pairing. Correct pairing is identified by mutual nearest neighbor.
    # Pairing could also be achieved by thresholding ratio of distances between nearest
    # and second nearest neighbors.
    mnn_score = soft_mnn_consistency(cdist)

    # Regularization terms
    norm = torch.sqrt(a**2 + b**2)
    rotation_penalty = (b / norm) ** 2 + (1 - a) ** 2 + b**2  # Penalty on rotation
    # Penalty on difference between sx and sy
    stretching_penalty = (sx - sy) ** 2 + (sx - 1) ** 2 + (sy - 1) ** 2
    return (torch.mean(min_distances), -mnn_score, rotation_penalty, stretching_penalty)


def soft_mnn_consistency(cdist: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
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
    if cdist.shape[0] > cdist.shape[1]:
        cdist = cdist.t()
    weights_A_to_B = (-cdist / temperature).softmax(dim=1)  # [n, m]
    weights_B_to_A = (-cdist / temperature).softmax(dim=0)  # [n, m]

    # Get the most likely match from array 1 to array 2 for each data point in array 1
    _, max_indices_A_to_B = weights_A_to_B.max(dim=1)
    consistency_scores = weights_B_to_A[
        torch.arange(len(max_indices_A_to_B)), max_indices_A_to_B
    ]
    # consistency_scores *= 
    # consistency_scores = (weights_B_to_A * weights_A_to_B)
    return consistency_scores.mean()


def affine_transform(params: torch.Tensor, keypoints: torch.Tensor):
    """Apply affine transformation to keypoints. The transformation consists of:
    - rotation and scaling around origin (hence order doesn't matter)
    - translation
    """
    a, b, tx, ty, sx, sy = params
    norm = torch.sqrt(a**2 + b**2)
    sin_t, cos_t = b / norm, a / norm
    transformed_keypoints = keypoints @ torch.stack(
        [torch.stack([sx * cos_t, -sy * sin_t]), torch.stack([sx * sin_t, sy * cos_t])],
        dim=0,
    ) + torch.stack([tx, ty])
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
    query: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float, float, float, float]:
    # hyperparameters
    lr = 0.0003
    max_epochs = 2000
    eps = 0.001
    patience = 10
    beta_d = 1
    beta_c = 2
    beta_t = 1
    beta_s = 1

    # Initialization
    # [a, b, tx, ty, sx, sy]
    params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], requires_grad=True)
    optimizer = optim.Adam([params], lr=lr)

    # Optimization loop
    last_loss = 0
    epoch = 0
    p = 0

    query = torch.from_numpy(query).float()
    target = torch.from_numpy(target).float()
    # standardize
    query_rescaled, mean_q, std_q = standardize(query, return_stats=True)
    target_rescaled, mean_t, std_t = standardize(target, return_stats=True)

    while epoch < max_epochs and p < patience:
        optimizer.zero_grad()
        nn_loss, mnn_loss, rot_loss, stre_loss = loss_function(
            params, query_rescaled, target_rescaled
        )
        loss = (
            nn_loss * beta_d
            + mnn_loss * beta_c
            + rot_loss * beta_t
            + stre_loss * beta_s
        )
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.item():.2f}, "
                f"Loss comps: {nn_loss.item():.2f}, {mnn_loss.item():.2f}, "
                f"{rot_loss.item():.2f}, {stre_loss.item():.2f}"
            )
        epoch += 1
        if loss < eps:
            p += 1
        else:
            p = 0
        last_loss = loss
    else:
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

    return np.array(mutual_pairs)


def align(
    center_rgb: np.ndarray, center_hsi: np.ndarray, center_taxa: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsi2rgb_param, *hsi2rgb_stats = find_affine(center_hsi, center_rgb)
    hsi2rgb_func = get_query2target_func(*hsi2rgb_param, *hsi2rgb_stats)
    center_hsi2rgb = hsi2rgb_func(center_hsi)
    map_hsi = find_mutual_pairs(center_rgb, center_hsi2rgb)

    taxa2rgb_param, *taxa2rgb_stats = find_affine(center_taxa, center_rgb)
    taxa2rgb_func = get_query2target_func(*taxa2rgb_param, *taxa2rgb_stats)
    center_taxa2rgb = taxa2rgb_func(center_taxa)
    map_taxa = find_mutual_pairs(center_rgb, center_taxa2rgb)

    return center_hsi2rgb, center_taxa2rgb, map_hsi, map_taxa
