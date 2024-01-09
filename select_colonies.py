#!/usr/bin/env python

"""This step selects a given number of colonies detected by the previous step from each 
    plate group.

Input data to this step:
    - metadata with plate barcode to group information in csv format
    - colony features (location, morphology, color, etc.) in csv format

Output data from this step:
    For each plate:
        - A gray scale image with colony segmentation. Selected colonies highlighted.
        - A RGB image with colony segmentation. Selected colonies highlighted.
        - A csv file with the same number of rows as the number of colonies in the 
            plate, where the first column is 0 if not selected, 1 if selected, and -1 if 
            excluded, and the second column is the picking order of the colony.
        - A 4 column csv file with xy coordinate of boundary points on the colony 
            polygons. Columns are index, Label, X, Y, where Label is the index of colony.
        - A two column csv file with xy coordinate of the center of selected colonies,
            ordered by picking order. No header.

This step involves human in the loop. Specifically, The 4 column csv file can be 
    iteratively refined in ImageJ to exclude unwanted colonies. When the user is 
    satistified with the picking, colony picking is finalized.
"""

import argparse
import os
from itertools import cycle
import json
import warnings
import warnings

import numpy as np
import numba
import cv2 as cv
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
import pandas as pd
import polars as pl
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print as rprint
from tqdm.auto import tqdm, trange


from utils import read_config, add_contours, _coco_to_contours


matplotlib.use("TkAgg")


def colony_feat_pca(df_contour: pl.DataFrame) -> pl.DataFrame:
    feats_for_pca = [
        "area",
        "perim",
        "radius",
        "circularity",
        "convexity",
        "inertia_ratio",
        "Graymean",
        "Graystd",
        "Repimean",
        "Repistd",
        "Gepimean",
        "Gepistd",
        "Bepimean",
        "Bepistd",
    ]
    contour_feat = df_contour[feats_for_pca].to_pandas()
    contour_feat = StandardScaler().fit_transform(contour_feat)
    pca = PCA(n_components=2)
    dr = pca.fit_transform(contour_feat)
    return df_contour.with_columns(pca1=pl.Series(dr[:, 0]), pca2=pl.Series(dr[:, 1]))


@numba.njit
def find_ith_false(arr: np.ndarray, i: int) -> int:
    p = 0
    idx = 0
    while p < i:
        if not arr[idx]:
            p += 1
        idx += 1
    return idx - 1


@numba.njit
def min_along_col(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[1]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = arr[:, i].min()
    return out


@numba.njit
def farthest_points_jit(
    dist: np.ndarray, k: int, seed: int
) -> tuple[np.ndarray, float]:
    n = dist.shape[0]
    np.random.seed(seed)
    idx = np.random.choice(n, k, replace=False)
    choices = np.zeros(n, dtype=numba.boolean)
    choices[idx] = True

    unchanged = 0
    # for i in range(np.iinfo(np.int64).max):
    i = 0
    while True:
        j = idx[i]
        choices[j] = False
        replace = min_along_col(dist[choices][:, ~choices]).argmax()
        replace_idx = find_ith_false(choices, replace + 1)
        choices[replace_idx] = True
        if replace_idx == j:
            unchanged += 1
        else:
            unchanged = 0
            idx[i] = replace_idx
        if unchanged == k:
            # if for k iterations, the farthest point is unchanged, we have converged
            break
        i = (i + 1) % k
    return idx, min_along_col(dist[choices][:, ~choices]).max()


def farthest_points_old(data, n, seed: int):
    dist_mat = cdist(data, data, metric="euclidean")
    rng = np.random.RandomState(seed)
    r = rng.choice(data.shape[0], n, replace=False).tolist()
    r_old = None
    while r_old != r:
        r_old = r[:]
        for i in range(n):
            no_i = r[:]
            no_i.pop(i)
            cols_in_play = np.asarray(range(dist_mat.shape[1]))[np.newaxis, :][
                :, list(filter(lambda n: n not in no_i, range(dist_mat.shape[1])))
            ]
            mm = dist_mat[no_i, :][
                :, list(filter(lambda n: n not in no_i, range(dist_mat.shape[1])))
            ]
            max_min_dist = np.argmax(np.min(mm, 0))
            r[i] = cols_in_play[0, :][max_min_dist]
    mm = dist_mat[r][:, list(filter(lambda n: n not in r, range(dist_mat.shape[1])))]
    return np.array(r), np.max(np.min(mm, 0))


# @singledispatch
# def farthest_points(*args, **kwargs):
#     raise NotImplementedError


def farthest_points(
    data: np.ndarray,
    k: int,
    seed: int,
    kmeans_init: bool = False,
    group_assignment: np.ndarray = None,
    group_max: dict = None,
) -> tuple[np.ndarray, float]:
    if k >= data.shape[0]:
        raise ValueError(
            f"Number of selected points ({k}) should be smaller than the number of data "
            f"points ({data.shape[0]})."
        )
    if group_assignment is None and group_max is None:
        return _farthest_points(data, k, seed, kmeans_init)
    elif group_assignment is not None and group_max is not None:
        return _farthest_points_with_max(
            data, k, seed, kmeans_init, group_assignment, group_max
        )
    else:
        raise ValueError(
            "Incompatible `group_assignment` and `group_max`."
            "They should be both None or both not None."
        )


# @farthest_points.register
def _farthest_points(
    data: np.ndarray, k: int, seed: int, kmeans_init: bool = False
) -> tuple[np.ndarray, float]:
    dist = squareform(pdist(data))
    n = dist.shape[0]
    rng = np.random.RandomState(seed)
    if kmeans_init:
        kmeans = KMeans(n_clusters=k, random_state=rng, n_init="auto").fit(data)
        cluster_centers = kmeans.cluster_centers_
        tree_data = KDTree(data)
        idx = tree_data.query(cluster_centers)[1]
    else:
        idx = rng.choice(n, k, replace=False)
    # np.random.seed(seed)
    # idx = np.random.choice(n, k, replace=False)
    choices = np.zeros(n, dtype=bool)
    choices[idx] = True

    unchanged = 0
    for i in cycle(range(k)):
        # i = i % k
        j = idx[i]
        choices[j] = False
        replace_idx = (~choices).nonzero()[0][
            dist[choices][:, ~choices].min(0).argmax()
        ]
        choices[replace_idx] = True
        if replace_idx == j:
            unchanged += 1
        else:
            unchanged = 0
            idx[i] = replace_idx
        if unchanged == k:
            # if for k iterations, the farthest point is unchanged, we have converged
            break
    return idx, dist[choices][:, ~choices].min(0).max()


# @farthest_points.register
def _farthest_points_with_max(
    data: np.ndarray,
    k: int,
    seed: int,
    kmeans_init: bool = False,
    group_assignment: np.ndarray = None,
    group_max: dict = None,
) -> tuple[np.ndarray, float]:
    """
    when choosing which point to replace with, consider these things:
    1. Only points that are not picked (or the point of interest) yet are
        considered.
    2. Only points that belong the a group that has not hit its max are considered.

    """
    # parameter sanity check
    group_max = {k: v if v > 0 else np.inf for k, v in group_max.items()}
    max_select = pd.concat(
        [pd.Series(group_assignment).value_counts(), pd.Series(group_max)], axis=1
    )
    max_select.columns = ["num_points", "max_selects"]
    if max_select.max(axis=1).sum() < k:
        raise ValueError(
            "The maximum number of selected points for groups "
            f"{max_select.index.tolist()} are {max_select.max(axis=1).astype(int).tolist()}, "
            f"so at most {max_select.max().sum().astype(int)} points can be selected, smaller than "
            f"what is intended ({k}).\n"
            "Please either increase the maximum number of selected points for some "
            "groups or decrease the number of points to select to make them compatible."
        )
    if not set(group_max).issubset(set(group_assignment)):
        warnings.warn(
            f"No data points belong to these groups: "
            f"{set(group_max) - set(group_assignment)}"
        )
    if not data.shape[0] == len(group_assignment):
        raise ValueError("Incompatible number of data points and group assignments.")

    group_max = {
        g: group_max[g] if isinstance(group_max.get(g), int) else np.inf
        for g in set(group_assignment)
    }

    dist = squareform(pdist(data))
    n = dist.shape[0]
    rng = np.random.RandomState(seed)
    if kmeans_init:
        kmeans = KMeans(n_clusters=k, random_state=rng, n_init="auto").fit(data)
        cluster_centers = kmeans.cluster_centers_
        tree_data = KDTree(data)
        idx = tree_data.query(cluster_centers)[1]
    else:
        idx = rng.choice(n, k, replace=False)
    # np.random.seed(seed)
    # idx = np.random.choice(n, k, replace=False)
    choices = np.zeros(n, dtype=bool)
    choices[idx] = True

    pick_counter = dict.fromkeys(group_assignment, 0)
    for i in idx:
        pick_counter[group_assignment[i]] += 1

    unchanged = 0
    for i in cycle(range(k)):
        j = idx[i]
        choices[j] = False
        pick_counter[group_assignment[j]] -= 1

        groups_good = [i for i in group_max if pick_counter[i] < group_max[i]]
        to_choose = np.logical_and(~choices, np.isin(group_assignment, groups_good))
        if not to_choose.any():
            # if no points are available, we have converged
            choices[j] = True
            pick_counter[group_assignment[j]] += 1
            if choices.sum() < k:
                warnings.warn(
                    f"Early stopping: only {choices.sum()} points are selected while "
                    f"{k} are intended."
                )
            break
        replace_idx = to_choose.nonzero()[0][
            dist[choices][:, to_choose].min(0).argmax()
        ]

        choices[replace_idx] = True
        pick_counter[group_assignment[replace_idx]] += 1
        if replace_idx == j:
            unchanged += 1
        else:
            unchanged = 0
            idx[i] = replace_idx
        if unchanged == k:
            # if for k iterations, the farthest point is unchanged, we have converged
            break
    return idx, dist[choices][:, ~choices].min(0).max().item()

    # choices_tortose = choices.copy()
    # pick_counter_tortose = pick_counter.copy()
    # idx_tortose = idx.copy()
    # choices_hare = choices.copy()
    # pick_counter_hare = pick_counter.copy()
    # idx_hare = idx.copy()

    # # tortose moves once
    # _get_new_idx(
    #     idx_tortose,
    #     0,
    #     group_assignment,
    #     group_max,
    #     pick_counter_tortose,
    #     dist,
    #     choices_tortose,
    # )
    # # hare moves twice
    # _get_new_idx(
    #     idx_hare,
    #     0,
    #     group_assignment,
    #     group_max,
    #     pick_counter_hare,
    #     dist,
    #     choices_hare,
    # )
    # _get_new_idx(
    #     idx_hare,
    #     1,
    #     group_assignment,
    #     group_max,
    #     pick_counter_hare,
    #     dist,
    #     choices_hare,
    # )

    # i_t = 1
    # i_h = 2
    # while not np.array_equal(idx_tortose, idx_hare):
    #     # tortose moves once
    #     _get_new_idx(
    #         idx_tortose,
    #         i,
    #         group_assignment,
    #         group_max,
    #         pick_counter_tortose,
    #         dist,
    #         choices_tortose,
    #     )
    #     i_t += 1
    #     # hare moves twice
    #     _get_new_idx(
    #         idx_hare,
    #         i_h,
    #         group_assignment,
    #         group_max,
    #         pick_counter_hare,
    #         dist,
    #         choices_hare,
    #     )
    #     _get_new_idx(
    #         idx_hare,
    #         i_h + 1,
    #         group_assignment,
    #         group_max,
    #         pick_counter_hare,
    #         dist,
    #         choices_hare,
    #     )
    #     i_h += 2
    # return idx_tortose, dist[choices_tortose][:, ~choices_tortose].min(0).max()


def _get_new_idx(
    idx: np.ndarray,  # will be modified in place
    i: int,
    group_assignment: np.ndarray,
    group_max: dict,
    pick_counter: dict,  # will be modified in place
    dist: np.ndarray,
    choices: np.ndarray,  # will be modified in place
) -> np.ndarray:
    i = i % idx.shape[0]
    j = idx[i]
    choices[j] = False
    pick_counter[group_assignment[j]] -= 1

    avoid_groups = [i for i in group_max if pick_counter[i] >= group_max[i]]
    to_choose = ~np.logical_or(choices, np.isin(group_assignment, avoid_groups))
    replace_idx = to_choose.nonzero()[0][dist[~to_choose][:, to_choose].min(0).argmax()]

    choices[replace_idx] = True
    pick_counter[group_assignment[replace_idx]] += 1
    if replace_idx != j:
        idx[i] = replace_idx
    return idx, choices, pick_counter


def process_metadata(metadata_path: str) -> dict[str, tuple[int, list[str], list[str]]]:
    metadata = pd.read_csv(metadata_path)
    g = metadata.groupby("group")
    group2barcodes = g["barcode"].apply(list).to_dict()
    group2num_colonies_plate = g["num_picks_plate"].apply(list).to_dict()
    group2num_colonies_group = g["num_picks_group"].first().to_dict()
    return {
        group: (
            group2barcodes[group],
            group2num_colonies_plate[group],
            group2num_colonies_group[group],
        )
        for group in group2barcodes
    }


def _get_empty_df(dfs: list[pl.DataFrame] | dict[str, pl.DataFrame]) -> pl.DataFrame:
    if isinstance(dfs, dict):
        non_empty_idx = [k for k, df in dfs.items() if df.shape[0] > 0]
    elif isinstance(dfs, list):
        non_empty_idx = [i for i, df in enumerate(dfs) if df.shape[0] > 0]
    else:
        raise TypeError("`dfs` should be either a list or a dict.")
    if not non_empty_idx:
        raise ValueError("At least one dataframe should be non-empty.")
    return dfs[non_empty_idx[0]].slice(0, 0)


def pick_colony_init(
    image_dir: str,
    input_dir: str,
    output_dir: str,
    metadata_path: str,
    config_path: str,
) -> None:
    config = read_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    for group, (barcodes, num_colonies_plate, num_colonies) in process_metadata(
        metadata_path
    ).items():
        dfs_contour = [
            pl.read_csv(
                os.path.join(input_dir, f"{b}_metadata.csv"),
                dtypes={
                    "post_pass": bool,
                    "pass_initial": bool,
                    "need_pp": bool,
                    "direct_pass": bool,
                },
            )
            .with_columns(barcode=pl.lit(b))
            .with_row_count("contour_idx")
            for b in barcodes
        ]
        dicts_border = []
        for b in barcodes:
            with open(os.path.join(input_dir, f"{b}_annot.json")) as f:
                dicts_border.append(json.load(f))

        # dfs_border = [
        #     pl.read_csv(
        #         os.path.join(input_dir, f"{b}_contour_border.csv")
        #     ).with_columns(barcode=pl.lit(b))
        #     for b in barcodes
        # ]

        # replace empty contour dataframes with a dummy dataframe with same schema
        df_dummy = _get_empty_df(dfs_contour)
        dfs_contour = [
            df if df.shape[0] > 0 else df_dummy.clone() for df in dfs_contour
        ]
        df_contour = pl.concat(dfs_contour).with_row_count("contour_idx_group")
        df_contour = colony_feat_pca(df_contour)
        contour_feat = df_contour[["pca1", "pca2"]].to_numpy()

        if num_colonies >= contour_feat.shape[0]:
            choices = list(range(contour_feat.shape[0]))
            warnings.warn(
                f"Number of colonies for picking ({num_colonies}) is larger than the "
                f"number of colonies detected ({contour_feat.shape[0]}) for group "
                f"{group}. As a result, all colonies are selected."
            )
        else:
            choices_list, min_dist_list = zip(
                *joblib.Parallel(n_jobs=8)(
                    joblib.delayed(farthest_points)(
                        contour_feat,
                        num_colonies,
                        seed=seed + 42,
                        group_assignment=df_contour["barcode"].to_numpy(),
                        group_max=dict(zip(barcodes, num_colonies_plate)),
                    )
                    for seed in range(config["farthest_points_iteration"])
                )
            )
            # choices_list, min_dist_list = zip(
            #     *[
            #         farthest_points(
            #             contour_feat,
            #             num_colonies,
            #             seed=seed + 42,
            #             group_assignment=df_contour["barcode"].to_numpy(),
            #             group_max=dict(zip(barcodes, num_colonies_plate)),
            #         )
            #         for seed in trange(config["farthest_points_iteration"])
            #     ]
            # )
            choices = choices_list[np.argmin(min_dist_list)]

        df_contour = df_contour.with_columns(
            picking_status=pl.when(pl.col("contour_idx_group").is_in(choices))
            .then(1)
            .otherwise(2)
        )

        dfs_contour = df_contour.drop(
            ["contour_idx", "contour_idx_group"]
        ).partition_by("barcode", as_dict=True)
        df_dummy = _get_empty_df(dfs_contour)
        dfs_contour = [
            dfs_contour[b] if b in dfs_contour else df_dummy.clone() for b in barcodes
        ]
        for barcode, df_contour, dict_border in zip(
            barcodes, dfs_contour, dicts_border
        ):
            _save_modified(
                image_dir,
                output_dir,
                df_contour,
                dict_border,
                barcode,
                annot_stage="init",
            )


def _save_modified(
    image_dir: str,
    output_dir: str,
    df_contour: pl.DataFrame,
    dict_border: dict,  # in coco format
    barcode: str,
    annot_stage: str,
) -> None:
    """These files will be saved to `output_dir`:
    - `<barcode>_annot_[init|final].json`: coco format segmentation annotation
    - `<barcode>_metadata_[init|final].csv`: metadata with picking status
    - `<barcode>_rgb_red_contour_[init|final].png`: RGB image under red light with
        colony segmentation. Selected colonies highlighted.
    - `<barcode>_rgb_white_contour_[init|final].png`: same as above for white light.

    The order of colonies in the annotation file and metadata file will first base on
    colony class and then by colony size.
    """
    picking_status = df_contour["picking_status"].to_list()

    for idx, ps in enumerate(picking_status):
        dict_border["annotations"][idx]["category_id"] = ps

    df_contour = df_contour.with_row_count("contour_idx")
    new_order = []
    for status in pd.Series(picking_status).unique():
        df = df_contour.filter(pl.col("picking_status") == status)
        df = df.sort("area", descending=True)
        new_order.extend(df["contour_idx"].to_list())
    dict_border["annotations"] = [dict_border["annotations"][i] for i in new_order]
    for idx, a in enumerate(dict_border["annotations"]):
        a["id"] = idx

    # a trick to reorder the rows of a polars dataframe by a list of indices, similar
    # to `.iloc` in pandas.
    df_contour = df_contour.join(
        pl.Series(new_order, dtype=pl.UInt32).to_frame("contour_idx"), on="contour_idx"
    ).drop("contour_idx")

    with open(
        os.path.join(output_dir, f"{barcode}_annot_{annot_stage}.json"), "w"
    ) as f_out:
        json.dump(dict_border, f_out)
    df_contour.write_csv(
        os.path.join(output_dir, f"{barcode}_metadata_{annot_stage}.csv")
    )
    for light in ["red", "white"]:
        image_contours = _add_contours(
            cv.imread(os.path.join(image_dir, f"{barcode}_rgb_{light}.png")),
            _coco_to_contours(dict_border),
            df_contour["picking_status"].to_list(),
            df_contour[["center_x", "center_y"]].to_numpy(),
        )
        cv.imwrite(
            os.path.join(
                output_dir, f"{barcode}_rgb_{light}_contour_{annot_stage}.png"
            ),
            image_contours,
        )


def _add_contours(
    image: np.ndarray,
    contours: list[np.ndarray],
    picking_status: list,
    centers: np.ndarray,
) -> np.ndarray:
    """Plot colony contours and centers on the image, with different color for colonies
    of different picking status.

    Picked colonies are green with red center, unpicked colonies are black with gray
    center, excluded colonies are orange with blue center.

    Annotations are added for picked colonies.
    """
    contours = np.array(contours, dtype=object)
    picking_status = np.array(picking_status)
    # picked colonies
    for status, border_color, center_color, annot_index in zip(
        [1, 2, 3],
        [(0, 0, 255), (0, 0, 0), (0, 165, 255)],
        [(0, 255, 0), (128, 128, 128), (255, 0, 0)],
        [True, False, False],
    ):
        image = add_contours(
            image,
            contours[picking_status == status],
            centers[picking_status == status],
            contour_pixel=1,
            center_pixel=1,
            border_color=border_color,
            center_color=center_color,
            annot_index=annot_index,
        )
    return image


def pick_colony_post(
    image_dir: str, data_dir: str, metadata_path: str, start_from: str = "init"
):
    """Resultant csv files from initial picking are subject to manual picking and
    colonies of poor quality are removed.

    This step finds colonies that are manually removed and mark them in the
    `picking_status` column of the metadata csv file. Then we make up for the removed
    colonies by picking the next best colonies, measured by minimum distance to the
    colonies that are not removed.
    """
    assert start_from in ["init", "final"]
    for group, (barcodes, num_colonies_plate, num_colonies) in process_metadata(
        metadata_path
    ).items():
        rprint(f"[bold green]Post-processing group {group}[/bold green]")
        dfs_contour = []
        dicts_border = []
        for b in barcodes:
            df_contour = (
                pl.read_csv(
                    os.path.join(data_dir, f"{b}_metadata_{start_from}.csv"),
                    dtypes={
                        "post_pass": bool,
                        "pass_initial": bool,
                        "need_pp": bool,
                        "direct_pass": bool,
                    },
                )
                .with_columns(barcode=pl.lit(b))
                .with_row_count("contour_idx")
            )
            with open(os.path.join(data_dir, f"{b}_annot_{start_from}.json")) as f:
                dict_border = json.load(f)
            with open(os.path.join(data_dir, f"{b}_annot_{start_from}_post.json")) as f:
                dict_border_post = json.load(f)

            cnt_ids = np.array([anno["id"] for anno in dict_border["annotations"]])
            cnt_ids_post = np.array(
                [anno["id"] for anno in dict_border_post["annotations"]]
            )
            picking_status_post = [
                anno["category_id"] for anno in dict_border_post["annotations"]
            ]

            if len(cnt_ids_post) < len(cnt_ids):
                raise ValueError(
                    "Number of colonies decreased after manual picking. "
                    "This is not supported. Colonies should only switch classes but not "
                    "be deleted."
                )
            if not (cnt_ids_post[: len(cnt_ids)] == cnt_ids).all():
                raise ValueError(
                    "Order of colonies changed after manual picking."
                    "This is not supported."
                )

            intersect = np.in1d(cnt_ids_post, cnt_ids)
            if not intersect.all():
                # update dict_border
                annots_new = np.array(dict_border_post["annotations"], dtype=object)[
                    ~intersect
                ].tolist()
                picking_status_new_vc = pd.Series(picking_status_post)[
                    ~intersect
                ].value_counts()
                dict_border["annotations"].extend(annots_new)

                # update df_contour
                from detect_colonies import _calc_contours_stats

                cnts_post = _coco_to_contours(dict_border_post)
                cnts_new = [c for i, c in zip(intersect, cnts_post) if not i]
                img = cv.imread(os.path.join(image_dir, f"{b}_rgb_red.png"))
                rng = np.random.default_rng(42)
                mock_image = rng.random(img.shape[:2])
                # get stats and set a few features to null
                df_contour_new = (
                    _calc_contours_stats(cnts_new, mock_image, -1)
                    .drop("close_to_border")
                    .with_columns(barcode=pl.lit(b))
                )
                df_contour_new = df_contour_new.select(
                    [c for c in df_contour.columns if c in df_contour_new.columns]
                )
                df_contour = pl.concat(
                    [df_contour.drop("contour_idx"), df_contour_new], how="diagonal"
                ).with_row_count("contour_idx")
                rprint(
                    "\tFound ",
                    len(cnts_new),
                    " new manually added colonies in plate ",
                    b,
                    ". Some features of these new colonies are calculated while others will be populated with null.\n",
                    "\t\tAmong these new colonies, ",
                    picking_status_new_vc.get(1, 0),
                    " are marked as [orange]`picked colony`[/orange], ",
                    picking_status_new_vc.get(2, 0),
                    " are marked as [orange]`unpicked colony`[/orange], ",
                    picking_status_new_vc.get(3, 0),
                    " are marked as [orange]`excluded colony`[/orange].",
                    sep="",
                )
            dicts_border.append(dict_border)
            df_contour = df_contour.with_columns(
                picking_status=pl.Series(picking_status_post)
            )
            dfs_contour.append(df_contour)
        df_dummy = _get_empty_df(dfs_contour)
        dfs_contour = [
            df if df.shape[0] > 0 else df_dummy.clone() for df in dfs_contour
        ]
        df_contour = pl.concat(dfs_contour).with_row_count("contour_idx_group")
        df_source = df_contour.filter(pl.col("picking_status") == 1)
        df_target = df_contour.filter(pl.col("picking_status") == 2)
        data_source = df_source[["pca1", "pca2"]].to_numpy()
        data_target = df_target[["pca1", "pca2"]].to_numpy()
        picking_status = df_contour["picking_status"].to_numpy().copy()
        dist, _ = KDTree(data_source).query(data_target)
        num_to_pick = num_colonies - df_source.shape[0]
        if num_to_pick > 0 and num_to_pick <= len(df_target):
            rprint(
                "\tDoing a quick picking of",
                num_to_pick,
                "colonies among",
                len(df_target),
                "unpicked ones to compensate for colonies manully removed.",
            )
            # index_to_modify = np.where(picking_status == 2)[0][
            #     np.argpartition(dist, -num_to_pick)[-num_to_pick:]
            # ]
            cnt_barcodes = df_target["barcode"].to_list()
            num_picked_barcodes = dict(zip(*df_source["barcode"].value_counts()))
            plate2num_cnts = {
                b: i if i > 0 else np.inf for b, i in zip(barcodes, num_colonies_plate)
            }
            index_to_modify = []
            for idx in np.argsort(dist)[::-1]:
                b = cnt_barcodes[idx]
                if num_picked_barcodes[b] < plate2num_cnts[b]:
                    index_to_modify.append(idx)
                    num_to_pick -= 1
                if num_to_pick == 0:
                    break
            picking_status[index_to_modify] = 1
        elif num_to_pick <= 0:
            rprint(
                "\tMore colonies are marked as picked after manual picking (",
                num_colonies,
                " vs. ",
                len(df_source),
                "). ",
                "Class of ",
                "all colonies will remain as is.",
                sep="",
            )
        elif len(df_target) == 0:
            rprint(
                "\tNo colonies are marked as unpicked after manual picking. "
                "Class of all colonies will remain as is."
            )
        else:
            import pdb; pdb.set_trace()
            rprint("Some other cases.")

        df_contour = df_contour.with_columns(picking_status=pl.Series(picking_status))

        dfs_contour_dict = df_contour.drop(
            ["contour_idx", "contour_idx_group"]
        ).partition_by("barcode", as_dict=True)
        # dfs_contour_dict = [dfs_contour_dict[b] for b in barcodes]
        df_dummy = _get_empty_df(dfs_contour_dict)
        dfs_contour_dict = [
            dfs_contour_dict[b] if b in dfs_contour_dict else df_dummy.clone()
            for b in barcodes
        ]
        for barcode, df_contour, dict_border in zip(
            barcodes, dfs_contour_dict, dicts_border
        ):
            _save_modified(
                image_dir,
                data_dir,
                df_contour,
                dict_border,
                barcode,
                annot_stage="final",
            )


def pick_colony_final(
    image_dir: str,
    data_dir: str,
    metadata_path: str,
    output_dir: str,
    tsp_method: str = None,
) -> None:
    """When the user is satisfied and no colonies need to be removed, we can reorder
    selected colonies to minimize movement (i.e., TSP problem) and save results
    includeing:
        - Two RGB image with colony segmentation. Selected colonies highlighted.
        - A two column csv file where each row is xy coordinate of the centers of
            selected colonies, ordered by picking order.
        - A plot of two dimensional colony feature from PCA, with colored mapping to
            origin plate and shape to whether the colony is picked, not picked, or
            manually excluded.
    """
    os.makedirs(output_dir, exist_ok=True)
    for group, (barcodes, num_colonies_plate, num_colonies) in process_metadata(
        metadata_path
    ).items():
        dfs_contour = []
        for b in barcodes:
            df_contour = pl.read_csv(
                os.path.join(data_dir, f"{b}_metadata_final.csv"),
                dtypes={
                    "post_pass": bool,
                    "picking_status": int,
                    "pass_initial": bool,
                    "need_pp": bool,
                    "direct_pass": bool,
                },
            )
            dfs_contour.append(df_contour)
        df_dummy = _get_empty_df(dfs_contour)
        dfs_contour = [
            df if df.shape[0] > 0 else df_dummy.clone() for df in dfs_contour
        ]

        for b, df_contour in zip(tqdm(barcodes), dfs_contour):
            image_rgb_red = cv.imread(os.path.join(image_dir, f"{b}_rgb_red.png"))
            image_rgb_white = cv.imread(os.path.join(image_dir, f"{b}_rgb_white.png"))
            image_gs_red = cv.imread(os.path.join(image_dir, f"{b}_gs_red.png"))
            image_gs_white = cv.imread(os.path.join(image_dir, f"{b}_gs_white.png"))
            centers = df_contour[["center_x", "center_y"]].to_numpy()
            ps = df_contour["picking_status"].to_numpy()
            with open(os.path.join(data_dir, f"{b}_annot_final.json")) as f:
                dict_border = json.load(f)
            contours = _coco_to_contours(dict_border)

            if tsp_method:
                idx = _tsp(centers[ps == 1]).argsort()
                picking_order = np.zeros(ps.shape[0], dtype=int)
                picking_order[ps == 1] = idx + 1
                df_contour = df_contour.with_columns(
                    picking_order=pl.Series(picking_order)
                )
                contours = np.array(contours, dtype=object)
                contours[ps == 1] = contours[ps == 1][idx]
                contours = contours.tolist()
                # ps and centers do not need to be reordered for plotting
            else:
                df_contour = df_contour.with_columns(picking_order=-1)

            # images with contours
            image_rgb_red_contours = _add_contours(image_rgb_red, contours, ps, centers)
            image_rgb_white_contours = _add_contours(
                image_rgb_white, contours, ps, centers
            )
            image_gs_red_contours = _add_contours(image_gs_red, contours, ps, centers)
            image_gs_white_contours = _add_contours(
                image_gs_white, contours, ps, centers
            )
            cv.imwrite(
                os.path.join(output_dir, f"{b}_rgb_red_contour.jpg"),
                image_rgb_red_contours,
            )
            cv.imwrite(
                os.path.join(output_dir, f"{b}_rgb_white_contour.jpg"),
                image_rgb_white_contours,
            )
            cv.imwrite(
                os.path.join(output_dir, f"{b}_gs_red_contour.jpg"),
                image_gs_red_contours,
            )
            cv.imwrite(
                os.path.join(output_dir, f"{b}_gs_white_contour.jpg"),
                image_gs_white_contours,
            )

            # metadata
            df_contour.write_csv(os.path.join(output_dir, f"{b}_metadata.csv"))

            # ordered picking coordinates
            df_contour.filter(pl.col("picking_status") == 1).sort(
                "picking_order"
            ).select(
                pl.col("center_x").round().cast(pl.Int32),
                pl.col("center_y").round().cast(pl.Int32),
            ).write_csv(
                os.path.join(output_dir, f"{b}_picking.csv"), has_header=False
            )

        # pca
        fig, ax = plt.subplots(figsize=(8, 8))
        data = pl.concat(dfs_contour)[
            ["pca1", "pca2", "barcode", "picking_status"]
        ].to_pandas()
        data.columns = ["PCA1", "PCA2", "Plate barcode", "Picking status"]

        # plot with no border and small size
        sns.scatterplot(
            x="PCA1",
            y="PCA2",
            data=data,
            hue="Plate barcode",
            style="Picking status",
            ax=ax,
            s=10,
            linewidth=0,
            alpha=0.5,
        )
        ax.set_title(f"PCA of colony features in group {group}")
        fig.savefig(os.path.join(output_dir, f"pca_{group}.png"), dpi=300)


def _tsp(data: np.ndarray, tsp_method: str = "heuristic") -> np.ndarray:
    """Reorder picked colonies to minimize movement"""
    if tsp_method not in ["heuristic", "exact"]:
        raise ValueError(
            f"Invalid tsp_method {tsp_method}. Choose from 'heuristic' or 'exact'."
        )
    if data.shape[0] <= 2:
        warnings.warn("TSP problem has only 2 or fewer points. No reordering needed.")
        return np.arange(data.shape[0])
    dm = squareform(pdist(data))
    dm[:, 0] = 0  # "open" TSP problem
    if tsp_method == "heuristic":
        permutation1, distance1 = solve_tsp_simulated_annealing(dm)
        permutation2, distance2 = solve_tsp_local_search(
            dm, x0=permutation1, perturbation_scheme="ps3"
        )
    elif tsp_method == "exact":
        permutation2, distance2 = solve_tsp_dynamic_programming(dm)
    else:
        raise ValueError(f"Invalid tsp_method {tsp_method}.")
    return np.array(permutation2)


if __name__ == "__main__":
    # pick_colony_init(
    #     image_dir="test_data/input_png",
    #     input_dir="test_data/output_01",
    #     output_dir="test_data/output_02",
    #     metadata_path="test_data/input/metadata.csv",
    #     config_path="test_data/configs/configure.yaml",
    # )

    # pick_colony_post(
    #     image_dir="test_data/input_png",
    #     data_dir="test_data/output_02",
    #     metadata_path="test_data/input/metadata.csv",
    #     start_from="init",
    # )

    # pick_colony_final(
    #     image_dir="test_data/input_png",
    #     data_dir="test_data/output_02",
    #     metadata_path="test_data/input/metadata.csv",
    #     output_dir="test_data/output_03",
    #     # tsp_method="heuristic",
    #     tsp_method="",
    # )
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_init = subparsers.add_parser("init")
    parser_init.add_argument(
        "-p",
        "--image_dir",
        required=True,
        type=str,
        help="path to the directory containing the images",
    )
    parser_init.add_argument(
        "-i", "--input_dir", required=True, type=str, help="path to the input directory"
    )
    parser_init.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="path to the output directory",
    )
    parser_init.add_argument(
        "-m",
        "--metadata_path",
        required=True,
        type=str,
        help="path to the metadata file",
    )
    parser_init.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="path to the configure file",
    )

    parser_post = subparsers.add_parser("post")
    parser_post.add_argument(
        "-p",
        "--image_dir",
        required=True,
        type=str,
        help="path to the directory containing the images",
    )
    parser_post.add_argument(
        "-i", "--input_dir", required=True, type=str, help="path to the input directory"
    )
    parser_post.add_argument(
        "-m",
        "--metadata_path",
        required=True,
        type=str,
        help="path to the metadata file",
    )
    parser_post.add_argument(
        "-s",
        "--start_from",
        required=True,
        type=str,
        help="start from init or final",
        choices=["init", "final"],
    )

    parser_final = subparsers.add_parser("final")
    parser_final.add_argument(
        "-p",
        "--image_dir",
        required=True,
        type=str,
        help="path to the directory containing the images",
    )
    parser_final.add_argument(
        "-i", "--input_dir", required=True, type=str, help="path to the input directory"
    )
    parser_final.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="path to the output directory",
    )
    parser_final.add_argument(
        "-m",
        "--metadata_path",
        required=True,
        type=str,
        help="path to the metadata file",
    )
    parser_final.add_argument(
        "-t",
        "--tsp_method",
        type=str,
        help="tsp method",
        choices=["heuristic", "exact"],
        default=None,
    )

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise ValueError(f"{args.image_dir} is not a valid directory.")
    if args.command == "init":
        pick_colony_init(
            image_dir=args.image_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            metadata_path=args.metadata_path,
            config_path=args.config_path,
        )
    elif args.command == "post":
        pick_colony_post(
            image_dir=args.image_dir,
            data_dir=args.input_dir,
            metadata_path=args.metadata_path,
            start_from=args.start_from,
        )
    elif args.command == "final":
        pick_colony_final(
            image_dir=args.image_dir,
            data_dir=args.input_dir,
            metadata_path=args.metadata_path,
            output_dir=args.output_dir,
            tsp_method=args.tsp_method,
        )
