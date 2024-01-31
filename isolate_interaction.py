#!/usr/bin/env python
import argparse
import subprocess
import warnings
import os
import sys
import glob
from collections import defaultdict, Counter
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse as ss
from scipy.stats import mannwhitneyu, false_discovery_control
import pandas as pd
import networkx as nx
from rich import print as rprint

from align import find_affine, get_query2target_func, find_mutual_pairs


def _read_colony_metadata(colony_metadata_dir: str) -> pd.DataFrame:
    # read colony metadata from CAMII picking pipeline
    colony_metadatas = []
    for f in glob.glob(f"{colony_metadata_dir}/*_metadata.csv"):
        src_plate_barcode = os.path.basename(f).split("_")[0]
        df_colony = pd.read_csv(f)
        df_colony["colony_barcode"] = (
            df_colony.plate_barcode
            + "_"
            + df_colony.center_x.round(3).astype(str)
            + "x"
            + df_colony.center_y.round(3).astype(str)
        )
        # if not all(df_colony.plate_barcode == src_plate_barcode):
        #     raise ValueError(
        #         f"Plate barcode in colony metadata file {f} does not match the filename."
        #     )
        df_colony = df_colony.set_index("colony_barcode", verify_integrity=True)
        colony_metadatas.append(df_colony)
    return pd.concat(colony_metadatas)


def _read_isolate_metadata(isolate_metadata_dir: str) -> pd.DataFrame:
    isolate_metadata_list = []
    for path in glob.glob(
        os.path.join(isolate_metadata_dir, "Destination * - * *.csv")
    ):
        dest_plate_barcode = os.path.basename(path).split(" ")[1]
        df = pd.read_csv(path, skiprows=2)
        df["dest_plate"] = dest_plate_barcode
        isolate_metadata_list.append(df)
    return pd.concat(isolate_metadata_list, ignore_index=True)


def _read_isolate_metadata_rich(
    isolate_metadata_dir: str, plate_metadata_path: str
) -> pd.DataFrame:
    plate_metadata = pd.read_csv(plate_metadata_path)
    if not plate_metadata.barcode.is_unique:
        raise ValueError("Plate metadata barcode column is not unique")
    isolate_metadata = _read_isolate_metadata(isolate_metadata_dir)
    isolate_metadata.columns = ["picking_coord", "src_plate", "dest_well", "dest_plate"]
    isolate_metadata = pd.merge(
        isolate_metadata,
        plate_metadata[["barcode", "group", "sample_type"]].rename(
            {"group": "medium_type", "barcode": "src_plate"}, axis=1
        ),
        on="src_plate",
    )
    isolate_metadata["dest_well_barcode"] = (
        isolate_metadata.dest_plate + "_" + isolate_metadata.dest_well
    )
    isolate_metadata[["src_x", "src_y"]] = (
        isolate_metadata["picking_coord"]
        .str.extract(r"\((\d+.\d+), (\d+.\d+)\)")
        .astype(float)
        .to_numpy()
    )
    return isolate_metadata.set_index("dest_well_barcode")


def map_to_network(map_t2b: list[int], map_b2t: list[int]) -> nx.Graph:
    """
    Constructs a directed network from two lists of integer indices.
    map_t2b: List of target indices in B for each node in A (-1 for no target).
    map_b2t: List of target indices in A for each node in B (-1 for no target).
    """
    G = nx.DiGraph()
    G.add_nodes_from([f"A{i}" for i in range(len(map_t2b))])
    G.add_nodes_from([f"B{i}" for i in range(len(map_b2t))])
    G.add_edges_from(
        [(f"A{i}", f"B{target}") for i, target in enumerate(map_t2b) if target != -1]
    )
    G.add_edges_from(
        [(f"B{i}", f"A{target}") for i, target in enumerate(map_b2t) if target != -1]
    )

    return G


def remove_bad_nodes(G: nx.Graph, remove_from: str) -> tuple[nx.Graph, list[str]]:
    """
    Removes all edges connected to bad nodes from the specified list (A or B) in the network.
    A bad node is one whose target in the other list is also targeted by other nodes.
    remove_from: 'A' to remove edges connected to bad nodes from A, 'B' to remove edges from B.
    """

    if remove_from not in ["A", "B"]:
        raise ValueError("remove_from must be 'top' or 'bottom'")

    the_other_set = "B" if remove_from == "A" else "A"
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


def _align_isolate_colony(
    colony_metadata: pd.DataFrame,
    isolate_metadata: pd.DataFrame,
    plate_barcodes: list[str] = None,
    colony_plate_key: str = "plate_barcode",
    isolate_plate_key: str = "src_plate",
    log: bool = True,
) -> tuple[list[str], pd.Series]:
    isolate2colony = []
    bad_colonies = []
    if plate_barcodes is None:
        plate_barcodes = sorted(
            set(colony_metadata[colony_plate_key])
            & set(isolate_metadata[isolate_plate_key])
        )
    for plate in plate_barcodes:
        print("Working on plate", plate)
        colony_plate = colony_metadata.query("plate_barcode == @plate")
        isolate_plate = isolate_metadata.query("src_plate == @plate")
        center_colony = colony_plate[["center_x", "center_y"]].to_numpy()
        center_isolate = isolate_plate[["src_x", "src_y"]].to_numpy()
        i2c_param, *i2c_stats = find_affine(center_isolate, center_colony)
        i2c_func = get_query2target_func(*i2c_param, *i2c_stats)
        center_i2c = i2c_func(center_isolate)
        map_c2i = find_mutual_pairs(center_colony, center_i2c)
        map_i2c = find_mutual_pairs(center_i2c, center_colony)

        g = map_to_network(map_c2i, map_i2c)
        h, bad_colony_idx = remove_bad_nodes(g, "A")
        isolate_out_d = np.array(
            [h.out_degree(f"B{i}") for i in range(len(center_isolate))]
        )
        map_i2cb = np.where(
            isolate_out_d == 0, np.nan, colony_plate.index[map_i2c].to_numpy()
        )
        bad_colonies.extend(colony_metadata.index[bad_colony_idx].to_list())
        isolate2colony.append(pd.Series(map_i2cb, index=isolate_plate.index))

        if log:
            colony_out_d_count = Counter(
                [v for k, v in g.out_degree() if k.startswith("A")]
            )
            isolate_out_d_count = Counter(
                [v for k, v in h.out_degree() if k.startswith("B")]
            )
            rprint(f"\tPlate {plate} alignment completed.")
            rprint(f"\t\tThe plate has {len(center_colony)} colonies.")
            rprint(
                f"\t\t\t{colony_out_d_count[0]} colonies have no paired isolate, "
                f"they are probably not picked."
            )
            if bad_colony_idx:
                warnings.warn(
                    f"\t\t\t{len(bad_colony_idx)} colonies have coliides with other colonies. "
                    f"These colonies will not be used."
                )

            rprint(f"\t\tThe plate has {len(center_isolate)} isolates.")
            rprint(
                f"\t\t\t{isolate_out_d_count[0]} isolates have no paired colony, "
                f"these isolates will not be used."
            )
            for i, n in isolate_out_d_count.items():
                if i:
                    rprint(f"\t\t\t{n} colonies are picked {i} times.")
    colony_metadata = colony_metadata.drop(bad_colonies)
    isolate_metadata["colony_barcode"] = pd.concat(isolate2colony)
    isolate_metadata = isolate_metadata.dropna(subset=["colony_barcode"])
    return colony_metadata, isolate_metadata


def read_camii_isolate_data(
    isolate_count_path: str,
    isolate_metadata_dir: str,
    plate_metadata_path: str,
    colony_metadata_dir: str,
    min_count: int = 10,
    min_purity: float = 0.3,
    log: bool = True,
):
    isolate_metadata = _read_isolate_metadata_rich(
        isolate_metadata_dir, plate_metadata_path
    )
    colony_metadata = _read_colony_metadata(colony_metadata_dir)

    src_plates_in_iso = isolate_metadata["src_plate"].unique()
    plates_in_colony = colony_metadata["plate_barcode"].unique()
    # throw warning if there are source plates missing in colony metadata
    no_colony_plates = set(src_plates_in_iso) - set(plates_in_colony)
    if no_colony_plates:
        warnings.warn(
            f"Source plates {no_colony_plates} are missing in colony metadata. "
            "Isolates from these plates will not be utilized."
        )
    colony_metadata, isolate_metadata = _align_isolate_colony(
        colony_metadata, isolate_metadata, plates_in_colony, log=log
    )

    isolate_count = pd.read_table(isolate_count_path, index_col=0).T
    isolate_count = isolate_count.loc[isolate_metadata.index]
    isolate_count["colony_barcode"] = isolate_metadata["colony_barcode"]
    have_count = isolate_metadata.index.isin(isolate_count.index)
    if not all(have_count):
        warnings.warn(
            "Isolates in isolate metadata are not a subset of colonies in isolate count "
            f"table, {sum(~have_count)} isolates are missing from isolate count table."
        )

    # do QC for colony
    isolate_count = isolate_count.groupby("colony_barcode").sum()
    total_count = isolate_count.sum(axis=1)
    purity = isolate_count.max(axis=1) / total_count
    good_isolates = np.logical_and(total_count >= min_count, purity >= min_purity)
    isolate_count = isolate_count.loc[good_isolates]

    colony_metadata = colony_metadata.loc[isolate_count.index]
    colony_metadata["otu"] = isolate_count.idxmax(axis=1).to_numpy()

    # do QC for ZOTUs.
    otu_count = colony_metadata.value_counts("otu")
    high_ab_otu = otu_count[otu_count >= min_count].index
    colony_metadata = colony_metadata.query("otu in @high_ab_otu")
    if log:
        rprint(
            f"{sum(~good_isolates)} isolates are filtered out by QC, {sum(good_isolates)} are left."
        )
        rprint(
            f"{len(otu_count) - len(high_ab_otu)} ZOTUs are filtered out by QC, {len(high_ab_otu)} are left."
        )
    return colony_metadata


def infer_pairwise_interaction(
    colony_metadata: pd.DataFrame,
    max_dist: int = 25,
) -> pd.DataFrame:
    otus = colony_metadata.otu.unique()
    interaction_stats = {}
    colonies_alone = []
    for plate, colony_metadata_p in colony_metadata.groupby("plate_barcode"):
        coords = colony_metadata_p[["center_x", "center_y"]]
        radius = colony_metadata_p["radius"].to_numpy()
        dist_mtx = cdist(coords, coords) - radius - radius[:, None]
        dist_mtx[dist_mtx > max_dist] = -1
        # set diagonal to -1 to avoid self-interaction
        dist_mtx[np.diag_indices_from(dist_mtx)] = -1
        colonies_alone.append(colony_metadata_p.index[(dist_mtx == -1).all(axis=1)])
        interaction_stats[plate] = {
            "area": colony_metadata_p.area.to_numpy(),
            "dist_mtx": dist_mtx,
            "otu2idxs": {
                otu: np.where(colony_metadata_p.otu == otu)[0] for otu in otus
            },
        }
    alone_areas = (
        colony_metadata.loc[np.concatenate(colonies_alone)]
        .groupby("otu")["area"]
        .agg(list)
    )
    receptors, donors, fcs, pvals, num_interactions = [], [], [], [], []
    for receptor_otu_idx, donor_otu_idx in product(range(len(otus)), repeat=2):
        receptor_otu = otus[receptor_otu_idx]
        donor_otu = otus[donor_otu_idx]
        if (
            receptor_otu == donor_otu
            or receptor_otu.endswith("unknown")
            or donor_otu.endswith("unknown")
        ):
            continue
        areas_list = []
        num_interaction = 0
        # aggregate all neighboring pairs of receptor_otu and donor_otu
        for plate, stats in interaction_stats.items():
            areas = stats["area"]
            dist_mtx = stats["dist_mtx"]
            otu2idxs = stats["otu2idxs"]
            receptor_idx, donor_idx = otu2idxs[receptor_otu], otu2idxs[donor_otu]
            pairing = np.where(dist_mtx[receptor_idx][:, donor_idx] != -1)
            num_interaction += len(pairing[0])
            areas_list.append(areas[receptor_idx[np.unique(pairing[0])]])
        areas = np.concatenate(areas_list)
        areas_alone = alone_areas.get(donor_otu, [])
        if not len(areas) or not len(areas_alone):
            fc = utest_pval = -1
        else:
            fc = areas.mean() / np.mean(areas_alone)
            utest_pval = mannwhitneyu(areas, areas_alone).pvalue
        fcs.append(fc)
        pvals.append(utest_pval)
        receptors.append(receptor_otu)
        donors.append(donor_otu)
        num_interactions.append(num_interaction)
    return pd.DataFrame(
        {
            "receptor": receptors,
            "donor": donors,
            "fc": fcs,
            "num_interactions": num_interactions,
            "pval": pvals,
        }
    ).sort_values(["receptor", "donor"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Isolate interaction inference from CAMII data."
    )
    parser.add_argument(
        "-c",
        "--isolate_count_path",
        type=str,
        required=True,
        help="Path to isolate count table.",
    )
    parser.add_argument(
        "-im",
        "--isolate_metadata_dir",
        type=str,
        required=True,
        help="Path to isolate metadata directory.",
    )
    parser.add_argument(
        "-pm",
        "--plate_metadata_path",
        type=str,
        required=True,
        help="Path to plate metadata table.",
    )
    parser.add_argument(
        "-cm",
        "--colony_metadata_dir",
        type=str,
        required=True,
        help="Path to colony metadata directory.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Minimum number of isolates per colony.",
    )
    parser.add_argument(
        "--min_purity",
        type=float,
        default=0.3,
        help="Minimum purity of colony.",
    )
    parser.add_argument(
        "--max_dist",
        type=int,
        default=50,
        help="Minimum distance between colonies.",
    )
    parser.add_argument(
        "--min_fc",
        type=float,
        default=1.2,
        help="Minimum fold change of interaction.",
    )
    parser.add_argument(
        "--min_num_interactions",
        type=int,
        default=3,
        help="Minimum number of interactions.",
    )
    parser.add_argument(
        "--max_qval",
        type=float,
        default=0.1,
        help="Minimum q-value of interaction.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path to output file.",
    )

    args = parser.parse_args()
    isolate_count_path = args.isolate_count_path
    isolate_metadata_dir = args.isolate_metadata_dir
    plate_metadata_path = args.plate_metadata_path
    colony_metadata_dir = args.colony_metadata_dir
    min_count = args.min_count
    min_purity = args.min_purity
    max_dist = args.max_dist
    min_fc = args.min_fc
    min_num_interactions = args.min_num_interactions
    max_qval = args.max_qval
    output_path = args.output_path

    colony_metadata = read_camii_isolate_data(
        isolate_count_path,
        isolate_metadata_dir,
        plate_metadata_path,
        colony_metadata_dir,
        min_count=min_count,
        min_purity=min_purity,
    )
    interaction_df = infer_pairwise_interaction(
        colony_metadata,
        max_dist=max_dist,
    )
    interaction_df = interaction_df.query(
        "(fc >= @min_fc or fc <= 1/@min_fc) and num_interactions >= @min_num_interactions"
    )
    interaction_df["qval"] = false_discovery_control(interaction_df["pval"])
    interaction_df = interaction_df.query("qval <= @max_qval")
    # add ZOTU count to interaction_df by adding rows to interaction_df with only
    # "receptor" and "num_interactions" columns.
    zotus_in_interaction = sorted(
        set(interaction_df.receptor.unique()) | set(interaction_df.donor.unique())
    )
    count_df = (
        colony_metadata.value_counts("otu").loc[zotus_in_interaction].reset_index()
    )
    count_df.columns = ["receptor", "num_interactions"]
    interaction_df = pd.concat([interaction_df, count_df])
    interaction_df.to_csv(output_path, index=False)
