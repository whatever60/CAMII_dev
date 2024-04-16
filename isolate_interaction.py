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

# from align import find_affine, get_query2target_func, find_mutual_pairs
from utils import _get_time_points


def _read_colony_metadata(
    colony_metadata_dir: str, time: int | str = "max"
) -> pd.DataFrame:
    # read colony metadata from CAMII picking pipeline
    colony_metadatas = []
    for p, f in zip(
        *_get_time_points(colony_metadata_dir, ext="_metadata.csv", time=time)
    ):
        df_colony = pd.read_csv(f)
        if (df_colony.plate_barcode != p).any():
            rprint(
                f"WARNING: plate barcode in colony metadata does not match file name: {p}."
            )
        df_colony["colony_barcode"] = (
            df_colony.plate_barcode
            + "_"
            + df_colony.center_x.round(3).astype(str)
            + "x"
            + df_colony.center_y.round(3).astype(str)
        )
        df_colony = df_colony.set_index("colony_barcode", verify_integrity=True)
        if len(df_colony):
            colony_metadatas.append(df_colony)
        else:
            rprint(f"WARNING: No colony metadata found for plate {p}.")
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
        plate_metadata[["barcode", "medium_type", "sample_type"]].rename(
            {"barcode": "src_plate"}, axis=1
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
    aligners = []
    if plate_barcodes is None:
        plate_barcodes = sorted(
            set(colony_metadata[colony_plate_key])
            & set(isolate_metadata[isolate_plate_key])
        )
    for plate in plate_barcodes:
        rprint("Working on plate", plate)
        colony_plate = colony_metadata.query("plate_barcode == @plate")
        isolate_plate = isolate_metadata.query("src_plate == @plate")
        aligner = Aligner()
        aligner._meta_rgb = colony_plate
        aligner._meta_isolate = isolate_plate
        query, target = "isolate", "rgb"
        aligner.fit(query=query, target=target, flip=False)
        aligner.transform(query=query, target=target)

        g = getattr(aligner, f"_graph_{target}_{query}")
        h = getattr(aligner, f"_graph_{target}_{query}_clean")
        bad_colony_idx = getattr(aligner, f"_bad_{target}_{query}2{target}_idx")
        map_i2cb = np.where(
            getattr(aligner, f"_map_{target}_{query}2{target}_clean") == -1,
            np.nan,
            colony_plate.index[aligner._map_rgb_isolate2rgb].to_numpy(),
        )
        bad_colonies.extend(colony_metadata.index[bad_colony_idx].to_list())
        isolate2colony.append(pd.Series(map_i2cb, index=isolate_plate.index))
        aligners.append(aligner)

        if log:
            colony_out_d_count = Counter(
                [v for k, v in g.out_degree() if k.startswith(g.name_top)]
            )
            isolate_out_d_count = Counter(
                [v for k, v in h.out_degree() if k.startswith(g.name_bottom)]
            )
            colony_in_d_count = Counter(
                [v for k, v in h.in_degree() if k.startswith(g.name_top)]
            )
            rprint(f"\tPlate {plate} alignment completed.")
            rprint(f"\t\tThe plate has {len(colony_plate)} colonies.")
            rprint(
                f"\t\t\t{colony_out_d_count[0]} colonies have no paired isolate, "
                f"they are probably not picked."
            )
            if bad_colony_idx:
                warnings.warn(
                    f"\t\t\t{len(bad_colony_idx)} colonies have collides with other "
                    "colonies. These colonies will not be used."
                )

            rprint(f"\t\tThe plate has {len(isolate_plate)} isolates.")
            rprint(
                f"\t\t\t{isolate_out_d_count[0]} isolates have no paired colony, "
                f"these isolates will not be used."
            )
            for i, n in colony_in_d_count.items():
                if i:
                    rprint(f"\t\t\t{n} colonies are picked {i} times.")
    colony_metadata = colony_metadata.drop(bad_colonies)
    isolate_metadata["colony_barcode"] = pd.concat(isolate2colony)
    isolate_metadata = isolate_metadata.dropna(subset=["colony_barcode"])
    return colony_metadata, isolate_metadata, aligners


def read_camii_isolate_data(
    isolate_count_path: str,
    isolate_metadata_path: str,
    colony_metadata_dir: str,
    time: str | int = "max",
    min_count: int = 10,
    min_purity: float = 0.3,
    log: bool = True,
):
    # isolate_metadata = _read_isolate_metadata_rich(
    #     isolate_metadata_dir, plate_metadata_path
    # )
    isolate_metadata = pd.read_table(isolate_metadata_path, index_col="sample")
    num_isolates_picked = len(isolate_metadata)
    colony_metadata = _read_colony_metadata(colony_metadata_dir)

    src_plates_in_iso = isolate_metadata["src_plate"].unique()
    plates_in_colony = colony_metadata["plate_barcode"].unique()
    # throw warning if there are source plates missing in colony metadata
    no_colony_plates = set(src_plates_in_iso) - set(plates_in_colony)
    extra_plates = set(plates_in_colony) - set(src_plates_in_iso)
    plates_in_colony = sorted(set(plates_in_colony) & set(src_plates_in_iso))
    if no_colony_plates:
        warnings.warn(
            f"Source plates {no_colony_plates} are missing in colony metadata. "
            "Isolates from these plates will not be utilized."
        )
    if extra_plates:
        warnings.warn(
            f"Colony plates {extra_plates} are not in isolate metadata, meaning that they "
            "are not picked. Colonies from these plates will not be utilized."
        )
    colony_metadata, isolate_metadata, aligners = _align_isolate_colony(
        colony_metadata, isolate_metadata, plates_in_colony, log=log
    )
    num_colonies_detected = sum(len(a._meta_rgb) for a in aligners)
    num_colonies_paired = sum(
        np.unique(
            a._map_rgb_isolate2rgb_clean[a._map_rgb_isolate2rgb_clean != -1]
        ).shape[0]
        for a in aligners
    )

    isolate_count = pd.read_table(isolate_count_path, index_col="#OTU ID").transpose()
    missing_isolates = np.setdiff1d(isolate_metadata.index, isolate_count.index)
    if missing_isolates.any():
        warnings.warn(
            "Isolates in isolate metadata are not a subset of colonies in isolate count "
            f"table, {len(missing_isolates)} isolates are missing from isolate count table."
        )
    isolate_count = isolate_count.loc[isolate_metadata.index]

    colony_count = pd.concat(
        [
            aligner.agg(query="isolate", target="rgb", data=isolate_count)
            for aligner in aligners
        ],
        axis=0,
    )
    total_count = colony_count.sum(axis=1)
    purity = colony_count.max(axis=1) / total_count
    good_isolates = np.logical_and(total_count >= min_count, purity >= min_purity)
    colony_count = colony_count.loc[good_isolates]

    colony_metadata = colony_metadata.loc[colony_count.index]
    num_colonies_paired_good = len(colony_metadata)
    colony_metadata["otu"] = colony_count.idxmax(axis=1).to_numpy()

    # do QC for ZOTUs.
    otu_count = colony_metadata.value_counts("otu")
    high_ab_otu = otu_count[otu_count >= min_count].index
    colony_metadata = colony_metadata.query("otu in @high_ab_otu")
    if log:
        rprint(f"{num_isolates_picked} isolates are picked.")
        rprint(
            f"{num_colonies_detected} colonies are detected, "
            f"{num_colonies_paired} are paired with isolates, "
            f"{num_colonies_paired_good} are paired with good isolates, and "
            f"{len(colony_metadata)} will be involved in interaction inference."
        )
        rprint(
            f"{len(otu_count) - len(high_ab_otu)} ZOTUs are filtered out by QC, {len(high_ab_otu)} are left."
        )
    return colony_metadata
    # isolate_count = pd.concat(
    #     [
    #         isolate_count,
    #         pd.DataFrame(0, index=missing_isolates, columns=isolate_count.columns),
    #     ],
    #     axis=0,
    # ).loc[isolate_metadata.index]
    # reorder
    isolate_count = isolate_count.loc[isolate_metadata.index]
    colony_count = pd.concat(
        [
            aligner.agg(query="isolate", target="rgb", data=isolate_count)
            for aligner in aligners
        ],
        axis=0,
    )
    # isolate_count["colony_barcode"] = isolate_metadata["colony_barcode"]

    # do QC for colony
    # isolate_count = isolate_count.groupby("colony_barcode").sum()
    total_count = colony_count.sum(axis=1)
    purity = colony_count.max(axis=1) / total_count
    good_isolates = np.logical_and(total_count >= min_count, purity >= min_purity)
    colony_count = colony_count.loc[good_isolates]

    colony_metadata = colony_metadata.loc[colony_count.index]
    colony_metadata["otu"] = colony_count.idxmax(axis=1).to_numpy()

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


def calc_interaction(
    colony_metadata: pd.DataFrame,
    max_dist: int = 25,
) -> pd.DataFrame:
    interaction_stats = []
    for plate, colony_metadata_p in colony_metadata.groupby("plate_barcode"):
        coords = colony_metadata_p[["center_x", "center_y"]]
        radius = colony_metadata_p["radius"].to_numpy()
        dist_mtx = cdist(coords, coords) - radius - radius[:, None]
        dist_mtx[dist_mtx > max_dist] = -1
        # set diagonal to -1 to avoid self-interaction
        dist_mtx[np.diag_indices_from(dist_mtx)] = -1
        receptors, donors = np.where(dist_mtx != -1)
        interaction_stats.append(
            pd.DataFrame(
                {
                    "plate_barcode": plate,
                    "receptor": colony_metadata_p.index[receptors],
                    "donor": colony_metadata_p.index[donors],
                    "distance": dist_mtx[receptors, donors],
                }
            )
        )
    return pd.concat(interaction_stats)


def infer_interaction(
    colony_metadata: pd.DataFrame, interaction_stats: pd.DataFrame
) -> pd.DataFrame:
    # colonies_alone.append(colony_metadata_p.index[(dist_mtx == -1).all(axis=1)])
    # interaction_stats[plate] = {
    #     "area": colony_metadata_p.area.to_numpy(),
    #     "dist_mtx": dist_mtx,
    #     "otu2idxs": {
    #         otu: np.where(colony_metadata_p.otu == otu)[0] for otu in otus
    #     },
    # }
    colony_idx2otu = colony_metadata["otu"].to_dict()
    colony_idx2area = colony_metadata["area"].to_dict()
    alone_areas = (
        colony_metadata.query("index not in @interaction_stats['receptor'].unique()")
        .groupby("otu")["area"]
        .agg(list)
    )

    interaction_stats = interaction_stats.copy()
    interaction_stats["receptor_otu"] = interaction_stats["receptor"].map(
        colony_idx2otu
    )
    interaction_stats["donor_otu"] = interaction_stats["donor"].map(colony_idx2otu)
    # otus = colony_metadata["otu"].unique()
    # alone_areas = (
    #     colony_metadata.loc[np.concatenate(colonies_alone)]
    #     .groupby("otu")["area"]
    #     .agg(list)
    # )
    receptors, donors, fcs, pvals, num_interactions = [], [], [], [], []
    # for receptor_otu_idx, donor_otu_idx in product(range(len(otus)), repeat=2):
    for (receptor_otu, donor_otu), stats in interaction_stats.groupby(
        ["receptor_otu", "donor_otu"],
    ):
        # receptor_otu = colony_idx2otu[receptor_otu_idx]
        # donor_otu = colony_idx2otu[donor_otu_idx]
        if (
            receptor_otu == donor_otu
            or receptor_otu.endswith("unknown")
            or donor_otu.endswith("unknown")
            or receptor_otu.endswith("_UNKNOWN")
            or donor_otu.endswith("_UNKNOWN")
        ):
            continue
        areas = (
            stats.drop_duplicates(["receptor", "plate_barcode"])["receptor"]
            .map(colony_idx2area)
            .to_numpy()
        )
        num_interaction = len(stats)
        # aggregate all neighboring pairs of receptor_otu and donor_otu
        # areas_list.append(areas[receptor_idx[np.unique(pairing[0])]])
        # areas = np.concatenate(areas_list)
        areas_alone = alone_areas.get(donor_otu, [])
        if not len(areas) or not len(areas_alone):
            continue
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
        help="Path to isolate count tsv.",
    )
    parser.add_argument(
        "-im",
        "--isolate_metadata_path",
        type=str,
        required=True,
        help="Path to isolate metadata tsv.",
    )
    parser.add_argument(
        "-cm",
        "--colony_metadata_dir",
        type=str,
        required=True,
        help="Path to colony metadata directory.",
    )
    parser.add_argument("-t", "--time_label", default="max", type=str)
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
        "--output_dir",
        type=str,
        required=True,
        help="Path to output file.",
    )

    args = parser.parse_args()
    isolate_count_path = args.isolate_count_path
    isolate_metadata_path = args.isolate_metadata_path
    colony_metadata_dir = args.colony_metadata_dir
    time_label = args.time_label
    min_count = args.min_count
    min_purity = args.min_purity
    max_dist = args.max_dist
    min_fc = args.min_fc
    min_num_interactions = args.min_num_interactions
    max_qval = args.max_qval
    output_dir = args.output_dir

    colony_metadata = read_camii_isolate_data(
        isolate_count_path,
        isolate_metadata_path,
        colony_metadata_dir,
        time=int(time_label) if time_label.isdigit() else time_label,
        min_count=min_count,
        min_purity=min_purity,
    )
    interaction_df = calc_interaction(colony_metadata, max_dist=max_dist)
    interaction_summary_df = infer_interaction(colony_metadata, interaction_df)
    interaction_summary_df = interaction_summary_df.query(
        "(fc >= @min_fc or fc <= 1/@min_fc) "
        "and num_interactions >= @min_num_interactions "
        "and pval >= 0"
    )
    interaction_summary_df["qval"] = false_discovery_control(
        interaction_summary_df["pval"]
    )
    interaction_summary_df = interaction_summary_df.query("qval <= @max_qval")
    os.makedirs(output_dir, exist_ok=True)
    colony_metadata[["otu", "plate_barcode", "area"]].to_csv(
        os.path.join(output_dir, "colony_metadata.csv")
    )
    interaction_df.to_csv(
        os.path.join(output_dir, "colony_interaction.csv"), index=False
    )
    interaction_summary_df.to_csv(
        os.path.join(output_dir, "colony_interaction_summary.csv"), index=False
    )
    # add ZOTU count to interaction_summary_df by adding rows to interaction_summary_df with only
    # "receptor" and "num_interactions" columns.
    # zotus_in_interaction = sorted(
    #     set(interaction_summary_df.receptor.unique()) | set(interaction_summary_df.donor.unique())
    # )
    # count_df = (
    #     colony_metadata.value_counts("otu").loc[zotus_in_interaction].reset_index()
    # )
    # count_df.columns = ["receptor", "num_interactions"]
    # interaction_summary_df = pd.concat([interaction_summary_df, count_df])
