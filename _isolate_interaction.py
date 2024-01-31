#!/usr/bin/env python

import argparse
import subprocess
import warnings
import os
import sys
import glob
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse as ss
from scipy.stats import mannwhitneyu, false_discovery_control
import pandas as pd
import polars as pl
from rich import print as rprint

from align import find_affine, get_query2target_func, find_mutual_pairs


def read_data(
    # otu_taxonomy_path: str,
    isolate_count_path: str,
    plate_metadata_path: str,
    isolate_metadata_dir: str,
    colony_metadata_dir: str,
):
    """
    Read relevant information into dataframes, including
        - Taxonomy classification of ASVs
        - Count table of ASVs in each isolate
        - Metadata of each plate, such as experiment group, medium kind, etc.
        - Metadata of each isolate, such as position on plate, purity, etc.
        - Metadata of each colony, such as position on plate, radius, color, etc.

    The core of this function other than reading data is to join relevant columns into
        isolate metadata table and perform quality control on isolates.
    """
    # otu_taxonomy = pd.read_table(otu_taxonomy_path, index_col="otu")
    plate_metadata = pd.read_csv(plate_metadata_path, index_col="barcode")

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
        df_colony = df_colony.set_index("colony_barcode")
        colony_metadatas.append(df_colony)
    colony_metadata = pd.concat(colony_metadatas)
    colony_metadata = colony_metadata.merge(
        plate_metadata[["group"]], left_on="plate_barcode", right_index=True, how="left"
    )

    # read picking log from RapidPick software log
    isolate_metadatas = []
    for f in glob.glob(f"{isolate_metadata_dir}/Destination * - * *.csv"):
        dest_plate_barcode = os.path.basename(f).split(" ")[1]
        # skip first 2 lines, and the 3rd line is the header line
        # the 3 columns are "Source", "Source Plate", "Destination Well"
        # The "Source" column is formatted as "(<x>, <y>)", extract x and y from it
        # Add an isolate ID column of "<dist_plate>w<well>"
        df_isolate = pd.read_csv(f, skiprows=2)
        df_isolate.columns = df_isolate.columns.str.strip()
        df_isolate[["src_x", "src_y"]] = (
            df_isolate.Source.str.extract(r"\((\d+.\d+), (\d+.\d+)\)")
            .astype(float)
            .to_numpy()
        )
        # import pdb; pdb.set_trace()
        del df_isolate["Source"]
        df_isolate["isolate_barcode"] = (
            dest_plate_barcode + "w" + df_isolate["Destination Well"]
        )
        df_isolate = df_isolate.set_index("isolate_barcode")
        isolate_metadatas.append(df_isolate)
    isolate_metadata = pd.concat(isolate_metadatas)

    #  ===== Align picking log coordinates onto colony metadata coordinates. =====
    # We need to align because there are several transformation from colony detection to
    # the final picking by the robot, including lens calibration and other internal
    # adjustment.
    # We need to pair because there is not necessarily a 1:1 mapping between
    # picking log and colony metadata Some colony might have 0, 1, or more associated
    # picking, similarly for picking. This is because of reasons such as double picking,
    # manual selection.
    # We are using colony metadata as the target since we have associated image features.
    src_plates_in_iso = isolate_metadata["Source Plate"].unique()
    plates_in_colony = colony_metadata["plate_barcode"].unique()
    # throw warning if there are source plates missing in colony metadata
    no_colony_plates = set(src_plates_in_iso) - set(plates_in_colony)
    if no_colony_plates:
        warnings.warn(
            f"Source plates {no_colony_plates} are missing in colony metadata. "
            "Isolates from these plates will not be utilized."
        )
    # colony2isolates = {}
    isolate2colony = []
    bad_colonies = []
    for plate in plates_in_colony:
        print("Working on plate", plate)
        colony_plate = colony_metadata.query("plate_barcode == @plate")
        isolate_plate = isolate_metadata.query("`Source Plate` == @plate")
        center_colony = colony_plate[["center_x", "center_y"]].to_numpy()
        center_isolate = isolate_plate[["src_x", "src_y"]].to_numpy()
        i2c_param, *i2c_stats = find_affine(center_isolate, center_colony)
        i2c_func = get_query2target_func(*i2c_param, *i2c_stats)
        center_i2c = i2c_func(center_isolate)
        map_c2i = find_mutual_pairs(center_colony, center_i2c)
        map_i2c = find_mutual_pairs(center_i2c, center_colony)
        # c_barcodes = colony_plate.index[map_i2c[good_isolates]]
        # s_barcode = isolate_plate.index[good_isolates]
        # for cb, sb in zip(c_barcodes, s_barcode):
        #     isolate2colony[sb].append(cb)

        num_picking = pd.Series(map_c2i[map_c2i != -1]).value_counts()
        collide_colonies = num_picking[num_picking >= 2].index.to_list()
        bad_colonies.extend(collide_colonies)
        map_i2c = np.where(np.isin(map_i2c, collide_colonies), -1, map_i2c)
        map_i2cb = colony_plate.index[map_i2c].to_numpy()
        # -1 to NaN
        map_i2cb = np.where(map_i2c == -1, np.nan, map_i2cb)
        isolate2colony.append(pd.Series(map_i2cb, index=isolate_plate.index))

        rprint(f"\tPlate {plate} alignment completed.")
        rprint(f"\t\tThe plate has {len(center_colony)} colonies.")
        rprint(
            f"\t\t{(map_c2i == -1).sum()} colonies have no paired isolate, "
            f"they are probably not picked."
        )
        if collide_colonies:
            warnings.warn(
                f"\t\t{len(collide_colonies)} colonies have coliides with other colonies. "
                f"These colonies will not be used."
            )

        num_colony = pd.Series(map_i2c[map_i2c != -1]).value_counts().value_counts()
        rprint(f"\t\tThe plate has {len(center_isolate)} isolates.")
        rprint(
            f"\t\t{(map_i2c == -1).sum()} isolates have no paired colony, "
            f"these isolates will not be used."
        )
        for i, n in num_colony.items():
            rprint(f"\t\t{n} colonies are picked {i} times.")

    colony_metadata = colony_metadata.drop(bad_colonies)
    isolate_metadata["colony_barcode"] = pd.concat(isolate2colony)
    isolate_metadata = isolate_metadata.dropna(subset=["colony_barcode"])

    # Read isolate count table. Indices are ZOTUs and columns are isolate barcodes
    isolate_count = pd.read_table(isolate_count_path, index_col=0).T
    in_table = isolate_metadata.index.isin(isolate_count.index)
    if not all(in_table):
        warnings.warn(
            "Isolates in isolate metadata are not a subset of colonies in isolate count "
            f"table, {sum(~in_table)} isolates are missing from isolate count table."
        )
    # aggregate to colony count
    isolate_count["colony_barcode"] = isolate_metadata.colony_barcode
    isolate_count = isolate_count.dropna(subset=["colony_barcode"])
    colony_count = isolate_count.groupby("colony_barcode").sum()
    # do QC
    total_count = colony_count.sum(axis=1)
    purity = colony_count.max(axis=1) / total_count
    good_colonies = np.logical_and(total_count >= cutoff_count, purity >= cutoff_purity)
    colony_count = colony_count.loc[good_colonies]
    colony_metadata = colony_metadata.loc[colony_count.index]
    rprint(f"{sum(~good_colonies)} colonies are filtered out by QC.")
    colony_metadata["otu"] = colony_count.idxmax(axis=1).to_numpy()

    return colony_metadata

    # pass_colony = isolate_count.index.to_list()
    # isolate_metadata = isolate_metadata.query("isolateID in @pass_colony")
    # isolate_metadata = isolate_metadata.merge(
    #     isolate_count[["OTU1"]].rename(columns={"OTU1": "otu"}),
    #     left_on="isolateID",
    #     right_index=True,
    #     how="left",
    # )
    # isolate_metadata = isolate_metadata.merge(
    #     otu_taxonomy[["genus", "family"]], left_on="otu", right_index=True, how="left"
    # )


def groupwise_pairwise_dist(
    df: pd.DataFrame, group_key: str | list[str]
) -> pd.DataFrame:
    """For colonies on each plate, calculate the distance between each pair of colonies."""
    # Initialize a list to store results
    results = []

    # Group the data by plate
    grouped = df.groupby(group_key)

    for g, group_df in grouped:
        # Number of colonies in the group
        n = len(group_df)

        # Create a grid of all combinations of indices
        ix1, ix2 = np.triu_indices(n, k=1)

        # Select the rows for each combination
        rows1 = grouped.iloc[ix1].reset_index(drop=True)
        rows2 = grouped.iloc[ix2].reset_index(drop=True)

        # Vectorized distance calculation
        coord_cols = ["center_x", "center_y"]
        distances = np.linalg.norm(
            rows1[coord_cols].to_numpy() - rows2[coord_cols].to_numpy(), axis=1
        )

        # Create a DataFrame from the results
        pair_data = pd.DataFrame(
            {
                "plate_barcode": plate,
                "colony_1": rows1.index.to_numpy(),
                "colony_2": rows2.index.to_numpy(),
                "radius_1": rows1.radius.to_numpy(),
                "radius_2": rows2.radius.to_numpy(),
                "distance": distances,
                "distance_clean": distances
                - rows1.radius.to_numpy()
                - rows2.radius.to_numpy(),
            }
        )
        # exchange main and adjacent
        pair_data2 = pair_data.copy()
        pair_data2.columns = [
            "plate_barcode",
            "colony_2",
            "colony_1",
            "radius_2",
            "radius_1",
            "distance",
            "distance_clean",
        ]

        results.append(pair_data)
        results.append(pair_data2)

    # Concatenate results from all groups
    return pd.concat(results, ignore_index=True)


def calc_interaction_df(
    colony_metadata: pd.DataFrame,
    colony_metadata_zotu_stat: pd.DataFrame,
    colony_adjacency_info: pd.DataFrame,
    count_cutoff: int,
    neig_count_cutoff: int,
    dist_cutoff: int,
) -> pd.DataFrame:
    colony_metadata_zotu_stat.query("count >= @count_cutoff"),
    colony_adjacency_info.query("distance_clean <= @distance_cutoff"),
    # Initializing an empty DataFrame
    interaction_df = pd.DataFrame()

    for i in range(len(colony_metadata_zotu_stat)):
        tmp_otu = colony_metadata_zotu_stat.index[i]
        # tmp_colony_metadata = colony_metadata[colony_metadata["otu"] == tmp_otu]

        # tmp_adjacency_info = colony_adjacency_info[
        #     colony_adjacency_info["colony_1"].isin(
        #         tmp_colony_metadata["colony_barcode"]
        #     )
        # ]
        tmp_colony_metadata = colony_metadata.query("otu == @tmp_otu")
        tmp_adjacency_info = (
            colony_adjacency_info.query("colony_1 in @tmp_colony_metadata.index")
            .merge(
                colony_metadata[["otu", "Area"]],
                left_on="colony_2",
                right_index=True,
                how="left",
            )
            .rename(columns={"otu": "otu_2", "Area": "area_1"})
        )
        tmp_adjacency_info["area_1"] = np.sqrt(tmp_adjacency_info.area_1)

        # tmp_adjacency_info["otu_2"] = colony_metadata.loc[
        #     tmp_adjacency_info["colony_2"].astype(str), "otu"
        # ].to_numpy()

        # tmp_adjacency_info["area_1"] = np.sqrt(
        #     colony_metadata.loc[
        #         tmp_adjacency_info["colony_2"].astype(str), "Area"
        #     ].to_numpy()
        # )

        # Grouping and summarising data
        tmp_adjacency_area_stat = (
            tmp_adjacency_info.groupby("otu_2")
            .agg({"area_1": "mean", "colony_1": "count"})
            .reset_index()
        )
        tmp_adjacency_area_stat.rename(columns={"colony_1": "count"}, inplace=True)
        tmp_adjacency_area_stat = tmp_adjacency_area_stat[
            (~tmp_adjacency_area_stat["otu_2"].isna())
            & (tmp_adjacency_area_stat["count"] >= interaction_cutoff)
        ]

        solo_colony_metadata = tmp_colony_metadata.query("num_of_adjacency == 0")

        # Statistical testing
        pvalue_up_list = []
        pvalue_down_list = []
        for j in range(len(tmp_adjacency_area_stat)):
            tmp_otu_add = tmp_adjacency_area_stat.iloc[j]["otu_2"]
            # tmp_withOTU_area = tmp_adjacency_info[
            #     tmp_adjacency_info["otu_2"] == tmp_otu_add
            # ]["area_1"]
            tmp_with_otu_area = tmp_adjacency_info.query("otu_2 == @tmp_otu_add")[
                "area_1"
            ]
            tmp_solo_area = np.sqrt(solo_colony_metadata["Area"])
            # if len(tmp_solo_area) == 0 or len(tmp_with_otu_area) == 0:
            test_up = mannwhitneyu(tmp_solo_area, tmp_with_otu_area, alternative="less")
            test_down = mannwhitneyu(
                tmp_solo_area, tmp_with_otu_area, alternative="greater"
            )
            pvalue_up_list.append(test_up.pvalue)
            pvalue_down_list.append(test_down.pvalue)

        tmp_adjacency_area_stat["pvalue_up"] = pvalue_up_list
        tmp_adjacency_area_stat["pvalue_down"] = pvalue_down_list
        tmp_adjacency_area_stat["area_solo"] = np.mean(
            np.sqrt(solo_colony_metadata["Area"])
        )
        tmp_adjacency_area_stat["total_interaction"] = tmp_adjacency_area_stat[
            "count"
        ].sum()
        tmp_adjacency_area_stat["otu_1"] = tmp_otu
        interaction_df = pd.concat([interaction_df, tmp_adjacency_area_stat])

    # Adjusting p-values for multiple testing
    interaction_df["fdr_up"] = interaction_df["pvalue_up"].transform(
        lambda p: pd.Series(p).apply(lambda x: min(1, x * len(p)))
    )
    interaction_df["fdr_down"] = interaction_df["pvalue_down"].transform(
        lambda p: pd.Series(p).apply(lambda x: min(1, x * len(p)))
    )

    # Calculating log2 fold change
    interaction_df["area_log2FC"] = np.log2(
        interaction_df.area_1 / interaction_df.area_solo
    )
    return interaction_df


if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Process data files")
    parser.add_argument(
        "-t",
        "--otu_taxonomy_path",
        type=str,
        required=True,
        help="Path to OTU taxonomy file",
    )
    parser.add_argument(
        "-ic",
        "--isolate_count_path",
        type=str,
        required=True,
        help="Path to isolate count file",
    )
    parser.add_argument(
        "-pm",
        "--plate_metadata_path",
        type=str,
        required=True,
        help="Path to plate metadata file",
    )
    parser.add_argument(
        "-im",
        "--isolate_metadata_dir",
        type=str,
        required=True,
        help="Path to second isolate metadata file",
    )
    parser.add_argument(
        "-cm",
        "--colony_metadata_dir",
        type=str,
        required=True,
        help="Path to colony metadata file",
    )

    args = parser.parse_args()

    otu_taxonomy_path = args.otu_taxonomy_path
    isolate_count_path = args.isolate_count_path
    plate_metadata_path = args.plate_metadata_path
    isolate_metadata_dir = args.isolate_metadata_dir
    colony_metadata_dir = args.colony_metadata_dir

    # Filtering based on conditions
    cutoff_count = 10
    cutoff_purity = 0.3
    cutoff_isolate_count = 10
    distance_cutoff = 25
    interaction_cutoff = 3
    min_lfc = np.log2(1.2)
    # target_input = "soil-2", "soil-6"

    colony_metadata = read_data(
        isolate_count_path,
        plate_metadata_path,
        isolate_metadata_dir,
        colony_metadata_dir,
    )

    # colony_adjacency_info = groupwise_pairwise_dist(colony_metadata.reset_index(), group_key="plate_barcode")
    # colony_adjacency_info_colony_stat = (
    #     colony_adjacency_info.query("distance_clean <= @distance_cutoff")
    #     .groupby(["plate_barcode", "colony_1"])
    #     .size()
    #     .reset_index(name="count")
    # ).set_index("colony_1")
    # colony_metadata["num_of_adjacency"] = colony_adjacency_info_colony_stat["count"]
    # colony_metadata.num_of_adjacency.fillna(0, inplace=True)
    # # Grouping and summarising data
    # isolate_metadata_zotu_stat = (
    #     colony_metadata.groupby("otu").size().reset_index(name="count").set_index("otu")
    # )
    # interaction_df = calc_interaction_df(
    #     colony_metadata,
    #     isolate_metadata_zotu_stat,
    #     colony_adjacency_info,
    #     count_cutoff=cutoff_isolate_count,
    #     dist_cutoff=distance_cutoff,
    #     neig_count_cutoff=interaction_cutoff,
    # )

    # The above commented code needs to be refactored by using more efficient functions
    #  such as scipy.spatial.distance.pdist
    # Steps are:
    # - Find ZOTUs with enough occurrences, i.e., high abundance ZOTUs.
    # - For each group (plate_barcode), calculate pairwise distance between centers of
    #    all colonies of high abundance ZOTUs on it.
    # - Calculate edge distance by subtracting the radius of each colony from the pairwise distance.
    # - For each pair of ZOTUs with enough pairing, treat one ZOTU as recipient and the
    #    other as donor, perform U test comparing the area of recipient ZOTU by itself
    #    and with the donor as a neighbor, and also calculate area log2 fold change.
    #    Swap the role of recipient and donor and repeat the test.
    #    This will result in a dirtected graph where each edge has a sign, strength and
    #    signifiance. Absence of an edge means one of the three things:
    #        - The two ZOTUs are not found to be neighbors often enough.
    #        - Area difference with or without neighbor is not significant.
    #        - Area difference is not large enough.
    # - Adjust p-values for multiple testing.

    otu_count = colony_metadata.value_counts("otu")
    high_ab_otu = otu_count.index[otu_count >= cutoff_isolate_count]
    colony_metadata = colony_metadata.query("otu in @high_ab_otu")
    plates = colony_metadata["plate_barcode"].unique()
    dist_mtxs = []
    otu_list = []
    area_list = []
    num_neighbors = []
    for p in plates:
        colony_meta_p = colony_metadata.query("plate_barcode == @p")
        coords = colony_meta_p[["center_x", "center_y"]]
        radius = colony_meta_p.radius.to_numpy()
        dist_mtx = cdist(coords, coords) - radius - radius[:, None]
        dist_mtx[dist_mtx > distance_cutoff] = 0
        num_neighbors.append(
            pd.Series((dist_mtx > 0).sum(axis=1), index=colony_meta_p.index)
        )
        dist_mtxs.append(dist_mtx)
        otu_list.append(colony_meta_p.otu.to_numpy())
        area_list.append(colony_meta_p.area.to_numpy())
    colony_metadata["num_of_adjacency"] = pd.concat(num_neighbors)

    recipient, donors, l2fcs, utest_pvals, num_connections = [], [], [], [], []
    for recip_otu in high_ab_otu:
        alone_areas = colony_metadata.query(
            "otu == @recip_otu and num_of_adjacency == 0"
        )["area"]
        for donor_otu in high_ab_otu:
            if recip_otu == donor_otu:
                continue
            areas = []
            nc = 0
            for dist_mtx, otus, areas_ in zip(dist_mtxs, otu_list, area_list):
                recip_ix = np.where(otus == recip_otu)[0]
                donor_ix = np.where(otus == donor_otu)[0]
                pairing = np.where(dist_mtx[recip_ix][:, donor_ix])
                nc += len(pairing[0])
                areas.append(areas_[recip_ix[np.unique(pairing[0])]])
            areas = np.concatenate(areas)
            if not len(areas) or not len(alone_areas):
                continue
            l2fc = np.log2(np.mean(areas) / np.mean(alone_areas))
            utest_pval = mannwhitneyu(alone_areas, areas).pvalue
            recipient.append(recip_otu)
            donors.append(donor_otu)
            l2fcs.append(l2fc)
            utest_pvals.append(utest_pval)
            num_connections.append(nc)
    utest_qvals = false_discovery_control(utest_pvals)

    import pdb; pdb.set_trace()
    # Filtering interactions
    interaction_df = interaction_df.query("abs(area_log2FC) > @min_lfc")

    # Creating otu order
    otu_order = pd.unique(interaction_df[["otu_2", "otu_1"]].to_numpy().ravel("K"))

    # Node annotation
    otu_taxonomy = pd.read_table(otu_taxonomy_path, index_col="otu")
    node_annotation = pd.DataFrame(
        {
            "otu": otu_order,
            "otu.label": otu_order + ", " + otu_taxonomy.loc[otu_order, "genus"],
            "genus": otu_taxonomy.loc[otu_order, "genus"],
            "family": otu_taxonomy.loc[otu_order, "family"],
            "order": otu_taxonomy.loc[otu_order, "order"],
            "class": otu_taxonomy.loc[otu_order, "class"],
            "phylum": otu_taxonomy.loc[otu_order, "phylum"],
            "isolate_count": isolate_metadata_zotu_stat.loc[
                otu_order.astype(str), "count"
            ],
        }
    ).sort_values(by=["phylum", "class", "order", "family", "genus"])

    # Edgelist creation
    edgelist = interaction_df[["otu_2", "otu_1", "area_log2FC", "fdr_up", "fdr_down"]]
    edgelist.columns = ["from", "to", "area_log2FC", "fdr_up", "fdr_down"]

    # Applying a minimum threshold for FDR values
    edgelist["fdr_up"] = edgelist["fdr_up"].clip(0.001)
    edgelist["fdr_down"] = edgelist["fdr_down"].clip(0.001)
