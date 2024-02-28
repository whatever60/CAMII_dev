#!/usr/bin/env python
"""Plot interaction plot for CAMII isolates.
"""

import argparse
import tempfile
import json
import subprocess
import sys
import os
import shutil
from typing import Callable

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio import Phylo
import dendropy
from transformers import AutoTokenizer
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from pycirclize import Circos, config, utils
from pycirclize.sector import Sector


matplotlib.use("TkAgg")
plt.rcParams["pdf.fonttype"] = 42


ARROWSTYLE = "Simple, tail_width={}, head_width={}, head_length={}"


def read_subtree(
    newick_path: str, nodes_of_interest: list[str] = None
) -> Phylo.BaseTree.Tree:
    if nodes_of_interest is None:
        return Phylo.read(newick_path, "newick")
    else:
        tree = dendropy.Tree.get(path=newick_path, schema="newick")
        tree.retain_taxa_with_labels(nodes_of_interest)
        # save to newick and read with Biopython to get a rooted tree
        with tempfile.NamedTemporaryFile() as f:
            tree.write(path=f.name, schema="newick")
            return Phylo.read(f.name, "newick")


def simplify_name(arr: list[str], max_length: int) -> list[str]:
    """Split the strings into subwords using cased BERT tokenizer from HuggingFace.
    Concat them back until the length reaches max.
    Note that BERT tokenizer uses ## to denote subwords, we remove them.
    If the word is truncated, add a "." at the end.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    new_arr = []
    for name in arr:
        subwords = tokenizer.tokenize(name)
        new_name = ""
        for subword in subwords:
            subword = subword.lstrip("##")
            if len(new_name) + len(subword) > max_length:
                new_name += "."
                break
            new_name += subword
        new_arr.append(new_name)
    return new_arr


def plot_gradient_ring(
    ax,
    start_angle,
    end_angle,
    inner_radius,
    outer_radius,
    start_color,
    end_color,
    alpha,
):
    """
    Plots a partial ring with a color gradient on the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on.
    start_angle (float): The starting angle of the ring in degrees.
    end_angle (float): The ending angle of the ring in degrees.
    inner_radius (float): The inner radius of the ring.
    outer_radius (float): The outer radius of the ring.
    start_color (str): The color at the start of the gradient.
    end_color (str): The color at the end of the gradient.
    """
    if not -360 < start_angle < 360 or not -360 < end_angle < 360:
        raise ValueError("Start and end angles must be between -360 and 360 degrees.")
    if not start_angle < end_angle:
        raise ValueError(
            f"Start angle ({start_angle}) must be less than end angle ({end_angle})."
        )

    # Number of slices to simulate the gradient
    num_slices = 100

    # Define the custom colormap for the gradient
    colors = [(0, start_color), (0.5, "white"), (1, end_color)]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors)

    # Create the gradient colors for the slices
    slice_colors = cmap(np.linspace(0, 1, num_slices))

    # Plot each slice within the specified angle range
    for i in range(num_slices):
        angle_range = end_angle - start_angle
        slice_angle = angle_range * i / num_slices + start_angle
        ax.add_patch(
            patches.Wedge(
                center=(0.5, 0.5),
                r=outer_radius,
                theta1=slice_angle,
                theta2=slice_angle + (angle_range / num_slices),
                width=outer_radius - inner_radius,
                color=slice_colors[i],
                alpha=alpha,
            )
        )


def sector_x_to_ax_deg(sector, x: float) -> float:
    """Transform x from polar system of a sector in radian to Circos ax coordinate
    system in degree.
    Radian 0 is 90 and move in clockwise direction, we need to transform it so that
    radian 0 is 0 and degree increases in counter-clockwise direction.
    """
    return (90 - np.rad2deg(sector.x_to_rad(x))) % 360


def sector_r_to_ax_r(r: float) -> float:
    """Transform r from polar system of a sector to Euclidean distance to Circos ax origin."""
    return (r / (config.MAX_R + config.R_PLOT_MARGIN) + 1) / 2 - 0.5


def get_fill_between_sectors_func(
    sectors: list[Sector], fcs: list, r1: float, r2: float, alpha: float
) -> Callable:
    """Return a function that fills the space between sectors with color gradient,
    defind by facecolors of the two adjacent sectors and white in the middle.
    """
    if not len(sectors) == len(fcs):
        raise ValueError(f"{len(sectors)=} and {len(fcs)=} must be equal.")
    if len(sectors) == 1:
        raise ValueError("Only one sector is given, cannot fill between sectors.")

    def func(ax: PolarAxes) -> None:
        bounds = [0, 0, 1, 1]
        axin = ax.inset_axes(bounds)
        axin.axis("off")
        sector1s = [sectors[-1]] + sectors[:-1]
        sector2s = sectors
        fc1s = [fcs[-1]] + fcs[:-1]
        fc2s = fcs
        fc1s, fc2s = fc2s, fc1s
        for sector1, sector2, fc1, fc2 in zip(sector1s, sector2s, fc1s, fc2s):
            start = sector_x_to_ax_deg(sector2, 0)
            end = sector_x_to_ax_deg(sector1, sector1.size)
            if start > end:
                end += 360
            ri, ro = sector_r_to_ax_r(r1), sector_r_to_ax_r(r2)
            plot_gradient_ring(axin, start, end, ri, ro, fc1, fc2, alpha)

        for sector, fc in zip(sectors, fcs):
            end = sector_x_to_ax_deg(sector, 0)
            start = sector_x_to_ax_deg(sector, sector.size)
            if start > end:
                end += 360
            ri, ro = sector_r_to_ax_r(r1), sector_r_to_ax_r(r2)
            axin.add_patch(
                patches.Wedge(
                    center=(0.5, 0.5),
                    r=ro,
                    theta1=start,
                    theta2=end,
                    width=ro - ri,
                    color=fc,
                    alpha=alpha,
                )
            )

    return func


# def _add_figure_properties(
#     interaction_df: pd.DataFrame,
#     sector2xs: dict[str, dict[str, float]],
#     # l2fc2width: Callable,
#     # mlog10qval2alpha: Callable,
#     # arr_color_pos,
#     # arr_color_neg,
# ) -> pd.DataFrame:
#     arr_data = interaction_df.copy()
#     arr_data["l2fc"] = arr_data["fc"].map(np.log2)
#     arr_data["mlog10qval"] = -arr_data["qval"].map(np.log10)
#     # arr_data["arr_width"] = arr_data["l2fc"].map(l2fc2width)
#     # arr_data["arr_alpha"] = arr_data["mlog10qval"].map(mlog10qval2alpha)
#     # arr_data["color"] = arr_data["l2fc"].map(
#     #     lambda x: arr_color_pos if x > 0 else arr_color_neg
#     # )
#     arr_data["sector_d"] = arr_data["donor"].map(
#         lambda x: prefix2sector[x.split("-")[0]]
#     )
#     arr_data["sector_r"] = arr_data["receptor"].map(
#         lambda x: prefix2sector[x.split("-")[0]]
#     )
#     arr_data["arr_start_c"] = arr_data.apply(
#         lambda x: sector2xs[x["sector_d"]][x["donor"]], axis=1
#     )
#     arr_data["arr_end_c"] = arr_data.apply(
#         lambda x: sector2xs[x["sector_r"]][x["receptor"]], axis=1
#     )
#     return arr_data


def get_arrow(
    posA: tuple[float, float],
    posB: tuple[float, float],
    width: float,
    head_length: float,
    color: str,
    alpha: float,
    connectionstyle: str = "arc3",
    **kwargs,
) -> FancyArrowPatch:
    tail_width = width
    head_width = tail_width * 3
    return FancyArrowPatch(
        posA=posA,
        posB=posB,
        arrowstyle=ARROWSTYLE.format(tail_width, head_width, head_length),
        fc=mcolors.to_rgb(color) + (alpha,),
        ec=mcolors.to_rgba(color),
        lw=1,
        connectionstyle=connectionstyle,
        **kwargs,
    )


def get_arrow_func(
    sname2sector: dict[str, Sector],
    sname2xs: dict[str, dict[str, float]],
    interaction_arr_r: float,
    interaction_df: pd.DataFrame,
    l2fc2width: Callable,
    mlog10qval2alpha: Callable,
    arr_color_pos,
    arr_color_neg,
) -> Callable:
    prefix2sname = {"ITS": "Fungi", "16S": "Bacteria"}
    interaction_arr_r = sector_r_to_ax_r(interaction_arr_r)

    def func(ax: PolarAxes) -> None:
        source_target_set = set()
        bounds = [0, 0, 1, 1]
        axin = ax.inset_axes(bounds)
        axin.axis("off")
        for _, row in interaction_df.iterrows():
            receptor = row["receptor"]
            donor = row["donor"]

            arr_width = l2fc2width(np.log2(row["fc"]))
            arr_alpha = mlog10qval2alpha(-np.log10(row["qval"]))
            arr_color = arr_color_pos if row["fc"] > 1 else arr_color_neg
            sector_r_sname = prefix2sname[receptor.split("-")[0]]
            sector_d_sname = prefix2sname[donor.split("-")[0]]
            arr_end_c = sname2xs[sector_r_sname][receptor]
            arr_start_c = sname2xs[sector_d_sname][donor]
            end = sector_x_to_ax_deg(sname2sector[sector_r_sname], arr_end_c)
            start = sector_x_to_ax_deg(sname2sector[sector_d_sname], arr_start_c)

            source_target_set.add((receptor, donor))
            rad_sign = ""
            if (donor, receptor) not in source_target_set:
                if (end - start) % 360 < 180:
                    rad_sign = "-"
            else:
                if (end - start) % 360 > 180:
                    rad_sign = "-"
            start_xy = (
                np.cos(np.deg2rad(start)) * interaction_arr_r + 0.5,
                np.sin(np.deg2rad(start)) * interaction_arr_r + 0.5,
            )
            end_xy = (
                np.cos(np.deg2rad(end)) * interaction_arr_r + 0.5,
                np.sin(np.deg2rad(end)) * interaction_arr_r + 0.5,
            )
            fancy_arrow_patch = get_arrow(
                posA=start_xy,
                posB=end_xy,
                width=arr_width,
                head_length=(180 - abs((end - start) % 360 - 180)) / 8 + 10,
                color=arr_color,
                alpha=arr_alpha,
                connectionstyle=f"arc3,rad={rad_sign}0.2",
                capstyle="round",
                joinstyle="round",
            )
            axin.add_patch(fancy_arrow_patch)

    return func


def get_legend_func(
    l2fcs: list[float],
    mlog10qvals: list[float],
    arr_color_pos,
    arr_color_neg,
    l2fc2width: Callable,
    mlog10qval2alpha: Callable,
) -> Callable:
    def legend_func(ax: PolarAxes) -> None:
        # put legend on the top right corner
        default_head_length = 20
        default_tail_width = l2fc2width(np.median(l2fcs))
        default_alpha = mlog10qval2alpha(np.median(mlog10qvals))
        legend_block_title_fontsize = 16
        bounds = [1.05, 0.0, 0.4, 1]
        delta = 0.05
        delta_between_blocks = 0.01
        y = 1
        title_start_x, arr_start_x, arr_end_x = 0.1, 0.2, 0.5

        axin = ax.inset_axes(bounds)
        axin.axis("off")

        axin.text(
            title_start_x,
            y,
            "-log10(q-value)",
            ha="left",
            va="center",
            fontsize=legend_block_title_fontsize,
        )
        y -= delta
        for mlog10_qval in mlog10qvals:
            arrow_alpha = mlog10qval2alpha(mlog10_qval)
            fancy_arrow_patch = get_arrow(
                posA=(arr_start_x, y),
                posB=(arr_end_x, y),
                width=default_tail_width,
                head_length=default_head_length,
                color="black",
                alpha=arrow_alpha,
            )
            axin.add_patch(fancy_arrow_patch)
            axin.text(
                arr_end_x + 0.1,
                y,
                f"{mlog10_qval}",
                ha="left",
                va="center",
                fontsize=16,
            )
            y -= delta
        y -= delta_between_blocks

        axin.text(
            title_start_x,
            y,
            "log2(fold change)",
            ha="left",
            va="center",
            fontsize=legend_block_title_fontsize,
        )
        y -= delta
        for l2fc in l2fcs:
            arrow_width = l2fc2width(l2fc)
            fancy_arrow_patch = get_arrow(
                posA=(arr_start_x, y),
                posB=(arr_end_x, y),
                width=arrow_width,
                head_length=default_head_length,
                color="black",
                alpha=default_alpha,
            )
            axin.add_patch(fancy_arrow_patch)
            axin.text(
                arr_end_x + 0.1,
                y,
                f"{l2fc}",
                ha="left",
                va="center",
                fontsize=16,
            )
            y -= delta
        y -= delta_between_blocks

        axin.text(
            title_start_x,
            y,
            "Effect",
            ha="left",
            va="center",
            fontsize=legend_block_title_fontsize,
        )
        y -= delta
        for color, direction in zip(
            [arr_color_pos, arr_color_neg], ["Positive", "Negative"]
        ):
            fancy_arrow_patch = get_arrow(
                posA=(arr_start_x, y),
                posB=(arr_end_x, y),
                width=default_tail_width,
                head_length=default_head_length,
                color=color,
                alpha=default_alpha,
            )
            axin.add_patch(fancy_arrow_patch)
            axin.text(
                arr_end_x + 0.1,
                y,
                f"{direction}",
                ha="left",
                va="center",
                fontsize=legend_block_title_fontsize,
            )
            y -= delta

    return legend_func


def add_title(sector: Sector, fontsize: int) -> None:
    rad = sector.x_to_rad(sector.center)
    params = utils.plot.get_label_params_by_rad(rad, orientation="horizontal")
    if rad > np.pi:
        params["rotation"] += 180
        params["va"] = "bottom"
    sector.text(
        f"{sector.name} ({sector.size})",
        size=fontsize,
        adjust_rotation=False,
        **params,
    )


def add_tree(sector: Sector, r_lim: tuple[float, float], tree: Phylo.BaseTree) -> None:
    tree_track = sector.add_track(r_lim)
    tree_track.axis(fc=None, alpha=0, lw=0)
    # tree_track.tree(tree, leaf_label_size=0)
    tree_track.tree(
        tree,
        outer=False,
        align_leaf_label=True,
        ignore_branch_length=True,
        line_kws=dict(lw=1),
        # leaf_label_rmargin=32,
        leaf_label_size=0,
    )


def add_bar(
    sector: Sector,
    r_lim: tuple[float, float],
    r_pad_ratio: float,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: np.ndarray,
    x_tick_labels: np.ndarray,
    ymin: float,
    ymax: float,
    bar_width: float,
    x_tick_label_fontsize: int,
    x_tick_label_margin: float,
) -> None:
    bar_track = sector.add_track(r_lim, r_pad_ratio=r_pad_ratio)
    # value here doesn't matter since there's no ytick at all.
    bar_track.yticks([], vmin=0, vmax=1)
    bar_track.bar(
        x=xs,
        height=ys - ymin,
        color=colors,
        width=bar_width,
        vmin=0,
        vmax=(ymax - ymin),
    )
    bar_track.xticks(
        xs,
        x_tick_labels,
        tick_length=0,
        label_size=x_tick_label_fontsize,
        label_margin=x_tick_label_margin,
        label_orientation="vertical",
        text_kws=dict(ma="center"),
    )


def add_bar_grid(
    circos,
    r_lim_bar: tuple[float, float],
    bar_grid_ys: list[float],
    bar_grid_rs: list[float],
    grid_alpha: float,
    tick_alpha: float,
    tick_delta: float,
    tick_font_size: int,
) -> None:
    # add grid to bar plot
    circos.line(r=r_lim_bar[0], ls="dashed", color="gray", alpha=tick_alpha)
    circos.line(r=r_lim_bar[1], ls="dashed", color="gray", alpha=tick_alpha)
    for y, r in zip(bar_grid_ys, bar_grid_rs):
        circos.text(10**y, deg=0, r=r + tick_delta, size=tick_font_size)
        circos.line(r=r, ls="dotted", color="gray", alpha=grid_alpha)


def add_bar_legend(
    circos,
    colormap: dict,
    marker_size: int,
    position: tuple[float, float],
    fontsize: int,
    title_fontsize: int,
) -> None:
    bar_handles = [
        Line2D([], [], color=color, marker="s", label=family, ms=marker_size, ls="None")
        for family, color in colormap.items()
    ]
    bar_legend = circos.ax.legend(
        handles=bar_handles,
        bbox_to_anchor=position,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        title="Family-level taxonomy",
        handlelength=2,
        alignment="left",
    )
    circos.ax.add_artist(bar_legend)


def normalize_to(
    in_min: float, in_max: float, out_min: float, out_max: float
) -> Callable:
    def func(v: float) -> float:
        return (np.clip(v, in_min, in_max) - in_min) / (in_max - in_min) * (
            out_max - out_min
        ) + out_min

    return func


def read_isolate_interaction(
    interaction_csv: str,
    taxon_16s_tsv: str,
    taxon_its_tsv: str,
    tree_16s_tsv: str,
    tree_its_tsv: str,
    label_by: str,
    color_by: str,
    cmap: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Phylo.BaseTree.Tree, Phylo.BaseTree.Tree]:
    interaction_df = pd.read_csv(interaction_csv)
    node_df = interaction_df.loc[interaction_df["donor"].isna()].copy().dropna(axis=1)
    node_df.columns = ["otu", "count"]
    node_df = node_df.set_index("otu")
    interaction_df = interaction_df.loc[~interaction_df["donor"].isna()].copy()
    taxon_df = pd.concat(
        [
            pd.read_table(taxon_16s_tsv, index_col=0),
            pd.read_table(taxon_its_tsv, index_col=0),
        ]
    )
    node_df = node_df.join(taxon_df, how="left").reset_index()

    zotus_16s, tax_16s = (
        node_df.query("otu.str.startswith('16S-')")[["otu", color_by]]
        .transpose()
        .to_numpy()
    )
    zotus_its, tax_its = (
        node_df.query("otu.str.startswith('ITS-')")[["otu", color_by]]
        .transpose()
        .to_numpy()
    )
    node_df = node_df.set_index("otu")
    node_df["label"] = (
        node_df.index.map(lambda x: x.split("-")[1])
        + "\n"
        + simplify_name(node_df[label_by].tolist(), max_length=8)
    )

    tree_16s = read_subtree(tree_16s_tsv, zotus_16s)
    tree_its = read_subtree(tree_its_tsv, zotus_its)

    cmap = matplotlib.colormaps.get_cmap(cmap).colors
    try:
        taxon_color_16s = {
            tax: cmap[i * 2] for i, tax in enumerate(np.sort(np.unique(tax_16s)))
        }
        taxon_color_its = {
            tax: cmap[i * 2 + 1] for i, tax in enumerate(np.sort(np.unique(tax_its)))
        }
    except IndexError:
        raise ValueError(
            f"Too many unique {color_by} values. Current color map has {len(cmap)} "
            f"colors, while there are {len(np.unique(tax_16s))} unique 16S taxa and "
            f"{len(np.unique(tax_its))} unique ITS taxa. Choose a colormap with at "
            f"least {max(len(np.unique(tax_16s)), len(np.unique(tax_its))) * 2} colors."
        )
    taxon_color = {**taxon_color_16s, **taxon_color_its}
    node_df["color"] = node_df[color_by].map(taxon_color)

    return (node_df, interaction_df, tree_16s, tree_its)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interaction", required=True, help="Interaction csv")
    parser.add_argument(
        "-tb", "--taxon_16s", required=True, help="ZOTU taxonomy tsv for 16S"
    )
    parser.add_argument(
        "-tf", "--taxon_its", required=True, help="ZOTU taxonomy tsv for ITS"
    )
    parser.add_argument(
        "-pb", "--tree_16s", required=True, help="Phylogenetic tree for 16S"
    )
    parser.add_argument(
        "-pf", "--tree_its", required=True, help="Phylogenetic tree for ITS"
    )
    parser.add_argument(
        "-c",
        "--color_by",
        type=str,
        default="family",
        help="Which taxonomy level to color by",
    )
    parser.add_argument(
        "-l",
        "--label_by",
        type=str,
        default="genus",
        help="Which taxonomy level to label by",
    )
    parser.add_argument("-o", "--output_fig", required=True, help="Output directory")

    # ===== clipping for better visualization =====
    max_l2fc, min_l2fc = 4, np.log2(1.2)
    max_mlog10_qval, min_mlog10_qval = 3, 1
    arr_legend_l2fcs = 1, 2, 3
    arr_legend_mlog10qvals = 1, 2, 3
    # ===== figure appearance =====
    figsize = (20, 10)
    sector_names = ["Bacteria", "Fungi"]
    # seq_name_prefix = ["16S-", "ITS-"]
    colors = ["limegreen", "orange"]
    sector_space = 10
    sector_title_fontsize = 16
    r_lim_tree = 85, 100
    r_lim_bar = 45, 65
    r_max_arr = r_lim_bar[0] - 3
    tree_fc_alpha = 0.2
    bar_r_pad = 0
    bar_width = 0.4
    bar_margin_line_alpha = 0.9
    bar_grid_line_alpha = 0.7
    # y axis range of the bar plot
    # y axis of these values will have a dotted line
    bar_ymax = np.log10(999)
    bar_grid_ys = np.arange(np.ceil(bar_ymax)).astype(int)
    bar_ymin = min(bar_grid_ys) - 0.1 * (bar_ymax - min(bar_grid_ys))
    # bar_ymax = max(bar_grid_ys) + 0.1 * (max(bar_grid_ys) - min(bar_grid_ys))
    # these values correspond to these radius values
    bar_grid_rs = [
        r_lim_bar[0]
        + (r_lim_bar[1] - r_lim_bar[0]) * (bar_y - bar_ymin) / (bar_ymax - bar_ymin)
        for bar_y in bar_grid_ys
    ]
    bar_x_tick_label_fontsize = 9
    bar_x_tick_label_margin = 2
    bar_y_tick_label_delta = 2
    bar_y_tick_label_fontsize = 8
    bar_color_map = "tab20"
    arr_alpha_max, arr_alpha_min = 1, 0.2
    arr_width_max, arr_width_min = 10, 2.5
    arr_pos_color, arr_neg_color = "brown", "steelblue"
    # ===== legend appearance =====
    legend_title_fontsize = 16
    legend_fontsize = 12
    bar_legend_position = 1.08, 0.45
    bar_legend_marker_size = 8

    l2fc2width = normalize_to(min_l2fc, max_l2fc, arr_width_min, arr_width_max)
    mlog10qval2alpha = normalize_to(
        min_mlog10_qval, max_mlog10_qval, arr_alpha_min, arr_alpha_max
    )

    # ===== data io =====
    args = parser.parse_args()
    interaction_path = args.interaction
    taxon_16s_path = args.taxon_16s
    taxon_its_path = args.taxon_its
    tree_16s_path = args.tree_16s
    tree_its_path = args.tree_its
    color_by = args.color_by
    label_by = args.label_by
    out_fig = args.output_fig

    # ===== read data =====
    node_df, interaction_df, tree_16s, tree_its = read_isolate_interaction(
        interaction_path,
        taxon_16s_path,
        taxon_its_path,
        tree_16s_path,
        tree_its_path,
        color_by=color_by,
        label_by=label_by,
        cmap=bar_color_map,
    )

    # ===== main function =====
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    axs = fig.subplots(1, 2, subplot_kw=dict(polar=True))

    interaction_df_pos = interaction_df.query("fc > 1")
    interaction_df_neg = interaction_df.query("fc < 1")
    for idx, (interaction_df, ax) in enumerate(
        zip([interaction_df_pos, interaction_df_neg], axs)
    ):
        sectors = {
            name: tree.count_terminals()
            for name, tree in zip(sector_names, [tree_16s, tree_its])
        }
        circos = Circos(
            sectors,
            space=sector_space,
            start=sector_space / 2 - 360,
            end=sector_space / 2,
        )

        sname2xs = (
            {}
        )  # map sector name to a dict that maps sequence name to x coordinate
        for i, (sector_name, tree, color) in enumerate(
            zip(sector_names, [tree_16s, tree_its], colors)
        ):
            sector = circos.get_sector(sector_name)
            leaf_names = [i.name for i in tree.get_terminals()]

            # Add sector title
            # add_title(sector, fontsize=sector_title_fontsize)
            add_tree(sector, r_lim=r_lim_tree, tree=tree)

            # Add bar plot
            xs = np.arange(0, sector.size) + 0.5
            # ys = np.log10(node_df.loc[leaf_names]["count"]).to_numpy()
            # colors = node_df.loc[leaf_names]["color"].to_numpy()
            # x_tick_labels = node_df.loc[leaf_names]["label"].to_numpy()
            add_bar(
                sector=sector,
                r_lim=r_lim_bar,
                r_pad_ratio=bar_r_pad,
                xs=xs,
                ys=np.log10(node_df.loc[leaf_names]["count"]).to_numpy(),
                colors=node_df.loc[leaf_names]["color"].to_numpy(),
                x_tick_labels=node_df.loc[leaf_names]["label"].to_numpy(),
                ymin=bar_ymin,
                ymax=bar_ymax,
                bar_width=bar_width,
                x_tick_label_fontsize=bar_x_tick_label_fontsize,
                x_tick_label_margin=bar_x_tick_label_margin,
            )
            sname2xs[sector.name] = {i: x for i, x in zip(leaf_names, xs)}
            # bar_track = sector.add_track(r_lim_bar, r_pad_ratio=bar_r_pad)
            # bar_track.yticks([], [], vmin=bar_ymin, vmax=bar_ymax)
            # xs = np.arange(0, sector.size) + 0.5
            # sector2xs[sector.name] = {i: x for i, x in zip(leaf_names, xs)}
            # bar_track.bar(
            #     x=xs,
            #     height=np.log10(node_df.loc[leaf_names]["count"]).to_numpy(),
            #     width=bar_width,
            #     color=node_df.loc[leaf_names]["color"].to_numpy(),
            # )
            # bar_track.xticks(
            #     np.arange(0, sector.size) + 0.5,
            #     node_df.loc[leaf_names]["label"].to_numpy(),
            #     tick_length=0,
            #     label_size=bar_x_tick_label_fontsize,
            #     label_margin=bar_x_tick_label_margin,
            #     label_orientation="vertical",
            #     text_kws=dict(ma="center"),
            #     # outer=False,
            # )

        # add grid to bar plot
        add_bar_grid(
            circos,
            r_lim_bar=r_lim_bar,
            bar_grid_ys=bar_grid_ys,
            bar_grid_rs=bar_grid_rs,
            grid_alpha=bar_grid_line_alpha,
            tick_alpha=bar_margin_line_alpha,
            tick_delta=bar_y_tick_label_delta,
            tick_font_size=bar_y_tick_label_fontsize,
        )

        sector_fill_func = get_fill_between_sectors_func(
            circos.sectors,
            fcs=colors,
            r1=r_lim_tree[0] - 1,
            r2=r_lim_tree[1] + 2,
            alpha=tree_fc_alpha,
        )

        # interaction_df = _add_figure_properties(
        #     interaction_df,
        #     # l2fc2width=l2fc2width,
        #     # mlog10qval2alpha=mlog10qval2alpha,
        #     # arr_color_pos=arr_pos_color,
        #     # arr_color_neg=arr_neg_color,
        # )
        interaction_arr_func = get_arrow_func(
            interaction_arr_r=r_max_arr,
            sname2sector={s.name: s for s in circos.sectors},
            sname2xs=sname2xs,
            interaction_df=interaction_df,
            l2fc2width=l2fc2width,
            mlog10qval2alpha=mlog10qval2alpha,
            arr_color_pos=arr_pos_color,
            arr_color_neg=arr_neg_color,
        )
        plot_funcs = [sector_fill_func, interaction_arr_func]
        if idx == 1:
            legend_func = get_legend_func(
                l2fcs=arr_legend_l2fcs,
                mlog10qvals=arr_legend_mlog10qvals,
                arr_color_pos=arr_pos_color,
                arr_color_neg=arr_neg_color,
                l2fc2width=l2fc2width,
                mlog10qval2alpha=mlog10qval2alpha,
            )
            plot_funcs.append(legend_func)
        circos._plot_funcs.extend(plot_funcs)
        fig = circos.plotfig(ax=ax)
        if idx == 1:
            # add bar legend
            add_bar_legend(
                circos,
                colormap=node_df[["family", "color"]]
                .drop_duplicates()
                .set_index("family")
                .squeeze()
                .to_dict(),
                marker_size=bar_legend_marker_size,
                position=bar_legend_position,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
            )

    # save figure
    out_dir = os.path.dirname(out_fig)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight", dpi=300)
    if not out_fig.endswith(".pdf"):
        fig.savefig(os.path.splitext(out_fig)[0] + ".pdf", bbox_inches="tight", dpi=300)
