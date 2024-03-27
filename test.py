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
import matplotlib.pyplot as plt
import seaborn as sns

from isolate_interaction import _read_colony_metadata, _align_isolate_colony
# from align import find_affine, get_query2target_func, find_mutual_pairs
from align import Aligner, remove_bad_nodes, network_to_map

data_dir = "/mnt/c/aws_data/20240224_logan_tyndall_boneyard_interaction/interaction"
isolate_count_path = f"{data_dir}/count.tsv"
isolate_metadata_path = f"{data_dir}/isolate_metadata.tsv"
colony_metadata_dir = "/mnt/c/Users/quym/Dropbox/Hyperspectral_imaging/data_collection/colony_detection/202401_darpa_arcadia_arl_boneyard_b2"

isolate_metadata = pd.read_table(isolate_metadata_path, index_col="sample")
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

isolate_count = pd.read_table(isolate_count_path, index_col=0)
poi = plates_in_colony
poi = ["ABYPSH3", "ABYRSG1", "ABYPSI3"]
_, _, aligers = _align_isolate_colony(colony_metadata, isolate_metadata, poi, log=True)