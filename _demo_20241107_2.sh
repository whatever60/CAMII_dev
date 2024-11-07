# Part 2 of the demo.
script_dir=$PWD


$script_dir/isolate_interaction.py \
    -c $data_dir/interaction/count.biom \
    -im $data_dir/metadata/isolate.tsv \
    -cm $data_dir/colony_detection_rgb/max \
    -o $working_dir/interaction \
    --min_purity 0.5 \
    --min_count 10

$script_dir/draw_interaction.py \
    -i $data_dir/interaction/colony_interaction_summary.csv \
    -m $data_dir/interaction/colony_metadata.csv \
    -tb $data_dir/output_unoise3_16s/unoise3_zotu_isolate_rrndb_processed.tsv \
    -tf $data_dir/output_unoise3_its/unoise3_zotu_isolate_rrndb_processed.tsv \
    -pb $data_dir/output_unoise3_16s/unoise3_zotu_isolate.newick \
    -pf $data_dir/output_unoise3_its/unoise3_zotu_isolate.newick \
    -o $data_dir/figs/interaction/circos.png
