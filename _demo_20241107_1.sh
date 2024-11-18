data_dir=/mnt/c/aws_data/data/camii_demo/demo_20241105_output
script_dir=$PWD


for time_point in 1 3 4 5; do
    $script_dir/data_transform.py process_bmp \
        --input_dir $data_dir/rgb_raw/d$time_point \
        --output_dir $data_dir/rgb \
        --time_point $time_point
done

for time_point in 1 3 4 5; do
    $script_dir/detect_colonies.py \
        -i $data_dir/rgb \
        -o $data_dir/colony_detection_rgb/d$time_point \
        -b $script_dir/test_data/parameters/calib_parameter.npz \
        -c $script_dir/test_data/configs/configure_small.yaml \
        -t $time_point
done
for time_point in min max; do
    $script_dir/detect_colonies.py \
        -i $data_dir/rgb \
        -o $data_dir/colony_detection_rgb/$time_point \
        -b $script_dir/test_data/parameters/calib_parameter.npz \
        -c $script_dir/test_data/configs/configure_small.yaml \
        -t $time_point
done


$script_dir/select_colonies.py init \
    -p $data_dir/rgb \
    -i $data_dir/colony_detection_rgb/max \
    -o $data_dir/colony_selection_rgb/max \
    -m $data_dir/metadata/plates.csv \
    -c $script_dir/test_data/configs/configure.yaml

for file in $data_dir/colony_selection_rgb/max/*_annot_init.json; do
    dir=$(dirname "$file")
    base_filename=$(basename "$file")
    newbase="${base_filename/_annot_init.json/_annot_init_post.json}"
    newfile="$dir/$newbase"
    cp "$file" "$newfile"
done

$script_dir/select_colonies.py post \
    -p $data_dir/rgb \
    -i $data_dir/colony_selection_rgb/max \
    -m $data_dir/metadata/plates.csv \
    -s init

$script_dir/select_colonies.py final \
    -p $data_dir/rgb \
    -i $data_dir/colony_selection_rgb/max \
    -m $data_dir/metadata/plates.csv \
    -o $data_dir/colony_picking_rgb/max \
    -t heuristic
    
$script_dir/correct_coords.py \
    -i $data_dir/colony_picking_rgb/max \
    -p $script_dir/test_data/parameters/correction_params.json


# HS
for time_point in 3 4 5; do
    # First convert bil to npz and extract png for visualization
    $script_dir/data_transform.py bil2npz \
        --input_dir $data_dir/hyperspectral_raw/d$time_point \
        --output_dir_npz $data_dir/hyperspectral \
        --output_dir_rgb $data_dir/hyperspectral_rgb \
        --time_point $time_point
done

# Now with the png files, we annotate the plate boundaries on makesense.ai, save in 
# yolo format.

for time_point in 3 4 5; do
    # Take the npz files, crop to plate using the yolo annotations, and do PCA, save the
    # cropping results and PCA results.
    $script_dir/data_transform.py bil2npz \
        --input_dir $data_dir/hyperspectral_raw/d$time_point \
        --output_dir_npz $data_dir/hyperspectral_cropped \
        --output_dir_rgb $data_dir/hyperspectral_rgb_cropped \
        --mask_crop $data_dir/hyperspectral_plate_detection \
        --time_point $time_point \
        --pca \
        --output_dir_pca $data_dir/hyperspectral_pca_cropped \
        -qp 0.005
done


files=$(find "$data_dir/hyperspectral_pca_cropped" -name '*_pc3.png' | sort)
for file in $files; do
    ./detect_colonies.py \
        -i "$file" \
        -o "$data_dir/colony_detection_hsi/max" \
        -b "$script_dir/test_data/parameters/calib_parameter.npz" \
        -c "$script_dir/test_data/configs/configure_small_hsi.yaml"
done
