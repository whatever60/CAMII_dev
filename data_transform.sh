#!/bin/bash

# Initialize variables
input_dir=""
output_dir=""

# Determine the directory of the script
script_dir=$(dirname "$0")

# Function to display usage
usage() {
    echo "Usage: $0 -i <input_directory> -o <output_directory>"
    exit 1
}

# Parse command-line options
while getopts 'i:o:' flag; do
    case "${flag}" in
        i) input_dir=${OPTARG} ;;
        o) output_dir=${OPTARG} ;;
        *) usage ;;
    esac
done

# Check if input and output directories are provided
if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
    echo "Error: Input and output directories must be specified."
    usage
fi

# Exporting variables and functions to be available in parallel subshell
export input_dir
export output_dir
export script_dir

# Defining the command to be parallelized as a function
process_file() {
    bil_file="$1"
    if [[ -f "$bil_file" ]]; then
        base_name=$(basename "$bil_file" .bil)
        metadata_file="${bil_file}.hdr"
        "$script_dir/data_transform.py" bil2npy "$bil_file" --metadata "$metadata_file" --output_dir "$output_dir"
    fi
}

export -f process_file

# Find all .bil files and pass them to GNU Parallel, specifying 4 cores
find "$input_dir" -name "*.bil" | parallel -j 4 process_file

