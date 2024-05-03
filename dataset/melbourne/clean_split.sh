#!/usr/bin/env bash

set -ex

zip_file="LasFiles_30-04-2024.zip"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <split_percentage>"
    exit 1
fi

data_dir="raw_data"
target_dir="tiles_data"
las_dir="las_data"

mkdir -p $data_dir
cp $zip_file $data_dir
unzip $data_dir/$zip_file -x '*/__MACOSX/*' -d $data_dir
rm $data_dir/$zip_file

mkdir -p $target_dir
mkdir -p $las_dir

# move all files to raw_data/
find $data_dir -maxdepth 4 -type f -name '*' -exec mv {} $target_dir \;
find $target_dir -maxdepth 4 -type f -name '*.las' -exec mv {} $las_dir \;

split_percentage=$1
echo "Splitting data with $split_percentage for training."

total_files=$(find $target_dir -maxdepth 1 -type f | wc -l)
echo "Total files: $total_files"

num_train_files=$(echo "scale=0; $total_files * $split_percentage" | bc)
num_train_files=${num_train_files%.*}
num_test_files=$(echo "scale=0; $total_files - $num_train_files" | bc)
num_test_files=${num_test_files%.*}

# split train and test
train_target="$target_dir/train"
test_target="$target_dir/test"

mkdir -p $train_target
mkdir -p $test_target

find $target_dir -maxdepth 1 -type f | head -n "$num_test_files" | xargs -I{} mv {} "$test_target"
find $target_dir -maxdepth 1 -type f | xargs -I{} mv {} "$train_target"

echo "Data split complete. $num_train_files files moved to training set and $num_test_files files moved to test set."
