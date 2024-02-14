#!/bin/bash

# List of integers
PLACEHOLDERS=(1 5 6 8 9 10 11 12)

# Loop through each integer
for placeholder in "${PLACEHOLDERS[@]}"
do
  # Source and destination paths
  source_path="results/lmo_combined_${placeholder}/LBBDM-f4/sample_to_eval/20"
  destination_path="results/lmo_combined/${placeholder}/xyz_images"

  # Create the destination directory if it doesn't exist
  mkdir -p "${destination_path}"

  # Move the contents of the source directory to the destination directory
  mv "${source_path}"/* "${destination_path}/"

  # Optionally, you can remove the empty source directory if needed
  # rm -r "${source_path}"

  echo "Moved files from ${source_path} to ${destination_path}"
done
