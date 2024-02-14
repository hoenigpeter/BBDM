#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# List of integers
PLACEHOLDERS=(1 5 6 8 9 10 11 12)

# Loop through each integer
for placeholder in "${PLACEHOLDERS[@]}"
do
  # Create a new YAML file with the corresponding integer
  config_file="${SCRIPT_DIR}/Template-LBBDM-f4_combined_lmo_${placeholder}.yaml"
  
  # Replace the placeholder in the template with the current integer
  sed "s/PLACEHOLDER/${placeholder}/g" "${SCRIPT_DIR}/Template-LBBDM-f4_augmented_lmo.yaml" > "${config_file}"
  
  # Optionally, you can customize the file further if needed
  
  echo "Generated ${config_file}"
done
