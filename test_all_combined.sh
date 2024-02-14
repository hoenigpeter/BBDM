#!/bin/bash

# List of integers
PLACEHOLDERS=(1 5 6 8 9 10 11 12)

# Loop through each integer
for placeholder in "${PLACEHOLDERS[@]}"
do
  config_file="configs/Template-LBBDM-f4_combined_lmo_${placeholder}.yaml"
  resume_model="results/lmo_xyz_combined/LBBDM-f4/checkpoint/last_model.pth"
  
  python3 main.py --config "${config_file}" --sample_to_eval --gpu_ids 0 --resume_model "${resume_model}"
done
