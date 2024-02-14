#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for i in {2..30}
do
    config_file="configs/Template-LBBDM-f4_tless_${i}.yaml"
    python3 main.py --config "$config_file" --train --sample_at_start --save_top --gpu_ids 0

    # Create a new YAML file with the corresponding integer
    config_file="${SCRIPT_DIR}/Template-LBBDM-f4_tless_${i}.yaml"
    
    # Replace the pilaceholder in the template with the current integer
    sed "s/PLACEHOLDER/${i}/g" "${SCRIPT_DIR}/Template-LBBDM-f4_tless.yaml" > "${config_file}"
    
    # Optionally, you can customize the file further if needed
    
    echo "Generated ${config_file}"
done