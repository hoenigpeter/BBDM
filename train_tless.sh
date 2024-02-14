#!/bin/bash

for i in {2..30}
do
    config_file="configs/Template-LBBDM-f4_tless_${i}.yaml"
    python3 main.py --config "$config_file" --train --sample_at_start --save_top --gpu_ids 0
done
