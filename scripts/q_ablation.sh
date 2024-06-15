#!/bin/bash

PROCESSES=48
Q_ARR=($(seq 0.1 0.1 0.2))
echo "Running coco experiments for target rates of: ${Q_ARR[@]}"

for q in "${Q_ARR[@]}"; do
    python coco.py \
        --trials 100 \
        --label_ct 400 \
        --target_rate $q \
        --q_min 0.05 \
        --processes $PROCESSES \
        --out_dir results/coco_d/q_ablation/target_rate=$q
done
for q in "${Q_ARR[@]}"; do
    python imagenet.py \
        --trials 100 \
        --label_ct 600 \
        --target_rate $q \
        --q_min 0.05 \
        --processes $PROCESSES \
        --out_dir results/imagenet_d/q_ablation/target_rate=$q
done

