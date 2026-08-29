#!/bin/bash
cd /home/gyf/CLIP-wgd/HiMo-CLIP/train

for ratio in 0.7 0.8 0.9 0.95; do
    JOB="pca_tau${ratio}-$(date +%Y%m%d_%H%M%S)"
    PORT=$((2780 + RANDOM % 100))
    echo "=== tau=${ratio} $(date) ==="
    PYTHONPATH=/home/gyf/CLIP-wgd/HiMo-CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=2,3 torchrun \
        --nnodes=1 --nproc_per_node=2 --master_port=${PORT} \
        -m train \
        --base_model '../weights/ViT-L-14.pt' \
        --jobname $JOB \
        --batch-size 32 --accum_steps 4 --epochs 10 \
        --pca_ratio ${ratio} \
        --queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 \
        > output/logs/train-log-${JOB} 2>&1
    echo "Done tau=${ratio} $(date)"
done
echo "ALL SWEEPS COMPLETE"
