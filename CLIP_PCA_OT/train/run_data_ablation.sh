#!/bin/bash
# 数据量消融：1K/5K/10K/25K/50K  ×  pure_clip(ratio=1.0) & pca_pure(ratio=0.9)
cd /home/gyf/CLIP-wgd/HiMo-CLIP/train

for n in 1000 5000 10000 25000 50000; do
    for mode in pure_clip pca_ot; do
        if [ "$mode" = "pure_clip" ]; then
            RATIO=1.0
            OT_LAMBDA=0.0
            LABEL="pureclip"
        else
            RATIO=0.9
            OT_LAMBDA=0.15
            LABEL="pcaot"
        fi
        JOB="abl_n${n}_${LABEL}-$(date +%Y%m%d_%H%M%S)"
        PORT=$((2800 + RANDOM % 100))
        echo "=== n=${n} ${mode} $(date) ==="
        PYTHONPATH=/home/gyf/CLIP-wgd/HiMo-CLIP:$PYTHONPATH \
        MAX_TRAIN_SAMPLES=${n} \
        CUDA_VISIBLE_DEVICES=2,3 torchrun \
            --nnodes=1 --nproc_per_node=2 --master_port=${PORT} \
            -m train_ablation \
            --base_model '../weights/ViT-L-14.pt' \
            --jobname $JOB --batch-size 32 --accum_steps 4 --epochs 10 \
            --pca_ratio ${RATIO} \
            --ot_lambda ${OT_LAMBDA} \
            --queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 \
            > output/logs/train-log-${JOB} 2>&1
        echo "Done n=${n} ${mode} $(date)"
    done
done
echo "ALL DATA ABLATION COMPLETE"
