#!/bin/bash
# 最终通宵：GPU 0,1 双卡，消融+pca_ot扫参+评测
cd /home/gyf/CLIP-wgd/HiMo-CLIP/train
RLOG=output/logs/final_results.txt
echo "=== START $(date) ===" > $RLOG

train_one() {
    local JOB=$1 R=$2 OT=$3
    local PORT=$((2900 + RANDOM % 50))
    echo "  $(date) $JOB ratio=$R ot=$OT" | tee -a $RLOG
    PYTHONPATH=/home/gyf/CLIP-wgd/HiMo-CLIP:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0,1 \
    torchrun --nnodes=1 --nproc_per_node=2 --master_port=${PORT} \
      -m train \
      --base_model '../weights/ViT-L-14.pt' --jobname $JOB \
      --batch-size 32 --accum_steps 4 --epochs 10 \
      --pca_ratio ${R} --ot_lambda ${OT} \
      --queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 \
      >> output/logs/train-log-${JOB} 2>&1
    echo "  Done $(date)" | tee -a $RLOG
}

# 1. pure_clip baseline
train_one "final_pureclip-$(date +%Y%m%d_%H%M%S)" 1.0 0.0

# 2. pca_ot (核心方法)
train_one "final_pcaot-$(date +%Y%m%d_%H%M%S)" 0.9 0.15

# 3. pca_ot τ 扫参
for tau in 0.7 0.8 0.9 0.95; do
    train_one "final_pcaot_tau${tau}-$(date +%Y%m%d_%H%M%S)" ${tau} 0.15
done

echo "=== ALL DONE $(date) ===" | tee -a $RLOG
