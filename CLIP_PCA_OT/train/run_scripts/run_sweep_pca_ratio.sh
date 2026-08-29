# PCA ratio 扫参：τ=0.7/0.8/0.9/0.95
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_num=2

WORKDIR=/home/gyf/CLIP-wgd/HiMo-CLIP
cd ${WORKDIR}
export PYTHONPATH=$PWD:$PYTHONPATH
cd train/

for ratio in 0.7 0.8 0.9 0.95; do
    JOB_NAME="pca_sweep_tau${ratio}"
    LOG_DIR=output/logs
    mkdir -p ${LOG_DIR}
    START_TIME=`date "+%Y%m%d_%H%M%S"`
    LOG_FILE=${LOG_DIR}/train-log-${JOB_NAME}-${START_TIME}

    echo "=== Running PCA ratio=${ratio} ==="
    torchrun -m \
        --nnodes=1 \
        --nproc_per_node=${gpu_num} \
        --master_port=2530 \
        train \
        --base_model '../weights/ViT-L-14.pt' \
        --jobname ${JOB_NAME}-${START_TIME} \
        --batch-size 32 \
        --accum_steps 4 \
        --epochs 10 \
        --pca_ratio ${ratio} \
        --queue_size 0 \
        --focal_gamma 0.0 \
        --gate_beta 0.0 \
        2>&1 | tee ${LOG_FILE} > /dev/null

    echo "=== Done ratio=${ratio} ==="
done

echo "All sweeps done!"
