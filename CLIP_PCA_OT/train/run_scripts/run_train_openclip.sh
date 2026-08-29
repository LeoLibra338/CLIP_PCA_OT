export CUDA_VISIBLE_DEVICES=$1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOB_NAME=$2
master_port=$3
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

echo use_gpu: ${CUDA_VISIBLE_DEVICES}
echo gpu_num=${gpu_num}

WORKDIR=/home/gyf/CLIP-wgd/HiMo-CLIP
cd ${WORKDIR}
export PYTHONPATH=$PWD:$PYTHONPATH:/home/gyf/CLIP-wgd/open_clip/src
cd train/

LOG_DIR=output/logs
mkdir -p ${LOG_DIR}
START_TIME=`date "+%Y%m%d_%H%M%S"`
LOG_FILE=${LOG_DIR}/train-log-$JOB_NAME-$START_TIME
torchrun -m \
    --nnodes=1 \
    --nproc_per_node=${gpu_num} \
    --master_port=${master_port} \
    train_openclip \
    --base_model '../weights/ViT-L-14.pt' \
    --jobname ${JOB_NAME}-${START_TIME} \
    --batch-size 32 \
    --accum_steps 4 \
    --epochs 10 \
    2>&1 | tee $LOG_FILE > /dev/null &
