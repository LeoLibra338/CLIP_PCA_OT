#!/bin/bash
# 通宵任务：数据消融 + pca_ot τ扫参
# GPU 1 单卡，全部串行
cd /home/gyf/CLIP-wgd/HiMo-CLIP/train
RLOG="output/logs/overnight_results.txt"
echo "=== START $(date) ===" > $RLOG

run_train() {
    local JOB=$1 RATIO=$2 OT=$3 MAXN=$4
    local PORT=$((2870 + RANDOM % 30))
    echo "  $(date) $JOB max_samples=$MAXN ratio=$RATIO ot=$OT" | tee -a $RLOG
    PYTHONPATH=/home/gyf/CLIP-wgd/HiMo-CLIP:$PYTHONPATH \
    MAX_TRAIN_SAMPLES=${MAXN} \
    CUDA_VISIBLE_DEVICES=1 \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=${PORT} \
      -m train_ablation \
      --base_model '../weights/ViT-L-14.pt' --jobname $JOB \
      --batch-size 16 --accum_steps 2 --epochs 10 \
      --pca_ratio ${RATIO} --ot_lambda ${OT} \
      --queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 \
      > output/logs/train-log-${JOB} 2>&1
    echo "  Done $(date)" | tee -a $RLOG
}

# ===== 数据量消融 =====
echo "=== Data Ablation $(date) ===" | tee -a $RLOG
for n in 1000 5000 10000 25000 50000; do
    JOB="abl_n${n}_pureclip-$(date +%Y%m%d_%H%M%S)"
    run_train $JOB 1.0 0.0 $n
    JOB="abl_n${n}_pcaot-$(date +%Y%m%d_%H%M%S)"
    run_train $JOB 0.9 0.15 $n
done

# ===== pca_ot τ 扫参 =====
echo "=== PCA-OT tau Sweep $(date) ===" | tee -a $RLOG
for tau in 0.7 0.8 0.9 0.95; do
    JOB="pcaot_tau${tau}-$(date +%Y%m%d_%H%M%S)"
    run_train $JOB ${tau} 0.15 50000
done

echo "=== ALL TRAINING DONE $(date) ===" | tee -a $RLOG

# ===== 评测所有 =====
cd /home/gyf/CLIP-wgd/HiMo-CLIP
eval_docci() {
    local CKPT=$1 LABEL=$2
    CUDA_VISIBLE_DEVICES=1 python3 -c "
import sys,torch,json,os; sys.path.insert(0,'.')
from model import himo as longclip; from PIL import Image
d='cuda:0'; m,p=longclip.load('${CKPT}',device=d); m.eval()
with open('data/docci/test_set.jsonl') as f: data=[json.loads(l) for l in f]
ifm=[];tfm=[]
for e in data:
    img=Image.open(os.path.join('data/docci/images/',e['image_file'])).convert('RGB')
    ifm.append(m.encode_image(p(img).unsqueeze(0).to(d)))
    tfm.append(m.encode_text(longclip.tokenize([e['description']],truncate=True).to(d)))
ifm=torch.cat(ifm);ifm=ifm/ifm.norm(dim=-1,keepdim=True)
tfm=torch.cat(tfm);tfm=tfm/tfm.norm(dim=-1,keepdim=True)
print(f'${LABEL} I2T={sum(1 for i in range(len(data)) if (ifm[i]@tfm.T).argmax().item()==i)/len(data):.4f}')
print(f'${LABEL} T2I={sum(1 for i in range(len(data)) if (tfm[i]@ifm.T).argmax().item()==i)/len(data):.4f}')
" 2>&1 | grep "I2T\|T2I" | tee -a $RLOG
    torch.cuda.empty_cache
}

echo "=== Evaluating All $(date) ===" | tee -a $RLOG
for n in 1000 5000 10000 25000 50000; do
    for m in pureclip pcaot; do
        CKPT=$(ls -td train/output/ckpts/abl_n${n}_${m}-*/ep=9_himo.pt 2>/dev/null | head -1)
        [ -n "$CKPT" ] && eval_docci "$CKPT" "abl_n${n}_${m}" || echo "abl_n${n}_${m}: NO CKPT" | tee -a $RLOG
    done
done
for tau in 0.7 0.8 0.9 0.95; do
    CKPT=$(ls -td train/output/ckpts/pcaot_tau${tau}-*/ep=9_himo.pt 2>/dev/null | head -1)
    [ -n "$CKPT" ] && eval_docci "$CKPT" "pcaot_tau${tau}" || echo "pcaot_tau${tau}: NO CKPT" | tee -a $RLOG
done
echo "=== ALL DONE $(date) ===" | tee -a $RLOG
