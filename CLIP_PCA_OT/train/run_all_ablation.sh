#!/bin/bash
# 数据量消融：pure_clip vs pca_ot(λ=0.15)，全部串行
# n=1K, 5K, 10K, 25K, 50K，每个跑 pure_clip + pca_ot
# 全部完成后自动评测 DOCCI I2T
cd /home/gyf/CLIP-wgd/HiMo-CLIP/train

RESULTS_FILE="output/logs/ablation_results.txt"
echo "=== Data Ablation Started $(date) ===" > $RESULTS_FILE

for n in 1000 5000 10000 25000 50000; do
  for method in pure_clip pca_ot; do
    if [ "$method" = "pure_clip" ]; then
      RATIO=1.0; OT=0.0; LABEL="pureclip"
    else
      RATIO=0.9; OT=0.15; LABEL="pcaot"
    fi
    JOB="abl_n${n}_${LABEL}-$(date +%Y%m%d_%H%M%S)"
    PORT=$((2830 + RANDOM % 50))
    echo "=== n=${n} ${method} ratio=${RATIO} ot=${OT} $(date) ===" | tee -a $RESULTS_FILE
    PYTHONPATH=/home/gyf/CLIP-wgd/HiMo-CLIP:$PYTHONPATH \
    MAX_TRAIN_SAMPLES=${n} \
    CUDA_VISIBLE_DEVICES=1,3 \
    torchrun --nnodes=1 --nproc_per_node=2 --master_port=${PORT} \
      -m train_ablation \
      --base_model '../weights/ViT-L-14.pt' \
      --jobname $JOB --batch-size 32 --accum_steps 4 --epochs 10 \
      --pca_ratio ${RATIO} --ot_lambda ${OT} \
      --queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 \
      > output/logs/train-log-${JOB} 2>&1
    echo "Done n=${n} ${method} $(date)" | tee -a $RESULTS_FILE
  done
done

echo "=== All training done, starting DOCCI eval $(date) ===" | tee -a $RESULTS_FILE

# 评测所有模型
cd /home/gyf/CLIP-wgd/HiMo-CLIP
for n in 1000 5000 10000 25000 50000; do
  for method in pure_clip pca_ot; do
    if [ "$method" = "pure_clip" ]; then LABEL="pureclip"; else LABEL="pcaot"; fi
    CKPT=$(ls -td train/output/ckpts/abl_n${n}_${LABEL}-*/ep=9_himo.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
      echo "n=${n} ${method}: NO CHECKPOINT" | tee -a $RESULTS_FILE
      continue
    fi
    echo "Evaluating n=${n} ${method}: $CKPT" | tee -a $RESULTS_FILE
    CUDA_VISIBLE_DEVICES=0 python3 -c "
import sys,torch,json,os; sys.path.insert(0,'.')
from model import himo as longclip; from PIL import Image
d='cuda:0'; m,p=longclip.load('$CKPT',device=d); m.eval()
with open('data/docci/test_set.jsonl') as f: data=[json.loads(l) for l in f]
ifm=[];tfm=[]
for e in data:
    img=Image.open(os.path.join('data/docci/images/',e['image_file'])).convert('RGB')
    ifm.append(m.encode_image(p(img).unsqueeze(0).to(d)))
    tfm.append(m.encode_text(longclip.tokenize([e['description']],truncate=True).to(d)))
ifm=torch.cat(ifm);ifm=ifm/ifm.norm(dim=-1,keepdim=True)
tfm=torch.cat(tfm);tfm=tfm/tfm.norm(dim=-1,keepdim=True)
print(f'DOCCI_I2T={sum(1 for i in range(len(data)) if (ifm[i]@tfm.T).argmax().item()==i)/len(data):.4f}')
print(f'DOCCI_T2I={sum(1 for i in range(len(data)) if (tfm[i]@ifm.T).argmax().item()==i)/len(data):.4f}')
" 2>&1 | grep "DOCCI" | tee -a $RESULTS_FILE
    torch.cuda.empty_cache()
  done
done
echo "=== ALL DONE $(date) ===" | tee -a $RESULTS_FILE

# ===== pca_ot τ 扫参（接在消融后面）=====
echo "=== PCA-OT tau sweep starting $(date) ===" | tee -a $RESULTS_FILE
for tau in 0.7 0.8 0.9 0.95; do
    JOB="pcaot_tau${tau}-$(date +%Y%m%d_%H%M%S)"
    PORT=$((2850 + RANDOM % 50))
    echo "=== pcaot tau=${tau} $(date) ===" | tee -a $RESULTS_FILE
    PYTHONPATH=/home/gyf/CLIP-wgd/HiMo-CLIP:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=1,3 \
    torchrun --nnodes=1 --nproc_per_node=2 --master_port=${PORT} \
      -m train \
      --base_model '../weights/ViT-L-14.pt' \
      --jobname $JOB --batch-size 32 --accum_steps 4 --epochs 10 \
      --pca_ratio ${tau} --ot_lambda 0.15 \
      --queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 \
      > output/logs/train-log-${JOB} 2>&1
    echo "Done pcaot tau=${tau} $(date)" | tee -a $RESULTS_FILE
done

# ===== 评测 pcaot τ 扫参 =====
echo "=== Evaluating pcaot tau sweep $(date) ===" | tee -a $RESULTS_FILE
cd /home/gyf/CLIP-wgd/HiMo-CLIP
for tau in 0.7 0.8 0.9 0.95; do
    CKPT=$(ls -td train/output/ckpts/pcaot_tau${tau}-*/ep=9_himo.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "pcaot tau=${tau}: NO CKPT" | tee -a $RESULTS_FILE; continue
    fi
    echo "Eval pcaot tau=${tau}" | tee -a $RESULTS_FILE
    CUDA_VISIBLE_DEVICES=0 python3 -c "
import sys,torch,json,os; sys.path.insert(0,'.')
from model import himo as longclip; from PIL import Image
d='cuda:0'; m,p=longclip.load('$CKPT',device=d); m.eval()
with open('data/docci/test_set.jsonl') as f: data=[json.loads(l) for l in f]
ifm=[];tfm=[]
for e in data:
    img=Image.open(os.path.join('data/docci/images/',e['image_file'])).convert('RGB')
    ifm.append(m.encode_image(p(img).unsqueeze(0).to(d)))
    tfm.append(m.encode_text(longclip.tokenize([e['description']],truncate=True).to(d)))
ifm=torch.cat(ifm);ifm=ifm/ifm.norm(dim=-1,keepdim=True)
tfm=torch.cat(tfm);tfm=tfm/tfm.norm(dim=-1,keepdim=True)
print(f'DOCCI_I2T={sum(1 for i in range(len(data)) if (ifm[i]@tfm.T).argmax().item()==i)/len(data):.4f}')
print(f'DOCCI_T2I={sum(1 for i in range(len(data)) if (tfm[i]@ifm.T).argmax().item()==i)/len(data):.4f}')
" 2>&1 | grep "DOCCI" | tee -a $RESULTS_FILE
    torch.cuda.empty_cache()
done
echo "=== ALL DONE $(date) ===" | tee -a $RESULTS_FILE
