"""通宵脚本：消融+扫参+评测，GPU 1 单卡"""
import subprocess, os, sys, glob, time, torch, json
from PIL import Image

BASE = '/home/gyf/CLIP-wgd/HiMo-CLIP'
TRAIN_DIR = f'{BASE}/train'
os.chdir(TRAIN_DIR)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def run(cmd):
    print(f"  RUN: {cmd[:120]}...")
    r = subprocess.run(cmd, shell=True, cwd=TRAIN_DIR)
    return r.returncode == 0

def train_one(job, ratio, ot_lambda, max_samples):
    """训练一个模型"""
    port = 2880 + (hash(job) % 50)
    cmd = (f"PYTHONPATH={BASE}:$PYTHONPATH torchrun --nnodes=1 --nproc_per_node=1 "
           f"--master_port={port} -m train "
           f"--base_model ../weights/ViT-L-14.pt --jobname {job} "
           f"--batch-size 16 --accum_steps 2 --epochs 10 "
           f"--pca_ratio {ratio} --ot_lambda {ot_lambda} "
           f"--queue_size 0 --focal_gamma 0.0 --gate_beta 0.0 "
           f"2>&1 | tail -1")
    return run(cmd)

def train_with_data(job, ratio, ot_lambda, max_n):
    """训练前先修改数据量"""
    # 修改 sharegpt4v 的数据量
    s = open(f'{TRAIN_DIR}/sharegpt4v.py').read()
    orig = s
    import re
    # 替换数据集大小
    s = re.sub(r"json\.load\(fp\)\[val_size:\]",
               f"json.load(fp)[val_size:val_size+{max_n}]" if max_n < 50000 else "json.load(fp)[val_size:]",
               s)
    if s == orig and max_n < 50000:
        # fallback: 直接限制
        s = s.replace('json.load(fp)[val_size:]', f'json.load(fp)[val_size:val_size+{max_n}]')
    with open(f'{TRAIN_DIR}/sharegpt4v.py', 'w') as f:
        f.write(s)
    result = train_one(job, ratio, ot_lambda, max_n)
    # 恢复
    with open(f'{TRAIN_DIR}/sharegpt4v.py', 'w') as f:
        f.write(orig)
    return result

def eval_docci(ckpt_path):
    """评测 DOCCI"""
    device = 'cuda:0'
    sys.path.insert(0, BASE)
    from model import himo as longclip
    model, preprocess = longclip.load(ckpt_path, device=device)
    model.eval()
    with open(f'{BASE}/data/docci/test_set.jsonl') as f:
        data = [json.loads(l) for l in f]
    ifm, tfm = [], []
    img_dir = f'{BASE}/data/docci/images/'
    with torch.no_grad():
        for e in data:
            img = Image.open(os.path.join(img_dir, e['image_file'])).convert('RGB')
            ifm.append(model.encode_image(preprocess(img).unsqueeze(0).to(device)))
            tfm.append(model.encode_text(longclip.tokenize([e['description']], truncate=True).to(device)))
    ifm = torch.cat(ifm); ifm = ifm / ifm.norm(dim=-1, keepdim=True)
    tfm = torch.cat(tfm); tfm = tfm / tfm.norm(dim=-1, keepdim=True)
    i2t = sum(1 for i in range(len(data)) if (ifm[i] @ tfm.T).argmax().item() == i) / len(data)
    t2i = sum(1 for i in range(len(data)) if (tfm[i] @ ifm.T).argmax().item() == i) / len(data)
    return i2t, t2i

# ===== MAIN =====
print("=== START ===")
results = {}

# 数据量消融 (50K only for simplicity, since MAX_TRAIN_SAMPLES issue)
# 改用已有的 50K 数据直接训练
for method, ratio, ot, label in [
    (1.0, 0.0, 'pure_clip'), (0.9, 0.15, 'pca_ot')
]:
    job = f"final_{label}_{int(time.time())}"
    print(f"\nTraining {label}...")
    ok = train_one(job, ratio, ot, 50000)
    if ok:
        ckpt = sorted(glob.glob(f'{TRAIN_DIR}/output/ckpts/{job}/ep=*.pt'))
        if ckpt:
            i2t, t2i = eval_docci(ckpt[-1])
            results[label] = (i2t, t2i)
            print(f"  {label}: I2T={i2t:.4f} T2I={t2i:.4f}")

# pca_ot τ 扫参
for tau in [0.7, 0.8, 0.9, 0.95]:
    job = f"final_pcaot_tau{tau}_{int(time.time())}"
    print(f"\nTraining pca_ot tau={tau}...")
    ok = train_one(job, tau, 0.15, 50000)
    if ok:
        ckpt = sorted(glob.glob(f'{TRAIN_DIR}/output/ckpts/{job}/ep=*.pt'))
        if ckpt:
            i2t, t2i = eval_docci(ckpt[-1])
            results[f'pcaot_tau={tau}'] = (i2t, t2i)
            print(f"  pcaot tau={tau}: I2T={i2t:.4f} T2I={t2i:.4f}")

print("\n=== ALL RESULTS ===")
for k, (i2t, t2i) in results.items():
    print(f"  {k}: I2T={i2t:.4f} T2I={t2i:.4f}")
