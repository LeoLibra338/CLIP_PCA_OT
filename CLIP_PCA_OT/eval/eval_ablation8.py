"""评测 8 个有效消融模型"""
import sys, torch, json, os
sys.path.insert(0, '/home/gyf/CLIP-wgd/HiMo-CLIP')
from model import himo as longclip
from PIL import Image

device = 'cuda:0'
base = '/home/gyf/CLIP-wgd/HiMo-CLIP/train/output/ckpts'

jobs = [
    ('50K pure', f'{base}/v2_pure_50k'),
    ('50K pca_ot', f'{base}/v2_pcaot_50k'),
    ('25K pure', f'{base}/v2_pure_n25000'),
    ('25K pca_ot', f'{base}/v2_pcaot_n25000'),
    ('10K pure', f'{base}/v2_pure_n10000'),
    ('10K pca_ot', f'{base}/v2_pcaot_n10000'),
    ('5K pure', f'{base}/v2_pure_n5000'),
    ('5K pca_ot', f'{base}/v2_pcaot_n5000'),
]

with open('/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/test_set.jsonl') as f:
    data = [json.loads(l) for l in f]
img_dir = '/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/images/'

results = []
for name, ckpt_dir in jobs:
    ckpt = f'{ckpt_dir}/ep=9_himo.pt'
    model, preprocess = longclip.load(ckpt, device=device)
    model.eval()
    ifm, tfm = [], []
    with torch.no_grad():
        for e in data:
            img = Image.open(os.path.join(img_dir, e['image_file'])).convert('RGB')
            ifm.append(model.encode_image(preprocess(img).unsqueeze(0).to(device)))
            tfm.append(model.encode_text(longclip.tokenize([e['description']], truncate=True).to(device)))
    ifm = torch.cat(ifm); ifm = ifm / ifm.norm(dim=-1, keepdim=True)
    tfm = torch.cat(tfm); tfm = tfm / tfm.norm(dim=-1, keepdim=True)
    i2t = sum(1 for i in range(len(data)) if (ifm[i] @ tfm.T).argmax().item() == i) / len(data)
    t2i = sum(1 for i in range(len(data)) if (tfm[i] @ ifm.T).argmax().item() == i) / len(data)
    results.append((name, i2t, t2i))
    print(f'{name:<12} I2T={i2t:.4f} T2I={t2i:.4f}', flush=True)
    torch.cuda.empty_cache()

# 保存结果
with open('/home/gyf/CLIP-wgd/HiMo-CLIP/eval/ablation_results.json', 'w') as f:
    json.dump([{'name': n, 'i2t': i, 't2i': t} for n, i, t in results], f, indent=2)
print('\nSaved to eval/ablation_results.json')
