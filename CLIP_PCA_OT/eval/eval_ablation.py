"""评测所有消融模型（10个）"""
import sys, torch, json, os, glob
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
    ('1K pure', f'{base}/v2_pure_n1000'),
    ('1K pca_ot', f'{base}/v2_pcaot_n1000'),
]

with open('/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/test_set.jsonl') as f:
    data = [json.loads(l) for l in f]
img_dir = '/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/images/'

print(f'{"Model":<14} {"I2T":>8} {"T2I":>8}')
print('-' * 34)
for name, ckpt_dir in jobs:
    ckpt = f'{ckpt_dir}/ep=9_himo.pt'
    if not os.path.exists(ckpt):
        print(f'{name:<14} {"MISSING":>8}')
        continue
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
    print(f'{name:<14} {i2t:>8.4f} {t2i:>8.4f}')
    torch.cuda.empty_cache()
