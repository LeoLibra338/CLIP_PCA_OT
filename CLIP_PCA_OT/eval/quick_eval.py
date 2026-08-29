import sys, torch, json, os
sys.path.insert(0, '/home/gyf/CLIP-wgd/HiMo-CLIP')
from model import himo as longclip
from PIL import Image
import glob

device = 'cuda:0'
models = {
    'pure_clip': 'train/output/ckpts/final_pure_50k/ep=9_himo.pt',
    'pca_ot': 'train/output/ckpts/final_pcaot_50k/ep=9_himo.pt',
}

with open('data/docci/test_set.jsonl') as f:
    data = [json.loads(l) for l in f]
img_dir = 'data/docci/images/'

for name, ckpt in models.items():
    print(f'=== {name} ===')
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
    print(f'  DOCCI I2T={i2t:.4f}')
    print(f'  DOCCI T2I={t2i:.4f}')
    torch.cuda.empty_cache()
