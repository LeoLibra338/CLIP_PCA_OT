"""Evaluate all PCA ratio sweeps on DOCCI"""
import sys, torch, json, os
sys.path.insert(0, '/home/gyf/CLIP-wgd/HiMo-CLIP')
from model import himo as longclip
from PIL import Image

base = '/home/gyf/CLIP-wgd/HiMo-CLIP/train/output/ckpts'
import glob

for tau in ['0.7', '0.8', '0.9', '0.95']:
    dirs = sorted(glob.glob(f'{base}/pca_tau{tau}-*'))
    if not dirs:
        print(f'tau={tau}: NO CHECKPOINT')
        continue
    ckpt = os.path.join(dirs[-1], 'ep=9_himo.pt')
    print(f'tau={tau}: {ckpt}')

    device = 'cuda:0'
    model, preprocess = longclip.load(ckpt, device=device)
    model.eval()

    with open('/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/test_set.jsonl') as f:
        data = [json.loads(l) for l in f]

    ifm, tfm = [], []
    img_dir = '/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/images/'
    with torch.no_grad():
        for e in data:
            img = Image.open(os.path.join(img_dir, e['image_file'])).convert('RGB')
            ifm.append(model.encode_image(preprocess(img).unsqueeze(0).to(device)))
            tok = longclip.tokenize([e['description']], truncate=True).to(device)
            tfm.append(model.encode_text(tok))

    ifm = torch.cat(ifm); ifm = ifm / ifm.norm(dim=-1, keepdim=True)
    tfm = torch.cat(tfm); tfm = tfm / tfm.norm(dim=-1, keepdim=True)

    i2t = sum(1 for i in range(len(data)) if (ifm[i] @ tfm.T).argmax().item() == i)
    t2i = sum(1 for i in range(len(data)) if (tfm[i] @ ifm.T).argmax().item() == i)
    print(f'  DOCCI I2T={i2t/len(data):.4f}  DOCCI T2I={t2i/len(data):.4f}')
    torch.cuda.empty_cache()
