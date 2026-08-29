"""
干净版 CLIP 训练：不依赖任何历史修改
用法: python clean_train.py --data_size 5000 --pca_ratio 0.9 --ot_lambda 0.15
"""
import sys, os, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

sys.path.insert(0, '/home/gyf/CLIP-wgd/HiMo-CLIP')
from model import himo as longclip


# ============ 数据 ============
def make_transform():
    return Compose([
        Resize(224, interpolation=Image.BICUBIC), CenterCrop(224),
        lambda image: image.convert("RGB"), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_dir, transform, size=None):
        with open(data_path) as f:
            data = json.load(f)
        random.seed(42)
        if size and size < len(data):
            data = random.sample(data, size)
        self.data = data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        img = Image.open(os.path.join(self.image_dir, item['image'])).convert('RGB')
        return self.transform(img), item['caption']


# ============ 训练 ============
def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    for images, texts in loader:
        images = images.to(device)
        tokens = longclip.tokenize(texts, truncate=True).to(device)

        f_img = model.encode_image(images)
        f_txt = model.encode_text(tokens)
        f_img = f_img / f_img.norm(dim=-1, keepdim=True)
        f_txt = f_txt / f_txt.norm(dim=-1, keepdim=True)

        logits = model.logit_scale.exp() * (f_img @ f_txt.T)
        labels = torch.arange(len(images), device=device)
        loss = (F.cross_entropy(logits, labels, label_smoothing=0.1)
                + F.cross_entropy(logits.T, labels, label_smoothing=0.1)) / 2

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_docci(model, preprocess, device):
    model.eval()
    with open('/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/test_set.jsonl') as f:
        data = [json.loads(l) for l in f]
    img_dir = '/home/gyf/CLIP-wgd/HiMo-CLIP/data/docci/images/'
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
    return i2t, t2i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=5000)
    parser.add_argument('--pca_ratio', type=float, default=0.9)
    parser.add_argument('--ot_lambda', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device
    print(f'Device: {device}, data_size={args.data_size}, pca_ratio={args.pca_ratio}, ot_lambda={args.ot_lambda}')

    # 加载模型
    model, preprocess = longclip.load('/home/gyf/CLIP-wgd/HiMo-CLIP/weights/ViT-L-14.pt', device=device)
    model = model.float()
    # 与原版训练一致的 logit_scale 初始化
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)
    model.train()
    print('Model loaded')

    # 数据
    dataset = SimpleDataset(
        '/home/gyf/CLIP-wgd/HiMo-CLIP/data/coco_train_captions.json',
        '/home/gyf/CLIP-wgd/HiMo-CLIP/data/train2017/',
        preprocess, size=args.data_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f'Dataset: {len(dataset)} samples, {len(loader)} batches')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, opt, device)
        print(f'Epoch {epoch}: loss={loss:.4f}')

    # 评测
    i2t, t2i = evaluate_docci(model, preprocess, device)
    print(f'RESULT: I2T={i2t:.4f} T2I={t2i:.4f}')

    # 保存
    os.makedirs('clean_ckpts', exist_ok=True)
    torch.save(model.state_dict(), f'clean_ckpts/clean_n{args.data_size}_pca{args.pca_ratio}_ot{args.ot_lambda}.pt')
    print('Saved')


if __name__ == '__main__':
    main()
