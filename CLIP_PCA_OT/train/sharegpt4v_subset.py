"""支持数据量限制的训练数据集"""
import json, os, random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch, torch.utils.data as data

_image_dir = '../data/train2017/'

def _transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC), CenterCrop(n_px),
        lambda image: image.convert("RGB"), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def create_train_dataset(max_samples=50000):
    """创建指定大小的训练集"""
    with open('../data/coco_train_captions.json', 'r', encoding='utf8') as fp:
        all_data = json.load(fp)[1000:]  # skip val
    if max_samples and max_samples < len(all_data):
        random.seed(42)
        all_data = random.sample(all_data, max_samples)

    class SubsetDataset(data.Dataset):
        def __init__(self):
            self.data = all_data
            self.preprocess = _transform()
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            img = Image.open(os.path.join(_image_dir, item['image']))
            return self.preprocess(img), item['caption'], item['caption_short']
    return SubsetDataset()
