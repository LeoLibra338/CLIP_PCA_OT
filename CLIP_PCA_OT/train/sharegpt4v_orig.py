import json
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch, torch.utils.data as data, os, random

captions_json = '../data/coco_train_captions.json'
image_dir = '../data/train2017/'

def _transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC), CenterCrop(n_px),
        lambda image: image.convert("RGB"), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class share4v_val_dataset(data.Dataset):
    def __init__(self, val_size=1000):
        with open(captions_json, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:val_size]
        self.preprocess = _transform()
    def __len__(self): return len(self.json_data)
    def __getitem__(self, index):
        item = self.json_data[index]
        img = Image.open(os.path.join(image_dir, item['image']))
        return self.preprocess(img), item['caption']

class share4v_train_dataset(data.Dataset):
    def __init__(self, val_size=1000):
        with open(captions_json, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[val_size:]
        self.preprocess = _transform()
    def __len__(self): return len(self.json_data)
    def __getitem__(self, index):
        try:
            item = self.json_data[index]
            img = Image.open(os.path.join(image_dir, item['image']))
            return self.preprocess(img), item['caption'], item['caption_short']
        except: return self[index+1]
