"""
零样本能力保持测试：CLIP原版 vs pca_pure vs pca_ot
用 COCO val 图片 + ImageNet 标准类别名 → 测零样本分类准确率
未经分类训练的模型，分类越准 = 零样本保持越好
"""
import sys
import torch
import numpy as np
from PIL import Image
import json
from model import himo as longclip

# 代表性的 200 个 ImageNet 类别（覆盖常见物体）
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
    "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
    "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
    "green snake", "king snake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba",
    "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider",
    "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
    "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
    "flatworm", "nematode", "conch", "snail", "slug",
    "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish",
    "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "great egret", "bittern", "crane",
    "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
    "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
    "king penguin", "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese",
    "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
    "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
    "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
    "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
    "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
    "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier",
    "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
    "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont",
    "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
    "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa",
    "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
]

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "a {} in the image.",
    "a photo of the {}.",
]


def zero_shot_classify(model, images, class_names, templates, device):
    """标准 CLIP 零样本分类"""
    # 编码所有类别文本
    text_features = []
    for name in class_names:
        texts = [t.format(name) for t in templates]
        text_input = longclip.tokenize(texts, truncate=True).to(device)
        feat = model.encode_text(text_input)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        text_features.append(feat.mean(dim=0))  # 模板平均
    text_features = torch.stack(text_features)  # [N_classes, D]

    # 编码图像
    img_feat = model.encode_image(images)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # 相似度
    logits = img_feat @ text_features.T  # [B, N_classes]
    return logits.argmax(dim=-1)


def run_zero_shot(model_path, device='cuda:0', num_images=500):
    print(f"\n{'='*50}")
    print(f"模型: {model_path}")

    model, preprocess = longclip.load(model_path, device=device)
    model.eval()

    # 加载 COCO val 图片
    from torchvision.datasets import CocoCaptions
    import os
    os.chdir('/home/gyf/CLIP-wgd/HiMo-CLIP')
    coco = CocoCaptions(
        root='data/coco/val2017/',
        annFile='data/coco/annotations/captions_val2017.json',
        transform=None
    )

    images = []
    coco_classes = set()
    for i in range(min(num_images, len(coco))):
        img, _ = coco[i]
        images.append(preprocess(img))

    images = torch.stack(images).to(device)

    # 零样本分类
    with torch.no_grad():
        preds = zero_shot_classify(model, images, IMAGENET_CLASSES, IMAGENET_TEMPLATES, device)

    # Top-5 准确率（零样本通常报 Top-5）
    # 因为没有 ground truth（ImageNet类别≠COCO标签），
    # 我们报预测分布的熵 + top-1 分布的集中度
    # 更集中的预测 = 更好的零样本能力

    # 用 softmax 温度 1 看预测置信度
    img_feat = model.encode_image(images)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    text_feats = []
    for name in IMAGENET_CLASSES:
        texts = [t.format(name) for t in IMAGENET_TEMPLATES]
        text_input = longclip.tokenize(texts, truncate=True).to(device)
        feat = model.encode_text(text_input)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        text_feats.append(feat.mean(dim=0))
    text_feats = torch.stack(text_feats)

    logits = img_feat @ text_feats.T
    probs = torch.softmax(logits, dim=-1)

    # 最高概率的平均值和分布熵
    top1_confidence = probs.max(dim=-1).values.mean().item()
    top5_confidence = probs.topk(5, dim=-1).values.sum(dim=-1).mean().item()
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()

    print(f"  零样本置信度 (越高=越不遗忘):")
    print(f"    Top-1 confidence: {top1_confidence:.4f}")
    print(f"    Top-5 confidence: {top5_confidence:.4f}")
    print(f"    预测熵 (越低越好): {entropy:.4f}")

    # 预测多样性（预测的类别种类数）
    unique_preds = len(set(preds.cpu().numpy()))
    print(f"    预测多样性: {unique_preds}/{len(IMAGENET_CLASSES)} unique classes")

    return top1_confidence, top5_confidence, entropy, unique_preds


if __name__ == "__main__":
    models = {
        "ViT-L/14 (original)": "/home/gyf/CLIP-wgd/HiMo-CLIP/weights/ViT-L-14.pt",
        "pure_clip (InfoNCE)": "/home/gyf/CLIP-wgd/HiMo-CLIP/output/ckpts/pure_clip-20260629_154440/ep=9_himo.pt",
        "pca_pure (PCA)": "/home/gyf/CLIP-wgd/HiMo-CLIP/output/ckpts/pca_pure-20260628_010254/ep=9_himo.pt",
        "pca_ot (PCA+OT)": "/home/gyf/CLIP-wgd/HiMo-CLIP/output/ckpts/pca_ot-20260627_225133/ep=9_himo.pt",
    }

    results = {}
    for name, path in models.items():
        t1, t5, ent, div = run_zero_shot(path)
        results[name] = {"top1": t1, "top5": t5, "entropy": ent, "diversity": div}

    print(f"\n{'='*50}")
    print("零样本能力对比")
    print(f"{'='*50}")
    print(f"{'模型':<25} {'Top-1':>7} {'Top-5':>7} {'Entropy':>8} {'Diversity':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['top1']:>7.4f} {r['top5']:>7.4f} {r['entropy']:>8.4f} {r['diversity']:>10}")
