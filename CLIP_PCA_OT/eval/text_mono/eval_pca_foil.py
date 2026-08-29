"""
PCA-Foil Detection Evaluation
==============================
Task: Given an image + 2 captions (one correct, one foil with 1 word swapped),
      detect which caption is wrong.

Key insight: PCA separates "common semantics" from "specific details".
  - Raw similarity: both captions look very similar → hard to distinguish
  - PCA similarity: the foil word mismatch shows up in the residual → easier to detect

Metric: Accuracy = % of pairs where correct caption gets higher similarity.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from model import himo as longclip


def load_foil_data(anno_path, image_dir, max_samples=None):
    """Load FOIL-COCO data, returning (image_path, correct_caption, foil_caption) triplets."""
    with open(anno_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Build image_id → file_name mapping
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    # Group annotations by (image_id, foil_id)
    anns = coco['annotations']
    pairs = defaultdict(list)
    for a in anns:
        key = (a['image_id'], a.get('foil_id', a['id']))
        pairs[key].append(a)

    triplets = []
    for (img_id, _), ann_list in pairs.items():
        if img_id not in id_to_file:
            continue
        corrects = [a for a in ann_list if not a.get('foil', False)]
        foils = [a for a in ann_list if a.get('foil', False)]
        if corrects and foils:
            img_path = os.path.join(image_dir, id_to_file[img_id])
            triplets.append((img_path, corrects[0]['caption'], foils[0]['caption']))

    if max_samples:
        triplets = triplets[:max_samples]

    return triplets


def evaluate_foil_accuracy(model, preprocess, triplets, device, pca_ratio=0.9, batch_size=64):
    """
    For each triplet (image, correct_caption, foil_caption), compute two scores:
      - raw_score:  cos(image, raw_caption) → correct > foil ?
      - pca_score:  cos(image, pca_caption) → correct > foil ?
    Returns accuracy for both scoring methods.
    """
    model.eval()
    raw_correct = 0
    pca_correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(triplets), batch_size):
            batch = triplets[start:start + batch_size]
            bs = len(batch)

            # Load images
            images = []
            for img_path, _, _ in batch:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(preprocess(img))
                except Exception:
                    images.append(torch.zeros(3, 224, 224))

            images = torch.stack(images).to(device)  # [B, 3, 224, 224]

            # Collect captions
            correct_caps = [c for _, c, _ in batch]
            foil_caps = [f for _, _, f in batch]

            # Encode
            img_feat = model.encode_image(images)  # [B, 768]
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # Raw text features
            text_correct_raw = model.encode_text(longclip.tokenize(correct_caps, truncate=True).to(device))
            text_foil_raw = model.encode_text(longclip.tokenize(foil_caps, truncate=True).to(device))

            # Combine for batch PCA
            all_texts = torch.cat([text_correct_raw, text_foil_raw], dim=0)  # [2B, 768]

            # Apply PCA (same as training)
            text_norm = all_texts.norm(dim=-1, keepdim=True)
            all_texts_l2 = all_texts / text_norm.clamp(min=1e-8)

            # Batch PCA
            mean = all_texts_l2.mean(dim=0, keepdim=True)
            X = all_texts_l2 - mean
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            s_sq = S ** 2
            total_var = s_sq.sum()
            if total_var < 1e-8:
                total_var = 1e-8
            cum_ratio = torch.cumsum(s_sq, dim=0) / total_var
            valid = torch.where(cum_ratio >= pca_ratio)[0]
            pca_dim = valid[0].item() + 1 if valid.numel() > 0 else len(S)
            pca_dim = min(max(1, pca_dim), min(X.shape))

            pc = Vt.T[:, :pca_dim]
            X_trans = torch.mm(X, pc)
            X_rev = torch.mm(X_trans, pc.T) + mean

            # Split back
            text_correct_pca = X_rev[:bs]
            text_foil_pca = X_rev[bs:]

            # Compute similarities
            # Raw
            raw_sim_correct = (img_feat * (text_correct_raw / text_correct_raw.norm(dim=-1, keepdim=True))).sum(-1)
            raw_sim_foil = (img_feat * (text_foil_raw / text_foil_raw.norm(dim=-1, keepdim=True))).sum(-1)

            # PCA (normalize PCA features)
            pca_sim_correct = (img_feat * (text_correct_pca / text_correct_pca.norm(dim=-1, keepdim=True).clamp(min=1e-8))).sum(-1)
            pca_sim_foil = (img_feat * (text_foil_pca / text_foil_pca.norm(dim=-1, keepdim=True).clamp(min=1e-8))).sum(-1)

            # Count correct
            raw_correct += (raw_sim_correct > raw_sim_foil).sum().item()
            pca_correct += (pca_sim_correct > pca_sim_foil).sum().item()
            total += bs

    return {
        'total': total,
        'raw_accuracy': raw_correct / total * 100,
        'pca_accuracy': pca_correct / total * 100,
        'raw_correct': raw_correct,
        'pca_correct': pca_correct,
    }


def run(model_path, foil_json, image_dir, max_samples=None, pca_ratio=0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    model, preprocess = longclip.load(model_path, device=device)
    model.eval()
    print("Model loaded.")

    triplets = load_foil_data(foil_json, image_dir, max_samples=max_samples)
    print(f"Loaded {len(triplets)} foil triplets")

    results = evaluate_foil_accuracy(model, preprocess, triplets, device, pca_ratio=pca_ratio)
    print(f"\n{'='*50}")
    print(f"FOIL Detection Accuracy ({len(triplets)} samples)")
    print(f"{'='*50}")
    print(f"  Raw similarity:  {results['raw_accuracy']:.2f}%")
    print(f"  PCA similarity:  {results['pca_accuracy']:.2f}%")
    print(f"  Delta (PCA-Raw): {results['pca_accuracy'] - results['raw_accuracy']:+.2f}%")

    # Per-category analysis: check how many swaps PCA helps
    print(f"\n  PCA helps: {(results['pca_correct'] - results['raw_correct'])} / {results['total']} cases")

    return results


if __name__ == "__main__":
    # Default paths (run from project root)
    FOIL_JSON = "data/foilv1.0_test_2017.json"
    IMAGE_DIR = "data/train2017/"  # FOIL uses COCO train2017 images

    model_path = sys.argv[1] if len(sys.argv) > 1 else "weights/ViT-L-14.pt"
    pca_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    run(model_path, FOIL_JSON, IMAGE_DIR,
        max_samples=max_samples if max_samples > 0 else None,
        pca_ratio=pca_ratio)
