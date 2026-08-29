"""
Docci I2T / T2I retrieval evaluation with R@1, R@5, R@10.
Usage: python eval_retrieval.py <model_path> <jobname>
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from model import himo as longclip


def load_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_retrieval(model, preprocess, data, device, image_root, print_freq=100):
    """Compute I2T and T2I R@1, R@5, R@10 on Docci."""
    model.eval()

    all_image_feats = []    # [N_img, D]
    all_text_feats = []     # [total_captions, D]
    img_to_caption_range = []  # (start, end) for each image's captions in all_text_feats

    with torch.no_grad():
        # --- Encode all images ---
        for idx, ele in enumerate(data):
            img_path = os.path.join(image_root, ele["img_path"])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[Warning] fail to open image {img_path}: {e}")
                all_image_feats.append(torch.zeros(model.visual.output_dim))
                continue

            image = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_image_feats.append(feat.cpu())

            if (idx + 1) % print_freq == 0:
                print(f"Image encoding: {idx + 1}/{len(data)}")

        all_image_feats = torch.cat(all_image_feats, dim=0)  # [N, D]

        # --- Encode all captions ---
        BATCH_SIZE = 128
        all_captions_flat = []
        caption_start = 0
        for idx, ele in enumerate(data):
            captions = ele["caption"]
            all_captions_flat.extend(captions)
            n = len(captions)
            img_to_caption_range.append((caption_start, caption_start + n))
            caption_start += n

        for start in range(0, len(all_captions_flat), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(all_captions_flat))
            batch_captions = all_captions_flat[start:end]
            text_tokens = longclip.tokenize(batch_captions, truncate=True).to(device)
            feats = model.encode_text(text_tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_text_feats.append(feats.cpu())
            if (start // BATCH_SIZE + 1) % print_freq == 0:
                print(f"Text encoding: {end}/{len(all_captions_flat)}")

        all_text_feats = torch.cat(all_text_feats, dim=0)  # [M, D]

    N_img = all_image_feats.shape[0]
    M_txt = all_text_feats.shape[0]

    # --- I2T: for each image, rank all captions ---
    all_image_feats_gpu = all_image_feats.cuda()
    all_text_feats_gpu = all_text_feats.cuda()

    i2t_r1, i2t_r5, i2t_r10 = 0, 0, 0
    for i in range(N_img):
        start, end = img_to_caption_range[i]
        sim = all_image_feats_gpu[i] @ all_text_feats_gpu.T  # [M]
        _, topk = sim.topk(max(10, end - start))
        for r in topk.tolist():
            if start <= r < end:
                i2t_r10 += 1
                break
    for i in range(N_img):
        start, end = img_to_caption_range[i]
        sim = all_image_feats_gpu[i] @ all_text_feats_gpu.T
        _, topk = sim.topk(5)
        for r in topk.tolist():
            if start <= r < end:
                i2t_r5 += 1
                break
    for i in range(N_img):
        start, end = img_to_caption_range[i]
        sim = all_image_feats_gpu[i] @ all_text_feats_gpu.T
        _, topk = sim.topk(1)
        for r in topk.tolist():
            if start <= r < end:
                i2t_r1 += 1
                break

    i2t_r1 /= N_img
    i2t_r5 /= N_img
    i2t_r10 /= N_img

    # --- T2I: for each caption, rank all images ---
    t2i_r1, t2i_r5, t2i_r10 = 0, 0, 0
    for i in range(N_img):
        start, end = img_to_caption_range[i]
        for j in range(start, end):
            sim = all_text_feats_gpu[j] @ all_image_feats_gpu.T  # [N]
            _, topk = sim.topk(max(10, 1))
            if i in topk.tolist():
                t2i_r10 += 1
    for i in range(N_img):
        start, end = img_to_caption_range[i]
        for j in range(start, end):
            sim = all_text_feats_gpu[j] @ all_image_feats_gpu.T
            _, topk = sim.topk(5)
            if i in topk.tolist():
                t2i_r5 += 1
    for i in range(N_img):
        start, end = img_to_caption_range[i]
        for j in range(start, end):
            sim = all_text_feats_gpu[j] @ all_image_feats_gpu.T
            _, topk = sim.topk(1)
            if i in topk.tolist():
                t2i_r1 += 1

    t2i_r1 /= M_txt
    t2i_r5 /= M_txt
    t2i_r10 /= M_txt

    print(f"\n{'='*50}")
    print(f"I2T R@1: {i2t_r1:.4f}  R@5: {i2t_r5:.4f}  R@10: {i2t_r10:.4f}")
    print(f"T2I R@1: {t2i_r1:.4f}  R@5: {t2i_r5:.4f}  R@10: {t2i_r10:.4f}")
    print(f"{'='*50}")

    return {
        "I2T_R@1": round(i2t_r1, 4),
        "I2T_R@5": round(i2t_r5, 4),
        "I2T_R@10": round(i2t_r10, 4),
        "T2I_R@1": round(t2i_r1, 4),
        "T2I_R@5": round(t2i_r5, 4),
        "T2I_R@10": round(t2i_r10, 4),
    }


def run(model_path, jobname, output_root, image_root):
    output_dir = os.path.join(output_root, jobname)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Load model from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(model_path, device=device)
    print("Model loaded.")

    data_file = "./himo_docci_data.json"
    task_data = load_jsonl(data_file)
    print(f"Docci num: {len(task_data)}")

    print("===> Evaluating I2T / T2I retrieval ...")
    metrics = evaluate_retrieval(model, preprocess, task_data, device, image_root)

    metrics["model"] = model_path
    metrics["HiMoK_pearson"] = None  # will be filled

    out_path = os.path.join(output_dir, "retrieval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    output_root = "./"
    image_root = "../../data/docci/images/"

    model_path = sys.argv[1]
    jobname = sys.argv[2]

    run(model_path, jobname, output_root, image_root)
