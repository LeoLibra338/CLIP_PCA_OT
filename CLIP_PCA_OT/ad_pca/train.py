"""
Training: OASIS-1 → Internal 80 evaluation
PCA-Regularized Multimodal AD Diagnosis
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

from dataset import OASIS1Dataset, InternalDataset
from model import PCACrossModal


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_ce = 0
    total_pca = 0

    for t1, clinical, labels in tqdm(loader, desc='Train', leave=False):
        t1 = t1.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, ce, pca = model.compute_loss(t1, clinical, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_pca += pca.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_pca / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for t1, clinical, labels, *_ in loader:
        t1 = t1.to(device)
        clinical = clinical.to(device)

        logits, _, _, _, _, _ = model(t1, clinical, return_features=True)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
    except:
        auc = 0.0

    return {'acc': acc, 'f1': f1, 'auc': auc, 'cm': cm,
            'preds': all_preds, 'labels': all_labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--pca_ratio', type=float, default=0.85)
    parser.add_argument('--pca_weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ===== 数据加载 =====
    print('Loading OASIS-1...')
    train_set = OASIS1Dataset(
        csv_path='/home/gyf/data/oasis/OASIS-1/final_dataset.csv',
        mri_root='/home/gyf/data/oasis/OASIS-1/mri_data_full',
    )
    train_set.fit_normalizer()

    print(f'OASIS-1: {len(train_set)} subjects')
    labels = [train_set[i][2] for i in range(len(train_set))]
    from collections import Counter
    print(f'  Label distribution: {dict(Counter(labels))}')

    # 加权采样（处理不平衡）
    class_counts = Counter(labels)
    weights = {c: 1.0 / class_counts[c] for c in class_counts}
    sample_weights = [weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2)

    print('Loading Internal 80...')
    test_set = InternalDataset(
        nii_dir='/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal',
        csv_path='/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal/patients.csv',
        clinical_mean=train_set.clinical_mean,
        clinical_std=train_set.clinical_std,
    )

    print(f'Internal: {len(test_set)} subjects')
    test_labels = [test_set[i][2] for i in range(len(test_set))]
    print(f'  Label distribution: {dict(Counter(test_labels))}')

    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # ===== 模型 =====
    model = PCACrossModal(
        img_dim=512, clinical_dim=64, num_classes=3,
        pca_ratio=args.pca_ratio, pca_weight=args.pca_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ===== 训练 =====
    best_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_ce, train_pca = train_epoch(
            model, train_loader, optimizer, device
        )
        scheduler.step()

        # 每 10 个 epoch 评测
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, test_loader, device)
            print(f'Epoch {epoch+1:3d} | '
                  f'Train Loss={train_loss:.4f} CE={train_ce:.4f} PCA={train_pca:.4f} | '
                  f'Test Acc={metrics["acc"]:.4f} F1={metrics["f1"]:.4f} AUC={metrics["auc"]:.4f}')

            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ===== 最终结果 =====
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)

    print(f'\n{"="*50}')
    print(f'Best Test Accuracy:  {best_acc:.4f}')
    print(f'Best Test F1 (macro): {final_metrics["f1"]:.4f}')
    print(f'Best Test AUC:       {final_metrics["auc"]:.4f}')
    print(f'Confusion Matrix:\n{final_metrics["cm"]}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
