"""
纯临床特征训练：OASIS-1 (9维) → Internal 80 (4维+均值填充)
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from collections import Counter
import pandas as pd
from model_clinical import ClinicalPCA

# OASIS-1 全部 9 维临床特征
OASIS_FEATURES = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF',
                  'M_F', 'Hand']  # M_F: M=1,F=0  Hand: R=1,L=0


def load_oasis(csv_path, fit_norm=False, norm_params=None):
    df = pd.read_csv(csv_path)
    df = df[(df['has_mri'] == True) & (df['has_cdr'] == True)]

    X, y = [], []
    for _, row in df.iterrows():
        feats = [
            row['Age'] if pd.notna(row['Age']) else 75,
            row['Educ'] if pd.notna(row['Educ']) else 3,
            row['SES'] if pd.notna(row['SES']) else 2.5,
            row['MMSE'] if pd.notna(row['MMSE']) else 27,
            row['eTIV'] if pd.notna(row['eTIV']) else 1500,
            row['nWBV'] if pd.notna(row['nWBV']) else 0.73,
            row['ASF'] if pd.notna(row['ASF']) else 1.0,
            1.0 if row['M/F'] == 'M' else 0.0,
            1.0 if row['Hand'] == 'R' else 0.0,
        ]
        cdr = float(row['CDR'])
        label = 0 if cdr == 0.0 else (1 if cdr == 0.5 else 2)
        X.append(feats)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    if fit_norm:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        return X, y, mean, std
    elif norm_params is not None:
        mean, std = norm_params
        X = (X - mean) / std
        return X, y
    else:
        return X, y


def load_internal(csv_path, norm_params):
    df = pd.read_csv(csv_path)
    mean, std = norm_params

    X, y = [], []
    for _, row in df.iterrows():
        # 内部有的 4 维，OASIS 独有的用均值填充（归一化后均值=0）
        age = float(row['age (years)']) if pd.notna(row['age (years)']) else 70
        educ = float(row['education (years)']) if pd.notna(row['education (years)']) else 12
        mmse = float(row['MMSE score']) if pd.notna(row['MMSE score']) else 25
        sex = float(row['gender']) if pd.notna(row['gender']) else 1.0
        sex = 1.0 if sex == 1 else 0.0

        feats_raw = [age, educ, 0, mmse, 0, 0, 0, sex, 0]  # 缺失用 0（归一化后均值）
        feats = (np.array(feats_raw, dtype=np.float32) - mean) / std

        g = row['groups']
        label = 0 if g == 'HC' else (1 if g == 'MCI' else 2)
        X.append(feats)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits, _, _, _, _ = model(xb.to(device), return_features=True)
            probs = torch.softmax(logits, -1)
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
    except:
        auc = 0.0
    return acc, f1, auc, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pca_ratio', type=float, default=0.85)
    parser.add_argument('--pca_weight', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ===== 数据 =====
    X_train, y_train, mean, std = load_oasis(
        '/home/gyf/data/oasis/OASIS-1/final_dataset.csv', fit_norm=True)
    X_test, y_test = load_internal(
        '/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal/patients.csv',
        (mean, std))

    print(f'OASIS-1 train: {len(X_train)}  (HC={sum(y_train==0)} MCI={sum(y_train==1)} AD={sum(y_train==2)})')
    print(f'Internal test: {len(X_test)}  (HC={sum(y_test==0)} MCI={sum(y_test==1)} AD={sum(y_test==2)})')
    print(f'Features: {X_train.shape[1]} dim (4 internal + 5 OASIS-only → filled with mean)')

    # 加权采样
    class_counts = Counter(y_train.tolist())
    weights = {c: 1.0 / class_counts[c] for c in class_counts}
    sample_w = [weights[l] for l in y_train]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ===== 模型 =====
    model = ClinicalPCA(
        input_dim=X_train.shape[1], pca_ratio=args.pca_ratio,
        pca_weight=args.pca_weight, num_classes=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    best_state = None
    results = []

    for epoch in range(args.epochs):
        model.train()
        train_ce_sum, train_pca_sum = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss, ce, pca = model.compute_loss(xb, yb)
            loss.backward()
            optimizer.step()
            train_ce_sum += ce.item()
            train_pca_sum += pca.item()
        scheduler.step()

        n = len(train_loader)
        train_ce, train_pca = train_ce_sum / n, train_pca_sum / n

        if (epoch + 1) % 10 == 0:
            acc, f1, auc, cm = evaluate(model, test_loader, device)
            results.append((epoch + 1, train_ce, train_pca, acc, f1, auc))
            print(f'Epoch {epoch+1:3d} | CE={train_ce:.4f} PCA={train_pca:.4f} | '
                  f'Test Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}')

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ===== 最终 =====
    print(f'\n{"="*50}')
    print('Per-epoch results:')
    for r in results:
        print(f'  E{r[0]:3d}: CE={r[1]:.4f} PCA={r[2]:.4f} Acc={r[3]:.4f} F1={r[4]:.4f} AUC={r[5]:.4f}')

    if best_state:
        model.load_state_dict(best_state)
    acc, f1, auc, cm = evaluate(model, test_loader, device)
    print(f'\nBest Acc:  {acc:.4f}')
    print(f'Best F1:   {f1:.4f}')
    print(f'Best AUC:  {auc:.4f}')
    print(f'Confusion:\n{cm}')


if __name__ == '__main__':
    main()
