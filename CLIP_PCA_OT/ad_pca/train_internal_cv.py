"""
内部 80 样本 5-fold CV：验证 PCA 正则化在小样本上的效果
对比：with PCA vs without PCA (pca_weight=0)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd


def load_internal(csv_path, nii_dir):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        import os
        name = row['Name']
        if not os.path.exists(os.path.join(nii_dir, f'{name}.nii')):
            continue
        feats = [
            float(row['age (years)']) if pd.notna(row['age (years)']) else 70,
            float(row['education (years)']) if pd.notna(row['education (years)']) else 12,
            float(row['MMSE score']) if pd.notna(row['MMSE score']) else 25,
            1.0 if float(row['gender']) == 1 else 0.0,
        ]
        g = row['groups']
        label = 0 if g == 'HC' else (1 if g == 'MCI' else 2)
        X.append(feats)
        y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # 归一化
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=3, use_pca=True, pca_ratio=0.85):
        super().__init__()
        self.use_pca = use_pca
        self.pca_ratio = pca_ratio
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _batch_pca(self, x, ratio=0.85):
        mean = x.mean(dim=0, keepdim=True)
        X = x - mean
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        s_sq = S ** 2
        total = s_sq.sum()
        if total < 1e-8:
            return x
        cum = torch.cumsum(s_sq, dim=0) / total
        valid = torch.where(cum >= ratio)[0]
        k = valid[0].item() + 1 if valid.numel() > 0 else len(S)
        k = min(max(1, k), len(S))
        return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :] + mean

    def forward(self, x):
        f = self.encoder(x)
        if self.use_pca:
            f_pca = self._batch_pca(f, self.pca_ratio)
            return self.classifier(f_pca), F.mse_loss(f_pca, f)
        return self.classifier(f), torch.tensor(0.0)

    def compute_loss(self, x, labels, pca_weight=0.5):
        logits, pca_loss = self.forward(x)
        ce = F.cross_entropy(logits, labels)
        return ce + pca_loss * pca_weight, ce, pca_loss * pca_weight


def train_fold(X_train, y_train, X_val, y_val, use_pca=True, pca_weight=0.5, epochs=200):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP(use_pca=use_pca).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.compute_loss(xb, yb, pca_weight)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits, _ = model(xb.to(device))
                    all_preds.extend(logits.argmax(-1).cpu().numpy())
                    all_labels.extend(yb.numpy())
            acc = accuracy_score(all_labels, all_preds)
            if acc > best_acc:
                best_acc = acc

    return best_acc


# ===== Main =====
X, y = load_internal(
    '/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal/patients.csv',
    '/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal'
)
print(f'Loaded: {len(X)} samples, {X.shape[1]} features')
print(f'Distribution: HC={sum(y==0)} MCI={sum(y==1)} AD={sum(y==2)}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 对比实验
for pca_w in [0.0, 0.1, 0.5, 1.0]:
    accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        acc = train_fold(X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                         use_pca=(pca_w > 0), pca_weight=pca_w)
        accs.append(acc)
    use_pca = pca_w > 0
    print(f'  PCA={"✅" if use_pca else "❌"} w={pca_w:.1f}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  ({accs})')
