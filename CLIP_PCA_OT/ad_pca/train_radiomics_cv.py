"""
T1 体素特征 + 临床特征 → PCA 降维 → 分类
80 样本，96³体素 ≈ 88万维 → PCA → 几十维 → MLP
这才是小样本 PCA 该有的场景
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom


def load_t1_features(nii_dir, patients_csv, target_size=48):
    """从 T1 提取特征：降采样体素 + 简单统计特征"""
    df = pd.read_csv(patients_csv)

    voxel_feats = []
    stat_feats = []
    clinical_feats = []
    labels = []

    for _, row in df.iterrows():
        name = row['Name']
        nii_path = os.path.join(nii_dir, f'{name}.nii')
        if not os.path.exists(nii_path):
            continue

        # 加载 T1
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)
        while data.ndim > 3:
            data = data.squeeze(-1)

        # 归一化
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # 降采样到 target_size
        factors = (target_size / data.shape[0], target_size / data.shape[1],
                   target_size / data.shape[2])
        data_small = zoom(data, factors, order=1)

        # 特征 1: 降采样体素（展平）
        voxel_feats.append(data_small.flatten())

        # 特征 2: 统计特征
        stats = [
            data.mean(), data.std(),
            np.percentile(data, 10), np.percentile(data, 25),
            np.percentile(data, 50), np.percentile(data, 75),
            np.percentile(data, 90),
            (data > 0.3).mean(), (data > 0.5).mean(), (data > 0.7).mean(),
            ((data - data.mean()) ** 3).mean() / (data.std() ** 3 + 1e-8),  # skew
            ((data - data.mean()) ** 4).mean() / (data.std() ** 4 + 1e-8),  # kurtosis
        ]
        stat_feats.append(stats)

        # 特征 3: 临床
        clinical = [
            float(row['age (years)']) if pd.notna(row['age (years)']) else 70,
            float(row['education (years)']) if pd.notna(row['education (years)']) else 12,
            float(row['MMSE score']) if pd.notna(row['MMSE score']) else 25,
            1.0 if float(row['gender']) == 1 else 0.0,
        ]
        clinical_feats.append(clinical)

        g = row['groups']
        labels.append(0 if g == 'HC' else (1 if g == 'MCI' else 2))

    return (np.array(voxel_feats, dtype=np.float32),
            np.array(stat_feats, dtype=np.float32),
            np.array(clinical_feats, dtype=np.float32),
            np.array(labels, dtype=np.int64))


class PCAClassifier(nn.Module):
    """PCA 降维 + 小 MLP 分类"""
    def __init__(self, input_dim, num_classes=3, pca_dim=32):
        super().__init__()
        self.fc = nn.Linear(input_dim, pca_dim)
        self.classifier = nn.Sequential(
            nn.Linear(pca_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)  # 线性降维（相当于学一个 PCA 投影）
        return self.classifier(x), x


def run_cv(X, y, name, epochs=100):
    """5-fold CV"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr = torch.from_numpy(X[train_idx]).to(device)
        y_tr = torch.from_numpy(y[train_idx]).to(device)
        X_te = torch.from_numpy(X[test_idx]).to(device)
        y_te = torch.from_numpy(y[test_idx])

        model = PCAClassifier(input_dim=X.shape[1], pca_dim=min(32, len(train_idx)))
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            logits, _ = model(X_tr)
            loss = F.cross_entropy(logits, y_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(X_te)
            preds = logits.argmax(-1).cpu().numpy()
            accs.append(accuracy_score(y_te, preds))
            f1s.append(f1_score(y_te, preds, average='macro'))

    print(f'  {name}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  ({[f"{a:.3f}" for a in accs]})')
    return np.mean(accs)


# ===== Main =====
print("Loading T1 features...")
voxel, stats, clinical, labels = load_t1_features(
    '/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal',
    '/home/gyf/CLIP-wgd/HiMo-CLIP/ad_pca/data/internal/patients.csv',
    target_size=48
)
print(f"Loaded: {len(labels)} samples, {len(labels[labels==0])} HC, {len(labels[labels==1])} MCI, {len(labels[labels==2])} AD")
print(f"Voxel dim: {voxel.shape[1]:,} ({48}³)")
print(f"Stats dim: {stats.shape[1]}")
print()

# 1. 纯临床基线
print("=== 5-fold CV ===")
run_cv(clinical, labels, "Clinical only (4D)")

# 2. 统计特征
run_cv(stats, labels, "T1 Stats (12D)")

# 3. 临床 + 统计
run_cv(np.hstack([clinical, stats]), labels, "Clinical+T1Stats (16D)")

# 4. PCA 降维的体素特征（offline PCA）
for n_comp in [16, 32, 64]:
    pca = PCA(n_components=min(n_comp, len(labels)))
    voxel_pca = pca.fit_transform(voxel).astype(np.float32)
    run_cv(voxel_pca, labels, f"T1 Voxel PCA-{n_comp}D")

# 5. 体素 PCA + 临床
for n_comp in [16, 32]:
    pca = PCA(n_components=min(n_comp, len(labels)))
    voxel_pca = pca.fit_transform(voxel).astype(np.float32)
    combined = np.hstack([voxel_pca, clinical, stats])
    run_cv(combined, labels, f"Voxel PCA-{n_comp}D + Clinical + Stats ({combined.shape[1]}D)")
