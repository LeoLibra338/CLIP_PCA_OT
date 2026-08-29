"""
PCA-Regularized Multimodal Model for AD Diagnosis
T1 MRI (3D ResNet) + Clinical features (MLP) → PCA alignment → classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Small3DCNN(nn.Module):
    """轻量 3D CNN：参数量 ~50K，适合 200+ 样本"""
    def __init__(self, in_channels=1, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            # 96³ → 48³
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            # 48³ → 24³
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            # 24³ → 12³
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            # 12³ → 6³
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            # 6³ → 3³
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # → (B, 256, 1, 1, 1)
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


class ClinicalEncoder(nn.Module):
    """临床特征编码器：4维 → 64维"""
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class PCACrossModal(nn.Module):
    """PCA 跨模态对齐 + 分类"""

    def __init__(self, img_dim=512, clinical_dim=64, num_classes=3,
                 pca_ratio=0.85, pca_weight=0.1):
        super().__init__()
        self.pca_ratio = pca_ratio
        self.pca_weight = pca_weight
        self.img_dim = img_dim
        self.clinical_dim = clinical_dim

        # 影像编码器（轻量 3D CNN，~50K 参数 vs ResNet ~5M）
        self.img_encoder = Small3DCNN(in_channels=1, out_dim=img_dim)

        # 临床编码器
        self.clinical_encoder = ClinicalEncoder(
            input_dim=4, hidden_dim=clinical_dim
        )

        # 分类头（PCA 融合后）
        joint_dim = img_dim + clinical_dim
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _batch_pca(self, x, ratio=0.85):
        """Batch PCA：保留 ratio 比例方差的低秩重建"""
        # x: [B, D], centered
        mean = x.mean(dim=0, keepdim=True)
        X = x - mean

        # SVD
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)

        # 累计方差比
        s_sq = S ** 2
        total_var = s_sq.sum()
        if total_var < 1e-8:
            return x  # 退化情况

        cum_ratio = torch.cumsum(s_sq, dim=0) / total_var
        valid = torch.where(cum_ratio >= ratio)[0]
        k = valid[0].item() + 1 if valid.numel() > 0 else len(S)
        k = min(max(1, k), len(S))

        # 低秩重建
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        X_recon = U_k @ torch.diag(S_k) @ Vt_k
        X_recon += mean

        return X_recon

    def forward(self, t1, clinical, return_features=False):
        """
        t1: [B, 1, 96, 96, 96]
        clinical: [B, 4]
        """
        # 编码
        f_img = self.img_encoder(t1)       # [B, 512]
        f_cli = self.clinical_encoder(clinical)  # [B, 64]

        # 归一化
        f_img = F.normalize(f_img, dim=-1)
        f_cli = F.normalize(f_cli, dim=-1)

        # === PCA 分别对两个模态做（关键：B<<D，PCA 真压缩） ===
        f_img_pca = self._batch_pca(f_img, ratio=self.pca_ratio)      # [B, 512]
        f_cli_pca = self._batch_pca(f_cli, ratio=self.pca_ratio)      # [B, 64]

        # PCA 残差（被 PCA 去掉的细节）
        residual_img = f_img - f_img_pca    # [B, 512]
        residual_cli = f_cli - f_cli_pca    # [B, 64]

        # 联合 PCA 特征做分类
        f_joint = torch.cat([f_img_pca, f_cli_pca], dim=-1)  # [B, 576]
        logits = self.classifier(f_joint)

        # PCA 正则化损失
        # 1. 重建损失：PCA 不应该丢掉太多信息
        loss_recon_img = F.mse_loss(f_img_pca, f_img)
        loss_recon_cli = F.mse_loss(f_cli_pca, f_cli)

        # 2. 跨模态一致性：同一病人的影像残差和临床残差应该相关
        #    残差大小反映"这个病人有多不典型"
        residual_norm_img = residual_img.norm(dim=-1)      # [B]
        residual_norm_cli = residual_cli.norm(dim=-1)      # [B]
        #    两个模态的残差范数应该正相关（Pearson correlation → 用 cosine）
        loss_cross = 1.0 - F.cosine_similarity(
            residual_norm_img.unsqueeze(0), residual_norm_cli.unsqueeze(0), dim=-1
        ).mean()

        if return_features:
            return logits, loss_recon_img, loss_recon_cli, loss_cross, f_joint, f_img

        return logits, loss_recon_img, loss_recon_cli, loss_cross

    def compute_loss(self, t1, clinical, labels):
        """完整训练损失：CE + PCA 重建 + 跨模态一致性"""
        logits, loss_img, loss_cli, loss_cross = self.forward(t1, clinical)

        ce_loss = F.cross_entropy(logits, labels)

        pca_reg = (loss_img + loss_cli + loss_cross) * self.pca_weight
        total_loss = ce_loss + pca_reg
        return total_loss, ce_loss, pca_reg
