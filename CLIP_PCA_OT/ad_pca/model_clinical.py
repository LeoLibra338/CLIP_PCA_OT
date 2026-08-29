"""
纯临床特征 + PCA 正则化分类
验证 PCA 在小样本多模态上的核心效果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalPCA(nn.Module):
    """临床特征 → MLP → PCA 去噪 → 分类"""

    def __init__(self, input_dim=9, hidden_dim=128, num_classes=3,
                 pca_ratio=0.85, pca_weight=0.5):
        super().__init__()
        self.pca_ratio = pca_ratio
        self.pca_weight = pca_weight

        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _batch_pca(self, x, ratio=0.85):
        mean = x.mean(dim=0, keepdim=True)
        X = x - mean
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        s_sq = S ** 2
        total_var = s_sq.sum()
        if total_var < 1e-8:
            return x
        cum_ratio = torch.cumsum(s_sq, dim=0) / total_var
        valid = torch.where(cum_ratio >= ratio)[0]
        k = valid[0].item() + 1 if valid.numel() > 0 else len(S)
        k = min(max(1, k), len(S))
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        return U_k @ torch.diag(S_k) @ Vt_k + mean

    def forward(self, x, return_features=False):
        """
        x: [B, input_dim]  临床特征
        """
        f = self.encoder(x)           # [B, hidden_dim]
        f = F.normalize(f, dim=-1)

        # PCA 去噪
        f_pca = self._batch_pca(f, ratio=self.pca_ratio)
        residual = f - f_pca

        # 分类
        logits = self.classifier(f_pca)

        # PCA 正则：重建损失 + 残差不应该太大
        loss_recon = F.mse_loss(f_pca, f)
        # 残差熵正则：鼓励残差在各个样本间均匀（不过度集中在某一样本）
        residual_norm = residual.norm(dim=-1)
        loss_residual = residual_norm.std()  # 残差方差太大 → 某些样本被 PCA 丢太多

        if return_features:
            return logits, loss_recon, loss_residual, f, f_pca

        return logits, loss_recon, loss_residual

    def compute_loss(self, x, labels):
        logits, loss_recon, loss_residual = self.forward(x)
        ce = F.cross_entropy(logits, labels)
        pca_reg = (loss_recon + loss_residual) * self.pca_weight
        return ce + pca_reg, ce, pca_reg
