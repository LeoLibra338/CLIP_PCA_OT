"""
AD Diagnosis Dataset: OASIS-1 (train) + Internal 80 (test)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom


CLINICAL_FEATURES = ['age', 'educ', 'mmse', 'sex']  # 共享的 4 个临床特征


def load_t1_volume(path, target_size=96):
    """加载 T1 MRI volume，统一 resize 到 (96, 96, 96)"""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # 去掉多余的尾部单维度 (256,256,128,1) → (256,256,128)
    while data.ndim > 3:
        data = data.squeeze(-1)

    # 归一化到 [0, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Resize 到统一尺寸
    factors = (target_size / data.shape[0],
               target_size / data.shape[1],
               target_size / data.shape[2])
    data = zoom(data, factors, order=1)

    # 加 channel 维度 → (1, 96, 96, 96)
    return torch.from_numpy(data.copy()).unsqueeze(0)


class OASIS1Dataset(Dataset):
    """OASIS-1 训练集：T1 MRI + 临床特征 + CDR 标签"""

    def __init__(self, csv_path, mri_root):
        df = pd.read_csv(csv_path)
        # 只保留有 MRI 且有 CDR 的
        self.df = df[(df['has_mri'] == True) & (df['has_cdr'] == True)].copy()
        self.mri_root = mri_root
        self.subjects = sorted(self.df['ID'].unique())

        # 临床特征归一化参数（在 fit_normalizer 中计算）
        self.clinical_mean = None
        self.clinical_std = None

    def fit_normalizer(self):
        """计算临床特征归一化参数"""
        values = []
        for _, row in self.df.iterrows():
            feats = self._get_clinical(row)
            values.append(feats)
        values = np.array(values)
        self.clinical_mean = values.mean(axis=0)
        self.clinical_std = values.std(axis=0) + 1e-8

    def _get_clinical(self, row):
        """提取 4 个共享临床特征，缺失值填均值"""
        age = row['Age'] if pd.notna(row['Age']) else 75
        educ = row['Educ'] if pd.notna(row['Educ']) else 14
        mmse = row['MMSE'] if pd.notna(row['MMSE']) else 27
        sex = 1.0 if row['M/F'] == 'M' else 0.0
        return np.array([age, educ, mmse, sex], dtype=np.float32)

    def _get_label(self, row):
        """CDR → 0=HC, 1=MCI, 2=AD"""
        cdr = float(row['CDR'])
        if cdr == 0.0:
            return 0
        elif cdr == 0.5:
            return 1
        else:
            return 2

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        row = self.df[self.df['ID'] == subj].iloc[0]

        # T1 MRI
        mpr_path = os.path.join(self.mri_root, subj, 'RAW',
                                f'{subj}_mpr-1_anon.img')
        # fallback: 有些受试者用不同命名
        if not os.path.exists(mpr_path):
            import glob
            candidates = glob.glob(os.path.join(
                self.mri_root, subj, 'RAW', '*mpr-1*anon*'))
            if candidates:
                mpr_path = candidates[0]
            else:
                # 用任何 mpr 文件
                any_mpr = glob.glob(os.path.join(
                    self.mri_root, subj, 'RAW', '*mpr*anon*.img'))
                mpr_path = any_mpr[0] if any_mpr else None

        t1 = load_t1_volume(mpr_path) if mpr_path else torch.zeros(1, 128, 128, 128)

        # 临床特征（归一化）
        clinical = self._get_clinical(row)
        if self.clinical_mean is not None:
            clinical = (clinical - self.clinical_mean) / self.clinical_std
        clinical = torch.from_numpy(clinical)

        label = self._get_label(row)
        return t1, clinical, label


class InternalDataset(Dataset):
    """内部 80 样本测试集"""

    def __init__(self, nii_dir, csv_path, clinical_mean=None, clinical_std=None):
        self.df = pd.read_csv(csv_path)
        self.nii_dir = nii_dir
        self.clinical_mean = clinical_mean
        self.clinical_std = clinical_std
        self.subjects = []

        # 匹配 CSV 和 nii 文件
        for _, row in self.df.iterrows():
            name = row['Name']  # e.g., Sub011
            nii_path = os.path.join(nii_dir, f'{name}.nii')
            if os.path.exists(nii_path):
                self.subjects.append(row)

    def _get_clinical(self, row):
        age = float(row['age (years)']) if pd.notna(row['age (years)']) else 70
        educ = float(row['education (years)']) if pd.notna(row['education (years)']) else 12
        mmse = float(row['MMSE score']) if pd.notna(row['MMSE score']) else 25
        sex = float(row['gender']) if pd.notna(row['gender']) else 1.0
        sex = 1.0 if sex == 1 else 0.0  # M=1, F=0
        return np.array([age, educ, mmse, sex], dtype=np.float32)

    def _get_label(self, row):
        g = row['groups']
        if g == 'HC':
            return 0
        elif g == 'MCI':
            return 1
        else:
            return 2

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        row = self.subjects[idx]
        name = row['Name']
        nii_path = os.path.join(self.nii_dir, f'{name}.nii')

        t1 = load_t1_volume(nii_path)
        clinical_raw = self._get_clinical(row)
        clinical = clinical_raw.copy()
        if self.clinical_mean is not None:
            clinical = (clinical - self.clinical_mean) / self.clinical_std
        clinical = torch.from_numpy(clinical)

        label = self._get_label(row)
        return t1, clinical, label, clinical_raw  # 返回原始临床值用于后续
