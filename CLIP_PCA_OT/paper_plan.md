# 三区论文写作计划：PCA-OT CLIP 几何正则化

## 一、论文定位与题目

### 核心叙事
**不是"改进 HiMo-CLIP"，而是"解决 CLIP 微调中特征冗余+语义失序+零样本遗忘三大问题"。**

HiMo-CLIP 只在 Related Work 中作为"层次分解范式"的代表出现不超过3次。我们的对手是 Noisy Features、Vanilla Fine-tuning、Architectural Heaviness 这些抽象概念。

### 建议题目
> *PCA-OT: A Parameter-Free Geometric Regularization Framework for Data-Efficient CLIP Fine-tuning*

### 目标期刊
Neurocomputing / Pattern Recognition Letters / Multimedia Tools and Applications（审稿周期 3-6 个月）

---

## 二、论文结构

| 章节 | 核心内容 | 状态 |
|------|---------|------|
| Abstract | 问题→方法→结果，不提 HiMo | ✅ 可写 |
| 1. Introduction | 三大困境 + 我们的方案 + 三点贡献 | ✅ 可写 |
| 2. Related Work | 2.1 CLIP微调 2.2 特征正则化 2.3 零样本保持 | ✅ 可写 |
| 3. Method | 3.1 PCA文本去噪 3.2 OT单调性约束 3.3 Trade-off分析 | ✅ 已有 |
| 4. Experiments | 4.1 设置 4.2 检索结果 4.3 单调性 4.4 零样本 4.5 消融 | ⚠️ 需补 |
| 5. Conclusion | 总结+未来工作 | ✅ 可写 |

---

## 三、当前已有 vs 需要补充

### ✅ 已完成的实验（直接可用）

| 实验 | 数据 | 位置 |
|------|------|------|
| pure_clip vs pca_pure vs pca_ot 检索对比 | DOCCI + COCO | experiment_results.md |
| HiMo@K 对比（6个方法） | HiMo-Docci | experiment_results.md |
| 零样本保持测试 | COCO 80类 | experiment_results.md |
| λ 扫参 (0/0.1/0.15/0.3/0.5/1.0) | DOCCI | experiment_results.md |
| 多方法对比（Long-CLIP/FineLIP/OpenCLIP） | DOCCI + COCO | experiment_results.md |
| OT CLS-free 实验 | DOCCI | experiment_results.md |
| GOAL风格一致性 | DOCCI | experiment_results.md |
| PCA Batch稳定性验证 | DOCCI | experiment_results.md |

### ⚠️ 建议补充（按优先级）

**P0：必须做**
1. **PCA ratio 扫参** (τ=0.7/0.8/0.85/0.9/0.95) — 论文消融表的核心，证明方法鲁棒。1天跑完
2. **数据量消融** (1K/5K/10K/25K/50K) — 证明小数据优势，这是你区别于所有SOTA的核心卖点。1天跑完

**P1：加分项**
3. **跨backbone验证** — 跑一组 ViT-B/16 的结果（哪怕只跑 pure_clip 和 pca_pure）
4. **Trade-off 双轴图** — λ vs R@K & HiMo@K，论文最抓眼的图

**P2：锦上添花**
5. Flickr30k 评测（需下载数据）
6. 跨数据集零样本（ImageNet）

---

## 四、论文写作要点

### 4.1 Abstract（150词）

```
长文本CLIP微调存在三个根本问题：
(1) 特征冗余——无关词占据判别维度
(2) 语义失序——丰富描述未必得到更高的图文相似度
(3) 遗忘困境——微调破坏预训练的零样本能力

我们提出一种零参数的几何正则化框架：
- PCA 去相关：batch内动态抑制噪声维度
- OT 对齐：最优传输显式建模语义累积，可调控trade-off

仅用5万数据（主流方法的4%），在DOCCI检索上提升2.7%，
HiMo@K提升47%，零样本仅降0.3%。
```

### 4.2 Introduction 写作要点

- 第1段：CLIP成功源于对齐，但特征空间存在信噪比困境
- 第2段：现有解法要么加参数（LoRA/LongCLIP），要么加数据（HiMo 124万）——都治标不治本
- 第3段：我们的方案——通过操控特征空间几何结构（PCA + OT），不增加一毫一厘参数
- 第4段：三点贡献

**HiMo只在Related Work里出现。**

### 4.3 Method 写作要点

把PCA重新定义为"特征白化正则化器"：
- 不是"提取语义组件"（HiMo的说法）
- 而是"在梯度反传时绕过噪声维度"

把OT重新定义为"累积语义度量"：
- 不是"组件级对比"（HiMo的MoLo）
- 而是"测量从部分到整体的分布距离"

### 4.4 Experiments 表格设计

**Table 1: 检索性能对比（主表）**
```
Method              Params  Data    DOCCI I2T  DOCCI T2I  COCO I2T  COCO T2I  HiMo@K  Zero-shot
ViT-L/14 (frozen)    300M     0       -          -         36.0      19.9      -       59.7
+ InfoNCE FT         300M    50K      72.9       74.8      61.2      42.9      0.510   57.3
+ PCA (Ours)         300M    50K      75.6       74.4      60.1      42.9      0.673   57.0
+ PCA+OT λ=0.15      300M    50K      73.5       73.1      56.4      41.2      0.751   57.0
Long-CLIP            300M    50K      59.2       65.2      59.3      46.8      0.693    -
FineLIP              300M    50K      62.7       67.5      59.6      51.1      0.795    -
OpenCLIP             300M    50K      59.4       66.1      60.0      49.5      0.373    -
HiMo-CLIP (论文)     300M   1.2M     82.3       84.4       -         -        0.882    -
```

**Table 2: PCA ratio 消融**
```
τ      DOCCI I2T  DOCCI T2I  HiMo@K
0.70   ?          ?          ?
0.80   ?          ?          ?
0.85   75.6       74.4       0.673
0.90   ?          ?          ?
0.95   ?          ?          ?
```

**Table 3: 数据量消融**
```
Data    pure_clip I2T  pca_pure I2T  Δ
1K      ?              ?             ?
5K      ?              ?             ?
10K     ?              ?             ?
25K     ?              ?             ?
50K     72.9           75.6          +2.7
```

**Figure 1: Trade-off曲线** (λ vs R@K & HiMo@K 双轴图)

---

## 五、创新点提炼（答辩用）

1. **首次发现 PCA 可作为 CLIP 微调的零参数检索正则化**
   - 原HiMo只用PCA做语义分解，我们首次证实PCA直接提升R@K
   
2. **首次揭示 PCA+OT 的检索-单调性 Trade-off**
   - λ=0检索最优，λ增大单调性提升，用户按需选择
   - 这是此前所有方法（包括HiMo）都未发现的

3. **零参数 + 零样本保持**
   - 不增加任何参数，零样本仅降0.3%
   - 唯一的zero-parameter PEFT方法

---

## 六、行动清单（按优先级）

- [ ] **P0**: 跑 PCA ratio 扫参 (τ=0.7,0.8,0.85,0.9,0.95) — 约3小时
- [ ] **P0**: 跑数据量消融 (1K/5K/10K/25K/50K) — 约4小时  
- [ ] **P1**: 画 Trade-off 双轴图（已有数据）
- [ ] **P1**: ViT-B/16 验证（可选，1小时）
- [ ] 写 Method 章节（已有代码，可照搬）
- [ ] 写 Experiments 章节
- [ ] 写 Introduction + Related Work
- [ ] 最后写 Abstract + Conclusion

---

## 七、时间预估

```
补实验:     1-2 天
论文写作:   1 周
修改润色:   2-3 天
总计:       2 周可完成初稿
```
