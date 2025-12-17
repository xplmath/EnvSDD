# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:59:54 2025

@author: lihs
"""
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt

import STAGATE

import sys
module_path = 'D:/Experiment/nyq/GCN_STAGATE_tf'
if module_path not in sys.path:
    sys.path.append(module_path)
from Train_EnvSDD import train_STAGATE
from utils import run_leiden
# import torch

import os
os.environ['R_HOME'] = 'D:/Users/lihs/anaconda3/envs/Community_domain/Lib/R'
os.environ['R_USER'] = 'D:/Users/lihs/anaconda3/envs/Community_domain/Lib/site-packages/ryp2'
# from utile import mclust_R

BASE = Path(r"D:/Experiment/nyq/DLPFC/dataset")
from pathlib import Path
OUT  = Path('D:/Experiment/nyq/DLPFC/results')
OUT.mkdir(exist_ok=True)

# ===== 找到所有样本文件夹 =====
sample_dirs = sorted([p for p in BASE.iterdir() if p.is_dir() and p.name.startswith("DLPFC")])
print("Found samples:", [d.name for d in sample_dirs])

# fusion_lambda_all = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
fusion_lambda_all = [0.1, 0]

for fusion_lambda in fusion_lambda_all:
    print(f"\n===== [Fusion Lambda = {fusion_lambda}] =====")

    # 为当前 fusion_lambda 创建总文件夹
    fusion_dir = OUT / f"fusion_{fusion_lambda:.1f}"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir in sample_dirs:
        name = sample_dir.name
        print(f"\n--- Processing sample: {name} ---")

        # ---- 1) 路径 ----
        counts_path = sample_dir / "counts_matrix.txt"
        meta_path   = sample_dir / "meta_data.txt"
        ct_path     = sample_dir / "rctd_res.txt"

        # ---- 2) 读取表达矩阵与元数据 ----
        adata = sc.read_text(str(counts_path)).T
        adata.var_names_make_unique()
        meta_data = pd.read_csv(meta_path, sep="\t")
        adata.obs = meta_data

        # ---- 3) 读取细胞类型丰度，并设置索引列 ----
        ct_abundance = pd.read_csv(ct_path, sep="\t")
        id_col = "Unnamed: 0" if "Unnamed: 0" in ct_abundance.columns else ct_abundance.columns[0]
        ct_abundance.index = ct_abundance[id_col].astype(str)
        ct_abundance = ct_abundance.drop(columns=[id_col])

        # ---- 4) 对齐行（交集，按 adata 顺序）----
        common = adata.obs_names.intersection(ct_abundance.index)
        adata = adata[common].copy()
        ct_abundance = ct_abundance.loc[common].copy()
        adata.obsm['microenv_prop'] = ct_abundance

        # ---- 5) 写入空间坐标 ----
        if {"pixel_x","pixel_y"}.issubset(adata.obs.columns):
            adata.obsm['spatial'] = np.asarray(adata.obs[['pixel_y','pixel_x']], dtype=float)
        else:
            raise KeyError(f"{name}: meta_data.txt 缺少空间坐标列")

        # ---- 6) 数据预处理 ----
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        from utils import Cal_Spatial_Net
        Cal_Spatial_Net(adata, rad_cutoff=150)

        # ---- 7) 训练模型 ----
        print(f"   >> Training STAGATE with fusion_lambda={fusion_lambda}")
        adata_tmp = train_STAGATE(adata.copy(), alpha=0, fusion_lambda=fusion_lambda)

        # ---- 8) 输出路径 ----
        out_dir = fusion_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        z_path = out_dir / f"{name}_Z.txt"
        z_df = pd.DataFrame(
            data=adata_tmp.obsm['STAGATE'],
            index=adata_tmp.obs_names.tolist(),
            columns=[f'L{i}' for i in range(30)]
        )
        z_df.to_csv(z_path, sep="\t", index=True)

        print(f"[SAVED] {name} -> {z_path}")






