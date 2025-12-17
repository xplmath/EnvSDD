# main_train.py
# 用 Dual-Head GAT-AE 进行空间域表征学习：X -> Z，并在图上重构 X
# 依赖：scanpy/anndata（可选）、torch, torch_geometric, torch_scatter, sklearn, scipy

import argparse
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import scanpy as sc

# === 1) 你自己的模块 ===
from network_multi_head import (
    DualHeadSTAGATE_AE,
    train_model,
    normalize_beta_by_dst,   # 如果想沿用其中的归一化
)  # ← 模型 & 训练（三损失）  :contentReference[oaicite:2]{index=2}

from utile import (
    cal_K_neighboorhood,     # KNN 构图（返回稀疏邻接/距离）
    cell_type_abundance,     # 计算邻域的细胞类型比例
)  # ← 图/邻域工具函数        :contentReference[oaicite:3]{index=3}

# === 2) 实用函数 ===
def ad_sparse_to_torch_dense(X):
    """把 AnnData.X（可能是稀疏）安全转成 torch.float32 的 dense 张量。"""
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32)
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=np.float32)
    return torch.from_numpy(X)

def adj_to_edge_index(A):
    """稀疏邻接 -> (2, E) edge_index。默认保留非对角非零，方向为 i->j 与 j->i 都会保留（取决于 A 是否对称）"""
    A = A.tocoo()
    mask = A.row != A.col
    src = A.row[mask].astype(np.int64)
    dst = A.col[mask].astype(np.int64)
    ei = np.vstack([src, dst])
    return torch.from_numpy(ei)

def cosine_beta_on_edges(props: np.ndarray, edge_index: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    在给定边上计算细胞类型比例的 cosine 相似度。
    props: (N, C) 每行和=1 的比例（cell_type_abundance 已经做了行归一）
    edge_index: (2, E)
    返回：beta (E,) ∈ [0,1]
    """
    src, dst = edge_index.numpy()
    Psrc = props[src]
    Pdst = props[dst]
    num = np.sum(Psrc * Pdst, axis=1)
    den = np.linalg.norm(Psrc, axis=1) * np.linalg.norm(Pdst, axis=1) + eps
    beta = (num / den).clip(0.0, 1.0).astype(np.float32)
    return torch.from_numpy(beta)

# === 3) 主流程 ===
def run_training(
    adata,
    coords_key: str,
    resolution: str, # 'low' or 'high'
    ct_key = None,
    ct_prop_df = None, 
    k: int = 8,
    in_dim: int | None = None,
    hidden1: int = 512,
    z_dim: int = 30,
    hidden2: int = 512,
    lambda_init: float = 0.5,
    learnable_lambda: bool = True,
    dropout: float = 0.0,
    act: str = "elu",
    last_act: str = "none",
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lam_rec: float = 1.0,
    lam_smooth: float = 0.5,
    lam_align: float = 0.1,
    device: str | None = None,
):
    """
    adata: AnnData（要求 .X 表达矩阵，.obsm[coords_key] 空间坐标，.obs[ct_key] 细胞类型）
    返回：Z (N, z_dim), X_hat (N, F), logs(list of dict)，并把 Z 写回 adata.obsm['X_dualheadgat']
    """
    # 1) 取数据
    coords = np.asarray(adata.obsm[coords_key], dtype=np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.shape[1] > 3000:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    
    X = adata_Vars.X  # 可能是稀疏

    # 2) KNN 构图（utile 提供）：取“原始图”的邻接矩阵
    ad_matrix, ad_diag1, distW, *_ = cal_K_neighboorhood(coords, k=k, cell_types=None)

    # 3) 邻域细胞类型比例（不用 1/d）：(N, C)，行归一
    if resolution == 'low':
        ## the user should get the cell type abundace from the RCTD
        if ct_prop_df is None:
            raise ValueError("ct_prop_df is None! Run RCTD to obtain cell type abundance in each spot firstly!")
        else:
            props = ct_prop_df.values if hasattr(ct_prop_df, "values") else np.asarray(ct_prop_df)
        
    else:
        ct_vec = np.asarray(adata.obs[ct_key])
        ct_prop_df = cell_type_abundance(adata_Vars, ct_vec, 
                                         ad_matrix, binarize=True, 
                                         include_self=False,
                                         return_dense=True, 
                                         as_dataframe=True)
        props = ct_prop_df.values if hasattr(ct_prop_df, "values") else np.asarray(ct_prop_df)
    
    # 4) 邻接 -> edge_index；基于 props 计算边上 β（cosine）
    edge_index = adj_to_edge_index(ad_matrix)                 # (2, E)
    beta = cosine_beta_on_edges(props, edge_index)            # (E,)

    # 5) 组装张量
    x = ad_sparse_to_torch_dense(X)                           # (N, F) float32
    if in_dim is None:
        in_dim = x.shape[1]

    # 6) 构建模型
    # model = DualHeadGAT_AE(
    #     in_dim=in_dim, hidden1=hidden1, z_dim=z_dim, hidden2=,
    #     lambda_init=lambda_init, learnable_lambda=learnable_lambda,
    #     dropout=dropout, act=act, last_act=last_act
    # )
    
    model = DualHeadSTAGATE_AE(in_dim=in_dim, hidden1=hidden1, 
                           z_dim=z_dim, hidden2=hidden2,
                           lambda_init=lambda_init, learnable_lambda=learnable_lambda,
                           dropout=dropout, act=act, last_act=last_act)

    # 7) 训练（内部会按 dst 归一 beta，并计算 MSE + smooth + align）
    Z, X_hat, logs = train_model(
        model, x, edge_index, beta,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        lambda_rec=lam_rec, lambda_smooth=lam_smooth, lambda_align=lam_align,
        log_every=10, device=device
    )

    # 8) 写回 & 返回
    adata.obsm["X_dualheadgat"] = Z.cpu().numpy()
    z_df = pd.DataFrame(Z.cpu().numpy(), index=adata_Vars.obs_names, columns=[f"Z{i+1}" for i in range(Z.shape[1])])
    xhat_df = pd.DataFrame(X_hat.cpu().numpy(), index=adata_Vars.obs_names, columns=adata_Vars.var_names.astype(str))
    
    return adata, z_df, xhat_df, logs

# === 4) 命令行封装（可选） ===
# def main():
#     parser = argparse.ArgumentParser(description="Train Dual-Head GAT-AE for spatial domain embedding")
#     parser.add_argument("--h5ad", type=str, required=True, help="Path to input AnnData .h5ad")
#     parser.add_argument("--coords_key", type=str, default="spatial", help="adata.obsm key for coordinates")
#     parser.add_argument("--ct_key", type=str, required=True, help="adata.obs key for cell/spot type")
#     parser.add_argument("--k", type=int, default=8)
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--wd", type=float, default=1e-5)
#     parser.add_argument("--lam_rec", type=float, default=1.0)
#     parser.add_argument("--lam_smooth", type=float, default=0.5)
#     parser.add_argument("--lam_align", type=float, default=0.1)
#     parser.add_argument("--lambda_init", type=float, default=0.5)
#     parser.add_argument("--learnable_lambda", action="store_true")
#     parser.add_argument("--z_dim", type=int, default=64)
#     parser.add_argument("--hidden1", type=int, default=256)
#     parser.add_argument("--hidden2", type=int, default=256)
#     parser.add_argument("--device", type=str, default=None)
#     parser.add_argument("--out_h5ad", type=str, default=None, help="Optional: write AnnData with Z to this path")
#     args = parser.parse_args()

#     # 读入 AnnData
#     import anndata as ad
#     adata = ad.read_h5ad(args.h5ad)

#     Z, X_hat, logs = run_training(
#         adata,
#         coords_key=args.coords_key,
#         ct_key=args.ct_key,
#         k=args.k,
#         hidden1=args.hidden1, z_dim=args.z_dim, hidden2=args.hidden2,
#         lambda_init=args.lambda_init, learnable_lambda=args.learnable_lambda,
#         epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
#         lam_rec=args.lam_rec, lam_smooth=args.lam_smooth, lam_align=args.lam_align,
#         device=args.device
#     )

#     if args.out_h5ad:
#         adata.write_h5ad(args.out_h5ad)
#         print(f"[done] wrote: {args.out_h5ad}")
#     else:
#         print("[done] training finished; embedding in adata.obsm['X_dualheadgat'].")

# if __name__ == "__main__":
#     main()
