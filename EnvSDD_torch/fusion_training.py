# train_fusion.py
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from network_fusion import STAGATE_Fusion, normalize_beta_by_dst


def reconstruction_loss_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)

def smoothness_loss_beta_laplacian(Z: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   beta_norm: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    diff = Z[src] - Z[dst]
    loss = (beta_norm.unsqueeze(1) * (diff ** 2)).sum() / Z.size(0)
    return loss

def align_loss_kl_alpha_beta(alpha: torch.Tensor,
                             beta_norm: torch.Tensor,
                             edge_index: torch.Tensor,
                             eps: float = 1e-8) -> torch.Tensor:
    """KL(α || β)，按每个 dst 的入边分布。"""
    _, dst = edge_index
    a = alpha.clamp_min(eps)
    a_sum = scatter_add(a, dst, dim=0, dim_size=int(dst.max()) + 1) + eps
    a = a / a_sum[dst]
    b = beta_norm.clamp_min(eps)
    kl = (a * (a.add(eps).log() - b.add(eps).log())).sum() / edge_index.size(1)
    return kl


@torch.no_grad()
def prepare_beta_norm(edge_index: torch.Tensor,
                      beta: torch.Tensor,
                      num_nodes: int,
                      device: torch.device) -> torch.Tensor:
    beta_norm = normalize_beta_by_dst(edge_index, beta, num_nodes=num_nodes)
    return beta_norm.to(device)


import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

# ===== 如果与你的工程在同一目录，按需调整导入路径 =====
# from network_fusion import STAGATE_Fusion, normalize_beta_by_dst
# 这里直接粘贴 normalize_beta_by_dst，避免循环依赖



# ---------- 工具：从 AnnData 构造 edge_index ----------

def _edge_index_from_adata(adata):
    """
    根据 adata.uns['Spatial_Net'] 构造 edge_index（与 Transfer_pytorch_Data 一致）.
    期望 Spatial_Net 为 DataFrame，含列 'Cell1','Cell2'（细胞名或已是整数索引）。
    返回:
        edge_index: torch.LongTensor，shape=(2, E)
    """
    if 'Spatial_Net' not in adata.uns:
        raise ValueError("`adata.uns['Spatial_Net']` 不存在，请先构建 Spatial_Net。")

    G_df = adata.uns['Spatial_Net'].copy()

    # 允许 'Cell1'/'Cell2' 已经是整数索引；否则按 obs_names 映射
    cells = np.array(adata.obs_names)
    name2id = dict(zip(cells, range(cells.shape[0])))

    if not np.issubdtype(np.asarray(G_df['Cell1']).dtype, np.integer):
        G_df['Cell1'] = G_df['Cell1'].map(name2id)
    if not np.issubdtype(np.asarray(G_df['Cell2']).dtype, np.integer):
        G_df['Cell2'] = G_df['Cell2'].map(name2id)

    # 安全检查：是否有 NaN（映射失败的名字）
    if G_df['Cell1'].isna().any() or G_df['Cell2'].isna().any():
        bad = G_df[G_df['Cell1'].isna() | G_df['Cell2'].isna()]
        raise ValueError(f"Spatial_Net 中存在无法映射到 obs_names 的细胞名，样例：\n{bad.head()}")

    G_df['Cell1'] = G_df['Cell1'].astype(int)
    G_df['Cell2'] = G_df['Cell2'].astype(int)

    # 构图（与 Transfer_pytorch_Data 一致）：G + I（添加自环）
    n = adata.n_obs
    G = sp.coo_matrix(
        (np.ones(G_df.shape[0], dtype=np.float32), (G_df['Cell1'].values, G_df['Cell2'].values)),
        shape=(n, n),
        dtype=np.float32
    )
    G = G + sp.eye(n, dtype=np.float32, format='coo')

    # 提取非零作为边
    edgeList = np.nonzero(G)
    edge_index = torch.as_tensor(
        np.vstack([edgeList[0], edgeList[1]]).astype(np.int64),
        dtype=torch.long
    )
    return edge_index



# ---------- 工具：从微环境矩阵构建 β ----------
def _beta_from_microenv(adata,
                        prop_key: str,
                        edge_index: torch.Tensor,
                        nonneg: bool = True) -> torch.Tensor:
    """
    从 adata.obsm[prop_key]（如 RCTD 的细胞类型比例 N×C）计算沿边的余弦相似度：
        beta_raw_e = cos(p_u, p_v) = <p_u, p_v> / (||p_u||·||p_v||)
    返回：beta_raw（未按 dst 归一），shape (E,)
    """
    if prop_key not in adata.obsm:
        raise ValueError(f"在 adata.obsm 中找不到 '{prop_key}'，无法构建微环境先验 β。")
    P = adata.obsm[prop_key]  # N×C
    if sp.issparse(P):
        P = P.todense()
    P = np.asarray(P, dtype=np.float32)
    # L2 归一化
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)

    src, dst = edge_index.numpy()
    val = (Pn[src] * Pn[dst]).sum(axis=1)  # 余弦相似度
    if nonneg:
        val = np.clip(val, 0.0, None)      # 截断负值，避免与注意力分布耦合不稳
    beta_raw = torch.from_numpy(val.astype(np.float32))
    return beta_raw


# ---------- 训练主过程：AnnData 风格 ----------

def train_STAGATE_Fusion(
    adata,
    hidden_dims: List[int] = [256, 64],      # 与 STAGATE 一致：in_dim 将自动从数据推断
    n_epochs: int = 1000,
    lr: float = 1e-3,
    key_added: str = 'STAGATE_Fusion',
    gradient_clipping: float = 5.0,
    weight_decay: float = 1e-4,
    verbose: bool = True,
    random_seed: int = 0,
    save_loss: bool = False,
    save_reconstruction: bool = False,
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    microenv_key: str = 'microenv_prop',     # 例如 RCTD 的细胞类型比例矩阵所在 obsm key
    use_microenv: bool = True,
    lambda_balance: float = 0.5,             # λ∈[0,1]；=1 等价 STAGATE
    learnable_lambda_balance: bool = False,
    lambda_rec: float = 1.0,
    lambda_smooth: float = 0.5,
    lambda_align: float = 0.1,
    log_every: int = 50,
    early_stop: bool = True,
    patience: int = 50,
    tol: float = 1e-4,
    min_epochs: int = 200
):
    """
    训练 STAGATE_Fusion（α–β 注意力融合），输入为 AnnData，接口对齐 STAGATE。
    需要：
      - adata.uns['Spatial_Net']（或你自定义的空间图），本函数用 _edge_index_from_adata() 构图
      - 若 use_microenv=True，需要 adata.obsm[microenv_key]（N×C 的细胞类型比例）用于构建 β

    训练完成后写回：
      - 低维表示 -> adata.obsm[key_added]
      - 可选：重构矩阵 -> adata.layers[f'{key_added}_ReX']
      - 可选：训练日志 -> adata.uns[f'{key_added}_loss']
    """
    # —— 随机种子
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # —— 特征矩阵（稀疏转 CSR）
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # —— 只取 HVGs（若有）
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']].copy()
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input:', adata_Vars.shape)

    # —— 图（来自 Spatial_Net），形状 (2, E) 的 long tensor
    edge_index = _edge_index_from_adata(adata_Vars).to(torch.long)

    # —— X -> torch
    X = adata_Vars.X
    if sp.issparse(X):
        X = X.toarray()
    X = torch.tensor(X, dtype=torch.float32)

    N, Fin = X.shape

    # —— β：微环境先验（或均匀先验）
    if use_microenv:
        # 需要你项目中的 _beta_from_microenv(adata, key, edge_index, nonneg=True)
        beta_raw = _beta_from_microenv(adata_Vars, microenv_key, edge_index, nonneg=True)
        # 期望返回 (E,) 的 torch.float 张量；若返回 np.ndarray 请转 tensor
        if isinstance(beta_raw, np.ndarray):
            beta_raw = torch.from_numpy(beta_raw.astype(np.float32))
    else:
        # 均匀先验：对每条边赋 1（后续 normalize_beta_by_dst 会按 dst 归一）
        E = edge_index.size(1)
        beta_raw = torch.ones(E, dtype=torch.float32)

    # —— 设备
    device = torch.device(device)
    X = X.to(device)
    edge_index = edge_index.to(device)
    beta_raw = beta_raw.to(device)

    # —— 模型（与 STAGATE 同构：Fin -> hidden -> z_dim -> hidden -> Fin）
    assert len(hidden_dims) == 2, "hidden_dims 期望形如 [hidden, z_dim]"
    hidden, z_dim = hidden_dims
    model = STAGATE_Fusion(
        hidden_dims=[Fin, hidden, z_dim],
        lambda_init=lambda_balance,
        learnable_lambda=learnable_lambda_balance
    ).to(device)

    # —— 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # —— β 归一（按目标节点 dst 的入边求和归一）
    N = X.size(0)
    beta_norm = normalize_beta_by_dst(edge_index, beta_raw, num_nodes=N, eps=1e-8).to(device)

    # —— 训练日志
    logs: Dict[str, List[float]] = {"epoch": [], "loss": [], "rec": [], "smooth": [], "align": []}
    best_loss = float('inf')
    no_improve = 0
    eps = 1e-8

    # —— 预解包一次即可
    src, dst = edge_index

    # —— 训练循环
    for ep in tqdm(range(1, n_epochs + 1), disable=not verbose):
        model.train()
        optim.zero_grad()

        # 前向：Z(编码), X_hat(重构), alpha(来自 conv1 的融合注意力；shape=(E,) 或 (E,H))
        Z, X_hat, alpha = model(X, edge_index, beta_norm)

        # 1) 重构损失（MSE）
        loss_rec = F.mse_loss(X_hat, X)

        # 2) 平滑损失：β-Laplacian on Z
        diff = Z[src] - Z[dst]                      # (E, z_dim)
        loss_smooth = (beta_norm.unsqueeze(1) * (diff ** 2)).sum() / N

        # 3) KL 对齐：对 alpha 先按 dst 归一（保留梯度）
        a = alpha.clamp_min(eps)                    # (E,) 或 (E,H)
        a_sum = scatter_add(a, dst, dim=0, dim_size=N) + eps
        a = a / a_sum[dst]                          # 与 β 在同一“入边分布”尺度
        b = beta_norm.clamp_min(eps)                # (E,)
        if a.dim() == 2:                            # 多头情况，将 β 扩展到 (E,H)
            b = b.unsqueeze(1).expand_as(a)
        loss_align = (a * (a.add(eps).log() - b.add(eps).log())).sum() / edge_index.size(1)

        # 总损失
        loss = lambda_rec * loss_rec + lambda_smooth * loss_smooth + lambda_align * loss_align
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optim.step()

        # 日志
        if (ep % log_every == 0) or (ep == 1) or (ep == n_epochs):
            logs["epoch"].append(ep)
            logs["loss"].append(float(loss))
            logs["rec"].append(float(loss_rec))
            logs["smooth"].append(float(loss_smooth))
            logs["align"].append(float(loss_align))
            if verbose:
                print(f"[{ep:04d}] loss={loss.item():.6f} | "
                      f"rec={loss_rec.item():.6f} | smooth={loss_smooth.item():.6f} | "
                      f"align={loss_align.item():.6f}")

        # 早停
        if early_stop:
            if best_loss == float('inf'):
                best_loss = float(loss); no_improve = 0
            else:
                rel_imp = (best_loss - float(loss)) / (abs(best_loss) + eps)
                if rel_imp > tol:
                    best_loss = float(loss); no_improve = 0
                else:
                    no_improve += 1
            if ep >= min_epochs and no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep}: no relative improvement > {tol} "
                          f"for {patience} epochs (best={best_loss:.6f}).")
                break

    # —— 最终前向（eval）
    model.eval()
    with torch.no_grad():
        Z, X_hat, _ = model(X, edge_index, beta_norm)

    # —— 写回 AnnData
    adata.obsm[key_added] = Z.detach().cpu().numpy()
    if save_reconstruction:
        ReX = X_hat.detach().cpu().numpy()
        ReX[ReX < 0] = 0
        adata.layers[f'{key_added}_ReX'] = ReX
    if save_loss:
        adata.uns[f'{key_added}_loss'] = logs

    return adata




# ============== Minimal Demo ==============
# def main_demo():
#     torch.manual_seed(0)
#     N, F = 500, 100
#     X = torch.randn(N, F)

#     # toy 边（随机）
#     E = 5000
#     src = torch.randint(0, N, (E,))
#     dst = torch.randint(0, N, (E,))
#     edge_index = torch.stack([src, dst], dim=0)

#     # toy 微环境比例 -> 余弦相似度 -> β_raw
#     C = 6
#     P = torch.rand(N, C)
#     P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
#     Pn = F.normalize(P, p=2, dim=1)
#     beta_raw = (Pn[src] * Pn[dst]).sum(dim=1).clamp_min(0.0)

#     model = STAGATE_Fusion(hidden_dims=[F, 256, 64], lambda_init=0.5, learnable_lambda=True)
#     logs = train_model(
#         model, X, edge_index, beta_raw,
#         epochs=200, lr=1e-3, weight_decay=1e-5,
#         lambda_rec=1.0, lambda_smooth=0.5, lambda_align=0.1,
#         log_every=20, early_stop=True, patience=20, tol=1e-4, min_epochs=60
#     )
#     print("Z shape:", model.Z_.shape, "Xhat shape:", model.Xhat_.shape)

# 若作为脚本直接运行，取消下一行注释：
# if __name__ == "__main__":
#     main_demo()
