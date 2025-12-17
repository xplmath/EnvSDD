# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors

class STAGATE(torch.nn.Module):
    """
    你的原始 STAGATE 框架（GCN 版），改成支持 edge_weight。
    hidden_dims = [in_dim, num_hidden, out_dim]
    forward 返回 (h2, h4) 分别是编码器输出与重构输出。
    """
    def __init__(self, hidden_dims):
        super().__init__()
        in_dim, num_hidden, out_dim = hidden_dims
        self.conv1 = GCNConv(in_dim,      num_hidden)  # Encoder 1
        self.conv2 = GCNConv(num_hidden,  out_dim)     # Encoder 2 (embedding h2=Z)
        self.conv3 = GCNConv(out_dim,     num_hidden)  # Decoder 1
        self.conv4 = GCNConv(num_hidden,  in_dim)      # Decoder 2 (recon h4=ĤX)
        self._edge_weight = None  # 可选：缓存权重，便于不每次都传

    def forward(self, features, edge_index, edge_weight=None):
        if edge_weight is None and self._edge_weight is not None:
            edge_weight = self._edge_weight
        # Encoder
        h1 = self.conv1(features, edge_index, edge_weight=edge_weight)
        h2 = self.conv2(h1,       edge_index, edge_weight=edge_weight)
        # Decoder
        h3 = self.conv3(h2,       edge_index, edge_weight=edge_weight)
        h4 = self.conv4(h3,       edge_index, edge_weight=edge_weight)
        return h2, h4

    @torch.no_grad()
    def set_edge_weight(self, edge_weight: torch.Tensor):
        """
        可选：把已经准备好的 edge_weight 存进模型，
        这样 forward 时可以不再传入 edge_weight。
        """
        self._edge_weight = edge_weight


def prepare_weighted_graph_from_rctd(
    coords: np.ndarray,
    ct_props: np.ndarray,
    k: int = 10,
    make_undirected: bool = True,
    device: torch.device | str | None = None
):
    """
    基于 spot 空间坐标 + RCTD 细胞类型比例，构建 '空间KNN + 余弦相似度权重' 的加权图。

    参数
    ----
    coords : (N, 2) numpy 数组，spot 的 (x, y) 坐标
    ct_props : (N, C) numpy 数组，每个 spot 的细胞类型比例（行向量非负、行和≈1）
    k : 每个点保留的空间近邻数（不含自身）
    make_undirected : 是否把每条边反向也加入（常见做法，推荐 True）
    device : None / "cpu" / "cuda"；返回的 PyTorch 张量所放设备

    返回
    ----
    edge_index : (2, E) torch.long，PyG 的边索引
    edge_weight: (E,)   torch.float，边权（邻接的“强度”=余弦相似度）
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords 需为 (N,2)"
    assert ct_props.ndim == 2, "ct_props 需为 (N,C)"
    n = coords.shape[0]
    if ct_props.shape[0] != n:
        raise ValueError("coords 与 ct_props 行数 (spot 数) 不一致")

    # 1) 空间 KNN（基于坐标；只决定连边不决定权重）
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
    nn_model.fit(coords)
    # indices: 每行第一个是自身，后面是最近的 k 个邻居
    neigh_dist, neigh_idx = nn_model.kneighbors(coords, return_distance=True)
    # 去掉自身邻居（列0）
    neigh_idx = neigh_idx[:, 1:]  # 形状 (N, k)

    # 2) 预归一化比例向量，用于余弦相似度
    #    由于是比例（非负，行和≈1），cosine ∈ [0,1]
    eps = 1e-8
    norm = np.linalg.norm(ct_props, axis=1, keepdims=True) + eps
    ct_unit = ct_props / norm  # (N, C)

    # 3) 生成边 (i -> j) 以及对应权重 w_ij = cos(pi, pj)
    #    我们只对 KNN 的邻居计算权重，避免 O(N^2)。
    src_list = []
    dst_list = []
    w_list = []

    for i in range(n):
        js = neigh_idx[i]  # 这个 i 的 k 个邻居
        # 余弦相似：向量点积（两端已单位化）
        w_ij = (ct_unit[i:i+1, :] @ ct_unit[js].T).ravel()  # (k,)
        # 保护性截断到 [0,1]
        w_ij = np.clip(w_ij, 0.0, 1.0)

        src_list.extend([i] * len(js))
        dst_list.extend(js.tolist())
        w_list.extend(w_ij.tolist())

    if make_undirected:
        # 加入反向边，权重对称（w_ji = w_ij），
        # 也可以改为用 ct_unit[j]·ct_unit[i] 重新算（数值等同）
        src_list.extend(dst_list)  # 反向
        dst_list.extend(src_list[:len(w_list)])  # 注意取反前的 src
        w_list.extend(w_list)  # 对称复制

    # 4) 转 PyTorch Geometric 需要的张量
    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=device
    )
    edge_weight = torch.tensor(
        w_list, dtype=torch.float32, device=device
    )

    return edge_index, edge_weight


# --------------------------
# 使用示例（放在你的训练脚本里运行）
# --------------------------
if __name__ == "__main__":
    # 伪造一些数据演示
    N, P, C = 100, 2000, 6
    coords = np.random.rand(N, 2) * 100.0                 # (N,2)
    X = torch.randn(N, P)                                  # 表达矩阵 (标准化后)
    ct = np.random.dirichlet(alpha=np.ones(C), size=N)     # 模拟 RCTD 输出 (N,C)

    # 1) 构图（KNN + 余弦权重）
    edge_index, edge_weight = prepare_weighted_graph_from_rctd(
        coords=coords, ct_props=ct, k=10, make_undirected=True, device="cpu"
    )

    # 2) 建模 & 前向
    model = STAGATE(hidden_dims=[P, 256, 64])
    # 也可缓存权重：model.set_edge_weight(edge_weight)
    z, X_hat = model(X, edge_index, edge_weight=edge_weight)
    print(z.shape, X_hat.shape)  # (N,64) (N,P)
