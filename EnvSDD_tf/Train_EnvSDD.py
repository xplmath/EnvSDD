import numpy as np
import scipy.sparse as sp
from EnvSDD_micro import STAGATE
import tensorflow.compat.v1 as tf
import pandas as pd
import scanpy as sc

tf.disable_eager_execution()

def train_STAGATE(adata, hidden_dims=[512, 30], alpha=0, n_epochs=500, lr=0.0001, key_added='STAGATE',
                  gradient_clipping=5, nonlinear=True, weight_decay=0.0001, verbose=True,
                  random_seed=2020, pre_labels=None, pre_resolution=0.2,
                  save_attention=False, save_loss=False, save_reconstrction=False,
                  # -------- 新增参数 --------
                  fusion_lambda=1.0, use_microenv=True, microenv_key='microenv_prop'):
    """Training graph attention auto-encoder (+ optional microenvironment prior β)."""

    tf.reset_default_graph()
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # 1) 基因选择
    adata_Vars = adata[:, adata.var['highly_variable']] if 'highly_variable' in adata.var.columns else adata
    X = pd.DataFrame(adata_Vars.X.toarray(), index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    if verbose:
        print('Size of Input: ', adata_Vars.shape)

    # 2) 构图（Spatial_Net -> 稀疏三元组）
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']

    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    N = adata.n_obs

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(N, N))
    G_tf = prepare_graph_data(G)  # indices = [col,row], 已加自环

    # 3) 构造 β（若不用则退化为均匀先验；均按 dst 归一）
    Beta_tf = build_beta_from_microenv(adata_Vars, G_df, N, microenv_key, use_microenv)

    # 4) 训练器
    trainer = STAGATE(hidden_dims=[X.shape[1]] + hidden_dims, alpha=alpha,
                      n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping,
                      nonlinear=nonlinear, weight_decay=weight_decay, verbose=verbose,
                      random_seed=random_seed, fusion_lambda=fusion_lambda)

    if alpha == 0:
        trainer(G_tf, G_tf, Beta_tf, X)
        embeddings, attentions, loss, ReX = trainer.infer(G_tf, G_tf, Beta_tf, X)
    else:
        # 预裁剪图
        G_df_prune = Spatial_Net.copy()
        if pre_labels is None:
            if verbose:
                print('------Pre-clustering using louvain with resolution=%.2f' % pre_resolution)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata)
            sc.tl.louvain(adata, resolution=pre_resolution, key_added='expression_louvain_label')
            pre_labels = 'expression_louvain_label'
        prune_G_df = prune_spatial_Net(G_df_prune, adata.obs[pre_labels])
        prune_G_df['Cell1'] = prune_G_df['Cell1'].map(cells_id_tran)
        prune_G_df['Cell2'] = prune_G_df['Cell2'].map(cells_id_tran)
        prune_G = sp.coo_matrix((np.ones(prune_G_df.shape[0]),
                                 (prune_G_df['Cell1'], prune_G_df['Cell2'])), shape=(N, N))
        prune_G_tf = prepare_graph_data(prune_G)
        prune_G_tf = (prune_G_tf[0], prune_G_tf[1], G_tf[2])  # 形状与 G_tf 保持一致

        trainer(G_tf, prune_G_tf, Beta_tf, X)
        embeddings, attentions, loss, ReX = trainer.infer(G_tf, prune_G_tf, Beta_tf, X)

    # 5) 回写
    cell_reps = pd.DataFrame(embeddings, index=cells)
    adata.obsm[key_added] = cell_reps.loc[adata.obs_names, :].values
    if save_attention:
        adata.uns['STAGATE_attention'] = attentions
    if save_loss:
        adata.uns['STAGATE_loss'] = loss
    if save_reconstrction:
        ReX = pd.DataFrame(ReX, index=X.index, columns=X.columns)
        ReX[ReX < 0] = 0
        adata.layers['STAGATE_ReX'] = ReX.values
    return adata


# ---------- 保持原有函数 ----------
def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' % Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label'] == Graph_df['Cell2_label'], ]
    print('%d edges after pruning.' % Graph_df.shape[0])
    return Graph_df

def prepare_graph_data(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # 注意顺序：[col,row]
    return (indices, adj.data, adj.shape)

def recovery_Imputed_Count(adata, size_factor):
    assert ('ReX' in adata.uns)
    temp_df = adata.uns['ReX'].copy()
    sf = size_factor.loc[temp_df.index]
    temp_df = np.expm1(temp_df)
    temp_df = (temp_df.T * sf).T
    adata.uns['ReX_Count'] = temp_df
    return adata

# ---------- 新增：构造 β（按 dst 归一），返回 TF 稀疏三元组 ----------
def build_beta_from_microenv(adata_Vars,
                             G_df_int,
                             N=None,
                             microenv_key='microenv_prop',
                             use_microenv=True):
    """
    基于 Spatial_Net 的有向边 (Cell1->Cell2) 与 obsm[microenv_key]（细胞类型比例）构建 β 先验：
      1) 对每条边 i->j：w_ij = max(0, cos(p_i, p_j))，p 为细胞类型比例向量
      2) 为每个节点添加自环 (j->j)，权重=1
      3) 在每个目标节点 j 的入边上做和为 1 的归一化：sum_{k->j} beta_kj = 1
    返回值：
      (indices, values, shape)：
        - indices: np.ndarray(E, 2) with [dst, src]（对齐 prepare_graph_data 的 [col, row] 约定）
        - values : np.ndarray(E,)
        - shape  : (N, N)
    说明：
      - 若 G_df_int['Cell1'/'Cell2'] 是细胞名，自动映射为 0..N-1；
      - 若已是整数索引，直接使用；
      - 若 obsm 中没有 microenv_key 或 use_microenv=False，则用均匀先验（边权=1）；
      - 若 P 为 DataFrame，会按 adata_Vars.obs_names 显式重排后再转 numpy。
    """
    if N is None:
        N = adata_Vars.n_obs

    # ---------- 1) 统一 Cell1/Cell2 为整数索引 ----------
    c1 = G_df_int['Cell1'].values
    c2 = G_df_int['Cell2'].values

    def _is_numeric(arr):
        try:
            return np.issubdtype(np.asarray(arr).dtype, np.number)
        except Exception:
            return False

    if not _is_numeric(c1):
        # 名称 -> 索引
        names = np.asarray(adata_Vars.obs_names)
        name2id = {n: i for i, n in enumerate(names)}
        try:
            src = np.asarray([name2id[x] for x in c1], dtype=np.int64)
            dst = np.asarray([name2id[x] for x in c2], dtype=np.int64)
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(f"Edge list contains a cell '{missing}' not found in adata.obs_names.")
    else:
        src = c1.astype(np.int64, copy=False)
        dst = c2.astype(np.int64, copy=False)

    # 丢弃越界与缺失
    mask = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
    if not np.all(mask):
        src = src[mask]; dst = dst[mask]

    # 去除已有自环，防止重复加（可选）
    non_self = src != dst
    src = src[non_self]; dst = dst[non_self]

    # 若没有边，后续只依赖自环
    # ---------- 2) 取微环境比例并计算 cos 相似 ----------
    if (not use_microenv) or (microenv_key not in adata_Vars.obsm):
        w_edges = np.ones_like(src, dtype=np.float32)
    else:
        P = adata_Vars.obsm[microenv_key]
        P = P.loc[adata_Vars.obs_names].to_numpy(dtype=np.float32)
        # DataFrame -> 按 obs_names 对齐行顺序再转 numpy
        # if isinstance(P, pd.DataFrame):
        #     # 若行索引不是 obs_names，按 obs_names 重新排序（缺失会触发 KeyError）
        #     try:
        #         P = P.loc[adata_Vars.obs_names].to_numpy(dtype=np.float32)
        #     except KeyError:
        #         # 容错：若部分缺失，先 reindex，缺失填 0
        #         P = P.reindex(index=adata_Vars.obs_names, fill_value=0.0).to_numpy(dtype=np.float32)
        # elif sp.issparse(P):
        #     P = P.toarray().astype(np.float32)
        # else:
        #     P = np.asarray(P, dtype=np.float32)

        if P.shape[0] != N:
            raise ValueError(f"obsm['{microenv_key}'] has {P.shape[0]} rows but adata has {N} cells.")

        if src.size == 0:
            w_edges = np.array([], dtype=np.float32)
        else:
            ps = P[src, :]    # (E, C)
            pd = P[dst, :]    # (E, C)
            num = (ps * pd).sum(axis=1)
            den = (np.linalg.norm(ps, axis=1) * np.linalg.norm(pd, axis=1)) + 1e-8
            w_edges = np.clip(num / den, 0.0, 1.0).astype(np.float32)

    # ---------- 3) 加自环并按目标节点归一 ----------
    self_idx = np.arange(N, dtype=np.int64)
    src_all = np.concatenate([src, self_idx], axis=0)
    dst_all = np.concatenate([dst, self_idx], axis=0)
    w_all = np.concatenate([w_edges, np.ones(N, dtype=np.float32)], axis=0)

    # 每个目标节点 j 的入边和为 1
    sums = np.bincount(dst_all, weights=w_all, minlength=N).astype(np.float32) + 1e-8
    w_all = (w_all / sums[dst_all]).astype(np.float32)

    # ---------- 4) 返回 TF 稀疏三元组（对齐 prepare_graph_data：indices=[dst,src]） ----------
    indices = np.vstack([dst_all, src_all]).T.astype(np.int64)
    shape = (N, N)
    return (indices, w_all, shape)
