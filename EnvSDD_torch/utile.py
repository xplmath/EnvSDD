# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:40:13 2025

@author: lihs
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
import sklearn.neighbors

import torch
from torch_geometric.data import Data

######## A part of code copying from https://github.com/zhanglabtools/STAGATE/blob/main/STAGATE/utils.py

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.
    
    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section, ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat(
            [adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section-1):
        section_1 = section_order[it]
        section_2 = section_order[it+1]
        if verbose:
            print('------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        Z_Net_ID = section_1+'-'+section_2
        temp_adata = adata[adata.obs[key_section].isin(
            [section_1, section_2]), ]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge, ]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, [
            'Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0]/temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat(
        [adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
            (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
            (adata.uns['Spatial_Net'].shape[0]/adata.n_obs))

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)
    
    
def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
    
#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']

#     res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
#     mclust_res = np.array(res[-2])

#     adata.obs['mclust'] = mclust_res
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata

# def cal_K_neighboorhood(cell_sort_loc, k=10, cell_types=None):
#     """
#     基于坐标构建KNN图，返回：
#       1) 原始KNN邻接矩阵/对角补1/距离加权矩阵
#       2) 仅保留“同类细胞”边后的邻接矩阵/对角补1/距离加权矩阵（若提供cell_types）

#     参数
#     ----
#     cell_sort_loc : (N, d) array-like
#         细胞的空间坐标
#     k : int
#         K近邻的K
#     cell_types : array-like, shape (N,) 
#         (N,)：int/str均可


#     返回
#     ----
#     ad_matrix : sp.csc_matrix (N, N)
#     ad_matrix_diag_one : sp.csc_matrix (N, N)
#     dist_weight_matrix : sp.csc_matrix (N, N)

#     ad_matrix_same : sp.csc_matrix (N, N)                # 仅同类边（若 cell_types 提供，否则为空稀疏矩阵）
#     ad_matrix_same_diag_one : sp.csc_matrix (N, N)
#     dist_weight_matrix_same : sp.csc_matrix (N, N)
#     """
#     def diag_with_one(A):
#         # 在稀疏矩阵 A 的对角线上补 1（如果已有对角元素，会变成1）
#         n = A.shape[0]
#         I = sp.identity(n, format="csc", dtype=np.float32)
#         B = A.copy().tocsc()
#         B.setdiag(0)           # 清空原对角，避免叠加>1
#         return (B + I).tocsc()

#     def normalize_labels(ct, n_nodes):
#         if ct is None:
#             return None
#         arr = np.asarray(ct)
#         if len(arr) != n_nodes:
#                 raise ValueError("The length of cell_type should been sample")
#         return arr
        

#     n_nodes = cell_sort_loc.shape[0]
#     labels = normalize_labels(cell_types, n_nodes)

#     # 1) KNN
#     nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(cell_sort_loc)
#     distances, indices = nbrs.kneighbors(cell_sort_loc)
#     distances = distances[:, 1:]  
#     indices = indices[:, 1:]

#     # 2) 距离阈值（均值 + 3*std）
#     mean_distance = np.mean(distances)
#     std_distance = np.std(distances)
#     threshold = mean_distance + 3 * std_distance

#     # 3) 构建边：edges_all 为原始筛选（按距离阈值），edges_same 进一步要求同类
#     edges_all = []
#     edges_same = []

#     if labels is None:
#         # 未提供 labels：只构建原始边
#         for i, (dists, neighs) in enumerate(zip(distances, indices)):
#             for dist, j in zip(dists, neighs):
#                 if dist <= threshold:
#                     edges_all.append((i, j, float(dist)))
#     else:
#         for i, (dists, neighs) in enumerate(zip(distances, indices)):
#             for dist, j in zip(dists, neighs):
#                 if dist <= threshold:
#                     edges_all.append((i, j, float(dist)))
#                     if labels[i] == labels[j]:
#                         edges_same.append((i, j, float(dist)))

#     def build_graph_from_edges(edges, n):
#         # 从 (u, v, w) 列表构建 0/1 邻接矩阵 与 距离加权矩阵（均做对称化）
#         if len(edges) == 0:
#             Z = sp.csc_matrix((n, n), dtype=np.float32)
#             return Z, diag_with_one(Z), Z

#         rows, cols, data = zip(*edges)
#         rows = np.fromiter(rows, dtype=np.int32)
#         cols = np.fromiter(cols, dtype=np.int32)
#         data = np.fromiter(data, dtype=np.float32)

#         # 0/1 邻接
#         A = sp.csc_matrix((np.ones_like(data, dtype=np.float32), (rows, cols)),
#                           shape=(n, n), dtype=np.float32)
#         A = (A + A.T).tocsc()
#         A.data[:] = 1.0  # 将>1 的项截断为1
#         A_diag1 = diag_with_one(A)

#         # 距离加权
#         W = sp.csc_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
#         W = (W + W.T).tocsc()

#         return A, A_diag1, W

#     # 原始图
#     ad_matrix, ad_matrix_diag_one, dist_weight_matrix = build_graph_from_edges(edges_all, n_nodes)

#     # 同类过滤图（若未提供 labels，则返回空矩阵）
#     ad_matrix_same, ad_matrix_same_diag_one, dist_weight_matrix_same = build_graph_from_edges(edges_same, n_nodes)

#     return (ad_matrix, ad_matrix_diag_one, dist_weight_matrix,
#             ad_matrix_same, ad_matrix_same_diag_one, dist_weight_matrix_same)


def cell_type_abundance(
    adata,
    ct_vec,
    ad_matrix,
    *,
    binarize: bool = True,      # True: 非零即相邻，权重置为1；False: 用原矩阵的权重（不做1/d）
    include_self: bool = False, # 是否把自身计入邻域（默认不包含）
    return_dense: bool = False, # 返回稠密数组或稀疏矩阵
    as_dataframe: bool = True   # 尽量返回 DataFrame，包含 obs_names 和类型名
):
    """
    计算每个细胞“周围（邻域）”的细胞类型比例（行和=1）。
    不对邻接做 1/d 变换；只按邻接关系统计，再行归一化。

    参数
    ----
    adata: AnnData，至少需要 n_obs / obs_names
    ct_vec: (N,) 细胞类型标签（字符串或类别）
    ad_matrix: (N,N) 稀疏邻接矩阵（KNN得到的距离矩阵也行，默认会二值化为邻接）
    binarize: True 时将 ad_matrix 的非零元素置为 1（仅表示相邻）
    include_self: 是否将自身计入邻域统计
    return_dense: True 返回 numpy.ndarray；False 返回 csr_matrix
    as_dataframe: True 则尽量返回 pandas.DataFrame

    返回
    ----
    (N,K) 的比例矩阵（每行和=1），行对应细胞，列对应细胞类型。
    """
    # 基本检查
    N = adata.n_obs if hasattr(adata, "n_obs") else adata.shape[0]
    if len(ct_vec) != N:
        raise ValueError(f"ct_vec 长度 {len(ct_vec)} 与样本数 {N} 不一致")
    if not sp.issparse(ad_matrix):
        ad_matrix = csr_matrix(ad_matrix)
    if ad_matrix.shape != (N, N):
        raise ValueError(f"ad_matrix 形状 {ad_matrix.shape} 不是 {(N, N)}")

    # 1) 邻接矩阵处理（是否二值化、是否包含自身）
    W = ad_matrix.tocsr().astype(np.float32).copy()
    if binarize:
        W.data[:] = 1.0
    if not include_self:
        # 去掉自环（自身不计入“周围”）
        W.setdiag(0.0)
        W.eliminate_zeros()

    # 2) 标签 → one-hot 指示矩阵 M (N,K)
    le = LabelEncoder()
    labels = le.fit_transform(np.asarray(ct_vec))
    K = len(le.classes_)
    M = csr_matrix(
        (np.ones(N, dtype=np.float32), (np.arange(N), labels)),
        shape=(N, K),
        dtype=np.float32
    )

    # 3) 邻域类型计数：W @ M  → (N,K)
    ct_counts = W @ M

    # 4) 行归一化为比例（每行和=1；若无邻居则整行保持为0）
    row_sums = np.asarray(ct_counts.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    inv[nonzero] = 1.0 / row_sums[nonzero]
    Dinv = sp.diags(inv)
    ct_prop = Dinv @ ct_counts  # (N,K)

    # 5) 组织输出
    out = ct_prop.toarray().astype(np.float32) if return_dense else ct_prop.tocsr()
    if as_dataframe:
        try:
            import pandas as pd
            index = getattr(adata, "obs_names", None)
            if index is None or len(index) != N:
                index = [f"cell_{i}" for i in range(N)]
            columns = list(le.classes_)
            out = (pd.DataFrame(out, index=index, columns=columns)
                   if return_dense
                   else pd.DataFrame.sparse.from_spmatrix(out, index=index, columns=columns))
        except Exception:
            pass
    return out


# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
    
#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']

#     res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
#     mclust_res = np.array(res[-2])

#     adata.obs['mclust'] = mclust_res
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata
def search_res_base(adata_base, n_setting, min_res = 1e-3, max_res = 5, 
                random_state = 2025, max_step = 20, tolerance = 0):
    this_step = 0
    this_min = float(min_res)
    this_max = float(max_res)
    
    while this_step < max_step:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.louvain(adata_base, resolution = this_resolution, random_state = random_state)
        res = adata_base.obs["louvain"].astype(int)
        this_cluster = len(np.unique(res))
        print(f"Step K: {this_step}, Louvain resolution: {this_resolution}, "
              f"Number of clusters: {this_cluster}, Ideal of clusters: {n_setting}")
        
        if this_cluster > n_setting + tolerance:
            this_max = this_resolution
        elif this_cluster < n_setting - tolerance:
            this_min = this_resolution
        else:
            print(f'Succeeded in finding clusters: {n_setting} with resolution: {this_resolution}')
            break

        this_step += 1
        
        if this_step >= max_step:
            print("Cannot find the desired number of clusters.")
    label = adata_base.obs['louvain']
    return label
            
    


def run_louvain(adata_base_methods, n_setting = 7, neigh = 50, use_rep = 'STAGATE_Fusion'):
    
    sc.pp.neighbors(adata_base_methods, n_neighbors = neigh, use_rep = use_rep)
    # clustering_df = pd.DataFrame(index=adata_base_methods.obs_names)
    
    # if len(n_setting) == 1:
    res = search_res_base(adata_base_methods, n_setting)
    res_final = res.astype('category')
    # else:
    #     for i in n_setting:
    #         res = search_res_base(adata_base_methods, i)
    #         clustering_df[f'Louvain_clustering_{i}'] = res.astype('category')
    return res_final