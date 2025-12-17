# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:59:54 2025

@author: lihs
"""
import scanpy as sc
import stlearn as st
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
from Train_STAGATE import train_STAGATE
from utils import run_leiden
# import torch

import os
os.environ['R_HOME'] = 'D:/Users/lihs/anaconda3/envs/Community_domain/Lib/R'
os.environ['R_USER'] = 'D:/Users/lihs/anaconda3/envs/Community_domain/Lib/site-packages/ryp2'
# from utile import mclust_R

BASE = Path(r"D:/Experiment/nyq/HBC/dataset")
from pathlib import Path
OUT  = Path('D:/Experiment/nyq/HBC/results')
OUT.mkdir(exist_ok=True)


counts_path = 'D:/Experiment/nyq/HBC/dataset/counts_matrix.txt'
meta_path = 'D:/Experiment/nyq/HBC/dataset/meta_data.txt'
ct_path = 'D:/Experiment/nyq/HBC/deconvolution/rctd_res.txt'
img_path = 'D:/Experiment/nyq/HBC/dataset/V1_Breast_Cancer_Block_A_Section_1_image.tif'
out_path = 'D:/Experiment/nyq/HBC/results'


adata = sc.read_text(str(counts_path)).T
adata.var_names_make_unique()
meta_data = pd.read_csv(meta_path, sep="\t")
adata.obs = meta_data

ct_abundance = pd.read_csv(ct_path, sep="\t", index_col = 0)

adata = adata[ct_abundance.index.tolist()]
# adata.obsm['spatial'] = np.asarray(adata.obs[['pixel_y','pixel_x']], dtype=float)


imgcol = meta_data.loc[:,"pixel_x"]
imgrow = meta_data.loc[:,"pixel_y"]
adata.obs["imagecol"] = imgrow
adata.obs["imagerow"] = imgcol
adata.obs["array_row"] = meta_data.loc[:,"x"]
adata.obs["array_col"] = meta_data.loc[:,"y"]
st.add.image(adata,library_id="151673",quality="fulres",
             imgpath=img_path,scale=1,spot_diameter_fullres=150)


sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


from utils import Cal_Spatial_Net
Cal_Spatial_Net(adata, model = 'KNN', k_cutoff = 6)


adata_tmp_0 = train_STAGATE(adata.copy(), alpha=0, fusion_lambda=0)
z_df_0 = pd.DataFrame(
    data=adata_tmp_0.obsm['STAGATE'],
    index=adata_tmp_0.obs_names.tolist(),
    columns=[f'L{i}' for i in range(30)]
)
z_df_0.to_csv('D:/Experiment/nyq/HBC/results/Z_0.txt', sep="\t", index=True)


adata_tmp_07 = train_STAGATE(adata.copy(), alpha=0, fusion_lambda=0.7)

z_df_07 = pd.DataFrame(
    data=adata_tmp_07.obsm['STAGATE'],
    index=adata_tmp_07.obs_names.tolist(),
    columns=[f'L{i}' for i in range(30)]
)
z_df_07.to_csv('D:/Experiment/nyq/HBC/results/Z_07.txt', sep="\t", index=True)



######### running with the mclust in the R environment ###################################
######### running with the mclust in the R environment ###################################

from PIL import Image
import os
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score


def ari_cal(true_lable, pred_lable):
    ari = adjusted_rand_score(true_lable, pred_lable)
    return ari

def nmi_score(classes, clusters):
    """
    classes  : 真实标签 (list/np.array/pd.Series)
    clusters : 聚类结果 (list/np.array/pd.Series)
    """
    return normalized_mutual_info_score(classes, clusters)


### ARI_str, Pur_str, sample_id: list str array
def visualization_function(counts_path, meta_path, img_path, dir_path,
                           cluster_num = 10, spot_diameter_fullres = 150, spot_size = 0.8):
    adata_ini = ad.read_text(counts_path).T
    sc.pp.normalize_total(adata_ini, inplace=True)
    sc.pp.log1p(adata_ini)
    meta_data_ini = pd.read_csv(meta_path, sep="\t", index_col = 0)
    
    Index_ann = adata_ini.obs_names
    Index_meta = meta_data_ini.index
    inter_index =  Index_ann.intersection(Index_meta)
    
    adata = adata_ini[inter_index,]
    meta_data = meta_data_ini.loc[inter_index]
    
    
    for col in meta_data.columns.values[16:26]:
        meta_data[col] = meta_data[col].astype("category")
    
    meta_data = meta_data.rename(columns={'DR.SC': 'DR-SC'})
    # meta_data = meta_data.drop(columns=['muse'])
    
    meta_data['old_fine_annot_type'] = meta_data['old_fine_annot_type'].astype("category")
    adata.obs = meta_data
    
    adata.obsm["spatial"] = meta_data.loc[:,["pixel_y", "pixel_x"]].values
    
    
    
    ### deal with image
    Image.MAX_IMAGE_PIXELS=None
    img = Image.open(img_path)
    img = np.array(img)
    spatial_key = "spatial"
    library_id = '151673' # 你的样本的name
    adata.uns.setdefault(spatial_key, {}).setdefault(library_id, {})
    adata.uns[spatial_key][library_id]["images"] = {"hires": None}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1.,
                                                          "spot_diameter_fullres": spot_diameter_fullres}
    
    methods_name_inte = ["BayesSpace", "DR-SC", "GraphST", "SpaGCN", "STAGATE", "stLearn", "EnvSDD"]
    
    ARI = np.array([])
    Pur = np.array([])
    
    for tt in methods_name_inte:
        if tt == 'layer':
            continue
        true_lable = np.array(meta_data['old_fine_annot_type'])
        pred_label = np.array(meta_data[tt]).astype(str)
        ari_res = ari_cal(true_lable, pred_label)
        purity_res = nmi_score(true_lable, pred_label)
        ARI = np.append(ARI, ari_res)
        Pur = np.append(Pur, purity_res)
    
    ARI_tmp = np.round(ARI, decimals=3)
    ARI_str = [f"{x:.3f}" for x in ARI_tmp]
    Pur_tmp = np.round(Pur, decimals=3)
    Pur_str = [f"{x:.3f}" for x in Pur_tmp]
    
    # ARI_str = np.array([0.55, 0.44, 0.43, 0.44, 0.47, 0.51, 0.52]).astype(str).tolist()
    ARI_title = ["ARI = " + x for x in ARI_str]
    # Pur_str = np.array([0.70, 0.52, 0.61, 0.57, 0.53, 0.65, 0.62]).astype(str).tolist()
    Pur_title = [", Pur = " + x for x in Pur_str]
    
    from functools import partial, reduce
    title_inte1 =  list(reduce(partial(map, str.__add__), (ARI_title, Pur_title)))
    

    
    # methods_name_inte = ['layer', "BayesSpace", "DR-SC", "GraphST", "SiGra", "SpaGCN", "STAGATE", "spaVAE", "stLearn", 'StaMarker', "EnSDD"]
    # ARI_title = ["ARI = " + x for x in ARI_str]
    # title_inte = np.array([151507]).astype(str).tolist() + list(title_inte1)
    title_inte = list(title_inte1)
    
    

    
    fig, axs = plt.subplots(2, 5, figsize = (8.2,4))
    fig.delaxes(axs[1, 4])
    
    if cluster_num == 20:
        for col_idx, mtd in enumerate(methods_name_inte):
            # print(col_idx)
            # print(mtd)
            # ms = methods_name_inte[col_idx] 
            # file_name = mtd + ".pdf"
            # ax = axs[col_idx]
            
            if col_idx < 5:
                row_pl = 0
                col_pl = col_idx
                ax = axs[0, col_pl]
                sc.pl.spatial(adata, img_key = "hires", color = mtd, size = spot_size, 
                              title = title_inte[col_idx], legend_loc = 'on data', legend_fontsize = 8, 
                              legend_fontweight = 'bold', ax = ax, save=False)
                ax.set(xlabel=None)
                ax.set(ylabel = None)
                ax.set_title(mtd, size = 6, pad = 10)  
                ax.annotate(title_inte[col_idx], xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom', fontsize=5)
            
            else:
                row_pl = 1
                col_pl = ((col_idx + 1) % 5) - 1
                
                ax = axs[row_pl, col_pl]
                sc.pl.spatial(adata, img_key = "hires", color = mtd, size = spot_size, 
                              title = title_inte[col_idx], legend_loc = 'on data', legend_fontsize = 8, 
                              legend_fontweight = 'bold' , ax = ax, save=False)
                ax.set(xlabel=None)
                ax.set(ylabel = None)
                ax.set_title(mtd, size = 6, pad = 10)  
                ax.annotate(title_inte[col_idx], xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom', fontsize=5)
        plt.subplots_adjust(wspace=0.1, hspace=0.01)
        plt.tight_layout()  
        save_file_name = os.path.join(dir_path, str(cluster_num) + ".pdf")
        fig.savefig(save_file_name, dpi = 300)
    
        
    if cluster_num != 20:
        for col_idx, mtd in enumerate(methods_name_inte):
            
            if col_idx < 5:
                row_pl = 0
                col_pl = col_idx
                ax = axs[0, col_pl]
                sc.pl.spatial(adata, img_key = "hires", color = mtd, size = spot_size, 
                              title = methods_name_inte[col_idx], legend_loc = 'on data', legend_fontsize = 8, 
                              legend_fontweight = 'bold' , ax = ax, save=False)
                ax.set(xlabel=None)
                ax.set(ylabel = None)
                ax.set_title(mtd, size = 6, pad = 10)  
                # ax.annotate(title_inte[col_idx], xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom', fontsize=5)
            
            else:
                row_pl = 1
                col_pl = ((col_idx + 1) % 5) - 1
                
                ax = axs[row_pl, col_pl]
                sc.pl.spatial(adata, img_key = "hires", color = mtd, size = spot_size, 
                              title = methods_name_inte[col_idx], legend_loc = 'on data', legend_fontsize = 8, 
                              legend_fontweight = 'bold' , ax = ax, save=False)
                ax.set(xlabel=None)
                ax.set(ylabel = None)
                ax.set_title(mtd, size = 6, pad = 10)  
                # ax.annotate(title_inte[col_idx], xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom', fontsize=5)
        plt.subplots_adjust(wspace=0.1, hspace=0.01)
        plt.tight_layout()  
        save_file_name = os.path.join(dir_path, str(cluster_num) + ".pdf")
        fig.savefig(save_file_name, dpi = 300)
        
        
    
    
###### combine the labels of base methods and     
# base_res_path = 'D:/Experiment/nyq/HBC/results/clustering.res.20.txt'
# base_results = pd.read_csv(base_res_path, sep="\t")   
   
# Env_res_path = 'D:/Experiment/nyq/HBC/results/meta_data_EnvSDD.txt'    
# EnvSDD_results = pd.read_csv(Env_res_path, sep="\t")


# base_results = base_results.loc[EnvSDD_results.index.tolist()]
# base_results['EnvSDD'] = EnvSDD_results['EnvSDD07']

# base_results.to_csv('D:/Experiment/nyq/HBC/results/meta_data_all.txt', sep="\t", index=True)



counts_path = 'D:/Experiment/nyq/HBC/dataset/counts_matrix.txt'
meta_path_20 = 'D:/Experiment/nyq/HBC/results/meta_data_all.txt'
img_path = 'D:/Experiment/nyq/HBC/dataset/V1_Breast_Cancer_Block_A_Section_1_image.tif'
dir_path = 'D:/Experiment/nyq/HBC/plot'

visualization_function(counts_path, meta_path_20, img_path, dir_path= dir_path, cluster_num = 20, spot_diameter_fullres = 150,
                       spot_size = 1.5)

# meta_path_25 = "/home/vision/Downloads/LHS/EnDecon/experiment/Human breast cancer/result_patch_HBC/clustering.res.25.txt"
# visualization_function(counts_path, meta_path_25, img_path, dir_path= dir_path, cluster_num = 25, spot_diameter_fullres = 150,
#                        spot_size = 1.5)

# meta_path_30 = "/home/vision/Downloads/LHS/EnDecon/experiment/Human breast cancer/result_patch_HBC/clustering.res.30.txt"
# visualization_function(counts_path, meta_path_30, img_path, dir_path= dir_path, cluster_num = 30, spot_diameter_fullres = 150,
#                        spot_size = 1.5)






#################  We will analysis 1 and 8 in IDC 5
################# We will analysis 6, 11, 15 in IDC 2
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
def enrichment_analysis(adata, id_key, val_key):
    print(f"Calculating the enrichment of each cluster ({id_key}) in group ({val_key})...")
    obs = adata.obs.copy()
    id_list = sorted(list(set(obs[id_key])))
    val_list = sorted(list(set(obs[val_key])))
    
    df_count = obs.groupby([id_key, val_key]).size().unstack().fillna(0)

    # 计算比例，避免误差
    MIN_NUM = 20
    df_count.loc[:, df_count.sum() < MIN_NUM] = 0
    df_normalized = df_count.div(df_count.sum(axis=0), axis=1)
        
    
    
    pval = []
    pval_adj = []
    N = adata.shape[0]
    for idx in id_list:
        K = df_count.loc[idx].sum()
        
        pval_tmp = []

        for val in val_list:
            n = df_count[val].sum()
            k = df_count.loc[idx,val]
            
            p_value = stats.hypergeom.sf(k-1, N, K, n)
            pval_tmp.append(p_value)
        
        _, p_adj_tmp, _, _ = multipletests(pval_tmp, method = 'fdr_bh')
        pval.append(pval_tmp)
        pval_adj.append(p_adj_tmp) 
    
    pval = pd.DataFrame(pval)
    pval_adj = pd.DataFrame(pval_adj)
    pval.columns = pval_adj.columns = val_list
    pval.index = pval_adj.index = id_list
    
    df_normalized = df_normalized.reindex(index=pval.index, columns=pval.columns)
    
    return df_normalized, pval, pval_adj



adata.obs['old_fine_annot_type'] = adata.obs['old_fine_annot_type'].astype('category')
adata.obs['EnvSDD'] = adata.obs['EnvSDD'].astype('category')

df_normalized, pval, pval_adj = enrichment_analysis(adata, id_key = 'old_fine_annot_type', 
                                                    val_key = 'EnvSDD')

# kwargs = {'figsize': (8, 8), 'vmax': 1, 'cmap': 'YlOrBr', 'linewidths': 0, 'linecolor': 'white', }
# enrichment_heatmap(cell_type_abundance = df_normalized, pval_adjust = pval, save = True,
#                    show_pval = True, kwargs=kwargs, 
#                    save_dir = 'E:/wlt_cd/MERFISH/figures/ct_domain_enrichment_heatmap.pdf')


#################  We will analysis 1 and 8 in IDC 5
#### marker gene for 1: SCGB1D2, SCGB2A2, HEBP1, KCNE4
#### marker gene for 8: MGST1, APOD, CCL19, TIMP1

counts_path = "D:/Experiment/nyq/HBC/dataset/counts_matrix.txt"
meta_path_20 = "D:/Experiment/nyq/HBC/results/meta_data_all.txt"
img_path = "D:/Experiment/nyq/HBC/dataset/V1_Breast_Cancer_Block_A_Section_1_image.tif"

adata = ad.read_text(counts_path).T
meta_data = pd.read_csv(meta_path_20, sep="\t", index_col = 0)
adata = adata[meta_data.index.tolist()]
adata.obs = meta_data
adata.obsm["spatial"] = meta_data.loc[:,["pixel_y", "pixel_x"]].values

condition4 = adata.obs['old_fine_annot_type'] == 'IDC_5'
adata_IDC5 = adata[condition4, ]
### extract 10 and 12 
condition1 = adata_IDC5.obs['EnvSDD'] == 1
condition2 = adata_IDC5.obs['EnvSDD'] == 8
# condition3 = adata_IDC5.obs['EnSDD'] == 17
# condition4 = adata.obs['old_fine_annot_type'] == 'IDC_5'


index_10_12 = np.concatenate([np.where(adata_IDC5.obs['EnvSDD'] == 1)[0], 
                             np.where(adata_IDC5.obs['EnvSDD'] == 8)[0]])

# index_10_12 = np.concatenate([np.where(adata.obs['EnSDD'] == 13)[0], np.where(adata.obs['EnSDD'] == 15)[0], np.where(adata.obs['EnSDD'] == 17)[0]])
# (adata.obs['EnSDD'] == 10).bool or (adata.obs['EnSDD'] == 12).bool
adata_sub = adata_IDC5[index_10_12,]

for col in meta_data.columns.values[16:26]:
    adata_sub.obs[col] = adata_sub.obs[col].astype("category")

adata_sub.obs.rename(columns = {'DR.SC': 'DR-SC'}, inplace = True)
adata_sub.obs['ground_truth'] = adata_sub.obs['ground_truth'].astype("category")

spatial_key = "spatial"
library_id = 'HBC' # 你的样本的name

# Image.MAX_IMAGE_PIXELS=None
# img = Image.open(img_path)
# img = np.array(img)

adata_sub.uns['{var}_colors'] = {"1": "FAE6E6","8": "EE82EE"}
adata_sub.uns.setdefault(spatial_key, {}).setdefault(library_id, {})
adata_sub.uns[spatial_key][library_id]["images"] = {"hires": None}
adata_sub.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1.,
                                                      "spot_diameter_fullres": 150}


folder_path = 'D:/Experiment/nyq/HBC/plot'
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path, exist_ok = True)  

sub_file = os.path.join(folder_path, '1_8_IDC5_plot.pdf')

plt.rcParams["figure.figsize"] = (2, 2)
# file_name_gt = os.path.join(path_no_pic, f'{col_gt}_no_pic.pdf')

###
'#%02x%02x%02x' % (172, 194, 225) #10
'#%02x%02x%02x' % (150, 204, 135) #12 
#{"10": '#acc2e1',"12": '#96cc87'}

sc.pl.spatial(adata_sub, img_key = "hires", color = 'EnvSDD', size = 1.5, show=False, 
              return_fig = True, legend_loc='on data', legend_fontsize = 8,
              legend_fontweight = 'bold', palette = ['#E6E6FA', '#EE82EE', '#FFA07A'])
plt.savefig(sub_file, bbox_inches='tight', dpi=300)
#### Sometimes need close more than one times.
plt.close()



# fig, axs = plt.subplots(2, 4, figsize = (4, 8))


# for index_pl in range(len(de_genes)):
#     if index_pl < 4:
#         row_pl = 0
#         col_pl = index_pl
#         ax = axs[row_pl, col_pl]
#         sc.pl.spatial(adata_sub, img_key = "hires", color = de_genes[index_pl], size = 1.5, 
#                       title = de_genes[index_pl], legend_loc= None, ax = ax, save=False, 
#                       colorbar_loc = None, color_map = 'viridis')
#         ax.set(xlabel=None)
#         ax.set(ylabel = None)
#         ax.set_title(de_genes[index_pl], size = 6)
#     elif 3 <= index_pl < 6:
#         row_pl = 1
#         col_pl = ((index_pl+1) % 3) - 1
#         ax = axs[row_pl, col_pl]
#         sc.pl.spatial(adata_sub, img_key = "hires", color = de_genes[index_pl], size = 1.5, 
#                       title = de_genes[index_pl], legend_loc= None, ax = ax, save=False, 
#                       colorbar_loc = None, color_map = 'viridis')
#         ax.set(xlabel=None)
#         ax.set(ylabel = None)
#         ax.set_title(de_genes[index_pl], size = 6)
        
#     else:
#         row_pl = 2
#         col_pl = ((index_pl+1) % 3) - 1
#         ax = axs[row_pl, col_pl]
#         if index_pl == 8:
#             sc.pl.spatial(adata_sub, img_key = "hires", color = de_genes[index_pl], size = 1.5, 
#                           title = de_genes[index_pl], legend_loc= None, ax = ax, save=False,  
#                           color_map = 'viridis')
#         else:
#             sc.pl.spatial(adata_sub, img_key = "hires", color = de_genes[index_pl], size = 1.5, 
#                           title = de_genes[index_pl], legend_loc= None, ax = ax, save=False,  
#                           colorbar_loc=None, color_map = 'viridis')
#         ax.set(xlabel=None)
#         ax.set(ylabel = None)
#         ax.set_title(de_genes[index_pl], size = 6)
    
#     plt.subplots_adjust(wspace=0.01, hspace = 0.01)
#     plt.tight_layout()  

sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

de_genes = ['SCGB1D2', 'SCGB2A2', 'HEBP1', 'KCNE4',
            'MGST1', 'APOD', 'CCL19', 'TIMP1']

fig, axs = plt.subplots(2, 4, figsize=(8, 4))  # 2 行 4 列

for index_pl in range(len(de_genes)):   # ✅ 修正 1
    row_pl, col_pl = divmod(index_pl, 4)
    ax = axs[row_pl, col_pl]

    gene = de_genes[index_pl]

    # ✅ 只在最后一个子图显示 colorbar
    if index_pl == len(de_genes) - 1:   # ✅ 修正 2
        sc.pl.spatial(
            adata_sub,
            img_key="hires",
            color=gene,
            size=1.5,
            ax=ax,
            legend_loc=None,
            color_map="viridis",
            show=False
        )
    else:
        sc.pl.spatial(
            adata_sub,
            img_key="hires",
            color=gene,
            size=1.5,
            ax=ax,
            legend_loc=None,
            color_map="viridis",
            colorbar_loc=None,
            show=False
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(gene, fontsize=6)

plt.tight_layout()
plt.savefig(
    "D:/Experiment/nyq/HBC/plot/spatial_DEgenes_IDC5.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


################# We will analysis 6, 11, 15 in IDC 2













