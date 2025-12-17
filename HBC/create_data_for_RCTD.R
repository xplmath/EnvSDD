setwd("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3")

counts_path <- "D:/Experiment/nyq/Human_Breast_Cancer/counts_matrix.txt"
loc_path <- "D:/Experiment/nyq/Human_Breast_Cancer/clustering.res.20.txt"
img_path <- "D:/Experiment/nyq/Human_Breast_Cancer/V1_Breast_Cancer_Block_A_Section_1_image.tif"
python_env <- "/home/lihs/.conda/envs/bio_software/bin/python"
data_process <- function(counts_path, loc_path, img_path, n_HVG = 2000, n_PCA = 20){
  
  ### input counts file
  counts = read.table(counts_path, sep = '\t')
  counts_ST <- as.matrix(counts)
  
  ### input location file
  coordinates <- data.frame(read.table(loc_path, sep = '\t'))
  
  colnames(counts_ST) <- rownames(coordinates)
  
  
  ### create Seurat object
  suppressWarnings({Seurat.ST <- Seurat::CreateSeuratObject(counts = counts_ST, assay = "Spatial",
                                                            meta.data = coordinates)})
  Seurat.ST <- Seurat::SCTransform(Seurat.ST, assay = "Spatial", verbose = FALSE)
  
  ### select HVG genes
  Seurat.ST <- Seurat::FindVariableFeatures(Seurat.ST, nfeatures = n_HVG, verbose = FALSE)
  ### PCA
  Seurat.ST <- Seurat::ScaleData(Seurat.ST, verbose = FALSE)
  Seurat.ST <- Seurat::RunPCA(Seurat.ST, npcs = n_PCA, verbose = FALSE)
  
  data.list <- list(counts_path = counts_path, loc_path = loc_path, img_path = img_path)
  Seurat.ST@images <- data.list
  return(Seurat.ST)
}


# source("/home/lihs/bio_software/code/utils.R")
Seurat.data <- data_process(counts_path, loc_path, img_path, n_HVG = 2000, n_PCA = 20)

# index.domain10_12 = Seurat.data$EnSDD == "10" | Seurat.data$EnSDD == "12"
# 
counts_ST <- as.matrix(Seurat.data@assays$SCT@counts)
meta_data_de <- Seurat.data@meta.data


#### load Wu scRNA-seq dataset
library(Matrix)
count_sc <- Matrix::readMM("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar/Wu_etal_2021_BRCA_scRNASeq/count_matrix_sparse.mtx")
gene_sc <- read.table("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar/Wu_etal_2021_BRCA_scRNASeq/count_matrix_genes.tsv")
cell_sc <- read.table("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar/Wu_etal_2021_BRCA_scRNASeq/count_matrix_barcodes.tsv")
rownames(count_sc) <- gene_sc$V1
colnames(count_sc) <- cell_sc$V1

meta_data_sc <- read.csv("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar/Wu_etal_2021_BRCA_scRNASeq/metadata.csv")
rownames(meta_data_sc) <- meta_data_sc[,1]
meta_data_sc$Cell_ID <- NULL

index_cell <- colSums(count_sc) >= 200
count_sc <- count_sc[,index_cell]
meta_data_sc <- meta_data_sc[index_cell,]

index_gene <- rowSums(count_sc) >= 0
count_sc <- count_sc[index_gene,]


library(Seurat)
Seurat_sc <- CreateSeuratObject(counts = count_sc, meta.data = meta_data_sc)
Seurat_sc <- Seurat::NormalizeData(Seurat_sc)
Seurat_sc <- Seurat::FindVariableFeatures(Seurat_sc, nfeatures = 5000)

sc_hvg <- VariableFeatures(Seurat_sc)
counts_sc_hvg <- as.matrix(Seurat_sc@assays$RNA$counts)[sc_hvg,]
ct_sc_major <- Seurat_sc$celltype_major[colnames(counts_sc_hvg)]
ct_sc_minor <- Seurat_sc$celltype_minor[colnames(counts_sc_hvg)]

### T-cells: Cycling T-cells, NKT cells,  NK cells, T cells CD4+, T cells CD8+
### Myeloid: Cycling_Myeloid, Monocyte, Macrophage, DCs
ct_sc <- ct_sc_major
library(dplyr)
index_T_cells <- ct_sc_minor %in% c("Cycling T-cells", "NKT cells",  "NK cells", "T cells CD4+", "T cells CD8+")
ct_sc[index_T_cells] <- ct_sc_minor[index_T_cells]
index_Myeloid <- ct_sc_minor %in% c("Cycling_Myeloid", "Monocyte", "Macrophage", "DCs")
ct_sc[index_Myeloid] <- ct_sc_minor[index_Myeloid]


common_genes <- intersect(rownames(counts_ST), rownames(counts_sc_hvg))
meta_data_rctd <- meta_data_de[,c(4, 5, 23)]

breastdata_rctd <- list()
breastdata_rctd$sp_loc <- meta_data_rctd
breastdata_rctd$sc_exp <- counts_sc_hvg[common_genes,]
breastdata_rctd$sp_exp <- counts_ST[common_genes,]
breastdata_rctd$sc_lable <- ct_sc


save(breastdata_rctd, file = "breastdata_rctd_all.RData")
