library(Seurat)

counts_path = '/home/lihs/sp_ct/DLPFC/dataset/DLPFC151507/counts_matrix.txt'
meta_path = '/home/lihs/sp_ct/DLPFC/dataset/DLPFC151507/meta_data.txt'
ref_path = '/home/lihs/sp_ct/DLPFC/dataset/reference/DLPFC_ref.rds'

counts = read.table(counts_path, sep = '\t')
counts_ST <- as.matrix(counts)



coordinates <- data.frame(read.table(loc_path, sep = '\t'))

colnames(counts_ST) <- rownames(coordinates)

#coordinates <- read.csv(file = loc_path) 
#coordinates <- as.data.frame(coordinates)
#rownames(coordinates) <- coordinates$X
#coordinates$X <- NULL

### create Seurat object 
suppressWarnings({Seurat.ST <- Seurat::CreateSeuratObject(counts = counts_ST, assay = "Spatial", 
                                                          meta.data = coordinates)})
Seurat.ST <- Seurat::SCTransform(Seurat.ST, assay = "Spatial", verbose = FALSE)

### select HVG genes
Seurat.ST <- Seurat::FindVariableFeatures(Seurat.ST, nfeatures = n_HVG, verbose = FALSE)
### PCA 
Seurat.ST <- Seurat::ScaleData(Seurat.ST, verbose = FALSE)
Seurat.ST <- Seurat::RunPCA(Seurat.ST, npcs = n_PCA, verbose = FALSE)


