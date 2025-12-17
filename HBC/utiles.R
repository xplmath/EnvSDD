data_process <- function(counts_path, loc_path, img_path, n_HVG = 3000, n_PCA = 30){
  
  ### input counts file
  counts = read.table(counts_path, sep = '\t')
  counts_ST <- as.matrix(counts)
  
  ### input location file
  coordinates <- data.frame(read.table(loc_path, sep = '\t', header = TRUE))
  rownames(coordinates) <-  gsub('-', '.', coordinates$X) 
  coordinates$X <- NULL
  
  counts_ST <- counts_ST[,rownames(coordinates)]
  
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


#' Local Getis and Ord's Gi for measure the spatial autocorrelation of selected gene
#'
#' @importFrom spdep localG knearneigh knn2nb
#'
#' @param Seurat_data a SeuratObject created by data_process function in EnSDD R package
#' @param gene_name a selected gene for the calculation of Local Getis and Ord's Gi
#' @param k  the number of nearest neighbours to be returned in the KNN
#'
#' @return a vector represents the Local Getis and Ord's Gi value for a selected gene across samples
#'
#' @export


LocalG_spa <- function(Seurat_data, gene_name, k = 6){
  
  normalized_exp_data <-  as.matrix(Seurat_data@assays$SCT@data)
  df <- Seurat_data@meta.data[, c("x", "y")][colnames(normalized_exp_data),]
  
  xycoords <- cbind(df$x, df$y)
  require(spdep)
  knn_object <- spdep::knn2nb(spdep::knearneigh(xycoords, k = k))
  w <- spdep::nb2listw(knn_object, style = "W")
  
  
  values <- normalized_exp_data[gene_name, ]
  G_value <- spdep::localG(values, w)
  
  G_value_re_order <- G_value[colnames(normalized_exp_data)]
  
  return(G_value)
  
}
