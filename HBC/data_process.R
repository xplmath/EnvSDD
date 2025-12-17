data_process <- function(sc_exp, sc_label, spot_exp, spot_loc,
                         gene_det_in_min_cells_per = 0.01, expression_threshold = 1,
                         nUMI = 100, verbose = FALSE, plot= FALSE){
  if(ncol(sc_exp) != length(sc_label))
    stop("Require cell labels!")
  
  if(ncol(spot_exp) != nrow(spot_loc))
    stop("Require x , y coordinations")
  
  #### scRNA-seq data process
  sc_matrix = t(cleanCounts(t(sc_exp), gene_det_in_min_cells_per = gene_det_in_min_cells_per,
                            expression_threshold = expression_threshold, nUMI = nUMI,
                            verbose = verbose, plot = plot))
  sc_matrix= as.matrix(sc_matrix)
  ind = match(colnames(sc_matrix), colnames(sc_exp))
  sc_label = sc_label[ind]
  # cell_type = sort(unique(sc_label))
  
  #### ST data process
  st_matrix = t(cleanCounts(t(spot_exp), gene_det_in_min_cells_per = gene_det_in_min_cells_per,
                            expression_threshold = expression_threshold, nUMI = nUMI,
                            verbose = verbose, plot = plot))
  st_matrix= as.matrix(st_matrix)
  ind_sp = match(colnames(st_matrix), colnames(spot_exp))
  spot_loc = spot_loc[ind_sp, ]
  
  #### find common genes
  com_gene = intersect(rownames(sc_matrix),rownames(st_matrix))
  sc_exp = sc_matrix[com_gene,]
  st_exp = st_matrix[com_gene,]
  
  ### rechecking nUMI
  index_sc <- colSums(sc_exp) >= nUMI
  sc_exp_filter <- sc_exp[,index_sc]
  sc_label_filter <- sc_label[index_sc]
  
  index_st <- colSums(st_exp) >= nUMI
  st_exp_filter = st_exp[,index_st]
  spot_loc_filter <- spot_loc[index_st,]
  
  database <- list(sc_exp = sc_exp_filter, sc_label = sc_label_filter,
                   spot_exp = st_exp_filter, spot_loc = spot_loc_filter)
  return(database)
}

cleanCounts <- function (counts, gene_det_in_min_cells_per = 0.01,
                         expression_threshold = 1 ,
                         nUMI = 100,
                         verbose = FALSE, plot= FALSE) {
  n = nrow(counts)
  ##### select of the genes
  filter_index_genes = Matrix::colSums(counts >= expression_threshold) >=
    gene_det_in_min_cells_per*n
  
  #### filter the cell
  filter_index_cells = Matrix::rowSums(counts[,filter_index_genes] >=
                                         expression_threshold) >= nUMI
  
  counts = counts[filter_index_cells, filter_index_genes]
  
  if (verbose) {
    message("Resulting matrix has ", nrow(counts), " cells and ", ncol(counts), " genes")
  }
  if (plot) {
    par(mfrow=c(1,2), mar=rep(5,4))
    hist(log10(Matrix::rowSums(counts)+1), breaks=20, main='Genes Per Dataset')
    hist(log10(Matrix::colSums(counts)+1), breaks=20, main='Datasets Per Gene')
  }
  return(counts)
}

####### Utils of data process of DWLS ##########
create_group_exp <- function(sc_exp,sc_label) {
  
  #sc_exp single cell gene expression datasets
  #sc_label  cell annotation of the single cells of the reference
  
  ##group cells
  # reference matrix (C) + refProfiles.var from TRAINING dataset
  cell_type = sort(unique(sc_label))
  group = list()
  for(i in 1:length(cell_type)){
    temp_use <- which(sc_label == cell_type[i])
    names(temp_use) <- NULL
    group[[i]] <- temp_use
  }
  sc_group_exp = sapply(group,function(x) Matrix::rowMeans(sc_exp[,x]))
  #sapply
  sc_group_exp = as.matrix(sc_group_exp)
  colnames(sc_group_exp) = cell_type
  return(sc_group_exp)
}