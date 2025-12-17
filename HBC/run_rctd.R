RCTD_run = function(database, CELL_MIN_INSTANCE = 20){
  ### extract SC, CL, ST from database
  sc_exp <- database$sc_exp
  sc_label <- database$sc_label
  spot_exp <- database$spot_exp
  cell_type <- sort(unique(sc_label))
  spot_loc <- database$spot_loc
  
  sparse_sc_exp <- as(sc_exp, "sparseMatrix")
  sparse_spot_exp <- as(spot_exp, "sparseMatrix")
  
  ## the reference scRNA-seq data
  cellnames <- colnames(sc_exp)
  cell_types <- as.factor(sc_label)
  names(cell_types) <- cellnames
  sc_nUMI <- as.numeric(colSums(sc_exp))
  names(sc_nUMI) <- cellnames
  reference <- spacexr::Reference(sparse_sc_exp, cell_types, nUMI = sc_nUMI)
  
  ### Create SpatialRNA object
  coords <- as.data.frame(spot_loc[,c(1,2)])
  # coords <- as.data.frame(matrix(1,dim(spot_exp)[2],2))
  rownames(coords) <- as.character(colnames(spot_exp))
  nUMI <- colSums(spot_exp)
  puck <- spacexr::SpatialRNA(coords, counts=sparse_spot_exp, nUMI=nUMI)
  
  myRCTD <- suppressMessages(spacexr::create.RCTD(puck, reference, max_cores = 1,CELL_MIN_INSTANCE = CELL_MIN_INSTANCE))
  myRCTD <- suppressMessages(spacexr::run.RCTD(myRCTD, doublet_mode = 'full'))
  results <- myRCTD@results
  
  temp <- as.matrix(results$weights)
  norm_weights_temp <- sweep(temp, 1, rowSums(temp), '/')
  RCTD_results <- norm_weights_temp[,cell_type]
  
  return(RCTD_results)
}

setwd("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3")
load("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/breastdata_rctd_all.RData")
source('D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/data_process.R')

sc_exp = breastdata_rctd$sc_exp
sc_label = breastdata_rctd$sc_lable
spot_exp = breastdata_rctd$sp_exp
spot_loc = breastdata_rctd$sp_loc

### change / to . for sc_label
#sc_label[sc_label == "macrophage/DC/monocyte"] = "macrophage_DC_monocyte"


database = data_process(sc_exp, sc_label, spot_exp, spot_loc)
rctd_res <- RCTD_run(database, CELL_MIN_INSTANCE = 20)
save(rctd_res, file = "rctd_HBC_all.RData")

write.table(rctd_res ,'D:/Experiment/nyq/Human_Breast_Cancer/rctd_res.txt', sep = '\t')

library(RColorBrewer)
library(ggplot2)
library(scatterpie)
library(cowplot)
library(gridExtra)
library(viridis)
library(rstatix)
library(ggpubr)

load("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/breastdata_rctd_all.RData")
load("D:/Experiment/nyq/Human_Breast_Cancer/deconvolution3/rctd_HBC_all.RData")

# EnDecon_res = Results.dec.mouse[[1]]
EnDecon = rctd_res
celltype = colnames(EnDecon)
# celltype

spot_loc = breastdata_rctd$sp_loc
colnames(spot_loc) <- c("x", "y", "region")
spot_loc$region <- paste0("domain", spot_loc$region)
# data("breast.spot.annotation")
# spot_loc$region = breast.spot.annotation
spot_exp = breastdata_rctd$sp_exp

#setting the color
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector  <-  unlist(mapply(brewer.pal, qual_col_pals$maxcolors, 
                              rownames(qual_col_pals)))
col_low <- "green"; col_high <- "red"



piedata_temp <- data.frame(x = spot_loc$x, y= spot_loc$y,
                           group =factor(1:length(spot_loc$x)))

pos_pie = as.data.frame(EnDecon)

piedata = cbind(piedata_temp,pos_pie)
##########
col_df <- data.frame(cell_types = celltype,
                     col_vector = col_vector[1:length(celltype)])

p1 = ggplot() + geom_scatterpie(aes(x=x, y=y, group=group),
                                cols= names(piedata)[-1:-3],
                                data = piedata,color=NA,
                                pie_scale = 1.3)+
  coord_equal() +theme_bw()+
  labs(y=" ",x=" ",title =" ")+
  theme(axis.ticks.x  = element_blank(), 
        axis.ticks.y  = element_blank(), 
        axis.text.x.bottom =element_blank(), 
        axis.text.y.left = element_blank(), 
        legend.text = element_text(size = 12,angle= 0.5), 
        legend.title = element_blank(), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank())

p2 = p1 + scale_fill_manual(values = col_df$col_vector,
                            breaks = col_df$cell_types)
# save picture
#ggsave(p2 , file='pirchart.pdf',width = 7, height = 5) 
print(p2)

### cell type abundance in different regions
anno_type = names(table(spot_loc[,3]))
# ind_caner = which(spot_loc[,3] == anno_type[3])
domain10 = which(spot_loc[,3] == anno_type[1])
domain12 = which(spot_loc[,3] == anno_type[2]) 

# immume_cell = c("B cell", "macrophage_DC_monocyte", "NK cell", "T cell")
# celltype <- immume_cell
# celltype <- colnames(EnDecon)

EnDecon_plot <- matrix(0, nrow = nrow(EnDecon), ncol = 4)
rownames(EnDecon_plot) <- rownames(EnDecon)
colnames(EnDecon_plot) <- c("Cancer Epithelial", "Macrophage", "DCs", "T cells CD4+")

EnDecon_plot[,1] <- EnDecon[,"Cancer Epithelial"]
EnDecon_plot[,2] <- EnDecon[,"Macrophage"]
EnDecon_plot[,3] <- EnDecon[,"DCs"]
# EnDecon_plot[,4] <- EnDecon[,"Cycling T-cells"]
# EnDecon_plot[,5] <- EnDecon[,"NKT cells"]
# EnDecon_plot[,6] <- EnDecon[,"NK cells"]
EnDecon_plot[,4] <- EnDecon[,"T cells CD4+"]
# EnDecon_plot[,8] <- EnDecon[,"T cells CD8+"]
# EnDecon_plot[,3] <- rowSums(EnDecon[, c("Macrophage", "DCs", "Monocyte")])
# EnDecon_plot[,4] <- EnDecon[,"NK cells"]
celltype <- colnames(EnDecon_plot)
pval = matrix(NA, length(celltype), 1)
rownames(pval) = celltype
colnames(pval) = c("comparing")
plotreg = list()
for( k in 1: length(celltype)){
  # print(k)
  cell_type_prop = EnDecon_plot[,celltype[k]]
  domain10_prop  = cell_type_prop[domain10]
  domain12_prop = cell_type_prop[domain12]
  # immu_prop = cell_type_prop[ind_immu ]
  pval[k,1] = wilcox.test( domain10_prop ,domain12_prop)$p.value
  # pval[k,2] = wilcox.test( cancer_prop ,immu_prop )$p.value
  # pval[k,3] = wilcox.test( conn_prop ,immu_prop )$p.value
  region = c("domain10","domain12")
  reg_lable =  rep(region,c(length(domain10_prop),length(domain12_prop)))
  data_regcan = data.frame(porp = c(domain10_prop, domain12_prop),
                           type = factor(reg_lable, levels = region))
  # Wilcox test
  stat.test <- data_regcan %>%
    wilcox_test(porp ~ type) %>%
    add_significance()
  bxp <- ggboxplot(data_regcan, x = "type", y = "porp", fill = "type", 
                   palette = c("#d9b1f0", "#FF99CC"))+
    theme_classic()+
    labs(y="",x=" ",title = celltype[k])+
    theme(
      axis.text.x.bottom = element_text(size = 9,hjust = 0.5,angle = 45), 
      axis.text.y.left = element_text(size = 9),
      axis.title.x = element_text(size = 9,hjust = 0.5), 
      axis.title.y = element_blank(),#element_text(size = 14),
      plot.title = element_text( size=9,hjust = 0.5),
      legend.title = element_blank(), 
      legend.position = "none",
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(colour = "black")
    ) + scale_x_discrete(
      breaks = c("domain10", "domain12"),
      label = c("domain10", "domain12")
    )
  # Box plot
  stat.test <- stat.test %>% add_xy_position(x = "type")
  plotreg [[k]]  = bxp + stat_pvalue_manual(stat.test, label = "p", 
                                            tip.length = 0.01)
}

plotregio = plot_grid(plotlist = plotreg,nrow = 1)

ggplot2::ggsave(filename = "immune_ct_1.pdf",width = 8, height = 5)
