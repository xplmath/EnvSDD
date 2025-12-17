library(ggplot2)
library(scatterpie)
library(RColorBrewer)

# EnDecon:  行 = spots，列 = 细胞类型比例
# spot_loc: 至少包含三列：x, y, region（region 可选）
plot_rctd_spatial_pie <- function(EnDecon,
                                  spot_loc,
                                  pie_scale = 1.3,
                                  palette_category = "qual",
                                  legend_text_size = 12) {
  # 1. 确保列名正确
  if (!all(c("x", "y") %in% colnames(spot_loc))) {
    stop("spot_loc 需要包含列名 'x' 和 'y'（坐标）")
  }
  
  # 2. 细胞类型名称
  celltype <- colnames(EnDecon)
  
  # 3. 生成颜色向量
  qual_col_pals <- brewer.pal.info[brewer.pal.info$category == palette_category, ]
  col_vector <- unlist(
    mapply(
      brewer.pal,
      qual_col_pals$maxcolors,
      rownames(qual_col_pals)
    )
  )
  if (length(col_vector) < length(celltype)) {
    warning("可用颜色数量少于细胞类型数量，部分颜色会重复。")
  }
  col_df <- data.frame(
    cell_types = celltype,
    col_vector = col_vector[1:length(celltype)]
  )
  
  # 4. 组织 scatterpie 所需数据
  piedata_temp <- data.frame(
    x = spot_loc$x,
    y = spot_loc$y,
    group = factor(seq_len(nrow(spot_loc)))
  )
  pos_pie <- as.data.frame(EnDecon)
  piedata <- cbind(piedata_temp, pos_pie)
  
  # 5. 画图主体
  p <- ggplot() +
    geom_scatterpie(
      aes(x = x, y = y, group = group),
      cols  = names(piedata)[-c(1:3)],  # 除 x,y,group 外的列作为扇区
      data  = piedata,
      color = NA,
      pie_scale = pie_scale
    ) +
    coord_equal() +
    theme_bw() +
    labs(x = " ", y = " ", title = " ") +
    theme(
      axis.ticks.x  = element_blank(),
      axis.ticks.y  = element_blank(),
      axis.text.x.bottom = element_blank(),
      axis.text.y.left   = element_blank(),
      legend.text  = element_text(size = legend_text_size, angle = 0),
      legend.title = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border     = element_blank(),
      axis.line        = element_blank()
    ) +
    scale_fill_manual(
      values = col_df$col_vector,
      breaks = col_df$cell_types
    )
  
  return(p)
}

setwd("D:/Experiment/nyq/HBC/deconvolution/")
# database151507_rctd <- readRDS("D:/Experiment/nyq/DLPFC/dataset/database151507_rctd.rds")
load("D:/Experiment/nyq/HBC/deconvolution/breastdata_rctd_all.RData")
load("D:/Experiment/nyq/HBC/deconvolution/rctd_HBC_all.RData")

# ct_abundance <- read.table('rctd_HBC_all.RData', sep = '\t', header = TRUE)
# rownames(ct_abundance) <- ct_abundance$X
# ct_abundance$X <- NULL


# counts_mat <- read.table('DLPFC151674/counts_matrix.txt', sep = '\t')
# counts_mat <- as.matrix(counts_mat)
# 
# meta_data <- read.table('DLPFC151674/meta_data.txt', sep = '\t', header = TRUE)
# meta_data<- meta_data[rownames(ct_abundance), ]

meta_data <- breastdata_rctd$sp_loc
spot_loc <- meta_data[,c('x', 'y')]
colnames(spot_loc) <- c('y', 'x')
# spot_loc$x <- -spot_loc$x
spot_loc$y <- - spot_loc$y


spot_loc <- spot_loc[rownames(rctd_res),]


# spot_loc <- meta_data[,c('pixel_y', 'pixel_x')]
# colnames(spot_loc) <- c('x', 'y')
# spot_loc$y <- -spot_loc$y

### 151507

p <- plot_rctd_spatial_pie(EnDecon = rctd_res, spot_loc = spot_loc, pie_scale = 0.3)
ggsave(p , file='pirchart_all.pdf',width = 8, height = 8) 



