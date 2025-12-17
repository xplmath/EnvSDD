setwd("D:/Experiment/nyq/HBC/results/")
base_dir <- getwd()
meta_dir <- "D:/Experiment/nyq/HBC/dataset/meta_data.txt"

# ----------------------------------
# 加载依赖包
# ----------------------------------
library(mclust)
library(aricode)
library(dplyr)

# ----------------------------------
# 初始化结果表
# ----------------------------------
metrics_df <- data.frame(
  Sample = character(),
  ARI = numeric(),
  NMI = numeric(),
  stringsAsFactors = FALSE
)

meta_data <- read.table(meta_dir, sep = "\t", header = TRUE)
low_dim_0 <- read.table('D:/Experiment/nyq/HBC/results/Z_0.txt', sep = "\t", header = TRUE)
rownames(low_dim_0) <- low_dim_0$X
low_dim_0$X <- NULL


low_dim_07 <- read.table('D:/Experiment/nyq/HBC/results/Z_07.txt', sep = "\t", header = TRUE)
rownames(low_dim_07) <- low_dim_07$X
low_dim_07$X <- NULL


meta_data <- meta_data[rownames(low_dim_0), ]


num_setting <- 20
labl_all0 <- Mclust(data = low_dim_0, G = num_setting, modelNames = "EEE")
label_mclust0 <- labl_all0$classification

labl_all07 <- Mclust(data = low_dim_07, G = num_setting, modelNames = "EEE")
label_mclust07 <- labl_all07$classification

# ARI_tmp <- ARI(meta_data[names(label_mclust0), "fine_annot_type"], label_mclust0)

meta_data$EnvSDD00 <- label_mclust0
meta_data$EnvSDD07 <- label_mclust07

write.table(meta_data, 'D:/Experiment/nyq/HBC/results/meta_data_EnvSDD.txt', sep = "\t")

#meta_data_EnvSDD <- read.table('D:/Experiment/nyq/HBC/results/meta_data_EnvSDD.txt', sep = '\t')

# ---- (4) 计算 ARI & NMI ----


