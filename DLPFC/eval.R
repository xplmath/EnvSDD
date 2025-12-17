
setwd("D:/Experiment/nyq/DLPFC/results/fusion_0.7")
base_dir <- getwd()
meta_dir <- "D:/Experiment/nyq/DLPFC/dataset"

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

# ----------------------------------
# 遍历样本文件夹
# ----------------------------------
folders <- list.dirs(base_dir, recursive = FALSE)

for (folder in folders) {
  folder_name <- basename(folder)
  cat("\n===== Dealing with sample:", folder_name, "=====\n")
  
  # ---- (1) 读取 Z.txt ----
  file_path <- file.path(folder, paste0(folder_name, "_Z.txt"))
  if (!file.exists(file_path)) {
    cat("⚠️ 找不到 Z 文件:", file_path, "\n")
    next
  }
  low_dim <- read.table(file_path, sep = "\t", header = TRUE)
  rownames(low_dim) <- low_dim$X
  low_dim$X <- NULL
  
  # ---- (2) 读取 meta_data.txt ----
  meta_path <- file.path(meta_dir, folder_name, "meta_data.txt")
  if (!file.exists(meta_path)) {
    cat("⚠️ 找不到 meta_data 文件:", meta_path, "\n")
    next
  }
  meta_data <- read.table(meta_path, sep = "\t", header = TRUE)
  
  # 对齐行顺序
  meta_data_filter <- meta_data[rownames(low_dim), , drop = FALSE]
  
  # ---- (3) Mclust 聚类 ----
  num_setting <- length(unique(meta_data$layer))
  labl_all <- Mclust(data = low_dim, G = num_setting, modelNames = "EEE")
  label_mclust <- labl_all$classification
  
  # ---- (4) 计算 ARI & NMI ----
  ARI_tmp <- ARI(meta_data_filter[names(label_mclust), "layer"], label_mclust)
  NMI_tmp <- NMI(meta_data_filter[names(label_mclust), "layer"], label_mclust)
  
  # 保存到 metrics_df
  metrics_df <- rbind(metrics_df, data.frame(
    Sample = folder_name,
    ARI = ARI_tmp,
    NMI = NMI_tmp
  ))
  
  # ---- (5) 把聚类结果并入 meta_data_filter ----
  meta_data_filter$Mclust_Label <- label_mclust[rownames(meta_data_filter)]
  
  # ---- (6) 保存新的 meta_data 文件 ----
  out_path <- file.path(folder, paste0(folder_name, "_meta_with_mclust.txt"))
  write.table(meta_data_filter, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
  cat("✅ 已保存聚类标签文件到:", out_path, "\n")
}



write.table(metrics_df, "D:/Experiment/nyq/DLPFC/results/fusion_0.7/metrics_summary.txt",
            sep = "\t", quote = FALSE, row.names = FALSE)

