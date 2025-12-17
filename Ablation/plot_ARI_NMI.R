# -------------------------------
# 1. 加载依赖包
# -------------------------------
library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------------
# 2. 设置路径
# -------------------------------
base_dir <- "D:/Experiment/nyq/DLPFC/results"  # 修改为 fusion 文件夹所在目录
fusion_list <- sprintf("fusion_%.1f", seq(0.1, 1.0, by = 0.1))  # 生成 fusion_0.1 ~ fusion_1.0

# -------------------------------
# 3. 读取所有 metrics_summary.txt 文件
# -------------------------------
all_metrics <- data.frame()

for (fusion_folder in fusion_list) {
  metrics_path <- file.path(base_dir, fusion_folder, "metrics_summary.txt")
  
  if (file.exists(metrics_path)) {
    df <- read.table(metrics_path, header = TRUE, sep = "", stringsAsFactors = FALSE)
    df$fusion_lambda <- sub("fusion_", "", fusion_folder)  # 提取数值部分，如 0.1
    all_metrics <- rbind(all_metrics, df)
  } else {
    cat("⚠️ 文件不存在:", metrics_path, "\n")
  }
}

# 检查汇总结果
head(all_metrics)

summary_stats <- all_metrics %>%
  group_by(fusion_lambda) %>%
  summarise(
    ARI_mean   = mean(ARI, na.rm = TRUE),
    ARI_median = median(ARI, na.rm = TRUE),
    ARI_var    = var(ARI, na.rm = TRUE),
    NMI_mean   = mean(NMI, na.rm = TRUE),
    NMI_median = median(NMI, na.rm = TRUE),
    NMI_var    = var(NMI, na.rm = TRUE)
  ) %>%
  arrange(as.numeric(fusion_lambda))  # 按数值排序

# -------------------------------
# 4. 整理为长格式以便 ggplot2 绘图
# -------------------------------
df_long <- all_metrics %>%
  pivot_longer(cols = c("ARI", "NMI"), names_to = "Metric", values_to = "Value")

# 将 fusion_lambda 按数值排序
df_long$fusion_lambda <- factor(df_long$fusion_lambda, 
                                levels = sprintf("%.1f", seq(0.1, 1.0, by = 0.1)))

# -------------------------------
# 5. 绘制 Boxplot
# -------------------------------
p <- ggplot(df_long, aes(x = fusion_lambda, y = Value, fill = Metric)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7, outlier.shape = 21) +
  scale_fill_manual(values = c("steelblue", "orange")) +
  labs(
    title = "ARI & NMI across Fusion Lambda Values",
    x = "Fusion Lambda (λ)",
    y = "Score",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 0)
  )

# 显示图形
print(p)

# -------------------------------
# 6. 可选：保存图形
# -------------------------------
ggsave(file.path(base_dir, "Fusion_ARI_NMI_Boxplot.pdf"), p, width = 9, height = 6, dpi = 300)

####### we think lambda= 0.7 will be the best