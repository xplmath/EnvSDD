setwd("D:/Experiment/nyq/HBC/plot/")

counts_path <- "D:/Experiment/nyq/HBC/dataset/counts_matrix.txt"
loc_path <- "D:/Experiment/nyq/HBC/results/meta_data_all.txt"
img_path <- 'D:/Experiment/nyq/HBC/dataset/V1_Breast_Cancer_Block_A_Section_1_image.tif'

source("D:/Experiment/nyq/HBC/utiles.R")
Seurat.data <- data_process(counts_path, loc_path, img_path, n_HVG = 3000, n_PCA = 30)

Seurat.data_IDC5 <- Seurat.data[, Seurat.data$old_fine_annot_type == 'IDC_5']

index.domain1_8 = c(which(Seurat.data_IDC5$EnvSDD == '1'), which(Seurat.data_IDC5$EnvSDD == '8'))

# 
counts_ST <- as.matrix(Seurat.data@assays$SCT@counts)[, names(index.domain1_8)]
meta_data_de <- Seurat.data@meta.data[names(index.domain1_8),]

suppressWarnings({Seurat.ST.DE <- Seurat::CreateSeuratObject(counts = counts_ST, assay = "Spatial", 
                                                             meta.data = meta_data_de)})
Seurat.ST.DE <- Seurat::SCTransform(Seurat.ST.DE, assay = "Spatial", verbose = FALSE)

### select HVG genes
Seurat.ST.DE <- Seurat::FindVariableFeatures(Seurat.ST.DE, nfeatures = 3000, verbose = FALSE)
### PCA 
Seurat.ST.DE <- Seurat::ScaleData(Seurat.ST.DE, verbose = FALSE)
Seurat.ST.DE <- Seurat::RunPCA(Seurat.ST.DE, npcs = 30, verbose = FALSE)
### UMAP of subtype
# Seurat.ST.DE <- Seurat::RunUMAP(Seurat.ST.DE, reduction = "pca", dims = 1:20)
# p1 <- Seurat::DimPlot(Seurat.ST.DE, reduction = "umap", group.by = "EnSDD", 
#                       pt.size = 3)


labels_de <- Seurat.ST.DE@meta.data$EnvSDD
markers <- Seurat::FindMarkers(Seurat.ST.DE, ident.1 = levels(as.factor(labels_de))[1], 
                            ident.2 = levels(as.factor(labels_de))[2], 
                            group.by = "EnvSDD", test.use = "wilcox", logfc.threshold = 0.25)

library(dplyr)
library(ggplot2)

markers$gene <- rownames(markers)

# 根据阈值给基因分组（可根据自己需求调整）
library(dplyr)
library(ggplot2)
library(ggrepel)

## 阈值可以自己调
logfc_cutoff <- 1
padj_cutoff  <- 1e-5

# 整理数据：计算 -log10(p_adj) 和显著性分组
markers <- markers %>%
  mutate(
    negLog10P = -log10(p_val_adj),
    sig = case_when(
      p_val_adj < padj_cutoff & avg_log2FC >=  logfc_cutoff ~ "Up-regulated",
      p_val_adj < padj_cutoff & avg_log2FC <= -logfc_cutoff ~ "Down-regulated",
      TRUE                                                   ~ "NS"
    ),
    sig = factor(sig, levels = c("Down-regulated", "NS", "Up-regulated"))
  )

# 选取要标注的 top 基因（只在显著基因中挑）
top_genes <- markers %>%
  filter(sig != "NS") %>%
  arrange(p_val_adj) %>%
  slice_head(n = 10)

p <- ggplot(markers, aes(x = avg_log2FC, y = negLog10P)) +
  ## 先画非显著点（灰色、透明一点）
  geom_point(
    data = subset(markers, sig == "NS"),
    color = "grey80",
    size  = 1.2,
    alpha = 0.6
  ) +
  ## 再画显著点（颜色醒目）
  geom_point(
    data = subset(markers, sig != "NS"),
    aes(color = sig),
    size  = 1.6,
    alpha = 0.9
  ) +
  ## 颜色：选色盲友好、对比明显
  scale_color_manual(
    values = c(
      "Down-regulated" = "#1f78b4",  # 蓝
      "Up-regulated"   = "#e31a1c"   # 红
    )
  ) +
  ## 阈值线（logFC & p 值）
  geom_vline(
    xintercept = c(-logfc_cutoff, logfc_cutoff),
    linetype   = "dashed",
    linewidth  = 0.4,
    color      = "grey50"
  ) +
  geom_hline(
    yintercept = -log10(padj_cutoff),
    linetype   = "dashed",
    linewidth  = 0.4,
    color      = "grey50"
  ) +
  ## 标注 top 基因
  geom_text_repel(
    data          = top_genes,
    aes(label = gene),
    size          = 3.5,
    max.overlaps  = 50,
    box.padding   = 0.4,
    point.padding = 0.2,
    segment.size  = 0.3,
    segment.color = "grey50",
    min.segment.length = 0
  ) +
  ## 轴标题 & 图例标题
  labs(
    x     = expression(log[2]~fold~change),
    y     = expression(-log[10]~adjusted~P),
    color = NULL
  ) +
  ## 期刊常用的简洁主题
  theme_classic(base_size = 14) +
  theme(
    axis.title   = element_text(face = "bold"),
    axis.text    = element_text(color = "black"),
    legend.position   = "top",
    legend.direction  = "horizontal",
    legend.text       = element_text(size = 12),
    plot.margin       = margin(5.5, 12, 5.5, 5.5),
    panel.border      = element_rect(colour = "black", fill = NA, linewidth = 0.5)
  )

p

## 保存为高分辨率图片（投稿用）
ggsave(
  filename = "D:/Experiment/nyq/HBC/plot/IDC5_marker.pdf",
  plot     = p,
  width    = 5,
  height   = 4,
  units    = "in",
  dpi      = 300
)

### HEBP1, FAM234b, CSTA, TMSB10
Loc_G_mat <- matrix(0, nrow = ncol(Seurat.ST.DE), ncol = 8)
rownames(Loc_G_mat) <- colnames(Seurat.ST.DE)
colnames(Loc_G_mat) <- c('SCGB1D2', 'SCGB2A2', 'HEBP1', 'KCNE4','MGST1', 'APOD', 'CCL19', 'TIMP1')

for (i in 1:ncol(Loc_G_mat)) {
  Loc_G_mat[,i] <- LocalG_spa(Seurat_data = Seurat.ST.DE, gene_name = colnames(Loc_G_mat)[i], k = 4)
  
}
write.table(Loc_G_mat, file = "D:/Experiment/nyq/HBC/results/Loc_G_mat_IDC5_1_8.txt", sep = "\t")

Loc_G_df <- data.frame(locG = as.vector(Loc_G_mat), 
                       gene = rep(colnames(Loc_G_mat), each = nrow(Loc_G_mat)),
                       label = as.character(rep(Seurat.ST.DE$EnvSDD, time = ncol(Loc_G_mat))))
library(ggplot2)
ggplot_LocG <- ggplot(Loc_G_df, aes(x = gene, y = locG, fill = label)) + 
  geom_boxplot() +
  theme_classic()+
  labs(y="LGO's Gi", title = "Local Getis and Ord's Gi")+
  theme(
    axis.text.x.bottom = element_text(size = 14,hjust = 1,angle =45), 
    axis.text.y.left = element_text(size = 14),
    axis.title.x = element_blank(), 
    axis.title.y = element_blank(),#element_text(size = 14),
    plot.title = element_text( size=14,hjust = 0.5),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    # legend.position = "none",
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(colour = "black")
  ) 

ggsave(ggplot_LocG, filename = "D:/Experiment/nyq/HBC/plot/LocG_box_IDC5_1_8.pdf", width = 10, height = 6)


##################











#### enrichment analysis
# library(biomaRt)
# library(clusterProfiler)
# library(dplyr)
# library(org.Hs.eg.db)
# library(ggplot2)
# 
# DE.genes <- toupper(res.de$gene_names)
# hmark = biomaRt::useEnsembl(biomart ="ensembl", dataset = "hsapiens_gene_ensembl", mirror = "asia")
# tmp <- biomaRt::getBM(attributes = c("hgnc_symbol", "entrezgene_id"), filters = "hgnc_symbol",
#                       values = DE.genes, mart = hmark, uniqueRows = TRUE)
# 
# #### For pathway analysis 
# tmp <- dplyr::filter(tmp, entrezgene_id != "NA")
# tmp.unique <- tmp[!duplicated(tmp$hgnc_symbol),]
# rownames(res.de) <- res.de$gene_names
# res.de <- res.de[tmp.unique$hgnc_symbol,]
# res.de$gene_names <- tmp.unique$entrezgene_id
# 
# genelist <- res.de$log2FC
# names(genelist) <- res.de$gene_names
# genelist = sort(genelist, decreasing = TRUE)
# 
# ego2 <- clusterProfiler::gseGO(geneList = genelist, 
#                                ont = "BP",
#                                OrgDb = org.Hs.eg.db,
#                                pvalueCutoff = 0.05,
#                                pAdjustMethod = "BH")
# 
# # ego2 <- clusterProfiler::gseKEGG(geneList = genelist, # ordered named vector of fold changes (Entrez IDs are the associated names)
# #                                  organism = "hsa", # supported organisms listed below
# #                                  nPerm = 1000, # default number permutations
# #                                  verbose = TRUE)
# #head(ego2, 3) 
# library(enrichplot)
# library(dplyr)
# 
# 
# # %>%  barplot(x = "qscore") + xlab(expression(-log[10](FDR)))
# ### show header 9 items 
# all_items <- ego2@result[order(ego2@result$p.adjust, decreasing = FALSE), ]
# ggplot2_enrich = all_items[1:10,]
# ggplot2_enrich$log_q.value = -log10(ggplot2_enrich$p.adjust)
# tt = expression(-log[10](FDR))
# 
# ggplot2_enrich$Description <- factor(ggplot2_enrich$Description, 
#                                      levels = ggplot2_enrich$Description[order(ggplot2_enrich$log_q.value, decreasing = FALSE)])
# 
# ggplot2_enrich$ID <- factor(ggplot2_enrich$ID, 
#                             levels = ggplot2_enrich$ID[order(ggplot2_enrich$log_q.value, decreasing = FALSE)])
# 
# plot_enrich <- ggplot(data = ggplot2_enrich, aes(x = log_q.value, y = ID)) +
#   geom_bar(stat="identity", width = 0.7) +
#   theme_classic() + 
#   labs(x = tt, y = '') +
#   theme(
#     axis.text.x.bottom = element_text(size = 10),
#     axis.text.y = element_text(size = 10),
#     # axis.title.x = element_text(size = 10),
#     # axis.title.y = element_text(size = 10),
#     # axis.ticks.x = element_blank(),
#     plot.title = element_text( size=14,hjust = 0.5),
#     legend.position = "none",
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank(),
#     panel.border = element_blank(),
#     axis.line = element_line(colour = "black")
#   )
