###############################################################
# Shared-beta0 version
# Modified so that each TCGA project uses the cell types
# explicitly specified in celltype_list_by_project[[project_cur]]
# when building W0.
#
# NEW:
#   - Add a switch parameter: use_gene_weight
#   - If TRUE, use gene_weight in weighted loss
#   - If FALSE, use the original uniform-weight loss
###############################################################

# -----------------------------
# Basic setup
# -----------------------------
work_dir <- "E:/TCGA_GDC"
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
setwd(work_dir)

suppressPackageStartupMessages({
  library(CVXR)
  library(osqp)
  library(Matrix)
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(edgeR)
})

# -----------------------------
# User switch
# -----------------------------
use_gene_weight <- TRUE   # TRUE: weighted loss; FALSE: original uniform loss

# -----------------------------
# Project list
# -----------------------------
TCGA_project_list <- c(
  "TCGA-BRCA",
  "TCGA-COAD",
  "TCGA-LIHC",
  "TCGA-LUAD",
  "TCGA-LUSC",
  "TCGA-SKCM",
  "TCGA-OV",
  "TCGA-PRAD",
  "TCGA-UCEC"
)

# -----------------------------
# Load loading matrices
# -----------------------------
Loading_intrinsic_use <- read.csv(
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/loading_intrinsic_use.csv",
  row.names = 1,
  check.names = FALSE
)

loading_sender_use <- read.csv(
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/loading_sender_use.csv",
  row.names = 1,
  check.names = FALSE
)

loading_receiver_use <- read.csv(
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/loading_receiver_use.csv",
  row.names = 1,
  check.names = FALSE
)

loading_MI <- loading_sender_use + loading_receiver_use
genename_list <- colnames(Loading_intrinsic_use)

# -----------------------------
# Build gene_weight
# (used only when use_gene_weight = TRUE)
# -----------------------------
eps_weight <- 1e-12

loading_MI_colsum <- colSums(loading_MI)
loading_MI_norm <- loading_MI / matrix(
  loading_MI_colsum + eps_weight,
  nrow = nrow(loading_MI),
  ncol = ncol(loading_MI),
  byrow = TRUE
)

KL_divergence <- function(p, q) {
  p <- p + 1e-10
  q <- q + 1e-10
  sum(p * log(p / q))
}

KL_divergence_vec <- numeric(ncol(loading_MI_norm))
uniform_vec <- rep(1 / nrow(loading_MI_norm), nrow(loading_MI_norm))

for (gene_index in seq_len(ncol(loading_MI_norm))) {
  loading_MI_norm_cur <- loading_MI_norm[, gene_index]
  KL_divergence_vec[gene_index] <- KL_divergence(loading_MI_norm_cur, uniform_vec)
}

gene_weight <- KL_divergence_vec / sum(KL_divergence_vec)
names(gene_weight) <- colnames(loading_MI)

# -----------------------------
# Project-specific cell types to use in W0
# IMPORTANT: do not overwrite this object later
# -----------------------------
celltype_list_by_project <- list(
  "TCGA-BRCA" = rownames(Loading_intrinsic_use)[c(1, 2, 3, 4, 6, 7, 9, 12, 13, 17)],
  "TCGA-COAD" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 5, 6, 7, 9, 12, 13, 17)],
  "TCGA-LIHC" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 17)],
  "TCGA-LUAD" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 7, 9, 11, 12, 17)],
  "TCGA-LUSC" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 7, 9, 11, 12, 17)],
  "TCGA-SKCM" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 6, 7, 9, 12, 13, 14, 17)],
  "TCGA-OV"   = rownames(Loading_intrinsic_use)[c(1, 3, 4, 6, 7, 9, 12, 13, 15, 17)],
  "TCGA-PRAD" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 7, 9, 12, 13, 16, 17)],
  "TCGA-UCEC" = rownames(Loading_intrinsic_use)[c(1, 3, 4, 7, 9, 12, 18, 17)]
)

# -----------------------------
# Helper functions
# -----------------------------
make_osqp_template <- function(D, C, M, gene_weight_vec = NULL) {
  K <- C + M
  G <- ncol(D)
  
  if (is.null(gene_weight_vec)) {
    gene_weight_vec <- rep(1, G)
  }
  
  gene_weight_vec <- as.numeric(gene_weight_vec)
  
  if (length(gene_weight_vec) != G) {
    stop("Length of gene_weight_vec does not match number of genes in D.")
  }
  
  if (any(!is.finite(gene_weight_vec))) {
    stop("gene_weight_vec contains non-finite values.")
  }
  
  if (any(gene_weight_vec < 0)) {
    stop("gene_weight_vec contains negative values.")
  }
  
  Wdiag <- Diagonal(x = gene_weight_vec)
  
  # Weighted quadratic term:
  #   || sqrt(W) (y - zD) ||^2
  # => P = 2 D W D^T
  Pmat <- 2 * (D %*% Wdiag %*% t(D))
  Pmat <- Matrix(Pmat, sparse = TRUE)
  
  # Constraints:
  #   sum of intrinsic proportions = 1
  #   intrinsic proportions >= 0
  #   MI scores in [0, 1]
  Aeq <- Matrix(c(rep(1, C), rep(0, M)), nrow = 1, sparse = TRUE)
  Aid <- Diagonal(K)
  A <- rbind(Aeq, Aid)
  
  l <- c(1, rep(0, K))
  u <- c(1, c(rep(Inf, C), rep(1, M)))
  
  list(
    P = Pmat,
    A = A,
    l = l,
    u = u,
    gene_weight_vec = gene_weight_vec
  )
}

init_model <- function(P, A, l, u, q_init) {
  osqp(
    P = P, q = q_init,
    A = A, l = l, u = u,
    pars = list(verbose = FALSE, warm_start = TRUE)
  )
}

# -----------------------------
# PASS 1:
# Build X_raw_list and project-specific W0/WMI
# -----------------------------
X_raw_list <- list()
gene_avail <- list()
proj_meta  <- list()

all_available_celltypes <- rownames(Loading_intrinsic_use)

for (project_cur in TCGA_project_list) {
  print(project_cur)
  message("====================================================")
  message("Analyzing project (prep): ", project_cur)
  
  # ===== Download & prepare bulk RNA-seq data =====
  query_exp <- GDCquery(
    project = project_cur,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts"
  )
  
  GDCdownload(query_exp)
  expdat <- GDCprepare(query_exp)
  
  expr <- assay(expdat)
  gene_annot <- rowData(expdat)
  coldata <- as.data.frame(colData(expdat))
  
  expr_cpm <- cpm(expr, log = TRUE, prior.count = 1)
  rownames(expr_cpm) <- gene_annot$gene_name
  expr_cpm <- expr_cpm[!is.na(rownames(expr_cpm)), , drop = FALSE]
  
  # Keep only genes present both in the loading matrix and this project
  genes_here <- intersect(genename_list, rownames(expr_cpm))
  gene_avail[[project_cur]] <- genes_here
  X_raw_list[[project_cur]] <- expr_cpm
  
  # ===== Build project-specific W0 using predefined celltype list =====
  if (!project_cur %in% names(celltype_list_by_project)) {
    stop("Project not found in celltype_list_by_project: ", project_cur)
  }
  
  celltype_list_use <- celltype_list_by_project[[project_cur]]
  
  missing_celltypes <- setdiff(celltype_list_use, all_available_celltypes)
  if (length(missing_celltypes) > 0) {
    stop(
      "These cell types are missing in Loading_intrinsic_use for ",
      project_cur, ": ",
      paste(missing_celltypes, collapse = ", ")
    )
  }
  
  # Preserve the exact user-defined order in celltype_list_by_project
  W0_full  <- as.matrix(Loading_intrinsic_use[celltype_list_use, , drop = FALSE])
  WMI_full <- as.matrix(loading_MI[, , drop = FALSE])
  
  mode(W0_full)  <- "numeric"
  mode(WMI_full) <- "numeric"
  
  proj_meta[[project_cur]] <- list(
    W0_full = W0_full,
    WMI_full = WMI_full,
    celltype_list_use = celltype_list_use
  )
}

# -----------------------------
# Determine a COMMON gene set across all projects
# -----------------------------
genes_common <- Reduce(intersect, gene_avail)
genes_common <- unique(genes_common)

if (length(genes_common) < 10) {
  stop("Too few common genes across projects. Consider checking the input matrices.")
}
message("Common genes across all projects: ", length(genes_common))

# -----------------------------
# Final gene weight for common genes
# -----------------------------
if (use_gene_weight) {
  if (!all(genes_common %in% names(gene_weight))) {
    stop("Some common genes are missing in gene_weight.")
  }
  gene_weight_common <- as.numeric(gene_weight[genes_common])
  names(gene_weight_common) <- genes_common
  
  if (any(!is.finite(gene_weight_common))) {
    stop("gene_weight_common contains non-finite values.")
  }
  if (any(gene_weight_common < 0)) {
    stop("gene_weight_common contains negative values.")
  }
  
  message("Using gene-weighted loss.")
} else {
  gene_weight_common <- rep(1, length(genes_common))
  names(gene_weight_common) <- genes_common
  message("Using original uniform-weight loss.")
}

# -----------------------------
# PASS 1b:
# Finalize X_list and D_list with SAME gene order
# -----------------------------
X_list <- list()
D_list <- list()
CDMK   <- list()

for (project_cur in TCGA_project_list) {
  message("Finalize matrices for: ", project_cur)
  
  expr_cpm <- X_raw_list[[project_cur]]
  
  # X: samples x genes
  Xg <- t(as.matrix(expr_cpm[genes_common, , drop = FALSE]))
  mode(Xg) <- "numeric"
  
  if (any(!is.finite(Xg))) {
    stop("Non-finite values in X for ", project_cur)
  }
  
  # Project-specific W0 and common W_MI, both subset to common genes
  W0_full  <- proj_meta[[project_cur]]$W0_full
  WMI_full <- proj_meta[[project_cur]]$WMI_full
  
  if (!all(genes_common %in% colnames(W0_full))) {
    stop("W0 missing common genes for ", project_cur)
  }
  if (!all(genes_common %in% colnames(WMI_full))) {
    stop("W_MI missing common genes for ", project_cur)
  }
  
  W0   <- W0_full[, genes_common, drop = FALSE]
  W_MI <- WMI_full[, genes_common, drop = FALSE]
  
  D <- rbind(W0, W_MI)
  D <- as.matrix(D)
  mode(D) <- "numeric"
  
  if (any(!is.finite(D))) {
    stop("Non-finite values in D for ", project_cur)
  }
  
  X_list[[project_cur]] <- Xg
  D_list[[project_cur]] <- D
  
  CDMK[[project_cur]] <- list(
    N = nrow(Xg),
    G = ncol(Xg),
    C = nrow(W0),
    M = nrow(W_MI),
    K = nrow(D),
    sample_names = rownames(Xg),
    intrinsic_names = rownames(W0),
    mi_names = rownames(W_MI)
  )
}

# -----------------------------
# Global scaling
# -----------------------------
use_scaling <- TRUE
scale_factor <- 1.0

if (use_scaling) {
  set.seed(1)
  sample_cap <- 1e6
  pool <- numeric(0)
  
  for (project_cur in TCGA_project_list) {
    X <- X_list[[project_cur]]
    xvec <- as.numeric(X)
    xvec <- xvec[is.finite(xvec)]
    if (length(xvec) == 0) next
    
    take <- min(length(xvec), ceiling(sample_cap / length(TCGA_project_list)))
    pool <- c(pool, sample(xvec, size = take))
  }
  
  pool <- abs(pool)
  pool <- pool[is.finite(pool)]
  if (length(pool) == 0) {
    stop("Cannot compute scale_factor (no finite values).")
  }
  
  scale_factor <- as.numeric(stats::quantile(pool, probs = 0.95, names = FALSE))
  if (!is.finite(scale_factor) || scale_factor <= 0) {
    scale_factor <- 1.0
  }
  
  message("Global scale_factor (95% |X|): ", signif(scale_factor, 4))
  
  for (project_cur in TCGA_project_list) {
    X_list[[project_cur]] <- X_list[[project_cur]] / scale_factor
    D_list[[project_cur]] <- D_list[[project_cur]] / scale_factor
  }
}

# -----------------------------
# PASS 2:
# Global outer loop with shared beta0
# -----------------------------
G <- length(genes_common)
N_total <- sum(vapply(TCGA_project_list, function(p) CDMK[[p]]$N, numeric(1)))

beta0 <- rep(0, G)
max_outer <- 1000
tol <- 1e-7

P_hat_list  <- list()
MI_hat_list <- list()
osqp_tpl    <- list()

for (project_cur in TCGA_project_list) {
  meta <- CDMK[[project_cur]]
  D <- D_list[[project_cur]]
  
  osqp_tpl[[project_cur]] <- make_osqp_template(
    D = D,
    C = meta$C,
    M = meta$M,
    gene_weight_vec = gene_weight_common
  )
  
  P_hat_list[[project_cur]]  <- matrix(0, meta$N, meta$C)
  MI_hat_list[[project_cur]] <- matrix(0, meta$N, meta$M)
}

prev_mse <- NA_real_

for (it in seq_len(max_outer)) {
  sum_e   <- numeric(G)
  sumsq_e <- 0.0
  
  for (project_cur in TCGA_project_list) {
    X <- X_list[[project_cur]]
    D <- D_list[[project_cur]]
    meta <- CDMK[[project_cur]]
    tpl <- osqp_tpl[[project_cur]]
    gw <- tpl$gene_weight_vec
    
    y1 <- as.numeric(X[1, ] - beta0)
    q1 <- as.numeric(-2 * (D %*% (gw * y1)))
    
    model <- init_model(
      P = tpl$P,
      A = tpl$A,
      l = tpl$l,
      u = tpl$u,
      q_init = q1
    )
    
    for (i in seq_len(meta$N)) {
      y <- as.numeric(X[i, ] - beta0)
      q <- as.numeric(-2 * (D %*% (gw * y)))
      
      model$Update(q = q)
      r <- model$Solve()
      
      if (r$info$status_val != 1L) {
        stop(
          sprintf(
            "OSQP failed | outer=%d project=%s row=%d status=%s",
            it, project_cur, i, r$info$status
          )
        )
      }
      
      z <- r$x
      if (any(!is.finite(z))) {
        stop(
          sprintf("Non-finite z | outer=%d project=%s row=%d", it, project_cur, i)
        )
      }
      
      C <- meta$C
      K <- meta$K
      
      P_hat_list[[project_cur]][i, ]  <- z[1:C]
      MI_hat_list[[project_cur]][i, ] <- z[(C + 1):K]
      
      pred_no_intercept <- as.numeric(z %*% D)
      e <- as.numeric(X[i, ] - pred_no_intercept)
      
      # beta0 update remains the same because gene weights are constant across samples
      sum_e <- sum_e + e
      
      # loss tracking: weighted when use_gene_weight = TRUE, uniform otherwise
      sumsq_e <- sumsq_e + sum((e ^ 2) * gw)
    }
  }
  
  beta0_new <- sum_e / N_total
  sse <- sumsq_e - N_total * sum((beta0_new ^ 2) * gene_weight_common)
  
  if (!is.finite(sse)) {
    stop("SSE became non-finite; try stronger scaling or check inputs.")
  }
  if (sse < 0) {
    sse <- 0
  }
  
  mse <- sse / (N_total * G)
  
  if (!is.finite(mse)) {
    stop("MSE became non-finite; try stronger scaling or check inputs.")
  }
  
  if (is.na(prev_mse)) {
    cat(sprintf("Outer %d | Global MSE=%.6g (init)\n", it, mse))
    prev_mse <- mse
    beta0 <- beta0_new
  } else {
    rel_change <- abs(prev_mse - mse) / max(1, abs(prev_mse))
    cat(sprintf("Outer %d | Global MSE=%.6g | rel_change=%.3g\n", it, mse, rel_change))
    beta0 <- beta0_new
    
    if (rel_change < tol) {
      break
    }
    prev_mse <- mse
  }
}

# -----------------------------
# Unscale beta0 back
# -----------------------------
if (use_scaling && is.finite(scale_factor) && scale_factor != 1.0) {
  beta0 <- beta0 * scale_factor
}

cat("✅ Global solve done. Shared beta0 learned across all projects.\n")

# -----------------------------
# Save outputs
# -----------------------------
output_dir <- "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (project_cur in TCGA_project_list) {
  # Save MI_hat
  MI_hat <- MI_hat_list[[project_cur]]
  MI_hat[MI_hat < 0] <- 0
  MI_hat[MI_hat > 1] <- 1
  
  rownames(MI_hat) <- CDMK[[project_cur]]$sample_names
  colnames(MI_hat) <- CDMK[[project_cur]]$mi_names
  
  write.csv(
    MI_hat,
    file.path(output_dir, paste0("MI_decomposition_", project_cur, ".csv"))
  )
  
  # Save intrinsic proportions as well
  P_hat <- P_hat_list[[project_cur]]
  P_hat[P_hat < 0] <- 0
  
  rownames(P_hat) <- CDMK[[project_cur]]$sample_names
  colnames(P_hat) <- CDMK[[project_cur]]$intrinsic_names
  
  write.csv(
    P_hat,
    file.path(output_dir, paste0("Intrinsic_decomposition_", project_cur, ".csv"))
  )
}

# Save shared beta0
beta0_df <- data.frame(
  gene = genes_common,
  beta0 = beta0,
  stringsAsFactors = FALSE
)

write.csv(
  beta0_df,
  file.path(output_dir, "shared_beta0_common_genes.csv"),
  row.names = FALSE
)

# Save gene weights actually used
gene_weight_df <- data.frame(
  gene = genes_common,
  gene_weight = gene_weight_common,
  stringsAsFactors = FALSE
)

write.csv(
  gene_weight_df,
  file.path(output_dir, "gene_weight_common_genes.csv"),
  row.names = FALSE
)

cat("✅ Saved MI decomposition per project + intrinsic decomposition + shared beta0.\n")
cat("✅ Saved gene weights used in optimization.\n")



##Load the MI_decomposition csv files and combine them into a list
MI_decomposition_list <- list()
for (project_cur in TCGA_project_list) {
  MI_decomposition_list[[project_cur]] <- read.csv(paste0("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_",project_cur,".csv"), row.names = 1)
}
##Get the mean MI intensity for each cluster (column) in each project, and combine them into a dataframe
MI_use_cur <- colnames(MI_decomposition_list$`TCGA-BRCA`)
Mean_MI_Intensity_cluster <- data.frame(matrix(nrow = length(MI_use_cur), ncol = length(TCGA_project_list)))
rownames(Mean_MI_Intensity_cluster) <- MI_use_cur
colnames(Mean_MI_Intensity_cluster) <- TCGA_project_list
for (project_cur in TCGA_project_list) {
  MI_decomposition_cur <- MI_decomposition_list[[project_cur]]
  for (MI_cur in MI_use_cur) {
    Mean_MI_Intensity_cluster[MI_cur, project_cur] <- mean(MI_decomposition_cur[, MI_cur], na.rm = TRUE)
  }
}

# MI_use_order <- c("MI3","MI9","MI1","MI2","MI6","MI7",
#                   "MI12","MI4","MI5","MI11","MI8","MI10")
# 
# Mean_MI_Intensity_cluster <- Mean_MI_Intensity_cluster[MI_use_order,]

##Show the heatmaop of Mean_MI_Intensity_cluster
# library(pheatmap)
# library(RColorBrewer)
# my_colors <- colorRampPalette(c("white", "#FFDFEF", "#EABDE6", "#AA60C8"))(100)
# pheatmap(Mean_MI_Intensity_cluster,
#          color = my_colors,
#          cluster_rows = FALSE,
#          cluster_cols = FALSE,
#          border_color = "grey70",
#          filename = paste0("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Mean_MI_Intensity_cluster_heatmap.png"),
#          width = 3,
#          height = 4)


library(pheatmap)

# my_colors <- colorRampPalette(c(
#   "#FFFFFF",
#   "#FBE6EC",
#   "#F4B6C2",
#   "#D96B8A",
#   "#B23A62"
# ))(100)
library(RColorBrewer)
my_colors <- RColorBrewer::brewer.pal(n = 9, name = "YlOrRd")

pheatmap(
  Mean_MI_Intensity_cluster,
  color = my_colors,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  border_color = "#D9D9D9",
  cellwidth = 16,
  cellheight = 16,
  fontsize = 8,
  fontsize_row = 8,
  fontsize_col = 8,
  angle_col = 45,
  treeheight_row = 0,
  treeheight_col = 0,
  filename = "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Mean_MI_Intensity_cluster_heatmap.pdf",
  width = 3.2,
  height = 4.2
)

##
Mean_MI_Intensity_cluster_sub <- Mean_MI_Intensity_cluster[c("MI2","MI10","MI8","MI3","MI4"),]
# Mean_MI_Intensity_cluster_sub <- Mean_MI_Intensity_cluster[c("MI2","MI10","MI8","MI4"),]


pheatmap(
  Mean_MI_Intensity_cluster_sub,
  color = my_colors,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  border_color = "#D9D9D9",
  cellwidth = 16,
  cellheight = 16,
  fontsize = 8,
  fontsize_row = 8,
  fontsize_col = 8,
  angle_col = 45,
  treeheight_row = 0,
  treeheight_col = 0,
  filename = "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Mean_MI_Intensity_cluster_heatmap_tumorMI.pdf",
  width = 3.2,
  height = 4.2
)



## Compare the selected MI with the others in terms of mean intensity across the 9 tumor types.
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)

selected_list <- list(
  MI2  = c("TCGA-COAD", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-SKCM", "TCGA-OV", "TCGA-PRAD", "TCGA-UCEC"),
  MI10 = c("TCGA-BRCA"),
  MI8  = c("TCGA-BRCA"),
  MI3  = c("TCGA-BRCA"),
  MI4  = c("TCGA-LIHC")
)

## -----------------------------
## 1. Convert to long format
## -----------------------------
df_long <- Mean_MI_Intensity_cluster_sub %>%
  rownames_to_column("MI") %>%
  pivot_longer(
    cols = -MI,
    names_to = "TumorType",
    values_to = "Value"
  )

# df_long <- Mean_MI_Intensity_cluster %>%
#   rownames_to_column("MI") %>%
#   pivot_longer(
#     cols = -MI,
#     names_to = "TumorType",
#     values_to = "Value"
#   )

## -----------------------------
## 2. Label Selected vs Others
## -----------------------------
df_long$Group <- "Others"

for (mi in names(selected_list)) {
  df_long$Group[df_long$MI == mi & df_long$TumorType %in% selected_list[[mi]]] <- "Selected"
}

df_long$Group <- factor(df_long$Group, levels = c("Selected", "Others"))

table(df_long$Group)

## -----------------------------
## 3. Two-sided Wilcoxon rank-sum test
## -----------------------------
wilcox_test <- wilcox.test(Value ~ Group, data = df_long, alternative = "two.sided")
print(wilcox_test)

df_long %>%
  group_by(Group) %>%
  summarise(
    n = n(),
    mean = mean(Value),
    median = median(Value),
    sd = sd(Value),
    .groups = "drop"
  ) %>%
  print()

pval <- wilcox_test$p.value
p_label <- if (pval < 0.001) {
  "Two-sided Wilcoxon\nP < 0.001"
} else {
  paste0("Two-sided Wilcoxon\nP = ", signif(pval, 3))
}

## -----------------------------
## 4. Set significance annotation positions
## -----------------------------
y_max <- max(df_long$Value, na.rm = TRUE)
y_min <- min(df_long$Value, na.rm = TRUE)
y_range <- y_max - y_min

if (y_range == 0) y_range <- 1

line_y <- y_max + 0.10 * y_range
text_y <- y_max + 0.16 * y_range

## -----------------------------
## 5. Plot boxplot
##    - Selected: fill #6bcce0, border #2C7FB8
##    - Others:   fill light gray, border dark gray
##    - points:   black
## -----------------------------
p <- ggplot(df_long, aes(x = Group, y = Value, fill = Group, color = Group)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    width = 0.12,
    size = 1,
    alpha = 0.85,
    shape = 16,
    color = "black"
  ) +
  scale_fill_manual(
    values = c(
      # "Selected" = "#6bcce0",
      "Selected" = "#44ACFF",
      "Others"   = "#D9D9D9"
    )
  ) +
  scale_color_manual(
    values = c(
      # "Selected" = "#2C7FB8",
      "Selected" = "#6bcce0",
      "Others"   = "#6E6E6E"
    )
  ) +
  labs(
    x = NULL,
    y = "Mean MI intensity"
  ) +
  annotate("segment", x = 1, xend = 2, y = line_y, yend = line_y, linewidth = 0.4) +
  annotate("segment", x = 1, xend = 1, y = line_y, yend = line_y - 0.03 * y_range, linewidth = 0.4) +
  annotate("segment", x = 2, xend = 2, y = line_y, yend = line_y - 0.03 * y_range, linewidth = 0.4) +
  annotate("text", x = 1.5, y = text_y, label = p_label, size = 3.5) +
  coord_cartesian(
    ylim = c(y_min, y_max + 0.22 * y_range),
    clip = "off"
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 11, color = "black"),
    axis.text.y = element_text(size = 11, color = "black"),
    axis.title.y = element_text(size = 12, color = "black"),
    axis.line = element_line(color = "black", linewidth = 0.4),
    plot.margin = margin(10, 10, 10, 10)
  )

print(p)

## -----------------------------
## 6. Export as Illustrator-friendly PDF
## -----------------------------
ggsave(
  filename = "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Selected_vs_Others_boxplot.pdf",
  plot = p,
  width = 3.2,
  height = 4.2,
  device = cairo_pdf
)









# ###############################
# ## Compare the selected MI with the others in terms of mean intensity across the 9 tumor types
# library(ggplot2)
# library(dplyr)
# library(tidyr)
# library(tibble)
# 
# selected_list <- list(
#   MI2  = c("TCGA-COAD", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-SKCM", "TCGA-OV", "TCGA-PRAD", "TCGA-UCEC"),
#   MI10 = c("TCGA-BRCA"),
#   MI8  = c("TCGA-BRCA"),
#   MI3  = c("TCGA-BRCA"),
#   MI4  = c("TCGA-LIHC")
# )
# 
# ## -----------------------------
# ## 1. Convert to long format
# ## -----------------------------
# df_long <- Mean_MI_Intensity_cluster_sub %>%
#   rownames_to_column("MI") %>%
#   pivot_longer(
#     cols = -MI,
#     names_to = "TumorType",
#     values_to = "Value"
#   )
# 
# # df_long <- Mean_MI_Intensity_cluster %>%
# #   rownames_to_column("MI") %>%
# #   pivot_longer(
# #     cols = -MI,
# #     names_to = "TumorType",
# #     values_to = "Value"
# #   )
# 
# ## -----------------------------
# ## 2. Label Selected vs Others
# ## -----------------------------
# df_long$Group <- "Others"
# 
# for (mi in names(selected_list)) {
#   df_long$Group[df_long$MI == mi & df_long$TumorType %in% selected_list[[mi]]] <- "Selected"
# }
# 
# df_long$Group <- factor(df_long$Group, levels = c("Selected", "Others"))
# 
# table(df_long$Group)
# 
# ## -----------------------------
# ## 3. Two-sided t test
# ## -----------------------------
# t_test <- t.test(Value ~ Group, data = df_long, alternative = "two.sided")
# print(t_test)
# 
# ## Group summary statistics
# df_long %>%
#   group_by(Group) %>%
#   summarise(
#     n = n(),
#     mean = mean(Value, na.rm = TRUE),
#     median = median(Value, na.rm = TRUE),
#     sd = sd(Value, na.rm = TRUE)
#   ) %>%
#   print()
# 
# ## p-value label
# pval <- t_test$p.value
# p_label <- if (pval < 0.001) {
#   "Two-sided t test\nP < 0.001"
# } else {
#   paste0("Two-sided t test\nP = ", signif(pval, 3))
# }
# 
# ## -----------------------------
# ## 4. Set significance annotation position
# ## -----------------------------
# y_max <- max(df_long$Value, na.rm = TRUE)
# y_min <- min(df_long$Value, na.rm = TRUE)
# y_range <- y_max - y_min
# 
# line_y <- y_max + 0.10 * y_range
# text_y <- y_max + 0.16 * y_range
# 
# ## -----------------------------
# ## 5. Draw boxplot
# ## -----------------------------
# p <- ggplot(df_long, aes(x = Group, y = Value, fill = Group)) +
#   geom_boxplot(
#     width = 0.55,
#     outlier.shape = NA,
#     color = "black",
#     linewidth = 0.4
#   ) +
#   geom_jitter(
#     width = 0.12,
#     size = 2,
#     alpha = 0.85,
#     shape = 16
#   ) +
#   scale_fill_manual(values = c("Selected" = "#2C7FB8", "Others" = "#D9D9D9")) +
#   labs(
#     x = NULL,
#     y = "Mean MI intensity"
#   ) +
#   annotate("segment", x = 1, xend = 2, y = line_y, yend = line_y, linewidth = 0.4) +
#   annotate("segment", x = 1, xend = 1, y = line_y, yend = line_y - 0.03 * y_range, linewidth = 0.4) +
#   annotate("segment", x = 2, xend = 2, y = line_y, yend = line_y - 0.03 * y_range, linewidth = 0.4) +
#   annotate("text", x = 1.5, y = text_y, label = p_label, size = 3.5) +
#   coord_cartesian(ylim = c(y_min, y_max + 0.22 * y_range), clip = "off") +
#   theme_classic(base_size = 12) +
#   theme(
#     legend.position = "none",
#     axis.text.x = element_text(size = 11, color = "black"),
#     axis.text.y = element_text(size = 11, color = "black"),
#     axis.title.y = element_text(size = 12, color = "black"),
#     axis.line = element_line(color = "black", linewidth = 0.4),
#     plot.margin = margin(10, 10, 10, 10)
#   )
# 
# print(p)
# 
# ## -----------------------------
# ## 6. Export Illustrator-friendly PDF
# ## -----------------------------
# ggsave(
#   filename = "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Selected_vs_Others_boxplot.pdf",
#   plot = p,
#   width = 3.2,
#   height = 4.2,
#   device = cairo_pdf
# )