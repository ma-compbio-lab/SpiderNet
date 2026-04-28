###############################################################
# Shared-beta0 version using precomputed bulk gene expression
# Input bulk expression matrix:
#   D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/geneexp_mean_df.csv
# Assumed dimension:
#   samples x genes
#
# Compared with the previous TCGA-based version:
#   - Remove TCGA cohort download/preparation
#   - Read bulk expression directly from geneexp_mean_df.csv
#   - Use one shared decomposition run across all samples
#   - Keep shared beta0, gene-weighted loss, and output structure
###############################################################

# -----------------------------
# Basic setup
# -----------------------------
suppressPackageStartupMessages({
  library(osqp)
  library(Matrix)
})

# -----------------------------
# User switch
# -----------------------------
use_gene_weight <- TRUE   # TRUE: weighted loss; FALSE: original uniform loss
use_scaling <- TRUE
max_outer <- 1000
tol <- 1e-7
if_traindata <- TRUE

# -----------------------------
# Input / output paths
# -----------------------------
input_dir <- "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11"
output_dir <- file.path(input_dir, "MI_decomposition")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

bulk_expr_path <- file.path(input_dir, "geneexp_mean_df.csv")
loading_intrinsic_path <- file.path(input_dir, "loading_intrinsic_use.csv")
loading_sender_path <- file.path(input_dir, "loading_sender_use.csv")
loading_receiver_path <- file.path(input_dir, "loading_receiver_use.csv")

# -----------------------------
# Load matrices
# -----------------------------
Loading_intrinsic_use <- read.csv(
  loading_intrinsic_path,
  row.names = 1,
  check.names = FALSE
)

loading_sender_use <- read.csv(
  loading_sender_path,
  row.names = 1,
  check.names = FALSE
)

loading_receiver_use <- read.csv(
  loading_receiver_path,
  row.names = 1,
  check.names = FALSE
)

bulk_expr_df <- read.csv(
  bulk_expr_path,
  row.names = 1,
  check.names = FALSE
)

if (if_traindata == TRUE){
  MI_mean_df <- read.csv(file.path(input_dir, "MI_mean_df.csv"), row.names = 1, check.names = FALSE)
  train_sample_name <- rownames(MI_mean_df )[which(MI_mean_df$Split == "Train")]
  bulk_expr_df <- bulk_expr_df[train_sample_name,]
}


loading_MI <- loading_sender_use + loading_receiver_use

# -----------------------------
# Basic checks
# -----------------------------
if (nrow(bulk_expr_df) == 0 || ncol(bulk_expr_df) == 0) {
  stop("bulk_expr_df is empty. Check geneexp_mean_df.csv.")
}

if (anyDuplicated(colnames(bulk_expr_df)) > 0) {
  stop("Duplicated gene names found in geneexp_mean_df.csv column names.")
}

if (anyDuplicated(colnames(Loading_intrinsic_use)) > 0) {
  stop("Duplicated gene names found in loading_intrinsic_use.csv column names.")
}

if (!identical(colnames(loading_sender_use), colnames(loading_receiver_use))) {
  stop("loading_sender_use and loading_receiver_use must have identical gene columns.")
}

if (!identical(colnames(Loading_intrinsic_use), colnames(loading_MI))) {
  stop("Loading_intrinsic_use and loading_MI must have identical gene columns.")
}

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
# Gene intersection and alignment
# -----------------------------
genes_common <- intersect(colnames(bulk_expr_df), colnames(Loading_intrinsic_use))
genes_common <- unique(genes_common)

if (length(genes_common) < 10) {
  stop("Too few overlapping genes between geneexp_mean_df.csv and loading matrices.")
}

message("Common genes used for decomposition: ", length(genes_common))

X <- as.matrix(bulk_expr_df[, genes_common, drop = FALSE])
mode(X) <- "numeric"

if (any(!is.finite(X))) {
  stop("Non-finite values found in bulk expression matrix after gene subsetting.")
}

W0 <- as.matrix(Loading_intrinsic_use[, genes_common, drop = FALSE])
W_MI <- as.matrix(loading_MI[, genes_common, drop = FALSE])
mode(W0) <- "numeric"
mode(W_MI) <- "numeric"

if (any(!is.finite(W0))) {
  stop("Non-finite values found in Loading_intrinsic_use after gene subsetting.")
}

if (any(!is.finite(W_MI))) {
  stop("Non-finite values found in loading_MI after gene subsetting.")
}

D <- rbind(W0, W_MI)
D <- as.matrix(D)
mode(D) <- "numeric"

# -----------------------------
# Final gene weights
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
# Metadata
# -----------------------------
N <- nrow(X)
G <- ncol(X)
C <- nrow(W0)
M <- nrow(W_MI)
K <- nrow(D)

sample_names <- rownames(X)
intrinsic_names <- rownames(W0)
mi_names <- rownames(W_MI)

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
    P = P,
    q = q_init,
    A = A,
    l = l,
    u = u,
    pars = list(verbose = FALSE, warm_start = TRUE)
  )
}

# -----------------------------
# Global scaling
# -----------------------------
scale_factor <- 1.0

if (use_scaling) {
  set.seed(1)
  xvec <- as.numeric(X)
  xvec <- xvec[is.finite(xvec)]

  if (length(xvec) == 0) {
    stop("Cannot compute scale_factor: no finite values in X.")
  }

  sample_cap <- 1e6
  take <- min(length(xvec), sample_cap)
  pool <- sample(xvec, size = take)
  pool <- abs(pool)
  pool <- pool[is.finite(pool)]

  scale_factor <- as.numeric(stats::quantile(pool, probs = 0.95, names = FALSE))
  if (!is.finite(scale_factor) || scale_factor <= 0) {
    scale_factor <- 1.0
  }

  message("Global scale_factor (95% |X|): ", signif(scale_factor, 4))

  X_scaled <- X / scale_factor
  D_scaled <- D / scale_factor
} else {
  X_scaled <- X
  D_scaled <- D
}

# -----------------------------
# Solve with shared beta0
# -----------------------------
beta0 <- rep(0, G)
P_hat <- matrix(0, nrow = N, ncol = C)
MI_hat <- matrix(0, nrow = N, ncol = M)

osqp_tpl <- make_osqp_template(
  D = D_scaled,
  C = C,
  M = M,
  gene_weight_vec = gene_weight_common
)

prev_mse <- NA_real_

for (it in seq_len(max_outer)) {
  sum_e <- numeric(G)
  sumsq_e <- 0.0

  y1 <- as.numeric(X_scaled[1, ] - beta0)
  q1 <- as.numeric(-2 * (D_scaled %*% (osqp_tpl$gene_weight_vec * y1)))

  model <- init_model(
    P = osqp_tpl$P,
    A = osqp_tpl$A,
    l = osqp_tpl$l,
    u = osqp_tpl$u,
    q_init = q1
  )

  for (i in seq_len(N)) {
    y <- as.numeric(X_scaled[i, ] - beta0)
    q <- as.numeric(-2 * (D_scaled %*% (osqp_tpl$gene_weight_vec * y)))

    model$Update(q = q)
    r <- model$Solve()

    if (r$info$status_val != 1L) {
      stop(
        sprintf(
          "OSQP failed | outer=%d row=%d status=%s",
          it, i, r$info$status
        )
      )
    }

    z <- r$x
    if (any(!is.finite(z))) {
      stop(sprintf("Non-finite z | outer=%d row=%d", it, i))
    }

    P_hat[i, ] <- z[1:C]
    MI_hat[i, ] <- z[(C + 1):K]

    pred_no_intercept <- as.numeric(z %*% D_scaled)
    e <- as.numeric(X_scaled[i, ] - pred_no_intercept)

    sum_e <- sum_e + e
    sumsq_e <- sumsq_e + sum((e ^ 2) * osqp_tpl$gene_weight_vec)
  }

  beta0_new <- sum_e / N
  sse <- sumsq_e - N * sum((beta0_new ^ 2) * gene_weight_common)

  if (!is.finite(sse)) {
    stop("SSE became non-finite; try stronger scaling or check inputs.")
  }
  if (sse < 0) {
    sse <- 0
  }

  mse <- sse / (N * G)

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

cat("✅ Decomposition finished. Shared beta0 learned across all samples.\n")

# -----------------------------
# Post-process and save outputs
# -----------------------------
MI_hat[MI_hat < 0] <- 0
MI_hat[MI_hat > 1] <- 1
rownames(MI_hat) <- sample_names
colnames(MI_hat) <- mi_names


##LOad the D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_mean_df.csv
MI_mean_df <- read.csv(file.path(input_dir, "MI_mean_df.csv"), row.names = 1, check.names = FALSE)
MI_mean_df <- MI_mean_df[rownames(MI_hat),]
##Remove the Split column from MI_mean_df
MI_mean_df_entrie <- MI_mean_df
MI_mean_df_entrie$Split <- NULL

##Rowwise Spearman correlation
corr_vec <- c()
for (row_index in seq_len(nrow(MI_hat))) {
  mi_vec <- MI_hat[row_index, ]
  mi_mean_vec <- as.numeric(MI_mean_df_entrie[row_index, ])

  if (length(mi_vec) != length(mi_mean_vec)) {
    stop("Length mismatch between MI_hat row and MI_mean_df_entrie row.")
  }

  if (all(mi_vec == 0) || all(mi_mean_vec == 0)) {
    corr_vec[row_index] <- NA
  } else {
    corr_vec[row_index] <- cor(mi_vec, mi_mean_vec, method = "spearman")
  }
}

summary(corr_vec)

##ColWisee Spearman correlation
corr_vec_col <- c()
for (col_index in seq_len(ncol(MI_hat))) {
  mi_vec <- MI_hat[, col_index]
  mi_mean_vec <- as.numeric(MI_mean_df_entrie[, col_index])
  
  if (length(mi_vec) != length(mi_mean_vec)) {
    stop("Length mismatch between MI_hat column and MI_mean_df_entrie column.")
  }
  
  if (all(mi_vec == 0) || all(mi_mean_vec == 0)) {
    corr_vec_col[col_index] <- NA
  } else {
    corr_vec_col[col_index] <- cor(mi_vec, mi_mean_vec, method = "spearman")
  }
}
summary(corr_vec_col)

MI_hat_flat <- as.vector(MI_hat)
MI_mean_flat <- as.vector(as.matrix(MI_mean_df_entrie))
corr_flat <- cor(MI_hat_flat, MI_mean_flat, method = "spearman")
cat("Overall Spearman correlation (flattened): ", signif(corr_flat, 4), "\n")
##Show the scatter plot between MI_hat_flat and MI_mean_flat
plot(MI_hat_flat, MI_mean_flat, xlab = "MI_hat (decomposed)", ylab = "MI_mean (original)", main = "Scatter plot of MI_hat vs MI_mean")
abline(lm(MI_mean_flat ~ MI_hat_flat), col = "red")
