
work_dir <- "E:/TCGA_GDC"
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
setwd(work_dir) 

library(CVXR)
library(osqp)
library(Matrix)
library(TCGAbiolinks)
library(SummarizedExperiment)
library(edgeR)

TCGA_project_list <- c("TCGA-BRCA",
                       "TCGA-COAD",
                       "TCGA-LIHC",
                       "TCGA-LUAD","TCGA-LUSC",
                       "TCGA-SKCM",
                       "TCGA-OV",
                       "TCGA-PRAD",
                       "TCGA-UCEC")
cancercell_list <-c("Breast-cancercell",
                    "Colon-cancercell",
                    "Liver-cancercell",
                    "Lung-cancercell",
                    "Lung-cancercell",
                    "Melanoma-cancercell",
                    "Ovarian-cancercell",
                    "Prostate-cancercell",
                    "Uterine-cancercell")

##Load the loading matries
Loading_intrinsic_use <- read.csv("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/loading_intrinsic_use.csv", row.names = 1)
loading_sender_use <- read.csv("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/loading_sender_use.csv", row.names = 1)
loading_receiver_use <- read.csv("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/loading_receiver_use.csv", row.names = 1)
loading_MI <- loading_sender_use + loading_receiver_use
genename_list <- colnames(Loading_intrinsic_use)

celltype_list <- list("TCGA-BRCA" = rownames(Loading_intrinsic_use)[c(1,2,3,4,6,7,9,12,13,17)],
                      "TCGA-COAD" = rownames(Loading_intrinsic_use)[c(1,3,4,5,6,7,9,12,13,17)],
                      "TCGA-LIHC" = rownames(Loading_intrinsic_use)[c(1,3,4,6,7,8,9,10,12,13,17)],
                      "TCGA-LUAD" = rownames(Loading_intrinsic_use)[c(1,3,4,7,9,11,12,17)],
                      "TCGA-LUSC" = rownames(Loading_intrinsic_use)[c(1,3,4,7,9,11,12,17)],
                      "TCGA-SKCM" = rownames(Loading_intrinsic_use)[c(1,3,4,6,7,9,12,13,14,17)],
                      "TCGA-OV"   = rownames(Loading_intrinsic_use)[c(1,3,4,6,7,9,12,13,15,17)],
                      "TCGA-PRAD" = rownames(Loading_intrinsic_use)[c(1,3,4,7,9,12,13,16,17)],
                      "TCGA-UCEC" = rownames(Loading_intrinsic_use)[c(1,3,4,7,9,12,18,17)]
)

aggMI_all <- list()

##################################################################################################
##Batch corrected version
for (project_cur in TCGA_project_list) {
  print(project_cur)
  project_cur_index <- which(TCGA_project_list==project_cur)
  # expr_cpm <- read.csv(paste0("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Expr_CPM_",project_cur,".csv"))
  message("====================================================")
  message("Analyzing project: ", project_cur)
  
  ## ===============================
  ## Download & prepare data
  ## ===============================
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
  expr_cpm <- as.data.frame((expr_cpm))
  expr_cpm$X <- rownames(expr_cpm)
  genename_list_index <-c()
  for (gene_cur in genename_list) {
    geneindex_cur <- which(expr_cpm$X==gene_cur)
    if (length(geneindex_cur)==0){
      genename_list_index <- c(genename_list_index, NA)
    }else{
      if (length(geneindex_cur)>1){
        genename_list_index <- c(genename_list_index, geneindex_cur[1])
      }else{
        genename_list_index <- c(genename_list_index, geneindex_cur)
      }
    }
  }
  # ##Choose the tumor samples only
  # tumor_sample_index <- which(coldata$tumor_descriptor != "Not Applicable")
  # expr_cpm <- expr_cpm[,c(tumor_sample_index)]
  # coldata <- coldata[tumor_sample_index,]
  ##
  genename_list_use <- genename_list[which(!is.na(genename_list_index))]
  expr_cpm_use <- expr_cpm[genename_list_index[which(!is.na(genename_list_index))],]
  # rownames(expr_cpm_use) <- expr_cpm_use$X
  ##Remove the X column
  expr_cpm_use$X <- NULL
  expr_cpm_use<- t(expr_cpm_use)
  ##
  celltype_list <- rownames(Loading_intrinsic_use)
  ##Get the index of celltype_list that contained "-cancercell"
  celltype_list_index <- c()
  for (celltype_cur in celltype_list) {
    if (grepl("-cancercell", celltype_cur)) {
      celltype_list_index <- c(celltype_list_index, which(celltype_list==celltype_cur))
    }
  }
  cancercell_cur <- cancercell_list[project_cur_index]
  celltype_list_index <- setdiff(celltype_list_index, which(celltype_list==cancercell_cur))
  celltype_list_index<- c(celltype_list_index,which(celltype_list=="low_exp"))
  celltype_list_use <- setdiff(celltype_list, celltype_list[celltype_list_index])
  Loading_intrinsic_use_cur <- Loading_intrinsic_use[celltype_list_use,colnames(expr_cpm_use)]
  ##
  loading_MI_use_cur <- loading_MI[,colnames(expr_cpm_use)]
  ##Use the CUPR to solve the optimization problem
  W_MI <- loading_MI_use_cur
  W0 <- Loading_intrinsic_use_cur
  
  # # -----------------------
  # # Inputs (given)
  # # X: N x G
  # # W0: C x G
  # # W_MI: M x G
  # # -----------------------
  # X <- expr_cpm_use
  # W0 <- W0
  # W_MI <- W_MI
  # 
  # # -----------------------
  # # Dimension checks
  # # -----------------------
  # N <- nrow(X); G <- ncol(X)
  # stopifnot(ncol(W0) == G)
  # stopifnot(ncol(W_MI) == G)
  # C <- nrow(W0)
  # M <- nrow(W_MI)
  # 
  # # -----------------------
  # # Decision variables
  # # -----------------------
  # P <- Variable(N, C, nonneg = TRUE)      # N x C, P >= 0
  # MI <- Variable(N, M, nonneg = TRUE)     # N x M, MI >= 0
  # beta0 <- Variable(1, G)                 # 1 x G, free (can be negative)
  # 
  # # -----------------------
  # # Build intercept matrix: 1_N %*% beta0  => N x G
  # # -----------------------
  # onesN <- matrix(1, nrow = N, ncol = 1)
  # B0mat <- onesN %*% beta0
  # 
  # # -----------------------
  # # Prediction
  # # -----------------------
  # X_hat <- P %*% W0 + MI %*% W_MI + B0mat
  # 
  # # -----------------------
  # # Objective: MSE
  # # -----------------------
  # mse <- sum_squares(X - X_hat) / (N * G)
  # 
  # # -----------------------
  # # Constraints
  # # (1) rowSums(P) == 1  -> P %*% 1_C = 1_N
  # # (2) 0 <= MI <= 1     -> MI <= 1 (lower bound already via nonneg)
  # # -----------------------
  # constraints <- list(
  #   P %*% rep(1, C) == rep(1, N),
  #   MI <= 1
  # )
  # 
  # prob <- Problem(Minimize(mse), constraints)
  # 
  # # For this quadratic objective + linear constraints, OSQP is usually a good pick
  # res <- solve(prob, solver = "OSQP", verbose = TRUE)
  # 
  # P_hat <- res$getValue(P)         # N x C
  # MI_hat <- res$getValue(MI)       # N x M
  # beta0_hat <- res$getValue(beta0) # 1 x G
  # 
  # cat("status:", res$status, "\n")
  # cat("objective (MSE):", res$value, "\n")
  # 
  # # Quick sanity checks:
  # cat("range(P_hat):", range(P_hat), "\n")
  # cat("rowSums(P_hat) summary:\n"); print(summary(rowSums(P_hat)))
  # cat("range(MI_hat):", range(MI_hat), "\n")
  

  
  ###############################################################
  # Convex optimization with constraints using OSQP
  # Goal:
  #   minimize || X - (P %*% W0 + MI %*% W_MI + 1_N %*% beta0) ||_F^2
  # subject to:
  #   P >= 0, rowSums(P) == 1
  #   0 <= MI <= 1
  ###############################################################
  
  ###############################################################
  # Solve:
  #   min || X - (P W0 + MI W_MI + 1 beta0) ||_F^2 / (N*G)
  # s.t.
  #   P >= 0, rowSums(P) == 1
  #   0 <= MI <= 1
  #
  # Approach:
  #   Outer loop updates beta0
  #   Inner loop solves per-row QP by OSQP (small K=C+M variables)
  ###############################################################
  
  # =======================
  # 0) Load & sanitize input
  # =======================
  X    <- as.matrix(expr_cpm_use)
  W0   <- as.matrix(W0)
  W_MI <- as.matrix(W_MI)
  
  mode(X) <- "numeric"
  mode(W0) <- "numeric"
  mode(W_MI) <- "numeric"
  
  N <- nrow(X); G <- ncol(X)
  stopifnot(ncol(W0) == G, ncol(W_MI) == G)
  C <- nrow(W0); M <- nrow(W_MI)
  K <- C + M
  
  # Optional quick NA/Inf check (comment out if X is huge and you worry about speed)
  if (any(!is.finite(X))) stop("X has NA/Inf.")
  if (any(!is.finite(W0))) stop("W0 has NA/Inf.")
  if (any(!is.finite(W_MI))) stop("W_MI has NA/Inf.")
  
  # =======================
  # 1) Optional scaling (STRONGLY recommended)
  #    Prevents Inf/NaN from huge magnitudes.
  # =======================
  use_scaling <- TRUE
  scale_factor <- 1.0
  
  if (use_scaling) {
    # Robust-ish scale: use a high quantile of |X| to avoid being dominated by a few outliers.
    # If quantile is too slow for you, replace with max(abs(X)).
    absX <- abs(as.numeric(X))
    absX <- absX[is.finite(absX)]
    if (length(absX) == 0) stop("X seems empty or all non-finite after filtering.")
    scale_factor <- as.numeric(stats::quantile(absX, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1.0
    
    X    <- X / scale_factor
    W0   <- W0 / scale_factor
    W_MI <- W_MI / scale_factor
  }
  
  # Dictionary D (K x G)
  D <- rbind(W0, W_MI)
  D <- as.matrix(D); mode(D) <- "numeric"
  
  # =======================
  # 2) Precompute QP matrices
  # =======================
  # Objective per row: minimize || y - z D ||^2
  # => (1/2) z' P z + q' z, where P = 2*(D D'), q = -2*D*y
  Pmat <- 2 * (D %*% t(D))
  Pmat <- Matrix(Pmat, sparse = TRUE)
  
  # Constraints in OSQP form: l <= A z <= u
  # - simplex for first C entries: sum z[1:C] = 1
  # - bounds: z[1:C] >= 0; 0 <= z[C+1:K] <= 1
  Aeq <- Matrix(c(rep(1, C), rep(0, M)), nrow = 1, sparse = TRUE)
  Aid <- Diagonal(K)
  A <- rbind(Aeq, Aid)               # (1+K) x K
  
  l <- c(1, rep(0, K))
  u <- c(1, c(rep(Inf, C), rep(1, M)))  # MI upper bound = 1
  
  make_model <- function(q_init) {
    osqp(
      P = Pmat, q = q_init,
      A = A, l = l, u = u,
      pars = list(verbose = FALSE, warm_start = TRUE)
    )
  }
  
  # =======================
  # 3) Outer loop for beta0
  # =======================
  beta0 <- colMeans(X)  # init (length G)
  max_outer <- 1000
  tol <- 1e-7
  
  P_hat  <- matrix(0, N, C)
  MI_hat <- matrix(0, N, M)
  
  prev_mse <- NA_real_
  
  for (it in seq_len(max_outer)) {
    
    sum_e   <- numeric(G)  # sum of e_i (no intercept)
    sumsq_e <- 0.0         # sum of ||e_i||^2
    
    # init OSQP model using row 1
    y1 <- as.numeric(X[1, ] - beta0)
    q1 <- as.numeric(-2 * (D %*% y1))
    model <- make_model(q1)
    
    for (i in seq_len(N)) {
      y <- as.numeric(X[i, ] - beta0)
      q <- as.numeric(-2 * (D %*% y))
      
      model$Update(q = q)
      r <- model$Solve()
      
      if (r$info$status_val != 1L) {
        stop(sprintf("OSQP failed at row %d (status: %s)", i, r$info$status))
      }
      
      z <- r$x
      if (any(!is.finite(z))) {
        stop(sprintf("Non-finite solution z at row %d. Try scaling more aggressively.", i))
      }
      
      P_hat[i, ]  <- z[1:C]
      MI_hat[i, ] <- z[(C + 1):K]
      
      pred_no_intercept <- as.numeric(z %*% D)         # length G
      e <- as.numeric(X[i, ] - pred_no_intercept)      # e = X - ZD
      
      sum_e <- sum_e + e
      # accumulate squared norm; use crossprod for stability
      sumsq_e <- sumsq_e + as.numeric(crossprod(e))
    }
    
    # beta0_new is mean(e)
    beta0_new <- sum_e / N
    
    # SSE after optimal intercept: sum||e - mean(e)||^2 = sum||e||^2 - N||mean(e)||^2
    sse <- sumsq_e - N * as.numeric(crossprod(beta0_new))
    
    # guard numeric issues
    if (!is.finite(sse)) {
      stop("SSE became non-finite (Inf/NaN). Strongly suggest scaling or checking data magnitude.")
    }
    if (sse < 0) sse <- 0  # clamp tiny negative due to floating error
    
    mse <- sse / (N * G)
    if (!is.finite(mse)) {
      stop("MSE became non-finite (Inf/NaN). Strongly suggest scaling or checking data magnitude.")
    }
    
    if (is.na(prev_mse)) {
      cat(sprintf("Outer %d | MSE=%.6g (init)\n", it, mse))
      prev_mse <- mse
      beta0 <- beta0_new
      next
    } else {
      rel_change <- abs(prev_mse - mse) / max(1, abs(prev_mse))
      cat(sprintf("Outer %d | MSE=%.6g | rel_change=%.3g\n", it, mse, rel_change))
      beta0 <- beta0_new
      if (rel_change < tol) break
      prev_mse <- mse
    }
  }
  
  # =======================
  # 4) Unscale back (only beta0 depends on scale; P/MI are scale-free here)
  # =======================
  if (use_scaling && is.finite(scale_factor) && scale_factor != 1.0) {
    # Model was: (X/scale) ≈ P*(W0/scale) + MI*(W_MI/scale) + beta0_scaled
    # => original beta0 = beta0_scaled * scale
    beta0 <- beta0 * scale_factor
  }
  
  # =======================
  # 5) Outputs & checks
  # =======================
  cat("✅ Done.\n")
  cat("Check constraints:\n")
  cat("  range(P_hat): ", paste(range(P_hat), collapse = " ~ "), "\n")
  cat("  rowSums(P_hat) summary:\n"); print(summary(rowSums(P_hat)))
  cat("  range(MI_hat):", paste(range(MI_hat), collapse = " ~ "), "\n")
  ##Change the negative elements in MI_hat to 0
  MI_hat[MI_hat<0] <- 0
  MI_hat[MI_hat>1] <- 1
  
  results <- list(P_hat = P_hat, MI_hat = MI_hat, beta0 = beta0)
  MI_hat_final <- results$MI_hat
  rownames(MI_hat_final) <- rownames(expr_cpm_use)
  colnames(MI_hat_final) <- rownames(loading_MI_use_cur)
  ##Save the MI_hat_final as csv
  write.csv(MI_hat_final, paste0("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_",project_cur,".csv"))
  
  aggMI_all[[project_cur]] <- results$MI_hat
  
}

##Batch corrected version & shared beta0
###############################################################
# Goal: make ALL projects share the SAME gene-level intercept beta0
# Strategy:
#   1) First pass: build per-project X (samples x genes) and per-project D = rbind(W0, W_MI)
#      but FORCE all projects to use the SAME gene set (same columns order).
#   2) Second pass: run ONE global outer-loop that updates a SINGLE beta0 using
#      residuals pooled across ALL rows from ALL projects.
#   3) Inner-loop: for each project, solve per-row QP with its own D (W0/W_MI can differ by project).
###############################################################

suppressPackageStartupMessages({
  library(Matrix)
  library(osqp)
  # make sure you already loaded:
  # TCGAbiolinks, SummarizedExperiment, edgeR, etc.
})

# -----------------------------
# helpers
# -----------------------------
make_osqp_template <- function(D, C, M) {
  K <- C + M
  Pmat <- 2 * (D %*% t(D))
  Pmat <- Matrix(Pmat, sparse = TRUE)
  
  # constraints: l <= A z <= u
  # sum_{c<=C} z_c = 1
  # z[1:C] >= 0 ; 0 <= z[C+1:K] <= 1
  Aeq <- Matrix(c(rep(1, C), rep(0, M)), nrow = 1, sparse = TRUE)
  Aid <- Diagonal(K)
  A <- rbind(Aeq, Aid)
  
  l <- c(1, rep(0, K))
  u <- c(1, c(rep(Inf, C), rep(1, M)))
  
  list(P = Pmat, A = A, l = l, u = u)
}

init_model <- function(P, A, l, u, q_init) {
  osqp(
    P = P, q = q_init,
    A = A, l = l, u = u,
    pars = list(verbose = FALSE, warm_start = TRUE)
  )
}

# -----------------------------
# PASS 1: build X_list and D_list (but DO NOT solve yet)
#         Also determine a COMMON gene set across all projects.
# -----------------------------
X_raw_list   <- list()
gene_avail   <- list()
proj_meta    <- list()  # store D, C, M, K, rownames, etc.

for (project_cur in TCGA_project_list) {
  print(project_cur)
  project_cur_index <- which(TCGA_project_list == project_cur)
  message("====================================================")
  message("Analyzing project (prep): ", project_cur)
  
  # ===== Download & prepare data =====
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
  
  # keep only genename_list genes that exist in this project
  genes_here <- intersect(genename_list, rownames(expr_cpm))
  gene_avail[[project_cur]] <- genes_here
  
  # We will subset later after we decide the common gene set.
  # For now store the CPM matrix to avoid re-downloading.
  # (If memory is tight, store only expr_cpm[genes_here, ] instead.)
  X_raw_list[[project_cur]] <- expr_cpm
  
  # ===== Build W0 and W_MI (project-specific) =====
  celltype_list <- rownames(Loading_intrinsic_use)
  
  celltype_list_index <- which(grepl("-cancercell", celltype_list))
  cancercell_cur <- cancercell_list[project_cur_index]
  celltype_list_index <- setdiff(celltype_list_index, which(celltype_list == cancercell_cur))
  celltype_list_index <- c(celltype_list_index, which(celltype_list == "low_exp"))
  
  celltype_list_use <- setdiff(celltype_list, celltype_list[celltype_list_index])
  
  # Note: genes/columns will be aligned later, so here just store full matrices
  W0_full   <- as.matrix(Loading_intrinsic_use[celltype_list_use, , drop = FALSE])
  WMI_full  <- as.matrix(loading_MI[, , drop = FALSE])
  
  mode(W0_full)  <- "numeric"
  mode(WMI_full) <- "numeric"
  
  proj_meta[[project_cur]] <- list(
    W0_full  = W0_full,
    WMI_full = WMI_full
  )
}

# ---- Decide a COMMON gene set across all projects ----
# safest: INTERSECTION so beta0 truly shared on same genes
genes_common <- Reduce(intersect, gene_avail)
genes_common <- unique(genes_common)

if (length(genes_common) < 10) {
  stop("Too few common genes across projects. Consider using union+imputation, but shared beta0 then becomes ill-posed.")
}
message("Common genes across all projects: ", length(genes_common))

# -----------------------------
# PASS 1b: finalize X_list and D_list with the SAME gene columns/order
# -----------------------------
X_list <- list()
D_list <- list()
CDMK   <- list()  # store C/M/K and also row/col names

for (project_cur in TCGA_project_list) {
  message("Finalize matrices for: ", project_cur)
  
  expr_cpm <- X_raw_list[[project_cur]]
  
  # X: samples x genes (COMMON)
  Xg <- t(as.matrix(expr_cpm[genes_common, , drop = FALSE]))
  mode(Xg) <- "numeric"
  if (any(!is.finite(Xg))) stop("Non-finite values in X for ", project_cur)
  
  # project-specific W0/WMI, but subset to COMMON genes
  W0_full  <- proj_meta[[project_cur]]$W0_full
  WMI_full <- proj_meta[[project_cur]]$WMI_full
  
  # both should have genes as columns
  if (!all(genes_common %in% colnames(W0_full)))  stop("W0 missing common genes for ", project_cur)
  if (!all(genes_common %in% colnames(WMI_full))) stop("W_MI missing common genes for ", project_cur)
  
  W0   <- W0_full[,  genes_common, drop = FALSE]
  W_MI <- WMI_full[, genes_common, drop = FALSE]
  
  D <- rbind(W0, W_MI)
  D <- as.matrix(D); mode(D) <- "numeric"
  if (any(!is.finite(D))) stop("Non-finite values in D for ", project_cur)
  
  X_list[[project_cur]] <- Xg
  D_list[[project_cur]] <- D
  
  CDMK[[project_cur]] <- list(
    N = nrow(Xg), G = ncol(Xg),
    C = nrow(W0), M = nrow(W_MI), K = nrow(D),
    sample_names = rownames(Xg),
    mi_names = rownames(W_MI)
  )
}

# -----------------------------
# GLOBAL scaling (recommended): compute ONE scale_factor for all projects
# -----------------------------
use_scaling <- TRUE
scale_factor <- 1.0
if (use_scaling) {
  # To avoid huge memory, sample up to 1e6 entries across projects
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
  if (length(pool) == 0) stop("Cannot compute scale_factor (no finite values).")
  
  scale_factor <- as.numeric(stats::quantile(pool, probs = 0.95, names = FALSE))
  if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1.0
  
  message("Global scale_factor (95% |X|): ", signif(scale_factor, 4))
  
  # apply same scaling to EVERY X and EVERY D
  for (project_cur in TCGA_project_list) {
    X_list[[project_cur]] <- X_list[[project_cur]] / scale_factor
    D_list[[project_cur]] <- D_list[[project_cur]] / scale_factor
  }
}

# -----------------------------
# PASS 2: ONE global outer loop => shared beta0
# Inner loop: per project, per row OSQP
# -----------------------------
G <- length(genes_common)
N_total <- sum(vapply(TCGA_project_list, function(p) CDMK[[p]]$N, numeric(1)))

beta0 <- rep(0, G)  # if you want init as zeros (shared)
max_outer <- 1000
tol <- 1e-7

# allocate outputs
P_hat_list  <- list()
MI_hat_list <- list()

# prepare per-project OSQP templates
osqp_tpl <- list()
for (project_cur in TCGA_project_list) {
  meta <- CDMK[[project_cur]]
  D <- D_list[[project_cur]]
  osqp_tpl[[project_cur]] <- make_osqp_template(D = D, C = meta$C, M = meta$M)
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
    
    # init model from row 1
    y1 <- as.numeric(X[1, ] - beta0)
    q1 <- as.numeric(-2 * (D %*% y1))
    model <- init_model(P = tpl$P, A = tpl$A, l = tpl$l, u = tpl$u, q_init = q1)
    
    for (i in seq_len(meta$N)) {
      y <- as.numeric(X[i, ] - beta0)
      q <- as.numeric(-2 * (D %*% y))
      
      model$Update(q = q)
      r <- model$Solve()
      if (r$info$status_val != 1L) {
        stop(sprintf("OSQP failed | outer=%d project=%s row=%d status=%s",
                     it, project_cur, i, r$info$status))
      }
      
      z <- r$x
      if (any(!is.finite(z))) {
        stop(sprintf("Non-finite z | outer=%d project=%s row=%d", it, project_cur, i))
      }
      
      C <- meta$C
      K <- meta$K
      
      P_hat_list[[project_cur]][i, ]  <- z[1:C]
      MI_hat_list[[project_cur]][i, ] <- z[(C + 1):K]
      
      pred_no_intercept <- as.numeric(z %*% D)    # length G
      e <- as.numeric(X[i, ] - pred_no_intercept) # e = X - ZD
      
      sum_e <- sum_e + e
      sumsq_e <- sumsq_e + as.numeric(crossprod(e))
    }
  }
  
  beta0_new <- sum_e / N_total
  sse <- sumsq_e - N_total * as.numeric(crossprod(beta0_new))
  if (!is.finite(sse)) stop("SSE became non-finite; try stronger scaling or check inputs.")
  if (sse < 0) sse <- 0
  
  mse <- sse / (N_total * G)
  if (!is.finite(mse)) stop("MSE became non-finite; try stronger scaling or check inputs.")
  
  if (is.na(prev_mse)) {
    cat(sprintf("Outer %d | Global MSE=%.6g (init)\n", it, mse))
    prev_mse <- mse
    beta0 <- beta0_new
  } else {
    rel_change <- abs(prev_mse - mse) / max(1, abs(prev_mse))
    cat(sprintf("Outer %d | Global MSE=%.6g | rel_change=%.3g\n", it, mse, rel_change))
    beta0 <- beta0_new
    if (rel_change < tol) break
    prev_mse <- mse
  }
}

# unscale beta0 back
if (use_scaling && is.finite(scale_factor) && scale_factor != 1.0) {
  beta0 <- beta0 * scale_factor
}

cat("✅ Global solve done. Shared beta0 learned across all projects.\n")

# -----------------------------
# Postprocess, save per project
# -----------------------------
for (project_cur in TCGA_project_list) {
  MI_hat <- MI_hat_list[[project_cur]]
  
  # clip numerical noise
  MI_hat[MI_hat < 0] <- 0
  MI_hat[MI_hat > 1] <- 1
  
  # attach names
  rownames(MI_hat) <- CDMK[[project_cur]]$sample_names
  colnames(MI_hat) <- CDMK[[project_cur]]$mi_names
  
  write.csv(
    MI_hat,
    paste0("D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_",
           project_cur, ".csv")
  )
}

# optionally save shared beta0 (gene-order = genes_common)
beta0_df <- data.frame(gene = genes_common, beta0 = beta0)
write.csv(
  beta0_df,
  "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/shared_beta0_common_genes.csv",
  row.names = FALSE
)

cat("✅ Saved MI decomposition per project + shared beta0.\n")

##Do not consider the batch effect & corr-based loss
for (project_cur in TCGA_project_list) {
  print(project_cur)
  project_cur_index <- which(TCGA_project_list == project_cur)
  message("====================================================")
  message("Analyzing project: ", project_cur)
  
  ## ===============================
  ## Download & prepare data
  ## ===============================
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
  expr_cpm <- as.data.frame(expr_cpm)
  expr_cpm$X <- rownames(expr_cpm)
  
  ## ===============================
  ## Subset genes to genename_list
  ## ===============================
  genename_list_index <- c()
  for (gene_cur in genename_list) {
    geneindex_cur <- which(expr_cpm$X == gene_cur)
    if (length(geneindex_cur) == 0) {
      genename_list_index <- c(genename_list_index, NA)
    } else {
      genename_list_index <- c(genename_list_index, geneindex_cur[1])
    }
  }
  
  genename_list_use <- genename_list[which(!is.na(genename_list_index))]
  expr_cpm_use <- expr_cpm[genename_list_index[which(!is.na(genename_list_index))], , drop = FALSE]
  
  ## remove gene-name column and transpose -> X: N(samples) x G(genes)
  expr_cpm_use$X <- NULL
  expr_cpm_use <- t(expr_cpm_use)
  
  ## ===============================
  ## Prepare W0 (intrinsic) and W_MI
  ## ===============================
  celltype_list <- rownames(Loading_intrinsic_use)
  
  ## find all "-cancercell" celltypes
  celltype_list_index <- c()
  for (celltype_cur in celltype_list) {
    if (grepl("-cancercell", celltype_cur)) {
      celltype_list_index <- c(celltype_list_index, which(celltype_list == celltype_cur))
    }
  }
  
  cancercell_cur <- cancercell_list[project_cur_index]
  celltype_list_index <- setdiff(celltype_list_index, which(celltype_list == cancercell_cur))
  celltype_list_index <- c(celltype_list_index, which(celltype_list == "low_exp"))
  celltype_list_use <- setdiff(celltype_list, celltype_list[celltype_list_index])
  
  ## align genes
  Loading_intrinsic_use_cur <- Loading_intrinsic_use[celltype_list_use, colnames(expr_cpm_use), drop = FALSE]
  loading_MI_use_cur <- loading_MI[, colnames(expr_cpm_use), drop = FALSE]
  
  W0   <- as.matrix(Loading_intrinsic_use_cur)  # C x G
  W_MI <- as.matrix(loading_MI_use_cur)         # M x G
  
  ## =======================
  ## Inputs
  ## =======================
  X <- as.matrix(expr_cpm_use)  # N x G
  mode(X) <- "numeric"
  mode(W0) <- "numeric"
  mode(W_MI) <- "numeric"
  
  N <- nrow(X); G <- ncol(X)
  stopifnot(ncol(W0) == G, ncol(W_MI) == G)
  C <- nrow(W0); M <- nrow(W_MI)
  K <- C + M
  
  if (any(!is.finite(X))) stop("X has NA/Inf.")
  if (any(!is.finite(W0))) stop("W0 has NA/Inf.")
  if (any(!is.finite(W_MI))) stop("W_MI has NA/Inf.")
  
  ## =======================
  ## Pearson corr objective via centered cosine
  ## =======================
  D <- rbind(W0, W_MI)   # K x G
  
  # Center dictionary rows across genes: D_c = D - rowMeans(D)
  D_row_mean <- rowMeans(D)
  D_c <- D - D_row_mean  # K x G
  
  # Q = D_c D_c^T (PSD)
  Q <- D_c %*% t(D_c)
  Q <- Matrix(Q, sparse = TRUE)
  
  ## =======================
  ## Constraints: l <= A z <= u
  ## =======================
  Aeq <- Matrix(c(rep(1, C), rep(0, M)), nrow = 1, sparse = TRUE)
  Aid <- Diagonal(K)
  A <- rbind(Aeq, Aid)
  
  l <- c(1, rep(0, K))
  u <- c(1, c(rep(Inf, C), rep(1, M)))
  
  # Solve one QP with OSQP and optional warm-start
  solve_qp <- function(Pmat, qvec, z_warm = NULL) {
    model <- osqp(
      P = Pmat,
      q = qvec,
      A = A, l = l, u = u,
      pars = list(verbose = FALSE, warm_start = TRUE)
    )
    if (!is.null(z_warm)) model$WarmStart(x = z_warm)
    r <- model$Solve()
    if (r$info$status_val != 1L) {
      stop(sprintf("OSQP failed (status: %s)", r$info$status))
    }
    r$x
  }
  
  ## ============================================================
  ## Initialization (YOUR REQUEST)
  ## - MI: all 0.1
  ## - P: cancercell column(s) total 0.5, others random summing to 0.5
  ## ============================================================
  cancer_idx <- grep("-cancercell", celltype_list_use)
  if (length(cancer_idx) == 0) {
    stop("No '-cancercell' found in celltype_list_use; cannot initialize P as requested.")
  }
  if (length(cancer_idx) > 1) {
    message("Multiple '-cancercell' columns found; splitting 0.5 equally among them.")
  }
  
  set.seed(1)  # deterministic init; change/remove if desired
  
  Z_init <- matrix(0, nrow = N, ncol = K)
  for (i in seq_len(N)) {
    # MI init
    MI0 <- rep(0.1, M)
    
    # P init
    P0 <- numeric(C)
    P0[cancer_idx] <- 0.5 / length(cancer_idx)
    
    other_idx <- setdiff(seq_len(C), cancer_idx)
    if (length(other_idx) > 0) {
      rnd <- runif(length(other_idx))
      rnd <- rnd / sum(rnd) * 0.5
      P0[other_idx] <- rnd
    } else {
      P0[cancer_idx] <- P0[cancer_idx] + 0.5 / length(cancer_idx)
    }
    
    P0[P0 < 0] <- 0
    s <- sum(P0)
    if (!is.finite(s) || s <= 0) {
      P0 <- rep(1 / C, C)
    } else {
      P0 <- P0 / s
    }
    
    Z_init[i, ] <- c(P0, MI0)
  }
  
  ## =======================
  ## Per-row optimization (iterative QP; print only AFTER finished)
  ## =======================
  P_hat  <- matrix(0, N, C)
  MI_hat <- matrix(0, N, M)
  corr_vec <- numeric(N)
  
  max_inner <- 50
  tol_inner <- 1e-6
  eps <- 1e-12
  lambda_min <- 1e-8
  lambda_max <- 1e8
  
  for (i in seq_len(N)) {
    y <- as.numeric(X[i, ])
    
    # center y
    y_c <- y - mean(y)
    y_norm <- sqrt(sum(y_c^2))
    if (!is.finite(y_norm) || y_norm < eps) {
      z <- Z_init[i, ]
      P_hat[i, ]  <- z[1:C]
      MI_hat[i, ] <- z[(C + 1):K]
      corr_vec[i] <- NA_real_
      next
    }
    
    # b = D_c %*% y_c
    b <- as.numeric(D_c %*% y_c)
    if (!all(is.finite(b)) || sum(abs(b)) < eps) {
      z <- Z_init[i, ]
      P_hat[i, ]  <- z[1:C]
      MI_hat[i, ] <- z[(C + 1):K]
      xhat <- as.numeric(z %*% D)
      corr_vec[i] <- suppressWarnings(stats::cor(y, xhat, method = "pearson"))
      next
    }
    
    # warm start: your custom init for this row
    z_prev <- Z_init[i, ]
    
    # initial QP with small lambda
    lambda <- 1e-2
    Pmat <- lambda * Q
    qvec <- -b
    z <- solve_qp(Pmat, qvec, z_warm = z_prev)
    
    prev_loss <- NA_real_
    
    for (t in seq_len(max_inner)) {
      zb  <- as.numeric(crossprod(z, b))
      zQz <- as.numeric(crossprod(z, Q %*% z))
      if (!is.finite(zb) || !is.finite(zQz) || zQz < eps) break
      
      lambda_new <- zb / zQz
      if (!is.finite(lambda_new)) break
      lambda_new <- min(lambda_max, max(lambda_min, lambda_new))
      
      Pmat <- lambda_new * Q
      qvec <- -b
      z_new <- solve_qp(Pmat, qvec, z_warm = z)
      
      # convergence check based on Pearson loss
      xhat_for_check <- as.numeric(z_new %*% D)
      corr_for_check <- suppressWarnings(stats::cor(y, xhat_for_check, method = "pearson"))
      loss_for_check <- -corr_for_check
      
      if (!is.na(prev_loss) && is.finite(loss_for_check) && is.finite(prev_loss)) {
        rel <- abs(prev_loss - loss_for_check) / max(1, abs(prev_loss))
        if (rel < tol_inner) {
          z <- z_new
          break
        }
      }
      prev_loss <- loss_for_check
      z <- z_new
      lambda <- lambda_new
    }
    
    P_hat[i, ]  <- z[1:C]
    MI_hat[i, ] <- z[(C + 1):K]
    
    # final Pearson corr for this row
    xhat_final <- as.numeric(z %*% D)
    corr_vec[i] <- suppressWarnings(stats::cor(y, xhat_final, method = "pearson"))
  }
  
  ## =======================
  ## Clamp & normalize
  ## =======================
  P_hat[P_hat < 0] <- 0
  rs <- rowSums(P_hat)
  bad <- which(!is.finite(rs) | rs <= 0)
  if (length(bad) > 0) {
    P_hat[bad, ] <- 1 / C
    rs[bad] <- 1
  }
  P_hat <- P_hat / rs
  
  MI_hat[MI_hat < 0] <- 0
  MI_hat[MI_hat > 1] <- 1
  
  ## =======================
  ## Print ONLY after training finished
  ## =======================
  total_corr_sum <- sum(corr_vec, na.rm = TRUE)
  mean_corr <- mean(corr_vec, na.rm = TRUE)
  
  message("----------------------------------------------------")
  message(sprintf("Project: %s", project_cur))
  message(sprintf("Total Pearson corr (sum over samples): %.6f", total_corr_sum))
  message(sprintf("Average Pearson corr per sample: %.6f", mean_corr))
  message(sprintf("Number of samples used: %d / %d", sum(!is.na(corr_vec)), length(corr_vec)))
  message("----------------------------------------------------")
  
  ## =======================
  ## Save MI_hat
  ## =======================
  MI_hat_final <- MI_hat
  rownames(MI_hat_final) <- rownames(expr_cpm_use)         # samples
  colnames(MI_hat_final) <- rownames(loading_MI_use_cur)   # MI dims
  
  write.csv(
    MI_hat_final,
    paste0(
      "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_",
      project_cur, ".csv"
    )
  )
  
  ## optional: save corr per sample
  corr_df <- data.frame(sample = rownames(expr_cpm_use), pearson_corr = corr_vec)
  write.csv(
    corr_df,
    paste0(
      "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_corr_",
      project_cur, ".csv"
    ),
    row.names = FALSE
  )
  
  aggMI_all[[project_cur]] <- MI_hat
}

##Batch corrected & corr-based loss
for (project_cur in TCGA_project_list) {
  print(project_cur)
  project_cur_index <- which(TCGA_project_list == project_cur)
  message("====================================================")
  message("Analyzing project: ", project_cur)
  
  ## ===============================
  ## Download & prepare data
  ## ===============================
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
  expr_cpm <- as.data.frame(expr_cpm)
  expr_cpm$X <- rownames(expr_cpm)
  
  ## ===============================
  ## Subset genes to genename_list
  ## ===============================
  genename_list_index <- c()
  for (gene_cur in genename_list) {
    geneindex_cur <- which(expr_cpm$X == gene_cur)
    if (length(geneindex_cur) == 0) {
      genename_list_index <- c(genename_list_index, NA)
    } else {
      genename_list_index <- c(genename_list_index, geneindex_cur[1])
    }
  }
  
  genename_list_use <- genename_list[which(!is.na(genename_list_index))]
  expr_cpm_use <- expr_cpm[genename_list_index[which(!is.na(genename_list_index))], , drop = FALSE]
  
  ## remove gene-name column and transpose -> X: N(samples) x G(genes)
  expr_cpm_use$X <- NULL
  expr_cpm_use <- t(expr_cpm_use)
  
  ## ===============================
  ## Prepare W0 (intrinsic) and W_MI
  ## ===============================
  celltype_list <- rownames(Loading_intrinsic_use)
  
  ## find all "-cancercell" celltypes
  celltype_list_index <- c()
  for (celltype_cur in celltype_list) {
    if (grepl("-cancercell", celltype_cur)) {
      celltype_list_index <- c(celltype_list_index, which(celltype_list == celltype_cur))
    }
  }
  
  cancercell_cur <- cancercell_list[project_cur_index]
  celltype_list_index <- setdiff(celltype_list_index, which(celltype_list == cancercell_cur))
  celltype_list_index <- c(celltype_list_index, which(celltype_list == "low_exp"))
  celltype_list_use <- setdiff(celltype_list, celltype_list[celltype_list_index])
  
  ## align genes
  Loading_intrinsic_use_cur <- Loading_intrinsic_use[celltype_list_use, colnames(expr_cpm_use), drop = FALSE]
  loading_MI_use_cur <- loading_MI[, colnames(expr_cpm_use), drop = FALSE]
  
  W0   <- as.matrix(Loading_intrinsic_use_cur)  # C x G
  W_MI <- as.matrix(loading_MI_use_cur)         # M x G
  
  ## =======================
  ## Inputs
  ## =======================
  X <- as.matrix(expr_cpm_use)  # N x G
  mode(X) <- "numeric"
  mode(W0) <- "numeric"
  mode(W_MI) <- "numeric"
  
  N <- nrow(X); G <- ncol(X)
  stopifnot(ncol(W0) == G, ncol(W_MI) == G)
  C <- nrow(W0); M <- nrow(W_MI)
  K <- C + M
  
  if (any(!is.finite(X))) stop("X has NA/Inf.")
  if (any(!is.finite(W0))) stop("W0 has NA/Inf.")
  if (any(!is.finite(W_MI))) stop("W_MI has NA/Inf.")
  
  ## =======================
  ## Optional scaling (recommended)
  ## =======================
  use_scaling <- TRUE
  scale_factor <- 1.0
  if (use_scaling) {
    absX <- abs(as.numeric(X))
    absX <- absX[is.finite(absX)]
    if (length(absX) == 0) stop("X seems empty or all non-finite after filtering.")
    scale_factor <- as.numeric(stats::quantile(absX, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1.0
    X    <- X / scale_factor
    W0   <- W0 / scale_factor
    W_MI <- W_MI / scale_factor
  }
  
  ## =======================
  ## Dictionary and centered version (for corr-loss on residual)
  ## =======================
  D <- rbind(W0, W_MI)  # K x G
  D <- as.matrix(D); mode(D) <- "numeric"
  
  D_row_mean <- rowMeans(D)
  D_c <- D - D_row_mean  # K x G
  
  Q <- D_c %*% t(D_c)
  Q <- Matrix(Q, sparse = TRUE)
  
  ## =======================
  ## Constraints: l <= A z <= u
  ## =======================
  Aeq <- Matrix(c(rep(1, C), rep(0, M)), nrow = 1, sparse = TRUE)
  Aid <- Diagonal(K)
  A <- rbind(Aeq, Aid)
  
  l <- c(1, rep(0, K))
  u <- c(1, c(rep(Inf, C), rep(1, M)))
  
  solve_qp <- function(Pmat, qvec, z_warm = NULL) {
    model <- osqp(
      P = Pmat,
      q = qvec,
      A = A, l = l, u = u,
      pars = list(verbose = FALSE, warm_start = TRUE)
    )
    if (!is.null(z_warm)) model$WarmStart(x = z_warm)
    r <- model$Solve()
    if (r$info$status_val != 1L) stop(sprintf("OSQP failed (status: %s)", r$info$status))
    r$x
  }
  
  ## ============================================================
  ## Initialization (YOUR REQUEST)
  ## - MI: all 0.1
  ## - P: cancercell column(s) total 0.5, others random summing to 0.5
  ## ============================================================
  cancer_idx <- grep("-cancercell", celltype_list_use)
  if (length(cancer_idx) == 0) stop("No '-cancercell' found in celltype_list_use.")
  if (length(cancer_idx) > 1) message("Multiple '-cancercell' columns found; splitting 0.5 equally among them.")
  
  set.seed(1)
  
  Z_init <- matrix(0, nrow = N, ncol = K)
  for (i in seq_len(N)) {
    MI0 <- rep(0.1, M)
    P0 <- numeric(C)
    P0[cancer_idx] <- 0.5 / length(cancer_idx)
    
    other_idx <- setdiff(seq_len(C), cancer_idx)
    if (length(other_idx) > 0) {
      rnd <- runif(length(other_idx))
      rnd <- rnd / sum(rnd) * 0.5
      P0[other_idx] <- rnd
    } else {
      P0[cancer_idx] <- P0[cancer_idx] + 0.5 / length(cancer_idx)
    }
    
    P0[P0 < 0] <- 0
    s <- sum(P0)
    if (!is.finite(s) || s <= 0) P0 <- rep(1 / C, C) else P0 <- P0 / s
    
    Z_init[i, ] <- c(P0, MI0)
  }
  
  ## ============================================================
  ## Add beta0 back with OUTER LOOP
  ## beta0 init = 0 vector  ✅ per your request
  ## ============================================================
  beta0 <- rep(0, G)   # in scaled space
  
  max_outer <- 200
  tol_outer <- 1e-6
  
  P_hat  <- matrix(0, N, C)
  MI_hat <- matrix(0, N, M)
  
  prev_mean_corr <- NA_real_
  
  max_inner <- 50
  tol_inner <- 1e-6
  eps <- 1e-12
  lambda_min <- 1e-8
  lambda_max <- 1e8
  
  for (out_it in seq_len(max_outer)) {
    sum_e <- numeric(G)
    corr_vec_outer <- numeric(N)
    
    for (i in seq_len(N)) {
      y_resid <- as.numeric(X[i, ] - beta0)
      
      y_c <- y_resid - mean(y_resid)
      y_norm <- sqrt(sum(y_c^2))
      if (!is.finite(y_norm) || y_norm < eps) {
        z <- Z_init[i, ]
        P_hat[i, ]  <- z[1:C]
        MI_hat[i, ] <- z[(C + 1):K]
        
        pred_no_intercept <- as.numeric(z %*% D)
        e <- as.numeric(X[i, ] - pred_no_intercept)
        sum_e <- sum_e + e
        
        xhat_full <- as.numeric(pred_no_intercept + beta0)
        corr_vec_outer[i] <- suppressWarnings(stats::cor(X[i, ], xhat_full, method = "pearson"))
        next
      }
      
      b <- as.numeric(D_c %*% y_c)
      
      if (!all(is.finite(b)) || sum(abs(b)) < eps) {
        z <- Z_init[i, ]
      } else {
        z_prev <- Z_init[i, ]
        lambda <- 1e-2
        z <- solve_qp(lambda * Q, -b, z_warm = z_prev)
        
        prev_loss <- NA_real_
        for (t in seq_len(max_inner)) {
          zb  <- as.numeric(crossprod(z, b))
          zQz <- as.numeric(crossprod(z, Q %*% z))
          if (!is.finite(zb) || !is.finite(zQz) || zQz < eps) break
          
          lambda_new <- zb / zQz
          if (!is.finite(lambda_new)) break
          lambda_new <- min(lambda_max, max(lambda_min, lambda_new))
          
          z_new <- solve_qp(lambda_new * Q, -b, z_warm = z)
          
          pred_resid <- as.numeric(z_new %*% D)
          corr_check <- suppressWarnings(stats::cor(y_resid, pred_resid, method = "pearson"))
          loss_check <- -corr_check
          
          if (!is.na(prev_loss) && is.finite(loss_check) && is.finite(prev_loss)) {
            rel <- abs(prev_loss - loss_check) / max(1, abs(prev_loss))
            if (rel < tol_inner) {
              z <- z_new
              break
            }
          }
          prev_loss <- loss_check
          z <- z_new
          lambda <- lambda_new
        }
      }
      
      P_hat[i, ]  <- z[1:C]
      MI_hat[i, ] <- z[(C + 1):K]
      
      pred_no_intercept <- as.numeric(z %*% D)
      e <- as.numeric(X[i, ] - pred_no_intercept)
      sum_e <- sum_e + e
      
      xhat_full <- as.numeric(pred_no_intercept + beta0)
      corr_vec_outer[i] <- suppressWarnings(stats::cor(X[i, ], xhat_full, method = "pearson"))
    }
    
    beta0_new <- sum_e / N
    
    mean_corr_outer <- mean(corr_vec_outer, na.rm = TRUE)
    
    if (is.na(prev_mean_corr)) {
      message(sprintf("Outer %d | mean corr(full recon) = %.6f (init)", out_it, mean_corr_outer))
      prev_mean_corr <- mean_corr_outer
      beta0 <- beta0_new
    } else {
      rel_change <- abs(prev_mean_corr - mean_corr_outer) / max(1, abs(prev_mean_corr))
      message(sprintf("Outer %d | mean corr(full recon) = %.6f | rel_change=%.3g",
                      out_it, mean_corr_outer, rel_change))
      beta0 <- beta0_new
      if (rel_change < tol_outer) break
      prev_mean_corr <- mean_corr_outer
    }
  }
  
  ## =======================
  ## Clamp & normalize
  ## =======================
  P_hat[P_hat < 0] <- 0
  rs <- rowSums(P_hat)
  bad <- which(!is.finite(rs) | rs <= 0)
  if (length(bad) > 0) {
    P_hat[bad, ] <- 1 / C
    rs[bad] <- 1
  }
  P_hat <- P_hat / rs
  
  MI_hat[MI_hat < 0] <- 0
  MI_hat[MI_hat > 1] <- 1
  
  ## =======================
  ## Unscale beta0 for saving/reporting
  ## =======================
  if (use_scaling && is.finite(scale_factor) && scale_factor != 1.0) {
    beta0_out <- beta0 * scale_factor
  } else {
    beta0_out <- beta0
  }
  
  ## =======================
  ## Final corr report (FULL reconstruction)
  ## =======================
  corr_vec_final <- numeric(N)
  for (i in seq_len(N)) {
    z <- c(P_hat[i, ], MI_hat[i, ])
    pred_no_intercept <- as.numeric(z %*% D)
    xhat_full <- as.numeric(pred_no_intercept + beta0)
    corr_vec_final[i] <- suppressWarnings(stats::cor(X[i, ], xhat_full, method = "pearson"))
  }
  
  total_corr_sum <- sum(corr_vec_final, na.rm = TRUE)
  mean_corr <- mean(corr_vec_final, na.rm = TRUE)
  
  message("----------------------------------------------------")
  message(sprintf("Project: %s", project_cur))
  message(sprintf("Total Pearson corr (sum over samples, full recon): %.6f", total_corr_sum))
  message(sprintf("Average Pearson corr per sample (full recon): %.6f", mean_corr))
  message(sprintf("Number of samples used: %d / %d", sum(!is.na(corr_vec_final)), length(corr_vec_final)))
  message("----------------------------------------------------")
  
  ## =======================
  ## Save MI_hat
  ## =======================
  MI_hat_final <- MI_hat
  rownames(MI_hat_final) <- rownames(expr_cpm_use)
  colnames(MI_hat_final) <- rownames(loading_MI_use_cur)
  
  write.csv(
    MI_hat_final,
    paste0(
      "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_",
      project_cur, ".csv"
    )
  )
  
  ## Save beta0
  beta0_df <- data.frame(gene = colnames(expr_cpm_use), beta0 = as.numeric(beta0_out))
  write.csv(
    beta0_df,
    paste0(
      "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/beta0_",
      project_cur, ".csv"
    ),
    row.names = FALSE
  )
  
  ## Save corr per sample (final, full recon)
  corr_df <- data.frame(sample = rownames(expr_cpm_use), pearson_corr = corr_vec_final)
  write.csv(
    corr_df,
    paste0(
      "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/MI_decomposition/MI_decomposition_corr_",
      project_cur, ".csv"
    ),
    row.names = FALSE
  )
  
  aggMI_all[[project_cur]] <- MI_hat
}








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

my_colors <- colorRampPalette(c(
  "#FFFFFF",
  "#FBE6EC",
  "#F4B6C2",
  "#D96B8A",
  "#B23A62"
))(100)

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


##Compare the selected MI (MI3, MI9, MI1, MI2, MI7, MI4) with the others in terms of mean intensity across the 9 tumor types.
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)

selected_list <- list(
  MI2 = c("TCGA-COAD", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-SKCM", "TCGA-OV", "TCGA-PRAD", "TCGA-UCEC"),
  MI10 = c("TCGA-BRCA"),
  MI8 = c("TCGA-BRCA"),
  MI3 = c("TCGA-BRCA"),
  MI4 = c("TCGA-LIHC")
)

## -----------------------------
## 1. 转成长表
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
## 2. 标记 Selected vs Others
## -----------------------------
df_long$Group <- "Others"

for (mi in names(selected_list)) {
  df_long$Group[df_long$MI == mi & df_long$TumorType %in% selected_list[[mi]]] <- "Selected"
}

## 让顺序变成 Selected 在前
df_long$Group <- factor(df_long$Group, levels = c("Selected", "Others"))

## 看一下数量
table(df_long$Group)

## -----------------------------
## 3. Two-sided Wilcoxon rank-sum test
## -----------------------------
wilcox_test <- wilcox.test(Value ~ Group, data = df_long, alternative = "two.sided")
print(wilcox_test)

## 组内统计
df_long %>%
  group_by(Group) %>%
  summarise(
    n = n(),
    mean = mean(Value),
    median = median(Value),
    sd = sd(Value)
  ) %>%
  print()

## p-value label
pval <- wilcox_test$p.value
p_label <- if (pval < 0.001) {
  "Two-sided Wilcoxon\nP < 0.001"
} else {
  paste0("Two-sided Wilcoxon\nP = ", signif(pval, 3))
}

## -----------------------------
## 4. 设置显著性标注位置
## -----------------------------
y_max <- max(df_long$Value, na.rm = TRUE)
y_min <- min(df_long$Value, na.rm = TRUE)
y_range <- y_max - y_min

line_y  <- y_max + 0.10 * y_range
text_y  <- y_max + 0.16 * y_range

## -----------------------------
## 5. 画 boxplot
## -----------------------------
p <- ggplot(df_long, aes(x = Group, y = Value, fill = Group)) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    color = "black",
    linewidth = 0.4
  ) +
  geom_jitter(
    width = 0.12,
    size = 2,
    alpha = 0.85,
    shape = 16
  ) +
  scale_fill_manual(values = c("Selected" = "#2C7FB8", "Others" = "#D9D9D9")) +
  labs(
    x = NULL,
    y = "Mean MI intensity"
  ) +
  annotate("segment", x = 1, xend = 2, y = line_y, yend = line_y, linewidth = 0.4) +
  annotate("segment", x = 1, xend = 1, y = line_y, yend = line_y - 0.03 * y_range, linewidth = 0.4) +
  annotate("segment", x = 2, xend = 2, y = line_y, yend = line_y - 0.03 * y_range, linewidth = 0.4) +
  annotate("text", x = 1.5, y = text_y, label = p_label, size = 3.5) +
  coord_cartesian(ylim = c(y_min, y_max + 0.22 * y_range), clip = "off") +
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
## 6. 导出为 Illustrator 友好的 PDF
## -----------------------------
ggsave(
  filename = "D:/SpiderNet/Results/Pancancer/V1/SpiderNet_Result_dim11/Selected_vs_Others_boxplot.pdf",
  plot = p,
  width = 3.2,
  height = 4.2,
  device = cairo_pdf
)