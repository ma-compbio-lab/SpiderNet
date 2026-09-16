# Project fixed spatial meta-interaction loadings onto TCGA tumor transcriptomes
# and validate recovery from spatial pseudo-bulk expression. The two projections
# retain their independent joint gene/LR fits and existing parameter settings.
# Inputs: four loading CSVs, pseudo-bulk expression/MI summaries, and GDC data.
# Fits, diagnostic tables, and figures are written under output/projection/.
# Use --check, --plot-only, or --stage tcga|pseudobulk; see README for input paths.

# Resolve the shared runtime relative to this script, independent of the working directory.
.pancancer_file <- if (sys.nframe() > 0L && !is.null(sys.frame(1)$ofile)) {
  sys.frame(1)$ofile
} else {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
}
.pancancer_dir <- dirname(normalizePath(.pancancer_file, winslash = "/", mustWork = TRUE))
source(file.path(.pancancer_dir, "plot_saved_R.R"), local = TRUE)
.pancancer <- pancancer_options("projection", .pancancer_dir)
if (.pancancer$mode != "full") {
  pancancer_dispatch(.pancancer)
  quit(save = "no", status = 0L)
}
pancancer_check(.pancancer)

# Run both independent projections unless a stage was selected.
run_tcga_projection <- .pancancer$stage %in% c("all", "tcga")
run_s26b_validation <- .pancancer$stage %in% c("all", "pseudobulk")
input_dir <- .pancancer$results
s26b_output_dir <- file.path(.pancancer$output, "projection", "MI_decomposition", "S26b_MI4")
s26b_mi_to_plot <- "MI-4"

if (isTRUE(run_tcga_projection)) {
  # -----------------------------
  # GDC download/cache directory
  # -----------------------------
  # Configure work_dir and input_dir for the local data layout before execution.
  # setwd() makes work_dir the base for GDC downloads and changes the R session directory.
  work_dir <- .pancancer$gdc
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  setwd(work_dir)

  suppressPackageStartupMessages({
    library(osqp)
    library(Matrix)
    library(TCGAbiolinks)
    library(SummarizedExperiment)
    library(edgeR)
  })

  # -----------------------------
  # Spatial loading inputs and projection outputs
  # -----------------------------
  # All four loading files must come from the same fitted MI basis and feature order.
  output_dir <- file.path(.pancancer$output, "projection", "MI_decomposition_tumor_only")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  loading_intrinsic_path <- file.path(input_dir, "loading_intrinsic_use.csv")
  loading_sender_path    <- file.path(input_dir, "loading_sender_use.csv")
  loading_receiver_path  <- file.path(input_dir, "loading_receiver_use.csv")
  loading_LR_path        <- file.path(input_dir, "loading_LR_use.csv")

  # -----------------------------
  # Projection configuration
  # -----------------------------
  # Cohort-specific gene and LR offsets (alpha and delta in Supplementary Note A.8)
  if_set_beta0_zero <- FALSE
  if_set_beta0_LR_zero <- FALSE
  if_beta0_nonnegative <- FALSE

  # Ridge-shrinkage and damping for cohort-specific beta0 updates
  use_beta0_ridge <- FALSE
  beta0_ridge_lambda_gene <- 10.0
  beta0_ridge_lambda_lr <- 50.0
  beta0_step_size <- 0.25

  # Feature-weight switches
  if_nongeneweight <- FALSE
  if_nonlrweight <- FALSE
  use_gene_weight <- !if_nongeneweight
  use_lr_weight <- !if_nonlrweight
  use_paper_omega_weighting <- TRUE

  # Joint-loss switches
  use_lr_loss <- TRUE
  lambda_gene <- 1.0
  lambda_lr <- 0.5

  # Cohort-balanced objective: each TCGA project contributes equal total weight
  use_cohort_balanced_loss <- TRUE

  # LR-pair parsing: FALSE keeps an LR pair if at least one ligand and one receptor
  # subunit are present in the TCGA expression matrix.
  lr_require_all_genes <- FALSE

  # Optional global scaling and outer-loop convergence settings
  use_scaling <- FALSE
  max_outer <- 1000
  tol <- 1e-7

  # -----------------------------
  # TCGA projects
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
  # Tumor-only sample filter
  # -----------------------------
  # All downstream analyses in this script are restricted to tumor_keep samples.
  # The filter is defined only from coldata$sample_type, without relying on sample IDs.
  tumor_sample_types <- c(
    "Primary Tumor",
    "Metastatic",
    "Recurrent Tumor",
    "Additional - New Primary"
  )

  nontumor_sample_types <- c(
    "Solid Tissue Normal",
    "Blood Derived Normal",
    "Buccal Cell Normal",
    "Bone Marrow Normal",
    "Normal"
  )

  # Cancer-cell labels corresponding to TCGA_project_list.
  # W0 selection below uses celltype_list_by_project directly.
  cancercell_list <- c(
    "Breast-cancercell",
    "Colon-cancercell",
    "Liver-cancercell",
    "Lung-cancercell",
    "Lung-cancercell",
    "Melanoma-cancercell",
    "Ovarian-cancercell",
    "Prostate-cancercell",
    "Uterine-cancercell"
  )

  # -----------------------------
  # Load and check the fixed spatial signatures
  # -----------------------------
  Loading_intrinsic_use <- read.csv(loading_intrinsic_path, row.names = 1, check.names = FALSE)
  loading_sender_use    <- read.csv(loading_sender_path,    row.names = 1, check.names = FALSE)
  loading_receiver_use  <- read.csv(loading_receiver_path,  row.names = 1, check.names = FALSE)
  loading_LR_use        <- read.csv(loading_LR_path,        row.names = 1, check.names = FALSE)

  loading_MI <- loading_sender_use + loading_receiver_use
  genename_list <- colnames(Loading_intrinsic_use)

  if (anyDuplicated(colnames(Loading_intrinsic_use)) > 0) {
    stop("Duplicated gene names found in loading_intrinsic_use.csv column names.")
  }
  if (!identical(colnames(loading_sender_use), colnames(loading_receiver_use))) {
    stop("loading_sender_use and loading_receiver_use must have identical gene columns.")
  }
  if (!identical(colnames(Loading_intrinsic_use), colnames(loading_MI))) {
    stop("Loading_intrinsic_use and loading_MI must have identical gene columns.")
  }
  if (nrow(loading_LR_use) != nrow(loading_MI)) {
    stop("loading_LR_use must have the same number of MI rows as loading_sender_use/loading_receiver_use.")
  }
  if (!is.null(rownames(loading_LR_use)) && !is.null(rownames(loading_MI)) &&
      all(rownames(loading_MI) %in% rownames(loading_LR_use))) {
    loading_LR_use <- loading_LR_use[rownames(loading_MI), , drop = FALSE]
  } else if (!identical(rownames(loading_LR_use), rownames(loading_MI))) {
    warning("Could not confidently align loading_LR_use rows by row name; using existing row order.")
  }

  # -----------------------------
  # Project-specific intrinsic programs used in W0
  # -----------------------------
  # Numeric indices refer to the row order of loading_intrinsic_use.csv.
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
  KL_divergence <- function(p, q) {
    eps <- 1e-10
    p <- p + eps
    p <- p / sum(p)
    q <- q + eps
    q <- q / sum(q)
    sum(p * log(p / q))
  }

  compute_kl_specificity_weights <- function(loading_mat, feature_label = "feature") {
    loading_mat <- as.matrix(loading_mat)
    mode(loading_mat) <- "numeric"
    if (any(!is.finite(loading_mat))) {
      stop(sprintf("%s loading matrix contains non-finite values.", feature_label))
    }
    if (any(loading_mat < 0)) {
      stop(sprintf("%s loading matrix contains negative values; KL-specificity weights assume non-negative loadings.", feature_label))
    }
    n_dim <- nrow(loading_mat)
    n_feature <- ncol(loading_mat)
    uniform_vec <- rep(1 / n_dim, n_dim)
    specificity <- numeric(n_feature)
    for (feature_index in seq_len(n_feature)) {
      x <- loading_mat[, feature_index]
      x_sum <- sum(x)
      if (!is.finite(x_sum) || x_sum <= 1e-12) {
        specificity[feature_index] <- 0
      } else {
        specificity[feature_index] <- KL_divergence(x / x_sum, uniform_vec)
      }
    }
    specificity_sum <- sum(specificity)
    if (!is.finite(specificity_sum) || specificity_sum <= 0) {
      stop(sprintf("Cannot normalize %s KL-specificity weights because all specificity scores are zero/non-finite.", feature_label))
    }
    weight <- n_feature * specificity / specificity_sum
    names(weight) <- colnames(loading_mat)
    weight
  }

  parse_lr_pair_name <- function(lr_name) {
    parts <- strsplit(lr_name, "\\s*->\\s*")[[1]]
    if (length(parts) != 2) {
      stop(sprintf("Cannot parse LR pair name '%s'. Expected format 'ligand -> receptor'.", lr_name))
    }
    ligands <- trimws(unlist(strsplit(parts[1], "\\+", fixed = FALSE)))
    receptors <- trimws(unlist(strsplit(parts[2], "\\+", fixed = FALSE)))
    ligands <- ligands[ligands != ""]
    receptors <- receptors[receptors != ""]
    if (length(ligands) == 0 || length(receptors) == 0) {
      stop(sprintf("LR pair name '%s' has empty ligand or receptor set.", lr_name))
    }
    list(ligands = unique(ligands), receptors = unique(receptors))
  }

  # Average duplicate symbols after log-CPM transformation.
  collapse_duplicate_gene_rows <- function(expr_mat) {
    expr_mat <- as.matrix(expr_mat)
    mode(expr_mat) <- "numeric"
    genes <- rownames(expr_mat)
    keep <- !is.na(genes) & genes != ""
    expr_mat <- expr_mat[keep, , drop = FALSE]
    genes <- genes[keep]
    if (length(unique(genes)) == length(genes)) {
      return(expr_mat)
    }
    sum_mat <- rowsum(expr_mat, group = genes, reorder = FALSE)
    n_per_gene <- as.numeric(table(genes)[rownames(sum_mat)])
    sweep(sum_mat, 1, n_per_gene, "/")
  }

  build_lr_coexpression_matrix <- function(expr_df, lr_pair_names, require_all_genes = TRUE) {
    if (nrow(expr_df) == 0 || ncol(expr_df) == 0) {
      stop("expr_df is empty; cannot build LR coexpression matrix.")
    }
    all_genes <- colnames(expr_df)
    sample_names <- rownames(expr_df)
    lr_values <- list()
    metadata_rows <- list()
    skipped_rows <- list()
    for (lr_name in lr_pair_names) {
      parsed <- parse_lr_pair_name(lr_name)
      ligands <- parsed$ligands
      receptors <- parsed$receptors
      ligand_present <- ligands %in% all_genes
      receptor_present <- receptors %in% all_genes
      if (isTRUE(require_all_genes)) {
        keep_pair <- all(ligand_present) && all(receptor_present)
      } else {
        keep_pair <- any(ligand_present) && any(receptor_present)
      }
      if (!keep_pair) {
        skipped_rows[[length(skipped_rows) + 1]] <- data.frame(
          lr_pair = lr_name,
          ligands = paste(ligands, collapse = "+"),
          receptors = paste(receptors, collapse = "+"),
          missing_ligands = paste(ligands[!ligand_present], collapse = "+"),
          missing_receptors = paste(receptors[!receptor_present], collapse = "+"),
          stringsAsFactors = FALSE,
          check.names = FALSE
        )
        next
      }
      ligands_use <- ligands[ligand_present]
      receptors_use <- receptors[receptor_present]
      ligand_expr <- rowMeans(as.matrix(expr_df[, ligands_use, drop = FALSE]))
      receptor_expr <- rowMeans(as.matrix(expr_df[, receptors_use, drop = FALSE]))
      if (any(!is.finite(ligand_expr)) || any(!is.finite(receptor_expr))) {
        stop(sprintf("Non-finite bulk expression values found for LR pair '%s'.", lr_name))
      }
      # Clamp negative ligand/receptor mean log-CPM values to zero before the geometric mean.
      ligand_expr <- pmax(ligand_expr, 0)
      receptor_expr <- pmax(receptor_expr, 0)
      coexpr <- sqrt(ligand_expr * receptor_expr)
      lr_values[[lr_name]] <- as.numeric(coexpr)
      metadata_rows[[length(metadata_rows) + 1]] <- data.frame(
        lr_pair = lr_name,
        ligands = paste(ligands, collapse = "+"),
        receptors = paste(receptors, collapse = "+"),
        ligands_used = paste(ligands_use, collapse = "+"),
        receptors_used = paste(receptors_use, collapse = "+"),
        n_ligands_used = length(ligands_use),
        n_receptors_used = length(receptors_use),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    }
    if (length(lr_values) == 0) {
      stop("No LR pairs could be computed from bulk expression. Check LR pair names and gene symbols.")
    }
    lr_mat <- do.call(cbind, lr_values)
    rownames(lr_mat) <- sample_names
    colnames(lr_mat) <- names(lr_values)
    mode(lr_mat) <- "numeric"
    metadata_df <- do.call(rbind, metadata_rows)
    skipped_df <- if (length(skipped_rows) > 0) do.call(rbind, skipped_rows) else data.frame()
    list(matrix = lr_mat, metadata = metadata_df, skipped = skipped_df)
  }

  make_osqp_template_joint <- function(D_gene, D_lr, C, M,
                                       gene_weight_vec,
                                       lr_weight_vec,
                                       lambda_gene = 1.0,
                                       lambda_lr = 1.0,
                                       use_lr_loss = TRUE) {
    K <- C + M
    G <- ncol(D_gene)
    R_lr <- ncol(D_lr)
    gene_weight_vec <- as.numeric(gene_weight_vec)
    lr_weight_vec <- as.numeric(lr_weight_vec)
    if (length(gene_weight_vec) != G) {
      stop("Length of gene_weight_vec does not match number of genes in D_gene.")
    }
    if (length(lr_weight_vec) != R_lr) {
      stop("Length of lr_weight_vec does not match number of LR pairs in D_lr.")
    }
    if (any(!is.finite(gene_weight_vec)) || any(gene_weight_vec < 0)) {
      stop("gene_weight_vec contains non-finite or negative values.")
    }
    if (any(!is.finite(lr_weight_vec)) || any(lr_weight_vec < 0)) {
      stop("lr_weight_vec contains non-finite or negative values.")
    }
    W_gene_diag <- Diagonal(x = gene_weight_vec)
    Pmat <- 2 * (lambda_gene / G) * (D_gene %*% W_gene_diag %*% t(D_gene))
    if (isTRUE(use_lr_loss)) {
      W_lr_diag <- Diagonal(x = lr_weight_vec)
      Pmat <- Pmat + 2 * (lambda_lr / R_lr) * (D_lr %*% W_lr_diag %*% t(D_lr))
    }
    Pmat <- Matrix(Pmat, sparse = TRUE)
    # Intrinsic abundances are nonnegative and sum to one; MI abundances lie in [0, 1].
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
      gene_weight_vec = gene_weight_vec,
      lr_weight_vec = lr_weight_vec,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      G = G,
      R_lr = R_lr,
      use_lr_loss = use_lr_loss
    )
  }

  make_q_joint <- function(D_gene, D_lr, y_gene, y_lr, osqp_tpl) {
    q <- as.numeric(
      -2 * (osqp_tpl$lambda_gene / osqp_tpl$G) *
        (D_gene %*% (osqp_tpl$gene_weight_vec * y_gene))
    )
    if (isTRUE(osqp_tpl$use_lr_loss)) {
      q <- q + as.numeric(
        -2 * (osqp_tpl$lambda_lr / osqp_tpl$R_lr) *
          (D_lr %*% (osqp_tpl$lr_weight_vec * y_lr))
      )
    }
    q
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

  compute_weighted_sse_with_beta0_by_group <- function(sumsq_no_intercept,
                                                       sum_e_mat,
                                                       beta0_mat,
                                                       feature_weight_vec,
                                                       objective_weight_by_group,
                                                       beta0_ridge_lambda = 0) {
    sum_e_mat <- as.matrix(sum_e_mat)
    beta0_mat <- as.matrix(beta0_mat)
    feature_weight_vec <- as.numeric(feature_weight_vec)
    objective_weight_by_group <- as.numeric(objective_weight_by_group)
    if (!identical(dim(sum_e_mat), dim(beta0_mat))) {
      stop("sum_e_mat and beta0_mat must have identical dimensions.")
    }
    if (ncol(beta0_mat) != length(feature_weight_vec)) {
      stop("Number of beta0 columns does not match length of feature_weight_vec.")
    }
    if (nrow(beta0_mat) != length(objective_weight_by_group)) {
      stop("Number of beta0 rows does not match length of objective_weight_by_group.")
    }
    if (any(objective_weight_by_group <= 0)) {
      stop("objective_weight_by_group must be positive for all groups.")
    }
    beta0_sq_weighted_by_feature <- sweep(beta0_mat ^ 2, 2, feature_weight_vec, "*")
    linear_term <- sum(sweep(beta0_mat * sum_e_mat, 2, feature_weight_vec, "*"))
    quadratic_term <- sum(objective_weight_by_group * rowSums(beta0_sq_weighted_by_feature))
    ridge_penalty <- beta0_ridge_lambda * sum(beta0_sq_weighted_by_feature)
    as.numeric(sumsq_no_intercept - 2 * linear_term + quadratic_term + ridge_penalty)
  }

  # -----------------------------
  # Pass 1: download TCGA data, filter tumor samples, and build expression/LR matrices
  # -----------------------------
  X_raw_list <- list()
  LR_raw_list <- list()
  gene_avail <- list()
  lr_avail <- list()
  proj_meta <- list()
  clinical_data_full_list <- list()
  clinical_data_list <- list()
  tumor_filter_record_list <- list()

  all_available_celltypes <- rownames(Loading_intrinsic_use)

  for (project_cur in TCGA_project_list) {
    message("====================================================")
    message("Preparing real TCGA bulk RNA-seq for project: ", project_cur)
    project_cur_index <- which(TCGA_project_list == project_cur)

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

    if (!"sample_type" %in% colnames(coldata)) {
      stop("coldata$sample_type is missing for ", project_cur,
           ". Cannot define tumor_keep from sample_type.")
    }

    # Align sample annotations with expression columns before applying the tumor filter.
    if (!identical(colnames(expr), rownames(coldata))) {
      if (all(colnames(expr) %in% rownames(coldata))) {
        coldata <- coldata[colnames(expr), , drop = FALSE]
      } else {
        stop("Expression column names do not match coldata rownames for ", project_cur,
             ". Cannot safely apply tumor_keep.")
      }
    }

    clinical_data_full_list[[project_cur]] <- coldata

    sample_type_chr <- as.character(coldata$sample_type)
    tumor_keep <- sample_type_chr %in% tumor_sample_types
    nontumor_keep <- sample_type_chr %in% nontumor_sample_types

    sample_filter_df <- data.frame(
      sample = colnames(expr),
      project = project_cur,
      sample_type = sample_type_chr,
      tumor_keep = tumor_keep,
      nontumor_keep = nontumor_keep,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    tumor_filter_record_list[[project_cur]] <- sample_filter_df
    write.csv(
      sample_filter_df,
      file.path(output_dir, paste0("sample_type_tumor_filter_", project_cur, ".csv")),
      row.names = FALSE
    )

    message("Sample types in ", project_cur, ":")
    print(table(sample_type_chr, useNA = "ifany"))
    message("Tumor-only filter for ", project_cur, ": keeping ",
            sum(tumor_keep), " / ", length(tumor_keep), " samples.")

    if (sum(tumor_keep) == 0) {
      stop(
        "No tumor samples were identified for ", project_cur,
        " using coldata$sample_type. Observed sample_type values: ",
        paste(sort(unique(sample_type_chr)), collapse = "; "),
        ". Expected tumor sample types: ",
        paste(tumor_sample_types, collapse = "; ")
      )
    }

    # Restrict expression and clinical annotations before building either reconstruction block.
    expr <- expr[, tumor_keep, drop = FALSE]
    coldata <- coldata[tumor_keep, , drop = FALSE]
    clinical_data_list[[project_cur]] <- coldata

    expr_cpm <- cpm(expr, log = TRUE, prior.count = 1)
    rownames(expr_cpm) <- gene_annot$gene_name
    expr_cpm <- collapse_duplicate_gene_rows(expr_cpm)

    genes_here <- intersect(genename_list, rownames(expr_cpm))
    gene_avail[[project_cur]] <- genes_here
    X_raw_list[[project_cur]] <- expr_cpm

    expr_sample_gene <- t(as.matrix(expr_cpm))
    mode(expr_sample_gene) <- "numeric"

    lr_build <- build_lr_coexpression_matrix(
      expr_df = expr_sample_gene,
      lr_pair_names = colnames(loading_LR_use),
      require_all_genes = lr_require_all_genes
    )
    LR_raw_list[[project_cur]] <- lr_build$matrix
    lr_avail[[project_cur]] <- colnames(lr_build$matrix)

    write.csv(lr_build$metadata,
              file.path(output_dir, paste0("LR_pair_parse_metadata_", project_cur, ".csv")),
              row.names = FALSE)
    if (nrow(lr_build$skipped) > 0) {
      write.csv(lr_build$skipped,
                file.path(output_dir, paste0("LR_pair_skipped_missing_genes_", project_cur, ".csv")),
                row.names = FALSE)
    }

    if (!project_cur %in% names(celltype_list_by_project)) {
      stop("Project not found in celltype_list_by_project: ", project_cur)
    }
    celltype_list_use <- celltype_list_by_project[[project_cur]]
    missing_celltypes <- setdiff(celltype_list_use, all_available_celltypes)
    if (length(missing_celltypes) > 0) {
      stop("These cell types are missing in Loading_intrinsic_use for ", project_cur, ": ",
           paste(missing_celltypes, collapse = ", "))
    }

    W0_full <- as.matrix(Loading_intrinsic_use[celltype_list_use, , drop = FALSE])
    WMI_full <- as.matrix(loading_MI[, , drop = FALSE])
    WLR_full <- as.matrix(loading_LR_use[, , drop = FALSE])
    mode(W0_full) <- "numeric"
    mode(WMI_full) <- "numeric"
    mode(WLR_full) <- "numeric"

    proj_meta[[project_cur]] <- list(
      W0_full = W0_full,
      WMI_full = WMI_full,
      WLR_full = WLR_full,
      celltype_list_use = celltype_list_use,
      cohort = project_cur
    )
  }

  # Save the tumor-only sample filter summary across all projects.
  tumor_filter_summary_df <- do.call(rbind, tumor_filter_record_list)
  write.csv(
    tumor_filter_summary_df,
    file.path(output_dir, "sample_type_tumor_filter_all_TCGA_projects.csv"),
    row.names = FALSE
  )

  tumor_filter_count_df <- do.call(
    rbind,
    lapply(names(tumor_filter_record_list), function(project_cur) {
      df <- tumor_filter_record_list[[project_cur]]
      data.frame(
        project = project_cur,
        n_total = nrow(df),
        n_tumor_keep = sum(df$tumor_keep),
        n_nontumor = sum(df$nontumor_keep),
        n_other_or_unknown = sum(!df$tumor_keep & !df$nontumor_keep),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    })
  )
  write.csv(
    tumor_filter_count_df,
    file.path(output_dir, "sample_type_tumor_filter_counts_TCGA.csv"),
    row.names = FALSE
  )

  # -----------------------------
  # Determine common gene and LR-pair sets across all TCGA projects
  # -----------------------------
  genes_common <- Reduce(intersect, gene_avail)
  genes_common <- unique(genes_common)
  if (length(genes_common) < 10) {
    stop("Too few common genes across TCGA projects and loading matrices.")
  }
  message("Common genes used for TCGA decomposition: ", length(genes_common))

  lr_pairs_common <- Reduce(intersect, lr_avail)
  lr_pairs_common <- unique(lr_pairs_common)
  if (length(lr_pairs_common) < 2) {
    stop("Too few common LR pairs could be computed across TCGA projects.")
  }
  message("Common LR pairs used for TCGA decomposition: ", length(lr_pairs_common))

  # -----------------------------
  # Feature weights
  # -----------------------------
  gene_weight <- NULL
  if (isTRUE(use_gene_weight) && !isTRUE(if_nongeneweight)) {
    gene_weight <- compute_kl_specificity_weights(loading_MI, feature_label = "gene")
  } else {
    message("Skipping KL-based gene-weight calculation; gene-expression loss will use uniform gene weights.")
  }

  if (isTRUE(if_nongeneweight)) {
    gene_weight_common <- rep(1, length(genes_common))
    names(gene_weight_common) <- genes_common
  } else if (isTRUE(use_gene_weight)) {
    if (!all(genes_common %in% names(gene_weight))) {
      stop("Some common genes are missing in gene_weight.")
    }
    gene_weight_common <- as.numeric(gene_weight[genes_common])
    names(gene_weight_common) <- genes_common
  } else {
    gene_weight_common <- rep(1, length(genes_common))
    names(gene_weight_common) <- genes_common
  }

  W_LR_for_weight <- as.matrix(loading_LR_use[, lr_pairs_common, drop = FALSE])
  mode(W_LR_for_weight) <- "numeric"
  lr_weight <- NULL
  if (isTRUE(use_lr_weight) && !isTRUE(if_nonlrweight)) {
    lr_weight <- compute_kl_specificity_weights(W_LR_for_weight, feature_label = "LR-pair")
  } else {
    message("Skipping KL-based LR-pair-weight calculation; LR loss will use uniform LR-pair weights.")
  }

  if (isTRUE(if_nonlrweight)) {
    lr_weight_common <- rep(1, length(lr_pairs_common))
    names(lr_weight_common) <- lr_pairs_common
  } else if (isTRUE(use_lr_weight)) {
    if (!all(lr_pairs_common %in% names(lr_weight))) {
      stop("Some common LR pairs are missing in lr_weight.")
    }
    lr_weight_common <- as.numeric(lr_weight[lr_pairs_common])
    names(lr_weight_common) <- lr_pairs_common
  } else {
    lr_weight_common <- rep(1, length(lr_pairs_common))
    names(lr_weight_common) <- lr_pairs_common
  }

  # Applying diagonal specificity weights to residuals squares their contribution to SSE.
  if (isTRUE(use_paper_omega_weighting)) {
    gene_loss_weight_common <- gene_weight_common ^ 2
    lr_loss_weight_common <- lr_weight_common ^ 2
    message("Using paper-consistent Omega/Psi weighting: squared-error weights are specificity_weight^2.")
  } else {
    gene_loss_weight_common <- gene_weight_common
    lr_loss_weight_common <- lr_weight_common
    message("Using legacy weighting: squared-error weights are specificity_weight.")
  }
  names(gene_loss_weight_common) <- genes_common
  names(lr_loss_weight_common) <- lr_pairs_common

  if (any(!is.finite(gene_loss_weight_common)) || any(gene_loss_weight_common < 0) || all(gene_loss_weight_common == 0)) {
    stop("gene_loss_weight_common is invalid.")
  }
  if (any(!is.finite(lr_loss_weight_common)) || any(lr_loss_weight_common < 0) || all(lr_loss_weight_common == 0)) {
    stop("lr_loss_weight_common is invalid.")
  }

  write.csv(
    data.frame(gene = genes_common, gene_weight = gene_weight_common, gene_loss_weight = gene_loss_weight_common),
    file.path(output_dir, "gene_loss_weights_TCGA.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(lr_pair = lr_pairs_common, lr_weight = lr_weight_common, lr_loss_weight = lr_loss_weight_common),
    file.path(output_dir, "LR_loss_weights_TCGA.csv"),
    row.names = FALSE
  )

  # -----------------------------
  # Pass 1b: align expression, LR proxies, and loadings to the common features
  # -----------------------------
  X_list <- list()
  C_LR_list <- list()
  D_gene_list <- list()
  D_lr_list <- list()
  CDMK <- list()
  sample_weight_list <- list()

  for (project_cur in TCGA_project_list) {
    message("Finalize TCGA decomposition matrices for: ", project_cur)

    expr_cpm <- X_raw_list[[project_cur]]
    Xg <- t(as.matrix(expr_cpm[genes_common, , drop = FALSE]))
    mode(Xg) <- "numeric"
    if (any(!is.finite(Xg))) {
      stop("Non-finite values in X for ", project_cur)
    }

    C_lr <- as.matrix(LR_raw_list[[project_cur]][rownames(Xg), lr_pairs_common, drop = FALSE])
    mode(C_lr) <- "numeric"
    if (any(!is.finite(C_lr))) {
      stop("Non-finite values in C_LR for ", project_cur)
    }

    W0_full <- proj_meta[[project_cur]]$W0_full
    WMI_full <- proj_meta[[project_cur]]$WMI_full
    WLR_full <- proj_meta[[project_cur]]$WLR_full

    W0 <- as.matrix(W0_full[, genes_common, drop = FALSE])
    W_MI <- as.matrix(WMI_full[, genes_common, drop = FALSE])
    W_LR <- as.matrix(WLR_full[, lr_pairs_common, drop = FALSE])
    mode(W0) <- "numeric"
    mode(W_MI) <- "numeric"
    mode(W_LR) <- "numeric"

    if (any(!is.finite(W0)) || any(!is.finite(W_MI)) || any(!is.finite(W_LR))) {
      stop("Non-finite loading values for ", project_cur)
    }
    if (any(W_LR < 0)) {
      stop("Negative values found in loading_LR_use; expected non-negative LR loadings.")
    }

    D_gene <- rbind(W0, W_MI)
    D_gene <- as.matrix(D_gene)
    mode(D_gene) <- "numeric"

    D_lr <- rbind(
      matrix(0, nrow = nrow(W0), ncol = ncol(W_LR), dimnames = list(rownames(W0), colnames(W_LR))),
      W_LR
    )
    D_lr <- as.matrix(D_lr)
    mode(D_lr) <- "numeric"

    X_list[[project_cur]] <- Xg
    C_LR_list[[project_cur]] <- C_lr
    D_gene_list[[project_cur]] <- D_gene
    D_lr_list[[project_cur]] <- D_lr

    if (isTRUE(use_cohort_balanced_loss)) {
      sample_weight_list[[project_cur]] <- rep(1 / nrow(Xg), nrow(Xg))
    } else {
      sample_weight_list[[project_cur]] <- rep(1, nrow(Xg))
    }

    CDMK[[project_cur]] <- list(
      N = nrow(Xg),
      G = ncol(Xg),
      C = nrow(W0),
      M = nrow(W_MI),
      K = nrow(D_gene),
      R_lr = ncol(W_LR),
      sample_names = rownames(Xg),
      intrinsic_names = rownames(W0),
      mi_names = rownames(W_MI),
      lr_pair_names = colnames(W_LR)
    )
  }

  # Save cohort/sample weights for inspection.
  cohort_weight_df <- do.call(
    rbind,
    lapply(TCGA_project_list, function(project_cur) {
      data.frame(
        sample = CDMK[[project_cur]]$sample_names,
        cohort = project_cur,
        sample_objective_weight = sample_weight_list[[project_cur]],
        row.names = NULL,
        check.names = FALSE
      )
    })
  )
  write.csv(cohort_weight_df, file.path(output_dir, "cohort_sample_weights_TCGA.csv"), row.names = FALSE)

  # -----------------------------
  # Optional global scaling for gene and LR blocks separately
  # -----------------------------
  scale_factor_gene <- 1.0
  scale_factor_lr <- 1.0

  # The sampling seed is set only when optional scaling is enabled.
  if (isTRUE(use_scaling)) {
    set.seed(1)
    sample_cap <- 1e6
    pool_gene <- numeric(0)
    pool_lr <- numeric(0)

    for (project_cur in TCGA_project_list) {
      xvec <- as.numeric(X_list[[project_cur]])
      xvec <- xvec[is.finite(xvec)]
      if (length(xvec) > 0) {
        take <- min(length(xvec), ceiling(sample_cap / length(TCGA_project_list)))
        pool_gene <- c(pool_gene, sample(xvec, size = take))
      }
      lrvec <- as.numeric(C_LR_list[[project_cur]])
      lrvec <- lrvec[is.finite(lrvec)]
      if (length(lrvec) > 0) {
        take <- min(length(lrvec), ceiling(sample_cap / length(TCGA_project_list)))
        pool_lr <- c(pool_lr, sample(lrvec, size = take))
      }
    }

    pool_gene <- abs(pool_gene[is.finite(pool_gene)])
    pool_lr <- abs(pool_lr[is.finite(pool_lr)])
    if (length(pool_gene) == 0 || length(pool_lr) == 0) {
      stop("Cannot compute scaling factors because gene/LR pools are empty.")
    }

    scale_factor_gene <- as.numeric(stats::quantile(pool_gene, probs = 0.95, names = FALSE))
    scale_factor_lr <- as.numeric(stats::quantile(pool_lr, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor_gene) || scale_factor_gene <= 0) scale_factor_gene <- 1.0
    if (!is.finite(scale_factor_lr) || scale_factor_lr <= 0) scale_factor_lr <- 1.0

    message("Global scale_factor_gene (95% |X|): ", signif(scale_factor_gene, 4))
    message("Global scale_factor_lr (95% |C_LR|): ", signif(scale_factor_lr, 4))

    for (project_cur in TCGA_project_list) {
      X_list[[project_cur]] <- X_list[[project_cur]] / scale_factor_gene
      D_gene_list[[project_cur]] <- D_gene_list[[project_cur]] / scale_factor_gene
      C_LR_list[[project_cur]] <- C_LR_list[[project_cur]] / scale_factor_lr
      D_lr_list[[project_cur]] <- D_lr_list[[project_cur]] / scale_factor_lr
    }
  }

  # -----------------------------
  # Pass 2: alternate constrained sample fits and cohort-specific offset updates
  # -----------------------------
  G <- length(genes_common)
  R_lr <- length(lr_pairs_common)
  B <- length(TCGA_project_list)
  cohort_levels_beta0 <- TCGA_project_list
  objective_weight_by_cohort <- vapply(TCGA_project_list, function(p) sum(sample_weight_list[[p]]), numeric(1))
  objective_weight_sum <- sum(objective_weight_by_cohort)

  beta0_gene <- matrix(0, nrow = B, ncol = G, dimnames = list(cohort_levels_beta0, genes_common))
  beta0_LR <- matrix(0, nrow = B, ncol = R_lr, dimnames = list(cohort_levels_beta0, lr_pairs_common))

  P_hat_list <- list()
  MI_hat_list <- list()
  osqp_tpl <- list()

  for (project_cur in TCGA_project_list) {
    meta <- CDMK[[project_cur]]
    osqp_tpl[[project_cur]] <- make_osqp_template_joint(
      D_gene = D_gene_list[[project_cur]],
      D_lr = D_lr_list[[project_cur]],
      C = meta$C,
      M = meta$M,
      gene_weight_vec = gene_loss_weight_common,
      lr_weight_vec = lr_loss_weight_common,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      use_lr_loss = use_lr_loss
    )
    P_hat_list[[project_cur]] <- matrix(0, nrow = meta$N, ncol = meta$C)
    MI_hat_list[[project_cur]] <- matrix(0, nrow = meta$N, ncol = meta$M)
  }

  message("Using cohort-specific beta0 with TCGA projects as cohorts: ", paste(cohort_levels_beta0, collapse = ", "))
  prev_objective <- NA_real_

  for (it in seq_len(max_outer)) {
    sum_e_gene_by_cohort <- matrix(0, nrow = B, ncol = G, dimnames = list(cohort_levels_beta0, genes_common))
    sum_e_lr_by_cohort <- matrix(0, nrow = B, ncol = R_lr, dimnames = list(cohort_levels_beta0, lr_pairs_common))
    sumsq_gene <- 0.0
    sumsq_lr <- 0.0

    for (project_cur in TCGA_project_list) {
      b <- match(project_cur, cohort_levels_beta0)
      X <- X_list[[project_cur]]
      C_LR <- C_LR_list[[project_cur]]
      D_gene <- D_gene_list[[project_cur]]
      D_lr <- D_lr_list[[project_cur]]
      meta <- CDMK[[project_cur]]
      tpl <- osqp_tpl[[project_cur]]
      sample_w <- sample_weight_list[[project_cur]]

      y1_gene <- as.numeric(X[1, ] - beta0_gene[b, ])
      y1_lr <- as.numeric(C_LR[1, ] - beta0_LR[b, ])
      q1 <- make_q_joint(D_gene, D_lr, y1_gene, y1_lr, tpl)
      model <- init_model(P = tpl$P, A = tpl$A, l = tpl$l, u = tpl$u, q_init = q1)

      for (i in seq_len(meta$N)) {
        y_gene <- as.numeric(X[i, ] - beta0_gene[b, ])
        y_lr <- as.numeric(C_LR[i, ] - beta0_LR[b, ])
        q <- make_q_joint(D_gene, D_lr, y_gene, y_lr, tpl)
        model$Update(q = q)
        r <- model$Solve()

        if (r$info$status_val != 1L) {
          stop(sprintf("OSQP failed | outer=%d project=%s row=%d status=%s", it, project_cur, i, r$info$status))
        }

        z <- r$x
        if (any(!is.finite(z))) {
          stop(sprintf("Non-finite z | outer=%d project=%s row=%d", it, project_cur, i))
        }

        C <- meta$C
        K <- meta$K
        P_hat_list[[project_cur]][i, ] <- z[1:C]
        MI_hat_list[[project_cur]][i, ] <- z[(C + 1):K]

        pred_gene_no_intercept <- as.numeric(z %*% D_gene)
        e_gene <- as.numeric(X[i, ] - pred_gene_no_intercept)
        pred_lr_no_intercept <- as.numeric(z %*% D_lr)
        e_lr <- as.numeric(C_LR[i, ] - pred_lr_no_intercept)

        row_w <- sample_w[i]
        sum_e_gene_by_cohort[b, ] <- sum_e_gene_by_cohort[b, ] + row_w * e_gene
        sumsq_gene <- sumsq_gene + row_w * sum((e_gene ^ 2) * tpl$gene_weight_vec)

        if (isTRUE(use_lr_loss)) {
          sum_e_lr_by_cohort[b, ] <- sum_e_lr_by_cohort[b, ] + row_w * e_lr
          sumsq_lr <- sumsq_lr + row_w * sum((e_lr ^ 2) * tpl$lr_weight_vec)
        }
      }
    }

    if (isTRUE(if_set_beta0_zero)) {
      beta0_gene_new <- matrix(0, nrow = B, ncol = G, dimnames = list(cohort_levels_beta0, genes_common))
    } else {
      beta0_gene_denominator <- objective_weight_by_cohort
      if (isTRUE(use_beta0_ridge)) beta0_gene_denominator <- beta0_gene_denominator + beta0_ridge_lambda_gene
      beta0_gene_proposed <- sweep(sum_e_gene_by_cohort, 1, beta0_gene_denominator, "/")
      if (isTRUE(if_beta0_nonnegative)) beta0_gene_proposed <- pmax(beta0_gene_proposed, 0)
      beta0_gene_new <- (1 - beta0_step_size) * beta0_gene + beta0_step_size * beta0_gene_proposed
    }

    sse_gene <- compute_weighted_sse_with_beta0_by_group(
      sumsq_no_intercept = sumsq_gene,
      sum_e_mat = sum_e_gene_by_cohort,
      beta0_mat = beta0_gene_new,
      feature_weight_vec = gene_loss_weight_common,
      objective_weight_by_group = objective_weight_by_cohort,
      beta0_ridge_lambda = if (isTRUE(use_beta0_ridge)) beta0_ridge_lambda_gene else 0
    )

    if (isTRUE(use_lr_loss)) {
      if (isTRUE(if_set_beta0_LR_zero)) {
        beta0_LR_new <- matrix(0, nrow = B, ncol = R_lr, dimnames = list(cohort_levels_beta0, lr_pairs_common))
      } else {
        beta0_LR_denominator <- objective_weight_by_cohort
        if (isTRUE(use_beta0_ridge)) beta0_LR_denominator <- beta0_LR_denominator + beta0_ridge_lambda_lr
        beta0_LR_proposed <- sweep(sum_e_lr_by_cohort, 1, beta0_LR_denominator, "/")
        if (isTRUE(if_beta0_nonnegative)) beta0_LR_proposed <- pmax(beta0_LR_proposed, 0)
        beta0_LR_new <- (1 - beta0_step_size) * beta0_LR + beta0_step_size * beta0_LR_proposed
      }
      sse_lr <- compute_weighted_sse_with_beta0_by_group(
        sumsq_no_intercept = sumsq_lr,
        sum_e_mat = sum_e_lr_by_cohort,
        beta0_mat = beta0_LR_new,
        feature_weight_vec = lr_loss_weight_common,
        objective_weight_by_group = objective_weight_by_cohort,
        beta0_ridge_lambda = if (isTRUE(use_beta0_ridge)) beta0_ridge_lambda_lr else 0
      )
    } else {
      beta0_LR_new <- beta0_LR
      sse_lr <- 0.0
    }

    if (!is.finite(sse_gene) || !is.finite(sse_lr)) {
      stop("SSE became non-finite; try stronger scaling or check inputs.")
    }
    if (sse_gene < 0) sse_gene <- 0
    if (sse_lr < 0) sse_lr <- 0

    gene_component <- ((lambda_gene / G) * sse_gene) / objective_weight_sum
    lr_component <- if (isTRUE(use_lr_loss)) ((lambda_lr / R_lr) * sse_lr) / objective_weight_sum else 0
    objective_value <- gene_component + lr_component

    if (!is.finite(objective_value)) {
      stop("Objective value became non-finite; try stronger scaling or check inputs.")
    }

    if (is.na(prev_objective)) {
      cat(sprintf("Outer %d | joint objective=%.6g | gene=%.6g | LR=%.6g (init)\n",
                  it, objective_value, gene_component, lr_component))
      prev_objective <- objective_value
      beta0_gene <- beta0_gene_new
      beta0_LR <- beta0_LR_new
      if (isTRUE(if_set_beta0_zero) && (!isTRUE(use_lr_loss) || isTRUE(if_set_beta0_LR_zero))) {
        cat("Fixed beta0 mode: no outer beta0 update is needed. Stopping after one outer iteration.\n")
        break
      }
    } else {
      # Normalize objective change by max(1, abs(previous objective)).
      rel_change <- abs(prev_objective - objective_value) / max(1, abs(prev_objective))
      cat(sprintf("Outer %d | joint objective=%.6g | gene=%.6g | LR=%.6g | rel_change=%.3g\n",
                  it, objective_value, gene_component, lr_component, rel_change))
      beta0_gene <- beta0_gene_new
      beta0_LR <- beta0_LR_new
      if (rel_change < tol) break
      prev_objective <- objective_value
    }
  }

  # -----------------------------
  # Unscale beta0 matrices back to original data scale
  # -----------------------------
  if (isTRUE(use_scaling) && is.finite(scale_factor_gene) && scale_factor_gene != 1.0) {
    beta0_gene <- beta0_gene * scale_factor_gene
  }
  if (isTRUE(use_scaling) && is.finite(scale_factor_lr) && scale_factor_lr != 1.0) {
    beta0_LR <- beta0_LR * scale_factor_lr
  }

  cat("✅ Joint TCGA decomposition finished.\n")

  # -----------------------------
  # Save outputs per TCGA project
  # -----------------------------
  for (project_cur in TCGA_project_list) {
    MI_hat <- MI_hat_list[[project_cur]]
    MI_hat[MI_hat < 0] <- 0
    MI_hat[MI_hat > 1] <- 1
    rownames(MI_hat) <- CDMK[[project_cur]]$sample_names
    colnames(MI_hat) <- CDMK[[project_cur]]$mi_names

    write.csv(
      MI_hat,
      file.path(output_dir, paste0("MI_decomposition_", project_cur, ".csv"))
    )

    P_hat <- P_hat_list[[project_cur]]
    P_hat[P_hat < 0] <- 0
    rownames(P_hat) <- CDMK[[project_cur]]$sample_names
    colnames(P_hat) <- CDMK[[project_cur]]$intrinsic_names

    write.csv(
      P_hat,
      file.path(output_dir, paste0("Intrinsic_decomposition_", project_cur, ".csv"))
    )

    write.csv(
      C_LR_list[[project_cur]],
      file.path(output_dir, paste0("LR_coexpression_bulk_", project_cur, ".csv"))
    )
  }

  write.csv(beta0_gene, file.path(output_dir, "beta0_gene_by_TCGA_project_matrix.csv"))
  write.csv(beta0_LR, file.path(output_dir, "beta0_LR_by_TCGA_project_matrix.csv"))

  settings_df <- data.frame(
    setting = c(
      "decomposition_input", "beta0_mode", "n_beta0_cohorts", "beta0_cohort_labels",
      "if_set_beta0_zero", "if_set_beta0_LR_zero", "if_beta0_nonnegative",
      "use_beta0_ridge", "beta0_ridge_lambda_gene", "beta0_ridge_lambda_lr", "beta0_step_size",
      "if_nongeneweight", "if_nonlrweight", "use_gene_weight", "use_lr_weight",
      "use_lr_loss", "lambda_gene", "lambda_lr", "use_paper_omega_weighting",
      "use_cohort_balanced_loss", "use_scaling", "scale_factor_gene", "scale_factor_lr",
      "lr_require_all_genes", "n_common_genes", "n_common_lr_pairs"
    ),
    value = c(
      "real_TCGA_STAR_Counts_logCPM_tumor_only", "TCGA_project_specific", as.character(B), paste(cohort_levels_beta0, collapse = ";"),
      as.character(if_set_beta0_zero), as.character(if_set_beta0_LR_zero), as.character(if_beta0_nonnegative),
      as.character(use_beta0_ridge), as.character(beta0_ridge_lambda_gene), as.character(beta0_ridge_lambda_lr), as.character(beta0_step_size),
      as.character(if_nongeneweight), as.character(if_nonlrweight), as.character(use_gene_weight), as.character(use_lr_weight),
      as.character(use_lr_loss), as.character(lambda_gene), as.character(lambda_lr), as.character(use_paper_omega_weighting),
      as.character(use_cohort_balanced_loss), as.character(use_scaling), as.character(scale_factor_gene), as.character(scale_factor_lr),
      as.character(lr_require_all_genes), as.character(G), as.character(R_lr)
    ),
    row.names = NULL,
    check.names = FALSE
  )

  settings_df <- rbind(
    settings_df,
    data.frame(
      setting = c(
        "use_tumor_only_samples",
        "tumor_sample_types",
        "tumor_filter_summary_file",
        "tumor_filter_count_file"
      ),
      value = c(
        "TRUE",
        paste(tumor_sample_types, collapse = ";"),
        "sample_type_tumor_filter_all_TCGA_projects.csv",
        "sample_type_tumor_filter_counts_TCGA.csv"
      ),
      row.names = NULL,
      check.names = FALSE
    )
  )
  write.csv(settings_df, file.path(output_dir, "decomposition_settings_TCGA_joint_gene_LR.csv"), row.names = FALSE)

  cat("✅ Saved tumor-only TCGA MI decomposition per project, intrinsic decomposition, LR coexpression, and project-specific beta0.\n")
  cat("TCGA projection outputs saved; spatial pseudo-bulk validation is a separate section below.\n")


  plot_tcga_summaries(output_dir, output_dir, TCGA_project_list)

}

# =============================================================================
# spatial pseudo-bulk validation of MI-4 recovery
# =============================================================================
# Inputs are exported by Pancancer_analysis_V2.ipynb: geneexp_mean_df.csv,
# MI_mean_df.csv, and the four loading_*_use.csv files from the same model run.
# Refit the original joint gene-LR model on spatial pseudo-bulks, then compare
# projected MI-4 with observed sub-slice-average MI-4 using type-7 tertiles and
# two-sided adjacent Wilcoxon tests (exact = FALSE). All MIs remain in the fit.
# This function does not download TCGA data or source another analysis script.
run_s26b_pseudobulk_validation <- function(input_dir, output_dir,
                                           mi_to_plot = "MI-4") {
  suppressPackageStartupMessages({
    library(osqp)
    library(Matrix)
  })

  # -----------------------------
  # User switches
  # -----------------------------
  # Internal testing switches:
  #   if_set_beta0_zero = TRUE: fix cohort-specific beta0_gene to zero during optimization.
  #   if_set_beta0_LR_zero = TRUE: fix cohort-specific beta0_LR to zero during optimization.
  #   if_beta0_nonnegative = TRUE: when beta0 is estimated, constrain both beta0_gene
  #       and beta0_LR to be non-negative by projecting each cohort-specific
  #       intercept update onto beta0 >= 0. This has no additional effect when the
  #       corresponding beta0 matrix is fixed to zero, because zero is already non-negative.
  #   if_nongeneweight = TRUE: force gene weights to 1 across all genes.
  #   if_nonlrweight   = TRUE: force LR-pair weights to 1 across all LR pairs.
  if_set_beta0_zero <- FALSE
  if_set_beta0_LR_zero <- FALSE
  if_beta0_nonnegative <- FALSE

  # Ridge-shrinkage and damping for estimated beta0.
  #   use_beta0_ridge = TRUE shrinks estimated beta0 toward zero so beta0 cannot
  #       freely absorb signal that should be explained by H/MI activities.
  #   beta0_ridge_lambda_* controls shrinkage strength. Larger values produce
  #       smaller beta0 estimates. The update becomes:
  #       beta0 = sum_i weight_i * residual_i / (sum_i weight_i + ridge_lambda).
  #   beta0_step_size controls damping of the outer beta0 update. Values below 1
  #       make beta0 change more gradually and usually stabilize the alternating
  #       optimization.
  use_beta0_ridge <- FALSE
  beta0_ridge_lambda_gene <- 10.0
  beta0_ridge_lambda_lr <- 50.0
  beta0_step_size <- 0.25

  if (!is.logical(use_beta0_ridge) || length(use_beta0_ridge) != 1L) {
    stop("use_beta0_ridge must be a single TRUE/FALSE value.")
  }
  if (!is.finite(beta0_ridge_lambda_gene) || beta0_ridge_lambda_gene < 0) {
    stop("beta0_ridge_lambda_gene must be finite and non-negative.")
  }
  if (!is.finite(beta0_ridge_lambda_lr) || beta0_ridge_lambda_lr < 0) {
    stop("beta0_ridge_lambda_lr must be finite and non-negative.")
  }
  if (!is.finite(beta0_step_size) || beta0_step_size <= 0 || beta0_step_size > 1) {
    stop("beta0_step_size must be in the interval (0, 1].")
  }


  if_nongeneweight <- FALSE
  if_nonlrweight <- FALSE


  # -----------------------------------------------------------------------------------------------------------

  # Joint-loss switches:
  use_lr_loss <- TRUE
  lambda_gene <- 1.0
  lambda_lr <- 0.5

  # Loss-type switch:
  #   loss_type = "MSE" uses the original weighted squared-error objective.
  #   loss_type = "MAE" uses a weighted absolute-error objective. Because the
  #       per-sample z update is then an L1 problem, it is solved as an LP/QP with
  #       auxiliary absolute-residual variables through OSQP. MAE is more robust
  #       to outlier genes/LR pairs, but is usually slower than MSE.
  loss_type <- "MSE"
  loss_type <- toupper(loss_type)
  if (!(loss_type %in% c("MSE", "MAE"))) {
    stop("loss_type must be either 'MSE' or 'MAE'.")
  }

  # Small quadratic regularization used only for MAE/LP mode to improve OSQP
  # numerical stability. This does not change the MSE objective.
  mae_qp_l2_epsilon <- 1e-8
  if (!is.finite(mae_qp_l2_epsilon) || mae_qp_l2_epsilon < 0) {
    stop("mae_qp_l2_epsilon must be finite and non-negative.")
  }

  # Paper-consistency switches:
  #   use_paper_omega_weighting = TRUE implements || residual %*% Omega ||_F^2
  #   with Omega = diag(w), i.e. the squared-error contribution is w^2 * e^2.
  #   If FALSE, use the legacy implementation w * e^2, equivalent to diag(sqrt(w)).
  use_paper_omega_weighting <- TRUE

  # Cohort-balanced objective:
  #   TRUE implements sum_p 1/J_p times each sample contribution, so each cohort
  #   has equal total weight regardless of sample size.
  use_cohort_balanced_loss <- TRUE

  # Cohort label settings.
  cohort_col <- NULL
  infer_cohort_from_sample_name <- TRUE

  # LR-pair parsing settings.
  # TRUE requires every ligand/receptor subunit in an LR-pair name to be present
  # in the bulk expression matrix. FALSE uses available subunits if at least one
  # ligand and one receptor gene are present.
  lr_require_all_genes <- FALSE

  use_gene_weight <- !if_nongeneweight
  use_lr_weight <- !if_nonlrweight
  use_scaling <- FALSE
  max_outer <- 1000
  tol <- 1e-7
  if_traindata <- FALSE

  # -----------------------------
  # Input / output paths
  # -----------------------------
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  bulk_expr_path <- file.path(input_dir, "geneexp_mean_df.csv")
  loading_intrinsic_path <- file.path(input_dir, "loading_intrinsic_use.csv")
  loading_sender_path <- file.path(input_dir, "loading_sender_use.csv")
  loading_receiver_path <- file.path(input_dir, "loading_receiver_use.csv")
  loading_LR_path <- file.path(input_dir, "loading_LR_use.csv")

  # -----------------------------
  # Helper functions
  # -----------------------------
  KL_divergence <- function(p, q) {
    eps <- 1e-10
    p <- p + eps
    p <- p / sum(p)
    q <- q + eps
    q <- q / sum(q)
    sum(p * log(p / q))
  }

  compute_weighted_sse_with_beta0 <- function(sumsq_no_intercept, sum_e_vec, beta0_vec, feature_weight_vec, objective_weight_sum, beta0_ridge_lambda = 0) {
    # Computes weighted SSE after applying a shared feature-specific intercept beta0.
    # Given residuals without intercept:
    #   e_ij = y_ij - prediction_without_beta0_ij
    # and sample weights a_i plus feature weights w_j, the objective contribution is:
    #   sum_i a_i sum_j w_j (e_ij - beta0_j)^2
    # The loop stores:
    #   sumsq_no_intercept = sum_i a_i sum_j w_j e_ij^2
    #   sum_e_vec[j]       = sum_i a_i e_ij
    # This function returns the exact SSE for any beta0 vector, including
    # constrained beta0 updates such as beta0 >= 0.
    # If beta0_ridge_lambda > 0, this returns:
    #   SSE(beta0) + beta0_ridge_lambda * sum_j w_j * beta0_j^2
    # which matches the ridge-shrunk beta0 update used below.
    beta0_vec <- as.numeric(beta0_vec)
    sum_e_vec <- as.numeric(sum_e_vec)
    feature_weight_vec <- as.numeric(feature_weight_vec)

    if (length(beta0_vec) != length(sum_e_vec) || length(beta0_vec) != length(feature_weight_vec)) {
      stop("Lengths of beta0_vec, sum_e_vec and feature_weight_vec do not match.")
    }
    if (any(!is.finite(beta0_vec)) || any(!is.finite(sum_e_vec)) ||
        any(!is.finite(feature_weight_vec)) || !is.finite(sumsq_no_intercept) ||
        !is.finite(objective_weight_sum) || !is.finite(beta0_ridge_lambda)) {
      stop("Non-finite input found when computing weighted SSE with beta0.")
    }
    if (beta0_ridge_lambda < 0) {
      stop("beta0_ridge_lambda must be non-negative.")
    }

    weighted_sse <- as.numeric(
      sumsq_no_intercept -
        2 * sum(feature_weight_vec * beta0_vec * sum_e_vec) +
        objective_weight_sum * sum(feature_weight_vec * (beta0_vec ^ 2))
    )

    ridge_penalty <- as.numeric(
      beta0_ridge_lambda * sum(feature_weight_vec * (beta0_vec ^ 2))
    )

    weighted_sse + ridge_penalty
  }


  compute_weighted_sse_with_beta0_by_group <- function(sumsq_no_intercept,
                                                       sum_e_mat,
                                                       beta0_mat,
                                                       feature_weight_vec,
                                                       objective_weight_by_group,
                                                       beta0_ridge_lambda = 0) {
    # Computes weighted SSE after applying cohort/batch-specific feature intercepts.
    # For group/cohort b and feature j:
    #   e_ij = y_ij - prediction_without_beta0_ij
    #   beta0_bj is the intercept used for samples in cohort b.
    # The loop stores:
    #   sumsq_no_intercept = sum_i a_i sum_j w_j e_ij^2
    #   sum_e_mat[b, j]    = sum_{i in b} a_i e_ij
    #   objective_weight_by_group[b] = sum_{i in b} a_i
    # This returns:
    #   sum_i a_i sum_j w_j (e_ij - beta0_{batch_i,j})^2
    #   + beta0_ridge_lambda * sum_b sum_j w_j beta0_bj^2.
    sum_e_mat <- as.matrix(sum_e_mat)
    beta0_mat <- as.matrix(beta0_mat)
    feature_weight_vec <- as.numeric(feature_weight_vec)
    objective_weight_by_group <- as.numeric(objective_weight_by_group)

    if (!identical(dim(sum_e_mat), dim(beta0_mat))) {
      stop("sum_e_mat and beta0_mat must have identical dimensions.")
    }
    if (ncol(beta0_mat) != length(feature_weight_vec)) {
      stop("Number of beta0 columns does not match length of feature_weight_vec.")
    }
    if (nrow(beta0_mat) != length(objective_weight_by_group)) {
      stop("Number of beta0 rows does not match length of objective_weight_by_group.")
    }
    if (any(!is.finite(beta0_mat)) || any(!is.finite(sum_e_mat)) ||
        any(!is.finite(feature_weight_vec)) || any(!is.finite(objective_weight_by_group)) ||
        !is.finite(sumsq_no_intercept) || !is.finite(beta0_ridge_lambda)) {
      stop("Non-finite input found when computing weighted SSE with group-specific beta0.")
    }
    if (any(objective_weight_by_group <= 0)) {
      stop("objective_weight_by_group must be positive for all groups.")
    }
    if (beta0_ridge_lambda < 0) {
      stop("beta0_ridge_lambda must be non-negative.")
    }

    beta0_sq_weighted_by_feature <- sweep(beta0_mat ^ 2, 2, feature_weight_vec, "*")

    linear_term <- sum(sweep(beta0_mat * sum_e_mat, 2, feature_weight_vec, "*"))
    quadratic_term <- sum(objective_weight_by_group * rowSums(beta0_sq_weighted_by_feature))
    ridge_penalty <- beta0_ridge_lambda * sum(beta0_sq_weighted_by_feature)

    as.numeric(sumsq_no_intercept - 2 * linear_term + quadratic_term + ridge_penalty)
  }

  compute_kl_specificity_weights <- function(loading_mat, feature_label = "feature") {
    loading_mat <- as.matrix(loading_mat)
    mode(loading_mat) <- "numeric"

    if (any(!is.finite(loading_mat))) {
      stop(sprintf("%s loading matrix contains non-finite values.", feature_label))
    }
    if (any(loading_mat < 0)) {
      stop(sprintf("%s loading matrix contains negative values; KL-specificity weights assume non-negative loadings.", feature_label))
    }

    n_dim <- nrow(loading_mat)
    n_feature <- ncol(loading_mat)
    uniform_vec <- rep(1 / n_dim, n_dim)
    specificity <- numeric(n_feature)

    for (feature_index in seq_len(n_feature)) {
      x <- loading_mat[, feature_index]
      x_sum <- sum(x)
      if (!is.finite(x_sum) || x_sum <= 1e-12) {
        specificity[feature_index] <- 0
      } else {
        specificity[feature_index] <- KL_divergence(x / x_sum, uniform_vec)
      }
    }

    specificity_sum <- sum(specificity)
    if (!is.finite(specificity_sum) || specificity_sum <= 0) {
      stop(sprintf("Cannot normalize %s KL-specificity weights because all specificity scores are zero/non-finite.", feature_label))
    }

    weight <- n_feature * specificity / specificity_sum
    names(weight) <- colnames(loading_mat)
    weight
  }

  parse_lr_pair_name <- function(lr_name) {
    parts <- strsplit(lr_name, "\\s*->\\s*")[[1]]
    if (length(parts) != 2) {
      stop(sprintf("Cannot parse LR pair name '%s'. Expected format 'ligand -> receptor'.", lr_name))
    }

    ligands <- trimws(unlist(strsplit(parts[1], "\\+", fixed = FALSE)))
    receptors <- trimws(unlist(strsplit(parts[2], "\\+", fixed = FALSE)))
    ligands <- ligands[ligands != ""]
    receptors <- receptors[receptors != ""]

    if (length(ligands) == 0 || length(receptors) == 0) {
      stop(sprintf("LR pair name '%s' has empty ligand or receptor set.", lr_name))
    }

    list(ligands = unique(ligands), receptors = unique(receptors))
  }

  build_lr_coexpression_matrix <- function(expr_df, lr_pair_names, require_all_genes = TRUE) {
    if (nrow(expr_df) == 0 || ncol(expr_df) == 0) {
      stop("expr_df is empty; cannot build LR coexpression matrix.")
    }

    all_genes <- colnames(expr_df)
    sample_names <- rownames(expr_df)

    lr_values <- list()
    metadata_rows <- list()
    skipped_rows <- list()

    for (lr_name in lr_pair_names) {
      parsed <- parse_lr_pair_name(lr_name)
      ligands <- parsed$ligands
      receptors <- parsed$receptors

      ligand_present <- ligands %in% all_genes
      receptor_present <- receptors %in% all_genes

      keep_pair <- FALSE
      if (isTRUE(require_all_genes)) {
        keep_pair <- all(ligand_present) && all(receptor_present)
      } else {
        keep_pair <- any(ligand_present) && any(receptor_present)
      }

      if (!keep_pair) {
        skipped_rows[[length(skipped_rows) + 1]] <- data.frame(
          lr_pair = lr_name,
          ligands = paste(ligands, collapse = "+"),
          receptors = paste(receptors, collapse = "+"),
          missing_ligands = paste(ligands[!ligand_present], collapse = "+"),
          missing_receptors = paste(receptors[!receptor_present], collapse = "+"),
          stringsAsFactors = FALSE,
          check.names = FALSE
        )
        next
      }

      ligands_use <- ligands[ligand_present]
      receptors_use <- receptors[receptor_present]

      expr_use <- as.matrix(expr_df[, unique(c(ligands_use, receptors_use)), drop = FALSE])
      mode(expr_use) <- "numeric"
      if (any(!is.finite(expr_use))) {
        stop(sprintf("Non-finite bulk expression values found for LR pair '%s'.", lr_name))
      }
      if (any(expr_use < 0)) {
        stop(sprintf("Negative bulk expression values found for LR pair '%s'. This script assumes non-negative expression.", lr_name))
      }

      ligand_expr <- rowMeans(as.matrix(expr_df[, ligands_use, drop = FALSE]))
      receptor_expr <- rowMeans(as.matrix(expr_df[, receptors_use, drop = FALSE]))
      coexpr <- sqrt(ligand_expr * receptor_expr)

      lr_values[[lr_name]] <- as.numeric(coexpr)
      metadata_rows[[length(metadata_rows) + 1]] <- data.frame(
        lr_pair = lr_name,
        ligands = paste(ligands, collapse = "+"),
        receptors = paste(receptors, collapse = "+"),
        ligands_used = paste(ligands_use, collapse = "+"),
        receptors_used = paste(receptors_use, collapse = "+"),
        n_ligands_used = length(ligands_use),
        n_receptors_used = length(receptors_use),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    }

    if (length(lr_values) == 0) {
      stop("No LR pairs could be computed from bulk expression. Check LR pair names and gene symbols.")
    }

    lr_mat <- do.call(cbind, lr_values)
    rownames(lr_mat) <- sample_names
    colnames(lr_mat) <- names(lr_values)
    mode(lr_mat) <- "numeric"

    metadata_df <- do.call(rbind, metadata_rows)
    skipped_df <- if (length(skipped_rows) > 0) do.call(rbind, skipped_rows) else data.frame()

    list(matrix = lr_mat, metadata = metadata_df, skipped = skipped_df)
  }

  get_cohort_labels <- function(sample_names, metadata_df = NULL, cohort_col = NULL,
                                infer_from_sample_name = TRUE) {
    if (!is.null(metadata_df)) {
      metadata_df <- metadata_df[sample_names, , drop = FALSE]

      if (!is.null(cohort_col)) {
        if (!(cohort_col %in% colnames(metadata_df))) {
          stop(sprintf("cohort_col='%s' was specified but was not found in MI_mean_df.csv.", cohort_col))
        }
        cohort_vec <- as.character(metadata_df[[cohort_col]])
        if (all(is.na(cohort_vec)) || length(unique(na.omit(cohort_vec))) == 0) {
          stop(sprintf("cohort_col='%s' contains no usable cohort labels.", cohort_col))
        }
        return(cohort_vec)
      }

      candidate_cols <- c(
        "CancerType", "Cancer_type", "cancer_type", "Cancer", "cancer",
        "Cohort", "cohort", "TCGA_cohort", "tcga_cohort",
        "Dataset", "dataset", "Study", "study", "Project", "project"
      )
      candidate_cols <- intersect(candidate_cols, colnames(metadata_df))

      for (cc in candidate_cols) {
        cohort_vec <- as.character(metadata_df[[cc]])
        usable <- !is.na(cohort_vec) & cohort_vec != ""
        n_unique <- length(unique(cohort_vec[usable]))
        if (sum(usable) > 0 && n_unique >= 1 && n_unique < length(cohort_vec)) {
          message("Auto-detected cohort column in MI_mean_df.csv: ", cc)
          return(cohort_vec)
        }
      }
    }

    if (isTRUE(infer_from_sample_name)) {
      cohort_vec <- sub("([_\\-\\.\\|].*)$", "", sample_names)
      n_unique <- length(unique(cohort_vec))
      if (n_unique >= 1 && n_unique < length(sample_names)) {
        message("No cohort column detected; inferred cohort labels from sample-name prefixes.")
        return(cohort_vec)
      }
    }

    message("No usable cohort labels found; treating all samples as one cohort.")
    rep("AllSamples", length(sample_names))
  }

  make_osqp_template_joint <- function(D_gene, D_lr, C, M,
                                       gene_weight_vec,
                                       lr_weight_vec,
                                       lambda_gene = 1.0,
                                       lambda_lr = 1.0,
                                       use_lr_loss = TRUE) {
    K <- C + M
    G <- ncol(D_gene)
    R_lr <- ncol(D_lr)

    gene_weight_vec <- as.numeric(gene_weight_vec)
    lr_weight_vec <- as.numeric(lr_weight_vec)

    if (length(gene_weight_vec) != G) {
      stop("Length of gene_weight_vec does not match number of genes in D_gene.")
    }
    if (length(lr_weight_vec) != R_lr) {
      stop("Length of lr_weight_vec does not match number of LR pairs in D_lr.")
    }
    if (any(!is.finite(gene_weight_vec)) || any(gene_weight_vec < 0)) {
      stop("gene_weight_vec contains non-finite or negative values.")
    }
    if (any(!is.finite(lr_weight_vec)) || any(lr_weight_vec < 0)) {
      stop("lr_weight_vec contains non-finite or negative values.")
    }

    W_gene_diag <- Diagonal(x = gene_weight_vec)
    Pmat <- 2 * (lambda_gene / G) * (D_gene %*% W_gene_diag %*% t(D_gene))

    if (isTRUE(use_lr_loss)) {
      W_lr_diag <- Diagonal(x = lr_weight_vec)
      Pmat <- Pmat + 2 * (lambda_lr / R_lr) * (D_lr %*% W_lr_diag %*% t(D_lr))
    }

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
      gene_weight_vec = gene_weight_vec,
      lr_weight_vec = lr_weight_vec,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      G = G,
      R_lr = R_lr,
      use_lr_loss = use_lr_loss
    )
  }

  make_q_joint <- function(D_gene, D_lr, y_gene, y_lr, osqp_tpl) {
    q <- as.numeric(
      -2 * (osqp_tpl$lambda_gene / osqp_tpl$G) *
        (D_gene %*% (osqp_tpl$gene_weight_vec * y_gene))
    )

    if (isTRUE(osqp_tpl$use_lr_loss)) {
      q <- q + as.numeric(
        -2 * (osqp_tpl$lambda_lr / osqp_tpl$R_lr) *
          (D_lr %*% (osqp_tpl$lr_weight_vec * y_lr))
      )
    }

    q
  }


  make_osqp_template_joint_mae <- function(D_gene, D_lr, C, M,
                                           gene_weight_vec,
                                           lr_weight_vec,
                                           lambda_gene = 1.0,
                                           lambda_lr = 1.0,
                                           use_lr_loss = TRUE,
                                           mae_qp_l2_epsilon = 1e-8) {
    # Exact MAE/L1 per-sample z update using auxiliary absolute-error variables.
    # Variables are:
    #   x = [z, t_gene, t_lr]
    # where z contains intrinsic proportions and MI activities, and t_* are
    # non-negative absolute residuals.
    K <- C + M
    G <- ncol(D_gene)
    R_lr <- ncol(D_lr)
    R_use <- if (isTRUE(use_lr_loss)) R_lr else 0
    nvar <- K + G + R_use

    gene_weight_vec <- as.numeric(gene_weight_vec)
    lr_weight_vec <- as.numeric(lr_weight_vec)

    if (length(gene_weight_vec) != G) {
      stop("Length of gene_weight_vec does not match number of genes in D_gene.")
    }
    if (length(lr_weight_vec) != R_lr) {
      stop("Length of lr_weight_vec does not match number of LR pairs in D_lr.")
    }
    if (any(!is.finite(gene_weight_vec)) || any(gene_weight_vec < 0)) {
      stop("gene_weight_vec contains non-finite or negative values.")
    }
    if (any(!is.finite(lr_weight_vec)) || any(lr_weight_vec < 0)) {
      stop("lr_weight_vec contains non-finite or negative values.")
    }
    if (!is.finite(mae_qp_l2_epsilon) || mae_qp_l2_epsilon < 0) {
      stop("mae_qp_l2_epsilon must be finite and non-negative.")
    }

    z_idx <- seq_len(K)
    t_gene_idx <- K + seq_len(G)
    if (R_use > 0) {
      t_lr_idx <- K + G + seq_len(R_use)
    } else {
      t_lr_idx <- integer(0)
    }

    # Linear objective: weighted absolute residuals.
    q <- numeric(nvar)
    q[t_gene_idx] <- (lambda_gene / G) * gene_weight_vec
    if (R_use > 0) {
      q[t_lr_idx] <- (lambda_lr / R_lr) * lr_weight_vec
    }

    # Small quadratic term for numerical stability in OSQP.
    Pmat <- Diagonal(nvar, x = rep(mae_qp_l2_epsilon, nvar))
    Pmat <- Matrix(Pmat, sparse = TRUE)

    zero_K_slack <- Matrix(0, nrow = K, ncol = G + R_use, sparse = TRUE)
    A_z_bounds <- cbind(Diagonal(K), zero_K_slack)

    Aeq <- Matrix(0, nrow = 1, ncol = nvar, sparse = TRUE)
    Aeq[1, seq_len(C)] <- 1

    # Slack non-negativity rows.
    n_slack <- G + R_use
    if (n_slack > 0) {
      A_slack <- cbind(
        Matrix(0, nrow = n_slack, ncol = K, sparse = TRUE),
        Diagonal(n_slack)
      )
    } else {
      A_slack <- Matrix(0, nrow = 0, ncol = nvar, sparse = TRUE)
    }

    D_gene_t <- Matrix(t(D_gene), sparse = TRUE)
    I_gene <- Diagonal(G)
    Z_gene_lr <- Matrix(0, nrow = G, ncol = R_use, sparse = TRUE)

    # t_gene >= y_gene - D_gene^T z  ->  D_gene^T z + t_gene >= y_gene
    # t_gene >= D_gene^T z - y_gene  -> -D_gene^T z + t_gene >= -y_gene
    A_gene_plus <- cbind(D_gene_t, I_gene, Z_gene_lr)
    A_gene_minus <- cbind(-D_gene_t, I_gene, Z_gene_lr)

    if (R_use > 0) {
      D_lr_t <- Matrix(t(D_lr), sparse = TRUE)
      Z_lr_gene <- Matrix(0, nrow = R_lr, ncol = G, sparse = TRUE)
      I_lr <- Diagonal(R_lr)
      A_lr_plus <- cbind(D_lr_t, Z_lr_gene, I_lr)
      A_lr_minus <- cbind(-D_lr_t, Z_lr_gene, I_lr)
      A <- rbind(Aeq, A_z_bounds, A_slack, A_gene_plus, A_gene_minus, A_lr_plus, A_lr_minus)
    } else {
      A <- rbind(Aeq, A_z_bounds, A_slack, A_gene_plus, A_gene_minus)
    }

    l <- c(
      1,
      rep(0, K),
      rep(0, n_slack),
      rep(0, G),
      rep(0, G),
      if (R_use > 0) rep(0, R_lr) else numeric(0),
      if (R_use > 0) rep(0, R_lr) else numeric(0)
    )
    u <- c(
      1,
      c(rep(Inf, C), rep(1, M)),
      rep(Inf, n_slack),
      rep(Inf, G),
      rep(Inf, G),
      if (R_use > 0) rep(Inf, R_lr) else numeric(0),
      if (R_use > 0) rep(Inf, R_lr) else numeric(0)
    )

    row_start_gene_plus <- 1 + K + n_slack + 1
    gene_plus_rows <- row_start_gene_plus + seq_len(G) - 1
    gene_minus_rows <- max(gene_plus_rows) + seq_len(G)

    if (R_use > 0) {
      lr_plus_rows <- max(gene_minus_rows) + seq_len(R_lr)
      lr_minus_rows <- max(lr_plus_rows) + seq_len(R_lr)
    } else {
      lr_plus_rows <- integer(0)
      lr_minus_rows <- integer(0)
    }

    list(
      P = Pmat,
      q = q,
      A = Matrix(A, sparse = TRUE),
      l = l,
      u = u,
      z_idx = z_idx,
      t_gene_idx = t_gene_idx,
      t_lr_idx = t_lr_idx,
      gene_plus_rows = gene_plus_rows,
      gene_minus_rows = gene_minus_rows,
      lr_plus_rows = lr_plus_rows,
      lr_minus_rows = lr_minus_rows,
      gene_weight_vec = gene_weight_vec,
      lr_weight_vec = lr_weight_vec,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      G = G,
      R_lr = R_lr,
      use_lr_loss = use_lr_loss,
      mae_qp_l2_epsilon = mae_qp_l2_epsilon
    )
  }

  make_l_joint_mae <- function(y_gene, y_lr, osqp_tpl) {
    l <- osqp_tpl$l
    l[osqp_tpl$gene_plus_rows] <- as.numeric(y_gene)
    l[osqp_tpl$gene_minus_rows] <- -as.numeric(y_gene)

    if (isTRUE(osqp_tpl$use_lr_loss)) {
      l[osqp_tpl$lr_plus_rows] <- as.numeric(y_lr)
      l[osqp_tpl$lr_minus_rows] <- -as.numeric(y_lr)
    }

    l
  }

  weighted_median <- function(x, w) {
    x <- as.numeric(x)
    w <- as.numeric(w)
    keep <- is.finite(x) & is.finite(w) & (w > 0)
    x <- x[keep]
    w <- w[keep]

    if (length(x) == 0) {
      return(0)
    }

    ord <- order(x)
    x <- x[ord]
    w <- w[ord]
    cs <- cumsum(w)
    cutoff <- 0.5 * sum(w)
    x[which(cs >= cutoff)[1]]
  }

  solve_l1_ridge_location <- function(x, w, ridge_lambda = 0, nonnegative = FALSE) {
    # Solves min_b sum_i w_i |x_i - b| + ridge_lambda * b^2.
    # With ridge_lambda=0 this is the weighted median.
    x <- as.numeric(x)
    w <- as.numeric(w)
    keep <- is.finite(x) & is.finite(w) & (w > 0)
    x <- x[keep]
    w <- w[keep]

    if (length(x) == 0) {
      return(0)
    }
    if (!is.finite(ridge_lambda) || ridge_lambda < 0) {
      stop("ridge_lambda must be finite and non-negative.")
    }

    if (ridge_lambda <= 0) {
      ans <- weighted_median(x, w)
      if (isTRUE(nonnegative)) ans <- max(ans, 0)
      return(ans)
    }

    lower <- min(c(x, 0))
    upper <- max(c(x, 0))
    if (isTRUE(nonnegative)) {
      lower <- max(lower, 0)
      upper <- max(upper, 0)
    }
    if (!is.finite(lower) || !is.finite(upper) || lower == upper) {
      ans <- lower
      if (isTRUE(nonnegative)) ans <- max(ans, 0)
      return(ans)
    }

    opt <- stats::optimize(
      f = function(b) sum(w * abs(x - b)) + ridge_lambda * (b ^ 2),
      interval = c(lower, upper)
    )
    ans <- opt$minimum
    if (isTRUE(nonnegative)) ans <- max(ans, 0)
    ans
  }

  update_beta0_by_group_mae <- function(residual_mat,
                                        sample_weight,
                                        cohort_index,
                                        B,
                                        current_beta0,
                                        fixed_zero = FALSE,
                                        nonnegative = FALSE,
                                        use_ridge = FALSE,
                                        ridge_lambda = 0,
                                        step_size = 1.0,
                                        feature_names = NULL,
                                        cohort_names = NULL) {
    residual_mat <- as.matrix(residual_mat)
    F <- ncol(residual_mat)

    if (isTRUE(fixed_zero)) {
      out <- matrix(0, nrow = B, ncol = F)
      if (!is.null(cohort_names)) rownames(out) <- cohort_names
      if (!is.null(feature_names)) colnames(out) <- feature_names
      return(out)
    }

    if (!is.finite(step_size) || step_size <= 0 || step_size > 1) {
      stop("step_size must be in the interval (0, 1].")
    }

    ridge_use <- if (isTRUE(use_ridge)) ridge_lambda else 0
    proposed <- matrix(0, nrow = B, ncol = F)

    for (b in seq_len(B)) {
      idx <- which(cohort_index == b)
      if (length(idx) == 0) {
        next
      }
      w_b <- sample_weight[idx]
      for (j in seq_len(F)) {
        proposed[b, j] <- solve_l1_ridge_location(
          x = residual_mat[idx, j],
          w = w_b,
          ridge_lambda = ridge_use,
          nonnegative = nonnegative
        )
      }
    }

    out <- (1 - step_size) * current_beta0 + step_size * proposed
    if (!is.null(cohort_names)) rownames(out) <- cohort_names
    if (!is.null(feature_names)) colnames(out) <- feature_names
    out
  }

  compute_weighted_mae_with_beta0_by_group <- function(residual_mat,
                                                       beta0_mat,
                                                       sample_weight,
                                                       cohort_index,
                                                       feature_weight_vec,
                                                       beta0_ridge_lambda = 0) {
    residual_mat <- as.matrix(residual_mat)
    beta0_mat <- as.matrix(beta0_mat)
    sample_weight <- as.numeric(sample_weight)
    feature_weight_vec <- as.numeric(feature_weight_vec)

    if (ncol(residual_mat) != ncol(beta0_mat) || ncol(residual_mat) != length(feature_weight_vec)) {
      stop("Feature dimensions do not match in compute_weighted_mae_with_beta0_by_group.")
    }
    if (nrow(residual_mat) != length(sample_weight) || nrow(residual_mat) != length(cohort_index)) {
      stop("Sample dimensions do not match in compute_weighted_mae_with_beta0_by_group.")
    }
    if (any(!is.finite(residual_mat)) || any(!is.finite(beta0_mat)) ||
        any(!is.finite(sample_weight)) || any(!is.finite(feature_weight_vec)) ||
        !is.finite(beta0_ridge_lambda)) {
      stop("Non-finite input found when computing weighted MAE with group-specific beta0.")
    }
    if (beta0_ridge_lambda < 0) {
      stop("beta0_ridge_lambda must be non-negative.")
    }

    total <- 0.0
    for (i in seq_len(nrow(residual_mat))) {
      b <- cohort_index[i]
      adj_abs <- abs(residual_mat[i, ] - beta0_mat[b, ])
      total <- total + sample_weight[i] * sum(feature_weight_vec * adj_abs)
    }

    ridge_penalty <- beta0_ridge_lambda * sum(sweep(beta0_mat ^ 2, 2, feature_weight_vec, "*"))
    as.numeric(total + ridge_penalty)
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

  loading_LR_use <- read.csv(
    loading_LR_path,
    row.names = 1,
    check.names = FALSE
  )

  bulk_expr_df <- read.csv(
    bulk_expr_path,
    row.names = 1,
    check.names = FALSE
  )

  # Observed sub-slice MI means provide cohort metadata and the recovery reference.
  MI_mean_path <- file.path(input_dir, "MI_mean_df.csv")
  if (!file.exists(MI_mean_path)) {
    stop("Missing spatial MI reference: ", MI_mean_path,
         "\nExport MI_mean_df.csv from Pancancer_analysis_V2.ipynb first.")
  }
  MI_mean_df_meta <- read.csv(MI_mean_path, row.names = 1, check.names = FALSE)
  if (!all(rownames(bulk_expr_df) %in% rownames(MI_mean_df_meta))) {
    stop("MI_mean_df.csv is missing sub-slices present in geneexp_mean_df.csv.")
  }
  MI_mean_df_meta$Split <- "Train"
  # if (isTRUE(if_traindata)) {
  #   if (is.null(MI_mean_df_meta)) {
  #     stop("if_traindata=TRUE but MI_mean_df.csv was not found.")
  #   }
  #   if (!("Split" %in% colnames(MI_mean_df_meta))) {
  #     stop("if_traindata=TRUE but MI_mean_df.csv does not contain a Split column.")
  #   }

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
  if (nrow(loading_LR_use) != nrow(loading_MI)) {
    stop("loading_LR_use must have the same number of MI rows as loading_sender_use/loading_receiver_use.")
  }

  # Align loading_LR rows to MI rows when row names are available.
  if (!is.null(rownames(loading_LR_use)) && !is.null(rownames(loading_MI)) &&
      all(rownames(loading_MI) %in% rownames(loading_LR_use))) {
    loading_LR_use <- loading_LR_use[rownames(loading_MI), , drop = FALSE]
  } else if (!identical(rownames(loading_LR_use), rownames(loading_MI))) {
    warning("Could not confidently align loading_LR_use rows by row name; using existing row order.")
  }

  # -----------------------------
  # Build LR-pair coexpression matrix from bulk expression
  # -----------------------------
  lr_build <- build_lr_coexpression_matrix(
    expr_df = bulk_expr_df,
    lr_pair_names = colnames(loading_LR_use),
    require_all_genes = lr_require_all_genes
  )

  LR_coexpr_df <- as.data.frame(lr_build$matrix, check.names = FALSE)
  LR_pair_metadata <- lr_build$metadata
  LR_pair_skipped <- lr_build$skipped

  if (ncol(LR_coexpr_df) < 2) {
    stop("Too few LR pairs could be computed from bulk expression.")
  }

  # Keep loading_LR columns that are computable from bulk expression.
  loading_LR_use <- loading_LR_use[, colnames(LR_coexpr_df), drop = FALSE]

  cat("Computed LR-pair coexpression matrix: ", nrow(LR_coexpr_df), " samples x ", ncol(LR_coexpr_df), " LR pairs\n", sep = "")
  write.csv(LR_coexpr_df, file.path(output_dir, "LR_coexpression_bulk.csv"))
  write.csv(LR_pair_metadata, file.path(output_dir, "LR_pair_parse_metadata.csv"), row.names = FALSE)
  if (nrow(LR_pair_skipped) > 0) {
    write.csv(LR_pair_skipped, file.path(output_dir, "LR_pair_skipped_missing_genes.csv"), row.names = FALSE)
    cat("Skipped ", nrow(LR_pair_skipped), " LR pairs due to missing ligand/receptor genes.\n", sep = "")
  }

  # -----------------------------
  # Gene intersection and alignment
  # -----------------------------
  genes_common <- intersect(colnames(bulk_expr_df), colnames(Loading_intrinsic_use))
  genes_common <- unique(genes_common)

  if (length(genes_common) < 10) {
    stop("Too few overlapping genes between geneexp_mean_df.csv and loading matrices.")
  }

  message("Common genes used for gene-expression decomposition: ", length(genes_common))

  X <- as.matrix(bulk_expr_df[, genes_common, drop = FALSE])
  mode(X) <- "numeric"

  if (any(!is.finite(X))) {
    stop("Non-finite values found in bulk expression matrix after gene subsetting.")
  }
  if (any(X < 0)) {
    stop("Negative values found in bulk expression matrix. This joint-loss script assumes non-negative expression.")
  }

  W0 <- as.matrix(Loading_intrinsic_use[, genes_common, drop = FALSE])
  W_MI <- as.matrix(loading_MI[, genes_common, drop = FALSE])
  W_LR <- as.matrix(loading_LR_use[, colnames(LR_coexpr_df), drop = FALSE])
  mode(W0) <- "numeric"
  mode(W_MI) <- "numeric"
  mode(W_LR) <- "numeric"

  if (any(!is.finite(W0))) {
    stop("Non-finite values found in Loading_intrinsic_use after gene subsetting.")
  }
  if (any(!is.finite(W_MI))) {
    stop("Non-finite values found in loading_MI after gene subsetting.")
  }
  if (any(!is.finite(W_LR))) {
    stop("Non-finite values found in loading_LR_use after LR-pair subsetting.")
  }
  if (any(W_LR < 0)) {
    stop("Negative values found in loading_LR_use; expected non-negative LR loadings.")
  }

  # -----------------------------
  # Metadata
  # -----------------------------
  N <- nrow(X)
  G <- ncol(X)
  C <- nrow(W0)
  M <- nrow(W_MI)
  K <- C + M
  R_lr <- ncol(W_LR)

  sample_names <- rownames(X)
  intrinsic_names <- rownames(W0)
  mi_names <- rownames(W_MI)
  lr_pair_names <- colnames(W_LR)

  C_LR <- as.matrix(LR_coexpr_df[sample_names, lr_pair_names, drop = FALSE])
  mode(C_LR) <- "numeric"

  # Design matrices:
  #   D_gene uses both intrinsic and MI gene loadings.
  #   D_lr has zero intrinsic block and LR-pair loading for MI block only.
  D_gene <- rbind(W0, W_MI)
  D_gene <- as.matrix(D_gene)
  mode(D_gene) <- "numeric"

  D_lr <- rbind(
    matrix(0, nrow = C, ncol = R_lr, dimnames = list(intrinsic_names, lr_pair_names)),
    W_LR
  )
  D_lr <- as.matrix(D_lr)
  mode(D_lr) <- "numeric"

  # -----------------------------
  # Gene-specific and LR-pair-specific weights
  # -----------------------------
  gene_weight <- NULL
  if (isTRUE(use_gene_weight) && !isTRUE(if_nongeneweight)) {
    gene_weight <- compute_kl_specificity_weights(loading_MI, feature_label = "gene")
  } else {
    message("Skipping KL-based gene-weight calculation; gene-expression loss will use uniform gene weights.")
  }

  if (isTRUE(if_nongeneweight)) {
    gene_weight_common <- rep(1, G)
    names(gene_weight_common) <- genes_common
    message("if_nongeneweight=TRUE: forcing gene weights to 1 across all common genes.")
  } else if (isTRUE(use_gene_weight)) {
    if (is.null(gene_weight)) {
      stop("gene_weight is NULL even though use_gene_weight=TRUE and if_nongeneweight=FALSE.")
    }
    if (!all(genes_common %in% names(gene_weight))) {
      stop("Some common genes are missing in gene_weight.")
    }
    gene_weight_common <- as.numeric(gene_weight[genes_common])
    names(gene_weight_common) <- genes_common
    message("Using KL-based gene-weighted gene-expression loss.")
  } else {
    gene_weight_common <- rep(1, G)
    names(gene_weight_common) <- genes_common
    message("Using uniform gene-expression loss.")
  }

  lr_weight <- NULL
  if (isTRUE(use_lr_weight) && !isTRUE(if_nonlrweight)) {
    lr_weight <- compute_kl_specificity_weights(W_LR, feature_label = "LR-pair")
  } else {
    message("Skipping KL-based LR-pair-weight calculation; LR loss will use uniform LR-pair weights.")
  }

  if (isTRUE(if_nonlrweight)) {
    lr_weight_common <- rep(1, R_lr)
    names(lr_weight_common) <- lr_pair_names
    message("if_nonlrweight=TRUE: forcing LR-pair weights to 1 across all LR pairs.")
  } else if (isTRUE(use_lr_weight)) {
    if (is.null(lr_weight)) {
      stop("lr_weight is NULL even though use_lr_weight=TRUE and if_nonlrweight=FALSE.")
    }
    if (!all(lr_pair_names %in% names(lr_weight))) {
      stop("Some LR pairs are missing in lr_weight.")
    }
    lr_weight_common <- as.numeric(lr_weight[lr_pair_names])
    names(lr_weight_common) <- lr_pair_names
    message("Using KL-based LR-pair-weighted LR coexpression loss.")
  } else {
    lr_weight_common <- rep(1, R_lr)
    names(lr_weight_common) <- lr_pair_names
    message("Using uniform LR coexpression loss.")
  }

  # Paper-consistent Omega/Psi weighting.
  if (isTRUE(use_paper_omega_weighting)) {
    gene_loss_weight_common <- gene_weight_common ^ 2
    lr_loss_weight_common <- lr_weight_common ^ 2
    message("Using paper-consistent Omega/Psi weighting: squared-error weights are specificity_weight^2.")
  } else {
    gene_loss_weight_common <- gene_weight_common
    lr_loss_weight_common <- lr_weight_common
    message("Using legacy weighting: squared-error weights are specificity_weight.")
  }
  names(gene_loss_weight_common) <- genes_common
  names(lr_loss_weight_common) <- lr_pair_names

  if (any(!is.finite(gene_loss_weight_common)) || any(gene_loss_weight_common < 0) || all(gene_loss_weight_common == 0)) {
    stop("gene_loss_weight_common is invalid.")
  }
  if (any(!is.finite(lr_loss_weight_common)) || any(lr_loss_weight_common < 0) || all(lr_loss_weight_common == 0)) {
    stop("lr_loss_weight_common is invalid.")
  }

  # Save weights for inspection.
  write.csv(
    data.frame(gene = genes_common, gene_weight = gene_weight_common, gene_loss_weight = gene_loss_weight_common),
    file.path(output_dir, "gene_loss_weights.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(lr_pair = lr_pair_names, lr_weight = lr_weight_common, lr_loss_weight = lr_loss_weight_common),
    file.path(output_dir, "LR_loss_weights.csv"),
    row.names = FALSE
  )

  # -----------------------------
  # Cohort-balanced sample weights
  # -----------------------------
  cohort_vec <- get_cohort_labels(
    sample_names = sample_names,
    metadata_df = MI_mean_df_meta,
    cohort_col = cohort_col,
    infer_from_sample_name = infer_cohort_from_sample_name
  )

  if (length(cohort_vec) != N) {
    stop("Length of cohort_vec does not match number of samples in X.")
  }
  if (any(is.na(cohort_vec)) || any(cohort_vec == "")) {
    stop("cohort_vec contains missing or empty cohort labels.")
  }

  cohort_vec <- as.character(cohort_vec)
  cohort_counts <- table(cohort_vec)

  if (isTRUE(use_cohort_balanced_loss)) {
    sample_objective_weight <- 1 / as.numeric(cohort_counts[cohort_vec])
    message("Using cohort-balanced objective. Cohorts and sample counts:")
    print(cohort_counts)
  } else {
    sample_objective_weight <- rep(1, N)
    message("Using pooled-sample objective; each sample has equal weight.")
  }

  if (any(!is.finite(sample_objective_weight)) || any(sample_objective_weight <= 0)) {
    stop("sample_objective_weight contains non-finite or non-positive values.")
  }

  objective_weight_sum <- sum(sample_objective_weight)

  cohort_weight_df <- data.frame(
    sample = sample_names,
    cohort = cohort_vec,
    sample_objective_weight = sample_objective_weight,
    row.names = NULL,
    check.names = FALSE
  )
  cohort_weight_path <- file.path(output_dir, "cohort_sample_weights.csv")
  write.csv(cohort_weight_df, cohort_weight_path, row.names = FALSE)
  cat("Saved cohort/sample weights to: ", cohort_weight_path, "\n", sep = "")

  # -----------------------------
  # Global scaling for gene and LR blocks separately
  # -----------------------------
  scale_factor_gene <- 1.0
  scale_factor_lr <- 1.0

  if (isTRUE(use_scaling)) {
    set.seed(1)

    xvec <- as.numeric(X)
    xvec <- xvec[is.finite(xvec)]
    if (length(xvec) == 0) {
      stop("Cannot compute scale_factor_gene: no finite values in X.")
    }
    sample_cap <- 1e6
    pool_gene <- abs(sample(xvec, size = min(length(xvec), sample_cap)))
    pool_gene <- pool_gene[is.finite(pool_gene)]
    scale_factor_gene <- as.numeric(stats::quantile(pool_gene, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor_gene) || scale_factor_gene <= 0) {
      scale_factor_gene <- 1.0
    }

    lrvec <- as.numeric(C_LR)
    lrvec <- lrvec[is.finite(lrvec)]
    if (length(lrvec) == 0) {
      stop("Cannot compute scale_factor_lr: no finite values in C_LR.")
    }
    pool_lr <- abs(sample(lrvec, size = min(length(lrvec), sample_cap)))
    pool_lr <- pool_lr[is.finite(pool_lr)]
    scale_factor_lr <- as.numeric(stats::quantile(pool_lr, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor_lr) || scale_factor_lr <= 0) {
      scale_factor_lr <- 1.0
    }

    message("Global scale_factor_gene (95% |X|): ", signif(scale_factor_gene, 4))
    message("Global scale_factor_lr (95% |C_LR|): ", signif(scale_factor_lr, 4))

    X_scaled <- X / scale_factor_gene
    D_gene_scaled <- D_gene / scale_factor_gene

    C_LR_scaled <- C_LR / scale_factor_lr
    D_lr_scaled <- D_lr / scale_factor_lr
  } else {
    X_scaled <- X
    D_gene_scaled <- D_gene
    C_LR_scaled <- C_LR
    D_lr_scaled <- D_lr
  }

  # -----------------------------
  # Solve with cohort-specific beta0_gene and beta0_LR
  # -----------------------------
  # Each cohort has its own intercept vector:
  #   beta0_gene[cohort, gene]
  #   beta0_LR[cohort, LR_pair]
  # This is closer to a batch-effect correction term than a single shared beta0.
  # Important caveat: if cohort labels correspond to biological groups such as
  # cancer types rather than technical batches, cohort-specific beta0 may remove
  # real biological between-cohort variation. Use cohort_col to provide a technical
  # batch column when available.
  cohort_levels_beta0 <- unique(cohort_vec)
  cohort_index <- match(cohort_vec, cohort_levels_beta0)
  B <- length(cohort_levels_beta0)

  if (any(is.na(cohort_index))) {
    stop("Failed to map samples to cohort-specific beta0 indices.")
  }

  objective_weight_by_cohort <- as.numeric(rowsum(sample_objective_weight, group = cohort_index, reorder = FALSE))
  names(objective_weight_by_cohort) <- cohort_levels_beta0

  if (length(objective_weight_by_cohort) != B || any(!is.finite(objective_weight_by_cohort)) || any(objective_weight_by_cohort <= 0)) {
    stop("Invalid objective_weight_by_cohort values.")
  }

  beta0_gene <- matrix(
    0,
    nrow = B,
    ncol = G,
    dimnames = list(cohort_levels_beta0, genes_common)
  )
  beta0_LR <- matrix(
    0,
    nrow = B,
    ncol = R_lr,
    dimnames = list(cohort_levels_beta0, lr_pair_names)
  )

  message("Using cohort-specific beta0 with ", B, " cohort(s): ", paste(cohort_levels_beta0, collapse = ", "))
  if (isTRUE(if_set_beta0_zero)) {
    message("if_set_beta0_zero=TRUE: cohort-specific beta0_gene is fixed to zero during optimization.")
  }
  if (isTRUE(if_set_beta0_LR_zero)) {
    message("if_set_beta0_LR_zero=TRUE: cohort-specific beta0_LR is fixed to zero during optimization.")
  }
  if (isTRUE(if_beta0_nonnegative)) {
    message("if_beta0_nonnegative=TRUE: estimated cohort-specific beta0_gene and beta0_LR are constrained to be non-negative.")
  }
  if (isTRUE(use_beta0_ridge)) {
    message(
      "use_beta0_ridge=TRUE: cohort-specific beta0 updates use ridge shrinkage. ",
      "lambda_gene=", beta0_ridge_lambda_gene,
      "; lambda_LR=", beta0_ridge_lambda_lr,
      "; step_size=", beta0_step_size
    )
  } else {
    message("use_beta0_ridge=FALSE: cohort-specific beta0 updates are unpenalized except for optional non-negativity.")
  }

  P_hat <- matrix(0, nrow = N, ncol = C)
  MI_hat <- matrix(0, nrow = N, ncol = M)

  if (loss_type == "MSE") {
    osqp_tpl <- make_osqp_template_joint(
      D_gene = D_gene_scaled,
      D_lr = D_lr_scaled,
      C = C,
      M = M,
      gene_weight_vec = gene_loss_weight_common,
      lr_weight_vec = lr_loss_weight_common,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      use_lr_loss = use_lr_loss
    )
  } else if (loss_type == "MAE") {
    osqp_tpl <- make_osqp_template_joint_mae(
      D_gene = D_gene_scaled,
      D_lr = D_lr_scaled,
      C = C,
      M = M,
      gene_weight_vec = gene_loss_weight_common,
      lr_weight_vec = lr_loss_weight_common,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      use_lr_loss = use_lr_loss,
      mae_qp_l2_epsilon = mae_qp_l2_epsilon
    )
  } else {
    stop("Unsupported loss_type.")
  }

  message("Using loss_type=", loss_type, ".")
  if (loss_type == "MAE") {
    message("MAE mode solves each sample-level update with auxiliary absolute-residual variables; this can be slower than MSE.")
  }

  prev_objective <- NA_real_

  for (it in seq_len(max_outer)) {
    sum_e_gene_by_cohort <- matrix(0, nrow = B, ncol = G, dimnames = list(cohort_levels_beta0, genes_common))
    sum_e_lr_by_cohort <- matrix(0, nrow = B, ncol = R_lr, dimnames = list(cohort_levels_beta0, lr_pair_names))
    sumsq_gene <- 0.0
    sumsq_lr <- 0.0

    if (loss_type == "MAE") {
      residual_gene_no_intercept_mat <- matrix(0, nrow = N, ncol = G, dimnames = list(sample_names, genes_common))
      residual_lr_no_intercept_mat <- matrix(0, nrow = N, ncol = R_lr, dimnames = list(sample_names, lr_pair_names))
    }

    b1 <- cohort_index[1]
    y1_gene <- as.numeric(X_scaled[1, ] - beta0_gene[b1, ])
    y1_lr <- as.numeric(C_LR_scaled[1, ] - beta0_LR[b1, ])

    if (loss_type == "MSE") {
      q1 <- make_q_joint(
        D_gene = D_gene_scaled,
        D_lr = D_lr_scaled,
        y_gene = y1_gene,
        y_lr = y1_lr,
        osqp_tpl = osqp_tpl
      )

      model <- init_model(
        P = osqp_tpl$P,
        A = osqp_tpl$A,
        l = osqp_tpl$l,
        u = osqp_tpl$u,
        q_init = q1
      )
    } else {
      l1 <- make_l_joint_mae(
        y_gene = y1_gene,
        y_lr = y1_lr,
        osqp_tpl = osqp_tpl
      )

      model <- init_model(
        P = osqp_tpl$P,
        A = osqp_tpl$A,
        l = l1,
        u = osqp_tpl$u,
        q_init = osqp_tpl$q
      )
    }

    # Bind the current solver's methods once per outer iteration. OSQP 1.x
    # exposes S7 properties; its legacy `$` accessor emits a warning on each call.
    # Earlier R6 versions expose the same bound methods through `$`.
    # A fresh solver is still initialized above at every outer iteration, preserving
    # the original warm-start sequence and numerical solver settings.
    if (inherits(model, "S7_object")) {
      update_model <- model@Update
      solve_model <- model@Solve
    } else {
      update_model <- model$Update
      solve_model <- model$Solve
    }

    for (i in seq_len(N)) {
      b <- cohort_index[i]

      y_gene <- as.numeric(X_scaled[i, ] - beta0_gene[b, ])
      y_lr <- as.numeric(C_LR_scaled[i, ] - beta0_LR[b, ])

      if (loss_type == "MSE") {
        q <- make_q_joint(
          D_gene = D_gene_scaled,
          D_lr = D_lr_scaled,
          y_gene = y_gene,
          y_lr = y_lr,
          osqp_tpl = osqp_tpl
        )
        update_model(q = q)
      } else {
        l_i <- make_l_joint_mae(
          y_gene = y_gene,
          y_lr = y_lr,
          osqp_tpl = osqp_tpl
        )
        update_model(l = l_i)
      }

      r <- solve_model()

      if (r$info$status_val != 1L) {
        stop(sprintf("OSQP failed | outer=%d row=%d status=%s", it, i, r$info$status))
      }

      if (loss_type == "MSE") {
        z <- r$x
      } else {
        z <- r$x[osqp_tpl$z_idx]
      }

      if (any(!is.finite(z))) {
        stop(sprintf("Non-finite z | outer=%d row=%d", it, i))
      }

      P_hat[i, ] <- z[1:C]
      MI_hat[i, ] <- z[(C + 1):K]

      pred_gene_no_intercept <- as.numeric(z %*% D_gene_scaled)
      e_gene <- as.numeric(X_scaled[i, ] - pred_gene_no_intercept)

      pred_lr_no_intercept <- as.numeric(z %*% D_lr_scaled)
      e_lr <- as.numeric(C_LR_scaled[i, ] - pred_lr_no_intercept)

      row_w <- sample_objective_weight[i]
      sum_e_gene_by_cohort[b, ] <- sum_e_gene_by_cohort[b, ] + row_w * e_gene
      sumsq_gene <- sumsq_gene + row_w * sum((e_gene ^ 2) * osqp_tpl$gene_weight_vec)

      if (loss_type == "MAE") {
        residual_gene_no_intercept_mat[i, ] <- e_gene
      }

      if (isTRUE(use_lr_loss)) {
        sum_e_lr_by_cohort[b, ] <- sum_e_lr_by_cohort[b, ] + row_w * e_lr
        sumsq_lr <- sumsq_lr + row_w * sum((e_lr ^ 2) * osqp_tpl$lr_weight_vec)
        if (loss_type == "MAE") {
          residual_lr_no_intercept_mat[i, ] <- e_lr
        }
      }
    }

    if (loss_type == "MSE") {
      if (isTRUE(if_set_beta0_zero)) {
        beta0_gene_new <- matrix(0, nrow = B, ncol = G, dimnames = list(cohort_levels_beta0, genes_common))
      } else {
        beta0_gene_denominator <- objective_weight_by_cohort
        if (isTRUE(use_beta0_ridge)) {
          beta0_gene_denominator <- beta0_gene_denominator + beta0_ridge_lambda_gene
        }

        beta0_gene_proposed <- sweep(sum_e_gene_by_cohort, 1, beta0_gene_denominator, "/")

        if (isTRUE(if_beta0_nonnegative)) {
          beta0_gene_proposed <- pmax(beta0_gene_proposed, 0)
        }

        beta0_gene_new <-
          (1 - beta0_step_size) * beta0_gene +
          beta0_step_size * beta0_gene_proposed
      }

      sse_gene <- compute_weighted_sse_with_beta0_by_group(
        sumsq_no_intercept = sumsq_gene,
        sum_e_mat = sum_e_gene_by_cohort,
        beta0_mat = beta0_gene_new,
        feature_weight_vec = osqp_tpl$gene_weight_vec,
        objective_weight_by_group = objective_weight_by_cohort,
        beta0_ridge_lambda = if (isTRUE(use_beta0_ridge)) beta0_ridge_lambda_gene else 0
      )

      if (isTRUE(use_lr_loss)) {
        if (isTRUE(if_set_beta0_LR_zero)) {
          beta0_LR_new <- matrix(0, nrow = B, ncol = R_lr, dimnames = list(cohort_levels_beta0, lr_pair_names))
        } else {
          beta0_LR_denominator <- objective_weight_by_cohort
          if (isTRUE(use_beta0_ridge)) {
            beta0_LR_denominator <- beta0_LR_denominator + beta0_ridge_lambda_lr
          }

          beta0_LR_proposed <- sweep(sum_e_lr_by_cohort, 1, beta0_LR_denominator, "/")

          if (isTRUE(if_beta0_nonnegative)) {
            beta0_LR_proposed <- pmax(beta0_LR_proposed, 0)
          }

          beta0_LR_new <-
            (1 - beta0_step_size) * beta0_LR +
            beta0_step_size * beta0_LR_proposed
        }

        sse_lr <- compute_weighted_sse_with_beta0_by_group(
          sumsq_no_intercept = sumsq_lr,
          sum_e_mat = sum_e_lr_by_cohort,
          beta0_mat = beta0_LR_new,
          feature_weight_vec = osqp_tpl$lr_weight_vec,
          objective_weight_by_group = objective_weight_by_cohort,
          beta0_ridge_lambda = if (isTRUE(use_beta0_ridge)) beta0_ridge_lambda_lr else 0
        )
      } else {
        beta0_LR_new <- beta0_LR
        sse_lr <- 0.0
      }

      if (!is.finite(sse_gene) || !is.finite(sse_lr)) {
        stop("SSE became non-finite; try stronger scaling or check inputs.")
      }
      if (sse_gene < 0) sse_gene <- 0
      if (sse_lr < 0) sse_lr <- 0

      gene_component <- ((lambda_gene / G) * sse_gene) / objective_weight_sum
      lr_component <- if (isTRUE(use_lr_loss)) ((lambda_lr / R_lr) * sse_lr) / objective_weight_sum else 0
      objective_value <- gene_component + lr_component
    } else {
      beta0_gene_new <- update_beta0_by_group_mae(
        residual_mat = residual_gene_no_intercept_mat,
        sample_weight = sample_objective_weight,
        cohort_index = cohort_index,
        B = B,
        current_beta0 = beta0_gene,
        fixed_zero = if_set_beta0_zero,
        nonnegative = if_beta0_nonnegative,
        use_ridge = use_beta0_ridge,
        ridge_lambda = beta0_ridge_lambda_gene,
        step_size = beta0_step_size,
        feature_names = genes_common,
        cohort_names = cohort_levels_beta0
      )

      mae_gene <- compute_weighted_mae_with_beta0_by_group(
        residual_mat = residual_gene_no_intercept_mat,
        beta0_mat = beta0_gene_new,
        sample_weight = sample_objective_weight,
        cohort_index = cohort_index,
        feature_weight_vec = osqp_tpl$gene_weight_vec,
        beta0_ridge_lambda = if (isTRUE(use_beta0_ridge)) beta0_ridge_lambda_gene else 0
      )

      if (isTRUE(use_lr_loss)) {
        beta0_LR_new <- update_beta0_by_group_mae(
          residual_mat = residual_lr_no_intercept_mat,
          sample_weight = sample_objective_weight,
          cohort_index = cohort_index,
          B = B,
          current_beta0 = beta0_LR,
          fixed_zero = if_set_beta0_LR_zero,
          nonnegative = if_beta0_nonnegative,
          use_ridge = use_beta0_ridge,
          ridge_lambda = beta0_ridge_lambda_lr,
          step_size = beta0_step_size,
          feature_names = lr_pair_names,
          cohort_names = cohort_levels_beta0
        )

        mae_lr <- compute_weighted_mae_with_beta0_by_group(
          residual_mat = residual_lr_no_intercept_mat,
          beta0_mat = beta0_LR_new,
          sample_weight = sample_objective_weight,
          cohort_index = cohort_index,
          feature_weight_vec = osqp_tpl$lr_weight_vec,
          beta0_ridge_lambda = if (isTRUE(use_beta0_ridge)) beta0_ridge_lambda_lr else 0
        )
      } else {
        beta0_LR_new <- beta0_LR
        mae_lr <- 0.0
      }

      if (!is.finite(mae_gene) || !is.finite(mae_lr)) {
        stop("MAE objective became non-finite; try stronger scaling or check inputs.")
      }

      gene_component <- ((lambda_gene / G) * mae_gene) / objective_weight_sum
      lr_component <- if (isTRUE(use_lr_loss)) ((lambda_lr / R_lr) * mae_lr) / objective_weight_sum else 0
      objective_value <- gene_component + lr_component
    }

    if (!is.finite(objective_value)) {
      stop("Objective value became non-finite; try stronger scaling or check inputs.")
    }

    if (is.na(prev_objective)) {
      cat(sprintf(
        "Outer %d | %s objective=%.6g | gene=%.6g | LR=%.6g (init)\n",
        it,
        loss_type,
        objective_value,
        gene_component,
        lr_component
      ))
      prev_objective <- objective_value
      beta0_gene <- beta0_gene_new
      beta0_LR <- beta0_LR_new

      if (isTRUE(if_set_beta0_zero) && (!isTRUE(use_lr_loss) || isTRUE(if_set_beta0_LR_zero))) {
        cat("Fixed beta0 mode: no outer beta0 update is needed. Stopping after one outer iteration.\n")
        break
      }
    } else {
      rel_change <- abs(prev_objective - objective_value) / max(1, abs(prev_objective))
      cat(sprintf(
        "Outer %d | %s objective=%.6g | gene=%.6g | LR=%.6g | rel_change=%.3g\n",
        it,
        loss_type,
        objective_value,
        gene_component,
        lr_component,
        rel_change
      ))
      beta0_gene <- beta0_gene_new
      beta0_LR <- beta0_LR_new

      if (rel_change < tol) {
        break
      }
      prev_objective <- objective_value
    }
  }

  # -----------------------------
  # Unscale beta0 matrices back to original data scale
  # -----------------------------
  if (use_scaling && is.finite(scale_factor_gene) && scale_factor_gene != 1.0) {
    beta0_gene <- beta0_gene * scale_factor_gene
  }
  if (use_scaling && is.finite(scale_factor_lr) && scale_factor_lr != 1.0) {
    beta0_LR <- beta0_LR * scale_factor_lr
  }

  rownames(beta0_gene) <- cohort_levels_beta0
  colnames(beta0_gene) <- genes_common
  rownames(beta0_LR) <- cohort_levels_beta0
  colnames(beta0_LR) <- lr_pair_names

  cat("✅ Joint decomposition finished.\n")

  # Print and save cohort-specific beta0_gene.
  beta0_gene_df <- data.frame(
    cohort = rep(cohort_levels_beta0, each = G),
    gene = rep(genes_common, times = B),
    beta0_gene = as.numeric(t(beta0_gene)),
    row.names = NULL,
    check.names = FALSE
  )
  cat("Final cohort-specific beta0_gene summary:\n")
  print(summary(beta0_gene_df$beta0_gene))
  cat("Final cohort-specific beta0_gene matrix dimension: ", nrow(beta0_gene), " cohorts x ", ncol(beta0_gene), " genes\n", sep = "")

  beta0_gene_output_path <- file.path(output_dir, "beta0_gene_by_cohort_long.csv")
  beta0_gene_matrix_output_path <- file.path(output_dir, "beta0_gene_by_cohort_matrix.csv")
  write.csv(beta0_gene_df, beta0_gene_output_path, row.names = FALSE)
  write.csv(beta0_gene, beta0_gene_matrix_output_path)
  cat("Saved cohort-specific beta0_gene long table to: ", beta0_gene_output_path, "\n", sep = "")
  cat("Saved cohort-specific beta0_gene matrix to: ", beta0_gene_matrix_output_path, "\n", sep = "")

  # Print and save cohort-specific beta0_LR.
  beta0_LR_df <- data.frame(
    cohort = rep(cohort_levels_beta0, each = R_lr),
    lr_pair = rep(lr_pair_names, times = B),
    beta0_LR = as.numeric(t(beta0_LR)),
    row.names = NULL,
    check.names = FALSE
  )
  cat("Final cohort-specific beta0_LR summary:\n")
  print(summary(beta0_LR_df$beta0_LR))
  cat("Final cohort-specific beta0_LR matrix dimension: ", nrow(beta0_LR), " cohorts x ", ncol(beta0_LR), " LR pairs\n", sep = "")

  beta0_LR_output_path <- file.path(output_dir, "beta0_LR_by_cohort_long.csv")
  beta0_LR_matrix_output_path <- file.path(output_dir, "beta0_LR_by_cohort_matrix.csv")
  write.csv(beta0_LR_df, beta0_LR_output_path, row.names = FALSE)
  write.csv(beta0_LR, beta0_LR_matrix_output_path)
  cat("Saved cohort-specific beta0_LR long table to: ", beta0_LR_output_path, "\n", sep = "")
  cat("Saved cohort-specific beta0_LR matrix to: ", beta0_LR_matrix_output_path, "\n", sep = "")

  settings_df <- data.frame(
    setting = c(
      "loss_type", "mae_qp_l2_epsilon",
      "beta0_mode", "n_beta0_cohorts", "beta0_cohort_labels",
      "if_set_beta0_zero", "if_set_beta0_LR_zero", "if_beta0_nonnegative",
      "use_beta0_ridge", "beta0_ridge_lambda_gene", "beta0_ridge_lambda_lr", "beta0_step_size",
      "if_nongeneweight", "if_nonlrweight",
      "use_gene_weight", "use_lr_weight", "use_lr_loss",
      "lambda_gene", "lambda_lr",
      "use_paper_omega_weighting", "use_cohort_balanced_loss",
      "use_scaling", "scale_factor_gene", "scale_factor_lr",
      "if_traindata", "lr_require_all_genes",
      "n_common_genes", "n_lr_pairs_used"
    ),
    value = c(
      as.character(loss_type), as.character(mae_qp_l2_epsilon),
      "cohort_specific", as.character(B), paste(cohort_levels_beta0, collapse = ";"),
      as.character(if_set_beta0_zero), as.character(if_set_beta0_LR_zero), as.character(if_beta0_nonnegative),
      as.character(use_beta0_ridge), as.character(beta0_ridge_lambda_gene), as.character(beta0_ridge_lambda_lr), as.character(beta0_step_size),
      as.character(if_nongeneweight), as.character(if_nonlrweight),
      as.character(use_gene_weight), as.character(use_lr_weight), as.character(use_lr_loss),
      as.character(lambda_gene), as.character(lambda_lr),
      as.character(use_paper_omega_weighting), as.character(use_cohort_balanced_loss),
      as.character(use_scaling), as.character(scale_factor_gene), as.character(scale_factor_lr),
      as.character(if_traindata), as.character(lr_require_all_genes),
      as.character(G), as.character(R_lr)
    ),
    row.names = NULL,
    check.names = FALSE
  )
  settings_output_path <- file.path(output_dir, "decomposition_settings.csv")
  write.csv(settings_df, settings_output_path, row.names = FALSE)
  cat("Saved decomposition settings to: ", settings_output_path, "\n", sep = "")

  # -----------------------------
  # Post-process and save outputs
  # -----------------------------
  MI_hat[MI_hat < 0] <- 0
  MI_hat[MI_hat > 1] <- 1
  rownames(MI_hat) <- sample_names
  colnames(MI_hat) <- mi_names

  P_hat[P_hat < 0] <- 0
  rownames(P_hat) <- sample_names
  colnames(P_hat) <- intrinsic_names

  MI_hat_output_path <- file.path(output_dir, "MI_hat_decomposed_joint_gene_LR.csv")
  write.csv(MI_hat, MI_hat_output_path)
  cat("Saved decomposed MI activities to: ", MI_hat_output_path, "\n", sep = "")

  P_hat_output_path <- file.path(output_dir, "H_hat_intrinsic_decomposed_joint_gene_LR.csv")
  write.csv(P_hat, P_hat_output_path)
  cat("Saved decomposed intrinsic proportions to: ", P_hat_output_path, "\n", sep = "")

  # -----------------------------
  # Benchmark comparison against MI_mean_df.csv
  # -----------------------------
  MI_mean_df <- read.csv(file.path(input_dir, "MI_mean_df.csv"), row.names = 1, check.names = FALSE)
  MI_mean_df <- MI_mean_df[rownames(MI_hat), , drop = FALSE]

  MI_mean_df_entrie <- align_observed_mi(MI_hat, MI_mean_df)

  # Row-wise Spearman correlation.
  corr_vec_row <- rep(NA_real_, nrow(MI_hat))
  for (row_index in seq_len(nrow(MI_hat))) {
    mi_vec <- as.numeric(MI_hat[row_index, ])
    mi_mean_vec <- as.numeric(MI_mean_df_entrie[row_index, ])
    if (length(unique(mi_vec)) > 1 && length(unique(mi_mean_vec)) > 1) {
      corr_vec_row[row_index] <- suppressWarnings(cor(mi_vec, mi_mean_vec, method = "spearman"))
    }
  }

  # Column-wise Spearman correlation.
  corr_vec_col <- rep(NA_real_, ncol(MI_hat))
  for (col_index in seq_len(ncol(MI_hat))) {
    mi_vec <- as.numeric(MI_hat[, col_index])
    mi_mean_vec <- as.numeric(MI_mean_df_entrie[, col_index])
    if (length(unique(mi_vec)) > 1 && length(unique(mi_mean_vec)) > 1) {
      corr_vec_col[col_index] <- suppressWarnings(cor(mi_vec, mi_mean_vec, method = "spearman"))
    }
  }
  names(corr_vec_col) <- colnames(MI_hat)

  MI_hat_flat <- as.vector(MI_hat)
  MI_mean_flat <- as.vector(as.matrix(MI_mean_df_entrie))
  corr_flat <- suppressWarnings(cor(MI_hat_flat, MI_mean_flat, method = "spearman"))
  cat("Overall Spearman correlation (flattened): ", signif(corr_flat, 4), "\n")
  cat("Mean column-wise Spearman correlation: ", signif(mean(corr_vec_col, na.rm = TRUE), 4), "\n")
  cat("Median row-wise Spearman correlation: ", signif(stats::median(corr_vec_row, na.rm = TRUE), 4), "\n")

  benchmark_summary <- data.frame(
    metric = c(
      "flat_spearman",
      "mean_column_spearman",
      "median_column_spearman",
      "mean_row_spearman",
      "median_row_spearman"
    ),
    value = c(
      corr_flat,
      mean(corr_vec_col, na.rm = TRUE),
      stats::median(corr_vec_col, na.rm = TRUE),
      mean(corr_vec_row, na.rm = TRUE),
      stats::median(corr_vec_row, na.rm = TRUE)
    ),
    row.names = NULL,
    check.names = FALSE
  )
  write.csv(benchmark_summary, file.path(output_dir, "benchmark_summary_joint_gene_LR.csv"), row.names = FALSE)
  write.csv(
    data.frame(MI = names(corr_vec_col), column_spearman = corr_vec_col, row.names = NULL),
    file.path(output_dir, "benchmark_columnwise_spearman_joint_gene_LR.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(sample = rownames(MI_hat), row_spearman = corr_vec_row, row.names = NULL),
    file.path(output_dir, "benchmark_rowwise_spearman_joint_gene_LR.csv"),
    row.names = FALSE
  )

  # Scatter plot between decomposed and benchmark MI activities.
  corr_flat_pearson <- suppressWarnings(cor(MI_hat_flat, MI_mean_flat, method = "pearson"))
  cat("Overall Pearson correlation (flattened): ", signif(corr_flat_pearson, 4), "\n")

  corr_flat_spearman <- suppressWarnings(cor(MI_hat_flat, MI_mean_flat, method = "spearman"))
  cat("Overall Spearman correlation (flattened): ", signif(corr_flat_spearman, 4), "\n")

  corr_flat_pearson_test <- suppressWarnings(cor.test(MI_hat_flat, MI_mean_flat, method = "pearson"))


  # Compute statistics
  lm_fit <- lm(MI_hat_flat ~ MI_mean_flat)
  lm_sum <- summary(lm_fit)

  lm_r2 <- lm_sum$r.squared
  lm_p <- lm_sum$coefficients[2, 4]   # p-value for slope

  spearman_test <- suppressWarnings(
    cor.test(MI_hat_flat, MI_mean_flat, method = "spearman")
  )
  spearman_rho <- as.numeric(spearman_test$estimate)
  spearman_p <- spearman_test$p.value

  # Pooled all-MI diagnostic scatter; the manuscript below selects one MI.
  pdf(file.path(output_dir, "pooled_MI_recovery_scatter.pdf"))
  plot(
    MI_mean_flat,
    MI_hat_flat,
    ylab = "MI_hat (joint decomposed)",
    xlab = "MI_mean (original)",
    main = "Scatter plot of MI_hat vs MI_mean"
  )

  abline(lm_fit, col = "red")

  # Add stats to top-right corner
  legend(
    "topright",
    legend = c(
      paste0("Linear regression: R^2 = ", signif(lm_r2, 3)),
      paste0("Linear regression: P = ", signif(lm_p, 3)),
      paste0("Spearman rho = ", signif(spearman_rho, 3)),
      paste0("Spearman P = ", signif(spearman_p, 3))
    ),
    bty = "n",
    cex = 0.9
  )
  dev.off()


  # ============================================================
  # projected MI-4 abundance by observed MI-4 tertile
  # ============================================================
  plot_pseudobulk_recovery(MI_hat, MI_mean_df_entrie, output_dir, mi_to_plot)
}

if (isTRUE(run_s26b_validation)) {
  s26b_results <- run_s26b_pseudobulk_validation(
    input_dir = input_dir, output_dir = s26b_output_dir, mi_to_plot = s26b_mi_to_plot
  )
}
