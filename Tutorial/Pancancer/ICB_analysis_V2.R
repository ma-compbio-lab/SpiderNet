# Project fixed spatial MI loadings onto three immune-checkpoint blockade cohorts
# and compare response and survival-time associations in four treatment settings.
# Preserve collector normalization, joint gene/LR fitting, Wilcoxon tests,
# Spearman associations, and Liptak summaries. Survival-time tests ignore censoring.
# Inputs: four loading CSVs and study collectors or source RData files.
# Collectors retain their input locations; other outputs are under output/icb/.
# Use --check or --plot-only; see README for input paths and cache behavior.

# Resolve the shared runtime relative to this script, independent of the working directory.
.pancancer_file <- if (sys.nframe() > 0L && !is.null(sys.frame(1)$ofile)) {
  sys.frame(1)$ofile
} else {
  sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
}
.pancancer_dir <- dirname(normalizePath(.pancancer_file, winslash = "/", mustWork = TRUE))
source(file.path(.pancancer_dir, "plot_saved_R.R"), local = TRUE)
.pancancer <- pancancer_options("icb", .pancancer_dir)
if (.pancancer$mode != "full") {
  pancancer_dispatch(.pancancer)
  quit(save = "no", status = 0L)
}
pancancer_check(.pancancer)

# ============================================================
# 0. User settings
# ============================================================

## ICB-portal root containing one subdirectory per cohort.
## Step 1 writes normalized collectors back to these input directories.
data_path <- .pancancer$icb

## Directory containing the fixed loadings exported by the pan-cancer analysis.
## Section 9 also assigns this path before reading the response table.
spidernet_result_dir <- .pancancer$results

## Projection, association, and plotting output directories.
icb_analysis_outdir <- file.path(.pancancer$output, "icb", "ICB_analysis_joint_gene_LR_all_datasets")
icb_decomp_outdir <- file.path(icb_analysis_outdir, "MI_decomposition_ICB")
dir.create(icb_analysis_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(icb_decomp_outdir, recursive = TRUE, showWarnings = FALSE)

## Whether to rebuild per-cohort *_ICB_data_collector.rds.
## If FALSE, reuse existing collectors and build missing ones. Reused collectors
## are still normalized and saved in place when Step 1 runs.
FORCE_REBUILD_DATA_COLLECTOR <- FALSE
RUN_STEP1_BUILD_COLLECTORS <- TRUE
RUN_STEP2_DECOMPOSITION <- TRUE
RUN_STEP3_ASSOCIATION <- TRUE

## Joint projection settings; each cohort-treatment group has separate offsets.
if_set_beta0_zero <- FALSE
if_set_beta0_LR_zero <- FALSE
if_beta0_nonnegative <- FALSE
## This switch applies damped offset updates; the ridge lambda values below
## are recorded in the settings table but are not used in the objective.
use_beta0_ridge <- FALSE
beta0_ridge_lambda_gene <- 10.0
beta0_ridge_lambda_lr <- 50.0
beta0_step_size <- 0.25

if_nongeneweight <- FALSE
if_nonlrweight <- FALSE
use_gene_weight <- !if_nongeneweight
use_lr_weight <- !if_nonlrweight
use_paper_omega_weighting <- TRUE

use_lr_loss <- TRUE
lambda_gene <- 1.0
lambda_lr <- 0.5
lr_require_all_genes <- FALSE

## Projection uses log2(max(TPM, 0) + 1), or edgeR logCPM with prior.count = 1
## for counts. Landscape biomarkers retain their collector expression scale.
tpm_transform <- "log2p1"
use_scaling <- FALSE
max_outer <- 1000
tol <- 1e-7

MIN_GENES_FOR_DECOMP <- 20
MIN_LR_PAIRS_FOR_DECOMP <- 2
MIN_SAMPLES_PER_DECOMP_GROUP <- 2
MIN_SAMPLES_PER_ASSOC_TEST <- 4
MIN_RESPONSE_PER_CLASS <- 2
MIN_NONMISSING_OS <- 4

## Heatmap settings.
HEATMAP_TOP_N_FEATURES <- 60
HEATMAP_FILL_NA_WITH_ZERO <- TRUE


## Four cohort-treatment settings for the the manuscript response analysis.
## Response and OS/PFS tests match these raw TreatmentICB labels exactly.
FOCUS_COHORT_TREATMENT_PAIRS <- data.frame(
  Project = c("Hugo et al", "Gide et al", "Jung et al", "Gide et al"),
  TreatmentICB = c("Anti-PD-1", "PD1", "anti-PD1/PDL1", "ipiPD1"),
  CancerType_expected = c("Melanoma", "Melanoma", "Lung", "Melanoma"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
FOCUS_COHORT_TREATMENT_PAIRS$CohortTreatment <- paste(
  FOCUS_COHORT_TREATMENT_PAIRS$Project,
  FOCUS_COHORT_TREATMENT_PAIRS$TreatmentICB,
  sep = " | "
)

## For both response heatmaps, rank features within each cohort-treatment panel
## by signed -log10(P), then order rows by the mean rank across panels.
## Rank 1 is the most negative value; the largest rank is the most positive.
ORDER_RESPONSE_HEATMAP_ROWS_BY_MEAN_RANK <- TRUE

## Exclude these features only from the response top-feature heatmap.
EXCLUDE_RESPONSE_TOP_FEATURES <- c("TMEscore")

## Display MI labels as MI-N instead of MIN in heatmaps and row-order CSVs.
FORMAT_MI_LABELS_WITH_DASH <- TRUE

## Fixed color scale for Heatmap_Response_top_features.pdf.
## Values outside [-2, 2] are clipped only for plotting; raw signed values are
## preserved in the output CSV tables.
HEATMAP_RESPONSE_TOP_FEATURES_COLOR_LIMIT <- 2

# ============================================================
# 1. Dependencies and data-format utilities
# ============================================================

load_pkg <- function(pkg, required = TRUE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    msg <- paste0("Package '", pkg, "' is required but not installed.")
    if (isTRUE(required)) stop(msg) else {
      warning(msg)
      return(FALSE)
    }
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  TRUE
}

## dplyr is attached separately in Section 9. RColorBrewer is needed whenever
## pheatmap is available and heatmaps are generated.
load_pkg("osqp", required = TRUE)
load_pkg("Matrix", required = TRUE)
load_pkg("edgeR", required = TRUE)
load_pkg("pheatmap", required = FALSE)
load_pkg("RColorBrewer", required = FALSE)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

safe_make_numeric_matrix <- function(x) {
  x <- as.matrix(x)
  suppressWarnings(mode(x) <- "numeric")
  x
}

sanitize_filename <- function(x) {
  x <- gsub("[^A-Za-z0-9_\\-]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  x
}

find_input_file <- function(base_dir, candidates) {
  paths <- file.path(base_dir, candidates)
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0) {
    stop(
      "Cannot find any of these files under ", base_dir, ":\n  ",
      paste(candidates, collapse = "\n  ")
    )
  }
  hit[1]
}

escape_regex <- function(x) {
  gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", x)
}

first_existing_col <- function(df, candidates) {
  hit <- candidates[candidates %in% colnames(df)]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

coerce_numeric <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  suppressWarnings(as.numeric(x))
}


## Read sample-by-gene expression from current or legacy collector schemas.
get_collector_expression <- function(data_collector, allow_null = FALSE) {
  expr_field_candidates <- c(
    "TPMexpression", "TPM_expression", "TPM", "tpm_expression",
    "countexpression", "countExpression", "CountExpression", "counts_expression",
    "expression", "Expression", "expr", "Expr", "X"
  )
  expr_field <- expr_field_candidates[vapply(expr_field_candidates, function(nm) {
    !is.null(data_collector[[nm]])
  }, logical(1))]

  if (length(expr_field) == 0) {
    if (isTRUE(allow_null)) return(NULL)
    stop("No expression matrix found in data_collector. Tried fields: ",
         paste(expr_field_candidates, collapse = ", "))
  }

  expr_field <- expr_field[1]
  expr <- as.matrix(data_collector[[expr_field]])
  suppressWarnings(mode(expr) <- "numeric")

  if (is.null(rownames(expr)) && !is.null(data_collector$metadata) &&
      "Sample" %in% colnames(data_collector$metadata) &&
      nrow(expr) == nrow(data_collector$metadata)) {
    rownames(expr) <- as.character(data_collector$metadata$Sample)
  }
  if (is.null(colnames(expr)) && !is.null(data_collector$genename) &&
      length(data_collector$genename) == ncol(expr)) {
    colnames(expr) <- as.character(data_collector$genename)
  }

  ## Guard against transposed legacy objects: sample IDs should be rows.
  if (!is.null(data_collector$metadata) && "Sample" %in% colnames(data_collector$metadata)) {
    sample_ids <- as.character(data_collector$metadata$Sample)
    row_hit <- sum(sample_ids %in% rownames(expr))
    col_hit <- sum(sample_ids %in% colnames(expr))
    if (col_hit > row_hit && col_hit >= 2) {
      expr <- t(expr)
    }
    common_samples <- intersect(sample_ids, rownames(expr))
    if (length(common_samples) > 0) {
      expr <- expr[common_samples, , drop = FALSE]
    }
  }

  if (!is.null(data_collector$genename) && length(data_collector$genename) == ncol(expr)) {
    colnames(expr) <- as.character(data_collector$genename)
  }

  list(matrix = expr, field = expr_field)
}

normalize_data_collector_schema <- function(data_collector, folder_name = NA_character_) {
  if (is.null(data_collector) || !is.list(data_collector)) return(data_collector)
  folder_name_use <- folder_name
  if (is.null(folder_name_use) || length(folder_name_use) == 0 || is.na(folder_name_use[1]) || folder_name_use[1] == "") {
    folder_name_use <- "Unknown"
  }
  folder_name_use <- as.character(folder_name_use[1])

  if (!is.null(data_collector$metadata)) {
    data_collector$metadata <- standardize_icb_metadata(data_collector$metadata, folder_name_use)
  }

  expr_info <- get_collector_expression(data_collector, allow_null = TRUE)
  if (is.null(expr_info)) return(data_collector)

  expr <- expr_info$matrix
  if (!is.null(data_collector$metadata) && "Sample" %in% colnames(data_collector$metadata)) {
    sample_ids <- as.character(data_collector$metadata$Sample)
    common_samples <- intersect(sample_ids, rownames(expr))
    if (length(common_samples) > 0) {
      data_collector$metadata <- data_collector$metadata[match(common_samples, sample_ids), , drop = FALSE]
      expr <- expr[common_samples, , drop = FALSE]
    }
  }

  data_collector$TPMexpression <- expr
  if (is.null(data_collector$genename) || length(data_collector$genename) != ncol(expr)) {
    data_collector$genename <- colnames(expr)
  }
  if (is.null(data_collector$expression_type) || length(data_collector$expression_type) == 0 || is.na(data_collector$expression_type)) {
    data_collector$expression_type <- ifelse(grepl("count", expr_info$field, ignore.case = TRUE), "counts", "TPM")
  }
  if (is.null(data_collector$source_folder) || length(data_collector$source_folder) == 0 ||
      is.na(data_collector$source_folder[1]) || data_collector$source_folder[1] == "") {
    data_collector$source_folder <- folder_name_use
  }

  data_collector
}

# ============================================================
# 2. Load the fixed pan-cancer spatial loading matrices
# ============================================================

loading_intrinsic_path <- find_input_file(
  spidernet_result_dir,
  c("loading_intrinsic_use.csv", "Loading_intrinsic_use.csv")
)
loading_sender_path <- find_input_file(
  spidernet_result_dir,
  c("loading_sender_use.csv", "Loading_sender_use.csv")
)
loading_receiver_path <- find_input_file(
  spidernet_result_dir,
  c("loading_receiver_use.csv", "Loading_receiver_use.csv")
)
loading_LR_path <- find_input_file(
  spidernet_result_dir,
  c("loading_LR_use.csv", "Loading_LR_use.csv", "LR_loading_use.csv", "loading_LR.csv")
)

Loading_intrinsic_use <- read.csv(loading_intrinsic_path, row.names = 1, check.names = FALSE)
loading_sender_use <- read.csv(loading_sender_path, row.names = 1, check.names = FALSE)
loading_receiver_use <- read.csv(loading_receiver_path, row.names = 1, check.names = FALSE)
loading_LR_use <- read.csv(loading_LR_path, row.names = 1, check.names = FALSE)

Loading_intrinsic_use <- safe_make_numeric_matrix(Loading_intrinsic_use)
loading_sender_use <- safe_make_numeric_matrix(loading_sender_use)
loading_receiver_use <- safe_make_numeric_matrix(loading_receiver_use)
loading_LR_use <- safe_make_numeric_matrix(loading_LR_use)

if (!identical(colnames(loading_sender_use), colnames(loading_receiver_use))) {
  stop("loading_sender_use and loading_receiver_use must have identical gene columns.")
}
if (!identical(colnames(Loading_intrinsic_use), colnames(loading_sender_use))) {
  stop("Loading_intrinsic_use and loading_sender/receiver must have identical gene columns.")
}
if (nrow(loading_LR_use) != nrow(loading_sender_use)) {
  stop("loading_LR_use must have the same number of MI rows as loading_sender_use.")
}

## Align LR loading rows to MI loading rows if rownames are available.
if (!is.null(rownames(loading_LR_use)) && all(rownames(loading_sender_use) %in% rownames(loading_LR_use))) {
  loading_LR_use <- loading_LR_use[rownames(loading_sender_use), , drop = FALSE]
} else if (!identical(rownames(loading_LR_use), rownames(loading_sender_use))) {
  warning("Could not confidently align loading_LR_use rows by row name; using existing row order.")
  rownames(loading_LR_use) <- rownames(loading_sender_use)
}

loading_MI <- loading_sender_use + loading_receiver_use
genename_list <- colnames(Loading_intrinsic_use)
mi_names <- rownames(loading_MI)
if (is.null(mi_names) || any(mi_names == "")) {
  mi_names <- paste0("MI", seq_len(nrow(loading_MI)))
  rownames(loading_MI) <- mi_names
  rownames(loading_sender_use) <- mi_names
  rownames(loading_receiver_use) <- mi_names
  rownames(loading_LR_use) <- mi_names
}

cat("Loaded latest SpiderNet pan-cancer loadings from:\n")
cat("  ", loading_intrinsic_path, "\n")
cat("  ", loading_sender_path, "\n")
cat("  ", loading_receiver_path, "\n")
cat("  ", loading_LR_path, "\n")
cat("Number of MIs:", nrow(loading_MI), "\n")
cat("Number of genes:", ncol(loading_MI), "\n")
cat("Number of LR pairs:", ncol(loading_LR_use), "\n")

# ============================================================
# 3. Locate the three study-cohort folders
# ============================================================

## Preserve the alphabetical cohort order used by the original folder traversal.
data_folder_names <- sort(unique(FOCUS_COHORT_TREATMENT_PAIRS$Project))
data_folders <- file.path(data_path, data_folder_names)
missing_cohort_dirs <- data_folders[!dir.exists(data_folders)]
if (length(missing_cohort_dirs) > 0) {
  stop("Missing required ICB cohort directories: ", paste(missing_cohort_dirs, collapse = ", "))
}

## The same three folders are used for collector preparation and projection.
dataset_index <- seq_along(data_folders)
data_folder_names_choose <- data_folder_names[dataset_index]
ICB_project_list <- data_folder_names_choose

cat("Found ", length(ICB_project_list), " ICB datasets.\n", sep = "")
print(data.frame(dataset_index = dataset_index, dataset = data_folder_names_choose))

# ============================================================
# 4. Cancer type mapping for pan-cancer intrinsic dictionary
# ============================================================

## Study-level cancer labels for the four cohort-treatment settings.
project_cancer_label_map <- c(
  "Gide et al" = "Melanoma",
  "Hugo et al" = "Melanoma",
  "Jung et al" = "Lung"
)

cancer_label_to_cancercell <- c(
  "Lung" = "Lung-cancercell",
  "Melanoma" = "Melanoma-cancercell"
)

infer_cancer_label <- function(folder_name, metadata = NULL) {
  if (!folder_name %in% names(project_cancer_label_map)) {
    stop("Cohort is not included in the four ICB settings: ", folder_name)
  }
  unname(project_cancer_label_map[[folder_name]])
}

get_celltype_list_for_project <- function(folder_name, metadata = NULL) {
  cancer_label <- infer_cancer_label(folder_name, metadata)
  cancercell_cur <- unname(cancer_label_to_cancercell[cancer_label])
  celltype_list <- rownames(Loading_intrinsic_use)
  cancercell_rows <- grep("-cancercell$", celltype_list, value = TRUE)
  low_exp_rows <- intersect("low_exp", celltype_list)

  if (!is.na(cancercell_cur) && length(cancercell_cur) == 1 && cancercell_cur %in% celltype_list) {
    drop_rows <- setdiff(cancercell_rows, cancercell_cur)
    drop_rows <- union(drop_rows, low_exp_rows)
    celltype_use <- setdiff(celltype_list, drop_rows)
    policy <- "matched_cancercell_only"
  } else {
    stop("Required tumor intrinsic program is missing for ", folder_name, ": ", cancercell_cur)
  }

  list(
    cancer_label = cancer_label,
    cancercell = cancercell_cur,
    celltype_use = celltype_use,
    policy = policy
  )
}

# ============================================================
# 5. ICB data collector construction
# ============================================================

## Select the first matching file and its first loaded object. Folder contents
## therefore determine the source used when a collector is rebuilt.
load_first_rdata_object <- function(folder_path) {
  rdata_files <- list.files(folder_path, pattern = "\\.Rdata$|\\.RData$", full.names = TRUE)
  if (length(rdata_files) == 0) {
    stop("No .Rdata/.RData file found under: ", folder_path)
  }
  rdata_files <- rdata_files[1]
  env <- new.env(parent = emptyenv())
  obj_names <- load(rdata_files, envir = env)
  if (length(obj_names) == 0) stop("No object loaded from: ", rdata_files)
  get(obj_names[1], envir = env)
}

match_samples_to_expr_cols <- function(samples, cols, allow_grep = TRUE) {
  out <- integer(length(samples))
  out[] <- NA_integer_
  for (i in seq_along(samples)) {
    s <- as.character(samples[i])
    idx <- which(cols == s)
    if (length(idx) == 0 && isTRUE(allow_grep)) {
      idx <- grep(escape_regex(s), cols)
    }
    if (length(idx) > 0) out[i] <- idx[1]
  }
  out
}

normalize_response_vector <- function(x) {
  x0 <- as.character(x)
  x_l <- tolower(trimws(x0))
  out <- rep(NA_character_, length(x_l))

  response_patterns <- c(
    "^response$", "^responder$", "^r$", "^cr$", "^pr$",
    "complete response", "partial response", "clinical benefit", "benefit", "sensitive"
  )
  no_response_patterns <- c(
    "^no response$", "^non.?response$", "^non.?responder$", "^nr$", "^pd$",
    "progressive disease", "progression", "resistant", "refractory", "no benefit"
  )

  out[Reduce(`|`, lapply(response_patterns, grepl, x = x_l))] <- "Response"
  out[Reduce(`|`, lapply(no_response_patterns, grepl, x = x_l))] <- "No response"

  ## Numeric binary fallback: many ICB files encode response as 1/0.
  x_num <- suppressWarnings(as.numeric(x0))
  if (any(is.na(out)) && all(stats::na.omit(unique(x_num)) %in% c(0, 1))) {
    out[is.na(out) & x_num == 1] <- "Response"
    out[is.na(out) & x_num == 0] <- "No response"
  }
  out
}

standardize_icb_metadata <- function(metadata, folder_name) {
  metadata <- as.data.frame(metadata, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"Sample" %in% colnames(metadata)) {
    sample_col <- first_existing_col(metadata, c("sample", "sample_id", "SampleID", "Run", "ID", "Patient"))
    if (is.na(sample_col)) {
      stop("No Sample column found for folder: ", folder_name)
    }
    metadata$Sample <- as.character(metadata[[sample_col]])
  }
  metadata$Sample <- as.character(metadata$Sample)

  ## ResponseICB
  if (!"ResponseICB" %in% colnames(metadata)) {
    resp_col <- first_existing_col(metadata, c(
      "Resp_NoResp", "Response", "response", "Responder", "BestResponse", "best_response",
      "BOR", "RECIST", "clinical_benefit", "Clinical Benefit", "CB", "Benefit"
    ))
    if (!is.na(resp_col)) {
      metadata$ResponseICB <- normalize_response_vector(metadata[[resp_col]])
    } else {
      metadata$ResponseICB <- NA_character_
    }
  } else {
    metadata$ResponseICB <- normalize_response_vector(metadata$ResponseICB)
  }

  ## TreatmentICB
  if (!"TreatmentICB" %in% colnames(metadata)) {
    treat_col <- first_existing_col(metadata, c(
      "TreatmentICB", "Treatment", "Treatment.x", "treatment", "therapy", "Therapy", "Drug", "Regimen"
    ))
    if (!is.na(treat_col)) {
      metadata$TreatmentICB <- as.character(metadata[[treat_col]])
    } else if (folder_name == "Hugo et al") {
      metadata$TreatmentICB <- "Anti-PD-1"
    } else {
      metadata$TreatmentICB <- "Unknown"
    }
  }
  metadata$TreatmentICB <- as.character(metadata$TreatmentICB)
  metadata$TreatmentICB[is.na(metadata$TreatmentICB) | metadata$TreatmentICB == ""] <- "Unknown"

  ## OS/PFS-like endpoint.
  if (!"OS" %in% colnames(metadata)) {
    os_col <- first_existing_col(metadata, c(
      "OS", "Overall Survival (Days)", "Overall Survival", "overall_survival.x",
      "Overall.survival..months.", "overall_survival", "PFS", "pfs", "Survival", "survival"
    ))
    if (!is.na(os_col)) {
      metadata$OS <- coerce_numeric(metadata[[os_col]])
    } else {
      metadata$OS <- NA_real_
    }
  } else {
    metadata$OS <- coerce_numeric(metadata$OS)
  }

  metadata$CancerTypeICB <- infer_cancer_label(folder_name, metadata)
  metadata
}

make_expr_sample_gene <- function(expr_gene_sample, gene_names, sample_names) {
  expr_gene_sample <- safe_make_numeric_matrix(expr_gene_sample)
  rownames(expr_gene_sample) <- gene_names
  colnames(expr_gene_sample) <- sample_names
  expr_sample_gene <- t(expr_gene_sample)
  colnames(expr_sample_gene) <- gene_names
  rownames(expr_sample_gene) <- sample_names
  expr_sample_gene
}

safe_get_gene_expression <- function(expr_sample_gene, candidates) {
  candidates <- candidates[candidates %in% colnames(expr_sample_gene)]
  if (length(candidates) == 0) return(rep(NA_real_, nrow(expr_sample_gene)))
  as.numeric(expr_sample_gene[, candidates[1]])
}

build_landscape_data <- function(ICB_datalist, data_collector) {
  sample_ids <- data_collector$metadata$Sample
  expr_info <- get_collector_expression(data_collector, allow_null = FALSE)
  expr <- expr_info$matrix

  if ("Landscape" %in% names(ICB_datalist) && !is.null(ICB_datalist$Landscape)) {
    Landscape_data <- as.data.frame(ICB_datalist$Landscape, stringsAsFactors = FALSE, check.names = FALSE)
    if ("Sample" %in% colnames(Landscape_data)) {
      rownames(Landscape_data) <- as.character(Landscape_data$Sample)
      keep <- sample_ids %in% rownames(Landscape_data)
      Landscape_data <- Landscape_data[sample_ids[keep], , drop = FALSE]
      if (!all(keep)) {
        warning("Some samples are missing from Landscape_data: ", paste(sample_ids[!keep], collapse = ", "))
      }
      sample_ids_landscape <- rownames(Landscape_data)
    } else {
      Landscape_data <- data.frame(Sample = sample_ids, row.names = sample_ids, check.names = FALSE)
      sample_ids_landscape <- sample_ids
    }
  } else {
    Landscape_data <- data.frame(Sample = sample_ids, row.names = sample_ids, check.names = FALSE)
    sample_ids_landscape <- sample_ids
  }

  ## Ecotyper one-hot columns.
  if ("Ecotype" %in% colnames(Landscape_data)) {
    for (j in seq_len(10)) {
      ce <- paste0("CE", j)
      Landscape_data[[paste0("Ecotyper_", ce)]] <- as.numeric(Landscape_data$Ecotype == ce)
    }
    Landscape_data$Ecotype <- NULL
  }

  ## MFP one-hot columns.
  if ("MFP" %in% colnames(Landscape_data)) {
    mfp_vals <- unique(Landscape_data$MFP)
    for (mfp_cur in mfp_vals) {
      if (is.na(mfp_cur) || mfp_cur == "") next
      col_cur <- paste0("MFP_", sanitize_filename(as.character(mfp_cur)))
      Landscape_data[[col_cur]] <- as.numeric(Landscape_data$MFP == mfp_cur)
    }
    Landscape_data$MFP <- NULL
  }

  ## Replace the listed biomarker columns with values from collector expression.
  drop_cols <- c("PD_L1", "PD_1", "PD_L2", "CX3CL1", "CTLA4", "CXCL9", "HLA_DRA", "IFN_gamma", "HRH1")
  Landscape_data <- Landscape_data[, setdiff(colnames(Landscape_data), drop_cols), drop = FALSE]

  expr_sub <- expr[sample_ids_landscape, , drop = FALSE]
  Landscape_data$PDCD1 <- safe_get_gene_expression(expr_sub, c("PDCD1", "PD-1"))
  Landscape_data$CD274 <- safe_get_gene_expression(expr_sub, c("CD274", "PD-L1", "PDL1"))
  Landscape_data$PDCD1LG2 <- safe_get_gene_expression(expr_sub, c("PDCD1LG2", "PD-L2", "PDL2"))
  Landscape_data$CX3CL1 <- safe_get_gene_expression(expr_sub, c("CX3CL1"))
  Landscape_data$CXCL9 <- safe_get_gene_expression(expr_sub, c("CXCL9"))
  Landscape_data$CTLA4 <- safe_get_gene_expression(expr_sub, c("CTLA4"))
  Landscape_data$HLA_DRA <- safe_get_gene_expression(expr_sub, c("HLA-DRA", "HLA_DRA", "HLA.DRA"))
  Landscape_data$IFNG <- safe_get_gene_expression(expr_sub, c("IFNG", "IFN-gamma", "IFN_gamma"))
  Landscape_data$HRH1 <- safe_get_gene_expression(expr_sub, c("HRH1"))

  ## Keep Sample plus numeric features only.
  if (!"Sample" %in% colnames(Landscape_data)) Landscape_data$Sample <- rownames(Landscape_data)
  sample_col <- Landscape_data$Sample
  feature_df <- Landscape_data[, setdiff(colnames(Landscape_data), "Sample"), drop = FALSE]
  numeric_keep <- vapply(feature_df, is.numeric, logical(1))
  feature_df <- feature_df[, numeric_keep, drop = FALSE]
  Landscape_data <- cbind(Sample = sample_col, feature_df)
  rownames(Landscape_data) <- as.character(sample_col)
  Landscape_data
}

build_data_collector_for_folder <- function(folder_path, folder_name) {
  if (!folder_name %in% FOCUS_COHORT_TREATMENT_PAIRS$Project) {
    stop("Cohort is not included in the four ICB settings: ", folder_name)
  }
  message("Building data collector for: ", folder_name)
  ICB_datalist <- load_first_rdata_object(folder_path)

  metadata <- NULL
  expr_sample_gene <- NULL
  gene_names <- NULL
  expression_type <- "TPM"

  if (folder_name == "Gide et al") {
    metadata <- as.data.frame(ICB_datalist$Samples, stringsAsFactors = FALSE, check.names = FALSE)
    if ("PREEDT" %in% colnames(metadata)) metadata <- metadata[metadata$PREEDT == "PRE", , drop = FALSE]
    metadata <- metadata[metadata$Sample %in% colnames(ICB_datalist$TPM), , drop = FALSE]
    gene_names <- rownames(ICB_datalist$TPM)
    idx <- match_samples_to_expr_cols(metadata$Sample, colnames(ICB_datalist$TPM), allow_grep = FALSE)
    keep <- !is.na(idx)
    metadata <- metadata[keep, , drop = FALSE]
    expr_sample_gene <- make_expr_sample_gene(ICB_datalist$TPM[, idx[keep], drop = FALSE], gene_names, metadata$Sample)

  } else if (folder_name == "Hugo et al") {
    metadata <- as.data.frame(ICB_datalist$Samples, stringsAsFactors = FALSE, check.names = FALSE)
    if ("Biopsy Time" %in% colnames(metadata)) metadata <- metadata[metadata$`Biopsy Time` == "pre-treatment", , drop = FALSE]
    metadata <- metadata[metadata$Sample %in% colnames(ICB_datalist$TPM), , drop = FALSE]
    gene_names <- rownames(ICB_datalist$TPM)
    idx <- match_samples_to_expr_cols(metadata$Sample, colnames(ICB_datalist$TPM), allow_grep = FALSE)
    keep <- !is.na(idx)
    metadata <- metadata[keep, , drop = FALSE]
    expr_sample_gene <- make_expr_sample_gene(ICB_datalist$TPM[, idx[keep], drop = FALSE], gene_names, metadata$Sample)

  } else if (folder_name == "Jung et al") {
    metadata_source <- if ("Clinical" %in% names(ICB_datalist)) ICB_datalist$Clinical else ICB_datalist$Samples
    metadata <- as.data.frame(metadata_source, stringsAsFactors = FALSE, check.names = FALSE)
    metadata <- metadata[metadata$Sample %in% colnames(ICB_datalist$TPM), , drop = FALSE]
    gene_names <- rownames(ICB_datalist$TPM)
    idx <- match_samples_to_expr_cols(metadata$Sample, colnames(ICB_datalist$TPM), allow_grep = TRUE)
    keep <- !is.na(idx)
    metadata <- metadata[keep, , drop = FALSE]
    expr_sample_gene <- make_expr_sample_gene(ICB_datalist$TPM[, idx[keep], drop = FALSE], gene_names, metadata$Sample)

  }

  if (is.null(metadata) || is.null(expr_sample_gene)) {
    stop("Failed to build collector for: ", folder_name)
  }

  metadata <- standardize_icb_metadata(metadata, folder_name)
  rownames(metadata) <- make.unique(metadata$Sample)

  ## Ensure expression and metadata are aligned.
  common_samples <- intersect(metadata$Sample, rownames(expr_sample_gene))
  metadata <- metadata[match(common_samples, metadata$Sample), , drop = FALSE]
  expr_sample_gene <- expr_sample_gene[common_samples, , drop = FALSE]

  if (!all(metadata$Sample == rownames(expr_sample_gene))) {
    stop("Sample names do not match after collector construction for: ", folder_name)
  }
  if (any(!is.finite(expr_sample_gene))) {
    stop("Expression matrix contains NA/Inf in folder: ", folder_name)
  }

  data_collector <- list(
    metadata = metadata,
    TPMexpression = expr_sample_gene,
    genename = colnames(expr_sample_gene),
    expression_type = expression_type,
    source_folder = folder_name
  )

  data_collector$Landscape_data <- build_landscape_data(ICB_datalist, data_collector)

  if (!all(data_collector$metadata$Sample == rownames(data_collector$TPMexpression))) {
    stop("Sample names do not match between metadata and expression in folder: ", folder_name)
  }

  rds_path <- file.path(folder_path, paste0(folder_name, "_ICB_data_collector.rds"))
  saveRDS(data_collector, file = rds_path)
  message("Saved: ", rds_path)
  data_collector
}

if (isTRUE(RUN_STEP1_BUILD_COLLECTORS)) {
  collector_summary <- list()
  for (ii in seq_along(data_folders)) {
    folder_path <- data_folders[ii]
    folder_name <- data_folder_names[ii]
    rds_path <- file.path(folder_path, paste0(folder_name, "_ICB_data_collector.rds"))
    reused_existing_collector <- FALSE
    if (file.exists(rds_path) && !isTRUE(FORCE_REBUILD_DATA_COLLECTOR)) {
      message("Reuse existing collector: ", rds_path)
      data_collector <- readRDS(rds_path)
      reused_existing_collector <- TRUE
    } else {
      data_collector <- tryCatch(
        build_data_collector_for_folder(folder_path, folder_name),
        error = function(e) {
          warning("Failed to build collector for ", folder_name, ": ", conditionMessage(e))
          NULL
        }
      )
    }
    if (!is.null(data_collector)) {
      data_collector <- normalize_data_collector_schema(data_collector, folder_name)
      expr_info <- get_collector_expression(data_collector, allow_null = TRUE)
      if (is.null(expr_info) || is.null(expr_info$matrix) || nrow(expr_info$matrix) == 0 || ncol(expr_info$matrix) == 0) {
        warning("Collector has no usable expression matrix; skip summary for: ", folder_name)
        next
      }
      ## Persist schema normalization even when a collector was reused.
      if (isTRUE(reused_existing_collector)) {
        saveRDS(data_collector, file = rds_path)
        message("Updated legacy collector schema: ", rds_path)
      }
      expr_cur <- expr_info$matrix
      collector_summary[[folder_name]] <- data.frame(
        Project = folder_name,
        N_sample = nrow(expr_cur),
        N_gene = ncol(expr_cur),
        Has_ResponseICB = "ResponseICB" %in% colnames(data_collector$metadata) && any(!is.na(data_collector$metadata$ResponseICB)),
        Has_OS = "OS" %in% colnames(data_collector$metadata) && any(!is.na(data_collector$metadata$OS)),
        CancerTypeICB = unique(data_collector$metadata$CancerTypeICB)[1] %||% infer_cancer_label(folder_name, data_collector$metadata),
        ExpressionType = data_collector$expression_type %||% ifelse(grepl("count", expr_info$field, ignore.case = TRUE), "counts", "TPM"),
        ExpressionField = expr_info$field,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    }
  }
  if (length(collector_summary) > 0) {
    collector_summary_df <- do.call(rbind, collector_summary)
    write.csv(collector_summary_df, file.path(icb_analysis_outdir, "ICB_data_collector_summary.csv"), row.names = FALSE)
  }
}

# ============================================================
# 6. Joint gene-expression and LR-proxy projection helpers
# ============================================================

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

collapse_duplicate_gene_columns <- function(expr_sample_gene) {
  expr_sample_gene <- as.matrix(expr_sample_gene)
  mode(expr_sample_gene) <- "numeric"
  genes <- colnames(expr_sample_gene)
  keep <- !is.na(genes) & genes != ""
  expr_sample_gene <- expr_sample_gene[, keep, drop = FALSE]
  genes <- genes[keep]
  if (length(unique(genes)) == length(genes)) {
    return(expr_sample_gene)
  }
  gene_sums <- rowsum(t(expr_sample_gene), group = genes, reorder = FALSE)
  gene_counts <- as.numeric(table(genes)[rownames(gene_sums)])
  gene_avg <- sweep(gene_sums, 1, gene_counts, "/")
  t(gene_avg)
}

prepare_expression_matrix_for_decomp <- function(data_collector) {
  folder_name <- data_collector$source_folder %||% NA_character_
  data_collector <- normalize_data_collector_schema(data_collector, folder_name)
  expr_info <- get_collector_expression(data_collector, allow_null = FALSE)
  expr <- as.matrix(expr_info$matrix)
  mode(expr) <- "numeric"
  if (!is.null(data_collector$metadata) && "Sample" %in% colnames(data_collector$metadata) &&
      nrow(expr) == nrow(data_collector$metadata)) {
    rownames(expr) <- as.character(data_collector$metadata$Sample)
  }
  if (!is.null(data_collector$genename) && length(data_collector$genename) == ncol(expr)) {
    colnames(expr) <- as.character(data_collector$genename)
  }
  expr <- collapse_duplicate_gene_columns(expr)

  expression_type <- tolower(data_collector$expression_type %||% ifelse(grepl("count", expr_info$field, ignore.case = TRUE), "counts", "TPM"))

  if (expression_type %in% c("counts", "count", "raw_counts")) {
    expr_gene_sample <- t(expr)
    expr_log <- edgeR::cpm(expr_gene_sample, log = TRUE, prior.count = 1)
    expr <- t(expr_log)
  } else {
    if (tpm_transform == "log2p1") {
      expr <- log2(pmax(expr, 0) + 1)
    } else if (tpm_transform == "none") {
      expr <- pmax(expr, 0)
    } else {
      stop("Unknown tpm_transform: ", tpm_transform)
    }
  }

  if (any(!is.finite(expr))) stop("Prepared expression matrix has NA/Inf.")
  expr
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
    ligand_expr <- pmax(ligand_expr, 0)
    receptor_expr <- pmax(receptor_expr, 0)
    ## Form the bulk LR proxy from nonnegative means on the transformed scale.
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

  if (length(gene_weight_vec) != G) stop("gene_weight_vec length mismatch.")
  if (length(lr_weight_vec) != R_lr) stop("lr_weight_vec length mismatch.")
  if (any(!is.finite(gene_weight_vec)) || any(gene_weight_vec < 0)) stop("Invalid gene weights.")
  if (any(!is.finite(lr_weight_vec)) || any(lr_weight_vec < 0)) stop("Invalid LR weights.")

  W_gene_diag <- Matrix::Diagonal(x = gene_weight_vec)
  Pmat <- 2 * (lambda_gene / G) * (D_gene %*% W_gene_diag %*% t(D_gene))
  if (isTRUE(use_lr_loss)) {
    W_lr_diag <- Matrix::Diagonal(x = lr_weight_vec)
    Pmat <- Pmat + 2 * (lambda_lr / R_lr) * (D_lr %*% W_lr_diag %*% t(D_lr))
  }
  Pmat <- Matrix::Matrix(Pmat, sparse = TRUE)

  ## Intrinsic abundances sum to one and are nonnegative; each MI lies in [0, 1].
  Aeq <- Matrix::Matrix(c(rep(1, C), rep(0, M)), nrow = 1, sparse = TRUE)
  Aid <- Matrix::Diagonal(K)
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

compute_joint_objective <- function(X, C_lr, pred_gene_no_intercept, pred_lr_no_intercept,
                                    beta0_gene, beta0_LR, gene_weight_vec, lr_weight_vec,
                                    lambda_gene, lambda_lr, use_lr_loss) {
  G <- ncol(X)
  R_lr <- ncol(C_lr)
  resid_gene <- sweep(X - pred_gene_no_intercept, 2, beta0_gene, "-")
  sse_gene <- sum(sweep(resid_gene^2, 2, gene_weight_vec, "*"))
  gene_component <- (lambda_gene / G) * sse_gene / nrow(X)
  if (isTRUE(use_lr_loss)) {
    resid_lr <- sweep(C_lr - pred_lr_no_intercept, 2, beta0_LR, "-")
    sse_lr <- sum(sweep(resid_lr^2, 2, lr_weight_vec, "*"))
    lr_component <- (lambda_lr / R_lr) * sse_lr / nrow(X)
  } else {
    lr_component <- 0
  }
  gene_component + lr_component
}

solve_joint_decomposition_group <- function(X, C_lr, W0, W_MI, W_LR,
                                            gene_weight_vec,
                                            lr_weight_vec,
                                            group_label = "group") {
  X <- as.matrix(X); mode(X) <- "numeric"
  C_lr <- as.matrix(C_lr); mode(C_lr) <- "numeric"
  W0 <- as.matrix(W0); mode(W0) <- "numeric"
  W_MI <- as.matrix(W_MI); mode(W_MI) <- "numeric"
  W_LR <- as.matrix(W_LR); mode(W_LR) <- "numeric"

  if (any(!is.finite(X))) stop("X has NA/Inf for ", group_label)
  if (any(!is.finite(C_lr))) stop("C_lr has NA/Inf for ", group_label)
  if (any(!is.finite(W0)) || any(!is.finite(W_MI)) || any(!is.finite(W_LR))) {
    stop("Loading matrices have NA/Inf for ", group_label)
  }

  N <- nrow(X)
  G <- ncol(X)
  R_lr <- ncol(C_lr)
  C <- nrow(W0)
  M <- nrow(W_MI)
  K <- C + M

  D_gene <- rbind(W0, W_MI)
  D_gene <- as.matrix(D_gene); mode(D_gene) <- "numeric"
  D_lr <- rbind(
    matrix(0, nrow = C, ncol = ncol(W_LR), dimnames = list(rownames(W0), colnames(W_LR))),
    W_LR
  )
  D_lr <- as.matrix(D_lr); mode(D_lr) <- "numeric"

  scale_factor_gene <- 1.0
  scale_factor_lr <- 1.0
  if (isTRUE(use_scaling)) {
    absX <- abs(as.numeric(X)); absX <- absX[is.finite(absX)]
    scale_factor_gene <- as.numeric(stats::quantile(absX, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor_gene) || scale_factor_gene <= 0) scale_factor_gene <- 1.0
    X <- X / scale_factor_gene
    W0 <- W0 / scale_factor_gene
    W_MI <- W_MI / scale_factor_gene
    D_gene <- rbind(W0, W_MI)

    absC <- abs(as.numeric(C_lr)); absC <- absC[is.finite(absC)]
    scale_factor_lr <- as.numeric(stats::quantile(absC, probs = 0.95, names = FALSE))
    if (!is.finite(scale_factor_lr) || scale_factor_lr <= 0) scale_factor_lr <- 1.0
    C_lr <- C_lr / scale_factor_lr
    W_LR <- W_LR / scale_factor_lr
    D_lr <- rbind(
      matrix(0, nrow = C, ncol = ncol(W_LR), dimnames = list(rownames(W0), colnames(W_LR))),
      W_LR
    )
  }

  tpl <- make_osqp_template_joint(
    D_gene = D_gene,
    D_lr = D_lr,
    C = C,
    M = M,
    gene_weight_vec = gene_weight_vec,
    lr_weight_vec = lr_weight_vec,
    lambda_gene = lambda_gene,
    lambda_lr = lambda_lr,
    use_lr_loss = use_lr_loss
  )

  beta0_gene <- if (isTRUE(if_set_beta0_zero)) rep(0, G) else colMeans(X)
  beta0_LR <- if (isTRUE(if_set_beta0_LR_zero) || !isTRUE(use_lr_loss)) rep(0, R_lr) else colMeans(C_lr)

  if (isTRUE(if_beta0_nonnegative)) {
    beta0_gene <- pmax(beta0_gene, 0)
    beta0_LR <- pmax(beta0_LR, 0)
  }

  P_hat <- matrix(0, N, C)
  MI_hat <- matrix(0, N, M)
  pred_gene_no_intercept <- matrix(0, N, G)
  pred_lr_no_intercept <- matrix(0, N, R_lr)

  q1 <- make_q_joint(D_gene, D_lr, X[1, ] - beta0_gene, C_lr[1, ] - beta0_LR, tpl)
  model <- osqp::osqp(
    P = tpl$P,
    q = q1,
    A = tpl$A,
    l = tpl$l,
    u = tpl$u,
    pars = list(verbose = FALSE, warm_start = TRUE)
  )

  prev_objective <- NA_real_
  objective_trace <- numeric(0)

  for (it in seq_len(max_outer)) {
    for (i in seq_len(N)) {
      y_gene <- as.numeric(X[i, ] - beta0_gene)
      y_lr <- as.numeric(C_lr[i, ] - beta0_LR)
      q <- make_q_joint(D_gene, D_lr, y_gene, y_lr, tpl)
      model$Update(q = q)
      r <- model$Solve()
      if (!(r$info$status_val %in% c(1L, 2L))) {
        stop(sprintf("OSQP failed in %s at sample %d, outer %d. Status: %s", group_label, i, it, r$info$status))
      }
      z <- r$x
      if (any(!is.finite(z))) stop("Non-finite OSQP solution in ", group_label)
      P_hat[i, ] <- z[seq_len(C)]
      MI_hat[i, ] <- z[(C + 1):K]
      pred_gene_no_intercept[i, ] <- as.numeric(z %*% D_gene)
      pred_lr_no_intercept[i, ] <- as.numeric(z %*% D_lr)
    }

    beta0_gene_new <- if (isTRUE(if_set_beta0_zero)) beta0_gene else colMeans(X - pred_gene_no_intercept)
    beta0_LR_new <- if (isTRUE(if_set_beta0_LR_zero) || !isTRUE(use_lr_loss)) beta0_LR else colMeans(C_lr - pred_lr_no_intercept)

    if (isTRUE(if_beta0_nonnegative)) {
      beta0_gene_new <- pmax(beta0_gene_new, 0)
      beta0_LR_new <- pmax(beta0_LR_new, 0)
    }

    if (isTRUE(use_beta0_ridge)) {
      beta0_gene_new <- beta0_step_size * beta0_gene_new + (1 - beta0_step_size) * beta0_gene
      beta0_LR_new <- beta0_step_size * beta0_LR_new + (1 - beta0_step_size) * beta0_LR
    }

    objective_value <- compute_joint_objective(
      X = X,
      C_lr = C_lr,
      pred_gene_no_intercept = pred_gene_no_intercept,
      pred_lr_no_intercept = pred_lr_no_intercept,
      beta0_gene = beta0_gene_new,
      beta0_LR = beta0_LR_new,
      gene_weight_vec = gene_weight_vec,
      lr_weight_vec = lr_weight_vec,
      lambda_gene = lambda_gene,
      lambda_lr = lambda_lr,
      use_lr_loss = use_lr_loss
    )
    if (!is.finite(objective_value)) stop("Objective became non-finite in ", group_label)

    objective_trace <- c(objective_trace, objective_value)
    if (is.na(prev_objective)) {
      cat(sprintf("%s | Outer %d | joint objective=%.6g (init)\n", group_label, it, objective_value))
      prev_objective <- objective_value
      beta0_gene <- beta0_gene_new
      beta0_LR <- beta0_LR_new
      if (isTRUE(if_set_beta0_zero) && (!isTRUE(use_lr_loss) || isTRUE(if_set_beta0_LR_zero))) break
    } else {
      rel_change <- abs(prev_objective - objective_value) / max(1, abs(prev_objective))
      cat(sprintf("%s | Outer %d | joint objective=%.6g | rel_change=%.3g\n", group_label, it, objective_value, rel_change))
      beta0_gene <- beta0_gene_new
      beta0_LR <- beta0_LR_new
      if (rel_change < tol) break
      prev_objective <- objective_value
    }
  }

  if (isTRUE(use_scaling)) {
    beta0_gene <- beta0_gene * scale_factor_gene
    beta0_LR <- beta0_LR * scale_factor_lr
  }

  MI_hat[MI_hat < 0] <- 0
  MI_hat[MI_hat > 1] <- 1
  P_hat[P_hat < 0] <- 0

  rownames(MI_hat) <- rownames(X)
  colnames(MI_hat) <- rownames(W_MI)
  rownames(P_hat) <- rownames(X)
  colnames(P_hat) <- rownames(W0)

  list(
    MI_hat = MI_hat,
    P_hat = P_hat,
    beta0_gene = beta0_gene,
    beta0_LR = beta0_LR,
    objective_trace = objective_trace,
    scale_factor_gene = scale_factor_gene,
    scale_factor_lr = scale_factor_lr
  )
}

# ============================================================
# 7. Project the selected ICB cohort-treatment groups
# ============================================================

if (isTRUE(RUN_STEP2_DECOMPOSITION)) {
  gene_weight_full <- NULL
  if (isTRUE(use_gene_weight) && !isTRUE(if_nongeneweight)) {
    gene_weight_full <- compute_kl_specificity_weights(loading_MI, feature_label = "gene")
  }
  lr_weight_full <- NULL
  if (isTRUE(use_lr_weight) && !isTRUE(if_nonlrweight)) {
    lr_weight_full <- compute_kl_specificity_weights(loading_LR_use, feature_label = "LR-pair")
  }

  decomp_summary_list <- list()

  for (project_cur in ICB_project_list) {
    message("====================================================")
    message("ICB MI decomposition: ", project_cur)
    folder_path <- file.path(data_path, project_cur)
    collector_path <- file.path(folder_path, paste0(project_cur, "_ICB_data_collector.rds"))
    if (!file.exists(collector_path)) {
      warning("Collector not found; skip decomposition for: ", project_cur)
      next
    }
    data_collector <- readRDS(collector_path)
    data_collector <- normalize_data_collector_schema(data_collector, project_cur)
    expr_all <- prepare_expression_matrix_for_decomp(data_collector)

    genes_use <- intersect(genename_list, colnames(expr_all))
    if (length(genes_use) < MIN_GENES_FOR_DECOMP) {
      warning(project_cur, ": too few common genes for decomposition: ", length(genes_use))
      next
    }

    lr_build <- tryCatch(
      build_lr_coexpression_matrix(
        expr_df = expr_all,
        lr_pair_names = colnames(loading_LR_use),
        require_all_genes = lr_require_all_genes
      ),
      error = function(e) {
        warning(project_cur, ": failed to build LR coexpression matrix: ", conditionMessage(e))
        NULL
      }
    )
    if (is.null(lr_build)) next
    lr_pairs_use <- intersect(colnames(loading_LR_use), colnames(lr_build$matrix))
    if (length(lr_pairs_use) < MIN_LR_PAIRS_FOR_DECOMP) {
      warning(project_cur, ": too few LR pairs for decomposition: ", length(lr_pairs_use))
      next
    }

    expr_use <- expr_all[, genes_use, drop = FALSE]
    C_lr_use <- lr_build$matrix[rownames(expr_use), lr_pairs_use, drop = FALSE]

    celltype_info <- get_celltype_list_for_project(project_cur, data_collector$metadata)
    if (length(celltype_info$celltype_use) == 0) {
      warning(project_cur, ": no compatible celltype dictionary; skip.")
      next
    }

    W0 <- Loading_intrinsic_use[celltype_info$celltype_use, genes_use, drop = FALSE]
    W_MI <- loading_MI[, genes_use, drop = FALSE]
    W_LR <- loading_LR_use[, lr_pairs_use, drop = FALSE]

    if (isTRUE(if_nongeneweight) || !isTRUE(use_gene_weight)) {
      gene_loss_weight <- rep(1, length(genes_use)); names(gene_loss_weight) <- genes_use
    } else {
      gene_loss_weight <- gene_weight_full[genes_use]
    }
    if (isTRUE(if_nonlrweight) || !isTRUE(use_lr_weight)) {
      lr_loss_weight <- rep(1, length(lr_pairs_use)); names(lr_loss_weight) <- lr_pairs_use
    } else {
      lr_loss_weight <- lr_weight_full[lr_pairs_use]
    }
    ## Squared weights implement diagonal weighting inside the squared norm.
    if (isTRUE(use_paper_omega_weighting)) {
      gene_loss_weight <- gene_loss_weight ^ 2
      lr_loss_weight <- lr_loss_weight ^ 2
    }

    TreatmentICB_cur <- as.character(data_collector$metadata$TreatmentICB)
    TreatmentICB_cur[is.na(TreatmentICB_cur) | TreatmentICB_cur == ""] <- "Unknown"
    treat_unique <- unique(TreatmentICB_cur)
    focus_treatments_cur <- FOCUS_COHORT_TREATMENT_PAIRS$TreatmentICB[
      FOCUS_COHORT_TREATMENT_PAIRS$Project == project_cur
    ]
    treat_unique <- treat_unique[treat_unique %in% focus_treatments_cur]
    if (length(treat_unique) == 0) {
      warning(project_cur, ": no requested focus treatment group was found; skip decomposition.")
      next
    }

    MI_hat_all <- matrix(NA_real_, nrow = nrow(expr_use), ncol = nrow(W_MI),
                         dimnames = list(rownames(expr_use), rownames(W_MI)))
    P_hat_all <- matrix(NA_real_, nrow = nrow(expr_use), ncol = nrow(W0),
                        dimnames = list(rownames(expr_use), rownames(W0)))

    beta0_gene_df_list <- list()
    beta0_LR_df_list <- list()
    objective_df_list <- list()

    for (treat in treat_unique) {
      idx <- which(TreatmentICB_cur == treat)
      if (length(idx) < MIN_SAMPLES_PER_DECOMP_GROUP) {
        warning(project_cur, " / ", treat, ": fewer than ", MIN_SAMPLES_PER_DECOMP_GROUP, " samples; skip this treatment group.")
        next
      }
      group_label <- paste0(project_cur, " | ", treat)
      fit <- solve_joint_decomposition_group(
        X = expr_use[idx, , drop = FALSE],
        C_lr = C_lr_use[idx, , drop = FALSE],
        W0 = W0,
        W_MI = W_MI,
        W_LR = W_LR,
        gene_weight_vec = gene_loss_weight,
        lr_weight_vec = lr_loss_weight,
        group_label = group_label
      )
      MI_hat_all[idx, ] <- fit$MI_hat
      P_hat_all[idx, ] <- fit$P_hat

      beta0_gene_df_list[[treat]] <- data.frame(
        TreatmentICB = treat,
        feature = names(fit$beta0_gene),
        beta0 = as.numeric(fit$beta0_gene),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      beta0_LR_df_list[[treat]] <- data.frame(
        TreatmentICB = treat,
        feature = names(fit$beta0_LR),
        beta0 = as.numeric(fit$beta0_LR),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      objective_df_list[[treat]] <- data.frame(
        TreatmentICB = treat,
        outer_iteration = seq_along(fit$objective_trace),
        objective = fit$objective_trace,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    }

    write.csv(MI_hat_all, file.path(icb_decomp_outdir, paste0("MI_decomposition_", sanitize_filename(project_cur), ".csv")))
    write.csv(P_hat_all, file.path(icb_decomp_outdir, paste0("Intrinsic_decomposition_", sanitize_filename(project_cur), ".csv")))
    write.csv(C_lr_use, file.path(icb_decomp_outdir, paste0("LR_coexpression_bulk_", sanitize_filename(project_cur), ".csv")))
    write.csv(data.frame(gene = genes_use), file.path(icb_decomp_outdir, paste0("genes_used_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)
    write.csv(data.frame(lr_pair = lr_pairs_use), file.path(icb_decomp_outdir, paste0("LR_pairs_used_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)
    write.csv(lr_build$metadata, file.path(icb_decomp_outdir, paste0("LR_pair_parse_metadata_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)
    if (nrow(lr_build$skipped) > 0) {
      write.csv(lr_build$skipped, file.path(icb_decomp_outdir, paste0("LR_pair_skipped_missing_genes_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)
    }
    if (length(beta0_gene_df_list) > 0) write.csv(do.call(rbind, beta0_gene_df_list), file.path(icb_decomp_outdir, paste0("beta0_gene_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)
    if (length(beta0_LR_df_list) > 0) write.csv(do.call(rbind, beta0_LR_df_list), file.path(icb_decomp_outdir, paste0("beta0_LR_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)
    if (length(objective_df_list) > 0) write.csv(do.call(rbind, objective_df_list), file.path(icb_decomp_outdir, paste0("objective_trace_", sanitize_filename(project_cur), ".csv")), row.names = FALSE)

    decomp_summary_list[[project_cur]] <- data.frame(
      Project = project_cur,
      N_sample = nrow(expr_use),
      N_gene_used = length(genes_use),
      N_LR_pair_used = length(lr_pairs_use),
      CancerTypeICB = celltype_info$cancer_label,
      CancerCellUsed = celltype_info$cancercell,
      CelltypePolicy = celltype_info$policy,
      N_treatment_groups = length(treat_unique),
      N_decomposed_samples = sum(rowSums(is.na(MI_hat_all)) == 0),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }

  if (length(decomp_summary_list) > 0) {
    decomp_summary_df <- do.call(rbind, decomp_summary_list)
    write.csv(decomp_summary_df, file.path(icb_analysis_outdir, "ICB_decomposition_summary.csv"), row.names = FALSE)
  }

  settings_df <- data.frame(
    setting = c(
      "data_path", "spidernet_result_dir", "loading_intrinsic_path", "loading_sender_path", "loading_receiver_path", "loading_LR_path",
      "dataset_selection", "n_datasets", "decomposition_only_focus_pairs", "if_set_beta0_zero", "if_set_beta0_LR_zero", "if_beta0_nonnegative",
      "use_beta0_ridge", "beta0_ridge_lambda_gene", "beta0_ridge_lambda_lr", "beta0_step_size",
      "if_nongeneweight", "if_nonlrweight", "use_gene_weight", "use_lr_weight",
      "use_lr_loss", "lambda_gene", "lambda_lr", "use_paper_omega_weighting", "use_scaling",
      "lr_require_all_genes", "tpm_transform", "celltype_policy",
      "response_sign_positive", "os_sign_positive"
    ),
    value = c(
      data_path, spidernet_result_dir, loading_intrinsic_path, loading_sender_path, loading_receiver_path, loading_LR_path,
      "four specified cohort-treatment settings", as.character(length(ICB_project_list)),
      "TRUE",
      as.character(if_set_beta0_zero), as.character(if_set_beta0_LR_zero), as.character(if_beta0_nonnegative),
      as.character(use_beta0_ridge), as.character(beta0_ridge_lambda_gene), as.character(beta0_ridge_lambda_lr), as.character(beta0_step_size),
      as.character(if_nongeneweight), as.character(if_nonlrweight), as.character(use_gene_weight), as.character(use_lr_weight),
      as.character(use_lr_loss), as.character(lambda_gene), as.character(lambda_lr), as.character(use_paper_omega_weighting), as.character(use_scaling),
      as.character(lr_require_all_genes), tpm_transform, "matched_cancercell_only",
      "+ signed -log10(P) = higher in ICB responders",
      "+ signed -log10(P) = higher feature associated with shorter OS/PFS"
    ),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  write.csv(settings_df, file.path(icb_analysis_outdir, "decomposition_settings_ICB_joint_gene_LR.csv"), row.names = FALSE)
}

# ============================================================
# 8. Response and OS/PFS time associations, aggregation, and heatmaps
# ============================================================

## Two-sided, asymptotic Wilcoxon rank-sum test with the default continuity
## correction. Direction is the responder-minus-non-responder median difference.
safe_signed_response_test <- function(feature_value, response_factor) {
  valid <- is.finite(feature_value) & !is.na(response_factor) & response_factor %in% c("No response", "Response")
  if (sum(valid) < MIN_SAMPLES_PER_ASSOC_TEST) return(c(signed_log10p = NA_real_, p = NA_real_, effect = NA_real_))
  y <- feature_value[valid]
  g <- factor(response_factor[valid], levels = c("No response", "Response"))
  tab <- table(g)
  if (length(tab) < 2 || any(tab < MIN_RESPONSE_PER_CLASS)) return(c(signed_log10p = NA_real_, p = NA_real_, effect = NA_real_))
  if (length(unique(y)) < 2) return(c(signed_log10p = NA_real_, p = NA_real_, effect = NA_real_))
  wt <- tryCatch(wilcox.test(y ~ g, exact = FALSE), error = function(e) NULL)
  if (is.null(wt)) return(c(signed_log10p = NA_real_, p = NA_real_, effect = NA_real_))
  med <- tapply(y, g, median, na.rm = TRUE)
  effect <- as.numeric(med["Response"] - med["No response"])
  sign_effect <- sign(effect)
  signed_log10p <- sign_effect * (-log10(max(wt$p.value, .Machine$double.xmin)))
  c(signed_log10p = signed_log10p, p = wt$p.value, effect = effect)
}

## Two-sided Spearman test on available times; OS/PFS endpoint fallback is
## defined by standardize_icb_metadata. Event/censoring indicators are not used.
safe_signed_os_test <- function(feature_value, os_vector) {
  os_num <- coerce_numeric(os_vector)
  valid <- is.finite(feature_value) & is.finite(os_num)
  if (sum(valid) < MIN_NONMISSING_OS) return(c(signed_log10p = NA_real_, p = NA_real_, rho = NA_real_))
  y <- feature_value[valid]
  os <- os_num[valid]
  if (length(unique(y)) < 2 || length(unique(os)) < 2) return(c(signed_log10p = NA_real_, p = NA_real_, rho = NA_real_))
  ct <- tryCatch(cor.test(y, os, method = "spearman", exact = FALSE), error = function(e) NULL)
  if (is.null(ct)) return(c(signed_log10p = NA_real_, p = NA_real_, rho = NA_real_))
  rho <- as.numeric(ct$estimate)
  ## Positive sign means inferior survival direction: higher feature -> shorter OS/PFS.
  sign_effect <- ifelse(rho < 0, 1, ifelse(rho > 0, -1, 0))
  signed_log10p <- sign_effect * (-log10(max(ct$p.value, .Machine$double.xmin)))
  c(signed_log10p = signed_log10p, p = ct$p.value, rho = rho)
}

make_numeric_landscape_matrix <- function(data_collector) {
  Landscape_data <- data_collector$Landscape_data
  if (is.null(Landscape_data) || nrow(Landscape_data) == 0) return(NULL)
  if ("Sample" %in% colnames(Landscape_data)) {
    rownames(Landscape_data) <- as.character(Landscape_data$Sample)
    Landscape_data$Sample <- NULL
  }
  numeric_keep <- vapply(Landscape_data, is.numeric, logical(1))
  Landscape_data <- Landscape_data[, numeric_keep, drop = FALSE]
  if (ncol(Landscape_data) == 0) return(NULL)
  as.matrix(Landscape_data)
}

feature_type_from_name <- function(x) {
  ifelse(grepl("^MI[0-9]+$", x), "MI", "Baseline_or_Landscape")
}


## Match only the four specified cohort-treatment settings.
is_focus_cohort_treatment_pair <- function(project_cur, treatment_cur) {
  any(
    FOCUS_COHORT_TREATMENT_PAIRS$Project == project_cur &
      FOCUS_COHORT_TREATMENT_PAIRS$TreatmentICB == treatment_cur
  )
}

make_focus_pair_description <- function(response_long_df) {
  out <- FOCUS_COHORT_TREATMENT_PAIRS
  out$Observed <- FALSE
  out$CancerType_observed <- NA_character_
  out$N <- NA_integer_
  out$N_Response <- NA_integer_
  out$N_No_response <- NA_integer_

  if (!is.null(response_long_df) && nrow(response_long_df) > 0) {
    response_long_df$CohortTreatment <- paste(response_long_df$Project, response_long_df$TreatmentICB, sep = " | ")
    detail_df <- unique(response_long_df[, c("CohortTreatment", "CancerType", "N", "N_Response", "N_No_response"), drop = FALSE])
    for (i in seq_len(nrow(out))) {
      idx <- which(detail_df$CohortTreatment == out$CohortTreatment[i])
      if (length(idx) > 0) {
        idx <- idx[1]
        out$Observed[i] <- TRUE
        out$CancerType_observed[i] <- as.character(detail_df$CancerType[idx])
        out$N[i] <- detail_df$N[idx]
        out$N_Response[i] <- detail_df$N_Response[idx]
        out$N_No_response[i] <- detail_df$N_No_response[idx]
      }
    }
  }
  out
}

if (isTRUE(RUN_STEP3_ASSOCIATION)) {
  response_rows <- list()
  os_rows <- list()

  for (project_cur in ICB_project_list) {
    message("Association testing: ", project_cur)
    folder_path <- file.path(data_path, project_cur)
    collector_path <- file.path(folder_path, paste0(project_cur, "_ICB_data_collector.rds"))
    mi_path <- file.path(icb_decomp_outdir, paste0("MI_decomposition_", sanitize_filename(project_cur), ".csv"))
    if (!file.exists(collector_path) || !file.exists(mi_path)) {
      warning("Missing collector or MI decomposition; skip association for: ", project_cur)
      next
    }

    data_collector <- readRDS(collector_path)
    data_collector <- normalize_data_collector_schema(data_collector, project_cur)
    MI_mat <- read.csv(mi_path, row.names = 1, check.names = FALSE)
    MI_mat <- as.matrix(MI_mat); mode(MI_mat) <- "numeric"

    common_samples <- intersect(data_collector$metadata$Sample, rownames(MI_mat))
    if (length(common_samples) == 0) {
      warning("No common samples between metadata and MI matrix for: ", project_cur)
      next
    }
    metadata <- data_collector$metadata[match(common_samples, data_collector$metadata$Sample), , drop = FALSE]
    MI_mat <- MI_mat[common_samples, , drop = FALSE]

    Landscape_mat <- make_numeric_landscape_matrix(data_collector)
    if (!is.null(Landscape_mat)) {
      landscape_aligned <- matrix(NA_real_, nrow = length(common_samples), ncol = ncol(Landscape_mat),
                                  dimnames = list(common_samples, colnames(Landscape_mat)))
      hit_landscape <- intersect(common_samples, rownames(Landscape_mat))
      if (length(hit_landscape) > 0) {
        landscape_aligned[hit_landscape, ] <- Landscape_mat[hit_landscape, , drop = FALSE]
      }
      feature_mat <- cbind(landscape_aligned, MI_mat)
    } else {
      feature_mat <- MI_mat
    }
    feature_mat <- as.matrix(feature_mat); mode(feature_mat) <- "numeric"

    response <- factor(metadata$ResponseICB, levels = c("No response", "Response"))
    os_vec <- metadata$OS
    treat_vec <- as.character(metadata$TreatmentICB)
    treat_vec[is.na(treat_vec) | treat_vec == ""] <- "Unknown"
    cancer_label <- unique(metadata$CancerTypeICB)[1] %||% infer_cancer_label(project_cur, metadata)

    for (treat in unique(treat_vec)) {
      if (!is_focus_cohort_treatment_pair(project_cur, treat)) next
      idx <- which(treat_vec == treat)
      if (length(idx) < MIN_SAMPLES_PER_ASSOC_TEST) next
      feature_sub <- feature_mat[idx, , drop = FALSE]
      response_sub <- response[idx]
      os_sub <- os_vec[idx]
      n_response <- sum(response_sub == "Response", na.rm = TRUE)
      n_no_response <- sum(response_sub == "No response", na.rm = TRUE)

      for (feature_name in colnames(feature_sub)) {
        x <- as.numeric(feature_sub[, feature_name])
        res_response <- safe_signed_response_test(x, response_sub)
        response_rows[[length(response_rows) + 1]] <- data.frame(
          Project = project_cur,
          TreatmentICB = treat,
          CancerType = cancer_label,
          Feature = feature_name,
          FeatureType = feature_type_from_name(feature_name),
          N = length(idx),
          N_Response = n_response,
          N_No_response = n_no_response,
          signed_log10p = as.numeric(res_response["signed_log10p"]),
          p_value = as.numeric(res_response["p"]),
          effect_median_Response_minus_NoResponse = as.numeric(res_response["effect"]),
          stringsAsFactors = FALSE,
          check.names = FALSE
        )

        res_os <- safe_signed_os_test(x, os_sub)
        os_rows[[length(os_rows) + 1]] <- data.frame(
          Project = project_cur,
          TreatmentICB = treat,
          CancerType = cancer_label,
          Feature = feature_name,
          FeatureType = feature_type_from_name(feature_name),
          N = length(idx),
          N_nonmissing_OS = sum(is.finite(coerce_numeric(os_sub)) & is.finite(x)),
          signed_log10p = as.numeric(res_os["signed_log10p"]),
          p_value = as.numeric(res_os["p"]),
          spearman_rho = as.numeric(res_os["rho"]),
          stringsAsFactors = FALSE,
          check.names = FALSE
        )
      }
    }
  }

  response_long <- if (length(response_rows) > 0) do.call(rbind, response_rows) else data.frame()
  os_long <- if (length(os_rows) > 0) do.call(rbind, os_rows) else data.frame()

  focus_pair_description_df <- make_focus_pair_description(response_long)
  if (nrow(focus_pair_description_df) > 0) {
    write.csv(
      focus_pair_description_df,
      file.path(icb_analysis_outdir, "ICB_focus_cohort_treatment_pairs_used.csv"),
      row.names = FALSE
    )
    cat("Focused cohort-treatment pairs used for association analysis:
")
    print(focus_pair_description_df)
  }

  write.csv(response_long, file.path(icb_analysis_outdir, "ICB_response_signed_log10p_long.csv"), row.names = FALSE)
  write.csv(os_long, file.path(icb_analysis_outdir, "ICB_OS_signed_log10p_long.csv"), row.names = FALSE)

  make_wide_matrix <- function(df, value_col = "signed_log10p") {
    if (nrow(df) == 0) return(matrix(nrow = 0, ncol = 0))
    df$CohortTreatment <- paste(df$Project, df$TreatmentICB, sep = " | ")
    features <- unique(df$Feature)
    cohorts <- unique(df$CohortTreatment)
    mat <- matrix(NA_real_, nrow = length(features), ncol = length(cohorts), dimnames = list(features, cohorts))
    for (i in seq_len(nrow(df))) {
      mat[df$Feature[i], df$CohortTreatment[i]] <- df[[value_col]][i]
    }
    mat
  }

  ## Convert raw two-sided P values and directions to signed normal scores.
  signed_log10p_to_z <- function(x) {
    sign_x <- sign(x)
    p_two_sided <- 10^(-abs(x))
    p_two_sided <- pmin(pmax(p_two_sided, .Machine$double.xmin), 1)
    sign_x * stats::qnorm(p_two_sided / 2, lower.tail = FALSE)
  }

  ## Weight each setting by its total aligned sample count N, including samples
  ## without a usable value for an individual test. No multiplicity adjustment
  ## is applied to the per-setting or combined P values.
  aggregate_liptak <- function(df) {
    if (nrow(df) == 0) return(data.frame())
    split_df <- split(df, df$Feature)
    out <- lapply(names(split_df), function(feature_cur) {
      d <- split_df[[feature_cur]]
      x <- d$signed_log10p
      valid <- is.finite(x)
      if (!any(valid)) {
        return(data.frame(
          Feature = feature_cur,
          FeatureType = unique(d$FeatureType)[1],
          N_tests = 0,
          Z_Liptak = NA_real_,
          p_Liptak = NA_real_,
          signed_log10p_Liptak = NA_real_,
          stringsAsFactors = FALSE
        ))
      }
      z <- signed_log10p_to_z(x[valid])
      w <- d$N[valid]
      w[!is.finite(w) | w <= 0] <- 1
      Z <- sum(w * z) / sqrt(sum(w^2))
      p <- 2 * stats::pnorm(-abs(Z))
      signed_log10p <- sign(Z) * (-log10(max(p, .Machine$double.xmin)))
      data.frame(
        Feature = feature_cur,
        FeatureType = unique(d$FeatureType)[1],
        N_tests = length(z),
        Z_Liptak = Z,
        p_Liptak = p,
        signed_log10p_Liptak = signed_log10p,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    })
    out <- do.call(rbind, out)
    out[order(out$signed_log10p_Liptak, decreasing = TRUE), , drop = FALSE]
  }

  response_mat <- make_wide_matrix(response_long, "signed_log10p")
  os_mat <- make_wide_matrix(os_long, "signed_log10p")

  ## Keep the response/OS matrix columns in the exact order shown in
  ## FOCUS_COHORT_TREATMENT_PAIRS.
  if (ncol(response_mat) > 0) {
    focus_col_order <- FOCUS_COHORT_TREATMENT_PAIRS$CohortTreatment
    focus_col_order <- focus_col_order[focus_col_order %in% colnames(response_mat)]
    response_mat <- response_mat[, focus_col_order, drop = FALSE]
  }
  if (ncol(os_mat) > 0) {
    focus_col_order <- FOCUS_COHORT_TREATMENT_PAIRS$CohortTreatment
    focus_col_order <- focus_col_order[focus_col_order %in% colnames(os_mat)]
    os_mat <- os_mat[, focus_col_order, drop = FALSE]
  }

  write.csv(response_mat, file.path(icb_analysis_outdir, "ICB_response_signed_log10p_matrix.csv"))
  write.csv(os_mat, file.path(icb_analysis_outdir, "ICB_OS_signed_log10p_matrix.csv"))

  response_liptak <- aggregate_liptak(response_long)
  os_liptak <- aggregate_liptak(os_long)
  write.csv(response_liptak, file.path(icb_analysis_outdir, "ICB_response_Liptak_summary.csv"), row.names = FALSE)
  write.csv(os_liptak, file.path(icb_analysis_outdir, "ICB_OS_Liptak_summary.csv"), row.names = FALSE)


  mi_features <- mi_names[mi_names %in% rownames(response_mat)]
  plot_signed_heatmap(
    icb_analysis_outdir = icb_analysis_outdir,
    HEATMAP_TOP_N_FEATURES = HEATMAP_TOP_N_FEATURES,
    HEATMAP_FILL_NA_WITH_ZERO = HEATMAP_FILL_NA_WITH_ZERO,
    FORMAT_MI_LABELS_WITH_DASH = FORMAT_MI_LABELS_WITH_DASH,
    response_mat,
    response_liptak,
    "Heatmap_Response_MI_only.pdf",
    "ICB response association: MI modules",
    features_keep = mi_features,
    order_rows_by_mean_rank = isTRUE(ORDER_RESPONSE_HEATMAP_ROWS_BY_MEAN_RANK)
  )
  plot_signed_heatmap(
    icb_analysis_outdir = icb_analysis_outdir,
    HEATMAP_TOP_N_FEATURES = HEATMAP_TOP_N_FEATURES,
    HEATMAP_FILL_NA_WITH_ZERO = HEATMAP_FILL_NA_WITH_ZERO,
    FORMAT_MI_LABELS_WITH_DASH = FORMAT_MI_LABELS_WITH_DASH,
    response_mat,
    response_liptak,
    "Heatmap_Response_top_features.pdf",
    "ICB response association: top features",
    features_keep = NULL,
    order_rows_by_mean_rank = isTRUE(ORDER_RESPONSE_HEATMAP_ROWS_BY_MEAN_RANK),
    features_drop = EXCLUDE_RESPONSE_TOP_FEATURES,
    color_limit = HEATMAP_RESPONSE_TOP_FEATURES_COLOR_LIMIT
  )
  mi_features_os <- mi_names[mi_names %in% rownames(os_mat)]
  plot_signed_heatmap(
    icb_analysis_outdir = icb_analysis_outdir,
    HEATMAP_TOP_N_FEATURES = HEATMAP_TOP_N_FEATURES,
    HEATMAP_FILL_NA_WITH_ZERO = HEATMAP_FILL_NA_WITH_ZERO,
    FORMAT_MI_LABELS_WITH_DASH = FORMAT_MI_LABELS_WITH_DASH,
    os_mat,
    os_liptak,
    "Heatmap_OS_MI_only.pdf",
    "OS/PFS association: MI modules",
    features_keep = mi_features_os
  )
  plot_signed_heatmap(
    icb_analysis_outdir = icb_analysis_outdir,
    HEATMAP_TOP_N_FEATURES = HEATMAP_TOP_N_FEATURES,
    HEATMAP_FILL_NA_WITH_ZERO = HEATMAP_FILL_NA_WITH_ZERO,
    FORMAT_MI_LABELS_WITH_DASH = FORMAT_MI_LABELS_WITH_DASH,
    os_mat,
    os_liptak,
    "Heatmap_OS_top_features.pdf",
    "OS/PFS association: top features",
    features_keep = NULL
  )

  cat("✅ ICB response/OS association analysis finished.\n")
  cat("Outputs saved to: ", icb_analysis_outdir, "\n")
}

# ============================================================
# 9. Export a separate MI-4 cohort-treatment order table
# ============================================================
# This section always runs and requires the response long table on disk,
# even when RUN_STEP3_ASSOCIATION is FALSE. The exported order does not
# change the heatmaps already generated in Section 8.

suppressPackageStartupMessages({
  library(dplyr)
})

export_icb_mi4_order(icb_analysis_outdir, FOCUS_COHORT_TREATMENT_PAIRS)
